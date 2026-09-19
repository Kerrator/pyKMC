"""Finite-difference mass-weighted partial Hessian from a forces callable.

Engine-agnostic validation oracle: the caller supplies ``forces_fn(positions)``
mapping full ``(N, 3)`` positions (Å) to full ``(N, 3)`` forces (eV/Å). Boundary
atoms are held fixed, so the result is a frozen-boundary *partial* Hessian with no
translational zero modes to project out.
"""

from __future__ import annotations

import math
from typing import Any, Callable

import numpy as np

from .request import HTSTRequestError

ForcesFn = Callable[[np.ndarray], np.ndarray]
"""``forces_fn(positions) -> forces``: full ``(N, 3)`` Å in, full ``(N, 3)`` eV/Å out.

The returned array may be a buffer the callable reuses on its next call (a common
native-adapter pattern); consumers in this package copy it before calling again.
"""

HessianFn = Callable[[np.ndarray, np.ndarray], np.ndarray]
"""``hessian_fn(positions, free_indices) -> H_mw``.

Returns the real ``(3F, 3F)`` mass-weighted Hessian in eV / (amu Å²) of the ``F``
free atoms, in ``free_indices`` order. The matrix must be symmetric to within
``pykmc.htst.normal_modes.SYMMETRY_ATOL`` (1e-8 absolute): an adapter assembling
it from raw engine output (for example LAMMPS ``dynamical_matrix ... eskm``) must
return ``0.5 * (H + H.T)``, because asymmetry above the tolerance is treated as a
plumbing error and raises ``ValueError`` rather than producing a rejection. A
scientific refusal is signalled by raising :class:`PrefactorRejected`.
"""


def _free_index_array(free_indices: Any, n_atoms: int) -> np.ndarray:
    """Return ``free_indices`` as a validated 1-D int array within ``[0, n_atoms)``."""
    free = np.asarray(free_indices)
    if free.ndim != 1:
        raise ValueError(
            f"free_indices must be one-dimensional, got shape {free.shape}"
        )
    if free.size == 0:
        raise ValueError("free_indices must not be empty")
    if not np.issubdtype(free.dtype, np.integer):
        raise ValueError(f"free_indices must be integers, got dtype {free.dtype}")
    free = free.astype(int)
    if np.any(free < 0) or np.any(free >= n_atoms):
        raise ValueError(f"free_indices out of range for {n_atoms} atoms: {free}")
    if len(np.unique(free)) != free.size:
        raise ValueError(f"free_indices contains duplicates: {free}")
    return free


def mass_weighted_partial_hessian(
    forces_fn: ForcesFn,
    positions: Any,
    masses_per_atom: Any,
    free_indices: Any,
    fd_step: float,
) -> np.ndarray:
    """Assemble the ``(3F, 3F)`` mass-weighted partial Hessian by central differences.

    ``H_ab = -(F_a(+dx_b) - F_a(-dx_b)) / (2 dx) / sqrt(m_a m_b)`` over the free
    atoms only; boundary atoms never move. The result is symmetrised.

    Parameters
    ----------
    forces_fn : Callable
        Maps full ``(N, 3)`` positions to full ``(N, 3)`` forces in eV/Å. The
        returned array may be reused by the callable on its next call; it is
        copied here before ``forces_fn`` is evaluated again, so a callable that
        writes into and returns one preallocated buffer is valid.
    positions : array_like
        ``(N, 3)`` reference positions in Å.
    masses_per_atom : array_like
        ``(N,)`` masses in amu, one per atom.
    free_indices : array_like
        Global indices of the ``F`` free atoms.
    fd_step : float
        Finite-difference displacement in Å; must be finite and > 0.

    Returns
    -------
    np.ndarray
        ``(3F, 3F)`` symmetric mass-weighted Hessian in eV / (amu Å²), with row and
        column ``3 * k + c`` belonging to component ``c`` of ``free_indices[k]``.

    Raises
    ------
    ValueError
        For an empty or malformed free selection, a non-positive step,
        non-positive masses on free atoms, or a ``forces_fn`` return whose shape
        differs from ``positions``.
    HTSTRequestError
        If ``positions`` or ``masses_per_atom`` have the wrong shape.

    """
    pos = np.asarray(positions, dtype=float)
    if pos.ndim != 2 or pos.shape[1] != 3:
        raise HTSTRequestError(f"positions must have shape (N, 3), got {pos.shape}")
    n_atoms = pos.shape[0]
    masses = np.asarray(masses_per_atom, dtype=float).reshape(-1)
    if masses.shape != (n_atoms,):
        raise HTSTRequestError(
            f"masses_per_atom must have shape ({n_atoms},), got {masses.shape}"
        )
    free = _free_index_array(free_indices, n_atoms)
    if isinstance(fd_step, bool) or not isinstance(fd_step, (int, float, np.floating)):
        raise ValueError(f"fd_step must be a real number, got {fd_step!r}")
    if not math.isfinite(fd_step) or fd_step <= 0.0:
        raise ValueError(f"fd_step must be finite and > 0, got {fd_step!r}")
    dx = float(fd_step)
    free_masses = masses[free]
    if not np.all(np.isfinite(free_masses)) or np.any(free_masses <= 0.0):
        raise ValueError(f"free-atom masses must be finite and > 0, got {free_masses}")

    n_free = free.size
    hessian = np.zeros((3 * n_free, 3 * n_free), dtype=float)
    for k, atom in enumerate(free):
        for comp in range(3):
            col = 3 * k + comp
            pos_p = pos.copy()
            pos_m = pos.copy()
            pos_p[atom, comp] += dx
            pos_m[atom, comp] -= dx
            # Copy each result: the callable may hand back one reused buffer, and
            # the second evaluation would otherwise overwrite the first.
            f_p = np.array(forces_fn(pos_p), dtype=float, copy=True)
            f_m = np.array(forces_fn(pos_m), dtype=float, copy=True)
            if f_p.shape != pos.shape or f_m.shape != pos.shape:
                raise ValueError(
                    f"forces_fn must return shape {pos.shape}, got {f_p.shape} and "
                    f"{f_m.shape}"
                )
            hessian[:, col] = -(f_p[free] - f_m[free]).reshape(-1) / (2.0 * dx)

    inv_sqrt_m = np.repeat(1.0 / np.sqrt(free_masses), 3)
    hessian *= np.outer(inv_sqrt_m, inv_sqrt_m)
    return 0.5 * (hessian + hessian.T)


def fd_hessian_fn(
    forces_fn: ForcesFn, masses_per_atom: Any, fd_step: float
) -> HessianFn:
    """Bind a forces callable into a ``hessian_fn(positions, free_indices)``.

    Parameters
    ----------
    forces_fn : Callable
        Maps full ``(N, 3)`` positions to full ``(N, 3)`` forces in eV/Å; its
        return value may be a reused buffer (see
        :func:`mass_weighted_partial_hessian`).
    masses_per_atom : array_like
        ``(N,)`` masses in amu.
    fd_step : float
        Finite-difference displacement in Å.

    Returns
    -------
    Callable
        ``hessian_fn(positions, free_indices) -> H_mw`` suitable for
        :func:`pykmc.htst.prefactor.compute_event_prefactors`.

    """
    masses = np.asarray(masses_per_atom, dtype=float).reshape(-1)

    def hessian_fn(positions: np.ndarray, free_indices: np.ndarray) -> np.ndarray:
        return mass_weighted_partial_hessian(
            forces_fn, positions, masses, free_indices, fd_step
        )

    return hessian_fn


__all__ = ["ForcesFn", "HessianFn", "fd_hessian_fn", "mass_weighted_partial_hessian"]
