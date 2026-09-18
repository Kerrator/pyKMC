"""Typed per-event request for an HTST prefactor calculation and its validation."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from .settings import HTSTSettings

ORTHORHOMBIC_ATOL: float = 1.0e-8
"""Absolute tolerance (Å) below which off-diagonal cell entries count as zero."""


class HTSTRequestError(ValueError):
    """A request payload violates the HTST contract (shape, index, finiteness, species)."""


class HTSTGeometryError(HTSTRequestError):
    """The cell geometry is not supported by the HTST kernels (non-orthorhombic)."""


def require_orthorhombic(cell: Any, *, atol: float = ORTHORHOMBIC_ATOL) -> np.ndarray:
    """Return ``cell`` as a float ``(3, 3)`` array, raising unless it is orthorhombic.

    Parameters
    ----------
    cell : array_like
        Simulation cell as three row vectors.
    atol : float, optional
        Absolute tolerance for off-diagonal entries.

    Returns
    -------
    np.ndarray
        The validated ``(3, 3)`` float cell.

    Raises
    ------
    HTSTRequestError
        If the cell is not a finite ``(3, 3)`` array.
    HTSTGeometryError
        If any off-diagonal entry exceeds ``atol`` in magnitude.

    """
    try:
        arr = np.asarray(cell, dtype=float)
    except (TypeError, ValueError) as exc:
        raise HTSTRequestError(f"cell is not numeric: {exc}") from exc
    if arr.shape != (3, 3):
        raise HTSTRequestError(f"cell must have shape (3, 3), got {arr.shape}")
    if not np.all(np.isfinite(arr)):
        raise HTSTRequestError("cell contains non-finite entries")
    off = arr - np.diag(np.diag(arr))
    if np.any(np.abs(off) > atol):
        raise HTSTGeometryError(
            "HTST v1 supports orthorhombic cells only; off-diagonal cell entries "
            f"exceed {atol} Å: {arr.tolist()}"
        )
    return arr


def _as_positions(name: str, value: Any) -> np.ndarray:
    """Return ``value`` as a finite float ``(N, 3)`` array or raise ``HTSTRequestError``."""
    try:
        arr = np.asarray(value, dtype=float)
    except (TypeError, ValueError) as exc:
        raise HTSTRequestError(f"{name} is not numeric: {exc}") from exc
    if arr.ndim != 2 or arr.shape[1] != 3 or arr.shape[0] < 1:
        raise HTSTRequestError(
            f"{name} must have shape (N, 3) with N >= 1, got {arr.shape}"
        )
    if not np.all(np.isfinite(arr)):
        raise HTSTRequestError(f"{name} contains non-finite entries")
    return arr


@dataclass(frozen=True, eq=False)
class HTSTEventRequest:
    """Everything the HTST kernels need for one event, in the caller's atom order.

    The three geometries share one atom ordering; ``types`` gives the chemical symbol
    of each atom and ``species``/``masses`` give the full potential species order
    with the authoritative engine masses. Positions are full-system ``(N, 3)`` arrays
    in Å; the request does not copy them, the caller supplies copies.

    Attributes
    ----------
    event_key : tuple
        Hashable logical identity supplied by the caller and echoed back unchanged.
    min1_positions, saddle_positions, min2_positions : np.ndarray
        ``(N, 3)`` positions of the initial minimum, saddle and final minimum.
    types : tuple of str
        ``(N,)`` chemical symbols; every entry must appear in ``species``.
    species : tuple of str
        Full potential species order (``pair_coeff`` order).
    masses : tuple of float
        Masses in amu, one per entry of ``species``, in ``species`` order
        (Python or NumPy reals; never bools).
    cell : np.ndarray
        ``(3, 3)`` simulation cell; orthorhombic only in v1.
    pbc : tuple of bool
        Periodicity per axis (Python or NumPy bools, so ``tuple(atoms.pbc)``
        from ASE is accepted; ``0``/``1`` are not).
    center_index : int
        Global index of the moving atom; centre of the free region.
    settings : HTSTSettings
        Validated numerical settings.

    """

    event_key: tuple
    min1_positions: np.ndarray
    saddle_positions: np.ndarray
    min2_positions: np.ndarray
    types: tuple[str, ...]
    species: tuple[str, ...]
    masses: tuple[float, ...]
    cell: np.ndarray
    pbc: tuple[bool, bool, bool]
    center_index: int
    settings: HTSTSettings

    def validate(self) -> None:
        """Check shapes, indices, finiteness, species membership and geometry.

        Raises
        ------
        HTSTRequestError
            For any shape, index, finiteness, species-membership or mass-length
            violation.
        HTSTGeometryError
            When the cell is not orthorhombic.

        """
        if not isinstance(self.event_key, tuple):
            raise HTSTRequestError(
                f"event_key must be a tuple, got {type(self.event_key).__name__}"
            )
        if not isinstance(self.settings, HTSTSettings):
            raise HTSTRequestError(
                f"settings must be an HTSTSettings, got {type(self.settings).__name__}"
            )
        min1 = _as_positions("min1_positions", self.min1_positions)
        sad = _as_positions("saddle_positions", self.saddle_positions)
        min2 = _as_positions("min2_positions", self.min2_positions)
        n_atoms = min1.shape[0]
        if sad.shape != min1.shape or min2.shape != min1.shape:
            raise HTSTRequestError(
                "min1/saddle/min2 positions must share one shape, got "
                f"{min1.shape}, {sad.shape}, {min2.shape}"
            )
        if not isinstance(self.types, tuple) or not all(
            isinstance(t, str) for t in self.types
        ):
            raise HTSTRequestError("types must be a tuple of str")
        if len(self.types) != n_atoms:
            raise HTSTRequestError(
                f"types has {len(self.types)} entries but positions have {n_atoms} atoms"
            )
        if not isinstance(self.species, tuple) or not all(
            isinstance(s, str) for s in self.species
        ):
            raise HTSTRequestError("species must be a tuple of str")
        if len(self.species) == 0:
            raise HTSTRequestError("species must not be empty")
        if len(set(self.species)) != len(self.species):
            raise HTSTRequestError(f"species contains duplicates: {self.species}")
        missing = sorted(set(self.types) - set(self.species))
        if missing:
            raise HTSTRequestError(
                f"types {missing} are not in the potential species {self.species}"
            )
        if not isinstance(self.masses, tuple):
            raise HTSTRequestError("masses must be a tuple of float in species order")
        if len(self.masses) != len(self.species):
            raise HTSTRequestError(
                f"masses has {len(self.masses)} entries but species has "
                f"{len(self.species)}"
            )
        for symbol, mass in zip(self.species, self.masses, strict=True):
            if (
                isinstance(mass, (bool, np.bool_))
                or not isinstance(mass, (int, float, np.integer, np.floating))
                or not np.isfinite(mass)
                or mass <= 0.0
            ):
                raise HTSTRequestError(
                    f"mass of species {symbol!r} must be finite and > 0, got {mass!r}"
                )
        cell = require_orthorhombic(self.cell)
        if (
            not isinstance(self.pbc, tuple)
            or len(self.pbc) != 3
            or not all(isinstance(p, (bool, np.bool_)) for p in self.pbc)
        ):
            raise HTSTRequestError("pbc must be a tuple of three bools")
        for axis in range(3):
            if self.pbc[axis] and cell[axis, axis] <= 0.0:
                raise HTSTRequestError(
                    f"periodic axis {axis} needs a positive cell length, got "
                    f"{cell[axis, axis]!r}"
                )
        if isinstance(self.center_index, bool) or not isinstance(
            self.center_index, (int, np.integer)
        ):
            raise HTSTRequestError(
                f"center_index must be an int, got {type(self.center_index).__name__}"
            )
        if not 0 <= int(self.center_index) < n_atoms:
            raise HTSTRequestError(
                f"center_index {self.center_index} out of range for {n_atoms} atoms"
            )

    def masses_per_atom(self) -> np.ndarray:
        """Return the ``(N,)`` per-atom masses in amu by mapping types through species.

        Returns
        -------
        np.ndarray
            Float array of per-atom masses in the request's atom order.

        Raises
        ------
        HTSTRequestError
            If a type is not in ``species`` or ``masses`` does not match ``species``.

        """
        if len(self.masses) != len(self.species):
            raise HTSTRequestError(
                f"masses has {len(self.masses)} entries but species has "
                f"{len(self.species)}"
            )
        lookup = dict(zip(self.species, self.masses, strict=True))
        try:
            return np.array([float(lookup[t]) for t in self.types], dtype=float)
        except KeyError as exc:
            raise HTSTRequestError(
                f"type {exc.args[0]!r} is not in the potential species {self.species}"
            ) from exc


def default_masses_for_species(species: tuple[str, ...]) -> tuple[float, ...]:
    """Return standard atomic masses (amu) for ``species`` from ASE's periodic table.

    This is the only place in ``pykmc.htst`` that touches ASE, and it imports it
    lazily. Use it only to build a request when the engine cannot supply its own
    masses; the engine's masses are authoritative whenever they exist.

    Parameters
    ----------
    species : tuple of str
        Chemical symbols in potential species order.

    Returns
    -------
    tuple of float
        Masses in amu, one per species.

    """
    from ase.data import atomic_masses, atomic_numbers

    return tuple(float(atomic_masses[atomic_numbers[s]]) for s in species)


__all__ = [
    "HTSTEventRequest",
    "HTSTGeometryError",
    "HTSTRequestError",
    "ORTHORHOMBIC_ATOL",
    "default_masses_for_species",
    "require_orthorhombic",
]
