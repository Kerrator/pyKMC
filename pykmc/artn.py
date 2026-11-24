# The ARTn search in this module is ported from pARTn v1.1.0
# (https://gitlab.com/mammasmias/artn-plugin, tag v1.1.0, commit 9b753517),
# which is distributed under the following notice:
#
#   Copyright (c) 2021, MAMMASMIAS Consortium
#   SPDX-License-Identifier: Apache-2.0
#
#   Licensed under the Apache License, Version 2.0 (the "License"); you may
#   not use this file except in compliance with the License. You may obtain a
#   copy of the License at
#
#       http://www.apache.org/licenses/LICENSE-2.0
#
#   Unless required by applicable law or agreed to in writing, software
#   distributed under the License is distributed on an "AS IS" BASIS, WITHOUT
#   WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the
#   License for the specific language governing permissions and limitations
#   under the License.
#
# This file is a rewrite in Python, not a copy of pARTn's Fortran. What it
# changes is listed in the module docstring.
"""ARTn saddle search on a plain energy and force callback (plain core).

From a minimum, the search pushes the atoms around the central atom in a
random direction until the lowest curvature, found by Lanczos, is negative
enough; then it climbs along the lowest mode to a first-order saddle point,
relaxing the other directions as it goes; then it pushes over the saddle on
both sides and relaxes each side to a minimum. Energies and forces come only
from a callback, so the search needs no LAMMPS, pARTn or MPI and runs on an
analytic surface.

Ported from pARTn v1.1.0 (Poberznik et al. 2024) as pyKMC's ``partn_search``
drives it, through the LAMMPS fix and LAMMPS FIRE. One ARTn step is one
callback call:

- the initial push: a random vector on the movable atoms within
  ``push_dist_thr`` of the central atom (``push_mode = "rad"``) or on the
  central atom only (``"list"``), scaled to a 3N norm of ``push_step_size``,
  applied ``ninit`` times, each followed by ``nperp`` FIRE steps
  perpendicular to it;
- Lanczos at the current point, from the last eigenvector (a random vector on
  every movable atom the first time), one force call per probe at
  ``lanczos_disp``, until the lowest eigenvalue settles (at least
  ``lanczos_min_size - 1`` calls, as in pARTn, and at most
  ``lanczos_max_size``);
- in the basin, another push and perpendicular relaxation until the lowest
  eigenvalue is below ``eigval_thr``; out of the basin, a push along the
  eigenvector, uphill, of ``min(eigen_step_size, |F.e| / max(|lambda|,
  0.48587 eV/Å²))``;
- FIRE relaxation perpendicular to the eigenvector, until the largest force
  component (or the force's 3N norm, ``convergence_property = "norm"``) is
  below ``forc_thr``, which is the saddle, or the perpendicular force is
  smaller than the parallel force, which calls Lanczos again;
- a push off the saddle along +eigenvector, of ``eigen_step_size * push_over
  * 0.8``, and FIRE relaxation to a minimum on the full force; then the same
  along -eigenvector.

FIRE is LAMMPS's (euler implicit) with the settings pARTn's fix gives it, in
LAMMPS metal units: time step 0.001 ps, masses in amu, ``alpha0`` 0.1,
``alphashrink`` 0.99, ``dtgrow`` 1.1, ``dtshrink`` 0.5, ``tmax`` 10, ``tmin``
0.02, half step back on, ``delaystep`` 5 for perpendicular relaxation and 0
for relaxation to a minimum, and the step time cut so that it times the
largest velocity component before the kick stays within ``dmax``. On a
restart FIRE reuses the earlier force scaled by (dt0/dt)², as pARTn's fix
does, instead of evaluating at the half-stepped positions. The time step dt0
is LAMMPS's 0.001 ps after pARTn's round trip through its time unit,
0.0010000000000000002 ps. A displacement move (push, eigenvector push, push
off the saddle) moves the atoms by the vector times 0.9999999432296087, scaled
by s² where s = min(1, dmax / largest component), as one FIRE step from rest
does; a Lanczos probe is placed directly. The factor is where pARTn's
displacement lands: pARTn sends it to LAMMPS as a force, through its Ry, bohr
and atomic-time constants and LAMMPS's metal ``ftm2v``.

This plain core leaves out, on purpose for now:

- smoothing from the push to the eigenvector (``nsmooth``);
- perpendicular step limits out of the basin (pARTn's list built from
  ``nperp`` and ``nperp_limitation``); ``nperp`` here is the basin's limit
  only, which the strategy sets to pARTn's;
- new chances in a convex region (``nnewchance``, ``alpha_mix_cr``): a lost
  negative curvature ends the search;
- the alignment test that stops a perpendicular relaxation when the
  eigenvector jumps, and the box-explosion guard.

Other differences from pARTn:

- eV and Å throughout; pARTn's constants, which are in Ry and bohr, are
  converted once (ruling of 2026-11-23), and its unit round trip is kept only
  where it moves the atoms (the displacement factor and dt0 above);
- ``neigen`` is 1, as pARTn's setup forces it;
- the random numbers come from NumPy's generator seeded with ``seed``; the
  push draws on the pushed atoms only and the first Lanczos vector on the
  movable atoms, each in order of the atom's minimum-image offset from the
  central atom (x, then y, then z), so the draws do not depend on the atom
  order;
- a Lanczos probe sits at the base point plus ``lanczos_disp`` times the
  basis vector, and the force at the base point is reused after Lanczos
  instead of being evaluated again; the evaluation pARTn makes at the saddle
  before the second push off it is skipped as well;
- each FIRE restart costs no evaluation, where pARTn's costs one it discards;
- ``nevalf_max`` caps the callback calls, and the force-call counts are
  every callback call.

Coordinates: the search keeps its own unwrapped positions and passes them to
the callback as they are. Distances use the minimum image on the periodic axes
only, and returned positions are wrapped on the periodic axes only.

Frozen atoms: forces on frozen atoms are discarded and the random draws are
zero on them, so their positions never change. A ``first_lanczos_vector``
passed in with components on them moves them, as pARTn's first Lanczos vector
moves the active volume's buffer atoms.

Every dot product, norm and largest component over the whole system goes
through ``vectors`` (:class:`NumpyVectorSpace` on one rank), so a driver over
several ranks can supply its own reductions.

References
----------
M. Poberznik, M. Gunde, N. Salles, A. Jay, A. Hemeryck, N. Richard,
N. Mousseau and L. Martin-Samos, "pARTn: a plugin implementation of the
Activation Relaxation Technique nouveau that takes over the FIRE minimisation
algorithm", Comput. Phys. Commun. 295, 108961 (2024).

E. Cancès, F. Legoll, M.-C. Marinica, K. Minoukadeh and F. Willaime, "Some
improvements of the activation-relaxation technique method for finding
transition pathways on potential energy surfaces", J. Chem. Phys. 130, 114711
(2009).

"""

import math
from typing import Callable, Literal, Optional

import numpy as np

from .dimer import DimerResult, _minimum_image, _wrap

__all__ = ["NumpyVectorSpace", "artn_search"]

# pARTn's internal units, Rydberg, bohr and Rydberg atomic time
# (artn:src/h_artn_units.f90:66, :71, :76, :78).
_RY2EV = 13.605691930242388
_B2A = 0.529177210903
_AMU_RY = 911.44421386718750
_AU_PS = 4.83776865317143e-5
_HESSIAN = _RY2EV / _B2A**2  # 1 Ry/bohr² in eV/Å²

# pARTn's hard-coded constants, converted once.
# Curvature floor of the eigenvector step (block_pusheigen.f90:69), 0.48587 eV/Å².
_CURVATURE_FLOOR = 0.01 * _HESSIAN
# etot_diff_limit default (d_artn_params.f90:90), 1088.455 eV above the start.
_ENERGY_LIMIT = 80.0 * _RY2EV
# Factor of the push off the saddle (push_over_procedure.f90:31).
_PUSH_OVER_FACTOR = 0.8
# Lanczos stops growing its basis below this new-vector norm (lanczos.f90:257).
_LANCZOS_BREAKDOWN = 1.0e-12 * _HESSIAN

# LAMMPS FIRE as pARTn's fix sets it, metal units (fix_artn.cpp:116-130, LAMMPS
# min.cpp defaults for the keys the fix does not pass).
_FTM2V = 1.0 / 1.0364269e-4  # LAMMPS metal force * time / mass to velocity
# LAMMPS's metal time step, 0.001 ps, after pARTn's move_mode converts it to
# atomic time and back (convert_units.f90:390, move_mode.f90:72, :179).
_DT0 = (1.0e-3 * (1.0 / _AU_PS)) * _AU_PS
# Fraction of a displacement that lands: move_mode writes it as the force
# displ * Mass / dt0², in bohr and Ry atomic units, the fix multiplies by the
# atom mass and FIRE moves dt0² * ftm2v * f / m (INVENTORY.md, "Units").
# Measured on issue 02's recordings: 0.2999999830 Å for a 0.3 Å push.
_DISPLACEMENT_FACTOR = _AMU_RY * _AU_PS**2 * _RY2EV * _FTM2V / _B2A**2
_ALPHA0 = 0.1
_ALPHA_SHRINK = 0.99
_DT_GROW = 1.1
_DT_SHRINK = 0.5
_DT_MAX = 10.0 * _DT0
_DT_MIN = 0.02 * _DT0
_PERP_DELAY = 5  # delaystep in perpendicular relaxation
_RELAX_DELAY = 0  # delaystep in relaxation to a minimum

EnergyAndForces = Callable[[np.ndarray], tuple[float, np.ndarray]]
# (move, FIRE state): the move that led to the positions about to be evaluated,
# and FIRE's (dt, alpha, velocities) after a FIRE step, else None.
Observer = Callable[[str, Optional[tuple[float, float, np.ndarray]]], None]


class NumpyVectorSpace:
    """Reductions over whole-system (N, 3) arrays held on one rank."""

    def dot(self, a: np.ndarray, b: np.ndarray) -> float:
        """Return the 3N dot product of ``a`` and ``b``."""
        return float(np.vdot(a, b))

    def norm(self, a: np.ndarray) -> float:
        """Return the 3N norm of ``a``."""
        return math.sqrt(self.dot(a, a))

    def max_abs(self, a: np.ndarray) -> float:
        """Return the largest absolute component of ``a``, 0 for an empty array."""
        return float(np.max(np.abs(a))) if a.size else 0.0


class _Failed(Exception):
    """The search stops with the reason given."""


def artn_search(
    positions: np.ndarray,
    cell: np.ndarray,
    pbc: np.ndarray,
    movable: np.ndarray,
    central_atom: int,
    seed: int,
    energy_and_forces: EnergyAndForces,
    *,
    masses: Optional[np.ndarray] = None,
    push_mode: Literal["list", "rad"] = "rad",
    push_dist_thr: float = 1.0,
    push_step_size: float = 0.4,
    ninit: int = 2,
    lanczos_min_size: int = 10,
    lanczos_max_size: int = 20,
    lanczos_disp: float = 0.0005,
    lanczos_eval_conv_thr: float = 0.001,
    eigval_thr: float = -0.01,
    eigen_step_size: float = 0.2,
    nperp: Optional[int] = 3,
    forc_thr: float = 0.001,
    convergence_property: Literal["maxval", "norm"] = "maxval",
    push_over: float = 1.0,
    nevalf_max: int = 9999,
    dmax: float = 6.0,
    vectors: Optional[NumpyVectorSpace] = None,
    initial_push: Optional[np.ndarray] = None,
    first_lanczos_vector: Optional[np.ndarray] = None,
    observe: Optional[Observer] = None,
) -> DimerResult:
    """Find a saddle point and the two minima it connects.

    The keyword parameters from ``push_mode`` to ``dmax`` are the
    ``[pARTn]`` parameters of the same names, with pyKMC's defaults, in eV and
    Å. A failed search returns a result with ``success`` False and a reason;
    it never raises. An exception raised by the callback is not caught.

    Parameters
    ----------
    positions : np.ndarray
        Starting positions, (N, 3), Å, normally a minimum.
    cell : np.ndarray
        Cell vectors as rows, (3, 3), Å.
    pbc : np.ndarray
        Periodicity of each cell axis, (3,) booleans.
    movable : np.ndarray
        Movable atoms, (N,) booleans. Other atoms never move.
    central_atom : int
        Atom the initial push is centred on (pARTn's ``push_ids``).
    seed : int
        Seed of the initial push and of the first Lanczos vector.
    energy_and_forces : Callable[[np.ndarray], tuple[float, np.ndarray]]
        Takes (N, 3) positions in Å and returns the energy in eV and the
        (N, 3) forces in eV/Å.
    masses : Optional[np.ndarray]
        Atom masses, (N,), amu, for FIRE. Default 1 amu each.
    push_mode : {"rad", "list"}
        Push the movable atoms within ``push_dist_thr`` (minimum image) of the
        central atom, or the central atom only.
    push_dist_thr : float
        Radius of the ``rad`` push, Å.
    push_step_size : float
        3N norm of the push vector, Å.
    ninit : int
        Pushes before the first Lanczos; 0 starts with Lanczos.
    lanczos_min_size, lanczos_max_size : int
        Lanczos convergence floor and most force calls per Lanczos run. As in
        pARTn, the first call can converge only from ``lanczos_min_size``
        calls and later calls from ``lanczos_min_size - 1``. The most is
        capped at three times the movable atoms.
    lanczos_disp : float
        Lanczos finite-difference step, Å.
    lanczos_eval_conv_thr : float
        Relative change of the lowest eigenvalue at which Lanczos stops.
    eigval_thr : float
        The search leaves the basin when the lowest eigenvalue is below this,
        eV/Å².
    eigen_step_size : float
        Largest push along the eigenvector, Å; it also sizes the push off the
        saddle.
    nperp : Optional[int]
        FIRE steps after each push in the basin; None or -1 for no limit (the
        basin then never ends before ``nevalf_max``). The `artn` strategy
        passes pARTn's basin limit, which is ``nperp_limitation[0]`` when
        ``[pARTn] nperp`` is unset or -1.
    forc_thr : float
        The saddle and both minima are converged when the force measure is
        below this, eV/Å.
    convergence_property : {"maxval", "norm"}
        Force measure: the largest absolute component, or the 3N norm. It is
        also the measure that compares the perpendicular and parallel forces.
    push_over : float
        Factor on the push off the saddle.
    nevalf_max : int
        Most callback calls; one more fails the search.
    dmax : float
        FIRE's ``dmax``, Å: the step time times the largest velocity
        component, and the largest component of a displacement move, stay
        within it. Raised to ``push_step_size`` if smaller, as pARTn's fix
        does.
    vectors : Optional[NumpyVectorSpace]
        Whole-system reductions; one rank by default.
    initial_push : Optional[np.ndarray]
        (N, 3) values that replace the random draw of the initial push, which
        is then cut to the pushed atoms and scaled to ``push_step_size`` as
        the draw is. For step replay of a recorded pARTn search.
    first_lanczos_vector : Optional[np.ndarray]
        (N, 3) vector that replaces the random draw of the first Lanczos
        vector, used as given. For step replay.
    observe : Optional[Callable[[str, Optional[tuple]], None]]
        Called before each callback call with the move that led to the
        positions (``"start"``, ``"init"``, ``"perp"``, ``"lanczos"``,
        ``"eigen"``, ``"over"`` or ``"relax"``) and, after a FIRE step,
        FIRE's ``(dt, alpha, velocities)``, else None. For step replay.

    Returns
    -------
    DimerResult
        The saddle, the two minima and the search's counters, or a failure
        with its reason: ``"empty_push"`` (no movable atom to push),
        ``"eigenvalue_lost"`` (positive lowest eigenvalue out of the basin),
        ``"max_force_calls"``, ``"energy_limit"`` (1088.455 eV above the
        start, pARTn's default) or ``"non_finite_force"``. As in pARTn,
        minimum 1 is reached by pushing off the saddle along the last Lanczos
        eigenvector as it was last oriented, minimum 2 the other way, so
        either may be the start. The mode is that eigenvector, pointed from
        the start to the saddle, and the curvature its eigenvalue; neither is
        recomputed at the saddle.

    Raises
    ------
    ValueError
        If an argument is malformed.

    """
    x_start = np.array(positions, dtype=float)
    n_atoms = len(x_start)
    cell = np.asarray(cell, dtype=float)
    pbc = np.asarray(pbc, dtype=bool)
    mask = np.asarray(movable, dtype=bool)
    if x_start.shape != (n_atoms, 3) or n_atoms == 0:
        raise ValueError("positions must be a non-empty (N, 3) array")
    if cell.shape != (3, 3) or pbc.shape != (3,) or mask.shape != (n_atoms,):
        raise ValueError("cell must be (3, 3), pbc (3,) and movable (N,)")
    if not 0 <= central_atom < n_atoms:
        raise ValueError("central_atom is not an atom index")
    if push_mode not in ("list", "rad"):
        raise ValueError(f"unknown push_mode {push_mode!r}")
    if convergence_property not in ("maxval", "norm"):
        raise ValueError(f"unknown convergence_property {convergence_property!r}")
    if push_step_size <= 0 or lanczos_disp <= 0 or eigen_step_size <= 0:
        raise ValueError("push_step_size, lanczos_disp and eigen_step_size must be > 0")
    masses = np.ones(n_atoms) if masses is None else np.asarray(masses, dtype=float)
    if masses.shape != (n_atoms,) or np.any(masses <= 0):
        raise ValueError("masses must be (N,) and positive")
    for given in (initial_push, first_lanczos_vector):
        if given is not None and np.shape(given) != (n_atoms, 3):
            raise ValueError("initial_push and first_lanczos_vector must be (N, 3)")

    vec = NumpyVectorSpace() if vectors is None else vectors
    weight = mask[:, None].astype(float)
    inv_mass = 1.0 / masses[:, None]
    dmax = max(dmax, push_step_size)
    basin_perp_steps = -1 if nperp is None else nperp
    nlanc = min(lanczos_max_size, 3 * int(mask.sum()))
    nlanc_min = min(lanczos_min_size, nlanc - 1)
    calls = 0
    e_start: Optional[float] = None

    def evaluate(
        x: np.ndarray,
        move: str,
        state: Optional[tuple[float, float, np.ndarray]] = None,
    ) -> tuple[float, np.ndarray]:
        nonlocal calls
        if calls >= nevalf_max:
            raise _Failed("max_force_calls")
        calls += 1
        if observe is not None:
            observe(move, state)
        energy, forces = energy_and_forces(x.copy())
        forces = np.asarray(forces, dtype=float)
        if forces.shape != (n_atoms, 3):
            raise ValueError("energy_and_forces must return (N, 3) forces")
        if not (np.isfinite(energy) and np.all(np.isfinite(forces))):
            raise _Failed("non_finite_force")
        if e_start is not None and energy - e_start > _ENERGY_LIMIT:
            raise _Failed("energy_limit")
        return float(energy), forces * weight

    def measure(f: np.ndarray) -> float:
        return vec.max_abs(f) if convergence_property == "maxval" else vec.norm(f)

    def along(a: np.ndarray, d: np.ndarray) -> np.ndarray:
        """Return the part of ``a`` along ``d``."""
        return d * (vec.dot(a, d) / vec.dot(d, d))

    def displace(x: np.ndarray, d: np.ndarray) -> np.ndarray:
        """Move by ``d`` as pARTn does, one FIRE step from rest: s² from dmax."""
        d = _DISPLACEMENT_FACTOR * d
        largest = vec.max_abs(d)
        s = min(1.0, dmax / largest) if largest > 0 else 1.0
        return x + s * s * d

    def fire(
        x: np.ndarray,
        energy: float,
        forces: np.ndarray,
        direction: Optional[np.ndarray],
        delaystep: int,
        stop: Callable[[np.ndarray, int], bool],
        move: str,
    ) -> tuple[np.ndarray, float, np.ndarray]:
        """Relax from rest with LAMMPS FIRE until ``stop(forces, steps)``.

        With ``direction`` the force and the velocity are kept perpendicular
        to it. ``stop`` is asked before each step, with the full force and the
        steps taken.
        """
        v = np.zeros_like(x)
        alpha, dt, last_negative, steps = _ALPHA0, _DT0, 0, 0
        while not stop(forces, steps):
            f = forces if direction is None else forces - along(forces, direction)
            if direction is not None and steps > 0:
                v = v - along(v, direction)
            steps += 1
            power = vec.dot(v, f)
            if power > 0:
                f_sq = vec.dot(f, f)
                mix_v = 1.0 - alpha
                mix_f = alpha * math.sqrt(vec.dot(v, v) / f_sq) if f_sq > 1e-20 else 0.0
                if steps - last_negative > delaystep:
                    dt = min(_DT_GROW * dt, _DT_MAX)
                    alpha *= _ALPHA_SHRINK
                kick, probe = f, v
            else:
                last_negative = steps
                if steps >= delaystep:
                    alpha = _ALPHA0
                    if _DT_SHRINK * dt >= _DT_MIN:
                        dt *= _DT_SHRINK
                x = x - 0.5 * dt * v
                v = np.zeros_like(x)
                # pARTn's fix replaces the force at the half-stepped positions
                # by the earlier one, scaled (fix_artn.cpp:688, :693).
                ratio = _DT0 / dt
                kick = f * ratio * ratio
                probe = dt * _FTM2V * kick * inv_mass
            largest = vec.max_abs(probe)
            dtv = dmax / largest if dt * largest > dmax else dt
            v = v + dtv * _FTM2V * kick * inv_mass
            if power > 0:
                v = mix_v * v + mix_f * f
            x = x + dtv * v
            energy, forces = evaluate(x, move, (dt, alpha, v.copy()))
        return x, energy, forces

    lowest_previous = 0.0  # pARTn carries the eigenvalue from one run to the next

    def lanczos(
        x0: np.ndarray, f0: np.ndarray, v_in: np.ndarray, push_dir: np.ndarray
    ) -> tuple[float, np.ndarray]:
        """Return the lowest eigenvalue at ``x0`` and its eigenvector."""
        nonlocal lowest_previous
        basis = [v_in / vec.norm(v_in)]
        alphas: list[float] = []
        betas: list[float] = []
        eigvec = basis[0]
        lowest = lowest_previous
        for k in range(1, nlanc + 1):
            _, f = evaluate(x0 + lanczos_disp * basis[k - 1], "lanczos")
            q = -(f - f0) / lanczos_disp
            alpha = vec.dot(basis[k - 1], q)
            alphas.append(alpha)
            lowest_old = lowest
            if k == 1:
                lowest = alpha
                floor = nlanc_min
            else:
                h = np.diag(alphas) + np.diag(betas, 1) + np.diag(betas, -1)
                values, vectors_h = np.linalg.eigh(h)
                lowest = float(values[0])
                eigvec = sum(c * b for c, b in zip(vectors_h[:, 0], basis, strict=True))
                if vec.dot(eigvec, push_dir) < 0:
                    eigvec = -eigvec
                floor = nlanc_min - 1
            converged = (
                k >= floor
                and lowest_old != 0.0
                and abs((lowest - lowest_old) / lowest_old) <= lanczos_eval_conv_thr
            )
            # Point it against the force at this probe (block_lanczos.f90:79).
            eigvec = -math.copysign(1.0, vec.dot(f, eigvec)) * eigvec
            if converged or k == nlanc:
                break
            v = q - alpha * basis[k - 1]
            if k > 1:
                v = v - betas[-1] * basis[k - 2]
                for b in basis[: k - 1]:
                    v = v - vec.dot(v, b) * b
            v_norm = vec.norm(v)
            if v_norm < _LANCZOS_BREAKDOWN:
                break
            basis.append(v / v_norm)
            betas.append(v_norm if k == 1 else vec.dot(q, basis[k]))
        lowest_previous = lowest
        return lowest, eigvec

    def result(reason: Optional[str], **found: object) -> DimerResult:
        fields: dict = dict.fromkeys(DimerResult._fields)
        fields.update(
            success=reason is None,
            reason=reason,
            e_start=e_start,
            force_calls_saddle=calls,
            force_calls_total=calls,
            seed=seed,
        )
        fields.update(found)
        return DimerResult(**fields)

    try:
        e_start, f = evaluate(x_start, "start")
    except _Failed as failure:
        return result(str(failure))

    rng = np.random.default_rng(seed)
    offsets = _minimum_image(x_start - x_start[central_atom], cell, pbc)
    if push_mode == "list":
        pushed = np.zeros(n_atoms, dtype=bool)
        pushed[central_atom] = True
    else:
        pushed = np.linalg.norm(offsets, axis=1) <= push_dist_thr
    pushed &= mask
    if not pushed.any():
        return result("empty_push")
    # Draws go to the atoms in order of their offset from the central atom
    # (x, then y, then z, to 1e-6 Å), not in atom order.
    key = np.round(offsets, 6)
    by_offset = np.lexsort((key[:, 2], key[:, 1], key[:, 0]))
    order = by_offset[pushed[by_offset]]
    push = np.zeros((n_atoms, 3))
    push[order] = 0.5 - rng.random((len(order), 3))
    if initial_push is not None:
        push = np.where(pushed[:, None], np.asarray(initial_push, dtype=float), 0.0)
    push *= push_step_size / vec.norm(push)
    order = by_offset[mask[by_offset]]
    eigvec = np.zeros((n_atoms, 3))
    eigvec[order] = rng.random((len(order), 3)) - 0.5
    if first_lanczos_vector is not None:
        eigvec = np.array(first_lanczos_vector, dtype=float)

    def basin_done(forces: np.ndarray, steps: int) -> bool:
        return basin_perp_steps >= 0 and steps >= basin_perp_steps

    at_saddle = False

    def climb_done(forces: np.ndarray, steps: int) -> bool:
        nonlocal at_saddle
        if measure(forces) < forc_thr:
            at_saddle = True
            return True
        parallel = along(forces, push)
        return measure(forces - parallel) < measure(parallel)

    def relaxed(forces: np.ndarray, steps: int) -> bool:
        return steps > 0 and measure(forces) < forc_thr

    found: dict = {}
    try:
        x, energy = x_start.copy(), e_start
        in_basin, pushes = True, 0
        # With ninit = 0 the first Lanczos runs at the start (block_pushinit.f90:35).
        skip_push = ninit == 0
        while not at_saddle:
            if in_basin and not skip_push:
                x = displace(x, push)
                energy, f = evaluate(x, "init")
                pushes += 1
                x, energy, f = fire(x, energy, f, push, _PERP_DELAY, basin_done, "perp")
                if pushes < ninit:
                    continue
            skip_push = False
            lowest, eigvec = lanczos(x, f, eigvec, push)
            if lowest < eigval_thr or (not in_basin and lowest < 0):
                in_basin = False
            elif not in_basin and lowest > 0:
                raise _Failed("eigenvalue_lost")
            else:
                in_basin = True
                continue
            push = eigvec
            f_para = vec.dot(f, push)
            step = -math.copysign(1.0, f_para) * min(
                eigen_step_size, abs(f_para) / max(abs(lowest), _CURVATURE_FLOOR)
            )
            x = displace(x, push * step)
            energy, f = evaluate(x, "eigen")
            x, energy, f = fire(x, energy, f, push, _PERP_DELAY, climb_done, "perp")

        saddle, e_saddle = x, energy
        offset = _minimum_image(saddle - x_start, cell, pbc)
        mode = eigvec / vec.norm(eigvec)
        found = dict(
            saddle_positions=_wrap(saddle, cell, pbc),
            e_saddle=e_saddle,
            curvature=lowest,
            mode=-mode if vec.dot(mode, offset) < 0 else mode,
            force_calls_saddle=calls,
        )
        minima = []
        for sign in (1.0, -1.0):
            x = displace(
                saddle, sign * eigen_step_size * push_over * _PUSH_OVER_FACTOR * eigvec
            )
            energy, f = evaluate(x, "over")
            x, energy, f = fire(x, energy, f, None, _RELAX_DELAY, relaxed, "relax")
            minima.append((x, energy))
    except _Failed as failure:
        return result(str(failure), **found, force_calls_total=calls)

    (x_min1, e_min1), (x_min2, e_min2) = minima
    return result(
        None,
        **found,
        min1_positions=_wrap(x_min1, cell, pbc),
        min2_positions=_wrap(x_min2, cell, pbc),
        e_min1=e_min1,
        e_min2=e_min2,
        delr1=vec.norm(_minimum_image(x_min1 - x_start, cell, pbc)),
        delr2=vec.norm(_minimum_image(x_min2 - x_start, cell, pbc)),
        force_calls_total=calls,
    )
