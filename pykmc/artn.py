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
  ``lanczos_max_size``), then one call back at the point it started from;
- in the basin, another push and perpendicular relaxation until the lowest
  eigenvalue is below ``eigval_thr``; out of the basin, a push of
  ``min(eigen_step_size, |F.p| / max(|lambda|, 0.48587 eV/Å²))`` along p,
  uphill, where p is the eigenvector, or for the first ``nsmooth`` pushes of
  the search a blend that turns from the last basin push to the eigenvector
  of the first push out of the basin;
- FIRE relaxation perpendicular to p, with the velocity kept perpendicular to
  the eigenvector, until the largest force component (or the force's 3N norm,
  ``convergence_property = "norm"``) is below ``forc_thr``, which is the
  saddle; Lanczos runs again after the out-of-basin step limit (pARTn's list
  from ``nperp`` and ``nperp_limitation``, one entry per relaxation), once
  the perpendicular force is smaller than the parallel force (not while the
  pushes are blended), or when the eigenvector has turned (``|e.e_prev| <
  0.8``) and the path from the start is nearly perpendicular to p;
- a positive eigenvalue out of the basin gives up to ``nnewchance`` new
  pushes from the start's push, mixed with a random part (``alpha_mix_cr``),
  back in the basin; after that the search fails;
- a push off the saddle along +eigenvector, of ``eigen_step_size * push_over
  * 0.8``, and FIRE relaxation to the first minimum on the full force. The
  search then returns to the saddle with zero velocity and keeps FIRE's time
  step. It takes one step with the first minimum's force before the opposite
  push and the relaxation to the second minimum.

FIRE is LAMMPS's (euler implicit) with the settings pARTn's fix gives it, in
LAMMPS metal units: time step 0.001 ps, masses in amu, ``alpha0`` 0.1,
``alphashrink`` 0.99, ``dtgrow`` 1.1, ``dtshrink`` 0.5, ``tmax`` 10, ``tmin``
0.02, half step back on, ``delaystep`` 5 for perpendicular relaxation and 0
for relaxation to a minimum, and the step time cut so that it times the
largest velocity component before the kick stays within ``dmax``. On a
restart FIRE reuses the earlier force scaled by (dt0/dt)², as pARTn's fix
does, instead of evaluating at the half-stepped positions. The time step dt0
is LAMMPS's 0.001 ps after pARTn's round trip through its time unit,
0.0010000000000000002 ps. A displacement move (push, Lanczos probe and the
move back, eigenvector push, push off the saddle) moves the atoms by the vector
times 0.9999999432296087, scaled by s² where s = min(1, dmax / largest
component), as one FIRE step from rest does. The factor is where pARTn's
displacement lands: pARTn sends it to LAMMPS as a force, through its Ry, bohr
and atomic-time constants and LAMMPS's metal ``ftm2v``.

pARTn holds the push in bohr. The blend mixes that bohr vector, of norm
``push_step_size`` / 0.529177 Å, with the unit eigenvector, and the alignment
test compares the bohr components of the blend; the core does both on the push
in bohr, so its path does not depend on the unit (finding of issue 06).

The alignment test takes the path from the start in the positions LAMMPS
holds, with no minimum image, as pARTn does. LAMMPS wraps atoms into the cell
only when it rebuilds its neighbour list: in pARTn's minimize, when an atom has
moved more than half the 2.0 Å skin since the last rebuild. After that, a
wrapped atom adds a cell length to the path (finding of issue 06). The core
applies the same rule to its own positions at each evaluation, unless the
callback reports the engine's positions. LAMMPS also checks at the evaluations
pARTn discards, which the core skips; on the eight recordings of issue 02 this
changes no step.

This core leaves out, on purpose for now, the box-explosion guard.

Other differences from pARTn:

- eV and Å throughout; pARTn's constants, which are in Ry and bohr, are
  converted once (ruling of 2026-11-23), and its units are kept only where
  it moves the atoms (the displacement factor and dt0 above) and inside a
  Lanczos block, whose force differences are in Ry/bohr (ruling of
  2026-11-24);
- ``neigen`` is 1, as pARTn's setup forces it;
- the random numbers come from NumPy's generator seeded with ``seed``; the
  push and each new chance draw on the pushed atoms only and the first
  Lanczos vector on the movable atoms, each in order of the atom's
  minimum-image offset from the central atom (x, then y, then z), so the draws
  do not depend on the atom order;
- each FIRE restart costs no evaluation, where pARTn's costs one it discards;
- the force-call counts are every callback call; as in pARTn, call number
  ``nevalf_max`` fails the search;

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
from typing import Callable, Literal, Optional, Sequence, cast

import numpy as np

from .dimer import DimerResult, _minimum_image, _wrap

__all__ = ["NumpyVectorSpace", "artn_search", "perp_limits"]

# pARTn's internal units, Rydberg, bohr and Rydberg atomic time
# (artn:src/h_artn_units.f90:66, :71, :76, :78).
_RY2EV = 13.605691930242388
_B2A = 0.529177210903
_AMU_RY = 911.44421386718750
_AU_PS = 4.83776865317143e-5
_HESSIAN = _RY2EV / _B2A**2  # 1 Ry/bohr² in eV/Å²
# pARTn's conversions from LAMMPS metal units (convert_units.f90:386-412).
# Lanczos takes its force differences in Ry/bohr and its eigenvalue in
# Ry/bohr², as pARTn does: in eV/Å the subtraction's round-off grows past
# 1e-10 Å over some climbs (finding of issue 06, ruling of 2026-11-24).
_F2AU = (1.0 / _RY2EV) / (1.0 / _B2A)
_L2AU = 1.0 / _B2A
_AU2H = 1.0 / (_F2AU / _L2AU)

# pARTn's hard-coded constants, converted once.
# Curvature floor of the eigenvector step (block_pusheigen.f90:69), 0.48587 eV/Å².
_CURVATURE_FLOOR = 0.01 * _HESSIAN
# etot_diff_limit default (d_artn_params.f90:90), 1088.455 eV above the start.
_ENERGY_LIMIT = 80.0 * _RY2EV
# Factor of the push off the saddle (push_over_procedure.f90:31).
_PUSH_OVER_FACTOR = 0.8
# Lanczos stops growing its basis below this new-vector norm, in Ry/bohr²
# (lanczos.f90:257).
_LANCZOS_BREAKDOWN = 1.0e-12
# Alignment test of an out-of-basin perpendicular relaxation
# (check_force_convergence.f90:127): the eigenvector has turned, |e.e_prev| <
# 0.8, and the path from the start is nearly perpendicular to the push.
_ALIGNMENT_TURNED = 0.8
_ALIGNMENT_PERPENDICULAR = 0.1
# The test takes the path from the start in the positions LAMMPS holds, with no
# minimum image. LAMMPS wraps atoms into the cell only when it rebuilds its
# neighbour list, which pARTn's minimize does when an atom has moved more than
# half the skin since the last rebuild (neigh_modify every 1 delay 0 check yes;
# LAMMPS Neighbor::decide, Domain::pbc). pyKMC sets no skin for pARTn, so it is
# LAMMPS's default for metal units, 2.0 Å.
_PARTN_SKIN = 2.0
# pARTn's default perpendicular step limits (d_artn_params.f90:92).
_DEFAULT_PERP_LIMITS = (4, 8, 12, 16, -1)

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

# Positions in, energy and forces out, and optionally the positions as the
# engine holds them (for the alignment test).
EnergyAndForces = Callable[[np.ndarray], tuple]
# (move, state): the move that led to the positions about to be evaluated, and
# FIRE's (dt, alpha, velocities) after a FIRE step, Lanczos's (eigenvalue,
# eigenvector) after a Lanczos move, or ("SMTH",) / ("EIGN",) after a climb push.
Observer = Callable[[str, Optional[tuple]], None]


class DrawOrder:
    """Map each atom's place in the serial draw order to its row on this rank."""

    def __init__(self, rows: np.ndarray, ordinals: np.ndarray, total: int) -> None:
        self.rows = rows
        self.ordinals = ordinals
        self.total = total

    def draw(self, rng: np.random.Generator, size: int, *, push: bool) -> np.ndarray:
        """Consume the same PCG64 draws in the same order on every rank."""
        out = np.zeros((size, 3))
        stream = cast(np.random.PCG64, rng.bit_generator)
        # Skip other ranks' draws by advancing the stream, without generating
        # them. Each run of consecutive ordinals gets NumPy's serial values.
        cursor = 0
        for group in np.split(
            np.arange(len(self.rows)), np.flatnonzero(np.diff(self.ordinals) != 1) + 1
        ):
            if not len(group):
                continue
            start = int(self.ordinals[group[0]])
            if start != cursor:
                stream.advance(3 * (start - cursor))
            values = rng.random((len(group), 3))
            out[self.rows[group]] = 0.5 - values if push else values - 0.5
            cursor = start + len(group)
        if cursor != self.total:
            stream.advance(3 * (self.total - cursor))
        return out


def _draw_rows(
    x: np.ndarray,
    central: int,
    cell: np.ndarray,
    pbc: np.ndarray,
    mask: np.ndarray,
    push_mode: str,
    radius: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Return the rows of the initial push and Lanczos draws in serial order."""
    offsets = _minimum_image(x - x[central], cell, pbc)
    pushed = (
        np.arange(len(x)) == central
        if push_mode == "list"
        else np.linalg.norm(offsets, axis=1) <= radius
    )
    key = np.round(offsets, 6)
    order = np.lexsort((key[:, 2], key[:, 1], key[:, 0]))
    return order[(pushed & mask)[order]], order[mask[order]]


class NumpyVectorSpace:
    """Reductions over whole-system (N, 3) arrays held on one rank."""

    def atom_count(self, local: int) -> int:
        """Return the number of atoms in the whole system."""
        return local

    def count(self, local: int) -> int:
        """Sum a count over all ranks."""
        return local

    def all_true(self, local: bool) -> bool:
        """Return whether a condition holds on every rank."""
        return local

    def draw_orders(
        self,
        x: np.ndarray,
        central: int,
        cell: np.ndarray,
        pbc: np.ndarray,
        mask: np.ndarray,
        push_mode: str,
        radius: float,
    ) -> tuple[DrawOrder, DrawOrder]:
        """Return the push and Lanczos draw orders in serial geometric order."""
        pushed, movable = _draw_rows(x, central, cell, pbc, mask, push_mode, radius)
        return (
            DrawOrder(pushed, np.arange(len(pushed)), len(pushed)),
            DrawOrder(movable, np.arange(len(movable)), len(movable)),
        )

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


def _lammps_images(
    cell: np.ndarray,
    pbc: np.ndarray,
    vec: NumpyVectorSpace,
    skin: float = _PARTN_SKIN,
) -> Callable[[np.ndarray], np.ndarray]:
    """Return a function giving the positions as LAMMPS holds them under pARTn.

    Called once per force evaluation, in order. The first call, and any call
    where an atom has moved more than ``skin / 2`` since the last rebuild,
    wraps every atom into the cell on the periodic axes; between rebuilds the
    atoms keep that image, even outside the cell. On several ranks each rank
    holds its own rows and the rebuild test is taken over all of them.
    """
    trigger = (skin / 2) ** 2
    rebuilt_at: Optional[np.ndarray] = None
    shift = np.zeros(0)

    def held(x: np.ndarray) -> np.ndarray:
        nonlocal rebuilt_at, shift
        if (
            rebuilt_at is None
            or vec.max_abs(np.sum((x - rebuilt_at) ** 2, axis=1)) > trigger
        ):
            rebuilt_at = x.copy()
            shift = _wrap(x, cell, pbc) - x
        return x + shift

    return held


def _wrap_result_positions(
    positions: np.ndarray, cell: np.ndarray, pbc: np.ndarray
) -> np.ndarray:
    """Return ARTn endpoint coordinates wrapped into the periodic catalogue box."""
    result = _wrap(positions, cell, pbc)
    lengths = np.diag(cell)
    if np.array_equal(cell, np.diag(lengths)) and np.all(lengths > 0):
        for axis in np.flatnonzero(pbc):
            length = lengths[axis]
            invalid = (result[:, axis] < 0) | (result[:, axis] >= length)
            if invalid.any():
                # Adding L to a tiny negative coordinate can round to L itself.
                # Take the image from the input coordinate instead, so a point
                # just below L stays there even if the reciprocal rounded it up.
                image = np.remainder(positions[invalid, axis], length)
                result[invalid, axis] = np.where(image == length, 0.0, image)
    return result


def perp_limits(
    nperp: Optional[int], nperp_limitation: Optional[Sequence[int]]
) -> list[int]:
    """Return pARTn's list of perpendicular step limits, -1 for none.

    The first entry is the basin's limit; out of the basin the n-th
    relaxation takes entry n, the last entry repeating. Built as pyKMC's
    ``partn_search`` has pARTn build it, ``nperp`` set before
    ``nperp_limitation`` (``nperp_limitation_init``, nperp_limitation.f90:96),
    and with ``nperp_limitation`` None, as ``lnperp_limitation = False``.
    An empty or all -2 list selects pARTn's defaults. Mixing -2 with explicit
    limits raises ValueError.
    """
    nperp = -1 if nperp is None else nperp
    if nperp_limitation is None:
        return [nperp if nperp > -1 else -1, -1]
    given = list(nperp_limitation)
    if not given or all(value == -2 for value in given):
        limits = list(_DEFAULT_PERP_LIMITS)
        if nperp > -1:
            limits[0] = nperp
        return limits
    if -2 in given:
        raise ValueError("nperp_limitation cannot mix -2 with other values")
    n = sum(value > -2 for value in given)
    return ([nperp] if nperp != -1 else []) + given[:n] + [given[n - 1]]


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
    nsmooth: int = 3,
    alpha_mix_cr: float = 0.2,
    nnewchance: int = 0,
    nperp: Optional[int] = 3,
    nperp_limitation: Optional[Sequence[int]] = (4, 8, 12, 16, -1),
    forc_thr: float = 0.001,
    convergence_property: Literal["maxval", "norm"] = "maxval",
    push_over: float = 1.0,
    nevalf_max: int = 9999,
    dmax: float = 6.0,
    vectors: Optional[NumpyVectorSpace] = None,
    initial_push: Optional[np.ndarray] = None,
    first_lanczos_vector: Optional[np.ndarray] = None,
    new_chance_pushes: Optional[Sequence[np.ndarray]] = None,
    recorded_draws_only: bool = False,
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
        Movable atoms, (N,) booleans. Forces and random draws are zero on
        frozen atoms. An injected ``first_lanczos_vector`` can move them.
    central_atom : int
        Atom the initial push is centred on (pARTn's ``push_ids``).
    seed : int
        Seed of the initial push, the first Lanczos vector and the new chances.
    energy_and_forces : Callable[[np.ndarray], tuple]
        Takes (N, 3) positions in Å and returns the energy in eV and the
        (N, 3) forces in eV/Å, and optionally, third, the (N, 3) positions as
        the engine holds them, which the alignment test then uses in place of
        the positions LAMMPS would hold under pARTn.
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
        Lanczos convergence floor and most probes per Lanczos run; one more
        call goes back to where the run started. As in pARTn, a run can stop
        at its first probe only if ``lanczos_min_size`` is 1, and at probe
        k > 1 once k >= ``lanczos_min_size - 1``. The most is capped at three
        times the movable atoms.
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
    nsmooth : int
        Pushes out of the basin, from the search's first, that blend the last
        basin push into the eigenvector, weight k / (nsmooth + 1) on the
        eigenvector at the k-th; 0 for none.
    alpha_mix_cr : float
        Weight of the random part of a new chance's push.
    nnewchance : int
        New chances after a positive eigenvalue out of the basin; 0 for none.
    nperp, nperp_limitation : Optional[int], Optional[Sequence[int]]
        FIRE step limits, as pARTn builds them (:func:`perp_limits`): the
        basin's limit, ``nperp`` or else the first of ``nperp_limitation``,
        then out of the basin one per relaxation, the last repeating. -1 is no
        limit; with none in the basin, the basin never ends before
        ``nevalf_max``.
    forc_thr : float
        The saddle and both minima are converged when the force measure is
        below this, eV/Å.
    convergence_property : {"maxval", "norm"}
        Force measure: the largest absolute component, or the 3N norm. It is
        also the measure that compares the perpendicular and parallel forces.
    push_over : float
        Factor on the push off the saddle.
    nevalf_max : int
        Most callback calls; as in pARTn, the last fails the search.
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
        direction. The search uses its direction without the movable mask.
        Its components on frozen atoms move those atoms during probes. They
        also enter the eigenvector, so perpendicular relaxation and
        eigenvector pushes can move frozen atoms too. Step replay needs this
        to reproduce pARTn's buffer motion.
    new_chance_pushes : Optional[Sequence[np.ndarray]]
        (N, 3) pushes, Å, that replace the pushes the new chances build, in
        order, used as given. For step replay.
    recorded_draws_only : bool
        Require both initial draws and end the search with recorded_draws_exhausted
        if its path needs a new-chance push absent from the supplied sequence.
    observe : Optional[Callable[[str, Optional[tuple]], None]]
        Called before each callback call with the move that led to the
        positions (``"start"``, ``"init"``, ``"perp"``, ``"lanczos"``,
        ``"eigen"``, ``"over"`` or ``"relax"``) and, after a FIRE step,
        FIRE's ``(dt, alpha, velocities)``; after a Lanczos move (a probe or
        the move back), the lowest eigenvalue and the eigenvector as pARTn
        holds them after the call that made the move; after an eigenvector
        push, ("SMTH",) during blending or ("EIGN",) otherwise. Other moves
        carry None. For step replay.

    Returns
    -------
    DimerResult
        The saddle, the two minima and the search's counters, or a failure
        with its reason: ``"empty_push"`` (no movable atom to push),
        ``"eigenvalue_lost"`` (positive lowest eigenvalue out of the basin,
        no new chance left),
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
    vec = NumpyVectorSpace() if vectors is None else vectors
    n_atoms = len(x_start)
    full_size = vec.atom_count(n_atoms)
    cell = np.asarray(cell, dtype=float)
    pbc = np.asarray(pbc, dtype=bool)
    mask = np.asarray(movable, dtype=bool)
    if x_start.shape != (n_atoms, 3) or full_size == 0:
        raise ValueError("positions must be a non-empty (N, 3) array")
    if cell.shape != (3, 3) or pbc.shape != (3,) or mask.shape != (n_atoms,):
        raise ValueError("cell must be (3, 3), pbc (3,) and movable (N,)")
    if not 0 <= central_atom < full_size:
        raise ValueError("central_atom is not an atom index")
    if push_mode not in ("list", "rad"):
        raise ValueError(f"unknown push_mode {push_mode!r}")
    if convergence_property not in ("maxval", "norm"):
        raise ValueError(f"unknown convergence_property {convergence_property!r}")
    if push_step_size <= 0 or lanczos_disp <= 0 or eigen_step_size <= 0:
        raise ValueError("push_step_size, lanczos_disp and eigen_step_size must be > 0")
    masses = np.ones(n_atoms) if masses is None else np.asarray(masses, dtype=float)
    if not vec.all_true(masses.shape == (n_atoms,) and not np.any(masses <= 0)):
        raise ValueError("masses must be (N,) and positive")
    for given in (initial_push, first_lanczos_vector, *(new_chance_pushes or ())):
        if given is not None and np.shape(given) != (n_atoms, 3):
            raise ValueError(
                "initial_push, first_lanczos_vector and new_chance_pushes must be (N, 3)"
            )
    if recorded_draws_only and (initial_push is None or first_lanczos_vector is None):
        raise ValueError(
            "recorded_draws_only requires the initial push and Lanczos vector"
        )

    weight = mask[:, None].astype(float)
    inv_mass = 1.0 / masses[:, None]
    dmax = max(dmax, push_step_size)
    limits = perp_limits(nperp, nperp_limitation)
    nlanc = min(lanczos_max_size, 3 * vec.count(int(mask.sum())))
    nlanc_min = min(lanczos_min_size, nlanc - 1)
    calls = 0
    e_start: Optional[float] = None
    held = x_start  # the positions as the engine holds them, last call
    lammps_holds = _lammps_images(cell, pbc, vec)

    def evaluate(
        x: np.ndarray,
        move: str,
        state: Optional[tuple] = None,
        *,
        restored_energy: Optional[float] = None,
    ) -> tuple[float, np.ndarray]:
        nonlocal calls, held
        calls += 1
        if observe is not None:
            observe(move, state)
        energy, forces, *engine = energy_and_forces(x.copy())
        forces = np.asarray(forces, dtype=float)
        if not vec.all_true(forces.shape == (n_atoms, 3)):
            raise ValueError("energy_and_forces must return (N, 3) forces")
        held = np.asarray(engine[0], dtype=float) if engine else lammps_holds(x)
        if not vec.all_true(bool(np.isfinite(energy) and np.all(np.isfinite(forces)))):
            raise _Failed("non_finite_force")
        if restored_energy is not None:
            energy = restored_energy  # block_pushover restores etot_sad on the return
        if e_start is not None and energy - e_start > _ENERGY_LIMIT:
            raise _Failed("energy_limit")
        # pARTn's step guard: its call number nevalf_max fails, whatever the
        # call found (artn.f90:729, with fix_artn.cpp:536's max_eval - 1).
        if calls >= nevalf_max:
            raise _Failed("max_force_calls")
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

    def bounded_dt(dt: float, velocity: np.ndarray) -> float:
        largest = vec.max_abs(velocity)
        return dmax / largest if dt * largest > dmax else dt

    def restart_fire(
        x: np.ndarray, v: np.ndarray, f: np.ndarray, dt: float, shrink: bool
    ) -> tuple[np.ndarray, np.ndarray, float]:
        """Take FIRE's half-step back, then kick from rest using the saved force."""
        if shrink and _DT_SHRINK * dt >= _DT_MIN:
            dt *= _DT_SHRINK
        x = x - 0.5 * dt * v
        # The force at the half-step is discarded (fix_artn.cpp:688-693).
        ratio = _DT0 / dt
        kick = f * ratio * ratio
        dtv = bounded_dt(dt, dt * _FTM2V * kick * inv_mass)
        v = dtv * _FTM2V * kick * inv_mass
        return x + dtv * v, v, dt

    def fire(
        x: np.ndarray,
        energy: float,
        forces: np.ndarray,
        direction: Optional[np.ndarray],
        delaystep: int,
        stop: Callable[[np.ndarray, int], bool],
        move: str,
        velocity_direction: Optional[np.ndarray] = None,
    ) -> tuple[np.ndarray, float, np.ndarray, float]:
        """Relax from rest with LAMMPS FIRE until ``stop(forces, steps)``.

        With ``direction`` the force is kept perpendicular to it, and the
        velocity perpendicular to ``velocity_direction`` (default
        ``direction``), as pARTn's split_field and move_mode do. ``stop`` is
        asked before each step, with the full force and the steps taken.
        """
        if velocity_direction is None:
            velocity_direction = direction
        v = np.zeros_like(x)
        alpha, dt, last_negative, steps = _ALPHA0, _DT0, 0, 0
        while not stop(forces, steps):
            f = forces if direction is None else forces - along(forces, direction)
            if velocity_direction is not None and steps > 0:
                v = v - along(v, velocity_direction)
            steps += 1
            power = vec.dot(v, f)
            if power > 0:
                f_sq = vec.dot(f, f)
                mix_v = 1.0 - alpha
                mix_f = alpha * math.sqrt(vec.dot(v, v) / f_sq) if f_sq > 1e-20 else 0.0
                if steps - last_negative > delaystep:
                    dt = min(_DT_GROW * dt, _DT_MAX)
                    alpha *= _ALPHA_SHRINK
                dtv = bounded_dt(dt, v)
                v = mix_v * (v + dtv * _FTM2V * f * inv_mass) + mix_f * f
                x = x + dtv * v
            else:
                last_negative = steps
                if steps >= delaystep:
                    alpha = _ALPHA0
                x, v, dt = restart_fire(x, v, f, dt, shrink=steps >= delaystep)
            energy, forces = evaluate(x, move, (dt, alpha, v.copy()))
        return x, energy, forces, dt

    # pARTn carries the eigenvalue from one Lanczos block into the next block's
    # first convergence test. A search starts with pARTn's "unset" value,
    # HUGE(1.0_DP) (m_block_lanczos.f90:29, :116; h_artn_units.f90:48).
    unset = float(np.finfo(float).max)
    lowest_previous = unset  # Ry/bohr², as pARTn holds it

    def against(e: np.ndarray, f: np.ndarray) -> np.ndarray:
        """Point ``e`` against the force ``f`` (block_lanczos.f90:79)."""
        return -math.copysign(1.0, vec.dot(f, e)) * e

    def lanczos(
        x0: np.ndarray, f0: np.ndarray, v_in: np.ndarray, push_dir: np.ndarray
    ) -> tuple[float, np.ndarray, np.ndarray, np.ndarray]:
        """Run one Lanczos block at ``x0`` as pARTn does (lanczos.f90:35).

        One probe per callback call, each reached by a displacement move from
        the last, the first ``lanczos_disp`` along ``v_in``. Return the lowest
        eigenvalue, its eigenvector as last pointed against the force, the
        position the last move returns to (``x0`` up to round-off), which is
        not evaluated here, and the force at the last probe. The observer
        sees, at each probe, the eigenvalue and eigenvector as pARTn holds
        them after the call that moved there. Inside the block forces are in
        Ry/bohr and eigenvalues in Ry/bohr², as in pARTn; the eigenvalue
        returned and observed is in eV/Å².
        """
        nonlocal lowest_previous
        lowest = lowest_previous
        size = nlanc
        basis = [v_in / vec.norm(v_in)]
        alphas: list[float] = []
        betas: list[float] = []
        eigvec = against(v_in, f0)
        f0_au, disp_au = f0 * _F2AU, lanczos_disp * _L2AU
        x = displace(x0, lanczos_disp * basis[0])
        k = 0
        while True:
            k += 1
            shown = lowest if lowest == unset else lowest * _AU2H
            _, f = evaluate(x, "lanczos", (shown, eigvec.copy()))
            q = -(f * _F2AU - f0_au) / disp_au
            alpha = vec.dot(basis[k - 1], q)
            alphas.append(alpha)
            lowest_old = lowest
            if k == 1:
                # One eigenvalue, and the eigenvector is left as it was.
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
            if (
                k >= floor
                and lowest_old != 0.0
                and abs((lowest - lowest_old) / lowest_old) <= lanczos_eval_conv_thr
            ):
                size = k
            # pARTn's v1: the next basis vector, or zero to go back to x0.
            target = np.zeros_like(x0)
            if k < size:
                v = q - alpha * basis[k - 1]
                if k > 1:
                    v = v - betas[-1] * basis[k - 2]
                    for b in basis[: k - 1]:
                        v = v - vec.dot(v, b) * b
                v_norm = vec.norm(v)
                if v_norm >= _LANCZOS_BREAKDOWN or (k == 1 and v_norm > 0):
                    target = v / v_norm
                    basis.append(target)
                    betas.append(v_norm if k == 1 else vec.dot(q, target))
                else:
                    # pARTn moves along the small vector but stores a zero basis
                    # vector and a zero beta (lanczos.f90:257); at its first
                    # probe it has no test and divides by zero.
                    target = v
                    basis.append(np.zeros_like(x0))
                    betas.append(0.0)
            eigvec = against(eigvec, f)
            x = displace(x, lanczos_disp * (target - basis[k - 1]))
            if k >= size:
                lowest_previous = lowest
                return lowest * _AU2H, eigvec, x, f

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

    rng = None if recorded_draws_only else np.random.Generator(np.random.PCG64(seed))
    pushed, mobile = vec.draw_orders(
        x_start, central_atom, cell, pbc, mask, push_mode, push_dist_thr
    )
    if not pushed.total:
        return result("empty_push")
    push = np.zeros((n_atoms, 3)) if rng is None else pushed.draw(rng, n_atoms, push=True)
    if initial_push is not None:
        push.fill(0)
        push[pushed.rows] = np.asarray(initial_push, dtype=float)[pushed.rows]
    push *= push_step_size / vec.norm(push)
    eigvec = np.zeros((n_atoms, 3)) if rng is None else mobile.draw(rng, n_atoms, push=False)
    if first_lanczos_vector is not None:
        eigvec = np.array(first_lanczos_vector, dtype=float)

    # pARTn's search state (d_artn_params, m_block_lanczos): the perpendicular
    # step limit in force and its place in the list, the blend counter and
    # ends, the new chances taken, and the last block's eigenvector with its
    # overlap with the one before (a1).
    nperp_now, nperp_step = limits[0], 1
    ismooth, inewchance = 0, 0
    blend_from: Optional[np.ndarray] = None
    blend_to: Optional[np.ndarray] = None
    turned = 0.0
    old_lanczos_vec = eigvec / vec.norm(eigvec)
    push_initial = push.copy()
    chance_pushes = iter(new_chance_pushes or ())
    held_start = held

    def next_perp_limit() -> None:
        """Go to the next out-of-basin limit (nperp_limitation_step(1))."""
        nonlocal nperp_now, nperp_step
        if nperp_step == 1:
            limits[0] = nperp_now
        nperp_step += 1
        nperp_now = limits[min(nperp_step, len(limits)) - 1]

    def basin_done(forces: np.ndarray, steps: int) -> bool:
        return nperp_now > -1 and steps >= nperp_now

    at_saddle = False

    def climb_done(forces: np.ndarray, steps: int) -> bool:
        """Return True when pARTn ends an out-of-basin perpendicular relaxation.

        The saddle (C0), else the step limit (C2), the perpendicular force
        below the parallel one (C3, off while the pushes are blended) or the
        alignment test (C4) (check_force_convergence.f90:104-131).
        """
        nonlocal at_saddle
        if measure(forces) < forc_thr:
            at_saddle = True
            return True
        if nperp_now >= 0 and steps >= nperp_now:
            return True
        parallel = along(forces, push)
        if not (nsmooth > 0 and ismooth <= nsmooth) and measure(
            forces - parallel
        ) < measure(parallel):
            return True
        # C4: the path from the start in the engine's positions, no minimum
        # image, against pARTn's push in bohr.
        path = held - held_start
        length = vec.norm(path)
        return (
            turned < _ALIGNMENT_TURNED
            and length > 0
            and abs(vec.dot(path, push) / (length * _B2A)) < _ALIGNMENT_PERPENDICULAR
        )

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
                x, energy, f, _ = fire(
                    x, energy, f, push, _PERP_DELAY, basin_done, "perp"
                )
                if inewchance == 0:
                    nperp_now = limits[0]
                if pushes < ninit:
                    continue
            skip_push = False
            lowest, eigvec, x, f_probe = lanczos(x, f, eigvec, push)
            turned = abs(vec.dot(eigvec, old_lanczos_vec))
            old_lanczos_vec = eigvec
            # pARTn decides at the last probe, before the move back is
            # evaluated (block_lanczos.f90:121); a lost eigenvalue stops there.
            if lowest < eigval_thr or (not in_basin and lowest < 0):
                in_basin = False
            else:
                if not in_basin and lowest > 0:
                    if inewchance >= nnewchance:
                        raise _Failed("eigenvalue_lost")
                    # A new chance: the start's push with a random part,
                    # pointed against the force at the last probe, back in
                    # the basin (block_lanczos.f90:144-161).
                    nperp_step, nperp_now = 1, limits[0]
                    inewchance += 1
                    ismooth = 0
                    push = next(chance_pushes, None)
                    if push is None:
                        if rng is None:
                            raise _Failed("recorded_draws_exhausted")
                        draw = pushed.draw(rng, n_atoms, push=True)
                        draw *= push_initial
                        draw /= vec.norm(draw)
                        push = (
                            1.0 - alpha_mix_cr
                        ) * push_initial + alpha_mix_cr * push_step_size * draw
                        push = (
                            -math.copysign(1.0, vec.dot(f_probe, push))
                            * push
                            / vec.norm(push)
                            * push_step_size
                        )
                    else:
                        push = np.array(push, dtype=float)
                in_basin = True
                nperp_step = 1
            energy, f = evaluate(x, "lanczos", (lowest, eigvec.copy()))
            if in_basin:
                continue
            # The push out of the basin (block_pusheigen.f90:41-73): for the
            # first nsmooth of the search, the last push blended into the
            # eigenvector of the first (smooth_interpol.f90:57-70), both as
            # pARTn holds them, the push in bohr.
            ismooth += 1
            if nsmooth > 0 and ismooth <= nsmooth:
                block = "SMTH"
                if ismooth == 1 or blend_from is None:
                    blend_from, blend_to = push / _B2A, eigvec
                s = ismooth / (nsmooth + 1)
                climb = (1.0 - s) * blend_from - math.copysign(
                    1.0, vec.dot(f, blend_to)
                ) * s * blend_to
                if ismooth == nsmooth:
                    blend_from = blend_to = None
            else:
                block = "EIGN"
                climb = eigvec
            f_para = vec.dot(f, climb)
            step = -math.copysign(1.0, f_para) * min(
                eigen_step_size, abs(f_para) / max(abs(lowest), _CURVATURE_FLOOR)
            )
            x = displace(x, climb * step)
            push = climb * _B2A
            energy, f = evaluate(x, "eigen", (block,))
            x, energy, f, _ = fire(
                x, energy, f, push, _PERP_DELAY, climb_done, "perp", eigvec
            )
            if not at_saddle:
                next_perp_limit()

        saddle, e_saddle = x, energy
        offset = _minimum_image(saddle - x_start, cell, pbc)
        mode = eigvec / vec.norm(eigvec)
        found = dict(
            saddle_positions=_wrap_result_positions(saddle, cell, pbc),
            e_saddle=e_saddle,
            curvature=lowest,
            mode=-mode if vec.dot(mode, offset) < 0 else mode,
            force_calls_saddle=calls,
        )
        push = eigen_step_size * push_over * _PUSH_OVER_FACTOR * eigvec

        def downhill(
            base: np.ndarray, step: np.ndarray
        ) -> tuple[np.ndarray, float, np.ndarray, float]:
            x = displace(base, step)
            energy, f = evaluate(x, "over")
            return fire(x, energy, f, None, _RELAX_DELAY, relaxed, "relax")

        x_min1, e_min1, f_min1, dt = downhill(saddle, push)
        found.update(
            min1_positions=_wrap_result_positions(x_min1, cell, pbc),
            e_min1=e_min1,
            delr1=vec.norm(_minimum_image(x_min1 - x_start, cell, pbc)),
        )
        # ARTn restores tau_sad and zeroes v at min1 (artn.f90:661-680,
        # move_mode.f90:148-156). The fix keeps FIRE's dt at irelax == 0.
        # Its next step therefore starts at the saddle using min1's force.
        back, v, dt = restart_fire(
            saddle, np.zeros_like(saddle), f_min1, dt, shrink=True
        )
        evaluate(back, "relax", (dt, _ALPHA0, v), restored_energy=e_saddle)
        x_min2, e_min2, _, _ = downhill(back, -push)
    except _Failed as failure:
        return result(str(failure), **found, force_calls_total=calls)

    return result(
        None,
        **found,
        min2_positions=_wrap_result_positions(x_min2, cell, pbc),
        e_min2=e_min2,
        delr2=vec.norm(_minimum_image(x_min2 - x_start, cell, pbc)),
        force_calls_total=calls,
    )
