# The dimer method in this module is ported from SEAKMC_py
# (https://github.com/SEAKMC/SEAKMC_py_parallel, seakmc_p/spsearch/SPSearch.py,
# class Dimer, and its dynamic active volume from class SPSearch; source file
# header: author Tao Liang, Copyright 2021), which is distributed under the
# following licence:
#
# BSD 3-Clause License
#
# Copyright (c) 2026 Haixuan Xu and Tao Liang
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# 1. Redistributions of source code must retain the above copyright notice,
#    this list of conditions and the following disclaimer.
#
# 2. Redistributions in binary form must reproduce the above copyright
#    notice, this list of conditions and the following disclaimer in the
#    documentation and/or other materials provided with the distribution.
#
# 3. Neither the name of the copyright holder nor the names of its
#    contributors may be used to endorse or promote products derived from
#    this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
# "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
# LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR
# A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT
# HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
# SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT
# LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
# DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY
# THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
# (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
"""Dimer saddle point search on a plain energy and force callback.

The search starts from a minimum, climbs along the lowest curvature mode to a
first-order saddle point, then pushes off both sides of the saddle along that
mode and relaxes each side to a minimum. Energies and forces come only from a
callback, so the search needs no LAMMPS, pARTn or MPI and can be tested on an
analytic surface.

Ported from SEAKMC_py's dimer (Liang and Xu 2023), which follows Henkelman and
Jónsson (1999):

- one image is evaluated, at a distance ``separation`` from the dimer centre
  along the mode; the force on the second image is extrapolated from the
  centre force;
- each rotation is a trial rotation by pi/4 followed by a rotation to the
  analytic angle, with at most three rotations per translation step;
- the centre moves by conjugate-gradient translation on the effective force
  (the force with its component along the mode inverted, or only minus that
  component while the curvature is positive), with SEAKMC's trial step,
  secant step and step cap.

Changes from SEAKMC:

- the search is converged on the force at the dimer centre (largest component
  over the movable atoms by default, as pARTn's ``forc_thr``; the 3N norm, as
  SEAKMC, on request), then the mode is re-rotated and the curvature
  recomputed at the centre, and the saddle is valid only with negative
  curvature;
- the saddle is then valid only with no second unstable direction: Lanczos on
  finite-difference Hessian-vector products (step ``separation``), restricted
  to the allowed directions (movable atoms, translation removed as for the
  mode) orthogonal to the mode, estimates the second-lowest curvature; below
  -1e-2 eV/Å² the search fails as a higher-order saddle. It costs one force
  call per Lanczos iteration, at most 40, about 13 to 25 on a 255-atom Ni
  vacancy hop;
- a start with no effective force, such as an atom on a symmetric site, is
  stepped ``trial_step`` along the mode up to three times before the first
  translation; the start stays the reference for energies and distances;
- the saddle is the dimer centre, not an image;
- the starting direction is drawn from a seeded generator, either on the
  movable atoms within a radius of the central atom (``local``) or on all
  movable atoms (``region``, as SEAKMC);
- after a rotation the curvature along the new mode is taken from the fit
  through the two evaluated orientations (Heyden, Bell and Keil 2005) instead
  of the value measured before the rotation;
- the push-off is a fixed distance along the mode, and both sides are
  relaxed with FIRE through the same callback;
- SEAKMC's TransHorizon rule, rotation conjugate gradient (unreachable there),
  step decay, angle check and preloaded starts are not ported.

Dynamic active volume (DAV, Liang and Xu 2023), on by default (``dav``): while
the dimer climbs out of the basin, each rotation's new mode is multiplied atom
by atom by a weight between 0 and 1 and renormalised, so the mode gathers on
the atoms that move most. It acts after the rotation to the analytic angle
when the curvature measured before that rotation is positive, the last
translation's line search found negative curvature, no translation step has
yet used a negative curvature, and more than four translation steps are done.
With N_i the atom's part of the mode and x_i = ln(|N_i|^4 + e^-20), the weight
is a sigmoid of x_i rescaled so that the smallest x_i on the movable atoms maps
to -4 and the largest to 16; when the range of x_i is below 4 every weight is
one. Weights compound from one rotation to the next. Only the mode is reweighted: forces, steps and the atom set are
unchanged. This is SEAKMC's ``RescaleVN`` with its default settings. Changes
from SEAKMC: the weights come from the movable atoms only, since the frozen
atoms' zero mode would set the range; and DAV is off in the final rotation at
the converged centre. SEAKMC's other weightings (``RAS``, ``SUM``), its step
scale and its moving average are not ported.

Coordinates: the search keeps its own unwrapped positions and passes them to
the callback as they are. Distances use the minimum image on the periodic axes
only, and returned positions are wrapped on the periodic axes only.

Frozen atoms: the movable mask is authoritative. Forces on frozen atoms are
discarded, the mode is zero on them, and their positions never change.

Translation: with every atom movable and a translation-invariant energy, rigid
translation is a zero-curvature mode. On request (``remove_translation``) the
net translation is kept out of the mode, as SEAKMC's ``CenterVN`` does.

References
----------
T. Liang and H. Xu, "Saddle point search with dynamic active volume",
Comput. Mater. Sci. 228, 112354 (2023).

G. Henkelman and H. Jónsson, "A dimer method for finding saddle points on high
dimensional potential surfaces using only first derivatives", J. Chem. Phys.
111, 7010 (1999).

A. Heyden, A. T. Bell and F. J. Keil, "Efficient methods for finding
transition states in chemical reactions: Comparison of improved dimer method
and partitioned rational function optimization method", J. Chem. Phys. 123,
224101 (2005).

"""

import math
from typing import Callable, Literal, NamedTuple, Optional

import numpy as np

__all__ = ["DimerResult", "dimer_search"]

# Rotation stops when the rotational force is below this (SEAKMC FMin4Rot), eV/Å².
_ROTATION_CONVERGED = 1e-3
# Rotation stops after a rotation that started below this (SEAKMC FThres4Rot), eV/Å².
_ROTATION_THRESHOLD = 0.1
# Line-search rotations per translation step (SEAKMC NMax4Rot).
_MAX_ROTATIONS = 3
# An effective force with a smaller 3N norm gives no direction to move, eV/Å.
_ZERO_FORCE = 1e-10
# Displacements along the mode tried when the start is stationary.
_MAX_KICKS = 3

# Saddle order check at the converged centre.
# A second curvature below minus this is a second unstable direction, eV/Å².
_ORDER_TOLERANCE = 1e-2
# Hessian-vector products (one force call each) before the check gives up.
_ORDER_MAX_ITERATIONS = 40
# The lowest Ritz value counts as converged once successive values agree to
# this fraction of its size, or to _ORDER_CONVERGED * _ORDER_TOLERANCE.
_ORDER_CONVERGED = 1e-2

# Dynamic active volume, SEAKMC's HandleVN defaults.
# DAV acts once more than this many translation steps are done (IgnoreSteps).
_DAV_IGNORE_STEPS = 4
# Per-atom value ln(|N_i|**_DAV_POWER + exp(_DAV_FLOOR)) (PowerOnV, MinValue4LOGV).
_DAV_POWER = 4
_DAV_FLOOR = -20.0
# Below this range of the value every weight is one (MinSpan4LOGV).
_DAV_MIN_RANGE = 4.0
# The sigmoid's argument runs over this many units across the range (XRange4LOGV).
_DAV_SIGMOID_RANGE = 20.0
# The sigmoid is centred this fraction of the range above its bottom (Ratio4Zero4LOGV).
_DAV_CENTRE = 0.2

EnergyAndForces = Callable[[np.ndarray], tuple[float, np.ndarray]]


class DimerResult(NamedTuple):
    """Outcome of one dimer search.

    Positions are full-system (N, 3) arrays in the input atom order, wrapped on
    the periodic axes only. Fields that a failed search did not reach are
    ``None``; no field is ever NaN.

    Attributes
    ----------
    success : bool
        Whether a first-order saddle and both minima were found.
    reason : Optional[str]
        Why the search failed, ``None`` on success: ``"empty_start"`` (no
        movable atom to draw the starting direction on, or nothing left of the
        direction once translation is removed), ``"zero_force"`` (the
        effective force gives no direction to move), ``"max_iterations"``,
        ``"positive_curvature"`` (the force converged where the lowest
        curvature is not negative), ``"higher_order_saddle"`` (a second
        curvature is negative too), ``"saddle_order_unverified"`` (the
        second curvature did not settle within the check's iterations),
        ``"minimisation_not_converged"`` or ``"non_finite_force"``. A start
        with no effective force is stepped along the mode before
        ``"zero_force"`` is given.
    saddle_positions : Optional[np.ndarray]
        Positions of the saddle (the dimer centre).
    min1_positions : Optional[np.ndarray]
        Minimum reached from the saddle against the mode, the side of the
        start.
    min2_positions : Optional[np.ndarray]
        Minimum reached from the saddle along the mode.
    e_start : Optional[float]
        Energy of the starting positions, eV; ``None`` only if the callback
        returned a non-finite value there.
    e_saddle : Optional[float]
        Energy of the saddle, eV.
    e_min1 : Optional[float]
        Energy of minimum 1, eV.
    e_min2 : Optional[float]
        Energy of minimum 2, eV.
    delr1 : Optional[float]
        3N minimum-image distance from the start to minimum 1, Å.
    delr2 : Optional[float]
        3N minimum-image distance from the start to minimum 2, Å.
    curvature : Optional[float]
        Curvature along the mode at the saddle, eV/Å².
    mode : Optional[np.ndarray]
        Unstable mode at the saddle, an (N, 3) unit vector in 3N space, zero on
        frozen atoms, pointing from the start towards the saddle.
    force_calls_saddle : int
        Callback calls up to the saddle, its order check included, or up to
        the failure if no saddle was found.
    force_calls_total : int
        Callback calls including the two minimisations.
    seed : int
        Seed of the starting direction.
    delr_sad : Optional[float]
        3N minimum-image distance from the start to the saddle, Å, pARTn's
        ``delr_sad``. Set by the ARTn core once it reaches a saddle; the
        dimer leaves it None.

    """

    success: bool
    reason: Optional[str]
    saddle_positions: Optional[np.ndarray]
    min1_positions: Optional[np.ndarray]
    min2_positions: Optional[np.ndarray]
    e_start: Optional[float]
    e_saddle: Optional[float]
    e_min1: Optional[float]
    e_min2: Optional[float]
    delr1: Optional[float]
    delr2: Optional[float]
    curvature: Optional[float]
    mode: Optional[np.ndarray]
    force_calls_saddle: int
    force_calls_total: int
    seed: int
    delr_sad: Optional[float] = None


class _NonFiniteForce(Exception):
    """The callback returned a non-finite energy or force."""


def dimer_search(
    positions: np.ndarray,
    cell: np.ndarray,
    pbc: np.ndarray,
    movable: np.ndarray,
    central_atom: int,
    seed: int,
    energy_and_forces: EnergyAndForces,
    *,
    start: Literal["local", "region"] = "local",
    start_radius: Optional[float] = None,
    convergence: Literal["max_component", "norm"] = "max_component",
    force_tolerance: float = 1e-3,
    separation: float = 0.005,
    trial_step: float = 0.015,
    max_step: float = 0.05,
    push_off: float = 0.1,
    max_translations: int = 1000,
    max_minimisation_steps: int = 5000,
    min_translations: int = 5,
    remove_translation: bool = False,
    dav: bool = True,
) -> DimerResult:
    """Find a saddle point and the two minima it connects.

    A failed search returns a result with ``success`` False and a reason; it
    never raises. An exception raised by the callback is not caught.

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
        Atom the ``local`` start is centred on.
    seed : int
        Seed of the starting direction.
    energy_and_forces : Callable[[np.ndarray], tuple[float, np.ndarray]]
        Takes (N, 3) positions in Å and returns the energy in eV and the
        (N, 3) forces in eV/Å.
    start : {"local", "region"}
        ``local`` draws the starting direction on the movable atoms within
        ``start_radius`` (minimum image) of the central atom, ``region`` on all
        movable atoms.
    start_radius : Optional[float]
        Radius of the ``local`` start, Å. Required for ``local``.
    convergence : {"max_component", "norm"}
        Force measure on the movable atoms: the largest absolute component, or
        the 3N norm.
    force_tolerance : float
        Convergence threshold on that measure, eV/Å, for the saddle and both
        minima.
    separation : float
        Distance from the dimer centre to the evaluated image, Å (SEAKMC
        ``DimerSep``).
    trial_step : float
        Trial translation step, Å (SEAKMC ``TrialStepsize``).
    max_step : float
        Translation step cap, Å (SEAKMC ``MaxStepsize``).
    push_off : float
        3N distance along the mode from the saddle to each side's starting
        point for the minimisation, Å.
    max_translations : int
        Translation steps before the search fails (SEAKMC ``NMax4Trans``).
    max_minimisation_steps : int
        FIRE steps per minimum before the search fails.
    min_translations : int
        Translation steps taken before the saddle convergence test applies, so
        the search cannot stop at its start (SEAKMC ``IgnoreSteps = 4``: it
        tests once more than four steps are done). With 0 the start itself is
        tested.
    remove_translation : bool
        Keep the net translation out of the starting direction, the
        rotational force and the mode after each rotation, when no atom is
        frozen. Set it for a
        translation-invariant energy, such as any interatomic potential:
        with every atom movable, rigid translation is then a zero-curvature
        mode the rotation would otherwise turn the dimer into. Leave it off
        for an external field. Ignored when an atom is frozen.
    dav : bool
        Reweight the mode atom by atom while climbing out of the basin
        (SEAKMC's dynamic active volume; see the module docstring). Applied
        before ``remove_translation``.

    Returns
    -------
    DimerResult
        The saddle, the two minima and the search's counters, or a failure
        with its reason.

    Raises
    ------
    ValueError
        If an argument is malformed, or ``start`` is ``local`` without a
        ``start_radius``.

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
    if start not in ("local", "region"):
        raise ValueError(f"unknown start {start!r}")
    if start == "local" and start_radius is None:
        raise ValueError("start='local' needs start_radius")
    if start_radius is not None and start_radius <= 0:
        raise ValueError("start_radius must be positive")
    if convergence not in ("max_component", "norm"):
        raise ValueError(f"unknown convergence measure {convergence!r}")
    if not 0 < trial_step < max_step:
        raise ValueError("need 0 < trial_step < max_step")

    weight = mask[:, None].astype(float)
    calls = 0

    def evaluate(x: np.ndarray) -> tuple[float, np.ndarray]:
        nonlocal calls
        calls += 1
        energy, forces = energy_and_forces(x.copy())
        forces = np.asarray(forces, dtype=float)
        if forces.shape != (n_atoms, 3):
            raise ValueError("energy_and_forces must return (N, 3) forces")
        if not (np.isfinite(energy) and np.all(np.isfinite(forces))):
            raise _NonFiniteForce
        return float(energy), forces * weight

    def measure(forces: np.ndarray) -> float:
        if convergence == "max_component":
            return float(np.max(np.abs(forces)))
        return float(np.linalg.norm(forces))

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

    e_start: Optional[float] = None
    try:
        e_start, f_start = evaluate(x_start)
    except _NonFiniteForce:
        return result("non_finite_force")

    rng = np.random.default_rng(seed)
    if start == "local":
        offsets = _minimum_image(x_start - x_start[central_atom], cell, pbc)
        selected = mask & (np.linalg.norm(offsets, axis=1) <= start_radius)
    else:
        selected = mask.copy()
    if not selected.any():
        return result("empty_start")
    # With no atom frozen, rigid translation of a translation-invariant
    # system is a zero-curvature mode that the rotation would turn the dimer
    # into (SEAKMC's CenterVN, off there because atoms beyond its buffer are
    # frozen); keep the mode free of it.
    translation_free = remove_translation and bool(mask.all())

    def untranslated(v: np.ndarray) -> np.ndarray:
        # The rotational force is projected too, so every orientation the
        # dimer tries stays free of translation.
        return v - v.mean(axis=0) if translation_free else v

    mode = np.zeros((n_atoms, 3))
    mode[selected] = rng.uniform(-0.5, 0.5, size=(int(selected.sum()), 3))
    mode = untranslated(mode)
    mode_norm = float(np.linalg.norm(mode))
    if mode_norm == 0.0:
        return result("empty_start")
    mode /= mode_norm

    def rotate(
        x0: np.ndarray, f0: np.ndarray, mode: np.ndarray, climbing: bool = False
    ) -> tuple[np.ndarray, float, bool]:
        """Rotate the dimer at ``x0``; return the mode, its curvature, exact.

        ``climbing``: the step meets DAV's conditions other than the curvature,
        which is tested before each rotation.
        """
        n_rotations = 0
        while True:
            _, f1 = evaluate(x0 + separation * mode)
            f_rot, curvature = _rotational_force(f0, f1, mode, separation)
            f_rot = untranslated(f_rot)
            f_rot_norm = float(np.linalg.norm(f_rot))
            if f_rot_norm < _ROTATION_CONVERGED:
                return mode, curvature, True

            # Trial rotation by pi/4 in the plane of the mode and the
            # rotational force.
            direction = f_rot / f_rot_norm
            trial, trial_direction = _rotate(mode, direction, 0.25 * math.pi)
            _, f1_trial = evaluate(x0 + separation * trial)
            f_rot_trial, curvature_trial = _rotational_force(
                f0, f1_trial, trial, separation
            )
            f_rot_trial = untranslated(f_rot_trial)
            if np.linalg.norm(f_rot_trial) < _ROTATION_CONVERGED:
                return trial, curvature_trial, True

            # Rotate to the analytic angle (SEAKMC's fit of the rotational
            # force, written with atan2 so that angle is phi in (0, pi/2]).
            f_rot1 = f_rot_norm
            f_rot2 = float(np.vdot(f_rot_trial, trial_direction))
            phi = 0.5 * math.atan2(f_rot1, -f_rot2)
            mode, _ = _rotate(trial, trial_direction, phi - 0.25 * math.pi)
            if climbing and curvature > 0:
                # DAV; ``curvature`` is still the value measured before this
                # rotation (SEAKMC's ROCURV).
                mode = mode * _dav_weights(mode, mask)[:, None]
            mode = untranslated(mode * weight)
            mode /= np.linalg.norm(mode)

            # Curvature along the new mode from the fit
            # c(phi) = a0 + a1 cos 2phi + b1 sin 2phi through c(0), c(pi/4)
            # and dc/dphi(0) = -2 f_rot1.
            b1 = -f_rot1
            a1 = curvature - curvature_trial + b1
            a0 = curvature - a1
            curvature = a0 + a1 * math.cos(2 * phi) + b1 * math.sin(2 * phi)

            n_rotations += 1
            if n_rotations >= _MAX_ROTATIONS or f_rot_norm < _ROTATION_THRESHOLD:
                return mode, curvature, False

    found: dict = {}
    try:
        x0, e0, f0 = x_start.copy(), e_start, f_start
        f_last = np.zeros_like(x0)  # effective force at the last step's start
        search_dir = np.zeros_like(x0)  # conjugate direction
        line_curvature = 0.0  # from the last line search (SEAKMC's TRCURV)
        in_basin = True  # no step has used a negative curvature yet
        kicks = 0  # displacements off a stationary start
        saddle = None
        for step in range(max_translations):
            if step > 0:
                e0, f0 = evaluate(x0)
            if step >= min_translations and measure(f0) < force_tolerance:
                mode, curvature, exact = rotate(x0, f0, mode)
                if not exact:
                    _, f1 = evaluate(x0 + separation * mode)
                    _, curvature = _rotational_force(f0, f1, mode, separation)
                if curvature >= 0:
                    return result("positive_curvature")
                saddle = x0
                break

            climbing = (
                dav and in_basin and line_curvature < 0 and step > _DAV_IGNORE_STEPS
            )
            mode, curvature, _ = rotate(x0, f0, mode, climbing)
            if curvature < 0:
                in_basin = False
            f_eff = _effective_force(f0, mode, curvature)

            # First half: Polak-Ribiere direction with Powell restart, then a
            # trial step.
            last_sq = float(np.vdot(f_last, f_last))
            if last_sq != 0.0 and abs(np.vdot(f_eff, f_last)) <= 0.5 * last_sq:
                gamma = float(np.vdot(f_eff, f_eff - f_last)) / last_sq
            else:
                gamma = 0.0
            search_dir = f_eff + gamma * search_dir
            dir_norm = float(np.linalg.norm(search_dir))
            if dir_norm < _ZERO_FORCE:
                # A stationary start, such as an atom on a symmetric site,
                # gives no effective force. Before any translation, step off
                # it along the mode (zero on frozen atoms, free of translation
                # when that is removed) and search from there; the start stays
                # the reference for energies and distances.
                if step != kicks or kicks == _MAX_KICKS:
                    return result("zero_force")
                kicks += 1
                x0 = x0 + trial_step * mode
                continue
            unit_dir = search_dir / dir_norm
            f_last = f_eff
            x_trial = x0 + trial_step * unit_dir
            _, f_trial = evaluate(x_trial)
            f_eff_trial = _effective_force(f_trial, mode, curvature)

            # Second half: secant step along the direction, capped.
            proj0 = float(np.vdot(f_last, unit_dir))
            proj1 = float(np.vdot(f_eff_trial, unit_dir))
            line_curvature = (proj0 - proj1) / trial_step
            x0 = x_trial + _secant_step(proj0, proj1, trial_step, max_step) * unit_dir
        if saddle is None:
            return result("max_iterations")

        if np.vdot(mode, saddle - x_start) < 0:
            mode = -mode
        found = dict(
            saddle_positions=_wrap(saddle, cell, pbc),
            e_saddle=e0,
            curvature=curvature,
            mode=mode,
        )

        # Saddle order: the dimer shows one negative curvature; a second
        # unstable direction can survive both relaxations.
        def project(v: np.ndarray) -> np.ndarray:
            v = untranslated(v * weight)
            return v - mode * np.vdot(mode, v)

        def hessian_product(v: np.ndarray) -> np.ndarray:
            _, f1 = evaluate(saddle + separation * v)
            return (f0 - f1) / separation

        second = _second_curvature(
            hessian_product, project, mode, curvature, rng.standard_normal(x0.shape)
        )
        found["force_calls_saddle"] = calls
        if second is None:
            return result("saddle_order_unverified", **found)
        if second < -_ORDER_TOLERANCE:
            return result("higher_order_saddle", **found)

        minima = []
        for sign in (-1.0, 1.0):
            x_min, e_min, converged = _fire(
                evaluate,
                saddle + sign * push_off * mode,
                measure,
                force_tolerance,
                max_minimisation_steps,
            )
            if not converged:
                return result(
                    "minimisation_not_converged",
                    **found,
                    force_calls_total=calls,
                )
            minima.append((x_min, e_min))
    except _NonFiniteForce:
        # After the saddle, keep it and count the minimisation's calls apart.
        return result("non_finite_force", **found, force_calls_total=calls)

    (x_min1, e_min1), (x_min2, e_min2) = minima
    return result(
        None,
        **found,
        min1_positions=_wrap(x_min1, cell, pbc),
        min2_positions=_wrap(x_min2, cell, pbc),
        e_min1=e_min1,
        e_min2=e_min2,
        delr1=_distance(x_min1, x_start, cell, pbc),
        delr2=_distance(x_min2, x_start, cell, pbc),
        force_calls_total=calls,
    )


def _rotational_force(
    f0: np.ndarray, f1: np.ndarray, mode: np.ndarray, separation: float
) -> tuple[np.ndarray, float]:
    """Return the rotational force and the curvature along ``mode``.

    The second image's force is extrapolated from the centre force ``f0`` and
    the evaluated image's force ``f1``.
    """
    f2 = 2.0 * f0 - f1
    df = f1 - f2
    f_rot = (df - mode * np.vdot(df, mode)) / (2.0 * separation)
    curvature = float(np.vdot(f2 - f1, mode)) / (2.0 * separation)
    return f_rot, curvature


def _second_curvature(
    hessian_product: Callable[[np.ndarray], np.ndarray],
    project: Callable[[np.ndarray], np.ndarray],
    mode: np.ndarray,
    curvature: float,
    start: np.ndarray,
) -> Optional[float]:
    """Return the second curvature at a saddle, or None if it is not established.

    Lanczos with full reorthogonalisation on the Hessian restricted to the
    allowed directions orthogonal to the unstable ``mode`` (``project``), one
    ``hessian_product`` per iteration. Its lowest eigenvalue lies between the
    Hessian's lowest and second-lowest (Cauchy interlacing), so once the
    lowest Ritz value has converged at or above ``-_ORDER_TOLERANCE`` there is
    no second unstable direction; that Ritz value is returned. Ritz values on
    the space spanned by ``mode`` (curvature ``curvature``) and the Lanczos
    vectors bound the Hessian's from above, so a second one below
    ``-_ORDER_TOLERANCE`` proves a second unstable direction; it is returned.
    """
    basis: list[np.ndarray] = []
    products: list[np.ndarray] = []
    q = project(start)
    norm = float(np.linalg.norm(q))
    if norm == 0.0:
        return math.inf  # no direction is left beside the mode
    lowest = math.inf
    for _ in range(_ORDER_MAX_ITERATIONS):
        basis.append(q / norm)
        products.append(hessian_product(basis[-1]))
        q = project(products[-1])
        for _ in range(2):
            for b in basis:
                q = q - b * np.vdot(b, q)
        norm = float(np.linalg.norm(q))

        # Rayleigh-Ritz on the mode and the Lanczos vectors, symmetrised
        # since the finite-difference products are not exactly symmetric.
        vectors = np.array([mode, *basis]).reshape(len(basis) + 1, -1)
        images = np.array(products).reshape(len(products), -1)
        ritz = np.empty((len(vectors), len(vectors)))
        ritz[1:, :] = images @ vectors.T
        ritz[0, :] = ritz[:, 0]
        ritz[0, 0] = curvature
        ritz = 0.5 * (ritz + ritz.T)
        second = float(np.linalg.eigvalsh(ritz)[1])
        if second < -_ORDER_TOLERANCE:
            return second
        previous, lowest = lowest, float(np.linalg.eigvalsh(ritz[1:, 1:])[0])
        # An invariant Krylov space makes its values exact.
        exhausted = norm < 1e-8 * max(1.0, abs(lowest))
        settled = abs(lowest - previous) <= _ORDER_CONVERGED * max(
            abs(lowest), _ORDER_TOLERANCE
        )
        if lowest >= -_ORDER_TOLERANCE and (settled or exhausted):
            return lowest
        if exhausted:
            return None
    return None


def _dav_weights(mode: np.ndarray, movable: np.ndarray) -> np.ndarray:
    """Return each atom's DAV weight, (N,), from the mode on the movable atoms.

    SEAKMC's ``compute_xRescale_logdisps`` and ``generate_sigmoid_scale`` with
    its default settings. Frozen atoms get weight one; their mode is zero.
    """
    weights = np.ones(len(mode))
    size = np.linalg.norm(mode[movable], axis=1)
    value = np.log(size**_DAV_POWER + math.exp(_DAV_FLOOR))
    bottom = float(value.min())
    value_range = float(value.max()) - bottom
    if value_range < _DAV_MIN_RANGE:
        return weights
    scale = value_range / _DAV_SIGMOID_RANGE
    centre = bottom + _DAV_CENTRE * value_range
    weights[movable] = 1.0 / (1.0 + np.exp(-(value - centre) / scale))
    return weights


def _rotate(
    v1: np.ndarray, v2: np.ndarray, theta: float
) -> tuple[np.ndarray, np.ndarray]:
    """Rotate the orthonormal pair (v1, v2) by ``theta`` in their plane."""
    cos, sin = math.cos(theta), math.sin(theta)
    return v1 * cos + v2 * sin, v2 * cos - v1 * sin


def _effective_force(
    forces: np.ndarray, mode: np.ndarray, curvature: float
) -> np.ndarray:
    """Return the translation force for the dimer.

    With negative curvature the component along the mode is inverted; with
    positive curvature only minus that component is kept, so the dimer climbs
    out of the basin.
    """
    parallel = mode * np.vdot(forces, mode)
    if curvature < 0:
        return forces - 2.0 * parallel
    return -parallel


def _secant_step(
    proj0: float, proj1: float, trial_step: float, max_step: float
) -> float:
    """Return SEAKMC's second translation step, measured from the trial point.

    ``proj0`` and ``proj1`` are the effective force along the step direction at
    the step's start and at the trial point. With negative curvature along the
    direction the step is ``max_step``; otherwise the centre goes to the
    secant estimate of the zero of the force, with the step capped.
    """
    curvature = (proj0 - proj1) / trial_step
    if curvature < 0:
        return max_step
    step = 0.5 * (proj0 + proj1) / (curvature + 1.0e-12)
    if abs(step) > max_step:
        return max_step - trial_step if step >= 0 else trial_step - max_step
    return step - 0.5 * trial_step


def _fire(
    evaluate: EnergyAndForces,
    x: np.ndarray,
    measure: Callable[[np.ndarray], float],
    tolerance: float,
    max_steps: int,
) -> tuple[np.ndarray, float, bool]:
    """Relax ``x`` with mass-free FIRE.

    Parameters follow Bitzek et al., Phys. Rev. Lett. 97, 170201 (2006), as in
    ASE's FIRE: time step 0.1 rising to 1.0, at most 0.2 Å of 3N displacement
    per step.

    Returns the relaxed positions, their energy and whether the force measure
    fell below ``tolerance`` within ``max_steps`` steps.
    """
    dt, dt_max, max_move = 0.1, 1.0, 0.2
    n_min, f_inc, f_dec, alpha_start, f_alpha = 5, 1.1, 0.5, 0.1, 0.99
    alpha, n_uphill_free = alpha_start, 0
    x = x.copy()
    v = np.zeros_like(x)
    energy, forces = evaluate(x)
    for _ in range(max_steps):
        if measure(forces) < tolerance:
            return x, energy, True
        if np.vdot(v, forces) > 0.0:
            v = (1.0 - alpha) * v + alpha * forces * (
                np.linalg.norm(v) / np.linalg.norm(forces)
            )
            if n_uphill_free > n_min:
                dt = min(dt * f_inc, dt_max)
                alpha *= f_alpha
            n_uphill_free += 1
        else:
            v[:] = 0.0
            alpha = alpha_start
            dt *= f_dec
            n_uphill_free = 0
        v += dt * forces
        move = dt * v
        move_norm = float(np.linalg.norm(move))
        if move_norm > max_move:
            move *= max_move / move_norm
        x += move
        energy, forces = evaluate(x)
    return x, energy, measure(forces) < tolerance


def _minimum_image(
    displacements: np.ndarray, cell: np.ndarray, pbc: np.ndarray
) -> np.ndarray:
    """Return the shortest displacements under translations by periodic rows.

    Round independent projections for an orthogonal basis. For a skew basis,
    QR gives an upper-triangular closest-lattice-point problem with at most
    three dimensions. Fixing integer shifts from the last axis to the first
    makes each partial squared distance a lower bound for all its descendants.
    """
    basis = np.asarray(cell, dtype=float)[np.asarray(pbc, dtype=bool)]
    if not len(basis) or not len(displacements):
        return displacements.copy()
    gram = basis @ basis.T
    if np.count_nonzero(gram) == len(basis):
        shifts = np.rint((displacements @ basis.T) / gram.diagonal())
        return displacements - shifts @ basis

    # In QR coordinates column k of the lattice basis affects only axes 0..k.
    # Start with the image obtained by rounding one plane at a time.
    basis = basis[np.argsort(gram.diagonal())]
    q, r = np.linalg.qr(basis.T)
    projected = displacements @ q
    initial = np.zeros_like(projected)
    for k in reversed(range(len(basis))):
        initial[:, k] = np.rint(projected[:, k] / r[k, k])
        projected -= initial[:, k, None] * r[:, k]

    # Search only lattice planes within the current best radius. Components
    # outside the periodic span are identical for every candidate image.
    best_squared = np.sum(projected**2, axis=1)
    best_shift = np.zeros_like(projected)
    shift = np.zeros_like(projected)

    def search(k: int, residual: np.ndarray) -> None:
        centre = residual[:, k] / r[k, k]
        nearest = np.rint(centre)
        reach = 0  # On the last axis the nearest integer is optimal.
        if k:
            remaining = best_squared - np.sum(residual[:, k + 1 :] ** 2, axis=1)
            radius = np.sqrt(np.maximum(remaining, 0.0)) / abs(r[k, k])
            reach = int(np.max(radius + np.abs(centre - nearest)))
        for offset in range(-reach, reach + 1):
            shift[:, k] = nearest + offset
            trial = residual - shift[:, k, None] * r[:, k]
            squared = np.sum(trial[:, k:] ** 2, axis=1)
            improved = squared < best_squared
            if k == 0:
                best_squared[improved] = squared[improved]
                best_shift[improved] = shift[improved]
            elif improved.any():
                search(k - 1, trial)

    search(len(basis) - 1, projected)
    return displacements - (initial + best_shift) @ basis


def _wrap(positions: np.ndarray, cell: np.ndarray, pbc: np.ndarray) -> np.ndarray:
    """Return positions wrapped into the cell on the periodic axes only."""
    fractional = positions @ np.linalg.inv(cell)
    return positions - np.floor(fractional[:, pbc]) @ cell[pbc]


def _distance(a: np.ndarray, b: np.ndarray, cell: np.ndarray, pbc: np.ndarray) -> float:
    """Return the 3N minimum-image distance between two configurations."""
    return float(np.linalg.norm(_minimum_image(a - b, cell, pbc)))
