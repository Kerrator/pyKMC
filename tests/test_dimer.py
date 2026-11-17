"""Dimer search on analytic surfaces with known saddles.

Every atom feels the same external field and atoms do not interact, so the
lowest saddle next to a minimum is one atom crossing one barrier, and its
position, energy and curvature are known in closed form. No LAMMPS is needed.

``curved_valley``: in x and y, a double well whose path between the minima
(x = +-1, y = 0.5) bends through the saddle at (0, 0); barrier 0.5 eV,
curvature -2 eV/Å² along x at the saddle. In z, wells every 3 Å.

``periodic_wells``: wells on a 2.5 Å grid offset by 0.5 Å in x and y and on a
2.5 Å grid in z. The x barrier (0.5 eV) is the lowest, so a hop goes along x,
to the saddle half a grid step away, with curvature -k_x along x.
"""

import ast
import inspect
import math
from pathlib import Path

import numpy as np
import pytest

from pykmc.dimer import DimerResult, _minimum_image, _wrap, dimer_search

# curved_valley parameters
VALLEY_A, VALLEY_B, VALLEY_C, VALLEY_BZ, VALLEY_PZ = 0.5, 2.0, 0.5, 0.5, 3.0
VALLEY_BARRIER = VALLEY_A
VALLEY_CURVATURE = -4.0 * VALLEY_A

# periodic_wells parameters
GRID, SHIFT = 2.5, 0.5
WELL_AX, WELL_AY, WELL_AZ = 0.25, 0.6, 0.6
WELL_K = 2.0 * math.pi / GRID
WELL_BARRIER = 2.0 * WELL_AX
WELL_CURVATURE = -WELL_AX * WELL_K**2


def curved_valley(x: np.ndarray) -> tuple[float, np.ndarray]:
    """Return the energy and forces of the curved valley field."""
    a, b, c, bz = VALLEY_A, VALLEY_B, VALLEY_C, VALLEY_BZ
    kz = 2.0 * math.pi / VALLEY_PZ
    px, py, pz = x[:, 0], x[:, 1], x[:, 2]
    bend = py - c * px**2
    energy = a * (px**2 - 1) ** 2 + b * bend**2 + bz * (1 - np.cos(kz * pz))
    forces = np.stack(
        [
            -(4 * a * px * (px**2 - 1) - 4 * b * c * px * bend),
            -2 * b * bend,
            -bz * kz * np.sin(kz * pz),
        ],
        axis=1,
    )
    return float(energy.sum()), forces


def periodic_wells(x: np.ndarray) -> tuple[float, np.ndarray]:
    """Return the energy and forces of the periodic wells field."""
    k = WELL_K
    px, py, pz = x[:, 0] - SHIFT, x[:, 1] - SHIFT, x[:, 2]
    energy = (
        WELL_AX * (1 - np.cos(k * px))
        + WELL_AY * (1 - np.cos(k * py))
        + WELL_AZ * (1 - np.cos(k * pz))
    )
    forces = -k * np.stack(
        [WELL_AX * np.sin(k * px), WELL_AY * np.sin(k * py), WELL_AZ * np.sin(k * pz)],
        axis=1,
    )
    return float(energy.sum()), forces


def counting(surface):  # noqa: ANN001, ANN201
    """Wrap a surface so the number of calls can be read back."""

    def call(x: np.ndarray) -> tuple[float, np.ndarray]:
        call.count += 1
        return surface(x)

    call.count = 0
    return call


OPEN_CELL = np.diag([20.0, 20.0, 20.0])
NO_PBC = np.array([False, False, False])


def valley_start(n_atoms: int = 4) -> np.ndarray:
    """Atoms in the x = 1 well, one z well apart; atom 0 nudged towards the saddle.

    The nudge (1e-3 Å) stands in for a minimiser's residual force; it sets the
    side the dimer climbs to.
    """
    x = np.array([[1.0, 0.5, VALLEY_PZ * i] for i in range(n_atoms)])
    x[0] += [-1e-3, -1e-3, 0.0]
    return x


def valley_search(x: np.ndarray, seed: int = 1, **kwargs: object) -> DimerResult:
    """Run the dimer on the curved valley, all atoms movable, local start."""
    kwargs.setdefault("start_radius", 1.5)
    return dimer_search(
        x, OPEN_CELL, NO_PBC, np.ones(len(x), bool), 0, seed, curved_valley, **kwargs
    )


def assert_no_nan(result: DimerResult) -> None:
    """Check every field is None or finite."""
    for name, value in result._asdict().items():
        if value is None or isinstance(value, (bool, str)):
            continue
        assert np.all(np.isfinite(value)), name


def test_defaults_are_the_design_record_ones() -> None:
    """Start local, max component below 1e-3 eV/Å, SEAKMC's separation and steps."""
    params = inspect.signature(dimer_search).parameters
    assert params["start"].default == "local"
    assert params["start_radius"].default is None
    assert params["convergence"].default == "max_component"
    assert params["force_tolerance"].default == 1e-3
    assert params["separation"].default == 0.005
    assert params["trial_step"].default == 0.015
    assert params["max_step"].default == 0.05


def test_core_imports_only_numpy_and_the_standard_library() -> None:
    """The core has no LAMMPS, pARTn or MPI import of its own."""
    source = Path(inspect.getfile(dimer_search)).read_text()
    imported = set()
    for node in ast.walk(ast.parse(source)):
        if isinstance(node, ast.Import):
            imported.update(alias.name.split(".")[0] for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            imported.add(node.module.split(".")[0])
    assert imported == {"math", "typing", "numpy"}


def test_seakmc_licence_and_citation_are_with_the_code() -> None:
    """SEAKMC's copyright notice, its licence text and Liang and Xu 2023."""
    source = Path(inspect.getfile(dimer_search)).read_text()
    assert "Copyright (c) 2026 Haixuan Xu and Tao Liang" in source
    assert "BSD 3-Clause License" in source
    assert "Neither the name of the copyright holder" in source
    assert "Comput. Mater. Sci. 228, 112354 (2023)" in source


def test_local_start_needs_a_radius() -> None:
    """The radius of the local start has no default."""
    with pytest.raises(ValueError, match="start_radius"):
        valley_search(valley_start(), start_radius=None)


def test_converges_to_the_known_saddle_and_minima() -> None:
    """Saddle, curvature, minima, energies, distances and mode match the surface."""
    x = valley_start()
    surface = counting(curved_valley)
    result = dimer_search(
        x, OPEN_CELL, NO_PBC, np.ones(4, bool), 0, 1, surface, start_radius=1.5
    )
    assert result.success and result.reason is None

    # Only atom 0 hops; the others stay in their wells.
    np.testing.assert_allclose(result.saddle_positions[0], [0.0, 0.0, 0.0], atol=2e-3)
    np.testing.assert_allclose(result.saddle_positions[1:], x[1:], atol=1e-6)
    assert result.e_saddle == pytest.approx(VALLEY_BARRIER, abs=1e-5)
    # Curvature at the saddle within 5e-3 eV/Å² of the analytic -2 eV/Å².
    assert result.curvature < 0
    assert result.curvature == pytest.approx(VALLEY_CURVATURE, abs=5e-3)

    # The mode is the x direction on atom 0, pointing from the start to the saddle.
    assert result.mode.shape == (4, 3)
    assert np.linalg.norm(result.mode) == pytest.approx(1.0)
    assert result.mode[0, 0] < -0.999
    np.testing.assert_allclose(result.mode[1:], 0.0, atol=1e-6)

    # Minimum 1 is the start's well, minimum 2 the other well.
    np.testing.assert_allclose(result.min1_positions[0], [1.0, 0.5, 0.0], atol=2e-3)
    np.testing.assert_allclose(result.min2_positions[0], [-1.0, 0.5, 0.0], atol=2e-3)
    assert result.e_min1 == pytest.approx(0.0, abs=1e-5)
    assert result.e_min2 == pytest.approx(0.0, abs=1e-5)
    assert result.delr1 < 0.01
    assert result.delr2 == pytest.approx(2.0, abs=5e-3)

    # Force calls: up to the saddle, then in total, as the callback counted them.
    assert 0 < result.force_calls_saddle < result.force_calls_total
    assert result.force_calls_total == surface.count
    assert result.seed == 1
    assert_no_nan(result)


@pytest.mark.parametrize("convergence", ["max_component", "norm"])
def test_both_convergence_measures(convergence: str) -> None:
    """The saddle and both minima meet the chosen force measure."""
    result = valley_search(valley_start(), convergence=convergence)
    assert result.success
    assert result.curvature == pytest.approx(VALLEY_CURVATURE, abs=5e-3)
    for positions in (result.saddle_positions, result.min1_positions):
        _, forces = curved_valley(positions)
        if convergence == "max_component":
            assert np.max(np.abs(forces)) < 1e-3
        else:
            assert np.linalg.norm(forces) < 1e-3


@pytest.mark.parametrize("seed", [1, 2, 3])
def test_region_start_finds_a_one_atom_saddle(seed: int) -> None:
    """Drawn over all movable atoms, the start still ends on a single hop."""
    x = np.array([[1.0, 0.5, VALLEY_PZ * i] for i in range(4)])
    x[:, 0] -= 1e-3 * (1 + np.arange(4))
    x[:, 1] -= 1e-3
    result = dimer_search(
        x, OPEN_CELL, NO_PBC, np.ones(4, bool), 0, seed, curved_valley, start="region"
    )
    assert result.success
    moved = np.linalg.norm(result.saddle_positions - x, axis=1)
    assert np.sum(moved > 0.5) == 1
    assert result.e_saddle == pytest.approx(VALLEY_BARRIER, abs=1e-5)
    assert result.curvature == pytest.approx(VALLEY_CURVATURE, abs=5e-3)


def test_positive_curvature_end_is_a_failure() -> None:
    """Converging where the curvature is positive (here, the start) fails."""
    x = np.array([[1.0, 0.5, VALLEY_PZ * i] for i in range(4)])
    x[0] += [-1e-5, -1e-5, 0.0]  # residual force below the threshold
    result = valley_search(x, min_translations=0)
    assert not result.success
    assert result.reason == "positive_curvature"
    assert result.saddle_positions is None and result.curvature is None
    assert result.force_calls_saddle == result.force_calls_total > 0
    assert_no_nan(result)


def test_same_seed_same_result() -> None:
    """A repeated seed repeats the search exactly; another seed draws differently."""
    first = valley_search(valley_start(), seed=7)
    again = valley_search(valley_start(), seed=7)
    for name, value in first._asdict().items():
        if isinstance(value, np.ndarray):
            assert np.array_equal(value, getattr(again, name)), name
        else:
            assert value == getattr(again, name), name

    other = valley_search(valley_start(), seed=8)
    assert other.success
    assert not np.array_equal(other.saddle_positions, first.saddle_positions)
    np.testing.assert_allclose(
        other.saddle_positions, first.saddle_positions, atol=2e-3
    )


def test_frozen_atoms_do_not_move_or_steer() -> None:
    """A frozen atom keeps its position, and its force changes nothing."""
    movable = np.array([True, False, True, True])
    on_site = valley_start()
    displaced = on_site.copy()
    displaced[1] += [0.2, -0.2, 0.1]  # off its minimum: a large force
    # The local radius reaches atom 1, so only the mask keeps it still.
    kwargs = dict(start_radius=3.5)

    results = []
    for x in (on_site, displaced):
        result = dimer_search(
            x, OPEN_CELL, NO_PBC, movable, 0, 3, curved_valley, **kwargs
        )
        assert result.success
        for positions in (
            result.saddle_positions,
            result.min1_positions,
            result.min2_positions,
        ):
            assert np.array_equal(positions[1], x[1])
        assert np.array_equal(result.mode[1], np.zeros(3))
        results.append(result)

    a, b = results
    for name in ("saddle_positions", "min1_positions", "min2_positions", "mode"):
        assert np.array_equal(getattr(a, name)[movable], getattr(b, name)[movable])
    assert a.e_saddle - a.e_start == pytest.approx(b.e_saddle - b.e_start)


def test_shuffled_atom_order_gives_the_same_saddle() -> None:
    """Reordering the atoms reorders the result and changes nothing else."""
    sites = [(2, 1, 1), (0, 0, 0), (1, 3, 2), (2, 1, 2), (3, 2, 0), (0, 2, 3)]
    x = np.array([[SHIFT + GRID * i, SHIFT + GRID * j, GRID * k] for i, j, k in sites])
    central = 3
    x[central, 0] -= 1e-3
    cell = np.diag([10.0, 10.0, 10.0])
    pbc = np.array([True, True, True])
    movable = np.ones(len(x), bool)
    result = dimer_search(
        x, cell, pbc, movable, central, 5, periodic_wells, start_radius=1.0
    )
    assert result.success

    order = np.array([4, 0, 3, 5, 1, 2])
    shuffled = dimer_search(
        x[order],
        cell,
        pbc,
        movable[order],
        int(np.flatnonzero(order == central)[0]),
        5,
        periodic_wells,
        start_radius=1.0,
    )
    assert shuffled.success
    for name in ("saddle_positions", "min1_positions", "min2_positions", "mode"):
        np.testing.assert_allclose(
            getattr(shuffled, name), getattr(result, name)[order], atol=1e-9
        )
    for name in ("e_saddle", "e_min1", "e_min2", "delr1", "delr2", "curvature"):
        assert getattr(shuffled, name) == pytest.approx(getattr(result, name))

    # The saddle is the analytic one: the central atom half a grid step back in x.
    expected = x[central] + [-(GRID / 2 - 1e-3), 0.0, 0.0]
    np.testing.assert_allclose(result.saddle_positions[central], expected, atol=2e-3)
    assert result.curvature == pytest.approx(WELL_CURVATURE, abs=5e-3)


def test_hop_across_the_periodic_boundary() -> None:
    """The hop crosses x = 0; z is not periodic and is never wrapped."""
    cell = np.diag([10.0, 10.0, 20.0])
    pbc = np.array([True, True, False])
    # Atom 0 sits in the well at x = 0.5, nudged towards x = 0, above the cell
    # in z; atom 1 sits below the cell in z.
    x = np.array(
        [[SHIFT - 1e-3, SHIFT, 22.5], [5.5, 5.5, -2.5], [3.0, 8.0, 5.0]],
    )
    seen_x = []

    def surface(positions: np.ndarray) -> tuple[float, np.ndarray]:
        seen_x.append(positions[0, 0])
        return periodic_wells(positions)

    result = dimer_search(
        x, cell, pbc, np.ones(3, bool), 0, 11, surface, start_radius=1.0
    )
    assert result.success
    # The search kept its own unwrapped coordinates.
    assert min(seen_x) < 0.0

    # Wrapped results: saddle at x = 0.5 - 1.25 + 10, minimum 2 at 0.5 - 2.5 + 10.
    np.testing.assert_allclose(
        result.saddle_positions[0], [9.25, SHIFT, 22.5], atol=2e-3
    )
    np.testing.assert_allclose(
        result.min1_positions[0], [SHIFT, SHIFT, 22.5], atol=2e-3
    )
    np.testing.assert_allclose(result.min2_positions[0], [8.0, SHIFT, 22.5], atol=2e-3)
    for positions in (
        result.saddle_positions,
        result.min1_positions,
        result.min2_positions,
    ):
        assert np.all((positions[:, :2] >= 0.0) & (positions[:, :2] < 10.0))
        assert positions[1, 2] == pytest.approx(-2.5, abs=1e-6)
        assert positions[0, 2] == pytest.approx(22.5, abs=1e-6)
    # Minimum-image distances: the hop is 2.5 Å, not 7.5 Å.
    assert result.delr1 < 0.01
    assert result.delr2 == pytest.approx(GRID, abs=5e-3)
    assert result.e_saddle == pytest.approx(WELL_BARRIER, abs=1e-5)
    assert result.curvature == pytest.approx(WELL_CURVATURE, abs=5e-3)


def test_minimum_image_and_wrap_skip_non_periodic_axes() -> None:
    """Only periodic axes are folded."""
    cell = np.diag([10.0, 10.0, 20.0])
    pbc = np.array([True, True, False])
    np.testing.assert_allclose(
        _minimum_image(np.array([[9.0, -6.0, 19.0]]), cell, pbc), [[-1.0, 4.0, 19.0]]
    )
    np.testing.assert_allclose(
        _wrap(np.array([[-1.0, 11.0, 25.0]]), cell, pbc), [[9.0, 1.0, 25.0]]
    )


def test_zero_force_start_fails_with_a_reason() -> None:
    """At an exact minimum the effective force gives no direction: no NaN."""
    x = np.array([[SHIFT, SHIFT, 0.0], [SHIFT + GRID, SHIFT, GRID]])
    result = dimer_search(
        x,
        np.diag([10.0, 10.0, 10.0]),
        np.array([True, True, True]),
        np.ones(2, bool),
        0,
        1,
        periodic_wells,
        start_radius=1.0,
    )
    assert not result.success
    assert result.reason == "zero_force"
    assert result.e_start == pytest.approx(0.0, abs=1e-12)
    assert_no_nan(result)


def test_flat_surface_fails_with_a_reason() -> None:
    """With no force and no curvature anywhere the search cannot move."""

    def flat(x: np.ndarray) -> tuple[float, np.ndarray]:
        return 0.0, np.zeros_like(x)

    result = dimer_search(
        valley_start(), OPEN_CELL, NO_PBC, np.ones(4, bool), 0, 1, flat, start="region"
    )
    assert result.reason == "zero_force"
    assert_no_nan(result)


def test_other_failures_return_a_reason() -> None:
    """Iteration caps, empty starts and non-finite forces fail without raising."""
    assert valley_search(valley_start(), max_translations=3).reason == (
        "max_iterations"
    )

    capped = valley_search(valley_start(), max_minimisation_steps=1)
    assert capped.reason == "minimisation_not_converged"
    assert capped.saddle_positions is not None and capped.min1_positions is None
    assert capped.force_calls_saddle < capped.force_calls_total

    movable = np.array([False, True, True, True])
    empty = dimer_search(
        valley_start(),
        OPEN_CELL,
        NO_PBC,
        movable,
        0,
        1,
        curved_valley,
        start_radius=1.5,
    )
    assert empty.reason == "empty_start"

    def goes_bad(x: np.ndarray) -> tuple[float, np.ndarray]:
        goes_bad.count += 1
        energy, forces = curved_valley(x)
        if goes_bad.count > 20:
            forces[0, 0] = np.nan
        return energy, forces

    goes_bad.count = 0
    bad = dimer_search(
        valley_start(),
        OPEN_CELL,
        NO_PBC,
        np.ones(4, bool),
        0,
        1,
        goes_bad,
        start_radius=1.5,
    )
    assert bad.reason == "non_finite_force"
    assert_no_nan(bad)


def test_callback_exception_is_not_caught() -> None:
    """A fatal engine error still aborts the run."""

    def broken(x: np.ndarray) -> tuple[float, np.ndarray]:
        raise RuntimeError("engine died")

    with pytest.raises(RuntimeError, match="engine died"):
        dimer_search(
            valley_start(),
            OPEN_CELL,
            NO_PBC,
            np.ones(4, bool),
            0,
            1,
            broken,
            start_radius=1.5,
        )


# A translation-invariant surface: the curved valley field acting on the
# separation of two atoms, d = x0 - x1 + PAIR_OFFSET. Every rigid translation
# is a zero-curvature mode. In the separation coordinate the saddle is the
# field's own (d = 0, barrier 0.5 eV); along the dimer's mode (v, -v) / sqrt(2)
# the curvature is twice the field's, -4 eV/Å², and the minima are
# d = (+-1, 0.5, 0), sqrt(2) Å apart in 6N space.
PAIR_OFFSET = np.array([0.0, 0.0, 3.0])


def valley_pair(x: np.ndarray) -> tuple[float, np.ndarray]:
    """Return the energy and forces of the curved valley on the pair separation."""
    energy, forces = curved_valley((x[0] - x[1] + PAIR_OFFSET)[None])
    return energy, np.vstack([forces, -forces])


def pair_start() -> np.ndarray:
    """Two atoms 3 Å apart with d in the x = 1 well, nudged towards the saddle."""
    x = np.array([[1.0, 0.5, 0.0], [0.0, 0.0, 3.0]])
    x[0] += [-1e-3, -1e-3, 0.0]
    return x


@pytest.mark.parametrize("seed", [1, 2])
def test_translation_removed_when_every_atom_moves(seed: int) -> None:
    """With every atom movable the mode stays free of translation and finds the saddle."""
    x = pair_start()
    centroid = x.mean(axis=0)
    result = dimer_search(
        x,
        OPEN_CELL,
        NO_PBC,
        np.ones(2, bool),
        0,
        seed,
        valley_pair,
        start_radius=1.5,
        remove_translation=True,
    )
    assert result.success
    assert_no_nan(result)
    assert result.e_saddle == pytest.approx(VALLEY_BARRIER, abs=1e-5)
    assert result.curvature == pytest.approx(2 * VALLEY_CURVATURE, abs=5e-3)
    # The centroid never moves, so the saddle is fixed in space.
    saddle = result.saddle_positions
    np.testing.assert_allclose(
        saddle, [centroid - PAIR_OFFSET / 2, centroid + PAIR_OFFSET / 2], atol=2e-3
    )
    np.testing.assert_allclose(saddle.mean(axis=0), centroid, atol=1e-12)
    np.testing.assert_allclose(result.mode.sum(axis=0), 0.0, atol=1e-12)
    minimum2 = result.min2_positions
    np.testing.assert_allclose(
        minimum2[0] - minimum2[1] + PAIR_OFFSET, [-1.0, 0.5, 0.0], atol=2e-3
    )
    assert result.delr1 < 2e-3
    assert result.delr2 == pytest.approx(math.sqrt(2), abs=2e-3)


def test_translation_left_in_drifts_on_a_translation_invariant_surface() -> None:
    """Without the removal the same search turns into translation and never converges."""
    result = dimer_search(
        pair_start(),
        OPEN_CELL,
        NO_PBC,
        np.ones(2, bool),
        0,
        1,
        valley_pair,
        start_radius=1.5,
        max_translations=200,
    )
    assert result.reason == "max_iterations"


def test_remove_translation_is_ignored_with_a_frozen_atom() -> None:
    """With an atom frozen translation is no zero mode, and the option changes nothing."""
    movable = np.array([True, False])
    on = dimer_search(
        pair_start(),
        OPEN_CELL,
        NO_PBC,
        movable,
        0,
        1,
        valley_pair,
        start_radius=1.5,
        remove_translation=True,
    )
    off = dimer_search(
        pair_start(),
        OPEN_CELL,
        NO_PBC,
        movable,
        0,
        1,
        valley_pair,
        start_radius=1.5,
    )
    assert on.success
    for name, value in on._asdict().items():
        if isinstance(value, np.ndarray):
            np.testing.assert_array_equal(value, getattr(off, name), err_msg=name)
        else:
            assert value == getattr(off, name), name
    np.testing.assert_array_equal(on.mode[1], 0.0)


def test_remove_translation_on_one_atom_is_an_empty_start() -> None:
    """One movable atom has nothing but translation to move along."""
    result = dimer_search(
        valley_start(1),
        OPEN_CELL,
        NO_PBC,
        np.ones(1, bool),
        0,
        1,
        curved_valley,
        start_radius=1.5,
        remove_translation=True,
    )
    assert result.reason == "empty_start"
    assert_no_nan(result)


def test_every_orientation_stays_free_of_translation(monkeypatch) -> None:  # noqa: ANN001
    """Force noise with a net component does not leak translation into the dimer.

    Real forces sum to zero only to rounding. Noise of 1e-9 eV/Å with a net
    part stands in for that: every orientation the dimer evaluates, trial
    rotations included, and the final mode keep a zero net translation.
    """
    import pykmc.dimer as dimer

    rng = np.random.default_rng(3)

    def noisy_pair(x: np.ndarray) -> tuple[float, np.ndarray]:
        energy, forces = valley_pair(x)
        return energy, forces + 1e-9 * rng.standard_normal(forces.shape)

    nets = []
    original = dimer._rotational_force

    def record(f0, f1, mode, separation):  # noqa: ANN001, ANN202
        nets.append(float(np.abs(mode.sum(axis=0)).max()))
        return original(f0, f1, mode, separation)

    monkeypatch.setattr(dimer, "_rotational_force", record)
    result = dimer_search(
        pair_start(),
        OPEN_CELL,
        NO_PBC,
        np.ones(2, bool),
        0,
        1,
        noisy_pair,
        start_radius=1.5,
        remove_translation=True,
    )
    assert result.success
    assert result.e_saddle == pytest.approx(VALLEY_BARRIER, abs=1e-5)
    assert max(nets) < 1e-14
    assert np.abs(result.mode.sum(axis=0)).max() < 1e-14


def test_worker_op_module_imports_without_lammps() -> None:
    """The dimer worker op can be imported where LAMMPS is not installed."""
    import subprocess
    import sys

    code = (
        "import sys; sys.modules['lammps'] = None\n"
        "from pykmc.engine.dimer_search import DimerSearchExtension\n"
        "assert 'pykmc.engine.lammps' not in sys.modules\n"
    )
    root = Path(__file__).resolve().parents[1]
    run = subprocess.run(
        [sys.executable, "-c", code], cwd=root, capture_output=True, text=True
    )
    assert run.returncode == 0, run.stderr
