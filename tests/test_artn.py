"""ARTn search (plain core) on the dimer tests' analytic surfaces.

Atoms do not interact and feel the same external field, so each saddle next
to a minimum is one atom crossing one barrier, with position, energy and
curvature known in closed form (see ``tests/test_dimer.py``). No LAMMPS is
needed. The push is random, so each test names a seed whose push leads to the
saddle it checks.
"""

import ast
import inspect
import math
from pathlib import Path

import numpy as np
import pytest

import pykmc.artn as artn
from pykmc.artn import NumpyVectorSpace, artn_search
from pykmc.config import PartnConfig
from pykmc.dimer import DimerResult, _minimum_image

from .test_dimer import (
    GRID,
    NO_PBC,
    OPEN_CELL,
    SHIFT,
    VALLEY_BARRIER,
    VALLEY_CURVATURE,
    WELL_BARRIER,
    WELL_CURVATURE,
    assert_no_nan,
    counting,
    curved_valley,
    periodic_wells,
    valley_start,
)

# A smaller push than pyKMC's default suits these 1 Å-wide wells.
PUSH = dict(push_mode="list", push_step_size=0.3)
# Seed whose push on the valley's atom 0 leads to the x saddle at the origin.
VALLEY_SEED = 5


def valley_search(
    x: np.ndarray, seed: int = VALLEY_SEED, **kwargs: object
) -> DimerResult:
    """Run ARTn on the curved valley, all atoms movable, push on atom 0."""
    options = {**PUSH, **kwargs}
    return artn_search(
        x, OPEN_CELL, NO_PBC, np.ones(len(x), bool), 0, seed, curved_valley, **options
    )


def test_core_has_no_lammps_partn_or_mpi_import() -> None:
    """The core imports only the standard library, NumPy and pykmc.dimer."""
    source = Path(inspect.getfile(artn_search)).read_text()
    imported = set()
    for node in ast.walk(ast.parse(source)):
        if isinstance(node, ast.Import):
            imported.update(alias.name.split(".")[0] for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            imported.add("." * node.level + (node.module or ""))
    assert imported == {"math", "typing", "numpy", ".dimer"}


def test_partn_licence_and_citation_are_with_the_code() -> None:
    """The copyright line, Apache-2.0 notice and citation of pARTn are there."""
    source = Path(inspect.getfile(artn_search)).read_text()
    assert "Copyright (c) 2021, MAMMASMIAS Consortium" in source
    assert "SPDX-License-Identifier: Apache-2.0" in source
    assert 'Licensed under the Apache License, Version 2.0 (the "License")' in source
    assert "Comput. Phys. Commun. 295, 108961 (2024)" in source


def test_defaults_are_pykmc_partn_defaults() -> None:
    """Each [pARTn] parameter the core takes defaults to pyKMC's value."""
    params = inspect.signature(artn_search).parameters
    defaults = PartnConfig()
    for name in (
        "push_mode",
        "push_dist_thr",
        "push_step_size",
        "ninit",
        "lanczos_min_size",
        "lanczos_max_size",
        "lanczos_disp",
        "lanczos_eval_conv_thr",
        "eigval_thr",
        "eigen_step_size",
        "nperp",
        "forc_thr",
        "convergence_property",
        "push_over",
        "nevalf_max",
        "dmax",
    ):
        assert params[name].default == getattr(defaults, name), name


def test_partn_constants_are_converted_to_ev_and_angstrom() -> None:
    """The ruling's two examples: 0.01 Ry/bohr² and 80 Ry."""
    assert artn._CURVATURE_FLOOR == pytest.approx(0.48587, abs=1e-5)
    assert artn._ENERGY_LIMIT == pytest.approx(1088.455, abs=1e-3)


def test_displacement_and_time_step_are_where_partn_puts_them() -> None:
    """The unit round trip of pARTn, as issue 02's recordings measured it."""
    # The first INIT move of every recording: 0.3 Å nominal.
    assert 0.3 * artn._DISPLACEMENT_FACTOR == pytest.approx(
        0.29999998296888336, abs=1e-15
    )
    # FIRE's time step after pARTn's first reset call.
    assert artn._DT0 == 0.0010000000000000002


def test_finds_the_saddle_and_its_two_minima() -> None:
    """Saddle with negative curvature, both minima, energies, mode and counts."""
    x = valley_start()
    surface = counting(curved_valley)
    result = artn_search(
        x, OPEN_CELL, NO_PBC, np.ones(4, bool), 0, VALLEY_SEED, surface, **PUSH
    )
    assert result.success and result.reason is None

    # Atom 0 crosses the saddle at the origin; the others stay in their wells.
    np.testing.assert_allclose(result.saddle_positions[0], [0.0, 0.0, 0.0], atol=2e-3)
    np.testing.assert_allclose(result.saddle_positions[1:], x[1:], atol=1e-5)
    assert result.e_saddle == pytest.approx(VALLEY_BARRIER, abs=1e-5)
    assert result.curvature < 0
    assert result.curvature == pytest.approx(VALLEY_CURVATURE, abs=0.05)
    _, forces = curved_valley(result.saddle_positions)
    assert np.max(np.abs(forces)) < 1e-3

    # The mode is x on atom 0, pointing from the start to the saddle.
    assert np.linalg.norm(result.mode) == pytest.approx(1.0)
    assert result.mode[0, 0] < -0.999

    # One minimum is the start's well, the other the far well, in either order
    # (pARTn's: minimum 1 lies along its last eigenvector, whose sign varies).
    start_first = result.delr1 < result.delr2
    near, far = (1, 2) if start_first else (2, 1)
    np.testing.assert_allclose(
        getattr(result, f"min{near}_positions")[0], [1.0, 0.5, 0.0], atol=2e-3
    )
    np.testing.assert_allclose(
        getattr(result, f"min{far}_positions")[0], [-1.0, 0.5, 0.0], atol=2e-3
    )
    assert result.e_min1 == pytest.approx(0.0, abs=1e-5)
    assert result.e_min2 == pytest.approx(0.0, abs=1e-5)
    assert min(result.delr1, result.delr2) < 0.01
    assert max(result.delr1, result.delr2) == pytest.approx(2.0, abs=5e-3)

    # Every callback call is counted, up to the saddle and in total.
    assert 0 < result.force_calls_saddle < result.force_calls_total
    assert result.force_calls_total == surface.count
    assert result.seed == VALLEY_SEED
    assert_no_nan(result)


@pytest.mark.parametrize("convergence_property", ["maxval", "norm"])
def test_both_convergence_properties(convergence_property: str) -> None:
    """The saddle and both minima meet the chosen force measure."""
    result = valley_search(valley_start(), convergence_property=convergence_property)
    assert result.success
    for positions in (
        result.saddle_positions,
        result.min1_positions,
        result.min2_positions,
    ):
        _, forces = curved_valley(positions)
        if convergence_property == "maxval":
            assert np.max(np.abs(forces)) < 1e-3
        else:
            assert np.linalg.norm(forces) < 1e-3


def test_same_seed_same_result() -> None:
    """A repeated seed repeats the search exactly; another seed pushes elsewhere."""
    first = valley_search(valley_start(), seed=VALLEY_SEED)
    again = valley_search(valley_start(), seed=VALLEY_SEED)
    for name, value in first._asdict().items():
        if isinstance(value, np.ndarray):
            assert np.array_equal(value, getattr(again, name)), name
        else:
            assert value == getattr(again, name), name

    other = valley_search(valley_start(), seed=1)
    assert other.success
    # Seed 1 pushes atom 0 to the z saddle instead (barrier 1 eV).
    assert other.e_saddle == pytest.approx(1.0, abs=1e-5)


def test_frozen_atoms_never_move_or_steer() -> None:
    """A frozen atom keeps its position, and its force changes nothing."""
    movable = np.array([True, False, True, True])
    on_site = valley_start()
    displaced = on_site.copy()
    displaced[1] += [0.2, -0.2, 0.1]  # off its minimum: a large force
    seen = []

    def surface(positions: np.ndarray) -> tuple[float, np.ndarray]:
        seen.append(positions[1].copy())
        return curved_valley(positions)

    results = []
    for x in (on_site, displaced):
        seen.clear()
        # The rad push reaches atom 1 (3 Å away), so only the mask keeps it still.
        result = artn_search(
            x,
            OPEN_CELL,
            NO_PBC,
            movable,
            0,
            VALLEY_SEED,
            surface,
            push_dist_thr=3.5,
            push_step_size=0.3,
        )
        assert result.success
        assert all(np.array_equal(p, x[1]) for p in seen)
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
    assert a.force_calls_total == b.force_calls_total


# Push on the central atom only, or rad over two atoms (atoms 0 and 3, 2.5 Å apart);
# the second order swaps atoms 0 and 3, so their draws would swap if handed out
# in atom order.
@pytest.mark.parametrize(
    ("push", "order"),
    [
        (PUSH, [4, 0, 3, 5, 1, 2]),
        (
            dict(push_mode="rad", push_dist_thr=2.6, push_step_size=0.3),
            [4, 3, 5, 0, 1, 2],
        ),
    ],
)
def test_shuffled_atom_order_gives_the_same_event(push: dict, order: list) -> None:
    """Reordering the atoms reorders the event and changes nothing else.

    The push and the first Lanczos vector draw on their atoms in order of the
    atoms' offsets from the central atom, so both are the same; only the order
    of summation differs, which is far below the force threshold.
    """
    sites = [(2, 1, 1), (0, 0, 0), (1, 3, 2), (2, 1, 2), (3, 2, 0), (0, 2, 3)]
    x = np.array([[SHIFT + GRID * i, SHIFT + GRID * j, GRID * k] for i, j, k in sites])
    central = 3
    cell = np.diag([10.0, 10.0, 10.0])
    pbc = np.array([True, True, True])
    movable = np.ones(len(x), bool)
    seen = []

    def surface(positions: np.ndarray) -> tuple[float, np.ndarray]:
        seen.append(positions.copy())
        return periodic_wells(positions)

    result = artn_search(x, cell, pbc, movable, central, 5, surface, **push)
    assert result.success
    first_push = seen[1] - seen[0]

    order = np.array(order)
    seen.clear()
    shuffled = artn_search(
        x[order],
        cell,
        pbc,
        movable[order],
        int(np.flatnonzero(order == central)[0]),
        5,
        surface,
        **push,
    )
    assert shuffled.success
    np.testing.assert_array_equal(seen[1] - seen[0], first_push[order])
    for name in ("saddle_positions", "min1_positions", "min2_positions"):
        difference = _minimum_image(
            getattr(shuffled, name) - getattr(result, name)[order], cell, pbc
        )
        assert np.max(np.abs(difference)) < 1e-6, name
    for name in ("e_saddle", "e_min1", "e_min2", "delr1", "delr2"):
        assert getattr(shuffled, name) == pytest.approx(getattr(result, name), abs=1e-9)
    assert shuffled.curvature == pytest.approx(result.curvature, abs=1e-6)

    # The saddle is an analytic one: one atom half a grid step from its well.
    moved = np.linalg.norm(
        _minimum_image(result.saddle_positions - x, cell, pbc), axis=1
    )
    assert np.sum(moved > 0.5) == 1
    assert moved.max() == pytest.approx(GRID / 2, abs=2e-3)
    # Barrier 2 a along the axis crossed (0.5 eV in x, 1.2 eV in y or z),
    # curvature -a k² there.
    assert result.e_saddle == pytest.approx(
        WELL_BARRIER if result.e_saddle < 1.0 else 1.2, abs=1e-5
    )
    assert result.curvature == pytest.approx(
        -0.5 * result.e_saddle * (2 * math.pi / GRID) ** 2, abs=0.05
    )


def test_hop_across_the_periodic_boundary() -> None:
    """The hop crosses x = 0; z is not periodic and is never wrapped."""
    cell = np.diag([10.0, 10.0, 20.0])
    pbc = np.array([True, True, False])
    # Atom 0 sits in the well at x = 0.5, above the cell in z; atom 1 below it.
    x = np.array([[SHIFT, SHIFT, 22.5], [5.5, 5.5, -2.5], [3.0, 8.0, 5.0]])
    seen_x = []

    def surface(positions: np.ndarray) -> tuple[float, np.ndarray]:
        seen_x.append(positions[0, 0])
        return periodic_wells(positions)

    result = artn_search(x, cell, pbc, np.ones(3, bool), 0, 4, surface, **PUSH)
    assert result.success
    # The search kept its own unwrapped coordinates.
    assert min(seen_x) < 0.0

    # Wrapped results: saddle at x = 0.5 - 1.25 + 10, far minimum at 0.5 - 2.5 + 10.
    np.testing.assert_allclose(
        result.saddle_positions[0], [9.25, SHIFT, 22.5], atol=2e-3
    )
    near, far = (1, 2) if result.delr1 < result.delr2 else (2, 1)
    np.testing.assert_allclose(
        getattr(result, f"min{near}_positions")[0], [SHIFT, SHIFT, 22.5], atol=2e-3
    )
    np.testing.assert_allclose(
        getattr(result, f"min{far}_positions")[0], [8.0, SHIFT, 22.5], atol=2e-3
    )
    for positions in (
        result.saddle_positions,
        result.min1_positions,
        result.min2_positions,
    ):
        assert np.all((positions[:, :2] >= 0.0) & (positions[:, :2] < 10.0))
        assert positions[1, 2] == pytest.approx(-2.5, abs=1e-6)
        assert positions[0, 2] == pytest.approx(22.5, abs=2e-3)
    # Minimum-image distances: the hop is 2.5 Å, not 7.5 Å.
    assert min(result.delr1, result.delr2) < 0.01
    assert max(result.delr1, result.delr2) == pytest.approx(GRID, abs=5e-3)
    assert result.e_saddle == pytest.approx(WELL_BARRIER, abs=1e-5)
    assert result.curvature == pytest.approx(WELL_CURVATURE, abs=0.05)


def test_block_order_push_then_perp_then_lanczos() -> None:
    """Pushes of push_step_size, nperp FIRE steps after each, then Lanczos probes."""
    x = valley_start()
    seen = []

    def surface(positions: np.ndarray) -> tuple[float, np.ndarray]:
        seen.append(positions.copy())
        return curved_valley(positions)

    artn_search(
        x,
        OPEN_CELL,
        NO_PBC,
        np.ones(4, bool),
        0,
        VALLEY_SEED,
        surface,
        ninit=1,
        nperp=2,
        nevalf_max=6,
        **PUSH,
    )
    # Calls: start, after the push, two perpendicular steps, two Lanczos probes.
    assert len(seen) == 6
    push = seen[1] - seen[0]
    assert np.linalg.norm(push) == pytest.approx(0.3)
    assert np.array_equal(push[1:], np.zeros((3, 3)))  # list push: atom 0 only
    for step in (seen[2] - seen[1], seen[3] - seen[2]):
        assert abs(np.vdot(step, push)) < 1e-12 * np.linalg.norm(step)
    for probe in seen[4:]:
        assert np.linalg.norm(probe - seen[3]) == pytest.approx(0.0005)

    # ninit = 0: Lanczos at the start.
    seen.clear()
    artn_search(
        x, OPEN_CELL, NO_PBC, np.ones(4, bool), 0, 1, surface, ninit=0, nevalf_max=3
    )
    assert np.linalg.norm(seen[1] - seen[0]) == pytest.approx(0.0005)


def test_rad_push_covers_the_movable_atoms_in_its_radius() -> None:
    """The rad push moves atoms 0 and 1 (3 Å apart), the list push atom 0 only."""
    x = valley_start()
    for mode, radius, moved in (("rad", 3.5, [0, 1]), ("list", 3.5, [0])):
        seen = []

        def surface(
            positions: np.ndarray, seen: list = seen
        ) -> tuple[float, np.ndarray]:
            seen.append(positions.copy())
            return curved_valley(positions)

        artn_search(
            x,
            OPEN_CELL,
            NO_PBC,
            np.ones(4, bool),
            0,
            2,
            surface,
            push_mode=mode,
            push_dist_thr=radius,
            nevalf_max=2,
        )
        push = seen[1] - seen[0]
        assert np.linalg.norm(push) == pytest.approx(0.4)
        assert np.flatnonzero(np.linalg.norm(push, axis=1) > 0).tolist() == moved


def test_force_call_budget_ends_the_search() -> None:
    """nevalf_max callback calls, then a failure that counts all of them."""
    surface = counting(curved_valley)
    result = artn_search(
        valley_start(),
        OPEN_CELL,
        NO_PBC,
        np.ones(4, bool),
        0,
        VALLEY_SEED,
        surface,
        nevalf_max=40,
        **PUSH,
    )
    assert not result.success and result.reason == "max_force_calls"
    assert result.force_calls_saddle == result.force_calls_total == 40 == surface.count
    assert result.saddle_positions is None
    assert_no_nan(result)

    # After the saddle the counts split, and the saddle is kept.
    full = valley_search(valley_start())
    surface = counting(curved_valley)
    result = artn_search(
        valley_start(),
        OPEN_CELL,
        NO_PBC,
        np.ones(4, bool),
        0,
        VALLEY_SEED,
        surface,
        nevalf_max=full.force_calls_saddle + 5,
        **PUSH,
    )
    assert result.reason == "max_force_calls"
    assert result.force_calls_saddle == full.force_calls_saddle
    assert result.force_calls_total == surface.count == full.force_calls_saddle + 5
    assert np.array_equal(result.saddle_positions, full.saddle_positions)


def test_lost_curvature_out_of_the_basin_is_a_failure() -> None:
    """With no new chance, a positive eigenvalue out of the basin ends the search."""
    surface = counting(curved_valley)
    result = artn_search(
        valley_start(), OPEN_CELL, NO_PBC, np.ones(4, bool), 0, 4, surface, **PUSH
    )
    assert not result.success and result.reason == "eigenvalue_lost"
    assert result.force_calls_total == surface.count
    assert_no_nan(result)


def test_other_failures_return_a_reason() -> None:
    """A frozen central atom gives nothing to push; a NaN start fails at once."""
    movable = np.array([False, True, True, True])
    frozen_centre = artn_search(
        valley_start(), OPEN_CELL, NO_PBC, movable, 0, 1, curved_valley, **PUSH
    )
    assert frozen_centre.reason == "empty_push"
    assert frozen_centre.force_calls_total == 1

    def nan_surface(positions: np.ndarray) -> tuple[float, np.ndarray]:
        return math.nan, np.zeros_like(positions)

    result = artn_search(
        valley_start(), OPEN_CELL, NO_PBC, np.ones(4, bool), 0, 1, nan_surface, **PUSH
    )
    assert result.reason == "non_finite_force" and result.e_start is None
    assert_no_nan(result)


def test_every_reduction_goes_through_the_vector_space() -> None:
    """A vector space that counts its calls sees them, and the result is unchanged."""

    class Counting(NumpyVectorSpace):
        calls = {"dot": 0, "norm": 0, "max_abs": 0}

        def dot(self, a: np.ndarray, b: np.ndarray) -> float:
            self.calls["dot"] += 1
            return super().dot(a, b)

        def max_abs(self, a: np.ndarray) -> float:
            self.calls["max_abs"] += 1
            return super().max_abs(a)

    space = Counting()
    counted = valley_search(valley_start(), vectors=space)
    plain = valley_search(valley_start())
    assert space.calls["dot"] > 100 and space.calls["max_abs"] > 100
    for name, value in plain._asdict().items():
        if isinstance(value, np.ndarray):
            assert np.array_equal(value, getattr(counted, name)), name
        else:
            assert value == getattr(counted, name), name
