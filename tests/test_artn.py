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
        "nsmooth",
        "alpha_mix_cr",
        "nnewchance",
        "nperp",
        "nperp_limitation",
        "forc_thr",
        "convergence_property",
        "push_over",
        "nevalf_max",
        "dmax",
    ):
        default = params[name].default
        if isinstance(default, tuple):
            default = list(default)
        assert default == getattr(defaults, name), name


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


def test_return_to_saddle_uses_minimum_force_and_counts_the_callback() -> None:
    """The opposite push follows one restart with min1's force and FIRE's time step."""
    moves = []
    states = []

    def observe(move: str, state: tuple | None) -> None:
        moves.append(move)
        states.append(state)

    x0 = valley_start()
    full = valley_search(x0, observe=observe)
    assert full.success
    first_over, second_over = (i for i, move in enumerate(moves) if move == "over")
    i_return = second_over - 1
    assert first_over < i_return and moves[i_return] == "relax"

    # On this unbounded analytic surface the restart is below dmax.
    _, force = curved_valley(full.min1_positions)
    dx = artn._DT0**2 * artn._FTM2V * force  # unit masses
    assert np.max(np.abs(dx)) > 1e-10
    dt = states[i_return - 1][0] * 0.5
    assert states[i_return][0:2] == pytest.approx((dt, 0.1))
    np.testing.assert_allclose(states[i_return][2], dx / dt, rtol=1e-12, atol=1e-13)

    calls = 0

    def surface(x: np.ndarray) -> tuple[float, np.ndarray]:
        nonlocal calls
        i = calls
        calls += 1
        if i == i_return:
            np.testing.assert_allclose(
                x, full.saddle_positions + dx, rtol=0, atol=1e-13
            )
            # block_pushover restores the saddle energy and ignores this force.
            return artn._ENERGY_LIMIT * 2, np.full_like(x, 50.0)
        return curved_valley(x)

    result = artn_search(
        x0, OPEN_CELL, NO_PBC, np.ones(len(x0), bool), 0, VALLEY_SEED, surface, **PUSH
    )
    assert (
        result.success and result.force_calls_total == calls == full.force_calls_total
    )
    np.testing.assert_array_equal(result.min2_positions, full.min2_positions)
    capped = valley_search(x0, nevalf_max=i_return + 1)
    assert (
        capped.reason == "max_force_calls" and capped.force_calls_total == i_return + 1
    )
    assert capped.min2_positions is None
    np.testing.assert_array_equal(capped.min1_positions, full.min1_positions)


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


def test_injected_first_lanczos_vector_documents_mask_exception() -> None:
    """Both parameter entries identify the replay exception to the movable mask."""
    doc = inspect.getdoc(artn_search)
    movable = doc.split("movable : ", 1)[1].split("central_atom : ", 1)[0]
    vector = doc.split("first_lanczos_vector : ", 1)[1].split(
        "new_chance_pushes : ", 1
    )[0]
    assert "first_lanczos_vector" in movable
    assert all(term in vector.lower() for term in ("mask", "frozen", "probe", "replay"))


def test_injected_first_lanczos_vector_moves_a_frozen_atom() -> None:
    """Replay keeps the buffer component of the first probe."""
    start = valley_start()
    movable = np.array([True, False, True, True])
    direction = np.zeros_like(start)
    direction[1, 0] = 1.0
    positions = []

    def surface(x: np.ndarray) -> tuple[float, np.ndarray]:
        positions.append(x.copy())
        return curved_valley(x)

    artn_search(
        start,
        OPEN_CELL,
        NO_PBC,
        movable,
        0,
        VALLEY_SEED,
        surface,
        ninit=0,
        nevalf_max=2,
        first_lanczos_vector=direction,
    )
    np.testing.assert_allclose(
        positions[1] - positions[0], direction * artn._DISPLACEMENT_FACTOR * 0.0005
    )


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
    # Probes are displacements, landed at pARTn's fraction of lanczos_disp.
    for probe in seen[4:]:
        assert np.linalg.norm(probe - seen[3]) == pytest.approx(
            artn._DISPLACEMENT_FACTOR * 0.0005, rel=1e-9
        )

    # ninit = 0: Lanczos at the start.
    seen.clear()
    artn_search(
        x, OPEN_CELL, NO_PBC, np.ones(4, bool), 0, 1, surface, ninit=0, nevalf_max=3
    )
    assert np.linalg.norm(seen[1] - seen[0]) == pytest.approx(0.0005)


def quadratic_lanczos(**kwargs: object) -> tuple[list, list, np.ndarray]:
    """One Lanczos block at the start of a quadratic surface (ninit = 0).

    Returns the positions of every call, the move and state the core showed
    before each call, and the Hessian. The callback stops the search at the
    first call after the block.
    """
    rng = np.random.default_rng(3)
    a = rng.random((12, 12)) - 0.5
    hessian = a + a.T + np.diag(np.linspace(-1.0, 4.0, 12))
    x0 = valley_start()
    seen, moves = [], []

    def surface(x: np.ndarray) -> tuple[float, np.ndarray]:
        if moves[-1][0] not in ("start", "lanczos"):
            raise StopIteration
        seen.append(x.copy())
        d = (x - x0).ravel()
        return 0.5 * d @ hessian @ d, -(hessian @ d).reshape(-1, 3)

    def observe(move: str, state: object) -> None:
        moves.append((move, state))

    options = dict(
        ninit=0,
        lanczos_max_size=12,
        first_lanczos_vector=rng.random((4, 3)) - 0.5,
        observe=observe,
    )
    with pytest.raises(StopIteration):
        artn_search(
            x0,
            OPEN_CELL,
            NO_PBC,
            np.ones(4, bool),
            0,
            0,
            surface,
            **{**options, **kwargs},
        )
    return seen, moves, hessian


def test_lanczos_block_on_a_quadratic_surface() -> None:
    """The lowest eigenpair, as pARTn's probes give it, then back to the start."""
    seen, moves, hessian = quadratic_lanczos(lanczos_eval_conv_thr=0.0)
    # The start, 12 probes (3N, the most) and the move back.
    assert [m for m, _ in moves] == ["start"] + ["lanczos"] * 13 + ["eigen"]
    assert len(seen) == 14
    assert np.max(np.abs(seen[-1] - seen[0])) < 1e-14
    for probe in seen[1:-1]:
        assert np.linalg.norm(probe - seen[0]) == pytest.approx(
            artn._DISPLACEMENT_FACTOR * 0.0005, rel=1e-9
        )
    values, vectors = np.linalg.eigh(hessian)
    lowest, eigvec = moves[-2][1]
    # The probe lands at kappa * lanczos_disp, but pARTn divides by
    # lanczos_disp, so its eigenvalue is kappa times the true one.
    assert lowest == pytest.approx(artn._DISPLACEMENT_FACTOR * values[0], rel=1e-9)
    assert lowest != pytest.approx(values[0], rel=1e-8)
    assert abs(np.vdot(eigvec.ravel(), vectors[:, 0])) == pytest.approx(1.0, abs=1e-9)
    # Pointed against the force at the last probe.
    d = (seen[-2] - seen[0]).ravel()
    assert np.vdot(-(hessian @ d), eigvec.ravel()) < 0


def test_first_lanczos_block_starts_from_partns_unset_eigenvalue() -> None:
    """HUGE(1.0_DP) before the first block: |(a - HUGE)/HUGE| is 1 at probe 1."""
    # One probe and the move back, then the block's outcome.
    _, moves, _ = quadratic_lanczos(lanczos_min_size=1, lanczos_eval_conv_thr=1.0)
    assert [m for m, _ in moves][:3] == ["start", "lanczos", "lanczos"]
    assert moves[3][0] != "lanczos"
    _, moves, _ = quadratic_lanczos(lanczos_min_size=1, lanczos_eval_conv_thr=0.99)
    assert [m for m, _ in moves][:4] == ["start"] + ["lanczos"] * 3


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


# Seed whose search loses its negative curvature out of the basin.
LOST_SEED = 166


def test_lost_curvature_out_of_the_basin_is_a_failure() -> None:
    """With no new chance, a positive eigenvalue out of the basin ends the search."""
    surface = counting(curved_valley)
    result = artn_search(
        valley_start(),
        OPEN_CELL,
        NO_PBC,
        np.ones(4, bool),
        0,
        LOST_SEED,
        surface,
        **PUSH,
    )
    assert not result.success and result.reason == "eigenvalue_lost"
    assert result.force_calls_total == surface.count
    assert_no_nan(result)


def moves_and_positions(seed: int, **kwargs: object) -> tuple[list, list, DimerResult]:
    """Run ARTn on the curved valley; return each call's move and positions."""
    moves, seen = [], []

    def surface(x: np.ndarray) -> tuple[float, np.ndarray]:
        seen.append(x.copy())
        return curved_valley(x)

    result = artn_search(
        valley_start(),
        OPEN_CELL,
        NO_PBC,
        np.ones(4, bool),
        0,
        seed,
        surface,
        observe=lambda move, state: moves.append(move),
        **{**PUSH, **kwargs},
    )
    return moves, seen, result


def runs(moves: list, name: str) -> list:
    """Lengths of the runs of ``name`` in ``moves``."""
    lengths, count = [], 0
    for move in moves + [None]:
        if move == name:
            count += 1
        elif count:
            lengths.append(count)
            count = 0
    return lengths


def test_a_new_chance_pushes_again_from_the_basin() -> None:
    """After the lost curvature: one more push and basin relaxation, then Lanczos."""
    lost, _, _ = moves_and_positions(LOST_SEED)
    moves, seen, result = moves_and_positions(LOST_SEED, nnewchance=1)
    assert result.success and result.e_saddle == pytest.approx(VALLEY_BARRIER, abs=1e-5)
    # The same search up to the lost block's last probe, then the move back and
    # a new push in place of the failure.
    k = len(lost)
    assert moves[:k] == lost
    assert moves[k : k + 2] == ["lanczos", "init"]
    assert moves[k + 2 : k + 5] == ["perp"] * 3 and moves[k + 5] == "lanczos"
    push = seen[k + 1] - seen[k]
    assert np.linalg.norm(push) == pytest.approx(0.3 * artn._DISPLACEMENT_FACTOR)
    assert np.array_equal(push[1:], np.zeros((3, 3)))  # the start's pushed atom only

    # A push given for the new chance is used as given.
    given = np.zeros((4, 3))
    given[0] = [0.0, 0.1, -0.2]
    moves, seen, _ = moves_and_positions(
        LOST_SEED, nnewchance=1, new_chance_pushes=[given], nevalf_max=k + 2
    )
    np.testing.assert_allclose(
        seen[k + 1] - seen[k], artn._DISPLACEMENT_FACTOR * given, rtol=0, atol=1e-15
    )


def test_recorded_draws_only_makes_no_generator(monkeypatch: pytest.MonkeyPatch) -> None:
    """With recorded draws only, a new chance with no push given ends the search."""
    lost, seen, _ = moves_and_positions(LOST_SEED)
    probe = lost.index("lanczos")
    recorded = dict(
        initial_push=(seen[1] - seen[0]) / artn._DISPLACEMENT_FACTOR,
        first_lanczos_vector=seen[probe] - seen[probe - 1],
        recorded_draws_only=True,
        nnewchance=1,
    )
    with pytest.raises(ValueError, match="recorded_draws_only"):
        valley_search(valley_start(), recorded_draws_only=True, nevalf_max=2)

    def no_generator(*args: object, **kwargs: object) -> None:
        raise AssertionError("a random generator was made")

    monkeypatch.setattr(np.random, "PCG64", no_generator)
    moves, _, result = moves_and_positions(LOST_SEED, **recorded)
    assert result.reason == "recorded_draws_exhausted" and moves == lost
    given = np.zeros((4, 3))
    given[0] = [0.0, 0.1, -0.2]
    moves, _, result = moves_and_positions(
        LOST_SEED, new_chance_pushes=[given], **recorded
    )
    assert result.reason != "recorded_draws_exhausted" and len(moves) > len(lost)


@pytest.mark.parametrize("limits", [[-2, 4], [4, -2, 8], [4, -2], [-1, -2]])
def test_mixed_default_perp_limits_raise(limits: list[int]) -> None:
    """A mixed list is a configuration error even when -2 is the last entry."""
    with pytest.raises(ValueError, match=r"nperp_limitation.*-2"):
        valley_search(valley_start(), nperp=3, nperp_limitation=limits, nevalf_max=2)


def test_all_default_perp_limits_keep_partn_defaults() -> None:
    """A list of only -2 entries selects the defaults with a basin override."""
    assert artn.perp_limits(3, [-2, -2]) == [3, 8, 12, 16, -1]


def test_out_of_basin_perpendicular_limits_follow_partns_list() -> None:
    """3 FIRE steps after each basin push; out of it 3, then 4, ... at most."""
    moves, _, result = moves_and_positions(
        VALLEY_SEED, nsmooth=0, nperp_limitation=[4, 8, 12, 16, -1]
    )
    assert result.success
    first_eigen = moves.index("eigen")
    basin = runs(moves[:first_eigen], "perp")
    assert basin and all(n == 3 for n in basin)
    # On this surface each relaxation runs to its limit, the last to the saddle.
    climb = runs(moves[first_eigen:], "perp")
    assert climb[:-1] == [3, 4, 8, 12] and climb[-1] <= 16

    # pARTn's list [3, 1, 1, -1, -1]: the basin's 3 first, then 1, 1, no limit.
    moves, _, result = moves_and_positions(
        VALLEY_SEED, nsmooth=0, nperp_limitation=[1, 1, -1]
    )
    assert result.success
    climb = runs(moves[moves.index("eigen") :], "perp")
    assert climb[:3] == [3, 1, 1] and max(climb[3:]) > 3


def test_the_first_pushes_out_of_the_basin_are_blended() -> None:
    """With nsmooth, the first eigen moves are not along the eigenvector."""

    def eigen_moves(**kwargs: object) -> tuple[list, list]:
        moves, states, seen = [], [], []

        def surface(x: np.ndarray) -> tuple[float, np.ndarray]:
            seen.append(x.copy())
            return curved_valley(x)

        artn_search(
            valley_start(),
            OPEN_CELL,
            NO_PBC,
            np.ones(4, bool),
            0,
            VALLEY_SEED,
            surface,
            observe=lambda move, state: (moves.append(move), states.append(state)),
            **{**PUSH, **kwargs},
        )
        cosines, eigvec = [], None
        for k, move in enumerate(moves):
            if move == "lanczos":
                eigvec = states[k][1]
            if move == "eigen":
                d = (seen[k] - seen[k - 1]).ravel()
                e = eigvec.ravel()
                cosines.append(abs(d @ e) / (np.linalg.norm(d) * np.linalg.norm(e)))
        return cosines

    plain = eigen_moves(nsmooth=0)
    assert plain and all(c == pytest.approx(1.0, abs=1e-12) for c in plain)
    blended = eigen_moves(nsmooth=3)
    assert len(blended) > 3
    assert all(c < 0.999 for c in blended[:3])
    assert all(c == pytest.approx(1.0, abs=1e-12) for c in blended[3:])


def test_the_call_at_nevalf_max_fails_even_at_the_saddle() -> None:
    """As in pARTn, call number nevalf_max fails, whatever it found."""
    full = valley_search(valley_start())
    result = valley_search(valley_start(), nevalf_max=full.force_calls_saddle)
    assert result.reason == "max_force_calls" and result.saddle_positions is None
    assert result.force_calls_total == full.force_calls_saddle
    result = valley_search(valley_start(), nevalf_max=full.force_calls_saddle + 1)
    assert result.force_calls_saddle == full.force_calls_saddle
    assert result.saddle_positions is not None


def test_atoms_are_wrapped_only_when_lammps_would_rebuild() -> None:
    """The alignment test's positions: wrapped at the first call, then only once
    an atom has moved more than half the 2.0 Å skin since the last wrap."""
    cell = np.diag([10.0, 10.0, 10.0])
    held = artn._lammps_images(cell, np.array([True, True, False]), NumpyVectorSpace())
    x = np.array([[9.9, 5.0, 12.0], [5.0, 5.0, 5.0], [-0.5, 5.0, 5.0]])
    # z is not periodic; the atom below x = 0 is wrapped at the first call.
    np.testing.assert_array_equal(held(x), [[9.9, 5.0, 12.0], [5.0, 5.0, 5.0], [9.5, 5.0, 5.0]])
    # Atom 0 crosses x = 10 and atom 1 moves exactly half the skin: no rebuild.
    x = x + [[0.2, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 0.0]]
    np.testing.assert_allclose(held(x)[:2], [[10.1, 5.0, 12.0], [5.0, 6.0, 5.0]])
    # Atom 1 passes half the skin: every atom is wrapped.
    x = x + [[0.0, 0.0, 0.0], [0.0, 0.01, 0.0], [0.0, 0.0, 0.0]]
    np.testing.assert_allclose(
        held(x), [[0.1, 5.0, 12.0], [5.0, 6.01, 5.0], [9.5, 5.0, 5.0]], atol=1e-12
    )
    # Atom 0 moves back across the face and keeps its image, outside the cell.
    x = x - [[0.15, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]]
    np.testing.assert_allclose(held(x)[0], [-0.05, 5.0, 12.0], atol=1e-12)


def test_positions_the_engine_reports_go_to_the_alignment_test_only() -> None:
    """A third value from the callback changes nothing when it equals the positions."""

    def reporting(x: np.ndarray) -> tuple:
        energy, forces = curved_valley(x)
        return energy, forces, x

    plain = valley_search(valley_start())
    reported = artn_search(
        valley_start(),
        OPEN_CELL,
        NO_PBC,
        np.ones(4, bool),
        0,
        VALLEY_SEED,
        reporting,
        **PUSH,
    )
    for name, value in plain._asdict().items():
        if isinstance(value, np.ndarray):
            assert np.array_equal(value, getattr(reported, name)), name
        else:
            assert value == getattr(reported, name), name


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


def test_fire_uses_each_atoms_mass() -> None:
    """The first PERP kick under equal forces gives each atom its own acceleration."""
    positions = []
    velocities = []
    masses = np.array([1.0, 4.0])
    force = np.array([[0.0, 1.0, 0.0], [0.0, 1.0, 0.0]])

    def surface(x: np.ndarray) -> tuple[float, np.ndarray]:
        positions.append(x.copy())
        return -float(np.sum(force * x)), force

    def observe(move: str, state: tuple) -> None:
        if move == "perp":
            velocities.append(state[2])

    artn_search(
        np.zeros((2, 3)),
        OPEN_CELL,
        NO_PBC,
        np.ones(2, bool),
        0,
        3,
        surface,
        masses=masses,
        initial_push=np.array([[1, 0, 0], [0, 0, 0]]),
        ninit=1,
        nperp=1,
        nevalf_max=3,
        observe=observe,
        **PUSH,
    )
    kick = artn._DT0 * artn._FTM2V * force / masses[:, None]
    np.testing.assert_allclose(velocities, [kick], rtol=1e-13, atol=0)
    np.testing.assert_allclose(
        positions[2] - positions[1], artn._DT0 * kick, rtol=1e-13, atol=0
    )


@pytest.mark.parametrize("mix", [0.2, 0.8])
def test_new_chance_mixes_the_seeded_draw(mix: float) -> None:
    """Rebuild the new push from its seed and the force at the lost block's last probe."""
    moves, _, _ = moves_and_positions(LOST_SEED)
    k = len(moves)
    moves, seen, _ = moves_and_positions(
        LOST_SEED, nnewchance=1, alpha_mix_cr=mix, nevalf_max=k + 2
    )
    assert moves[k : k + 2] == ["lanczos", "init"]
    rng = np.random.default_rng(LOST_SEED)
    initial = 0.5 - rng.random(3)  # list push on the central atom
    initial /= np.linalg.norm(initial)
    rng.random(seen[0].shape)  # first Lanczos vector
    draw = initial * (0.5 - rng.random(3))
    direction = (1 - mix) * initial + mix * draw / np.linalg.norm(draw)
    _, probe_force = curved_valley(seen[k - 1])
    sign = -np.sign(np.dot(probe_force[0], direction))
    expected = np.zeros_like(seen[k])
    expected[0] = sign * PUSH["push_step_size"] * direction / np.linalg.norm(direction)
    np.testing.assert_allclose(
        seen[k + 1] - seen[k], artn._DISPLACEMENT_FACTOR * expected, rtol=0, atol=1e-15
    )


@pytest.mark.parametrize("coupled", [False, True], ids=["zero-first", "tiny-second"])
def test_lanczos_continues_after_small_residual(coupled: bool) -> None:
    """A zero first residual or a tiny second one still produces another finite probe."""
    coupling = 1e-11
    hessian = np.diag([1.0, 3.0, 2.0])
    if coupled:
        hessian[0, 1] = hessian[1, 0] = 1.0
        hessian[1, 2] = hessian[2, 1] = coupling
    seen = []
    moves = []

    def surface(x: np.ndarray) -> tuple[float, np.ndarray]:
        seen.append(x.copy())
        gradient = x @ hessian
        return float(np.sum(x * gradient) / 2), -gradient

    disp = 0.01
    artn_search(
        np.zeros((2, 3)),
        OPEN_CELL,
        NO_PBC,
        np.ones(2, bool),
        0,
        5,
        surface,
        ninit=0,
        lanczos_disp=disp,
        lanczos_min_size=4,
        lanczos_max_size=4,
        lanczos_eval_conv_thr=0.0,
        nevalf_max=4,
        first_lanczos_vector=np.array([[1, 0, 0], [0, 0, 0]]),
        observe=lambda move, state: moves.append(move),
    )
    assert moves == ["start", "lanczos", "lanczos", "lanczos"]
    expected = np.zeros((2, 3))
    if coupled:
        # Probe two follows +y. Its remaining z component in Ry/bohr² is
        # below the breakdown threshold, so probe three moves by this unscaled residual.
        residual = artn._DISPLACEMENT_FACTOR * coupling / artn._AU2H
        expected[0, 2] = artn._DISPLACEMENT_FACTOR * disp * residual
        assert seen[2][0, 1] == pytest.approx(artn._DISPLACEMENT_FACTOR * disp)
    else:
        np.testing.assert_array_equal(seen[2], expected)
    np.testing.assert_allclose(seen[3], expected, rtol=1e-12, atol=0)


def test_result_positions_keep_box_half_open_and_interior_points_exact() -> None:
    """Repair misrounded periodic faces without changing valid or nonperiodic entries."""
    from pykmc.dimer import _wrap

    cell = np.diag([28.55312, 0.11, 9.0])
    pbc = np.array([True, True, False])
    lx, ly, _ = cell.diagonal()
    inside_x, inside_y = np.nextafter([lx, ly], 0.0)
    outside_x = np.nextafter(lx, np.inf)
    x = np.array(
        [
            [-1e-17, inside_y, -1e-17],
            [lx, ly, 9.0],
            [inside_x, -ly, -12.0],
            [outside_x, 0.037, 11.0],
        ]
    )
    original = x.copy()
    old = _wrap(x, cell, pbc)
    assert old[0, 0] == lx
    wrapped = artn._wrap_result_positions(x, cell, pbc)
    expected = [
        [0.0, inside_y, -1e-17],
        [0.0, 0.0, 9.0],
        [inside_x, 0.0, -12.0],
        [outside_x - lx, 0.037, 11.0],
    ]
    np.testing.assert_array_equal(wrapped, expected)
    np.testing.assert_array_equal(
        artn._wrap_result_positions(wrapped, cell, pbc), wrapped
    )
    np.testing.assert_array_equal(x, original)
    ordinary = (old >= 0) & (old < cell.diagonal())
    np.testing.assert_array_equal(wrapped[ordinary], old[ordinary])
    np.testing.assert_array_equal(
        artn._wrap_result_positions(x, cell, ~np.ones(3, bool)), original
    )
    skew = np.array([[5.0, 0, 0], [1.5, 5.0, 0], [0.2, 0.3, 9.0]])
    np.testing.assert_array_equal(
        artn._wrap_result_positions(x, skew, pbc), _wrap(x, skew, pbc)
    )
