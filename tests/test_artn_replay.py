"""Step replay of pARTn v1.1.0 recordings through the ARTn core (no LAMMPS).

The recorded tests replay issue 02's eight recordings, cut to the
initial-push window, from ``tests/data/artn_step_replay`` (see its README).
``PYKMC_ARTN_RECORDINGS`` names a folder of full recordings to read instead
(611 MB, outside git); the Lanczos block tests need them and skip without.
The synthetic tests check the harness itself on a recording the core makes on
an analytic surface, with repeat evaluations added as pARTn's FIRE makes them.
"""

import math
import os
from pathlib import Path

import numpy as np
import pytest

import pykmc.artn as artn

from .artn_replay import (
    CORE_BLOCKS,
    RECORDINGS_ENV,
    TOLERANCE,
    check_blocks,
    lanczos_blocks,
    load_recording,
    replay,
    starting_vector,
)
from .test_dimer import NO_PBC, OPEN_CELL, curved_valley, valley_start

RECORDINGS = Path(
    os.environ.get(RECORDINGS_ENV)
    or Path(__file__).parent / "data" / "artn_step_replay"
)


def recorded(search: int) -> dict:
    """One recording, from the replay set or ``PYKMC_ARTN_RECORDINGS``."""
    return load_recording(RECORDINGS / f"search{search}.npz")


def first_probe_step(rec: dict) -> int:
    """Return the first step a Lanczos move led to."""
    steps = np.flatnonzero(rec["partn_step"] >= 0)
    return (["start"] + [rec["block_out"][e] for e in steps[:-1]]).index("LANC")


@pytest.mark.parametrize("search", range(8))
def test_initial_push_phase_matches_partn(search: int) -> None:
    """Start, pushes and basin perpendicular steps to the first Lanczos probe."""
    rec = recorded(search)
    report = replay(rec, last_step=first_probe_step(rec))
    assert report.passed, report.describe()
    blocks = [s.block for s in report.steps]
    assert blocks[:2] == ["start", "INIT"] and blocks[-1] == "LANC"
    assert blocks.count("LANC") == 1
    fire_steps = [s for s in report.steps if s.block == "PERP"]
    assert len(fire_steps) == blocks.count("PERP") > 0
    assert all(s.dt is not None and s.velocity is not None for s in fire_steps)
    assert report.repeats_skipped > 0


def full_recording(search: int) -> dict:
    """Load a recording that holds its first Lanczos block whole, else skip."""
    rec = recorded(search)
    if not lanczos_blocks(rec):
        pytest.skip(f"set {RECORDINGS_ENV} to the full recordings for Lanczos blocks")
    return rec


@pytest.mark.parametrize("search", range(8))
def test_first_lanczos_block_matches_partn(search: int) -> None:
    """Every step from the start to the first Lanczos block's move back."""
    rec = full_recording(search)
    block = lanczos_blocks(rec)[0]
    report = replay(rec)
    assert report.passed, report.describe()
    assert report.last_step == block.end and block.returns
    lanczos = [s for s in report.steps if s.block == "LANC"]
    assert len(lanczos) == block.probes + 1 == block.evaluations - block.repeats
    assert lanczos[-1].block_end and not any(s.block_end for s in lanczos[:-1])
    # The move back lands where the block started.
    start = report.steps[block.start]
    assert (
        np.max(np.abs(rec["x"][lanczos[-1].evaluation] - rec["x"][start.evaluation]))
        < 1e-12
    )


@pytest.mark.parametrize("search", range(8))
def test_every_lanczos_block_matches_partn(search: int) -> None:
    """Each block alone: steps, evaluations, end eigenvalue and mode up to sign."""
    checks = check_blocks(full_recording(search))
    problems = {c.block.start: c.problems() for c in checks if c.problems()}
    assert not problems, problems


def synthetic_recording(
    repeats_after: tuple = (1, 4, 12), nevalf_max: int = 40
) -> dict:
    """Record the core's own search on the curved valley, pARTn-style.

    A setup evaluation comes first, and a repeat of the positions follows each
    step in ``repeats_after``, flagged as fix artn flags its blank calls.
    pARTn's state after each call (eigenvalue, eigenvector, Lanczos flag) is
    read from the state the core showed with its next move.
    """
    x0 = valley_start()
    n = len(x0)
    rng = np.random.default_rng(7)
    params = dict(push_mode="list", push_step_size=0.3, ninit=2, nperp=3)
    seen: dict = {}
    rows = []

    def observe(move: str, state: object) -> None:
        seen.update(move=move, state=state)

    def surface(x: np.ndarray) -> tuple[float, np.ndarray]:
        energy, forces = curved_valley(x)
        rows.append((x.copy(), forces, energy, seen["move"], seen["state"]))
        return energy, forces

    push, first = rng.random((n, 3)) - 0.5, rng.random((n, 3)) - 0.5
    artn.artn_search(
        x0,
        OPEN_CELL,
        NO_PBC,
        np.ones(n, bool),
        0,
        0,
        surface,
        nevalf_max=nevalf_max,
        initial_push=push,
        first_lanczos_vector=first,
        observe=observe,
        **params,
    )
    blocks = {move: names[0] for move, names in CORE_BLOCKS.items()}
    move = [r[3] for r in rows] + [None, None]
    after = []  # pARTn's (eigenvalue, eigenvector, in Lanczos) after each call
    eigval, eigvec = math.inf, np.zeros((n, 3))
    for k in range(len(rows)):
        if move[k + 1] == "lanczos":
            eigval, eigvec = rows[k + 1][4]
        after.append((eigval, eigvec, move[k + 1] == move[k + 2] == "lanczos"))
    evals = [(rows[0], -1, False, 0, None, after[0])]  # setup
    for k, row in enumerate(rows):
        written = blocks[rows[k + 1][3]] if k + 1 < len(rows) else None
        evals.append((row, k, False, 2, written, after[k]))
        if k in repeats_after:
            evals.append((row, -1, True, 1, None, after[k]))
    fire = [e[0][4] if e[0][3] in ("perp", "relax") else None for e in evals]
    nan = float("nan")
    return dict(
        search=-1,
        x=np.array([e[0][0] for e in evals]),
        f=np.array([e[0][1] for e in evals]),
        v_in=np.array([f[2] if f else np.zeros((n, 3)) for f in fire]),
        energy=np.array([e[0][2] for e in evals]),
        fire_dt=np.array([f[0] if f else nan for f in fire]),
        fire_alpha=np.array([f[1] if f else nan for f in fire]),
        block_out=[e[4] for e in evals],
        eigval=np.array([e[5][0] for e in evals]),
        in_lanczos=np.array([e[5][2] for e in evals]),
        eigenvec=np.array([e[5][1] for e in evals]),
        eigenvec_index=np.arange(len(evals)),
        partn_step=np.array([e[1] for e in evals]),
        repeat=np.array([e[2] for e in evals]),
        kind=np.array([e[3] for e in evals]),
        initial_push=push,
        first_lanczos_vector=first,
        movable=np.ones(n, bool),
        cell=OPEN_CELL,
        pbc=NO_PBC,
        central_atom=0,
        masses=np.ones(n),
        params=dict(params, eigen_step_size=0.2),
    )


def test_replay_of_the_cores_own_search_matches() -> None:
    """Every step to the first Lanczos block's move back, repeats counted."""
    rec = synthetic_recording()
    report = replay(rec)
    assert report.passed, report.describe()
    block = lanczos_blocks(rec)[0]
    assert block.start == 8 and block.returns  # start, 2 x (push, 3 perp)
    assert report.last_step == block.end == 9 + block.probes
    assert [s.block for s in report.steps] == (
        ["start"] + ["INIT", "PERP", "PERP", "PERP"] * 2 + ["LANC"] * (block.probes + 1)
    )
    assert all(s.position == 0.0 for s in report.steps)
    lanczos = [s for s in report.steps if s.block == "LANC"]
    assert all(s.mode == 0.0 for s in lanczos)
    assert lanczos[0].eigval is None  # still unset at the first probe
    assert all(s.eigval == 0.0 for s in lanczos[1:]) and lanczos[-1].block_end
    assert report.repeats_skipped == 3  # after steps 1, 4 and 12
    assert report.repeats_total == 3
    assert report.other_skipped == 1  # the setup evaluation


def test_lanczos_blocks_replay_alone() -> None:
    """Each block from its own start: the same steps, state and evaluations."""
    rec = synthetic_recording(nevalf_max=80)
    checks = check_blocks(rec)
    assert len(checks) >= 2
    for c in checks:
        assert not c.problems(), c.problems()
        assert all(s.position == 0.0 for s in c.report.steps)
        assert c.end(0) == c.end(2) == 0.0
        assert c.core_evaluations == c.block.probes + 1
    # A later block starts from the eigenvector the one before ended with.
    steps = np.flatnonzero(rec["partn_step"] >= 0)
    end = steps[checks[0].block.end - 1]
    assert np.array_equal(starting_vector(rec, checks[1].block), rec["eigenvec"][end])


def test_replay_checks_the_lanczos_state_at_the_block_end() -> None:
    """Only the state the move back carries is judged; earlier ones are reported."""
    rec = synthetic_recording()
    block = lanczos_blocks(rec)[0]
    steps = np.flatnonzero(rec["partn_step"] >= 0)
    last_probe, middle = steps[block.end - 1], steps[block.start + 3]
    rec["eigenvec"][middle, 0, 0] += 1e-6
    report = replay(rec)
    assert report.passed and report.steps[block.start + 4].mode == pytest.approx(1e-6)

    rec["eigenvec"][last_probe, 0, 0] += 1e-9
    bad = replay(rec).first_mismatch
    assert bad.step == block.end and "eigenvector differs" in bad.problem

    rec = synthetic_recording()
    rec["eigval"][last_probe] += 1e-9 * max(abs(rec["eigval"][last_probe]), 1.0)
    bad = replay(rec).first_mismatch
    assert bad.step == block.end and "eigenvalue differs" in bad.problem


def test_replay_names_the_first_differing_step() -> None:
    """Its step, evaluation, block and the size of the difference."""
    rec = synthetic_recording()
    e = int(np.flatnonzero(rec["partn_step"] == 5)[0])
    rec["x"][e, 0, 1] += 3 * TOLERANCE
    report = replay(rec)
    bad = report.first_mismatch
    assert not report.passed and bad is not None
    assert (bad.step, bad.evaluation, bad.block) == (5, e, "INIT")
    assert bad.position == pytest.approx(3 * TOLERANCE, rel=1e-3)
    assert "first differing step 5" in report.describe()
    assert len(report.steps) == 6  # the replay stops there


def test_replay_checks_fire_state_and_blocks() -> None:
    """A changed FIRE alpha or a changed block is a mismatch."""
    rec = synthetic_recording()
    e = int(np.flatnonzero(rec["partn_step"] == 3)[0])
    rec["fire_alpha"][e] *= 1.01
    assert replay(rec).first_mismatch.problem.startswith("FIRE alpha")

    rec = synthetic_recording()
    e = int(np.flatnonzero(rec["partn_step"] == 4)[0])
    rec["block_out"][e] = "LANC"  # pARTn would have started Lanczos a step early
    bad = replay(rec).first_mismatch
    assert bad.step == 5 and "block LANC against core move init" in bad.problem
