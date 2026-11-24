"""Step replay of pARTn v1.1.0 recordings through the ARTn core (no LAMMPS).

The recorded tests read issue 02's eight recordings (611 MB, outside git) from
the folder named by ``PYKMC_ARTN_RECORDINGS`` and skip when it is unset. The
synthetic tests check the harness itself on a recording the core makes on an
analytic surface, with repeat evaluations added as pARTn's FIRE makes them.
"""

import os
from pathlib import Path

import numpy as np
import pytest

import pykmc.artn as artn

from .artn_replay import CORE_BLOCKS, RECORDINGS_ENV, TOLERANCE, load_recording, replay
from .test_dimer import NO_PBC, OPEN_CELL, curved_valley, valley_start

RECORDINGS = os.environ.get(RECORDINGS_ENV)


@pytest.mark.skipif(not RECORDINGS, reason=f"{RECORDINGS_ENV} is not set")
@pytest.mark.parametrize("search", range(8))
def test_initial_push_phase_matches_partn(search: int) -> None:
    """Start, pushes and basin perpendicular steps to the first Lanczos probe."""
    path = Path(RECORDINGS) / f"search{search}.npz"
    if not path.exists():
        pytest.skip(f"{path} not found")
    report = replay(load_recording(path))
    assert report.passed, report.describe()
    blocks = [s.block for s in report.steps]
    assert blocks[:2] == ["start", "INIT"] and blocks[-1] == "LANC"
    fire_steps = [s for s in report.steps if s.block == "PERP"]
    assert len(fire_steps) == blocks.count("PERP") > 0
    assert all(s.dt is not None and s.velocity is not None for s in fire_steps)
    assert report.repeats_skipped > 0


def synthetic_recording(repeats_after: tuple = (1, 4, 12)) -> dict:
    """Record the core's own search on the curved valley, pARTn-style.

    A setup evaluation comes first, and a repeat of the positions follows each
    step in ``repeats_after``, flagged as fix artn flags its blank calls.
    """
    x0 = valley_start()
    n = len(x0)
    rng = np.random.default_rng(7)
    params = dict(push_mode="list", push_step_size=0.3, ninit=2, nperp=3)
    seen: dict = {}
    rows = []

    def observe(move: str, fire_state: object) -> None:
        seen.update(move=move, fire=fire_state)

    def surface(x: np.ndarray) -> tuple[float, np.ndarray]:
        energy, forces = curved_valley(x)
        rows.append((x.copy(), forces, energy, seen["move"], seen["fire"]))
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
        nevalf_max=40,
        initial_push=push,
        first_lanczos_vector=first,
        observe=observe,
        **params,
    )
    blocks = {move: names[0] for move, names in CORE_BLOCKS.items()}
    evals = [(rows[0], -1, False, 0, None)]  # setup
    for k, row in enumerate(rows):
        written = blocks[rows[k + 1][3]] if k + 1 < len(rows) else None
        evals.append((row, k, False, 2, written))
        if k in repeats_after:
            evals.append((row, -1, True, 1, None))
    nan = float("nan")
    return dict(
        search=-1,
        x=np.array([e[0][0] for e in evals]),
        f=np.array([e[0][1] for e in evals]),
        v_in=np.array([e[0][4][2] if e[0][4] else np.zeros((n, 3)) for e in evals]),
        energy=np.array([e[0][2] for e in evals]),
        fire_dt=np.array([e[0][4][0] if e[0][4] else nan for e in evals]),
        fire_alpha=np.array([e[0][4][1] if e[0][4] else nan for e in evals]),
        block_out=[e[4] for e in evals],
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
        params=params,
    )


def test_replay_of_the_cores_own_search_matches() -> None:
    """Every step to the first Lanczos probe, repeats in that window counted."""
    rec = synthetic_recording()
    report = replay(rec)
    assert report.passed, report.describe()
    assert report.last_step == 9  # start, 2 x (push, 3 perp), first probe
    assert [s.block for s in report.steps] == (
        ["start"] + ["INIT", "PERP", "PERP", "PERP"] * 2 + ["LANC"]
    )
    assert all(s.position == 0.0 for s in report.steps)
    assert (
        report.repeats_skipped == 2
    )  # after steps 1 and 4; step 12 is past the window
    assert report.repeats_total == 3
    assert report.other_skipped == 1  # the setup evaluation


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
