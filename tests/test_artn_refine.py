"""The ARTn refiner (refine_style = artn), without LAMMPS or pARTn.

The core's stop at the saddle and the retry loop run on the curved valley,
whose saddle for atom 0 is at the origin, 0.5 eV above the minimum at
(1, 0.5, 0). The engine operation runs on a fake worker engine, and both
callers on a recording manager.
"""

import logging
import random
from concurrent.futures import Future
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import pandas as pd
import pytest
from mpi4py import MPI

import pykmc.artn
from pykmc.artn import RefineResult, artn_refine, artn_search
from pykmc.basins.basin import BasinsGenericEvents, StateData
from pykmc.config import Config, EventSearchConfig, PartnConfig, RegionConfig
from pykmc.dimer import DimerResult
from pykmc.engine.artn_ranks import DistributedSpace
from pykmc.engine.artn_search import ArtnSearchExtension, with_refiner
from pykmc.engine.dimer_search import DimerSearchExtension, _movable_mask
from pykmc.eventsearch.strategies.artn import (
    ArtnStrategy,
    artn_refine_options,
    ignored_partn_parameters,
)
from pykmc.initializer import Initializer
from pykmc.refinement import (
    Refinement,
    artn_refine_result,
    refine_seed_stream,
    submit_artn_refine,
)
from pykmc.result import ErrorType, EventRefinementOutput, Ok
from pykmc.system import System
from tests.engine.test_artn_ranks_offline import ThreadComm, parallel
from tests.test_dimer import (
    NO_PBC,
    OPEN_CELL,
    VALLEY_BARRIER,
    counting,
    curved_valley,
    valley_start,
)

OPTIONS = artn_refine_options(PartnConfig())
MASK = np.ones(4, bool)
MINIMUM = np.array([[1.0, 0.5, 3.0 * i] for i in range(4)])


def at_saddle(offset: float = 0.0) -> np.ndarray:
    """Return the minimum with atom 0 at the valley's saddle, or ``offset`` off it."""
    x = MINIMUM.copy()
    x[0] = [offset, 0.3 * offset, 0.0]
    return x


# ---------------------------------------------------------------- the core


def test_the_core_started_at_the_saddle_stops_there() -> None:
    """Stop as pARTn's lpush_final = .false. does: the saddle, its eigenvalue, no minima."""
    surface = counting(curved_valley)
    result = artn_search(
        at_saddle(),
        OPEN_CELL,
        NO_PBC,
        MASK,
        0,
        7,
        surface,
        stop_at_saddle=True,
        **OPTIONS,
    )

    assert result.success and result.reason is None
    np.testing.assert_allclose(result.saddle_positions, at_saddle(), atol=1e-12)
    assert result.e_saddle == pytest.approx(VALLEY_BARRIER, abs=1e-12)
    assert result.delr_sad < 1e-12
    assert result.curvature == pytest.approx(-2.0, abs=1e-3)
    assert result.min1_positions is None and result.min2_positions is None
    assert result.force_calls_saddle == result.force_calls_total == surface.count


@pytest.mark.parametrize("offset", [0.02, 0.05])
def test_the_core_started_off_the_saddle_climbs_back_and_stops(offset: float) -> None:
    """Off the saddle, it climbs back to it and stops; delr_sad is pARTn's 3N norm."""
    start = at_saddle(offset)
    result = artn_search(
        start,
        OPEN_CELL,
        NO_PBC,
        MASK,
        0,
        7,
        curved_valley,
        stop_at_saddle=True,
        **OPTIONS,
    )

    assert result.success
    np.testing.assert_allclose(result.saddle_positions[0], [0, 0, 0], atol=5e-4)
    np.testing.assert_allclose(result.saddle_positions[1:], start[1:], atol=1e-6)
    assert result.e_saddle == pytest.approx(VALLEY_BARRIER, abs=1e-6)
    assert result.delr_sad == pytest.approx(
        np.linalg.norm(result.saddle_positions - start), rel=1e-12
    )
    assert result.min1_positions is None


def test_the_full_search_is_unchanged_and_reports_delr_sad() -> None:
    """Without the stop the search still pushes over; delr_sad is now set too."""
    options = dict(push_mode="list", push_step_size=0.3)
    x = valley_start()
    full = artn_search(x, OPEN_CELL, NO_PBC, MASK, 0, 5, curved_valley, **options)
    stopped = artn_search(
        x, OPEN_CELL, NO_PBC, MASK, 0, 5, curved_valley, stop_at_saddle=True, **options
    )

    assert full.success and full.min2_positions is not None
    assert stopped.min1_positions is None
    np.testing.assert_array_equal(stopped.saddle_positions, full.saddle_positions)
    assert stopped.force_calls_total == full.force_calls_saddle
    assert stopped.delr_sad == full.delr_sad
    assert full.delr_sad == pytest.approx(np.linalg.norm(full.saddle_positions - x))


def test_a_saddle_on_the_last_allowed_call_still_fails() -> None:
    """As pARTn's guard: call nevalf_max fails even if it finds the saddle."""
    calls = artn_search(
        at_saddle(),
        OPEN_CELL,
        NO_PBC,
        MASK,
        0,
        7,
        curved_valley,
        stop_at_saddle=True,
        **OPTIONS,
    ).force_calls_total
    capped = artn_search(
        at_saddle(),
        OPEN_CELL,
        NO_PBC,
        MASK,
        0,
        7,
        curved_valley,
        stop_at_saddle=True,
        **{**OPTIONS, "nevalf_max": calls},
    )

    assert capped.reason == "max_force_calls"


def _first_relaxation(start: np.ndarray, **options: Any) -> tuple[DimerResult, int]:
    """Return the search and its first out-of-basin perpendicular relaxation's steps."""
    moves = []
    result = artn_search(
        start, OPEN_CELL, NO_PBC, MASK, 0, 7, curved_valley, stop_at_saddle=True,
        observe=lambda move, state: moves.append(move), **{**OPTIONS, **options},
    )  # fmt: skip
    steps = 0
    for move in moves[moves.index("eigen") + 1 :]:
        if move != "perp":
            break
        steps += 1
    return result, steps


def test_nperp_start_replaces_only_the_limit_in_force() -> None:
    """Replay pARTn's retry state: no cap on the first relaxation, the list unchanged."""
    start = at_saddle(0.02)
    start[1, 1] += 0.05  # a perpendicular strain the first relaxation works off

    fresh, fresh_steps = _first_relaxation(start)
    carried, carried_steps = _first_relaxation(start, nperp_start=-1)
    same, same_steps = _first_relaxation(start, nperp_start=OPTIONS["nperp"])

    assert fresh.success and carried.success
    assert fresh_steps == OPTIONS["nperp"] < carried_steps
    assert same_steps == fresh_steps
    np.testing.assert_array_equal(same.saddle_positions, fresh.saddle_positions)
    assert same.force_calls_total == fresh.force_calls_total


# ---------------------------------------------------------------- retry loop


def _scripted(outcomes: dict[int, tuple[bool, float | None, str | None]]) -> Any:
    """Return a fake core giving ``outcomes[seed]`` = (success, delr_sad, reason)."""
    seen = []

    def fake(positions, cell, pbc, movable, central, seed, callback, **options):  # noqa: ANN001, ANN003, ANN202
        seen.append((seed, options))
        success, delr, reason = outcomes[seed]
        fields = dict.fromkeys(DimerResult._fields)
        fields.update(
            success=success,
            reason=reason,
            seed=seed,
            force_calls_saddle=10 + seed,
            force_calls_total=10 + seed,
            delr_sad=delr,
        )
        if success:
            fields.update(saddle_positions=positions + seed, e_saddle=5.0 + seed)
            fields.update(curvature=-1.0)
        return DimerResult(**fields)

    fake.seen = seen
    return fake


def test_the_retry_loop_accepts_on_attempt_k(monkeypatch: pytest.MonkeyPatch) -> None:
    """Attempts run in seed order until one is closer than delr_sad_thr."""
    fake = _scripted(
        {
            1: (False, None, "eigenvalue_lost"),
            2: (True, 0.4, None),
            3: (True, 0.1, None),
        }
    )
    monkeypatch.setattr(pykmc.artn, "artn_search", fake)
    result = artn_refine(
        MINIMUM, OPEN_CELL, NO_PBC, MASK, 0, [1, 2, 3, 4], curved_valley,
        delr_sad_thr=0.4, e_start=2.0, ninit=0,
    )  # fmt: skip

    assert result.success and result.reason is None
    assert [a.seed for a in result.attempts] == [1, 2, 3]
    assert [a.reason for a in result.attempts] == ["eigenvalue_lost", None, None]
    assert result.attempts[1].delr_sad == 0.4  # at the threshold: retried
    assert result.barrier == pytest.approx(8.0 - 2.0)
    assert result.delr_sad == 0.1
    np.testing.assert_array_equal(result.saddle_positions, MINIMUM + 3)
    assert result.force_calls == 11 + 12 + 13
    assert all(opts["stop_at_saddle"] and opts["ninit"] == 0 for _, opts in fake.seen)


def test_the_retry_loop_fails_after_the_last_seed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """After r_max_attempts the last attempt's end is the failure."""
    fake = _scripted({1: (False, None, "max_force_calls"), 2: (True, 0.9, None)})
    monkeypatch.setattr(pykmc.artn, "artn_search", fake)
    far = artn_refine(
        MINIMUM, OPEN_CELL, NO_PBC, MASK, 0, [1, 2], curved_valley, e_start=0.0
    )
    lost = artn_refine(
        MINIMUM, OPEN_CELL, NO_PBC, MASK, 0, [2, 1], curved_valley, e_start=0.0
    )

    assert not far.success and far.reason == "far_saddle" and far.delr_sad == 0.9
    assert far.barrier is None and far.saddle_positions is None
    assert len(far.attempts) == 2
    assert lost.reason == "max_force_calls" and lost.delr_sad is None


def test_attempt_options_go_to_their_attempt_only(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """For step replay: an injected first Lanczos vector and nperp per attempt."""
    fake = _scripted({1: (True, 0.9, None), 2: (True, 0.9, None), 3: (True, 0.9, None)})
    monkeypatch.setattr(pykmc.artn, "artn_search", fake)
    carried = np.ones((4, 3))
    artn_refine(
        MINIMUM, OPEN_CELL, NO_PBC, MASK, 0, [1, 2, 3], curved_valley, e_start=0.0,
        attempt_options=[{}, {"first_lanczos_vector": carried, "nperp_start": -1}],
        nperp=3,
    )  # fmt: skip

    first, second, third = (opts for _, opts in fake.seen)
    assert "first_lanczos_vector" not in first and "nperp_start" not in first
    assert second["first_lanczos_vector"] is carried and second["nperp_start"] == -1
    assert "nperp_start" not in third
    assert first["nperp"] == second["nperp"] == third["nperp"] == 3


def test_retries_draw_fresh_and_a_failed_attempt_retries_for_real() -> None:
    """A capped first attempt fails; the second, with its own seed, is accepted."""
    surface = counting(curved_valley)
    result = artn_refine(
        at_saddle(0.02), OPEN_CELL, NO_PBC, MASK, 0, [11, 12, 13], surface,
        minimum=MINIMUM, attempt_options=[{"nevalf_max": 5}], **OPTIONS,
    )  # fmt: skip

    assert result.success
    assert [(a.seed, a.reason) for a in result.attempts] == [
        (11, "max_force_calls"),
        (12, None),
    ]
    # The start energy is the callback's at the minimum, one call apart.
    assert result.e_start == pytest.approx(0.0, abs=1e-12)
    assert result.barrier == pytest.approx(VALLEY_BARRIER, abs=1e-6)
    assert (
        result.force_calls
        == surface.count
        == sum(a.force_calls for a in result.attempts) + 1
    )


def test_refine_arguments_are_checked() -> None:
    """Exactly one start energy source, a seed, and no more options than seeds."""
    args = (MINIMUM, OPEN_CELL, NO_PBC, MASK, 0)
    with pytest.raises(ValueError, match="exactly one"):
        artn_refine(*args, [1], curved_valley)
    with pytest.raises(ValueError, match="exactly one"):
        artn_refine(*args, [1], curved_valley, e_start=0.0, minimum=MINIMUM)
    with pytest.raises(ValueError, match="seed"):
        artn_refine(*args, [], curved_valley, e_start=0.0)
    with pytest.raises(ValueError, match="seed"):
        artn_refine(*args, [1], curved_valley, e_start=0.0, attempt_options=[{}, {}])


def test_a_refinement_over_ranks_matches_the_serial_one() -> None:
    """With a rank-local callback, the refinement and its barrier equal one rank's."""
    start = at_saddle(0.02)
    serial = artn_refine(
        start,
        OPEN_CELL,
        NO_PBC,
        MASK,
        0,
        [3, 4],
        curved_valley,
        minimum=MINIMUM,
        **OPTIONS,
    )

    def run(comm: ThreadComm) -> Any:
        ids = np.flatnonzero(np.arange(4) % 2 == comm.rank)
        space = DistributedSpace(comm, ids, MASK, start)

        def query(local: np.ndarray) -> tuple[float, np.ndarray]:
            energy, forces = curved_valley(local)
            return comm.allreduce(energy, MPI.SUM), forces

        local = artn_refine(
            start[ids], OPEN_CELL, NO_PBC, MASK[ids], 0, [3, 4], query,
            minimum=MINIMUM[ids], vectors=space, **OPTIONS,
        )  # fmt: skip
        return space.collect_result(local, OPEN_CELL, NO_PBC)

    combined, peer = parallel(2, run)
    assert peer is None
    assert combined.success
    for ours, theirs in zip(combined.attempts, serial.attempts, strict=True):
        assert ours[:2] == theirs[:2] and ours.force_calls == theirs.force_calls
        assert ours.delr_sad == pytest.approx(theirs.delr_sad, rel=1e-12)
    np.testing.assert_allclose(
        combined.saddle_positions, serial.saddle_positions, atol=1e-12
    )
    assert combined.barrier == pytest.approx(serial.barrier, abs=1e-12)
    assert combined.delr_sad == pytest.approx(serial.delr_sad, rel=1e-12)


# ---------------------------------------------------------------- engine operation


class FakeWorker:
    """The worker's own engine: records the pre-relaxation, never reads etotal."""

    comm = None

    def __init__(self) -> None:
        self.config = SimpleNamespace(pair_style="eam/alloy")
        self.x: np.ndarray | None = None
        self.held: list[np.ndarray] = []
        self.steps = 0
        self.lmp = SimpleNamespace(extract_global=lambda name: self.steps)

    def register(self, extension: object) -> None:
        """Accept the extension."""

    def set_positions(self, positions: np.ndarray) -> None:
        """Hold the positions."""
        self.x = np.array(positions, dtype=float)

    def minimize_freeze_core(self, core_idx: np.ndarray) -> None:
        """Record the held atoms; the valley's other atoms are already relaxed."""
        self.held.append(np.array(core_idx))
        self.steps += 7

    def get_positions(self) -> np.ndarray:
        """Return the positions held."""
        return self.x.copy()

    def get_total_energy(self, *args: Any, **kwargs: Any) -> float:
        """Fail: the refiner must not read etotal."""
        raise AssertionError("the refiner reads the potential energy, not etotal")


# The instance's potential energy: the valley plus an offset, as a total pe is.
OFFSET = -7.0


def offset_valley(x: np.ndarray) -> tuple[float, np.ndarray]:
    """Return the curved valley's energy plus OFFSET, and its forces."""
    energy, forces = curved_valley(x)
    return energy + OFFSET, forces


@pytest.fixture
def operation(monkeypatch: pytest.MonkeyPatch) -> tuple[ArtnSearchExtension, dict]:
    """Build the extension on a fake worker, its force path on the offset valley."""
    seen: dict = {}

    def search(  # noqa: ANN202
        self,  # noqa: ANN001
        core,  # noqa: ANN001
        positions,  # noqa: ANN001
        types,  # noqa: ANN001
        cell,  # noqa: ANN001
        pbc,  # noqa: ANN001
        movable,  # noqa: ANN001
        central,  # noqa: ANN001
        seed,  # noqa: ANN001
        options,  # noqa: ANN001
    ):
        seen.update(start=positions, options=dict(options), central=central)
        core_options = {
            k: v
            for k, v in options.items()
            if k not in ("reuse_neighbor_list", "eam_force_crop")
        }
        mask = _movable_mask(movable, len(positions))
        return core(
            positions, cell, pbc, mask, central, seed, offset_valley, **core_options
        )

    monkeypatch.setattr(ArtnSearchExtension, "_search", search)
    monkeypatch.setattr(
        ArtnSearchExtension, "_instance_masses", lambda self: np.ones(4)
    )
    return ArtnSearchExtension(FakeWorker()), seen


@pytest.mark.parametrize("relax_radius", [None, 4.0])
def test_the_operation_returns_the_barrier_itself(
    operation: tuple[ArtnSearchExtension, dict], relax_radius: float | None
) -> None:
    """Saddle energy minus the minimum's potential energy, with or without the AV."""
    extension, seen = operation
    result = extension.artn_refine(
        MINIMUM, ["Fe"] * 4, OPEN_CELL, NO_PBC, np.arange(4), 0,
        saddle_idx=[0], saddle_positions=np.zeros((1, 3)), seeds=[5, 6],
        delr_sad_thr=0.4, relax_radius=relax_radius, options=OPTIONS,
    )  # fmt: skip

    assert result.success and result.prerelax_iterations == 7
    assert result.e_start == pytest.approx(OFFSET, abs=1e-12)
    assert result.e_saddle == pytest.approx(OFFSET + VALLEY_BARRIER, abs=1e-12)
    assert result.barrier == pytest.approx(VALLEY_BARRIER, abs=1e-12)
    # The core starts from the pre-relaxed positions, rcut atoms at the saddle.
    np.testing.assert_array_equal(seen["start"], at_saddle())
    assert seen["options"]["reuse_neighbor_list"] is True
    held = extension.engine.held[0]
    if relax_radius is None:
        np.testing.assert_array_equal(held, [0])  # minimize_freeze_core(saddle_idx)
    else:
        np.testing.assert_array_equal(held, [0, 2, 3])  # and the atoms beyond rmov


class RankInstance:
    """A distributed LAMMPS instance on the curved valley, rows split over ranks."""

    def __init__(self, config: Any, comm: ThreadComm, engine_id: str, **_: Any) -> None:
        self.comm, self.lmp, self.numpy = comm, self, self

    def start(self) -> None:
        """Start nothing."""

    initialize_parameters = initialize_potential = close = start

    def initialize_system(
        self, types: Any, x: np.ndarray, *args: Any, **kw: Any
    ) -> None:
        """Keep every size-th row, from this rank's."""
        self.ids = np.arange(self.comm.rank, len(x), self.comm.Get_size())
        self.x, self.n = x[self.ids].copy(), len(x)

    def get_natoms(self) -> int:
        """Return the atom count."""
        return self.n

    def extract_global(self, key: str) -> int:
        """Return the local atom count."""
        return len(self.ids)

    def extract_atom(self, key: str) -> np.ndarray:
        """Return a per-atom array of this rank's rows."""
        return dict(
            id=self.ids + 1,
            x=self.x,
            f=curved_valley(self.x)[1],
            type=np.ones(len(self.ids), int),
            mass=np.array([0.0, 1.0]),
        )[key]

    def command(self, cmd: str) -> None:
        """Accept a run or an atom migration."""
        assert cmd in ("run 0 post no", "displace_atoms all move 0 0 0 units box")

    def get_thermo(self, key: str) -> float:
        """Return the potential energy summed over ranks."""
        return self.comm.allreduce(curved_valley(self.x)[0], MPI.SUM)

    @staticmethod
    def _positions_to_lammps(x: np.ndarray) -> np.ndarray:
        return x

    _positions_from_lammps = _positions_to_lammps


class RankWorker(FakeWorker):
    """The worker's own engine on one rank of a multi-rank worker."""

    def __init__(self, comm: ThreadComm) -> None:
        super().__init__()
        self.comm, self.engine_id = comm, 0
        self.config = SimpleNamespace(pair_style="eam/fs")

    def get_positions(self) -> np.ndarray | None:
        """Return the positions on rank 0 only, as LAMMPS's gather does."""
        return self.x.copy() if self.comm.rank == 0 else None


def test_a_multi_rank_worker_refines_as_one_rank(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The rank driver runs the refiner: same attempts, saddle and barrier."""
    import sys

    monkeypatch.setitem(
        sys.modules, "pykmc.engine.lammps", SimpleNamespace(LammpsEngine=RankInstance)
    )
    monkeypatch.setattr(
        ArtnSearchExtension, "_instance_masses", lambda self: np.ones(4)
    )
    request = dict(
        positions=MINIMUM, types=["Fe"] * 4, cell=OPEN_CELL, pbc=NO_PBC,
        movable=np.arange(4), central_atom=0, saddle_idx=[0],
        saddle_positions=[[0.02, 0.006, 0.0]], seeds=[3, 4], delr_sad_thr=0.4,
        options={**OPTIONS, "reuse_neighbor_list": False},
    )  # fmt: skip

    def invoke(comm: ThreadComm) -> Any:
        return ArtnSearchExtension(RankWorker(comm)).artn_refine(**request)

    leader, *others = parallel(3, invoke)
    serial = artn_refine(
        at_saddle(0.02), OPEN_CELL, NO_PBC, MASK, 0, [3, 4], curved_valley,
        minimum=MINIMUM, masses=np.ones(4), **OPTIONS,
    )  # fmt: skip

    assert others == [None, None]
    assert leader.success and leader.prerelax_iterations == 7
    assert [a[:2] for a in leader.attempts] == [a[:2] for a in serial.attempts]
    assert leader.force_calls == serial.force_calls
    np.testing.assert_allclose(
        leader.saddle_positions, serial.saddle_positions, atol=1e-12
    )
    assert leader.barrier == pytest.approx(serial.barrier, abs=1e-12)
    with pytest.raises(ValueError, match="single-rank"):
        parallel(
            2,
            lambda comm: ArtnSearchExtension(RankWorker(comm)).artn_refine(
                **{
                    **request,
                    "options": {**request["options"], "attempt_options": [{}]},
                }
            ),
        )


# ---------------------------------------------------------------- the switch


def test_refine_style_defaults_to_partn_and_takes_artn() -> None:
    """[EventSearch] refine_style: partn by default, artn by choice, nothing else."""
    assert EventSearchConfig(style="artn", nsearch=1).refine_style == "partn"
    assert (
        EventSearchConfig(style="dimer", nsearch=1, refine_style="artn").refine_style
        == "artn"
    )
    with pytest.raises(ValueError):
        EventSearchConfig(style="partn", nsearch=1, refine_style="dimer")


def test_refine_style_artn_takes_either_force_path(tmp_path: Path) -> None:
    """refine_style reads from the input; with the AV on, full and av both load."""
    text = (Path(__file__).parent / "data" / "input.in").read_text()
    text = text.replace("emin_event = 0.0", "emin_event = 0.0\nrefine_style = artn")
    text = text.replace("basin = True", "basin = True\nactive_volume = True")
    path = tmp_path / "input.in"
    for forces in ("full", "av"):
        av = f"\n[ActiveVolume]\nrmov = 6.0\nract = 10.0\nartn_forces = {forces}\n"
        path.write_text(text + av)
        config = Config.from_ini_file(str(path))
        assert config.eventsearch.refine_style == "artn"
        assert config.eventsearch.style == "partn"
        assert config.activevolume.artn_forces == forces


_GPU_ARGS = "-k on g 1 -sf kk"


def _gpu_refine_input(
    tmp_path: Path,
    lammps_options: str,
    *,
    active: bool = True,
    refine_style: str | None = None,
    artn_forces: str | None = None,
    min_style: str = "cg",
) -> Config:
    """Read an ARTn input with the given LAMMPS instance and refinement settings."""
    text = (Path(__file__).parent / "data" / "input.in").read_text()
    text = text.replace("n_steps = 10", f"n_steps = 10\nactive_volume = {active}")
    text = text.replace("min_style = cg", f"min_style = {min_style}")
    text = text.replace(
        "pair_style = eam/alloy", f"{lammps_options}\npair_style = eam/alloy"
    )
    text = text.replace("style = partn", "style = artn")
    if refine_style is not None:
        text = text.replace(
            "nsearch = 10", f"nsearch = 10\nrefine_style = {refine_style}"
        )
    text += "\n[ActiveVolume]\nrmov = 6\nract = 10\n"
    if artn_forces is not None:
        text += f"artn_forces = {artn_forces}\n"
    path = tmp_path / "gpu_refinement.in"
    path.write_text(text)
    return Config.from_ini_file(str(path))


@pytest.mark.parametrize(
    ("lammps_options", "active", "expected"),
    [
        (f"search_cmdargs = {_GPU_ARGS}", True, "artn"),
        ("cmdargs = -kokkos yes gpus 2 -suffix kk", True, "artn"),
        ("library = mtpkk", True, "partn"),
        ("search_cmdargs = -k on t 4 -sf kk", True, "partn"),
        (f"group_cmdargs = {_GPU_ARGS}", True, "partn"),
        (f"search_cmdargs = {_GPU_ARGS}", False, "partn"),
        (f"cmdargs = {_GPU_ARGS}\nsearch_cmdargs = -k off", True, "partn"),
        ("search_cmdargs = -k on g 1 -sf hybrid kk omp", True, "artn"),
        ("search_cmdargs = -k on g 1 -pk kokkos neigh half", True, "partn"),
        (f"search_cmdargs = {_GPU_ARGS} -sf omp", True, "partn"),
    ],
    ids=[
        "search-gpu",
        "global-gpu",
        "library-only",
        "cpu-kokkos",
        "group-only",
        "av-off",
        "search-disables-gpu",
        "hybrid-kk",
        "no-kk-suffix",
        "last-suffix-not-kk",
    ],
)
def test_refinement_default_uses_the_active_volume_search_gpu(
    tmp_path: Path, lammps_options: str, active: bool, expected: str
) -> None:
    config = _gpu_refine_input(tmp_path, lammps_options, active=active)
    assert config.eventsearch.refine_style == expected


def test_a_gpu_input_kokkos_cannot_minimize_keeps_pre_change_refiners(
    tmp_path: Path,
) -> None:
    """With min_style = fire the AV pre-relaxation would stop LAMMPS: keep the CPU paths."""
    gpu = f"search_cmdargs = {_GPU_ARGS}"
    config = _gpu_refine_input(tmp_path, gpu, min_style="fire")
    assert config.eventsearch.refine_style == "partn"
    config = _gpu_refine_input(tmp_path, gpu, min_style="fire", refine_style="artn")
    assert config.eventsearch.refine_style == "artn"
    assert config.artn_refine_forces == "full"


@pytest.mark.parametrize(
    ("lammps_options", "min_style"),
    [
        (f"search_cmdargs = {_GPU_ARGS}", "fire"),
        ("search_cmdargs = -k on g 1", "cg"),
        ("search_cmdargs = -k on t 4", "cg"),
    ],
    ids=["gpu-fire", "gpu-no-suffix", "cpu-kokkos-no-suffix"],
)
def test_explicit_av_refinement_needs_a_kokkos_minimizer(
    tmp_path: Path, lammps_options: str, min_style: str
) -> None:
    with pytest.raises(ValueError, match="min_style = cg with -sf kk"):
        _gpu_refine_input(
            tmp_path,
            lammps_options,
            refine_style="artn",
            artn_forces="av",
            min_style=min_style,
        )


def test_the_gpu_default_survives_a_dump_and_reload(tmp_path: Path) -> None:
    config = _gpu_refine_input(tmp_path, f"search_cmdargs = {_GPU_ARGS}")
    reloaded = Config.model_validate(config.model_dump())
    for c in (config, reloaded):
        assert c.eventsearch.refine_style == "artn"
        assert c.activevolume.artn_forces is None
        assert c.artn_refine_forces == "av"


def test_the_log_says_why_a_gpu_input_keeps_partn(tmp_path: Path) -> None:
    lines: list[str] = []
    for min_style in ("fire", "cg"):
        config = _gpu_refine_input(
            tmp_path, f"search_cmdargs = {_GPU_ARGS}", min_style=min_style
        )
        kmc = SimpleNamespace(
            config=config,
            loggers=SimpleNamespace(info=lambda name, line: lines.append(line)),
        )
        Initializer(kmc).log_refine_region()
    assert len(lines) == 2
    assert "min_style = fire), so pARTn refines" in lines[0]
    assert lines[1] == "ARTn refinement force region = av"


def test_explicit_partn_refiner_on_gpu(tmp_path: Path) -> None:
    config = _gpu_refine_input(
        tmp_path, f"search_cmdargs = {_GPU_ARGS}", refine_style="partn"
    )
    assert config.eventsearch.refine_style == "partn"


@pytest.mark.parametrize(
    ("refine_style", "artn_forces", "search_radii", "refine_radii"),
    [
        (None, None, None, (6, 10)),
        ("artn", None, None, (6, 10)),
        (None, "av", (6, 10), (6, 10)),
        (None, "full", None, None),
        ("artn", "av", (6, 10), (6, 10)),
        ("artn", "full", None, None),
    ],
)
def test_gpu_default_and_explicit_force_regions_reach_the_worker(
    tmp_path: Path,
    refine_style: str | None,
    artn_forces: str | None,
    search_radii: tuple[int, int] | None,
    refine_radii: tuple[int, int] | None,
) -> None:
    config = _gpu_refine_input(
        tmp_path,
        f"search_cmdargs = {_GPU_ARGS}",
        refine_style=refine_style,
        artn_forces=artn_forces,
    )
    assert config.eventsearch.refine_style == "artn"
    assert ArtnStrategy(config).options.get("av_radii") == search_radii

    manager = RecordingManager()
    system = _system()
    submit_artn_refine(
        manager,
        config,
        refine_seed_stream(9),
        0,
        system.positions,
        system.cell,
        system.types,
        system.pbc,
        NEIGHBORS,
        SADDLE,
    ).result()
    ((op, sent),) = manager.calls
    assert op == "artn_refine"
    assert sent["options"].get("av_radii") == refine_radii
    assert sent["relax_radius"] == (None if refine_radii else 6)


@pytest.mark.parametrize("artn_forces", ["av", "full"])
def test_av_refinement_carries_the_frozen_atoms(artn_forces: str) -> None:
    """[frozen_atoms] leave the movable set, and go with the active volume."""
    config = _config("artn", active_volume=True, artn_forces=artn_forces)
    config.frozen_atoms = RegionConfig(indices=[1, 4])
    manager = RecordingManager()
    submit_artn_refine(
        manager,
        config,
        refine_seed_stream(9),
        0,
        POSITIONS,
        CELL,
        np.array(["Ni"] * len(POSITIONS)),
        [True] * 3,
        NEIGHBORS,
        SADDLE,
    ).result()

    ((_, sent),) = manager.calls
    np.testing.assert_array_equal(sent["movable"], [0, 2, 3])
    if artn_forces == "av":
        np.testing.assert_array_equal(sent["options"]["av_frozen"], [1, 4])
    else:
        assert "av_frozen" not in sent["options"]


def test_the_artn_extension_is_loaded_for_any_search_style() -> None:
    """refine_style = artn adds the ARTn extension, in place of the dimer's."""
    assert with_refiner([], "partn") == []
    assert with_refiner([DimerSearchExtension], "partn") == [DimerSearchExtension]
    assert with_refiner([], "artn") == [ArtnSearchExtension]
    assert with_refiner([DimerSearchExtension], "artn") == [ArtnSearchExtension]
    assert with_refiner([ArtnSearchExtension], "artn") == [ArtnSearchExtension]


def test_r_parameters_reach_the_core_under_its_names() -> None:
    """Every r_* field partn_refine passes to pARTn maps to the core's keyword."""
    partn = PartnConfig(
        r_push_mode="rad", r_push_dist_thr=2.5, r_nperp=None, r_nevalf_max=77
    )
    options = artn_refine_options(partn)

    assert options["push_mode"] == "rad" and options["push_dist_thr"] == 2.5
    assert options["nperp"] is None and options["nevalf_max"] == 77
    assert options["dmax"] == partn.r_dmax and options["forc_thr"] == partn.r_forc_thr
    assert "convergence_property" not in options and "push_over" not in options
    artn_search(
        at_saddle(),
        OPEN_CELL,
        NO_PBC,
        MASK,
        0,
        1,
        curved_valley,
        stop_at_saddle=True,
        **options,
    )


# ---------------------------------------------------------------- seeds


def test_refinement_seeds_repeat_from_the_run_seed_and_shift_no_search_seed() -> None:
    """A child stream of [Control] seed: the same seeds each run, random untouched."""
    random.seed(42)
    searches = [random.getrandbits(32) for _ in range(3)]
    random.seed(42)
    first = refine_seed_stream(42).integers(0, 2**32, size=10)
    again = refine_seed_stream(42).integers(0, 2**32, size=10)
    assert [random.getrandbits(32) for _ in range(3)] == searches
    np.testing.assert_array_equal(first, again)
    assert not np.array_equal(first, refine_seed_stream(43).integers(0, 2**32, size=10))


# ---------------------------------------------------------------- callers


class RecordingManager:
    """Records each operation submitted and answers with a prepared result."""

    def __init__(self, workers: int = 1) -> None:
        self.calls: list[tuple[str, dict]] = []
        self.workers = workers

    def _answer(self, op: str, kwargs: dict) -> Future:
        self.calls.append((op, kwargs))
        future: Future = Future()
        if op == "artn_refine":
            future.set_result(
                RefineResult(
                    success=True, reason=None, saddle_positions=np.array(kwargs["positions"]),
                    e_saddle=-99.4, e_start=-100.0, barrier=0.6, curvature=-1.0,
                    delr_sad=0.01, attempts=(), force_calls=3,
                )
            )  # fmt: skip
        elif op == "get_total_energy":
            future.set_result(-100.0)
        else:
            future.set_result(
                Ok(
                    EventRefinementOutput(
                        central_atom_index=kwargs["central_atom_idx"],
                        saddle_positions=np.array(kwargs["positions"]),
                        E_saddle=-99.4,
                        refined="T",
                    )
                )
            )
        return future

    def __getattr__(self, op: str) -> Any:
        """Submit any operation by name, as the manager does."""
        return lambda **kwargs: self._answer(op, kwargs)


CELL = np.diag([10.0, 10.0, 10.0])
POSITIONS = np.array(
    [
        [5.0, 5.0, 5.0],
        [6.2, 5.0, 5.0],
        [5.0, 6.2, 5.0],
        [5.0, 5.0, 6.2],
        [1.0, 1.0, 1.0],
    ]
)
NEIGHBORS = np.array([0, 1, 2, 3])
SADDLE = POSITIONS[NEIGHBORS] + [[0.6, 0.0, 0.0], [-0.1, 0, 0], [0, 0, 0], [0, 0, 0]]


def _config(
    refine_style: str, active_volume: bool, artn_forces: str = "full"
) -> SimpleNamespace:
    return SimpleNamespace(
        control=SimpleNamespace(active_volume=active_volume),
        activevolume=SimpleNamespace(rmov=3.0, ract=6.0),
        artn_refine_forces=artn_forces,
        atomicenvironment=SimpleNamespace(atom_coloring_mode="full"),
        eventsearch=SimpleNamespace(refined_energy_thr=0.05, refine_style=refine_style),
        psr=SimpleNamespace(style="ira", matching_score_thr=0.1),
        ira=None,
        partn=PartnConfig(),
        frozen_atoms=None,
        reconstruction=SimpleNamespace(saddle_disp_thr=0.001),
    )


class IdentityPSR:
    """A registration that matches the event to the site as it is."""

    @staticmethod
    def create(style: str, config: Any) -> "IdentityPSR":
        """Build one, whatever the style."""
        return IdentityPSR()

    def match_and_check(self, *args: Any, **kwargs: Any) -> Ok:
        """Match with no rotation, translation or permutation."""
        return Ok(
            SimpleNamespace(
                rotation_matrix=np.eye(3),
                translation_matrix=np.zeros(3),
                permutation_matrix=np.arange(len(NEIGHBORS)),
            )
        )


def _event() -> pd.Series:
    initial = POSITIONS[NEIGHBORS]
    return pd.Series(
        {
            "event_id": "e",
            "idx_ref": 3,
            "initial_positions": initial,
            "saddle_positions": SADDLE.copy(),
            "final_positions": initial + [[1.2, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0]],
            "types": ["Fe"] * 4,
            "sym_matrix": [np.eye(3)],
            "sym_perm": [np.arange(4)],
            "energy_barrier": 0.6,
            "move_atom_idx": 0,
            "k_pref": 1e13,
            "k": 1.0,
        }
    )


def _system() -> System:
    return System(
        positions=POSITIONS.copy(),
        types=["Fe"] * 5,
        cell=CELL,
        pbc=np.array([True, True, True]),
        index=np.arange(5),
    )


def _refinement_call(  # noqa: ANN202
    refine_style: str,
    active_volume: bool,
    monkeypatch,  # noqa: ANN001
    artn_forces: str = "full",
):
    monkeypatch.setattr("pykmc.refinement.PointSetRegistration", IdentityPSR)
    manager = RecordingManager()
    neighbors = SimpleNamespace(get_neighbors=lambda kind, atom: NEIGHBORS.copy())
    refinement = Refinement(
        _config(refine_style, active_volume, artn_forces),
        SimpleNamespace(info=lambda *a: None, progress_bar=lambda *a: None),
        _system(),
        neighbors,
        SimpleNamespace(get_atoms_with_id=lambda event_id: [0]),
        manager,
        refine_seeds=refine_seed_stream(9),
    )
    refinement.get_total_refinements_todo = lambda df: (1, 1.0)
    refinement.get_energy_thr_refine = lambda df, k: 10.0
    refinement.execute(pd.DataFrame([_event()]), -100.0)
    return manager.calls, refinement.results


@pytest.mark.parametrize(
    ("active_volume", "artn_forces"),
    [(False, "full"), (True, "full"), (True, "av")],
    ids=["av-off", "av-full", "av-region"],
)
def test_refinement_routes_by_refine_style(
    monkeypatch,  # noqa: ANN001
    active_volume: bool,
    artn_forces: str,
) -> None:
    """refine_style = artn sends artn_refine the minimum and the rcut saddle."""
    calls, results = _refinement_call("artn", active_volume, monkeypatch, artn_forces)

    assert [op for op, _ in calls] == ["artn_refine"]
    sent = calls[0][1]
    np.testing.assert_array_equal(sent["positions"], POSITIONS)  # the minimum
    np.testing.assert_array_equal(sent["saddle_idx"], NEIGHBORS)
    np.testing.assert_allclose(sent["saddle_positions"], SADDLE)
    assert sent["central_atom"] == 0 and sent["delr_sad_thr"] == 0.4
    assert sent["seeds"] == [
        int(s) for s in refine_seed_stream(9).integers(0, 2**32, size=5)
    ]
    if artn_forces == "av":
        # The AV's own pre-relaxation holds its buffer; no radius is sent.
        assert sent["options"] == {**OPTIONS, "av_radii": (3.0, 6.0)}
        assert sent["relax_radius"] is None
    else:
        assert sent["options"] == OPTIONS
    if active_volume:
        if artn_forces == "full":
            assert sent["relax_radius"] == 3.0
        np.testing.assert_array_equal(sent["movable"], [0, 1, 2, 3])  # within rmov
    else:
        assert sent["relax_radius"] is None
        np.testing.assert_array_equal(sent["movable"], np.arange(5))
    # The refiner's barrier is used as it is, total energy or not.
    assert results[0].ok_value().dE_forward == pytest.approx(0.6)


def test_refine_style_partn_keeps_the_partn_call(monkeypatch) -> None:  # noqa: ANN001
    """The default route is unchanged: partn_refine with the saddle system."""
    calls, results = _refinement_call("partn", False, monkeypatch)

    assert [op for op, _ in calls] == ["partn_refine"]
    sent = calls[0][1]
    assert set(sent) == {
        "config",
        "central_atom_idx",
        "positions",
        "types",
        "saddle_idx",
    }
    expected = POSITIONS.copy()
    expected[NEIGHBORS] = SADDLE
    np.testing.assert_allclose(sent["positions"], expected)
    assert results[0].ok_value().dE_forward == pytest.approx(-99.4 + 100.0)


def test_the_basin_reaches_the_refiner_with_the_same_arguments(monkeypatch) -> None:  # noqa: ANN001
    """basin.py's refinement of an absorbing event sends what refinement.py sends."""
    monkeypatch.setattr("pykmc.basins.basin.PointSetRegistration", IdentityPSR)
    from_refinement, _ = _refinement_call("artn", False, monkeypatch)

    config = _config("artn", False)
    config.rateconstant = None
    config.atomicenvironment.style = "cna"
    manager = RecordingManager()
    rates = SimpleNamespace(
        compute=lambda requests: [Ok(SimpleNamespace(k=2.0)) for _ in requests]
    )
    basin = BasinsGenericEvents(
        config, SimpleNamespace(table=pd.DataFrame([_event()])), set(), manager,
        rate_constant=rates, refine_seeds=refine_seed_stream(9),
    )  # fmt: skip
    neighbors = SimpleNamespace(get_neighbors=lambda kind, atom: NEIGHBORS.copy())
    basin.states = {
        0: StateData(_system(), environment=object(), neighbors_list=neighbors)
    }
    basin.connectivity_table = SimpleNamespace(
        df=pd.DataFrame(
            [
                dict(
                    transient=False, state=0, state_connexion=1, event_connexion=3,
                    central_atom=0, sym=0, dE_forward=np.nan, k_forward=np.nan,
                )
            ]
        )
    )  # fmt: skip

    assert basin.refine_absorbing(None).is_ok()
    refine_calls = [kw for op, kw in manager.calls if op == "artn_refine"]
    assert len(refine_calls) == 1
    sent, expected = refine_calls[0], from_refinement[0][1]
    assert sent.keys() == expected.keys()
    for key, value in expected.items():
        if isinstance(value, np.ndarray):
            np.testing.assert_allclose(sent[key], value, err_msg=key)
        else:
            assert sent[key] == value, key
    # The barrier is the refiner's, not minus the state's energy.
    assert basin.connectivity_table.df.loc[0, "dE_forward"] == pytest.approx(0.6)


def test_refine_style_artn_needs_the_run_seeds() -> None:
    """Both callers refuse to draw seeds of their own."""
    config = _config("artn", False)
    with pytest.raises(ValueError, match="seeds"):
        Refinement(config, None, None, None, None, None)
    config.rateconstant = None
    with pytest.raises(ValueError, match="seeds"):
        BasinsGenericEvents(config, None, set(), None, rate_constant=object())


def test_seeds_do_not_depend_on_the_workers() -> None:
    """Seeds are drawn at submit, in submit order, whatever runs the jobs."""
    sent = []
    for workers in (1, 4):
        manager = RecordingManager(workers)
        stream = refine_seed_stream(5)
        for atom in (0, 1, 2):
            submit_artn_refine(
                manager, _config("artn", False), stream, atom, POSITIONS, CELL,
                ["Fe"] * 5, True, NEIGHBORS, SADDLE,
            )  # fmt: skip
        sent.append([kw["seeds"] for _, kw in manager.calls])
    assert sent[0] == sent[1]
    assert len({seed for seeds in sent[0] for seed in seeds}) == 15


# ---------------------------------------------------------------- logging


def test_ignored_settings_name_what_still_reads_zseed() -> None:
    """r_neigen is ignored by the refiner; zseed's reader is named, or nothing."""
    partn = PartnConfig(zseed=7, neigen=2, r_neigen=3)

    search_only = ignored_partn_parameters(partn, "artn", "partn")
    assert set(search_only) == {"neigen", "zseed"}
    assert "pARTn's refinement still reads zseed" in search_only["zseed"]

    both = ignored_partn_parameters(partn, "artn", "artn")
    assert set(both) == {"neigen", "r_neigen", "zseed"}
    assert "searches and refinements" in both["zseed"]
    assert "nothing reads zseed" in both["zseed"]

    refine_only = ignored_partn_parameters(partn, "partn", "artn")
    assert set(refine_only) == {"r_neigen", "zseed"}
    assert "pARTn's search still reads zseed" in refine_only["zseed"]
    assert (
        "nothing reads zseed"
        in ignored_partn_parameters(partn, "dimer", "artn")["zseed"]
    )

    assert ignored_partn_parameters(PartnConfig(), "partn", "artn") == {}
    assert ignored_partn_parameters(partn, "partn", "partn") == {}


def test_the_refiner_line_is_logged_once_at_start(caplog) -> None:  # noqa: ANN001
    """With style != artn the initializer logs the refiner's line; with artn, the search does."""
    warnings = []
    for style in ("partn", "artn"):
        config = SimpleNamespace(
            eventsearch=SimpleNamespace(style=style, refine_style="artn"),
            partn=PartnConfig(zseed=7, r_neigen=2),
        )
        kmc = SimpleNamespace(
            config=config,
            loggers=SimpleNamespace(warning=lambda name, line: warnings.append(line)),
        )
        Initializer(kmc).log_ignored_refine_settings()
    assert len(warnings) == 1
    assert warnings[0].startswith("[EventSearch] refine_style = artn ignores")
    assert (
        "r_neigen = 2" in warnings[0]
        and "pARTn's search still reads zseed" in warnings[0]
    )


def test_the_search_line_covers_the_refiner_with_both_artn() -> None:
    """With style = artn, the search's line lists r_neigen and that nothing reads zseed."""
    from pykmc.eventsearch import SearchRequest
    from tests.eventsearch.test_dimer_strategy import CELL as SEARCH_CELL
    from tests.eventsearch.test_dimer_strategy import PBC, OrderedManager, _av_config
    from tests.eventsearch.test_dimer_strategy import POSITIONS as SEARCH_X

    config = _av_config(rmov=None)
    config.eventsearch.refine_style = "artn"
    config.partn = PartnConfig(zseed=7, r_neigen=2)
    lines = []

    class Recorder(logging.Handler):
        def emit(self, record: logging.LogRecord) -> None:
            lines.append(record.getMessage())

    logger = logging.getLogger("log")
    recorder = Recorder(logging.WARNING)
    logger.addHandler(recorder)
    try:
        request = SearchRequest(
            1, SEARCH_X, SEARCH_CELL, np.array(["Ni"] * 5), PBC, 9, np.arange(5)
        )
        ArtnStrategy(config).submit([request], OrderedManager(1, [0]))
    finally:
        logger.removeHandler(recorder)
    assert len(lines) == 1
    assert "r_neigen = 2" in lines[0] and "nothing reads zseed" in lines[0]


def test_each_attempt_is_logged_at_positive_verbosity(caplog) -> None:  # noqa: ANN001
    """One debug line per attempt: seed, end, delr_sad and force calls."""
    raw = RefineResult(
        success=False, reason="far_saddle", saddle_positions=None, e_saddle=None,
        e_start=None, barrier=None, curvature=None, delr_sad=0.7,
        attempts=(
            pykmc.artn.RefineAttempt(11, "max_force_calls", None, 299),
            pykmc.artn.RefineAttempt(12, None, 0.7, 80),
        ),
        force_calls=379,
    )  # fmt: skip
    with caplog.at_level(logging.DEBUG, logger="log"):
        result = artn_refine_result(raw, 4, verbosity=2)
        artn_refine_result(raw, 4, verbosity=0)
    lines = [r.getMessage() for r in caplog.records if r.name == "log"]

    assert lines == [
        "ARTn refinement atom 4 attempt 1 seed 11: max_force_calls, delr_sad=None A, 299 force calls",
        "ARTn refinement atom 4 attempt 2 seed 12: saddle, delr_sad=0.7 A, 80 force calls",
    ]
    error = result.err_value()
    assert error.type == ErrorType.EVENT_NOT_FOUND
    assert error.variables["delr_sad"] == 0.7 and error.variables["seeds"] == [11, 12]
