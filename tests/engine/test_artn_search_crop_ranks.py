"""The ARTn active-volume search on a worker of one rank and of several, with no LAMMPS.

Serial: pytest tests/engine/test_artn_search_crop_ranks.py

As for the dimer, a search with ``av_radii`` (``[ActiveVolume] artn_forces =
av``) runs on the worker's leader rank, on a private one-rank instance. The
other ranks return at once and wait at the operation's closing barrier. LAMMPS,
pypARTn and the ARTn core are replaced by stand-ins that record their calls.
"""

import sys
from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest
from mpi4py import MPI

from pykmc.dimer import DimerResult
from pykmc.engine.artn_search import ArtnSearchExtension

CELL = np.diag([10.0, 10.0, 10.0])
# Atom 0 is the central atom; atom 1 is within rmov, atom 2 in the buffer
# (beyond rmov, within ract) and atom 3 beyond ract.
START = np.array([[1.0, 1.0, 1.0], [2.5, 1.0, 1.0], [1.0, 3.5, 1.0], [6.0, 6.0, 6.0]])
CROP_RADII = (2.0, 3.0)
CROP = np.array([0, 1, 2])
SEED = 7


class RankComm:
    """The rank and size of one rank of a worker; the crop needs no collective."""

    def __init__(self, rank: int, size: int) -> None:
        self.rank, self.size = rank, size

    def Get_rank(self) -> int:  # noqa: N802
        """Return the rank."""
        return self.rank

    def Get_size(self) -> int:  # noqa: N802
        """Return the worker's size."""
        return self.size


@pytest.fixture
def calls(monkeypatch: pytest.MonkeyPatch) -> dict[str, list]:
    """Replace the instance, the crop builder and the core; record their calls."""
    seen: dict[str, list] = {"instances": [], "crops": [], "cores": []}

    class Instance:
        def __init__(self, config: Any, comm: Any, engine_id: str, **_: Any) -> None:
            self.comm = comm
            self.lmp = self.numpy = self
            self.natoms = 0
            seen["instances"].append(self)

        def start(self) -> None:
            pass

        close = start

        def command(self, text: str) -> None:
            """Accept the neighbor-reuse setup the active volume sends."""

        def gather_atoms(self, name: str, kind: int, count: int) -> list[int]:
            assert name == "type"
            return [1] * self.natoms

        def extract_atom(self, name: str) -> np.ndarray:
            assert name == "mass"
            return np.array([0.0, 58.71])

    def partn_search_AV(engine, config, central, positions, cell, types):  # noqa: ANN001, ANN202, N802
        av = config.activevolume
        seen["crops"].append((engine, (av.rmov, av.ract), central))
        engine.natoms = len(CROP)
        return CROP, None

    def core(positions, cell, pbc, mask, central, seed, query, masses, **options):  # noqa: ANN001, ANN003, ANN202
        seen["cores"].append(
            dict(positions=positions, pbc=pbc, mask=mask, central=central, seed=seed)
        )
        seen["cores"][-1].update(masses=masses, options=options)
        fields = dict.fromkeys(DimerResult._fields)
        fields.update(
            success=True,
            saddle_positions=positions + 0.1,
            min1_positions=positions,
            min2_positions=positions + 0.2,
            mode=np.ones_like(positions),
        )
        return DimerResult(**fields)

    def distributed_search(*args: Any, **kwargs: Any) -> None:
        raise AssertionError("a crop search must not take the multi-rank path")

    monkeypatch.setitem(
        sys.modules, "pykmc.engine.lammps", SimpleNamespace(LammpsEngine=Instance)
    )
    monkeypatch.setitem(
        sys.modules,
        "pykmc.activevolume.active_volume",
        SimpleNamespace(partn_search_AV=partn_search_AV, set_positions=None),
    )
    monkeypatch.setitem(
        sys.modules,
        "pykmc.engine.artn_ranks",
        SimpleNamespace(distributed_search=distributed_search),
    )
    monkeypatch.setattr("pykmc.engine.artn_search.run_artn", core)
    return seen


def _crop_search(comm: Any) -> DimerResult | None:
    """Run one crop search on a new worker extension with ``comm``."""
    extension = ArtnSearchExtension(
        SimpleNamespace(
            comm=comm,
            config=SimpleNamespace(pair_style="eam/alloy"),
            engine_id=0,
            register=lambda extension: None,
        )
    )
    return extension.artn_search(
        START,
        ["Ni"] * len(START),
        CELL,
        [True] * 3,
        np.arange(len(START)),
        0,
        SEED,
        {"av_radii": CROP_RADII, "ninit": 2},
    )


def _check_one_crop_search(seen: dict[str, list], result: DimerResult) -> None:
    """One instance on one rank, one crop, one core call, mapped back."""
    (instance,) = seen["instances"]
    assert instance.comm is MPI.COMM_SELF
    assert seen["crops"] == [(instance, CROP_RADII, 0)]
    (core,) = seen["cores"]
    np.testing.assert_array_equal(core["positions"], START[CROP])
    assert core["pbc"].tolist() == [True] * 3
    assert core["mask"].tolist() == [True] * 3
    assert (core["central"], core["seed"]) == (0, SEED)
    assert core["options"] == {"ninit": 2}
    np.testing.assert_array_equal(core["masses"], [58.71] * 3)

    assert result.success
    np.testing.assert_array_equal(result.saddle_positions[CROP], START[CROP] + 0.1)
    np.testing.assert_array_equal(result.min2_positions[CROP], START[CROP] + 0.2)
    np.testing.assert_array_equal(result.mode[CROP], np.ones((3, 3)))
    for name in ("saddle_positions", "min1_positions", "min2_positions"):
        np.testing.assert_array_equal(getattr(result, name)[3], START[3])
    np.testing.assert_array_equal(result.mode[3], np.zeros(3))


@pytest.mark.parametrize("comm", [None, RankComm(0, 1)], ids=["no-comm", "one-rank"])
def test_single_rank_worker_runs_the_crop(
    calls: dict[str, list], comm: RankComm | None
) -> None:
    """A one-rank worker runs the crop search on a private one-rank instance."""
    _check_one_crop_search(calls, _crop_search(comm))


def test_multi_rank_worker_runs_the_crop_on_its_leader(
    calls: dict[str, list],
) -> None:
    """On four ranks the leader runs the crop as one rank would; the rest return."""
    for rank in (1, 2, 3):
        assert _crop_search(RankComm(rank, 4)) is None
    assert calls == {"instances": [], "crops": [], "cores": []}
    _check_one_crop_search(calls, _crop_search(RankComm(0, 4)))
