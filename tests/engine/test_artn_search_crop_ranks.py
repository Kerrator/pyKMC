"""The ARTn active-volume search and refinement on a worker of one rank and of several, with no LAMMPS.

Serial: pytest tests/engine/test_artn_search_crop_ranks.py

As for the dimer, a search or refinement with ``av_radii`` (``[ActiveVolume]
artn_forces = av``) runs on the worker's leader rank, on a private one-rank
instance. The other ranks return at once and wait at the operation's closing
barrier. LAMMPS, pypARTn and the ARTn core are replaced by stand-ins that record
their calls.
"""

import sys
from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest
from mpi4py import MPI

from pykmc.artn import RefineResult
from pykmc.dimer import DimerResult
from pykmc.engine.artn_search import ArtnSearchExtension

CELL = np.diag([10.0, 10.0, 10.0])
# Atom 0 is the central atom; atom 1 is within rmov, atom 2 in the buffer
# (beyond rmov, within ract) and atom 3 beyond ract.
START = np.array([[1.0, 1.0, 1.0], [2.5, 1.0, 1.0], [1.0, 3.5, 1.0], [6.0, 6.0, 6.0]])
CROP_RADII = (2.0, 3.0)
CROP = np.array([0, 1, 2])
SEED = 7
# Refinement: the rcut atoms, their reference saddle, the AV after the
# pre-relaxation, its start energy and the pre-relaxation's iterations.
SADDLE_IDX = np.array([0, 1])
SADDLE_POSITIONS = START[SADDLE_IDX] + 0.3
AV_START = START[CROP] + 0.05
E_INIT = -12.5
NTIMESTEP = 9
SEEDS = [11, 12]
LAMMPS = dict(
    pair_style="zero 5.7",
    search_pair_style="mtp/small/kk",
    min_style="cg",
    frz_min="1e-6 1e-8 100 1000",
)


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
        def __init__(
            self, config: Any, comm: Any, engine_id: str, *, search_instance: bool
        ) -> None:
            assert search_instance
            self.commands: list[str] = []
            self.comm = comm
            self.lmp = self.numpy = self
            self.natoms = 0
            self.ntimestep = 0
            self.x = np.zeros((0, 3))
            seen["instances"].append(self)

        def start(self) -> None:
            pass

        close = start

        def command(self, text: str) -> None:
            self.commands.append(text)
            if text == "run 1 pre no post no":
                self.ntimestep += 1

        def scatter_atoms(self, name: str, kind: int, count: int, values: Any) -> None:
            assert name == "v"

        def get_thermo(self, name: str) -> float:
            assert name == "pe"
            return E_INIT

        def gather_atoms(self, name: str, kind: int, count: int) -> Any:
            if name == "f":
                return np.ones((self.natoms, 3)).ravel()
            if name == "x":
                return self.x.ravel().copy()
            assert name == "type"
            return [1] * self.natoms

        def extract_global(self, name: str) -> int:
            assert name == "ntimestep"
            return self.ntimestep

        def extract_atom(self, name: str) -> np.ndarray:
            assert name == "mass"
            return np.array([0.0, 58.71])

    def partn_search_AV(engine, config, central, positions, cell, types):  # noqa: ANN001, ANN202, N802
        av = config.activevolume
        seen["crops"].append((engine, (av.rmov, av.ract), central))
        engine.natoms = len(CROP)
        return CROP, None

    def partn_refine_AV(engine, config, central, positions, cell, types, idx, saddle):  # noqa: ANN001, ANN202, N802
        av, lmp = config.activevolume, config.lammps
        seen["crops"].append(
            (
                engine,
                (av.rmov, av.ract),
                central,
                av.AV_debug,
                lmp.min_style,
                lmp.frz_min,
            )
        )
        np.testing.assert_array_equal(idx, SADDLE_IDX)
        np.testing.assert_array_equal(saddle, SADDLE_POSITIONS)
        engine.natoms, engine.ntimestep, engine.x = len(CROP), NTIMESTEP, AV_START
        return E_INIT, CROP, None

    def refine(positions, cell, pbc, mask, central, seeds, query, masses, **options):  # noqa: ANN001, ANN003, ANN202
        seen["cores"].append(
            dict(positions=positions, pbc=pbc, mask=mask, central=central, seeds=seeds)
        )
        seen["cores"][-1].update(masses=masses, options=options)
        _, forces = query(positions)
        np.testing.assert_array_equal(forces, [[1, 1, 1], [1, 1, 1], [0, 0, 0]])
        query(positions + 0.001)
        return RefineResult(
            success=True, reason=None, saddle_positions=positions + 0.1,
            e_saddle=E_INIT + 0.6, e_start=options["e_start"], barrier=0.6,
            curvature=-1.0, delr_sad=0.01, attempts=(), force_calls=5,
        )  # fmt: skip

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
        SimpleNamespace(
            partn_search_AV=partn_search_AV,
            partn_refine_AV=partn_refine_AV,
            set_positions=lambda engine, positions: None,
        ),
    )
    monkeypatch.setitem(
        sys.modules,
        "pykmc.engine.artn_ranks",
        SimpleNamespace(distributed_search=distributed_search),
    )
    monkeypatch.setattr("pykmc.engine.artn_search.run_artn", core)
    monkeypatch.setattr("pykmc.engine.artn_search.run_refine", refine)
    return seen


def _extension(comm: Any) -> ArtnSearchExtension:
    """Return the ARTn extension on a worker engine that is never driven.

    The worker engine has no LAMMPS methods: the crop path must not touch it.
    """
    return ArtnSearchExtension(
        SimpleNamespace(
            comm=comm,
            config=SimpleNamespace(**LAMMPS),
            engine_id=0,
            register=lambda extension: None,
        )
    )


def _crop_search(comm: Any) -> DimerResult | None:
    """Run one crop search on a new worker extension with ``comm``."""
    return _extension(comm).artn_search(
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


# ---------------------------------------------------------------- refinement


def _av_refine(comm: Any) -> RefineResult | None:
    """Run one AV refinement on a new worker extension with ``comm``."""
    return _extension(comm).artn_refine(
        START,
        ["Ni"] * len(START),
        CELL,
        [True] * 3,
        np.arange(len(START)),
        0,
        SADDLE_IDX,
        SADDLE_POSITIONS,
        SEEDS,
        0.4,
        relax_radius=None,
        options={
            "av_radii": CROP_RADII,
            "ninit": 0,
            "attempt_options": [{"nperp_start": -1}],
        },
    )


def _check_one_av_refinement(seen: dict[str, list], result: RefineResult) -> None:
    """One instance on one rank, partn_refine_AV's build, one refiner call, mapped back."""
    (instance,) = seen["instances"]
    assert instance.comm is MPI.COMM_SELF
    assert seen["crops"] == [
        (instance, CROP_RADII, 0, False, LAMMPS["min_style"], LAMMPS["frz_min"])
    ]
    (core,) = seen["cores"]
    # The refiner starts where the pre-relaxation left LAMMPS, on the AV.
    np.testing.assert_array_equal(core["positions"], AV_START)
    assert core["pbc"].tolist() == [True] * 3
    assert core["mask"].tolist() == [True] * 3  # the buffer moves, as under pARTn
    assert (core["central"], core["seeds"]) == (0, SEEDS)
    assert core["options"] == {
        "delr_sad_thr": 0.4,
        "e_start": E_INIT,  # partn_refine_AV's E_init, not the core's own
        "ninit": 0,
        "attempt_options": [{"nperp_start": -1}],
    }
    np.testing.assert_array_equal(core["masses"], [58.71] * 3)

    assert instance.commands[-2:] == ["run 0 post no", "run 1 pre no post no"]
    assert result.success and result.barrier == 0.6
    assert result.prerelax_iterations == NTIMESTEP
    np.testing.assert_array_equal(result.saddle_positions[CROP], AV_START + 0.1)
    np.testing.assert_array_equal(result.saddle_positions[3], START[3])


@pytest.mark.parametrize("comm", [None, RankComm(0, 1)], ids=["no-comm", "one-rank"])
def test_single_rank_worker_refines_on_the_av(
    calls: dict[str, list], comm: RankComm | None
) -> None:
    """A one-rank worker refines on the AV that partn_refine_AV builds."""
    _check_one_av_refinement(calls, _av_refine(comm))


def test_multi_rank_worker_refines_on_the_av_on_its_leader(
    calls: dict[str, list],
) -> None:
    """On four ranks the leader refines as one rank would; the rest build nothing."""
    for rank in (1, 2, 3):
        assert _av_refine(RankComm(rank, 4)) is None
    assert calls == {"instances": [], "crops": [], "cores": []}
    _check_one_av_refinement(calls, _av_refine(RankComm(0, 4)))
