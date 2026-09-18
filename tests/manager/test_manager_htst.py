"""``LammpsHTSTExtension`` through ``EngineManagerFactory`` under real MPI.

Launch contract (same file, two layouts, skip guards on the world size):

- ``mpirun -n 4``: ``n_workers=3``, one rank per worker.
- ``mpirun -n 5``: ``n_workers=2``, two ranks per worker (the non-root rank of
  each worker returns ``None`` and the manager reports the root's result).

Rank 0 is the manager; it also computes the serial reference on its own
``comm=None`` engine while the workers wait for messages. The workers are
initialised with the tracked SW-Si vacancy-hop fixture.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pytest

pytest.importorskip("lammps")

from mpi4py import MPI  # noqa: E402

from pykmc.engine.htst_lammps import LammpsHTSTExtension  # noqa: E402
from pykmc.engine.lammps import LammpsEngine  # noqa: E402
from pykmc.factory import EngineManagerFactory  # noqa: E402
from pykmc.htst import EventPrefactors, HTSTEventRequest, HTSTSettings  # noqa: E402

_ROOT = Path(__file__).resolve().parents[2]
_SI_SW = _ROOT / "examples" / "Si_vac" / "Si.sw"
_HOP_FIXTURE = _ROOT / "tests" / "data" / "htst_si_vacancy_hop.npz"
_ALL_PERIODIC = (True, True, True)
_LAYOUTS = {4: 3, 5: 2}  # world size -> n_workers


@dataclass
class _SWConfig:
    """Stillinger-Weber Si (masses come from pyKMC's species map)."""

    pair_style: str = "sw"
    pair_coeff: str = f"* * {_SI_SW} Si"
    min_style: str = "cg"
    minimize: str = "1e-6 1e-8 1000 10000"
    frz_min: str = "1e-4 1e-6 100 1000"
    verbosity: int = 0


def _local_comm_size(comm: MPI.Comm, **_: Any) -> int:
    """extra_op: size of the worker's active communicator."""
    return int(comm.Get_size())


def _require_layout() -> int:
    """Skip unless the world size is one of the two documented layouts."""
    size = MPI.COMM_WORLD.Get_size()
    if size not in _LAYOUTS:
        pytest.skip("TestManagerHTST runs with mpirun -n 4 or mpirun -n 5")
    if not _SI_SW.is_file() or not _HOP_FIXTURE.is_file():
        pytest.skip("SW-Si potential or hop fixture not present")
    return _LAYOUTS[size]


@pytest.fixture(scope="session")
def si_hop() -> dict[str, Any]:
    """Load the tracked SW-Si vacancy-hop fixture as plain arrays."""
    if not _HOP_FIXTURE.is_file():
        pytest.skip(f"{_HOP_FIXTURE.name} not present")
    with np.load(_HOP_FIXTURE, allow_pickle=False) as data:
        return {key: np.array(data[key]) for key in data.files}


def _request(si_hop: dict[str, Any], **settings: Any) -> HTSTEventRequest:
    """Fixture request; SW sets no masses, so the species map's are authoritative."""
    types = tuple(str(t) for t in si_hop["types"])
    return HTSTEventRequest(
        event_key=("ref", int(si_hop["idx_ref_forward"])),
        min1_positions=si_hop["min1_positions"].copy(),
        saddle_positions=si_hop["saddle_positions"].copy(),
        min2_positions=si_hop["min2_positions"].copy(),
        types=types,
        species=("Si",),
        masses=(28.085,),
        cell=si_hop["cell"].copy(),
        pbc=_ALL_PERIODIC,
        center_index=int(si_hop["central_atom_idx"]),
        settings=HTSTSettings(**settings),
    )


def _serial_reference(
    config: _SWConfig, si_hop: dict[str, Any], request: HTSTEventRequest
) -> EventPrefactors:
    """Compute the same event on a single-rank engine with the extension (rank 0).

    ``comm=None`` would make the lammps wrapper use ``MPI_COMM_WORLD`` (a
    collective the workers never join), so the reference runs on ``COMM_SELF``.
    """
    engine = LammpsEngine(config=config, comm=MPI.COMM_SELF, engine_id=0)
    engine.start()
    try:
        engine.initialize_parameters()
        engine.initialize_system(
            types=list(request.types),
            positions=si_hop["min1_positions"],
            cell=si_hop["cell"],
            pbc=_ALL_PERIODIC,
        )
        engine.initialize_potential()
        assert engine.full_system.masses == request.masses
        return LammpsHTSTExtension(engine).compute_event_prefactors(request)
    finally:
        engine.close()


def _assert_same_prefactors(
    result: EventPrefactors, reference: EventPrefactors, rel: float = 1e-6
) -> None:
    """Both directions accepted and equal within ``rel``."""
    assert isinstance(result, EventPrefactors)
    assert result.method == reference.method == "lammps_eskm"
    assert result.n_free == reference.n_free == 43
    assert result.event_key == reference.event_key
    for got, want in (
        (result.forward, reference.forward),
        (result.backward, reference.backward),
    ):
        assert got.ok and want.ok, (got.reason, want.reason)
        assert got.nu0_hz == pytest.approx(want.nu0_hz, rel=rel)
        assert got.n_positive_min == want.n_positive_min
        assert got.n_negative_saddle == want.n_negative_saddle


@pytest.mark.mpi
class TestManagerHTST:
    """Factory-launched pool with the HTST extension registered on every engine."""

    @pytest.fixture(autouse=True)
    def setup(self, si_hop: dict[str, Any]) -> None:
        """Launch the pool, initialise the workers, preflight, tear down."""
        self.n_workers = _require_layout()
        self.config = _SWConfig()
        self.si_hop = si_hop
        self.rank = MPI.COMM_WORLD.Get_rank()
        MPI.COMM_WORLD.Barrier()
        self.manager = EngineManagerFactory(
            engine_style="lammps",
            engine_config=self.config,
            n_workers=self.n_workers,
            comm=MPI.COMM_WORLD,
            engine_extensions=[LammpsHTSTExtension],
            extra_ops={"local_comm_size": _local_comm_size},
        ).launch()
        if self.rank == 0:
            types = [str(t) for t in si_hop["types"]]
            self.manager.broadcast("start")
            self.manager.broadcast("initialize_parameters")
            self.manager.broadcast(
                "initialize_system",
                types=types,
                positions=si_hop["min1_positions"],
                cell=si_hop["cell"],
                pbc=_ALL_PERIODIC,
            )
            self.manager.broadcast("initialize_potential")
            self.manager.broadcast("htst_preflight")
        yield
        if self.manager is not None:
            self.manager.shutdown()
        MPI.COMM_WORLD.Barrier()

    def test_layout_and_operations(self) -> None:
        """Each worker's local communicator has the documented size; ops are listed."""
        if self.rank != 0:
            return
        expected = (MPI.COMM_WORLD.Get_size() - 1) // self.n_workers
        sizes = [
            self.manager.submit("local_comm_size").result()
            for _ in range(2 * self.n_workers)
        ]
        assert sizes == [expected] * (2 * self.n_workers)
        ops = self.manager.list_ops()
        assert "compute_event_prefactors" in ops
        assert "htst_preflight" in ops

    def test_preflight_result_reaches_the_manager(self) -> None:
        """``htst_preflight`` returns the root's report through a Future."""
        if self.rank != 0:
            return
        report = self.manager.submit("htst_preflight").result()
        assert report["phonon"] is True
        assert report["species"] == ("Si",)

    def test_prefactors_match_the_serial_result(self) -> None:
        """Concurrent submissions all equal the serial extension to 1e-6 relative."""
        if self.rank != 0:
            return
        request = _request(self.si_hop, free_radius=6.0, zone_radius=10.0)
        reference = _serial_reference(self.config, self.si_hop, request)
        futures = [
            self.manager.submit("compute_event_prefactors", request=request)
            for _ in range(self.n_workers + 1)
        ]
        for future in futures:
            _assert_same_prefactors(future.result(), reference)
        # A full-system request through the pool agrees with the crop as well.
        full = self.manager.submit(
            "compute_event_prefactors", request=_request(self.si_hop, free_radius=6.0)
        ).result()
        _assert_same_prefactors(full, reference, rel=0.02)

    def test_invalid_request_fails_the_future_and_the_worker_survives(self) -> None:
        """Validation errors surface on the Future; every worker stays usable."""
        if self.rank != 0:
            return
        good = _request(self.si_hop, free_radius=4.0, zone_radius=8.0)
        bad = HTSTEventRequest(
            event_key=good.event_key,
            min1_positions=good.min1_positions,
            saddle_positions=good.saddle_positions,
            min2_positions=good.min2_positions,
            types=good.types,
            species=good.species,
            masses=good.masses,
            cell=good.cell,
            pbc=good.pbc,
            center_index=len(good.types),
            settings=good.settings,
        )
        failures = [
            self.manager.submit("compute_event_prefactors", request=bad)
            for _ in range(self.n_workers)
        ]
        for future in failures:
            with pytest.raises(RuntimeError, match="HTSTRequestError"):
                future.result()
        with pytest.raises(RuntimeError, match="zone_radius"):
            self.manager.submit(
                "compute_event_prefactors",
                request=_request(self.si_hop, free_radius=4.0, zone_radius=4.0),
            ).result()
        results = [
            self.manager.submit("compute_event_prefactors", request=good).result()
            for _ in range(self.n_workers)
        ]
        for result in results:
            assert result.n_free == 13
            assert result.forward.ok and result.backward.ok
            assert result.forward.nu0_hz == pytest.approx(results[0].forward.nu0_hz)
