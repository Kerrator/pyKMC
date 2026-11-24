"""The ARTn search worker operation on the dimer's dedicated serial LAMMPS instance.

Serial: pytest tests/engine/test_engine_artn_search.py

The system is the dimer op tests' fcc Ni vacancy (4x4x4 cubic cells minus one
atom, shifted so no atom sits on a face); the central atom's hop into the
vacancy crosses the x and y boundaries, 1.185 eV with this potential.
"""

import numpy as np
import pytest

lammps = pytest.importorskip("lammps")

from mpi4py import MPI  # noqa: E402

import pykmc.artn  # noqa: E402
from pykmc.config import PartnConfig  # noqa: E402
from pykmc.dimer import DimerResult  # noqa: E402
from pykmc.engine import Engine  # noqa: E402
from pykmc.engine.artn_search import ArtnSearchExtension  # noqa: E402
from pykmc.eventsearch.strategies.artn import artn_options  # noqa: E402

from .test_engine_dimer_search import (  # noqa: E402, F401 (fixtures)
    DELR_THR,
    HOP_BARRIER,
    _frozen_movable,
    _make_engine,
    eam_config,
    vacancy,
)

# pyKMC's [pARTn] defaults; the rad push (1 Å) reaches the central atom only.
OPTIONS = artn_options(PartnConfig())
# A seed whose search reaches the hop. With these defaults most seeds end in
# eigenvalue_lost: the plain core has no perpendicular step limit out of the
# basin, so its first relaxation there undoes the strain of the two pushes.
HOP_SEED = 3


def _close(engine: Engine) -> None:
    instance = engine._extensions["ArtnSearchExtension"]._force_engine
    if instance is not None:
        instance.close()
    engine.close()


def _search(engine: Engine, system, seed: int = HOP_SEED, **kwargs) -> DimerResult:  # noqa: ANN001
    request = dict(
        positions=system.positions,
        types=system.types,
        cell=system.cell,
        pbc=system.pbc,
        movable=np.arange(len(system.positions)),
        central_atom=system.central,
        seed=seed,
        options=OPTIONS,
    )
    request.update(kwargs)
    return engine.artn_search(**request)


class TestArtnSearchSerial:
    @pytest.fixture(autouse=True)
    def require_serial(self) -> None:
        if MPI.COMM_WORLD.Get_size() > 1:
            pytest.skip("serial tests must run without mpirun")

    @pytest.fixture
    def engine(self, eam_config, vacancy):  # noqa: ANN001, ANN201, F811
        engine = _make_engine(
            eam_config,
            None,
            vacancy.positions,
            vacancy.types,
            vacancy.cell,
            vacancy.pbc,
            ArtnSearchExtension,
        )
        yield engine
        _close(engine)

    def test_finds_the_hop_and_counts_every_force_call(  # noqa: D102
        self,
        engine,  # noqa: ANN001
        vacancy,  # noqa: ANN001, F811
        monkeypatch,  # noqa: ANN001
    ) -> None:
        core = pykmc.artn.artn_search
        seen = {}

        def counted(positions, cell, pbc, mask, central, seed, query, **options):  # noqa: ANN001, ANN003, ANN202
            seen["calls"] = 0
            seen["masses"] = options["masses"]

            def query_counted(x):  # noqa: ANN001, ANN202
                seen["calls"] += 1
                return query(x)

            return core(positions, cell, pbc, mask, central, seed, query_counted, **options)

        monkeypatch.setattr("pykmc.engine.artn_search.run_artn", counted)
        result = _search(engine, vacancy)

        assert isinstance(result, DimerResult)
        assert result.success and result.reason is None
        assert result.seed == HOP_SEED
        assert result.e_saddle - result.e_start == pytest.approx(HOP_BARRIER, abs=0.02)
        assert result.curvature < 0
        assert min(result.delr1, result.delr2) < DELR_THR
        assert 0 < result.force_calls_saddle < result.force_calls_total
        assert result.force_calls_total == seen["calls"]
        # FIRE gets LAMMPS's masses (ASE's table).
        np.testing.assert_allclose(seen["masses"], 58.6934, atol=1e-3)

    def test_shares_the_dimer_instance_and_repeats_exactly(  # noqa: D102
        self,
        engine,  # noqa: ANN001
        vacancy,  # noqa: ANN001, F811
    ) -> None:
        extension = engine._extensions["ArtnSearchExtension"]
        first = _search(engine, vacancy)
        instance = extension._force_engine
        assert instance is not None and instance.comm is MPI.COMM_SELF
        again = _search(engine, vacancy)
        assert extension._force_engine is instance
        for name, value in first._asdict().items():
            if isinstance(value, np.ndarray):
                assert np.array_equal(value, getattr(again, name)), name
            else:
                assert value == getattr(again, name), name

    def test_frozen_atoms_stay(self, engine, vacancy) -> None:  # noqa: ANN001, D102, F811
        movable = _frozen_movable(vacancy)
        frozen = np.setdiff1d(np.arange(len(vacancy.positions)), movable)
        result = _search(engine, vacancy, movable=movable)

        assert result.success
        for name in ("saddle_positions", "min1_positions", "min2_positions"):
            np.testing.assert_allclose(
                getattr(result, name)[frozen], vacancy.positions[frozen], atol=1e-12
            )
        assert np.array_equal(result.mode[frozen], np.zeros((len(frozen), 3)))
