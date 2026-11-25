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
    _minimum_image,
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

            return core(
                positions, cell, pbc, mask, central, seed, query_counted, **options
            )

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
        # FIRE gets the instance's masses: the EAM file's 58.710, not ASE's 58.6934.
        instance = engine._extensions["ArtnSearchExtension"]._force_engine
        expected = np.asarray(instance.lmp.numpy.extract_atom("mass"))[1]
        assert expected == pytest.approx(58.710)
        np.testing.assert_array_equal(seen["masses"], expected)
        assert seen["masses"].shape == (len(vacancy.positions),)

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


# pARTn's active volume on the Ni cell (edge 14.1 Å): movable within 4.0 Å,
# buffer to 6.5 Å.
AV_RADII = (4.0, 6.5)
AV_OPTIONS = {**OPTIONS, "av_radii": AV_RADII}
# A seed whose active-volume search reaches the hop; its barrier is 1.220 eV.
AV_HOP_SEED = 3


def _av_sets(system):  # noqa: ANN001, ANN202
    """The active volume's atoms and buffer, from the builder `partn_search` uses."""
    from types import SimpleNamespace

    from pykmc.activevolume.active_volume import define_AV

    rmov, ract = AV_RADII
    config = SimpleNamespace(activevolume=SimpleNamespace(rmov=rmov, ract=ract))
    _, av_idx, buffer_idx = define_AV(
        config, system.central, system.positions, np.array(system.cell)
    )
    return av_idx, np.isin(av_idx, buffer_idx)


class TestArtnSearchAVSerial:
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

    def test_the_core_gets_partns_active_volume_and_forces(  # noqa: D102
        self,
        engine,  # noqa: ANN001
        eam_config,  # noqa: ANN001, F811
        vacancy,  # noqa: ANN001, F811
        monkeypatch,  # noqa: ANN001
    ) -> None:
        av_idx, buffer = _av_sets(vacancy)
        # The strategy's movable set, the atoms within rmov, less one neighbour
        # of the central atom, as [frozen_atoms] would hold it.
        offsets = _minimum_image(
            vacancy.positions[av_idx] - vacancy.positions[vacancy.central],
            vacancy.cell,
        )
        held = av_idx[np.argsort(np.linalg.norm(offsets, axis=1))[1]]
        movable = np.setdiff1d(av_idx[~buffer], [held])
        rng = np.random.default_rng(0)
        moved = vacancy.positions[av_idx] + rng.normal(0, 0.05, (len(av_idx), 3))
        seen = {}

        def fake(positions, cell, pbc, mask, central, seed, query, **options):  # noqa: ANN001, ANN003, ANN202
            seen.update(positions=positions, pbc=pbc, mask=mask, central=central)
            seen["start"] = query(positions)
            seen["moved"] = query(moved)
            fields = dict.fromkeys(DimerResult._fields)
            fields.update(success=False, reason="stub", seed=seed)
            fields.update(force_calls_saddle=0, force_calls_total=0)
            return DimerResult(**fields)

        monkeypatch.setattr("pykmc.engine.artn_search.run_artn", fake)
        result = _search(engine, vacancy, movable=movable, options=AV_OPTIONS)

        assert result.reason == "stub" and result.saddle_positions is None
        np.testing.assert_array_equal(seen["positions"], vacancy.positions[av_idx])
        assert av_idx[seen["central"]] == vacancy.central
        assert seen["pbc"].all()
        # The core moves the buffer, as pARTn does, but not the held atom.
        np.testing.assert_array_equal(seen["mask"], av_idx != held)
        assert 0 < buffer.sum() < len(av_idx) < len(vacancy.positions)
        # The forces of the active volume alone, built independently: zero on the buffer
        # and on the held atom.
        reference = _make_engine(
            eam_config,
            MPI.COMM_SELF,
            vacancy.positions[av_idx],
            [vacancy.types[i] for i in av_idx],
            vacancy.cell,
            np.ones(3, dtype=bool),
        )
        try:
            for name, x in (("start", vacancy.positions[av_idx]), ("moved", moved)):
                energy, forces = reference.get_energy_and_forces(x)
                forces[buffer | (av_idx == held)] = 0.0
                assert seen[name][0] == pytest.approx(energy, rel=1e-12), name
                np.testing.assert_allclose(seen[name][1], forces, rtol=0, atol=1e-10)
        finally:
            reference.close()
        assert np.abs(seen["start"][1][~buffer & (av_idx != held)]).max() > 0

    def test_a_av_search_maps_back_to_the_full_system(  # noqa: D102
        self,
        engine,  # noqa: ANN001
        vacancy,  # noqa: ANN001, F811
    ) -> None:
        av_idx, _ = _av_sets(vacancy)
        outside = np.setdiff1d(np.arange(len(vacancy.positions)), av_idx)
        result = _search(engine, vacancy, seed=AV_HOP_SEED, options=AV_OPTIONS)

        assert result.success and result.reason is None
        assert result.e_saddle - result.e_start == pytest.approx(HOP_BARRIER, abs=0.05)
        assert min(result.delr1, result.delr2) < DELR_THR
        for name in ("saddle_positions", "min1_positions", "min2_positions"):
            value = getattr(result, name)
            assert value.shape == vacancy.positions.shape
            np.testing.assert_array_equal(value[outside], vacancy.positions[outside])
        assert np.array_equal(result.mode[outside], np.zeros((len(outside), 3)))
        assert np.linalg.norm(result.mode) == pytest.approx(1.0)

    def test_only_the_search_instance_gets_search_cmdargs(  # noqa: D102
        self,
        eam_config,  # noqa: ANN001, F811
        vacancy,  # noqa: ANN001, F811
        monkeypatch,  # noqa: ANN001
    ) -> None:
        import dataclasses

        import pykmc.engine.lammps as engine_module

        opened = []
        real = engine_module.lammps

        def recording(**kwargs):  # noqa: ANN003, ANN202
            opened.append(kwargs["cmdargs"])
            return real(**kwargs)

        monkeypatch.setattr(engine_module, "lammps", recording)
        config = dataclasses.replace(eam_config)
        config.search_cmdargs = ["-echo", "none"]
        # The full path and the active-volume path each build the instance.
        for seed, options in ((HOP_SEED, OPTIONS), (AV_HOP_SEED, AV_OPTIONS)):
            opened.clear()
            engine = _make_engine(
                config,
                None,
                vacancy.positions,
                vacancy.types,
                vacancy.cell,
                vacancy.pbc,
                ArtnSearchExtension,
            )
            try:
                _search(engine, vacancy, seed=seed, options=options)
            finally:
                _close(engine)
            assert [args[4:] for args in opened] == [[], ["-echo", "none"]]

    def test_neighbor_list_reuse_leaves_the_av_search_unchanged(  # noqa: D102
        self,
        engine,  # noqa: ANN001
        vacancy,  # noqa: ANN001, F811
    ) -> None:
        results = [
            _search(
                engine,
                vacancy,
                seed=AV_HOP_SEED,
                options={**AV_OPTIONS, "reuse_neighbor_list": reuse},
            )
            for reuse in (False, True)
        ]

        assert results[0].success and results[1].success
        assert results[1].force_calls_total == results[0].force_calls_total
        for name in ("e_start", "e_saddle", "e_min1", "e_min2"):
            assert getattr(results[1], name) == pytest.approx(
                getattr(results[0], name), abs=1e-8
            )
        np.testing.assert_allclose(
            results[1].saddle_positions, results[0].saddle_positions, atol=1e-6
        )

    def test_the_full_path_after_a_av_search_is_unchanged(  # noqa: D102
        self,
        engine,  # noqa: ANN001
        eam_config,  # noqa: ANN001, F811
        vacancy,  # noqa: ANN001, F811
    ) -> None:
        _search(engine, vacancy, options=AV_OPTIONS)
        after_av = _search(engine, vacancy)
        fresh = _make_engine(
            eam_config,
            None,
            vacancy.positions,
            vacancy.types,
            vacancy.cell,
            vacancy.pbc,
            ArtnSearchExtension,
        )
        try:
            expected = _search(fresh, vacancy)
        finally:
            _close(fresh)
        for name, value in expected._asdict().items():
            if isinstance(value, np.ndarray):
                assert np.array_equal(value, getattr(after_av, name)), name
            else:
                assert value == getattr(after_av, name), name

    def test_a_skewed_cell_is_refused(self, engine, vacancy) -> None:  # noqa: ANN001, D102, F811
        cell = np.array(vacancy.cell)
        cell[1, 0] = 0.5
        with pytest.raises(ValueError, match="orthogonal cell"):
            _search(engine, vacancy, cell=cell, options=AV_OPTIONS)
