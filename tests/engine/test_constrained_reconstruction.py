"""A stationary coupled LJ event reconstructs under its original restriction."""

from types import SimpleNamespace

from mpi4py import MPI
import numpy as np
import pytest
from pykmc.config import RegionConfig
from pykmc.engine.lammps import LammpsEngine
from pykmc.physics import ResolvedConstraints
from pykmc.reconstruction import Reconstruction


def test_native_constrained_two_endpoint_transaction():
    b = np.sqrt(2 ** (1 / 3) - 1)
    first = (
        np.array([[-b, 0, 0], [0, 1, 0], [0, -1, 0], [0, 0, 1], [0, 0, -1]])
        + [4, -0.8, 4]
    ) % 8
    saddle, second = first.copy(), first.copy()
    saddle[0, 0], second[0, 0] = 4, 4 + b
    cell, pbc, types = np.eye(3) * 8, (True, True, True), ("Ni",) * 5
    cfg = SimpleNamespace(
        lammps=SimpleNamespace(
            pair_style="lj/cut 2.5",
            pair_coeff="* * .05 1",
            min_style="cg",
            minimize="0.0 1e-12 1000 10000",
            frz_min="0.0 1e-12 1000 10000",
            verbosity=0,
        ),
        frozen_atoms=RegionConfig(indices=[1]),
        reconstruction=SimpleNamespace(push_fraction=0.1),
        psr=SimpleNamespace(matching_score_thr=1e-7),
    )
    constraints = ResolvedConstraints.resolve(
        first,
        types,
        cfg.frozen_atoms,
        (42, 8, 91, 17, 63),
        cell=cell,
        pbc=pbc,
        center_id=8,
        rmov=1.3,
    )
    engine = LammpsEngine(cfg.lammps, comm=MPI.COMM_SELF)
    calls = []
    try:
        engine.start()
        engine.initialize_parameters()
        engine.initialize_system(
            types, first.copy(), cell, pbc=pbc, species=("Ni",), masses=(58.6934,)
        )
        engine.initialize_potential()
        engine.command("group user_keep id 2")
        engine.command("fix user_keep user_keep setforce 0 0 0")
        resources = {k: tuple(engine.lmp.available_ids(k)) for k in ("group", "fix")}

        def endpoint(**kwargs):
            assert kwargs["constraints"] is constraints
            calls.append(kwargs["positions"].copy())
            result = engine.minimize_with_results(**kwargs)
            np.testing.assert_allclose(
                engine.get_positions(), first, rtol=0, atol=1e-12
            )
            for k, ids in resources.items():
                assert tuple(engine.lmp.available_ids(k)) == ids
            return result

        result = Reconstruction(
            cfg,
            SimpleNamespace(group_minimize_with_results=endpoint),
            types=types,
            constraints=constraints,
            pbc=pbc,
        ).reconstruct(first.copy(), second.copy(), saddle.copy(), cell, 1e-7)
        assert result.is_ok()
        assert len(calls) == 2
        out = result.ok_value()
        for actual, expected in (
            (out.min1_positions, first),
            (out.min2_positions, second),
        ):
            np.testing.assert_allclose(actual, expected, rtol=0, atol=1e-7)
            np.testing.assert_allclose(actual[1:], first[1:], rtol=0, atol=1e-12)
        assert out.min2_etot == pytest.approx(-0.29365234375, abs=1e-8, rel=0)
        assert out.min2_positions[0, 0] - out.min1_positions[0, 0] == pytest.approx(
            2 * b, abs=2e-7, rel=0
        )
        assert engine.get_total_energy(recompute=False) == pytest.approx(
            -0.29365234375, abs=1e-8, rel=0
        )
    finally:
        engine.close()
