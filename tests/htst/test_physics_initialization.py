"""Constraint validation precedes native initialization; snapshots stay immutable."""

import hashlib
from types import SimpleNamespace

import numpy as np
import pytest

from pykmc.config import RegionConfig
from pykmc.initializer import Initializer
from pykmc.physics import EnginePhysics, ResolvedConstraints
from tests.htst.test_physics_contract import api, configured, request, service


def test_invalid_global_constraint_precedes_native_initialization():
    calls = []
    kmc = SimpleNamespace(
        system=SimpleNamespace(positions=np.zeros((2, 3)), types=("Ni", "Ni")),
        config=SimpleNamespace(frozen_atoms=RegionConfig(indices=[2])),
        manager=SimpleNamespace(broadcast=lambda *a, **k: calls.append((a, k))),
    )
    with pytest.raises(ValueError, match="range"):
        Initializer(kmc).initialize_engine()
    assert calls == []


def test_potential_digest_and_pinned_constraint_policy(tmp_path):
    cfg, potential = configured(tmp_path, frozen=RegionConfig(indices=[2]))
    worker = service(api(), cfg, object())
    descriptor = worker.descriptor_for(("Ni",))
    expected = hashlib.sha256(potential.read_bytes()).hexdigest()
    assert descriptor.engine.force_model.file_digests == ((2, expected),)
    cfg.frozen_atoms.indices[:] = [0]
    built = request(worker)
    assert built.constraints.fixed_ids == (2,)
    assert descriptor.descriptor_id == worker.current_descriptor.descriptor_id


def test_preflight_constraints_remain_in_source_coordinates(tmp_path):
    cfg, _ = configured(tmp_path, frozen=RegionConfig(indices=[2]))
    from pykmc.rate_constant import create_rate_constant
    from pykmc.rate_constant.prefactors import PrefactorService

    source = np.array([[1.0, 1.0, 1.0], [2.0, 1.0, 1.0], [8.0, 8.0, 8.0]])
    constraints = ResolvedConstraints.resolve(source, ("Ni",) * 3, cfg.frozen_atoms)
    engine = EnginePhysics.capture(cfg.lammps, ("Ni", "Cu", "H"), (61.0, 65.0, 2.0))
    svc = PrefactorService(
        cfg,
        object(),
        create_rate_constant(cfg.rateconstant),
        engine_physics=engine,
        global_constraints=constraints,
    )
    assert request(svc).constraints is constraints
    source[2] = 0.0
    assert request(svc).constraints.fixed_positions == ((8.0, 8.0, 8.0),)


def test_descriptor_map_disagreement_rejects(tmp_path):
    from pykmc.rate_constant import create_rate_constant
    from pykmc.rate_constant.prefactors import PrefactorService

    cfg, _ = configured(tmp_path)
    engine = EnginePhysics.capture(cfg.lammps, ("Ni",), (61.0,))
    with pytest.raises(ValueError, match="disagree"):
        PrefactorService(
            cfg,
            object(),
            create_rate_constant(cfg.rateconstant),
            engine_physics=engine,
            species_masses=(("Ni",), (62.0,)),
        )
