"""Saved source AV constraints survive relaxed endpoints and cache reloads.

Real service, schema2 persistence, analytic FD, Vineyard and Active/BKL run.
Scheduling and native force evaluation use the analytic catalogue helper.
"""

import hashlib

import numpy as np
import pytest
from pykmc.config import ActiveVolume

from . import prefactor_catalogue_helpers as helper

IDS = (31, 77)
PBC = (False, False, False)
SOURCE_CENTER = (-1.0007, 0.0002, -0.0004)


def av_service(potential, user, *, saved=None, mass=helper.MASS_A):
    cfg = helper.config(potential, saved)
    cfg = cfg.model_copy(
        update={
            "control": cfg.control.model_copy(update={"active_volume": True}),
            "activevolume": ActiveVolume(rmov=2.0, ract=15.0),
        }
    )
    worker = helper.PolynomialWorker(potential)
    service = helper.PrefactorService(
        cfg,
        worker,
        helper.create_rate_constant(cfg.rateconstant),
        species_masses=(("Si",), (mass,)),
        global_constraints=user,
        method="fd",
    )
    return service, worker


@pytest.mark.parametrize("mass_scale", [1.0, 4.0], ids=["same-input", "changed-mass"])
def test_saved_presearch_center_survives_minimum_relaxation(
    tmp_path, monkeypatch, mass_scale
):
    helper.require_schema2()
    potential = tmp_path / "actual-polynomial.json"
    helper.write_potential(potential, a=0.0, scale=1.0, bias=0.3)
    presearch = helper.FIRST.copy()
    presearch[0] = SOURCE_CENTER
    source_before = presearch.copy()
    user = helper.ResolvedConstraints.resolve(
        presearch,
        ("Si", "Si"),
        atom_ids=IDS,
        cell=helper.CELL,
        pbc=PBC,
    )
    execution = helper.ResolvedConstraints.resolve(
        presearch,
        ("Si", "Si"),
        atom_ids=IDS,
        cell=helper.CELL,
        pbc=PBC,
        center_id=31,
        rmov=2.0,
    )
    assert user.fixed_ids == () and execution.fixed_ids == (77,)
    assert execution.center_position == SOURCE_CENTER
    assert execution.center_position != tuple(helper.FIRST[0])
    service, original_worker = av_service(potential, user)
    request = service.build_request(
        event_key=("stationary-triplet-after-source-relaxation",),
        min1_positions=helper.FIRST,
        saddle_positions=helper.SADDLE,
        min2_positions=helper.FINAL,
        types=("Si", "Si"),
        cell=helper.CELL,
        pbc=PBC,
        center_index=0,
        constraints=execution,
    )
    # The actual triplet is stationary; only its source restriction predates
    # relaxation. The frozen worker independently checks all analytic forces.
    result = service.compute([request], compute_energies=True)[request.event_key]
    assert result.forward.ok and result.backward.ok
    assert (
        result.provenance.free_indices == (0,) and result.provenance.source.is_complete
    )
    assert result.provenance.source.constraints == execution
    table = helper.table_for(
        service, [(17, result, "forward", 47), (47, result, "backward", 17)]
    )
    assert len(original_worker.calls) == 1
    path = tmp_path / "actual-source-center.pkl"
    table.save(str(path))
    old_facts = dict(table.prefactor_archive.calculations)
    old_geometry = helper.geometric_rows(table.table)
    input_hash = hashlib.sha256(path.read_bytes()).hexdigest()

    current, worker = av_service(
        potential, user, saved=path, mass=helper.MASS_A * mass_scale
    )

    # Behavioral guard: a changed physical model may recompute, but it must use
    # exactly the original source AV mask/references rather than min1's center.
    def before_submit(rebuilt):
        assert rebuilt.constraints == execution, (
            "saved presearch AV restriction was rebased to relaxed min1"
        )
        assert rebuilt.user_constraints == user
        assert rebuilt.constraints.atom_ids == IDS
        np.testing.assert_array_equal(rebuilt.min1_positions, helper.FIRST)
        np.testing.assert_array_equal(rebuilt.saddle_positions, helper.SADDLE)
        np.testing.assert_array_equal(rebuilt.min2_positions, helper.FINAL)

    worker.before_submit = before_submit
    loaded = helper.ReferenceEventTable(current.config, prefactor_service=current)
    expected_calls = int(mass_scale != 1.0)
    assert len(worker.calls) == expected_calls, (
        "same-input load must reuse the actual producing restriction"
    )
    if expected_calls:
        _, backward, energies, computed = worker.calls[0]
        assert backward and energies
        assert computed.provenance.energies == pytest.approx(
            (0.04, 1.0, -0.04), rel=1e-14
        )
    for idx, direction in ((17, "forward"), (47, "backward")):
        linked = helper.linked_calculation(loaded, idx)
        assert linked.provenance.source.constraints == execution
        assert linked.identity.free_ids == (31,) and linked.identity.atom_ids == IDS
        assert linked.provenance.source.user_constraints == user
        assert linked.estimate.nu0_hz == pytest.approx(
            getattr(result, direction).nu0_hz / mass_scale**0.5, rel=1e-10
        )
        if expected_calls == 0:
            assert linked.calculation_id == result.calculation(direction).calculation_id
        active = helper.active_and_draw(loaded, idx, monkeypatch)
        assert active.nu0_status == "ok" and active.nu0_source == "reference"
    assert len(worker.calls) == expected_calls
    assert all(
        loaded.prefactor_archive.calculations[k] == v for k, v in old_facts.items()
    )
    helper.assert_geometry_unchanged(loaded.table, old_geometry)
    np.testing.assert_array_equal(presearch, source_before)
    assert hashlib.sha256(path.read_bytes()).hexdigest() == input_hash
