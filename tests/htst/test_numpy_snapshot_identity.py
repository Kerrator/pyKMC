"""Valid immutable NumPy scalars must have usable persistent content identity.

Direct ResolvedConstraints construction is deliberate. Its validation and the
snapshot immutable validator both accept these scalar types; no ndarray is used
inside the stored tuples. No worker or native resource is created.
"""

import importlib
import importlib.util
import pickle
from dataclasses import replace
from types import SimpleNamespace

import numpy as np
import pytest
from pykmc.htst.request import HTSTEventRequest
from pykmc.htst.result import DirectionalPrefactor, EventPrefactors
from pykmc.htst.settings import HTSTSettings
from pykmc.physics import EnginePhysics, PhysicalDescriptor, ResolvedConstraints


def request():
    native = SimpleNamespace(
        pair_style="zero 5.0",
        pair_coeff="* *",
        min_style="cg",
        frz_min="0 1e-12 10 100",
    )
    settings = HTSTSettings(free_radius=1.0)
    descriptor = PhysicalDescriptor.from_config(
        SimpleNamespace(frozen_atoms=None),
        EnginePhysics.capture(native, ("Si",), (28.0,)),
        settings,
    )
    first = np.array([[0.0, 0.0, 0.0], [2.0, 2.0, 0.0]])
    saddle, final = first.copy(), first.copy()
    saddle[0, 0], final[0, 0] = 0.1, 0.2
    cell = 5.0 * np.eye(3)
    pbc = (False, True, False)
    constraints = ResolvedConstraints.resolve(
        first,
        ("Si", "Si"),
        atom_ids=(8, 19),
        cell=cell,
        pbc=pbc,
        center_id=8,
        rmov=1.0,
    )
    return HTSTEventRequest(
        event_key=("numpy-scalar-identity",),
        min1_positions=first,
        saddle_positions=saddle,
        min2_positions=final,
        types=("Si", "Si"),
        species=("Si",),
        masses=(28.0,),
        cell=cell,
        pbc=pbc,
        center_index=0,
        settings=settings,
        descriptor=descriptor,
        constraints=constraints,
    )


@pytest.mark.parametrize("field", ["atom-identities", "cell"])
def test_immutable_numpy_context_has_stable_ids_across_pickle_roundtrip(field):
    name = "pykmc.htst.provenance"
    assert importlib.util.find_spec(name) is not None, "Producing API is required"
    api = importlib.import_module(name)
    source = request()
    if field == "atom-identities":
        nested = replace(
            source.constraints,
            source_ids=tuple(np.int64(i) for i in source.constraints.source_ids),
            atom_ids=tuple(np.int64(i) for i in source.constraints.atom_ids),
            fixed_ids=tuple(np.int64(i) for i in source.constraints.fixed_ids),
        )
        assert all(isinstance(i, np.integer) for i in nested.atom_ids)
    else:
        nested = replace(
            source.constraints,
            cell=tuple(
                tuple(np.int64(v) for v in row) for row in source.constraints.cell
            ),
        )
        assert all(isinstance(v, np.integer) for row in nested.cell for v in row)
    source = replace(source, constraints=nested)
    source.validate()
    snapshot = api.RequestSnapshot.capture(source)
    snapshot.validate()
    assert snapshot.is_complete
    snapshot_id = snapshot.snapshot_id
    assert isinstance(snapshot_id, str) and snapshot_id
    restored = pickle.loads(pickle.dumps(snapshot, protocol=pickle.HIGHEST_PROTOCOL))
    restored.validate()
    assert restored.snapshot_id == snapshot_id
    assert restored.to_request(event_key=()).constraints.atom_ids == (8, 19)
    np.testing.assert_array_equal(restored.to_request(event_key=()).cell, source.cell)

    provenance = api.CalculationProvenance.capture(
        source,
        source,
        method="typed-protocol",
        free_indices=(0,),
        energies=(0.0, 1.0, 0.25),
    )
    estimate = DirectionalPrefactor.accepted(
        1e12, n_free=1, n_positive_min=3, n_negative_saddle=1
    )
    result = EventPrefactors(
        source.event_key,
        estimate,
        estimate,
        "typed-protocol",
        1,
        source.settings,
        provenance=provenance,
    )
    calculation = result.calculation("forward")
    assert calculation is not None and calculation.reusable
    assert calculation.identity.free_ids == (8,)
    assert calculation.identity.atom_ids == (8, 19)
    assert calculation.identity.center_id == 8
    identities = (
        provenance.provenance_id,
        calculation.identity.identity_id,
        calculation.calculation_id,
    )
    restored = pickle.loads(pickle.dumps(calculation, protocol=pickle.HIGHEST_PROTOCOL))
    restored.validate()
    assert (
        restored.provenance.provenance_id,
        restored.identity.identity_id,
        restored.calculation_id,
    ) == identities
    assert restored.provenance.source.snapshot_id == snapshot_id
