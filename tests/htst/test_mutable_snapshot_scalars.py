"""Reject mutable scalar arrays inside supposedly immutable producing records.

No worker, Hessian, native engine or MPI pool is constructed. The cases exercise
public dataclass construction/replacement and RequestSnapshot.capture/validate.
"""

import importlib
import importlib.util
from dataclasses import replace
from types import SimpleNamespace

import numpy as np
import pytest
from pykmc.htst.request import HTSTEventRequest
from pykmc.htst.settings import HTSTSettings
from pykmc.physics import EnginePhysics, PhysicalDescriptor, ResolvedConstraints


def request():
    native = SimpleNamespace(
        pair_style="zero 5.0",
        pair_coeff="* *",
        min_style="cg",
        frz_min="0 1e-12 10 100",
    )
    cfg = SimpleNamespace(lammps=native, frozen_atoms=None)
    settings = HTSTSettings(free_radius=1.0)
    descriptor = PhysicalDescriptor.from_config(
        cfg, EnginePhysics.capture(native, ("Si",), (28.0,)), settings
    )
    first = np.array([[0.0, 0.0, 0.0], [2.0, 2.0, 0.0]])
    saddle, final = first.copy(), first.copy()
    saddle[0, 0], final[0, 0] = 0.1, 0.2
    return HTSTEventRequest(
        event_key=("mutable-scalar-review",),
        min1_positions=first,
        saddle_positions=saddle,
        min2_positions=final,
        types=("Si", "Si"),
        species=("Si",),
        masses=(28.0,),
        cell=5.0 * np.eye(3),
        pbc=(False, True, False),
        center_index=0,
        settings=settings,
        descriptor=descriptor,
    )


def snapshot_api():
    name = "pykmc.htst.provenance"
    assert importlib.util.find_spec(name) is not None, "Snapshot API is required"
    return importlib.import_module(name).RequestSnapshot


def replace_entry(matrix, scalar):
    rows = [list(row) for row in matrix]
    rows[0][0] = scalar
    return tuple(tuple(row) for row in rows)


@pytest.mark.parametrize("field", ["min1_positions", "cell"])
def test_snapshot_rejects_zero_dimensional_array_inside_coordinate_tuple(field):
    snapshot = snapshot_api().capture(request())
    original = getattr(snapshot, field)
    scalar = np.array(original[0][0])
    assert scalar.ndim == 0 and scalar.flags.writeable
    # Tuples alone do not make the element immutable. np.array(..., dtype=float)
    # accepts this input and hides the alias while validating request geometry.
    with pytest.raises(ValueError):
        replace(snapshot, **{field: replace_entry(original, scalar)})


@pytest.mark.parametrize("field", ["cell", "center_position"])
def test_snapshot_rejects_mutable_scalar_in_nested_constraint_context(field):
    api = snapshot_api()
    source = request()
    constraints = ResolvedConstraints.resolve(
        source.min1_positions,
        source.types,
        cell=source.cell,
        pbc=source.pbc,
        center_id=0,
        rmov=1.0,
    )
    # The genuine source has row1 fixed and unchanged in all three geometries.
    assert constraints.fixed_ids == (1,)
    good = api.capture(replace(source, constraints=constraints))
    assert good.is_complete
    with pytest.raises(ValueError):
        if field == "cell":
            nested = replace(
                constraints,
                cell=replace_entry(constraints.cell, np.array(constraints.cell[0][0])),
            )
        else:
            nested = replace(
                constraints,
                center_position=(np.array(0.0), 0.0, 0.0),
            )
        # Either the nested constructor or the producing boundary may reject;
        # accepting the alias as an immutable snapshot is the forbidden result.
        api.capture(replace(source, constraints=nested))
