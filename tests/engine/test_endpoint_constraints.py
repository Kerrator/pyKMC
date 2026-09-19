"""Source-resolved endpoint restrictions survive mapping, pushing and failures."""

import ctypes
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

import pykmc.basins.basin as basin_module
import pykmc.reconstruction as reconstruction_module
from pykmc.config import RegionConfig
from pykmc.engine.lammps import FullSystem, LammpsEngine
from pykmc.result import Ok

from pykmc.physics import ResolvedConstraints
from pykmc.reconstruction import Reconstruction


def resolve(positions, types, config, atom_ids, cell, pbc, center_id):
    return ResolvedConstraints.resolve(
        positions,
        types,
        config.frozen_atoms,
        atom_ids=atom_ids,
        cell=cell,
        pbc=pbc,
        center_id=center_id if config.control.active_volume else None,
        rmov=config.activevolume.rmov if config.control.active_volume else None,
    )


def reconstruction(config, manager, types, constraints, pbc):
    return Reconstruction(
        config, manager, types=types, constraints=constraints, pbc=pbc
    )


def center_position(constraints):
    return constraints.center_position


def minimize(engine, positions, config, types, constraints):
    return engine.minimize_with_results(
        positions=positions, config=config, types=types, constraints=constraints
    )


IDS = (40, 8, 19, 2, 77)
TYPES = ("Ni", "Fe", "Ni", "Ni", "Ni")
FIXED = (8, 19, 2, 77)
CELL = np.diag([8.0, 8.0, 8.0])
PBC = (True, True, True)
NEIGHBORS = np.array([3, 0, 4, 1, 2])  # local event row != source/global ID


def event():
    b = np.sqrt(2.0 ** (1.0 / 3.0) - 1.0)
    saddle = np.array(
        [
            [4.0, 7.2, 4.0],
            [4.0, 0.2, 4.0],
            [4.0, 6.2, 4.0],
            [4.0, 7.2, 5.0],
            [4.0, 7.2, 3.0],
        ]
    )
    first, second = saddle.copy(), saddle.copy()
    first[0, 0] -= b
    second[0, 0] += b
    return first, saddle, second


def config(style="global/reconstruction", active=True, radius=1.3):
    return SimpleNamespace(
        frozen_atoms=RegionConfig(types=["Fe"]),
        control=SimpleNamespace(active_volume=active),
        activevolume=SimpleNamespace(rmov=radius, ract=2.25),
        basin=SimpleNamespace(style=style),
        reconstruction=SimpleNamespace(push_fraction=0.1),
        psr=SimpleNamespace(matching_score_thr=1e-7),
    )


def payload(cfg=None):
    first, _, _ = event()
    return resolve(first, TYPES, cfg or config(), IDS, CELL, PBC, 8)


def fixed_map(constraints):
    return dict(zip(constraints.fixed_ids, constraints.fixed_positions, strict=True))


def assert_payload(got, expected):
    assert isinstance(got, ResolvedConstraints)
    assert got.source_ids == expected.source_ids
    assert got.atom_ids == expected.atom_ids
    assert got.fixed_ids == expected.fixed_ids
    assert got.fixed_positions == expected.fixed_positions
    assert got.constraint_id == expected.constraint_id
    assert tuple(got.pbc) == PBC
    np.testing.assert_array_equal(got.cell, CELL)
    assert got.center_id == 8
    np.testing.assert_array_equal(center_position(got), event()[0][1])


def test_global_union_uses_source_ids_and_retains_reference_context_through_crop():
    first, _, _ = event()
    source = first.copy()
    source_cell, source_pbc = CELL.copy(), np.array(PBC)
    got = resolve(source, TYPES, config(), IDS, source_cell, source_pbc, 8)
    assert got.source_ids == IDS
    assert got.atom_ids == IDS
    assert set(got.fixed_ids) == set(FIXED)
    expected_fixed = {IDS[i]: tuple(first[i]) for i in range(1, 5)}
    assert fixed_map(got) == expected_fixed
    assert got.center_id == 8
    assert got.rmov == 1.3
    np.testing.assert_array_equal(center_position(got), first[1])
    cropped = got.crop((3, 0, 1))
    assert cropped.source_ids == IDS
    assert cropped.atom_ids == (2, 40, 8)
    assert cropped.fixed_ids == got.fixed_ids
    assert cropped.local_fixed_indices == (0, 2)
    assert cropped.fixed_positions == got.fixed_positions
    source[:] = 123.0
    source_cell[:] = 123.0
    source_pbc[:] = False
    assert fixed_map(got) == expected_fixed
    np.testing.assert_array_equal(center_position(cropped), first[1])
    np.testing.assert_array_equal(got.cell, CELL)
    assert tuple(got.pbc) == PBC


@pytest.mark.parametrize("active,radius", [(False, 1.3), (True, 20.0)])
def test_av_off_or_no_outer_keeps_the_nonempty_global_mask(active, radius):
    first, _, _ = event()
    cfg = config(active=active, radius=radius)
    got = resolve(first, TYPES, cfg, IDS, CELL, PBC, 8)
    assert got.fixed_ids == (8,)
    assert got.fixed_positions == (tuple(first[1]),)


def test_thin_nonperiodic_axis_changes_membership_without_changing_source_ids():
    first, _, _ = event()
    # Same physical LJ cage, wholly inside a fixed y box. Full-PBC MIC would
    # turn center-to-opposite-anchor distance 2 into .4 and wrongly unfreeze it.
    first[:, 1] = [1.2, 2.2, 0.2, 1.2, 1.2]
    cell = np.diag([8.0, 2.4, 8.0])
    fixed_y = resolve(first, TYPES, config(), IDS, cell, (True, False, True), 8)
    periodic_y = resolve(first, TYPES, config(), IDS, cell, PBC, 8)
    assert set(fixed_y.fixed_ids) == set(FIXED)
    assert set(periodic_y.fixed_ids) == {8, 2, 77}
    assert fixed_y.source_ids == periodic_y.source_ids == IDS
    # This resolver assertion does not certify mixed-PBC reconstruction.


def test_fixed_reference_validation_and_protection_are_distinct_and_nonmutating():
    first, _, _ = event()
    got = payload()
    valid_image = first.copy()
    valid_image[1] += CELL[0]  # representation-equivalent fixed position
    got.validate_positions(valid_image)
    invalid = first.copy()
    invalid[1, 0] += 0.2
    invalid[0, 0] += 0.15  # movable proposal must survive protection
    saved = invalid.copy()
    with pytest.raises(ValueError):
        got.validate_positions(invalid)
    protected = got.protect_positions(invalid)
    np.testing.assert_array_equal(invalid, saved)
    np.testing.assert_array_equal(protected[1:], first[1:])
    np.testing.assert_array_equal(protected[0], saved[0])
    assert not np.shares_memory(protected, invalid)


class RecordingManager:
    def __init__(self, expected, outputs):
        self.expected = expected
        self.outputs = [p.copy() for p in outputs]
        self.calls = []

    def group_minimize_with_results(self, **kwargs):
        assert "constraints" in kwargs, "missing resolved endpoint restriction"
        assert_payload(kwargs["constraints"], self.expected)
        assert tuple(kwargs.get("types", ())) == TYPES
        positions = np.array(kwargs["positions"], copy=True)
        np.testing.assert_allclose(positions[1:], event()[0][1:], rtol=0, atol=1e-12)
        self.calls.append((positions, kwargs["constraints"]))
        return self.outputs[len(self.calls) - 1].copy(), -0.29365234375


def invoke(recon, first, saddle, second):
    return recon.reconstruct(
        first[NEIGHBORS], second[NEIGHBORS], saddle, CELL, 1e-7, NEIGHBORS
    )


def test_both_endpoint_pushes_share_the_resolved_union_and_preserve_input_triplet():
    first, saddle, second = event()
    saved = [p.copy() for p in (first, saddle, second)]
    expected = payload()
    manager = RecordingManager(expected, [first, second])
    recon = reconstruction(config(), manager, TYPES, expected, PBC)
    result = invoke(recon, first, saddle, second)
    assert result.is_ok()
    assert len(manager.calls) == 2
    assert manager.calls[0][0][0, 0] < saddle[0, 0]
    assert manager.calls[1][0][0, 0] > saddle[0, 0]
    assert manager.calls[0][1].constraint_id == manager.calls[1][1].constraint_id
    for actual, before in zip((first, saddle, second), saved):
        np.testing.assert_array_equal(actual, before)


@pytest.mark.parametrize("vertex", [0, 1, 2])
def test_incompatible_claimed_fixed_triplet_rejects_before_any_dispatch(vertex):
    triplet = list(event())
    expected = payload()
    triplet[vertex][1, 0] += 0.2
    manager = RecordingManager(expected, [triplet[0], triplet[2]])
    recon = reconstruction(config(), manager, TYPES, expected, PBC)
    try:
        result = invoke(recon, *triplet)
    except ValueError:
        pass
    else:
        assert not result.is_ok(), (
            "incompatible stationary event was silently projected"
        )
    assert manager.calls == []


def test_injected_fixed_push_proposal_cannot_reach_endpoint_minimization(monkeypatch):
    first, saddle, second = event()
    expected = payload()
    manager = RecordingManager(expected, [first, second])
    original = reconstruction_module.push_towards

    def proposed_push(current, target, *args, **kwargs):
        result = original(current, target, *args, **kwargs)
        # If the operation passes the whole neighbor patch, propose a wrong
        # displacement of source row1 (neighbor row3). Movable-only callers
        # bypass this proposal; masking/rejection callers must stop it.
        if len(result) == len(NEIGHBORS):
            result[3, 0] += 0.25
        return result

    monkeypatch.setattr(reconstruction_module, "push_towards", proposed_push)
    recon = reconstruction(config(), manager, TYPES, expected, PBC)
    try:
        result = invoke(recon, first, saddle, second)
    except ValueError:
        assert manager.calls == []
    else:
        assert result.is_ok() or manager.calls == []
        if result.is_ok():
            assert len(manager.calls) == 2  # manager independently checked fixed rows


class ProtocolSystem:
    """Explicitly excludes R05 wrapping; positions here are already canonical."""

    def __init__(self, positions, types, cell, pbc, index):
        self.positions = np.array(positions, copy=True)
        self.types, self.cell, self.pbc, self.index = types, cell, pbc, index

    def update_positions(self, new_positions, atom_idx=None):
        if atom_idx is None:
            self.positions = np.array(new_positions, copy=True)
        else:
            self.positions[atom_idx] = new_positions


@pytest.mark.parametrize("style", ["global", "global/reconstruction"])
@pytest.mark.parametrize("bad_fixed_overlay", [False, True])
def test_basin_callers_resolve_before_overlay_and_transport_types_and_both_endpoints(
    monkeypatch, style, bad_fixed_overlay
):
    first, saddle, second = event()
    if bad_fixed_overlay:
        second[1, 0] += 0.2
    cfg, expected = config(style), payload()
    manager = RecordingManager(
        expected, [second] if style == "global" else [first, second]
    )
    source = ProtocolSystem(first, TYPES, CELL, np.array(PBC), np.array(IDS))
    neighbors = SimpleNamespace(get_neighbors=lambda *_: NEIGHBORS)
    state = SimpleNamespace(
        system=source, neighbors_list=neighbors, ensure_full_state=lambda *_: None
    )
    ref = pd.DataFrame(
        [
            dict(
                idx_ref=31,
                initial_positions=first[NEIGHBORS],
                saddle_positions=saddle[NEIGHBORS],
                final_positions=second[NEIGHBORS],
            )
        ]
    )
    psr = SimpleNamespace(
        rotation_matrix=np.eye(3),
        translation_matrix=np.zeros(3),
        permutation_matrix=np.arange(5),
    )
    monkeypatch.setattr(basin_module, "System", ProtocolSystem)
    monkeypatch.setattr(
        basin_module,
        "PointSetRegistration",
        lambda *_args, **_kw: SimpleNamespace(match=lambda: Ok(psr)),
    )
    monkeypatch.setattr(basin_module, "check_match", lambda result, *_: result)
    basin = basin_module.BasinsGenericEvents.__new__(basin_module.BasinsGenericEvents)
    basin.config, basin.manager = cfg, manager
    basin.reference_table = SimpleNamespace(table=ref)
    basin.states = {0: state}
    try:
        result = basin.system_from_state(0, 31, 1, 0)
    except ValueError:
        assert bad_fixed_overlay
        assert manager.calls == []
        np.testing.assert_array_equal(source.positions, first)
        return
    if bad_fixed_overlay:
        assert not result.is_ok(), (
            "basin accepted a physically different fixed-atom event"
        )
        assert manager.calls == []
        np.testing.assert_array_equal(source.positions, first)
        return
    assert result.is_ok()
    assert len(manager.calls) == (1 if style == "global" else 2)
    np.testing.assert_array_equal(source.positions, first)
    np.testing.assert_allclose(result.ok_value().positions, second, rtol=0, atol=1e-12)


class CommandEndpoint:
    """Native command recorder; no lammps constructor, library or communicator."""

    def __init__(self, positions, failure=None):
        self.positions = positions.copy()
        self.groups = {"user_keep": {3}, "g_frozen": {3}}
        self.fixes = {"user_keep_fix": "user_keep", "f_frozen_min": "g_frozen"}
        self.computes = set()
        self.initial_groups = {k: set(v) for k, v in self.groups.items()}
        self.initial_fixes = dict(self.fixes)
        self.failure, self.minimizations, self.scatters = failure, [], 0

    def get_natoms(self):
        return len(self.positions)

    def has_id(self, kind, name):
        return (
            name
            in {"group": self.groups, "fix": self.fixes, "compute": self.computes}[kind]
        )

    def available_ids(self, kind):
        return list(
            {"group": self.groups, "fix": self.fixes, "compute": self.computes}[kind]
        )

    def command(self, command):
        fields = command.split()
        if fields[0] == "group":
            name, operation = fields[1:3]
            if operation == "delete":
                assert name not in self.initial_groups, "deleted pre-existing group"
                assert name not in self.fixes.values(), (
                    "deleted a group with an active fix"
                )
                self.groups.pop(name)
            elif operation == "id":
                assert name not in self.initial_groups, "changed pre-existing group"
                self.groups.setdefault(name, set()).update(map(int, fields[3:]))
            elif operation == "empty":
                assert name not in self.initial_groups, "changed pre-existing group"
                self.groups[name] = set()
            else:
                raise AssertionError(
                    f"ADAPTER_PROTOCOL: unsupported group command {command}"
                )
        elif fields[0] == "fix":
            name, group = fields[1:3]
            assert name not in self.initial_fixes, "replaced pre-existing fix"
            if self.failure == "fix":
                raise RuntimeError("primary injected setup failure")
            assert fields[3] == "setforce"
            assert list(map(float, fields[4:])) == [0.0, 0.0, 0.0]
            self.fixes[name] = group
        elif fields[0] == "unfix":
            assert fields[1] not in self.initial_fixes, "removed pre-existing fix"
            self.fixes.pop(fields[1])
        elif fields[0] == "minimize":
            frozen = set().union(*(self.groups[g] for g in self.fixes.values()))
            assert frozen == {1, 3}, "global IDs were not mapped to current native rows"
            self.minimizations.append(frozen)
            self.positions[1, 0] += 0.15  # only the one movable native row
            if self.failure == "minimize":
                raise RuntimeError("primary injected minimization failure")
        elif fields[0] == "compute":
            self.computes.add(fields[1])
        elif fields[0] == "uncompute":
            self.computes.remove(fields[1])
        elif fields[0] not in ("min_style", "run"):
            raise AssertionError(f"ADAPTER_PROTOCOL: unsupported command {command}")

    def scatter_atoms(self, name, kind, count, array):
        assert (name, kind, count) == ("x", 1, 3)
        self.positions = np.array(np.ctypeslib.as_array(array), copy=True).reshape(
            -1, 3
        )
        self.scatters += 1

    def gather_atoms(self, name, kind, count):
        assert (name, kind, count) == ("x", 1, 3)
        return (ctypes.c_double * self.positions.size)(*self.positions.ravel())

    def get_thermo(self, _name):
        return 0.0

    def extract_compute(self, name, style, result_type):
        assert name in self.computes
        assert (style, result_type) == (0, 0)
        return 0.0  # energy is not a numerical oracle in this command protocol


@pytest.mark.parametrize("failure", [None, "fix", "minimize"])
def test_engine_maps_crop_ids_and_cleans_only_owned_resources(failure):
    first, _, _ = event()
    rows = np.array([3, 0, 1])
    positions = first[rows].copy()
    constraints = payload().crop(tuple(rows))
    native = SimpleNamespace(min_style="cg", minimize="1e-8 1e-8 100 1000")
    engine = LammpsEngine(native)
    endpoint = CommandEndpoint(positions, failure)
    engine.lmp = endpoint
    engine._is_orthorhombic = True
    engine.full_system = FullSystem(
        tuple(np.array(TYPES)[rows]), ("Fe", "Ni"), (56.0, 60.0), CELL.copy(), PBC
    )
    cfg = config()
    proposed = positions.copy()
    proposed[1, 0] += 0.1  # native entry differs from the proposed pushed state
    if failure is None:
        result, _ = minimize(
            engine, proposed.copy(), cfg, tuple(np.array(TYPES)[rows]), constraints
        )
        assert endpoint.minimizations == [{1, 3}]
        expected = positions.copy()
        expected[1, 0] += 0.25
        np.testing.assert_allclose(result, expected, rtol=0, atol=1e-12)
        assert not np.shares_memory(result, endpoint.positions)
    else:
        with pytest.raises(RuntimeError, match="primary injected"):
            minimize(
                engine, proposed.copy(), cfg, tuple(np.array(TYPES)[rows]), constraints
            )
    assert endpoint.groups == endpoint.initial_groups
    assert endpoint.fixes == endpoint.initial_fixes
    np.testing.assert_allclose(endpoint.positions, positions, rtol=0, atol=1e-12)
    assert constraints.fixed_positions == payload().fixed_positions


def test_engine_rejects_fixed_coordinate_violation_before_scatter_or_resource_mutation():
    first, _, _ = event()
    rows = np.array([3, 0, 1])
    positions = first[rows].copy()
    engine = LammpsEngine(
        SimpleNamespace(min_style="cg", minimize="1e-8 1e-8 100 1000")
    )
    endpoint = CommandEndpoint(positions)
    engine.lmp = endpoint
    engine._is_orthorhombic = True
    engine.full_system = FullSystem(
        tuple(np.array(TYPES)[rows]), ("Fe", "Ni"), (56.0, 60.0), CELL.copy(), PBC
    )
    bad = positions.copy()
    bad[0, 0] += 0.2  # source/global ID2, currently native atom1
    with pytest.raises(ValueError):
        minimize(
            engine,
            bad,
            config(),
            tuple(np.array(TYPES)[rows]),
            payload().crop(tuple(rows)),
        )
    assert endpoint.scatters == 0
    assert endpoint.minimizations == []
    assert endpoint.groups == endpoint.initial_groups
    assert endpoint.fixes == endpoint.initial_fixes
    np.testing.assert_array_equal(endpoint.positions, positions)
