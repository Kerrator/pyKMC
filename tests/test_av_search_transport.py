"""Independent N02 caller/first-mutation protocol, never a native physics gate.

Real System, callers, tables, service request validation, AV helpers and engine
scatter run. Manager execution, PSR mapping and the native command endpoint are
recording seams. The only new contract assumed is optional source constraints
on search/refine/results and optional global_constraints on caller constructors.
"""

from concurrent.futures import Future
from copy import deepcopy
import ctypes
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from pykmc.activevolume import active_volume as av
from pykmc.config import RateConstantConfig, RegionConfig
from pykmc.engine.lammps import FullSystem, LammpsEngine
from pykmc.event_table import ActiveEventTable, ReferenceEventTable
from pykmc.eventsearch import EventSearch
from pykmc.physics import ResolvedConstraints
from pykmc.rate_constant import create_rate_constant
from pykmc.rate_constant.prefactors import PrefactorService
from pykmc.refinement import Refinement
import pykmc.refinement as refinement_module
from pykmc.result import EventRefinementOutput, EventSearchOutput, Ok, PSROutput
from pykmc.system import System


IDS = (42, 17, 8, 91)
TYPES = ("Ni",) * 4
CELL = np.diag([10.0, 12.0, 14.0])
PBC = (True, False, True)
SEARCH_CENTER = 1  # source ID 17; mover is row 0/source ID 42
FIELDS = ("min1_positions", "saddle_positions", "min2_positions")


def config():
    return SimpleNamespace(
        lammps=SimpleNamespace(
            pair_style="zero 10.0",
            pair_coeff="* *",
            min_style="cg",
            frz_min="0.0 1e-12 100 1000",
            minimize="0.0 1e-12 100 1000",
            verbosity=0,
        ),
        frozen_atoms=RegionConfig(
            region_type="plane", normal="y", side="above", threshold=3.3
        ),
        control=SimpleNamespace(active_volume=True),
        activevolume=SimpleNamespace(rmov=1.6, ract=5.0, AV_debug=False),
        atomicenvironment=SimpleNamespace(rcut=2.0),
        eventsearch=SimpleNamespace(refined_energy_thr=1e-7),
        psr=SimpleNamespace(matching_score_thr=1e-7),
        rateconstant=RateConstantConfig(
            style="htst", k0=7, free_radius=4.0, nu0_min_THz=0.01
        ),
    )


def geometries(crossed=False):
    source = np.array(
        [[2.25, 3.0, 3.0], [3.5, 3.0, 3.0], [3.5, 3.5, 3.0], [8.0, 3.0, 3.0]]
    )
    if crossed:
        source[0, 1] = 3.4  # enters configured region after snapshot capture
    saddle, final = source.copy(), source.copy()
    saddle[0, 0] += 0.4
    final[0, 0] += 0.8
    return source, saddle, final


def user_snapshot(cfg):
    return ResolvedConstraints.resolve(
        geometries()[0], TYPES, cfg.frozen_atoms, IDS, cell=CELL, pbc=PBC
    )


def system(source):
    return System(
        types=np.array(TYPES),
        positions=source.copy(),
        cell=CELL.copy(),
        pbc=PBC,
        index=np.array(IDS),
    )


def logger():
    return SimpleNamespace(info=lambda *a, **k: None, progress_bar=lambda *a, **k: None)


def assert_union(payload, source, user):
    assert isinstance(payload, ResolvedConstraints)
    assert payload.source_ids == IDS and payload.atom_ids == IDS
    assert set(payload.fixed_ids) == {8, 91}, "known user ID plus source AV outer ID"
    refs = dict(zip(payload.fixed_ids, payload.fixed_positions, strict=True))
    assert refs == {8: tuple(source[2]), 91: tuple(source[3])}
    assert payload.user_policy == user.user_policy
    assert payload.center_id == IDS[SEARCH_CENTER]
    np.testing.assert_array_equal(payload.center_position, source[SEARCH_CENTER])
    assert payload.rmov == 1.6
    assert tuple(payload.pbc) == PBC
    np.testing.assert_array_equal(payload.cell, CELL)
    payload.require_preserves(user, cell=CELL, pbc=PBC)


class CaptureComplete(RuntimeError):
    """Stops at actual validated request dispatch; no Hessian or native launch."""


def capturing_service(cfg, user):
    service = PrefactorService(
        cfg,
        SimpleNamespace(),
        create_rate_constant(cfg.rateconstant),
        species_masses=(("Ni",), (58.6934,)),
        global_constraints=user,
    )
    service.captured = []
    service.direction = []

    def capture(requests, *, compute_backward=True):
        for request in requests:
            request.validate()
        service.captured.extend(requests)
        service.direction.append(compute_backward)
        raise CaptureComplete("validated protocol request captured")

    service.compute = capture
    return service


def assert_user_only(payload, source, user):
    # contracts 7f policy 5: outputs and HTST requests carry the USER
    # constraints only; the AV outer ID and the AV centre context never enter
    # the request (the Vineyard free region stays free_radius). Formerly
    # asserted the user+AV union (assert_union).
    assert isinstance(payload, ResolvedConstraints)
    assert payload.source_ids == IDS and payload.atom_ids == IDS
    assert payload.fixed_ids == (8,), "known user ID only, no source AV outer ID"
    assert payload.fixed_positions == (tuple(source[2]),)
    assert payload.user_policy == user.user_policy
    assert payload.center_id is None and payload.rmov is None
    assert tuple(payload.pbc) == PBC
    np.testing.assert_array_equal(payload.cell, CELL)
    payload.require_preserves(user, cell=CELL, pbc=PBC)


def assert_request(request, triplet, user):
    for name, expected in zip(FIELDS, triplet, strict=True):
        np.testing.assert_array_equal(getattr(request, name), expected)
    assert_user_only(request.constraints, triplet[0], user)
    assert request.user_constraints == user
    assert request.constraints.atom_ids == IDS


class SearchManager:
    def __init__(self, triplet, user):
        self.triplet, self.user = triplet, user
        self.calls, self.outputs = [], []

    def partn_search(self, **kwargs):
        assert_union(kwargs.get("constraints"), self.triplet[0], self.user)
        assert kwargs["central_atom_idx"] == SEARCH_CENTER
        np.testing.assert_array_equal(kwargs["positions"], self.triplet[0])
        self.calls.append(kwargs)
        output = EventSearchOutput(
            central_atom_index=SEARCH_CENTER,
            move_atom_index=0,
            dE_forward=0.2,
            dE_backward=0.2,
            **{
                key: value.copy()
                for key, value in zip(FIELDS, self.triplet, strict=True)
            },
            cell=CELL.copy(),
            types=np.array(TYPES),
            # The engine attaches the user view of the union it transported.
            constraints=kwargs["constraints"].user_view(),
        )
        self.outputs.append(output)
        result = Future()
        result.set_result(Ok(output))
        return result


@pytest.mark.parametrize("crossed", [False, True], ids=["source-frame", "known-region"])
def test_search_result_to_reference_request_keeps_original_frame_and_known_union(
    crossed,
):
    cfg = config()
    triplet = geometries(crossed)
    user = user_snapshot(cfg)
    assert user.fixed_ids == (8,)
    before = deepcopy(user)
    source = system(triplet[0])
    manager = SearchManager(triplet, user)
    caller = EventSearch(cfg, source, manager, logger(), global_constraints=user)
    caller.execute([SEARCH_CENTER])
    outputs = caller.get_successes_results()
    assert len(outputs) == 1
    # The legacy centering translation is deliberately not a lattice vector.
    translation = np.diag(CELL) / 2.0 - triplet[0][0]
    assert np.all(translation != 0.0) and translation[1] == 3.0 - (
        0.4 if crossed else 0
    )
    service = capturing_service(cfg, user)
    table = ReferenceEventTable.__new__(ReferenceEventTable)
    table.config, table.prefactor_service = cfg, service
    with pytest.raises(CaptureComplete):
        table._resolve_prefactors([(5, 6, None, outputs[0])], PBC)
    assert len(service.captured) == 1 and service.direction == [True]
    request = service.captured[0]
    assert_request(request, triplet, user)
    assert (
        request.center_index == 0
    )  # vibrational center may differ from source AV center
    np.testing.assert_array_equal(source.positions, triplet[0])
    assert user == before
    assert_user_only(outputs[0].constraints, triplet[0], user)


class Neighbors:
    rows = np.array([2, 0, 1])  # excludes AV outer row 3; intentionally permuted

    def get_neighbors(self, kind, center):
        assert kind == "rcut" and center == SEARCH_CENTER
        return self.rows.copy()


def reference_row(triplet):
    rows = Neighbors.rows
    return pd.Series(
        {
            "idx_ref": 5,
            "event_id": "fixture",
            "energy_barrier": 0.2,
            "initial_positions": triplet[0][rows].copy(),
            "saddle_positions": triplet[1][rows].copy(),
            "final_positions": triplet[2][rows].copy(),
            "types": np.array(TYPES)[rows],
            "sym_matrix": [np.eye(3)],
            "sym_perm": [np.arange(len(rows))],
            "nu0_status": "rejected",
            "nu0_reason": "protocol has no frequency",
            "nu0": np.nan,
        }
    )


def identity_psr(monkeypatch):
    class Matching:
        def __init__(self, *args, **kwargs):
            pass

        def match(self):
            return Ok(
                PSROutput(
                    rotation_matrix=np.eye(3),
                    translation_matrix=np.zeros(3),
                    permutation_matrix=np.arange(len(Neighbors.rows)),
                    matching_score=0.0,
                )
            )

    monkeypatch.setattr(refinement_module, "PointSetRegistration", Matching)


class RefineManager:
    def __init__(self, triplet, user, fail=False):
        self.triplet, self.user, self.fail = triplet, user, fail
        self.calls = []

    def partn_refine(self, **kwargs):
        assert_union(kwargs.get("constraints"), self.triplet[0], self.user)
        np.testing.assert_array_equal(kwargs["positions"], self.triplet[0])
        np.testing.assert_array_equal(kwargs["saddle_idx"], Neighbors.rows)
        np.testing.assert_array_equal(
            kwargs["saddle_positions"], self.triplet[1][Neighbors.rows]
        )
        self.calls.append(kwargs)
        if self.fail:
            raise RuntimeError("injected immediate manager submission failure")
        output = EventRefinementOutput(
            central_atom_index=SEARCH_CENTER,
            saddle_positions=self.triplet[1].copy(),
            E_saddle=0.2,
            refined="T",
            # The engine attaches the user view of the union it transported.
            constraints=kwargs["constraints"].user_view(),
        )
        result = Future()
        result.set_result(Ok(output))
        return result


def refinery(monkeypatch, *, fail=False):
    identity_psr(monkeypatch)
    cfg, triplet = config(), geometries()
    user = user_snapshot(cfg)
    source = system(triplet[0])
    manager = RefineManager(triplet, user, fail)
    environment = SimpleNamespace(get_atoms_with_id=lambda _: [SEARCH_CENTER])
    caller = Refinement(
        cfg,
        logger(),
        source,
        Neighbors(),
        environment,
        manager,
        global_constraints=user,
    )
    # Enumeration/energy scheduling are not under test; actual execute/refine_single are.
    caller.get_total_refinements_todo = lambda _: (1, 0.0)
    caller.get_energy_thr_refine = lambda *args: 1.0
    return cfg, triplet, user, source, manager, caller


def test_refinement_result_to_site_request_keeps_user_constraints(monkeypatch):
    cfg, triplet, user, source, manager, caller = refinery(monkeypatch)
    caller.execute(pd.DataFrame([reference_row(triplet)]), total_energy=0.0)
    assert len(manager.calls) == 1 and caller.results[0].is_ok()
    output = caller.results[0].ok_value()
    assert_user_only(output.constraints, triplet[0], user)
    np.testing.assert_array_equal(output.full_saddle_positions, triplet[1])
    service = capturing_service(cfg, user)
    table = ActiveEventTable(cfg, prefactor_service=service)
    table.add_events([output])
    with pytest.raises(CaptureComplete):
        table.request_site_prefactors(source, Neighbors())
    assert len(service.captured) == 1 and service.direction == [False]
    assert_request(service.captured[0], (triplet[0], triplet[1], triplet[0]), user)
    np.testing.assert_array_equal(source.positions, triplet[0])


def test_immediate_refinement_submission_failure_restores_python_source(monkeypatch):
    _, triplet, user, source, manager, caller = refinery(monkeypatch, fail=True)
    user_before = deepcopy(user)
    with pytest.raises(RuntimeError, match="injected immediate manager"):
        caller.execute(pd.DataFrame([reference_row(triplet)]), total_energy=0.0)
    assert len(manager.calls) == 1
    np.testing.assert_array_equal(source.positions, triplet[0])
    assert user == user_before


CROP_IDS = (999, 42, 8, 17, 91)
CROP_CELL = np.diag([40.0, 40.0, 40.0])
CROP_SOURCE = np.array(
    [
        [15.0, 15.0, 15.0],
        [5.0, 5.0, 5.0],
        [5.5, 5.0, 5.0],
        [5.0, 5.4, 5.0],
        [7.4, 5.0, 5.0],
    ]
)


class CommandEndpoint:
    """Native seam: record locks at minimization and move only unlocked rows.

    Values are protocol markers, never physical energies/forces or convergence.
    Command names for groups/fixes are deliberately not prescribed.
    """

    def __init__(self, positions):
        self.positions = positions.copy()
        self.commands, self.scatters, self.minimizations = [], [], []
        self.groups, self.fixes, self.computes = {}, {}, set()

    def command(self, command):
        self.commands.append(command)
        words = command.split()
        if words[0] == "clear":
            self.positions = np.empty((0, 3))
            self.groups.clear()
            self.fixes.clear()
            self.computes.clear()
        elif words[0] == "group":
            name, style = words[1:3]
            if style == "id":
                self.groups[name] = {int(i) for i in words[3:]}
            elif style == "empty":
                self.groups[name] = set()
            elif style == "delete":
                self.groups.pop(name, None)
            elif style == "union":
                self.groups[name] = set().union(*(self.members(n) for n in words[3:]))
            elif style == "subtract":
                self.groups[name] = self.members(words[3]) - set().union(
                    *(self.members(n) for n in words[4:])
                )
            else:
                raise AssertionError(f"unmodeled group command: {command}")
        elif words[0] == "fix":
            assert words[3] == "setforce", "pARTn itself is outside this protocol seam"
            self.fixes[words[1]] = words[2]
        elif words[0] == "unfix":
            self.fixes.pop(words[1], None)
        elif words[0] == "compute":
            self.computes.add(words[1])
        elif words[0] == "uncompute":
            self.computes.discard(words[1])
        elif words[0] == "minimize":
            locked = self.locked()
            self.minimizations.append((self.positions.copy(), locked.copy()))
            for row in range(len(self.positions)):
                if row + 1 not in locked:
                    self.positions[row, 0] += 0.01

    def members(self, group):
        return (
            set(range(1, len(self.positions) + 1))
            if group == "all"
            else self.groups[group].copy()
        )

    def locked(self):
        return set().union(*(self.members(group) for group in self.fixes.values()))

    def has_id(self, kind, name):
        return name in self.available_ids(kind)

    def available_ids(self, kind):
        return list(
            {"group": self.groups, "fix": self.fixes, "compute": self.computes}[kind]
        )

    def create_atoms(self, n, ids, types, x):
        self.positions = np.array(x, dtype=float).reshape(n, 3).copy()

    def get_natoms(self):
        return len(self.positions)

    def scatter_atoms(self, name, dtype, count, data):
        assert name == "x"
        self.positions = np.array(list(data), dtype=float).reshape(-1, 3)
        self.scatters.append(self.positions.copy())

    def gather_atoms(self, name, dtype, count):
        assert name == "x"
        flat = self.positions.ravel()
        return (ctypes.c_double * len(flat))(*flat)

    def extract_compute(self, *args):
        return -1.0

    def get_thermo(self, name):
        return -1.0


def crop_context():
    cfg = config()
    cfg.frozen_atoms = RegionConfig(indices=[2])
    cfg.activevolume.rmov, cfg.activevolume.ract = 1.0, 3.0
    types = ("Ni",) * 5
    payload = ResolvedConstraints.resolve(
        CROP_SOURCE,
        types,
        cfg.frozen_atoms,
        CROP_IDS,
        cell=CROP_CELL,
        pbc=PBC,
        center_id=42,
        rmov=1.0,
    )
    assert set(payload.fixed_ids) == {999, 8, 91}
    engine = LammpsEngine(cfg.lammps, comm=None)
    engine.full_system = FullSystem(
        types=types, species=("Ni",), masses=(58.6934,), cell=CROP_CELL.copy(), pbc=PBC
    )
    engine._is_orthorhombic = True
    engine.lmp = CommandEndpoint(CROP_SOURCE)
    return cfg, types, payload, engine


@pytest.mark.parametrize("operation", ["search", "refine"])
def test_native_preparation_maps_user_and_buffer_before_first_minimization(operation):
    cfg, types, payload, engine = crop_context()
    before = deepcopy(payload)
    if operation == "search":
        atom_map, center = av.partn_search_AV(
            engine, cfg, 1, CROP_SOURCE.copy(), CROP_CELL, types, constraints=payload
        )
        assert {2, 4}.issubset(engine.lmp.locked())
    else:
        saddle = CROP_SOURCE[[3]].copy()
        saddle[0, 1] += 0.1
        _, atom_map, center = av.partn_refine_AV(
            engine,
            cfg,
            1,
            CROP_SOURCE.copy(),
            CROP_CELL,
            types,
            np.array([3]),
            saddle,
            constraints=payload,
        )
        assert engine.lmp.minimizations, "exercise preparation minimization"
        for positions, locked in engine.lmp.minimizations:
            assert {2, 4}.issubset(locked), "user native ID 2 and buffer native ID 4"
            np.testing.assert_array_equal(positions[[1, 3]], CROP_SOURCE[[2, 4]])
        for scattered in engine.lmp.scatters:
            np.testing.assert_array_equal(scattered[[1, 3]], CROP_SOURCE[[2, 4]])
    np.testing.assert_array_equal(atom_map, [1, 2, 3, 4])
    np.testing.assert_array_equal(center, [1])
    assert payload == before and payload.source_ids == CROP_IDS


def test_public_refinement_invalid_user_overlay_rejects_before_native_mutation():
    cfg, types, payload, engine = crop_context()
    indices = np.array([3, 2])  # row 2 is the user-declared frozen atom
    saddle = CROP_SOURCE[indices].copy()
    saddle[1, 0] += 0.2
    with pytest.raises(ValueError):
        engine.partn_refine(
            cfg,
            1,
            positions=CROP_SOURCE.copy(),
            cell=CROP_CELL,
            types=types,
            saddle_idx=indices,
            saddle_positions=saddle,
            constraints=payload,
        )
    assert engine.lmp.commands == [] and engine.lmp.scatters == []
    np.testing.assert_array_equal(engine.lmp.positions, CROP_SOURCE)
    assert not engine.system_is_cropped


class _CropBuilt(RuntimeError):
    """Raised by the pARTn stand-in: the overlay passed the gate and was placed."""


def test_public_refinement_outer_buffer_overlay_is_placed_not_rejected(monkeypatch):
    # contracts 7f policy 5: the AV shell (row 4, beyond rmov) is a crop
    # restriction held by fix setforce, not a coordinate contract. Its overlay
    # reaches the crop and is scattered as given; formerly asserted ValueError.
    from pykmc.engine import lammps as lammps_module

    cfg, types, payload, engine = crop_context()
    indices = np.array([3, 4])
    saddle = CROP_SOURCE[indices].copy()
    saddle[1, 0] += 0.2
    monkeypatch.setattr(
        lammps_module,
        "pypARTn",
        SimpleNamespace(artn=lambda engine: (_ for _ in ()).throw(_CropBuilt())),
    )
    monkeypatch.setattr(engine, "ensure_full_system", lambda positions: True)
    with pytest.raises(_CropBuilt):
        engine.partn_refine(
            cfg,
            1,
            positions=CROP_SOURCE.copy(),
            cell=CROP_CELL,
            types=types,
            saddle_idx=indices,
            saddle_positions=saddle,
            constraints=payload,
        )
    assert engine.lmp.scatters, "the crop was built and the saddle overlay placed"
    # Crop rows follow atom_map [1, 2, 3, 4]; source row 4 is native row 3.
    np.testing.assert_array_equal(engine.lmp.scatters[-1][3], saddle[1])
    assert 4 in engine.lmp.locked(), "the buffer row stays held by setforce"
