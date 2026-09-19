"""Physical descriptor transport oracles; no native engine or MPI pool is launched.

Fake coefficient bytes are identity fixtures only. The recording worker below
returns a labelled protocol result and makes no numerical/HTST support claim.
Missing new interfaces must appear as ordinary assertion failures on the base.
"""

from concurrent.futures import Future
from dataclasses import replace
import importlib
import importlib.util
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest


def api():
    spec = importlib.util.find_spec("pykmc.physics")
    assert spec is not None, (
        "Physical descriptor requires the shared pykmc.physics contract"
    )
    module = importlib.import_module("pykmc.physics")
    for name, method in (
        ("ForceModel", "capture"),
        ("EnginePhysics", "capture"),
        ("PhysicalDescriptor", "from_config"),
        ("ResolvedConstraints", "resolve"),
    ):
        cls = getattr(module, name, None)
        assert cls is not None, f"Physical descriptor requires {name}"
        assert callable(getattr(cls, method, None)), (
            f"Physical descriptor requires {name}.{method}"
        )
    return module


def configured(tmp_path, *, premin=False, frozen=None):
    import pykmc
    from pykmc.config import Config, RateConstantConfig

    coefficient = tmp_path / "same-path.eam.alloy"
    coefficient.write_bytes(
        b"Physical descriptor coefficient identity fixture version A\n"
    )
    cfg = Config.from_ini_file(
        str(Path(pykmc.__file__).resolve().parent.parent / "tests/data/input.in")
    )
    cfg = cfg.model_copy(
        update={
            "rateconstant": RateConstantConfig(style="htst", k0=1.0, premin=premin),
            "control": cfg.control.model_copy(update={"reference_table": None}),
            "lammps": cfg.lammps.model_copy(
                update={
                    "pair_style": "eam/alloy",
                    "pair_coeff": f"* * {coefficient} Ni Cu H",
                    "min_style": "cg",
                    "frz_min": "1e-6 1e-8 10 10",
                }
            ),
            "frozen_atoms": frozen,
        }
    )
    return cfg, coefficient


def descriptor(
    module, cfg, *, species=("Ni", "Cu", "H"), masses=(61.0, 65.0, 2.0), settings=None
):
    from pykmc.rate_constant.prefactors import settings_from_config

    engine = module.EnginePhysics.capture(cfg.lammps, species, masses)
    settings = settings or settings_from_config(cfg.rateconstant)
    return module.PhysicalDescriptor.from_config(cfg, engine, settings)


def comparison(actual, expected):
    assert getattr(actual, "status", None) == expected
    reasons = getattr(actual, "reasons", None)
    assert reasons is not None, "comparison must carry explicit reasons"
    if expected != "compatible":
        assert reasons, f"{expected} comparison needs its reason"


def service(module, cfg, manager):
    from pykmc.rate_constant import create_rate_constant
    from pykmc.rate_constant.prefactors import PrefactorService

    engine = module.EnginePhysics.capture(
        cfg.lammps, ("Ni", "Cu", "H"), (61.0, 65.0, 2.0)
    )
    return PrefactorService(
        cfg, manager, create_rate_constant(cfg.rateconstant), engine_physics=engine
    )


def geometry():
    initial = np.array([[1.0, 1.0, 1.0], [2.0, 1.0, 1.0], [8.0, 8.0, 8.0]])
    saddle = initial.copy()
    saddle[0, 0] += 0.2
    final = initial.copy()
    final[0, 0] += 0.4
    return initial, saddle, final


def request(svc, *, key=("batch", 1), positions=None, pbc=(True, False, True)):
    first, saddle, final = geometry() if positions is None else positions
    return svc.build_request(
        event_key=key,
        min1_positions=first,
        saddle_positions=saddle,
        min2_positions=final,
        types=("Ni", "Ni", "Ni"),  # Cu and H slots are deliberately absent.
        cell=20.0 * np.eye(3),
        pbc=pbc,
        center_index=0,
    )


def test_descriptor_snapshot_and_potential_content_identity(tmp_path):
    module = api()
    cfg, coefficient = configured(tmp_path)
    old = descriptor(module, cfg)
    old_id = old.descriptor_id
    assert old.reusable, "complete, known file-backed model must be reusable"
    comparison(old.compare(descriptor(module, cfg)), "compatible")
    coefficient.write_bytes(
        b"Physical descriptor coefficient identity fixture version B\n"
    )
    current = descriptor(module, cfg)
    assert old.descriptor_id == old_id, (
        "captured identity must not follow later file edits"
    )
    assert current.descriptor_id != old_id
    comparison(current.compare(old), "incompatible")
    with pytest.raises((AttributeError, TypeError)):
        old.descriptor_id = "changed"


@pytest.mark.parametrize(
    "changed", ["species_order", "absent_slot", "mass", "force_definition"]
)
def test_engine_inputs_change_descriptor(tmp_path, changed):
    module = api()
    cfg, coefficient = configured(tmp_path)
    old = descriptor(module, cfg)
    if changed == "species_order":
        cfg = cfg.model_copy(
            update={
                "lammps": cfg.lammps.model_copy(
                    update={"pair_coeff": f"* * {coefficient} Cu Ni H"}
                )
            }
        )
        current = descriptor(
            module, cfg, species=("Cu", "Ni", "H"), masses=(65.0, 61.0, 2.0)
        )
    elif changed == "absent_slot":
        cfg = cfg.model_copy(
            update={
                "lammps": cfg.lammps.model_copy(
                    update={"pair_coeff": f"* * {coefficient} Ni Cu"}
                )
            }
        )
        current = descriptor(module, cfg, species=("Ni", "Cu"), masses=(61.0, 65.0))
    elif changed == "mass":
        current = descriptor(module, cfg, masses=(62.0, 65.0, 2.0))
    else:
        cfg = cfg.model_copy(
            update={"lammps": cfg.lammps.model_copy(update={"pair_style": "eam/fs"})}
        )
        current = descriptor(module, cfg)
    assert current.descriptor_id != old.descriptor_id
    comparison(current.compare(old), "incompatible")


def test_unknown_force_and_missing_provenance_are_not_compatible(tmp_path):
    module = api()
    cfg, _ = configured(tmp_path)
    known = descriptor(module, cfg)
    comparison(known.compare(None), "unknown")
    opaque = cfg.model_copy(
        update={
            "lammps": cfg.lammps.model_copy(
                update={
                    "pair_style": "opaque/example",
                    "pair_coeff": "* * unresolved-driver-token",
                }
            )
        }
    )
    unknown = descriptor(module, opaque)
    assert not unknown.reusable
    comparison(unknown.compare(unknown), "unknown")


@pytest.mark.parametrize("premin", [False, True])
@pytest.mark.parametrize(
    "field,value", [("min_style", "fire"), ("frz_min", "1e-12 1e-14 1000 1000")]
)
def test_premin_dependencies_are_conditional(tmp_path, premin, field, value):
    module = api()
    cfg, _ = configured(tmp_path, premin=premin)
    changed = cfg.model_copy(
        update={"lammps": cfg.lammps.model_copy(update={field: value})}
    )
    old, current = descriptor(module, cfg), descriptor(module, changed)
    comparison(current.compare(old), "incompatible" if premin else "compatible")
    assert (current.descriptor_id != old.descriptor_id) is premin


def test_numerical_inputs_but_not_rate_or_window_policy_define_physics(tmp_path):
    module = api()
    from pykmc.rate_constant.prefactors import settings_from_config

    cfg, _ = configured(tmp_path)
    base_settings = settings_from_config(cfg.rateconstant)
    old = descriptor(module, cfg, settings=base_settings)
    for change in (
        {"fd_step": 0.02},
        {"free_radius": 5.5},
        {"free_region_center": "min1"},
        {"zone_radius": 9.0},
        {"zero_mode_tol": 2e-6},
    ):
        current = descriptor(module, cfg, settings=replace(base_settings, **change))
        comparison(current.compare(old), "incompatible")
    changed_cfg = cfg.model_copy(
        update={
            "rateconstant": cfg.rateconstant.model_copy(update={"T": 600.0, "k0": 3.0})
        }
    )
    widened = replace(base_settings, nu0_min_hz=1e11, nu0_max_hz=1e15)
    current = descriptor(module, changed_cfg, settings=widened)
    comparison(current.compare(old), "compatible")
    assert current.descriptor_id == old.descriptor_id
    assert not ({"nu0_min_hz", "nu0_max_hz", "T", "k0"} & set(old.numerical_settings()))


def test_constraint_crop_keeps_global_ids_and_immutable_fixed_coordinates():
    module = api()
    from pykmc.config import RegionConfig

    source = np.array(
        [[0.0, 0.0, 0.0], [1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]]
    )
    resolved = module.ResolvedConstraints.resolve(
        source, ("Ni",) * 4, region=RegionConfig(indices=[2, 3])
    )
    assert resolved.source_ids == (0, 1, 2, 3)
    assert resolved.atom_ids == (0, 1, 2, 3)
    assert resolved.fixed_ids == (2, 3)
    assert resolved.fixed_positions == ((4.0, 5.0, 6.0), (7.0, 8.0, 9.0))
    source[2:, :] = -99.0
    assert resolved.fixed_positions == ((4.0, 5.0, 6.0), (7.0, 8.0, 9.0))
    cropped = resolved.crop((3, 0, 2))
    cropped.validate(3)
    assert cropped.source_ids == (0, 1, 2, 3)
    assert cropped.atom_ids == (3, 0, 2)
    assert cropped.fixed_ids == (2, 3)
    assert cropped.fixed_positions == ((4.0, 5.0, 6.0), (7.0, 8.0, 9.0))
    assert cropped.local_fixed_indices == (0, 2)
    again = cropped.crop((2, 1))
    assert again.atom_ids == (2, 0)
    assert again.local_fixed_indices == (0,)
    with pytest.raises((AttributeError, TypeError)):
        resolved.fixed_positions[0][0] = 0.0


@pytest.mark.parametrize("indices", [(-1,), (4,), (1, 1), (True,), (1.5,)])
def test_bad_constraint_indices_reject_before_crop(indices):
    module = api()
    # Deliberately bypass RegionConfig's integer coercion to test the boundary.
    region = SimpleNamespace(indices=indices, types=(), region_type=None)
    with pytest.raises((ValueError, TypeError)):
        module.ResolvedConstraints.resolve(np.zeros((4, 3)), ("Ni",) * 4, region=region)


def test_constraint_source_identity_and_validation():
    module = api()
    positions = np.arange(12.0).reshape(4, 3)
    resolved = module.ResolvedConstraints.resolve(
        positions, ("Ni",) * 4, atom_ids=(2, 0, 3, 1)
    )
    assert resolved.source_ids == (2, 0, 3, 1)
    assert resolved.atom_ids == (2, 0, 3, 1)
    with pytest.raises((ValueError, TypeError)):
        resolved.validate(3)
    for bad in ((0, 0, 2, 3), (0, 1, 2), (-1, 0, 1, 2), (0, 1, 2, True)):
        with pytest.raises((ValueError, TypeError)):
            module.ResolvedConstraints.resolve(positions, ("Ni",) * 4, atom_ids=bad)
    for bad_crop in ((0, 0), (-1,), (4,), (True,), (1.5,)):
        with pytest.raises((ValueError, TypeError)):
            resolved.crop(bad_crop)
    invalid = positions.copy()
    invalid[0, 0] = np.nan
    with pytest.raises((ValueError, TypeError)):
        module.ResolvedConstraints.resolve(invalid, ("Ni",) * 4)


def test_service_table_and_request_share_authoritative_descriptor(tmp_path):
    module = api()
    from pykmc.config import RegionConfig
    from pykmc.event_table import ReferenceEventTable

    cfg, _ = configured(tmp_path, frozen=RegionConfig(indices=[2]))
    svc = service(module, cfg, object())
    req = request(svc)
    req.validate()
    assert req.species == ("Ni", "Cu", "H")
    assert req.masses == (61.0, 65.0, 2.0)
    assert req.constraints.fixed_ids == (2,)
    assert req.constraints.fixed_positions == ((8.0, 8.0, 8.0),)
    current = svc.descriptor_for(req.types)
    assert req.descriptor.descriptor_id == current.descriptor_id
    table = ReferenceEventTable(cfg, prefactor_service=svc)
    assert table.current_descriptor.descriptor_id == current.descriptor_id
    comparison(table.compare_physics(current), "compatible")
    comparison(table.compare_physics(None), "unknown")
    comparison(
        table.compare_physics(descriptor(module, cfg, masses=(62.0, 65.0, 2.0))),
        "incompatible",
    )
    other_cfg = cfg.model_copy(update={"frozen_atoms": RegionConfig(indices=[1])})
    changed = request(service(module, other_cfg, object()))
    assert changed.constraints.fixed_ids == (1,)
    comparison(changed.descriptor.compare(req.descriptor), "incompatible")


def test_calculation_identity_ignores_batch_labels_but_covers_full_dependencies(
    tmp_path,
):
    module = api()
    cfg, _ = configured(tmp_path)
    svc = service(module, cfg, object())
    req = request(svc)
    first = req.calculation_identity(direction="forward", free_indices=(0, 1))
    relabelled = request(svc, key=("other-row", 992)).calculation_identity(
        direction="forward", free_indices=(0, 1)
    )
    assert first.identity_id == relabelled.identity_id
    assert first.descriptor_id == req.descriptor.descriptor_id
    assert (
        first.identity_id
        != req.calculation_identity(
            direction="backward", free_indices=(0, 1)
        ).identity_id
    )
    assert (
        first.identity_id
        != req.calculation_identity(direction="forward", free_indices=(0,)).identity_id
    )
    positions = list(geometry())
    positions[0][2, 2] += 0.3  # Noncentral atom outside the changed event core.
    neighbor = request(svc, positions=positions).calculation_identity(
        direction="forward", free_indices=(0, 1)
    )
    assert first.identity_id != neighbor.identity_id
    pbc = request(svc, pbc=(False, False, True)).calculation_identity(
        direction="forward", free_indices=(0, 1)
    )
    assert first.identity_id != pbc.identity_id
    with pytest.raises((AttributeError, TypeError)):
        first.identity_id = "changed"


def test_initializer_retains_preflight_engine_snapshot(tmp_path):
    module = api()
    from pykmc.initializer import Initializer
    from pykmc.rate_constant import create_rate_constant

    cfg, _ = configured(tmp_path)
    engine = module.EnginePhysics.capture(
        cfg.lammps, ("Ni", "Cu", "H"), (61.0, 65.0, 2.0)
    )
    report = {
        "phonon": True,
        "lammps_version": 20250722,
        "pair_style": "eam/alloy",
        "species": ("Ni", "Cu", "H"),
        "masses": (61.0, 65.0, 2.0),
        "engine_physics": engine,
    }
    validated = Initializer._preflight_report(report)
    assert validated["engine_physics"] == engine
    kmc = SimpleNamespace(
        config=cfg,
        manager=object(),
        rate_constant=create_rate_constant(cfg.rateconstant),
        htst_preflight=validated,
        uses_event_prefactors=True,
        loggers=SimpleNamespace(info=lambda *args: None),
    )
    Initializer(kmc).initialize_prefactor_service()
    built = request(kmc.prefactor_service)
    expected = module.PhysicalDescriptor.from_config(
        cfg, engine, kmc.prefactor_service.settings
    )
    assert built.descriptor.descriptor_id == expected.descriptor_id
    assert built.species == ("Ni", "Cu", "H") and built.masses == (61.0, 65.0, 2.0)


class RecordingWorker:
    """No force calculation; records actual request dispatch at table boundaries."""

    def __init__(self):
        self.requests = []

    def submit(self, operation, *, request, compute_backward=True):
        from pykmc.htst.result import DirectionalPrefactor, EventPrefactors

        assert operation == "compute_event_prefactors"
        self.requests.append((request, compute_backward))
        forward = DirectionalPrefactor.accepted(
            8e12, n_free=1, n_positive_min=3, n_negative_saddle=1
        )
        backward = (
            forward
            if compute_backward
            else DirectionalPrefactor.not_requested(n_free=1, n_negative_saddle=1)
        )
        result = EventPrefactors(
            request.event_key,
            forward,
            backward,
            "Physical descriptor_protocol_stub",
            1,
            request.settings,
        )
        future = Future()
        future.set_result(result)
        return future


def test_reference_and_site_callers_transport_the_contract(tmp_path, monkeypatch):
    module = api()
    from pykmc.config import RegionConfig
    from pykmc.event_table import ActiveEventTable, EventAdmission, ReferenceEventTable
    from pykmc.result import EventRefinementOutput

    cfg, _ = configured(tmp_path, frozen=RegionConfig(indices=[2]))
    worker = RecordingWorker()
    svc = service(module, cfg, worker)
    first, saddle, final = geometry()
    ref = ReferenceEventTable(cfg, prefactor_service=svc)
    patched = []

    def record_patch(idx, estimate, *, calculation=None, fresh=False):
        assert calculation is None  # RecordingWorker never calculated a spectrum.
        assert fresh is True
        patched.append(idx)

    monkeypatch.setattr(ref, "_patch_row", record_patch)
    monkeypatch.setattr(ref, "_log_direction", lambda *args: None)
    event = SimpleNamespace(
        min1_positions=first,
        saddle_positions=saddle,
        min2_positions=final,
        types=("Ni",) * 3,
        cell=20.0 * np.eye(3),
        move_atom_index=0,
    )
    ref._resolve_prefactors(
        [(17, 42, EventAdmission(pd.DataFrame()), event)], pbc=(True, False, True)
    )
    assert patched == [17, 42]
    active = ActiveEventTable(cfg, prefactor_service=svc)
    active.add_events(
        EventRefinementOutput(
            central_atom_index=0,
            saddle_positions=saddle[:2].copy(),
            E_saddle=1.0,
            min2_positions=final[:2].copy(),
            dE_forward=1.0,
            num_reference_event=17,
            refined="T",
            nu0_hz=8e12,
            nu0_status="ok",
            nu0_reason="",
            nu0_source="reference",
            full_saddle_positions=saddle.copy(),
        )
    )
    system = SimpleNamespace(
        positions=first,
        types=("Ni",) * 3,
        cell=20.0 * np.eye(3),
        pbc=(True, False, True),
    )
    neighbors = SimpleNamespace(
        get_neighbors=lambda kind, atom: np.array([0, 1], dtype=int)
    )
    active.request_site_prefactors(system, neighbors)
    assert len(worker.requests) == 2
    reference_request, backward = worker.requests[0]
    site_request, site_backward = worker.requests[1]
    assert backward is True and site_backward is False
    for observed in (reference_request, site_request):
        assert observed.descriptor.descriptor_id == ref.current_descriptor.descriptor_id
        assert observed.species == ("Ni", "Cu", "H")
        assert observed.masses == (61.0, 65.0, 2.0)
        assert observed.constraints.source_ids == (0, 1, 2)
        assert observed.constraints.fixed_ids == (2,)
        assert observed.constraints.fixed_positions == ((8.0, 8.0, 8.0),)
        assert np.array_equal(observed.saddle_positions, saddle)


def scratch_spy(monkeypatch, cfg):
    """Exercise actual HTST adapter orchestration with zero native mutations."""
    import pykmc.engine.htst_lammps as adapter

    calls = []
    extension = object.__new__(adapter.LammpsHTSTExtension)
    extension.engine = SimpleNamespace(config=cfg.lammps, comm=None, engine_id=0)

    class Scratch:
        def start(self):
            calls.append(("start",))

        def initialize_parameters(self):
            calls.append(("parameters",))

        def initialize_system(self, **kwargs):
            calls.append(("system", kwargs))
            self.full_system = SimpleNamespace(
                species=kwargs["species"], masses=kwargs["masses"], physics=None
            )

        def initialize_potential(self):
            calls.append(("potential",))
            from pykmc.physics import EnginePhysics

            self.full_system.physics = EnginePhysics.capture(
                cfg.lammps, self.full_system.species, self.full_system.masses
            )

        def close(self):
            calls.append(("close",))

    scratch = Scratch()

    def create():
        calls.append(("create",))
        return scratch

    results = []

    def kernel(local_request, hessian, **kwargs):
        local_request.validate()
        calls.append(("kernel", local_request, kwargs))
        from pykmc.htst.result import DirectionalPrefactor, EventPrefactors

        n_free = len(kwargs["free_indices"])
        estimate = DirectionalPrefactor.accepted(
            1e12, n_free=n_free, n_positive_min=3 * n_free, n_negative_saddle=1
        )
        result = EventPrefactors(
            local_request.event_key,
            estimate,
            estimate,
            kwargs["method"],
            n_free,
            local_request.settings,
        )
        results.append(result)
        return result

    monkeypatch.setattr(extension, "_new_scratch", create)
    monkeypatch.setattr(extension, "_hessian_fn", lambda *args: None)
    monkeypatch.setattr(adapter, "_kernel_compute_event_prefactors", kernel)
    return extension, calls, results


@pytest.mark.parametrize("zone_radius", [None, 3.0])
def test_native_adapter_transports_descriptor_and_crop_mapping_without_native_calls(
    tmp_path, monkeypatch, zone_radius
):
    module = api()
    from pykmc.config import RegionConfig

    cfg, _ = configured(tmp_path, frozen=RegionConfig(indices=[1, 2]))
    cfg = cfg.model_copy(
        update={
            "rateconstant": cfg.rateconstant.model_copy(
                update={"free_radius": 1.0, "zone_radius": zone_radius}
            )
        }
    )
    req = request(service(module, cfg, object()))
    extension, calls, results = scratch_spy(monkeypatch, cfg)
    result = extension.compute_event_prefactors(req)
    assert result == results[0]
    assert result.forward == results[0].forward
    assert result.backward == results[0].backward
    assert result.provenance.source.is_complete
    assert result.provenance.produced.is_complete
    assert (
        result.provenance.source.to_request().calculation_identity()
        == req.calculation_identity()
    )
    assert [call[0] for call in calls] == [
        "create",
        "start",
        "parameters",
        "system",
        "potential",
        "kernel",
        "close",
    ]
    built = next(call[1] for call in calls if call[0] == "system")
    local = next(call[1] for call in calls if call[0] == "kernel")
    expected_ids = (0, 1, 2) if zone_radius is None else (0, 1)
    assert built["species"] == ("Ni", "Cu", "H")
    assert built["masses"] == (61.0, 65.0, 2.0)
    assert built["types"] == ("Ni",) * len(expected_ids)
    assert local.descriptor.descriptor_id == req.descriptor.descriptor_id
    assert local.constraints.source_ids == (0, 1, 2)
    assert local.constraints.atom_ids == expected_ids
    assert local.constraints.fixed_ids == (1, 2)
    assert local.constraints.fixed_positions == ((2.0, 1.0, 1.0), (8.0, 8.0, 8.0))
    assert local.constraints.local_fixed_indices == (
        (1, 2) if zone_radius is None else (1,)
    )
    assert np.array_equal(local.min1_positions, req.min1_positions[list(expected_ids)])
    # Deliberately no assertion that free_indices exclude fixed atoms: The constrained Hessian consumer owns
    # that numerical behavior. This test proves the payload reaches its consumer.


@pytest.mark.parametrize("failure", ["mismatched_mass", "changed_file"])
def test_native_adapter_rejects_inconsistent_physics_before_scratch(
    tmp_path, monkeypatch, failure
):
    module = api()
    cfg, coefficient = configured(tmp_path)
    req = request(service(module, cfg, object()))
    if failure == "mismatched_mass":
        req = replace(req, masses=(62.0, 65.0, 2.0))
        expected_diagnostic = "(?i)mass|map|descriptor|physics"
    else:
        coefficient.write_bytes(
            b"Physical descriptor coefficient identity fixture changed after capture\n"
        )
        expected_diagnostic = "(?i)force|potential|content|physics|model"
    extension, calls, _ = scratch_spy(monkeypatch, cfg)
    with pytest.raises(
        (ValueError, TypeError, RuntimeError), match=expected_diagnostic
    ):
        extension.compute_event_prefactors(req)
    assert calls == [], (
        "bad provenance/map must reject before scratch creation or mutation"
    )
