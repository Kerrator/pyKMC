"""Optional full-energy protocol; no native engine or MPI job is opened.

Extends the ``producing_helpers`` recording scratch. Real native-adapter
orchestration, request/provenance validation and Vineyard kernel execute; only
scratch resource operations and the terminal Hessian are protocol doubles. These
are ordering and resource assertions, not a native or potential-energy accuracy
claim.
"""

from __future__ import annotations

import inspect
from types import SimpleNamespace

import numpy as np
import pytest

from . import producing_helpers as base


def require_option():
    base.require_api()
    assert (
        "compute_energies"
        in inspect.signature(
            base.LammpsHTSTExtension.compute_event_prefactors
        ).parameters
    ), "An explicit optional full-energy operation argument is required"


class EnergyScratch(base.Scratch):
    def __init__(self, native, *, fail_energy=False):
        super().__init__(native)
        self.timeline = []
        self.energy_calls = []
        self.matrix_calls = []
        self.premin_calls = []
        self.fail_energy = fail_energy
        self.initiating_error = RuntimeError(
            "injected full-system saddle energy failure"
        )

    def initialize_system(self, **kwargs):
        super().initialize_system(**kwargs)
        self.timeline.append(("build", self.types))

    def command(self, command):
        super().command(command)
        self.timeline.append(("clear", self.types))

    def minimize_freeze_core(self, core):
        before = self.positions.copy()
        super().minimize_freeze_core(core)
        self.premin_calls.append((tuple(core), before, self.positions.copy()))
        self.timeline.append(("premin", self.types))

    def get_potential_energy(self, positions=None, *, recompute=True):
        assert recompute, "fresh current-geometry potential energy is required"
        assert self.types == base.TYPES, "energy must precede the native zone crop"
        assert positions is not None, "each energy must explicitly select its vertex"
        assert np.asarray(positions).shape == (4, 3)
        self.set_positions(positions)
        self.timeline.append(("energy", self.types))
        self.energy_calls.append(self.positions.copy())
        # Explicitly depends on outer O (absent from the crop) and relaxed Fe.
        # The three vertex levels and premin shift are independently declared.
        x = float(self.positions[2, 0])
        value = {2.0: 0.0, 3.0: 1.0, 4.0: 0.25}[x]
        value += 7.0 * self.positions[0, 1] + 11.0 * (self.positions[3, 1] - 1.5)
        if self.fail_energy and len(self.energy_calls) == 2:
            raise self.initiating_error
        return float(value)

    def get_total_energy(self, *args, **kwargs):
        raise AssertionError("barrier provenance requires potential energy, not etotal")

    def close(self):
        self.timeline.append(("close", self.types))
        super().close()


def setup(premin, zone, *, fail_energy=False):
    request, native = base.request_case(premin=premin, zone=zone)
    original = base.require_api().RequestSnapshot.capture(request).to_request()
    scratch = EnergyScratch(native, fail_energy=fail_energy)
    extension = object.__new__(base.LammpsHTSTExtension)
    extension.engine = SimpleNamespace(config=native, comm=None, engine_id=0)
    extension._new_scratch = lambda: scratch

    def local_matrix(resource, positions, free, step):
        assert resource is scratch
        local_ni = scratch.types.index("Ni")
        assert tuple(free) == (local_ni,)
        scratch.matrix_calls.append(
            (scratch.types, np.array(positions, copy=True), tuple(free))
        )
        scratch.timeline.append(("hessian", scratch.types))
        x = positions[local_ni, 0]
        return np.diag([-1.0, 2.0, 3.0] if x == 3.0 else [4.0, 5.0, 6.0])

    extension._eskm_hessian = local_matrix
    return request, original, scratch, extension


def assert_common(request, original, scratch, result, premin, zone, enabled, backward):
    assert scratch.closed and scratch.timeline[-1][0] == "close"
    base.assert_geometry(request, original)
    assert result.forward.ok and result.forward.n_free == 1
    assert result.backward.ok if backward else result.backward.skipped
    assert len(scratch.matrix_calls) == (3 if backward else 2)
    expected_types = ("Cu", "Ni", "Fe") if zone else base.TYPES
    assert all(types == expected_types for types, _, _ in scratch.matrix_calls)
    expected_prepared = base.produced_request(request) if premin else request
    record = result.provenance
    assert record is not None
    base.assert_geometry(record.source.to_request(), request)
    base.assert_geometry(record.produced.to_request(), expected_prepared)
    assert record.source.is_complete and record.produced.is_complete
    assert record.free_indices == (2,)
    assert record.zone_indices == ((1, 2, 3) if zone else (0, 1, 2, 3))
    assert result.calculation("forward").identity.free_ids == (42,)
    assert len(scratch.premin_calls) == (3 if premin else 0)
    for core, before, after in scratch.premin_calls:
        assert core == (1, 2)
        np.testing.assert_array_equal(after[list(core)], before[list(core)])
    if enabled:
        assert len(scratch.energy_calls) == 3
        for actual, key in zip(scratch.energy_calls, base.GEOMETRIES, strict=True):
            np.testing.assert_array_equal(actual, getattr(expected_prepared, key))
        expected_energies = (1.8, 2.8, 2.05) if premin else (0.0, 1.0, 0.25)
        np.testing.assert_allclose(
            record.energies, expected_energies, rtol=0.0, atol=2e-14
        )
        stages = [name for name, _ in scratch.timeline]
        energy_indices = [i for i, name in enumerate(stages) if name == "energy"]
        assert max(energy_indices) < stages.index("hessian")
        if premin:
            assert max(i for i, name in enumerate(stages) if name == "premin") < min(
                energy_indices
            )
        if zone:
            assert max(energy_indices) < stages.index("clear")
            assert scratch.builds == [base.TYPES, expected_types]
        else:
            assert scratch.builds == [base.TYPES]
    else:
        assert scratch.energy_calls == []
        assert record.energies is None
        assert not any(name == "energy" for name, _ in scratch.timeline)
        # The false option preserves ordinary build/premin/Hessian counts.
        assert scratch.builds == (
            [base.TYPES, expected_types] if premin and zone else [expected_types]
        )
    if not backward:
        assert result.calculation("backward") is None
        assert all(
            pos[types.index("Ni"), 0] != 4.0 for types, pos, _ in scratch.matrix_calls
        )


@pytest.mark.parametrize("premin", [False, True], ids=["no-premin", "premin"])
@pytest.mark.parametrize("zone", [None, 2.0], ids=["full", "crop"])
@pytest.mark.parametrize("enabled", [False, True], ids=["ordinary", "full-energies"])
def test_optional_energies_use_full_prepared_vertices_without_ordinary_cost(
    premin, zone, enabled
):
    require_option()
    request, original, scratch, extension = setup(premin, zone)
    # Ordinary lane deliberately omits the keyword, proving its default.
    options = {"compute_energies": True} if enabled else {}
    result = extension.compute_event_prefactors(request, **options)
    assert_common(request, original, scratch, result, premin, zone, enabled, True)


@pytest.mark.parametrize("zone", [None, 2.0], ids=["full", "crop"])
def test_full_energy_triplet_does_not_enable_backward_hessian(zone):
    require_option()
    request, original, scratch, extension = setup(True, zone)
    result = extension.compute_event_prefactors(
        request, compute_backward=False, compute_energies=True
    )
    assert_common(request, original, scratch, result, True, zone, True, False)


@pytest.mark.parametrize("premin", [False, True], ids=["no-premin", "premin"])
@pytest.mark.parametrize("zone", [None, 2.0], ids=["full", "crop"])
def test_full_energy_error_propagates_original_cause_and_closes_scratch(premin, zone):
    require_option()
    request, original, scratch, extension = setup(premin, zone, fail_energy=True)
    with pytest.raises(RuntimeError) as caught:
        extension.compute_event_prefactors(request, compute_energies=True)
    assert caught.value is scratch.initiating_error
    assert scratch.closed and scratch.timeline[-1][0] == "close"
    assert scratch.matrix_calls == []
    assert scratch.builds == [base.TYPES]
    assert len(scratch.energy_calls) == 2
    assert len(scratch.premin_calls) == (3 if premin else 0)
    assert not any(name == "clear" for name, _ in scratch.timeline)
    prepared = base.produced_request(request) if premin else request
    for actual, key in zip(scratch.energy_calls, base.GEOMETRIES[:2], strict=True):
        np.testing.assert_array_equal(actual, getattr(prepared, key))
    base.assert_geometry(request, original)
