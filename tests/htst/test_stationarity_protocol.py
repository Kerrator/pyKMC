"""Independent native-event stationarity protocol; no native handle opens.

The literal matrices only establish known valid saddle/minimum spectra. Forces
are independent protocol inputs, so no matrix can stand in for stationarity.
Actual extension request validation, build/premin/crop, event force guard and
kernel execute. Only scratch resources and the terminal matrix call are fake.
"""

from dataclasses import replace
from types import SimpleNamespace

import numpy as np
import pytest

from pykmc.config import RateConstantConfig, RegionConfig
from pykmc.engine.htst_lammps import LammpsHTSTExtension
from pykmc.htst import HTSTRequestError, HTSTSettings
from pykmc.physics import EnginePhysics, ResolvedConstraints
from pykmc.rate_constant import create_rate_constant
from pykmc.rate_constant.prefactors import PrefactorService


TOL = 0.005
SPECIES = ("Ni", "Fe", "Cu", "O")
MASSES = (10.0, 20.0, 30.0, 40.0)
TYPES = ("O", "Cu", "Ni", "Fe")
IDS = (8, 91, 42, 17)
SOURCE = np.array([[9.0, 0.0, 0.0], [3.0, 0.25, 0.0], [2.0, 0.0, 0.0], [3.0, 1.5, 0.0]])
CELL = np.eye(3) * 20.0
PBC = (False, False, False)


def require_force_api():
    assert "force_tol" in HTSTSettings.__dataclass_fields__, (
        "HTSTSettings must declare force_tol"
    )
    assert "force_tol" in RateConstantConfig.model_fields, (
        "RateConstantConfig must declare force_tol"
    )


def label_of(positions, types):
    coordinate = float(positions[types.index("Ni"), 0])
    assert coordinate in (2.0, 3.0, 4.0)
    return {2.0: "min1", 3.0: "saddle", 4.0: "min2"}[coordinate]


class FakeScratch:
    def __init__(self, config, forces, calls):
        self.config, self.force_vectors, self.calls = config, forces, calls
        self.types = ()
        self.positions = None
        self.closed = False
        self.observed = []
        self.lmp = SimpleNamespace(command=self.command)

    def start(self):
        self.calls.append(("start",))

    def command(self, text):
        assert text == "clear", "unexpected native command escaped the fake boundary"
        self.calls.append(("clear",))

    def initialize_parameters(self):
        self.calls.append(("parameters",))

    def initialize_system(self, *, types, positions, cell, pbc, species, masses):
        self.types = tuple(types)
        self.positions = np.array(positions, copy=True)
        self.full_system = SimpleNamespace(species=species, masses=masses, physics=None)
        self.calls.append(("build", len(types), tuple(types)))

    def initialize_potential(self):
        self.full_system.physics = EnginePhysics.capture(
            self.config, self.full_system.species, self.full_system.masses
        )

    def set_positions(self, positions):
        self.positions = np.array(positions, copy=True)

    def get_positions(self):
        return self.positions.copy()

    def minimize_freeze_core(self, core):
        # Deterministic relaxation of noncore/nonfixed surrounding rows.
        assert tuple(core) == (1, 2), "premin must protect event core plus fixed rows"
        self.calls.append(("premin", label_of(self.positions, self.types)))
        for row in set(range(len(self.types))) - set(core):
            self.positions[row, 1] += 0.1

    def get_forces(self, *, positions):
        geometry = np.array(positions, copy=True)
        label = label_of(geometry, self.types)
        forces = np.zeros_like(geometry)
        forces[self.types.index("Ni")] = self.force_vectors.get(label, (0.0, 0.0, 0.0))
        # Both a fixed atom in the core and a movable atom outside the core
        # may carry large reactions in this partial constrained problem.
        forces[self.types.index("Cu")] = (100.0, -200.0, 300.0)
        forces[self.types.index("Fe")] = (-50.0, 60.0, 70.0)
        self.observed.append((label, geometry, self.types, forces.copy()))
        self.calls.append(("force", label))
        return forces

    def close(self):
        self.closed = True
        self.calls.append(("close",))


def setup_case(*, forces=None, premin=False, zone=None, force_tol=TOL):
    require_force_api()
    native = SimpleNamespace(
        pair_style="zero 10.0",
        pair_coeff="* *",
        min_style="cg",
        minimize="0.0 1e-12 100 1000",
        frz_min="0.0 1e-12 100 1000",
        verbosity=0,
    )
    config = SimpleNamespace(
        lammps=native,
        frozen_atoms=RegionConfig(indices=[1]),
        rateconstant=RateConstantConfig(
            style="htst",
            k0=1.0,
            free_radius=1.1,
            zone_radius=zone,
            premin=premin,
            force_tol=force_tol,
            nu0_min_THz=1e-4,
            nu0_max_THz=1e8,
        ),
    )
    authority = ResolvedConstraints.resolve(
        SOURCE, TYPES, config.frozen_atoms, IDS, cell=CELL, pbc=PBC
    )
    service = PrefactorService(
        config,
        object(),
        create_rate_constant(config.rateconstant),
        engine_physics=EnginePhysics.capture(native, SPECIES, MASSES),
        global_constraints=authority,
    )
    saddle, final = SOURCE.copy(), SOURCE.copy()
    saddle[2, 0], final[2, 0] = 3.0, 4.0
    request = service.build_request(
        event_key=("N04-stationarity-protocol",),
        min1_positions=SOURCE.copy(),
        saddle_positions=saddle,
        min2_positions=final,
        types=TYPES,
        cell=CELL.copy(),
        pbc=PBC,
        center_index=2,
    )
    calls = []
    scratch = FakeScratch(native, forces or {}, calls)
    extension = object.__new__(LammpsHTSTExtension)
    extension.engine = SimpleNamespace(config=native, comm=None, engine_id=0)

    def new_scratch():
        calls.append(("create",))
        return scratch

    def matrix(resource, positions, free, step):
        assert resource is scratch
        label = label_of(positions, scratch.types)
        assert tuple(free) == (scratch.types.index("Ni"),)
        calls.append(("hessian", label))
        # Mass-weighted units, intentionally independent of supplied forces.
        return np.diag([-1.0, 2.0, 3.0] if label == "saddle" else [4.0, 5.0, 6.0])

    extension._new_scratch = new_scratch
    extension._eskm_hessian = matrix
    return SimpleNamespace(
        service=service,
        request=request,
        extension=extension,
        scratch=scratch,
        calls=calls,
        authority=authority,
    )


def execute(case, *, backward=True):
    original = [
        getattr(case.request, key).copy()
        for key in ("min1_positions", "saddle_positions", "min2_positions")
    ]
    result = case.extension.compute_event_prefactors(
        case.request, compute_backward=backward
    )
    for key, expected in zip(
        ("min1_positions", "saddle_positions", "min2_positions"), original
    ):
        np.testing.assert_array_equal(getattr(case.request, key), expected)
    assert case.scratch.closed
    assert case.request.constraints == case.authority
    return result


def scientific_rejection(direction):
    assert direction.status == "rejected"
    assert direction.reason_code.value == "nonstationary_geometry"
    assert direction.nu0_hz is None
    assert "force" in direction.reason.lower()
    assert "0.005" in direction.reason


@pytest.mark.parametrize("bad_geometry", ["saddle", "min1", "min2"])
def test_vector_norm_rejection_is_directional_and_precedes_bad_hessian(bad_geometry):
    # Every component is below tolerance; the vector norm sqrt(2)*.004 is not.
    case = setup_case(forces={bad_geometry: (0.004, 0.004, 0.0)})
    result = execute(case)
    expected_labels = (
        ["saddle"] if bad_geometry == "saddle" else ["saddle", "min1", "min2"]
    )
    assert [c[1] for c in case.calls if c[0] == "force"] == expected_labels
    assert [c[1] for c in case.calls if c[0] == "hessian"] == [
        g for g in expected_labels if g != bad_geometry
    ]
    for direction, geometry in ((result.forward, "min1"), (result.backward, "min2")):
        if bad_geometry in ("saddle", geometry):
            scientific_rejection(direction)
        else:
            assert direction.ok


@pytest.mark.parametrize("vector", [(0.0, 0.0, 0.0), (TOL, 0.0, 0.0)])
def test_fixed_reactions_and_outer_forces_do_not_reject_inclusive_free_bound(vector):
    case = setup_case(forces={g: vector for g in ("saddle", "min1", "min2")})
    result = execute(case)
    assert result.forward.ok and result.backward.ok and result.n_free == 1
    assert len(case.scratch.observed) == 3
    for _, _, types, forces in case.scratch.observed:
        assert np.linalg.norm(forces[types.index("Cu")]) > 100
        assert np.linalg.norm(forces[types.index("Fe")]) > 50


@pytest.mark.parametrize("value", [float("nan"), float("inf")], ids=["nan", "infinity"])
def test_nonfinite_allowed_force_rejects_only_its_minimum(value):
    case = setup_case(forces={"min2": (value, 0.0, 0.0)})
    result = execute(case)
    assert result.forward.ok
    scientific_rejection(result.backward)
    assert [c[1] for c in case.calls if c[0] == "hessian"] == ["saddle", "min1"]


@pytest.mark.parametrize("premin", [False, True])
@pytest.mark.parametrize("zone", [None, 2.0], ids=["full", "crop"])
def test_force_checks_use_each_final_geometry_after_premin_and_crop(premin, zone):
    case = setup_case(premin=premin, zone=zone)
    result = execute(case)
    assert result.forward.ok and result.backward.ok
    assert [c for c in case.calls if c[0] in ("force", "hessian")] == [
        ("force", "saddle"),
        ("hessian", "saddle"),
        ("force", "min1"),
        ("hessian", "min1"),
        ("force", "min2"),
        ("hessian", "min2"),
    ]
    first_force = next(i for i, c in enumerate(case.calls) if c[0] == "force")
    assert all(
        i < first_force for i, c in enumerate(case.calls) if c[0] in ("premin", "build")
    )
    assert len([c for c in case.calls if c[0] == "premin"]) == (3 if premin else 0)
    for label, geometry, types, _ in case.scratch.observed:
        assert len(geometry) == (3 if zone else 4)
        assert types.index("Ni") == (1 if zone else 2)
        assert geometry[types.index("Fe"), 1] == pytest.approx(1.6 if premin else 1.5)
        assert (
            geometry[types.index("Ni"), 0]
            == {"min1": 2.0, "saddle": 3.0, "min2": 4.0}[label]
        )
        np.testing.assert_array_equal(geometry[types.index("Cu")], SOURCE[1])


def test_forward_only_never_checks_or_builds_unused_minimum_hessian():
    case = setup_case(forces={"min2": (float("nan"), 0.0, 0.0)})
    result = execute(case, backward=False)
    assert result.forward.ok and result.backward.skipped
    assert [c[1] for c in case.calls if c[0] == "force"] == ["saddle", "min1"]
    assert [c[1] for c in case.calls if c[0] == "hessian"] == ["saddle", "min1"]


@pytest.mark.parametrize(
    "bad",
    [0.0, -0.1, None, float("nan"), float("inf"), True],
    ids=["zero", "negative", "none", "nan", "infinity", "boolean"],
)
def test_force_tolerance_rejects_nonpositive_nonfinite_and_boolean_values(bad):
    require_force_api()
    with pytest.raises((ValueError, TypeError)):
        HTSTSettings(force_tol=bad)
    with pytest.raises((ValueError, TypeError)):
        RateConstantConfig(style="htst", force_tol=bad)


def test_default_force_tolerance_is_the_declared_absolute_value():
    require_force_api()
    assert HTSTSettings().force_tol == TOL
    assert RateConstantConfig(style="htst").force_tol == TOL


def test_nondefault_producing_tolerance_is_transported_hashed_and_enforced():
    low = setup_case(forces={"min1": (0.006, 0.0, 0.0)})
    high = setup_case(forces={"min1": (0.006, 0.0, 0.0)}, force_tol=0.007)
    assert high.service.settings.force_tol == 0.007
    assert high.request.settings.force_tol == 0.007
    assert dict(high.request.descriptor.numerical)["force_tol"] == 0.007
    assert low.request.descriptor.descriptor_id != high.request.descriptor.descriptor_id
    scientific_rejection(execute(low).forward)
    assert execute(high).forward.ok
    # Changing only settings must not silently reinterpret producing identity.
    changed = setup_case()
    changed.request = replace(
        changed.request, settings=replace(changed.request.settings, force_tol=0.007)
    )
    with pytest.raises(HTSTRequestError, match="settings.*descriptor"):
        changed.extension.compute_event_prefactors(changed.request)
    assert changed.calls == [], "mismatched producing tolerance reached scratch"
