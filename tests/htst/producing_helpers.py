"""Shared recording scratch and request fixtures for producing contracts."""

import importlib
import importlib.util
from dataclasses import replace
from types import SimpleNamespace

import numpy as np
from pykmc.config import RateConstantConfig, RegionConfig
from pykmc.engine.htst_lammps import LammpsHTSTExtension as LammpsHTSTExtension
from pykmc.htst.result import EventPrefactors
from pykmc.physics import EnginePhysics, ResolvedConstraints
from pykmc.rate_constant import create_rate_constant
from pykmc.rate_constant.prefactors import PrefactorService

SPECIES = ("Ni", "Fe", "Cu", "O")

MASSES = (10.0, 20.0, 30.0, 40.0)

TYPES = ("O", "Cu", "Ni", "Fe")

IDS = (8, 91, 42, 17)

SOURCE = np.array([[9.0, 0.0, 0.0], [3.0, 0.25, 0.0], [2.0, 0.0, 0.0], [3.0, 1.5, 0.0]])

CELL = np.eye(3) * 20.0

PBC = (True, False, True)

GEOMETRIES = ("min1_positions", "saddle_positions", "min2_positions")


def require_api():
    # Called inside each test, so the unchanged baseline reports a clear
    # contract assertion failure rather than an import/collection/setup error.
    name = "pykmc.htst.provenance"
    assert importlib.util.find_spec(name) is not None, (
        "Producing-record API is required"
    )
    api = importlib.import_module(name)
    assert callable(getattr(getattr(api, "RequestSnapshot", None), "capture", None))
    assert callable(
        getattr(getattr(api, "CalculationProvenance", None), "capture", None)
    )
    assert "provenance" in EventPrefactors.__dataclass_fields__
    assert callable(getattr(EventPrefactors, "calculation", None))
    return api


def request_case(*, premin=False, zone=None):
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
        event_key=("producing-contract", 17),
        min1_positions=SOURCE.copy(),
        saddle_positions=saddle,
        min2_positions=final,
        types=TYPES,
        cell=CELL.copy(),
        pbc=PBC,
        center_index=2,
    )
    return request, native


def produced_request(request):
    geometry = {}
    for key in GEOMETRIES:
        value = getattr(request, key).copy()
        value[[0, 3], 1] += 0.1
        geometry[key] = value
    return replace(request, **geometry)


def assert_geometry(actual, expected):
    for key in GEOMETRIES:
        np.testing.assert_array_equal(getattr(actual, key), getattr(expected, key))
    np.testing.assert_array_equal(actual.cell, expected.cell)
    for key in ("types", "species", "masses", "pbc", "center_index", "settings"):
        assert getattr(actual, key) == getattr(expected, key)
    assert actual.descriptor == expected.descriptor
    assert actual.constraints == expected.constraints
    assert actual.user_constraints == expected.user_constraints


def capture(api, source, produced=None, **changes):
    kwargs = dict(method="protocol-matrix", free_indices=(2,), zone_indices=(1, 2, 3))
    kwargs.update(changes)
    return api.CalculationProvenance.capture(source, produced or source, **kwargs)


def matrix(positions, free):
    assert tuple(free) == (2,)
    x = positions[2, 0]
    return np.diag(
        [-1.0, 2.0, 3.0]
        if x == 3.0
        else [4.0, 5.0, 6.0]
        if x == 2.0
        else [9.0, 10.0, 11.0]
    )


class Scratch:
    """Protocol endpoint only; no LAMMPS construction or native execution."""

    def __init__(self, native):
        self.config = native
        self.types = ()
        self.positions = None
        self.closed = False
        self.builds = []
        self.lmp = SimpleNamespace(command=self.command)

    def command(self, command):
        assert command == "clear"

    def start(self):
        pass

    def initialize_parameters(self):
        pass

    def initialize_system(self, *, types, positions, cell, pbc, species, masses):
        self.types = tuple(types)
        self.positions = np.array(positions, copy=True)
        self.full_system = SimpleNamespace(species=species, masses=masses, physics=None)
        self.builds.append(tuple(types))

    def initialize_potential(self):
        self.full_system.physics = EnginePhysics.capture(
            self.config, self.full_system.species, self.full_system.masses
        )

    def set_positions(self, positions):
        self.positions = np.array(positions, copy=True)

    def get_positions(self):
        return self.positions.copy()

    def minimize_freeze_core(self, core):
        assert tuple(core) == (1, 2)
        self.positions[[0, 3], 1] += 0.1

    def get_forces(self, *, positions):
        return np.zeros_like(positions)

    def get_total_energy(self, positions=None, *, recompute=True):
        if positions is not None:
            self.set_positions(positions)
        x = self.positions[self.types.index("Ni"), 0]
        return {2.0: 0.0, 3.0: 1.0, 4.0: 0.25}[float(x)]

    def close(self):
        self.closed = True
