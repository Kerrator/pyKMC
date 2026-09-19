"""Public restore/retry protocol; fake native endpoint, real recovery and physics.

No native engine or communicator starts. The native constructor and geometry
replay are replaced, while start/close, ensure_full_system, potential capture,
mass refresh, and search/refine exception policy remain production methods.
"""

from dataclasses import replace
from types import SimpleNamespace

import numpy as np
import pytest

import pykmc.engine.lammps as engine_module
from pykmc.engine.lammps import FullSystem, LammpsEngine
from pykmc.physics import EnginePhysics


POSITIONS = np.array([[2.0, 2.0, 2.0], [3.3, 2.0, 2.0]])
CELL = np.diag([10.0, 11.0, 12.0])
SPECIES = ("Fe", "Ni")
MASSES = (56.0, 60.0)


class ExportedLAMMPSException(Exception):
    pass


class InitiatingFailure(RuntimeError):
    pass


class Endpoint:
    def __init__(self, harness, natoms=0):
        self.harness = harness
        self.natoms = natoms
        self.closed = False

    def get_natoms(self):
        return self.natoms

    def command(self, command):
        self.harness.trace.append(("command", command))
        if command == "clear":
            self.harness.fail_once("clear")
            self.natoms = 0
        if command.startswith("pair_style "):
            self.harness.fail_once("potential")

    def extract_atom(self, name):
        assert name == "mass"
        return [0.0, *self.harness.live_masses]

    def close(self):
        self.closed = True
        self.harness.trace.append(("close",))


class Harness:
    def __init__(self, monkeypatch, config=None):
        self.config = config or SimpleNamespace(
            pair_style="lj/cut 2.5",
            pair_coeff="* * 1.0 1.0",
            min_style="cg",
            frz_min="1e-8 1e-8 100 1000",
            verbosity=0,
        )
        self.trace = []
        self.failures = {}
        self.live_masses = MASSES
        self.engine = LammpsEngine(self.config, comm=None)
        self.original = FullSystem(
            types=("Ni", "Ni"),
            species=SPECIES,
            masses=MASSES,
            cell=CELL.copy(),
            pbc=(True, True, False),
            physics=EnginePhysics.capture(self.config, SPECIES, MASSES),
        )
        self.engine.full_system = self.original
        self.engine.lmp = Endpoint(self, natoms=2)
        monkeypatch.setattr(engine_module, "lammps", self.factory)
        monkeypatch.setattr(
            engine_module, "_LAMMPS_EXCEPTIONS", (ExportedLAMMPSException,)
        )
        monkeypatch.setattr(self.engine, "initialize_parameters", self.parameters)
        monkeypatch.setattr(self.engine, "initialize_system", self.system)

    def fail_once(self, stage):
        failure = self.failures.pop(stage, None)
        if failure is not None:
            raise failure

    def factory(self, **kwargs):
        self.trace.append(("start",))
        self.fail_once("start")
        return Endpoint(self)

    def parameters(self):
        self.trace.append(("parameters",))
        self.fail_once("parameters")

    def system(self, **kwargs):
        self.trace.append(("system",))
        assert tuple(kwargs["types"]) == self.original.types
        assert tuple(kwargs["species"]) == SPECIES
        assert tuple(kwargs["masses"]) == MASSES
        assert tuple(kwargs["pbc"]) == self.original.pbc
        np.testing.assert_array_equal(kwargs["cell"], CELL)
        np.testing.assert_array_equal(kwargs["positions"], POSITIONS)
        self.engine.lmp.natoms = len(kwargs["types"])
        # Actual initialize_system replaces the old descriptor before potential
        # replay. A following failure must put the original snapshot back.
        self.engine.full_system = replace(self.original, physics=None)
        self.fail_once("system")

    def mark_pending(self, *, closed=False):
        self.engine.command("clear")
        assert self.engine.system_is_cropped
        if closed:
            self.engine.close()
            assert self.engine.lmp is None

    def assert_pending(self):
        assert self.engine.system_is_cropped is True
        assert self.engine.full_system is self.original
        assert self.engine.full_system.physics is self.original.physics

    def assert_restored(self):
        actual = self.engine.full_system
        assert self.engine.system_is_cropped is False
        assert actual.types == self.original.types
        assert actual.species == self.original.species
        assert actual.masses == self.original.masses
        assert actual.pbc == self.original.pbc
        np.testing.assert_array_equal(actual.cell, self.original.cell)
        assert actual.physics == self.original.physics
        assert self.engine.lmp.get_natoms() == 2
        assert not self.engine.lmp.closed


def test_closed_pending_engine_restarts_then_is_an_intact_noop(monkeypatch):
    h = Harness(monkeypatch)
    h.mark_pending(closed=True)
    h.assert_pending()
    assert h.engine.ensure_full_system(POSITIONS) is True
    h.assert_restored()
    assert h.trace.count(("start",)) == 1
    before = list(h.trace)
    assert h.engine.ensure_full_system(POSITIONS + 0.1) is False
    assert h.trace == before


@pytest.mark.parametrize(
    "stage", ["start", "clear", "parameters", "system", "potential"]
)
def test_each_failed_replay_stage_retains_descriptor_and_later_retries(
    monkeypatch, stage
):
    h = Harness(monkeypatch)
    h.mark_pending(closed=stage == "start")
    h.failures[stage] = RuntimeError(f"one-shot {stage} failure")
    with pytest.raises(RuntimeError, match=f"one-shot {stage} failure"):
        h.engine.ensure_full_system(POSITIONS)
    h.assert_pending()
    assert h.engine.ensure_full_system(POSITIONS) is True
    h.assert_restored()


def test_exported_exception_closes_handle_but_next_public_retry_rebuilds(monkeypatch):
    h = Harness(monkeypatch)
    h.mark_pending()
    failed_handle = h.engine.lmp
    h.failures["potential"] = ExportedLAMMPSException("one-shot exported failure")
    with pytest.raises(RuntimeError, match="one-shot exported failure"):
        h.engine.ensure_full_system(POSITIONS)
    assert failed_handle.closed and h.engine.lmp is None
    h.assert_pending()
    assert h.engine.ensure_full_system(POSITIONS) is True
    assert h.engine.lmp is not failed_handle
    h.assert_restored()
    assert h.trace.count(("start",)) == 1


def test_explicit_close_of_intact_engine_does_not_start_a_new_one(monkeypatch):
    h = Harness(monkeypatch)
    h.engine.close()
    before = list(h.trace)
    assert h.engine.system_is_cropped is False
    assert h.engine.ensure_full_system(POSITIONS) is False
    assert h.engine.lmp is None and h.trace == before


@pytest.mark.parametrize("operation", ["partn_search", "partn_refine"])
def test_initiating_failure_survives_failed_restore_then_public_retry_works(
    monkeypatch, operation
):
    h = Harness(monkeypatch)
    initiating = InitiatingFailure(f"original {operation} failure")

    def fail_operation(*args, **kwargs):
        h.mark_pending(closed=True)
        raise initiating

    monkeypatch.setattr(h.engine, "_" + operation + "_impl", fail_operation)
    h.failures["start"] = RuntimeError("secondary restart failure")
    config = SimpleNamespace(
        control=SimpleNamespace(active_volume=True),
        activevolume=SimpleNamespace(rmov=0.5, ract=5.0),
        frozen_atoms=None,
    )
    with pytest.warns(
        RuntimeWarning, match="restoring the full system afterwards failed"
    ) as warnings:
        with pytest.raises(InitiatingFailure) as raised:
            getattr(h.engine, operation)(
                config, 0, positions=POSITIONS, cell=CELL, types=h.original.types
            )
    assert raised.value is initiating
    assert any("secondary restart failure" in str(w.message) for w in warnings)
    h.assert_pending()
    assert h.engine.ensure_full_system(POSITIONS) is True
    h.assert_restored()


@pytest.mark.parametrize(
    "change", ["numeric_coefficients", "potential_bytes", "live_masses"]
)
def test_retry_cannot_relabel_a_different_physical_system(
    monkeypatch, tmp_path, change
):
    config = None
    if change == "potential_bytes":
        potential = tmp_path / "same_path.eam"
        potential.write_bytes(b"original force-model bytes\n")
        config = SimpleNamespace(
            pair_style="eam/alloy",
            pair_coeff=f"* * {potential} Fe Ni",
            min_style="cg",
            frz_min="1e-8 1e-8 100 1000",
            verbosity=0,
        )
    h = Harness(monkeypatch, config=config)
    h.mark_pending(closed=True)
    if change == "numeric_coefficients":
        h.config.pair_coeff = "* * 2.0 1.0"
    elif change == "potential_bytes":
        potential.write_bytes(b"changed force-model bytes\n")
    else:
        h.live_masses = (56.0, 61.0)
    # Repeated calls must report the mismatch; neither may claim intact/no-op.
    for _ in range(2):
        with pytest.raises(
            (ValueError, RuntimeError),
            match=r"(?i)(physic|force|mass|descriptor|potential|coefficient)",
        ):
            h.engine.ensure_full_system(POSITIONS)
        h.assert_pending()
    if change == "numeric_coefficients":
        h.config.pair_coeff = "* * 1.0 1.0"
    elif change == "potential_bytes":
        potential.write_bytes(b"original force-model bytes\n")
    else:
        h.live_masses = MASSES
    assert h.engine.ensure_full_system(POSITIONS) is True
    h.assert_restored()


def test_equal_opaque_force_snapshot_does_not_preclude_fresh_restoration(monkeypatch):
    config = SimpleNamespace(
        pair_style="custom/opaque",
        pair_coeff="* * token",
        min_style="cg",
        frz_min="1e-8 1e-8 100 1000",
        verbosity=0,
    )
    h = Harness(monkeypatch, config=config)
    assert h.original.physics.force_model.reusable is False
    h.mark_pending(closed=True)
    assert h.engine.ensure_full_system(POSITIONS) is True
    h.assert_restored()
