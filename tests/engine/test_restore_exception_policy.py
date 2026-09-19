"""Pure-Python recovery probe: no LAMMPS instance or MPI communicator starts.

The fake command endpoint covers the two wrapper policies explicitly supported
by lammps_error_handler: plain Python errors leave the endpoint open; exported
LAMMPSException errors close it. The production recovery and decorator run.
"""

from types import SimpleNamespace

import numpy as np
import pytest

import pykmc.engine.lammps as engine_module
from pykmc.engine.lammps import FullSystem, LammpsEngine


class ExportedLAMMPSException(Exception):
    pass


class Endpoint:
    def __init__(self, exc_type):
        self.exc_type = exc_type
        self.natoms = 2
        self.fail = True
        self.closed = False

    def get_natoms(self):
        return self.natoms

    def command(self, command):
        if command == "clear":
            self.natoms = 0
        if command.startswith("pair_style ") and self.fail:
            self.fail = False
            raise self.exc_type("one-shot potential replay failure")

    def close(self):
        self.closed = True

    def extract_atom(self, name):
        assert name == "mass"
        return [0.0, 58.6934]


@pytest.mark.parametrize("close_on_failure", [False, True])
def test_restore_must_not_report_a_failed_closed_engine_intact(
    monkeypatch, close_on_failure
):
    monkeypatch.setattr(engine_module, "_LAMMPS_EXCEPTIONS", (ExportedLAMMPSException,))
    config = SimpleNamespace(
        pair_style="lj/cut 2.5",
        pair_coeff="* * 1 1",
        min_style="cg",
        frz_min="1e-8 1e-8 100 1000",
    )
    engine = LammpsEngine(config)
    endpoint = Endpoint(ExportedLAMMPSException if close_on_failure else RuntimeError)
    engine.lmp = endpoint
    descriptor = FullSystem(
        types=("Ni", "Ni"),
        species=("Ni",),
        masses=(58.6934,),
        cell=np.diag([10.0] * 3),
        pbc=(True,) * 3,
    )
    engine.full_system = descriptor
    monkeypatch.setattr(engine, "initialize_parameters", lambda: None)

    def initialize_system(**kwargs):
        endpoint.natoms = len(kwargs["types"])
        engine.full_system = descriptor

    monkeypatch.setattr(engine, "initialize_system", initialize_system)
    positions = np.array([[2.0, 2.0, 2.0], [3.3, 2.0, 2.0]])
    engine.command("clear")
    with pytest.raises(RuntimeError, match="one-shot potential replay failure"):
        engine.ensure_full_system(positions)
    assert engine.full_system is descriptor
    # A correct implementation may recreate the closed endpoint or report a
    # clear unrecoverable state. It must not declare restoration unnecessary.
    assert engine.system_is_cropped is True, {
        "endpoint_closed": endpoint.closed,
        "engine_lmp_is_none": engine.lmp is None,
        "dirty_marker": engine._cleared_since_init,
        "retry_result": engine.ensure_full_system(positions),
    }
    if not close_on_failure:
        assert engine.ensure_full_system(positions) is True
        assert engine.system_is_cropped is False
