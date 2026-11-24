"""Controlled (no native engine, registration, MPI or potential) refinement fixture."""

from concurrent.futures import Future
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np
import pandas as pd
import pytest

import pykmc.engine.lammps as lmp_module
from pykmc import Config, System
from pykmc.config import ActiveVolume
from pykmc.engine.lammps import LammpsEngine
from pykmc.event_table import ActiveEventTable
from pykmc.kmc import KMC
from pykmc.psr import PointSetRegistration, PSROutput
from pykmc.result import Ok

CELL = np.eye(3) * 10
MIN = np.array([[5, 5, 5], [1, 2, 2], [6, 6, 6], [0.25, 2, 2], [9.5, 2, 2]])
# Basin source is not the ordinary incoming system: only absolute id 0
# changes, well outside the neighbour shell and active-volume crop.
SAVED_MIN = MIN.copy()
SAVED_MIN[0, 2] += 1.25
TYPES = np.array(["Ni", "Fe", "Cr", "Ni", "Fe"])
ORDER = np.array([4, 3, 1])
BULK, BASIN, CROP, DE, PREF = -18.0, -36.0, -7.25, 0.75, 2.5
# Expected geometry in ORDER, independently tabulated (never call production transforms).
RAW_X = np.array([[-0.4, -0.25, 0.8], [-0.3, 0.75, 0.9]])
FINAL_X = np.array([[-0.3, -0.75, 0.6], [-0.1, 1.25, 0.8]])


def coords(x, *, wrapped=False, y=2.0):
    return np.column_stack((np.mod(x, 10) if wrapped else x, np.full(3, y), [2] * 3))


def catalogue():
    initial = np.array([[0.0, 0, 0], [-0.75, 0, 0], [0.75, 0, 0]])
    disp = np.zeros_like(initial)
    disp[:, 0] = [-0.5, 0.1, -0.2]
    return pd.DataFrame(
        [
            dict(
                event_id="env",
                idx_ref=37,
                initial_positions=initial,
                types=["Ni", "Fe", "Fe"],
                move_atom_idx=0,
                saddle_positions=initial + disp,
                final_positions=initial + 2 * disp,
                sym_matrix=[np.eye(3), np.diag([-1, 1, 1])],
                sym_perm=[[0, 1, 2], [0, 2, 1]],
                energy_barrier=DE,
                k=1.0,
                k_pref=PREF,
            )
        ]
    )


def ready(value):
    f = Future()
    f.set_result(value)
    return f


class CheckedFuture(Future):
    def result(self):
        assert self.done(), "unexpected submission count/order in deterministic batch"
        return super().result()


class ScriptedWorker:
    """Run real Python engine preparation/retry/reassembly with only native I/O scripted."""

    def __init__(self, cfg, monkeypatch):
        self.cfg = cfg
        self.full_min = BULK
        self.calls = []
        self.min_calls = []
        self.artns = []
        self.pending = []
        self.batch = 1
        self.completed = []
        self.trace = [(0, 0.125)]  # (Fortran error flag, saddle displacement)
        self.exception = None
        self.engine = LammpsEngine(cfg.lammps)
        self.engine._is_orthorhombic = True
        self.engine.lmp = Mock()
        self.engine.lmp.scatter_atoms.side_effect = self.scatter
        self.engine.lmp.extract_compute.return_value = CROP
        monkeypatch.setattr(lmp_module.pypARTn, "artn", self.artn)

    def scatter(self, field, typ, n, data):
        assert (field, typ, n) == ("x", 1, 3)
        self.at_saddle = np.array(data).reshape(-1, 3)

    def artn(self, engine):
        assert engine == "lmp"
        art = Mock()
        art.lib._name = "scripted"
        self.artns.append(art)
        attempt = -1

        def error():
            nonlocal attempt
            attempt = min(attempt + 1, len(self.trace) - 1)
            return (self.trace[attempt][0], "scripted")

        art.get_error.side_effect = error

        def extract(key):
            if key == "delr_sad":
                return self.trace[attempt][1]
            if key == "etot_sad":
                return (CROP if self.cfg.control.active_volume else self.full_min) + DE
            if key == "tau_sad":
                xyz = self.at_saddle.copy()
                xyz[:, 0] %= 10
                xyz[:, 1] += (
                    0.02  # distinguish returned saddle from the submitted guess
                )
                return xyz
            raise AssertionError(key)

        art.extract.side_effect = extract
        return art

    def get_total_energy(self, **kwargs):
        return self.submit("get_total_energy", **kwargs)

    def partn_refine(self, **kwargs):
        return self.submit("partn_refine", **kwargs)

    def group_minimize_with_results(self, **kwargs):
        return self.submit_group("minimize_with_results", **kwargs)

    def submit_group(self, op_name, **kwargs):
        if op_name != "minimize_with_results":
            raise AssertionError(f"unexpected group operation {op_name}")
        return self.group_minimizer(**kwargs)

    def submit(self, op_name, **kwargs):
        if op_name == "get_total_energy":
            positions = kwargs["positions"]
            self.min_calls.append(positions.copy())
            for known, energy in ((SAVED_MIN, BASIN), (MIN, BULK)):
                if np.allclose(positions, known, atol=1e-12, rtol=0):
                    return ready(energy)
            raise AssertionError("no controlled energy for this geometry")
        if op_name != "partn_refine":
            raise AssertionError(f"unexpected local operation {op_name}")
        idx = len(self.calls)
        self.calls.append(kwargs)
        f = CheckedFuture()
        # Reverse result readiness without relying on arrival times or OS scheduling.
        self.pending.append((idx, f, kwargs))
        if len(self.pending) == self.batch:
            for j, task, kw in reversed(self.pending):
                if self.exception is None:
                    task.set_result(self.engine.partn_refine(**kw))
                else:
                    task.set_exception(self.exception)
                self.completed.append(j)
            self.pending.clear()
        return f


@pytest.fixture
def make_case(monkeypatch):
    def build(crop=False):
        cfg = Config.from_ini_file(str(Path(__file__).parents[1] / "data" / "input.in"))
        cfg.control.recycle = False
        cfg.control.active_volume = crop
        cfg.control.refine_thr = 1.0
        cfg.activevolume = ActiveVolume(ract=1.5, rmov=0.5)
        cfg.eventsearch.refined_energy_thr = 0.125
        cfg.partn.r_delr_sad_thr = 0.25
        cfg.partn.r_max_attempts = 2
        cfg.partn.zseed = 193
        cfg.rateconstant.k0 = 11.0
        cfg.rateconstant.T = 900.0
        sim = KMC(cfg)
        sim.system = System(TYPES.copy(), MIN.copy(), CELL.copy(), True, np.arange(5))
        sim.neighbors_list = Mock()
        sim.neighbors_list.get_neighbors.return_value = ORDER.copy()
        sim.atomic_environment = Mock()
        sim.atomic_environment.get_atoms_with_id.return_value = [3]
        sim.manager = ScriptedWorker(cfg, monkeypatch)
        sim.reference_table = SimpleNamespace(table=catalogue())
        sim.loggers = Mock()
        sim.total_energy = BULK
        sim.active_table = ActiveEventTable(cfg)
        match = Mock()
        match.match_and_check.return_value = Ok(
            PSROutput(0.0, np.eye(3), np.array([0.25, 2.0, 2.0]), np.array([1, 0, 2]))
        )
        monkeypatch.setattr(PointSetRegistration, "create", lambda *a, **kw: match)
        return sim, match.match_and_check

    return build
