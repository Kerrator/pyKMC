"""Mocked tests for the active-volume full-system restore in the pARTn operations."""

from types import SimpleNamespace

import numpy as np

from pykmc.enginemanager.lmpi import lammps_operations as ops
from pykmc.result import Err, ErrorType, Ok


class _FakeComm:
    def bcast(self, value, root=0):
        return value


class _FakeLammps:
    def __init__(self, natoms):
        self._natoms = natoms

    def get_natoms(self):
        return self._natoms


class _FakeEngine:
    def __init__(self, natoms):
        self.rank = 0
        self.engine_id = 0
        self.local_engine_comm = _FakeComm()
        self.lmp = _FakeLammps(natoms)
        self.commands = []

    def command(self, cmd):
        self.commands.append(cmd)


class _SearchArtn:
    lib = SimpleNamespace(_name="libartn-lmp.so")

    def reset_input(self):
        pass

    def set(self, *_args):
        pass

    def get_error(self):
        return [0]

    def extract(self, key):
        values = {
            "delr_min1": 0.1,
            "delr_min2": 1.0,
            "etot_sad": 1.5,
            "etot_min1": 1.0,
            "etot_min2": 0.8,
        }
        return values[key]


class _RefineArtnSuccess:
    lib = SimpleNamespace(_name="libartn-lmp.so")

    def reset_input(self):
        pass

    def set(self, *_args):
        pass

    def get_error(self):
        return [0]

    def extract(self, key):
        values = {
            "delr_sad": 0.1,
            "etot_sad": 1.5,
            "tau_sad": np.array([[0.2, 0.2, 0.2], [0.6, 0.6, 0.6]]),
        }
        return values[key]


class _RefineArtnFailure:
    lib = SimpleNamespace(_name="libartn-lmp.so")

    def reset_input(self):
        pass

    def set(self, *_args):
        pass

    def get_error(self):
        return [77, "failed"]

    def extract(self, key):
        if key == "delr_sad":
            return 1.0
        raise KeyError(key)


def _config():
    return SimpleNamespace(
        control=SimpleNamespace(active_volume=True),
        frozen_atoms=None,
        eventsearch=SimpleNamespace(delr_thr=0.5),
        partn=SimpleNamespace(
            path_artnso="libartn.so",
            verbosity=0,
            dmax=0.2,
            delr_thr=0.5,
            zseed=7,
            nevalf_max=1000,
            r_nevalf_max=1000,
            push_mode="rad",
            push_dist_thr=0.1,
            push_step_size=0.1,
            ninit=1,
            lanczos_min_size=1,
            lanczos_max_size=2,
            lanczos_disp=0.1,
            lanczos_eval_conv_thr=1e-3,
            eigval_thr=-1.0,
            eigen_step_size=0.1,
            nsmooth=1,
            neigen=1,
            alpha_mix_cr=0.1,
            nnewchance=1,
            nperp=None,
            nperp_limitation=None,
            forc_thr=0.1,
            push_over=0.1,
            r_push_mode="rad",
            r_push_dist_thr=0.1,
            r_push_step_size=0.1,
            r_ninit=1,
            r_lanczos_min_size=1,
            r_lanczos_max_size=2,
            r_lanczos_disp=0.1,
            r_lanczos_eval_conv_thr=1e-3,
            r_eigval_thr=-1.0,
            r_eigen_step_size=0.1,
            r_nsmooth=1,
            r_neigen=1,
            r_alpha_mix_cr=0.1,
            r_nnewchance=1,
            r_nperp=None,
            r_nperp_limitation=None,
            r_forc_thr=0.1,
            r_max_attempts=1,
            r_dmax=0.2,
            r_delr_sad_thr=0.5,
        ),
        lammps=SimpleNamespace(
            boundary="p p p",
            pair_style="eam/alloy",
            pair_coeff="* * dummy.eam Ni",
            min_style="cg",
            minimize="1e-6 1e-8 10 10",
        ),
        atomicenvironment=SimpleNamespace(rcut=6.5),
        activevolume=SimpleNamespace(AV_debug=False),
    )


def _patch_stdout_redirect(monkeypatch):
    monkeypatch.setattr(ops.os, "dup", lambda _fd: 10)
    monkeypatch.setattr(ops.os, "open", lambda _path, _flags: 11)
    monkeypatch.setattr(ops.os, "dup2", lambda _src, _dst: None)
    monkeypatch.setattr(ops.os, "close", lambda _fd: None)


def test_ensure_full_system_reinitializes_after_atom_count_mismatch(monkeypatch):
    captured = {}

    def fake_reinitialize(engine, config, system):
        captured["engine"] = engine
        captured["config"] = config
        captured["system"] = system

    engine = _FakeEngine(natoms=2)
    config = SimpleNamespace(lammps=SimpleNamespace(boundary="p p p"))
    positions = np.array([[0.0, 0.0, 0.0], [0.4, 0.4, 0.4], [0.8, 0.8, 0.8]])
    cell = np.eye(3)
    types = np.array(["Ni", "Ni", "Ni"])

    monkeypatch.setattr(ops, "reinitialize_system", fake_reinitialize)

    ops._ensure_full_system(engine, config, positions, cell, types)

    system = captured["system"]
    assert np.array_equal(system.positions, positions)
    assert np.array_equal(system.cell, cell)
    assert np.array_equal(system.types, types)
    assert np.array_equal(system.pbc, np.array([True, True, True]))
    assert np.array_equal(system.index, np.arange(len(positions)))


def test_partn_search_restores_full_system_after_active_volume(monkeypatch):
    positions = np.array([[0.0, 0.0, 0.0], [0.4, 0.4, 0.4], [0.8, 0.8, 0.8]])
    cell = np.eye(3)
    types = np.array(["Ni", "Ni", "Ni"])
    engine = _FakeEngine(natoms=2)
    config = _config()
    restore_calls = []

    _patch_stdout_redirect(monkeypatch)
    monkeypatch.setattr(ops, "partn_search_AV", lambda *_args: (np.array([0, 2]), np.array([1])))
    monkeypatch.setattr(
        ops,
        "position_results_AV",
        lambda *_args: (positions.copy(), positions.copy(), positions.copy(), 0),
    )
    monkeypatch.setattr(ops.pypARTn, "artn", lambda engine="lmp": _SearchArtn())
    monkeypatch.setattr(
        ops,
        "_ensure_full_system",
        lambda engine, config, positions, cell, types: restore_calls.append(
            (positions.copy(), cell.copy(), np.array(types, copy=True))
        ),
    )

    result = ops.partn_search(
        engine,
        config,
        central_atom_idx=0,
        positions=positions,
        cell=cell,
        types=types,
    )

    assert isinstance(result, Ok)
    assert len(restore_calls) == 1
    assert np.array_equal(restore_calls[0][0], positions)


def test_partn_refine_restores_full_system_after_successful_active_volume(monkeypatch):
    positions = np.array([[0.0, 0.0, 0.0], [0.4, 0.4, 0.4], [0.8, 0.8, 0.8]])
    cell = np.eye(3)
    types = np.array(["Ni", "Ni", "Ni"])
    engine = _FakeEngine(natoms=2)
    config = _config()
    restore_calls = []

    monkeypatch.setattr(
        ops,
        "partn_refine_AV",
        lambda *_args: (0.3, np.array([0, 2]), np.array([1])),
    )
    monkeypatch.setattr(ops.pypARTn, "artn", lambda engine="lmp": _RefineArtnSuccess())
    monkeypatch.setattr(
        ops,
        "_ensure_full_system",
        lambda engine, config, positions, cell, types: restore_calls.append(
            (positions.copy(), cell.copy(), np.array(types, copy=True))
        ),
    )

    result = ops.partn_refine(
        engine,
        config,
        central_atom_idx=0,
        positions=positions,
        cell=cell,
        types=types,
        saddle_idx=np.array([0, 2]),
        saddle_positions=positions[[0, 2]],
    )

    assert isinstance(result, Ok)
    assert result.ok_value().E_saddle == 1.2
    assert len(restore_calls) == 1
    assert np.array_equal(result.ok_value().saddle_positions[1], positions[1])


def test_partn_refine_restores_full_system_after_failed_active_volume(monkeypatch):
    positions = np.array([[0.0, 0.0, 0.0], [0.4, 0.4, 0.4], [0.8, 0.8, 0.8]])
    cell = np.eye(3)
    types = np.array(["Ni", "Ni", "Ni"])
    engine = _FakeEngine(natoms=2)
    config = _config()
    restore_calls = []

    monkeypatch.setattr(
        ops,
        "partn_refine_AV",
        lambda *_args: (0.3, np.array([0, 2]), np.array([1])),
    )
    monkeypatch.setattr(ops.pypARTn, "artn", lambda engine="lmp": _RefineArtnFailure())
    monkeypatch.setattr(
        ops,
        "_ensure_full_system",
        lambda engine, config, positions, cell, types: restore_calls.append(
            (positions.copy(), cell.copy(), np.array(types, copy=True))
        ),
    )

    result = ops.partn_refine(
        engine,
        config,
        central_atom_idx=0,
        positions=positions,
        cell=cell,
        types=types,
        saddle_idx=np.array([0, 2]),
        saddle_positions=positions[[0, 2]],
    )

    assert isinstance(result, Err)
    assert result.err_value().type == ErrorType.EVENT_NOT_FOUND
    assert len(restore_calls) == 1
