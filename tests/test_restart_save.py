"""Tests for the atomic periodic restart save (KMC._save_restart_file)."""

from unittest.mock import Mock

import numpy as np
from ase.io import read

from pykmc import System
from pykmc.kmc import KMC


def _toy_system() -> System:
    return System(
        positions=np.array([[1.0, 0.0, 0.0], [2.0, 0.0, 0.0]], dtype=float),
        types=np.array(["Ni", "Ni"]),
        cell=np.diag([20.0, 20.0, 20.0]),
        pbc=np.array([True, True, True]),
        index=np.array([0, 1]),
    )


def test_restart_save_writes_loadable_files(tmp_path, monkeypatch):
    """restart_latest.npz/.xyz are written atomically and load back correctly."""
    monkeypatch.chdir(tmp_path)
    kmc = KMC(config=Mock())
    kmc.system = _toy_system()

    kmc._save_restart_file(42, 1.5e-7)

    data = np.load(tmp_path / "restart_latest.npz")
    assert int(data["last_step"]) == 42
    assert float(data["last_time"]) == 1.5e-7

    atoms = read(tmp_path / "restart_latest.xyz")
    assert len(atoms) == 2
    assert np.allclose(atoms.get_positions(), kmc.system.positions)

    # no tmp leftovers, no legacy file unless final=True
    leftovers = [p.name for p in tmp_path.iterdir() if ".tmp." in p.name]
    assert leftovers == []
    assert not (tmp_path / "restart_42.npz").exists()


def test_restart_save_final_writes_legacy_npz(tmp_path, monkeypatch):
    """final=True also writes the legacy end-of-run restart_<step>.npz."""
    monkeypatch.chdir(tmp_path)
    kmc = KMC(config=Mock())
    kmc.system = _toy_system()

    kmc._save_restart_file(7, 3.0e-9, final=True)

    assert (tmp_path / "restart_latest.npz").exists()
    legacy = np.load(tmp_path / "restart_7.npz")
    assert int(legacy["last_step"]) == 7


def test_restart_save_overwrites_previous_latest(tmp_path, monkeypatch):
    """A later interval replaces restart_latest atomically."""
    monkeypatch.chdir(tmp_path)
    kmc = KMC(config=Mock())
    kmc.system = _toy_system()

    kmc._save_restart_file(10, 1.0e-9)
    kmc.system.positions[0, 0] = 5.0
    kmc._save_restart_file(20, 2.0e-9)

    data = np.load(tmp_path / "restart_latest.npz")
    assert int(data["last_step"]) == 20
    atoms = read(tmp_path / "restart_latest.xyz")
    assert np.isclose(atoms.get_positions()[0, 0], 5.0)
