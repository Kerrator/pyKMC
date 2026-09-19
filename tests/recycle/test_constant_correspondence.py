"""Explicit constant crop identities must not become legacy positional data.

Real active-table and KMC mapping code; Reconstruction is a recording boundary.
No native minimization, event eligibility or HTST calculation is claimed.
"""

from pathlib import Path

import numpy as np
import pytest

import pykmc
from pykmc.config import Config, RateConstantConfig
from pykmc.event_table import ActiveEventTable
from pykmc.kmc import KMC
from pykmc.neighbors_list import NeighborsList
from pykmc.result import EventRefinementOutput, ErrorType, Ok
from pykmc.system import System


CONSTANT_COLUMNS = {
    "atom_index",
    "saddle_positions",
    "final_positions",
    "energy_barrier",
    "k",
    "num_reference_event",
    "refined",
}


def setup_case(monkeypatch, crop_ids, saddle, final):
    root = Path(pykmc.__file__).resolve().parent.parent
    cfg = Config.from_ini_file(str(root / "tests/data/input.in"))
    cfg.control.active_volume = False
    cfg.control.reference_table = None
    cfg.control.seed = None
    cfg.frozen_atoms = None
    cfg.rateconstant = RateConstantConfig(style="constant", k0=7.0)
    cfg.atomicenvironment.rnei = 2.0
    cfg.atomicenvironment.rcut = 2.0
    source = np.array([[10.0, 10.0, 10.0], [11.0, 10.0, 10.0], [20.0, 10.0, 10.0]])
    system = System(
        types=np.array(["Si"] * 3),
        positions=source.copy(),
        cell=np.eye(3) * 100,
        pbc=(False, False, False),
        index=np.array([42, 8, 91]),
    )
    neighbors = NeighborsList(system, rnei=2.0, rcut=2.0)
    assert neighbors.get_neighbors("rcut", 0) == [0, 1]
    table = ActiveEventTable(cfg)
    table.add_events(
        EventRefinementOutput(
            central_atom_index=0,
            saddle_positions=np.array(saddle, dtype=float),
            E_saddle=0.1,
            min2_positions=np.array(final, dtype=float),
            dE_forward=0.1,
            num_reference_event=47,
            refined="F",
            crop_atom_ids=crop_ids,
        )
    )
    assert not table.uses_prefactors
    assert set(table.table.columns) == CONSTANT_COLUMNS
    calls = {"constructor": [], "reconstruct": []}

    class RecordingReconstruction:
        def __init__(self, config, manager, **kwargs):
            calls["constructor"].append(kwargs)

        def reconstruct(self, initial, final, working, cell, threshold, indices):
            calls["reconstruct"].append(
                {
                    "initial": np.array(initial, copy=True),
                    "final": np.array(final, copy=True),
                    "working": np.array(working, copy=True),
                    "indices": np.array(indices, copy=True),
                }
            )
            return Ok("recorded-reconstruction-boundary")

    monkeypatch.setattr("pykmc.kmc.Reconstruction", RecordingReconstruction)
    sim = KMC(cfg, manager=object())
    sim.system, sim.neighbors_list = system, neighbors
    return sim, table, source, calls


@pytest.mark.parametrize(
    "crop_ids", [(999,), (42, 42)], ids=["unknown-id", "duplicate-id"]
)
def test_invalid_explicit_constant_ids_reject_before_reconstruction(
    monkeypatch, crop_ids
):
    saddle = [[10.2, 10.0, 10.0]] * len(crop_ids)
    final = [[10.5, 10.0, 10.0]] * len(crop_ids)
    sim, table, source, calls = setup_case(monkeypatch, crop_ids, saddle, final)
    result = sim._reconstruction_active_event(0, table)
    assert not result.is_ok(), (
        "Explicit invalid crop IDs must never use legacy neighbor order"
    )
    assert result.err_value().type == ErrorType.RECONSTRUCTION_INVALID_EVENT_DATA
    assert calls == {"constructor": [], "reconstruct": []}
    np.testing.assert_array_equal(sim.system.positions, source)
    np.testing.assert_array_equal(table.table.iloc[0].saddle_positions, saddle)
    assert set(table.table.columns) == CONSTANT_COLUMNS


def test_valid_reordered_constant_ids_scatter_by_identity(monkeypatch):
    saddle = [[11.4, 10.0, 10.0], [10.2, 10.0, 10.0]]
    final = [[11.7, 10.0, 10.0], [10.5, 10.0, 10.0]]
    sim, table, source, calls = setup_case(monkeypatch, (8, 42), saddle, final)
    result = sim._reconstruction_active_event(0, table)
    assert result.is_ok()
    assert result.ok_value() == "recorded-reconstruction-boundary"
    assert len(calls["constructor"]) == len(calls["reconstruct"]) == 1
    observed = calls["reconstruct"][0]
    np.testing.assert_array_equal(observed["indices"], [1, 0])
    np.testing.assert_array_equal(observed["initial"], source[[1, 0]])
    np.testing.assert_array_equal(observed["final"], final)
    expected = source.copy()
    expected[[1, 0]] = saddle
    np.testing.assert_array_equal(observed["working"], expected)
    np.testing.assert_array_equal(sim.system.positions, source)
    assert set(table.table.columns) == CONSTANT_COLUMNS


def test_legacy_constant_without_ids_keeps_positional_path(monkeypatch):
    saddle = [[10.2, 10.0, 10.0], [11.4, 10.0, 10.0]]
    final = [[10.5, 10.0, 10.0], [11.7, 10.0, 10.0]]
    sim, table, source, calls = setup_case(monkeypatch, None, saddle, final)
    result = sim._reconstruction_active_event(0, table)
    assert result.is_ok()
    assert result.ok_value() == "recorded-reconstruction-boundary"
    assert len(calls["constructor"]) == len(calls["reconstruct"]) == 1
    observed = calls["reconstruct"][0]
    np.testing.assert_array_equal(observed["indices"], [0, 1])
    np.testing.assert_array_equal(observed["initial"], source[[0, 1]])
    np.testing.assert_array_equal(observed["final"], final)
    expected = source.copy()
    expected[[0, 1]] = saddle
    np.testing.assert_array_equal(observed["working"], expected)
    np.testing.assert_array_equal(sim.system.positions, source)
    assert set(table.table.columns) == CONSTANT_COLUMNS
