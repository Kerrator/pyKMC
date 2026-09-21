"""A basin-path ``Err(RECONSTRUCTION_INVALID_EVENT_DATA)`` purges its reference.

Contracts 7f policy 5: a violated user constraint on the reconstruction or
basin path returns ``Err(RECONSTRUCTION_INVALID_EVENT_DATA)`` *so the
catalogue purge runs*. ``KMC.reconstruction`` purges on the reconstruction
path; here a real ``KMC.run`` step (engine stages stubbed, no LAMMPS, no MPI)
enters the basin branch with a basin double returning that ``Err`` and the
same purge must follow: reference removed with its reverse-link closure,
every other active row referencing it dropped before the end-of-step prune,
its topology forgotten, the selected event applied as before. The executed
row is already reconstructed on the current state and is read by the step
log, so it is kept through the log and its label stays valid.
"""

from __future__ import annotations

import copy
import random
import types
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pytest

import pykmc.kmc as kmc_module
from pykmc import NeighborsList
from pykmc.config import Config, RateConstantConfig
from pykmc.event_table import ActiveEventTable, ReferenceEventTable
from pykmc.kmc import KMC
from pykmc.result import Err, ErrorInfo, ErrorType, Ok
from tests.lifecycle.conftest import DATA_INPUT, FakeManager
from tests.lifecycle.test_kmc_orchestration import (
    SEED,
    _FakeEventSearch,
    _FakeRefinement,
    _Recorder,
    _refined,
)

OFFENDER_TOPOLOGY = "offender-topology"
BYSTANDER_TOPOLOGY = "bystander-topology"


def _basin_config() -> Config:
    """One constant-mode basin step (threshold above every barrier used here)."""
    config = Config.from_ini_file(DATA_INPUT)
    control = config.control.model_copy(
        update={"n_steps": 1, "basin": True, "recycle": False}
    )
    basin = config.basin.model_copy(update={"energy_thr": 1.0})
    rate = RateConstantConfig(style="constant", k0=5.0, T=config.rateconstant.T)
    return config.model_copy(
        update={"control": control, "basin": basin, "rateconstant": rate}
    )


def _reference_row(
    table: ReferenceEventTable, system: Any, idx_ref: int, topology: str
) -> None:
    """Insert one self-linked constant-mode row with a synthetic topology."""
    pos = system.positions
    fwd, _ = table._build_event_series(
        min1_positions=pos,
        saddle_positions=pos,
        min2_positions=pos,
        index_move=0,
        dE_forward=0.5,
        dE_backward=0.5,
        cell=system.cell,
        types=list(system.types),
    )
    fwd["idx_ref"] = idx_ref
    fwd["idx_backward"] = idx_ref
    fwd["event_id"] = topology
    fwd["id_final"] = topology
    table.table = pd.concat([table.table, fwd.to_frame().T], ignore_index=True)


class _PolicyFiveBasin:
    """Basin double: ``execute`` returns the policy-5 ``Err`` for ``offender``."""

    offender: int = 1
    instances: list[Any] = []

    def __init__(
        self,
        config: Any,
        reference_table: Any,
        visited: Any,
        manager: Any,
        global_constraints: Any = None,
    ) -> None:
        self.connectivity_table = None
        self.states: dict[int, Any] = {}
        type(self).instances.append(self)

    def execute(self, system: Any) -> Err:
        """The Err ``BasinsGenericEvents.system_from_state`` produces."""
        return Err(
            ErrorInfo(
                type=ErrorType.RECONSTRUCTION_INVALID_EVENT_DATA,
                message="generic event {} changes a user-fixed reference "
                "coordinate: test".format(self.offender),
                variables={"idx_ref": self.offender},
            )
        )


def _run_basin_step(
    system: Any,
    neighbors: NeighborsList,
    offender: int,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    reconstruction: Any = None,
    basin_class: Any = _PolicyFiveBasin,
) -> tuple[KMC, _Recorder, dict[str, Any]]:
    """One real ``KMC.run`` step entering the basin branch on active row 0."""
    config = _basin_config()
    system = copy.deepcopy(system)
    sim = KMC(config, manager=FakeManager())
    sim.system = system
    sim.loggers = _Recorder()
    sim.reference_table = ReferenceEventTable(config)
    _reference_row(sim.reference_table, system, 0, BYSTANDER_TOPOLOGY)
    _reference_row(sim.reference_table, system, 1, OFFENDER_TOPOLOGY)
    sim.visited_environments = {"crystal", BYSTANDER_TOPOLOGY, OFFENDER_TOPOLOGY}
    positions = np.array(system.positions, copy=True)

    # Three active rows: the selected one (reference 0, 0.5 eV, the only
    # channel with a non-negligible rate), a second row on reference 0 and a
    # row on reference 1, both far too slow to be drawn.
    second = _refined(system, neighbors, 5, 2.0)
    third = _refined(system, neighbors, 10, 2.0)
    third.num_reference_event = 1
    refined = [_refined(system, neighbors, 0, 0.5), second, third]

    monkeypatch.setattr(sim, "minimize_system", lambda: None)
    monkeypatch.setattr(
        sim, "execute_event_searches", lambda atoms: _FakeEventSearch([])
    )
    monkeypatch.setattr(sim, "add_reference_events", lambda results: [])
    monkeypatch.setattr(
        sim,
        "execute_refinements",
        lambda subset, existing_pairs=None: _FakeRefinement(refined),
    )
    if reconstruction is None:

        def reconstruction(idx: int, table: Any) -> Any:
            return Ok(
                types.SimpleNamespace(
                    min1_positions=positions.copy(),
                    saddle_positions=positions.copy(),
                    min2_positions=positions.copy(),
                    min2_etot=-1.0,
                )
            )

    monkeypatch.setattr(sim, "_reconstruction_active_event", reconstruction)
    monkeypatch.setattr(sim, "_save", lambda: None)
    monkeypatch.setattr(sim, "_append_snapshot_to_trajectory", lambda: None)
    monkeypatch.setattr(sim, "_save_restart_file", lambda step, total_time: None)
    monkeypatch.setattr(sim, "_close", lambda: None)

    _PolicyFiveBasin.offender = offender
    _PolicyFiveBasin.instances = []
    monkeypatch.setattr(kmc_module, "BasinsGenericEvents", basin_class)

    # Observe the active table exactly when the end-of-step prune reads it:
    # after the step log, before the table is cleared.
    at_prune: dict[str, Any] = {}
    original_prune = ActiveEventTable.prune_for_recycling

    def observing_prune(self: ActiveEventTable, executed_idx: int, *args: Any) -> None:
        at_prune["references"] = [
            int(r) for r in self.table["num_reference_event"].tolist()
        ]
        at_prune["executed_reference"] = int(
            self.table.loc[executed_idx].at["num_reference_event"]
        )
        at_prune["executed_atom"] = int(self.table.loc[executed_idx].at["atom_index"])
        original_prune(self, executed_idx, *args)

    monkeypatch.setattr(ActiveEventTable, "prune_for_recycling", observing_prune)
    monkeypatch.chdir(tmp_path)
    random.seed(SEED)
    sim.run()
    return sim, sim.loggers, at_prune


@pytest.fixture
def fcc_neighbors(system_single_type_fcc: Any) -> NeighborsList:
    """The step's rcut neighbour list on the test crystal."""
    config = Config.from_ini_file(DATA_INPUT)
    return NeighborsList(
        system_single_type_fcc,
        config.atomicenvironment.rnei,
        config.atomicenvironment.rcut,
    )


def test_basin_err_purges_another_reference_and_keeps_the_selected_event(
    system_single_type_fcc: Any,
    fcc_neighbors: NeighborsList,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """Offender 1 leaves both tables; reference 0 and its rows survive."""
    sim, log, at_prune = _run_basin_step(
        system_single_type_fcc, fcc_neighbors, 1, monkeypatch, tmp_path
    )
    assert len(_PolicyFiveBasin.instances) == 1, "the basin branch ran once"
    remaining = sorted(int(i) for i in sim.reference_table.table["idx_ref"])
    assert remaining == [0], remaining
    assert OFFENDER_TOPOLOGY not in sim.visited_environments
    assert BYSTANDER_TOPOLOGY in sim.visited_environments
    # Active rows of the purged reference are gone before the prune reads
    # the table; both reference-0 rows are still there, the executed one
    # under a valid label.
    assert at_prune["references"] == [0, 0]
    assert at_prune["executed_reference"] == 0 and at_prune["executed_atom"] == 0
    # The selected event was applied and logged as before.
    assert len(log.step_lines) == 1
    assert log.step_lines[0]["num_reference_event"] == 0
    assert log.step_lines[0]["energy_barrier"] == 0.5
    purge_lines = [m for _, m in log.messages if "1" in m and "purg" in m.lower()]
    assert purge_lines, [m for _, m in log.messages if "Basin" in m]
    assert any("back to original event" in m for _, m in log.messages)


def test_basin_err_naming_the_selected_reference_purges_it_but_logs_the_step(
    system_single_type_fcc: Any,
    fcc_neighbors: NeighborsList,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """Offender 0 is the executed row's own reference: catalogue purged, the
    other reference-0 row dropped, the executed row kept for the step log."""
    sim, log, at_prune = _run_basin_step(
        system_single_type_fcc, fcc_neighbors, 0, monkeypatch, tmp_path
    )
    remaining = sorted(int(i) for i in sim.reference_table.table["idx_ref"])
    assert remaining == [1], remaining
    assert BYSTANDER_TOPOLOGY not in sim.visited_environments
    assert OFFENDER_TOPOLOGY in sim.visited_environments
    assert at_prune["references"] == [0, 1], at_prune
    assert at_prune["executed_reference"] == 0 and at_prune["executed_atom"] == 0
    assert len(log.step_lines) == 1
    assert log.step_lines[0]["num_reference_event"] == 0
    assert log.step_lines[0]["energy_barrier"] == 0.5


def test_basin_err_without_the_offending_reference_purges_nothing_loudly(
    system_single_type_fcc: Any,
    fcc_neighbors: NeighborsList,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """The policy-5 type without ``variables['idx_ref']`` cannot purge: one
    WARNING says so and the catalogue is untouched."""

    class _AnonymousBasin(_PolicyFiveBasin):
        def execute(self, system: Any) -> Err:
            return Err(
                ErrorInfo(
                    type=ErrorType.RECONSTRUCTION_INVALID_EVENT_DATA,
                    message="anonymous violation",
                )
            )

    sim, log, at_prune = _run_basin_step(
        system_single_type_fcc,
        fcc_neighbors,
        1,
        monkeypatch,
        tmp_path,
        basin_class=_AnonymousBasin,
    )
    remaining = sorted(int(i) for i in sim.reference_table.table["idx_ref"])
    assert remaining == [0, 1], remaining
    assert at_prune["references"] == [0, 0, 1]
    warnings = [
        m
        for _, m in log.messages
        if "anonymous violation" in m and "nothing is purged" in m
    ]
    assert len(warnings) == 1, [m for _, m in log.messages if "Basin" in m]
    assert len(log.step_lines) == 1
    assert log.step_lines[0]["num_reference_event"] == 0


class _ExitingBasin(_PolicyFiveBasin):
    """Basin double whose exploration succeeds and exits through reference 1."""

    system: Any = None
    neighbors: Any = None

    def execute(self, system: Any) -> Ok:
        positions = np.array(type(self).system.positions, copy=True)
        state = types.SimpleNamespace(
            neighbors_list=type(self).neighbors,
            system=types.SimpleNamespace(positions=positions.copy()),
        )
        self.states = {0: state}
        crop = np.asarray(type(self).neighbors.get_neighbors("rcut", 0), dtype=int)
        return Ok(
            types.SimpleNamespace(
                initial_system_positions=positions.copy(),
                from_state=0,
                num_reference_event=1,
                central_atom=0,
                saddle_positions=positions[crop] + 0.1,
                final_positions=positions[crop] + 0.2,
                energy_barrier=0.5,
                neighbors=crop,
                t_exit=1.0,
                k_tot=1.0,
                exit_state=1,
            )
        )


def test_exit_event_constraint_violation_purges_the_exit_reference(
    system_single_type_fcc: Any,
    fcc_neighbors: NeighborsList,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """The exit-state reconstruction is the second policy-5 seam on the basin
    path: its ``Err(RECONSTRUCTION_INVALID_EVENT_DATA)`` purges the exit
    event's reference and the run falls back to the selected event."""
    _ExitingBasin.system = system_single_type_fcc
    _ExitingBasin.neighbors = fcc_neighbors
    positions = np.array(system_single_type_fcc.positions, copy=True)
    calls: list[int] = []

    def reconstruction(idx: int, table: Any) -> Any:
        calls.append(int(table.table.loc[idx].at["num_reference_event"]))
        if len(calls) == 1:
            return Ok(
                types.SimpleNamespace(
                    min1_positions=positions.copy(),
                    saddle_positions=positions.copy(),
                    min2_positions=positions.copy(),
                    min2_etot=-1.0,
                )
            )
        return Err(
            ErrorInfo(
                type=ErrorType.RECONSTRUCTION_INVALID_EVENT_DATA,
                message="frozen atom moved",
            )
        )

    sim, log, at_prune = _run_basin_step(
        system_single_type_fcc,
        fcc_neighbors,
        1,
        monkeypatch,
        tmp_path,
        reconstruction=reconstruction,
        basin_class=_ExitingBasin,
    )
    assert calls == [0, 1], "selected event, then the exit event on reference 1"
    remaining = sorted(int(i) for i in sim.reference_table.table["idx_ref"])
    assert remaining == [0], remaining
    assert OFFENDER_TOPOLOGY not in sim.visited_environments
    assert at_prune["references"] == [0, 0]
    assert at_prune["executed_reference"] == 0
    assert len(log.step_lines) == 1
    assert log.step_lines[0]["num_reference_event"] == 0
    assert any("Reconstruction Exit State Basin fails" in m for _, m in log.messages)
    assert any(
        "purging reference events [1]" in m and "frozen atom moved" in m
        for _, m in log.messages
    )
