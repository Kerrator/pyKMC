"""F4: catalogue invalidation must remain coherent through a real KMC step.

Promoted unchanged in substance from the independent review's red-to-green
pack (``test_catalogue_selection.py``); every assertion is the pack's.

These are behavioural regressions, not a requirement to keep deleted refs
alive. The implementation may invalidate and retry selection, or defer a
coherent purge until the selected event's consumers finish. The separate
closure test prevents making this green by reverting to an incomplete
reference-table removal.

Runs real ``KMC.run``/``reconstruction``, event tables, basin detection, step
logging and distance recycling. Search/refinement/native reconstruction are
deterministic boundaries; selection chooses a prescribed available reference
so the failure is independent of the random seed. No live LAMMPS instance or
MPI launch is needed.
"""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import pandas as pd
import pytest
from ase.build import bulk

import pykmc
from pykmc import System
from pykmc.config import Config, EventRecyclingConfig, RateConstantConfig
from pykmc.event_recycling import DistanceRecycling
from pykmc.event_table import ReferenceEventTable
from pykmc.kmc import KMC
from pykmc.result import Err, EventRefinementOutput, Ok


def _config(*, basin: bool) -> Config:
    """One-step htst config with recycling on and the basin flag as given."""
    root = Path(pykmc.__file__).resolve().parent.parent
    original = Config.from_ini_file(str(root / "tests/data/input.in"))
    return original.model_copy(
        update={
            "control": original.control.model_copy(
                update={"n_steps": 1, "basin": basin, "recycle": True}
            ),
            "basin": original.basin.model_copy(update={"energy_thr": 0.15}),
            "rateconstant": RateConstantConfig(style="htst", k0=1.0, T=300.0),
            "eventrecycling": EventRecyclingConfig(style="displacement"),
        }
    )


def _rows(*, incoming_alias: bool) -> pd.DataFrame:
    """Return a small catalogue with a non-reciprocal (alias) or reciprocal graph."""
    # The nonreciprocal graph is allowed by HTST admission: a new direction can
    # point to an already catalogued reverse. Ref 9 is an unrelated viable event.
    # Reference barriers exceed the basin threshold; the selected refined barrier
    # is lower, forcing real detection to dereference the catalogue, then say no.
    ids = [0, 1, 2, 9] if incoming_alias else [0, 1, 2, 3, 9]
    links = [1, 0, 0, 9] if incoming_alias else [1, 0, 3, 2, 9]
    return pd.DataFrame(
        {
            "idx_ref": ids,
            "idx_backward": links,
            "event_id": ["X"] * len(ids),
            "id_final": ["X"] * len(ids),
            "energy_barrier": [0.5] * len(ids),
            "dra": [0.1] * len(ids),
        }
    )


def _assert_reference_closure(table: pd.DataFrame) -> None:
    """Every ``idx_backward`` names a row that is still in the table."""
    ids = set(table["idx_ref"].astype(int))
    assert set(table["idx_backward"].astype(int)) <= ids, (
        "Reference invalidation left a dangling backward link"
    )


class _Log:
    """Logger double recording the reference id of every executed step line."""

    def __init__(self, sim: KMC) -> None:
        self.sim = sim
        self.selected_refs: list[int] = []

    def __getattr__(self, name: str) -> Any:
        """Swallow every other logger call."""
        return lambda *args, **kwargs: None

    def table_line_info_kmc(
        self,
        name: str,
        step: int,
        delta_t: float,
        total_time: float,
        num_reference_event: Any,
        *args: Any,
        **kwargs: Any,
    ) -> None:
        """Record the executed event's reference id."""
        selected = int(num_reference_event)
        # A coherent fix may log an immutable snapshot of an executed event
        # after purging its live catalogue component. Historical IDs need not
        # remain live forever; the detector/recycler invariants below are the
        # requirements, rather than any chosen deletion order.
        self.selected_refs.append(selected)


def _run_invalidation_step(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    *,
    basin: bool,
    incoming_alias: bool,
) -> KMC:
    """Run one real KMC step whose first selected reconstruction fails."""
    config = _config(basin=basin)
    atoms = bulk("Ni", "fcc", a=3.52, cubic=True).repeat((2, 2, 2))
    system = System(
        types=atoms.get_chemical_symbols(),
        positions=atoms.get_positions(),
        cell=np.asarray(atoms.cell),
        pbc=atoms.get_pbc(),
        index=np.arange(len(atoms)),
    )
    manager = SimpleNamespace(broadcast=lambda *args, **kwargs: None)
    sim = KMC(config, manager=manager)
    sim.system = system
    sim.reference_table = ReferenceEventTable(config)
    sim.reference_table.table = _rows(incoming_alias=incoming_alias)
    sim.visited_environments = {"crystal"}
    sim.loggers = _Log(sim)
    # Every unexecuted central atom is far enough and unmoved to be recyclable.
    # This makes stale active rows observable instead of clearing them by default.
    sim.recycler = DistanceRecycling(movement_thr=0.01, distance_thr=0.1)
    attempts: list[int] = []
    empty_stage = SimpleNamespace(results=[], get_successes_results=lambda: [])

    def refinements(*args: Any, **kwargs: Any) -> Any:
        outputs = []
        for atom, reference in enumerate((2, 1, 9, 0)):
            neighbors = sim.neighbors_list.get_neighbors("rcut", atom)
            crop = np.array(system.positions[neighbors], copy=True)
            outputs.append(
                EventRefinementOutput(
                    central_atom_index=atom,
                    saddle_positions=crop,
                    E_saddle=0.1,
                    min2_positions=crop,
                    dE_forward=0.1,
                    num_reference_event=reference,
                    refined="F",  # No site Hessian is needed for this lifecycle test.
                )
            )
        return SimpleNamespace(results=[], get_successes_results=lambda: outputs)

    def select(table: Any) -> tuple[int, float, float]:
        for reference in (2, 1, 9, 0):
            matches = table.table.index[table.table["num_reference_event"] == reference]
            if len(matches):
                return int(matches[0]), 1.0, float(table.table["k"].sum())
        raise AssertionError("Invalidation discarded the unrelated viable ref 9")

    def reconstruct(index: int, table: Any) -> Any:
        reference = int(table.table.loc[index, "num_reference_event"])
        attempts.append(reference)
        if reference == 2:
            return Err(
                SimpleNamespace(message="injected native reconstruction failure")
            )
        positions = np.array(system.positions, copy=True)
        return Ok(
            SimpleNamespace(
                min1_positions=positions,
                saddle_positions=positions,
                min2_positions=positions,
                min2_etot=-1.0,
            )
        )

    monkeypatch.setattr(sim, "minimize_system", lambda: None)
    monkeypatch.setattr(sim, "execute_event_searches", lambda atoms: empty_stage)
    monkeypatch.setattr(sim, "add_reference_events", lambda results: [])
    monkeypatch.setattr(sim, "execute_refinements", refinements)
    monkeypatch.setattr(sim, "_select_event", select)
    monkeypatch.setattr(sim, "_reconstruction_active_event", reconstruct)
    monkeypatch.setattr(sim, "_save", lambda: None)
    monkeypatch.setattr(sim, "_append_snapshot_to_trajectory", lambda: None)
    monkeypatch.setattr(sim, "_save_restart_file", lambda *args: None)
    monkeypatch.setattr(sim, "_close", lambda: None)
    monkeypatch.chdir(tmp_path)
    sim.run()

    assert attempts[0] == 2, "The intended rejected reference was never exercised"
    assert len(sim.loggers.selected_refs) == 1, "No coherent KMC step completed"
    assert sim.loggers.selected_refs[0] in {1, 9}, (
        "The selected event must be the accepted ref 1 or a valid retry using ref 9"
    )
    refs = set(sim.reference_table.table["idx_ref"].astype(int))
    assert 2 not in refs, "The failed reference must eventually be invalidated"
    assert 9 in refs, "An unrelated viable reference was incorrectly invalidated"
    _assert_reference_closure(sim.reference_table.table)
    assert set(sim.active_table.table["num_reference_event"].astype(int)) <= refs, (
        "Recycling retained active rows whose reference was invalidated"
    )
    return sim


@pytest.mark.parametrize(
    "basin", [True, False], ids=["basin-consumer", "recycle-consumer"]
)
def test_selected_event_and_recycling_stay_coherent_after_reverse_link_purge(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, basin: bool
) -> None:
    """F4 red: a rejected alias must not strand the successful selected event."""
    _run_invalidation_step(monkeypatch, tmp_path, basin=basin, incoming_alias=True)


def test_ordinary_reciprocal_component_does_not_disrupt_other_selected_event(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Positive control: deleting a separate reciprocal pair remains safe."""
    sim = _run_invalidation_step(
        monkeypatch, tmp_path, basin=True, incoming_alias=False
    )
    assert sim.loggers.selected_refs == [1]
    assert set(sim.reference_table.table["idx_ref"].astype(int)) == {0, 1, 9}


def test_removal_preserves_reverse_link_closure_for_incoming_alias() -> None:
    """Positive guard: an F4 fix must not reintroduce dangling reverse links."""
    table = ReferenceEventTable(_config(basin=False))
    table.table = _rows(incoming_alias=True)
    table.remove([2])
    _assert_reference_closure(table.table)
    assert set(table.table["idx_ref"].astype(int)) == {9}
