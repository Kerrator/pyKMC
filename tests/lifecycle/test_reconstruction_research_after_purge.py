"""``KMC.reconstruction``: purge the failed reference, then search again once.

A failed reconstruction removes its reference from the catalogue (with the
row its backward link names and, in the htst/rpa styles, the reverse-link
closure) and drops every active row of a removed reference. When that
purge empties the active table, the step does not end the simulation: the
purged topologies are forgotten, a fresh event search runs for the same
step, and the selection continues on the rebuilt table. The simulation
exits with status 1 only when that single re-search leaves nothing to
reconstruct.

Default setup (cases that need another catalogue or other rows say so): a
constant-style catalogue with ONE reference (``idx_ref`` 2, topology
``"X"``, its own backward event, as for a symmetric vacancy hop) and four
active rows on atoms 10-13 that all point at it. ``_select_event`` takes
the first row, ``_reconstruction_active_event`` records the attempted
atom, and the step's search stage (``KMC._search_and_refine_step_events``)
is replaced by a stub that catalogues new references and adds their
active rows. No LAMMPS, no MPI.
"""

from __future__ import annotations

import copy
import types
from typing import Any, Callable

import numpy as np
import pandas as pd
import pytest
from pykmc.event_table import ActiveEventTable, ReferenceEventTable
from pykmc.info_simulation import info_active_events
from pykmc.kmc import KMC
from pykmc.result import Err, ErrorInfo, ErrorType, EventRefinementOutput, Ok
from tests.lifecycle.conftest import FakeManager

TOPOLOGIES = ("X", "Y", "Z")
"""Topologies present in the fake current configuration."""


class _Recorder:
    """Logger double recording every message with its level."""

    def __init__(self) -> None:
        self.lines: list[tuple[str, str]] = []

    def info(self, name: str, msg: str, *args: Any, **kwargs: Any) -> None:
        """Record an info message."""
        self.lines.append(("info", str(msg)))

    def warning(self, name: str, msg: str, *args: Any, **kwargs: Any) -> None:
        """Record a warning."""
        self.lines.append(("warning", str(msg)))

    def error(self, name: str, msg: str, *args: Any, **kwargs: Any) -> None:
        """Record an error."""
        self.lines.append(("error", str(msg)))


def _row(atom: int, ref: int) -> EventRefinementOutput:
    """Return a generic (unrefined) active row of ``ref`` on ``atom``."""
    crop = np.zeros((1, 3))
    return EventRefinementOutput(
        central_atom_index=atom,
        saddle_positions=crop.copy(),
        E_saddle=0.5,
        min2_positions=crop.copy(),
        dE_forward=0.5,
        num_reference_event=ref,
        refined="F",
    )


def _catalogue_rows(refs: list[tuple[int, str]]) -> pd.DataFrame:
    """Return self-linked catalogue rows ``(idx_ref, event_id)``."""
    return pd.DataFrame(
        {
            "idx_ref": [ref for ref, _ in refs],
            "idx_backward": [ref for ref, _ in refs],
            "event_id": [topology for _, topology in refs],
            "energy_barrier": [0.5] * len(refs),
            "dra": [0.1] * len(refs),
        }
    )


class _Step:
    """One constant-style ``reconstruction`` call with the search stage stubbed.

    Parameters
    ----------
    config : Config
        The committed constant-style test input.
    system : System
        The test crystal (read by ``validate_recycled`` and, in the
        real-stage tests, by ``remove_duplicates``).
    fails : Callable[[int, int], bool]
        ``fails(atom, reference)`` decides whether an attempt fails.
    research_adds : list[tuple[int, str, list[int]]]
        What the stubbed re-search catalogues: ``(idx_ref, topology, atoms)``
        per new self-linked reference, with one active row per atom.
    catalogue : pd.DataFrame, optional
        The catalogue before the step; defaults to reference 2 (``X``) only.
    first_rows : list[tuple[int, int]], optional
        The step's active rows ``(atom, reference)``; defaults to atoms
        10-13 on reference 2.
    research_rows : list[tuple[int, int]], optional
        Rows ``(atom, reference)`` the stubbed re-search refines on entries
        already catalogued, added before the rows of ``research_adds``.
    stub_research : bool, optional
        ``False`` keeps the real ``_search_and_refine_step_events``; the
        caller then stubs the engine stages it drives.

    """

    def __init__(
        self,
        config: Any,
        system: Any,
        fails: Callable[[int, int], bool],
        research_adds: list[tuple[int, str, list[int]]],
        catalogue: pd.DataFrame | None = None,
        first_rows: list[tuple[int, int]] | None = None,
        research_rows: list[tuple[int, int]] | None = None,
        stub_research: bool = True,
    ) -> None:
        self.kmc = KMC(config, manager=FakeManager())
        kmc = self.kmc
        kmc.system = copy.deepcopy(system)
        kmc.loggers = _Recorder()
        kmc.neighbors_list = types.SimpleNamespace(
            get_neighbors=lambda cutoff, atom: np.array([atom], dtype=int)
        )
        # The current configuration shows the topologies X, Y and Z; only X
        # has been searched before this step.
        kmc.atomic_environment = types.SimpleNamespace(
            get_new_environments=lambda visited: set(TOPOLOGIES) - set(visited),
            atomic_environment_list=list(TOPOLOGIES),
        )
        kmc.visited_environments = {"X"}
        kmc.reference_table = ReferenceEventTable(kmc.config)
        kmc.reference_table.table = (
            _catalogue_rows([(2, "X")]) if catalogue is None else catalogue
        )
        if first_rows is None:
            first_rows = [(atom, 2) for atom in (10, 11, 12, 13)]
        kmc.active_table = ActiveEventTable(kmc.config)
        kmc.active_table.add_events([_row(atom, ref) for atom, ref in first_rows])
        self.table = kmc.active_table
        self.attempts: list[tuple[int, int]] = []
        self.research_calls: list[tuple[list[str], int]] = []
        self.visited_at_research: list[set[str]] = []
        self.fails = fails
        self.research_adds = research_adds
        self.research_rows = research_rows or []
        kmc._select_event = self._select
        kmc._reconstruction_active_event = self._reconstruct
        if stub_research:
            kmc._search_and_refine_step_events = self._research

    @staticmethod
    def _select(active: ActiveEventTable) -> tuple[int, float, float]:
        """Select the first active row."""
        return int(active.table.index[0]), 1.0, float(len(active.table))

    def _reconstruct(self, label: int, active: ActiveEventTable) -> Any:
        """Record the attempt and fail or succeed as ``fails`` says."""
        atom = int(active.table.loc[label, "atom_index"])
        ref = int(active.table.loc[label, "num_reference_event"])
        self.attempts.append((atom, ref))
        if self.fails(atom, ref):
            return Err(
                ErrorInfo(
                    type=ErrorType.RECONSTRUCTION_INVALID_MIN2,
                    message="injected failure",
                )
            )
        return Ok(atom)

    def _research(self, new_environments: list[str], nsearch: int) -> tuple:
        """Stand in for the step's search/refine stage.

        Records its arguments and the known topologies it was called with,
        catalogues ``research_adds``, adds the ``research_rows`` and the new
        references' active rows to the persistent table and marks every
        current topology visited, as the real stage does.
        """
        kmc = self.kmc
        self.research_calls.append((list(new_environments), int(nsearch)))
        self.visited_at_research.append(set(kmc.visited_environments))
        if self.research_rows:
            kmc.active_table.add_events(
                [_row(atom, ref) for atom, ref in self.research_rows]
            )
        if self.research_adds:
            kmc.reference_table.table = pd.concat(
                [
                    kmc.reference_table.table,
                    _catalogue_rows([(ref, top) for ref, top, _ in self.research_adds]),
                ],
                ignore_index=True,
            )
            kmc.active_table.add_events(
                [
                    _row(atom, ref)
                    for ref, _, atoms in self.research_adds
                    for atom in atoms
                ]
            )
        kmc.visited_environments.update(TOPOLOGIES)
        return kmc.active_table, None, [], None

    def errors(self) -> list[str]:
        """Return the error messages logged so far."""
        return [msg for level, msg in self.kmc.loggers.lines if level == "error"]


def _first_attempt_fails(atom: int, ref: int) -> bool:
    """Fail the first row (atom 10) only."""
    return atom == 10


def test_one_failure_purges_then_re_searches_and_the_step_completes(
    constant_config: Any, system_single_type_fcc: Any
) -> None:
    """(a) One failure: purge, one re-search, a re-searched row is executed.

    The three sibling rows of reference 2 were purged with it and are never
    attempted; the executed row belongs to reference 3, which the re-search
    produced. The purged topology ``X`` is forgotten before the re-search
    (so the search sees it as new again) and stays visited afterwards: it
    is not returned for step-level forgetting.
    """
    step = _Step(
        constant_config,
        system_single_type_fcc,
        fails=_first_attempt_fails,
        research_adds=[(3, "X", [20])],
    )
    kmc = step.kmc
    try:
        result, _, _, label, err_reference, err_ae = kmc.reconstruction(step.table)
    except SystemExit as exc:
        pytest.fail(
            f"one failed reconstruction ended the simulation (SystemExit "
            f"{exc.code!r}; errors {step.errors()}); attempts {step.attempts}, "
            f"re-search calls {step.research_calls}"
        )

    assert result.is_ok() and result.ok_value() == 20
    assert step.attempts == [(10, 2), (20, 3)]
    assert int(step.table.table.loc[label, "num_reference_event"]) == 3
    assert int(step.table.table.loc[label, "atom_index"]) == 20
    nsearch = constant_config.eventsearch.nsearch
    assert step.research_calls == [(["X", "Y", "Z"], nsearch)]
    assert "X" not in step.visited_at_research[0]
    assert list(kmc.reference_table.table["idx_ref"].astype(int)) == [3]
    assert err_reference == [2]
    assert err_ae == []
    assert "X" in kmc.visited_environments.difference(set(err_ae))
    assert step.errors() == []
    assert kmc.manager.shutdowns == 0


def test_re_search_that_yields_no_row_exits_with_status_one(
    constant_config: Any, system_single_type_fcc: Any
) -> None:
    """(b) The re-search finds nothing to execute: the run ends with status 1."""
    step = _Step(
        constant_config,
        system_single_type_fcc,
        fails=_first_attempt_fails,
        research_adds=[],
    )
    with pytest.raises(SystemExit) as raised:
        step.kmc.reconstruction(step.table)

    assert raised.value.code == 1
    assert len(step.research_calls) == 1
    assert step.attempts == [(10, 2)]
    assert "All event reconstuctions failed." in step.errors()
    assert step.kmc.manager.shutdowns == 1


def test_every_re_searched_row_failing_exits_with_status_one_without_a_third_pass(
    constant_config: Any, system_single_type_fcc: Any
) -> None:
    """(b) Every re-searched row fails: exit 1 after exactly one re-search.

    The re-search catalogues two references (3 on atom 20, 4 on atom 21);
    both fail in turn, each purge dropping its own row. The second
    exhaustion ends the run; no further search is attempted.
    """
    step = _Step(
        constant_config,
        system_single_type_fcc,
        fails=lambda atom, ref: True,
        research_adds=[(3, "X", [20]), (4, "Y", [21])],
    )
    with pytest.raises(SystemExit) as raised:
        step.kmc.reconstruction(step.table)

    assert raised.value.code == 1
    assert len(step.research_calls) == 1
    assert step.attempts == [(10, 2), (20, 3), (21, 4)]
    assert step.table.table.empty
    assert "All event reconstuctions failed." in step.errors()
    assert step.kmc.manager.shutdowns == 1


def test_no_failure_runs_no_re_search(
    constant_config: Any, system_single_type_fcc: Any
) -> None:
    """(c) The first row reconstructs: nothing is purged or searched again."""
    step = _Step(
        constant_config,
        system_single_type_fcc,
        fails=lambda atom, ref: False,
        research_adds=[(3, "X", [20])],
    )
    result, _, _, label, err_reference, err_ae = step.kmc.reconstruction(step.table)

    assert result.is_ok() and result.ok_value() == 10
    assert step.research_calls == []
    assert step.attempts == [(10, 2)]
    assert label == 0
    assert err_reference == [] and err_ae == []
    assert list(step.table.table["atom_index"].astype(int)) == [10, 11, 12, 13]
    assert step.kmc.visited_environments == {"X"}


def test_only_second_pass_topologies_are_returned_for_step_level_forgetting(
    constant_config: Any, system_single_type_fcc: Any
) -> None:
    """First-pass topologies are forgotten before the re-search, not again.

    The atom-10 row of reference 2 (``X``) fails and empties the table; the
    re-search catalogues 3 (``Y``, atom 20) and 4 (``Z``, atom 21). The
    atom-20 row fails too, reference 3 is purged, and the atom-21 row is
    executed. Every removed id is reported for the step log, but only the
    second-pass topology ``Y`` is returned for the step-level
    ``visited_environments`` difference: ``X`` was forgotten inside
    ``reconstruction`` and re-searched, so forgetting it again would search
    it a third time next step.
    """
    step = _Step(
        constant_config,
        system_single_type_fcc,
        fails=lambda atom, ref: atom in (10, 20),
        research_adds=[(3, "Y", [20]), (4, "Z", [21])],
    )
    kmc = step.kmc
    result, _, _, label, err_reference, err_ae = kmc.reconstruction(step.table)

    assert result.is_ok() and result.ok_value() == 21
    assert step.attempts == [(10, 2), (20, 3), (21, 4)]
    assert int(step.table.table.loc[label, "num_reference_event"]) == 4
    assert len(step.research_calls) == 1
    assert err_reference == [2, 3]
    assert err_ae == ["Y"]
    after_step = kmc.visited_environments.difference(set(err_ae))
    assert after_step == {"X", "Z"}


def test_rebuilt_rows_of_an_entry_that_lost_its_backward_event_are_dropped(
    constant_config: Any, system_single_type_fcc: Any
) -> None:
    """The purge's backward-link rule also applies to the re-searched rows.

    A loaded catalogue need not have reciprocal links: 0 <-> 1 is a pair
    and 2 -> 0 an alias of it. The only active row (atom 10 -> 2) fails; the
    constant style removes 2 and the row its backward link names (0) and
    keeps 1, whose backward event 0 is gone. The re-search refines every
    applicable catalogue entry, so it produces a row of 1 again (atom 30),
    and catalogues a new event under the freed id 2 (the constant style
    allocates above the surviving ids), with a row on atom 20. The ref-1
    row is dropped as the first-pass purge would have dropped it; the new
    ref-2 row survives although its id was removed earlier in the call,
    is executed, and the step's events block can be built.
    """
    step = _Step(
        constant_config,
        system_single_type_fcc,
        fails=lambda atom, ref: atom == 10,
        research_adds=[(2, "X", [20])],
        catalogue=pd.DataFrame(
            {
                "idx_ref": [0, 1, 2],
                "idx_backward": [1, 0, 0],
                "event_id": ["A", "B", "C"],
                "energy_barrier": [0.5] * 3,
                "dra": [0.1] * 3,
            }
        ),
        first_rows=[(10, 2)],
        research_rows=[(30, 1)],
    )
    kmc = step.kmc
    result, _, _, label, err_reference, err_ae = kmc.reconstruction(step.table)
    info = info_active_events(kmc.system.types, kmc.reference_table, step.table)
    info.output_msg()

    assert result.is_ok() and result.ok_value() == 20
    assert step.attempts == [(10, 2), (20, 2)]
    assert list(step.table.table["atom_index"].astype(int)) == [20]
    assert int(step.table.table.loc[label, "num_reference_event"]) == 2
    assert list(kmc.reference_table.table["idx_ref"].astype(int)) == [1, 2]
    assert "A" not in step.visited_at_research[0]
    assert "C" not in step.visited_at_research[0]
    assert err_reference == [0, 2] and err_ae == []


def test_orphaned_entry_rows_are_dropped_even_when_its_removed_backward_id_is_reused(
    constant_config: Any, system_single_type_fcc: Any
) -> None:
    """An entry that lost its backward event stays dropped after an id reuse.

    A loaded catalogue need not have reciprocal links: 0 -> 2, 1 -> 1 and
    2 -> 2. The active rows are atom 10 -> 2 and atom 11 -> 0. The atom-10
    row fails; the constant style removes 2 (its own backward event), and
    the purge drops the ref-2 row and, by the backward-link rule, the ref-0
    row, so the table is empty. The re-search refines every applicable
    catalogue entry, entry 0 among them (atom 30), and catalogues an
    UNRELATED event (``Y``, barrier 1.9 eV) under the freed id 2 (the
    constant style allocates above the surviving ids 0 and 1), with a row
    on atom 20.

    Id 2 is catalogued again, but it names the new event, not the backward
    event entry 0 lost. The rebuilt ref-0 row must be dropped as in the
    first pass; executing it would report the unrelated event as its
    backward in the events block and in basin detection. The new ref-2 row
    is executed and ``info_active_events`` resolves its backward to itself.
    """
    step = _Step(
        constant_config,
        system_single_type_fcc,
        fails=lambda atom, ref: atom == 10,
        research_adds=[(2, "Y", [20])],
        catalogue=pd.DataFrame(
            {
                "idx_ref": [0, 1, 2],
                "idx_backward": [2, 1, 2],
                "event_id": ["A", "B", "X"],
                "energy_barrier": [0.5, 0.6, 0.7],
                "dra": [0.1] * 3,
            }
        ),
        first_rows=[(10, 2), (11, 0)],
        research_rows=[(30, 0)],
    )
    kmc = step.kmc
    stub = step._research

    def research(new_environments: list[str], nsearch: int) -> tuple:
        out = stub(new_environments, nsearch)
        catalogue = kmc.reference_table.table
        catalogue.loc[catalogue["event_id"] == "Y", "energy_barrier"] = 1.9
        return out

    kmc._search_and_refine_step_events = research
    result, _, _, label, err_reference, err_ae = kmc.reconstruction(step.table)
    info = info_active_events(kmc.system.types, kmc.reference_table, step.table)
    info.output_msg()

    assert step.attempts == [(10, 2), (20, 2)]
    assert result.is_ok() and result.ok_value() == 20
    rows = step.table.table[["atom_index", "num_reference_event"]].astype(int)
    assert rows.values.tolist() == [[20, 2]]
    assert int(step.table.table.loc[label, "num_reference_event"]) == 2
    assert [int(ref) for ref in info.reference_events] == [2]
    assert [float(e) for e in info.dE_backward] == [1.9]
    # Entry 0 stays catalogued, as in the first pass (the constant style
    # removes an event and the row its backward link names only).
    assert list(kmc.reference_table.table["idx_ref"].astype(int)) == [0, 1, 2]
    assert err_reference == [2] and err_ae == []


def test_new_event_under_the_freed_id_of_a_removed_orphan_is_kept(
    constant_config: Any, system_single_type_fcc: Any
) -> None:
    """An orphan removed later in the pass is not an orphan after the re-search.

    Catalogue: 0 (``X``, self-linked), 1 (``W``) -> 3, 2 (``A``,
    self-linked) and 3 (``Z``) -> 0, a chain an htst/rpa run can write (a
    forward row linked to an already catalogued reverse). The active rows
    are atom 10 -> 0, atom 11 -> 1 and atom 12 -> 3. The atom-10 row fails:
    0 is removed, and the ref-3 row is dropped because entry 3 lost its
    backward event. The atom-11 row fails: 1 and the row its backward link
    names, 3, are removed, and the table is empty. The re-search catalogues
    an unrelated event under the freed id 3 (above the surviving id 2) with
    a row on atom 20.

    Entry 3 was removed in this call, so id 3 now names only the new event:
    its row is executed, not dropped as the rows of the removed orphan were.
    """
    step = _Step(
        constant_config,
        system_single_type_fcc,
        fails=lambda atom, ref: atom in (10, 11),
        research_adds=[(3, "Y", [20])],
        catalogue=pd.DataFrame(
            {
                "idx_ref": [0, 1, 2, 3],
                "idx_backward": [0, 3, 2, 0],
                "event_id": ["X", "W", "A", "Z"],
                "energy_barrier": [0.5] * 4,
                "dra": [0.1] * 4,
            }
        ),
        first_rows=[(10, 0), (11, 1), (12, 3)],
    )
    kmc = step.kmc
    try:
        result, _, _, label, err_reference, err_ae = kmc.reconstruction(step.table)
    except SystemExit as exc:
        pytest.fail(
            f"the new event's row was dropped and the run ended (SystemExit "
            f"{exc.code!r}; errors {step.errors()}); attempts {step.attempts}"
        )
    info = info_active_events(kmc.system.types, kmc.reference_table, step.table)
    info.output_msg()

    assert step.attempts == [(10, 0), (11, 1), (20, 3)]
    assert result.is_ok() and result.ok_value() == 20
    assert int(step.table.table.loc[label, "num_reference_event"]) == 3
    assert list(kmc.reference_table.table["idx_ref"].astype(int)) == [2, 3]
    assert err_reference == [0, 1, 3] and err_ae == []


class _NoSearch:
    """Event-search double: the searches find nothing."""

    results: list[Any] = []

    def get_successes_results(self) -> list[Any]:
        """Return no successful search."""
        return []


class _Refined:
    """Refinement double returning canned refined rows."""

    def __init__(self, outputs: list[EventRefinementOutput]) -> None:
        self._outputs = outputs
        self.results: list[Any] = []
        self.superseded_rows: list[Any] = []

    def get_successes_results(self) -> list[EventRefinementOutput]:
        """Return the canned rows."""
        return self._outputs


def test_real_re_search_stage_refines_the_surviving_catalogue_and_completes(
    constant_config: Any, system_single_type_fcc: Any
) -> None:
    """The real search stage runs inside ``reconstruction`` after the purge.

    The catalogue holds 2 (``X``, the four rows of the step) and 5 (``Y``,
    whose refinement produced no row this step). The atom-10 row fails and
    reference 2 is purged with every row. The real
    ``_search_and_refine_step_events`` then searches the forgotten ``X``
    with ``Y`` and ``Z`` (nothing is found), marks the current topologies
    visited and refines the applicable catalogue, now reference 5 only,
    against an empty table; its row on atom 1 is executed.
    """
    step = _Step(
        constant_config,
        system_single_type_fcc,
        fails=_first_attempt_fails,
        research_adds=[],
        catalogue=_catalogue_rows([(2, "X"), (5, "Y")]),
        stub_research=False,
    )
    kmc = step.kmc
    kmc.inactive_ae = None
    searched: list[list[int]] = []
    refined: list[tuple[list[int], Any]] = []

    def search(atoms: list[int]) -> _NoSearch:
        searched.append(list(atoms))
        return _NoSearch()

    def refine(subset: pd.DataFrame, existing_pairs: Any = None) -> _Refined:
        refined.append((list(subset["idx_ref"].astype(int)), existing_pairs))
        return _Refined([_row(1, 5)])

    kmc.execute_event_searches = search
    kmc.execute_refinements = refine
    result, _, _, label, err_reference, err_ae = kmc.reconstruction(step.table)

    assert result.is_ok() and result.ok_value() == 1
    assert step.attempts == [(10, 2), (1, 5)]
    assert int(step.table.table.loc[label, "num_reference_event"]) == 5
    # One search per new environment and nsearch atoms each: X on atom 0,
    # Y on atom 1, Z on atom 2 (the fake configuration's three atoms).
    nsearch = constant_config.eventsearch.nsearch
    assert len(searched) == 1
    assert sorted(set(searched[0])) == [0, 1, 2]
    assert len(searched[0]) == 3 * nsearch
    assert refined == [([5], set())]
    assert kmc.visited_environments == set(TOPOLOGIES)
    assert err_reference == [2] and err_ae == []
    assert any(
        "forgetting topologies ['X'] and searching again" in msg
        for _, msg in kmc.loggers.lines
    )


def test_real_re_search_leaving_the_catalogue_empty_exits_with_status_one(
    constant_config: Any, system_single_type_fcc: Any
) -> None:
    """(b) The purge empties the catalogue and the re-search finds nothing.

    The real search stage ends the run through its empty-catalogue check,
    the same "nothing found" condition as at the start of a step, before
    any refinement.
    """
    step = _Step(
        constant_config,
        system_single_type_fcc,
        fails=_first_attempt_fails,
        research_adds=[],
        stub_research=False,
    )
    kmc = step.kmc
    kmc.inactive_ae = None
    searched: list[list[int]] = []

    def search(atoms: list[int]) -> _NoSearch:
        searched.append(list(atoms))
        return _NoSearch()

    def refine(subset: pd.DataFrame, existing_pairs: Any = None) -> _Refined:
        raise AssertionError("no refinement after an empty catalogue")

    kmc.execute_event_searches = search
    kmc.execute_refinements = refine
    with pytest.raises(SystemExit) as raised:
        kmc.reconstruction(step.table)

    assert raised.value.code == 1
    assert len(searched) == 1
    assert step.attempts == [(10, 2)]
    assert any("empty reference events table" in msg for msg in step.errors())
    assert kmc.manager.shutdowns == 1
