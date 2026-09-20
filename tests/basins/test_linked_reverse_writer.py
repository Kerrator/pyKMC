"""The detector resolves the reverse the production link writer actually wrote.

Both catalogues are built through ``ReferenceEventTable.add_events`` (no
hand-written ``idx_backward``). In constant mode the writer self-links a
forward whose reverse is already catalogued (``event_table.add`` with
``reverse_idx_ref=None``); that placeholder must resolve to the catalogued
reciprocal reverse row (initial topology of the forward's final one AND final
topology of the forward's initial one), never to the forward itself and never
to another channel that merely leaves the forward's final topology. In htst
mode the reciprocal pair carries explicit links. Strict threshold equality
stays absorbing.
"""

from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import pandas as pd
import pytest

import pykmc
from pykmc.basins.detection import DetectorThreshold, resolve_linked_pair
from pykmc.basins.exploration import BasinGenericEventExplorer
from pykmc.config import Config, RateConstantConfig
from pykmc.event_table import ReferenceEventTable
from pykmc.rate_constant import create_rate_constant
from pykmc.rate_constant.prefactors import PrefactorService
from pykmc.result import EventSearchOutput
from tests.lifecycle.conftest import FakeManager, accepted
from tests.lifecycle.protocol_producers import protocol_event_prefactors

HOP = np.array([1.2, 0.3, 0.0])
_INPUT = Path(pykmc.__file__).resolve().parent.parent / "tests" / "data" / "input.in"


@pytest.fixture
def constant_config() -> Config:
    """The committed test input (constant style)."""
    return Config.from_ini_file(str(_INPUT))


@pytest.fixture
def htst_config(constant_config: Config) -> Config:
    """The committed test input switched to htst with ``k0 = 1.0``."""
    rate = RateConstantConfig(style="htst", k0=1.0, T=constant_config.rateconstant.T)
    return constant_config.model_copy(update={"rateconstant": rate})


def _event(system: Any, move: int, dE_forward: float, dE_backward: float):
    """Search output on ``system`` moving ``move`` by ``HOP`` (saddle halfway)."""
    pos = np.asarray(system.positions, dtype=float)
    min2 = pos.copy()
    saddle = pos.copy()
    min2[move] += HOP
    saddle[move] += 0.5 * HOP
    return EventSearchOutput(
        central_atom_index=move,
        min1_positions=pos.copy(),
        saddle_positions=saddle,
        min2_positions=min2,
        dE_forward=dE_forward,
        dE_backward=dE_backward,
        move_atom_index=move,
        cell=np.asarray(system.cell, dtype=float),
        types=list(system.types),
    )


def _row(table: ReferenceEventTable, idx_ref: int) -> pd.Series:
    rows = table.table[table.table["idx_ref"] == idx_ref]
    assert len(rows) == 1, idx_ref
    return rows.iloc[0]


def _assert_detector_contract(frame: pd.DataFrame, forward: pd.Series) -> None:
    """Forward 0.60 or 0.30 with a catalogued reverse of 0.90 eV."""
    detector = DetectorThreshold()
    resolved_forward, reverse = resolve_linked_pair(forward, frame)
    assert int(resolved_forward["idx_ref"]) == int(forward["idx_ref"])
    assert int(reverse["idx_ref"]) != int(forward["idx_ref"]), (
        "the physical reverse is another catalogued row, not the forward itself"
    )
    assert reverse["event_id"] == forward["id_final"]
    assert reverse["energy_barrier"] == 0.90
    # Low forward, high reverse: absorbing, whatever the forward barrier says.
    assert not detector.detect(forward, frame, 0.7)
    # Both below the threshold: transient.
    assert detector.detect(forward, frame, 0.95)
    # Strict equality at the reverse barrier is outside the basin.
    assert not detector.detect(forward, frame, 0.90)
    active = pd.Series(
        {"num_reference_event": int(forward["idx_ref"]), "energy_barrier": 0.55}
    )
    assert not detector.detect(active, frame, 0.7, is_refined=True)
    assert detector.detect(active, frame, 0.95, is_refined=True)


def _explorer_edge(frame: pd.DataFrame, forward: pd.Series, threshold: float):
    subset = frame[frame["idx_ref"] == int(forward["idx_ref"])]
    reference = SimpleNamespace(table=frame, has_id_subset_table=lambda ids: subset)
    environment = SimpleNamespace(
        atomic_environment_list=[forward["event_id"]],
        get_atoms_with_id=lambda topology: [0],
    )
    explorer = BasinGenericEventExplorer(
        SimpleNamespace(basin=SimpleNamespace(energy_thr=threshold)), reference
    )
    explorer.explore(
        SimpleNamespace(environment=environment), state_index=2, start_index=11
    )
    edges = explorer.get_connectivity_table()
    # One edge per recorded symmetry of the catalogued environment; every edge
    # carries the same linked reverse.
    assert len(edges) == len(forward["sym_matrix"]) >= 1
    assert edges["dE_backward"].nunique() == 1 and edges["transient"].nunique() == 1
    return edges.iloc[0]


def constant_placeholder(constant_config: Any, system: Any):
    """Admit a reciprocal pair, then a forward whose reverse is already known."""
    table = ReferenceEventTable(constant_config)
    first = table.add_events([_event(system, 0, 0.30, 0.90)], pbc=system.pbc)
    assert first[0].is_ok() and len(first[0].ok_value()) == 2
    pair = _row(table, 0), _row(table, 1)
    assert pair[0]["event_id"] != pair[0]["id_final"], (
        "the hop must change the mover's topology for this control"
    )
    assert int(pair[0]["idx_backward"]) == 1 and int(pair[1]["idx_backward"]) == 0
    # Same geometry, forward barrier outside the 0.25 eV duplicate window,
    # backward inside it: the production writer admits the forward alone and
    # self-links it (its reverse is the catalogued row 1).
    second = table.add_events([_event(system, 0, 0.60, 1.00)], pbc=system.pbc)
    assert second[0].is_ok()
    admitted = second[0].ok_value()
    assert len(admitted) == 1
    forward = _row(table, int(admitted.iloc[0]["idx_ref"]))
    assert int(forward["idx_backward"]) == int(forward["idx_ref"])
    assert forward["event_id"] != forward["id_final"]
    assert forward["energy_barrier"] == 0.60
    return table, forward, pair


def test_constant_placeholder_self_link_resolves_the_catalogued_reverse(
    constant_config: Any, system_single_type_fcc: Any
) -> None:
    table, forward, _ = constant_placeholder(constant_config, system_single_type_fcc)
    before = table.table.copy(deep=True)
    _assert_detector_contract(table.table, forward)
    edge = _explorer_edge(table.table, forward, 0.7)
    assert edge.dE_forward == 0.60 and edge.dE_backward == 0.90
    assert bool(edge.transient) is False
    pd.testing.assert_frame_equal(table.table, before)


def test_constant_reciprocal_pair_follows_its_written_link(
    constant_config: Any, system_single_type_fcc: Any
) -> None:
    table, _, (forward, reverse) = constant_placeholder(
        constant_config, system_single_type_fcc
    )
    assert forward["energy_barrier"] == 0.30 and reverse["energy_barrier"] == 0.90
    _assert_detector_contract(table.table, forward)
    edge = _explorer_edge(table.table, forward, 0.95)
    assert edge.dE_forward == 0.30 and edge.dE_backward == 0.90
    assert bool(edge.transient) is True


def _third_channel(table: ReferenceEventTable, system: Any, forward: pd.Series):
    """Admit ``B -> C`` through the writer: it leaves the placeholder's final
    topology ``B`` for a third topology ``C`` with a barrier below the
    reciprocal ``B -> A`` reverse (0.90 eV)."""
    pos = np.asarray(system.positions, dtype=float)
    side = np.array([0.0, 1.2, 0.3])
    b = pos.copy()
    b[0] += HOP
    saddle = b.copy()
    saddle[0] += 0.5 * side
    c = b.copy()
    c[0] += side
    result = table.add_events(
        [
            EventSearchOutput(
                central_atom_index=0,
                min1_positions=b,
                saddle_positions=saddle,
                min2_positions=c,
                dE_forward=0.20,
                dE_backward=0.95,
                move_atom_index=0,
                cell=np.asarray(system.cell, dtype=float),
                types=list(system.types),
            )
        ],
        pbc=system.pbc,
    )
    assert result[0].is_ok() and len(result[0].ok_value()) == 2
    third = _row(table, int(result[0].ok_value().iloc[0]["idx_ref"]))
    assert third["energy_barrier"] == 0.20
    assert third["event_id"] == forward["id_final"], "B -> C starts from B"
    assert third["id_final"] not in {forward["event_id"], forward["id_final"]}, (
        "C is a third topology, so B -> C is not the reverse of A -> B"
    )
    return third


def test_constant_placeholder_ignores_non_reciprocal_channels_from_its_final_topology(
    constant_config: Any, system_single_type_fcc: Any
) -> None:
    """The lowest barrier leaving ``B`` is not the reverse of ``A -> B``."""
    table, forward, (_, reciprocal) = constant_placeholder(
        constant_config, system_single_type_fcc
    )
    third = _third_channel(table, system_single_type_fcc, forward)
    assert third["energy_barrier"] < reciprocal["energy_barrier"]
    before = table.table.copy(deep=True)
    _, reverse = resolve_linked_pair(forward, table.table)
    assert int(reverse["idx_ref"]) == int(reciprocal["idx_ref"])
    assert reverse["id_final"] == forward["event_id"], "the reverse returns to A"
    assert reverse["energy_barrier"] == 0.90
    _assert_detector_contract(table.table, forward)
    # Forward 0.60 below 0.7, reciprocal reverse 0.90 above it: absorbing;
    # following B -> C (0.20) would wrongly declare the state transient.
    assert not DetectorThreshold().detect(forward, table.table, 0.7)
    edge = _explorer_edge(table.table, forward, 0.7)
    assert edge.dE_forward == 0.60 and edge.dE_backward == 0.90
    assert edge.k_backward == reciprocal["k"]
    assert bool(edge.transient) is False
    pd.testing.assert_frame_equal(table.table, before)


def test_constant_placeholder_with_only_non_reciprocal_channels_is_explicit(
    constant_config: Any, system_single_type_fcc: Any
) -> None:
    """Rows leaving ``B`` elsewhere do not stand in for a missing ``B -> A``."""
    table, forward, (_, reciprocal) = constant_placeholder(
        constant_config, system_single_type_fcc
    )
    _third_channel(table, system_single_type_fcc, forward)
    frame = table.table[table.table["idx_ref"] != int(reciprocal["idx_ref"])].copy()
    assert (frame["event_id"] == forward["id_final"]).sum() == 1, (
        "B -> C is still catalogued; only the reciprocal B -> A is gone"
    )
    with pytest.raises(ValueError) as captured:
        DetectorThreshold().detect(forward, frame, 0.95)
    assert str(int(forward["idx_ref"])) in str(captured.value)
    assert str(forward["id_final"]) in str(captured.value)


def test_constant_placeholder_without_any_catalogued_reverse_is_explicit(
    constant_config: Any, system_single_type_fcc: Any
) -> None:
    table, forward, _ = constant_placeholder(constant_config, system_single_type_fcc)
    frame = table.table[table.table["idx_ref"] == int(forward["idx_ref"])].copy()
    with pytest.raises(ValueError) as captured:
        DetectorThreshold().detect(forward, frame, 0.95)
    assert str(int(forward["idx_ref"])) in str(captured.value)
    assert str(forward["id_final"]) in str(captured.value)


def test_htst_reciprocal_pair_follows_its_written_link(
    htst_config: Any, system_single_type_fcc: Any
) -> None:
    responses = {(0, 1): (accepted(5.0e12), accepted(3.0e12))}
    fake = FakeManager(
        lambda req: protocol_event_prefactors(req, *responses[req.event_key])
    )
    service = PrefactorService(
        htst_config, fake, create_rate_constant(htst_config.rateconstant), method="fd"
    )
    table = ReferenceEventTable(htst_config, prefactor_service=service)
    system = system_single_type_fcc
    results = table.add_events([_event(system, 0, 0.30, 0.90)], pbc=system.pbc)
    assert results[0].is_ok() and len(results[0].ok_value()) == 2
    forward, reverse = _row(table, 0), _row(table, 1)
    assert int(forward["idx_backward"]) == 1 and int(reverse["idx_backward"]) == 0
    _assert_detector_contract(table.table, forward)
    edge = _explorer_edge(table.table, forward, 0.7)
    assert edge.dE_backward == 0.90 and bool(edge.transient) is False
