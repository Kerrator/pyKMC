"""ActiveEventTable refined-event nu0 backfill (post-dedup, refined rows only).

Team point: prefactors must be computed for REFINED events, after the active
table is built and deduplicated, so duplicates and improbable events never
cost a Hessian. Rows with ``refined == "T"`` (the e_thr-gated, probable ones)
get a site-specific nu0 at the refined saddle through ONE batch; ``"F"``/"B"
rows keep the values inherited from the reference table. No LAMMPS, no MPI.
"""

from __future__ import annotations

import logging
import math
from concurrent.futures import Future
from typing import Any

import numpy as np
import pytest

from pykmc.event_table import ActiveEventTable
from pykmc.rate_constant.prefactor import EventPrefactors
from pykmc.result import EventRefinementOutput


def _prefactors(nu0_f: "float | None") -> EventPrefactors:
    return EventPrefactors(
        nu0_forward=nu0_f,
        nu0_backward=None,
        n_free=5,
        n_neg_saddle=1,
        ok_forward=nu0_f is not None,
        ok_backward=False,
        reason="" if nu0_f is not None else "test fallback",
    )


class FakeManager:
    """Record fan-out calls; return pre-resolved futures (one per payload)."""

    def __init__(self, results: list[EventPrefactors]) -> None:
        self.calls: list[tuple[Any, list[dict[str, Any]]]] = []
        self._results = results

    def compute_event_prefactors(
        self, config: Any, events: list[dict[str, Any]]
    ) -> list[Future]:
        """Record the call and return one resolved future per event."""
        self.calls.append((config, list(events)))
        futures: list[Future] = []
        for pre in self._results[: len(events)]:
            f: Future = Future()
            f.set_result(pre)
            futures.append(f)
        return futures


class _FakeNL:
    """Neighbors-list stub: a fixed rcut neighborhood per atom."""

    def __init__(self, neighbors: dict[int, np.ndarray]) -> None:
        self._neighbors = neighbors

    def get_neighbors(self, cutoff_type: str, idx: int) -> np.ndarray:
        """Return the canned neighbor indices for this atom."""
        return self._neighbors[idx]


def _refined_event(
    system: Any,
    neighbors: np.ndarray,
    atom_index: int,
    refined: str,
    nu0_inherited: float,
    de: float = 0.5,
) -> EventRefinementOutput:
    """Build a refinement output with neighbor-cropped geometry, as production does."""
    saddle_crop = system.positions[neighbors] + 0.1  # displaced refined saddle
    final_crop = system.positions[neighbors] + 0.2
    out = EventRefinementOutput(
        central_atom_index=atom_index,
        saddle_positions=saddle_crop,
        E_saddle=de,
        refined=refined,
    )
    out.min2_positions = final_crop
    out.num_reference_event = 0
    out.k_prefactor = nu0_inherited
    out.nu0 = nu0_inherited
    out.dE_forward = de
    out.neighbors = neighbors  # persisted ordering backfill now reads from the row
    return out


@pytest.fixture
def setup(
    config_Ni_4000at_monovacancy_sia: Any, system_single_type_fcc: Any
) -> "tuple[Any, Any, _FakeNL, np.ndarray]":
    """Htst config + system + a stub neighbors list around atoms 0 and 1."""
    config = config_Ni_4000at_monovacancy_sia
    config.rateconstant.style = "htst"
    sys_ = system_single_type_fcc
    neighbors = np.arange(0, 6, dtype=np.int64)
    nl = _FakeNL({0: neighbors, 1: neighbors})
    return config, sys_, nl, neighbors


def test_refined_rows_patched_f_rows_untouched(setup: Any) -> None:
    """refined=T gets site nu0 + recomputed k; refined=F keeps inherited values."""
    config, sys_, nl, neighbors = setup
    fake = FakeManager([_prefactors(5.0e12)])
    table = ActiveEventTable(config, manager=fake)
    table.add_events(
        [
            _refined_event(sys_, neighbors, 0, "T", nu0_inherited=7.0e11),
            _refined_event(sys_, neighbors, 1, "F", nu0_inherited=7.0e11),
        ]
    )
    k_f_before = table.table.iloc[1]["k"]

    table.backfill_refined_prefactors(sys_, nl)

    assert len(fake.calls) == 1
    assert len(fake.calls[0][1]) == 1  # only the refined=T row cost a payload
    t_row = table.table.iloc[0]
    assert t_row["nu0"] == 5.0e12
    assert math.isclose(
        t_row["k"], table.rate_constant.compute_rate(0.5, nu0=5.0e12).rate
    )
    f_row = table.table.iloc[1]
    assert f_row["nu0"] == 7.0e11  # inherited, untouched
    assert f_row["k"] == k_f_before


def test_payload_reconstructs_full_geometry(setup: Any) -> None:
    """The payload saddle is the full system with the crop written into neighbors."""
    config, sys_, nl, neighbors = setup
    fake = FakeManager([_prefactors(5.0e12)])
    table = ActiveEventTable(config, manager=fake)
    table.add_events([_refined_event(sys_, neighbors, 0, "T", nu0_inherited=7.0e11)])

    table.backfill_refined_prefactors(sys_, nl)

    payload = fake.calls[0][1][0]
    full_saddle = payload["saddle_positions"]
    full_min2 = payload["min2_positions"]
    n = len(sys_.positions)
    assert full_saddle.shape == (n, 3)
    outside = np.setdiff1d(np.arange(n), neighbors)
    assert np.allclose(full_saddle[outside], sys_.positions[outside])
    assert np.allclose(full_saddle[neighbors], sys_.positions[neighbors] + 0.1)
    assert np.allclose(full_min2[neighbors], sys_.positions[neighbors] + 0.2)
    assert np.allclose(payload["min1_positions"], sys_.positions)
    assert payload["central_atom_idx"] == 0


def test_fallback_keeps_inherited_and_logs(
    setup: Any, caplog: pytest.LogCaptureFixture
) -> None:
    """nu0=None from the batch -> refined row keeps inherited values + a log."""
    config, sys_, nl, neighbors = setup
    fake = FakeManager([_prefactors(None)])
    table = ActiveEventTable(config, manager=fake)
    table.add_events([_refined_event(sys_, neighbors, 0, "T", nu0_inherited=7.0e11)])
    k_before = table.table.iloc[0]["k"]

    with caplog.at_level(logging.INFO, logger="pykmc.event_table"):
        table.backfill_refined_prefactors(sys_, nl)

    row = table.table.iloc[0]
    assert row["nu0"] == 7.0e11
    assert row["k"] == k_before
    assert any("fallback" in r.message for r in caplog.records)


def test_constant_style_is_noop(
    config_Ni_4000at_monovacancy_sia: Any, system_single_type_fcc: Any
) -> None:
    """Constant style: the backfill is a no-op and never touches the manager."""
    config = config_Ni_4000at_monovacancy_sia  # style=constant
    fake = FakeManager([])
    table = ActiveEventTable(config, manager=fake)
    sys_ = system_single_type_fcc
    nl = _FakeNL({0: np.arange(0, 6, dtype=np.int64)})

    table.backfill_refined_prefactors(sys_, nl)

    assert fake.calls == []


def test_recycled_rows_are_not_recomputed(setup: Any) -> None:
    """Rows that already got a site-specific nu0 never cost a second Hessian.

    Simulates event recycling (the active table persists across steps and
    refined rows survive pruning): a second backfill pass must submit ZERO
    payloads for already-computed rows.
    """
    config, sys_, nl, neighbors = setup
    fake = FakeManager([_prefactors(5.0e12), _prefactors(9.0e12)])
    table = ActiveEventTable(config, manager=fake)
    table.add_events([_refined_event(sys_, neighbors, 0, "T", nu0_inherited=7.0e11)])

    table.backfill_refined_prefactors(sys_, nl)  # step N: computes
    table.backfill_refined_prefactors(sys_, nl)  # step N+1: row was recycled

    assert len(fake.calls) == 1  # no second fan-out
    assert table.table.iloc[0]["nu0"] == 5.0e12  # first result kept


def test_fallback_rows_are_not_retried(setup: Any) -> None:
    """A failed nu0 is attempted once, not re-attempted every recycled step."""
    config, sys_, nl, neighbors = setup
    fake = FakeManager([_prefactors(None), _prefactors(5.0e12)])
    table = ActiveEventTable(config, manager=fake)
    table.add_events([_refined_event(sys_, neighbors, 0, "T", nu0_inherited=7.0e11)])

    table.backfill_refined_prefactors(sys_, nl)
    table.backfill_refined_prefactors(sys_, nl)

    assert len(fake.calls) == 1
    assert table.table.iloc[0]["nu0"] == 7.0e11  # inherited value kept
