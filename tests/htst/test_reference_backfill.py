"""ReferenceEventTable batch nu0 backfill: accepted events only, rows patched.

The table builds each accepted event with a k0 placeholder, collects
full-geometry payloads during the add loop, fans ONE batch through
``rate_constant.compute_prefactors_batch`` (htst/rpa only), and patches
``k``/``k_prefactor``/``nu0`` on the rows by ``idx_ref`` mask. Rejected and
duplicate events never cost a payload. Constant style never touches the
manager. No LAMMPS, no MPI — a FakeManager resolves the futures.
"""

from __future__ import annotations

import logging
import math
from concurrent.futures import Future
from typing import Any

import numpy as np
import pandas as pd
import pytest
from pytest import MonkeyPatch

from pykmc.event_table import ReferenceEventTable
from pykmc.rate_constant.prefactor import EventPrefactors
from pykmc.result import EventSearchOutput


def _prefactors(nu0_f: "float | None", nu0_b: "float | None") -> EventPrefactors:
    return EventPrefactors(
        nu0_forward=nu0_f,
        nu0_backward=nu0_b,
        n_free=5,
        n_neg_saddle=1,
        ok_forward=nu0_f is not None,
        ok_backward=nu0_b is not None,
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


def _patch_numpy_nl(monkeypatch: MonkeyPatch) -> None:
    """NeighborsList lists -> numpy arrays (same workaround as the nu0 tests)."""
    import pykmc.event_table as _et
    from pykmc.neighbors_list import NeighborsList as _RealNL

    class _NumpyNL(_RealNL):
        def __init__(self, *args: Any, **kwargs: Any) -> None:
            super().__init__(*args, **kwargs)
            for key in ("rnei", "rcut"):
                if key in self.neighbors_list:
                    self.neighbors_list[key] = [
                        np.array(lst, dtype=np.int64)
                        for lst in self.neighbors_list[key]
                    ]

    monkeypatch.setattr(_et, "NeighborsList", _NumpyNL)


def _mid_barrier(config: Any) -> float:
    emin = config.eventsearch.emin_event
    emax = config.eventsearch.emax_event
    return 0.5 * (emin + emax)


def _event(system: Any, config: Any, dE: "float | None" = None) -> EventSearchOutput:
    """Trivial self-backward event (min1 == saddle == min2) at atom 0."""
    de = _mid_barrier(config) if dE is None else dE
    return EventSearchOutput(
        central_atom_index=0,
        min1_positions=system.positions,
        saddle_positions=system.positions,
        min2_positions=system.positions,
        dE_forward=de,
        dE_backward=de,
        move_atom_index=0,
        cell=system.cell,
    )


def test_constant_style_never_calls_manager(
    config_Ni_4000at_monovacancy_sia: Any,
    system_single_type_fcc: Any,
    monkeypatch: MonkeyPatch,
) -> None:
    """Constant style: rows keep k0, the manager fan-out is never invoked."""
    _patch_numpy_nl(monkeypatch)
    config = config_Ni_4000at_monovacancy_sia  # style=constant
    fake = FakeManager([])
    table = ReferenceEventTable(config, manager=fake)
    sys_ = system_single_type_fcc

    table.add_events([_event(sys_, config)], types=list(sys_.types))

    assert fake.calls == []
    assert len(table.table) == 1
    assert table.table.iloc[0]["k_prefactor"] == config.rateconstant.k0


def test_htst_backfills_accepted_event(
    config_Ni_4000at_monovacancy_sia: Any,
    system_single_type_fcc: Any,
    monkeypatch: MonkeyPatch,
) -> None:
    """Accepted event gets nu0/k/k_prefactor patched from the batch result."""
    _patch_numpy_nl(monkeypatch)
    config = config_Ni_4000at_monovacancy_sia
    config.rateconstant.style = "htst"
    fake = FakeManager([_prefactors(5.0e12, 3.0e12)])
    table = ReferenceEventTable(config, manager=fake)
    sys_ = system_single_type_fcc
    de = _mid_barrier(config)

    table.add_events([_event(sys_, config)], types=list(sys_.types))

    assert len(fake.calls) == 1
    (cfg_used, payloads), = fake.calls
    assert cfg_used is config  # the FULL config, not the sub-config
    assert len(payloads) == 1
    assert payloads[0]["central_atom_idx"] == 0
    assert payloads[0]["min1_positions"] is not None
    row = table.table.iloc[0]
    assert row["nu0"] == 5.0e12  # nu0 column stays Hz (diagnostic; pylatkmc consumer)
    assert row["k_prefactor"] == 5.0e12 * 1e-12  # resolved prefactor is ps^-1
    assert math.isclose(row["k"], table.rate_constant.compute_rate(de, nu0=5.0e12).rate)


def test_htst_fallback_keeps_k0_and_logs(
    config_Ni_4000at_monovacancy_sia: Any,
    system_single_type_fcc: Any,
    monkeypatch: MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    """nu0=None from the batch -> row keeps the k0 placeholder + a fallback log."""
    _patch_numpy_nl(monkeypatch)
    config = config_Ni_4000at_monovacancy_sia
    config.rateconstant.style = "htst"
    fake = FakeManager([_prefactors(None, None)])
    table = ReferenceEventTable(config, manager=fake)
    sys_ = system_single_type_fcc

    with caplog.at_level(logging.INFO, logger="pykmc.event_table"):
        table.add_events([_event(sys_, config)], types=list(sys_.types))

    row = table.table.iloc[0]
    assert row["k_prefactor"] == config.rateconstant.k0
    assert row["nu0"] is None or (isinstance(row["nu0"], float) and math.isnan(row["nu0"]))
    assert any("fallback" in r.message for r in caplog.records)


def test_backfill_only_accepted_events(
    config_Ni_4000at_monovacancy_sia: Any,
    system_single_type_fcc: Any,
    monkeypatch: MonkeyPatch,
) -> None:
    """{accepted, energy-rejected, duplicate} -> exactly ONE payload in ONE batch."""
    _patch_numpy_nl(monkeypatch)
    config = config_Ni_4000at_monovacancy_sia
    config.rateconstant.style = "htst"
    fake = FakeManager([_prefactors(5.0e12, None)])
    table = ReferenceEventTable(config, manager=fake)
    sys_ = system_single_type_fcc
    emax = config.eventsearch.emax_event

    accepted = _event(sys_, config)
    rejected = _event(sys_, config, dE=emax * 2.0)  # energy gate
    duplicate = _event(sys_, config)  # same topology + energy as `accepted`

    results = table.add_events(
        [accepted, rejected, duplicate], types=list(sys_.types)
    )

    assert [r.is_ok() for r in results] == [True, False, False]
    assert len(fake.calls) == 1
    assert len(fake.calls[0][1]) == 1  # only the accepted event cost a payload


def test_htst_requires_types(
    config_Ni_4000at_monovacancy_sia: Any,
    system_single_type_fcc: Any,
    monkeypatch: MonkeyPatch,
) -> None:
    """Htst add_events without types is a misconfiguration -> RuntimeError."""
    _patch_numpy_nl(monkeypatch)
    config = config_Ni_4000at_monovacancy_sia
    config.rateconstant.style = "htst"
    table = ReferenceEventTable(config, manager=FakeManager([_prefactors(5e12, None)]))

    with pytest.raises(RuntimeError, match="types"):
        table.add_events([_event(system_single_type_fcc, config)])


def test_backfill_patches_directional_rows(
    config_Ni_4000at_monovacancy_sia: Any,
    system_single_type_fcc: Any,
    monkeypatch: MonkeyPatch,
) -> None:
    """Forward row gets nu0_forward, backward row gets nu0_backward (by idx_ref)."""
    _patch_numpy_nl(monkeypatch)
    config = config_Ni_4000at_monovacancy_sia
    config.rateconstant.style = "htst"
    fake = FakeManager([_prefactors(5.0e12, 3.0e12)])
    table = ReferenceEventTable(config, manager=fake)
    sys_ = system_single_type_fcc
    positions = sys_.positions

    # Build a 2-row (forward+backward) event directly and add it; then drive the
    # private backfill with explicit refs, the way add_events does internally.
    fwd, bwd = table._build_event_series(
        min1_positions=positions,
        saddle_positions=positions,
        min2_positions=positions,
        index_move=0,
        dE_forward=0.5,
        dE_backward=0.7,
        cell=sys_.cell,
    )
    df = pd.concat([fwd.to_frame().T, bwd.to_frame().T], ignore_index=True)
    table.add(df)
    fwd_ref = int(df.iloc[0]["idx_ref"])
    bwd_ref = int(df.iloc[1]["idx_ref"])

    table._backfill_prefactors([(fwd_ref, bwd_ref, {"central_atom_idx": 0})])

    fwd_row = table.table[table.table["idx_ref"] == fwd_ref].iloc[0]
    bwd_row = table.table[table.table["idx_ref"] == bwd_ref].iloc[0]
    assert fwd_row["nu0"] == 5.0e12  # nu0 column stays Hz (diagnostic)
    assert fwd_row["k_prefactor"] == 5.0e12 * 1e-12  # resolved prefactor is ps^-1
    assert math.isclose(fwd_row["k"], table.rate_constant.compute_rate(0.5, nu0=5.0e12).rate)
    assert bwd_row["nu0"] == 3.0e12  # nu0 column stays Hz (diagnostic)
    assert bwd_row["k_prefactor"] == 3.0e12 * 1e-12  # resolved prefactor is ps^-1
    assert math.isclose(bwd_row["k"], table.rate_constant.compute_rate(0.7, nu0=3.0e12).rate)
