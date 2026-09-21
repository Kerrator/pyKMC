"""Every k0 demotion inside ``_ensure_current_estimate`` is visible.

Five early-return paths set a reference row to ``stale`` or ``legacy`` (the
constant ``k0`` fallback) through ``_set_estimate``, which writes the table
only. A run whose service cannot rebuild a stored source would otherwise
convert every affected catalogued event to ``k0`` at selection time with
nothing above the aggregate ``stale=``/``legacy=`` counts. Each demotion
emits one WARNING naming ``idx_ref``, the new status and the reason; a
row already carrying that status and reason is not demoted again.
"""

from __future__ import annotations

import logging
import types
from typing import Any

import pandas as pd
import pytest

from pykmc.event_table import (
    NU0_LEGACY,
    NU0_OK,
    NU0_PENDING,
    NU0_STALE,
    ReferenceEventTable,
)
from pykmc.htst.request import HTSTRequestError
from tests.lifecycle.conftest import accepted
from tests.lifecycle.protocol_producers import protocol_patch, protocol_service


def _row(table: ReferenceEventTable, system: Any, idx_ref: int = 0) -> None:
    """Insert one pending row (id ``idx_ref``) built from the test crystal."""
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
    table.table = pd.concat([table.table, fwd.to_frame().T], ignore_index=True)


def _accepted_table(htst_config: Any, system: Any) -> ReferenceEventTable:
    """A table whose row 0 carries an accepted 5 THz estimate with provenance."""
    table = ReferenceEventTable(
        htst_config, prefactor_service=protocol_service(htst_config)
    )
    _row(table, system)
    table._protocol_source = system
    protocol_patch(table, 0, accepted(5.0e12))
    assert table.table.loc[0, "nu0_status"] == NU0_OK
    return table


def _demotions(caplog: Any) -> list[str]:
    return [
        r.getMessage()
        for r in caplog.records
        if r.levelno == logging.WARNING and "demoted" in r.getMessage()
    ]


def _assert_one_demotion(
    caplog: Any, table: ReferenceEventTable, status: str, fragment: str
) -> str:
    row = table.table.loc[0]
    assert row["nu0_status"] == status
    assert fragment in str(row["nu0_reason"])
    assert row["k_prefactor"] == float(table.config.rateconstant.k0)
    messages = _demotions(caplog)
    assert len(messages) == 1, [r.getMessage() for r in caplog.records]
    message = messages[0]
    assert "reference event 0" in message
    assert f"'{status}'" in message
    assert fragment in message
    return message


def test_missing_producing_calculation_demotes_to_legacy_once(
    htst_config: Any, system_single_type_fcc: Any, caplog: Any
) -> None:
    table = ReferenceEventTable(
        htst_config, prefactor_service=protocol_service(htst_config)
    )
    _row(table, system_single_type_fcc)
    assert table.table.loc[0, "nu0_status"] == NU0_PENDING
    with caplog.at_level(logging.INFO, logger="log"):
        table._ensure_current_estimate(0)
    _assert_one_demotion(caplog, table, NU0_LEGACY, "missing producing calculation")
    # Already legacy with that reason: a second pass is not a demotion.
    caplog.clear()
    with caplog.at_level(logging.INFO, logger="log"):
        table._ensure_current_estimate(0)
    assert _demotions(caplog) == []
    assert table.table.loc[0, "nu0_status"] == NU0_LEGACY


def test_missing_service_demotes_to_stale(
    htst_config: Any, system_single_type_fcc: Any, caplog: Any
) -> None:
    table = _accepted_table(htst_config, system_single_type_fcc)
    table.prefactor_service = None
    with caplog.at_level(logging.INFO, logger="log"):
        table._ensure_current_estimate(0)
    _assert_one_demotion(caplog, table, NU0_STALE, "missing current physical context")


def test_unbuildable_source_demotes_to_stale(
    htst_config: Any,
    system_single_type_fcc: Any,
    caplog: Any,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    table = _accepted_table(htst_config, system_single_type_fcc)

    def refuse(snapshot: Any, *, event_key: Any) -> Any:
        raise HTSTRequestError("species map cannot describe the stored source")

    monkeypatch.setattr(table.prefactor_service, "request_from_snapshot", refuse)
    with caplog.at_level(logging.INFO, logger="log"):
        table._ensure_current_estimate(0)
    message = _assert_one_demotion(
        caplog, table, NU0_STALE, "cannot rebuild complete current source"
    )
    assert "species map cannot describe the stored source" in message


def _stale_with_recompute(
    table: ReferenceEventTable, monkeypatch: pytest.MonkeyPatch, calculation: Any
) -> None:
    """Force the recompute path and make the worker answer ``calculation``."""
    service = table.prefactor_service
    # The patched row is "fresh" (its calculation was produced by this
    # service): drop that exemption so the physics comparison decides.
    table._fresh_calculations.clear()
    monkeypatch.setattr(
        table,
        "compare_physics",
        lambda descriptor: types.SimpleNamespace(
            status="incompatible", reasons=("test: producing physics differs",)
        ),
    )

    def compute(
        requests: Any, *, compute_backward: bool, compute_energies: bool
    ) -> Any:
        request = requests[0]
        return {
            request.event_key: types.SimpleNamespace(
                calculation=lambda direction: calculation(request)
            )
        }

    monkeypatch.setattr(service, "compute", compute)
    monkeypatch.setattr(
        service, "calculation_context_matches", lambda calc, request: True
    )


def test_recomputation_without_a_calculation_demotes_to_stale(
    htst_config: Any,
    system_single_type_fcc: Any,
    caplog: Any,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    table = _accepted_table(htst_config, system_single_type_fcc)
    _stale_with_recompute(table, monkeypatch, lambda request: None)
    with caplog.at_level(logging.INFO, logger="log"):
        table._ensure_current_estimate(0)
    _assert_one_demotion(
        caplog, table, NU0_STALE, "worker returned no producing calculation"
    )


def test_recomputation_without_energies_demotes_to_stale(
    htst_config: Any,
    system_single_type_fcc: Any,
    caplog: Any,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from pykmc.htst.provenance import RequestSnapshot

    table = _accepted_table(htst_config, system_single_type_fcc)
    service = table.prefactor_service

    def energyless(request: Any) -> Any:
        return types.SimpleNamespace(
            provenance=types.SimpleNamespace(
                source=RequestSnapshot.capture(request),
                method=service.method,
                energies=None,
            )
        )

    _stale_with_recompute(table, monkeypatch, energyless)
    with caplog.at_level(logging.INFO, logger="log"):
        table._ensure_current_estimate(0)
    _assert_one_demotion(
        caplog, table, NU0_STALE, "lacks current full-system potential energies"
    )
