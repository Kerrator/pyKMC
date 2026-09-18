"""Active-table prefactor lifecycle (architecture rules 2-4).

Inheritance from the reference estimate, one site request per newly accepted
refined row after dedup, success override, rejection fallback hierarchy, one
attempt per row, recycled rows untouched, and the k/k_prefactor/nu0/status
consistency invariant. No LAMMPS, no MPI.
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np
import pytest

from pykmc import NeighborsList
from pykmc.event_table import (
    ACTIVE_BASE_COLUMNS,
    ACTIVE_HTST_COLUMNS,
    ActiveEventTable,
)
from pykmc.rate_constant import create_rate_constant, rate_from_prefactor
from pykmc.rate_constant.prefactors import PrefactorService
from pykmc.result import EventRefinementOutput
from tests.lifecycle.conftest import (
    FakeManager,
    accepted,
    event_prefactors,
    rejected,
)


@pytest.fixture
def neighbors_list(system_single_type_fcc: Any, htst_config: Any) -> NeighborsList:
    """Build the rcut neighbour list refinement would crop with."""
    return NeighborsList(
        system_single_type_fcc,
        htst_config.atomicenvironment.rnei,
        htst_config.atomicenvironment.rcut,
    )


def _refined(
    system: Any,
    neighbors_list: NeighborsList,
    atom: int,
    refined: str = "T",
    dE: float = 0.5,
    ref: int = 0,
    **estimate: Any,
) -> EventRefinementOutput:
    """Refinement output with neighbour-cropped geometry, as production builds it."""
    neighbors = np.asarray(neighbors_list.get_neighbors("rcut", atom), dtype=int)
    pos = np.asarray(system.positions, dtype=float)
    return EventRefinementOutput(
        central_atom_index=atom,
        saddle_positions=pos[neighbors] + 0.1,
        E_saddle=dE,
        min2_positions=pos[neighbors] + 0.2,
        dE_forward=dE,
        num_reference_event=ref,
        refined=refined,
        **estimate,
    )


def _service(config: Any, responder: Any) -> tuple[PrefactorService, FakeManager]:
    fake = FakeManager(responder)
    return (
        PrefactorService(config, fake, create_rate_constant(config.rateconstant)),
        fake,
    )


def _site_ok(nu0: float) -> Any:
    return lambda req: event_prefactors(req.event_key, accepted(nu0), rejected("n/a"))


def _site_rejected(reason: str = "saddle_not_first_order") -> Any:
    return lambda req: event_prefactors(
        req.event_key, rejected(reason), rejected(reason)
    )


def _assert_consistent(table: ActiveEventTable, config: Any) -> None:
    """k, k_prefactor, nu0 and nu0_status always agree on every row."""
    k0 = config.rateconstant.k0
    T = config.rateconstant.T
    for _, row in table.table.iterrows():
        assert row["k"] == rate_from_prefactor(
            row["k_prefactor"], row["energy_barrier"], T
        )
        if row["nu0_status"] == "ok":
            assert row["k_prefactor"] == row["nu0"] * 1.0e-12
            assert row["nu0_reason"] == ""
            assert row["nu0_source"] in ("reference", "site")
        else:
            assert row["k_prefactor"] == k0
            assert math.isnan(row["nu0"])
            assert row["nu0_source"] == "k0"


class TestSchema:
    """Constant active tables keep 7 columns; htst/rpa append six."""

    def test_constant_schema(self, constant_config: Any) -> None:
        """Exactly the baseline 7 columns."""
        assert list(ActiveEventTable(constant_config).table.columns) == list(
            ACTIVE_BASE_COLUMNS
        )
        assert len(ACTIVE_BASE_COLUMNS) == 7

    def test_htst_schema(self, htst_config: Any) -> None:
        """Baseline 7 plus the six prefactor columns."""
        assert list(ActiveEventTable(htst_config).table.columns) == list(
            ACTIVE_BASE_COLUMNS
        ) + list(ACTIVE_HTST_COLUMNS)
        assert ACTIVE_HTST_COLUMNS == (
            "k_prefactor",
            "nu0",
            "nu0_status",
            "nu0_reason",
            "nu0_source",
            "nu0_site_attempted",
        )


class TestInheritance:
    """A refined event inherits its reference estimate, rated at its own barrier."""

    def test_accepted_reference_estimate_is_inherited(
        self, htst_config: Any, system_single_type_fcc: Any, neighbors_list: Any
    ) -> None:
        """Source 'reference', k recomputed from the inherited prefactor."""
        table = ActiveEventTable(htst_config)
        table.add_events(
            _refined(
                system_single_type_fcc,
                neighbors_list,
                0,
                dE=0.6,
                nu0_hz=7.0e11,
                nu0_status="ok",
                nu0_reason="",
                nu0_source="reference",
            )
        )
        row = table.table.iloc[0]
        T = htst_config.rateconstant.T
        assert row["k_prefactor"] == 0.7
        assert row["nu0"] == 7.0e11
        assert row["nu0_status"] == "ok"
        assert row["nu0_source"] == "reference"
        assert bool(row["nu0_site_attempted"]) is False
        assert row["k"] == rate_from_prefactor(0.7, 0.6, T)
        _assert_consistent(table, htst_config)

    @pytest.mark.parametrize(
        "status,reason",
        [("rejected", "out_of_window: x"), ("legacy", "legacy table"), ("pending", "")],
    )
    def test_non_accepted_reference_estimate_resolves_to_k0(
        self,
        status: str,
        reason: str,
        htst_config: Any,
        system_single_type_fcc: Any,
        neighbors_list: Any,
    ) -> None:
        """Any non-ok inherited status keeps its label and uses k0."""
        table = ActiveEventTable(htst_config)
        table.add_events(
            _refined(
                system_single_type_fcc,
                neighbors_list,
                0,
                nu0_hz=None,
                nu0_status=status,
                nu0_reason=reason,
                nu0_source="reference",
            )
        )
        row = table.table.iloc[0]
        assert row["k_prefactor"] == htst_config.rateconstant.k0
        assert math.isnan(row["nu0"])
        assert row["nu0_status"] == status
        assert row["nu0_reason"] == reason
        assert row["nu0_source"] == "k0"
        _assert_consistent(table, htst_config)

    def test_no_inherited_estimate_resolves_to_k0(
        self, htst_config: Any, system_single_type_fcc: Any, neighbors_list: Any
    ) -> None:
        """An output without any estimate is a k0 row flagged as such."""
        table = ActiveEventTable(htst_config)
        table.add_events(_refined(system_single_type_fcc, neighbors_list, 0))
        row = table.table.iloc[0]
        assert row["nu0_status"] == "rejected"
        assert row["nu0_reason"] == "no reference estimate"
        assert row["nu0_source"] == "k0"
        assert row["k_prefactor"] == htst_config.rateconstant.k0

    def test_ok_status_without_frequency_is_an_error(
        self, htst_config: Any, system_single_type_fcc: Any, neighbors_list: Any
    ) -> None:
        """An 'ok' label without a finite nu0 is a programming error."""
        table = ActiveEventTable(htst_config)
        with pytest.raises(ValueError, match="nu0_status 'ok'"):
            table.add_events(
                _refined(
                    system_single_type_fcc,
                    neighbors_list,
                    0,
                    nu0_hz=None,
                    nu0_status="ok",
                )
            )

    def test_constant_mode_ignores_estimates(
        self, constant_config: Any, system_single_type_fcc: Any, neighbors_list: Any
    ) -> None:
        """Constant rows carry no prefactor columns whatever the output holds."""
        table = ActiveEventTable(constant_config)
        table.add_events(
            _refined(
                system_single_type_fcc,
                neighbors_list,
                0,
                nu0_hz=7.0e11,
                nu0_status="ok",
            )
        )
        assert list(table.table.columns) == list(ACTIVE_BASE_COLUMNS)
        k0 = constant_config.rateconstant.k0
        T = constant_config.rateconstant.T
        assert table.table.iloc[0]["k"] == rate_from_prefactor(k0, 0.5, T)


class TestSiteRequests:
    """One request per newly accepted refined row, after dedup."""

    def test_site_success_overrides_the_inherited_estimate(
        self, htst_config: Any, system_single_type_fcc: Any, neighbors_list: Any
    ) -> None:
        """k, k_prefactor, nu0, status and source move together to 'site'."""
        service, fake = _service(htst_config, _site_ok(5.0e12))
        table = ActiveEventTable(htst_config, prefactor_service=service)
        table.add_events(
            _refined(
                system_single_type_fcc,
                neighbors_list,
                0,
                nu0_hz=7.0e11,
                nu0_status="ok",
            )
        )
        summary = table.request_site_prefactors(system_single_type_fcc, neighbors_list)
        assert summary == {"attempted": 1, "ok": 1, "rejected": 0}
        row = table.table.iloc[0]
        T = htst_config.rateconstant.T
        assert row["nu0"] == 5.0e12
        assert row["k_prefactor"] == 5.0
        assert row["k"] == rate_from_prefactor(5.0, 0.5, T)
        assert row["nu0_status"] == "ok"
        assert row["nu0_source"] == "site"
        assert bool(row["nu0_site_attempted"]) is True
        assert table.prefactor_summary() == {
            "reference": 0,
            "site": 1,
            "k0": 0,
            "site_attempted": 1,
        }
        _assert_consistent(table, htst_config)

    def test_request_geometry_uses_the_crop_mapping(
        self, htst_config: Any, system_single_type_fcc: Any, neighbors_list: Any
    ) -> None:
        """Crops are written back at the neighbour indices, in their order."""
        service, fake = _service(htst_config, _site_ok(5.0e12))
        table = ActiveEventTable(htst_config, prefactor_service=service)
        table.add_events(
            _refined(
                system_single_type_fcc,
                neighbors_list,
                3,
                ref=11,
                nu0_hz=7.0e11,
                nu0_status="ok",
            )
        )
        table.request_site_prefactors(system_single_type_fcc, neighbors_list)

        (req,) = fake.prefactor_requests
        pos = np.asarray(system_single_type_fcc.positions, dtype=float)
        neighbors = np.asarray(neighbors_list.get_neighbors("rcut", 3), dtype=int)
        outside = np.setdiff1d(np.arange(len(pos)), neighbors)
        assert req.event_key == ("site", 0, 3, 11)
        assert req.center_index == 3
        assert np.array_equal(req.min1_positions, pos)
        assert np.allclose(req.saddle_positions[neighbors], pos[neighbors] + 0.1)
        assert np.allclose(req.min2_positions[neighbors], pos[neighbors] + 0.2)
        assert np.array_equal(req.saddle_positions[outside], pos[outside])
        assert np.array_equal(req.min2_positions[outside], pos[outside])
        assert req.types == tuple(system_single_type_fcc.types)
        assert req.pbc == tuple(bool(p) for p in system_single_type_fcc.pbc)

    def test_foreign_frame_without_htst_columns_is_refused(
        self,
        htst_config: Any,
        constant_config: Any,
        system_single_type_fcc: Any,
        neighbors_list: Any,
    ) -> None:
        """A caller-supplied 7-column frame in htst mode is a contract error."""
        seed = ActiveEventTable(constant_config)
        seed.add_events(_refined(system_single_type_fcc, neighbors_list, 0))
        assert list(seed.table.columns) == list(ACTIVE_BASE_COLUMNS)
        service, fake = _service(htst_config, _site_ok(5.0e12))
        table = ActiveEventTable(
            htst_config, event_dataframe=seed.table.copy(), prefactor_service=service
        )
        with pytest.raises(ValueError, match="nu0_site_attempted"):
            table.request_site_prefactors(system_single_type_fcc, neighbors_list)
        with pytest.raises(ValueError, match="k_prefactor"):
            table.add_events(_refined(system_single_type_fcc, neighbors_list, 1))
        assert fake.prefactor_requests == []
        assert list(table.table.columns) == list(ACTIVE_BASE_COLUMNS)

    def test_crop_mapping_mismatch_is_an_error(
        self, htst_config: Any, system_single_type_fcc: Any, neighbors_list: Any
    ) -> None:
        """A crop that no longer matches the mapping cannot be rebuilt silently."""
        service, _ = _service(htst_config, _site_ok(5.0e12))
        table = ActiveEventTable(htst_config, prefactor_service=service)
        out = _refined(system_single_type_fcc, neighbors_list, 0)
        out.saddle_positions = out.saddle_positions[:-1]
        table.add_events(out)
        with pytest.raises(RuntimeError, match="do not match the current rcut mapping"):
            table.request_site_prefactors(system_single_type_fcc, neighbors_list)

    def test_site_rejection_keeps_the_reference_estimate(
        self, htst_config: Any, system_single_type_fcc: Any, neighbors_list: Any
    ) -> None:
        """A valid inherited estimate survives a rejected site estimate."""
        service, _ = _service(htst_config, _site_rejected())
        table = ActiveEventTable(htst_config, prefactor_service=service)
        table.add_events(
            _refined(
                system_single_type_fcc,
                neighbors_list,
                0,
                nu0_hz=7.0e11,
                nu0_status="ok",
            )
        )
        k_before = table.table.iloc[0]["k"]
        summary = table.request_site_prefactors(system_single_type_fcc, neighbors_list)
        assert summary == {"attempted": 1, "ok": 0, "rejected": 1}
        row = table.table.iloc[0]
        assert row["nu0"] == 7.0e11 and row["k_prefactor"] == 0.7
        assert row["k"] == k_before
        assert row["nu0_status"] == "ok" and row["nu0_source"] == "reference"
        assert bool(row["nu0_site_attempted"]) is True
        _assert_consistent(table, htst_config)

    def test_site_rejection_without_inherited_estimate_is_k0(
        self, htst_config: Any, system_single_type_fcc: Any, neighbors_list: Any
    ) -> None:
        """No valid inherited estimate: the row stays at k0, marked attempted."""
        service, _ = _service(htst_config, _site_rejected())
        table = ActiveEventTable(htst_config, prefactor_service=service)
        table.add_events(
            _refined(
                system_single_type_fcc,
                neighbors_list,
                0,
                nu0_hz=None,
                nu0_status="rejected",
                nu0_reason="out_of_window: ref",
            )
        )
        table.request_site_prefactors(system_single_type_fcc, neighbors_list)
        row = table.table.iloc[0]
        assert row["k_prefactor"] == htst_config.rateconstant.k0
        assert row["nu0_source"] == "k0" and row["nu0_status"] == "rejected"
        assert bool(row["nu0_site_attempted"]) is True
        _assert_consistent(table, htst_config)

    def test_one_attempt_per_row(
        self, htst_config: Any, system_single_type_fcc: Any, neighbors_list: Any
    ) -> None:
        """Success or rejection, a row is never re-submitted."""
        outcomes = iter([_site_rejected(), _site_ok(9.0e12)])
        current = {"responder": next(outcomes)}
        service, fake = _service(htst_config, lambda req: current["responder"](req))
        table = ActiveEventTable(htst_config, prefactor_service=service)
        table.add_events(
            [
                _refined(
                    system_single_type_fcc,
                    neighbors_list,
                    0,
                    nu0_hz=7e11,
                    nu0_status="ok",
                ),
                _refined(
                    system_single_type_fcc,
                    neighbors_list,
                    5,
                    nu0_hz=None,
                    nu0_status="rejected",
                ),
            ]
        )
        table.request_site_prefactors(system_single_type_fcc, neighbors_list)
        assert len(fake.prefactor_requests) == 2
        current["responder"] = next(outcomes)
        summary = table.request_site_prefactors(system_single_type_fcc, neighbors_list)
        assert summary == {"attempted": 0, "ok": 0, "rejected": 0}
        assert len(fake.prefactor_requests) == 2
        assert table.table.iloc[0]["nu0"] == 7e11  # inherited value kept
        assert math.isnan(table.table.iloc[1]["nu0"])

    def test_unrefined_rows_are_never_requested(
        self, htst_config: Any, system_single_type_fcc: Any, neighbors_list: Any
    ) -> None:
        """Only refined == 'T' rows cost a site request."""
        service, fake = _service(htst_config, _site_ok(5e12))
        table = ActiveEventTable(htst_config, prefactor_service=service)
        table.add_events(
            [
                _refined(
                    system_single_type_fcc,
                    neighbors_list,
                    0,
                    refined="F",
                    nu0_hz=7e11,
                    nu0_status="ok",
                ),
                _refined(
                    system_single_type_fcc,
                    neighbors_list,
                    5,
                    refined="B",
                    nu0_hz=7e11,
                    nu0_status="ok",
                ),
                _refined(
                    system_single_type_fcc,
                    neighbors_list,
                    9,
                    refined="T",
                    nu0_hz=7e11,
                    nu0_status="ok",
                ),
            ]
        )
        table.request_site_prefactors(system_single_type_fcc, neighbors_list)
        assert [r.center_index for r in fake.prefactor_requests] == [9]
        assert list(table.table["nu0_source"]) == ["reference", "reference", "site"]

    def test_recycled_rows_keep_values_and_are_not_reattempted(
        self, htst_config: Any, system_single_type_fcc: Any, neighbors_list: Any
    ) -> None:
        """A row surviving pruning keeps its estimate and attempt flag."""
        service, fake = _service(htst_config, _site_ok(5e12))
        table = ActiveEventTable(htst_config, prefactor_service=service)
        table.add_events(
            [
                _refined(
                    system_single_type_fcc,
                    neighbors_list,
                    0,
                    nu0_hz=7e11,
                    nu0_status="ok",
                ),
                _refined(
                    system_single_type_fcc,
                    neighbors_list,
                    40,
                    nu0_hz=7e11,
                    nu0_status="ok",
                ),
            ]
        )
        table.request_site_prefactors(system_single_type_fcc, neighbors_list)
        # Simulate the recycler keeping the second row for the next step.
        table.table = table.table.loc[[1]].reset_index(drop=True).copy()
        kept = table.table.iloc[0].copy()
        table.add_events(
            _refined(
                system_single_type_fcc, neighbors_list, 20, nu0_hz=7e11, nu0_status="ok"
            )
        )
        table.request_site_prefactors(system_single_type_fcc, neighbors_list)
        assert [r.center_index for r in fake.prefactor_requests] == [0, 40, 20]
        assert table.table.iloc[0]["nu0"] == kept["nu0"] == 5e12
        assert table.table.iloc[0]["nu0_source"] == "site"
        assert bool(table.table.iloc[0]["nu0_site_attempted"]) is True
        _assert_consistent(table, htst_config)

    def test_duplicates_are_removed_before_any_request(
        self, htst_config: Any, system_single_type_fcc: Any, neighbors_list: Any
    ) -> None:
        """Two identical rows on one atom cost one request after dedup."""
        service, fake = _service(htst_config, _site_ok(5e12))
        table = ActiveEventTable(htst_config, prefactor_service=service)
        out = _refined(
            system_single_type_fcc, neighbors_list, 0, nu0_hz=7e11, nu0_status="ok"
        )
        table.add_events([out, out])
        assert len(table.table) == 2
        table.remove_duplicates(system_single_type_fcc.cell, neighbors_list)
        assert len(table.table) == 1
        table.request_site_prefactors(system_single_type_fcc, neighbors_list)
        assert len(fake.prefactor_requests) == 1

    def test_missing_service_with_eligible_rows_raises(
        self, htst_config: Any, system_single_type_fcc: Any, neighbors_list: Any
    ) -> None:
        """An htst table without a service cannot silently skip site estimates."""
        table = ActiveEventTable(htst_config)
        table.add_events(_refined(system_single_type_fcc, neighbors_list, 0))
        with pytest.raises(RuntimeError, match="PrefactorService"):
            table.request_site_prefactors(system_single_type_fcc, neighbors_list)

    def test_constant_mode_is_a_noop(
        self, constant_config: Any, system_single_type_fcc: Any, neighbors_list: Any
    ) -> None:
        """Constant tables never request anything and report zeros."""
        table = ActiveEventTable(constant_config)
        table.add_events(_refined(system_single_type_fcc, neighbors_list, 0))
        assert table.request_site_prefactors(
            system_single_type_fcc, neighbors_list
        ) == {
            "attempted": 0,
            "ok": 0,
            "rejected": 0,
        }
        assert table.prefactor_summary() == {}
