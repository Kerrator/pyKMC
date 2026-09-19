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
from .protocol_producers import protocol_event_prefactors
from tests.lifecycle.conftest import (
    FakeManager,
    accepted,
    event_prefactors,
    rejected,
    skipped,
)


@pytest.fixture
def neighbors_list(system_single_type_fcc: Any, htst_config: Any) -> NeighborsList:
    """Build the rcut neighbour list refinement would crop with."""
    return NeighborsList(
        system_single_type_fcc,
        htst_config.atomicenvironment.rnei,
        htst_config.atomicenvironment.rcut,
    )


def _full_saddle(system: Any, neighbors: np.ndarray) -> np.ndarray:
    """Full refined saddle: the crop atoms shifted by 0.1, everything else by 0.01.

    The far-field shift makes the full saddle distinguishable from the old
    construction (crop pasted into the minimum, far atoms at minimum positions).
    """
    pos = np.asarray(system.positions, dtype=float)
    full = pos + 0.01
    full[neighbors] = pos[neighbors] + 0.1
    return full


def _refined(
    system: Any,
    neighbors_list: NeighborsList,
    atom: int,
    refined: str = "T",
    dE: float = 0.5,
    ref: int = 0,
    full_saddle: str = "auto",
    **estimate: Any,
) -> EventRefinementOutput:
    """Refinement output with neighbour-cropped geometry, as production builds it.

    ``full_saddle="auto"`` attaches the full refined saddle exactly as
    ``Refinement.execute`` does for ``refined == "T"`` outputs in htst/rpa;
    ``"none"`` leaves it out.
    """
    neighbors = np.asarray(neighbors_list.get_neighbors("rcut", atom), dtype=int)
    pos = np.asarray(system.positions, dtype=float)
    full = _full_saddle(system, neighbors)
    return EventRefinementOutput(
        central_atom_index=atom,
        saddle_positions=full[neighbors],
        E_saddle=dE,
        min2_positions=pos[neighbors] + 0.2,
        dE_forward=dE,
        num_reference_event=ref,
        refined=refined,
        full_saddle_positions=full if full_saddle == "auto" else None,
        crop_atom_ids=tuple(int(system.index[i]) for i in neighbors),
        **estimate,
    )


def _service(config: Any, responder: Any) -> tuple[PrefactorService, FakeManager]:
    fake = FakeManager(responder)
    return (
        PrefactorService(
            config, fake, create_rate_constant(config.rateconstant), method="fd"
        ),
        fake,
    )


def _site_ok(nu0: float) -> Any:
    return lambda req: event_prefactors(req.event_key, accepted(nu0), skipped())


def _site_rejected(reason: str = "saddle_not_first_order") -> Any:
    return lambda req: event_prefectors_rejected(req, reason)


def event_prefectors_rejected(req: Any, reason: str) -> Any:
    """Forward rejected, backward skipped: what a forward-only request returns."""
    return event_prefactors(req.event_key, rejected(reason), skipped())


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
        assert summary == {"attempted": 1, "ok": 1, "rejected": 0, "no_geometry": 0}
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

    def test_request_geometry_is_the_full_refined_saddle_forward_only(
        self, htst_config: Any, system_single_type_fcc: Any, neighbors_list: Any
    ) -> None:
        """The request carries the full pARTn saddle and asks for the forward direction only."""
        service, fake = _service(htst_config, _site_ok(5.0e12))
        table = ActiveEventTable(htst_config, prefactor_service=service)
        out = _refined(
            system_single_type_fcc,
            neighbors_list,
            3,
            ref=11,
            nu0_hz=7.0e11,
            nu0_status="ok",
        )
        table.add_events(out)
        table.request_site_prefactors(system_single_type_fcc, neighbors_list)

        (req,) = fake.prefactor_requests
        assert fake.prefactor_backward_flags == [False]
        pos = np.asarray(system_single_type_fcc.positions, dtype=float)
        neighbors = np.asarray(neighbors_list.get_neighbors("rcut", 3), dtype=int)
        outside = np.setdiff1d(np.arange(len(pos)), neighbors)
        assert req.event_key == ("site", 0, 3, 11)
        assert req.center_index == 3
        assert np.array_equal(req.min1_positions, pos)
        assert np.array_equal(req.saddle_positions, out.full_saddle_positions)
        assert req.saddle_positions is not out.full_saddle_positions  # copied
        assert np.array_equal(req.saddle_positions[neighbors], out.saddle_positions)
        # the old construction (crop pasted into the minimum) is NOT what is sent
        assert not np.array_equal(req.saddle_positions[outside], pos[outside])
        # min2 is unused for a forward-only request: a copy of the minimum
        assert np.array_equal(req.min2_positions, pos)
        assert req.types == tuple(system_single_type_fcc.types)
        assert req.pbc == tuple(bool(p) for p in system_single_type_fcc.pbc)

    def test_missing_full_saddle_keeps_the_inherited_estimate(
        self, htst_config: Any, system_single_type_fcc: Any, neighbors_list: Any
    ) -> None:
        """A crop-only refined row is no error: inherited estimate, no request (F1)."""
        import logging

        records: list[logging.LogRecord] = []

        class _Collect(logging.Handler):
            def emit(self, record: logging.LogRecord) -> None:
                records.append(record)

        handler = _Collect(level=logging.DEBUG)
        logger = logging.getLogger("log")
        previous = logger.level
        logger.addHandler(handler)
        logger.setLevel(logging.DEBUG)
        try:
            service, fake = _service(htst_config, _site_ok(5.0e12))
            table = ActiveEventTable(htst_config, prefactor_service=service)
            table.add_events(
                [
                    _refined(
                        system_single_type_fcc,
                        neighbors_list,
                        0,
                        full_saddle="none",
                        nu0_hz=7.0e11,
                        nu0_status="ok",
                    ),
                    _refined(
                        system_single_type_fcc,
                        neighbors_list,
                        5,
                        ref=3,
                        full_saddle="none",
                        nu0_hz=None,
                        nu0_status="rejected",
                        nu0_reason="unstable_minimum: x",
                    ),
                ]
            )
            before = table.table.copy(deep=True)
            summary = table.request_site_prefactors(
                system_single_type_fcc, neighbors_list
            )
        finally:
            logger.removeHandler(handler)
            logger.setLevel(previous)
        assert summary == {"attempted": 2, "ok": 0, "rejected": 0, "no_geometry": 2}
        assert fake.prefactor_requests == []
        assert fake.submitted == []
        for column in (
            "nu0",
            "nu0_status",
            "nu0_source",
            "nu0_reason",
            "k",
            "k_prefactor",
        ):
            assert list(table.table[column].astype(object)) == list(
                before[column].astype(object)
            ) or (
                column == "nu0"
                and all(
                    math.isnan(v) for v in (table.table[column][1], before[column][1])
                )
                and table.table[column][0] == before[column][0]
            ), column
        assert list(table.table["nu0_site_attempted"]) == [True, True]
        assert list(table.table["nu0_source"]) == ["reference", "k0"]
        assert table.table.iloc[0]["nu0"] == 7.0e11
        assert table.prefactor_summary()["site_attempted"] == 2
        _assert_consistent(table, htst_config)
        lines = [
            r.getMessage()
            for r in records
            if "no full refined saddle available" in r.getMessage()
        ]
        assert len(lines) == 2
        assert lines[0].startswith("[htst] active event (atom 0, reference 0)")
        assert "keeping the inherited reference estimate" in lines[0]
        assert "keeping the inherited k0 estimate" in lines[1]
        assert all(
            r.levelno == logging.INFO for r in records if r.getMessage() in lines
        )
        # never re-attempted, still no request
        again = table.request_site_prefactors(system_single_type_fcc, neighbors_list)
        assert again == {"attempted": 0, "ok": 0, "rejected": 0, "no_geometry": 0}
        assert fake.submitted == []

    def test_crop_inconsistent_with_the_full_saddle_is_an_error(
        self, htst_config: Any, system_single_type_fcc: Any, neighbors_list: Any
    ) -> None:
        """The stored crop must be the full saddle at the current mapping."""
        service, fake = _service(htst_config, _site_ok(5.0e12))
        table = ActiveEventTable(htst_config, prefactor_service=service)
        out = _refined(system_single_type_fcc, neighbors_list, 0)
        out.saddle_positions = out.saddle_positions + 1.0e-3
        table.add_events(out)
        with pytest.raises(RuntimeError, match="not the full refined saddle"):
            table.request_site_prefactors(system_single_type_fcc, neighbors_list)
        assert fake.prefactor_requests == []

    def test_full_saddles_follow_their_rows_and_are_released(
        self, htst_config: Any, system_single_type_fcc: Any, neighbors_list: Any
    ) -> None:
        """The side store is keyed by row label, remapped on removal, cleared after the batch."""
        service, fake = _service(htst_config, _site_ok(5.0e12))
        table = ActiveEventTable(htst_config, prefactor_service=service)
        outs = [
            _refined(
                system_single_type_fcc,
                neighbors_list,
                atom,
                nu0_hz=7e11,
                nu0_status="ok",
            )
            for atom in (0, 5, 9)
        ]
        table.add_events(outs)
        assert sorted(table._full_saddles) == [0, 1, 2]
        table.remove(1)
        assert sorted(table._full_saddles) == [0, 1]
        assert np.array_equal(table._full_saddles[1], outs[2].full_saddle_positions)
        assert list(table.table["atom_index"]) == [0, 9]
        table.request_site_prefactors(system_single_type_fcc, neighbors_list)
        assert table._full_saddles == {}
        assert [r.center_index for r in fake.prefactor_requests] == [0, 9]
        assert all(
            np.array_equal(r.saddle_positions, o.full_saddle_positions)
            for r, o in zip(fake.prefactor_requests, (outs[0], outs[2]), strict=True)
        )
        # the DataFrame never carries the full array
        assert "full_saddle_positions" not in table.table.columns
        table.add_events(_refined(system_single_type_fcc, neighbors_list, 20))
        assert sorted(table._full_saddles) == [2]
        table.prune_for_recycling(0, system_single_type_fcc, None)
        assert table._full_saddles == {}

    def test_worker_failure_releases_the_full_saddles(
        self, htst_config: Any, system_single_type_fcc: Any, neighbors_list: Any
    ) -> None:
        """A raising worker propagates and still drops the transient arrays."""

        def boom(req: Any) -> Any:
            raise OSError("scratch failed")

        service, _ = _service(htst_config, boom)
        table = ActiveEventTable(htst_config, prefactor_service=service)
        table.add_events(_refined(system_single_type_fcc, neighbors_list, 0))
        with pytest.raises(OSError):
            table.request_site_prefactors(system_single_type_fcc, neighbors_list)
        assert table._full_saddles == {}

    def test_site_log_lines_carry_n_free_and_batch_time(
        self, htst_config: Any, system_single_type_fcc: Any, neighbors_list: Any
    ) -> None:
        """Success and rejection lines report the free-atom count and the batch wall time."""
        import logging

        records: list[logging.LogRecord] = []

        class _Collect(logging.Handler):
            def emit(self, record: logging.LogRecord) -> None:
                records.append(record)

        logger = logging.getLogger("log")
        handler = _Collect(level=logging.DEBUG)
        previous = logger.level
        logger.addHandler(handler)
        logger.setLevel(logging.DEBUG)
        try:
            outcomes = {0: _site_ok(5e12), 5: _site_rejected("bad")}
            service, _ = _service(
                htst_config, lambda req: outcomes[req.center_index](req)
            )
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
                        nu0_hz=7e11,
                        nu0_status="ok",
                    ),
                ]
            )
            table.request_site_prefactors(system_single_type_fcc, neighbors_list)
        finally:
            logger.removeHandler(handler)
            logger.setLevel(previous)
        lines = [
            r.getMessage() for r in records if "active event (atom" in r.getMessage()
        ]
        assert len(lines) == 2
        assert all(
            "n_free 5" in line and "batch " in line and " s)" in line for line in lines
        )
        assert "site nu0 = 5.0000e+12 Hz" in lines[0]
        assert (
            "site prefactor rejected (out_of_window: bad); keeping the reference"
            in lines[1]
        )

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
        with pytest.raises(RuntimeError, match="not match the current rcut mapping"):
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
        assert summary == {"attempted": 1, "ok": 0, "rejected": 1, "no_geometry": 0}
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
        # Preserve the original inheritance observation before the subsequent
        # current-policy guard: 7e11 is below this fixture's default 1e12 bound.
        assert table.table.iloc[0]["nu0"] == 7e11
        assert 7e11 < service.settings.nu0_min_hz
        current["responder"] = next(outcomes)
        summary = table.request_site_prefactors(system_single_type_fcc, neighbors_list)
        assert summary == {"attempted": 0, "ok": 0, "rejected": 0, "no_geometry": 0}
        assert len(fake.prefactor_requests) == 2
        # R09 rechecks current admissibility before reuse. The disallowed
        # inherited row is unavailable; the unchanged rejected row is not retried.
        assert list(table.table["atom_index"].astype(int)) == [5]
        assert math.isnan(table.table.iloc[0]["nu0"])
        assert table.table.iloc[0]["nu0_source"] == "k0"

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
        # This reuse assertion needs an actual immutable producing record;
        # the isolated scalar/log tests retain their original naked workers.
        service, fake = _service(
            htst_config,
            lambda request: protocol_event_prefactors(
                request, accepted(5e12), skipped()
            ),
        )
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
        # Use the real mutation boundary so producing context is remapped
        # with the row; direct DataFrame replacement is not a valid producer.
        table.remove(0)
        kept = table.table.iloc[0].copy()
        table.add_events(
            _refined(
                system_single_type_fcc, neighbors_list, 20, nu0_hz=7e11, nu0_status="ok"
            )
        )
        table.request_site_prefactors(system_single_type_fcc, neighbors_list)
        assert [r.center_index for r in fake.prefactor_requests] == [0, 40, 20]
        assert int(table.table.iloc[0]["atom_index"]) == int(kept["atom_index"]) == 40
        assert table.site_calculation(table.table.index[0]) is not None
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
            "no_geometry": 0,
        }
        assert table.prefactor_summary() == {}


class TestDropReferenceEvents:
    """``drop_reference_events`` keeps the active table in step with the catalogue (F4)."""

    def test_drops_rows_of_removed_references_and_relabels(
        self, htst_config: Any, system_single_type_fcc: Any, neighbors_list: Any
    ) -> None:
        """Every row whose reference was removed goes; survivors are relabelled."""
        service, fake = _service(htst_config, _site_ok(5.0e12))
        table = ActiveEventTable(htst_config, prefactor_service=service)
        outs = [
            _refined(
                system_single_type_fcc,
                neighbors_list,
                atom,
                ref=ref,
                nu0_hz=7e11,
                nu0_status="ok",
            )
            for atom, ref in ((0, 2), (5, 1), (9, 9), (14, 0))
        ]
        table.add_events(outs)
        assert sorted(table._full_saddles) == [0, 1, 2, 3]
        assert table.drop_reference_events((0, 1, 2)) == 3
        assert list(table.table.index) == [0]
        assert list(table.table["atom_index"]) == [9]
        assert list(table.table["num_reference_event"]) == [9]
        assert sorted(table._full_saddles) == [0]
        assert np.array_equal(table._full_saddles[0], outs[2].full_saddle_positions)
        assert table.drop_reference_events([]) == 0
        assert table.drop_reference_events([2]) == 0
        table.request_site_prefactors(system_single_type_fcc, neighbors_list)
        assert [r.center_index for r in fake.prefactor_requests] == [9]
        assert np.array_equal(
            fake.prefactor_requests[0].saddle_positions, outs[2].full_saddle_positions
        )

    def test_recycled_rows_are_dropped_too(
        self, htst_config: Any, system_single_type_fcc: Any, neighbors_list: Any
    ) -> None:
        """An attempted (recycled) row is only as valid as its reference."""
        service, _ = _service(htst_config, _site_ok(5.0e12))
        table = ActiveEventTable(htst_config, prefactor_service=service)
        table.add_events(
            [
                _refined(system_single_type_fcc, neighbors_list, 0, ref=4),
                _refined(system_single_type_fcc, neighbors_list, 5, ref=6),
            ]
        )
        table.request_site_prefactors(system_single_type_fcc, neighbors_list)
        assert list(table.table["nu0_site_attempted"]) == [True, True]
        assert table.drop_reference_events({4}) == 1
        assert list(table.table["num_reference_event"]) == [6]
        assert list(table.table.index) == [0]

    def test_constant_mode_table_is_handled_the_same_way(
        self, constant_config: Any, system_single_type_fcc: Any, neighbors_list: Any
    ) -> None:
        """No HTST columns are needed: the drop is keyed on the reference id only."""
        table = ActiveEventTable(constant_config)
        table.add_events(
            [
                _refined(system_single_type_fcc, neighbors_list, 0, ref=1),
                _refined(system_single_type_fcc, neighbors_list, 5, ref=2),
                _refined(system_single_type_fcc, neighbors_list, 9, ref=1),
            ]
        )
        assert table.drop_reference_events([1]) == 2
        assert list(table.table["atom_index"]) == [5]
        assert list(table.table.index) == [0]
        assert ActiveEventTable(constant_config).drop_reference_events([1]) == 0
