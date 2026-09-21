# Promoted from the frozen refinement-coverage oracle of the 2026-09-19 HTST
# audit (sha256 93af76e60a510c47...) through its versioned adapter 01 (R12,
# contracts 7f policy 3). The assertions are the adapter's, unchanged; the
# harness edits are documented in the adapter's ADAPTER.md. Do not widen.
"""Actual dispatch/active-table/site-path coverage with deterministic boundaries.

Native pARTn and PSR are protocol doubles; selection, refinement.execute,
active insertion/dedup, rate resolution, site eligibility and rate accounting
are production code. No Hessians or engines. The literal advertised
refine_thr fraction is challenged; green controls characterize the inherited
heuristic without asserting it proves a cumulative coverage guarantee.
"""

from concurrent.futures import Future
import math
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

import pykmc
from pykmc.config import Config, RateConstantConfig
from pykmc.event_table import ActiveEventTable
from pykmc.refinement import Refinement
import pykmc.refinement as refinement_module
from pykmc.result import EventRefinementOutput, Ok, PSROutput

# Adapter 01: the site boundary is now the run's PrefactorService (R07..R09:
# typed requests, request snapshots and producing calculations); the
# synthetic typed worker of the repository's own unit fixtures stands in for
# the Hessian kernel with this file's per-row accepted/rejected frequencies.
from pykmc.rate_constant import create_rate_constant
from pykmc.rate_constant.prefactors import PrefactorService
from tests.lifecycle.conftest import FakeManager, accepted, rejected, skipped
from tests.lifecycle.protocol_producers import protocol_event_prefactors


# The documented application constant is config.py::Parameters.kb, inherited
# for constant-path compatibility. Use its literal value in an independent
# Arrhenius calculation; this test is not a CODATA-precision assessment.
KB = 8.6173303e-5


def run_case(monkeypatch, rowspecs, style="htst", temperature=300.0, threshold=0.9999):
    """rowspec = (barrier, accepted nu0 Hz or None, number of sites)."""
    cfg = Config.from_ini_file(
        str(Path(pykmc.__file__).resolve().parents[1] / "tests/data/input.in")
    )
    cfg.rateconstant = RateConstantConfig(style=style, k0=1.0, T=temperature)
    cfg.control.active_volume = False
    cfg.control.refine_thr = threshold
    cfg.psr.matching_score_thr = 0.001
    frame_rows = []
    sites = {}
    total_sites = 0
    for ordinal, (barrier, nu0, multiplicity) in enumerate(rowspecs):
        idx = 17 + ordinal * 7
        event = "event-" + str(idx)
        sites[event] = np.arange(total_sites, total_sites + multiplicity)
        total_sites += multiplicity
        prefactor = 1.0 if style == "constant" or nu0 is None else nu0 * 1e-12
        rate = prefactor * math.exp(-barrier / (KB * temperature))
        row = {
            "idx_ref": idx,
            "event_id": event,
            "initial_positions": np.zeros((1, 3)),
            "saddle_positions": np.array([[ordinal + 1.0, 0.0, 0.0]]),
            "final_positions": np.array([[ordinal + 1.25, 0.0, 0.0]]),
            "energy_barrier": barrier,
            "k": rate,
            "sym_matrix": [np.eye(3)],
            "sym_perm": [np.array([0])],
        }
        if style != "constant":
            row.update(
                nu0_status="ok" if nu0 is not None else "rejected",
                nu0=np.nan if nu0 is None else nu0,
                nu0_reason="" if nu0 is not None else "test reference rejection",
            )
        frame_rows.append(row)
    frame = pd.DataFrame(frame_rows)

    class SystemBoundary:
        positions = np.zeros((total_sites, 3))
        cell = np.eye(3) * 1000.0
        types = ["Si"] * total_sites
        pbc = [False, False, False]
        # Adapter 01 (R09, `1aa110a`..`fe1dbe1`): htst/rpa refinement requires
        # stable atom identities (System.index); the double predates R09.
        index = np.arange(total_sites)

        def update_positions(self, new_positions, atom_idx=None):
            if atom_idx is None:
                self.positions = np.array(new_positions, copy=True)
            else:
                self.positions[atom_idx] = new_positions

    class NativeBoundary:
        def __init__(self):
            self.submitted = []

        def partn_refine(self, **kwargs):
            positions = np.array(kwargs["positions"], copy=True)
            atom = kwargs["central_atom_idx"]
            ordinal = round(float(positions[atom, 0])) - 1
            self.submitted.append((atom, 17 + ordinal * 7))
            future = Future()
            future.set_result(
                Ok(
                    EventRefinementOutput(
                        central_atom_index=atom,
                        saddle_positions=positions,
                        E_saddle=rowspecs[ordinal][0],
                        refined="T",
                    )
                )
            )
            return future

    class SiteBoundary(PrefactorService):
        # Adapter 01: a real PrefactorService (rate facade, settings, typed
        # requests, producing calculations) over a synthetic typed worker.
        def __init__(self):
            def worker(request):
                ordinal = (request.event_key[-1] - 17) // 7
                nu0 = rowspecs[ordinal][1]
                forward = (
                    accepted(nu0)
                    if nu0 is not None
                    else rejected("test site rejection")
                )
                return protocol_event_prefactors(request, forward, skipped())

            super().__init__(
                cfg,
                FakeManager(worker),
                create_rate_constant(cfg.rateconstant),
                species_masses=(("Si",), (28.0855,)),
                method="fd",
            )
            self.submitted = []

        def compute(self, requests, compute_backward):
            assert compute_backward is False
            self.submitted.extend(request.event_key for request in requests)
            return super().compute(requests, compute_backward=compute_backward)

    match = Ok(
        PSROutput(
            rotation_matrix=np.eye(3),
            translation_matrix=np.zeros(3),
            permutation_matrix=np.array([0]),
            matching_score=0.0,
        )
    )
    monkeypatch.setattr(
        refinement_module,
        "PointSetRegistration",
        lambda *args: SimpleNamespace(match=lambda: match),
    )
    system = SystemBoundary()
    neighbors = SimpleNamespace(
        get_neighbors=lambda kind, atom: np.array([atom]), system=system
    )
    manager = NativeBoundary()
    refinement = Refinement(
        cfg,
        SimpleNamespace(info=lambda *args: None, progress_bar=lambda *args: None),
        system,
        neighbors,
        SimpleNamespace(get_atoms_with_id=lambda event: sites[event]),
        manager,
    )
    count, supposed_total = refinement.get_total_refinements_todo(frame)
    refinement.execute(frame, total_energy=0.0)
    # Adapter 01: the constant path never carries a PrefactorService (the
    # service refuses that backend); only htst/rpa attach the site boundary.
    site = None if style == "constant" else SiteBoundary()
    active = ActiveEventTable(cfg, prefactor_service=site)
    for output in refinement.get_successes_results():
        active.add_events(output)
    active.remove_duplicates(system.cell, neighbors)
    assert len(active.table) == count  # no accidental duplicate losses
    before = active.table.copy(deep=True)
    summary = active.request_site_prefactors(system, neighbors)
    actual_total = active.table.k.sum()
    assert actual_total == pytest.approx(supposed_total, rel=1e-13)
    refined_rate = float(active.table.loc[active.table.refined == "T", "k"].sum())
    coverage = refined_rate / actual_total if actual_total else None
    submitted_refs = [item[1] for item in manager.submitted]
    report = {
        "style": style,
        "T_K": temperature,
        "refine_thr": threshold,
        "rowspecs": rowspecs,
        "active_count": len(active.table),
        "pARTn_submitted_references": submitted_refs,
        "site_count": 0 if site is None else len(site.submitted),
        "summary": summary,
        "actual_total_rate_per_ps": float(actual_total),
        "refined_rate_coverage": coverage,
        "rows": active.table[
            ["num_reference_event", "energy_barrier", "k", "refined"]
        ].to_dict("records"),
    }
    print(report)
    return SimpleNamespace(
        config=cfg,
        active=active.table,
        before=before,
        site=site,
        coverage=coverage,
        manager=manager,
        frame=frame,
        refinement=refinement,
        summary=summary,
        report=report,
    )


@pytest.mark.parametrize("style", ["htst", "rpa"])
def test_single_dominant_row_meets_advertised_refine_fraction(monkeypatch, style):
    result = run_case(monkeypatch, [(0.4, 1e12, 1)], style=style)
    assert result.coverage >= result.config.control.refine_thr
    assert result.summary["attempted"] == 1


@pytest.mark.parametrize("style", ["htst", "rpa"])
def test_many_small_site_contributions_meet_advertised_refine_fraction(
    monkeypatch, style
):
    # 100 slow-site copies together supply half the total rate. Their barrier
    # is 0.11905eV above the fast row, beyond the fixed 0.1eV window.
    slow_barrier = 0.4 + KB * 300.0 * math.log(100.0)
    result = run_case(
        monkeypatch, [(0.4, 1e12, 1), (slow_barrier, 1e12, 100)], style=style
    )
    # The expected 50% is an input contribution, not a demand that the
    # production method forever refine exactly half the rate.
    assert 100 * result.frame.iloc[1].k / (
        result.frame.iloc[0].k + 100 * result.frame.iloc[1].k
    ) == pytest.approx(0.5, rel=2e-6)
    assert result.coverage >= result.config.control.refine_thr


@pytest.mark.parametrize("style", ["htst", "rpa"])
def test_rate_inversion_refines_actual_large_contributor(monkeypatch, style):
    result = run_case(monkeypatch, [(0.4, 1e12, 1), (0.525, 100e12, 1)], style=style)
    second = result.active.query("num_reference_event == 24").iloc[0]
    assert second.k / result.active.k.sum() > 0.4
    assert second.refined == "T"
    assert any(key[-1] == 24 for key in result.site.submitted)


@pytest.mark.parametrize("style", ["constant", "htst", "rpa"])
def test_equal_rate_tie_control_refines_all_sites(monkeypatch, style):
    result = run_case(monkeypatch, [(0.4, 1e12, 1)] * 3, style=style)
    assert result.coverage == pytest.approx(1.0)
    assert len(result.manager.submitted) == 3


@pytest.mark.parametrize("style", ["htst", "rpa"])
def test_rejected_reference_and_site_keep_fallback_control(monkeypatch, style):
    result = run_case(monkeypatch, [(0.4, None, 1), (0.4, 1e12, 1)], style=style)
    assert result.coverage == pytest.approx(1.0)
    row = result.active.query("num_reference_event == 17").iloc[0]
    assert row.nu0_status == "rejected" and row.nu0_source == "k0"
    assert row.k_prefactor == 1.0
    # Adapter 01 (R09): the summary also reports identity-less rows dropped
    # before selection; none here. The four original counts are unchanged.
    assert result.summary == {
        "attempted": 2,
        "ok": 1,
        "rejected": 1,
        "no_geometry": 0,
        "identityless": 0,
    }


@pytest.mark.parametrize("style", ["constant", "htst", "rpa"])
def test_equal_prefactor_low_rate_heuristic_control(monkeypatch, style):
    result = run_case(monkeypatch, [(0.4, 1e12, 1), (0.525, 1e12, 1)], style=style)
    if style == "constant":
        assert result.active.refined.tolist() == ["T", "F"]
        assert len(result.manager.submitted) == 1
    else:
        # Adapter 01 (contracts 7f policy 3, decided 2026-09-19): htst/rpa
        # select by cumulative pre-dispatch rate coverage. The .525 eV row
        # carries 0.78 % of the snapshot total at 300 K, above the 0.01 %
        # the default refine_thr leaves uncovered, so both rows are
        # selected. The former "equal prefactors select exactly the
        # constant set" requirement is withdrawn with 7e R4.
        assert result.active.refined.tolist() == ["T", "T"]
        assert len(result.manager.submitted) == 2


@pytest.mark.parametrize("style", ["htst", "rpa"])
def test_all_underflowed_rates_control(monkeypatch, style):
    result = run_case(monkeypatch, [(100.0, 1e12, 1)], style=style)
    assert result.coverage is None
    assert result.active.k.tolist() == [0.0]
    assert not result.manager.submitted and not result.site.submitted


@pytest.mark.parametrize("temperature", [300.0, 1000.0])
def test_temperature_changed_rates_still_refine_both_material_contributors(
    monkeypatch, temperature
):
    # Heterogeneous prefactors invert the fastest row with temperature.
    result = run_case(
        monkeypatch, [(0.4, 1e12, 1), (0.525, 100e12, 1)], temperature=temperature
    )
    assert result.active.refined.tolist() == ["T", "T"]
