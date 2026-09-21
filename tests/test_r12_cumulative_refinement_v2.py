"""R12 / N07: cumulative pre-dispatch refinement coverage (contracts 7f policy 3).

The actual dispatcher (``Refinement.execute``), the active-table transport and
the site-prefactor path run with pARTn, PSR and the Hessian kernel substituted
by protocol doubles, as the frozen refinement-coverage oracle does. Every N07
bullet of the acceptance table has a case here. The constant style is a
byte-identity control, not a participant in the ledger.
"""

from __future__ import annotations

import hashlib
import inspect
import math
from concurrent.futures import Future
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import pandas as pd
import pytest

import pykmc
import pykmc.refinement as refinement_module
from pykmc.config import Config, RateConstantConfig
from pykmc.event_table import ActiveEventTable, ReferenceEventTable
from pykmc.kmc import KMC
from pykmc.rate_constant import create_rate_constant
from pykmc.rate_constant.prefactors import PrefactorService
from pykmc.refinement import Refinement
from pykmc.result import (
    Err,
    ErrorInfo,
    ErrorType,
    EventRefinementOutput,
    Ok,
    PSROutput,
)
from tests.lifecycle.conftest import FakeManager, accepted, rejected, skipped
from tests.lifecycle.protocol_producers import protocol_event_prefactors

KB = 8.6173303e-5
INPUT = Path(pykmc.__file__).resolve().parents[1] / "tests" / "data" / "input.in"
ROT90 = np.array([[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]])
GET_ENERGY_THR_REFINE_SHA256 = (
    "276331434ebc83ce13f3476f2bb9680b31b4d657c6d1a4c343d346598b78d658"
)
"""sha256 of ``inspect.getsource(Refinement.get_energy_thr_refine)`` at ``fe1dbe1``."""


# --------------------------------------------------------------------------- #
# Harness: reference rows, doubles, one execute() round through the table.
# --------------------------------------------------------------------------- #


def row(ref, barrier, sites, *, nu0=1e12, syms=None, k=None, event=None):
    """One reference row: logical id, barrier, its sites and its prefactor."""
    return {
        "ref": ref,
        "barrier": barrier,
        "sites": list(sites),
        "nu0": nu0,
        "syms": [np.eye(3)] if syms is None else list(syms),
        "k": k,
        "event": event or "event-{}".format(ref),
    }


def rate(spec, style, temperature):
    if spec["k"] is not None:
        return spec["k"]
    prefactor = (
        1.0 if style == "constant" or spec["nu0"] is None else spec["nu0"] * 1e-12
    )
    return prefactor * math.exp(-spec["barrier"] / (KB * temperature))


def config(style, temperature, threshold):
    cfg = Config.from_ini_file(str(INPUT))
    cfg.rateconstant = RateConstantConfig(style=style, k0=1.0, T=temperature)
    cfg.control.active_volume = False
    cfg.control.refine_thr = threshold
    cfg.psr.matching_score_thr = 0.001
    return cfg


def frame(specs, style, temperature):
    rows = []
    for spec in specs:
        entry = {
            "idx_ref": spec["ref"],
            "event_id": spec["event"],
            "initial_positions": np.zeros((1, 3)),
            "saddle_positions": np.array([[float(spec["ref"]), 0.0, 0.0]]),
            "final_positions": np.array([[spec["ref"] + 0.25, 0.0, 0.0]]),
            "energy_barrier": spec["barrier"],
            "k": rate(spec, style, temperature),
            "sym_matrix": list(spec["syms"]),
            "sym_perm": [np.array([0]) for _ in spec["syms"]],
        }
        if style != "constant":
            ok = spec["nu0"] is not None
            entry.update(
                nu0_status="ok" if ok else "rejected",
                nu0=spec["nu0"] if ok else np.nan,
                nu0_reason="" if ok else "test reference rejection",
            )
        rows.append(entry)
    return pd.DataFrame(rows)


class SystemBoundary:
    def __init__(self, n_atoms):
        self.positions = np.zeros((n_atoms, 3))
        self.cell = np.eye(3) * 1000.0
        self.types = ["Si"] * n_atoms
        self.pbc = [False, False, False]
        self.index = np.arange(n_atoms)

    def update_positions(self, new_positions, atom_idx=None):
        if atom_idx is None:
            self.positions = np.array(new_positions, copy=True)
        else:
            self.positions[atom_idx] = new_positions


class NativeBoundary:
    """pARTn double: the reference id is encoded in the saddle displacement."""

    def __init__(self, barriers, failing=(), *, fail_rotated=False):
        self.barriers = barriers
        self.failing = set(failing)
        self.fail_rotated = fail_rotated
        self.submitted = []
        self.geometries = []

    def partn_refine(self, **kwargs):
        positions = np.array(kwargs["positions"], copy=True)
        atom = int(kwargs["central_atom_idx"])
        ref = int(round(float(np.abs(positions[atom]).max())))
        self.submitted.append((atom, ref))
        rotated = abs(positions[atom, 1]) > abs(positions[atom, 0])
        self.geometries.append("rotated" if rotated else "identity")
        future = Future()
        if (atom, ref) in self.failing or (self.fail_rotated and rotated):
            future.set_result(
                Err(
                    ErrorInfo(
                        type=ErrorType.REFINEMENT_INVALID_MINIMA,
                        message="injected pARTn failure",
                    )
                )
            )
        else:
            future.set_result(
                Ok(
                    EventRefinementOutput(
                        central_atom_index=atom,
                        saddle_positions=positions,
                        E_saddle=self.barriers[ref],
                        refined="T",
                    )
                )
            )
        return future


class SiteBoundary(PrefactorService):
    """The run's PrefactorService over a typed worker answering per reference."""

    def __init__(self, cfg, frequencies):
        self.frequencies = dict(frequencies)

        def worker(request):
            nu0 = self.frequencies.get(int(request.event_key[-1]))
            forward = (
                accepted(nu0) if nu0 is not None else rejected("test site rejection")
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


class Log:
    def __init__(self):
        self.lines = []

    def info(self, name, msg, *args, **kwargs):
        self.lines.append(("info", str(msg)))

    def warning(self, name, msg, *args, **kwargs):
        self.lines.append(("warning", str(msg)))

    def progress_bar(self, *args, **kwargs):
        pass


def psr_double(monkeypatch, failing_pairs):
    def factory(config, system, dfevent, neighbors, at_idx):
        score = 1.0 if (int(at_idx), int(dfevent["idx_ref"])) in failing_pairs else 0.0
        return SimpleNamespace(
            match=lambda: Ok(
                PSROutput(
                    rotation_matrix=np.eye(3),
                    translation_matrix=np.zeros(3),
                    permutation_matrix=np.array([0]),
                    matching_score=score,
                )
            )
        )

    monkeypatch.setattr(refinement_module, "PointSetRegistration", factory)


def retained_output(atom, ref, barrier, refined, n_atoms, *, nu0=1e12, rotated=False):
    """A producer output as the previous step left it (crop = the atom itself)."""
    ok = nu0 is not None
    full = np.zeros((n_atoms, 3))
    full[atom] = [0.0, float(ref), 0.0] if rotated else [float(ref), 0.0, 0.0]
    return EventRefinementOutput(
        central_atom_index=atom,
        saddle_positions=full[[atom]].copy(),
        E_saddle=barrier,
        min2_positions=np.array([[ref + 0.25, 0.0, 0.0]]),
        dE_forward=barrier,
        num_reference_event=ref,
        refined=refined,
        nu0_hz=nu0 if ok else None,
        nu0_status="ok" if ok else "rejected",
        nu0_reason="" if ok else "test reference rejection",
        nu0_source="reference",
        full_saddle_positions=full if refined == "T" else None,
        crop_atom_ids=(atom,),
    )


RETAINED_SITE_HZ = 5e12
"""Site frequency answered while retained rows are built: 5x the reference 1e12 Hz."""


def build_retained(active, site, system, neighbors, barriers, rows):
    """Carry rows over as production does: add, then capture their site state.

    A refined row obtains its site rate through the service (frequency
    ``RETAINED_SITE_HZ``); an unrefined row obtains its fallback context. Both
    then pass ``validate_recycled`` at the next step. Returns ``{label: k}``.
    """
    outputs = [
        retained_output(
            spec[0],
            spec[1],
            barriers[spec[1]],
            spec[2],
            len(system.positions),
            rotated=len(spec) > 3 and spec[3] == "rotated",
        )
        for spec in rows
    ]
    active.add_events(outputs)
    if site is not None:
        run_map = dict(site.frequencies)
        site.frequencies = {ref: RETAINED_SITE_HZ for ref in run_map}
        active.request_site_prefactors(system, neighbors)
        site.frequencies = run_map
        site.submitted.clear()
    return {int(label): float(r.k) for label, r in active.table.iterrows()}


def run(
    monkeypatch,
    specs,
    *,
    style="htst",
    temperature=300.0,
    threshold=0.9999,
    retained_rows=(),
    psr_fail=(),
    partn_fail=(),
    n_atoms=None,
    alias_ids=False,
    fail_rotated=False,
):
    """One refinement round: execute -> supersede -> add -> dedup -> site path.

    ``retained_rows`` are ``(atom, ref, refined[, "rotated"])`` tuples carried
    over in the active table before the round, as recycling leaves them
    (attempted in the step that built them; a refined row carries its site
    rate; ``"rotated"`` is the ROT90 symmetric application's generic saddle).
    """
    cfg = config(style, temperature, threshold)
    df = frame(specs, style, temperature)
    if alias_ids:
        df["idx_ref"] = df["idx_ref"].astype(np.int64)
    sites: dict[str, list[int]] = {}
    for spec in specs:
        for atom in spec["sites"]:
            if atom not in sites.setdefault(spec["event"], []):
                sites[spec["event"]].append(atom)
    atoms = [a for spec in specs for a in spec["sites"]] + [r[0] for r in retained_rows]
    retained_rates: dict[int, float] = {}
    system = SystemBoundary(n_atoms or (1 + max(atoms)))
    neighbors = SimpleNamespace(
        get_neighbors=lambda kind, atom: np.array([atom]), system=system
    )
    psr_double(monkeypatch, set(psr_fail))
    barriers = {spec["ref"]: spec["barrier"] for spec in specs}
    manager = NativeBoundary(barriers, partn_fail, fail_rotated=fail_rotated)
    log = Log()
    refinement = Refinement(
        cfg,
        log,
        system,
        neighbors,
        SimpleNamespace(get_atoms_with_id=lambda event: np.array(sites[event])),
        manager,
    )
    site = (
        None
        if style == "constant"
        else SiteBoundary(cfg, {spec["ref"]: spec["nu0"] for spec in specs})
    )
    active = ActiveEventTable(cfg, prefactor_service=site)
    if retained_rows:
        retained_rates = build_retained(
            active, site, system, neighbors, barriers, retained_rows
        )
    if alias_ids and len(active.table):
        # pandas may hold the logical id as a float column after a NaN visit.
        active.table["num_reference_event"] = active.table[
            "num_reference_event"
        ].astype(float)
    kwargs: dict[str, Any] = {}
    if retained_rows:
        kwargs = {
            "existing_pairs": active.existing_pairs(),
            "retained_channels": active.retained_channels(),
        }
    refinement.execute(df, total_energy=0.0, **kwargs)
    dropped = active.drop_unrefined_rows(refinement.superseded_rows)
    active.add_events(refinement.get_successes_results())
    active.remove_duplicates(system.cell, neighbors)
    summary = (
        None if site is None else active.request_site_prefactors(system, neighbors)
    )
    return SimpleNamespace(
        cfg=cfg,
        frame=df,
        refinement=refinement,
        coverage=refinement.coverage,
        manager=manager,
        active=active.table,
        table=active,
        site=site,
        log=log,
        system=system,
        neighbors=neighbors,
        dropped=dropped,
        summary=summary,
        results=refinement.results,
        retained_rates=retained_rates,
    )


def fraction(result, ref):
    table = result.active
    return float(table.loc[table.num_reference_event == ref, "k"].sum()) / float(
        table.k.sum()
    )


STYLES = ["htst", "rpa"]


# --------------------------------------------------------------------------- #
# Selection over the immutable ledger
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("style", STYLES)
def test_inversion_44_percent_row_is_selected_and_site_requested(monkeypatch, style):
    result = run(
        monkeypatch, [row(17, 0.4, [0]), row(24, 0.525, [1], nu0=100e12)], style=style
    )
    assert fraction(result, 24) == pytest.approx(0.443, abs=0.002)
    assert sorted(result.manager.submitted) == [(0, 17), (1, 24)]
    assert result.coverage["selected_ids"] == (17, 24)
    assert result.coverage["selected_fraction"] == pytest.approx(1.0)
    assert result.coverage["refined_fraction"] == pytest.approx(1.0)
    assert any(key[-1] == 24 for key in result.site.submitted)


@pytest.mark.parametrize("style", STYLES)
def test_dominant_singleton_is_selected(monkeypatch, style):
    result = run(monkeypatch, [row(17, 0.4, [0])], style=style)
    assert result.manager.submitted == [(0, 17)]
    cov = result.coverage
    assert cov["applicable"] and cov["n_groups"] == 1 and cov["n_channels"] == 1
    assert cov["selected_fraction"] == pytest.approx(1.0)
    assert cov["snapshot_total"] == pytest.approx(float(result.frame.k.sum()))
    assert result.active.refined.tolist() == ["T"]


@pytest.mark.parametrize("threshold, expect_small", [(0.5, False), (0.9999, True)])
def test_one_fast_plus_100_small_sites(monkeypatch, threshold, expect_small):
    # 100 slow sites together carry 40 % of the snapshot total.
    slow = 0.4 + KB * 300.0 * math.log(150.0)
    result = run(
        monkeypatch,
        [row(17, 0.4, [0]), row(24, slow, range(1, 101))],
        threshold=threshold,
    )
    assert result.coverage["n_channels"] == 101 and result.coverage["n_groups"] == 2
    small = sum(1 for atom, ref in result.manager.submitted if ref == 24)
    assert (0, 17) in result.manager.submitted
    assert small == (100 if expect_small else 0)
    assert result.coverage["selected_ids"] == ((17, 24) if expect_small else (17,))
    assert result.coverage["selected_fraction"] == pytest.approx(
        1.0 if expect_small else 0.6, abs=1e-12
    )
    # Unselected channels still enter the table with the generic saddle.
    assert len(result.active) == 101
    assert (result.active.refined == "T").sum() == (101 if expect_small else 1)


def test_repeated_logical_reference_rows_group_and_dispatch_once(monkeypatch):
    specs = [row(17, 0.4, [0]), row(17, 0.4, [0])]
    result = run(monkeypatch, specs)
    cov = result.coverage
    assert cov["n_groups"] == 1 and cov["n_channels"] == 1
    assert cov["n_duplicates_removed"] == 1
    assert cov["snapshot_total"] == pytest.approx(float(result.frame.k.iloc[0]))
    assert result.manager.submitted == [(0, 17)]


def test_symmetry_duplicates_are_removed_before_accounting(monkeypatch):
    duplicate = run(monkeypatch, [row(17, 0.4, [0], syms=[np.eye(3), np.eye(3)])])
    assert duplicate.coverage["n_channels"] == 1
    assert duplicate.coverage["n_duplicates_removed"] == 1
    assert duplicate.manager.submitted == [(0, 17)]
    assert duplicate.coverage["snapshot_total"] == pytest.approx(
        float(duplicate.frame.k.iloc[0])
    )
    distinct = run(monkeypatch, [row(17, 0.4, [0], syms=[np.eye(3), ROT90])])
    assert distinct.coverage["n_channels"] == 2
    assert distinct.coverage["n_duplicates_removed"] == 0
    assert distinct.manager.submitted == [(0, 17), (0, 17)]
    assert distinct.coverage["snapshot_total"] == pytest.approx(
        2.0 * float(distinct.frame.k.iloc[0])
    )


def test_aliased_logical_ids_share_one_group(monkeypatch):
    # A retained row whose id column was upcast to float and a numpy-typed
    # frame id name the same logical reference.
    result = run(
        monkeypatch,
        [row(17, 0.4, [0, 1])],
        retained_rows=[(0, 17, "T")],
        alias_ids=True,
    )
    cov = result.coverage
    assert cov["n_groups"] == 1 and cov["n_retained"] == 1 and cov["n_prospective"] == 1
    assert result.manager.submitted == [(1, 17)]
    assert cov["selected_ids"] == (17,)


def test_noncontiguous_ids_rank_descending_with_deterministic_tie_order(monkeypatch):
    specs = [
        row(31, 0.4, [0], k=5.0),
        row(3, 0.4, [1], k=3.0),
        row(17, 0.4, [2], k=2.0),
    ]
    result = run(monkeypatch, specs, threshold=0.75)
    assert result.coverage["selected_ids"] == (31, 3)
    assert result.coverage["selected_fraction"] == pytest.approx(0.8)
    assert sorted(result.manager.submitted) == [(0, 31), (1, 3)]
    ties = run(monkeypatch, [row(31, 0.4, [0]), row(3, 0.4, [1]), row(17, 0.4, [2])])
    assert ties.coverage["selected_ids"] == (3, 17, 31)


@pytest.mark.parametrize("temperature, expected", [(300.0, (17,)), (1000.0, (24,))])
def test_temperature_moves_the_cumulative_cut(monkeypatch, temperature, expected):
    result = run(
        monkeypatch,
        [row(17, 0.4, [0]), row(24, 0.525, [1], nu0=100e12)],
        temperature=temperature,
        threshold=0.5,
    )
    assert result.coverage["selected_ids"] == expected
    assert [ref for _, ref in result.manager.submitted] == list(expected)


def test_boundary_ties_are_included(monkeypatch):
    specs = [
        row(3, 0.4, [0], k=2.0),
        row(17, 0.4, [1], k=1.0),
        row(24, 0.4, [2], k=1.0),
    ]
    cut_at_first = run(monkeypatch, specs, threshold=0.5)
    assert cut_at_first.coverage["selected_ids"] == (3,)
    cut_inside_tie = run(monkeypatch, specs, threshold=0.75)
    assert cut_inside_tie.coverage["selected_ids"] == (3, 17, 24)
    assert cut_inside_tie.coverage["selected_fraction"] == pytest.approx(1.0)


def test_threshold_one_selects_every_positive_group(monkeypatch):
    specs = [row(17, 0.4, [0]), row(24, 0.6, [1]), row(31, 100.0, [2])]
    result = run(monkeypatch, specs, threshold=1.0)
    assert float(result.frame.k.iloc[2]) == 0.0
    assert result.coverage["selected_ids"] == (17, 24)
    assert sorted(result.manager.submitted) == [(0, 17), (1, 24)]
    assert result.coverage["selected_fraction"] == pytest.approx(1.0)
    assert result.active.query("num_reference_event == 31").refined.tolist() == ["F"]


@pytest.mark.parametrize("style", STYLES)
def test_zero_total_dispatches_nothing_and_claims_no_coverage(monkeypatch, style):
    result = run(monkeypatch, [row(17, 100.0, [0]), row(24, 100.0, [1])], style=style)
    cov = result.coverage
    assert cov["applicable"] is False and cov["snapshot_total"] == 0.0
    assert cov["selected_fraction"] is None and cov["refined_fraction"] is None
    assert cov["selected_ids"] == ()
    assert result.manager.submitted == [] and result.site.submitted == []
    assert result.active.k.tolist() == [0.0, 0.0]
    assert result.active.refined.tolist() == ["F", "F"]


# --------------------------------------------------------------------------- #
# Ledger contents: rejected, excluded and retained channels
# --------------------------------------------------------------------------- #


def test_nonfinite_and_negative_rates_are_rejected_not_counted(monkeypatch):
    specs = [
        row(17, 0.4, [0]),
        row(24, 0.4, [1], k=float("nan")),
        row(31, 0.4, [2], k=-1.0),
    ]
    result = run(monkeypatch, specs)
    cov = result.coverage
    assert cov["n_rejected"] == 2 and cov["n_channels"] == 1
    assert cov["snapshot_total"] == pytest.approx(float(result.frame.k.iloc[0]))
    assert result.manager.submitted == [(0, 17)]
    errors = [r.err_value() for r in result.results if not r.is_ok()]
    assert len(errors) == 2
    assert {e.type for e in errors} == {ErrorType.REFINEMENT_INVALID_RATE}
    assert any(
        "rejected" in msg and "rate" in msg
        for level, msg in result.log.lines
        if level == "warning"
    )
    assert len(result.active) == 1


@pytest.mark.parametrize("style", STYLES)
def test_fallback_k0_rates_count_like_any_other(monkeypatch, style):
    result = run(
        monkeypatch, [row(17, 0.4, [0], nu0=None), row(24, 0.4, [1])], style=style
    )
    assert result.coverage["n_channels"] == 2 and result.coverage["n_rejected"] == 0
    assert sorted(result.manager.submitted) == [(0, 17), (1, 24)]
    fallback = result.active.query("num_reference_event == 17").iloc[0]
    assert fallback.nu0_status == "rejected" and fallback.nu0_source == "k0"
    assert result.summary == {
        "attempted": 2,
        "ok": 1,
        "rejected": 1,
        "no_geometry": 0,
        "identityless": 0,
    }


def test_psr_failures_are_excluded_before_the_denominator_is_fixed(monkeypatch):
    result = run(monkeypatch, [row(17, 0.4, [0, 1, 2])], psr_fail=[(1, 17)])
    cov = result.coverage
    assert cov["n_excluded"] == 1 and cov["n_channels"] == 2
    assert cov["snapshot_total"] == pytest.approx(2.0 * float(result.frame.k.iloc[0]))
    assert cov["selected_fraction"] == pytest.approx(1.0)
    assert sorted(result.manager.submitted) == [(0, 17), (2, 17)]
    assert sum(1 for r in result.results if not r.is_ok()) == 1
    assert len(result.active) == 2


def test_retained_refined_rows_count_once_with_their_site_rate(monkeypatch):
    result = run(monkeypatch, [row(17, 0.4, [0, 1])], retained_rows=[(0, 17, "T")])
    k_ref = float(result.frame.k.iloc[0])
    k_site = result.retained_rates[0]
    assert k_site == pytest.approx(5.0 * k_ref), (
        "the retained row carries its site rate"
    )
    cov = result.coverage
    assert (
        cov["n_retained"] == 1 and cov["n_prospective"] == 1 and cov["n_channels"] == 2
    )
    assert cov["snapshot_total"] == pytest.approx(k_site + k_ref)
    assert result.manager.submitted == [(1, 17)]
    assert cov["refined_fraction"] == pytest.approx(1.0)
    assert result.refinement.superseded_rows == set()
    table = result.active
    assert table.query("atom_index == 0").k.tolist() == [k_site]
    assert table.refined.tolist() == ["T", "T"]


def test_retained_unrefined_row_dispatches_when_selected_and_is_superseded(monkeypatch):
    result = run(monkeypatch, [row(17, 0.4, [0])], retained_rows=[(0, 17, "F")])
    cov = result.coverage
    assert cov["n_duplicates_removed"] == 1 and cov["n_channels"] == 1
    assert result.manager.submitted == [(0, 17)]
    assert result.refinement.superseded_rows == {0}
    assert result.dropped == 1
    assert result.active.refined.tolist() == ["T"]
    assert result.active.atom_index.tolist() == [0]


def test_retained_unrefined_row_outside_the_cut_stays_as_fallback(monkeypatch):
    specs = [row(17, 0.4, [0], k=0.01), row(24, 0.4, [1], k=1.0)]
    result = run(monkeypatch, specs, retained_rows=[(0, 17, "F")], threshold=0.5)
    assert result.coverage["selected_ids"] == (24,)
    assert result.manager.submitted == [(1, 24)]
    assert result.refinement.superseded_rows == set() and result.dropped == 0
    rows = result.active.query("num_reference_event == 17")
    assert rows.refined.tolist() == ["F"], "no duplicate F output for a retained pair"


def test_retained_unrefined_row_whose_psr_fails_now_counts_as_retained(monkeypatch):
    result = run(
        monkeypatch,
        [row(17, 0.4, [0])],
        retained_rows=[(0, 17, "F")],
        psr_fail=[(0, 17)],
    )
    cov = result.coverage
    assert cov["n_retained"] == 1 and cov["n_duplicates_removed"] == 0
    assert cov["n_excluded"] == 1 and cov["n_prospective"] == 0
    assert result.manager.submitted == []
    assert result.active.refined.tolist() == ["F"]
    # Selected but not refinable this step: an explicit shortfall, never a
    # silent "refined 0 %, shortfall 0 %" (review R12, minor).
    assert cov["selected_ids"] == (17,) and cov["n_selected_unrefined"] == 1
    assert cov["refined_fraction"] == pytest.approx(0.0)
    assert cov["shortfall_fraction"] == pytest.approx(1.0)


def test_partial_failure_of_a_multisymmetry_retained_pair_keeps_the_failed_fallback(
    monkeypatch,
):
    # Reference 17 has two symmetric applications on atom 0, both retained as
    # generic rows; the rotated one fails on re-dispatch. Only the fallback
    # row of the successful application may leave (review R12, blocker).
    result = run(
        monkeypatch,
        [row(17, 0.4, [0], syms=[np.eye(3), ROT90])],
        retained_rows=[(0, 17, "F"), (0, 17, "F", "rotated")],
        fail_rotated=True,
    )
    k_ref = float(result.frame.k.iloc[0])
    cov = result.coverage
    assert result.manager.submitted == [(0, 17), (0, 17)]
    assert sorted(result.manager.geometries) == ["identity", "rotated"]
    assert cov["n_duplicates_removed"] == 2 and cov["n_channels"] == 2
    assert cov["snapshot_total"] == pytest.approx(2.0 * k_ref)
    assert cov["n_dispatch_ok"] == 1 and cov["n_dispatch_failed"] == 1
    assert cov["refined_fraction"] == pytest.approx(0.5)
    assert cov["shortfall_fraction"] == pytest.approx(0.5)
    assert len(result.refinement.superseded_rows) == 1
    table = result.active
    assert sorted(table.refined.tolist()) == ["F", "T"], table
    assert float(table.k.sum()) == pytest.approx(2.0 * k_ref), (
        "the failed application keeps its generic row in the BKL sum"
    )
    kept = table[table.refined == "F"].iloc[0]
    assert np.asarray(kept.saddle_positions)[0, 1] == pytest.approx(17.0), (
        "the surviving generic row is the rotated (failed) application"
    )


def test_retained_unrefined_pair_with_a_rejected_prospective_rate_still_counts(
    monkeypatch,
):
    # The reference row lost its rate this step (NaN): its prospective channel
    # is rejected, the retained generic row is still a valid channel with its
    # own rate and the ledger says so (review R12, minor).
    result = run(
        monkeypatch,
        [row(17, 0.4, [0], k=float("nan"))],
        retained_rows=[(0, 17, "F")],
    )
    cov = result.coverage
    k_row = result.retained_rates[0]
    assert cov["n_rejected"] == 1 and cov["n_duplicates_removed"] == 0
    assert cov["n_retained"] == 1 and cov["n_prospective"] == 0
    assert cov["applicable"] and cov["snapshot_total"] == pytest.approx(k_row)
    assert cov["selected_ids"] == (17,)
    assert cov["n_selected_unrefined"] == 1
    assert cov["shortfall_fraction"] == pytest.approx(1.0)
    assert result.manager.submitted == []
    assert result.active.refined.tolist() == ["F"]


def test_mixed_flag_retained_pair_redispatches_only_the_unrefined_application(
    monkeypatch,
):
    # One application of the pair is already refined (T), the other retained
    # generic (F): only the generic one is re-dispatched and the refined one
    # counts once with its site rate (review R12, latent minor).
    result = run(
        monkeypatch,
        [row(17, 0.4, [0], syms=[np.eye(3), ROT90])],
        retained_rows=[(0, 17, "T"), (0, 17, "F", "rotated")],
    )
    k_ref = float(result.frame.k.iloc[0])
    k_site = result.retained_rates[0]
    cov = result.coverage
    assert result.manager.submitted == [(0, 17)]
    assert result.manager.geometries == ["rotated"]
    assert cov["n_retained"] == 1 and cov["n_prospective"] == 1
    assert cov["n_duplicates_removed"] == 2
    assert cov["snapshot_total"] == pytest.approx(k_site + k_ref)
    table = result.active
    assert table.refined.tolist() == ["T", "T"]
    assert float(table.k.sum()) == pytest.approx(k_site + k_ref)


def test_failed_refinement_is_a_shortfall_not_a_new_denominator(monkeypatch):
    specs = [row(17, 0.4, [0], k=3.0), row(24, 0.4, [1], k=2.0)]
    result = run(monkeypatch, specs, partn_fail=[(1, 24)])
    cov = result.coverage
    assert cov["snapshot_total"] == pytest.approx(5.0)
    assert cov["selected_fraction"] == pytest.approx(1.0)
    assert cov["n_dispatched"] == 2 and cov["n_dispatch_ok"] == 1
    assert cov["n_dispatch_failed"] == 1
    assert cov["refined_fraction"] == pytest.approx(0.6)
    assert cov["shortfall_fraction"] == pytest.approx(0.4)
    assert len(result.active) == 1
    assert any("shortfall" in msg for _, msg in result.log.lines)


@pytest.mark.parametrize("style", STYLES)
def test_ledger_and_coverage_are_logged(monkeypatch, style):
    result = run(
        monkeypatch, [row(17, 0.4, [0]), row(24, 0.525, [1], nu0=100e12)], style=style
    )
    ledger = [m for level, m in result.log.lines if "refinement ledger:" in m]
    coverage = [m for level, m in result.log.lines if "refinement coverage:" in m]
    assert len(ledger) == 1 and len(coverage) == 1
    assert "2 channels" in ledger[0] and "2 reference groups" in ledger[0]
    assert "selected 2 groups" in ledger[0]
    assert "refined 100.00 %" in coverage[0] and "shortfall 0.00 %" in coverage[0]


def test_htst_and_rpa_agree(monkeypatch):
    specs = [row(17, 0.4, [0, 1]), row(24, 0.525, [2], nu0=100e12), row(31, 0.6, [3])]
    htst = run(monkeypatch, specs, style="htst", threshold=0.9)
    rpa = run(monkeypatch, specs, style="rpa", threshold=0.9)
    assert htst.manager.submitted == rpa.manager.submitted
    left = {k: v for k, v in htst.coverage.items() if k != "style"}
    right = {k: v for k, v in rpa.coverage.items() if k != "style"}
    assert left == right


# --------------------------------------------------------------------------- #
# Constant mode is untouched
# --------------------------------------------------------------------------- #


def test_constant_barrier_rule_is_byte_identical():
    source = inspect.getsource(Refinement.get_energy_thr_refine)
    assert hashlib.sha256(source.encode()).hexdigest() == GET_ENERGY_THR_REFINE_SHA256


def test_constant_mode_keeps_the_barrier_rule_and_no_ledger(monkeypatch):
    result = run(
        monkeypatch, [row(17, 0.4, [0]), row(24, 0.525, [1])], style="constant"
    )
    assert result.active.refined.tolist() == ["T", "F"]
    assert result.manager.submitted == [(0, 17)]
    assert result.coverage is None
    assert result.refinement.superseded_rows == set()
    assert not any("refinement ledger" in m for _, m in result.log.lines)


def test_constant_mode_still_skips_every_existing_pair(monkeypatch):
    result = run(
        monkeypatch,
        [row(17, 0.4, [0])],
        style="constant",
        retained_rows=[(0, 17, "F")],
    )
    assert result.manager.submitted == []
    assert result.active.refined.tolist() == ["F"]


# --------------------------------------------------------------------------- #
# Active-table helpers and the KMC wiring
# --------------------------------------------------------------------------- #


def test_retained_channels_and_drop_unrefined_rows(monkeypatch):
    cfg = config("htst", 300.0, 0.9999)
    site = SiteBoundary(cfg, {17: 1e12, 24: 1e12})
    active = ActiveEventTable(cfg, prefactor_service=site)
    assert active.retained_channels().empty
    active.add_events(
        [
            retained_output(0, 17, 0.4, "T", 3),
            retained_output(1, 17, 0.4, "F", 3),
            retained_output(2, 24, 0.5, "F", 3),
        ]
    )
    channels = active.retained_channels()
    assert list(channels.columns) == [
        "label",
        "atom_index",
        "num_reference_event",
        "k",
        "refined",
        "crop_atom_ids",
        "saddle_positions",
    ]
    assert channels.label.tolist() == [0, 1, 2]
    assert channels.atom_index.tolist() == [0, 1, 2]
    assert channels.num_reference_event.tolist() == [17, 17, 24]
    assert channels.refined.tolist() == ["T", "F", "F"]
    assert channels.k.tolist() == active.table.k.tolist()
    assert channels.crop_atom_ids.tolist() == [(0,), (1,), (2,)]
    assert np.asarray(channels.saddle_positions[2]).tolist() == [[24.0, 0.0, 0.0]]
    # Only unrefined rows leave, by label; the refined row 0 stays even if named.
    assert active.drop_unrefined_rows({0, 1, 5}) == 1
    assert active.table.atom_index.tolist() == [0, 2]
    assert active.drop_unrefined_rows(set()) == 0


def test_execute_refinements_passes_retained_rows_and_drops_superseded(monkeypatch):
    cfg = config("htst", 300.0, 0.9999)
    specs = [row(17, 0.4, [0, 1])]
    df = frame(specs, "htst", 300.0)
    system = SystemBoundary(2)
    neighbors = SimpleNamespace(
        get_neighbors=lambda kind, atom: np.array([atom]), system=system
    )
    psr_double(monkeypatch, set())
    manager = NativeBoundary({17: 0.4})
    site = SiteBoundary(cfg, {17: 1e12})
    active = ActiveEventTable(cfg, prefactor_service=site)
    build_retained(
        active, site, system, neighbors, {17: 0.4}, [(0, 17, "F"), (1, 17, "T")]
    )
    sim = SimpleNamespace(
        config=cfg,
        loggers=Log(),
        system=system,
        neighbors_list=neighbors,
        atomic_environment=SimpleNamespace(
            get_atoms_with_id=lambda event: np.array([0, 1])
        ),
        manager=manager,
        global_constraints=None,
        total_energy=0.0,
        active_table=active,
        refinement_coverage=None,
    )
    refinement = KMC.execute_refinements(
        sim, df, existing_pairs=active.existing_pairs()
    )
    assert manager.submitted == [(0, 17)], "retained F pair re-dispatched, T pair kept"
    assert refinement.superseded_rows == {0}
    assert active.table.atom_index.tolist() == [1], "superseded F row left before add"
    assert sim.refinement_coverage is refinement.coverage
    assert sim.refinement_coverage["n_retained"] == 1


def test_step_summary_line_reports_refinement_coverage():
    cfg = config("htst", 300.0, 0.9999)
    site = SiteBoundary(cfg, {})
    sim = KMC(cfg, manager=FakeManager())
    assert sim.uses_event_prefactors
    sim.active_table = ActiveEventTable(cfg, prefactor_service=site)
    sim.prefactor_service = site
    sim.reference_table = ReferenceEventTable(cfg, prefactor_service=site)
    summary = {
        "attempted": 0,
        "ok": 0,
        "rejected": 0,
        "no_geometry": 0,
        "identityless": 0,
    }
    sim.loggers = Log()
    sim.refinement_coverage = None
    sim._log_htst_step_summary(summary)
    (line,) = [m for _, m in sim.loggers.lines if "HTST prefactors:" in m]
    assert "refinement coverage: not available" in line
    sim.loggers = Log()
    sim.refinement_coverage = {
        "style": "htst",
        "applicable": True,
        "target": 0.9999,
        "snapshot_total": 5.0,
        "n_channels": 2,
        "n_retained": 0,
        "n_prospective": 2,
        "n_groups": 2,
        "n_excluded": 0,
        "n_rejected": 0,
        "n_duplicates_removed": 0,
        "n_predispatched": 0,
        "selected_ids": (17, 24),
        "selected_fraction": 1.0,
        "n_dispatched": 2,
        "n_dispatch_ok": 1,
        "n_dispatch_failed": 1,
        "n_selected_unrefined": 0,
        "refined_fraction": 0.6,
        "shortfall_fraction": 0.4,
    }
    sim._log_htst_step_summary(summary)
    (line,) = [m for _, m in sim.loggers.lines if "HTST prefactors:" in m]
    assert "refinement coverage: selected=1.0000" in line
    assert "refined=0.6000" in line and "shortfall=0.4000" in line
    assert "failed=1" in line and "dispatched=2" in line
    assert "left_unrefined=0" in line
    sim.loggers = Log()
    sim.refinement_coverage = {
        "style": "htst",
        "applicable": False,
        "snapshot_total": 0.0,
    }
    sim._log_htst_step_summary(summary)
    (line,) = [m for _, m in sim.loggers.lines if "HTST prefactors:" in m]
    assert "refinement coverage: not applicable" in line
