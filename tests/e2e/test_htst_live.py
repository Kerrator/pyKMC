"""Live Si_vac runs: constant baseline, htst with one- and two-rank workers.

Opt-in (``PYKMC_RUN_E2E=1``, see ``conftest.py``). Each case launches the
real ``pykmc.run.main`` under ``mpirun`` on ``examples/Si_vac`` (511-atom SW
silicon, one vacancy) for three KMC steps and checks the run from the
outside: exit code, the imported tree, the reference table (statuses, units,
consistency of ``k`` with ``k_prefactor``), the clock in the output file, the
per-step HTST summary in the log (no active row on the ``k0`` fallback, site
attempts accepted) and the absence of scratch-engine log files. The asserted
numbers are unit and status invariants, not oracles for the prefactor values:
those live in ``tests/engine/test_htst_lammps.py``.
"""

from __future__ import annotations

import glob
import pickle
import re

import numpy as np
import pandas as pd
import pytest

from tests.e2e.conftest import REPO_ROOT, LiveRun

pytest.importorskip("lammps")

KB_EV_K = 8.6173303e-05
"""``pykmc.config.PhysicalConstants.kb``; imported lazily below to keep collection cheap."""
T_K = 300.0
K0_PS = 10.0
K_PREFACTOR_MAX_PS = 1.0e4
"""Above this a prefactor was written in Hz, not ps^-1 (``RateConstantConfig.K0_MAX_PS_INV``)."""
SCRATCH_ID_MIN = 1_000_000
"""Scratch engines derive ids from ``_SCRATCH_ID_STRIDE``; a log with such an id leaked."""
N_STEPS = 3
SUMMARY_RE = re.compile(
    r"HTST prefactors: reference ok=(?P<ref_ok>\d+) rejected=(?P<ref_rejected>\d+) "
    r"legacy=(?P<ref_legacy>\d+) pending=(?P<ref_pending>\d+); "
    r"active sources reference=(?P<src_reference>\d+) site=(?P<src_site>\d+) "
    r"k0=(?P<src_k0>\d+); site attempts this step=(?P<attempted>\d+) "
    r"\(ok=(?P<site_ok>\d+), rejected=(?P<site_rejected>\d+)\)"
)
"""The per-step line written by ``KMC._log_htst_step_summary``."""


def _reference_table(run: LiveRun) -> pd.DataFrame:
    """Load the run's saved reference table.

    Parameters
    ----------
    run : LiveRun
        A completed child run.

    Returns
    -------
    pandas.DataFrame
        The unpickled ``reference_table.pickle`` (with its ``attrs``).

    """
    path = run.directory / "reference_table.pickle"
    assert path.is_file(), f"no reference table saved in {run.directory}"
    with open(path, "rb") as handle:
        return pickle.load(handle)


def _output_rows(run: LiveRun) -> pd.DataFrame:
    """Parse the numeric rows of ``pykmc.out`` (step, dT[s], T[s], ..., k_tot).

    Parameters
    ----------
    run : LiveRun
        A completed child run.

    Returns
    -------
    pandas.DataFrame
        One row per completed step with ``step``, ``dT``, ``T``, ``k_evt``
        and ``k_tot``; the step-0 header row is excluded.

    """
    rows = []
    for line in (run.directory / "pykmc.out").read_text().splitlines():
        parts = line.split()
        if len(parts) >= 7 and parts[0].isdigit() and int(parts[0]) > 0:
            rows.append(
                {
                    "step": int(parts[0]),
                    "dT": float(parts[1]),
                    "T": float(parts[2]),
                    "k_evt": float(parts[5]),
                    "k_tot": float(parts[6]),
                }
            )
    return pd.DataFrame(rows)


def _assert_common(run: LiveRun) -> pd.DataFrame:
    """Check the style-independent invariants and return the parsed output rows.

    Parameters
    ----------
    run : LiveRun
        A completed child run.

    Returns
    -------
    pandas.DataFrame
        The parsed ``pykmc.out`` rows (see :func:`_output_rows`).

    """
    assert run.returncode == 0, f"exit {run.returncode}\n{run.stderr[-4000:]}"
    imported = [
        line for line in run.stdout.splitlines() if "[e2e] pykmc.__file__" in line
    ]
    assert len(imported) == run.n_ranks, (
        f"expected one import line per rank:\n{run.stdout}"
    )
    for line in imported:
        assert str(REPO_ROOT) in line, line
    out = _output_rows(run)
    assert list(out["step"]) == list(range(1, N_STEPS + 1)), (
        f"expected {N_STEPS} completed steps, got\n{out}\nlog tail:\n{run.log[-3000:]}"
    )
    # The clock is accumulated in seconds from a ps interval: strictly
    # increasing, a running sum of dT, and dT * k_tot * 1e12 is the O(1)
    # exponential draw (a unit slip would make it 1e12 off).
    assert np.all(np.diff(np.concatenate([[0.0], out["T"].to_numpy()])) > 0)
    assert np.allclose(np.cumsum(out["dT"]), out["T"], rtol=1e-6)
    draws = out["dT"].to_numpy() * out["k_tot"].to_numpy() * 1e12
    assert np.all((draws > 1e-6) & (draws < 1e3)), draws
    leaked = [
        p
        for p in glob.glob(str(run.directory / "lammps.log.*"))
        if p.rsplit(".", 1)[1].isdigit() and int(p.rsplit(".", 1)[1]) >= SCRATCH_ID_MIN
    ]
    assert leaked == [], f"scratch engine log files leaked: {leaked}"
    return out


def _assert_htst_table(table: pd.DataFrame) -> None:
    """HTST schema, statuses and unit invariants of a saved reference table.

    Parameters
    ----------
    table : pandas.DataFrame
        A reference table saved by an ``htst`` run.

    """
    for column in ("k_prefactor", "nu0", "nu0_status", "nu0_reason"):
        assert column in table.columns, table.columns
    status = table["nu0_status"].astype(str)
    assert (status == "ok").sum() >= 1, status.value_counts()
    assert not (status == "pending").any(), status.value_counts()
    assert set(status) <= {"ok", "rejected"}, status.value_counts()
    prefactor = table["k_prefactor"].astype(float).to_numpy()
    assert np.all(np.isfinite(prefactor)) and np.all(prefactor > 0.0)
    assert prefactor.max() <= K_PREFACTOR_MAX_PS, "k_prefactor is not in ps^-1"
    ok = (status == "ok").to_numpy()
    nu0 = table["nu0"].astype(float).to_numpy()
    assert np.allclose(prefactor[ok], nu0[ok] * 1e-12, rtol=1e-12)
    assert np.allclose(prefactor[~ok], K0_PS) if (~ok).any() else True
    barrier = table["energy_barrier"].astype(float).to_numpy()
    rate = table["k"].astype(float).to_numpy()
    assert np.allclose(rate, prefactor * np.exp(-barrier / (KB_EV_K * T_K)), rtol=1e-6)
    meta = dict(table.attrs)
    assert meta.get("nu0_units") == "Hz" and meta.get("k_prefactor_units") == "ps^-1", (
        meta
    )
    assert (
        meta.get("style") == "htst" and meta.get("T") == T_K and meta.get("k0") == K0_PS
    )


def _htst_summaries(log: str) -> list[dict[str, int]]:
    """Parse the per-step ``HTST prefactors:`` summary lines of ``pykmc.log``.

    Parameters
    ----------
    log : str
        The log text with colour codes removed.

    Returns
    -------
    list of dict
        One ``{field: count}`` mapping per summary line, in log order, with
        the group names of :data:`SUMMARY_RE` as keys.

    """
    parsed = []
    for line in log.splitlines():
        if "HTST prefactors:" not in line:
            continue
        match = SUMMARY_RE.search(line)
        assert match is not None, f"unparsable HTST summary line: {line!r}"
        parsed.append({key: int(value) for key, value in match.groupdict().items()})
    return parsed


def _assert_htst_summaries(log: str) -> None:
    """Every completed step priced its active events with live HTST estimates.

    A saved reference table can satisfy :func:`_assert_htst_table` while every
    active row is on the ``k0`` fallback (an accepted reference row that no
    active event inherits, or an inherited row that was rejected): the active
    table is not saved, so the per-step summary is the only handle on what
    entered the live rates. Hence, on every step, no active row may be
    ``k0``-sourced, at least one must be ``site``- or ``reference``-sourced
    and no site attempt may be rejected (every Si_vac site estimate lies
    inside the 1-100 THz window); at least one step must have an accepted
    site attempt.

    Parameters
    ----------
    log : str
        The log text with colour codes removed.

    """
    summaries = _htst_summaries(log)
    assert len(summaries) == N_STEPS, log[-3000:]
    for step, counts in enumerate(summaries, start=1):
        assert counts["ref_pending"] == 0, (step, counts)
        assert counts["ref_legacy"] == 0, (step, counts)
        assert counts["src_k0"] == 0, f"step {step}: active rows on k0: {counts}"
        assert counts["src_site"] + counts["src_reference"] >= 1, (step, counts)
        assert counts["site_rejected"] == 0, (step, counts)
        assert counts["attempted"] == counts["site_ok"] + counts["site_rejected"], (
            step,
            counts,
        )
    assert any(counts["site_ok"] >= 1 for counts in summaries), summaries
    assert "site prefactor rejected" not in log


def test_constant_baseline(run_si_vac) -> None:  # noqa: ANN001
    """Constant style completes with the base schema and ``k = k0 exp(-Ea/kT)``.

    Parameters
    ----------
    run_si_vac : Callable[..., LiveRun]
        The runner fixture from ``conftest.py``.

    """
    run = run_si_vac(style="constant", n_sessions=3, ranks_per_worker=1)
    _assert_common(run)
    table = _reference_table(run)
    assert len(table) >= 1
    for column in ("k_prefactor", "nu0", "nu0_status"):
        assert column not in table.columns, "constant mode must keep the base schema"
    assert dict(table.attrs) == {}
    rate = table["k"].astype(float).to_numpy()
    barrier = table["energy_barrier"].astype(float).to_numpy()
    assert np.allclose(rate, K0_PS * np.exp(-barrier / (KB_EV_K * T_K)), rtol=1e-6)
    assert "HTST prefactors:" not in run.log
    assert "[htst]" not in run.log


@pytest.mark.parametrize(
    ("n_sessions", "ranks_per_worker"),
    [
        pytest.param(3, 1, id="one-rank-workers"),
        pytest.param(2, 2, id="two-rank-workers"),
    ],
)
def test_htst_live(run_si_vac, n_sessions: int, ranks_per_worker: int) -> None:  # noqa: ANN001
    """HTST style completes with accepted prefactors on 1- and 2-rank workers.

    ``mpirun -n 4`` with three one-rank workers and ``mpirun -n 5`` with two
    two-rank workers (the non-root rank of a worker returns ``None`` and the
    manager reports the root's result).

    Parameters
    ----------
    run_si_vac : Callable[..., LiveRun]
        The runner fixture from ``conftest.py``.
    n_sessions : int
        Number of worker sessions.
    ranks_per_worker : int
        MPI ranks per worker session.

    """
    run = run_si_vac(
        style="htst", n_sessions=n_sessions, ranks_per_worker=ranks_per_worker
    )
    _assert_common(run)
    _assert_htst_table(_reference_table(run))
    _assert_htst_summaries(run.log)
    assert "[htst] reference event" in run.log
    assert "HTST prefactor service ready" in run.log
    assert "is a legacy table" not in run.log
