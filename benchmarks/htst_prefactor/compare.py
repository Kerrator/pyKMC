"""Turn one or more run_profile CSVs into a markdown report + recommendation.

Per system, aggregates the per-event total time (sum of the 3 Hessians + the
shared eigh) for each (mode, free_radius), reports median +/- spread over
repeats, the fd/eskm speedup, the cross-mode nu0 agreement, and the
production-relevant engine round-trip counts (eskm = 3 per event; fd grows with
n_free). Prints the fastest mode at the production free_radius and flags the
messenger-round-trip asymmetry that matters once the prefactor runs through a
remote/global engine (the Option A backend). See README.md for usage.
"""

from __future__ import annotations

import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_REPO_ROOT))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

_PROD_RADIUS = 6.0


def _event_totals(df: pd.DataFrame) -> pd.DataFrame:
    """Collapse per-Hessian rows to one per (system, mode, event, radius, repeat).

    t_event = sum of the 3 Hessian wall times + the shared eigh time (stored on
    hessian_idx 0). n_free / nu0 are constant within the group.
    """
    grp = df.groupby(["system", "mode", "event", "free_radius", "repeat"], as_index=False)
    out = grp.agg(
        t_event=("t_hessian", "sum"),
        t_eigh=("t_eigh", "sum"),
        n_free=("n_free", "first"),
        nu0_thz=("nu0_thz", "first"),
        round_trips=("round_trips", "sum"),
    )
    out["t_event"] = out["t_event"] + out["t_eigh"]
    return out


def _median_spread(values: "pd.Series[float]") -> tuple[float, float]:
    """Return (median, half-range spread) of a numeric series."""
    arr = np.asarray(values, dtype=float)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return float("nan"), float("nan")
    return float(np.median(arr)), float((arr.max() - arr.min()) / 2.0)


def report(df: pd.DataFrame) -> None:
    """Print the per-system markdown table and the recommendation."""
    totals = _event_totals(df)
    for system in sorted(totals["system"].unique()):
        sdf = totals[totals["system"] == system]
        radii = sorted(sdf["free_radius"].unique())
        print(f"\n## {system}\n")
        print("| radius (A) | n_free | t_event fd (s) | t_event eskm (s) | speedup fd/eskm | nu0 fd (THz) | nu0 eskm (THz) | rel_err | rt fd | rt eskm |")
        print("|---|---|---|---|---|---|---|---|---|---|")
        for r in radii:
            rdf = sdf[sdf["free_radius"] == r]
            fd = rdf[rdf["mode"] == "fd"]
            es = rdf[rdf["mode"] == "eskm"]
            n_free = int(rdf["n_free"].iloc[0])
            t_fd, s_fd = _median_spread(fd["t_event"]) if len(fd) else (float("nan"), float("nan"))
            t_es, s_es = _median_spread(es["t_event"]) if len(es) else (float("nan"), float("nan"))
            speedup = (t_fd / t_es) if (t_es and np.isfinite(t_es) and t_es > 0) else float("nan")
            nu0_fd = float(fd["nu0_thz"].iloc[0]) if len(fd) and pd.notna(fd["nu0_thz"].iloc[0]) else float("nan")
            nu0_es = float(es["nu0_thz"].iloc[0]) if len(es) and pd.notna(es["nu0_thz"].iloc[0]) else float("nan")
            rel = abs(nu0_fd - nu0_es) / nu0_es if (np.isfinite(nu0_es) and nu0_es != 0) else float("nan")
            flag = " (>1%!)" if (np.isfinite(rel) and rel >= 0.01) else ""
            rt_fd = int(fd["round_trips"].iloc[0]) if len(fd) else 0
            rt_es = int(es["round_trips"].iloc[0]) if len(es) else 0
            print(
                f"| {r:g} | {n_free} | {t_fd:.4f}±{s_fd:.4f} | {t_es:.4f}±{s_es:.4f} | "
                f"{speedup:.2f} | {nu0_fd:.2f} | {nu0_es:.2f} | {rel:.3%}{flag} | {rt_fd} | {rt_es} |"
            )

    print("\n## Recommendation\n")
    for system in sorted(totals["system"].unique()):
        sdf = totals[totals["system"] == system]
        radii = sorted(sdf["free_radius"].unique())
        target = min(radii, key=lambda x: abs(x - _PROD_RADIUS))
        rdf = sdf[sdf["free_radius"] == target]
        t_fd, _ = _median_spread(rdf[rdf["mode"] == "fd"]["t_event"])
        t_es, _ = _median_spread(rdf[rdf["mode"] == "eskm"]["t_event"])
        n_free = int(rdf["n_free"].iloc[0])
        if np.isfinite(t_fd) and np.isfinite(t_es):
            rt_fd_ev = int(rdf[rdf["mode"] == "fd"]["round_trips"].iloc[0])
            rt_es_ev = int(rdf[rdf["mode"] == "eskm"]["round_trips"].iloc[0])
            faster = "eskm" if t_es < t_fd else "fd"
            ratio = (max(t_fd, t_es) / min(t_fd, t_es)) if min(t_fd, t_es) > 0 else float("nan")
            rt_ratio = (rt_fd_ev / rt_es_ev) if rt_es_ev else float("nan")
            print(
                f"- **{system}** @ r={target:g} (n_free={n_free}): serial-fastest = **{faster}** "
                f"({ratio:.1f}x). Engine round-trips/event: eskm={rt_es_ev} vs fd={rt_fd_ev} "
                f"-> through a remote/global engine (Option A backend) eskm wins by the "
                f"messenger-traffic ratio {rt_ratio:.0f}x regardless of local compute."
            )
        elif np.isfinite(t_es):
            print(f"- **{system}** @ r={target:g}: only eskm ran (FD missing).")
        elif np.isfinite(t_fd):
            print(f"- **{system}** @ r={target:g}: only fd ran (eskm/PHONON unavailable).")


def main(argv: "list[str] | None" = None) -> None:
    """Read the CSV path(s) and print the report."""
    paths = argv if argv is not None else sys.argv[1:]
    if not paths:
        raise SystemExit("usage: compare.py <profile.csv> [more.csv ...]")
    frames = [pd.read_csv(pth) for pth in paths]
    df = pd.concat(frames, ignore_index=True)
    report(df)


if __name__ == "__main__":
    main()
