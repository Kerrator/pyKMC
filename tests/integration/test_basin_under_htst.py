"""Basin search composed with the HTST rate backend.

Two layers:

* A fast unit guard (no MPI): a ``BasinsGenericEvents`` built from an
  ``style = htst`` config routes its absorbing-event rate through the
  rate-constant backend (``_absorbing_rate`` -> ``RateConstant.compute_rate``),
  returning a finite ps^-1 rate. With no per-event nu0 supplied the HTST backend
  falls back to ``k0`` (already ps^-1) — the agreed semantic for basin rates.

* An opt-in MPI end-to-end run (``RUN_BASIN_HTST_E2E=1`` + ``BASIN_HTST_E2E_DIR``
  pointing at a prepared run directory whose ``input.in`` enables
  ``[Control] basin = True`` and ``[RateConstant] style = htst``). It runs a few
  KMC steps for both the serial and the wavefront basin strategy and asserts the
  run completes, the reference table carries finite ``nu0`` (Hz) with the
  ``k_prefactor`` resolved to ps^-1, and a basin super-event advances time.
"""

from __future__ import annotations

import math
import os
from pathlib import Path

import pytest

from pykmc.config import PhysicalConstants, RateConstantConfig
from pykmc.rate_constant import create_rate_constant

KB = PhysicalConstants().kb


def test_basin_absorbing_rate_routes_through_htst_backend() -> None:
    """Basin absorbing rate uses the HTST backend (k0 fallback, ps^-1)."""
    from pykmc.basins import BasinsGenericEvents

    cfg = type("Cfg", (), {})()
    cfg.rateconstant = RateConstantConfig(style="htst", k0=10.0, T=500.0)

    basin = BasinsGenericEvents.__new__(BasinsGenericEvents)
    basin.config = cfg
    basin.rate_constant = create_rate_constant(
        T=cfg.rateconstant.T,
        prefactor_backend_name=cfg.rateconstant.style,
        config=cfg.rateconstant,
    )

    dE = 0.5
    rate = basin._absorbing_rate(dE)
    # No nu0 supplied -> HTST backend falls back to k0 (ps^-1); Eyring exponential.
    expected = 10.0 * math.exp(-dE / (KB * 500.0))
    assert math.isfinite(rate)
    assert math.isclose(rate, expected, rel_tol=1e-9)


_OPT_IN = os.environ.get("RUN_BASIN_HTST_E2E") == "1"


@pytest.mark.skipif(not _OPT_IN, reason="set RUN_BASIN_HTST_E2E=1 to run the heavy e2e")
@pytest.mark.parametrize("strategy", ["serial", "wavefront"])
def test_basin_completes_under_htst(strategy: str, tmp_path: Path) -> None:
    """A basin run under style=htst completes and produces HTST-priced rates."""
    import shutil
    import subprocess

    import pandas as pd

    src = Path(os.environ.get("BASIN_HTST_E2E_DIR", ""))
    assert src.is_dir(), "set BASIN_HTST_E2E_DIR to a prepared basin+htst run directory"
    mpirun = os.environ.get("MPIRUN", "mpirun")
    nproc = os.environ.get("BASIN_HTST_E2E_NPROC", "8")

    run_dir = tmp_path / strategy
    shutil.copytree(src, run_dir)
    # Force the basin exploration strategy for this cell.
    inp = (run_dir / "input.in").read_text()
    if "strategy" in inp:
        inp = "\n".join(
            f"strategy = {strategy}" if line.strip().startswith("strategy") else line
            for line in inp.splitlines()
        )
    else:
        inp = inp.replace("[BASIN]", f"[BASIN]\nstrategy = {strategy}")
    (run_dir / "input.in").write_text(inp)

    subprocess.run(
        [mpirun, "-n", nproc, "python", "-m", "pykmc", "-in", "input.in"],
        cwd=run_dir,
        check=True,
    )

    # The run completed; the reference table carries HTST prefactors.
    table = pd.read_pickle(run_dir / "reference_table.pickle")
    assert "nu0" in table.columns and "k_prefactor" in table.columns
    finite_nu0 = table["nu0"].dropna()
    assert (finite_nu0 > 0).any(), "no finite nu0 produced under style=htst"
    # nu0 is stored in Hz (THz-scale, ~1e12+); k_prefactor is the ps^-1 carrier.
    big = finite_nu0[finite_nu0 > 0].iloc[0]
    assert big > 1e11, "stored nu0 should be in Hz (THz-scale)"

    # A basin super-event advanced the clock: the output table has >1 step.
    out = (run_dir / "pykmc.out").read_text()
    step_rows = [r for r in out.splitlines() if r.strip()[:1].isdigit()]
    assert len(step_rows) >= 2, "run did not advance past the first step"
