"""Opt-in wrapper for the portable constant-vs-HTST A/B benchmark.

Set ``RUN_HTST_AB=1`` and ``HTST_AB_DIR`` to a directory containing one
portable ``input.in`` plus its referenced structure and potential. The
standalone runner generates both styles, isolates outputs, performs dependency
checks, and writes durable reports.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
import subprocess
import sys

import pytest

pytest.importorskip("lammps")

_OPT_IN = os.environ.get("RUN_HTST_AB") == "1"
_ROOT = Path(__file__).resolve().parents[2]
_RUNNER = _ROOT / "scripts" / "run_htst_ab.py"


@pytest.mark.skipif(not _OPT_IN, reason="set RUN_HTST_AB=1 to run the heavy A/B")
def test_ab_characterization() -> None:
    """Run one paired benchmark and assert that the generated report passes."""
    ab_dir = Path(os.environ.get("HTST_AB_DIR", ""))
    assert ab_dir.is_dir(), "set HTST_AB_DIR to a prepared run directory"
    input_path = ab_dir / os.environ.get("HTST_AB_INPUT", "input.in")
    assert input_path.is_file(), f"base input not found: {input_path}"
    output = Path(os.environ.get("HTST_AB_OUTPUT", str(ab_dir / "ab_results")))
    launcher = os.environ.get("MPIRUN", "mpirun")
    command = [
        sys.executable,
        str(_RUNNER),
        "--input",
        str(input_path),
        "--output",
        str(output),
        "--python",
        sys.executable,
        "--launcher",
        launcher,
        "--nproc",
        os.environ.get("HTST_AB_NPROC", "8"),
        "--steps",
        os.environ.get("HTST_AB_STEPS", "1"),
        "--nsearch",
        os.environ.get("HTST_AB_NSEARCH", "1"),
        "--repeats",
        os.environ.get("HTST_AB_REPEATS", "1"),
        "--overwrite",
    ]
    if nproc_flag := os.environ.get("HTST_AB_NPROC_FLAG"):
        command.extend(["--nproc-flag", nproc_flag])
    if os.environ.get("HTST_AB_SKIP_PREFLIGHT") == "1":
        command.append("--skip-preflight")
    subprocess.run(command, cwd=_ROOT, check=True)

    summary = json.loads((output / "summary.json").read_text())
    assert summary["status"] == "passed", summary["failures"]
