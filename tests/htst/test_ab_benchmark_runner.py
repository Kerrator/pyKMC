"""Fast tests for the portable HTST A/B benchmark runner."""

from __future__ import annotations

import configparser
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys


_ROOT = Path(__file__).resolve().parents[2]
_RUNNER = _ROOT / "scripts" / "run_htst_ab.py"


def _config(path: Path) -> configparser.ConfigParser:
    parser = configparser.ConfigParser()
    parser.read(path)
    return parser


def test_dry_run_builds_self_contained_paired_inputs(tmp_path: Path) -> None:
    """A dry run copies assets and changes only the prefactor style."""
    source = tmp_path / "source"
    source.mkdir()
    (source / "initial_config.xyz").write_text("1\n\nNi 0 0 0\n")
    (source / "Ni.eam").write_text("placeholder\n")
    (source / "input.in").write_text(
        """
[Control]
initial_config = ./initial_config.xyz
n_steps = 4
engine = lammps
n_sessions = 1
engine_use_rank_0 = True
basin = True
recycle = True

[Lammps]
pair_style = eam/alloy
pair_coeff = * * ./Ni.eam Ni

[EventSearch]
style = partn
nsearch = 10

[pARTn]
zseed = 0

[RateConstant]
style = constant
k0 = 1e13
T = 500

[EventRecycling]
style = displacement
""".strip()
        + "\n"
    )

    output = tmp_path / "output"
    subprocess.run(
        [
            sys.executable,
            str(_RUNNER),
            "--input",
            str(source / "input.in"),
            "--output",
            str(output),
            "--python",
            sys.executable,
            "--launcher",
            "direct",
            "--nproc",
            "1",
            "--steps",
            "1",
            "--nsearch",
            "2",
            "--repeats",
            "2",
            "--skip-preflight",
            "--dry-run",
        ],
        cwd=_ROOT,
        check=True,
    )

    for repeat in ("repeat_001", "repeat_002"):
        constant = output / repeat / "constant"
        htst = output / repeat / "htst"
        constant_config = _config(constant / "input.in")
        htst_config = _config(htst / "input.in")
        assert constant_config["RateConstant"]["style"] == "constant"
        assert htst_config["RateConstant"]["style"] == "htst"
        assert constant_config["Control"].getboolean("basin") is False
        assert constant_config["Control"].getboolean("recycle") is False
        assert constant_config["EventSearch"].getint("nsearch") == 2
        assert not constant_config.has_section("EventRecycling")
        assert (constant / "initial_config.xyz").is_file()
        assert (constant / "Ni.eam").is_file()
        assert (htst / "initial_config.xyz").is_file()
        assert (htst / "Ni.eam").is_file()

        constant_text = (constant / "input.in").read_text()
        htst_text = (htst / "input.in").read_text()
        assert constant_text.replace("style = constant", "style = htst") == htst_text

    metadata = json.loads((output / "metadata.json").read_text())
    assert metadata["nproc"] == 1
    assert metadata["n_sessions"] == 1
    assert metadata["engine_use_rank_0"] is True
    assert metadata["basin_enabled"] is False
    assert metadata["recycling_enabled"] is False
    assert metadata["execution_mode"] == "sequential"
    assert metadata["nsearch_override"] == 2


def test_run_checked_records_elapsed_wall_time(tmp_path: Path) -> None:
    """The launcher timer measures elapsed wall time and captures output."""
    spec = importlib.util.spec_from_file_location("run_htst_ab", _RUNNER)
    assert spec and spec.loader
    runner = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(runner)

    stdout = tmp_path / "stdout"
    stderr = tmp_path / "stderr"
    returncode, wall_s = runner._run_checked(
        [
            sys.executable,
            "-c",
            "import time; time.sleep(0.05); print('timed')",
        ],
        tmp_path,
        stdout,
        stderr,
        2.0,
        dict(os.environ),
    )

    assert returncode == 0
    assert wall_s >= 0.04
    assert stdout.read_text().strip() == "timed"
