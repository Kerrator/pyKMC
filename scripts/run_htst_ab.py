"""Run a reproducible pyKMC constant-vs-HTST A/B benchmark.

The runner accepts one pyKMC input file and creates paired runs that differ only
in ``[RateConstant] style``. It copies the initial structure and potential files
into isolated work directories, seeds Python/NumPy/pARTn, measures wall time,
and writes durable CSV/JSON reports.
"""

from __future__ import annotations

import argparse
import configparser
import csv
from datetime import datetime
import hashlib
import json
import math
import os
import platform
import shlex
import shutil
import statistics
import subprocess
import sys
import time
from pathlib import Path
from typing import Any


HTST_DEFAULTS = {
    "free_radius": "6.0",
    "fd_step": "0.01",
    "nu0_min_THz": "1.0",
    "nu0_max_THz": "100.0",
    "require_one_negative_mode": "True",
    "premin": "True",
}

REPORT_FIELDS = [
    "repeat",
    "style",
    "seed",
    "returncode",
    "started_at",
    "finished_at",
    "wall_s",
    "wall_per_event_s",
    "steps_completed",
    "step_wall_sum_s",
    "step_wall_median_s",
    "n_events",
    "k_min",
    "k_median",
    "k_max",
    "has_nu0_column",
    "n_finite_nu0",
    "n_fallback_nu0",
    "nu0_coverage",
    "nu0_min_THz",
    "nu0_median_THz",
    "nu0_max_THz",
    "run_dir",
]


def _parser(path: Path) -> configparser.ConfigParser:
    parser = configparser.ConfigParser()
    parser.optionxform = str
    if not parser.read(path):
        raise ValueError(f"could not read input file: {path}")
    return parser


def _section(parser: configparser.ConfigParser, name: str) -> str:
    for section in parser.sections():
        if section.lower() == name.lower():
            return section
    raise ValueError(f"missing required [{name}] section")


def _get_option(
    parser: configparser.ConfigParser, section: str, option: str
) -> str | None:
    for key, value in parser.items(section):
        if key.lower() == option.lower():
            return value
    return None


def _set_option(
    parser: configparser.ConfigParser, section: str, option: str, value: object
) -> None:
    for key in parser[section]:
        if key.lower() == option.lower():
            parser[section][key] = str(value)
            return
    parser[section][option] = str(value)


def _remove_option(
    parser: configparser.ConfigParser, section: str, option: str
) -> None:
    for key in list(parser[section]):
        if key.lower() == option.lower():
            parser.remove_option(section, key)


def _remove_section(parser: configparser.ConfigParser, name: str) -> None:
    for section in parser.sections():
        if section.lower() == name.lower():
            parser.remove_section(section)
            return


def _resolve_source(raw: str, base_dir: Path) -> Path:
    source = Path(raw).expanduser()
    if not source.is_absolute():
        source = base_dir / source
    source = source.resolve()
    if not source.is_file():
        raise FileNotFoundError(f"required input asset not found: {source}")
    return source


def _copy_asset(source: Path, work_dir: Path, copied: dict[str, Path]) -> str:
    name = source.name
    previous = copied.get(name)
    if previous is not None and previous != source:
        raise ValueError(
            f"asset filename collision: {previous} and {source} are both named {name}"
        )
    copied[name] = source
    shutil.copy2(source, work_dir / name)
    return f"./{name}"


def _rewrite_pair_coeff(
    raw: str, base_dir: Path, work_dir: Path, copied: dict[str, Path]
) -> str:
    tokens = shlex.split(raw)
    rewritten: list[str] = []
    for token in tokens:
        candidate = Path(token).expanduser()
        if not candidate.is_absolute():
            candidate = base_dir / candidate
        if candidate.is_file():
            rewritten.append(_copy_asset(candidate.resolve(), work_dir, copied))
        else:
            rewritten.append(token)
    return " ".join(rewritten)


def prepare_input(
    base_input: Path,
    work_dir: Path,
    style: str,
    seed: int,
    steps: int | None,
    nsearch: int | None,
    htst_options: dict[str, str],
) -> Path:
    """Create one self-contained run directory and return its input path."""
    parser = _parser(base_input)
    control = _section(parser, "Control")
    lammps = _section(parser, "Lammps")
    event_search = _section(parser, "EventSearch")
    partn = _section(parser, "pARTn")
    rate = _section(parser, "RateConstant")
    base_dir = base_input.resolve().parent

    work_dir.mkdir(parents=True, exist_ok=False)
    copied: dict[str, Path] = {}

    initial_raw = _get_option(parser, control, "initial_config")
    if not initial_raw:
        raise ValueError("[Control] initial_config is required")
    initial = _resolve_source(initial_raw, base_dir)
    _set_option(
        parser, control, "initial_config", _copy_asset(initial, work_dir, copied)
    )

    pair_coeff = _get_option(parser, lammps, "pair_coeff")
    if not pair_coeff:
        raise ValueError("[Lammps] pair_coeff is required")
    _set_option(
        parser,
        lammps,
        "pair_coeff",
        _rewrite_pair_coeff(pair_coeff, base_dir, work_dir, copied),
    )

    for option in ("reference_table", "visited_environments", "restart_file"):
        _remove_option(parser, control, option)
    _set_option(parser, control, "basin", False)
    _set_option(parser, control, "recycle", False)
    _remove_section(parser, "EventRecycling")
    _set_option(parser, control, "trajectory_output", "./trajkmc.xyz")
    _set_option(parser, control, "reference_table_output", "./reference_table.pickle")
    _set_option(
        parser, control, "visited_environments_output", "./visited_environments.pickle"
    )
    if steps is not None:
        _set_option(parser, control, "n_steps", steps)
    if nsearch is not None:
        _set_option(parser, event_search, "nsearch", nsearch)

    _set_option(parser, partn, "zseed", seed)
    for option, value in htst_options.items():
        _set_option(parser, rate, option, value)
    _set_option(parser, rate, "style", style)

    output = work_dir / "input.in"
    with output.open("w") as handle:
        parser.write(handle)
    return output


def _input_signature(path: Path, exclude_style: bool = False) -> dict[str, Any]:
    parser = _parser(path)
    result: dict[str, Any] = {}
    for section in parser.sections():
        values = dict(parser.items(section))
        if exclude_style and section.lower() == "rateconstant":
            values = {k: v for k, v in values.items() if k.lower() != "style"}
        result[section.lower()] = values
    return result


def assert_paired_inputs(constant_input: Path, htst_input: Path) -> None:
    """Ensure generated inputs differ only in the prefactor style."""
    if _input_signature(constant_input, exclude_style=True) != _input_signature(
        htst_input, exclude_style=True
    ):
        raise ValueError("generated A/B inputs differ outside [RateConstant] style")
    constant = _parser(constant_input)
    htst = _parser(htst_input)
    constant_style = _get_option(
        constant, _section(constant, "RateConstant"), "style"
    )
    htst_style = _get_option(htst, _section(htst, "RateConstant"), "style")
    if (constant_style, htst_style) != ("constant", "htst"):
        raise ValueError(
            f"unexpected generated styles: constant={constant_style}, htst={htst_style}"
        )


def _minimum_nproc(base_input: Path) -> tuple[int, int, bool]:
    parser = _parser(base_input)
    control = _section(parser, "Control")
    n_sessions = int(_get_option(parser, control, "n_sessions") or "1")
    raw_use_rank_0 = (_get_option(parser, control, "engine_use_rank_0") or "False")
    use_rank_0 = raw_use_rank_0.strip().lower() in {"1", "true", "yes", "on"}
    minimum = n_sessions if use_rank_0 else n_sessions + 1
    return minimum, n_sessions, use_rank_0


def _launcher_prefix(
    launcher: str, nproc_flag: str, nproc: int, launcher_extra: str
) -> list[str]:
    if launcher.lower() in {"none", "direct"}:
        if nproc != 1:
            raise ValueError("direct launch requires --nproc 1")
        return []
    executable = shutil.which(launcher) if not Path(launcher).is_file() else launcher
    if not executable:
        raise FileNotFoundError(f"launcher not found: {launcher}")
    return [str(executable), nproc_flag, str(nproc), *shlex.split(launcher_extra)]


def _executable_path(raw: str) -> str:
    """Return an absolute executable path without resolving virtualenv symlinks."""
    candidate = Path(raw).expanduser()
    if candidate.parent != Path("."):
        candidate = candidate.absolute()
        if candidate.is_file():
            return str(candidate)
    discovered = shutil.which(raw)
    if discovered:
        return discovered
    raise FileNotFoundError(f"executable not found: {raw}")


def _run_checked(
    command: list[str],
    cwd: Path,
    stdout_path: Path,
    stderr_path: Path,
    timeout_seconds: float | None,
    env: dict[str, str],
) -> tuple[int, float]:
    start = time.perf_counter()
    with stdout_path.open("w") as stdout, stderr_path.open("w") as stderr:
        try:
            completed = subprocess.run(
                command,
                cwd=cwd,
                env=env,
                stdout=stdout,
                stderr=stderr,
                timeout=timeout_seconds,
                check=False,
            )
            returncode = completed.returncode
        except subprocess.TimeoutExpired:
            returncode = 124
            stderr.write(
                f"\nBenchmark timeout after {timeout_seconds:.1f} seconds.\n"
            )
    return returncode, time.perf_counter() - start


def _now_iso() -> str:
    return datetime.now().astimezone().isoformat(timespec="seconds")


def run_preflight(
    python: str,
    launcher_prefix: list[str],
    cwd: Path,
    env: dict[str, str],
    timeout_seconds: float | None,
) -> dict[str, Any]:
    """Validate Python modules, native libraries, LAMMPS packages, and MPI."""
    local_code = r"""
import json
import pandas
import mpi4py
import pypARTn
import ira_mod
import pykmc
from lammps import lammps
from pykmc.config import RateConstantConfig

lmp = lammps(cmdargs=["-log", "none", "-screen", "none"])
packages = {
    "PHONON": bool(lmp.has_package("PHONON")),
    "PLUGIN": bool(lmp.has_package("PLUGIN")),
}
lammps_library = lmp.lib._name
lammps_version = lmp.version()
lmp.close()
artn = pypARTn.artn(engine="lmp")
artn_library = artn.lib._name
artn.destroy()
ira = ira_mod.IRA()
RateConstantConfig(style="htst")
print(json.dumps({
    "packages": packages,
    "lammps_library": lammps_library,
    "lammps_version": lammps_version,
    "artn_library": artn_library,
    "pandas": pandas.__version__,
    "mpi4py": mpi4py.__version__,
    "pykmc": pykmc.__file__,
}))
if not all(packages.values()):
    raise SystemExit("LAMMPS must include PHONON and PLUGIN")
"""
    local = subprocess.run(
        [python, "-c", local_code],
        cwd=cwd,
        env=env,
        text=True,
        capture_output=True,
        timeout=timeout_seconds,
        check=False,
    )
    if local.returncode != 0:
        raise RuntimeError(
            "local dependency preflight failed:\n"
            f"stdout:\n{local.stdout}\nstderr:\n{local.stderr}"
        )

    mpi_code = r"""
from mpi4py import MPI
from lammps import lammps
lmp = lammps(cmdargs=["-log", "none", "-screen", "none"])
ok = lmp.has_package("PHONON") and lmp.has_package("PLUGIN")
lmp.close()
if not ok:
    raise SystemExit("LAMMPS package check failed")
if MPI.COMM_WORLD.rank == 0:
    print(f"MPI preflight passed with {MPI.COMM_WORLD.size} ranks")
"""
    mpi = subprocess.run(
        [*launcher_prefix, python, "-c", mpi_code],
        cwd=cwd,
        env=env,
        text=True,
        capture_output=True,
        timeout=timeout_seconds,
        check=False,
    )
    if mpi.returncode != 0:
        raise RuntimeError(
            "MPI/LAMMPS preflight failed:\n"
            f"stdout:\n{mpi.stdout}\nstderr:\n{mpi.stderr}"
        )
    return {
        "local": json.loads(local.stdout.strip().splitlines()[-1]),
        "mpi": mpi.stdout.strip(),
    }


def _parse_step_timings(path: Path) -> dict[str, Any]:
    wall_times: list[float] = []
    if path.is_file():
        for line in path.read_text(errors="replace").splitlines():
            fields = line.split()
            if len(fields) != 10:
                continue
            try:
                step = int(fields[0])
                wall = float(fields[-1])
            except ValueError:
                continue
            if step > 0 and math.isfinite(wall):
                wall_times.append(wall)
    return {
        "steps_completed": len(wall_times),
        "step_wall_sum_s": sum(wall_times) if wall_times else None,
        "step_wall_median_s": statistics.median(wall_times) if wall_times else None,
    }


def _finite_summary(values: list[float]) -> tuple[float | None, float | None, float | None]:
    finite = [value for value in values if math.isfinite(value)]
    if not finite:
        return None, None, None
    return min(finite), statistics.median(finite), max(finite)


def _reference_table_stats(run_dir: Path) -> dict[str, Any]:
    import pandas as pd

    table_path = run_dir / "reference_table.pickle"
    if not table_path.is_file():
        return {
            "n_events": 0,
            "k_min": None,
            "k_median": None,
            "k_max": None,
            "has_nu0_column": False,
            "n_finite_nu0": 0,
            "n_fallback_nu0": 0,
            "nu0_coverage": 0.0,
            "nu0_min_THz": None,
            "nu0_median_THz": None,
            "nu0_max_THz": None,
        }
    table = pd.read_pickle(table_path)
    rates = pd.to_numeric(table.get("k"), errors="coerce").dropna().tolist()
    finite_rates = [float(value) for value in rates if math.isfinite(float(value))]
    k_min, k_median, k_max = _finite_summary(finite_rates)
    has_nu0 = "nu0" in table.columns
    raw_nu0_hz = (
        pd.to_numeric(table["nu0"], errors="coerce").dropna().tolist()
        if has_nu0
        else []
    )
    nu0_hz = [
        float(value) for value in raw_nu0_hz if math.isfinite(float(value))
    ]
    nu0_thz = [float(value) * 1e-12 for value in nu0_hz]
    nu0_min, nu0_median, nu0_max = _finite_summary(nu0_thz)
    n_events = int(len(table))
    n_finite_nu0 = len(nu0_thz)
    return {
        "n_events": n_events,
        "k_min": k_min,
        "k_median": k_median,
        "k_max": k_max,
        "has_nu0_column": has_nu0,
        "n_finite_nu0": n_finite_nu0,
        "n_fallback_nu0": (
            max(n_events - n_finite_nu0, 0) if has_nu0 else 0
        ),
        "nu0_coverage": (n_finite_nu0 / n_events) if n_events else 0.0,
        "nu0_min_THz": nu0_min,
        "nu0_median_THz": nu0_median,
        "nu0_max_THz": nu0_max,
    }


def _median(values: list[float | None]) -> float | None:
    finite = [
        float(value)
        for value in values
        if value is not None and math.isfinite(float(value))
    ]
    return statistics.median(finite) if finite else None


def _write_reports(output: Path, runs: list[dict[str, Any]], metadata: dict[str, Any]) -> dict[str, Any]:
    with (output / "runs.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=REPORT_FIELDS)
        writer.writeheader()
        writer.writerows({field: row.get(field) for field in REPORT_FIELDS} for row in runs)

    constant = [row for row in runs if row["style"] == "constant"]
    htst = [row for row in runs if row["style"] == "htst"]
    constant_wall = _median([row["wall_s"] for row in constant])
    htst_wall = _median([row["wall_s"] for row in htst])
    comparison = {
        "constant_wall_median_s": constant_wall,
        "htst_wall_median_s": htst_wall,
        "htst_over_constant_wall_ratio": (
            htst_wall / constant_wall
            if constant_wall is not None and htst_wall is not None and constant_wall > 0
            else None
        ),
        "constant_step_wall_median_s": _median(
            [row["step_wall_median_s"] for row in constant]
        ),
        "htst_step_wall_median_s": _median(
            [row["step_wall_median_s"] for row in htst]
        ),
        "constant_events_median": _median([row["n_events"] for row in constant]),
        "htst_events_median": _median([row["n_events"] for row in htst]),
        "constant_wall_per_event_median_s": _median(
            [row["wall_per_event_s"] for row in constant]
        ),
        "htst_wall_per_event_median_s": _median(
            [row["wall_per_event_s"] for row in htst]
        ),
        "htst_nu0_coverage_median": _median(
            [row["nu0_coverage"] for row in htst]
        ),
        "total_launcher_wall_s": sum(float(row["wall_s"]) for row in runs),
    }

    failures: list[str] = []
    warnings: list[str] = []
    for row in runs:
        label = f"repeat {row['repeat']} {row['style']}"
        if row["returncode"] != 0:
            failures.append(f"{label}: return code {row['returncode']}")
        if row["n_events"] <= 0:
            failures.append(f"{label}: no reference events")
        if row["style"] == "constant" and row["has_nu0_column"]:
            failures.append(f"{label}: constant table unexpectedly contains nu0")
        if row["style"] == "htst":
            if not row["has_nu0_column"]:
                failures.append(f"{label}: HTST table has no nu0 column")
            elif row["n_finite_nu0"] <= 0:
                failures.append(f"{label}: HTST produced no finite nu0 values")
            elif row["nu0_coverage"] < 1.0:
                warnings.append(
                    f"{label}: HTST coverage is {row['nu0_coverage']:.1%}; "
                    f"{row['n_fallback_nu0']} events used k0 fallback"
                )

    summary = {
        "status": "passed" if not failures else "failed",
        "failures": failures,
        "warnings": warnings,
        "metadata": metadata,
        "comparison": comparison,
        "runs": runs,
    }
    (output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    return summary


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    digest.update(path.read_bytes())
    return digest.hexdigest()


def _git_revision(repo_root: Path) -> str | None:
    completed = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=repo_root,
        text=True,
        capture_output=True,
        check=False,
    )
    return completed.stdout.strip() if completed.returncode == 0 else None


def build_parser() -> argparse.ArgumentParser:
    """Build the command-line parser."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, type=Path, help="Base pyKMC input file.")
    parser.add_argument("--output", required=True, type=Path, help="New benchmark output directory.")
    parser.add_argument("--python", default=sys.executable, help="Python executable used for pyKMC.")
    parser.add_argument("--launcher", default="mpirun", help="MPI launcher, or 'direct'.")
    parser.add_argument("--nproc-flag", default="-n", help="Launcher task-count flag.")
    parser.add_argument("--launcher-extra", default="", help="Extra launcher arguments.")
    parser.add_argument("--nproc", type=int, default=None, help="MPI ranks; defaults to the input minimum.")
    parser.add_argument("--repeats", type=int, default=1)
    parser.add_argument("--steps", type=int, default=None, help="Override [Control] n_steps.")
    parser.add_argument(
        "--nsearch",
        type=int,
        default=None,
        help="Override [EventSearch] nsearch; useful for a quick MPI smoke.",
    )
    parser.add_argument("--seed", type=int, default=19073, help="Base paired-run seed.")
    parser.add_argument("--timeout-seconds", type=float, default=0.0)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--dry-run", action="store_true", help="Prepare and validate without running pyKMC.")
    parser.add_argument("--preflight-only", action="store_true")
    parser.add_argument("--skip-preflight", action="store_true")
    parser.add_argument("--free-radius", default=HTST_DEFAULTS["free_radius"])
    parser.add_argument("--fd-step", default=HTST_DEFAULTS["fd_step"])
    parser.add_argument("--nu0-min-THz", default=HTST_DEFAULTS["nu0_min_THz"])
    parser.add_argument("--nu0-max-THz", default=HTST_DEFAULTS["nu0_max_THz"])
    parser.add_argument("--premin", choices=["True", "False"], default=HTST_DEFAULTS["premin"])
    return parser


def main(argv: list[str] | None = None) -> int:
    """Prepare, execute, and summarize the paired benchmark."""
    args = build_parser().parse_args(argv)
    base_input = args.input.expanduser().resolve()
    output = args.output.expanduser().resolve()
    if not base_input.is_file():
        raise SystemExit(f"input not found: {base_input}")
    try:
        python = _executable_path(args.python)
    except FileNotFoundError as exc:
        raise SystemExit(str(exc)) from exc
    if args.repeats < 1:
        raise SystemExit("--repeats must be at least 1")
    if args.nsearch is not None and args.nsearch < 1:
        raise SystemExit("--nsearch must be at least 1")

    minimum, n_sessions, use_rank_0 = _minimum_nproc(base_input)
    nproc = args.nproc if args.nproc is not None else minimum
    if nproc < minimum:
        raise SystemExit(
            f"--nproc {nproc} is too small for n_sessions={n_sessions}, "
            f"engine_use_rank_0={use_rank_0}; minimum is {minimum}"
        )
    try:
        launcher_prefix = _launcher_prefix(
            args.launcher, args.nproc_flag, nproc, args.launcher_extra
        )
    except (ValueError, FileNotFoundError) as exc:
        raise SystemExit(str(exc)) from exc

    if output.exists():
        if not args.overwrite:
            raise SystemExit(f"output already exists: {output} (use --overwrite)")
        shutil.rmtree(output)
    output.mkdir(parents=True)

    env = os.environ.copy()
    env["PYTHONUNBUFFERED"] = "1"
    repo_root = Path(__file__).resolve().parents[1]
    existing_pythonpath = env.get("PYTHONPATH")
    env["PYTHONPATH"] = (
        f"{repo_root}{os.pathsep}{existing_pythonpath}"
        if existing_pythonpath
        else str(repo_root)
    )
    timeout = args.timeout_seconds if args.timeout_seconds > 0 else None
    htst_options = {
        **HTST_DEFAULTS,
        "free_radius": str(args.free_radius),
        "fd_step": str(args.fd_step),
        "nu0_min_THz": str(args.nu0_min_THz),
        "nu0_max_THz": str(args.nu0_max_THz),
        "premin": str(args.premin),
    }
    metadata: dict[str, Any] = {
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "host": platform.node(),
        "platform": platform.platform(),
        "base_input": str(base_input),
        "base_input_sha256": _sha256(base_input),
        "repo_root": str(repo_root),
        "repo_commit": _git_revision(repo_root),
        "python": python,
        "launcher_prefix": launcher_prefix,
        "nproc": nproc,
        "n_sessions": n_sessions,
        "engine_use_rank_0": use_rank_0,
        "repeats": args.repeats,
        "steps_override": args.steps,
        "nsearch_override": args.nsearch,
        "seed": args.seed,
        "execution_mode": "sequential",
        "basin_enabled": False,
        "recycling_enabled": False,
        "htst_options": htst_options,
    }

    if not args.skip_preflight:
        print("[htst-ab] running dependency and MPI preflight")
        try:
            metadata["preflight"] = run_preflight(
                python, launcher_prefix, output, env, timeout
            )
        except (RuntimeError, subprocess.TimeoutExpired) as exc:
            (output / "preflight_error.txt").write_text(f"{exc}\n")
            raise SystemExit(str(exc)) from exc
    else:
        metadata["preflight"] = "skipped"

    (output / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
    if args.preflight_only:
        print(f"[htst-ab] preflight passed; metadata -> {output / 'metadata.json'}")
        return 0

    prepared: list[tuple[int, int, str, Path]] = []
    for repeat in range(1, args.repeats + 1):
        seed = args.seed + repeat - 1
        repeat_dir = output / f"repeat_{repeat:03d}"
        repeat_dir.mkdir()
        inputs: dict[str, Path] = {}
        for style in ("constant", "htst"):
            work_dir = repeat_dir / style
            inputs[style] = prepare_input(
                base_input,
                work_dir,
                style,
                seed,
                args.steps,
                args.nsearch,
                htst_options,
            )
        assert_paired_inputs(inputs["constant"], inputs["htst"])
        order = ("constant", "htst") if repeat % 2 else ("htst", "constant")
        for style in order:
            prepared.append((repeat, seed, style, inputs[style].parent))

    print(
        f"[htst-ab] prepared {len(prepared)} runs in {output}; "
        "paired inputs differ only by prefactor style"
    )
    if args.dry_run:
        print("[htst-ab] dry run complete; pyKMC was not launched")
        return 0

    runs: list[dict[str, Any]] = []
    for repeat, seed, style, run_dir in prepared:
        seed_code = (
            "import random, numpy as np; "
            f"random.seed({seed}); np.random.seed({seed}); "
            "from pykmc.run import main; main()"
        )
        command = [
            *launcher_prefix,
            python,
            "-c",
            seed_code,
            "-in",
            "input.in",
        ]
        print(f"[htst-ab] repeat {repeat}/{args.repeats} style={style}")
        started_at = _now_iso()
        returncode, wall_s = _run_checked(
            command,
            run_dir,
            run_dir / "launcher.stdout",
            run_dir / "launcher.stderr",
            timeout,
            env,
        )
        finished_at = _now_iso()
        row: dict[str, Any] = {
            "repeat": repeat,
            "style": style,
            "seed": seed,
            "returncode": returncode,
            "started_at": started_at,
            "finished_at": finished_at,
            "wall_s": wall_s,
            "run_dir": str(run_dir),
        }
        row.update(_parse_step_timings(run_dir / "pykmc.out"))
        row.update(_reference_table_stats(run_dir))
        row["wall_per_event_s"] = (
            wall_s / row["n_events"] if row["n_events"] > 0 else None
        )
        runs.append(row)

    runs.sort(key=lambda row: (row["repeat"], row["style"]))
    summary = _write_reports(output, runs, metadata)
    ratio = summary["comparison"]["htst_over_constant_wall_ratio"]
    ratio_text = f"{ratio:.3f}" if ratio is not None else "unavailable"
    print(f"[htst-ab] status={summary['status']} HTST/constant wall ratio={ratio_text}")
    print(f"[htst-ab] reports: {output / 'runs.csv'} and {output / 'summary.json'}")
    for warning in summary["warnings"]:
        print(f"[htst-ab] WARN: {warning}", file=sys.stderr)
    if summary["failures"]:
        for failure in summary["failures"]:
            print(f"[htst-ab] FAIL: {failure}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
