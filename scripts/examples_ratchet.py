"""Run the examples for a few steps and compare them with a committed baseline.

Each selected example is copied to a scratch folder and its shipped input is
run for a few KMC steps under ``mpirun``. The run passes if

- it completes every step and logs no error,
- the barrier of the first selected event is the baseline one,
- the event searches and the refinements succeed about as often as in the
  baseline (``scripts/examples_baseline.json``).

Event searches are random, so two runs of the same code differ, and now and
then a run never finds the event the others select first. The script allows
for both. A success count fails only when it is below what the baseline rate
can give by chance. An example that fails is run a second time before the
failure counts. A baseline entry pools several runs and keeps the ones that
agree on the first barrier.

Usage, from anywhere in the repository::

    python scripts/examples_ratchet.py                # the default examples
    python scripts/examples_ratchet.py --all          # every baseline example
    python scripts/examples_ratchet.py NAME [NAME ...]
    python scripts/examples_ratchet.py --update       # rewrite their baseline
    python scripts/examples_ratchet.py --update --runs 1 NAME

The runs need LAMMPS, pARTn and as many MPI ranks as the input asks for
(``n_sessions`` + 1).
"""

from __future__ import annotations

import argparse
import configparser
import json
import math
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parent.parent
EXAMPLES = ROOT / "examples"
BASELINE = Path(__file__).resolve().with_name("examples_baseline.json")
SELF = "python scripts/examples_ratchet.py"

DEFAULT_STEPS = 3
DEFAULT_RUNS = 3  # runs pooled into a baseline entry
COUNT_KEYS = ("searches", "search_successes", "refinements", "refinement_successes")
BARRIER_TOLERANCE = 0.005  # eV
SIGMAS = 3.0  # allowed drop of a success count, in standard deviations
ERROR_MARKS = ("Traceback", "Reconstruction fails", "reconstuctions failed")

# Runs pyKMC from this checkout, whatever pykmc is installed in the environment.
BOOTSTRAP = """
import os, runpy, sys
sys.path.insert(0, {root!r})
import pykmc
assert os.path.realpath(pykmc.__file__).startswith({root!r} + os.sep), pykmc.__file__
runpy.run_module("pykmc", run_name="__main__")
"""


def input_file(example: Path) -> Path:
    """Return the tracked input of an example: the template, else ``input.in``."""
    template = example / "input.in.template"
    return template if template.is_file() else example / "input.in"


def run_example(name: str, steps: int, folder: Path, mpirun: str, timeout: int) -> dict:
    """Run one example for ``steps`` steps in ``folder`` and return what it did."""
    example = EXAMPLES / name
    source = input_file(example)
    folder.mkdir(parents=True)
    for path in example.iterdir():
        if path.is_file() and not path.name.startswith("input.in"):
            shutil.copy(path, folder)

    parser = configparser.ConfigParser()
    parser.optionxform = str
    parser.read(source)
    parser["Control"]["n_steps"] = str(steps)
    with (folder / "input.in").open("w") as handle:
        parser.write(handle)
    ranks = int(parser["Control"].get("n_sessions", "1")) + 1

    command = [mpirun, "-n", str(ranks), sys.executable, "-c"]
    command += [BOOTSTRAP.format(root=str(ROOT)), "-in", "input.in"]
    problems = []
    with (
        (folder / "run.stdout").open("w") as out,
        (folder / "run.stderr").open("w") as err,
    ):
        try:
            code = subprocess.run(
                command, cwd=folder, stdout=out, stderr=err, timeout=timeout
            ).returncode
        except subprocess.TimeoutExpired:
            code = None
            problems.append(f"still running after {timeout} s")
    if code:
        problems.append(f"mpirun exited with code {code}")

    result = read_outputs(folder)
    if result["steps_done"] < steps:
        problems.append(f"stopped after step {result['steps_done']} of {steps}")
    for file in ("pykmc.log", "run.stderr"):
        text = (folder / file).read_text() if (folder / file).is_file() else ""
        problems += [f"'{mark}' in {file}" for mark in ERROR_MARKS if mark in text]
    if result["first_barrier"] is None:
        problems.append("no barrier read from pykmc.out")
    result["steps"] = steps
    result["problems"] = problems
    return result


def read_outputs(folder: Path) -> dict:
    """Read the barriers, the wall time and the success counts of a run."""
    barriers: list[float] = []
    steps_done = 0
    wall = 0.0
    out = folder / "pykmc.out"
    for line in out.read_text().splitlines() if out.is_file() else []:
        fields = line.split()
        if not fields or not fields[0].isdigit() or int(fields[0]) == 0:
            continue
        steps_done = int(fields[0])
        wall = float(fields[-2])
        if len(fields) == 14:
            barriers.append(float(fields[4]))

    counts = dict.fromkeys(COUNT_KEYS, 0)
    info = folder / "pykmc.info"
    for step in yaml.safe_load_all(info.read_text()) if info.is_file() else []:
        if not step:
            continue
        searches = step["reference_event_searches_info"]
        refinements = step["refinements_info"]
        counts["searches"] += searches["total_event_searches"]
        counts["search_successes"] += searches["n_successes"]
        counts["refinements"] += refinements["n_attempts"]
        # pyKMC writes "n_sucesses" here; accept the corrected spelling too.
        counts["refinement_successes"] += refinements.get(
            "n_sucesses", refinements.get("n_successes", 0)
        )
    return {
        "steps_done": steps_done,
        "first_barrier": round(barriers[0], 4) if barriers else None,
        "barriers": [round(b, 4) for b in barriers],
        **counts,
        "wall_seconds": round(wall, 1),
    }


def lowest_allowed(successes: int, total: int, new_total: int) -> int:
    """Return the smallest success count still consistent with the baseline.

    The baseline rate is taken as ``(successes + 1) / (total + 2)``, so that a
    baseline of 38 out of 38 does not turn a single failure into an error. The
    count may then fall ``SIGMAS`` binomial standard deviations below its mean.
    """
    rate = (successes + 1) / (total + 2)
    spread = math.sqrt(new_total * rate * (1 - rate))
    return max(0, math.ceil(new_total * rate - SIGMAS * spread))


def compare(result: dict, baseline: dict) -> list[str]:
    """List what a run did worse than its baseline entry."""
    problems = []
    barrier, expected = result["first_barrier"], baseline["first_barrier"]
    if barrier is None or abs(barrier - expected) > BARRIER_TOLERANCE:
        problems.append(f"first barrier {barrier} eV, baseline {expected} eV")
    for kind, total_key, ok_key in (
        ("event searches", "searches", "search_successes"),
        ("refinements", "refinements", "refinement_successes"),
    ):
        total, ok = result[total_key], result[ok_key]
        if baseline[total_key] and not total:
            problems.append(f"no {kind} ran")
            continue
        floor = lowest_allowed(baseline[ok_key], baseline[total_key], total)
        if ok < floor:
            problems.append(
                f"{kind}: {ok} of {total} succeeded, baseline "
                f"{baseline[ok_key]} of {baseline[total_key]}, lowest allowed {floor}"
            )
    return problems


def summary(result: dict) -> str:
    """Describe a run in one line."""
    return (
        f"barrier {result['first_barrier']} eV, "
        f"searches {result['search_successes']}/{result['searches']}, "
        f"refinements {result['refinement_successes']}/{result['refinements']}, "
        f"{result['wall_seconds']} s"
    )


def pool(steps: int, results: list[dict]) -> dict:
    """Pool several runs of one example into a baseline entry.

    The entry keeps the runs that agree with most of the others on the first
    barrier and adds up their counts. It holds ``problems`` instead if no first
    barrier has a majority.
    """

    def agreeing(reference: dict) -> list[dict]:
        barrier = reference["first_barrier"]
        return [
            r for r in results if abs(r["first_barrier"] - barrier) <= BARRIER_TOLERANCE
        ]

    kept = max((agreeing(reference) for reference in results), key=len)
    barriers = [r["first_barrier"] for r in results]
    if 2 * len(kept) <= len(results):
        return {"problems": [f"the runs disagree on the first barrier: {barriers}"]}
    if len(kept) < len(results):
        print(f"    left out {len(results) - len(kept)} run(s): barriers {barriers}")
    entry = {
        "steps": steps,
        "runs": len(kept),
        "first_barrier": kept[0]["first_barrier"],
    }
    entry.update({key: sum(r[key] for r in kept) for key in COUNT_KEYS})
    entry["wall_seconds"] = round(sum(r["wall_seconds"] for r in kept) / len(kept), 1)
    return entry


def baseline_entry(
    name: str, steps: int, scratch: Path, args: argparse.Namespace
) -> dict:
    """Run an example ``args.runs`` times and pool the runs into a baseline entry."""
    results = []
    for number in range(1, args.runs + 1):
        folder = scratch / f"{name}_{number}"
        result = run_example(name, steps, folder, args.mpirun, args.timeout)
        print(f"{name}, run {number}: {summary(result)}")
        if result["problems"]:
            return {"problems": result["problems"]}
        results.append(result)
    return pool(steps, results)


def check(
    name: str, baseline: dict, scratch: Path, args: argparse.Namespace
) -> tuple[bool, bool]:
    """Run an example against its baseline; return (failed, differs from baseline).

    A run that completes but compares badly is repeated once, since the
    searches are random. A run that crashes is not repeated.
    """
    for again in ("", "_again"):
        folder = scratch / (name + again)
        result = run_example(name, baseline["steps"], folder, args.mpirun, args.timeout)
        crashed = bool(result["problems"])
        problems = result["problems"] or compare(result, baseline)
        print(f"{name}: {'FAILED' if problems else 'ok'} ({summary(result)})")
        for problem in problems:
            print(f"    {problem}")
        if not problems or crashed or again:
            break
        print("    searches are random: running it a second time")
    return bool(problems), bool(problems) and not crashed


def main() -> int:
    """Run the chosen examples, then compare them with the baseline or rewrite it."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("examples", nargs="*", help="example folders to run")
    parser.add_argument("--all", action="store_true", help="run every baseline example")
    parser.add_argument("--update", action="store_true", help="rewrite the baseline")
    parser.add_argument("--steps", type=int, help="steps to run, with --update")
    parser.add_argument(
        "--runs", type=int, default=DEFAULT_RUNS, help="runs to pool, with --update"
    )
    parser.add_argument("--mpirun", default="mpirun", help="MPI launcher")
    parser.add_argument("--timeout", type=int, default=1800, help="seconds per run")
    parser.add_argument("--keep", type=Path, help="folder to keep the runs in")
    args = parser.parse_args()

    data = json.loads(BASELINE.read_text())
    entries: dict[str, dict] = data["examples"]
    names = args.examples or (sorted(entries) if args.all else data["default"])
    unknown = [n for n in names if not (EXAMPLES / n).is_dir()]
    unknown += [n for n in names if n not in entries and not args.update]
    if unknown:
        print(f"Not an example, or not in the baseline: {sorted(set(unknown))}")
        return 2

    scratch = args.keep or Path(tempfile.mkdtemp(prefix="pykmc-examples-"))
    failed = differs = False
    for name in names:
        if not args.update:
            bad, different = check(name, entries[name], scratch, args)
            failed, differs = failed or bad, differs or different
            continue
        steps = args.steps or entries.get(name, {}).get("steps", DEFAULT_STEPS)
        entry = baseline_entry(name, steps, scratch, args)
        if "problems" in entry:
            print(f"{name}: FAILED, baseline entry not changed")
            for problem in entry["problems"]:
                print(f"    {problem}")
            failed = True
            continue
        old = entries.get(name, {}).get("first_barrier")
        if old and abs(old - entry["first_barrier"]) > BARRIER_TOLERANCE:
            print(f"    note: the baseline had a first barrier of {old} eV")
        entries[name] = entry
        data["examples"] = dict(sorted(entries.items()))
        BASELINE.write_text(json.dumps(data, indent=2) + "\n")
        print(f"{name}: baseline entry written from {entry['runs']} run(s)")

    if failed or args.keep:
        print(f"Run folders: {scratch}")
    else:
        shutil.rmtree(scratch)
    if differs:
        print(f"\nIf the change is intended, run `{SELF} --update` and commit it.")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
