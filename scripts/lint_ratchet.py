"""Fail when ruff reports more findings than the committed baseline.

The baseline (``scripts/lint_baseline.json``) records how many findings
``ruff check`` reported per file and per rule when it was last updated. The
check fails if any count grew, or if a file has a rule it did not have before.
New code therefore has to be clean, while old findings can be fixed over time.

Usage, from anywhere in the repository::

    python scripts/lint_ratchet.py            # compare with the baseline
    python scripts/lint_ratchet.py --update   # rewrite the baseline
"""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
BASELINE = Path(__file__).resolve().with_name("lint_baseline.json")
SELF = "python scripts/lint_ratchet.py"

# {file path relative to the repository root: {rule code: number of findings}}
Counts = dict[str, dict[str, int]]


def ruff_command() -> list[str]:
    """Return the command that runs ruff: the executable, else the module."""
    executable = shutil.which("ruff")
    return [executable] if executable else [sys.executable, "-m", "ruff"]


def ruff_version() -> str:
    """Return the version of the ruff that will be run, such as ``0.15.4``."""
    result = subprocess.run(
        [*ruff_command(), "--version"], capture_output=True, text=True, check=True
    )
    return result.stdout.split()[-1]


def current_counts() -> Counts:
    """Run ``ruff check`` on the repository and count findings by file and rule."""
    result = subprocess.run(
        [*ruff_command(), "check", "--output-format", "json", "--exit-zero", "."],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=True,
    )
    tally: Counter[tuple[str, str]] = Counter()
    for finding in json.loads(result.stdout):
        path = Path(finding["filename"]).resolve().relative_to(ROOT).as_posix()
        tally[(path, finding["code"] or "syntax-error")] += 1
    counts: Counts = {}
    for (path, code), number in sorted(tally.items()):
        counts.setdefault(path, {})[code] = number
    return counts


def regressions(baseline: Counts, current: Counts) -> list[str]:
    """List every file and rule whose count is above the baseline."""
    lines = []
    for path, rules in current.items():
        for code, number in rules.items():
            allowed = baseline.get(path, {}).get(code, 0)
            if number > allowed:
                lines.append(f"{path}: {code} {allowed} -> {number}")
    return lines


def total(counts: Counts) -> int:
    """Return the number of findings in ``counts``."""
    return sum(number for rules in counts.values() for number in rules.values())


def main() -> int:
    """Compare the current findings with the baseline, or rewrite it."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--update",
        action="store_true",
        help="rewrite the baseline from the current findings",
    )
    args = parser.parse_args()

    version = ruff_version()
    current = current_counts()

    if args.update:
        data = {"ruff_version": version, "counts": current}
        BASELINE.write_text(json.dumps(data, indent=2) + "\n")
        print(
            f"Baseline updated: {total(current)} findings in {len(current)} files "
            f"(ruff {version})."
        )
        return 0

    data = json.loads(BASELINE.read_text())
    baseline: Counts = data["counts"]
    if version != data["ruff_version"]:
        print(
            f"Note: the baseline was written with ruff {data['ruff_version']} and "
            f"this is ruff {version}. Counts can differ between versions."
        )

    worse = regressions(baseline, current)
    if worse:
        print("New ruff findings (file: rule baseline -> now):")
        for line in worse:
            print(f"  {line}")
        print(
            "\nFix them; `ruff check <file> --select <rule>` lists them.\n"
            f"If a file was only moved or renamed, run `{SELF} --update` and "
            "commit the baseline."
        )
        return 1

    print(f"Lint ratchet OK: {total(current)} findings, baseline {total(baseline)}.")
    if current != baseline:
        print(f"The baseline can be lowered: run `{SELF} --update` and commit it.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
