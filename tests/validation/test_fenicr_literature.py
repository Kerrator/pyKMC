"""FeNiCr literature validation of the HTST prefactors of a pyKMC campaign.

Opt-in (``PYKMC_RUN_VALIDATION=1``, see ``conftest.py``). The campaign itself
(hours of MPI runs) never runs inside pytest: these tests evaluate its output.
The alloy is fcc FeNiCr 55/28/17 at.% with the Bonny-2013 EAM at T = 600 K,
8x8x8 cells with one vacancy (2047 atoms) or a first-neighbour divacancy (2046
atoms), ``style = htst``. The targets are Lefèvre López et al., Phys. Rev.
Materials 8, 013609 (2024), Table I, and Rahman et al., Phys. Rev. Materials
10, 053606 (2026) for the divacancy; extraction, statistics and the report
live in :mod:`tests.validation.fenicr_literature`, whose module docstring
states the sets and how the executed events are recovered.

Both tests assert every acceptance criterion; the failure message names each
failed criterion with its numbers:

1. catalogue size: at least 500 catalogued directed reference rows over the
   monovacancy runs;
2. all-events mean prefactor: log10 of the linear mean nu0 of the accepted
   rows within 0.3 dex of 6.6e12 s^-1 (forward-only and pair-averaged means
   reported);
3. selected-events means: the KMC-executed events with E > 0.3 eV and an
   accepted nu0 have log10 of their linear mean nu0 within 0.3 dex of 1e13 s^-1
   and mean barrier 0.79 +/- 0.05 eV (catalogue-only E > 0.3 eV means
   reported);
4. element ordering: per mover species, the linear mean nu0 of the accepted
   catalogue rows ordered Fe > Ni > Cr (executed-set ordering reported);
5. prefactor span: at least 4 decades over the accepted catalogue rows;
6. status hygiene: no ``legacy`` or ``stale`` row, every ``rejected`` row has
   a reason (``pending`` and rows outside the code-default 1e12-1e14 Hz window
   reported);
7. divacancy transitions: 1NN->1NN (0.43 eV, 1.07e13 s^-1) and 1NN->4NN (0.72
   eV, 6.76e12 s^-1): mean barrier within 0.05 eV and log10 of the linear mean
   nu0 within 0.3 dex, asserted for a class with at least 5 accepted rows and
   reported as insufficient otherwise.

The papers' prefactor averages are linear means, so criteria 2-4 and 7 assert the
linear mean (like for like) and report the mean of log10(nu0), which sits
lower on a wide distribution, beside it; the width of the log10(nu0)
distribution (standard deviation and smoothed-histogram FWHM, paper 1.3 and 1.2
decades) is reported, not asserted. The pair-classifier consistency check on
the monovacancy rows is reported in the same report but not asserted (see the
helper module's docstring).

The Markdown report (Table I layout, span, status counts, divacancy classes,
per-run counts, criteria) is printed in the terminal summary through the
``validation_tables`` fixture whether the test passes or fails.

Environment
-----------
``PYKMC_FENICR_RUN_DIRS``
    Colon-separated run directories for
    :func:`test_literature_targets_on_run_dirs`; each holds
    ``reference_table.pickle``, ``input.in``, ``initial_config.xyz`` and
    ``pykmc.out`` (``pykmc.events`` and ``pykmc.log`` are read when present).
    Divacancy cells are recognised by their atom count. Unset or empty: the
    test is skipped.
``PYKMC_FENICR_SUMMARY``
    Summary file for :func:`test_literature_targets_on_committed_summary`;
    default ``tests/data/fenicr_literature_summary.json``. The test is skipped
    when the default file is absent and fails when a configured file is
    absent.
``PYKMC_FENICR_REPORT_DIR``
    Directory for the report (and, for the run-directory test, the summary it
    built); default a pytest temporary directory.
"""

from __future__ import annotations

import os
from collections.abc import Callable
from pathlib import Path

import pytest

from tests.validation.fenicr_literature import (
    Evaluation,
    evaluate,
    failure_message,
    load_summary,
    render_report,
    summary_from_runs,
    write_summary,
)

pytestmark = pytest.mark.validation

RUN_DIRS_ENV = "PYKMC_FENICR_RUN_DIRS"
SUMMARY_ENV = "PYKMC_FENICR_SUMMARY"
REPORT_DIR_ENV = "PYKMC_FENICR_REPORT_DIR"

COMMITTED_SUMMARY = (
    Path(__file__).resolve().parents[1] / "data" / "fenicr_literature_summary.json"
)


def _report_dir(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """Return ``PYKMC_FENICR_REPORT_DIR`` (created) or a fresh temporary directory."""
    configured = os.environ.get(REPORT_DIR_ENV, "").strip()
    if not configured:
        return tmp_path_factory.mktemp("fenicr_literature")
    directory = Path(configured)
    directory.mkdir(parents=True, exist_ok=True)
    return directory


def _report_and_assert(
    evaluation: Evaluation,
    path: Path,
    title: str,
    validation_tables: Callable[[str, Path, str], None],
) -> None:
    """Write and record the report, then assert every acceptance criterion."""
    text = render_report(evaluation)
    path.write_text(text)
    validation_tables(title, path, text)
    failed = [name for name, passed, _ in evaluation.criteria if not passed]
    assert not failed, failure_message(evaluation)


def test_literature_targets_on_run_dirs(
    tmp_path_factory: pytest.TempPathFactory,
    validation_tables: Callable[[str, Path, str], None],
) -> None:
    """The campaign run directories meet every acceptance criterion."""
    raw = os.environ.get(RUN_DIRS_ENV, "")
    run_dirs = [Path(item) for item in raw.split(":") if item.strip()]
    if not run_dirs:
        pytest.skip(
            f"{RUN_DIRS_ENV} is unset or empty: no FeNiCr campaign run directories "
            "to evaluate"
        )
    summary = summary_from_runs(run_dirs)
    report_dir = _report_dir(tmp_path_factory)
    write_summary(report_dir / "fenicr_literature_summary.json", summary)
    _report_and_assert(
        evaluate(summary),
        report_dir / "fenicr_literature_report_run_dirs.md",
        "fenicr_literature_run_dirs",
        validation_tables,
    )


def test_literature_targets_on_committed_summary(
    tmp_path_factory: pytest.TempPathFactory,
    validation_tables: Callable[[str, Path, str], None],
) -> None:
    """The committed campaign summary meets every acceptance criterion."""
    configured = os.environ.get(SUMMARY_ENV, "").strip()
    path = Path(configured) if configured else COMMITTED_SUMMARY
    if not path.is_file():
        if configured:
            pytest.fail(f"{SUMMARY_ENV} names {path}, which does not exist")
        pytest.skip(
            f"no committed summary at tests/data/{COMMITTED_SUMMARY.name}: nothing "
            "to evaluate (set PYKMC_FENICR_SUMMARY to evaluate another file)"
        )
    report_dir = _report_dir(tmp_path_factory)
    _report_and_assert(
        evaluate(load_summary(path)),
        report_dir / "fenicr_literature_report_committed_summary.md",
        "fenicr_literature_committed_summary",
        validation_tables,
    )
