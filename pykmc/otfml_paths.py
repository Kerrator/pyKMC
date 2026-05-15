"""Hardcoded paths and helpers for OTF MLIP artifacts."""

from __future__ import annotations

from pathlib import Path


OTFML_BASE_DIR = Path("./otfml_runtime")
OTFML_DUMP_DIR = OTFML_BASE_DIR / "dumps"

OTFML_EXTRAPOLATION_TOLERANCE = 1.2
OTFML_MAX_GAMMA = 25.0
OTFML_TOL_FLAG_INTERNAL = "saw_gamma_over_tol"
OTFML_MAX_FLAG_INTERNAL = "saw_gamma_over_max"
OTFML_TOL_FLAG_VARIABLE = "otf_flag_tol"
OTFML_MAX_FLAG_VARIABLE = "otf_flag_max"


def ensure_otfml_dirs() -> None:
    """Create hardcoded OTF runtime directories."""
    OTFML_DUMP_DIR.mkdir(parents=True, exist_ok=True)


def session_dump_path(session_id: int) -> Path:
    """Return the append-only raw dump path for one session."""
    ensure_otfml_dirs()
    return OTFML_DUMP_DIR / f"extrapolating_dump.{session_id}.lammps"
