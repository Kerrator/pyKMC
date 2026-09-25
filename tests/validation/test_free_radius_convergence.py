"""Free-radius convergence of the HTST prefactor on bcc Fe and fcc Ni vacancy hops.

Opt-in (``PYKMC_RUN_VALIDATION=1``, see ``conftest.py``). For each tracked hop
fixture and each ``free_radius`` in :data:`RADII` the Vineyard prefactor nu0 of
both directions is computed offline with ``LammpsHTSTExtension`` on a serial
engine (no MPI), three times:

* ``zone_radius = "auto"`` with ``interaction_range`` declared as twice the
  potential cutoff (the production default: ``free_radius + interaction_range``);
* ``zone_radius = None`` (the full periodic cell, the reference);
* ``zone_radius = free_radius + cutoff`` (one cutoff, not two), measured only:
  it is the candidate "approximate" crop, adopted only if its error is below
  1 percent.

``fd_step = 0.01``, ``premin = False``, ``free_region_center = "saddle"`` and
the default ``force_tol`` are fixed for every call. For Fe the acceptance
window is widened to ``nu0_max_hz = 1e16``: the forward nu0 of the Fe hop is
about 1.4e14 Hz at ``free_radius = 6``, above the default 1e14. The window is
acceptance policy, not a convergence parameter; the table header states it.

Assertions, per system:

1. every direction at every radius is ``ok`` for the auto and full zones (the
   rejection reason is in the failure message);
2. auto and full agree to 1e-9 relative in both directions at every radius,
   on the same free set;
3. with nu_ref the full-system nu0 at the largest radius present (12 A for the
   full set), ``err(r) = |nu(r) - nu_ref| / nu_ref`` is non-increasing over
   increasing ``r`` in each direction, allowing at most one inversion of at
   most 0.01 in ``err``;
4. a recommended radius exists and is at most 12 A: the smallest ``r`` with
   ``err <= 5 %`` in both directions at ``r`` and at every larger radius (it is
   printed and written into the table; ``err`` of the reference itself is 0,
   so the value, not its existence, is the result);
5. for Fe only, the shape of Lazauskas et al. (2014) for the Ackland-2004
   potential (2 A 1.42e13, 7 A 1.00e14, 20 A 1.07e14 s^-1; small radii
   under-estimate): nu0 at the smallest radius is below nu_ref in both
   directions. Nothing numeric is compared.

Environment (all optional):

``PYKMC_FREE_RADIUS_RADII``
    Comma list restricting the radii, e.g. ``2,3,4,5,6,7,8``. Without 12 A the
    reference is the largest radius present and the recommendation is marked
    provisional.
``PYKMC_FREE_RADIUS_SYSTEMS``
    Comma list of ``fe``, ``ni``; a system left out is skipped.
``PYKMC_FREE_RADIUS_TABLE_DIR``
    Directory for the Markdown tables (``fe_mendelev2003.md``,
    ``ni_v6_lkbeland2016.md``); default a pytest temporary directory. The
    tables are also printed in the terminal summary.

Fixtures and potentials are described in ``tests/data/README.md``. The full
set takes a few minutes per system on a serial engine, most of it at 10 and
12 A. The Fe hop is the first half of the first-neighbour vacancy migration
(its ``min2`` is the split-vacancy minimum of this potential).
"""

from __future__ import annotations

import datetime
import itertools
import os
import platform
import socket
import time
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

import numpy as np
import pytest

pytestmark = pytest.mark.validation

pytest.importorskip("lammps")

from pykmc.engine.htst_lammps import LammpsHTSTExtension  # noqa: E402
from pykmc.engine.lammps import LammpsEngine  # noqa: E402
from pykmc.htst import (  # noqa: E402
    DirectionalPrefactor,
    EventPrefactors,
    HTSTEventRequest,
    HTSTSettings,
)

_DATA = Path(__file__).resolve().parents[1] / "data"
_ALL_PERIODIC = (True, True, True)
_THZ = 1.0e12

RADII: tuple[float, ...] = (2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 10.0, 12.0)
"""Free radii (A) of the full table; ``PYKMC_FREE_RADIUS_RADII`` selects a subset."""
FD_STEP = 0.01
AUTO_TOL = 1.0e-9
"""Largest relative difference allowed between the auto zone and the full cell."""
INVERSION_MAX = 0.01
"""Largest single increase of ``err`` allowed between consecutive radii."""
CONVERGED_TOL = 0.05
"""``err`` at and above the recommended radius."""
ONE_CUTOFF_ADOPT_TOL = 0.01
"""Error below which the one-cutoff zone would be adoptable (reported only)."""

RADII_ENV = "PYKMC_FREE_RADIUS_RADII"
SYSTEMS_ENV = "PYKMC_FREE_RADIUS_SYSTEMS"
TABLE_DIR_ENV = "PYKMC_FREE_RADIUS_TABLE_DIR"

MODES = ("auto", "full", "one_cutoff")
DIRECTIONS = ("forward", "backward")


@dataclass(frozen=True)
class HopSystem:
    """One tracked vacancy-hop fixture with its potential.

    Attributes
    ----------
    key : str
        Selector used by ``PYKMC_FREE_RADIUS_SYSTEMS``.
    table_name : str
        Stem of the table file.
    description : str
        Crystal, cell and defect.
    event : str
        What the hop is.
    fixture : str
        ``.npz`` file name under ``tests/data``.
    potential : str
        Potential file name under ``tests/data``.
    potential_label : str
        Citation of the potential.
    pair_style : str
        LAMMPS pair style.
    element : str
        Element name in the potential file.
    cutoff : float
        Cutoff of the potential (A).
    nu0_max_hz : float or None
        Widened upper acceptance bound, or ``None`` for the default window.

    """

    key: str
    table_name: str
    description: str
    event: str
    fixture: str
    potential: str
    potential_label: str
    pair_style: str
    element: str
    cutoff: float
    nu0_max_hz: float | None

    @property
    def interaction_range(self) -> float:
        """Return the declared interaction range: twice the EAM cutoff (A)."""
        return 2.0 * self.cutoff


SYSTEMS: dict[str, HopSystem] = {
    "fe": HopSystem(
        key="fe",
        table_name="fe_mendelev2003",
        description="bcc Fe, 10x10x10 conventional cell, one vacancy",
        event=(
            "first half of the first-neighbour vacancy hop (min2 is the "
            "split-vacancy minimum of this potential; not symmetric)"
        ),
        fixture="htst_fe_bcc_vacancy_hop.npz",
        potential="Fe_mm.eam.fs",
        potential_label=(
            "Mendelev, Han, Srolovitz, Ackland, Sun, Asta, Phil. Mag. 83 (2003) "
            "3977, potential 2 (Fe_mm.eam.fs, eam/fs)"
        ),
        pair_style="eam/fs",
        element="Fe",
        cutoff=5.3,
        nu0_max_hz=1.0e16,
    ),
    "ni": HopSystem(
        key="ni",
        table_name="ni_v6_lkbeland2016",
        description="fcc Ni, 8x8x8 conventional cell, one vacancy",
        event="symmetric first-neighbour vacancy hop",
        fixture="htst_ni_fcc_vacancy_hop.npz",
        potential="Ni_v6_2.0_LKBeland2016.eam",
        potential_label=(
            "Ni v6, restiffened Ni EAM after Stoller et al. 2016 (doi "
            "10.1021/acs.jctc.5b01194), based on Mishin et al. 1999 (doi "
            "10.1103/PhysRevB.59.3393), as the file header states "
            "(Ni_v6_2.0_LKBeland2016.eam, eam/alloy)"
        ),
        pair_style="eam/alloy",
        element="Ni",
        cutoff=6.5,
        nu0_max_hz=None,
    ),
}


@dataclass
class _EAMConfig:
    """Engine inputs for one EAM potential (the fields ``LammpsEngine`` reads)."""

    pair_style: str
    pair_coeff: str
    min_style: str = "cg"
    minimize: str = "1e-6 1e-8 1000 10000"
    frz_min: str = "1e-4 1e-6 100 1000"
    verbosity: int = 0


@dataclass
class RadiusRow:
    """Everything measured at one free radius.

    Attributes
    ----------
    radius : float
        ``free_radius`` (A).
    results : dict
        Result per mode (``auto``, ``full``, ``one_cutoff``).
    wall : dict
        Wall time (s) of each ``compute_event_prefactors`` call, per mode.

    """

    radius: float
    results: dict[str, EventPrefactors]
    wall: dict[str, float]

    def direction(self, mode: str, direction: str) -> DirectionalPrefactor:
        """Return one direction of one mode."""
        result = self.results[mode]
        return result.forward if direction == "forward" else result.backward

    def nu0(self, mode: str, direction: str) -> float | None:
        """Return nu0 (Hz) of one direction of one mode, ``None`` unless ok."""
        estimate = self.direction(mode, direction)
        return estimate.nu0_hz if estimate.ok else None

    def zone_size(self, mode: str) -> int:
        """Return the number of atoms in the scratch system of one mode."""
        provenance = self.results[mode].provenance
        assert provenance is not None
        return len(provenance.zone_indices)


@dataclass
class Convergence:
    """The measured table of one system and what follows from it.

    Attributes
    ----------
    system : HopSystem
        The system measured.
    rows : list of RadiusRow
        One row per radius, in increasing order.
    reference_radius : float
        Largest radius present; its full-system nu0 is ``nu_ref``.
    errors : dict
        ``err(r)`` per direction (``None`` where a value is missing).
    recommended : float or None
        Smallest radius with ``err <= 5 %`` there and above, both directions.
    table_path : Path
        Where the Markdown table was written.

    """

    system: HopSystem
    rows: list[RadiusRow]
    reference_radius: float
    errors: dict[str, dict[float, float | None]]
    recommended: float | None
    table_path: Path

    def __repr__(self) -> str:
        """Return a short summary (pytest prints it in failure tracebacks)."""
        return (
            f"Convergence({self.system.key}, radii={self.radii}, "
            f"reference={self.reference_radius:g}, recommended={self.recommended})"
        )

    @property
    def radii(self) -> list[float]:
        """Return the radii present, increasing."""
        return [row.radius for row in self.rows]

    @property
    def provisional(self) -> bool:
        """Return True when the reference is not the full set's 12 A."""
        return self.reference_radius != max(RADII)


# ---------------------------------------------------------------------------
# Pure helpers
# ---------------------------------------------------------------------------


def _parse_list(name: str, default: list[str]) -> list[str]:
    """Return the comma list in environment variable ``name`` or ``default``."""
    raw = os.environ.get(name, "").strip()
    if not raw:
        return default
    return [item.strip() for item in raw.split(",") if item.strip()]


def _selected_radii() -> list[float]:
    """Return the radii selected by ``PYKMC_FREE_RADIUS_RADII``, sorted and unique."""
    items = _parse_list(RADII_ENV, [f"{r:g}" for r in RADII])
    radii: set[float] = set()
    for item in items:
        try:
            value = float(item)
        except ValueError:
            pytest.fail(f"{RADII_ENV}: {item!r} is not a number")
        if not np.isfinite(value) or value <= 0.0:
            pytest.fail(f"{RADII_ENV}: {item!r} is not a positive radius")
        radii.add(value)
    return sorted(radii)


def _selected_systems() -> list[str]:
    """Return the system keys selected by ``PYKMC_FREE_RADIUS_SYSTEMS``."""
    keys = [key.lower() for key in _parse_list(SYSTEMS_ENV, list(SYSTEMS))]
    unknown = sorted(set(keys) - set(SYSTEMS))
    if unknown:
        pytest.fail(
            f"{SYSTEMS_ENV}: unknown system(s) {unknown}; known {list(SYSTEMS)}"
        )
    return keys


def _relative(value: float, reference: float) -> float:
    """Return ``|value - reference| / |reference|``."""
    return abs(value - reference) / abs(reference)


def _errors(
    rows: list[RadiusRow], reference: RadiusRow
) -> dict[str, dict[float, float | None]]:
    """Return ``err(r)`` of the full-system nu0 per direction."""
    errors: dict[str, dict[float, float | None]] = {}
    for direction in DIRECTIONS:
        ref = reference.nu0("full", direction)
        errors[direction] = {}
        for row in rows:
            value = row.nu0("full", direction)
            errors[direction][row.radius] = (
                None if value is None or ref is None else _relative(value, ref)
            )
    return errors


def _inversions(
    radii: list[float], err: dict[float, float | None]
) -> list[tuple[float, float, float]]:
    """Return ``(r_prev, r_next, increase)`` wherever ``err`` grows with ``r``."""
    found = []
    for previous, following in zip(radii[:-1], radii[1:], strict=True):
        a, b = err[previous], err[following]
        if a is not None and b is not None and b > a:
            found.append((previous, following, b - a))
    return found


def _recommended_radius(
    radii: list[float], errors: dict[str, dict[float, float | None]]
) -> float | None:
    """Return the smallest radius with ``err <= 5 %`` there and at every larger one."""
    recommended = None
    for radius in reversed(radii):
        values = [errors[d][radius] for d in DIRECTIONS]
        if all(v is not None and v <= CONVERGED_TOL for v in values):
            recommended = radius
        else:
            break
    return recommended


# ---------------------------------------------------------------------------
# Table
# ---------------------------------------------------------------------------


def _fmt_nu(estimate: DirectionalPrefactor) -> str:
    """Return nu0 in THz, or the rejection code."""
    if estimate.ok:
        assert estimate.nu0_hz is not None
        return f"{estimate.nu0_hz / _THZ:.6g}"
    code = estimate.reason_code.value if estimate.reason_code else estimate.status
    return f"{estimate.status} ({code})"


def _fmt_rel(row: RadiusRow, mode: str, direction: str) -> str:
    """Return ``|nu(mode) - nu(full)| / nu(full)`` of one direction, or ``-``."""
    value, full = row.nu0(mode, direction), row.nu0("full", direction)
    if value is None or full is None:
        return "-"
    return f"{_relative(value, full):.1e}"


def _fmt_hz(value: float) -> str:
    """Return a frequency in Hz as ``1e16``."""
    return f"{value:.0e}".replace("e+", "e")


def _fmt_err(err: float | None) -> str:
    """Return an error as a percentage, or ``-``."""
    return "-" if err is None else f"{100.0 * err:.2f} %"


def _one_cutoff_errors(rows: list[RadiusRow]) -> list[tuple[float, float]]:
    """Return ``(radius, error)`` of every one-cutoff value that could be compared."""
    found = []
    for row in rows:
        for direction in DIRECTIONS:
            value, full = row.nu0("one_cutoff", direction), row.nu0("full", direction)
            if value is not None and full is not None:
                found.append((row.radius, _relative(value, full)))
    return found


def _render_table(
    conv: Convergence, hop: dict[str, np.ndarray], lammps_version: int
) -> str:
    """Return the Markdown table of one system (Lazauskas 2014, Table 1 layout)."""
    system = conv.system
    cell = np.diag(hop["cell"])
    natoms = len(hop["types"])
    box = min(float(x) for x in cell)
    window = (
        "default, nu0_min_hz 1e12, nu0_max_hz 1e14"
        if system.nu0_max_hz is None
        else f"nu0_min_hz 1e12, nu0_max_hz {_fmt_hz(system.nu0_max_hz)} (widened "
        "from the default 1e14: the forward nu0 of this hop is about 1.4e14 Hz "
        "at free_radius 6; the window is acceptance policy, not a convergence "
        "parameter)"
    )
    reference = (
        f"{conv.reference_radius:g} A"
        if not conv.provisional
        else f"{conv.reference_radius:g} A (subset run: 12 A absent, provisional)"
    )
    lines = [
        f"# Free-radius convergence of the HTST prefactor: {system.table_name}",
        "",
        "| Item | Value |",
        "| --- | --- |",
        f"| System | {system.description} |",
        f"| Event | {system.event}; mover index {int(hop['central_atom_idx'])}; "
        f"dE_forward {float(hop['dE_forward']):.6f} eV, dE_backward "
        f"{float(hop['dE_backward']):.6f} eV |",
        f"| Fixture | tests/data/{system.fixture} |",
        f"| Potential | {system.potential_label} |",
        f"| Cell | {' x '.join(f'{x:.5f}' for x in cell)} A, periodic |",
        f"| N atoms | {natoms} |",
        f"| Cutoff | {system.cutoff:g} A |",
        f"| interaction_range | {system.interaction_range:g} A (2 x cutoff); "
        f"auto zone = free_radius + {system.interaction_range:g} A |",
        f"| One-cutoff zone | free_radius + {system.cutoff:g} A (measured only) |",
        f"| Acceptance window | {window} |",
        "| Fixed settings | fd_step 0.01 A, premin False, free_region_center "
        "saddle, force_tol default (0.005 eV/A), eskm Hessian |",
        "| Convergence quantity | full-cell (zone_radius None) nu0; err vs ref = "
        "abs(nu(r) - nu(ref)) / nu(ref) |",
        f"| Reference radius | {reference} |",
        "| Periodic self-coupling | free atoms couple directly to periodic "
        f"images of free atoms above free_radius {(box - system.cutoff) / 2:.2f} A "
        "and through a shared neighbour's density above "
        f"{(box - 2 * system.cutoff) / 2:.2f} A |",
        f"| Host | {socket.gethostname()} |",
        f"| Python | {platform.python_version()} |",
        f"| LAMMPS | {lammps_version} |",
        "| Date | "
        f"{datetime.datetime.now(datetime.timezone.utc).isoformat(timespec='seconds')}"
        " |",
        "",
        "nu0 in THz; rel diff = abs(nu(auto) - nu(full)) / nu(full); wall = one "
        "compute_event_prefactors call (both directions, three Hessians) on a "
        "serial engine.",
        "",
        "| free_radius (A) | n_free | zone atoms (auto) | zone atoms (one-cutoff) "
        "| nu0 fwd auto (THz) | nu0 fwd full | rel diff | nu0 bwd auto "
        "| nu0 bwd full | rel diff | err vs ref fwd | err bwd "
        "| one-cutoff zone rel err fwd/bwd | wall auto (s) | wall full (s) "
        "| wall one-cutoff (s) |",
        "| " + " | ".join(["---"] * 16) + " |",
    ]
    for row in conv.rows:
        cells = [
            f"{row.radius:g}",
            str(row.results["full"].n_free),
            str(row.zone_size("auto")),
            str(row.zone_size("one_cutoff")),
            _fmt_nu(row.direction("auto", "forward")),
            _fmt_nu(row.direction("full", "forward")),
            _fmt_rel(row, "auto", "forward"),
            _fmt_nu(row.direction("auto", "backward")),
            _fmt_nu(row.direction("full", "backward")),
            _fmt_rel(row, "auto", "backward"),
            _fmt_err(conv.errors["forward"][row.radius]),
            _fmt_err(conv.errors["backward"][row.radius]),
            f"{_fmt_rel(row, 'one_cutoff', 'forward')} / "
            f"{_fmt_rel(row, 'one_cutoff', 'backward')}",
            f"{row.wall['auto']:.2f}",
            f"{row.wall['full']:.2f}",
            f"{row.wall['one_cutoff']:.2f}",
        ]
        lines.append("| " + " | ".join(cells) + " |")
    lines.append("")
    if conv.recommended is None:
        lines.append(
            "Recommended free_radius: none (err <= 5 % is not reached in both "
            "directions at any radius and every larger one)"
        )
    else:
        notes = [
            "smallest radius with err <= 5 % in both directions there and at every larger radius"
        ]
        if conv.provisional:
            notes.append("provisional: reference below 12 A")
        if conv.recommended == conv.reference_radius and len(conv.rows) > 1:
            notes.append(
                "equal to the reference radius, so convergence below it is not shown"
            )
        lines.append(
            f"Recommended free_radius: {conv.recommended:g} A ({'; '.join(notes)})"
        )
    lines.append(f"Reference radius = {conv.reference_radius:g} A")
    one_cutoff = _one_cutoff_errors(conv.rows)
    if one_cutoff:
        worst = max(err for _, err in one_cutoff)
        above = [
            err
            for r, err in one_cutoff
            if conv.recommended is not None and r >= conv.recommended
        ]
        verdict = "below" if worst < ONE_CUTOFF_ADOPT_TOL else "not below"
        tail = (
            f"; largest at and above the recommended radius {max(above):.1e}"
            if above
            else ""
        )
        lines.append(
            f"One-cutoff zone: largest relative error vs the full cell {worst:.1e} "
            f"over all radii and both directions ({verdict} 1 %){tail}"
        )
    not_ok = [
        f"{row.radius:g} {direction}"
        for row in conv.rows
        for direction in DIRECTIONS
        if not row.direction("one_cutoff", direction).ok
    ]
    if not_ok:
        lines.append(f"One-cutoff zone: no accepted value at {', '.join(not_ok)}")
    lines.append("")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Measurement
# ---------------------------------------------------------------------------


def _load_hop(system: HopSystem) -> dict[str, np.ndarray]:
    """Load one tracked fixture as plain arrays; fail when a file is missing."""
    for name in (system.fixture, system.potential):
        if not (_DATA / name).is_file():
            pytest.fail(f"tests/data/{name} is missing")
    with np.load(_DATA / system.fixture, allow_pickle=False) as data:
        return {key: np.array(data[key]) for key in data.files}


def _settings(system: HopSystem, radius: float, mode: str) -> HTSTSettings:
    """Return the settings of one call; only the zone differs between modes."""
    zone: float | Literal["auto"] | None
    if mode == "auto":
        zone = "auto"
    elif mode == "full":
        zone = None
    else:
        zone = radius + system.cutoff
    window: dict[str, Any] = (
        {} if system.nu0_max_hz is None else {"nu0_max_hz": system.nu0_max_hz}
    )
    return HTSTSettings(
        free_radius=radius,
        fd_step=FD_STEP,
        premin=False,
        free_region_center="saddle",
        zone_radius=zone,
        interaction_range=system.interaction_range,
        **window,
    )


def _measure(system: HopSystem, radii: list[float], table_dir: Path) -> Convergence:
    """Compute every radius and mode, write the table and return the result."""
    hop = _load_hop(system)
    config = _EAMConfig(
        pair_style=system.pair_style,
        pair_coeff=f"* * {_DATA / system.potential} {system.element}",
    )
    engine = LammpsEngine(config=config, comm=None, engine_id=0)
    engine.start()
    try:
        engine.initialize_parameters()
        engine.initialize_system(
            types=[str(t) for t in hop["types"]],
            positions=hop["min1_positions"],
            cell=hop["cell"],
            pbc=_ALL_PERIODIC,
        )
        engine.initialize_potential()
        lammps_version = int(engine.lmp.version())
        full_system = engine.full_system
        extension = LammpsHTSTExtension(engine)
        rows = []
        for radius in radii:
            results: dict[str, EventPrefactors] = {}
            wall: dict[str, float] = {}
            for mode in MODES:
                request = HTSTEventRequest(
                    event_key=("ref", int(hop["idx_ref_forward"])),
                    min1_positions=hop["min1_positions"].copy(),
                    saddle_positions=hop["saddle_positions"].copy(),
                    min2_positions=hop["min2_positions"].copy(),
                    types=tuple(str(t) for t in hop["types"]),
                    species=full_system.species,
                    masses=full_system.masses,
                    cell=hop["cell"].copy(),
                    pbc=_ALL_PERIODIC,
                    center_index=int(hop["central_atom_idx"]),
                    settings=_settings(system, radius, mode),
                )
                start = time.perf_counter()
                results[mode] = extension.compute_event_prefactors(request)
                wall[mode] = time.perf_counter() - start
            rows.append(RadiusRow(radius, results, wall))
    finally:
        engine.close()
    reference = rows[-1]
    errors = _errors(rows, reference)
    radii_present = [row.radius for row in rows]
    conv = Convergence(
        system=system,
        rows=rows,
        reference_radius=reference.radius,
        errors=errors,
        recommended=_recommended_radius(radii_present, errors),
        table_path=table_dir / f"{system.table_name}.md",
    )
    table_dir.mkdir(parents=True, exist_ok=True)
    conv.table_path.write_text(_render_table(conv, hop, lammps_version))
    return conv


@pytest.fixture(scope="module", params=list(SYSTEMS), ids=list(SYSTEMS))
def convergence(
    request: pytest.FixtureRequest,
    tmp_path_factory: pytest.TempPathFactory,
    validation_tables: Callable[[str, Path, str], None],
) -> Convergence:
    """Measure one system at every selected radius (once per module)."""
    from mpi4py import MPI

    if MPI.COMM_WORLD.Get_size() > 1:
        pytest.skip("the free-radius validation runs without mpirun")
    system = SYSTEMS[request.param]
    if system.key not in _selected_systems():
        pytest.skip(f"{system.key} not selected by {SYSTEMS_ENV}")
    radii = _selected_radii()
    configured = os.environ.get(TABLE_DIR_ENV, "").strip()
    table_dir = (
        Path(configured) if configured else tmp_path_factory.mktemp("free_radius")
    )
    conv = _measure(system, radii, table_dir)
    text = conv.table_path.read_text()
    validation_tables(system.table_name, conv.table_path, text)
    print(text)
    return conv


# ---------------------------------------------------------------------------
# Assertions
# ---------------------------------------------------------------------------


def test_every_direction_is_accepted(convergence: Convergence) -> None:
    """Both directions are ``ok`` at every radius with the auto and full zones."""
    failures = [
        f"free_radius {row.radius:g} {mode} {direction}: "
        f"{row.direction(mode, direction).status} "
        f"({row.direction(mode, direction).reason_code}): "
        f"{row.direction(mode, direction).reason}"
        for row in convergence.rows
        for mode in ("auto", "full")
        for direction in DIRECTIONS
        if not row.direction(mode, direction).ok
    ]
    assert not failures, "\n".join(failures)


def test_auto_zone_matches_full_system(convergence: Convergence) -> None:
    """The auto zone reproduces the full cell to 1e-9 relative at every radius."""
    counts = [row.results["full"].n_free for row in convergence.rows]
    assert all(a < b for a, b in itertools.pairwise(counts)), (
        f"n_free does not grow with free_radius: {counts}"
    )
    failures = []
    for row in convergence.rows:
        auto, full = row.results["auto"], row.results["full"]
        if auto.n_free != full.n_free:
            failures.append(
                f"free_radius {row.radius:g}: n_free auto {auto.n_free} != full "
                f"{full.n_free}"
            )
        for direction in DIRECTIONS:
            a, f = row.nu0("auto", direction), row.nu0("full", direction)
            if a is None or f is None:
                failures.append(
                    f"free_radius {row.radius:g} {direction}: no value to compare "
                    f"(auto {a}, full {f})"
                )
            elif _relative(a, f) > AUTO_TOL:
                failures.append(
                    f"free_radius {row.radius:g} {direction}: auto {a!r} Hz vs full "
                    f"{f!r} Hz, relative {_relative(a, f):.3e} > {AUTO_TOL:g}"
                )
    assert not failures, "\n".join(failures)


def test_error_decreases_with_radius(convergence: Convergence) -> None:
    """``err(r)`` vs the largest radius is non-increasing, one 1 % inversion allowed.

    Radii whose free set is the moving atom alone (``n_free == 1``) are left
    out: a one-atom partial Hessian is not on the converging branch (on both
    fixtures err rises from 2 to 3 A, where the first shell joins the free set).
    """
    failures = []
    radii = [row.radius for row in convergence.rows if row.results["full"].n_free > 1]
    for direction in DIRECTIONS:
        err = convergence.errors[direction]
        missing = [r for r in convergence.radii if err[r] is None]
        if missing:
            failures.append(f"{direction}: no full-cell value at radii {missing}")
            continue
        inversions = _inversions(radii, err)
        too_large = [inv for inv in inversions if inv[2] > INVERSION_MAX]
        if len(inversions) > 1 or too_large:
            failures.append(
                f"{direction}: err rises at "
                + ", ".join(f"{a:g}->{b:g} A by {d:.4f}" for a, b, d in inversions)
                + f" (at most one rise of at most {INVERSION_MAX} allowed); err = "
                + ", ".join(f"{r:g}: {err[r]:.4f}" for r in convergence.radii)
            )
    assert not failures, "\n".join(failures)


def test_recommended_radius_is_reported(convergence: Convergence) -> None:
    """A radius with ``err <= 5 %`` there and above exists and is at most 12 A."""
    recommended = convergence.recommended
    status = " (provisional)" if convergence.provisional else ""
    print(
        f"{convergence.system.table_name}: recommended free_radius "
        f"{recommended} A{status}, reference radius "
        f"{convergence.reference_radius:g} A, table {convergence.table_path}"
    )
    assert recommended is not None, (
        "err <= 5 % is not reached in both directions at any radius and every "
        f"larger one: {convergence.errors}"
    )
    assert recommended <= max(RADII)
    if not convergence.provisional:
        # err(reference) is 0 by construction, so convergence must be shown
        # below the reference radius, not only at it.
        assert recommended < convergence.reference_radius, (
            f"err <= 5 % only at the reference radius {convergence.reference_radius:g} "
            f"A: {convergence.errors}"
        )


def test_small_radius_underestimates_iron(convergence: Convergence) -> None:
    """Fe: nu0 at the smallest radius is below the reference (shape only)."""
    if convergence.system.key != "fe":
        pytest.skip("the Lazauskas 2014 shape oracle is for bcc Fe")
    smallest, reference = convergence.rows[0], convergence.rows[-1]
    if smallest is reference:
        pytest.skip("the shape check needs at least two radii")
    failures = []
    for direction in DIRECTIONS:
        small = smallest.nu0("full", direction)
        ref = reference.nu0("full", direction)
        if small is None or ref is None:
            failures.append(
                f"{direction}: no value (smallest {small}, reference {ref})"
            )
        elif not small < ref:
            failures.append(
                f"{direction}: nu0 at {smallest.radius:g} A ({small:.4e} Hz) is not "
                f"below the reference at {reference.radius:g} A ({ref:.4e} Hz)"
            )
    assert not failures, "\n".join(failures)
