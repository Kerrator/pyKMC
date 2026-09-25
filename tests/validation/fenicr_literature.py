"""FeNiCr literature validation: extraction, statistics and report of HTST prefactors.

Importable helper of ``test_fenicr_literature.py`` and of the scripts that build
the committed summary ``tests/data/fenicr_literature_summary.json``. Nothing
here runs pytest or LAMMPS. Reading a ``reference_table.pickle`` imports
``pykmc`` (the table ``attrs`` hold ``pykmc.physics`` dataclasses), so run from a
checkout whose ``import pykmc`` resolves to the tree that wrote the tables.

Targets
-------
Lefèvre López, Mousseau, Adjanor, Domain, Phys. Rev. Materials 8, 013609 (2024),
Table I: FeNiCr 55/28/17 at.%, Bonny-2013 EAM, one vacancy in a 2047-atom
8x8x8 fcc cell, harmonic TST prefactors. All generated events: linear mean
6.6e12 s^-1, range 4.2e7-2.3e15 s^-1, mean barrier 1.11 eV. Selected
(executed) events with E > 0.3 eV: 1.0e13 s^-1, mean barrier 0.79 eV, range
1e10-4.8e14 s^-1. Diffusing Fe / Ni / Cr: 1.94e13 / 0.82e13 / 0.53e13 s^-1 at
0.84 / 0.78 / 0.77 eV. Rahman, Adjanor, Domain, Mousseau, Phys. Rev.
Materials 10, 053606 (2026): divacancy transitions 1NN->1NN 0.43 +/- 0.11 eV,
1.07e13 s^-1 and 1NN->4NN 0.72 eV, 6.76e12 s^-1.

Sets
----
``rows``
    Every catalogued directed reference row (one per direction) of the
    monovacancy runs; forward and backward rows are both generated events.
``executed``
    One row per KMC step of the monovacancy runs: the event the step executed.
``divacancy``
    Every catalogued directed row of the divacancy runs, with the separation
    class of the vacancy pair before and after the row.

Acceptance criteria (evaluated by :func:`evaluate`)
---------------------------------------------------
1. catalogue size: at least 500 directed rows over the monovacancy runs;
2. all-events mean prefactor: log10 of the linear mean nu0 of the accepted
   (``ok``) rows within 0.3 dex of log10(6.6e12);
3. selected-events means: the executed events with E > 0.3 eV and an accepted
   nu0 have log10 of their linear mean nu0 within 0.3 dex of 13 and mean
   barrier 0.79 +/- 0.05 eV;
4. element ordering: linear mean nu0 of the accepted catalogue rows by mover
   species ordered Fe > Ni > Cr;
5. prefactor span: accepted catalogue rows span at least 4 decades;
6. status hygiene: no ``legacy`` or ``stale`` row and every ``rejected`` row
   carries a reason (every run, both cell kinds);
7. divacancy transitions: 1NN->1NN and 1NN->4NN accepted rows have mean
   barrier within 0.05 eV and log10 of their linear mean nu0 within 0.3 dex of
   the targets, asserted for a class with at least 5 rows;

Reported, not asserted: the pair-classifier consistency check on the
monovacancy runs (every on-lattice first-neighbour hop shows exactly one vacant
site before and after, the mover's final site is the initial vacancy and its
initial site the final one). Its failures are listed in the report; they flag
the divacancy classification as unreliable rather than fail the validation,
because a concerted event with a first-neighbour mover breaks the check
without contradicting the paper.

The paper's prefactor averages are linear means (prefactors enter the rate
linearly), so criteria 2-4 and 7 assert the linear mean, the like-for-like statistic,
because on a distribution several decades wide the mean of log10(nu0) sits
about half a decade below log10 of the linear mean. The mean of log10(nu0) is
reported beside it everywhere, as are the width of the log10(nu0) distribution
of the accepted catalogue rows and of the selected set (standard deviation, and
the FWHM of a Gaussian-smoothed histogram with a 0.1-dex bandwidth, beside the
paper's 1.3 and 1.2 decades). Rahman 2026 follows the same group's kART/hTST
convention, so its divacancy averages are compared the same way.

Executed events
---------------
No file written by pyKMC records the executed row's ``nu0``, ``nu0_status`` or
``nu0_source``. ``pykmc.out`` (``LogKMC.table_line_info_kmc``, called from
``KMC.run``) carries per step the executed row's reference index, barrier and
rate ``k`` (ps^-1); ``k = prefactor * exp(-E / (kb T))`` exactly
(``rate_from_prefactor``) with ``prefactor = nu0 * 1e-12`` for an accepted
estimate and ``k0`` otherwise, so the prefactor the step used is recovered as
``k * exp(E / (kb T))`` to about 1e-5 relative (the table prints seven
significant digits). ``pykmc.events`` (``LogKMC.events_applicable_info_line`` and
``EventsInfo.output_msg``) names the selected row's central atom and its
species; ``pykmc.log`` carries ``site nu0`` lines
(``ActiveEventTable`` site refinement) for every accepted site estimate. The
source is ``site`` when a site line of the same atom and reference matches the
recovered value, ``reference`` when the catalogue row's accepted nu0 matches,
``k0`` when the value is the fallback, otherwise ``unknown``.

Vacancy-pair classes
--------------------
A reference row stores a crop: the atoms within ``rcut`` (``[AtomicEnvironment]``)
of the mover in the row's initial geometry, rigidly translated. On a
cube-aligned fcc supercell with ``a0 = L / n`` (``n`` from the atom count, the
number of vacancies ``4 n^3 - N``) ideal sites are rebuilt around the mover's
initial position and every site closer than ``rcut - 0.5 A`` to it is labelled
vacant or occupied in both geometries; a crop atom farther than ``a0 / 4``
from its nearest site, or two atoms on one site, marks the geometry
off-lattice. The class of a two-vacancy geometry is the fcc shell of the pair
vector (``1NN`` ... ); one vacancy inside the sphere is ``unresolved``.
"""

from __future__ import annotations

import configparser
import hashlib
import itertools
import json
import math
import re
from collections import Counter
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from pykmc.config import PhysicalConstants, RateConstantConfig
from pykmc.event_table import (
    NU0_LEGACY,
    NU0_OK,
    NU0_REJECTED,
    NU0_STALE,
    NU0_STATUSES,
)
from pykmc.rate_constant.units import thz_to_hz

SCHEMA = "fenicr-literature-summary/1"
"""Value of the summary's ``schema`` key."""

# ---------------------------------------------------------------------------
# Targets and tolerances
# ---------------------------------------------------------------------------

ALL_EVENTS_NU_HZ = 6.6e12
"""Lefèvre López 2024 Table I, all events: linear mean prefactor (s^-1)."""
ALL_EVENTS_BARRIER_EV = 1.11
"""Lefèvre López 2024 Table I, all events: mean barrier (eV); reported only."""
SELECTED_NU_HZ = 1.0e13
"""Lefèvre López 2024 Table I, selected events with E > 0.3 eV (s^-1)."""
SELECTED_BARRIER_EV = 0.79
"""Lefèvre López 2024 Table I, selected events with E > 0.3 eV (eV)."""
SELECTED_MIN_BARRIER_EV = 0.3
"""Lower barrier bound of the selected set (eV, exclusive)."""
ELEMENT_TARGETS: dict[str, tuple[float, float]] = {
    "Fe": (0.84, 1.94e13),
    "Ni": (0.78, 0.82e13),
    "Cr": (0.77, 0.53e13),
}
"""Lefèvre López 2024 Table I, diffusing element: (mean barrier eV, prefactor s^-1)."""
ELEMENT_ORDER: tuple[str, ...] = ("Fe", "Ni", "Cr")
"""Required order of the per-element linear mean nu0, largest first."""
PAPER_FWHM_DECADES: dict[str, float] = {"all": 1.3, "selected": 1.2}
"""Lefèvre López 2024: FWHM of the log10(nu0) distribution, all and selected events."""
FWHM_BANDWIDTH_DEX = 0.1
"""Standard deviation of the Gaussian that smooths the log10(nu0) histogram (dex)."""
FWHM_BIN_DEX = 0.005
"""Bin width of the histogram smoothed for the FWHM (dex)."""
DIVACANCY_TARGETS: dict[tuple[str, str], tuple[float, float]] = {
    ("1NN", "1NN"): (0.43, 1.07e13),
    ("1NN", "4NN"): (0.72, 6.76e12),
}
"""Rahman 2026 divacancy transitions: (mean barrier eV, prefactor s^-1)."""
DEX_TOL = 0.3
"""Largest deviation of an asserted log10 prefactor statistic from its target (dex)."""
BARRIER_TOL_EV = 0.05
"""Largest deviation of a mean barrier from its target (eV)."""
MIN_ROWS = 500
"""Smallest number of catalogued directed rows (monovacancy runs)."""
MIN_SPAN_DECADES = 4.0
"""Smallest max - min of log10(nu0) over the accepted catalogue rows."""
DIVACANCY_MIN_ROWS = 5
"""Smallest number of accepted rows for a divacancy class to be asserted."""

TRUST_MARGIN = 0.5
"""Sites closer than ``rcut - TRUST_MARGIN`` (A) to the mover are classified."""
SITE_TOL_FRACTION = 0.25
"""An atom farther than ``SITE_TOL_FRACTION * a0`` from every site is off-lattice."""
SITE_NU0_RTOL = 2.0e-4
"""Match of a logged site nu0 (five significant digits) with the recovered one."""
VALUE_RTOL = 1.0e-4
"""Match of the recovered prefactor with a stored nu0 or with ``k0``."""

CRITERIA: tuple[str, ...] = (
    "catalogue size",
    "all-events mean prefactor",
    "selected-events means",
    "element ordering",
    "prefactor span",
    "status hygiene",
    "divacancy transitions",
)
"""Names of the acceptance criteria, in evaluation order."""

_ROW_LISTS: tuple[str, ...] = ("rows", "executed", "divacancy", "divacancy_executed")
_STEP_RE = re.compile(r"Step : (\d+)")
_SITE_RE = re.compile(
    r"\[htst\] active event \(atom (\d+), reference (\d+)\): site nu0 = "
    r"([0-9.eE+-]+) Hz"
)
_EVENTS_STEP_RE = re.compile(r"^#Step: (\d+)")
_EVENTS_SELECTED_RE = re.compile(r"Applicable Events \(Selected=(\d+)\)")
_LATTICE_KEYS: tuple[str, ...] = (
    "on_lattice",
    "hop_shell",
    "vacancies_initial",
    "vacancies_final",
    "mover_final_site_vacant",
    "mover_initial_site_vacated",
    "pair_initial",
    "pair_final",
)
_REQUIRED_COLUMNS = frozenset(
    {
        "idx_ref",
        "idx_backward",
        "types",
        "move_atom_idx",
        "energy_barrier",
        "initial_positions",
        "final_positions",
        "nu0",
        "nu0_status",
        "nu0_reason",
    }
)


def default_window_hz() -> tuple[float, float]:
    """Return the code-default nu0 acceptance window of ``RateConstantConfig`` (Hz).

    Returns
    -------
    tuple of float
        ``(nu0_min, nu0_max)`` in Hz.

    """
    fields = RateConstantConfig.model_fields
    return (
        thz_to_hz(float(fields["nu0_min_THz"].default)),
        thz_to_hz(float(fields["nu0_max_THz"].default)),
    )


# ---------------------------------------------------------------------------
# fcc lattice analysis of a reference crop
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class FccLattice:
    """Ideal cube-aligned fcc lattice of a periodic supercell.

    Attributes
    ----------
    a0 : float
        Lattice parameter (A).
    cell_length : float
        Edge of the cubic periodic cell (A).
    rcut : float
        Crop radius of the reference rows (A).
    sites : numpy.ndarray
        Integer site vectors in units of ``a0 / 2`` (even coordinate sum) closer
        than :attr:`trust_radius` to the origin, origin included.
    shells : dict of int to int
        fcc shell index keyed by squared length in units of ``(a0 / 2)^2``.

    """

    a0: float
    cell_length: float
    rcut: float
    sites: np.ndarray
    shells: dict[int, int]

    @property
    def half(self) -> float:
        """Return half the lattice parameter (A)."""
        return self.a0 / 2.0

    @property
    def trust_radius(self) -> float:
        """Return the radius (A) inside which sites are classified."""
        return self.rcut - TRUST_MARGIN

    @property
    def site_tol(self) -> float:
        """Return the largest distance (A) of an on-lattice atom from its site."""
        return SITE_TOL_FRACTION * self.a0

    @classmethod
    def build(cls, a0: float, cell_length: float, rcut: float) -> FccLattice:
        """Enumerate the trusted sites and the shell table.

        Parameters
        ----------
        a0 : float
            Lattice parameter (A).
        cell_length : float
            Edge of the cubic cell (A).
        rcut : float
            Crop radius (A).

        Returns
        -------
        FccLattice
            The lattice.

        """
        half = a0 / 2.0
        reach = int(math.ceil((rcut - TRUST_MARGIN) / half)) + 1
        span = range(-2 * reach, 2 * reach + 1)
        norms = sorted(
            {
                i * i + j * j + k * k
                for i, j, k in itertools.product(span, repeat=3)
                if (i + j + k) % 2 == 0 and (i, j, k) != (0, 0, 0)
            }
        )
        shells = {n2: rank + 1 for rank, n2 in enumerate(norms)}
        limit = ((rcut - TRUST_MARGIN) / half) ** 2 + 1e-9
        sites = np.array(
            [
                (i, j, k)
                for i, j, k in itertools.product(range(-reach, reach + 1), repeat=3)
                if (i + j + k) % 2 == 0 and i * i + j * j + k * k <= limit
            ],
            dtype=np.int64,
        )
        return cls(
            a0=a0, cell_length=cell_length, rcut=rcut, sites=sites, shells=shells
        )

    def shell(self, vector: np.ndarray) -> int:
        """Return the fcc shell of an integer site vector (0 for the null vector).

        Parameters
        ----------
        vector : numpy.ndarray
            Integer vector in units of ``a0 / 2``.

        Returns
        -------
        int
            Shell index (1 for first neighbours).

        """
        n2 = int(np.dot(vector, vector))
        return 0 if n2 == 0 else self.shells[n2]


def nearest_fcc_sites(u: np.ndarray) -> np.ndarray:
    """Return the nearest fcc lattice vectors (Conway-Sloane decoding of the fcc lattice).

    Parameters
    ----------
    u : numpy.ndarray
        ``(M, 3)`` coordinates in units of ``a0 / 2``.

    Returns
    -------
    numpy.ndarray
        ``(M, 3)`` integer vectors with an even coordinate sum.

    """
    n = np.rint(u)
    odd = (n.sum(axis=1) % 2) != 0
    if np.any(odd):
        err = u[odd] - n[odd]
        column = np.argmax(np.abs(err), axis=1)
        rows = np.arange(err.shape[0])
        fixed = n[odd]
        fixed[rows, column] += np.where(err[rows, column] >= 0.0, 1.0, -1.0)
        n[odd] = fixed
    return n.astype(np.int64)


def _vacant_sites(
    displacement: np.ndarray, lattice: FccLattice
) -> set[tuple[int, ...]] | None:
    """Return the vacant trusted sites of one geometry, ``None`` when off-lattice."""
    sites = nearest_fcc_sites(displacement / lattice.half)
    residual = np.linalg.norm(displacement - sites * lattice.half, axis=1)
    radius = np.linalg.norm(sites, axis=1) * lattice.half
    inside = radius <= lattice.trust_radius + 1e-9
    if np.any(inside & (residual > lattice.site_tol)):
        return None
    counts = Counter(tuple(int(x) for x in site) for site in sites[inside])
    if any(count > 1 for count in counts.values()):
        return None
    return {tuple(int(x) for x in site) for site in lattice.sites} - set(counts)


def _pair_class(vacant: set[tuple[int, ...]] | None, lattice: FccLattice) -> str:
    """Return the separation class of the vacancy pair of one geometry."""
    if vacant is None:
        return "off-lattice"
    if len(vacant) == 2:
        first, second = (np.array(site) for site in sorted(vacant))
        return f"{lattice.shell(first - second)}NN"
    if len(vacant) == 1:
        return "unresolved"
    return "other"


def lattice_analysis(
    initial: np.ndarray,
    final: np.ndarray,
    mover: int,
    lattice: FccLattice,
    n_vacancies: int,
) -> dict[str, Any]:
    """Classify the vacant sites of one reference row before and after the event.

    Parameters
    ----------
    initial, final : numpy.ndarray
        ``(M, 3)`` crop positions (A) of the row's initial and final geometry.
    mover : int
        Crop index of the moving atom.
    lattice : FccLattice
        Lattice of the cell.
    n_vacancies : int
        Vacancies in the cell; the pair classes are set only for two.

    Returns
    -------
    dict
        ``on_lattice``, ``hop_shell``, ``vacancies_initial``,
        ``vacancies_final``, ``mover_final_site_vacant``,
        ``mover_initial_site_vacated``, ``pair_initial``, ``pair_final``.

    """
    length = lattice.cell_length

    def image(delta: np.ndarray) -> np.ndarray:
        return delta - length * np.rint(delta / length)

    first = image(initial - initial[mover])
    sites = nearest_fcc_sites(first / lattice.half)
    residual = first - sites * lattice.half
    close = np.linalg.norm(residual, axis=1) <= lattice.site_tol
    shift = np.median(residual[close], axis=0) if np.any(close) else np.zeros(3)
    anchor = initial[mover] + shift
    before = image(initial - anchor)
    after = image(final - anchor)
    vacant_before = _vacant_sites(before, lattice)
    vacant_after = _vacant_sites(after, lattice)
    on_lattice = vacant_before is not None and vacant_after is not None
    result: dict[str, Any] = {
        "on_lattice": on_lattice,
        "hop_shell": None,
        "vacancies_initial": None if vacant_before is None else len(vacant_before),
        "vacancies_final": None if vacant_after is None else len(vacant_after),
        "mover_final_site_vacant": None,
        "mover_initial_site_vacated": None,
        "pair_initial": None,
        "pair_final": None,
    }
    if on_lattice:
        assert vacant_before is not None and vacant_after is not None
        start = nearest_fcc_sites(before[mover][None, :] / lattice.half)[0]
        end = nearest_fcc_sites(after[mover][None, :] / lattice.half)[0]
        result["hop_shell"] = lattice.shell(end - start)
        result["mover_final_site_vacant"] = tuple(int(x) for x in end) in vacant_before
        result["mover_initial_site_vacated"] = (
            tuple(int(x) for x in start) in vacant_after
        )
    if n_vacancies == 2:
        result["pair_initial"] = _pair_class(vacant_before, lattice)
        result["pair_final"] = _pair_class(vacant_after, lattice)
    return result


# ---------------------------------------------------------------------------
# Run-directory readers
# ---------------------------------------------------------------------------


def _sha256(path: Path) -> str:
    """Return the sha256 of a file."""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def read_xyz_cell(path: Path) -> tuple[np.ndarray, list[str]]:
    """Read the cell and the species of an extended-XYZ file (first frame).

    Parameters
    ----------
    path : Path
        The ``.xyz`` file.

    Returns
    -------
    tuple
        ``(cell, species)``: the ``(3, 3)`` lattice (A) and one species per atom.

    Raises
    ------
    ValueError
        If the comment line has no ``Lattice`` or the frame is truncated.

    """
    lines = path.read_text().splitlines()
    n_atoms = int(lines[0].split()[0])
    lattice = re.search(r'Lattice="([^"]+)"', lines[1])
    if lattice is None:
        raise ValueError(f"{path}: no Lattice= in the comment line")
    cell = np.array([float(x) for x in lattice.group(1).split()]).reshape(3, 3)
    column = 0
    properties = re.search(r"Properties=(\S+)", lines[1])
    if properties is not None:
        fields = properties.group(1).split(":")
        for name, count in zip(fields[0::3], fields[2::3], strict=False):
            if name == "species":
                break
            column += int(count)
    species = [line.split()[column] for line in lines[2 : 2 + n_atoms]]
    if len(species) != n_atoms:
        raise ValueError(f"{path}: {len(species)} atom lines, header says {n_atoms}")
    return cell, species


def fcc_supercell(cell: np.ndarray, n_atoms: int) -> tuple[int, float, int] | None:
    """Return ``(n, a0, vacancies)`` of a cube-aligned n x n x n fcc supercell.

    Parameters
    ----------
    cell : numpy.ndarray
        ``(3, 3)`` lattice (A).
    n_atoms : int
        Number of atoms in the cell.

    Returns
    -------
    tuple or None
        ``None`` unless the cell is cubic and axis-aligned and ``4 n^3 - N`` is
        between 0 and 4 for ``n = round((N / 4)^(1/3))``.

    """
    lengths = np.linalg.norm(cell, axis=1)
    if not np.allclose(cell, np.diag(np.diag(cell)), atol=1e-6):
        return None
    if not np.allclose(lengths, lengths[0], rtol=1e-6):
        return None
    n = round((n_atoms / 4.0) ** (1.0 / 3.0))
    vacancies = 4 * n**3 - n_atoms
    if n < 1 or not 0 <= vacancies <= 4:
        return None
    return n, float(lengths[0]) / n, vacancies


def _read_ini(path: Path) -> dict[str, dict[str, str]]:
    """Return an INI file as ``{section (lower case): {key: value}}``."""
    parser = configparser.ConfigParser(inline_comment_prefixes=("#", ";"))
    parser.optionxform = str  # type: ignore[assignment,method-assign]
    parser.read(path)
    return {
        section.lower(): {key: value.strip() for key, value in parser.items(section)}
        for section in parser.sections()
    }


def _step_table(text: str) -> list[dict[str, Any]]:
    """Parse the ``pykmc.out`` step table (steps >= 1)."""
    rows = []
    for line in text.splitlines():
        parts = line.split()
        if len(parts) < 8 or not parts[0].isdigit():
            continue
        try:
            rows.append(
                {
                    "step": int(parts[0]),
                    "ref_index": int(parts[3]),
                    "barrier_ev": float(parts[4]),
                    "k_per_ps": float(parts[5]),
                }
            )
        except ValueError:
            continue
    return rows


def _selected_events(text: str) -> dict[int, dict[str, Any]]:
    """Parse the selected row of every step of ``pykmc.events``.

    Raises
    ------
    ValueError
        On a basin step (its executed event is not the listed selection) or a
        repeated step.

    """
    selected: dict[int, dict[str, Any]] = {}
    step: int | None = None
    index: int | None = None
    for line in text.splitlines():
        match = _EVENTS_STEP_RE.match(line.strip())
        if match:
            step, index = int(match.group(1)), None
            if step in selected:
                raise ValueError(f"pykmc.events repeats step {step} (restarted run?)")
            continue
        if "Basin Exit Events" in line:
            raise ValueError(f"pykmc.events step {step}: basin steps are not supported")
        match = _EVENTS_SELECTED_RE.search(line)
        if match:
            index = int(match.group(1))
            continue
        parts = line.split()
        if (
            step is not None
            and index is not None
            and len(parts) == 11
            and parts[0].isdigit()
            and int(parts[0]) == index
        ):
            selected[step] = {
                "species": parts[1],
                "atom_index": int(parts[2]),
                "ref_index": int(parts[3]),
                "barrier_ev": float(parts[4]),
                "k_per_ps": float(parts[7]),
            }
    return selected


def _site_lines(text: str) -> dict[tuple[int, int], list[tuple[int, float]]]:
    """Return logged site nu0 values as ``{(atom, reference): [(step, Hz), ...]}``."""
    found: dict[tuple[int, int], list[tuple[int, float]]] = {}
    step = 0
    for line in text.splitlines():
        match = _SITE_RE.search(line)
        if match:
            key = (int(match.group(1)), int(match.group(2)))
            found.setdefault(key, []).append((step, float(match.group(3))))
            continue
        match = _STEP_RE.search(line)
        if match and "[htst]" not in line:
            step = int(match.group(1))
    return found


def _resolve_run_dir(path: str | Path) -> Path:
    """Return the directory holding ``reference_table.pickle`` (``run/`` accepted)."""
    directory = Path(path)
    if not directory.is_dir():
        raise FileNotFoundError(f"run directory {directory} does not exist")
    if (directory / "reference_table.pickle").is_file():
        return directory
    if (directory / "run" / "reference_table.pickle").is_file():
        return directory / "run"
    raise FileNotFoundError(f"{directory}: no reference_table.pickle")


def _run_label(directory: Path) -> str:
    """Return a path-free label of a run directory."""
    if directory.name in ("run", "fixture"):
        return f"{directory.parent.name}/{directory.name}"
    return directory.name


def _float_or_none(value: Any) -> float | None:
    """Return a finite float, or ``None`` for a missing or non-finite value."""
    if value is None:
        return None
    number = float(value)
    return number if math.isfinite(number) else None


def _directions(table: pd.DataFrame) -> dict[int, tuple[str, str]]:
    """Return ``{idx_ref: (direction, basis)}`` of every row.

    The direction of a row is that of the calculation that produced its
    estimate (``attrs["estimate_references"]`` -> ``attrs["calculations"]``):
    ``forward`` is the search direction, min1 -> saddle -> min2 as pARTn found
    it. A row without one takes the opposite of a partner that has one
    (``partner``); otherwise the smaller ``idx_ref`` of the pair is forward, as
    ``ReferenceEventTable.add`` assigns them (``pair_order``); a self-linked
    row is ``forward`` (``self``).
    """
    calculations = table.attrs.get("calculations") or {}
    references = table.attrs.get("estimate_references") or {}
    partner = {
        int(idx): int(back)
        for idx, back in zip(table["idx_ref"], table["idx_backward"], strict=True)
    }
    found: dict[int, tuple[str, str]] = {}
    for idx in partner:
        link = references.get(idx)
        calculation = (
            calculations.get(link.calculation_id) if link is not None else None
        )
        direction = getattr(calculation, "direction", None)
        if direction in ("forward", "backward"):
            found[idx] = (str(direction), "calculation")
    opposite = {"forward": "backward", "backward": "forward"}
    for idx, other in partner.items():
        if idx in found:
            continue
        if other == idx:
            found[idx] = ("forward", "self")
        elif other in found and found[other][1] == "calculation":
            found[idx] = (opposite[found[other][0]], "partner")
        elif other in partner:
            found[idx] = ("forward" if idx < other else "backward", "pair_order")
        else:
            found[idx] = ("unknown", "none")
    return found


@dataclass
class RunRows:
    """Everything extracted from one run directory.

    Attributes
    ----------
    run_id : str
        Path-free label of the run.
    kind : str
        ``monovacancy``, ``divacancy`` or ``other`` (not a 1- or 2-vacancy
        cube-aligned fcc supercell).
    meta : dict
        Per-run facts and counts (JSON-serialisable).
    rows : list of dict
        One per catalogued directed reference row.
    executed : list of dict
        One per KMC step: the executed event.

    """

    run_id: str
    kind: str
    meta: dict[str, Any]
    rows: list[dict[str, Any]]
    executed: list[dict[str, Any]]


def extract_run(run_dir: str | Path, run_id: str | None = None) -> RunRows:
    """Extract the catalogue and the executed events of one pyKMC htst run.

    Parameters
    ----------
    run_dir : str or Path
        Directory with ``reference_table.pickle``, ``input.in``,
        ``initial_config.xyz`` and ``pykmc.out`` (``pykmc.events`` and
        ``pykmc.log`` are read when present); a directory whose ``run/``
        subdirectory holds them is accepted.
    run_id : str, optional
        Label of the run; default the directory name (``<parent>/run`` for a
        directory named ``run``).

    Returns
    -------
    RunRows
        The rows, the executed events and the per-run facts.

    Raises
    ------
    FileNotFoundError
        If a required file is missing.
    ValueError
        If the table is not an htst table, a status is unknown, an accepted
        row has no finite nu0, or the step files disagree.

    """
    directory = _resolve_run_dir(run_dir)
    label = run_id or _run_label(directory)
    for name in ("input.in", "initial_config.xyz", "pykmc.out"):
        if not (directory / name).is_file():
            raise FileNotFoundError(f"{directory}: no {name}")
    table = pd.read_pickle(directory / "reference_table.pickle")
    if not isinstance(table, pd.DataFrame):
        raise ValueError(f"{label}: reference_table.pickle is not a DataFrame")
    missing = _REQUIRED_COLUMNS - set(table.columns)
    if missing:
        raise ValueError(
            f"{label}: not an htst reference table (missing {sorted(missing)})"
        )
    if table.attrs.get("nu0_units", "Hz") != "Hz":
        raise ValueError(f"{label}: nu0 units {table.attrs.get('nu0_units')!r}")
    ini = _read_ini(directory / "input.in")
    rate = ini.get("rateconstant", {})
    temperature = float(table.attrs.get("T", rate.get("T", "nan")))
    k0 = float(table.attrs.get("k0", rate.get("k0", "nan")))
    for key, stored in (("T", temperature), ("k0", k0)):
        if key in rate and not math.isclose(float(rate[key]), stored, rel_tol=1e-12):
            raise ValueError(
                f"{label}: input.in {key} = {rate[key]} but the table was built "
                f"with {stored}"
            )
    if not (math.isfinite(temperature) and math.isfinite(k0)):
        raise ValueError(f"{label}: no temperature or k0 in the table or input.in")
    rcut_text = ini.get("atomicenvironment", {}).get("rcut")
    rcut = (
        float(rcut_text)
        if rcut_text is not None and rcut_text not in ("", "None")
        else None
    )
    settings = dict(table.attrs.get("settings") or {})
    window = (
        _float_or_none(settings.get("nu0_min_hz")),
        _float_or_none(settings.get("nu0_max_hz")),
    )
    cell, species = read_xyz_cell(directory / "initial_config.xyz")
    supercell = fcc_supercell(cell, len(species))
    n_vacancies = supercell[2] if supercell is not None else None
    kind = {1: "monovacancy", 2: "divacancy"}.get(n_vacancies or 0, "other")
    lattice = (
        FccLattice.build(supercell[1], float(cell[0, 0]), rcut)
        if supercell is not None and rcut is not None
        else None
    )
    low, high = default_window_hz()

    directions = _directions(table)
    rows: list[dict[str, Any]] = []
    by_idx: dict[int, dict[str, Any]] = {}
    for _, record in table.sort_values("idx_ref").iterrows():
        idx = int(record["idx_ref"])
        status = str(record["nu0_status"])
        if status not in NU0_STATUSES:
            raise ValueError(f"{label} row {idx}: unknown nu0_status {status!r}")
        nu0 = _float_or_none(record["nu0"])
        if status == NU0_OK and nu0 is None:
            raise ValueError(f"{label} row {idx}: accepted without a finite nu0")
        mover = int(record["move_atom_idx"])
        direction, basis = directions[idx]
        row: dict[str, Any] = {
            "run": label,
            "idx_ref": idx,
            "idx_backward": int(record["idx_backward"]),
            "direction": direction,
            "direction_basis": basis,
            "species": str(record["types"][mover]),
            "barrier_ev": float(record["energy_barrier"]),
            "nu0_hz": nu0 if status == NU0_OK else None,
            "status": status,
            "reason": record["nu0_reason"]
            if isinstance(record["nu0_reason"], str)
            else "",
            "outside_default_window": (
                None if status != NU0_OK else not (low <= float(nu0 or 0.0) <= high)
            ),
        }
        row.update(dict.fromkeys(_LATTICE_KEYS))
        if lattice is not None and n_vacancies is not None:
            row.update(
                lattice_analysis(
                    np.asarray(record["initial_positions"], dtype=float),
                    np.asarray(record["final_positions"], dtype=float),
                    mover,
                    lattice,
                    n_vacancies,
                )
            )
        rows.append(row)
        by_idx[idx] = row

    executed, species_basis = _executed_events(
        directory, label, temperature, k0, species, by_idx
    )
    files = [
        name
        for name in (
            "reference_table.pickle",
            "input.in",
            "initial_config.xyz",
            "pykmc.out",
            "pykmc.events",
            "pykmc.log",
        )
        if (directory / name).is_file()
    ]
    meta: dict[str, Any] = {
        "run": label,
        "kind": kind,
        "n_atoms": len(species),
        "n_vacancies": n_vacancies,
        "composition": dict(sorted(Counter(species).items())),
        "cell_length_a": float(cell[0, 0]),
        "a0_a": None if supercell is None else supercell[1],
        "rcut_a": rcut,
        "trust_radius_a": None if lattice is None else lattice.trust_radius,
        "temperature_k": temperature,
        "k0_per_ps": k0,
        "style": table.attrs.get("style"),
        "settings": settings,
        "window_hz": list(window),
        "n_rows": len(rows),
        "status_counts": dict(sorted(Counter(r["status"] for r in rows).items())),
        "direction_counts": dict(sorted(Counter(r["direction"] for r in rows).items())),
        "direction_basis_counts": dict(
            sorted(Counter(r["direction_basis"] for r in rows).items())
        ),
        "n_steps": len(executed),
        "executed_source_counts": dict(
            sorted(Counter(e["source"] for e in executed).items())
        ),
        "executed_species_basis": species_basis,
        "files_sha256": {name: _sha256(directory / name) for name in files},
    }
    return RunRows(label, kind, meta, rows, executed)


def _executed_events(
    directory: Path,
    label: str,
    temperature: float,
    k0: float,
    species: list[str],
    by_idx: dict[int, dict[str, Any]],
) -> tuple[list[dict[str, Any]], str]:
    """Return the executed event of every step and how its species was found."""
    steps = _step_table((directory / "pykmc.out").read_text(errors="replace"))
    numbers = [s["step"] for s in steps]
    if len(numbers) != len(set(numbers)):
        raise ValueError(f"{label}: pykmc.out repeats steps (restarted run?)")
    events_path = directory / "pykmc.events"
    selected = (
        _selected_events(events_path.read_text(errors="replace"))
        if events_path.is_file()
        else {}
    )
    log_path = directory / "pykmc.log"
    sites = (
        _site_lines(log_path.read_text(errors="replace")) if log_path.is_file() else {}
    )
    by_ref: dict[int, list[tuple[int, float]]] = {}
    for (_, ref_key), hits in sites.items():
        by_ref.setdefault(ref_key, []).extend(hits)
    kb = PhysicalConstants.kb
    executed = []
    for step in steps:
        number, ref = step["step"], step["ref_index"]
        barrier, rate = step["barrier_ev"], step["k_per_ps"]
        chosen = selected.get(number)
        atom: int | None = None
        mover_species: str | None = None
        if chosen is not None:
            if (
                chosen["ref_index"] != ref
                or not math.isclose(chosen["barrier_ev"], barrier, abs_tol=1e-5)
                or not math.isclose(chosen["k_per_ps"], rate, rel_tol=1e-5)
            ):
                raise ValueError(
                    f"{label} step {number}: pykmc.events selected row {chosen} "
                    f"disagrees with pykmc.out {step}"
                )
            atom = int(chosen["atom_index"])
            mover_species = species[atom]
            if mover_species != chosen["species"]:
                raise ValueError(
                    f"{label} step {number}: atom {atom} is {mover_species} in "
                    f"initial_config.xyz but {chosen['species']} in pykmc.events"
                )
        elif ref in by_idx:
            mover_species = by_idx[ref]["species"]
        nu0 = (
            rate * math.exp(barrier / (kb * temperature)) * 1.0e12 if rate > 0 else None
        )
        source, status = "unknown", "unresolved"
        if nu0 is not None:
            hits = (
                sites.get((atom, ref), []) if atom is not None else by_ref.get(ref, [])
            )
            logged = [value for logged_step, value in hits if logged_step <= number]
            reference = by_idx.get(ref)
            if any(math.isclose(v, nu0, rel_tol=SITE_NU0_RTOL) for v in logged):
                source, status = "site", NU0_OK
            elif (
                reference is not None
                and reference["status"] == NU0_OK
                and math.isclose(reference["nu0_hz"], nu0, rel_tol=VALUE_RTOL)
            ):
                source, status = "reference", NU0_OK
            elif math.isclose(nu0, k0 * 1.0e12, rel_tol=VALUE_RTOL):
                source = "k0"
                status = (
                    reference["status"]
                    if reference is not None and reference["status"] != NU0_OK
                    else "unresolved"
                )
        executed.append(
            {
                "run": label,
                "step": number,
                "ref_index": ref,
                "atom_index": atom,
                "species": mover_species,
                "barrier_ev": barrier,
                "k_per_ps": rate,
                "nu0_hz": nu0,
                "status": status,
                "source": source,
            }
        )
    if not selected:
        basis = "reference row (no pykmc.events)"
    else:
        basis = "initial_config.xyz at the pykmc.events central atom"
    return executed, basis


# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------


def summary_from_runs(run_dirs: Sequence[str | Path]) -> dict[str, Any]:
    """Build the JSON-serialisable summary of a set of run directories.

    Parameters
    ----------
    run_dirs : sequence of str or Path
        Run directories (see :func:`extract_run`); monovacancy and divacancy
        cells are told apart by the atom count.

    Returns
    -------
    dict
        ``schema``, ``meta`` (per-run facts, cell counts, windows, settings),
        ``rows`` and ``executed`` (monovacancy runs), ``divacancy`` and
        ``divacancy_executed`` (divacancy runs).

    Raises
    ------
    ValueError
        If no directory is given or a run is neither a monovacancy nor a
        divacancy fcc cell.

    """
    if not run_dirs:
        raise ValueError("no run directories")
    runs: list[RunRows] = []
    seen: Counter[str] = Counter()
    for path in run_dirs:
        directory = _resolve_run_dir(path)
        label = _run_label(directory)
        seen[label] += 1
        if seen[label] > 1:
            label = f"{label}#{seen[label]}"
        run = extract_run(directory, label)
        if run.kind not in ("monovacancy", "divacancy"):
            raise ValueError(
                f"{run.run_id}: not a monovacancy or divacancy cube-aligned fcc "
                f"cell ({run.meta['n_atoms']} atoms)"
            )
        runs.append(run)
    mono = [run for run in runs if run.kind == "monovacancy"]
    di = [run for run in runs if run.kind == "divacancy"]

    def unique(values: Iterable[Any]) -> list[Any]:
        out: list[Any] = []
        for value in values:
            if value not in out:
                out.append(value)
        return out

    return {
        "schema": SCHEMA,
        "meta": {
            "runs": [run.meta for run in runs],
            "cells": {"monovacancy": len(mono), "divacancy": len(di)},
            "temperature_k": unique(run.meta["temperature_k"] for run in runs),
            "window_hz": unique(run.meta["window_hz"] for run in runs),
            "settings": unique(run.meta["settings"] for run in runs),
            "default_window_hz": list(default_window_hz()),
        },
        "rows": [row for run in mono for row in run.rows],
        "executed": [row for run in mono for row in run.executed],
        "divacancy": [row for run in di for row in run.rows],
        "divacancy_executed": [row for run in di for row in run.executed],
    }


def write_summary(path: str | Path, summary: dict[str, Any]) -> None:
    """Write a summary as JSON, one row per line in the row lists.

    Parameters
    ----------
    path : str or Path
        Output file; parent directories are created.
    summary : dict
        A summary from :func:`summary_from_runs` (or of the same schema).

    Raises
    ------
    ValueError
        If the summary holds a non-finite float.

    """
    chunks = [
        f'"schema": {json.dumps(summary["schema"])}',
        '"meta": '
        + json.dumps(summary["meta"], indent=1, sort_keys=True, allow_nan=False),
    ]
    for key in _ROW_LISTS:
        body = ",\n".join(
            json.dumps(row, sort_keys=True, allow_nan=False)
            for row in summary.get(key, [])
        )
        chunks.append(f'"{key}": [\n{body}\n]' if body else f'"{key}": []')
    text = "{\n" + ",\n".join(chunks) + "\n}\n"
    json.loads(text)
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(text)


def load_summary(path: str | Path) -> dict[str, Any]:
    """Load a summary written by :func:`write_summary`.

    Parameters
    ----------
    path : str or Path
        The JSON file.

    Returns
    -------
    dict
        The summary.

    Raises
    ------
    ValueError
        If the schema is not :data:`SCHEMA` or a row list is missing.

    """
    summary = json.loads(Path(path).read_text())
    if summary.get("schema") != SCHEMA:
        raise ValueError(f"{path}: schema {summary.get('schema')!r}, expected {SCHEMA}")
    missing = [key for key in ("meta", *_ROW_LISTS) if key not in summary]
    if missing:
        raise ValueError(f"{path}: missing {missing}")
    return summary


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------


@dataclass
class Evaluation:
    """Every statistic of a summary and the verdict of each acceptance criterion.

    Attributes
    ----------
    meta : dict
        The summary's ``meta``.
    stats : dict
        Statistics per set (see :func:`evaluate`).
    criteria : list of tuple
        ``(criterion, passed, message)`` in :data:`CRITERIA` order.

    """

    meta: dict[str, Any]
    stats: dict[str, Any]
    criteria: list[tuple[str, bool, str]]

    def __repr__(self) -> str:
        """Return a short summary (pytest prints it in failure tracebacks)."""
        failed = [name for name, _, _ in self.failures]
        return f"Evaluation({len(self.criteria)} criteria, failed={failed})"

    @property
    def failures(self) -> list[tuple[str, bool, str]]:
        """Return the failed criteria."""
        return [item for item in self.criteria if not item[1]]

    @property
    def passed(self) -> bool:
        """Return True when every criterion passed."""
        return not self.failures


def smoothed_fwhm_decades(logs: Sequence[float]) -> float | None:
    """Return the FWHM (decades) of a Gaussian-smoothed histogram of log10 values.

    The values are binned at :data:`FWHM_BIN_DEX` and the histogram is
    convolved with a Gaussian of standard deviation :data:`FWHM_BANDWIDTH_DEX`;
    the width is that of the contiguous region around the highest point where
    the smoothed density is at least half its maximum, with the two crossings
    interpolated linearly.

    Parameters
    ----------
    logs : sequence of float
        log10 of the prefactors (s^-1).

    Returns
    -------
    float or None
        The FWHM in decades; ``None`` for an empty set.

    """
    if len(logs) == 0:
        return None
    values = np.asarray(logs, dtype=float)
    pad = 6.0 * FWHM_BANDWIDTH_DEX
    edges = np.arange(
        values.min() - pad, values.max() + pad + FWHM_BIN_DEX, FWHM_BIN_DEX
    )
    counts, _ = np.histogram(values, bins=edges)
    reach = int(round(5.0 * FWHM_BANDWIDTH_DEX / FWHM_BIN_DEX))
    offsets = np.arange(-reach, reach + 1) * FWHM_BIN_DEX
    kernel = np.exp(-0.5 * (offsets / FWHM_BANDWIDTH_DEX) ** 2)
    density = np.convolve(counts.astype(float), kernel, mode="same")
    centres = 0.5 * (edges[:-1] + edges[1:])
    peak = int(np.argmax(density))
    half = 0.5 * density[peak]
    left = peak
    while left > 0 and density[left - 1] >= half:
        left -= 1
    right = peak
    while right < len(density) - 1 and density[right + 1] >= half:
        right += 1

    def crossing(inside: int, outside: int) -> float:
        a, b = density[inside], density[outside]
        fraction = (a - half) / (a - b) if a != b else 0.0
        return float(centres[inside] + fraction * (centres[outside] - centres[inside]))

    low = crossing(left, left - 1) if left > 0 else float(centres[0])
    high = (
        crossing(right, right + 1) if right < len(density) - 1 else float(centres[-1])
    )
    return high - low


def _set_stats(rows: Sequence[dict[str, Any]]) -> dict[str, Any]:
    """Return ``n``, mean barrier and the nu0 statistics of accepted rows.

    ``log10_linear_mean`` (log10 of the linear mean nu0) is the statistic the
    criteria assert; ``mean_log10_nu`` (mean of log10 nu0) is reported beside
    it; ``std_log10`` and ``fwhm_decades`` are the width of the log10(nu0)
    distribution.
    """
    nus = [float(r["nu0_hz"]) for r in rows]
    if not nus:
        return {
            "n": 0,
            "mean_barrier_ev": None,
            "log10_linear_mean": None,
            "linear_mean_hz": None,
            "mean_log10_nu": None,
            "std_log10": None,
            "fwhm_decades": None,
            "min_hz": None,
            "max_hz": None,
        }
    logs = np.log10(nus)
    linear = float(np.mean(nus))
    return {
        "n": len(nus),
        "mean_barrier_ev": float(np.mean([float(r["barrier_ev"]) for r in rows])),
        "log10_linear_mean": math.log10(linear),
        "linear_mean_hz": linear,
        "mean_log10_nu": float(np.mean(logs)),
        "std_log10": float(np.std(logs, ddof=1)) if len(nus) > 1 else None,
        "fwhm_decades": smoothed_fwhm_decades(logs.tolist()),
        "min_hz": float(np.min(nus)),
        "max_hz": float(np.max(nus)),
    }


def _accepted(rows: Iterable[dict[str, Any]]) -> list[dict[str, Any]]:
    """Return the rows with an accepted nu0."""
    return [r for r in rows if r["status"] == NU0_OK and r.get("nu0_hz") is not None]


def _pair_mean(rows: Sequence[dict[str, Any]]) -> dict[str, Any]:
    """Return pair-averaged prefactor statistics (each reciprocal pair counts once).

    ``log10_linear_mean`` is log10 of the mean over pairs of the pair's linear
    mean nu0; ``mean_log10_nu`` the mean over pairs of the pair's mean log10(nu0).
    """
    by_key = {(r["run"], int(r["idx_ref"])): r for r in rows}
    groups: dict[tuple[str, int, int], list[float]] = {}
    for (run, idx), row in by_key.items():
        other = int(row["idx_backward"])
        partner = by_key.get((run, other))
        linked = partner is not None and int(partner["idx_backward"]) == idx
        key = (run, min(idx, other), max(idx, other)) if linked else (run, idx, idx)
        if row["status"] == NU0_OK and row.get("nu0_hz") is not None:
            groups.setdefault(key, []).append(float(row["nu0_hz"]))
    linear = [float(np.mean(values)) for values in groups.values()]
    logs = [float(np.mean(np.log10(values))) for values in groups.values()]
    return {
        "n": len(linear),
        "log10_linear_mean": math.log10(float(np.mean(linear))) if linear else None,
        "mean_log10_nu": float(np.mean(logs)) if logs else None,
    }


def _dev(mean_log10: float | None, target_hz: float) -> float | None:
    """Return ``mean_log10 - log10(target_hz)``."""
    return None if mean_log10 is None else mean_log10 - math.log10(target_hz)


def _fmt(value: float | None, spec: str) -> str:
    """Format a number, ``-`` for ``None``."""
    return "-" if value is None else format(value, spec)


def _ids(rows: Iterable[dict[str, Any]], limit: int = 8) -> str:
    """Return ``run:idx_ref`` labels of rows, truncated."""
    labels = [f"{r['run']}:{r['idx_ref']}" for r in rows]
    tail = f" (+{len(labels) - limit} more)" if len(labels) > limit else ""
    return ", ".join(labels[:limit]) + tail


def evaluate(summary: dict[str, Any]) -> Evaluation:
    """Compute every statistic of a summary once and judge the acceptance criteria.

    Parameters
    ----------
    summary : dict
        A summary from :func:`summary_from_runs` or :func:`load_summary`.

    Returns
    -------
    Evaluation
        Statistics and ``(criterion, passed, message)`` per criterion.

    """
    rows = list(summary["rows"])
    executed = list(summary["executed"])
    divacancy = list(summary["divacancy"])
    ok_rows = _accepted(rows)
    stats: dict[str, Any] = {}
    criteria: list[tuple[str, bool, str]] = []

    # 1. catalogue size
    runs = [
        r for r in summary["meta"].get("runs", []) if r.get("kind") == "monovacancy"
    ]
    stats["n_rows"] = len(rows)
    criteria.append(
        (
            "catalogue size",
            len(rows) >= MIN_ROWS,
            f"{len(rows)} catalogued directed rows in {len(runs)} monovacancy run(s) "
            f"(required >= {MIN_ROWS}); {len(ok_rows)} with an accepted nu0",
        )
    )

    # 2. all events
    stats["all"] = _set_stats(ok_rows)
    stats["all_forward"] = _set_stats(
        [r for r in ok_rows if r["direction"] == "forward"]
    )
    stats["all_pairs"] = _pair_mean(rows)
    block = stats["all"]
    all_dev = _dev(block["log10_linear_mean"], ALL_EVENTS_NU_HZ)
    criteria.append(
        (
            "all-events mean prefactor",
            all_dev is not None and abs(all_dev) <= DEX_TOL,
            f"log10 <nu> (asserted) {_fmt(block['log10_linear_mean'], '.3f')} "
            f"(linear mean {_fmt(block['linear_mean_hz'], '.3g')} s^-1) over "
            f"{block['n']} accepted rows vs log10({ALL_EVENTS_NU_HZ:.3g}) = "
            f"{math.log10(ALL_EVENTS_NU_HZ):.3f}: deviation {_fmt(all_dev, '+.3f')} "
            f"dex (tolerance {DEX_TOL} dex); <log10 nu> (reported) "
            f"{_fmt(block['mean_log10_nu'], '.3f')} (deviation "
            f"{_fmt(_dev(block['mean_log10_nu'], ALL_EVENTS_NU_HZ), '+.3f')} dex); "
            f"forward rows only: log10 <nu> "
            f"{_fmt(stats['all_forward']['log10_linear_mean'], '.3f')}, <log10 nu> "
            f"{_fmt(stats['all_forward']['mean_log10_nu'], '.3f')} "
            f"({stats['all_forward']['n']}); pair-averaged: log10 <nu> "
            f"{_fmt(stats['all_pairs']['log10_linear_mean'], '.3f')}, <log10 nu> "
            f"{_fmt(stats['all_pairs']['mean_log10_nu'], '.3f')} "
            f"({stats['all_pairs']['n']} pairs); width (reported): std "
            f"{_fmt(block['std_log10'], '.2f')}, FWHM "
            f"{_fmt(block['fwhm_decades'], '.2f')} decades (paper FWHM "
            f"{PAPER_FWHM_DECADES['all']})",
        )
    )

    # 3. selected (executed) events with E > 0.3 eV
    selected = [
        r
        for r in _accepted(executed)
        if float(r["barrier_ev"]) > SELECTED_MIN_BARRIER_EV
    ]
    stats["selected"] = _set_stats(selected)
    stats["catalogue_above"] = _set_stats(
        [r for r in ok_rows if float(r["barrier_ev"]) > SELECTED_MIN_BARRIER_EV]
    )
    stats["executed_counts"] = {
        "total": len(executed),
        "source": dict(sorted(Counter(str(r["source"]) for r in executed).items())),
        "status": dict(sorted(Counter(str(r["status"]) for r in executed).items())),
        "accepted_above": len(selected),
    }
    sel = stats["selected"]
    sel_dev = _dev(sel["log10_linear_mean"], SELECTED_NU_HZ)
    sel_barrier_dev = (
        None
        if sel["mean_barrier_ev"] is None
        else sel["mean_barrier_ev"] - SELECTED_BARRIER_EV
    )
    above = stats["catalogue_above"]
    criteria.append(
        (
            "selected-events means",
            sel_dev is not None
            and sel_barrier_dev is not None
            and abs(sel_dev) <= DEX_TOL
            and abs(sel_barrier_dev) <= BARRIER_TOL_EV,
            f"{sel['n']} executed events with E > {SELECTED_MIN_BARRIER_EV} eV and an "
            f"accepted nu0: log10 <nu> (asserted) "
            f"{_fmt(sel['log10_linear_mean'], '.3f')} (linear mean "
            f"{_fmt(sel['linear_mean_hz'], '.3g')} s^-1) vs "
            f"log10({SELECTED_NU_HZ:.3g}): deviation {_fmt(sel_dev, '+.3f')} dex "
            f"(tolerance {DEX_TOL}); mean barrier {_fmt(sel['mean_barrier_ev'], '.3f')} "
            f"eV vs {SELECTED_BARRIER_EV} +/- {BARRIER_TOL_EV} eV (deviation "
            f"{_fmt(sel_barrier_dev, '+.3f')}); <log10 nu> (reported) "
            f"{_fmt(sel['mean_log10_nu'], '.3f')} (deviation "
            f"{_fmt(_dev(sel['mean_log10_nu'], SELECTED_NU_HZ), '+.3f')} dex); "
            f"width (reported): std {_fmt(sel['std_log10'], '.2f')}, FWHM "
            f"{_fmt(sel['fwhm_decades'], '.2f')} decades (paper FWHM "
            f"{PAPER_FWHM_DECADES['selected']}); "
            f"executed total {len(executed)}, sources "
            f"{stats['executed_counts']['source']}; catalogue E > "
            f"{SELECTED_MIN_BARRIER_EV} eV: mean barrier "
            f"{_fmt(above['mean_barrier_ev'], '.3f')} eV, log10 <nu> "
            f"{_fmt(above['log10_linear_mean'], '.3f')}, <log10 nu> "
            f"{_fmt(above['mean_log10_nu'], '.3f')} ({above['n']} rows)",
        )
    )

    # 4. element ordering
    stats["elements"] = {
        element: _set_stats([r for r in ok_rows if r["species"] == element])
        for element in sorted({str(r["species"]) for r in ok_rows} | set(ELEMENT_ORDER))
    }
    stats["elements_selected"] = {
        element: _set_stats([r for r in selected if r["species"] == element])
        for element in sorted(
            {str(r["species"]) for r in selected} | set(ELEMENT_ORDER)
        )
    }

    def in_order(block: dict[str, Any], key: str) -> bool:
        values = [block[e][key] for e in ELEMENT_ORDER]
        return all(v is not None for v in values) and all(
            a > b for a, b in itertools.pairwise(values)
        )

    ordered = in_order(stats["elements"], "log10_linear_mean")
    sel_ordered = in_order(stats["elements_selected"], "log10_linear_mean")
    stats["element_order"] = {
        "catalogue_linear": ordered,
        "catalogue_log_mean": in_order(stats["elements"], "mean_log10_nu"),
        "selected_linear": sel_ordered,
        "selected_log_mean": in_order(stats["elements_selected"], "mean_log10_nu"),
    }

    def describe(block: dict[str, Any], key: str) -> str:
        return ", ".join(
            f"{e} {_fmt(block[e][key], '.3f')} ({block[e]['n']})" for e in ELEMENT_ORDER
        )

    def verdict(flag: bool) -> str:
        return "ordered" if flag else "not ordered"

    order = stats["element_order"]
    criteria.append(
        (
            "element ordering",
            ordered,
            "catalogue log10 <nu> (asserted) by mover: "
            f"{describe(stats['elements'], 'log10_linear_mean')}; required "
            f"{' > '.join(ELEMENT_ORDER)}; catalogue <log10 nu> (reported): "
            f"{describe(stats['elements'], 'mean_log10_nu')}, "
            f"{verdict(order['catalogue_log_mean'])}; executed E > "
            f"{SELECTED_MIN_BARRIER_EV} eV (reported): log10 <nu> "
            f"{describe(stats['elements_selected'], 'log10_linear_mean')}, "
            f"{verdict(order['selected_linear'])}; <log10 nu> "
            f"{describe(stats['elements_selected'], 'mean_log10_nu')}, "
            f"{verdict(order['selected_log_mean'])}",
        )
    )

    # 5. span
    if ok_rows:
        logs = np.log10([float(r["nu0_hz"]) for r in ok_rows])
        span = float(np.max(logs) - np.min(logs))
        stats["span"] = {
            "min_hz": float(10 ** np.min(logs)),
            "max_hz": float(10 ** np.max(logs)),
            "decades": span,
        }
    else:
        stats["span"] = {"min_hz": None, "max_hz": None, "decades": None}
    decades = stats["span"]["decades"]
    criteria.append(
        (
            "prefactor span",
            decades is not None and decades >= MIN_SPAN_DECADES,
            f"accepted catalogue rows span {_fmt(stats['span']['min_hz'], '.3g')} - "
            f"{_fmt(stats['span']['max_hz'], '.3g')} s^-1 = {_fmt(decades, '.2f')} "
            f"decades (required >= {MIN_SPAN_DECADES:g})",
        )
    )

    # 6. status hygiene (every catalogued row, both cell kinds)
    catalogue = rows + divacancy
    bad_status = [r for r in catalogue if r["status"] in (NU0_LEGACY, NU0_STALE)]
    silent = [
        r
        for r in catalogue
        if r["status"] == NU0_REJECTED and not str(r.get("reason") or "").strip()
    ]
    low, high = default_window_hz()
    accepted_all = _accepted(catalogue)
    below = sum(1 for r in accepted_all if float(r["nu0_hz"]) < low)
    over = sum(1 for r in accepted_all if float(r["nu0_hz"]) > high)
    stats["status_counts"] = {
        status: sum(1 for r in catalogue if r["status"] == status)
        for status in NU0_STATUSES
    }
    stats["outside_default_window"] = {
        "below": below,
        "above": over,
        "accepted": len(accepted_all),
        "window_hz": [low, high],
    }
    stats["rejection_reasons"] = dict(
        Counter(
            str(r.get("reason") or "").split(":")[0] or "(empty)"
            for r in catalogue
            if r["status"] == NU0_REJECTED
        ).most_common()
    )
    parts = [
        f"status counts {stats['status_counts']}",
        f"legacy/stale rows {len(bad_status)}"
        + (f" [{_ids(bad_status)}]" if bad_status else ""),
        f"rejected rows without a reason {len(silent)}"
        + (f" [{_ids(silent)}]" if silent else ""),
        f"pending {stats['status_counts'].get('pending', 0)}",
        f"accepted rows outside the code-default window {low:.0e}-{high:.0e} Hz: "
        f"{below} below, {over} above (of {len(accepted_all)}; reported)",
    ]
    criteria.append(("status hygiene", not bad_status and not silent, "; ".join(parts)))

    # 7. divacancy transitions
    classes = Counter(
        (str(r.get("pair_initial")), str(r.get("pair_final"))) for r in divacancy
    )
    stats["divacancy_counts"] = {
        f"{a}->{b}": n for (a, b), n in sorted(classes.items())
    }
    stats["divacancy"] = {}
    dv_ok = True
    dv_parts = []
    for (start, end), (barrier, nu) in DIVACANCY_TARGETS.items():
        members = _accepted(
            r
            for r in divacancy
            if r.get("pair_initial") == start and r.get("pair_final") == end
        )
        block = _set_stats(members)
        name = f"{start}->{end}"
        dev = _dev(block["log10_linear_mean"], nu)
        e_dev = (
            None
            if block["mean_barrier_ev"] is None
            else block["mean_barrier_ev"] - barrier
        )
        if block["n"] < DIVACANCY_MIN_ROWS:
            dv_verdict = "insufficient"
        elif (
            dev is not None
            and e_dev is not None
            and abs(dev) <= DEX_TOL
            and abs(e_dev) <= BARRIER_TOL_EV
        ):
            dv_verdict = "pass"
        else:
            dv_verdict = "fail"
            dv_ok = False
        block.update({"dex": dev, "barrier_dev_ev": e_dev, "verdict": dv_verdict})
        stats["divacancy"][name] = block
        dv_parts.append(
            f"{name}: {block['n']} accepted rows, mean barrier "
            f"{_fmt(block['mean_barrier_ev'], '.3f')} eV vs {barrier} +/- "
            f"{BARRIER_TOL_EV} (deviation {_fmt(e_dev, '+.3f')}), log10 <nu> "
            f"(asserted) {_fmt(block['log10_linear_mean'], '.3f')} vs "
            f"log10({nu:.3g}) (deviation {_fmt(dev, '+.3f')} dex), <log10 nu> "
            f"(reported) {_fmt(block['mean_log10_nu'], '.3f')} (deviation "
            f"{_fmt(_dev(block['mean_log10_nu'], nu), '+.3f')} dex): {dv_verdict}"
            + (
                f" (fewer than {DIVACANCY_MIN_ROWS} rows)"
                if dv_verdict == "insufficient"
                else ""
            )
        )
    dv_parts.append(
        f"{len(divacancy)} divacancy rows, classes {stats['divacancy_counts']}"
    )
    criteria.append(("divacancy transitions", dv_ok, "; ".join(dv_parts)))

    # Reported diagnostic (not a criterion): pair classifier on the monovacancy rows
    analysed = [r for r in rows if r.get("on_lattice") is not None]
    hops = [r for r in analysed if r.get("on_lattice") and r.get("hop_shell") == 1]
    broken = [
        r
        for r in hops
        if not (
            r.get("vacancies_initial") == 1
            and r.get("vacancies_final") == 1
            and r.get("mover_final_site_vacant")
            and r.get("mover_initial_site_vacated")
        )
    ]
    shells = Counter(
        str(r.get("hop_shell")) if r.get("on_lattice") else "off-lattice"
        for r in analysed
    )
    stats["classifier"] = {
        "analysed": len(analysed),
        "checked": len(hops),
        "failed": len(broken),
        "failed_ids": _ids(broken) if broken else "",
        "hop_shells": dict(sorted(shells.items())),
    }
    return Evaluation(meta=dict(summary["meta"]), stats=stats, criteria=criteria)


# ---------------------------------------------------------------------------
# Report
# ---------------------------------------------------------------------------


def _window(value: Any) -> str:
    """Format a ``[low, high]`` window (or a list of them) in Hz."""
    if (
        isinstance(value, (list, tuple))
        and len(value) == 2
        and not isinstance(value[0], (list, tuple))
    ):
        return f"{_fmt(value[0], '.0e')} - {_fmt(value[1], '.0e')}"
    if isinstance(value, (list, tuple)):
        return "; ".join(_window(item) for item in value)
    return str(value)


def _cell(text: str) -> str:
    """Escape a Markdown table cell."""
    return text.replace("|", "\\|")


def _table_header(asserted: bool) -> list[str]:
    """Return the header and rule of a Table-I-layout table.

    ``log10 <nu>`` is log10 of the linear mean nu0, ``<log10 nu>`` the mean of
    log10(nu0); in the asserted table the first is the asserted statistic.
    """
    linear = "log10 <nu> (asserted)" if asserted else "log10 <nu> (reported)"
    columns = [
        "Set",
        "N",
        "<E> (eV)",
        "paper <E> (eV)",
        linear,
        "linear <nu> (s^-1)",
        "paper linear <nu> (s^-1)",
        "deviation of log10 <nu> (dex)",
        "<log10 nu> (reported)",
        "10^<log10 nu> (s^-1)",
        "deviation of <log10 nu> (dex, reported)",
        "asserted here",
    ]
    return [
        "| " + " | ".join(columns) + " |",
        "| " + " | ".join(["---"] * len(columns)) + " |",
    ]


def _table_row(
    label: str,
    block: dict[str, Any],
    paper: tuple[float, float] | None,
    asserted: str = "-",
) -> str:
    """Return one Table-I-layout row (columns of :func:`_table_header`)."""
    linear = block.get("log10_linear_mean")
    mean = block.get("mean_log10_nu")
    paper_e = "-" if paper is None else f"{paper[0]:.2f}"
    paper_nu = "-" if paper is None else f"{paper[1]:.3g}"
    dev_linear = None if paper is None else _dev(linear, paper[1])
    dev_mean = None if paper is None else _dev(mean, paper[1])
    cells = [
        label,
        str(block.get("n", 0)),
        _fmt(block.get("mean_barrier_ev"), ".3f"),
        paper_e,
        _fmt(linear, ".3f"),
        _fmt(block.get("linear_mean_hz"), ".3g"),
        paper_nu,
        _fmt(dev_linear, "+.3f"),
        _fmt(mean, ".3f"),
        _fmt(None if mean is None else 10**mean, ".3g"),
        _fmt(dev_mean, "+.3f"),
        asserted,
    ]
    return "| " + " | ".join(cells) + " |"


def render_report(evaluation: Evaluation) -> str:
    """Return the Markdown acceptance report (Table I layout).

    Parameters
    ----------
    evaluation : Evaluation
        The result of :func:`evaluate`.

    Returns
    -------
    str
        Markdown text.

    """
    stats, meta = evaluation.stats, evaluation.meta
    runs = meta.get("runs", [])
    failed = evaluation.failures
    verdict = (
        "PASS (every acceptance criterion holds)"
        if not failed
        else f"FAIL ({len(failed)} of {len(evaluation.criteria)} criteria failed: "
        + ", ".join(name for name, _, _ in failed)
        + ")"
    )
    tolerance = f"within {DEX_TOL} dex"
    lines = [
        "# FeNiCr literature validation: acceptance report",
        "",
        f"**Verdict: {verdict}**",
        "",
        "Targets: Lefèvre López et al., Phys. Rev. Materials 8, 013609 (2024), "
        "Table I (FeNiCr 55/28/17, Bonny-2013 EAM); divacancy: Rahman et al., "
        "Phys. Rev. Materials 10, 053606 (2026). The paper's prefactor averages "
        "are linear means, so the asserted statistic is log10 <nu>, log10 of the "
        "linear mean nu0 (like for like); <log10 nu>, the mean of log10(nu0), is "
        "reported beside it and sits below it on a wide distribution; the "
        "divacancy classes are compared the same way. Accepted rows only "
        "(nu0_status ok).",
        "",
        "| Item | Value |",
        "| --- | --- |",
        f"| Cells | {meta.get('cells', {})} |",
        f"| Catalogued directed rows (monovacancy) | {stats['n_rows']} |",
        "| Executed events (monovacancy) | "
        f"{stats['executed_counts']['total']} (sources "
        f"{stats['executed_counts']['source']}) |",
        f"| Temperature (K) | {meta.get('temperature_k')} |",
        f"| nu0 window of the runs (Hz) | {_window(meta.get('window_hz'))} |",
        "| Code-default window (Hz) | "
        f"{_window(stats['outside_default_window']['window_hz'])} |",
        "",
        "## Table I layout",
        "",
        *_table_header(asserted=True),
        _table_row(
            "All (catalogue)",
            stats["all"],
            (ALL_EVENTS_BARRIER_EV, ALL_EVENTS_NU_HZ),
            f"log10 <nu> {tolerance}",
        ),
        _table_row(
            f"Selected, E > {SELECTED_MIN_BARRIER_EV} eV (executed)",
            stats["selected"],
            (SELECTED_BARRIER_EV, SELECTED_NU_HZ),
            f"log10 <nu> {tolerance}; <E> within {BARRIER_TOL_EV} eV",
        ),
    ]
    for element in ELEMENT_ORDER:
        lines.append(
            _table_row(
                f"Diffusing {element} (catalogue)",
                stats["elements"][element],
                ELEMENT_TARGETS[element],
                f"order {' > '.join(ELEMENT_ORDER)} of log10 <nu>",
            )
        )
    pairs = stats["all_pairs"]
    pair_mean = pairs["mean_log10_nu"]
    lines += [
        "",
        "Reported beside (not asserted):",
        "",
        *_table_header(asserted=False),
        _table_row("All, forward rows only", stats["all_forward"], None),
        f"| All, pair-averaged | {pairs['n']} pairs | - | - "
        f"| {_fmt(pairs['log10_linear_mean'], '.3f')} "
        f"| {_fmt(None if pairs['log10_linear_mean'] is None else 10 ** pairs['log10_linear_mean'], '.3g')} "
        f"| - | - | {_fmt(pair_mean, '.3f')} "
        f"| {_fmt(None if pair_mean is None else 10**pair_mean, '.3g')} | - | - |",
        _table_row(
            f"Catalogue, E > {SELECTED_MIN_BARRIER_EV} eV",
            stats["catalogue_above"],
            (ALL_EVENTS_BARRIER_EV, ALL_EVENTS_NU_HZ),
        ),
    ]
    for element in ELEMENT_ORDER:
        lines.append(
            _table_row(
                f"Diffusing {element} (executed, E > {SELECTED_MIN_BARRIER_EV} eV)",
                stats["elements_selected"][element],
                ELEMENT_TARGETS[element],
            )
        )
    others = [e for e in stats["elements"] if e not in ELEMENT_ORDER]
    for element in others:
        lines.append(
            _table_row(
                f"Diffusing {element} (catalogue)", stats["elements"][element], None
            )
        )
    span = stats["span"]
    window = stats["outside_default_window"]
    lines += [
        "",
        "Width of the log10(nu0) distribution (reported, not asserted); FWHM of "
        f"the histogram smoothed by a Gaussian of {FWHM_BANDWIDTH_DEX} dex:",
        "",
        "| Set | N | std of log10(nu0) (decades) | FWHM (decades) | paper FWHM "
        "(decades) |",
        "| --- | --- | --- | --- | --- |",
    ]
    for label, key in (
        ("All (catalogue)", "all"),
        (f"Selected, E > {SELECTED_MIN_BARRIER_EV} eV (executed)", "selected"),
    ):
        block = stats[key]
        lines.append(
            f"| {label} | {block['n']} | {_fmt(block['std_log10'], '.2f')} "
            f"| {_fmt(block['fwhm_decades'], '.2f')} | {PAPER_FWHM_DECADES[key]} |"
        )
    lines += [
        "",
        f"Span of the accepted catalogue rows: {_fmt(span['min_hz'], '.3g')} - "
        f"{_fmt(span['max_hz'], '.3g')} s^-1 = {_fmt(span['decades'], '.2f')} decades "
        "(paper: all 4.2e7 - 2.3e15, selected 1e10 - 4.8e14).",
        "",
        f"Status counts (all catalogued rows, both cell kinds): {stats['status_counts']}; "
        f"rejection reasons {stats['rejection_reasons']}.",
        "",
        f"Accepted rows outside the code-default window: {window['below']} below, "
        f"{window['above']} above, of {window['accepted']}.",
        "",
        "## Divacancy (Rahman 2026)",
        "",
        "| Initial -> final | Rows | Accepted | <E> (eV) | log10 <nu> (asserted) "
        "| linear <nu> (s^-1) | <log10 nu> (reported) | target <E> (eV) "
        "| target linear nu (s^-1) | deviation of log10 <nu> (dex) | verdict |",
        "| " + " | ".join(["---"] * 11) + " |",
    ]
    targets = {f"{a}->{b}": value for (a, b), value in DIVACANCY_TARGETS.items()}
    names = list(targets) + [n for n in stats["divacancy_counts"] if n not in targets]
    for name in names:
        count = stats["divacancy_counts"].get(name, 0)
        block = stats["divacancy"].get(name)
        target = targets.get(name)
        if block is None:
            lines.append(
                f"| {name} | {count} | - | - | - | - | - | - | - | - | reported |"
            )
            continue
        lines.append(
            f"| {name} | {count} | {block['n']} | {_fmt(block['mean_barrier_ev'], '.3f')} "
            f"| {_fmt(block['log10_linear_mean'], '.3f')} "
            f"| {_fmt(block['linear_mean_hz'], '.3g')} "
            f"| {_fmt(block['mean_log10_nu'], '.3f')} "
            f"| {target[0] if target else '-'} | {f'{target[1]:.3g}' if target else '-'} "
            f"| {_fmt(block['dex'], '+.3f')} | {block['verdict']} |"
        )
    classifier = stats["classifier"]
    lines += [
        "",
        "## Pair classifier on the monovacancy rows (reported, not asserted)",
        "",
        f"{classifier['analysed']} rows analysed, {classifier['checked']} on-lattice "
        f"first-neighbour hops checked, {classifier['failed']} failed"
        + (f" [{classifier['failed_ids']}]" if classifier["failed"] else "")
        + f"; mover hop shells {classifier['hop_shells']}. A failure here flags the "
        "divacancy classification as unreliable; it does not fail the validation.",
        "",
        "## Runs",
        "",
        "| Run | Kind | Atoms | a0 (A) | rcut (A) | T (K) | window (Hz) | Rows | "
        "Statuses | Steps | Executed sources |",
        "| " + " | ".join(["---"] * 11) + " |",
    ]
    for run in runs:
        lines.append(
            f"| {_cell(str(run.get('run')))} | {run.get('kind')} | {run.get('n_atoms')} "
            f"| {_fmt(run.get('a0_a'), '.4f')} | {run.get('rcut_a')} "
            f"| {run.get('temperature_k')} | {_window(run.get('window_hz'))} "
            f"| {run.get('n_rows')} "
            f"| {run.get('status_counts')} | {run.get('n_steps')} "
            f"| {run.get('executed_source_counts')} |"
        )
    lines += [
        "",
        "## Acceptance criteria",
        "",
        "| # | Criterion | Result | Detail |",
        "| --- | --- | --- | --- |",
    ]
    for number, (name, passed, message) in enumerate(evaluation.criteria, start=1):
        lines.append(
            f"| {number} | {name} | {'PASS' if passed else 'FAIL'} | {_cell(message)} |"
        )
    lines.append("")
    return "\n".join(lines)


def failure_message(evaluation: Evaluation) -> str:
    """Return the assertion message naming every failed criterion with its numbers.

    Parameters
    ----------
    evaluation : Evaluation
        The result of :func:`evaluate`.

    Returns
    -------
    str
        One line per failed criterion, ``- <criterion>: <message>``.

    """
    failed = evaluation.failures
    lines = [
        f"FeNiCr literature validation: {len(failed)} of {len(evaluation.criteria)} "
        "acceptance criteria failed"
    ]
    lines += [f"- {name}: {message}" for name, _, message in failed]
    return "\n".join(lines)
