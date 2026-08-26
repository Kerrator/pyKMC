"""Real-system tests for the #90 ``atom_coloring_mode`` full-default behaviour.

These tests load the real generated multi-species FCC fixtures (NiCr, NiFe; each a
3x3x3 a=3.52 FCC supercell with a central vacancy and ~10 at.% solute) and prove that
species-resolved ("full") environment coloring yields strictly MORE distinct
environment classes than species-blind ("grey") coloring, because the real element
types split otherwise-identical geometric environments.

Design notes
------------
* The ``"graph"`` AtomicEnvironment style is used deliberately (see ``compute_graph``
  in ``pykmc/atomic_environment.py``). Unlike ``"cna/graph"``, the pure ``"graph"``
  style graph-hashes EVERY atom (``atom_idx=None`` in
  ``pykmc/environments/graph_nauty.py::graph``), so the colored graph hash is
  exercised for all 107 atoms rather than only the handful of non-crystalline atoms
  near the vacancy. That makes the full-vs-grey margin large and robust (grey ~ 4
  classes vs full ~ 105 classes on these fixtures) rather than a marginal
  vacancy-local effect that the ``cna/graph`` style would give on a near-perfect bulk.
* ``coloring_mode="full"`` threads ``types`` into ``pynauty`` vertex coloring;
  ``"grey"`` passes ``types=None`` (species-blind). This mirrors the live call sites
  in ``pykmc/kmc.py`` and ``pykmc/basins/basin.py``.
* Fully deterministic: fixed on-disk fixtures, fixed cutoffs, no RNG.
"""

from __future__ import annotations

from collections import Counter
from pathlib import Path

import pytest

from pykmc.atomic_environment import AtomicEnvironment
from pykmc.config import AtomicEnvironmentConfig
from pykmc.neighbors_list import NeighborsList
from pykmc.system import System

DATA_DIR = Path(__file__).parent / "data"

# FCC a=3.52: 1NN ~= 2.49 A. rnei comfortably brackets the first shell only;
# rcut defines the per-atom graph environment (a few shells).
RNEI = 3.0
RCUT = 4.5

FIXTURES = [
    ("NiCr", DATA_DIR / "nicr_fcc_3x3x3_1vac.xyz"),
    ("NiFe", DATA_DIR / "nife_fcc_3x3x3_1vac.xyz"),
]


def test_default_coloring_mode_is_full() -> None:
    """Pin the #90 behaviour change: the default ``atom_coloring_mode`` is ``full``.

    A config that omits the field resolves to species-resolved colouring; this guards
    against a silent revert of the default back to ``grey``.
    """
    field = AtomicEnvironmentConfig.model_fields["atom_coloring_mode"]
    assert field.default == "full"


def _build_neighbors(path: Path) -> tuple[System, list, list]:
    """Load a fixture and return (system, rnei_list, rcut_environment_list)."""
    system = System.create_from_file(str(path))
    nl = NeighborsList(system, RNEI, RCUT)
    return system, nl.neighbors_list["rnei"], nl.neighbors_list["rcut"]


def _graph_classes(rnei: list, rcut: list, types: list[str], mode: str) -> list[str]:
    """Per-atom 'graph'-style environment IDs under the given coloring mode."""
    ae = AtomicEnvironment(
        "graph",
        rnei,
        rcut,
        0,
        types=types,
        coloring_mode=mode,
    )
    return ae.atomic_environment_list


@pytest.mark.parametrize("name,path", FIXTURES, ids=[f[0] for f in FIXTURES])
def test_full_coloring_splits_more_classes_than_grey(name: str, path: Path) -> None:
    """Full (species-resolved) coloring yields strictly more classes than grey."""
    assert path.exists(), f"missing fixture for {name}: {path}"
    system, rnei, rcut = _build_neighbors(path)
    types = list(system.types)

    # Sanity: the fixture is genuinely multi-species, otherwise the test is vacuous.
    assert len(set(types)) >= 2, f"{name} fixture is not multi-species: {set(types)}"

    grey_ids = _graph_classes(rnei, rcut, types, "grey")
    full_ids = _graph_classes(rnei, rcut, types, "full")

    grey_n = len(set(grey_ids))
    full_n = len(set(full_ids))

    # Strictly more, with a real margin -- not a 1-vs-2 fluke. On these fixtures the
    # observed counts are grey=4, full=105; require a comfortable lower bound so the
    # assertion is robust but not brittle to incidental hash collisions.
    assert full_n > grey_n, f"{name}: full ({full_n}) should exceed grey ({grey_n})"
    assert full_n >= grey_n + 10, (
        f"{name}: margin too small -- grey={grey_n}, full={full_n}"
    )


@pytest.mark.parametrize("name,path", FIXTURES, ids=[f[0] for f in FIXTURES])
def test_grey_is_species_blind_full_is_not(name: str, path: Path) -> None:
    """Relabeling species leaves the GREY ID multiset unchanged but changes FULL.

    Swap the two element symbols on the (fixed) positions. Grey coloring forces all
    types to a single colour, so it must be invariant under any relabeling. Full
    coloring threads real types into the graph hash, so swapping the (asymmetric)
    species reassigns vertex colours and the ID multiset must change.
    """
    system, rnei, rcut = _build_neighbors(path)
    orig = list(system.types)
    elems = sorted(set(orig))
    assert len(elems) == 2, f"{name}: expected exactly 2 species, got {elems}"
    a, b = elems
    swapped = [b if t == a else (a if t == b else t) for t in orig]

    # Positions are held fixed; only the labels move -> neighbor lists are identical.
    grey_orig = Counter(_graph_classes(rnei, rcut, orig, "grey"))
    grey_swap = Counter(_graph_classes(rnei, rcut, swapped, "grey"))
    full_orig = Counter(_graph_classes(rnei, rcut, orig, "full"))
    full_swap = Counter(_graph_classes(rnei, rcut, swapped, "full"))

    # Grey is species-blind: identical multiset of environment IDs.
    assert grey_orig == grey_swap, (
        f"{name}: grey IDs changed under species relabel -- not species-blind"
    )
    # Full is species-resolved: the multiset must change under the swap.
    assert full_orig != full_swap, (
        f"{name}: full IDs unchanged under species relabel -- coloring not applied"
    )
