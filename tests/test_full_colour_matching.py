"""Species colouring must reach every IRA/SOFI call, not just the PSR match.

``atom_coloring_mode = full`` was honoured by the PSR match and by graph
hashing, but three IRA/SOFI call sites were hard-wired species-blind: the two
saddle-dedup calls in :mod:`pykmc.event_table` (``typ = nat * ['X']``) and the
SOFI call in :mod:`pykmc.symmetries` (``typ = nat * [1]``).

Colour-blindness is *permissive* -- it never rejects a match, it accepts a
wrong one -- so the failure mode is a silently wrong catalogue, never a crash.
The central case is exercised below: one atom sitting between two vacancies
that a C2 rotation maps onto each other. The two hops are the same shape, so a
species-blind saddle match calls them one event; a Cr solute off the rotation
axis makes them chemically distinct, with different barriers. Both hops share a
single ``event_id`` by construction (same initial state, same central atom), so
the saddle match is the only thing standing between them.
"""

from unittest.mock import Mock

import numpy as np
import pandas as pd
import pytest

from pykmc.atomic_environment import AtomicEnvironment, match_types
from pykmc.basins.basin import BasinsGenericEvents
from pykmc.event_table import ReferenceEventTable
from pykmc.neighbors_list import NeighborsList
from pykmc.point_set_registration import PointSetRegistration
from pykmc.result import ErrorType
from pykmc.symmetries import unique_symmetries
from pykmc.system import System

A = 3.52  # Ni lattice parameter
REPEAT = 5
RNEI, RCUT = 2.8, 6.5
KMAX, SYM_THR, SCORE_THR = 1.8, 0.01, 0.1


def _config(mode: str) -> Mock:
    cfg = Mock()
    cfg.rateconstant.style = "constant"
    cfg.rateconstant.T = 300.0
    cfg.rateconstant.k0 = 10.0
    cfg.atomicenvironment.rnei = RNEI
    cfg.atomicenvironment.rcut = RCUT
    cfg.atomicenvironment.atom_coloring_mode = mode
    cfg.ira.kmax_factor = KMAX
    cfg.ira.sym_thr = SYM_THR
    cfg.psr.style = "ira"
    cfg.psr.matching_score_thr = SCORE_THR
    cfg.control.reference_table = None
    return cfg


def _fcc(repeat: int = REPEAT) -> "tuple[np.ndarray, np.ndarray]":
    basis = np.array([[0, 0, 0], [0.5, 0.5, 0], [0.5, 0, 0.5], [0, 0.5, 0.5]]) * A
    pos = [
        basis + np.array([i, j, k]) * A
        for i in range(repeat)
        for j in range(repeat)
        for k in range(repeat)
    ]
    return np.concatenate(pos), np.diag([A * repeat] * 3)


def _site(positions: np.ndarray, target: np.ndarray, box: float) -> int:
    """Index of the lattice site at `target` (minimum image)."""
    d = positions - target
    d -= box * np.round(d / box)
    idx = int(np.argmin(np.linalg.norm(d, axis=1)))
    assert np.linalg.norm(d[idx]) < 1e-8, "no lattice site at the requested position"
    return idx


def _two_vacancy_hops(mode: str) -> "tuple[pd.Series, pd.Series]":
    """Build the two competing hops of the docstring, as event rows.

    Atom ``m`` has a vacancy on either side along [110]. It can hop into
    either; the C2 rotation about [001] through ``m`` maps one hop onto the
    other, so the two saddle geometries are identical. A Cr solute at
    ``m + (a/2)(0, 1, 1)`` is *not* fixed by that rotation, so the two hops
    differ chemically -- and in a real run, in barrier height.
    """
    pos, cell = _fcc()
    box = cell[0][0]
    m = int(np.argmin(np.linalg.norm(pos - A * REPEAT / 2, axis=1)))
    v1 = _site(pos, pos[m] + A / 2 * np.array([1, 1, 0]), box)
    v2 = _site(pos, pos[m] + A / 2 * np.array([-1, -1, 0]), box)
    cr = _site(pos, pos[m] + A / 2 * np.array([0, 1, 1]), box)

    keep = np.array([i for i in range(len(pos)) if i not in (v1, v2)])
    remap = {old: new for new, old in enumerate(keep)}
    min1 = pos[keep]
    mm = remap[m]
    types = ["Ni"] * len(min1)
    types[remap[cr]] = "Cr"

    table = ReferenceEventTable(_config(mode))
    rows = []
    for vacancy, barrier in ((pos[v1], 0.60), (pos[v2], 0.65)):
        step = vacancy - min1[mm]
        step -= box * np.round(step / box)
        saddle, final = min1.copy(), min1.copy()
        saddle[mm] = min1[mm] + 0.5 * step
        final[mm] = min1[mm] + step
        forward, _ = table._build_event_series(
            min1_positions=min1.copy(),
            saddle_positions=saddle,
            min2_positions=final,
            index_move=np.int64(mm),
            dE_forward=barrier,
            dE_backward=0.60,
            cell=cell,
            types=types,
        )
        rows.append(forward)
    return rows[0], rows[1]


def test_the_two_hops_are_indistinguishable_before_the_saddle_match() -> None:
    """Both hops share an event_id and a barrier window -> dedup rests on PSR.

    Guards the premise of the test below: if these ever stopped colliding, the
    dedup test would pass for the wrong reason.
    """
    hop_a, hop_b = _two_vacancy_hops("full")
    assert hop_a["event_id"] == hop_b["event_id"]
    assert abs(hop_a["energy_barrier"] - hop_b["energy_barrier"]) < 0.25


def test_full_colour_dedup_keeps_chemically_distinct_hops() -> None:
    """RED before the fix: the second hop was discarded as a duplicate."""
    hop_a, hop_b = _two_vacancy_hops("full")
    table = ReferenceEventTable(_config("full"))
    table.add(hop_a.to_frame().T)

    assert table.is_new_event(hop_b), (
        "the two hops differ only in where the Cr solute sits relative to the "
        "hop direction; a species-blind saddle match collapses them into one "
        "reference event and the second barrier is lost"
    )


def test_grey_still_collapses_the_same_two_hops() -> None:
    """Grey is unchanged: species-blind by request, so the hops are one event.

    Also shows the fix is colour-driven -- the geometry alone still matches.
    """
    hop_a, hop_b = _two_vacancy_hops("grey")
    table = ReferenceEventTable(_config("grey"))
    table.add(hop_a.to_frame().T)

    assert not table.is_new_event(hop_b)


def test_full_colour_dedup_still_rejects_a_true_duplicate() -> None:
    """The fix must not turn every event into a new one."""
    hop_a, _ = _two_vacancy_hops("full")
    table = ReferenceEventTable(_config("full"))
    table.add(hop_a.to_frame().T)

    assert not table.is_new_event(hop_a)


# --- symmetries.py: SOFI was called with a hardcoded uniform type list -------


def _nn_cluster() -> "tuple[np.ndarray, list[str]]":
    """Central atom + its 12 FCC nearest neighbours, one of them Cr."""
    offsets = [
        (i, j, 0) for i in (1, -1) for j in (1, -1)
    ] + [
        (i, 0, k) for i in (1, -1) for k in (1, -1)
    ] + [
        (0, j, k) for j in (1, -1) for k in (1, -1)
    ]
    positions = np.array([[0.0, 0.0, 0.0]] + [np.array(o) * A / 2 for o in offsets])
    types = ["Ni"] * len(positions)
    types[1] = "Cr"
    return positions, types


def test_unique_symmetries_never_maps_one_species_onto_another() -> None:
    """RED before the fix: symmetry copies could put Ni where Cr was."""
    positions, types = _nn_cluster()
    final = positions.copy()
    final[0] += np.array([0.1, 0.1, 0.0])  # the moving atom
    labels = np.array(types)

    _, perms = unique_symmetries(positions, final, SYM_THR, types=types)
    for perm in perms:
        assert list(labels[perm]) == types

    _, grey_perms = unique_symmetries(positions, final, SYM_THR)
    assert any(list(labels[perm]) != types for perm in grey_perms), (
        "the species-blind call is expected to produce species-breaking "
        "operations -- that is the defect being fixed"
    )


def test_unique_symmetries_without_types_is_unchanged() -> None:
    """Grey path is bit-identical: no types -> the old uniform type list."""
    positions, types = _nn_cluster()
    final = positions.copy()
    final[0] += np.array([0.1, 0.1, 0.0])

    mats, perms = unique_symmetries(positions, final, SYM_THR)
    explicit, explicit_perms = unique_symmetries(
        positions, final, SYM_THR, types=None
    )
    assert np.array_equal(mats, explicit)
    assert np.array_equal(perms, explicit_perms)
    assert len(mats) >= len(unique_symmetries(positions, final, SYM_THR, types=types)[0])


def test_unique_symmetries_rejects_mismatched_types() -> None:
    """Types that do not belong to these positions are a caller bug."""
    positions, types = _nn_cluster()
    with pytest.raises(ValueError, match="element types"):
        unique_symmetries(positions, positions, SYM_THR, types=types[:-1])


# --- the typ2 = typ1 fallback ------------------------------------------------


def _psr(mode: str, stored_types: "list[str] | None | object") -> PointSetRegistration:
    """Build a PSR against a 13-atom reference event with the given types."""
    pos, cell = _fcc(repeat=3)
    system = System()
    system.positions = pos
    system.cell = cell
    system.types = ["Ni"] * len(pos)
    system.types[7] = "Cr"

    reference, _ = _nn_cluster()
    row = {"initial_positions": reference}
    if stored_types is not ...:
        row["initial_types"] = stored_types

    return PointSetRegistration(
        _config(mode),
        system,
        pd.Series(row),
        NeighborsList(system, RNEI, RCUT),
        central_atom_index=0,
    )


def test_psr_refuses_a_reference_event_with_no_stored_types() -> None:
    """RED before the fix: typ2 fell back to typ1 -- the target's own species.

    That labels the reference cluster with the *target's* elements in the
    target's index order, and at the target's length. IRA accepts it silently.
    """
    result = _psr("full", None).match()

    assert not result.is_ok()
    assert result.err_value().type == ErrorType.PSR_REFERENCE_TYPES_MISSING


def test_psr_refuses_a_reference_event_with_no_types_column() -> None:
    """An older reference table has no initial_types column at all."""
    result = _psr("full", ...).match()

    assert not result.is_ok()
    assert result.err_value().type == ErrorType.PSR_REFERENCE_TYPES_MISSING


def test_psr_refuses_wrong_length_reference_types() -> None:
    """Types that do not size the reference cluster cannot label it."""
    result = _psr("full", ["Ni"] * 4).match()

    assert not result.is_ok()
    assert result.err_value().type == ErrorType.PSR_REFERENCE_TYPES_MISSING


def test_psr_in_grey_mode_ignores_missing_reference_types() -> None:
    """Grey never needed the types, and must not start failing without them."""
    result = _psr("grey", None).match()

    assert result.is_ok() or result.err_value().type in (
        ErrorType.PSR_NO_MATCH_FOUND,
        ErrorType.PSR_MATCHING_SCORE_ABOVE_ACCEPTANCE_THRESHOLD,
    )


# --- the shared helper -------------------------------------------------------


# --- environment IDs: the graph styles must all colour alike -----------------


def _environment_ids(style: str, mode: str, solute_site: int) -> "list[str]":
    """Environment IDs for an FCC box with one Cr, under the given style/mode."""
    positions, cell = _fcc(repeat=3)
    system = System()
    system.positions = positions
    system.cell = cell
    types = ["Ni"] * len(positions)
    types[solute_site] = "Cr"
    system.types = types
    neighbors = NeighborsList(system, RNEI, RCUT)

    env = AtomicEnvironment(
        style,
        neighbors.neighbors_list["rnei"],
        neighbors.neighbors_list["rcut"],
        0,
        # threshold above the FCC coordination number -> every atom is
        # "noncrystal", so every atom takes the graph-hash path under test
        coordination_threshold=13,
        types=types if mode == "full" else None,
    )
    return env.atomic_environment_list


def test_coordination_graph_environment_ids_are_coloured_in_full_mode() -> None:
    """RED before the fix: compute_coordinationgraph dropped `types`.

    This is the style the NiCr production inputs actually run
    (``style = coordination/graph`` + ``atom_coloring_mode = full``), and the
    environment ID is the key the whole event catalogue is built on.
    """
    here = _environment_ids("coordination/graph", "full", solute_site=0)
    moved = _environment_ids("coordination/graph", "full", solute_site=5)

    assert here != moved, (
        "moving the solute must change some environment ID under full colour; "
        "identical IDs mean the graph hash ignored the species"
    )


def test_coordination_graph_environment_ids_are_grey_when_asked() -> None:
    """Grey is unchanged: species are invisible to the environment ID."""
    here = _environment_ids("coordination/graph", "grey", solute_site=0)
    moved = _environment_ids("coordination/graph", "grey", solute_site=5)

    assert here == moved


@pytest.mark.parametrize("style", ["graph", "cna/graph", "coordination/graph", "diamond/graph"])
def test_every_graph_style_forwards_its_types(style: str, monkeypatch: "pytest.MonkeyPatch") -> None:
    """Every graph-bearing style must hand `types` to the hash, or none may.

    Locks all four siblings together: two of them silently omitted the kwarg.
    """
    import pykmc.atomic_environment as ae

    seen: "list[object]" = []

    def _spy(neighbors_list, environment_list, atom_idx=None, types=None):  # noqa: ANN001, ANN202
        seen.append(types)
        idx = range(len(neighbors_list)) if atom_idx is None else atom_idx
        return ["hash"] * len(list(idx))

    monkeypatch.setattr(ae, "graph", _spy)
    types = ["Ni", "Cr", "Ni", "Ni"]
    neighbors = [[1, 2, 3], [0, 2, 3], [0, 1, 3], [0, 1, 2]]

    AtomicEnvironment(style, neighbors, neighbors, 0, types=types, coordination_threshold=13)

    assert seen, "the style did not reach graph() at all"
    assert all(t == types for t in seen), (
        "style {} called graph() without its types -> species-blind IDs".format(style)
    )


# --- basin state identity ----------------------------------------------------


def _basin() -> BasinsGenericEvents:
    """Build a BasinsGenericEvents shell; the identity helpers need only config."""
    obj = BasinsGenericEvents.__new__(BasinsGenericEvents)
    obj.config = _config("full")
    return obj


def _swapped_pair() -> "tuple[np.ndarray, list[str], list[str]]":
    """One site set, two species assignments -- a Ni hop vs a Cr hop.

    Same occupied positions, different atom on the site that moved.
    """
    positions, _ = _fcc(repeat=3)
    types_a = ["Ni"] * len(positions)
    types_a[4] = "Cr"
    types_b = ["Ni"] * len(positions)
    types_b[9] = "Cr"
    return positions, types_a, types_b


def test_basin_state_identity_is_species_aware_in_full_mode() -> None:
    """RED before the fix: two chemically distinct states were one state."""
    positions, types_a, types_b = _swapped_pair()
    cell = np.diag([A * 3] * 3)

    assert not _basin().are_structures_equivalent(
        positions, positions, cell, types1=types_a, types2=types_b
    )


def test_basin_state_identity_still_matches_a_true_repeat() -> None:
    """Same positions and same species is still the same state."""
    positions, types_a, _ = _swapped_pair()
    cell = np.diag([A * 3] * 3)

    assert _basin().are_structures_equivalent(
        positions, positions, cell, types1=types_a, types2=types_a
    )


def test_basin_state_identity_without_types_is_unchanged() -> None:
    """Grey path is bit-identical: no types -> positions-only comparison."""
    positions, types_a, types_b = _swapped_pair()
    cell = np.diag([A * 3] * 3)
    basin = _basin()

    assert basin.are_structures_equivalent(positions, positions, cell)
    assert basin.are_structures_equivalent(
        positions, positions, cell, types1=None, types2=types_b
    )


def test_species_agree_follows_geometry_not_index() -> None:
    """Species are compared on the MATCHED pairs, not index-wise.

    Atom i of one state sits where atom matched[i] of the other sits; comparing
    ``types1[i]`` to ``types2[i]`` would be the wrong question.
    """
    basin = _basin()
    matched = np.array([1, 0, 2])

    assert basin._species_agree(["Ni", "Cr", "Ni"], ["Cr", "Ni", "Ni"], matched)
    assert not basin._species_agree(["Ni", "Cr", "Ni"], ["Ni", "Cr", "Ni"], matched)


def test_match_types_grey_sizes_labels_to_its_own_structure() -> None:
    """Grey labels are uniform, but still one per atom of THIS structure."""
    assert match_types("grey", None, 3) == ["X"] * 3
    # grey ignores the types it is handed, but still sizes per structure
    assert match_types("grey", ["Ni", "Cr"], 5) == ["X"] * 5


def test_match_types_full_returns_the_real_species() -> None:
    """Full colour hands IRA the element symbols unchanged."""
    assert match_types("full", ["Ni", "Cr"], 2) == ["Ni", "Cr"]


@pytest.mark.parametrize(
    ("types", "nat"), [(None, 2), (["Ni"], 2), (["Ni", "Cr", "Ni"], 2)]
)
def test_match_types_full_refuses_to_guess(types: "list[str] | None", nat: int) -> None:
    """Missing or wrong-length types are an error, never a fabricated label."""
    with pytest.raises(ValueError):
        match_types("full", types, nat)
