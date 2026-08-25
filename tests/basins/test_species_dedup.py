"""Species-aware basin state deduplication.

``are_structures_equivalent`` compared positions only, so in an alloy two states that
differ *only* by a species swap between two sites merged into one (review
2026-08-25 § 4b): for a Cr-vacancy flicker X <-> Y, a Ni common neighbour S with both
hops catalogued (S->X from state 0, S->Y from state 1) yields two states whose position
sets agree to << tol but whose occupancies of X and Y are swapped. One product state
then silently absorbed both channels' probability.

The tests below pin the new behaviour and, just as importantly, pin that pure
single-species deduplication is *bit-identical* to the pre-fix decision: the oracle in
``_oracle_equivalent`` is the verbatim ``c4b8aea`` body.
"""

from types import SimpleNamespace

import numpy as np
import pytest
from scipy.spatial import cKDTree

from pykmc import System
from pykmc.basins import BasinStatesConnectivity, BasinsGenericEvents
from pykmc.basins import fingerprinting
from pykmc.basins.basin import StateData

A0 = 3.52  # Ni lattice parameter (A)
N_CELLS = 3


# ──────────────────────────────────────────────────────────────────────────
# The pre-fix oracle: verbatim body of are_structures_equivalent at c4b8aea
# ──────────────────────────────────────────────────────────────────────────
def _oracle_equivalent(pos1: np.ndarray, pos2: np.ndarray, cell: np.ndarray,
                       tol: float = 0.3) -> bool:
    """Positions-only equivalence exactly as it stood at ``c4b8aea``."""
    if len(pos1) != len(pos2):
        return False

    box = np.diag(cell).tolist()
    wrapped1 = np.mod(pos1, np.diag(cell))
    wrapped2 = np.mod(pos2, np.diag(cell))
    tree2 = cKDTree(wrapped2, boxsize=box)
    distances, _ = tree2.query(wrapped1, k=1)

    return np.max(distances) < tol


# ──────────────────────────────────────────────────────────────────────────
# Geometry helpers
# ──────────────────────────────────────────────────────────────────────────
def _fcc_sites(n_cells: int = N_CELLS) -> np.ndarray:
    """Conventional fcc lattice sites, ``n_cells``**3 cells."""
    basis = np.array([[0.0, 0.0, 0.0], [0.0, 0.5, 0.5], [0.5, 0.0, 0.5], [0.5, 0.5, 0.0]])
    origins = np.array(
        [[i, j, k] for i in range(n_cells) for j in range(n_cells) for k in range(n_cells)],
        dtype=float,
    )
    return ((origins[:, None, :] + basis[None, :, :]).reshape(-1, 3)) * A0


def _cell(n_cells: int = N_CELLS) -> np.ndarray:
    return np.diag([n_cells * A0] * 3)


def _mic_distances(sites: np.ndarray, idx: int, box: np.ndarray) -> np.ndarray:
    """Minimum-image distances from ``sites[idx]`` to every site."""
    diffs = sites - sites[idx]
    diffs -= np.round(diffs / box) * box
    return np.linalg.norm(diffs, axis=1)


def _flicker_triplet(sites: np.ndarray, box: np.ndarray) -> tuple[int, int, int]:
    """Return (X, Y, S): X-Y nearest neighbours, S a common nearest neighbour."""
    nn = A0 / np.sqrt(2.0)
    i_x = 0
    d_x = _mic_distances(sites, i_x, box)
    neighbours_x = np.where(np.abs(d_x - nn) < 1e-6)[0]
    i_y = int(neighbours_x[0])
    d_y = _mic_distances(sites, i_y, box)
    neighbours_y = np.where(np.abs(d_y - nn) < 1e-6)[0]
    common = sorted(set(neighbours_x.tolist()) & set(neighbours_y.tolist()))
    assert common, "fcc nearest-neighbour pairs always share common neighbours"
    return i_x, i_y, int(common[0])


def _make_system(positions: np.ndarray, types: np.ndarray | list[str],
                 cell: np.ndarray | None = None) -> System:
    cell = _cell() if cell is None else cell
    return System(
        positions=np.asarray(positions, dtype=float),
        types=np.asarray(types),
        cell=cell,
        pbc=np.array([True, True, True]),
        index=np.arange(len(positions)),
    )


def _flicker_states(jitter: float = 0.02, seed: int = 7) -> tuple[System, System]:
    """Build the review's construction: two states differing only by a Ni/Cr swap.

    Sites X and Y are the Cr-vacancy flicker pair, S their common Ni neighbour.
    State A: S has hopped into X, so Ni sits at X and Cr at Y (vacancy at S).
    State B: S has hopped into Y, so Ni sits at Y and Cr at X (vacancy at S).
    Atom identities (and hence the ``types`` array) are shared: only positions move.
    """
    sites = _fcc_sites()
    cell = _cell()
    box = np.diag(cell)
    i_x, i_y, i_s = _flicker_triplet(sites, box)

    keep = [i for i in range(len(sites)) if i != i_s]  # vacancy at S
    base = sites[keep]
    slot_x = keep.index(i_x)  # atom "s", the Ni that hopped
    slot_y = keep.index(i_y)  # atom "c", the Cr

    types = np.array(["Ni"] * len(keep))
    types[slot_y] = "Cr"

    rng = np.random.default_rng(seed)
    pos_a = base + rng.uniform(-jitter, jitter, base.shape)
    # State B: the two atoms exchange sites (Cr now at X, the Ni hopper at Y)
    swapped = base.copy()
    swapped[[slot_x, slot_y]] = swapped[[slot_y, slot_x]]
    pos_b = swapped + rng.uniform(-jitter, jitter, base.shape)

    return _make_system(pos_a, types, cell), _make_system(pos_b, types, cell)


# ──────────────────────────────────────────────────────────────────────────
# Basin instance without an engine/manager (cf. tests/basins/test_wavefront_units.py)
# ──────────────────────────────────────────────────────────────────────────
def _config(fingerprint_mode: str = "auto") -> SimpleNamespace:
    return SimpleNamespace(
        basin=SimpleNamespace(fingerprint_mode=fingerprint_mode,
                              fingerprint_coordination_thr=None, fingerprint_tolerance=1.0,
                              strategy="wavefront", n_workers=2,
                              max_states=None, max_total_states=None,
                              max_basin_walltime_s=None, max_frontier_size=None,
                              max_failed_fraction=0.2),
        atomicenvironment=SimpleNamespace(style="graph", coordination_threshold=None, rnei=3.0),
    )


def _basin(config: SimpleNamespace | None = None) -> BasinsGenericEvents:
    """BasinsGenericEvents without __init__ — no manager, no LAMMPS, no MPI."""
    basin = BasinsGenericEvents.__new__(BasinsGenericEvents)
    basin.config = _config() if config is None else config
    basin.states = {}
    basin._state_fingerprints = {}
    basin.connectivity_table = BasinStatesConnectivity()
    return basin


def _register(basin: BasinsGenericEvents, index: int, system: System) -> None:
    basin.states[index] = StateData(system=system, environment=None, neighbors_list=None)
    fp = fingerprinting.compute_fingerprint(basin.config, system.positions, system.cell, system.pbc)
    if fp is not None:
        basin._state_fingerprints[index] = fp


# ──────────────────────────────────────────────────────────────────────────
# (a) the review's constructible failure
# ──────────────────────────────────────────────────────────────────────────
class TestFlickerSpeciesSwap:
    """The review's constructible failure (§ 4b): a Ni/Cr swap between X and Y."""

    def test_positions_only_oracle_merges_them(self) -> None:
        """The hazard: pre-fix, the two states are indistinguishable."""
        sys_a, sys_b = _flicker_states()
        # positions agree far inside tol = 0.3 A
        assert _oracle_equivalent(sys_a.positions, sys_b.positions, sys_a.cell)

    def test_species_swap_is_not_equivalent(self) -> None:
        """Species-aware comparison must keep the two flicker products apart."""
        basin = _basin()
        sys_a, sys_b = _flicker_states()
        assert not basin.are_structures_equivalent(
            sys_a.positions, sys_b.positions, cell=sys_a.cell,
            types1=sys_a.types, types2=sys_b.types)

    def test_species_swap_is_new_state(self) -> None:
        """Serial dedup must register the swapped product as a new state."""
        basin = _basin()
        sys_a, sys_b = _flicker_states()
        _register(basin, 0, sys_a)
        assert basin.is_new_state(sys_b) == -1

    def test_fingerprint_prefilter_does_not_separate_them(self) -> None:
        """The pre-filter stays species-blind: it must hand both to the full test."""
        basin = _basin()
        sys_a, sys_b = _flicker_states()
        fp_a = fingerprinting.compute_fingerprint(basin.config, sys_a.positions, sys_a.cell, sys_a.pbc)
        fp_b = fingerprinting.compute_fingerprint(basin.config, sys_b.positions, sys_b.cell, sys_b.pbc)
        tol = fingerprinting.fingerprint_tolerance(basin.config)
        assert np.max(np.abs(fp_a - fp_b)) <= tol


# ──────────────────────────────────────────────────────────────────────────
# (b) + (c) identity, and displacement either side of tol, at fixed species
# ──────────────────────────────────────────────────────────────────────────
class TestSameSpeciesGeometry:
    """Geometry-only decisions at fixed species: identity and the 0.3 A tolerance."""

    def test_identical_states_are_equivalent(self) -> None:
        """A state compared with its own copy is a duplicate."""
        basin = _basin()
        sys_a, _ = _flicker_states()
        sys_copy = _make_system(sys_a.positions.copy(), sys_a.types.copy(), sys_a.cell)
        assert basin.are_structures_equivalent(
            sys_a.positions, sys_copy.positions, cell=sys_a.cell,
            types1=sys_a.types, types2=sys_copy.types)
        _register(basin, 0, sys_a)
        assert basin.is_new_state(sys_copy) == 0

    @pytest.mark.parametrize("shift,expected", [(0.2, True), (0.4, False)])
    def test_single_atom_displacement(self, shift: float, expected: bool) -> None:
        """Same species everywhere: only the 0.3 A tolerance decides."""
        basin = _basin()
        sys_a, _ = _flicker_states(jitter=0.0)
        moved = sys_a.positions.copy()
        moved[3] += np.array([shift, 0.0, 0.0])
        assert basin.are_structures_equivalent(
            sys_a.positions, moved, cell=sys_a.cell,
            types1=sys_a.types, types2=sys_a.types) is expected

    def test_length_mismatch_is_not_equivalent(self) -> None:
        """The atom-count guard still short-circuits before any tree work."""
        basin = _basin()
        sys_a, _ = _flicker_states()
        assert not basin.are_structures_equivalent(
            sys_a.positions, sys_a.positions[:-1], cell=sys_a.cell,
            types1=sys_a.types, types2=sys_a.types[:-1])


# ──────────────────────────────────────────────────────────────────────────
# (d) pure single-species behaviour is bit-identical to the pre-fix decision
# ──────────────────────────────────────────────────────────────────────────
class TestSingleSpeciesOracleIdentity:
    """One species => the decision must equal the pre-fix positions-only one."""

    def test_matches_oracle_on_random_perturbations(self) -> None:
        """One species => the new decision must equal the old one on every input.

        This is the proof that the toolkit/basin_testing/BENCHMARKS.md pure-Ni
        reference values are unchanged: they cannot be re-run here (MPI, and the box
        is loaded), so structural identity of the decision stands in for a re-run.
        """
        basin = _basin()
        sites = _fcc_sites()
        cell = _cell()
        n = len(sites)
        types = np.array(["Ni"] * n)
        rng = np.random.default_rng(2026)

        n_true = 0
        n_false = 0
        for _ in range(60):
            scale = rng.choice([0.01, 0.05, 0.1, 0.2, 0.29, 0.31, 0.5, 1.0])
            pos1 = sites + rng.uniform(-0.01, 0.01, sites.shape)
            pos2 = sites.copy()
            # one atom straddling tol, plus a global rattle
            pos2[int(rng.integers(n))] += np.array([scale, 0.0, 0.0])
            pos2 += rng.uniform(-0.01, 0.01, sites.shape)
            new = basin.are_structures_equivalent(pos1, pos2, cell=cell,
                                                  types1=types, types2=types)
            old = _oracle_equivalent(pos1, pos2, cell)
            assert bool(new) == bool(old), f"diverged at scale {scale}"
            n_true += bool(old)
            n_false += not bool(old)
        assert n_true > 0 and n_false > 0, "the sweep must cover both verdicts"

    def test_matches_oracle_straddling_tol_exactly(self) -> None:
        """The `< tol` comparison itself is untouched (0.3 is excluded, as before)."""
        basin = _basin()
        sites = _fcc_sites()
        cell = _cell()
        types = np.array(["Ni"] * len(sites))
        for shift in (0.29, 0.2999999, 0.3, 0.3000001, 0.31):
            pos2 = sites.copy()
            pos2[0] += np.array([shift, 0.0, 0.0])
            new = basin.are_structures_equivalent(sites, pos2, cell=cell,
                                                  types1=types, types2=types)
            assert bool(new) == bool(_oracle_equivalent(sites, pos2, cell)), shift

    def test_types_none_falls_back_to_positions_only(self) -> None:
        """A System without species must still deduplicate as before."""
        basin = _basin()
        sites = _fcc_sites()
        cell = _cell()
        pos2 = sites.copy()
        pos2[0] += np.array([0.2, 0.0, 0.0])
        assert basin.are_structures_equivalent(sites, pos2, cell=cell,
                                               types1=None, types2=None)


# ──────────────────────────────────────────────────────────────────────────
# (e) batch/serial parity, including the within-batch cross-check
# ──────────────────────────────────────────────────────────────────────────
class TestBatchSerialParity:
    """is_new_state and is_new_state_batch must return the same verdict."""

    def test_batch_fast_path_is_species_aware(self) -> None:
        """The cached-kd-tree fast path in is_new_state_batch must check species too."""
        basin = _basin()
        sys_a, sys_b = _flicker_states()
        _register(basin, 0, sys_a)
        sys_a_copy = _make_system(sys_a.positions.copy(), sys_a.types.copy(), sys_a.cell)

        results = basin.is_new_state_batch({5: sys_b, 6: sys_a_copy})
        assert results[5] == -1  # species swap: a genuinely different state
        assert results[6] == 0   # same species, same positions: a duplicate

    def test_within_batch_cross_check_is_species_aware(self) -> None:
        """Two new states that are species swaps of each other must both be new."""
        basin = _basin()
        sys_a, sys_b = _flicker_states()

        results = basin.is_new_state_batch({5: sys_a, 6: sys_b})
        assert results[5] == -1
        assert results[6] == -1

        # control: a true duplicate inside the batch is still collapsed
        sys_a_copy = _make_system(sys_a.positions.copy(), sys_a.types.copy(), sys_a.cell)
        results = basin.is_new_state_batch({5: sys_a, 6: sys_a_copy})
        assert results[5] == -1
        assert results[6] == 5

    def test_serial_and_batch_agree(self) -> None:
        """Same (positions, types) inputs => same verdict through both paths."""
        sys_a, sys_b = _flicker_states()
        sys_a_copy = _make_system(sys_a.positions.copy(), sys_a.types.copy(), sys_a.cell)
        sys_b_copy = _make_system(sys_b.positions.copy(), sys_b.types.copy(), sys_b.cell)
        far = _make_system(sys_a.positions + 1.7, sys_a.types.copy(), sys_a.cell)

        for registered, probes in ((sys_a, (sys_a_copy, sys_b, far)),
                                   (sys_b, (sys_b_copy, sys_a, far))):
            serial = _basin()
            _register(serial, 0, registered)
            batch = _basin()
            _register(batch, 0, registered)
            batch_results = batch.is_new_state_batch(dict(enumerate(probes, start=5)))
            for offset, probe in enumerate(probes):
                assert serial.is_new_state(probe) == batch_results[5 + offset]

    def test_off_mode_parity(self) -> None:
        """With the pre-filter off, both paths must still separate the swap."""
        config = _config(fingerprint_mode="off")
        sys_a, sys_b = _flicker_states()

        serial = _basin(config)
        _register(serial, 0, sys_a)
        assert serial.is_new_state(sys_b) == -1

        batch = _basin(config)
        _register(batch, 0, sys_a)
        assert batch.is_new_state_batch({5: sys_b}) == {5: -1}
