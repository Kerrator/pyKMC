"""Same-hop double counting: geometric collapse of duplicate connectivity rows.

The explorer emits one connectivity row per (generic event, central atom, symmetry),
so the same physical hop catalogued under two central atoms produces two rows with the
same ``(state, state_connexion)``. Both rows are summed into the FPTA generator and
into ``k_tot``, doubling that channel's rate. Two *different* saddles into the same
product state are parallel channels whose rates must add, so the collapse has to be
geometric.
"""

import logging
from types import SimpleNamespace

import numpy as np
import numpy.testing as npt
import pytest

from pykmc import NeighborsList, System
from pykmc.basins.basin import BasinsGenericEvents, StateData
from pykmc.basins.connectivity import BasinStatesConnectivity
from pykmc.basins.selection import FPTASelector

A0 = 3.52  # Ni lattice parameter (A)
RCUT = 4.2  # first + second neighbour shell
THR = 0.4  # stands in for config.psr.matching_score_thr

C1 = 0  # central atom of the first row of a duplicated pair
C2 = 1  # nearest neighbour of C1: its rcut shell also contains C1
C3 = 5  # central atom of an unrelated row
C4 = 9  # central atom of another unrelated row
MOVER = C1  # the hopping atom, inside the rcut shell of both C1 and C2


def _fcc_system(n_cells: int = 3) -> System:
    """Perfect fcc cell, ``n_cells``**3 conventional cells of pure Ni."""
    basis = np.array([[0.0, 0.0, 0.0], [0.0, 0.5, 0.5], [0.5, 0.0, 0.5], [0.5, 0.5, 0.0]])
    origins = np.array(
        [[i, j, k] for i in range(n_cells) for j in range(n_cells) for k in range(n_cells)],
        dtype=float,
    )
    positions = ((origins[:, None, :] + basis[None, :, :]).reshape(-1, 3)) * A0
    return System(
        types=np.ones(len(positions), dtype=int),
        positions=positions,
        cell=np.diag([n_cells * A0] * 3),
        pbc=np.array([True, True, True]),
        index=np.arange(len(positions)),
    )


def _basin(rows: list[tuple], n_states: int = 1) -> BasinsGenericEvents:
    """Basin with a real fcc state geometry and the given connectivity rows.

    rows are (state, state_connexion, event, central_atom, sym, transient, dE_f, k_f).
    """
    system = _fcc_system()
    neighbors_list = NeighborsList(system=system, rnei=2.6, rcut=RCUT)
    basin = object.__new__(BasinsGenericEvents)  # no __init__: no manager/engine needed
    basin.config = SimpleNamespace(psr=SimpleNamespace(matching_score_thr=THR))
    basin.reference_table = None  # no PSR fallback: every row's saddle is stored below
    basin.states = {
        idx: StateData(system=system, environment=None, neighbors_list=neighbors_list, transient=True)
        for idx in range(n_states)
    }
    basin.absorbing_saddle_positions = {}
    basin._creating_row = {}
    table = BasinStatesConnectivity()
    for state, target, event, central_atom, sym, transient, dE, k in rows:
        table.add_connectivity(state, target, event, central_atom, sym, transient, dE, k, dE, k)
    basin.connectivity_table = table
    return basin


def _store_saddle(
    basin: BasinsGenericEvents, row: tuple, mover_shift: np.ndarray, mover: "int | None" = None
) -> None:
    """Store a saddle for one row: the state geometry with the hopping atom displaced."""
    state, target, event, central_atom, sym = row[0], row[1], row[2], row[3], row[4]
    state_data = basin.states[state]
    neighbors = state_data.neighbors_list.get_neighbors("rcut", central_atom)
    saddle = np.array(state_data.system.positions[neighbors], copy=True)
    saddle[list(neighbors).index(central_atom if mover is None else mover)] += mover_shift
    key = basin._saddle_key(state, target, central_atom, event, sym)
    basin.absorbing_saddle_positions[key] = saddle


def _generator(basin: BasinsGenericEvents) -> np.ndarray:
    """Build the generator matrix the FPTA selector derives from the basin's table."""
    selector = FPTASelector()
    selector.build_absorbing_matrix_from_connectivity(basin.connectivity_table)
    return selector.M_abs


def _k_tot(basin: BasinsGenericEvents) -> float:
    """k_tot exactly as execute() snapshots it."""
    df = basin.connectivity_table.df
    return float(df.loc[df["transient"] == False, "k_forward"].sum())  # noqa: E712


# (state, target, event, central_atom, sym, transient, dE_forward, k_forward)
ROW_A = (0, 1, 14, C1, 0, False, 0.791, 5.0)
ROW_B = (0, 1, 16, C2, 0, False, 0.792, 5.0)  # same hop, catalogued on a neighbour
ROW_C = (0, 2, 20, C3, 0, False, 0.850, 3.0)  # a different exit


class TestDuplicateRowCollapse:
    """collapse_duplicate_rows(): geometric same-hop dedup of the connectivity table."""

    def test_same_hop_counted_once_in_M_and_k_tot(self, test_logger: logging.Logger) -> None:
        """Two rows carrying the same saddle contribute their rate once, to M and to k_tot."""
        basin = _basin([ROW_A, ROW_B, ROW_C])
        shift = np.array([0.6, 0.0, 0.0])
        _store_saddle(basin, ROW_A, shift, mover=MOVER)
        _store_saddle(basin, ROW_B, shift, mover=MOVER)
        _store_saddle(basin, ROW_C, np.array([0.0, 0.0, 0.6]))

        npt.assert_allclose(_generator(basin)[1, 0], -10.0)  # doubled before the collapse
        npt.assert_allclose(_k_tot(basin), 13.0)

        assert basin.collapse_duplicate_rows() == 1
        assert len(basin.connectivity_table.df) == 2
        npt.assert_allclose(_generator(basin)[1, 0], -5.0)
        npt.assert_allclose(_generator(basin)[2, 0], -3.0)  # the unique exit is untouched
        npt.assert_allclose(_k_tot(basin), 8.0)

    def test_different_saddles_into_same_state_add(self, test_logger: logging.Logger) -> None:
        """Two distinct saddles into one product state are parallel channels: rates add."""
        basin = _basin([ROW_A, ROW_B, ROW_C])
        _store_saddle(basin, ROW_A, np.array([0.6, 0.0, 0.0]), mover=MOVER)
        _store_saddle(basin, ROW_B, np.array([0.0, 0.6, 0.0]), mover=MOVER)  # 0.85 A apart > THR
        _store_saddle(basin, ROW_C, np.array([0.0, 0.0, 0.6]))

        assert basin.collapse_duplicate_rows() == 0
        npt.assert_allclose(_generator(basin)[1, 0], -10.0)
        npt.assert_allclose(_k_tot(basin), 13.0)

    def test_table_without_collisions_is_untouched(self, test_logger: logging.Logger) -> None:
        """A pure-Ni table with one row per (state, exit) keeps a bit-identical generator."""
        rows = [
            (0, 1, 14, C1, 0, False, 0.79, 5.0),
            (0, 2, 20, C3, 0, False, 0.85, 3.0),
            (0, 3, 21, C4, 0, False, 0.88, 1.0),
        ]
        basin = _basin(rows)
        for row, shift in zip(rows, [[0.6, 0, 0], [0, 0.6, 0], [0, 0, 0.6]], strict=True):
            _store_saddle(basin, row, np.array(shift, dtype=float))
        before = _generator(basin).copy()
        df_before = basin.connectivity_table.df.copy()

        assert basin.collapse_duplicate_rows() == 0
        assert np.array_equal(_generator(basin), before)
        assert basin.connectivity_table.df.equals(df_before)

    def test_transient_duplicate_is_collapsed(self, test_logger: logging.Logger) -> None:
        """A transient -> transient hop catalogued twice is collapsed too, not only exits."""
        row_t1 = (0, 1, 14, C1, 0, True, 0.20, 100.0)
        row_t2 = (0, 1, 16, C2, 0, True, 0.20, 100.0)  # same hop
        rows = [row_t1, row_t2, (1, 0, 30, C3, 0, True, 0.20, 90.0), (0, 2, 20, C4, 0, False, 0.85, 1.0)]
        basin = _basin(rows, n_states=2)
        shift = np.array([0.6, 0.0, 0.0])
        _store_saddle(basin, row_t1, shift, mover=MOVER)
        _store_saddle(basin, row_t2, shift, mover=MOVER)
        _store_saddle(basin, rows[2], np.array([0.0, 0.6, 0.0]))
        _store_saddle(basin, rows[3], np.array([0.0, 0.0, 0.6]))

        npt.assert_allclose(_generator(basin)[1, 0], -200.0)
        assert basin.collapse_duplicate_rows() == 1
        npt.assert_allclose(_generator(basin)[1, 0], -100.0)
        npt.assert_allclose(_generator(basin)[0, 1], -90.0)  # the reverse hop is untouched

    def test_row_without_geometry_is_kept(self, test_logger: logging.Logger) -> None:
        """No comparable saddle means not-comparable: keep both rows (never a blind collapse)."""
        basin = _basin([ROW_A, ROW_B, ROW_C])
        _store_saddle(basin, ROW_A, np.array([0.6, 0.0, 0.0]), mover=MOVER)
        _store_saddle(basin, ROW_C, np.array([0.0, 0.0, 0.6]))
        # ROW_B has no stored saddle and reference_table is None -> no PSR fallback.

        assert basin.collapse_duplicate_rows() == 0
        npt.assert_allclose(_generator(basin)[1, 0], -10.0)

    def test_creating_row_survives_the_collapse(self, test_logger: logging.Logger) -> None:
        """The row that reconstructed the target's geometry is the one kept, so _select_exit_row still pairs row and saddle."""
        basin = _basin([ROW_A, ROW_B, ROW_C])
        shift = np.array([0.6, 0.0, 0.0])
        _store_saddle(basin, ROW_A, shift, mover=MOVER)
        _store_saddle(basin, ROW_B, shift, mover=MOVER)
        _store_saddle(basin, ROW_C, np.array([0.0, 0.0, 0.6]))
        basin._creating_row = {1: (0, C2, 16, 0)}  # state 1 was reconstructed by ROW_B

        assert basin.collapse_duplicate_rows() == 1
        kept = basin.connectivity_table.df[basin.connectivity_table.df["state_connexion"] == 1]
        assert len(kept) == 1
        assert int(kept.iloc[0]["central_atom"]) == C2
        res = basin._select_exit_row(1)
        assert res.is_ok()
        from_state, event_idx, central_atom, sym, _dE, _saddle = res.ok_value()
        assert (from_state, event_idx, central_atom, sym) == (0, 16, C2, 0)


@pytest.mark.parametrize("central_atom", [C1, C2])
def test_mover_is_inside_both_shells(central_atom: int) -> None:
    """Guard on the fixture: the two central atoms share the hopping atom."""
    system = _fcc_system()
    neighbors_list = NeighborsList(system=system, rnei=2.6, rcut=RCUT)
    assert MOVER in neighbors_list.get_neighbors("rcut", central_atom)
