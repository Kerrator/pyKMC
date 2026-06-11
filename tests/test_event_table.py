from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np
import pandas as pd

from pykmc.event_table import ActiveEventTable


def _minimal_config() -> SimpleNamespace:
    return SimpleNamespace(psr=SimpleNamespace(matching_score_thr=0.1))


def _migration_rows_without_event_type() -> pd.DataFrame:
    # Mimics a table loaded from an older pickle: no "event_type" column.
    return pd.DataFrame(
        {
            "atom_index": [0, 1],
            "energy_barrier": [0.5, 0.9],
            "saddle_positions": [
                np.zeros((2, 3)),
                np.ones((2, 3)),
            ],
            "num_reference_event": [0, 1],
        }
    )


def test_remove_duplicates_without_event_type_column():
    table = ActiveEventTable(
        config=_minimal_config(), event_dataframe=_migration_rows_without_event_type()
    )

    # Must not raise KeyError when the column is absent (old pickled tables).
    table.remove_duplicates(cell=np.diag([10.0, 10.0, 10.0]))

    assert len(table.table) == 2


def test_remove_duplicates_skips_dealloying_rows():
    df = _migration_rows_without_event_type()
    df["event_type"] = ["migration", "dealloying"]
    table = ActiveEventTable(config=_minimal_config(), event_dataframe=df)

    table.remove_duplicates(cell=np.diag([10.0, 10.0, 10.0]))

    # Both rows survive; the dealloying row is never compared for duplicates.
    assert len(table.table) == 2


def test_dealloying_active_table_clear_primitive():
    # The KMC loop drops all carried-over rows after a dealloying step because
    # remove_atom shifts every index above the removed atom. This covers the
    # clear expression used there: empty the rows, keep the schema.
    df = _migration_rows_without_event_type()
    table = ActiveEventTable(config=_minimal_config(), event_dataframe=df)
    columns_before = list(table.table.columns)

    table.table = table.table.iloc[0:0].reset_index(drop=True)

    assert len(table.table) == 0
    assert list(table.table.columns) == columns_before


def test_is_valid_new_event_threads_pbc_to_temp_systems(monkeypatch):
    """_build_event_series must carry the source system's pbc into its temp
    Systems, otherwise NeighborsList assumes full periodicity and surface
    systems get graph IDs computed with wrong boundary conditions."""
    import pykmc.event_table as et
    from pykmc.event_table import ReferenceEventTable

    seen_pbc = []

    class _SpyNeighborsList:
        def __init__(self, system, rnei, rcut):
            seen_pbc.append(system.pbc)
            raise RuntimeError("stop after recording pbc")

    monkeypatch.setattr(et, "NeighborsList", _SpyNeighborsList)

    config = SimpleNamespace(
        atomicenvironment=SimpleNamespace(
            atom_coloring_mode="grey", rnei=3.0, rcut=6.0, neighbors_add=0.0
        ),
        eventsearch=SimpleNamespace(
            min_energy_barrier=0.0, max_energy_barrier=100.0, energy_asymmetry=100.0
        ),
    )
    table = ReferenceEventTable.__new__(ReferenceEventTable)
    table.config = config

    pbc = np.array([True, True, False])
    positions = np.random.default_rng(0).random((4, 3)) * 5.0
    try:
        table._build_event_series(
            min1_positions=positions,
            saddle_positions=positions,
            min2_positions=positions,
            index_move=0,
            dE_forward=0.5,
            dE_backward=0.5,
            cell=np.diag([5.0, 5.0, 5.0]),
            pbc=pbc,
        )
    except RuntimeError:
        pass  # the spy aborts after the first NeighborsList construction

    assert len(seen_pbc) == 1
    assert np.array_equal(seen_pbc[0], pbc)
