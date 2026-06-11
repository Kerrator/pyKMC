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
