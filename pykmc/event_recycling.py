"""Event recycling: carry events with unmoved, distant central atoms into the next KMC step.

For each candidate event in step N's active table, recycle it into step N+1 iff both:
  1. Its central atom's displacement from pre- to post-execution is below `movement_thr`.
  2. Its central atom is farther than `distance_thr` from the executed event's central atom
     (minimum-image PBC distance).
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .event_table import ActiveEventTable
from .system import System


def _minimum_image(disp: np.ndarray, cell: np.ndarray) -> np.ndarray:
    """Apply minimum-image PBC wrapping to a displacement array in-place style.

    Parameters
    ----------
    disp : np.ndarray
        Shape (..., 3) raw displacement (pos_b - pos_a).
    cell : np.ndarray
        3x3 simulation cell (orthorhombic; row-wise vectors).

    Returns
    -------
    np.ndarray
        Wrapped displacement with each component in (-L_i/2, L_i/2].

    """
    cell_lengths = np.linalg.norm(cell, axis=1)
    wrapped = disp.copy()
    for i in range(3):
        wrapped[..., i] -= cell_lengths[i] * np.round(wrapped[..., i] / cell_lengths[i])
    return wrapped


def select_recyclable_events(
    active_table: ActiveEventTable,
    executed_idx: int,
    system: System,
    positions_pre: np.ndarray,
    movement_thr: float,
    distance_thr: float,
) -> pd.DataFrame:
    """Return the rows of `active_table` that pass both the movement and distance checks.

    Parameters
    ----------
    active_table : ActiveEventTable
        The active event table from the current step (post-refinement).
    executed_idx : int
        Row index in `active_table.table` of the event that was executed.
    system : System
        The atomic system with positions already updated to post-execution.
    positions_pre : np.ndarray
        Snapshot of `system.positions` taken before the event was applied.
    movement_thr : float
        Displacement threshold in Angstroms. Atoms moving less than this are "unmoved".
    distance_thr : float
        Distance threshold in Angstroms (PBC-aware) from the executed event's central atom.

    Returns
    -------
    pd.DataFrame
        Subset of `active_table.table` containing the rows that can be recycled into the
        next step. Empty DataFrame if nothing is recyclable.

    """
    table = active_table.table
    if executed_idx not in table.index or len(table) <= 1:
        return table.iloc[0:0].copy()

    cell = np.asarray(system.cell)
    disp_per_atom = np.linalg.norm(
        _minimum_image(system.positions - positions_pre, cell), axis=1
    )

    executed_atom = int(table.loc[executed_idx, "atom_index"])
    executed_pos = system.positions[executed_atom]

    keep_rows: list[int] = []
    for i in table.index:
        if i == executed_idx:
            continue
        atom_idx = int(table.loc[i, "atom_index"])
        if disp_per_atom[atom_idx] >= movement_thr:
            continue
        dvec = _minimum_image(system.positions[atom_idx] - executed_pos, cell)
        if np.linalg.norm(dvec) <= distance_thr:
            continue
        keep_rows.append(i)

    if not keep_rows:
        return table.iloc[0:0].copy()
    return table.loc[keep_rows].reset_index(drop=True).copy()
