from __future__ import annotations

import logging
from abc import ABC, abstractmethod

import pandas as pd

logger = logging.getLogger("log")


def resolve_linked_pair(
    selected_event: pd.Series,
    reference_table: pd.DataFrame,
    *,
    is_refined: bool = False,
) -> tuple[pd.Series, pd.Series | None]:
    """Resolve the unique canonical forward row and its declared reverse.

    Logical IDs are independent of DataFrame labels and copied row metadata.
    A reverse may be self-linked or shared by aliases; its link need not point
    back to the selected forward.

    A self-link is authoritative only for a genuine self-reverse, whose
    initial and final topologies coincide (``event_id == id_final``). The
    constant-mode writer also self-links a forward whose reverse was already
    catalogued (``ReferenceEventTable.add`` with ``reverse_idx_ref=None``);
    that placeholder carries no reverse barrier of its own, so the physical
    reverse is resolved by topology among the reciprocal rows only: those
    whose ``event_id`` is the forward's ``id_final`` AND whose ``id_final`` is
    the forward's ``event_id``, the lowest barrier when several exist. A row
    that merely leaves the forward's final topology for a third one is another
    channel, not the reverse.

    A placeholder with no reciprocal row is not a malformed catalogue: the
    constant-mode purge (``ReferenceEventTable.remove``, base rule, no
    closure) legitimately removes the reciprocal and leaves the placeholder
    behind. Its reverse is then unknown, and an unknown reverse cannot be
    shown fast, so the pair resolves to ``(forward, None)`` and the callers
    treat the transition as absorbing; one WARNING names the row, the missing
    reciprocal and the reason. Only a genuinely malformed catalogue raises:
    a missing or duplicated forward identity, or a missing or duplicated
    explicitly linked reverse.
    """
    forward_id = selected_event["num_reference_event" if is_refined else "idx_ref"]
    forward_rows = reference_table[reference_table["idx_ref"] == forward_id]
    if len(forward_rows) != 1:
        raise ValueError(
            f"Basin event {forward_id}: expected one forward row for logical "
            f"idx_ref {forward_id}, found {len(forward_rows)}."
        )
    forward = forward_rows.iloc[0]
    reverse_id = forward["idx_backward"]
    if reverse_id == forward["idx_ref"] and forward["event_id"] != forward["id_final"]:
        candidates = reference_table[
            (reference_table["event_id"] == forward["id_final"])
            & (reference_table["id_final"] == forward["event_id"])
        ]
        if len(candidates) == 0:
            logger.warning(
                "Basin event %s: its self-link is a placeholder for an already "
                "catalogued reverse, but no row returns from topology %r to %r "
                "(the reciprocal was purged or never written); the reverse "
                "barrier is unknown, so the transition is treated as absorbing.",
                forward_id,
                forward["id_final"],
                forward["event_id"],
            )
            return forward, None
        lowest = candidates["energy_barrier"].astype(float).idxmin()
        return forward, candidates.loc[lowest]
    reverse_rows = reference_table[reference_table["idx_ref"] == reverse_id]
    if len(reverse_rows) != 1:
        raise ValueError(
            f"Basin event {forward_id}: expected one linked reverse row for "
            f"logical idx_ref {reverse_id}, found {len(reverse_rows)}."
        )
    return forward, reverse_rows.iloc[0]


class Detector(ABC):
    """Abstract base class for basin detection algorithms"""

    @abstractmethod
    def detect(self) -> bool:
        """Detect if current configuration is in a basin"""
        pass


class DetectorThreshold(Detector):
    def detect(
        self,
        pds_selected_active_event: pd.Series,
        df_reference_table: pd.DataFrame,
        energy_threshold: float,
        is_refined: bool = False,
    ) -> bool:
        """Check if the current configuration is in a basin.

        Returns True if the active event's barrier is below `energy_threshold`
        and its explicitly linked reverse also has a barrier below this
        threshold. Both logical identities must exist uniquely, even if the
        forward barrier is already too high. A placeholder whose reciprocal
        reverse is gone resolves to no reverse row and is absorbing (see
        :func:`resolve_linked_pair`).

        Parameters
        ----------
        pds_selected_active_event : pd.Series
            A pandas Series of the selected active event.
        df_reference_table : pd.DataFrame
            A pandas DataFrame with all generic events.
        energy_threshold : float
            Energy threshold to considere the system in a basin.
        is_refined : bool, optional
            Resolve the active row's ``num_reference_event`` and retain its
            refined forward barrier. Otherwise use the canonical generic row.
        """

        forward, reverse = resolve_linked_pair(
            pds_selected_active_event, df_reference_table, is_refined=is_refined
        )
        if reverse is None:
            return False
        dE_forward = (
            pds_selected_active_event["energy_barrier"]
            if is_refined
            else forward["energy_barrier"]
        )
        return bool(
            dE_forward < energy_threshold
            and reverse["energy_barrier"] < energy_threshold
        )
