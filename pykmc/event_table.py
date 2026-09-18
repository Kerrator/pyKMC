"""Module implementing Classes to manage reference events and active events."""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import pandas as pd
from .rate_constant import (
    RateConstant,
    create_rate_constant,
    rate_from_prefactor,
    thz_to_hz,
)
from .config import Config
import numpy as np
from .environments.graph_nauty import graph
from .system import System
from .neighbors_list import NeighborsList
from .symmetries import unique_symmetries
from .result import (
    Result,
    ErrorInfo,
    Ok,
    Err,
    ErrorType,
    EventSearchOutput,
    EventRefinementOutput,
)
from .point_set_registration import simple_ira, check_match
from .utils.geometry import compute_delr

if TYPE_CHECKING:
    from .event_recycling import Recycling
    from .htst.result import DirectionalPrefactor
    from .rate_constant.prefactors import PrefactorService

logger = logging.getLogger("log")
"""HTST lifecycle diagnostics go to the KMC ``log`` logger (see ``pykmc.log``)."""

TABLE_SCHEMA_VERSION: int = 1
"""Version of the HTST reference-table metadata persisted next to the pickle."""

REFERENCE_BASE_COLUMNS: tuple[str, ...] = (
    "idx_ref",
    "event_id",
    "initial_positions",
    "saddle_positions",
    "final_positions",
    "types",
    "energy_barrier",
    "k",
    "id_saddle",
    "id_final",
    "move_atom_idx",
    "sym_matrix",
    "sym_perm",
    "idx_backward",
    "dra",
)
"""Constant-mode reference schema (S0 baseline, contracts section 7)."""

REFERENCE_HTST_COLUMNS: tuple[str, ...] = (
    "k_prefactor",
    "nu0",
    "nu0_status",
    "nu0_reason",
)
"""Columns appended to the reference schema by the htst/rpa styles."""

ACTIVE_BASE_COLUMNS: tuple[str, ...] = (
    "atom_index",
    "saddle_positions",
    "final_positions",
    "energy_barrier",
    "k",
    "num_reference_event",
    "refined",
)
"""Constant-mode active schema (S0 baseline, contracts section 7)."""

ACTIVE_HTST_COLUMNS: tuple[str, ...] = (
    "k_prefactor",
    "nu0",
    "nu0_status",
    "nu0_reason",
    "nu0_source",
    "nu0_site_attempted",
)
"""Columns appended to the active schema by the htst/rpa styles."""

NU0_OK: str = "ok"
NU0_REJECTED: str = "rejected"
NU0_PENDING: str = "pending"
NU0_LEGACY: str = "legacy"
NU0_STATUSES: tuple[str, ...] = (NU0_OK, NU0_REJECTED, NU0_PENDING, NU0_LEGACY)
"""``nu0_status`` values: accepted estimate, scientific rejection (``k0``),
not yet resolved (``k0`` placeholder), loaded without provenance (``k0``)."""

SOURCE_REFERENCE: str = "reference"
SOURCE_SITE: str = "site"
SOURCE_K0: str = "k0"
"""``nu0_source`` values of an active row: inherited reference estimate,
site-specific estimate at the refined saddle, or the ``k0`` fallback."""

SELF_REVERSE_NU0_RTOL: float = 0.05
"""Relative tolerance under which two directional Vineyard prefactors count as equal.

Used only by the htst/rpa directional identity gate: an event whose endpoint
topologies match and whose saddle crops map onto each other (the IRA check) is
collapsed to one self-linked catalogue row only when both directional
prefactors were accepted and agree within this tolerance. For a genuinely
self-reverse event the two minimum Hessians are related by the symmetry that
maps min1 onto min2, so their spectra differ only by finite-difference and
relaxation noise (well below one percent); 5 % leaves room for that noise while
rejecting physically distinct spectra. The value is a documented constant, not
a validated calibration.
"""

SAME_TOPOLOGY_BARRIER_TOL: float = 0.25
"""Barrier gap (eV) below which same-topology directions may be the same event."""


@dataclass(frozen=True)
class EventAdmission:
    """Outcome of admitting one search result into the reference catalogue.

    Attributes
    ----------
    frame : pd.DataFrame
        One (forward only) or two (forward, backward) event rows whose logical
        ids ``idx_ref``/``idx_backward`` are still unassigned (``-1``).
    reverse_idx_ref : int or None
        Logical id of an already catalogued reverse event that the single
        forward row must link to; ``None`` when the frame carries its own
        reverse or the reverse is unknown.
    self_reverse_candidate : bool
        htst/rpa only: the endpoint topologies match and the saddle crops map
        onto each other, so constant-mode admission would have kept one
        self-linked row. Both directional rows are kept until their
        prefactors have been resolved; see
        :meth:`ReferenceEventTable.finalize_self_reverse`.

    """

    frame: pd.DataFrame
    reverse_idx_ref: int | None = None
    self_reverse_candidate: bool = False


def self_reverse_prefactors_agree(
    forward: DirectionalPrefactor, backward: DirectionalPrefactor
) -> bool:
    """Return True when two directional prefactors support a self-reverse collapse.

    Parameters
    ----------
    forward : DirectionalPrefactor
        Resolved ``min1 -> saddle`` estimate.
    backward : DirectionalPrefactor
        Resolved ``min2 -> saddle`` estimate.

    Returns
    -------
    bool
        ``True`` only when both directions were accepted and their ``nu0_hz``
        agree within :data:`SELF_REVERSE_NU0_RTOL`. A rejected direction never
        supports a collapse: equal fallbacks do not demonstrate equal spectra.

    """
    if not (forward.ok and backward.ok):
        return False
    return math.isclose(
        forward.nu0_hz, backward.nu0_hz, rel_tol=SELF_REVERSE_NU0_RTOL, abs_tol=0.0
    )


class ReferenceEventTable:
    """Store reference events and manage them.

    Parameters
    ----------
    config : Config
        The atomic simulations configuration.

    prefactor_service : PrefactorService, optional
        Batching service resolving per-event prefactors (htst/rpa). Required
        as soon as an event is accepted in those styles; never built or used
        by the constant style.

    Attributes
    ----------
    rate_constant : RateConstant
        Rate facade built from ``config.rateconstant``; its backend decides
        whether the catalogue carries per-event prefactors.
    uses_prefactors : bool
        ``True`` for the htst/rpa styles. Selects the directional identity
        rules of :meth:`_admit_series` and the HTST columns of the schema;
        constant-mode admission and schema are unchanged.
    metadata : dict
        Table-level HTST metadata read from a loaded pickle (empty when the
        table was created fresh or loaded from a legacy file).

    Notes
    -----
    Lifecycle of a reference prefactor (architecture rule 1): admission and
    species-aware dedup run first; the full search geometry of every accepted
    event is kept until its directional logical ids are known; exactly one
    request per accepted event (both directions) is submitted; rows are
    patched by logical id; the batch is resolved before :meth:`add_events`
    returns, so refinement never reads an unresolved reference ``k``.

    """

    def __init__(
        self, config: Config, prefactor_service: PrefactorService | None = None
    ) -> None:
        self.config = config
        self.rate_constant = create_rate_constant(config.rateconstant)
        self.uses_prefactors: bool = bool(
            self.rate_constant.backend.requires_event_prefactors
        )
        self.prefactor_service = prefactor_service
        self.metadata: dict[str, Any] = {}
        self._initialize_table()

    def add_events(
        self, events: list[EventSearchOutput], pbc: Any = None
    ) -> list[Result[pd.DataFrame, ErrorInfo]]:
        """Admit events into the table and, for htst/rpa, resolve their prefactors.

        Parameters
        ----------
        events : list[EventSearchOutput]
            list of EventSearchOutput dataclass with events to be added to the table dataframe.
        pbc : array_like of bool, optional
            Actual periodicity of the system, required by the htst/rpa styles
            to build the prefactor requests; ignored by the constant style.

        Returns
        -------
        list[Result[pd.DataFrame, ErrorInfo]]
            One result per event: the admitted rows (as returned by admission,
            before prefactor resolution) or the rejection.

        """
        results_is_valid_events = []
        accepted: list[tuple[int, int | None, bool, EventSearchOutput]] = []
        # Check if the event is valid based on is_valid_new_event conditions
        for ev in events:
            res = self._admit(
                min1_positions=ev.min1_positions,
                saddle_positions=ev.saddle_positions,
                min2_positions=ev.min2_positions,
                move_atom_idx=ev.move_atom_index,
                dE_forward=ev.dE_forward,
                dE_backward=ev.dE_backward,
                cell=ev.cell,
                types=ev.types,
            )
            if res.is_ok():
                admission = res.ok_value()
                self.add(admission.frame, reverse_idx_ref=admission.reverse_idx_ref)
                results_is_valid_events.append(Ok(admission.frame))
                if self.uses_prefactors:
                    frame = admission.frame
                    fwd_id = int(frame.iloc[0]["idx_ref"])
                    bwd_id = int(frame.iloc[1]["idx_ref"]) if len(frame) > 1 else None
                    accepted.append(
                        (fwd_id, bwd_id, admission.self_reverse_candidate, ev)
                    )
            else:
                results_is_valid_events.append(res)

        if self.uses_prefactors and accepted:
            self._resolve_prefactors(accepted, pbc)

        return results_is_valid_events

    def _resolve_prefactors(
        self,
        accepted: list[tuple[int, int | None, bool, EventSearchOutput]],
        pbc: Any,
    ) -> None:
        """Submit one request per accepted event and patch the directional rows.

        Parameters
        ----------
        accepted : list of (forward id, backward id or None, candidate, event)
            Logical ids assigned by :meth:`add`, the self-reverse flag of the
            admission and the full search geometry.
        pbc : array_like of bool
            Actual periodicity of the system.

        Raises
        ------
        RuntimeError
            If no prefactor service is attached, if ``pbc`` is missing or if a
            search result carries no ``types`` (the request cannot be built).

        """
        if self.prefactor_service is None:
            raise RuntimeError(
                f"rateconstant style {self.config.rateconstant.style!r} needs a "
                "PrefactorService on the reference table to resolve the prefactors "
                "of accepted events; none was attached"
            )
        if pbc is None:
            raise RuntimeError(
                "add_events needs the system pbc to build HTST requests in "
                f"style {self.config.rateconstant.style!r}"
            )
        requests = []
        for fwd_id, bwd_id, _candidate, ev in accepted:
            if ev.types is None:
                raise RuntimeError(
                    "EventSearchOutput.types is required to build the HTST request "
                    f"of reference event {fwd_id}"
                )
            requests.append(
                self.prefactor_service.build_request(
                    event_key=(fwd_id, bwd_id),
                    min1_positions=ev.min1_positions,
                    saddle_positions=ev.saddle_positions,
                    min2_positions=ev.min2_positions,
                    types=ev.types,
                    cell=ev.cell,
                    pbc=pbc,
                    center_index=ev.move_atom_index,
                )
            )
        results = self.prefactor_service.compute(requests)
        for fwd_id, bwd_id, candidate, _ev in accepted:
            pre = results[(fwd_id, bwd_id)]
            self._patch_row(fwd_id, pre.forward)
            self._log_direction(fwd_id, "forward", pre.forward)
            if bwd_id is not None:
                self._patch_row(bwd_id, pre.backward)
                self._log_direction(bwd_id, "backward", pre.backward)
            else:
                # The reverse is already catalogued; its own estimate stands.
                logger.info(
                    "[htst] reference event %d: reverse already catalogued as "
                    "event %d, backward estimate of this search discarded",
                    fwd_id,
                    int(
                        self.table.loc[
                            self.table["idx_ref"] == fwd_id, "idx_backward"
                        ].iloc[0]
                    ),
                )
            if candidate:
                agree = self_reverse_prefactors_agree(pre.forward, pre.backward)
                self.finalize_self_reverse(fwd_id, bwd_id, agree)
                logger.info(
                    "[htst] reference events (%d, %d): equal endpoint topologies; %s",
                    fwd_id,
                    bwd_id,
                    "prefactors agree, collapsed to one self-linked row"
                    if agree
                    else "prefactors differ, two directional rows kept",
                )

    @staticmethod
    def _log_direction(
        idx_ref: int, direction: str, estimate: DirectionalPrefactor
    ) -> None:
        """Log the outcome of one directional reference estimate."""
        if estimate.ok:
            logger.info(
                "[htst] reference event %d (%s): nu0 = %.4e Hz",
                idx_ref,
                direction,
                estimate.nu0_hz,
            )
        else:
            logger.info(
                "[htst] reference event %d (%s): prefactor rejected (%s: %s), "
                "falling back to k0",
                idx_ref,
                direction,
                estimate.reason_code.value,
                estimate.reason,
            )

    def _patch_row(self, idx_ref: int, estimate: DirectionalPrefactor) -> None:
        """Write one directional estimate on the row with logical id ``idx_ref``.

        ``k``, ``k_prefactor``, ``nu0``, ``nu0_status`` and ``nu0_reason`` are
        always updated together: an accepted estimate stores its Hz value and
        the prefactor resolved through the backend (ps^-1); a rejected one
        stores ``NaN``, ``k0`` and the reason.

        Parameters
        ----------
        idx_ref : int
            Logical id of the row (never a positional index).
        estimate : DirectionalPrefactor
            The direction's resolved estimate.

        Raises
        ------
        ValueError
            If no row carries ``idx_ref``.

        """
        mask = self.table["idx_ref"] == idx_ref
        if not mask.any():
            raise ValueError(f"idx_ref {idx_ref} is not in the reference table")
        dE = float(self.table.loc[mask, "energy_barrier"].iloc[0])
        nu0_hz = float(estimate.nu0_hz) if estimate.ok else None
        rc = self.rate_constant.compute_rate(dE, nu0_hz)
        self.table.loc[mask, "k"] = rc.rate
        self.table.loc[mask, "k_prefactor"] = rc.prefactor
        self.table.loc[mask, "nu0"] = nu0_hz if nu0_hz is not None else float("nan")
        self.table.loc[mask, "nu0_status"] = NU0_OK if estimate.ok else NU0_REJECTED
        self.table.loc[mask, "nu0_reason"] = (
            "" if estimate.ok else f"{estimate.reason_code.value}: {estimate.reason}"
        )

    def prefactor_summary(self) -> dict[str, int]:
        """Count the reference rows per ``nu0_status`` (empty in constant mode).

        Returns
        -------
        dict[str, int]
            ``{status: count}`` for every status in :data:`NU0_STATUSES`.

        """
        if not self.uses_prefactors or "nu0_status" not in self.table.columns:
            return {}
        counts = self.table["nu0_status"].value_counts()
        return {status: int(counts.get(status, 0)) for status in NU0_STATUSES}

    def reference_estimate(self, idx_ref: int) -> dict[str, Any]:
        """Return the stored estimate of the reference row ``idx_ref``.

        Parameters
        ----------
        idx_ref : int
            Logical id of the row.

        Returns
        -------
        dict[str, Any]
            Empty in constant mode; otherwise ``{"nu0_hz", "nu0_status",
            "nu0_reason", "nu0_source"}`` with ``nu0_hz`` a float only when
            the status is ``ok`` (``None`` otherwise) and ``nu0_source`` set to
            ``reference``, ready to seed an ``EventRefinementOutput``.

        Raises
        ------
        ValueError
            If no row carries ``idx_ref``.

        """
        if not self.uses_prefactors:
            return {}
        rows = self.table[self.table["idx_ref"] == idx_ref]
        if rows.empty:
            raise ValueError(f"idx_ref {idx_ref} is not in the reference table")
        row = rows.iloc[0]
        status = str(row["nu0_status"])
        nu0 = row["nu0"]
        nu0_hz = float(nu0) if status == NU0_OK else None
        return {
            "nu0_hz": nu0_hz,
            "nu0_status": status,
            "nu0_reason": str(row["nu0_reason"]) if status != NU0_OK else "",
            "nu0_source": SOURCE_REFERENCE,
        }

    def _admit(
        self,
        min1_positions: np.ndarray,
        saddle_positions: np.ndarray,
        min2_positions: np.ndarray,
        move_atom_idx: int,
        dE_forward: float,
        dE_backward: float,
        cell: np.ndarray,
        types: list[str] = None,
    ) -> Result[EventAdmission, ErrorInfo]:
        """Apply the energy gates, build the directional series and admit them.

        Parameters
        ----------
        min1_positions : np.ndarray
            event's positions of the first minimum.
        saddle_positions : np.ndarray
            event's positions of the saddle point.
        min2_positions : np.ndarray
            event's positions of the second minimum.
        move_atom_idx : int
            index of the atom that move the most during the event.
        dE_forward : float
            Energy barrier of the foward event.
        dE_backward : float
            Energy barrier of the backward event.
        cell : np.ndarray
            Simulation box cell.
        types : list[str]
            Event's atom types.

        Returns
        -------
        Result[EventAdmission, ErrorInfo]
            The admitted rows with their linking metadata (see
            :meth:`_admit_series`), or the rejection.

        """
        # Energy bounds
        emin = self.config.eventsearch.emin_event
        emax = self.config.eventsearch.emax_event
        backward_emin = self.config.eventsearch.backward_emin_event
        energy_asymmetry = self.config.eventsearch.energy_asymmetry

        if dE_forward > emax:  # barrier energy too high, reject the event
            return Err(
                ErrorInfo(
                    type=ErrorType.EVENT_ENERGY_HIGHER_THAN_THRESHOLD,
                    message="Energy barrier of the event higher than emax_event",
                    details="Energy barrier = {}, energy max threshold = {}".format(
                        dE_forward, emax
                    ),
                )
            )

        elif dE_forward < emin:  # barrier energy too low, reject the event
            return Err(
                ErrorInfo(
                    type=ErrorType.EVENT_ENERGY_LOWER_THAN_THRESHOLD,
                    message="Energy barrier of the event lower than emin_event",
                    details="Energy barrier = {}, energy min threshold = {}".format(
                        dE_forward, emin
                    ),
                )
            )

        elif (
            dE_backward < emin
        ):  # backard reaction energy barrier too low, reject the event
            return Err(
                ErrorInfo(
                    type=ErrorType.EVENT_BACKWARD_ENERGY_LOWER_THAN_THRESHOLD,
                    message="Backward energy barrier of the event lower than emin_event",
                    details="Backward Energy barrier = {}, energy min threshold = {}".format(
                        dE_backward, emin
                    ),
                )
            )

        # TODO Maybe REMOVE THIS, IT SHOULD NOT HAPPEN
        elif (dE_forward > energy_asymmetry * backward_emin) and (
            dE_backward < backward_emin
        ):  # Asymmetric event, reject
            return Err(
                ErrorInfo(
                    type=ErrorType.EVENT_ASYMMETRIC,
                    message="Found event is highly asymmetric",
                    details="Foward barrier eneryg > {} and backward barrier energy < {}".format(
                        energy_asymmetry * backward_emin, backward_emin
                    ),
                )
            )

        else:  # Event is valid, construct event Series
            dfevent_forward, dfevent_backward = self._build_event_series(
                min1_positions=min1_positions,
                saddle_positions=saddle_positions,
                min2_positions=min2_positions,
                index_move=move_atom_idx,
                dE_forward=dE_forward,
                dE_backward=dE_backward,
                cell=cell,
                types=types,
            )
            return self._admit_series(dfevent_forward, dfevent_backward)

    def is_valid_new_event(
        self,
        min1_positions: np.ndarray,
        saddle_positions: np.ndarray,
        min2_positions: np.ndarray,
        move_atom_idx: int,
        dE_forward: float,
        dE_backward: float,
        cell: np.ndarray,
        types: list[str] = None,
    ) -> Result[pd.DataFrame, ErrorInfo]:
        """Check if the event has the required conditions to be added to the table DataFrame based on the configuration's parameters.

        Thin wrapper over :meth:`_admit` returning only the admitted frame;
        the linking metadata is consumed by :meth:`add_events`.

        Parameters
        ----------
        min1_positions : np.ndarray
            event's positions of the first minimum.
        saddle_positions : np.ndarray
            event's positions of the saddle point.
        min2_positions : np.ndarray
            event's positions of the second minimum.
        move_atom_idx : int
            index of the atom that move the most during the event.
        dE_forward : float
            Energy barrier of the foward event.
        dE_backward : float
            Energy barrier of the backward event.
        cell : np.ndarray
            Simulation box cell.
        types : list[str]
            Event's atom types.

        Returns
        -------
        Result[pd.DataFrame, ErrorInfo]
            The results of the operation.

        """
        res = self._admit(
            min1_positions=min1_positions,
            saddle_positions=saddle_positions,
            min2_positions=min2_positions,
            move_atom_idx=move_atom_idx,
            dE_forward=dE_forward,
            dE_backward=dE_backward,
            cell=cell,
            types=types,
        )
        if res.is_ok():
            return Ok(res.ok_value().frame)
        return res

    def _admit_series(
        self, dfevent_forward: pd.Series, dfevent_backward: pd.Series
    ) -> Result[EventAdmission, ErrorInfo]:
        """Decide how the forward/backward series of one search enter the catalogue.

        Constant mode reproduces the base admission exactly:

        - forward already catalogued (:meth:`find_matching_event`) -> rejected;
        - equal endpoint topologies (``event_id == id_final``) -> one
          self-linked forward row, no geometric check;
        - otherwise both rows when the backward is new, else the forward only
          (self-linked by :meth:`add`).

        htst/rpa mode keeps the same energy and duplicate rules but treats the
        directional identity as a scientific decision (architecture rule
        "Direction identity is a scientific review gate"):

        - a backward direction already in the catalogue links the forward row
          to that logical id instead of self-linking it;
        - equal endpoint topologies collapse to one row only when the saddle
          crops map onto each other (the IRA check, within
          :data:`SAME_TOPOLOGY_BARRIER_TOL`) **and**, after resolution, both
          directional prefactors agree (:meth:`finalize_self_reverse`); until
          then both directional rows are kept with reciprocal links.

        Parameters
        ----------
        dfevent_forward : pd.Series
            Forward event series from :meth:`_build_event_series`.
        dfevent_backward : pd.Series
            Backward event series from :meth:`_build_event_series`.

        Returns
        -------
        Result[EventAdmission, ErrorInfo]
            The admitted rows and their linking metadata, or the rejection.

        """
        if self.find_matching_event(dfevent_forward) is not None:
            return Err(
                ErrorInfo(
                    type=ErrorType.EVENT_NOT_NEW,
                    message="Found event already in reference table",
                    details="Same topology",
                )
            )
        same_topology = dfevent_forward["event_id"] == dfevent_forward["id_final"]

        if not self.uses_prefactors:
            # Constant mode: unchanged base behaviour.
            if same_topology:
                # We are sure that the backward reaction same as forward
                return Ok(EventAdmission(frame=dfevent_forward.to_frame().T))
            if self.is_new_event(dfevent=dfevent_backward):
                return Ok(
                    EventAdmission(
                        frame=self._two_rows(dfevent_forward, dfevent_backward)
                    )
                )
            # backward is already known: forward only (self-linked by add)
            return Ok(EventAdmission(frame=dfevent_forward.to_frame().T))

        # htst/rpa: directional identity gate.
        reverse_idx_ref = self.find_matching_event(dfevent_backward)
        if reverse_idx_ref is not None:
            return Ok(
                EventAdmission(
                    frame=dfevent_forward.to_frame().T,
                    reverse_idx_ref=reverse_idx_ref,
                )
            )
        candidate = same_topology and self._saddle_crops_match(
            dfevent_forward, dfevent_backward
        )
        return Ok(
            EventAdmission(
                frame=self._two_rows(dfevent_forward, dfevent_backward),
                self_reverse_candidate=candidate,
            )
        )

    @staticmethod
    def _two_rows(
        dfevent_forward: pd.Series, dfevent_backward: pd.Series
    ) -> pd.DataFrame:
        """Stack the forward and backward series into a two-row frame."""
        return pd.concat(
            [dfevent_forward.to_frame().T, dfevent_backward.to_frame().T],
            ignore_index=True,
        )

    def _saddle_crops_match(
        self, dfevent_forward: pd.Series, dfevent_backward: pd.Series
    ) -> bool:
        """Return True when the two directional saddle crops map onto each other.

        This is the geometric self-reverse check of the base admission code
        (IRA match of the forward saddle crop against the backward one,
        accepted through ``psr.matching_score_thr``), applied only when the
        two barriers lie within :data:`SAME_TOPOLOGY_BARRIER_TOL`. On the
        base it sat behind an unreachable branch; the htst/rpa gate uses it as
        the "physical mapping" half of the collapse decision. Species are fed
        to IRA exactly as :meth:`find_matching_event` does: the local element
        types in ``full`` colouring mode, a single grey label otherwise, so a
        species-swapped pair of directional crops is never a candidate in
        full mode.

        Parameters
        ----------
        dfevent_forward : pd.Series
            Forward event series.
        dfevent_backward : pd.Series
            Backward event series.

        Returns
        -------
        bool
            Whether the crops match.

        """
        gap = abs(
            float(dfevent_forward["energy_barrier"])
            - float(dfevent_backward["energy_barrier"])
        )
        if gap >= SAME_TOPOLOGY_BARRIER_TOL:
            return False
        ref_saddle = np.array(dfevent_forward["saddle_positions"], copy=True)
        event_saddle = np.array(dfevent_backward["saddle_positions"], copy=True)
        nat_ref = len(ref_saddle)
        nat_event = len(event_saddle)
        full = self.config.atomicenvironment.atom_coloring_mode == "full"
        typ_ref = (
            list(dfevent_forward["types"])
            if full and dfevent_forward["types"] is not None
            else nat_ref * ["X"]
        )
        typ_event = (
            list(dfevent_backward["types"])
            if full and dfevent_backward["types"] is not None
            else nat_event * ["X"]
        )
        result = simple_ira(
            nat_event,
            typ_event,
            event_saddle,
            nat_ref,
            typ_ref,
            ref_saddle,
            self.config.ira.kmax_factor,
        )
        if not result.is_ok():
            return False
        return check_match(result, self.config.psr.matching_score_thr).is_ok()

    def is_new_event(self, dfevent: pd.Series) -> bool:
        """Check if the constructed event Series is already in the table.

        Parameters
        ----------
        dfevent : pd.Series
            the event's Serie.

        Returns
        -------
        bool
            if the event is in the table.

        """
        return self.find_matching_event(dfevent) is None

    def find_matching_event(self, dfevent: pd.Series) -> int | None:
        """Return the logical id of the catalogued event matching ``dfevent``.

        Same rules as the base duplicate check: same ``event_id``, barrier
        within 0.25 eV, and a species-aware (``full`` colouring) or grey IRA
        match of the saddle crops through ``psr.matching_score_thr``.

        Parameters
        ----------
        dfevent : pd.Series
            The event series to look up.

        Returns
        -------
        int or None
            ``idx_ref`` of the first matching row, ``None`` when the event is
            new. The id is logical: rows are never addressed by position.

        """
        # Only select rows with same event_id as dfenvent :
        subset = self.table[self.table["event_id"] == dfevent["event_id"]]
        if len(subset) == 0:
            return None

        # if same  id, chekc if same dE
        tol = 0.25
        dE = dfevent["energy_barrier"]
        subset = subset[(subset["energy_barrier"] - dE).abs() <= tol]
        if len(subset) == 0:
            return None

        # if all same, check PSR  saddle_initial
        event_saddle = dfevent["saddle_positions"]
        nat_event = len(event_saddle)
        # Mirror the PSR / classification paths: only feed real element types to IRA
        # in "full" coloring mode. In "grey" mode every atom is greyed to a single
        # dummy label ('X'), so geometrically-identical species-swapped saddles
        # de-duplicate as one event (grey-alloy approximation). The None-"types"
        # fallback keeps list(None) from raising when a row stores no types.
        full = self.config.atomicenvironment.atom_coloring_mode == "full"
        typ_event = (
            list(dfevent["types"])
            if full and dfevent["types"] is not None
            else nat_event * ["X"]
        )

        for _, ev in subset.iterrows():
            ref_saddle = ev["saddle_positions"]
            nat_ref = len(ref_saddle)
            typ_ref = (
                list(ev["types"])
                if full and ev["types"] is not None
                else nat_ref * ["X"]
            )

            result = simple_ira(
                nat_event,
                typ_event,
                event_saddle,
                nat_ref,
                typ_ref,
                ref_saddle,
                self.config.ira.kmax_factor,
            )

            if not result.is_ok():  # no match
                continue

            result = check_match(result, self.config.psr.matching_score_thr)
            if not result.is_ok():  # matching score > thr
                continue

            return int(ev["idx_ref"])
        return None

    def get_valid_events(
        self, results_is_valid_event: list[Result[pd.Series, ErrorInfo]]
    ) -> list[pd.Series]:
        """Return the list of successful Result.

        Parameters
        ----------
        results_is_valid_event : list[Result[pd.Series, ErrorInfo]]
            list of Result containing event to be added to the table, or ErrorInfo.

        Returns
        -------
        list[pd.Series]
            list of successful Result.

        """
        return [e.ok_value() for e in results_is_valid_event if e.is_ok()]

    def add(self, dfevent: pd.DataFrame, reverse_idx_ref: int | None = None) -> None:
        """Add one or two event rows to the table and assign their logical ids.

        Parameters
        ----------
        dfevent : pd.DataFrame
            One row (forward only) or two rows (forward, backward). Modified in
            place: ``idx_ref`` and ``idx_backward`` are assigned.
        reverse_idx_ref : int or None, optional
            For a single row, the logical id of its already catalogued reverse
            event. ``None`` (the default, and always the case in constant
            mode) self-links the row exactly as the base did.

        """
        # Check if only one or two events (if event is its own backard or not)
        ref = self.max_idx_ref()
        if len(dfevent) == 1:
            dfevent["idx_ref"] = ref
            dfevent["idx_backward"] = (
                ref if reverse_idx_ref is None else reverse_idx_ref
            )
        else:
            dfevent.loc[0, "idx_ref"] = ref
            dfevent.loc[0, "idx_backward"] = ref + 1
            dfevent.loc[1, "idx_ref"] = ref + 1
            dfevent.loc[1, "idx_backward"] = ref

        self.table = pd.concat([self.table, dfevent], ignore_index=True)

    def finalize_self_reverse(
        self, forward_idx_ref: int, backward_idx_ref: int, agree: bool
    ) -> None:
        """Resolve a self-reverse candidate once its directional prefactors are known.

        Parameters
        ----------
        forward_idx_ref : int
            Logical id of the forward row.
        backward_idx_ref : int
            Logical id of the backward row admitted alongside it.
        agree : bool
            Result of :func:`self_reverse_prefactors_agree` on the resolved
            directional prefactors. ``True`` collapses the pair to the single
            self-linked forward row (constant-mode representation): the
            backward row is dropped and any row that linked to it as its
            reverse is re-pointed at the forward row. ``False`` keeps both
            directional rows with their reciprocal links; the logical ids stay
            as assigned (the catalogue may therefore be sparse).

        """
        if not agree:
            return
        fwd_mask = self.table["idx_ref"] == forward_idx_ref
        bwd_mask = self.table["idx_ref"] == backward_idx_ref
        if not fwd_mask.any() or not bwd_mask.any():
            raise ValueError(
                f"self-reverse pair ({forward_idx_ref}, {backward_idx_ref}) is not "
                "in the reference table"
            )
        self.table.loc[fwd_mask, "idx_backward"] = forward_idx_ref
        relink = self.table["idx_backward"] == backward_idx_ref
        self.table.loc[relink, "idx_backward"] = forward_idx_ref
        self.table = self.table[~bwd_mask].reset_index(drop=True)

    def has_id_subset_table(self, ids: list[str]) -> pd.DataFrame:
        """Return subset table with event having id in ids.

        Parameters
        ----------
        ids : list[str]
            list of IDs.

        Returns
        -------
        pd.DataFrame
            Subset of the reference table dataframe with only event having IDs in ids.

        """
        return self.table[self.table["event_id"].isin(ids)]

    def _build_event_series(
        self,
        min1_positions: np.ndarray,
        saddle_positions: np.ndarray,
        min2_positions: np.ndarray,
        index_move: int,
        dE_forward: float,
        dE_backward: float,
        cell: np.ndarray,
        types: list[str] = None,
    ) -> tuple[pd.Series, pd.Series]:
        """Build foward and backward events Series.

        Parameters
        ----------
        min1_positions : np.ndarray
            event's positions of the first minimum.
        saddle_positions : np.ndarray
            event's positions of the saddle point.
        min2_positions : np.ndarray
            event's positions of the second minimum.
        index_move : int
            index of the atom that move the most during the event.
        dE_forward : float
            Energy barrier of the foward event.
        dE_backward : float
            Energy barrier of the backward event.
        cell : np.ndarray
            Simulation box cell.
        types : list[str], optional
            Element type of each atom. When provided, the per-event local types are
            always stored in the ``types`` column (both coloring modes, so the schema
            is mode-independent). Colouring is only *applied* to graph
            hashing/symmetry detection when the configured coloring mode is 'full'.

        Returns
        -------
        tuple[pd.Series, pd.Series]
            tuple containing :
            - a pd.Series of the foward reaction.
            - a pd.Series of the backward reaction.

        """
        full = self.config.atomicenvironment.atom_coloring_mode == "full"
        # Only use element types for graph/symmetry computation in full coloring mode
        graph_types = types if full else None

        # compute neighbors list for initial, saddle and final positions -> to compute graphs
        min1system = System()
        min1system.positions = min1_positions
        min1system.cell = cell
        min1neighbors_list = NeighborsList(
            min1system,
            self.config.atomicenvironment.rnei,
            self.config.atomicenvironment.rcut,
        )

        saddlesystem = System()
        saddlesystem.positions = saddle_positions
        saddlesystem.cell = cell
        saddleneighbors_list = NeighborsList(
            saddlesystem,
            self.config.atomicenvironment.rnei,
            self.config.atomicenvironment.rcut,
        )

        min2system = System()
        min2system.positions = min2_positions
        min2system.cell = cell
        min2neighbors_list = NeighborsList(
            min2system,
            self.config.atomicenvironment.rnei,
            self.config.atomicenvironment.rcut,
        )

        # TODO need to see how to deal with different style for atomic environment ID
        # Compute all needed topology ID :
        id_min1 = graph(
            min1neighbors_list.neighbors_list["rnei"],
            min1neighbors_list.neighbors_list["rcut"],
            atom_idx=[index_move],
            types=graph_types,
        )[0]
        id_saddle = graph(
            saddleneighbors_list.neighbors_list["rnei"],
            saddleneighbors_list.neighbors_list["rcut"],
            atom_idx=[index_move],
            types=graph_types,
        )[0]
        id_min2 = graph(
            min2neighbors_list.neighbors_list["rnei"],
            min2neighbors_list.neighbors_list["rcut"],
            atom_idx=[index_move],
            types=graph_types,
        )[0]

        # query_ball_point can hand back Python lists; coerce to arrays so the
        # element-wise comparisons (np.where) and type indexing below behave.
        neighbor_list_forward = np.asarray(
            min1neighbors_list.neighbors_list["rcut"][index_move]
        )
        neighbor_list_backward = np.asarray(
            min2neighbors_list.neighbors_list["rcut"][index_move]
        )

        # Element types of each neighbor. These are ALWAYS stored (both modes) so
        # the reference-table schema is mode-independent; colouring is only *applied*
        # in matching/symmetry, which is gated below.
        local_types_forward = (
            list(np.array(types)[neighbor_list_forward]) if types is not None else None
        )
        local_types_backward = (
            list(np.array(types)[neighbor_list_backward]) if types is not None else None
        )

        # Colour symmetry detection only in full coloring mode (usage gate).
        sym_types_forward = local_types_forward if full else None
        sym_types_backward = local_types_backward if full else None

        # Symmetries :
        sym_matrix, sym_perm = unique_symmetries(
            min1_positions[neighbor_list_forward],
            min2_positions[neighbor_list_forward],
            self.config.ira.sym_thr,
            types=sym_types_forward,
        )

        # dr :
        move_atom_idx_forward = np.where(neighbor_list_forward == index_move)[0][0]
        dra_forward = np.linalg.norm(
            min1_positions[neighbor_list_forward][move_atom_idx_forward]
            - saddle_positions[neighbor_list_forward][move_atom_idx_forward]
        )
        move_atom_idx_backward = np.where(neighbor_list_backward == index_move)[0][0]
        dra_backward = np.linalg.norm(
            min2_positions[neighbor_list_backward][move_atom_idx_backward]
            - saddle_positions[neighbor_list_backward][move_atom_idx_backward]
        )

        # Rates are built without a per-event estimate: the constant backend
        # returns k0 and the htst/rpa backends resolve to their k0 placeholder;
        # _resolve_prefactors patches the accepted rows afterwards.
        dfevent_forward = pd.Series(
            {
                "idx_ref": -1,  # unknown yet
                "event_id": id_min1,
                "initial_positions": min1_positions[neighbor_list_forward],
                "saddle_positions": saddle_positions[neighbor_list_forward],
                "final_positions": min2_positions[neighbor_list_forward],
                "types": local_types_forward,
                "energy_barrier": dE_forward,
                "k": self.rate_constant.compute_rate(dE_forward).rate,
                "id_saddle": id_saddle,
                "id_final": id_min2,
                "move_atom_idx": np.where(neighbor_list_forward == index_move)[0][0],
                "sym_matrix": sym_matrix,
                "sym_perm": sym_perm,
                "idx_backward": -1,  # unknown yet,
                "dra": dra_forward,
            }
        )

        sym_matrix, sym_perm = unique_symmetries(
            min2_positions[neighbor_list_backward],
            min1_positions[neighbor_list_backward],
            self.config.ira.sym_thr,
            types=sym_types_backward,
        )
        dfevent_backward = pd.Series(
            {
                "idx_ref": -1,  # unknown yet
                "event_id": id_min2,
                "initial_positions": min2_positions[neighbor_list_backward],
                "saddle_positions": saddle_positions[neighbor_list_backward],
                "final_positions": min1_positions[neighbor_list_backward],
                "types": local_types_backward,
                "energy_barrier": dE_backward,
                "k": self.rate_constant.compute_rate(dE_backward).rate,
                "id_saddle": id_saddle,
                "id_final": id_min1,
                "move_atom_idx": np.where(neighbor_list_backward == index_move)[0][0],
                "sym_matrix": sym_matrix,
                "sym_perm": sym_perm,
                "idx_backward": -1,  # unknown yet
                "dra": dra_backward,
            }
        )
        if self.uses_prefactors:
            for series in (dfevent_forward, dfevent_backward):
                series["k_prefactor"] = self.config.rateconstant.k0
                series["nu0"] = float("nan")
                series["nu0_status"] = NU0_PENDING
                series["nu0_reason"] = ""

        return dfevent_forward, dfevent_backward

    def max_idx_ref(self) -> int:
        """Return max value of idx_ref"""
        if len(self.table) == 0:
            return 0
        else:
            return int(self.table["idx_ref"].max()) + 1

    def _initialize_table(self) -> None:
        """Initialize the reference event table.

        If a path to a reference table is in the configurations it reads it, otherwise initialize an empty dataframe.
        """
        if self.config.control.reference_table is not None:
            self._load(self.config.control.reference_table)
        else:
            columns = {
                "idx_ref": pd.Series(dtype="int64"),
                "event_id": pd.Series(dtype="str"),
                "initial_positions": pd.Series(dtype="object"),
                "saddle_positions": pd.Series(dtype="object"),
                "final_positions": pd.Series(dtype="object"),
                "types": pd.Series(dtype="object"),
                "energy_barrier": pd.Series(dtype="float64"),
                "k": pd.Series(dtype="float64"),
                "id_saddle": pd.Series(dtype="str"),
                "id_final": pd.Series(dtype="str"),
                "move_atom_idx": pd.Series(dtype="int64"),
                "sym_matrix": pd.Series(dtype="object"),
                "sym_perm": pd.Series(dtype="object"),
                "idx_backward": pd.Series(dtype="int64"),
                "dra": pd.Series(dtype="float64"),
            }
            if self.uses_prefactors:
                columns["k_prefactor"] = pd.Series(dtype="float64")
                columns["nu0"] = pd.Series(dtype="float64")
                columns["nu0_status"] = pd.Series(dtype="str")
                columns["nu0_reason"] = pd.Series(dtype="str")
            self.table = pd.DataFrame(columns)

    def _load(self, path: str) -> None:
        """Load a pickled reference table deliberately, per style and provenance.

        Constant style: a constant-mode pickle is loaded exactly as the base
        did (no new columns, no metadata, rates untouched). A pickle carrying
        any HTST column (the complete set or a partial one such as the
        donor-era ``k_prefactor`` + ``nu0`` pair) or metadata is stripped of
        them and its rates are recomputed with ``k0`` at the current
        temperature, with a warning: constant runs never reuse per-event
        prefactors.

        htst/rpa style: a table without the HTST columns, or with them but
        without table-level metadata (Hz ``nu0`` of unknown provenance), is a
        legacy table: every row gets ``nu0_status = "legacy"``, ``k_prefactor
        = k0`` and ``k`` recomputed from ``energy_barrier`` at the current
        temperature, and one warning is logged. A table with metadata is
        validated (schema version and units) and its rates are recomputed
        from the stored ``nu0`` (accepted rows, through the rate backend,
        which must reproduce the stored ``k_prefactor``) or the current
        ``k0`` (every other status) and ``energy_barrier`` at the current
        temperature, so a temperature change on reload is never silently
        ignored and a row whose ``k_prefactor`` disagrees with its ``nu0`` is
        refused.

        Parameters
        ----------
        path : str
            Pickle file written by :meth:`save`.

        Raises
        ------
        ValueError
            If the metadata declares an unsupported schema version or units,
            or if an accepted row's ``k_prefactor`` is not the resolution of
            its ``nu0`` (or its status is outside the vocabulary).

        """
        df = pd.read_pickle(path)
        metadata = dict(df.attrs) if df.attrs else {}
        df.attrs = {}
        present_htst = [c for c in REFERENCE_HTST_COLUMNS if c in df.columns]
        has_htst_columns = len(present_htst) == len(REFERENCE_HTST_COLUMNS)
        k0 = self.config.rateconstant.k0
        T = self.config.rateconstant.T

        if not self.uses_prefactors:
            # Any HTST provenance is dropped: the complete S6 column set, a
            # partial one (the donor-era ``k_prefactor`` + ``nu0`` pair) or
            # table metadata. A constant run never reuses per-event
            # prefactors, whatever subset of them a pickle carries.
            if present_htst or metadata:
                logger.warning(
                    "Reference table %s carries HTST prefactor data (columns: %s%s) "
                    "but the run uses the constant style: dropping the HTST "
                    "columns and recomputing every rate with k0 = %g ps^-1 at "
                    "T = %g K",
                    path,
                    ", ".join(present_htst) if present_htst else "none",
                    "; table metadata" if metadata else "",
                    k0,
                    T,
                )
                df = df.drop(columns=present_htst)
                df["k"] = [
                    rate_from_prefactor(k0, float(dE), T) for dE in df["energy_barrier"]
                ]
            self.table = df
            return

        if not has_htst_columns or not metadata:
            if has_htst_columns:
                why = "no HTST metadata"
            elif present_htst:
                why = "incomplete HTST columns (" + ", ".join(present_htst) + ")"
            else:
                why = "no HTST columns"
            logger.warning(
                "Reference table %s is a legacy table (%s): every event gets "
                "nu0_status = 'legacy', k_prefactor = k0 = %g ps^-1 and k "
                "recomputed from its barrier at T = %g K; no prefactor of this "
                "table is an HTST estimate",
                path,
                why,
                k0,
                T,
            )
            df["k_prefactor"] = float(k0)
            if "nu0" not in df.columns:
                df["nu0"] = float("nan")
            df["nu0_status"] = NU0_LEGACY
            df["nu0_reason"] = f"legacy table: {why}"
            df["k"] = [
                rate_from_prefactor(k0, float(dE), T) for dE in df["energy_barrier"]
            ]
            # Canonical layout: the HTST columns follow the others in schema order.
            other = [c for c in df.columns if c not in REFERENCE_HTST_COLUMNS]
            self.table = df[other + list(REFERENCE_HTST_COLUMNS)]
            return

        version = metadata.get("schema_version")
        if version != TABLE_SCHEMA_VERSION:
            raise ValueError(
                f"reference table {path} has schema_version {version!r}; this "
                f"code reads version {TABLE_SCHEMA_VERSION}"
            )
        if metadata.get("nu0_units") != "Hz" or (
            metadata.get("k_prefactor_units") != "ps^-1"
        ):
            raise ValueError(
                f"reference table {path} declares nu0_units="
                f"{metadata.get('nu0_units')!r} and k_prefactor_units="
                f"{metadata.get('k_prefactor_units')!r}; expected 'Hz' and 'ps^-1' "
                "(units are never inferred from magnitudes)"
            )
        if metadata.get("T") != T:
            logger.info(
                "Reference table %s was saved at T = %s K; rates recomputed at "
                "T = %g K from the stored prefactors",
                path,
                metadata.get("T"),
                T,
            )
        if metadata.get("k0") != k0:
            logger.info(
                "Reference table %s was saved with k0 = %s ps^-1; fallback rows "
                "re-based on k0 = %g ps^-1",
                path,
                metadata.get("k0"),
                k0,
            )
        labels = df["idx_ref"] if "idx_ref" in df.columns else df.index
        prefactors = []
        rates = []
        for label, status, k_prefactor, nu0, dE in zip(
            labels,
            df["nu0_status"],
            df["k_prefactor"],
            df["nu0"],
            df["energy_barrier"],
            strict=True,
        ):
            if status not in NU0_STATUSES:
                raise ValueError(
                    f"reference table {path}: event {label} has nu0_status "
                    f"{status!r}; expected one of {NU0_STATUSES}"
                )
            if status == NU0_OK:
                nu0 = float(nu0)
                if not math.isfinite(nu0) or nu0 <= 0.0:
                    raise ValueError(
                        f"reference table {path}: event {label} has nu0_status "
                        f"'ok' but nu0 = {nu0!r} Hz; an accepted estimate must be "
                        "a finite positive frequency"
                    )
                # Single conversion point: the stored prefactor must be the
                # backend's own resolution of the stored frequency, otherwise
                # k, k_prefactor and nu0 would disagree on the loaded row.
                rc = self.rate_constant.compute_rate(float(dE), nu0)
                if not math.isclose(
                    rc.prefactor, float(k_prefactor), rel_tol=1e-9, abs_tol=0.0
                ):
                    raise ValueError(
                        f"reference table {path}: event {label} stores k_prefactor "
                        f"= {float(k_prefactor)!r} ps^-1 but its nu0 = {nu0!r} Hz "
                        f"resolves to {rc.prefactor!r} ps^-1; k, k_prefactor and "
                        "nu0 must agree (the table was edited or corrupted)"
                    )
            else:
                rc = self.rate_constant.compute_rate(float(dE))  # k0 fallback
            prefactors.append(rc.prefactor)
            rates.append(rc.rate)
        df["k_prefactor"] = prefactors
        df["k"] = rates
        self.metadata = metadata
        self.table = df

    def table_metadata(self) -> dict[str, Any]:
        """Return the table-level HTST metadata persisted with the pickle.

        Returns
        -------
        dict[str, Any]
            ``schema_version``, ``style``, ``nu0_units`` (Hz),
            ``k_prefactor_units`` (ps^-1), ``T`` (K), ``k0`` (ps^-1) and the
            kernel ``settings`` (radii and step in Angstrom, window in Hz).

        """
        rc = self.config.rateconstant
        return {
            "schema_version": TABLE_SCHEMA_VERSION,
            "style": rc.style,
            "nu0_units": "Hz",
            "k_prefactor_units": "ps^-1",
            "T": float(rc.T),
            "k0": float(rc.k0),
            "settings": {
                "free_radius": float(rc.free_radius),
                "fd_step": float(rc.fd_step),
                "zone_radius": None
                if rc.zone_radius is None
                else float(rc.zone_radius),
                "premin": bool(rc.premin),
                "nu0_min_hz": thz_to_hz(rc.nu0_min_THz),
                "nu0_max_hz": thz_to_hz(rc.nu0_max_THz),
            },
        }

    def remove(self, idx_refs: list[int]) -> None:
        """Remove events with ind == idx_ref as well as its backward event

        Constant mode: exactly the base rule (the event and the row its
        ``idx_backward`` names; links are reciprocal or self there).

        htst/rpa mode: links are not always reciprocal (a new forward row may
        link to an already catalogued reverse), so the removal is closed
        under reverse links: every surviving row whose ``idx_backward`` names
        a removed id is removed as well, iterated to a fixed point. This may
        remove more than the base pair, but never leaves a dangling link
        (``info_active_events`` and the basin exploration dereference
        ``idx_backward`` by logical id).

        Parameters
        ----------
        idx_refs : list[int]
            logical ids of the events to be removed
        """

        idx_refs = set(idx_refs)  # make a set if there are doublons

        backward_refs = set(
            self.table.loc[self.table["idx_ref"].isin(idx_refs), "idx_backward"].astype(
                int
            )
        )  # find set idx backwards

        all_refs = idx_refs | backward_refs  # all ref to remove
        if self.uses_prefactors:
            while True:
                dangling = (
                    set(
                        self.table.loc[
                            self.table["idx_backward"].astype(int).isin(all_refs),
                            "idx_ref",
                        ].astype(int)
                    )
                    - all_refs
                )
                if not dangling:
                    break
                all_refs = all_refs | dangling

        self.table = self.table[~self.table["idx_ref"].isin(all_refs)].reset_index(
            drop=True
        )  # keep event not (~) in all refs

    def save(self, outfile: str = "reference_table.pickle") -> None:
        """Save the reference event table to a pickle file.

        In the htst/rpa styles the table-level metadata of
        :meth:`table_metadata` travels inside the pickle as
        ``DataFrame.attrs`` (pandas persists ``attrs`` through
        ``to_pickle``/``read_pickle``; ``concat`` drops them, which is why they
        are re-attached here at every save). Constant-mode pickles carry no
        attrs and are byte-identical to the base.

        Parameters
        ----------
        outfile : str, optional
            path to the output file, by default 'reference_table.pickle'.

        """
        if self.uses_prefactors:
            self.table.attrs = self.table_metadata()
        self.table.to_pickle(outfile)


class ActiveEventTable:
    """Store active events and manage them.

    Parameters
    ----------
    config : Config
        The atomic simulations configuration.
    event_dataframe : pd.DataFrame, optional
        An table with active event use to initialize the table. by default 'None'.
        In the htst/rpa styles it must carry :data:`ACTIVE_HTST_COLUMNS`;
        the first htst-only operation (:meth:`add_events`,
        :meth:`request_site_prefactors`) refuses a frame that lacks any of
        them with a ``ValueError`` naming the missing columns.
    recycler : Recycling, optional
        Recycling plugin deciding which rows survive between steps.
    prefactor_service : PrefactorService, optional
        Batching service for the site-specific estimates (htst/rpa only).

    Notes
    -----
    Lifecycle of an active prefactor (architecture rules 2-4): a row is built
    from the estimate inherited through ``EventRefinementOutput`` (source
    ``reference`` when the reference estimate is accepted, else ``k0``) with
    ``k`` recomputed at the refined barrier. After duplicates are removed,
    :meth:`request_site_prefactors` submits one request per newly accepted
    ``refined == "T"`` row; success overrides the estimate (source ``site``),
    a scientific rejection keeps the row as it is (a valid inherited
    reference estimate, else ``k0``). ``nu0_site_attempted`` records the
    attempt, not its success, so a recycled row is never re-attempted. A
    row's geometry is never rebuilt in place: a changed geometry is a new row
    built by :meth:`add_events` from its reference estimate.

    """

    def __init__(
        self,
        config: Config,
        event_dataframe: pd.DataFrame = None,
        recycler: "Recycling | None" = None,
        prefactor_service: PrefactorService | None = None,
    ):
        self.config = config
        # Optional recycling plugin. If attached, `prune_for_recycling` keeps
        # the rows the recycler selects between KMC steps. If None, the table
        # is cleared at the end of each step (matching prior behavior).
        self.recycler = recycler
        # The rate facade is built on first use so that a table wrapped around
        # an existing DataFrame (recycling tests, tooling) never touches the
        # rate configuration, exactly as before.
        self._rate_constant: RateConstant | None = None
        self.prefactor_service = prefactor_service

        if event_dataframe is not None:
            if not isinstance(event_dataframe, pd.DataFrame):
                raise TypeError("event_dataframe must be a pandas DataFrame or None.")
            self.table = event_dataframe
        else:
            columns = {
                "atom_index": pd.Series(dtype="int64"),
                "saddle_positions": pd.Series(dtype="object"),
                "final_positions": pd.Series(dtype="object"),
                "energy_barrier": pd.Series(dtype="float64"),
                "k": pd.Series(dtype="float64"),
                "num_reference_event": pd.Series(dtype="int64"),
                "refined": pd.Series(dtype="str"),
            }
            if self.uses_prefactors:
                columns["k_prefactor"] = pd.Series(dtype="float64")
                columns["nu0"] = pd.Series(dtype="float64")
                columns["nu0_status"] = pd.Series(dtype="str")
                columns["nu0_reason"] = pd.Series(dtype="str")
                columns["nu0_source"] = pd.Series(dtype="str")
                columns["nu0_site_attempted"] = pd.Series(dtype="bool")
            self.table = pd.DataFrame(columns)

    @property
    def rate_constant(self) -> RateConstant:
        """Return the rate facade of ``config.rateconstant`` (built on first use)."""
        if self._rate_constant is None:
            self._rate_constant = create_rate_constant(self.config.rateconstant)
        return self._rate_constant

    @property
    def uses_prefactors(self) -> bool:
        """Return True when the rate backend needs per-event prefactors (htst/rpa)."""
        return bool(self.rate_constant.backend.requires_event_prefactors)

    def _require_htst_columns(self, operation: str) -> None:
        """Refuse an htst/rpa table whose frame lacks the HTST columns.

        Parameters
        ----------
        operation : str
            Name of the operation about to run, for the error message.

        Raises
        ------
        ValueError
            Naming the missing :data:`ACTIVE_HTST_COLUMNS`.

        """
        missing = [c for c in ACTIVE_HTST_COLUMNS if c not in self.table.columns]
        if missing:
            raise ValueError(
                f"{operation}: the active table of style "
                f"{self.config.rateconstant.style!r} lacks the HTST columns "
                f"{missing}; a caller-supplied event_dataframe must carry "
                f"{list(ACTIVE_HTST_COLUMNS)}"
            )

    def prune_for_recycling(
        self,
        executed_idx: int,
        system: System,
        positions_pre: np.ndarray,
    ) -> None:
        """Replace `self.table` with the rows that survive the recycler's filter.

        If no recycler is attached, clear the table (matches the prior
        end-of-step `del active_table` behavior).
        """
        if self.recycler is None:
            self.table = self.table.iloc[0:0].reset_index(drop=True)
        else:
            self.table = self.recycler.select_recyclable(
                self,
                executed_idx,
                system,
                positions_pre,
            )

    def existing_pairs(self) -> set[tuple[int, int]]:
        """Return `(atom_index, num_reference_event)` tuples already in the table.

        Used by `Refinement.execute` to skip pairs that survived the last step
        and don't need to be refined again.
        """
        if len(self.table) == 0:
            return set()
        return set(
            zip(
                self.table["atom_index"].astype(int).tolist(),
                self.table["num_reference_event"].astype(int).tolist(),
            )
        )

    def add_events(
        self, events: EventRefinementOutput | list[EventRefinementOutput]
    ) -> None:
        """Add active events to the table.

        Parameters
        ----------
        events : EventRefinementOutput | list[EventRefinementOutput]
            An EventRefinementOuput dataclass, or a list of it, with active event to be added to the table.

        Raises
        ------
        TypeError
            if events is not a EventRefinementOuput dataclass or a list of it.

        """
        if isinstance(events, list):
            dfactive = []
            for e in events:
                dfactive.append(self.build_event_series(e))
        elif isinstance(events, EventRefinementOutput):
            dfactive = self.build_event_series(events)
        else:
            raise TypeError(
                "Input 'events' must be an EventRefinementOutput dataclass or a list of it."
            )
        if self.uses_prefactors:
            self._require_htst_columns("add_events")
        self.add(dfactive)

    def add(self, dfevents: pd.Series | list[pd.Series]) -> None:
        """Add a pd.Series of the active events.

        Parameters
        ----------
        dfevents : pd.Series | list[pd.Series]
            a pd.Series of an event to be added to the table, or a list of it.

        Raises
        ------
        TypeError
            if dfevents is not a pd.Series.

        """
        if isinstance(dfevents, pd.Series):
            df_to_add = dfevents.to_frame().T
        elif isinstance(dfevents, list):
            if not all(isinstance(s, pd.Series) for s in dfevents):
                raise TypeError("All elements in the input list must be pandas Series.")
            df_to_add = pd.DataFrame(dfevents)
        else:
            raise TypeError(
                "Input 'dfevents' must be a pandas Series or a list of pandas Series."
            )

        self.table = pd.concat([self.table, df_to_add], ignore_index=True)

    def build_event_series(
        self, event_refinement_output: EventRefinementOutput
    ) -> pd.Series:
        """Build an event Series based on the EventRefinementOuput dataclass.

        Parameters
        ----------
        event_search_output : EventRefinementOutput
            The dataclass with the active event informations.

        Returns
        -------
        pd.Series
            The pd.Series of the event.

        """

        dE = event_refinement_output.dE_forward
        if self.uses_prefactors:
            status, nu0_hz, reason, source = self._inherited_estimate(
                event_refinement_output
            )
            rc = self.rate_constant.compute_rate(dE, nu0_hz)
        else:
            rc = self.rate_constant.compute_rate(dE)
        dfactive = pd.Series(
            {
                "atom_index": event_refinement_output.central_atom_index,
                "saddle_positions": event_refinement_output.saddle_positions,
                "final_positions": event_refinement_output.min2_positions,
                "energy_barrier": dE,
                "k": rc.rate,
                "num_reference_event": event_refinement_output.num_reference_event,
                "refined": event_refinement_output.refined,
            }
        )
        if self.uses_prefactors:
            dfactive["k_prefactor"] = rc.prefactor
            dfactive["nu0"] = nu0_hz if nu0_hz is not None else float("nan")
            dfactive["nu0_status"] = status
            dfactive["nu0_reason"] = reason
            dfactive["nu0_source"] = source
            dfactive["nu0_site_attempted"] = False
        return dfactive

    @staticmethod
    def _inherited_estimate(
        event_refinement_output: EventRefinementOutput,
    ) -> tuple[str, float | None, str, str]:
        """Normalise the estimate a refinement output inherited from its reference.

        Parameters
        ----------
        event_refinement_output : EventRefinementOutput
            Refinement output carrying ``nu0_hz``/``nu0_status``/``nu0_reason``.

        Returns
        -------
        tuple[str, float | None, str, str]
            ``(nu0_status, nu0_hz, nu0_reason, nu0_source)``: an accepted
            reference estimate is inherited as is (source ``reference``); any
            other status, or no information at all, resolves to ``k0``
            (source ``k0``) while keeping the status and reason.

        Raises
        ------
        ValueError
            If the status is ``ok`` without a finite positive ``nu0_hz``.

        """
        status = event_refinement_output.nu0_status
        if status == NU0_OK:
            nu0 = event_refinement_output.nu0_hz
            if nu0 is None or not math.isfinite(nu0) or nu0 <= 0.0:
                raise ValueError(
                    "EventRefinementOutput has nu0_status 'ok' but nu0_hz "
                    f"{nu0!r}; an accepted estimate must be a finite positive Hz"
                )
            return NU0_OK, float(nu0), "", SOURCE_REFERENCE
        if status is None:
            return NU0_REJECTED, None, "no reference estimate", SOURCE_K0
        if status not in NU0_STATUSES:
            raise ValueError(f"unknown nu0_status {status!r}")
        return status, None, event_refinement_output.nu0_reason or "", SOURCE_K0

    def request_site_prefactors(
        self, system: System, neighbors_list: NeighborsList
    ) -> dict[str, int]:
        """Request one site-specific estimate per newly accepted refined row.

        Call after :meth:`remove_duplicates`, so duplicates never cost a
        Hessian. Only ``refined == "T"`` rows that have not been attempted
        participate; ``"F"``/``"B"`` rows and recycled rows keep their values.

        Ordering invariant: a refined row stores its saddle and final
        positions cropped by ``neighbors_list.get_neighbors("rcut",
        atom_index)`` evaluated on the neighbour list refinement ran with,
        in that list's order (``Refinement.refine_single`` crops with
        ``ctx["neighbors"]`` and ``KMC._reconstruction_active_event`` reads
        them back through the same call). The caller must therefore pass that
        same neighbour list; the full geometries are rebuilt by writing the
        crops into a copy of the current minimum at those indices, and the
        crop length is checked against the mapping before any request.

        Crop approximation: only the atoms inside the ``rcut`` crop carry
        refined saddle/final positions; every other atom sits at its current
        minimum position in the rebuilt geometries. Free atoms of the Hessian
        beyond ``rcut`` (``rateconstant.free_radius > atomicenvironment.rcut``)
        are therefore evaluated off their stationary point, which biases the
        site spectra towards ``saddle_not_first_order``/``unstable_minimum``
        rejections (and the reference/``k0`` fallback). ``Initializer``
        warns once about that configuration.

        Parameters
        ----------
        system : System
            Current system; its positions are the initial minimum of every
            active event.
        neighbors_list : NeighborsList
            The neighbour list refinement cropped with.

        Returns
        -------
        dict[str, int]
            ``{"attempted", "ok", "rejected"}`` counts for this call (all zero
            in constant mode or when nothing was eligible).

        Raises
        ------
        RuntimeError
            If a row is eligible but no prefactor service is attached, or if
            a row's crop does not match the current neighbour mapping.
        ValueError
            If the table (a caller-supplied frame) lacks the HTST columns.

        """
        summary = {"attempted": 0, "ok": 0, "rejected": 0}
        if not self.uses_prefactors or len(self.table) == 0:
            return summary
        self._require_htst_columns("request_site_prefactors")
        eligible = (self.table["refined"] == "T") & ~self.table[
            "nu0_site_attempted"
        ].astype(bool)
        rows = self.table[eligible]
        if rows.empty:
            return summary
        if self.prefactor_service is None:
            raise RuntimeError(
                f"rateconstant style {self.config.rateconstant.style!r} needs a "
                "PrefactorService on the active table to request site estimates; "
                "none was attached"
            )
        positions = np.asarray(system.positions, dtype=float)
        requests = []
        keys: list[tuple[Any, tuple]] = []
        for idx, row in rows.iterrows():
            atom = int(row["atom_index"])
            neighbors = np.asarray(
                neighbors_list.get_neighbors("rcut", atom), dtype=int
            )
            saddle_crop = np.asarray(row["saddle_positions"], dtype=float)
            final_crop = np.asarray(row["final_positions"], dtype=float)
            if (
                saddle_crop.shape != (len(neighbors), 3)
                or final_crop.shape != (len(neighbors), 3)
                or atom not in neighbors
            ):
                raise RuntimeError(
                    f"active row {idx} (atom {atom}): stored crops of shape "
                    f"{saddle_crop.shape}/{final_crop.shape} do not match the "
                    f"current rcut mapping of {len(neighbors)} neighbours; the "
                    "site request must use the neighbour list refinement cropped with"
                )
            full_saddle = positions.copy()
            full_saddle[neighbors] = saddle_crop
            full_min2 = positions.copy()
            full_min2[neighbors] = final_crop
            key = ("site", int(idx), atom, int(row["num_reference_event"]))
            requests.append(
                self.prefactor_service.build_request(
                    event_key=key,
                    min1_positions=positions,
                    saddle_positions=full_saddle,
                    min2_positions=full_min2,
                    types=system.types,
                    cell=system.cell,
                    pbc=system.pbc,
                    center_index=atom,
                )
            )
            keys.append((idx, key))
        results = self.prefactor_service.compute(requests)
        for idx, key in keys:
            estimate = results[key].forward
            self.table.loc[idx, "nu0_site_attempted"] = True
            summary["attempted"] += 1
            atom = int(self.table.loc[idx, "atom_index"])
            ref = int(self.table.loc[idx, "num_reference_event"])
            if estimate.ok:
                dE = float(self.table.loc[idx, "energy_barrier"])
                rc = self.rate_constant.compute_rate(dE, float(estimate.nu0_hz))
                self.table.loc[idx, "k"] = rc.rate
                self.table.loc[idx, "k_prefactor"] = rc.prefactor
                self.table.loc[idx, "nu0"] = float(estimate.nu0_hz)
                self.table.loc[idx, "nu0_status"] = NU0_OK
                self.table.loc[idx, "nu0_reason"] = ""
                self.table.loc[idx, "nu0_source"] = SOURCE_SITE
                summary["ok"] += 1
                logger.info(
                    "[htst] active event (atom %d, reference %d): site nu0 = %.4e Hz",
                    atom,
                    ref,
                    estimate.nu0_hz,
                )
            else:
                summary["rejected"] += 1
                logger.info(
                    "[htst] active event (atom %d, reference %d): site prefactor "
                    "rejected (%s: %s); keeping the %s estimate",
                    atom,
                    ref,
                    estimate.reason_code.value,
                    estimate.reason,
                    self.table.loc[idx, "nu0_source"],
                )
        return summary

    def prefactor_summary(self) -> dict[str, int]:
        """Count the active rows per ``nu0_source`` and the attempted ones.

        Returns
        -------
        dict[str, int]
            ``{"reference", "site", "k0", "site_attempted"}``; empty in
            constant mode.

        """
        if not self.uses_prefactors or "nu0_source" not in self.table.columns:
            return {}
        counts = self.table["nu0_source"].value_counts()
        return {
            SOURCE_REFERENCE: int(counts.get(SOURCE_REFERENCE, 0)),
            SOURCE_SITE: int(counts.get(SOURCE_SITE, 0)),
            SOURCE_K0: int(counts.get(SOURCE_K0, 0)),
            "site_attempted": int(self.table["nu0_site_attempted"].astype(bool).sum()),
        }

    def remove(self, ind: int | list[int]) -> None:
        """Remove event at row = ind

        Parameters
        ----------
        ind : int
            index of the row to be removed
        """
        self.table = self.table.drop(ind)
        self.table = self.table.reset_index(drop=True)

    def remove_duplicates(self, cell, neighbors_list: NeighborsList = None) -> None:
        """Loop over all active events in the DataFrame, check if there are duplicates by computing delr."""

        duplicates = []
        # 1. Check duplicates on central atoms : to be sure
        # Sub dataframes with events grouped by central_atom and dE
        tol_energy = 0.1  # eV
        grouped = []

        for idx, row in self.table.iterrows():
            central_atom = row["atom_index"]
            dE = row["energy_barrier"]

            subset = self.table[
                (self.table["atom_index"] == central_atom)
                & (abs(self.table["energy_barrier"] - dE) < tol_energy)
            ]
            grouped.append((idx, subset))

        # For each group, check duplicated by computing delr

        for idx, subset in grouped:
            pos_ref = np.array(self.table.loc[idx, "saddle_positions"])
            for jdx in subset.index:
                if jdx <= idx:
                    continue  # dont compute twice
                pos_comp = np.array(self.table.loc[jdx, "saddle_positions"])
                delr = compute_delr(pos_ref, pos_comp, cell)
                if delr < self.config.psr.matching_score_thr:
                    # print('Removing event with delr',delr)
                    duplicates.append(jdx)

        # 2. Check duplicates due to symmetric events applied on different central atoms.
        # Group by same generic event if generic event has symmetries meaning that the same generic event has been applied to same central atom
        if (
            neighbors_list is not None
        ):  # need neighbors list to remove symmetric duplicates
            counts = self.table.groupby(["atom_index", "num_reference_event"]).size()
            symmetric_num_ref = counts[counts > 1].index.get_level_values(1).unique()

            # Loop on all num_ref symmetric event
            for num_ref in symmetric_num_ref:
                subset = self.table[self.table["num_reference_event"] == num_ref]
                indices = subset.index.to_list()

                for i, idx in enumerate(indices):  # Loop over indice of subset
                    central_atom1 = subset.loc[idx, "atom_index"]
                    env1 = neighbors_list.get_neighbors(
                        "rcut", central_atom1
                    )  # list of atom in env1

                    for jdx in indices[i + 1 :]:  # to not compare two times
                        central_atom2 = subset.loc[jdx, "atom_index"]
                        if (
                            central_atom1 != central_atom2
                        ):  # if yes already done in part 1.
                            env2 = neighbors_list.get_neighbors("rcut", central_atom2)
                            # intersection of atoms in atomic environments
                            common = set(env1) & set(env2)

                            if not common:  # it's not a duplicate since they don't share atoms in their atomic environments
                                continue

                            if (
                                central_atom1 not in env2
                            ):  # TODO : To check, but should not be a duplicate
                                continue

                            # extract saddle positions
                            sad_pos1 = subset.loc[idx, "saddle_positions"]
                            sad_pos2 = subset.loc[jdx, "saddle_positions"]

                            # know we want to compare positions of share atoms, need to map.
                            map1 = {
                                a: k for k, a in enumerate(env1)
                            }  # so we know that the first position is atom xxx, ect, eg {345:0, 439:1, ....}
                            map2 = {a: k for k, a in enumerate(env2)}  # same for env2

                            # map atom when they are in common
                            index1 = [map1[a] for a in common]
                            index2 = [map2[a] for a in common]

                            # get subarray of sad_pos
                            sad_pos1 = sad_pos1[index1]
                            sad_pos2 = sad_pos2[index2]

                            # now we can compare
                            delr = compute_delr(sad_pos1, sad_pos2, cell)
                            if delr < self.config.psr.matching_score_thr:
                                duplicates.append(jdx)

        self.remove(duplicates)

    def save(self, outfile: str = "active_table.pickle") -> None:
        """Save the reference event table to a pickle file.

        Parameters
        ----------
        outfile : str, optional
            path to the output file, by default 'active_table.pickle'.

        """
        self.table.to_pickle(outfile)
