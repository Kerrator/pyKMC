"""Module implementing Classes to manage reference events and active events."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import TYPE_CHECKING

import pandas as pd
from .rate_constant import compute_rate_Eyring, create_rate_constant
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

    Attributes
    ----------
    rate_constant : RateConstant
        Rate facade built from ``config.rateconstant``; its backend decides
        whether the catalogue carries per-event prefactors.
    uses_prefactors : bool
        ``True`` for the htst/rpa styles. Selects the directional identity
        rules of :meth:`_admit_series`; constant-mode admission is unchanged.

    """

    def __init__(self, config: Config) -> None:
        self.config = config
        self.rate_constant = create_rate_constant(config.rateconstant)
        self.uses_prefactors: bool = bool(
            self.rate_constant.backend.requires_event_prefactors
        )
        self._initialize_table()

    def add_events(
        self, events: list[EventSearchOutput]
    ) -> Result[pd.DataFrame, ErrorInfo]:
        """Events events to the table dataframe.

        Parameters
        ----------
        events : list[EventSearchOutput]
            list of EventSearchOutput dataclass with events to be added to the table dataframe.

        Returns
        -------
        Result[pd.DataFrame, ErrorInfo]
            The results of the operation.

        """
        results_is_valid_events = []
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
            else:
                results_is_valid_events.append(res)
        # df_valid_events = self.get_valid_events(results_is_valid_events)

        # Check if events in results are not the same :

        # for df in df_valid_events:
        #    self.add(df)

        return results_is_valid_events

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

        dfevent_forward = pd.Series(
            {
                "idx_ref": -1,  # unknown yet
                "event_id": id_min1,
                "initial_positions": min1_positions[neighbor_list_forward],
                "saddle_positions": saddle_positions[neighbor_list_forward],
                "final_positions": min2_positions[neighbor_list_forward],
                "types": local_types_forward,
                "energy_barrier": dE_forward,
                "k": compute_rate_Eyring(dE_forward, self.config),
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
                "k": compute_rate_Eyring(dE_backward, self.config),
                "id_saddle": id_saddle,
                "id_final": id_min1,
                "move_atom_idx": np.where(neighbor_list_backward == index_move)[0][0],
                "sym_matrix": sym_matrix,
                "sym_perm": sym_perm,
                "idx_backward": -1,  # unknown yet
                "dra": dra_backward,
            }
        )

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
            self.table = pd.read_pickle(self.config.control.reference_table)
        else:
            self.table = pd.DataFrame(
                {
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
            )

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

        Parameters
        ----------
        outfile : str, optional
            path to the output file, by default 'reference_table.pickle'.

        """
        self.table.to_pickle(outfile)


class ActiveEventTable:
    """Store active events and manage them.

    Parameters
    ----------
    config : Config
        The atomic simulations configuration.
    event_dataframe : pd.DataFrame, optional
        An table with active event use to initialize the table. by default 'None'.

    """

    def __init__(
        self,
        config: Config,
        event_dataframe: pd.DataFrame = None,
        recycler: "Recycling | None" = None,
    ):
        self.config = config
        # Optional recycling plugin. If attached, `prune_for_recycling` keeps
        # the rows the recycler selects between KMC steps. If None, the table
        # is cleared at the end of each step (matching prior behavior).
        self.recycler = recycler

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
            self.table = pd.DataFrame(columns)

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

        dfactive = pd.Series(
            {
                "atom_index": event_refinement_output.central_atom_index,
                "saddle_positions": event_refinement_output.saddle_positions,
                "final_positions": event_refinement_output.min2_positions,
                "energy_barrier": event_refinement_output.dE_forward,
                "k": compute_rate_Eyring(
                    event_refinement_output.dE_forward, self.config
                ),
                "num_reference_event": event_refinement_output.num_reference_event,
                "refined": event_refinement_output.refined,
            }
        )
        return dfactive

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
