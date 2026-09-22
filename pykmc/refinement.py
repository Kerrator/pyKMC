"""Module implementing the Refinement class that deals with the event refinement procedure.

Which reference rows are refined
--------------------------------
Not every applicable reference row is sent to pARTn; an unselected application
enters the active table with the generic (reference) saddle as a
``refined="F"`` row.

``constant`` style
    The inherited barrier rule, unchanged: the fastest reference row whose
    multiplicity-weighted rate lies under ``refine_thr * ktot`` fixes a barrier
    threshold ``E_ref + 0.1 eV`` (:meth:`Refinement.get_energy_thr_refine`);
    rows above it are not refined. Every ``(atom, reference)`` pair already in
    the active table is skipped.

``htst``/``rpa`` styles (contracts 7f policy 3)
    ``refine_thr`` is a cumulative pre-dispatch rate coverage over an immutable
    candidate ledger: the retained valid active channels (their current site or
    reference rate, once each) plus every prospective PSR-valid
    (centre, reference, symmetry) application (the current resolved reference
    or ``k0`` fallback rate), with duplicate representations removed and
    negative or non-finite rates rejected. Contributions are summed per logical
    reference id, the groups are ranked by decreasing total with ascending id
    as the tie-break, and groups are selected in that order until the running
    sum reaches ``refine_thr * snapshot_total``; every group tied with the one
    at the cut is included, a zero total selects nothing and ``refine_thr >= 1``
    selects every positive group. The prospective channels of the selected
    groups are dispatched, including a retained ``refined="F"`` pair: each of
    its generic rows is matched to its own symmetric application by saddle
    geometry (within the overlay tolerance plus the recycler's drift, never
    to a sibling application) and superseded only by the successful
    refinement of that application (:attr:`Refinement.superseded_rows`); a
    failed or unmatched one keeps its generic row as the fallback and is
    reported as left unrefined; retained refined channels are never
    re-dispatched. The selected snapshot fraction, the fraction actually
    refined and the shortfall (failed refinements and selected retained rows
    that could not be re-dispatched) are reported separately in
    :attr:`Refinement.coverage`; the denominator is never recomputed from
    rates that arrive later.
"""

from .result import Result, EventRefinementOutput, ErrorInfo, ErrorType, Err, Ok
from .point_set_registration import PointSetRegistration, check_match
from .utils import geometry
from .config import Config
from .system import System
from .neighbors_list import NeighborsList
from .log import LogKMC
from .atomic_environment import AtomicEnvironment
from .manager import Manager
from .physics import (
    ConstraintViolationError,
    _indices,
    overlay_tolerance,
    resolve_event_constraints,
)
from .utils.geometry import compute_delr
import math
import numpy as np
import pandas as pd
import concurrent.futures


def _finite_rate(value) -> bool:
    """Return whether ``value`` is a finite, non-negative rate."""
    try:
        rate = float(value)
    except (TypeError, ValueError):
        return False
    return math.isfinite(rate) and rate >= 0.0


def _logical_id(value) -> int | None:
    """Return ``value`` as a logical id, or ``None`` when it names none.

    Accepts the integer-valued floats a pandas column may hold after a NaN
    visit and numpy integers; rejects non-finite and fractional values.
    """
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(number) or number != int(number):
        return None
    return int(number)


class _Channel:
    """One prospective htst/rpa refinement whose dispatch waits for the ledger.

    ``refine_single`` prepares the overlay (``working``: the full system with
    the PSR-transformed saddle pasted onto ``neighbors``) and its resolved
    constraints; ``Refinement._select_and_dispatch`` either submits it to
    pARTn or resolves it with the generic saddle. ``result()`` proxies the
    future either way, so the result loop of ``execute`` treats a channel
    like any other future.

    Attributes
    ----------
    at_idx : int
        Central atom (row index in the current system).
    ref : int
        Logical reference id (``idx_ref``).
    key : tuple
        ``(at_idx, ref, symmetry signature)``; a repeated key is a duplicate
        representation of the same channel.
    rate : float
        Pre-dispatch contribution (ps^-1): the reference row's resolved rate.
    retained_unrefined : bool
        This application is represented by a retained ``refined="F"`` row of
        the active table (label ``superseded_label``), matched by saddle
        geometry; the row is superseded when this channel refines.

    """

    def __init__(
        self,
        at_idx,
        dfevent: pd.Series,
        total_energy,
        working: np.ndarray,
        neighbors: "np.ndarray | list[int]",
        constraints,
        current_positions: np.ndarray,
        signature: tuple,
    ) -> None:
        self.at_idx = int(at_idx)
        self.ref = int(dfevent["idx_ref"])
        self.key = (self.at_idx, self.ref, signature)
        self.dfevent = dfevent
        self.total_energy = total_energy
        self.working = working
        self.neighbors = neighbors
        self.constraints = constraints
        self.current_positions = current_positions
        self.rate = dfevent["k"] if "k" in dfevent.index else float("nan")
        self.retained_unrefined = False
        self.superseded_label: int | None = None
        self.duplicate = False
        self.rejected = False
        self.dispatched = False
        self.ok = False
        self.future: concurrent.futures.Future | None = None

    def result(self):
        """Return the refinement result once the ledger has resolved the channel."""
        if self.future is None:
            raise RuntimeError(
                "refinement channel read before the ledger dispatched or declined it"
            )
        return self.future.result()


class Refinement:
    """Perfrom event refinements and deal with results.

    Parameters
    ----------
    config : Config
        The configuration of the simulation.
    loggers : LogKMC
        The logger of the KMC simulation.
    system : System
        The atomic system.
    neighbors_list : NeighborsList
        The neighbors lists of the system.
    atomic_environment : AtomicEnvironment
        The atomic environment of the system.
    engine : Engine
        The engine to use for the refinement.

    """

    def __init__(
        self,
        config: Config,
        loggers: LogKMC,
        system: System,
        neighbors_list: NeighborsList,
        atomic_environment: AtomicEnvironment,
        manager: Manager,
        global_constraints=None,
    ) -> None:
        self.config = config
        self.loggers = loggers
        self.system = system
        self.neighbors_list = neighbors_list
        self.atomic_environment = atomic_environment
        self.manager = manager
        self.results = None
        self.global_constraints = global_constraints
        # htst/rpa ledger report of the last execute (module docstring); None
        # in the constant style. Keys: style, applicable, target,
        # snapshot_total, n_channels, n_retained, n_prospective, n_groups,
        # n_excluded, n_rejected, n_duplicates_removed, n_predispatched,
        # selected_ids, selected_fraction, n_dispatched, n_dispatch_ok,
        # n_dispatch_failed, n_selected_unrefined, refined_fraction,
        # shortfall_fraction.
        self.coverage: dict | None = None
        # Labels of retained ``refined="F"`` rows whose own application was
        # refined successfully this execute: superseded by the refined output.
        self.superseded_rows: set[int] = set()
        self._carry_prefactors = False
        self._record_crop_ids = False
        self._ledger: dict | None = None

    def execute(
        self,
        df_reference_events: pd.DataFrame,
        total_energy,
        existing_pairs: set[tuple[int, int]] | None = None,
        retained_channels: pd.DataFrame | None = None,
    ) -> None:
        """Execute event refinements for each reference event in the df_reference_events dataframe.

        It stores the results of the event refinements in self.results. Which
        applications are sent to pARTn is decided per rate style (module
        docstring): the ``constant`` barrier rule, or the htst/rpa cumulative
        ledger whose report is left in :attr:`coverage`.

        Parameters
        ----------
        df_reference_events : pd.DataFrame
            dataframe of reference events to refine.
        total_energy : float
            Total energy of the current minimum (eV); without active volume
            the saddle energies are measured against it.
        existing_pairs : set[tuple[int, int]] | None, optional
            `(atom_index, num_reference_event)` pairs already present in the
            active event table (carried over from the previous step). The
            constant style skips them all; the htst/rpa styles skip only the
            pairs that ``retained_channels`` does not mark ``refined == "F"``.
        retained_channels : pd.DataFrame | None, optional
            The retained active rows as ledger channels
            (``ActiveEventTable.retained_channels``: ``label``, ``atom_index``,
            ``num_reference_event``, ``k``, ``refined``, ``crop_atom_ids``,
            ``saddle_positions``). htst/rpa only: each contributes its current
            rate once; a ``refined == "F"`` pair is re-dispatched when its
            group is selected and each generic row is superseded by the
            refinement of its own application (:attr:`superseded_rows`).
            Without it the ledger holds the prospective channels only.

        """
        existing_pairs = set(existing_pairs or ())
        self.results = []
        self.coverage = None
        self.superseded_rows = set()
        self._ledger = None
        # htst/rpa reference tables carry the resolved prefactor columns; the
        # inherited estimate travels with each refinement through its context.
        self._carry_prefactors = "nu0_status" in df_reference_events.columns
        # Refined rows record the source identities of their crop
        # (``crop_atom_ids``) so recycling and reconstruction can re-address
        # them after atoms move. ``System.index`` is optional: a constant-mode
        # reconstruction falls back to the rcut neighbours and skips the
        # record, while htst/rpa cannot recycle or reconstruct without stable
        # identities and must say so before any refinement is dispatched.
        self._record_crop_ids = getattr(self.system, "index", None) is not None
        if self._carry_prefactors and not self._record_crop_ids:
            raise RuntimeError(
                "Refinement in the {} rate style requires stable atom identities "
                "(System.index is None): site prefactors, recycling and "
                "reconstruction address refined rows by their crop identities. "
                "Build the System from a file or give it an index.".format(
                    self.config.rateconstant.style
                )
            )

        total_refinements, supposed_ktot = self.get_total_refinements_todo(
            df_reference_events
        )
        if self._carry_prefactors:
            # htst/rpa: no barrier threshold; the ledger decides once every
            # application has been matched (contracts 7f policy 3). A retained
            # unrefined pair stays a candidate.
            e_thr = None
            skipped_pairs = existing_pairs - self._redispatchable_pairs(
                retained_channels
            )
        else:
            e_thr = self.get_energy_thr_refine(df_reference_events, supposed_ktot)
            skipped_pairs = existing_pairs
        self.loggers.info("log", "\t :=> Refining {} events".format(total_refinements))

        all_futures = []
        future_context = {}  # mapping future -> contexte

        # Launch all refine Jobs
        for idx, dfevent in df_reference_events.iterrows():
            ###=>Find atoms with same atomic environment as the generic event
            atoms_refine_idx = self.atomic_environment.get_atoms_with_id(
                dfevent["event_id"]
            )
            ref_idx = int(dfevent["idx_ref"])

            for at_idx in atoms_refine_idx:
                if (at_idx, ref_idx) in skipped_pairs:
                    continue
                ###=>refine single generic
                futures = self.refine_single(
                    at_idx, dfevent, total_energy, future_context, e_thr
                )
                if isinstance(futures, list):  # If symmetries
                    all_futures.extend(futures)
                else:
                    all_futures.append(futures)

        if self._carry_prefactors:
            all_futures = self._select_and_dispatch(
                all_futures, future_context, retained_channels
            )

        # Get results and update values :
        for f in all_futures:
            # get results
            res = f.result()
            # get specific results info
            ctx = future_context[f]
            _ = future_context.pop(f)

            self.loggers.progress_bar(
                "progress", total_refinements - len(future_context), total_refinements
            )

            # update result
            if res.is_ok():
                res.ok_value().min2_positions = ctx["min2_positions"]
                res.ok_value().num_reference_event = ctx["num_reference_event"]
                if self._record_crop_ids:
                    res.ok_value().crop_atom_ids = tuple(
                        int(self.system.index[i]) for i in ctx["neighbors"]
                    )
                estimate = ctx["estimate"]
                res.ok_value().nu0_hz = estimate["nu0_hz"]
                res.ok_value().nu0_status = estimate["nu0_status"]
                res.ok_value().nu0_reason = estimate["nu0_reason"]
                res.ok_value().nu0_source = estimate["nu0_source"]
                if self._carry_prefactors and res.ok_value().refined == "T":
                    # htst/rpa: the site-specific prefactor request needs the
                    # full pARTn saddle, not the rcut crop stored on the row.
                    # Kept only in those styles so the constant path never
                    # holds an extra (N, 3) array per refinement.
                    res.ok_value().full_saddle_positions = (
                        res.ok_value().saddle_positions
                    )
                res.ok_value().saddle_positions = res.ok_value().saddle_positions[
                    ctx["neighbors"]
                ]
                # Now check if energy barrier consistent with generic one
                # TODO partn should not return different things depending on AV or not. We get the total energy at the saddle point or dE, but not both.
                # TODO and to be consistent, you should modify res.ok_value().E_saddle.
                if self.config.control.active_volume == True:
                    res.ok_value().dE_forward = res.ok_value().E_saddle
                else:
                    res.ok_value().dE_forward = res.ok_value().E_saddle - total_energy
                res = self.check_refinement_energy(
                    res,
                    abs(res.ok_value().dE_forward - ctx["reference_energy_barrier"]),
                    self.config.eventsearch.refined_energy_thr,
                )

            else:
                err = res.err_value()
                if not isinstance(err.variables, dict):
                    err.variables = {}
                err.variables["n_ref_event"] = ctx["num_reference_event"]

            self.results.append(res)
            if isinstance(f, _Channel) and f.dispatched:
                # Ledger bookkeeping: a dispatched channel either refined its
                # pre-dispatch contribution or left it as a shortfall; only a
                # refined application supersedes its own generic row.
                f.ok = res.is_ok()
                if f.ok and f.superseded_label is not None:
                    self.superseded_rows.add(f.superseded_label)
        if self._carry_prefactors:
            self._finish_coverage()

    def refine_single(
        self,
        at_idx: int,
        dfevent: pd.Series,
        total_energy: float,
        future_context: dict,
        e_thr: float | None,
    ) -> "concurrent.futures.Future | _Channel":
        """Perform a single reference event refinement.

        If a reference event has symmetries, it also refine those symmetric events.
        In the constant style each symmetric application is submitted (barrier
        at most ``e_thr``) or resolved with the generic saddle at once; in the
        htst/rpa styles each one becomes a ledger channel whose dispatch
        :meth:`execute` decides after every application has been matched. A
        symmetry listed twice on a reference row is one channel (the ledger
        removes the duplicate representation).

        Parameters
        ----------
        at_idx : int
            index of the central atom for which we perform the refinement.
        dfevent : pd.Series
            a Series of the reference event to refine.
        total_energy : float
            Total energy of the current minimum (eV).
        future_context : dict
            Mapping future -> context, filled for every returned future.
        e_thr : float or None
            Barrier threshold (eV) of :meth:`get_energy_thr_refine` in the
            constant style; ``None`` in the htst/rpa styles.

        Returns
        -------
        list
            Futures (constant style) or ledger channels (htst/rpa) resolving
            to ``Result[EventRefinementOutput, ErrorInfo]``; a failed match
            is a single resolved ``Err`` future.
        """

        ##=>PSR between generic event and at_idx environments
        result_psr = PointSetRegistration(
            self.config, self.system, dfevent, self.neighbors_list, at_idx
        ).match()

        ##=>Check results if match or match < matching_score
        result_psr = check_match(result_psr, self.config.psr.matching_score_thr)
        if not result_psr.is_ok():
            f: concurrent.futures.Future | _Channel = concurrent.futures.Future()
            f.set_result(result_psr)
            future_context[f] = {"num_reference_event": dfevent["idx_ref"]}
            return f

        else:
            ##=> Get Saddle positions to refine

            output_psr = result_psr.ok_value()

            displacement_saddle = (
                dfevent.at["saddle_positions"].copy()
                - dfevent.at["initial_positions"].copy()
            )
            displacement_final = (
                dfevent.at["final_positions"].copy()
                - dfevent.at["initial_positions"].copy()
            )

            # all_results = []
            futures = []

            ###=>Apply symmetries

            current_positions = (
                self.system.positions.copy()
            )  # save to restore system after

            try:
                constraints = resolve_event_constraints(
                    self.config,
                    current_positions,
                    self.system.types,
                    self.system.cell,
                    self.system.pbc,
                    at_idx,
                    self.system.index,
                    user_constraints=self.global_constraints,
                )
            except ConstraintViolationError as exc:
                # A user-frozen atom drifted from its initialisation reference:
                # the same recoverable Err as an overlay that moves one, never
                # a bare ValueError out of the KMC loop (contracts 7f policy 5).
                self.loggers.warning(
                    "log",
                    "\t :=> Reference event {} on atom {}: a user-fixed atom has "
                    "drifted from its initialisation reference; refinement "
                    "skipped ({})".format(dfevent["idx_ref"], at_idx, exc),
                )
                f = concurrent.futures.Future()
                f.set_result(
                    Err(
                        ErrorInfo(
                            type=ErrorType.RECONSTRUCTION_INVALID_EVENT_DATA,
                            message="user-fixed reference coordinate drifted "
                            "since initialisation: {}".format(exc),
                        )
                    )
                )
                future_context[f] = {"num_reference_event": dfevent["idx_ref"]}
                return [f]
            tolerance = overlay_tolerance(self.config)
            for sym_matrix, perm_matrix in zip(
                dfevent.at["sym_matrix"], dfevent.at["sym_perm"], strict=False
            ):
                ###=> Apply symmetries to displacements
                new_displacement_saddle = geometry.transform_positions(
                    displacement_saddle, sym_matrix, 0, perm_matrix
                )
                new_displacement_final = geometry.transform_positions(
                    displacement_final, sym_matrix, 0, perm_matrix
                )

                ###=> Get symmetric saddle and final positions
                saddle_positions = (
                    dfevent.at["initial_positions"].copy() + new_displacement_saddle
                )
                final_positions = (
                    dfevent.at["initial_positions"].copy() + new_displacement_final
                )

                ###=> Apply PSR to the saddle and final positions do get specific saddle and final positions (before refinement)
                new_positions_saddle = geometry.transform_positions(
                    saddle_positions,
                    output_psr.rotation_matrix,
                    output_psr.translation_matrix,
                    output_psr.permutation_matrix,
                )
                new_positions_final = geometry.transform_positions(
                    final_positions,
                    output_psr.rotation_matrix,
                    output_psr.translation_matrix,
                    output_psr.permutation_matrix,
                )
                neighbors = self.neighbors_list.get_neighbors("rcut", at_idx).copy()

                ###=> move the system to the saddle point
                working = current_positions.copy()
                working[neighbors] = new_positions_saddle
                final = current_positions.copy()
                final[neighbors] = new_positions_final
                # Only user-declared fixed atoms are a coordinate contract: a
                # PSR residual within the matching tolerance is re-clamped, a
                # larger one is a different event and is reported as an Err
                # (never a ValueError out of the KMC loop). The AV shell is
                # placed as given and held by fix setforce (contracts 7f
                # policy 5).
                try:
                    constraints.validate_positions(
                        working, tolerance=tolerance, user_only=True
                    )
                    constraints.validate_positions(
                        final, tolerance=tolerance, user_only=True
                    )
                except ConstraintViolationError as exc:
                    self.loggers.warning(
                        "log",
                        "\t :=> Reference event {} on atom {} moves a user-fixed "
                        "atom beyond {} A; refinement skipped ({})".format(
                            dfevent["idx_ref"], at_idx, tolerance, exc
                        ),
                    )
                    f = concurrent.futures.Future()
                    f.set_result(
                        Err(
                            ErrorInfo(
                                type=ErrorType.RECONSTRUCTION_INVALID_EVENT_DATA,
                                message="generic event changes a user-fixed "
                                "reference coordinate: {}".format(exc),
                            )
                        )
                    )
                    futures.append(f)
                    future_context[f] = {"num_reference_event": dfevent["idx_ref"]}
                    continue
                working = constraints.protect_positions(working, user_only=True)
                final = constraints.protect_positions(final, user_only=True)
                new_positions_final = final[neighbors]
                if self._carry_prefactors:
                    # htst/rpa: the ledger dispatches or declines the channel
                    # once every application of this execute has been matched.
                    signature = (
                        np.round(np.asarray(sym_matrix, dtype=float), 8).tobytes(),
                        np.asarray(perm_matrix).tobytes(),
                    )
                    f = _Channel(
                        at_idx,
                        dfevent,
                        total_energy,
                        working,
                        neighbors,
                        constraints,
                        current_positions,
                        signature,
                    )
                elif (
                    dfevent.at["energy_barrier"] > e_thr
                ):  # We dont refine, we use generic date
                    # create a fake future to store the result
                    f = concurrent.futures.Future()
                    f.set_result(
                        Ok(
                            self._generic_output(
                                at_idx, dfevent, total_energy, working, constraints
                            )
                        )
                    )

                else:  # we refine
                    f = self._submit(
                        at_idx, working, neighbors, constraints, current_positions
                    )
                futures.append(f)

                # NOTE: TEMPORARY, NEED TO FIND A BETTER WAY
                future_context[f] = {
                    "min2_positions": geometry.wrap_positions(
                        new_positions_final,
                        cell=self.system.cell,
                        pbc=self.system.pbc,
                    ),
                    "num_reference_event": dfevent["idx_ref"],
                    "reference_energy_barrier": dfevent["energy_barrier"],
                    "neighbors": neighbors.copy(),
                    "estimate": self._inherited_estimate(dfevent),
                }

            return futures

    def _generic_output(
        self, at_idx, dfevent: pd.Series, total_energy, working, constraints
    ) -> EventRefinementOutput:
        """Return the unrefined output of an application: the generic saddle."""
        # TODO I don't like that we don't gibe the same information to E_saddle depending on AV or not
        return EventRefinementOutput(
            central_atom_index=at_idx,
            saddle_positions=working.copy(),
            E_saddle=dfevent["energy_barrier"]
            if self.config.control.active_volume
            else total_energy + dfevent["energy_barrier"],
            refined="F",
            constraints=constraints,
        )

    def _submit(self, at_idx, working, neighbors, constraints, current_positions):
        """Queue the pARTn refinement of one application and return its future."""
        # TODO : same here, we should send the same information to the partn_refine function
        # TODO : when AV, partn_refine needs the minimum positions to compute the initial energy with AV
        # TODO : but this is the third parameter here, and without AV, the third parameter is the saddle positions.
        # TODO : and with AV, you only need to send saddle positions in the rcut, while without we send all saddle positions, this is just too confusing
        if self.config.control.active_volume == True:
            # add a job to manager queue
            return self.manager.partn_refine(
                config=self.config,
                central_atom_idx=at_idx,
                positions=current_positions.copy(),
                cell=self.system.cell,
                types=self.system.types.copy(),
                saddle_idx=neighbors.copy(),
                saddle_positions=working[neighbors].copy(),
                constraints=constraints,
                user_constraints=self.global_constraints,
            )  # send copy not reference !
        # add a job to manager queue
        return self.manager.partn_refine(
            config=self.config,
            central_atom_idx=at_idx,
            positions=working.copy(),
            types=self.system.types.copy(),
            cell=self.system.cell,
            saddle_idx=neighbors.copy(),
            constraints=constraints,
            user_constraints=self.global_constraints,
        )  # send copy not reference !

    @staticmethod
    def _redispatchable_pairs(
        retained_channels: pd.DataFrame | None,
    ) -> set[tuple[int, int]]:
        """Return the retained ``refined == "F"`` pairs (candidates again)."""
        if retained_channels is None or len(retained_channels) == 0:
            return set()
        pairs: set[tuple[int, int]] = set()
        for atom, ref, refined in zip(
            retained_channels["atom_index"],
            retained_channels["num_reference_event"],
            retained_channels["refined"],
            strict=True,
        ):
            if str(refined) != "F":
                continue
            atom_id, ref_id = _logical_id(atom), _logical_id(ref)
            if atom_id is not None and ref_id is not None:
                pairs.add((atom_id, ref_id))
        return pairs

    def _reject_rate(self, kind: str, atom: int, ref: int, rate) -> None:
        self.loggers.warning(
            "log",
            "\t :=> [{}] refinement ledger: rejected the rate {!r} of the {} "
            "channel (atom {}, reference {}); a negative or non-finite "
            "contribution never enters the snapshot".format(
                self.config.rateconstant.style, rate, kind, atom, ref
            ),
        )

    def _match_retained_rows(
        self, rows: list[dict], channels: list[_Channel]
    ) -> list[tuple[dict, _Channel]]:
        """Pair retained rows of one ``(atom, reference)`` with their applications.

        Each retained row stores the generic (or refined) saddle crop of one
        symmetric application; the prospective channel of the same
        application is the one whose generic overlay is nearest to it
        (``compute_delr`` over the crop aligned by stable atom identities)
        and within :meth:`_matching_tolerance`: the overlay of the same
        application differs from the stored crop by the recycled atoms'
        drift only, a sibling application by the saddle displacement itself.
        Greedy one-to-one assignment by increasing distance; a row whose crop
        cannot be aligned (no identities or saddle stored) is matched only
        when the pair has exactly one row and one channel, and a row without
        a channel within tolerance stays a retained channel.

        Returns
        -------
        list[tuple[dict, _Channel]]
            The assigned ``(row, channel)`` pairs.

        """
        alignable = [
            row["saddle_positions"] is not None and row["crop_atom_ids"] is not None
            for row in rows
        ]
        if len(rows) == 1 and len(channels) == 1 and not alignable[0]:
            return [(rows[0], channels[0])]
        tolerance = self._matching_tolerance()
        distances: list[tuple[float, int, int]] = []
        for i, row in enumerate(rows):
            if not alignable[i]:
                continue
            saddle = row["saddle_positions"]
            ids = row["crop_atom_ids"]
            try:
                ids_row = _indices(ids)
                row_crop = np.asarray(saddle, dtype=float)
            except (TypeError, ValueError):
                continue
            for j, channel in enumerate(channels):
                ids_channel = tuple(
                    int(self.system.index[k]) for k in channel.neighbors
                )
                if set(ids_row) != set(ids_channel) or len(ids_row) != len(ids_channel):
                    continue
                order = [ids_channel.index(k) for k in ids_row]
                crop = channel.working[channel.neighbors][order]
                if crop.shape != row_crop.shape:
                    continue
                delr = float(
                    compute_delr(row_crop, crop, self.system.cell, self.system.pbc)
                )
                if delr <= tolerance:
                    distances.append((delr, i, j))
        assigned: list[tuple[dict, _Channel]] = []
        used_rows: set[int] = set()
        used_channels: set[int] = set()
        for _delr, i, j in sorted(distances):
            if i in used_rows or j in used_channels:
                continue
            used_rows.add(i)
            used_channels.add(j)
            assigned.append((rows[i], channels[j]))
        return assigned

    def _matching_tolerance(self) -> float:
        """Return the largest overlay distance that names the same application.

        The PSR overlay tolerance (``psr.matching_score_thr``, the residual a
        registration may leave) plus twice the recycler's ``movement_thr``
        (each recycled atom may have drifted that much since the row was
        built); zero drift allowance without recycling, where no row is
        retained.
        """
        tolerance = overlay_tolerance(self.config)
        recycling = getattr(self.config, "eventrecycling", None)
        if getattr(self.config.control, "recycle", False) and recycling is not None:
            tolerance += 2.0 * float(recycling.movement_thr)
        return tolerance

    def _select_and_dispatch(
        self,
        all_futures: list,
        future_context: dict,
        retained_channels: pd.DataFrame | None,
    ) -> list:
        """Build the immutable ledger, select by cumulative coverage, dispatch.

        Returns the resolved sequence of futures/channels for the result loop:
        duplicate representations are removed, channels with an invalid rate
        are replaced by an ``Err`` future, selected channels are submitted to
        pARTn and the others resolved with the generic saddle. The report is
        left in :attr:`coverage` (completed by :meth:`_finish_coverage`).
        """
        style = self.config.rateconstant.style
        channels = [f for f in all_futures if isinstance(f, _Channel)]
        # Prospective channels: PSR-valid applications, each representation
        # once (a repeated reference row or symmetry is the same channel).
        seen: set[tuple] = set()
        kept: list[_Channel] = []
        duplicates = 0
        for channel in channels:
            if channel.key in seen:
                channel.duplicate = True
                duplicates += 1
                continue
            seen.add(channel.key)
            kept.append(channel)
        # Their rates, before any retained row is matched to them: a row
        # whose application lost its rate this step stays a retained channel.
        rejected = 0
        counted: list[_Channel] = []
        for channel in kept:
            if not _finite_rate(channel.rate):
                channel.rejected = True
                rejected += 1
                self._reject_rate(
                    "prospective", channel.at_idx, channel.ref, channel.rate
                )
                continue
            channel.rate = float(channel.rate)
            counted.append(channel)
        by_pair: dict[tuple[int, int], list[_Channel]] = {}
        for channel in counted:
            by_pair.setdefault((channel.at_idx, channel.ref), []).append(channel)
        # Retained channels: their current rate, once each. A retained row of
        # a pair that is matched again is represented by the channel of its
        # own application: a generic row is re-dispatched with it (and
        # superseded only by its success), a refined row keeps the channel
        # from being dispatched at all.
        retained: list[tuple[int, float, bool]] = []
        candidates: dict[tuple[int, int], list[dict]] = {}
        if retained_channels is not None and len(retained_channels) > 0:
            has_geometry = "crop_atom_ids" in retained_channels.columns
            for _, entry in retained_channels.iterrows():
                atom_id = _logical_id(entry["atom_index"])
                ref_id = _logical_id(entry["num_reference_event"])
                rate = entry["k"]
                if atom_id is None or ref_id is None:
                    rejected += 1
                    self._reject_rate(
                        "retained",
                        entry["atom_index"],
                        entry["num_reference_event"],
                        rate,
                    )
                    continue
                row = {
                    "label": entry["label"] if "label" in entry.index else None,
                    "ref": ref_id,
                    "rate": rate,
                    "unrefined": str(entry["refined"]) == "F",
                    "crop_atom_ids": entry["crop_atom_ids"] if has_geometry else None,
                    "saddle_positions": entry["saddle_positions"]
                    if has_geometry
                    else None,
                }
                if (atom_id, ref_id) in by_pair:
                    candidates.setdefault((atom_id, ref_id), []).append(row)
                    continue
                if not _finite_rate(rate):
                    rejected += 1
                    self._reject_rate("retained", atom_id, ref_id, rate)
                    continue
                retained.append((ref_id, float(rate), not row["unrefined"]))
        for pair, rows in candidates.items():
            assigned = self._match_retained_rows(rows, by_pair[pair])
            matched_rows = set()
            for row, channel in assigned:
                matched_rows.add(id(row))
                duplicates += 1
                if row["unrefined"]:
                    channel.retained_unrefined = True
                    label = _logical_id(row["label"])
                    channel.superseded_label = label
                    continue
                # Already refined: the retained row is the channel.
                channel.duplicate = True
                if not _finite_rate(row["rate"]):
                    rejected += 1
                    self._reject_rate("retained", pair[0], pair[1], row["rate"])
                    continue
                retained.append((pair[1], float(row["rate"]), True))
            for row in rows:
                if id(row) in matched_rows:
                    continue
                if not _finite_rate(row["rate"]):
                    rejected += 1
                    self._reject_rate("retained", pair[0], pair[1], row["rate"])
                    continue
                retained.append((pair[1], float(row["rate"]), not row["unrefined"]))
                if row["unrefined"]:
                    self.loggers.warning(
                        "log",
                        "\t :=> [{}] refinement ledger: retained generic row {} "
                        "(atom {}, reference {}) matches none of the {} "
                        "application(s) dispatched for its pair within {:.3g} A; "
                        "it stays as the fallback and is left unrefined".format(
                            style,
                            row["label"],
                            pair[0],
                            pair[1],
                            len(by_pair[pair]),
                            self._matching_tolerance(),
                        ),
                    )
        counted = [channel for channel in counted if not channel.duplicate]
        # Group by logical reference id; rank by decreasing total, ascending id.
        groups: dict[int, float] = {}
        retained_refined: dict[int, float] = {}
        retained_unrefined: dict[int, list[float]] = {}
        for ref_id, rate, refined in retained:
            groups[ref_id] = groups.get(ref_id, 0.0) + rate
            if refined:
                retained_refined[ref_id] = retained_refined.get(ref_id, 0.0) + rate
            else:
                retained_unrefined.setdefault(ref_id, []).append(rate)
        for channel in counted:
            groups[channel.ref] = groups.get(channel.ref, 0.0) + channel.rate
        ranked = sorted(groups.items(), key=lambda item: (-item[1], item[0]))
        positive = [(ref_id, total) for ref_id, total in ranked if total > 0.0]
        snapshot_total = 0.0
        for _ref_id, total in positive:
            snapshot_total += total
        threshold = float(self.config.control.refine_thr)
        selected: list[int] = []
        if snapshot_total > 0.0:
            if threshold >= 1.0:
                selected = [ref_id for ref_id, _total in positive]
            else:
                target = threshold * snapshot_total
                cumulative = 0.0
                boundary = None
                for ref_id, total in positive:
                    if boundary is not None:
                        if total == boundary:
                            selected.append(ref_id)  # tied at the cut
                            continue
                        break
                    cumulative += total
                    selected.append(ref_id)
                    if cumulative >= target:
                        boundary = total
        selected_set = set(selected)
        selected_rate = 0.0
        for ref_id, total in positive:
            if ref_id in selected_set:
                selected_rate += total
        # Resolve every channel in dispatch order.
        resolved: list = []
        dispatched: list[_Channel] = []
        excluded = 0
        predispatched = 0
        for f in all_futures:
            if not isinstance(f, _Channel):
                # A resolved Err (PSR or constraint failure) is excluded from
                # the snapshot; a pending or Ok future comes from a
                # substituted dispatcher and is collected outside the ledger.
                if not f.done() or f.result().is_ok():
                    predispatched += 1
                else:
                    excluded += 1
                resolved.append(f)
                continue
            if f.duplicate:
                future_context.pop(f, None)
                continue
            if f.rejected:
                err: concurrent.futures.Future = concurrent.futures.Future()
                err.set_result(
                    Err(
                        ErrorInfo(
                            type=ErrorType.REFINEMENT_INVALID_RATE,
                            message="reference rate {!r} is negative or non-finite; "
                            "the channel was not dispatched".format(f.rate),
                        )
                    )
                )
                future_context[err] = {"num_reference_event": f.dfevent["idx_ref"]}
                future_context.pop(f, None)
                resolved.append(err)
                continue
            if f.ref in selected_set:
                self._dispatch_channel(f)
                dispatched.append(f)
            else:
                self._decline_channel(f)
            resolved.append(f)
        applicable = snapshot_total > 0.0
        selected_unrefined = [
            rate for ref_id in selected for rate in retained_unrefined.get(ref_id, ())
        ]
        self._ledger = {
            "snapshot_total": snapshot_total,
            "groups": groups,
            "selected": selected,
            "selected_rate": selected_rate,
            "retained_refined": retained_refined,
            "dispatched": dispatched,
        }
        self.coverage = {
            "style": style,
            "applicable": applicable,
            "target": threshold,
            "snapshot_total": snapshot_total,
            "n_channels": len(retained) + len(counted),
            "n_retained": len(retained),
            "n_prospective": len(counted),
            "n_groups": len(groups),
            "n_excluded": excluded,
            "n_rejected": rejected,
            "n_duplicates_removed": duplicates,
            "n_predispatched": predispatched,
            "selected_ids": tuple(selected),
            "selected_fraction": selected_rate / snapshot_total if applicable else None,
            "n_dispatched": len(dispatched),
            "n_dispatch_ok": 0,
            "n_dispatch_failed": 0,
            "n_selected_unrefined": len(selected_unrefined),
            "refined_fraction": None,
            "shortfall_fraction": None,
        }
        self.loggers.info(
            "log",
            "\t :=> [{}] refinement ledger: {} channels ({} retained, {} "
            "prospective), {} reference groups, snapshot total {:.4e} ps^-1; "
            "excluded {} (PSR/constraints), rejected {} (invalid rates), "
            "duplicates removed {}, collected outside the ledger {}; selected "
            "{} groups covering {} (target {:.4f}); dispatching {} channels, "
            "{} selected retained rows cannot be re-dispatched".format(
                style,
                self.coverage["n_channels"],
                len(retained),
                len(counted),
                len(groups),
                snapshot_total,
                excluded,
                rejected,
                duplicates,
                predispatched,
                len(selected),
                "{:.2f} %".format(100.0 * self.coverage["selected_fraction"])
                if applicable
                else "not applicable",
                threshold,
                len(dispatched),
                len(selected_unrefined),
            ),
        )
        return resolved

    def _dispatch_channel(self, channel: _Channel) -> None:
        """Submit a selected channel to pARTn."""
        channel.future = self._submit(
            channel.at_idx,
            channel.working,
            channel.neighbors,
            channel.constraints,
            channel.current_positions,
        )
        channel.dispatched = True

    def _decline_channel(self, channel: _Channel) -> None:
        """Resolve an unselected channel with its generic saddle."""
        f = concurrent.futures.Future()
        f.set_result(
            Ok(
                self._generic_output(
                    channel.at_idx,
                    channel.dfevent,
                    channel.total_energy,
                    channel.working,
                    channel.constraints,
                )
            )
        )
        channel.future = f

    def _finish_coverage(self) -> None:
        """Complete :attr:`coverage` with the dispatch outcome and log it.

        ``refined_fraction`` is the share of the snapshot that ends the step
        validly refined (retained refined rows of every group plus the
        successful refinements); ``shortfall_fraction`` is the share of the
        selected groups that does not: failed refinements and selected
        retained generic rows that could not be re-dispatched.
        """
        if self.coverage is None or self._ledger is None:
            return
        ledger = self._ledger
        total = ledger["snapshot_total"]
        ok_by_group: dict[int, float] = {}
        n_ok = n_failed = 0
        for channel in ledger["dispatched"]:
            if channel.ok:
                n_ok += 1
                ok_by_group[channel.ref] = (
                    ok_by_group.get(channel.ref, 0.0) + channel.rate
                )
            else:
                n_failed += 1
        self.coverage["n_dispatch_ok"] = n_ok
        self.coverage["n_dispatch_failed"] = n_failed
        style = self.coverage["style"]
        if not self.coverage["applicable"]:
            self.loggers.info(
                "log",
                "\t :=> [{}] refinement coverage: not applicable (snapshot total "
                "0, nothing dispatched)".format(style),
            )
            return
        refined_rate = sum(ledger["retained_refined"].values()) + sum(
            ok_by_group.values()
        )
        refined_selected = 0.0
        for ref_id in ledger["selected"]:
            refined_selected += ledger["retained_refined"].get(ref_id, 0.0)
            refined_selected += ok_by_group.get(ref_id, 0.0)
        shortfall_rate = max(ledger["selected_rate"] - refined_selected, 0.0)
        refined = refined_rate / total
        shortfall = shortfall_rate / total
        self.coverage["refined_fraction"] = refined
        self.coverage["shortfall_fraction"] = shortfall
        self.loggers.info(
            "log",
            "\t :=> [{}] refinement coverage: selected {:.2f} % of the "
            "pre-dispatch snapshot, refined {:.2f} % (retained refined rows and "
            "successful refinements), shortfall {:.2f} % ({} failed of {} "
            "dispatched, {} selected retained rows left unrefined)".format(
                style,
                100.0 * self.coverage["selected_fraction"],
                100.0 * refined,
                100.0 * shortfall,
                n_failed,
                self.coverage["n_dispatched"],
                self.coverage["n_selected_unrefined"],
            ),
        )

    def _inherited_estimate(self, dfevent: pd.Series) -> dict:
        """Return the reference prefactor estimate a refinement inherits.

        Parameters
        ----------
        dfevent : pd.Series
            The reference event row being refined.

        Returns
        -------
        dict
            ``nu0_hz`` (Hz, only when the reference status is ``ok``),
            ``nu0_status``, ``nu0_reason`` and ``nu0_source`` (``reference``);
            every value is ``None`` in the constant style, whose reference
            table has no prefactor columns.

        """
        if not self._carry_prefactors:
            return {
                "nu0_hz": None,
                "nu0_status": None,
                "nu0_reason": None,
                "nu0_source": None,
            }
        status = str(dfevent["nu0_status"])
        ok = status == "ok"
        return {
            "nu0_hz": float(dfevent["nu0"]) if ok else None,
            "nu0_status": status,
            "nu0_reason": "" if ok else str(dfevent["nu0_reason"]),
            "nu0_source": "reference",
        }

    def check_refinement_energy(
        self,
        result_refine: Result[EventRefinementOutput, ErrorInfo],
        energy_mismatch: float,
        refined_energy_thr: float,
    ) -> Result[EventRefinementOutput, ErrorInfo]:
        """Check if the energy barrier of the refinement correspond the one of the reference event.

        Parameters
        ----------
        result_refine : Result[EventRefinementOutput, ErrorInfo]
            Results of the refinement procedure.
        energy_mismatch : float
            Difference between the reference event energy barrier and the refine one.
        refined_energy_thr : float
            maximum allowed difference (in eV) between a reference event's initial barrier energy and its refined barrier energy

        Returns
        -------
        Result[EventRefinementOutput, ErrorInfo]
            list of results of the procedure.

        """
        if energy_mismatch > refined_energy_thr:
            return Err(
                ErrorInfo(
                    type=ErrorType.REFINEMENT_INVALID_ENERGY_BARRIER,
                    message="refinement energy barrier does not match reference one",
                )
            )
        else:
            return result_refine

    def get_total_refinements_todo(self, df_reference_events: pd.DataFrame) -> int:
        """Give the total number of refinements to do.

        Parameters
        ----------
        df_reference_events : pd.DataFrame
            dataframe with reference events to refine.

        Returns
        -------
        int
            total number of refinements to do.

        """
        total = 0
        supposed_ktot = 0
        for _idx, dfevent in df_reference_events.iterrows():
            ###=>Find atoms with same atomic environment as the generic event
            n_atoms = len(
                self.atomic_environment.get_atoms_with_id(dfevent["event_id"])
            ) * len(dfevent["sym_matrix"])
            total += n_atoms
            supposed_ktot += dfevent.at["k"] * n_atoms
        return total, supposed_ktot

    def get_energy_thr_refine(self, df_reference_events, supposed_ktot):
        tol = self.config.control.refine_thr
        k_thr = supposed_ktot * tol

        # get energy corresponding to the first k value just under k_thr
        mask = df_reference_events["k"] < k_thr
        if mask.any():
            e_value = (
                df_reference_events.loc[mask]
                .sort_values("k")
                .iloc[-1]["energy_barrier"]
            )
        else:  # refine no event
            e_value = 0.0
        e_value += 0.1  # to be sure want using condition
        return e_value

    def get_successes_results(self) -> list[EventRefinementOutput]:
        """Return successful results.

        Returns
        -------
        list[EventRefinementOutput]
            list of EventRefinementOutpout dataclass with refine event's informations.

        """
        return [e.ok_value() for e in self.results if e.is_ok()]
