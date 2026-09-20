"""Module implementing Classes to manage reference events and active events."""

from __future__ import annotations

import logging
import math
from collections.abc import Iterable
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
    from .physics import DescriptorComparison, PhysicalDescriptor
    from .event_recycling import Recycling
    from .htst.result import DirectionalPrefactor, EventPrefactors
    from .rate_constant.prefactors import PrefactorService

logger = logging.getLogger("log")
"""HTST lifecycle diagnostics go to the KMC ``log`` logger (see ``pykmc.log``)."""

TABLE_SCHEMA_VERSION: int = 2
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
NU0_STALE: str = "stale"
NU0_STATUSES: tuple[str, ...] = (
    NU0_OK,
    NU0_REJECTED,
    NU0_PENDING,
    NU0_LEGACY,
    NU0_STALE,
)
"""``nu0_status`` values: accepted estimate, scientific rejection (``k0``),
not yet resolved (``k0`` placeholder), loaded without provenance (``k0``),
loaded under different kernel settings (``k0``; the stored estimate is
discarded on reload, see :meth:`ReferenceEventTable._load`). Every consumer
treats ``stale`` exactly like ``legacy``: there is no estimate to inherit."""

RELOAD_SETTINGS: tuple[str, ...] = (
    "free_radius",
    "free_region_center",
    "fd_step",
    "zone_radius",
    "premin",
)
"""Kernel settings compared on reload (contracts section 7d, F2).

A stored table whose ``settings`` differ from the current configuration in any
of these invalidates every accepted row (``stale``). The acceptance window is
not a compatibility setting: it is re-applied to every accepted row on reload
instead. A stored table written before ``free_region_center`` existed counts
as ``"min1"`` (the centring of that era).
"""

SOURCE_REFERENCE: str = "reference"
SOURCE_SITE: str = "site"
SOURCE_K0: str = "k0"
"""``nu0_source`` values of an active row: inherited reference estimate,
site-specific estimate at the refined saddle, or the ``k0`` fallback."""

SELF_REVERSE_NU0_RTOL: float = 0.05
"""Required agreement of two accepted directional Vineyard frequencies.

This existing 5% tolerance is not a validated calibration. Frequency agreement
alone never proves identity: one map must preserve the complete physical event,
its constraints and actual vibrational selection. Estimates are never averaged.
"""

SELF_REVERSE_BARRIER_TOL: float = 0.01
"""Maximum barrier gap (eV) for a proposed directional merge.

This numerical equality gate is necessary, but equal barriers do not establish
symmetry-equivalent minima. Unproven events retain reciprocal directional rows.
"""

SAME_TOPOLOGY_BARRIER_TOL: float = 0.25
"""Coarse barrier window (eV) for a geometric lookup; not an identity proof."""

GEOMETRY_INVALIDATING_REJECTIONS: frozenset[str] = frozenset(
    {
        "nonstationary_geometry",
        "unstable_minimum",
        "saddle_not_first_order",
        "mode_count_mismatch",
        "nonfinite_hessian",
    }
)
"""``PrefactorRejection`` codes that disqualify the recomputed energies.

A recomputation under the current physics that reports one of these has
found the saved triplet not to be a minimum-saddle-minimum of the current
potential, so its energy differences are not a barrier: the catalogued
barrier is kept and the row falls back to ``k0``. The other codes (window,
empty free region, non-finite prefactor) leave the stationary geometry
intact and its recomputed barrier is adopted.
"""


@dataclass(frozen=True)
class EventAdmission:
    """Provisional rows and candidate links, before actual prefactor resolution.

    HTST/RPA admit two reciprocal provisional rows, or one forward row when
    ``reverse_idx_ref`` names the catalogued row the geometric gate matched
    for the backward direction (contracts 7f policy 6, as in constant mode).
    ``same_topology`` and ``self_reverse_candidate`` propose self-reversal
    only; the whole-event proof after resolution decides the collapse, and an
    unproven candidate falls back to one self-linked forward row. Constant
    mode keeps its original row policy.
    """

    frame: pd.DataFrame
    reverse_idx_ref: int | None = None
    same_topology: bool = False
    self_reverse_candidate: bool = False


@dataclass(frozen=True)
class RemovedReferences:
    """What :meth:`ReferenceEventTable.remove` took out of the catalogue.

    Attributes
    ----------
    idx_refs : tuple[int, ...]
        Every logical id that left the table, sorted ascending: the requested
        ids, the rows their ``idx_backward`` named and, in htst/rpa, the
        reverse-link closure. A requested id that had no row is not reported.
    event_ids : tuple[str, ...]
        ``event_id`` (initial topology) of each removed row, aligned with
        ``idx_refs``; ``KMC.run`` forgets these topologies in
        ``visited_environments``.

    """

    idx_refs: tuple[int, ...] = ()
    event_ids: tuple[str, ...] = ()

    def __len__(self) -> int:
        """Return the number of removed rows."""
        return len(self.idx_refs)


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
        agree within :data:`SELF_REVERSE_NU0_RTOL`. A rejected or skipped
        direction never agrees: equal fallbacks do not demonstrate equal
        spectra.

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
    returns, so refinement never reads an unresolved reference ``k``. The
    admission duplicate gate is :meth:`find_matching_event` in every style
    (contracts 7f policy 6), so a catalogued event stays a duplicate whatever
    its prefactor status. A self-reverse candidate starts with two provisional
    rows; only accepted, agreeing estimates and a common full physical map
    permit a proven collapse, and an unproven candidate collapses back to one
    self-linked forward row with the backward estimate kept in the archive.
    Two rows sharing one ``event_id`` are never exposed as selectable
    channels of one site. Both original producing records remain archived.

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
        self._fresh_calculations: set[tuple[int, str]] = set()
        # idx_ref -> cheap validation key (see _ensure_current_estimate); a hit
        # means the row was validated under this exact service, physics,
        # producing record, row content and rate policy, so selection skips
        # the full-source rebuild and hashing. Row writers pop their entry.
        self._resolved_contexts: dict[int, tuple] = {}
        self._recomputed: dict[str, Any] = {}
        self.prefactor_archive = None
        if self.uses_prefactors:
            from .htst.catalogue import PrefactorArchive

            self.prefactor_archive = PrefactorArchive()
        self._initialize_table()

    @property
    def current_descriptor(self) -> PhysicalDescriptor | None:
        """Return service context without assigning it to historical estimates."""
        return (
            None
            if self.prefactor_service is None
            else self.prefactor_service.current_descriptor
        )

    def compare_physics(
        self, producing: PhysicalDescriptor | None
    ) -> DescriptorComparison:
        """Compare producing physics using the common descriptor contract.

        This is a comparison interface. Persistence and selection callers must
        separately preserve provenance and act on unknown/incompatible results.
        """
        from .physics import DescriptorComparison

        current = self.current_descriptor
        if current is None:
            return DescriptorComparison("unknown", ("missing current descriptor",))
        return current.compare(producing)

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
            One result per event: surviving resolved rows or the rejection.

        """
        results_is_valid_events = []
        accepted: list[tuple[int, int | None, EventAdmission, EventSearchOutput]] = []
        # Admission is geometric (contracts 7f policy 6): the full physical
        # source is built once per accepted event, in _resolve_prefactors.
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
                pbc=pbc,
            )
            if res.is_ok():
                admission = res.ok_value()
                self.add(admission.frame, reverse_idx_ref=admission.reverse_idx_ref)
                results_is_valid_events.append(Ok(admission.frame))
                if self.uses_prefactors:
                    frame = admission.frame
                    fwd_id = int(frame.iloc[0]["idx_ref"])
                    bwd_id = int(frame.iloc[1]["idx_ref"]) if len(frame) > 1 else None
                    accepted.append((fwd_id, bwd_id, admission, ev))
            else:
                results_is_valid_events.append(res)

        if self.uses_prefactors and accepted:
            self._resolve_prefactors(accepted, pbc)
            # Resolution can merge a proven reverse. Return only surviving,
            # resolved rows and their current links, not provisional frames.
            resolved_results = []
            for res in results_is_valid_events:
                if not res.is_ok():
                    resolved_results.append(res)
                    continue
                surviving = self.table[
                    self.table.idx_ref.isin(res.ok_value().idx_ref)
                ].copy()
                resolved_results.append(
                    Ok(surviving)
                    if len(surviving)
                    else Err(
                        ErrorInfo(
                            type=ErrorType.EVENT_NOT_NEW,
                            message="Found event already in reference table",
                            details="Both resolved directions match an earlier whole event",
                        )
                    )
                )
            results_is_valid_events = resolved_results

        return results_is_valid_events

    def _event_request(self, ev, pbc, *, event_key):
        """Build the same full physical source for lookup and calculation."""
        geometry = getattr(ev, "prefactor_geometry", None) or (
            ev.min1_positions,
            ev.saddle_positions,
            ev.min2_positions,
        )
        return self.prefactor_service.build_request(
            event_key=event_key,
            min1_positions=geometry[0],
            saddle_positions=geometry[1],
            min2_positions=geometry[2],
            types=ev.types,
            cell=ev.cell,
            pbc=pbc,
            center_index=ev.move_atom_index,
            constraints=getattr(ev, "constraints", None),
        )

    def _resolve_prefactors(
        self,
        accepted: list[tuple[int, int | None, EventAdmission, EventSearchOutput]],
        pbc: Any,
    ) -> None:
        """Submit one request per accepted event and patch the directional rows.

        Parameters
        ----------
        accepted : list of (forward id, backward id or None, admission, event)
            Logical ids assigned by :meth:`add`, the admission (its linking
            flags) and the full search geometry.
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
        for fwd_id, bwd_id, _admission, ev in accepted:
            if ev.types is None:
                raise RuntimeError(
                    "EventSearchOutput.types is required to build the HTST request "
                    f"of reference event {fwd_id}"
                )
            requests.append(self._event_request(ev, pbc, event_key=(fwd_id, bwd_id)))
        results = self.prefactor_service.compute(requests)
        wall = self.prefactor_service.last_batch_wall_s
        for fwd_id, bwd_id, admission, _ev in accepted:
            pre = results[(fwd_id, bwd_id)]
            self._patch_row(
                fwd_id, pre.forward, calculation=pre.calculation("forward"), fresh=True
            )
            self._log_direction(fwd_id, "forward", pre.forward, pre.n_free, wall)
            if bwd_id is not None:
                self._patch_row(
                    bwd_id,
                    pre.backward,
                    calculation=pre.calculation("backward"),
                    fresh=True,
                )
                self._log_direction(bwd_id, "backward", pre.backward, pre.n_free, wall)
                # Earlier entries in this same batch were pending at lookup.
                # Refresh identity only after both actual producers exist.
                known_forward = self._matching_resolved_direction(fwd_id, before=fwd_id)
                known_backward = self._matching_resolved_direction(
                    bwd_id, before=fwd_id
                )
                if known_forward is not None and known_backward is not None:
                    if self._coherent_existing_pair(
                        fwd_id, bwd_id, known_forward, known_backward, pre
                    ):
                        self._merge_direction(fwd_id, known_forward)
                        self._merge_direction(bwd_id, known_backward)
                    else:
                        logger.info(
                            "[htst] reference event %d: existing directional matches "
                            "do not prove one equivalent pair; retaining both directions "
                            "(n_free %d, batch %.3f s)",
                            fwd_id,
                            pre.n_free,
                            wall,
                        )
                elif known_backward is not None:
                    self._merge_direction(bwd_id, known_backward)
                    logger.info(
                        "[htst] reference event %d: reverse already catalogued as event %d; actual backward estimate agrees (n_free %d, batch %.3f s)",
                        fwd_id,
                        known_backward,
                        pre.n_free,
                        wall,
                    )
                elif admission.self_reverse_candidate:
                    self._record_self_reverse(fwd_id, bwd_id, pre, wall)
            else:
                # The reverse is already catalogued (geometric gate); its own
                # estimate stands and this search's backward is archived only.
                reverse_id = int(
                    self.table.loc[
                        self.table["idx_ref"] == fwd_id, "idx_backward"
                    ].iloc[0]
                )
                self._archive_discarded_backward(fwd_id, reverse_id, _ev, pre)
                logger.info(
                    "[htst] reference event %d: reverse already catalogued as "
                    "event %d, backward estimate of this search archived, not "
                    "selectable (n_free %d, batch %.3f s)",
                    fwd_id,
                    reverse_id,
                    pre.n_free,
                    wall,
                )

    def _archive_discarded_backward(self, fwd_id, reverse_id, ev, pre) -> None:
        """Keep a computed backward estimate that no selectable row carries."""
        backward = pre.backward
        if backward.skipped:
            return
        nu0_hz = float(backward.nu0_hz) if backward.ok else None
        rate = self.rate_constant.compute_rate(float(ev.dE_backward), nu0_hz)
        self.prefactor_archive.retain(
            fwd_id,
            {
                "nu0": nu0_hz,
                "nu0_status": NU0_OK if backward.ok else NU0_REJECTED,
                "nu0_reason": ""
                if backward.ok
                else f"{backward.reason_code.value}: {backward.reason}",
                "energy_barrier": float(ev.dE_backward),
                "k_prefactor": rate.prefactor,
            },
            f"backward estimate of this search discarded: reverse already "
            f"catalogued as reference {reverse_id}",
        )

    @staticmethod
    def _log_direction(
        idx_ref: int,
        direction: str,
        estimate: DirectionalPrefactor,
        n_free: int,
        wall_s: float,
    ) -> None:
        """Log the outcome of one directional reference estimate.

        Parameters
        ----------
        idx_ref : int
            Logical id of the row.
        direction : str
            ``"forward"`` or ``"backward"``.
        estimate : DirectionalPrefactor
            The resolved estimate.
        n_free : int
            Free atoms of the event's Hessians.
        wall_s : float
            Wall time (s) of the batch that produced the estimate.

        """
        if estimate.ok:
            logger.info(
                "[htst] reference event %d (%s): nu0 = %.4e Hz (n_free %d, "
                "batch %.3f s)",
                idx_ref,
                direction,
                estimate.nu0_hz,
                n_free,
                wall_s,
            )
        elif estimate.skipped:
            logger.info(
                "[htst] reference event %d (%s): not requested (n_free %d, "
                "batch %.3f s)",
                idx_ref,
                direction,
                n_free,
                wall_s,
            )
        else:
            logger.info(
                "[htst] reference event %d (%s): prefactor rejected (%s: %s), "
                "falling back to k0 (n_free %d, batch %.3f s)",
                idx_ref,
                direction,
                estimate.reason_code.value,
                estimate.reason,
                n_free,
                wall_s,
            )

    def _eligible_identity_calculation(self, idx_ref):
        """Read-only current acceptance check; identity lookup never launches work."""
        from .htst.request import HTSTRequestError

        rows = self.table[self.table.idx_ref == idx_ref]
        service = self.prefactor_service
        if len(rows) != 1 or service is None:
            return None
        row = rows.iloc[0]
        calculation = self.prefactor_archive.calculation_for(idx_ref, row)
        if (
            calculation is None
            or not calculation.estimate.ok
            or row.nu0_status != NU0_OK
        ):
            return None
        nu0 = calculation.estimate.nu0_hz
        if (
            not np.isfinite(row.nu0)
            or float(row.nu0) != nu0
            or not service.settings.nu0_min_hz <= nu0 <= service.settings.nu0_max_hz
            or self.prefactor_archive.descriptors.get(calculation.descriptor_id) is None
            or self.compare_physics(calculation.provenance.produced.descriptor).status
            != "compatible"
        ):
            return None
        try:
            current = service.request_from_snapshot(
                calculation.provenance.source, event_key=("identity", idx_ref)
            )
        except HTSTRequestError:
            return None
        return (
            calculation
            if service.calculation_context_matches(calculation, current)
            else None
        )

    def _merge_direction(self, discarded, survivor, reason=None):
        """Redirect every alias while keeping both original producing records."""
        mask = self.table.idx_ref == discarded
        row = self.table.loc[mask].iloc[0]
        self.prefactor_archive.retain(
            discarded,
            row,
            f"whole-event equivalent to reference {survivor}"
            if reason is None
            else reason,
        )
        self.table.loc[self.table.idx_backward == discarded, "idx_backward"] = survivor
        self.table = self.table.loc[~mask].copy()
        self._resolved_contexts.pop(discarded, None)

    def finalize_self_reverse(self, fwd_id, bwd_id, *, prefactors):
        """Collapse only the actual resolved pair with one full reversal witness."""
        from .htst.event_identity import calculations_equivalent

        if not self.uses_prefactors or fwd_id == bwd_id:
            return False
        first = self._eligible_identity_calculation(fwd_id)
        second = self._eligible_identity_calculation(bwd_id)
        if (
            first is None
            or second is None
            or first != prefactors.calculation("forward")
            or second != prefactors.calculation("backward")
            or not self_reverse_prefactors_agree(first.estimate, second.estimate)
        ):
            return False
        fwd = self.table[self.table.idx_ref == fwd_id].iloc[0]
        bwd = self.table[self.table.idx_ref == bwd_id].iloc[0]
        if (
            int(fwd.idx_backward) != bwd_id
            or int(bwd.idx_backward) != fwd_id
            or fwd.event_id != fwd.id_final
            or abs(float(fwd.energy_barrier) - float(bwd.energy_barrier))
            > SELF_REVERSE_BARRIER_TOL
            or not calculations_equivalent(
                first,
                second,
                tolerance=self.config.psr.matching_score_thr,
                kmax_factor=self.config.ira.kmax_factor,
                crop_radius=self.config.atomicenvironment.rcut,
            )
        ):
            return False
        self._merge_direction(bwd_id, fwd_id)
        return True

    def _coherent_existing_pair(
        self, fwd_id, bwd_id, known_forward, known_backward, pre
    ):
        """Do not splice independent pairs or make approximate equality transitive."""
        from .htst.event_identity import calculations_equivalent

        old_fwd = self.table[self.table.idx_ref == known_forward].iloc[0]
        old_bwd = self.table[self.table.idx_ref == known_backward].iloc[0]
        if (
            int(old_fwd.idx_backward) != known_backward
            or int(old_bwd.idx_backward) != known_forward
        ):
            return False
        if known_forward == known_backward:
            fwd = self.table[self.table.idx_ref == fwd_id].iloc[0]
            bwd = self.table[self.table.idx_ref == bwd_id].iloc[0]
            # Each new value may be close to the old value while the new
            # pair itself fails the barrier or spectral equality requirement.
            return (
                fwd.event_id == fwd.id_final
                and abs(float(fwd.energy_barrier) - float(bwd.energy_barrier))
                <= SELF_REVERSE_BARRIER_TOL
                and self_reverse_prefactors_agree(pre.forward, pre.backward)
                and calculations_equivalent(
                    pre.calculation("forward"),
                    pre.calculation("backward"),
                    tolerance=self.config.psr.matching_score_thr,
                    kmax_factor=self.config.ira.kmax_factor,
                    crop_radius=self.config.atomicenvironment.rcut,
                )
            )
        first = self._eligible_identity_calculation(known_forward)
        second = self._eligible_identity_calculation(known_backward)
        # With one canonical producing context and opposite directions, the
        # first full-triplet witness also maps the entire reversed triplet.
        # Different producers need a stronger joint proof; retain the new pair.
        return (
            first is not None
            and second is not None
            and first.provenance == second.provenance
            and first.direction != second.direction
        )

    def _matching_resolved_direction(self, idx_ref, *, before):
        """Find an older, current accepted producer after this batch resolves.

        Both directional spectra and the complete source/produced map must
        agree. Pending later rows cannot become a merge target.
        """
        from .htst.event_identity import calculations_equivalent

        first = self._eligible_identity_calculation(idx_ref)
        if first is None:
            return None
        row = self.table[self.table.idx_ref == idx_ref].iloc[0]
        candidates = self.table[
            (self.table.idx_ref < before)
            & (self.table.event_id == row.event_id)
            & (
                (self.table.energy_barrier - float(row.energy_barrier)).abs()
                <= SELF_REVERSE_BARRIER_TOL
            )
        ]
        for known_id in candidates.idx_ref:
            second = self._eligible_identity_calculation(int(known_id))
            if (
                second is not None
                and self_reverse_prefactors_agree(first.estimate, second.estimate)
                and calculations_equivalent(
                    first,
                    second,
                    tolerance=self.config.psr.matching_score_thr,
                    kmax_factor=self.config.ira.kmax_factor,
                    crop_radius=self.config.atomicenvironment.rcut,
                )
            ):
                return int(known_id)
        return None

    def _record_self_reverse(self, idx_ref, bwd_id, pre, wall_s):
        """Finalize a candidate, preserving each directional estimate on failure."""
        if self.finalize_self_reverse(idx_ref, bwd_id, prefactors=pre):
            logger.info(
                "[htst] reference event %d: whole-event self-reverse, backward nu0 = %.4e Hz agrees within %.0f%% (n_free %d, batch %.3f s)",
                idx_ref,
                pre.backward.nu0_hz,
                100.0 * SELF_REVERSE_NU0_RTOL,
                pre.n_free,
                wall_s,
            )
            return
        fwd, bwd = pre.forward, pre.backward
        # Policy 6: a rejected or disagreeing backward estimate collapses the
        # candidate back to one self-linked forward row (the backward value
        # stays in the archive). Accepted agreeing spectra without one common
        # whole-event map keep both rows (policy 4 counterexample).
        collapse = True
        if fwd.ok and bwd.ok:
            differs = 100.0 * abs(fwd.nu0_hz - bwd.nu0_hz) / fwd.nu0_hz
            if self_reverse_prefactors_agree(fwd, bwd):
                collapse = False
                note = f"self-reverse unproven: no common whole-event map, backward nu0 = {bwd.nu0_hz:.4e} Hz"
            else:
                note = f"self-reverse unproven: backward nu0 = {bwd.nu0_hz:.4e} Hz, differs by {differs:.1f}%"
        elif bwd.ok:
            note = f"self-reverse unproven: backward nu0 = {bwd.nu0_hz:.4e} Hz (forward rejected)"
        elif bwd.skipped:
            note = "self-reverse unproven: backward prefactor not requested"
        else:
            note = f"self-reverse unproven: backward prefactor rejected ({bwd.reason_code.value}: {bwd.reason})"
        mask = self.table.idx_ref == idx_ref
        current = str(self.table.loc[mask, "nu0_reason"].iloc[0])
        self.table.loc[mask, "nu0_reason"] = f"{current}; {note}" if current else note
        if collapse:
            self._merge_direction(bwd_id, idx_ref, reason=note)
            action = "collapsed to one self-linked row, backward estimate archived"
        else:
            action = "retaining both directions"
        logger.warning(
            "[htst] reference event %d: %s; %s (n_free %d, batch %.3f s)",
            idx_ref,
            note,
            action,
            pre.n_free,
            wall_s,
        )

    def _patch_row(
        self,
        idx_ref: int,
        estimate: DirectionalPrefactor,
        *,
        calculation=None,
        fresh: bool = False,
    ) -> None:
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
            If no row carries ``idx_ref`` or if the estimate was skipped
            (``status == "skipped"`` is never stored: it is no estimate).

        Notes
        -----
        An accepted estimate written without ``calculation`` has no producer
        and is demoted to ``legacy``/``k0`` with a warning; production callers
        always pass the producing calculation.

        """
        if estimate.skipped:
            raise ValueError(
                f"reference event {idx_ref}: a skipped direction carries no "
                "estimate and is never written to the table"
            )
        mask = self.table["idx_ref"] == idx_ref
        if not mask.any():
            raise ValueError(f"idx_ref {idx_ref} is not in the reference table")
        if calculation is not None:
            calculation.validate()
            if calculation.estimate != estimate:
                raise ValueError("estimate differs from its producing calculation")
        archive = self.prefactor_archive
        previous = self.table.loc[mask].iloc[0]
        if previous["nu0_status"] != NU0_PENDING:
            archive.retain(idx_ref, previous, "estimate superseded")
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
        self._resolved_contexts.pop(int(idx_ref), None)
        if calculation is None:
            archive.references[int(idx_ref)] = None
            if estimate.ok:
                # Never a production path (every caller passes calculation=);
                # a value without its producer is not selectable, and says so.
                archive.retain(
                    idx_ref,
                    self.table.loc[mask].iloc[0],
                    "missing producing calculation",
                )
                self._set_estimate(
                    idx_ref, NU0_LEGACY, None, "legacy: missing producing calculation"
                )
                logger.warning(
                    "[htst] reference event %d: accepted nu0 = %.4e Hz written "
                    "without its producing calculation; demoted to 'legacy' with "
                    "the k0 fallback (the value stays in the archive history)",
                    idx_ref,
                    float(estimate.nu0_hz),
                )
        else:
            archive.record(idx_ref, self.table.loc[mask].iloc[0], calculation)
            if fresh:
                self._fresh_calculations.add(
                    (id(self.prefactor_service), calculation.calculation_id)
                )

    def _set_estimate(self, idx_ref, status, nu0, reason) -> None:
        """Update every selectable rate field together, in the current rate policy."""
        mask = self.table["idx_ref"] == idx_ref
        barrier = float(self.table.loc[mask, "energy_barrier"].iloc[0])
        rate = self.rate_constant.compute_rate(barrier, nu0)
        self.table.loc[mask, "nu0_status"] = status
        self.table.loc[mask, "nu0"] = float("nan") if nu0 is None else float(nu0)
        self.table.loc[mask, "nu0_reason"] = reason
        self.table.loc[mask, "k_prefactor"] = rate.prefactor
        self.table.loc[mask, "k"] = rate.rate

    def _ensure_current_estimate(self, idx_ref: int) -> None:
        """Validate producing context before reference inheritance or selection.

        The validation is memoised per row on a cheap key (service identity,
        method, current physical descriptor, producing calculation id, row
        digest, ``T``, ``k0``); a hit returns before the full source is
        rebuilt or hashed, so repeated selection of an unchanged row costs one
        crop digest. Any change of service, physics, producing record, row
        content or rate policy misses and revalidates.
        """
        from .htst.provenance import RequestSnapshot
        from .htst.request import HTSTRequestError
        from .physics import _digest

        mask = self.table["idx_ref"] == idx_ref
        row = self.table.loc[mask].iloc[0]
        archive = self.prefactor_archive
        calculation = archive.calculation_for(idx_ref, row)
        if calculation is None:
            reason = "legacy: missing producing calculation or row correspondence"
            archive.retain(idx_ref, row, reason)
            # Preserve an existing scientific rejection/fallback diagnostic.
            status = str(row["nu0_status"])
            if status in (NU0_PENDING, NU0_OK):
                status = NU0_LEGACY
            prior = str(row["nu0_reason"])
            self._set_estimate(idx_ref, status, None, prior or reason)
            return
        service = self.prefactor_service
        if service is None:
            archive.retain(idx_ref, row, "missing current prefactor service")
            self._set_estimate(
                idx_ref, NU0_STALE, None, "stale: missing current physical context"
            )
            return
        # Cheap memo first: calculation_for already bound the row digest to the
        # producing record, so the key needs no request rebuild and no
        # full-geometry hash. Every component is a stored string or scalar.
        link = archive.references[int(idx_ref)]
        descriptor = service.current_descriptor
        signature = (
            id(service),
            service.method,
            None if descriptor is None else descriptor.descriptor_id,
            link.calculation_id,
            link.row_digest,
            float(self.config.rateconstant.T),
            float(self.config.rateconstant.k0),
        )
        if self._resolved_contexts.get(int(idx_ref)) == signature:
            return
        try:
            request = service.request_from_snapshot(
                calculation.provenance.source, event_key=("reload", int(idx_ref))
            )
        except HTSTRequestError as exc:
            reason = f"stale: cannot rebuild complete current source: {exc}"
            archive.retain(idx_ref, row, reason)
            self._set_estimate(idx_ref, NU0_STALE, None, reason)
            return
        comparison = self.compare_physics(calculation.provenance.produced.descriptor)
        registered = archive.descriptors.get(calculation.descriptor_id)
        fresh = (id(service), calculation.calculation_id) in self._fresh_calculations
        physics_match = comparison.status == "compatible" or (
            fresh and calculation.descriptor_id == request.descriptor.descriptor_id
        )
        context_match = service.calculation_context_matches(calculation, request)
        compatible = registered is not None and physics_match and context_match
        estimate = calculation.estimate
        old_settings = calculation.provenance.produced.settings
        window_changed = (old_settings.nu0_min_hz, old_settings.nu0_max_hz) != (
            service.settings.nu0_min_hz,
            service.settings.nu0_max_hz,
        )
        needs_window_recompute = (
            not estimate.ok
            and estimate.reason_code.value == "out_of_window"
            and window_changed
        )
        if compatible and not needs_window_recompute:
            if estimate.ok:
                nu0 = float(estimate.nu0_hz)
                if service.settings.nu0_min_hz <= nu0 <= service.settings.nu0_max_hz:
                    self._set_estimate(idx_ref, NU0_OK, nu0, "")
                else:
                    reason = (
                        f"out_of_window (reload): nu0 = {nu0:.4e} Hz outside "
                        f"[{service.settings.nu0_min_hz:.4e}, "
                        f"{service.settings.nu0_max_hz:.4e}] Hz"
                    )
                    archive.retain(idx_ref, row, reason)
                    self._set_estimate(idx_ref, NU0_REJECTED, None, reason)
            else:
                self._set_estimate(
                    idx_ref,
                    NU0_REJECTED,
                    None,
                    f"{estimate.reason_code.value}: {estimate.reason}",
                )
            self._resolved_contexts[int(idx_ref)] = signature
            return
        reason = "stale: " + "; ".join(
            comparison.reasons or ("producing calculation context differs",)
        )
        archive.retain(idx_ref, row, reason)
        self._set_estimate(idx_ref, NU0_STALE, None, reason)
        # The old value is unavailable before dispatch. Both directions share
        # one full-source recalculation, independent of sparse table labels.
        request_id = _digest(
            (id(service), RequestSnapshot.capture(request).snapshot_id, service.method)
        )
        if request_id not in self._recomputed:
            self._recomputed[request_id] = service.compute(
                [request], compute_backward=True, compute_energies=True
            )[request.event_key]
        result = self._recomputed[request_id]
        current = result.calculation(calculation.direction)
        if current is None:
            self._set_estimate(
                idx_ref,
                NU0_STALE,
                None,
                "stale: worker returned no producing calculation",
            )
            self._resolved_contexts[int(idx_ref)] = signature
            return
        if (
            current.provenance.source != RequestSnapshot.capture(request)
            or current.provenance.method != service.method
        ):
            raise RuntimeError(
                "recomputed prefactor does not describe the submitted current source"
            )
        if not service.calculation_context_matches(current, request):
            raise RuntimeError(
                "recomputed prefactor used a different free/crop or constraint context"
            )
        energies = current.provenance.energies
        if energies is None:
            self._set_estimate(
                idx_ref,
                NU0_STALE,
                None,
                "stale: recomputation lacks current full-system potential energies",
            )
            self._resolved_contexts[int(idx_ref)] = signature
            return
        minimum = energies[0] if calculation.direction == "forward" else energies[2]
        recomputed_barrier = float(energies[1] - minimum)
        verdict = current.estimate
        if verdict.ok or verdict.reason_code.value not in (
            GEOMETRY_INVALIDATING_REJECTIONS
        ):
            self.table.loc[mask, "energy_barrier"] = recomputed_barrier
        else:
            # The kernel found the saved triplet non-stationary under the
            # current physics: its energies are not a barrier. Keep the
            # catalogued barrier, record the attempt and say so.
            kept = float(self.table.loc[mask, "energy_barrier"].iloc[0])
            reason = (
                f"recomputation rejected ({verdict.reason_code.value}: "
                f"{verdict.reason}); catalogued barrier {kept:.4f} eV kept, "
                f"recomputed {recomputed_barrier:.4f} eV not adopted"
            )
            archive.retain(idx_ref, row, reason)
            logger.warning(
                "[htst] reference event %d: recomputed prefactor rejected (%s: %s) "
                "under the current physics; keeping the catalogued barrier %.4f eV "
                "(recomputed %.4f eV not adopted), rate falls back to k0",
                idx_ref,
                verdict.reason_code.value,
                verdict.reason,
                kept,
                recomputed_barrier,
            )
        self._patch_row(idx_ref, current.estimate, calculation=current, fresh=True)
        # Apply the current window and rate policy even to a worker whose
        # numerical acceptance contract was implemented separately.
        self._ensure_current_estimate(idx_ref)

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
        self._ensure_current_estimate(idx_ref)
        rows = self.table[self.table["idx_ref"] == idx_ref]
        row = rows.iloc[0]
        status = str(row["nu0_status"])
        nu0 = row["nu0"]
        nu0_hz = float(nu0) if status == NU0_OK else None
        return {
            "nu0_hz": nu0_hz,
            "nu0_status": status,
            "nu0_reason": str(row["nu0_reason"]) if status != NU0_OK else "",
            "nu0_source": SOURCE_REFERENCE if status == NU0_OK else SOURCE_K0,
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
        *,
        request=None,
        pbc: Any = None,
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
        request : HTSTEventRequest, optional
            Full physical source of the event (htst/rpa lookups).
        pbc : array_like of bool, optional
            Periodic axes of the system the event was found in, handed to
            :meth:`_build_event_series`.

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
                pbc=pbc,
            )
            return self._admit_series(
                dfevent_forward, dfevent_backward, request=request
            )

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
        *,
        pbc: Any = None,
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
        pbc : array_like of bool, optional
            Periodic axes of the system the event was found in.

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
            pbc=pbc,
        )
        if res.is_ok():
            return Ok(res.ok_value().frame)
        return res

    def _admit_series(
        self,
        dfevent_forward: pd.Series,
        dfevent_backward: pd.Series,
        *,
        request=None,
    ) -> Result[EventAdmission, ErrorInfo]:
        """Apply the geometric admission gate, then the style's row policy.

        :meth:`find_matching_event` (topology + 0.25 eV + IRA saddle-crop
        match) is the duplicate gate in every style (contracts 7f policy 6):
        a catalogued event is a duplicate whatever its prefactor status, so a
        k0-fallback row is never re-admitted. Constant mode keeps its original
        row policy. HTST/RPA admit two provisional reciprocal rows so both
        directions are actually calculated, except when the backward direction
        already matches a catalogued row geometrically: then one forward row is
        admitted and linked to that row, as in constant mode. Whether a
        resolved pair may be collapsed is decided after resolution by the
        whole-event proof (policy 4), never here. ``request`` is accepted for
        signature compatibility and ignored: admission never reads the full
        source, which is built once per accepted event for the worker.
        """
        duplicate = self.find_matching_event(dfevent_forward)
        if duplicate is not None:
            return Err(
                ErrorInfo(
                    type=ErrorType.EVENT_NOT_NEW,
                    message="Found event already in reference table",
                    details="Same topology",
                )
            )
        same_topology = dfevent_forward["event_id"] == dfevent_forward["id_final"]
        if not self.uses_prefactors:
            if same_topology:
                return Ok(
                    EventAdmission(
                        frame=dfevent_forward.to_frame().T, same_topology=True
                    )
                )
            if self.is_new_event(dfevent=dfevent_backward):
                return Ok(
                    EventAdmission(
                        frame=self._two_rows(dfevent_forward, dfevent_backward)
                    )
                )
            return Ok(EventAdmission(frame=dfevent_forward.to_frame().T))

        gap = abs(
            float(dfevent_forward.energy_barrier)
            - float(dfevent_backward.energy_barrier)
        )
        if same_topology and gap <= SELF_REVERSE_BARRIER_TOL:
            # Both directions are calculated; finalize_self_reverse decides
            # the collapse and _record_self_reverse the one-row fallback.
            return Ok(
                EventAdmission(
                    frame=self._two_rows(dfevent_forward, dfevent_backward),
                    same_topology=True,
                    self_reverse_candidate=True,
                )
            )
        reverse_idx_ref = self.find_matching_event(dfevent_backward)
        if reverse_idx_ref is not None:
            return Ok(
                EventAdmission(
                    frame=dfevent_forward.to_frame().T,
                    reverse_idx_ref=reverse_idx_ref,
                )
            )
        return Ok(
            EventAdmission(frame=self._two_rows(dfevent_forward, dfevent_backward))
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
        """Coarse saddle-crop proposal, never proof of whole-event identity.

        Kept for geometric diagnostics and compatibility with classification
        callers. HTST/RPA admission and collapse require full physical maps.
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

    def has_id_subset_table(self, ids: list[str]) -> pd.DataFrame:
        """Return subset table with event having id in ids.

        In prefactor modes, validate the requested rows and their linked
        reverses before callers read either direction's barriers and rates.

        Parameters
        ----------
        ids : list[str]
            list of IDs.

        Returns
        -------
        pd.DataFrame
            Subset of the reference table dataframe with only event having IDs in ids.

        """
        mask = self.table["event_id"].isin(ids)
        if self.uses_prefactors:
            required = self.table.loc[mask, ["idx_ref", "idx_backward"]].to_numpy()
            linked = self.table["idx_ref"].isin(required.ravel())
            for idx_ref in self.table.loc[linked, "idx_ref"].tolist():
                self._ensure_current_estimate(int(idx_ref))
        return self.table[mask]

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
        pbc: Any = None,
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
        pbc : array_like of bool, optional
            Periodic axes of the system the event was found in. The scratch
            systems that compute the neighbour lists, graph ids, stored crops
            and symmetry sets carry these axes, so a catalogued ``event_id``
            equals the runtime environment id of a boundary-crossing atom.
            ``None`` keeps the catalogue's historical all-periodic convention
            (a bare ``System()`` is non-periodic); ``add_events`` always passes
            the real axes.

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
        axes = True if pbc is None else pbc

        # compute neighbors list for initial, saddle and final positions -> to compute graphs
        min1system = System(pbc=axes)
        min1system.positions = min1_positions
        min1system.cell = cell
        min1neighbors_list = NeighborsList(
            min1system,
            self.config.atomicenvironment.rnei,
            self.config.atomicenvironment.rcut,
        )

        saddlesystem = System(pbc=axes)
        saddlesystem.positions = saddle_positions
        saddlesystem.cell = cell
        saddleneighbors_list = NeighborsList(
            saddlesystem,
            self.config.atomicenvironment.rnei,
            self.config.atomicenvironment.rcut,
        )

        min2system = System(pbc=axes)
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
        """Allocate above surviving and archived logical IDs in prefactor modes."""
        ids = [int(i) for i in self.table.idx_ref]
        if self.uses_prefactors:
            ids.extend(self.prefactor_archive.references)
            ids.extend(self.prefactor_archive.history)
        return max(ids, default=-1) + 1

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

    def _validate_stored_estimates(self, df, path: str) -> None:
        """Reject corrupt declared estimates before considering reuse policy."""
        if df["idx_ref"].duplicated().any():
            raise ValueError(f"reference table {path}: duplicate logical reference IDs")
        for _, row in df.iterrows():
            status = row["nu0_status"]
            if status not in NU0_STATUSES:
                raise ValueError(
                    f"reference table {path}: unknown nu0_status {status!r}"
                )
            if status != NU0_OK:
                continue
            value = row["nu0"]
            if value is None:
                raise ValueError(
                    f"reference table {path}: accepted estimate has no nu0 value; "
                    "its frequency in Hz is required"
                )
            if (
                isinstance(value, (bool, np.bool_))
                or not math.isfinite(float(value))
                or float(value) <= 0
            ):
                raise ValueError(
                    f"reference table {path}: accepted estimate must be a finite "
                    "positive frequency"
                )
            rate = self.rate_constant.compute_rate(
                float(row["energy_barrier"]), float(value)
            )
            if not math.isclose(
                rate.prefactor, float(row["k_prefactor"]), rel_tol=1e-9, abs_tol=0.0
            ):
                raise ValueError(
                    f"reference table {path}: k_prefactor = {float(row['k_prefactor'])!r} "
                    f"ps^-1 but its nu0 = {float(value)!r} Hz resolves to "
                    f"{rate.prefactor!r} ps^-1 (edited or corrupted)"
                )

    def _load(self, path: str) -> None:
        """Migrate unknown estimates; validate schema-2 producers before reuse.

        Event crops always survive. A complete producing source can be rebuilt
        under current physics; missing context uses explicit k0 fallback.
        Saving never supplies the missing scientific evidence.
        """
        df = pd.read_pickle(path)
        metadata = dict(df.attrs) if df.attrs else {}
        df.attrs = {}
        present = [name for name in REFERENCE_HTST_COLUMNS if name in df.columns]
        if not self.uses_prefactors:
            if present or metadata:
                logger.warning(
                    "Reference table %s carries HTST data (columns: %s); "
                    "constant style drops it and recomputes rates",
                    path,
                    ", ".join(present) if present else "none",
                )
                df = df.drop(columns=present)
                df["k"] = [
                    self.rate_constant.compute_rate(float(barrier)).rate
                    for barrier in df["energy_barrier"]
                ]
            self.table = df
            return
        from .htst.catalogue import PrefactorArchive

        complete_columns = len(present) == len(REFERENCE_HTST_COLUMNS)
        version = metadata.get("schema_version")
        if "idx_ref" not in df.columns and version is None:
            # Pre-schema catalogues used the dataframe index for reverse links.
            # Promote that logical label without inventing physical provenance.
            if not df.index.is_unique or any(
                isinstance(idx, (bool, np.bool_))
                or not isinstance(idx, (int, np.integer))
                or idx < 0
                for idx in df.index
            ):
                raise ValueError(
                    f"reference table {path}: legacy reference labels must be "
                    "unique nonnegative integers"
                )
            df.insert(0, "idx_ref", df.index.to_numpy(copy=True))
        if metadata and complete_columns:
            if version not in (1, TABLE_SCHEMA_VERSION):
                raise ValueError(
                    f"reference table {path} has unsupported schema_version {version!r}"
                )
            if (
                metadata.get("nu0_units") != "Hz"
                or metadata.get("k_prefactor_units") != "ps^-1"
            ):
                raise ValueError(
                    f"reference table {path}: expected nu0_units='Hz' and "
                    "k_prefactor_units='ps^-1' (units are never inferred from magnitudes)"
                )
            self._validate_stored_estimates(df, path)
            if metadata.get("T") != self.config.rateconstant.T:
                logger.info(
                    "Reference table %s was saved at T = %s K; rates recomputed "
                    "at T = %g K after validating current prefactors",
                    path,
                    metadata.get("T"),
                    self.config.rateconstant.T,
                )
        if complete_columns and version == TABLE_SCHEMA_VERSION:
            self.prefactor_archive = PrefactorArchive.from_metadata(metadata)
            for _, row in df.iterrows():
                calculation = self.prefactor_archive.calculation_for(
                    int(row["idx_ref"]), row
                )
                if (
                    row["nu0_status"] == NU0_OK
                    and calculation is not None
                    and (
                        not calculation.estimate.ok
                        or float(row["nu0"]) != calculation.estimate.nu0_hz
                    )
                ):
                    raise ValueError("accepted row differs from its producing estimate")
        else:
            self.prefactor_archive = PrefactorArchive()
            if metadata:
                self.prefactor_archive.legacy_metadata.append(metadata)
            reason = "legacy table: missing per-calculation producing provenance"
            # Policy 2: invalidation is allowed, silence is not. One WARNING
            # per table says what was lost, why, and how to get it back.
            accepted_rows = (
                int((df["nu0_status"] == NU0_OK).sum())
                if "nu0_status" in df.columns
                else 0
            )
            logger.warning(
                "Reference table %s: %s (stored %s, current schema %d); %d row(s) "
                "kept with their event geometry, %d accepted estimate(s) demoted to "
                "status 'legacy' with the k0 fallback (%g ps^-1) because their "
                "producing calculations were never persisted (contracts policy 2). "
                "Recovery: reference prefactors cannot be rebuilt from local crops; "
                "run with a prefactor service so refined sites get site estimates, "
                "or regenerate the catalogue under schema %d. Re-saving never "
                "restores the estimates.",
                path,
                reason,
                "no schema metadata"
                if version is None
                else f"schema_version {version}",
                TABLE_SCHEMA_VERSION,
                len(df),
                accepted_rows,
                float(self.config.rateconstant.k0),
                TABLE_SCHEMA_VERSION,
            )
            for _, row in df.iterrows():
                self.prefactor_archive.retain(int(row["idx_ref"]), row, reason)
            df["nu0"] = float("nan")
            df["nu0_status"] = NU0_LEGACY
            df["nu0_reason"] = reason
            df["k_prefactor"] = float(self.config.rateconstant.k0)
        other = [name for name in df.columns if name not in REFERENCE_HTST_COLUMNS]
        self.table = df[other + list(REFERENCE_HTST_COLUMNS)]
        self.metadata = metadata
        self._fresh_calculations.clear()
        self._resolved_contexts.clear()
        self._recomputed.clear()
        for idx_ref in self.table["idx_ref"].tolist():
            self._ensure_current_estimate(int(idx_ref))
        outside_window = self.table["nu0_reason"].str.startswith(
            "out_of_window (reload):", na=False
        )
        if outside_window.any():
            logger.warning(
                "Reference table %s: %d accepted estimate(s) lie outside the "
                "current nu0 window; using k0 fallback",
                path,
                int(outside_window.sum()),
            )

    def _changed_settings(self, stored: Any) -> list[str]:
        """Compare stored kernel settings with the current configuration.

        Parameters
        ----------
        stored : Any
            The ``settings`` entry of a loaded table's metadata (a dict, or
            anything else for a table without one).

        Returns
        -------
        list[str]
            One ``"<name> changed on reload (stored X, current Y)"`` entry per
            setting of :data:`RELOAD_SETTINGS` that differs, in that order;
            empty when the table is compatible. A missing
            ``free_region_center`` counts as ``"min1"`` (tables written before
            the saddle-centred default); any other missing setting counts as
            changed (unknown provenance is never treated as compatible).

        """
        stored = dict(stored) if isinstance(stored, dict) else {}
        current = self.table_metadata()["settings"]
        changed: list[str] = []
        for name in RELOAD_SETTINGS:
            now = current[name]
            if name in stored:
                was = stored[name]
            elif name == "free_region_center":
                was = "min1"
            else:
                changed.append(
                    f"{name} changed on reload (stored absent, current {now})"
                )
                continue
            if not self._same_setting(was, now):
                changed.append(
                    f"{name} changed on reload (stored {was}, current {now})"
                )
        return changed

    @staticmethod
    def _same_setting(was: Any, now: Any) -> bool:
        """Return True when a stored setting equals the current one.

        Parameters
        ----------
        was : Any
            The stored value.
        now : Any
            The current value.

        Returns
        -------
        bool
            Booleans compare as booleans, ``None`` only equals ``None``,
            numbers compare within ``1e-12`` relative, everything else by
            string.

        """
        if isinstance(was, bool) or isinstance(now, bool):
            return isinstance(was, bool) and isinstance(now, bool) and was == now
        if was is None or now is None:
            return was is None and now is None
        if isinstance(was, (int, float)) and isinstance(now, (int, float)):
            return math.isclose(float(was), float(now), rel_tol=1e-12, abs_tol=0.0)
        return str(was) == str(now)

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
        descriptor = self.current_descriptor
        numerical = {} if descriptor is None else descriptor.numerical_settings()
        return {
            "schema_version": TABLE_SCHEMA_VERSION,
            "style": rc.style,
            "nu0_units": "Hz",
            "k_prefactor_units": "ps^-1",
            "T": float(rc.T),
            "k0": float(rc.k0),
            "context_role": "serialization_policy",
            **(
                {}
                if self.prefactor_archive is None
                else self.prefactor_archive.metadata()
            ),
            "settings": {
                "free_radius": numerical.get("free_radius", float(rc.free_radius)),
                "free_region_center": numerical.get(
                    "free_region_center", str(rc.free_region_center)
                ),
                "fd_step": numerical.get("fd_step", float(rc.fd_step)),
                "zone_radius": numerical.get("zone_radius", rc.zone_radius),
                "premin": numerical.get("premin", bool(rc.premin)),
                "nu0_min_hz": thz_to_hz(rc.nu0_min_THz),
                "nu0_max_hz": thz_to_hz(rc.nu0_max_THz),
            },
        }

    def remove(self, idx_refs: list[int]) -> RemovedReferences:
        """Remove events with ind == idx_ref as well as its backward event.

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

        Returns
        -------
        RemovedReferences
            The complete set of removed logical ids (sorted) with their
            ``event_id``; ``KMC.reconstruction`` hands it to
            :meth:`ActiveEventTable.drop_reference_events` so no active row
            ever references a removed catalogue entry (contracts section 7d,
            F4).

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

        removed_mask = self.table["idx_ref"].isin(all_refs)
        removed = self.table.loc[removed_mask]
        ids = removed["idx_ref"].astype(int).to_numpy()
        order = np.argsort(ids, kind="stable")
        report = RemovedReferences(
            idx_refs=tuple(int(i) for i in ids[order]),
            event_ids=tuple(str(e) for e in removed["event_id"].to_numpy()[order]),
        )
        self.table = self.table[~removed_mask].reset_index(
            drop=True
        )  # keep event not (~) in all refs
        return report

    def save(self, outfile: str = "reference_table.pickle") -> None:
        """Serialize producing facts and current rate-policy metadata without work.

        An attached service is not a producer of old rows. Unknown numerical
        values are preserved only in history, never made selectable by resave.
        """
        if not self.uses_prefactors:
            self.table.to_pickle(outfile)
            return
        frame = self.table.copy(deep=True)
        self._validate_stored_estimates(frame, outfile)
        for label, row in frame.iterrows():
            idx_ref = int(row["idx_ref"])
            if self.prefactor_archive.calculation_for(idx_ref, row) is None:
                reason = "legacy: missing producing calculation or row correspondence"
                self.prefactor_archive.retain(idx_ref, row, reason)
                if row["nu0_status"] == NU0_OK:
                    frame.loc[label, "nu0_status"] = NU0_LEGACY
                    frame.loc[label, "nu0_reason"] = reason
                frame.loc[label, "nu0"] = float("nan")
                rate = self.rate_constant.compute_rate(float(row["energy_barrier"]))
                frame.loc[label, "k_prefactor"] = rate.prefactor
                frame.loc[label, "k"] = rate.rate
        frame.attrs = self.table_metadata()
        frame.to_pickle(outfile)


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
    attempt, not its success. An unchanged complete producing context keeps
    that attempt; changed or unknown dependencies remove the row before it
    can suppress refinement or enter selection. Rebuilding produces a new
    row through :meth:`add_events`, with a current full saddle and crop IDs.

    Site geometry: the request is built from the current full minimum and the
    full pARTn-refined saddle that ``Refinement.execute`` hands over on
    ``EventRefinementOutput.full_saddle_positions`` (only the forward
    direction is computed). That array is kept in a transient side store keyed
    by row label, never in the DataFrame, and released as soon as the site
    batch has been submitted; every table mutation that relabels rows
    (:meth:`remove`, :meth:`drop_reference_events`,
    :meth:`prune_for_recycling`) keeps the store consistent, and the request
    path checks the stored crop against the full saddle before submitting
    anything. Crop identities are stable global atom IDs, so source or crop
    ordering changes do not silently change correspondence. A refined row
    with explicit crop IDs but no full saddle keeps its inherited estimate, is
    marked attempted and is counted as ``no_geometry`` (contracts section 7d,
    F1); no request is ever built from an ``rcut`` crop pasted into the
    minimum.

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
        # Transient full refined saddles of rows awaiting their site request,
        # keyed by the row's current label (see the class notes).
        self._full_saddles: dict[int, np.ndarray] = {}
        self._full_saddle_constraints: dict[int, Any] = {}
        self._site_states: dict[int, Any] = {}
        self._pending_site_rows: set[int] = set()
        self._constant_crop_ids: dict[int, tuple[int, ...]] = {}
        # Site rejections of the most recent request_site_prefactors call,
        # by kernel reason code (for the per-step [htst] summary).
        self.step_site_rejections: dict[str, int] = {}

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
        if self.prefactor_service is not None:
            return self.prefactor_service.rate_constant
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
        end-of-step `del active_table` behavior). Rows surviving a prune were
        attempted in the step that built them, so the transient full saddles
        are dropped here.
        """
        if self.recycler is None:
            self.remove(list(self.table.index))
        else:
            selected = self.recycler.select_recyclable(
                self,
                executed_idx,
                system,
                positions_pre,
            )
            self.remove([i for i in self.table.index if i not in selected.index])
            self._pending_site_rows.clear()
            from dataclasses import replace

            # Immediate use of fresh opaque physics is permitted only in its
            # producing step. Equal opaque descriptors cannot justify recycling.
            self._site_states = {
                label: replace(state, fresh_service=None)
                for label, state in self._site_states.items()
            }
            self.validate_recycled(system, allow_pending=False)
        self._full_saddles = {}
        self._full_saddle_constraints = {}

    def has_crop_correspondence(self, label: int) -> bool:
        """Distinguish absent legacy metadata from invalid declared identities."""
        if int(label) in self._constant_crop_ids:
            return True
        stored = self.table.loc[label].get("crop_atom_ids")
        return stored is not None and not (
            isinstance(stored, float) and math.isnan(stored)
        )

    def crop_indices(self, label, system, neighbors_list=None, *, capture=False):
        """Resolve stored stable crop identities without using a new crop order."""
        from .physics import _indices

        row = self.table.loc[label]
        ids = _indices(system.index)
        stored = row.get("crop_atom_ids")
        if not self.uses_prefactors:
            stored = self._constant_crop_ids.get(int(label), stored)
        if stored is None or (isinstance(stored, float) and math.isnan(stored)):
            if not capture or neighbors_list is None:
                raise ValueError("active event has no stored crop identities")
            indices = _indices(
                neighbors_list.get_neighbors("rcut", int(row.atom_index)),
                upper=len(ids),
            )
            if self.uses_prefactors:
                full = self._full_saddles.get(int(label))
                if full is None:
                    raise ValueError(
                        "active event without a full saddle must supply its crop atom identities"
                    )
                if np.asarray(row["saddle_positions"]).shape != (len(indices), 3):
                    raise RuntimeError(
                        "stored crop does not match the current rcut mapping"
                    )
                if np.asarray(full).shape != np.asarray(
                    system.positions
                ).shape or not np.array_equal(
                    np.asarray(full)[list(indices)], row["saddle_positions"]
                ):
                    raise RuntimeError(
                        "stored crop is not the full refined saddle at the current rcut mapping"
                    )
            stored = tuple(ids[i] for i in indices)
            if "crop_atom_ids" not in self.table.columns:
                self.table["crop_atom_ids"] = pd.Series(
                    [None] * len(self.table), index=self.table.index, dtype=object
                )
            self.table.at[label, "crop_atom_ids"] = stored
        stored = _indices(stored)
        position = {atom_id: k for k, atom_id in enumerate(ids)}
        try:
            indices = np.array([position[i] for i in stored], dtype=int)
        except KeyError as exc:
            raise ValueError(
                "stored crop identities are not in the current source"
            ) from exc
        if int(row.atom_index) not in indices:
            raise ValueError("active event center is outside its stored crop")
        for name in ("saddle_positions", "final_positions"):
            if row[name] is not None and np.asarray(row[name]).shape != (
                len(indices),
                3,
            ):
                raise RuntimeError(
                    "stored crop does not match the current rcut mapping: "
                    "crop and identities have different sizes"
                )
        return indices

    def _warn_crop_identity(self, label, exc: Exception, stage: str) -> None:
        """Report a row that cannot declare stable crop identities (never raise).

        Such a row is not a crop-only fallback (contracts section 7d, F1
        covers rows with stable identities and no full saddle): reconstruction
        resolves the stored identities against the current source, so the row
        cannot be reconstructed and selecting it would purge its reference.
        :meth:`request_site_prefactors` drops it before selection
        (:meth:`_drop_identityless_rows`); its ``(atom, reference)`` pair is
        re-refined next step.
        """
        row = self.table.loc[label]
        logger.warning(
            "[htst] active event (atom %d, reference %d): no stable crop "
            "identities at %s (%s); the row cannot be reconstructed from the "
            "current source and is dropped before selection, its (atom, "
            "reference) pair is re-refined next step",
            int(row["atom_index"]),
            int(row["num_reference_event"]),
            stage,
            exc,
        )

    def _drop_identityless_rows(self, labels: Iterable[Any]) -> None:
        """Remove the rows reported by :meth:`_warn_crop_identity` this call."""
        if not labels:
            return
        self.remove(list(labels))
        logger.info(
            "active table: dropped %d row(s) without stable crop identities "
            "before selection; their (atom, reference) pairs are re-refined "
            "next step",
            len(labels),
        )

    def site_calculation(self, label):
        """Return the actual row-bound site producer, never a scalar reconstruction."""
        from .htst.site_state import row_signature

        state = self._site_states.get(int(label))
        if state is None or label not in self.table.index:
            return None
        try:
            return (
                state.calculation
                if row_signature(self.table.loc[label], state.center_id)
                == state.signature
                else None
            )
        except (ValueError, TypeError, KeyError):
            return None

    def validate_recycled(self, system, neighbors_list=None, *, allow_pending=True):
        """Drop obsolete dependencies before refinement skips or rate selection.

        A dropped pair can run through the normal refinement dispatcher again.
        Fresh unattempted producer outputs are allowed only until their first
        site/fallback context is captured. They are never exempt across a prune.
        """
        if not self.uses_prefactors or self.table.empty:
            return 0
        from .htst.request import HTSTRequestError
        from .htst.site_state import row_signature, source_index_map
        from dataclasses import replace

        # One identity -> row map per validation pass, shared by every row. A
        # source without stable identities cannot validate any recycled row:
        # the documented conservative fallback drops them (re-refined next
        # step) with one WARNING naming the cause, never an abort of the step.
        try:
            index_map = source_index_map(system)
        except (TypeError, ValueError) as exc:
            index_map = None
            bound = [
                label for label in self.table.index if int(label) in self._site_states
            ]
            logger.warning(
                "[htst] active table: source identities unavailable (%s: %s); "
                "dropping %d recycled row(s) with a site context instead of "
                "validating them, their (atom, reference) pairs are re-refined "
                "next step",
                type(exc).__name__,
                exc,
                len(bound),
            )
        dropped = []
        for label, row in self.table.iterrows():
            state = self._site_states.get(int(label))
            if state is None:
                if (
                    allow_pending
                    and label in self._pending_site_rows
                    and not bool(row["nu0_site_attempted"])
                    and row["nu0_source"] != SOURCE_SITE
                ):
                    continue
                dropped.append(label)
                continue
            if index_map is None:
                dropped.append(label)
                continue
            try:
                if not state.matches(
                    row, system, self.prefactor_service, index_map=index_map
                ):
                    dropped.append(label)
                    continue
                self.table.at[label, "atom_index"] = index_map[state.center_id]
                if neighbors_list is not None:
                    crop = self.crop_indices(label, system)
                    current = neighbors_list.get_neighbors(
                        "rcut", int(self.table.at[label, "atom_index"])
                    )
                    if set(crop) != set(current):
                        dropped.append(label)
                        continue
                nu0 = float(row["nu0"]) if row["nu0_status"] == NU0_OK else None
                if (
                    nu0 is not None
                    and not self.prefactor_service.settings.nu0_min_hz
                    <= nu0
                    <= self.prefactor_service.settings.nu0_max_hz
                ):
                    # Rebuild under the new acceptance policy; the old site
                    # value must not become a reference estimate by relabeling.
                    dropped.append(label)
                    continue
                rate = self.rate_constant.compute_rate(
                    float(row["energy_barrier"]), nu0
                )
                self.table.loc[label, "k"] = rate.rate
                self.table.loc[label, "k_prefactor"] = rate.prefactor
                self._site_states[int(label)] = replace(
                    state,
                    signature=row_signature(self.table.loc[label], state.center_id),
                )
            except HTSTRequestError as exc:
                # A configuration/authority error of the current service is not
                # an ordinary dependency change: report which row it hit and why.
                dropped.append(label)
                logger.warning(
                    "[htst] active event (atom %d, reference %d): dropped, the "
                    "prefactor service cannot describe its source: %s",
                    int(row["atom_index"]),
                    int(row["num_reference_event"]),
                    exc,
                )
            except (ValueError, TypeError, KeyError, IndexError, RuntimeError):
                dropped.append(label)
        if dropped:
            self.remove(dropped)
            logger.info(
                "active table: invalidated %d rows with changed or unknown full-source dependencies",
                len(dropped),
            )
        return len(dropped)

    def drop_reference_events(self, removed: Iterable[int]) -> int:
        """Drop every row whose ``num_reference_event`` left the catalogue.

        Recycled rows are dropped too: a row is only as valid as its
        reference. The surviving rows are relabelled ``0..n-1`` through
        :meth:`remove`, so the transient full-saddle store follows them.

        Parameters
        ----------
        removed : Iterable[int]
            Logical ids removed from the reference table
            (``RemovedReferences.idx_refs``).

        Returns
        -------
        int
            Number of rows dropped (logged when non-zero).

        """
        removed_ids = {int(i) for i in removed}
        if not removed_ids or len(self.table) == 0:
            return 0
        mask = self.table["num_reference_event"].astype(int).isin(removed_ids)
        labels = [int(label) for label in self.table.index[mask]]
        if labels:
            self.remove(labels)
            logger.info(
                "active table: dropped %d row(s) whose reference event was "
                "removed (%s)",
                len(labels),
                sorted(removed_ids),
            )
        return len(labels)

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
            outputs = list(events)
            dfactive = [self.build_event_series(e) for e in outputs]
        elif isinstance(events, EventRefinementOutput):
            outputs = [events]
            dfactive = self.build_event_series(events)
        else:
            raise TypeError(
                "Input 'events' must be an EventRefinementOutput dataclass or a list of it."
            )
        if self.uses_prefactors:
            self._require_htst_columns("add_events")
        first_label = len(self.table)
        self.add(dfactive)
        if not self.uses_prefactors:
            for offset, output in enumerate(outputs):
                if output.crop_atom_ids is not None:
                    self._constant_crop_ids[first_label + offset] = tuple(
                        output.crop_atom_ids
                    )
        if self.uses_prefactors:
            # The full refined saddle travels beside the row (never in it) until
            # request_site_prefactors consumes it.
            for offset, output in enumerate(outputs):
                self._pending_site_rows.add(first_label + offset)
                full = output.full_saddle_positions
                if full is not None:
                    self._full_saddles[first_label + offset] = np.asarray(
                        full, dtype=float
                    ).copy()
                    self._full_saddle_constraints[first_label + offset] = (
                        output.constraints
                    )

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

        self._remap_stores(list(self.table.index))
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
        if self.uses_prefactors and event_refinement_output.crop_atom_ids is not None:
            dfactive["crop_atom_ids"] = tuple(event_refinement_output.crop_atom_ids)
        return dfactive

    def _capture_site_state(self, label, request, calculation=None):
        from .htst.provenance import RequestSnapshot
        from .htst.site_state import SiteState, row_signature

        source = RequestSnapshot.capture(request)
        if calculation is not None:
            calculation.validate()
            if (
                calculation.direction != "forward"
                or calculation.provenance.source != source
                or calculation.provenance.method != self.prefactor_service.method
            ):
                raise RuntimeError(
                    "site result does not belong to its submitted source and method"
                )
        center_id = source.constraints.atom_ids[source.center_index]
        self._site_states[int(label)] = SiteState(
            source,
            self.prefactor_service.method,
            row_signature(self.table.loc[label], center_id),
            calculation,
            fresh_service=self.prefactor_service,
        )
        self._pending_site_rows.discard(int(label))

    def _site_request(self, label, system, saddle):
        from .physics import resolve_event_constraints

        atom = int(self.table.at[label, "atom_index"])
        constraints = self._full_saddle_constraints.get(int(label))
        if constraints is None:
            # Under active volume the resolver returns the user+AV union; a
            # request receives the USER constraints only, so the rmov shell
            # never enters free_region selection (contracts 7f policy 5).
            constraints = resolve_event_constraints(
                self.prefactor_service.config,
                system.positions,
                system.types,
                system.cell,
                system.pbc,
                atom,
                system.index,
                user_constraints=self.prefactor_service.global_constraints,
            ).user_view()
        return self.prefactor_service.build_request(
            event_key=(
                "site",
                int(label),
                atom,
                int(self.table.at[label, "num_reference_event"]),
            ),
            min1_positions=system.positions,
            saddle_positions=saddle,
            min2_positions=system.positions,
            types=system.types,
            cell=system.cell,
            pbc=system.pbc,
            center_index=atom,
            constraints=constraints,
        )

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
        participate. All retained rows first pass the full-source dependency
        guard; valid ``"F"``/``"B"`` approximations require no site Hessian.

        Geometry: the request carries the current full minimum as ``min1``
        and the full pARTn-refined saddle handed over by ``Refinement.execute``
        (``EventRefinementOutput.full_saddle_positions``, held in the
        transient side store) as ``saddle``; only the forward direction is
        computed (``compute_backward=False``), so ``min2`` is a copy of
        ``min1`` and is never used. Rebuilding the saddle from the ``rcut``
        crop pasted into the minimum (the previous construction) left every
        atom between ``rcut`` and ``free_radius`` plus the potential cutoff at
        its minimum position and biased the site estimate (-4.5 % on SW-Si at
        ``rcut`` 6.3 Å, -31 % on Cu); the full saddle removes that bias and
        makes ``free_radius`` independent of ``rcut``.

        Refinement records stable global crop IDs in the same order as its
        saddle and final arrays. Requests and reconstruction resolve those IDs
        against the current source, independent of neighbor-list ordering.
        The stored crop is checked against the full saddle before submission;
        changed crop membership invalidates a retained row. A legacy fresh
        output can acquire IDs only when its full saddle proves the mapping.

        The full saddles are released once the batch has been submitted and
        resolved (also when a worker failure propagates).

        Parameters
        ----------
        system : System
            Current system; its positions are the initial minimum of every
            active event.
        neighbors_list : NeighborsList
            The neighbour list refinement cropped with.

        Crop-only rows (contracts section 7d, F1): an eligible row whose
        producer handed over no full refined saddle keeps its inherited
        estimate untouched (``nu0``, ``nu0_status``, ``nu0_source``, ``k``,
        ``k_prefactor``), is marked ``nu0_site_attempted``, is counted under
        ``no_geometry`` and logged once at info level; nothing is submitted
        for it. Explicit crop IDs are required without a full saddle. The
        fallback's current source is recorded without inventing a calculation.
        Later changes invalidate it under the same dependency guard.

        Rows without stable crop identities (none stored and no full saddle
        to prove a mapping, or identities that do not resolve in the current
        source) are not crop-only fallbacks: reconstruction cannot resolve
        them, so they are dropped before selection with one WARNING per row
        (``_warn_crop_identity``), counted under no summary key and never
        kept behind a fallback context. Their ``(atom, reference)`` pair is
        re-refined next step. The mapping-mismatch ``RuntimeError`` is
        unchanged.

        Returns
        -------
        dict[str, int]
            ``{"attempted", "ok", "rejected", "no_geometry"}`` counts for this
            call (all zero in constant mode or when nothing was eligible);
            ``attempted == ok + rejected + no_geometry``.

        Raises
        ------
        RuntimeError
            If a row is eligible but no prefactor service is attached, or if a
            row's crop does not match the full saddle at the current neighbour
            mapping.
        ValueError
            If the table (a caller-supplied frame) lacks the HTST columns.

        """
        summary = {"attempted": 0, "ok": 0, "rejected": 0, "no_geometry": 0}
        self.step_site_rejections = {}
        if not self.uses_prefactors or len(self.table) == 0:
            return summary
        from .htst.result import PrefactorRejection

        self._require_htst_columns("request_site_prefactors")
        self.validate_recycled(system, neighbors_list)
        # Rows without stable crop identities cannot be reconstructed; they
        # are removed once, after every label-addressed update of this call.
        identityless: list[Any] = []
        if self.prefactor_service is not None:
            # Even an unrefined reference approximation needs a full dependency
            # context before it may suppress work in a later step. This is a
            # fallback context, not a claim that its spectrum was calculated here.
            for label, row in self.table.iterrows():
                if row["refined"] != "T" and label in self._pending_site_rows:
                    try:
                        self.crop_indices(label, system, neighbors_list, capture=True)
                    except ValueError as exc:
                        self._warn_crop_identity(label, exc, "fallback context")
                        identityless.append(label)
                        continue
                    request = self._site_request(label, system, system.positions)
                    self._capture_site_state(label, request)
        eligible = (self.table["refined"] == "T") & ~self.table[
            "nu0_site_attempted"
        ].astype(bool)
        rows = self.table[eligible]
        if rows.empty:
            self._drop_identityless_rows(identityless)
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
        no_geometry: list[Any] = []
        submitted = {}
        try:
            for idx, row in rows.iterrows():
                atom = int(row["atom_index"])
                try:
                    neighbors = self.crop_indices(
                        idx, system, neighbors_list, capture=True
                    )
                except ValueError as exc:
                    # Not a crop-only fallback: without stable identities the
                    # row cannot be reconstructed, so it leaves the table
                    # before selection instead of being counted as attempted.
                    self._warn_crop_identity(idx, exc, "site request")
                    identityless.append(idx)
                    continue
                saddle_crop = np.asarray(row["saddle_positions"], dtype=float)
                if saddle_crop.shape != (len(neighbors), 3) or atom not in neighbors:
                    raise RuntimeError(
                        f"active row {idx} (atom {atom}): stored crop of shape "
                        f"{saddle_crop.shape} does not match the current rcut "
                        f"mapping of {len(neighbors)} neighbours; the site request "
                        "must use the neighbour list refinement cropped with"
                    )
                full_saddle = self._full_saddles.get(int(idx))
                if full_saddle is None:
                    # Crop-only output: no stationary geometry to request a
                    # site Hessian from. The inherited estimate stands.
                    no_geometry.append(idx)
                    submitted[idx] = self._site_request(idx, system, positions)
                    continue
                if full_saddle.shape != positions.shape or not np.array_equal(
                    full_saddle[neighbors], saddle_crop
                ):
                    raise RuntimeError(
                        f"active row {idx} (atom {atom}): the stored saddle crop is "
                        "not the full refined saddle at the current rcut mapping; "
                        "the site request must use the neighbour list refinement "
                        "cropped with"
                    )
                request = self._site_request(idx, system, full_saddle)
                key = request.event_key
                requests.append(request)
                submitted[idx] = request
                keys.append((idx, key))
            results = self.prefactor_service.compute(requests, compute_backward=False)
        finally:
            # Release the full arrays: the requests hold their own copies and
            # a row is attempted at most once.
            self._full_saddles = {}
            self._full_saddle_constraints = {}
            self._pending_site_rows.clear()
        wall = self.prefactor_service.last_batch_wall_s
        for idx in no_geometry:
            self.table.loc[idx, "nu0_site_attempted"] = True
            if idx in submitted:
                self._capture_site_state(idx, submitted[idx])
            summary["attempted"] += 1
            summary["no_geometry"] += 1
            logger.info(
                "[htst] active event (atom %d, reference %d): no full refined "
                "saddle available (crop-only refinement output); keeping the "
                "inherited %s estimate, no site request submitted",
                int(self.table.loc[idx, "atom_index"]),
                int(self.table.loc[idx, "num_reference_event"]),
                self.table.loc[idx, "nu0_source"],
            )
        for idx, key in keys:
            pre = results[key]
            estimate = pre.forward
            calculation = pre.calculation("forward")
            if calculation is not None and calculation.estimate != estimate:
                raise RuntimeError(
                    "site result estimate and producing calculation disagree"
                )
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
                    "[htst] active event (atom %d, reference %d): site nu0 = %.4e Hz "
                    "(n_free %d, batch %.3f s)",
                    atom,
                    ref,
                    estimate.nu0_hz,
                    pre.n_free,
                    wall,
                )
            else:
                # A skipped forward direction cannot happen (only the backward
                # one is skipped); a rejected one keeps the row as it is.
                summary["rejected"] += 1
                code = estimate.reason_code.value if estimate.reason_code else "skipped"
                self.step_site_rejections[code] = (
                    self.step_site_rejections.get(code, 0) + 1
                )
                if estimate.reason_code is PrefactorRejection.NONSTATIONARY_GEOMETRY:
                    # A stationarity rejection is a silent k0 fallback unless
                    # it is reported here, with the tolerance that decided it.
                    logger.warning(
                        "[htst] active event (atom %d, reference %d): site "
                        "prefactor rejected (%s: %s; force_tol %s eV/A); keeping "
                        "the %s estimate (n_free %d, batch %.3f s)",
                        atom,
                        ref,
                        code,
                        estimate.reason,
                        self.prefactor_service.settings.force_tol,
                        self.table.loc[idx, "nu0_source"],
                        pre.n_free,
                        wall,
                    )
                else:
                    logger.info(
                        "[htst] active event (atom %d, reference %d): site "
                        "prefactor rejected (%s: %s); keeping the %s estimate "
                        "(n_free %d, batch %.3f s)",
                        atom,
                        ref,
                        code,
                        estimate.reason,
                        self.table.loc[idx, "nu0_source"],
                        pre.n_free,
                        wall,
                    )
            self._capture_site_state(idx, submitted[idx], calculation)
        nonstationary = self.step_site_rejections.get(
            PrefactorRejection.NONSTATIONARY_GEOMETRY.value, 0
        )
        if nonstationary:
            logger.warning(
                "[htst] %d site prefactor request(s) rejected as %s this step "
                "(force_tol %s eV/A): those events use the constant k0 prefactor",
                nonstationary,
                PrefactorRejection.NONSTATIONARY_GEOMETRY.value,
                self.prefactor_service.settings.force_tol,
            )
        self._drop_identityless_rows(identityless)
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

        The surviving rows are relabelled ``0..n-1``; the transient full
        saddles follow their rows.

        Parameters
        ----------
        ind : int
            index of the row to be removed
        """
        dropped = {int(ind)} if np.isscalar(ind) else {int(i) for i in ind}
        kept = [label for label in self.table.index if int(label) not in dropped]
        self._remap_stores(kept)
        self.table = self.table.drop(ind)
        self.table = self.table.reset_index(drop=True)

    def _remap_stores(self, kept) -> None:
        """Apply the same row-label map to every transient store."""
        self._constant_crop_ids = {
            new: self._constant_crop_ids[int(old)]
            for new, old in enumerate(kept)
            if int(old) in self._constant_crop_ids
        }
        self._site_states = {
            new: self._site_states[int(old)]
            for new, old in enumerate(kept)
            if int(old) in self._site_states
        }
        self._pending_site_rows = {
            new for new, old in enumerate(kept) if int(old) in self._pending_site_rows
        }
        if self._full_saddles:
            self._full_saddles = {
                new: self._full_saddles[int(old)]
                for new, old in enumerate(kept)
                if int(old) in self._full_saddles
            }

        self._full_saddle_constraints = {
            new: self._full_saddle_constraints[int(old)]
            for new, old in enumerate(kept)
            if int(old) in self._full_saddle_constraints
        }

    def remove_duplicates(self, cell, neighbors_list: NeighborsList = None) -> None:
        """Loop over all active events in the DataFrame, check if there are duplicates by computing delr."""

        from .physics import _indices

        if neighbors_list is not None and self.uses_prefactors:
            for label in list(self._pending_site_rows):
                if label in self.table.index:
                    try:
                        self.crop_indices(
                            label, neighbors_list.system, neighbors_list, capture=True
                        )
                    except ValueError as exc:
                        # A row without stable identities cannot be compared by
                        # identity; it is left to the geometric checks below and
                        # to the no_geometry fallback, never aborting the step.
                        self._warn_crop_identity(label, exc, "duplicate removal")

        def align(first, second, *, shared=False):
            a, b = self.table.loc[first], self.table.loc[second]
            left, right = np.asarray(a.saddle_positions), np.asarray(b.saddle_positions)
            try:
                ids_a, ids_b = _indices(a.crop_atom_ids), _indices(b.crop_atom_ids)
            except (AttributeError, TypeError, ValueError):
                if (
                    not self.uses_prefactors
                    and not shared
                    and left.shape == right.shape
                ):
                    return left, right
                return None
            if left.shape != (len(ids_a), 3) or right.shape != (len(ids_b), 3):
                return None
            if not shared and set(ids_a) != set(ids_b):
                return None
            common = sorted(set(ids_a) & set(ids_b))
            if not common:
                return None
            return left[[ids_a.index(i) for i in common]], right[
                [ids_b.index(i) for i in common]
            ]

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
            for jdx in subset.index:
                if jdx <= idx:
                    continue  # dont compute twice
                aligned = align(idx, jdx)
                if aligned is None:
                    continue
                pos_ref, pos_comp = aligned
                delr = compute_delr(
                    pos_ref,
                    pos_comp,
                    cell,
                    True if neighbors_list is None else neighbors_list.system.pbc,
                )
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
                    if "crop_atom_ids" in self.table.columns:
                        for jdx in indices[i + 1 :]:
                            if central_atom1 == subset.loc[jdx, "atom_index"]:
                                continue
                            try:
                                other_ids = _indices(
                                    self.table.at[jdx, "crop_atom_ids"]
                                )
                                center_id = int(
                                    neighbors_list.system.index[int(central_atom1)]
                                )
                            except (TypeError, ValueError, IndexError):
                                continue
                            if center_id not in other_ids:
                                continue
                            aligned = align(idx, jdx, shared=True)
                            if (
                                aligned is not None
                                and compute_delr(
                                    *aligned, cell, neighbors_list.system.pbc
                                )
                                < self.config.psr.matching_score_thr
                            ):
                                duplicates.append(jdx)
                        continue
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
