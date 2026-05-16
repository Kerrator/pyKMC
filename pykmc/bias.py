"""Event selection bias for KMC simulations."""

from __future__ import annotations
from abc import ABC, abstractmethod
import numpy as np
import pandas as pd
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .event_table import ActiveEventTable, ReferenceEventTable
    from .system import System
    from .atomic_environment import AtomicEnvironment


class Bias(ABC):
    """Abstract base class for event selection bias.

    A bias decides whether a candidate event is accepted or rejected.
    :meth:`select` drives the loop: it draws a candidate from the current pool,
    checks :meth:`accept`, removes rejected events, and recomputes ``ktot`` and
    ``delta_t`` from the shrinking pool at each step.  The returned ``delta_t``
    and ``ktot`` therefore reflect the effective total rate at the moment of
    acceptance.  If every event is rejected the selection falls back to an
    unbiased draw from the full pool.

    Subclasses must implement :meth:`accept`.  Subclasses that need to cache
    per-step context (e.g. topology lookups) should override :meth:`_prepare`.

    Attributes
    ----------
    enabled : bool
        When ``False``, :meth:`select` falls back to an unbiased draw without
        calling :meth:`accept`.  Defaults to ``True``.
    """

    def __init__(self) -> None:
        self.enabled: bool = True

    @abstractmethod
    def accept(
        self,
        event: pd.Series,
        system: System,
        reference_table: ReferenceEventTable,
    ) -> bool:
        """Return True to accept the candidate event, False to reject it.

        Parameters
        ----------
        event : pd.Series
            One row of the active event table representing the candidate event.
        system : System
            Current atomic configuration used to read atom positions.
        reference_table : ReferenceEventTable
            Reference event table used to retrieve the moving-atom index
            within the event's local neighbourhood array.

        Returns
        -------
        bool
            True if the event satisfies the bias condition, False otherwise.
        """
        pass

    def _prepare(
        self,
        system: System,
        reference_table: ReferenceEventTable,
        atomic_environment: AtomicEnvironment,
    ) -> None:
        """Pre-loop hook called once per :meth:`select` invocation.

        Override in subclasses that need to cache per-step context (e.g. look
        up which atoms currently carry a given topology).  The base
        implementation is a no-op.

        Parameters
        ----------
        system : System
            Current atomic configuration.
        reference_table : ReferenceEventTable
            Reference event table.
        atomic_environment : AtomicEnvironment
            Current atomic environment (topology IDs per atom).
        """
        pass

    def select(
        self,
        selection_algorithm: callable,
        l_k: np.ndarray,
        active_table: ActiveEventTable,
        system: System,
        reference_table: ReferenceEventTable,
        atomic_environment: AtomicEnvironment | None = None,
    ) -> tuple[int, float, float]:
        """Select an event using the bias condition.

        If ``self.enabled`` is ``False`` the bias is bypassed and
        ``selection_algorithm`` is called directly on the full pool.

        Otherwise, :meth:`_prepare` is called once, then candidates are drawn
        one by one, removing each rejected event before the next draw.
        ``delta_t`` and ``ktot`` are recomputed from the current pool at every
        iteration.  Falls back to an unbiased draw when all events are rejected.

        Parameters
        ----------
        selection_algorithm : callable
            Function with signature ``(rates) -> (index, delta_t, ktot)``.
        l_k : np.ndarray
            Rate constants for all active events.
        active_table : ActiveEventTable
            Active event table providing event metadata.
        system : System
            Current atomic configuration.
        reference_table : ReferenceEventTable
            Reference event table.
        atomic_environment : AtomicEnvironment or None, optional
            Current atomic environment; forwarded to :meth:`_prepare`.

        Returns
        -------
        tuple[int, float, float]
            - int: index of the selected event in the active table.
            - float: time increment (from the pool at acceptance).
            - float: total rate constant (from the pool at acceptance).
        """
        if not self.enabled:
            return selection_algorithm(l_k)
        self._prepare(system, reference_table, atomic_environment)
        candidate_events = list(range(len(l_k)))
        while candidate_events:
            idx_in_candidates, delta_t, ktot = selection_algorithm(
                l_k[candidate_events]
            )
            idx = candidate_events[idx_in_candidates]
            if self.accept(active_table.table.loc[idx], system, reference_table):
                return idx, delta_t, ktot
            candidate_events.remove(idx)
        # fallback: all events rejected, unbiased selection from full pool
        return selection_algorithm(l_k)

    def _moving_atom_displacement(
        self,
        event: pd.Series,
        system: System,
        reference_table: ReferenceEventTable,
    ) -> np.ndarray:
        """Return the displacement of the moving atom for a candidate event.

        Parameters
        ----------
        event : pd.Series
            One row of the active event table.
        system : System
            Current atomic configuration.
        reference_table : ReferenceEventTable
            Reference table used to look up ``move_atom_idx``.

        Returns
        -------
        np.ndarray, shape (3,)
            Displacement vector (final − initial) of the moving atom.
        """
        atom_idx = int(event["atom_index"])
        num_ref = int(event["num_reference_event"])
        ref_row = reference_table.table[
            reference_table.table["idx_ref"] == num_ref
        ].iloc[0]
        move_atom_idx = int(ref_row["move_atom_idx"])
        final_pos = event["final_positions"][move_atom_idx]
        current_pos = system.positions[atom_idx]
        return final_pos - current_pos


class DirectionBias(Bias):
    """Accept only events that move targeted atoms along a prescribed direction.

    An event is accepted when the projection of the moving atom's displacement
    onto *direction* is greater than or equal to *threshold*.  The default
    threshold of 0 keeps any event that has a non-negative component along
    the desired direction.

    Parameters
    ----------
    direction : array-like, shape (3,)
        Desired direction vector (normalised internally).
    atom_indices : list[int] or None, optional
        Global indices of atoms to bias.  Events whose central atom is not
        in this set are always accepted.  When *None* (default) all atoms
        are subject to the bias.
    threshold : float, optional
        Minimum required projection onto *direction*.  Default is 0.
    """

    def __init__(
        self,
        direction: np.ndarray,
        atom_indices: list[int] | None = None,
        threshold: float = 0.0,
    ) -> None:
        super().__init__()
        d = np.asarray(direction, dtype=float)
        self._direction = d / np.linalg.norm(d)
        self._atom_set = set(atom_indices) if atom_indices is not None else None
        self._threshold = threshold

    def accept(
        self,
        event: pd.Series,
        system: System,
        reference_table: ReferenceEventTable,
    ) -> bool:
        if (
            self._atom_set is not None
            and int(event["atom_index"]) not in self._atom_set
        ):
            return True
        displacement = self._moving_atom_displacement(event, system, reference_table)
        return float(np.dot(displacement, self._direction)) >= self._threshold


class PointBias(Bias):
    """Accept only events that move targeted atoms toward or away from a point.

    For each candidate event the local direction toward *target_point* is
    computed from the atom's current position.  The event is accepted when
    the projection of the displacement onto that direction is greater than
    or equal to *threshold*.  To push atoms **away** from the target, pass a
    negative *threshold* (or flip *direction* by negating it explicitly and
    using :class:`DirectionBias`).

    Parameters
    ----------
    target_point : array-like, shape (3,)
        Reference point in Cartesian coordinates.
    atom_indices : list[int] or None, optional
        Global indices of atoms to bias.  Events whose central atom is not
        in this set are always accepted.  When *None* (default) all atoms
        are subject to the bias.
    threshold : float, optional
        Minimum required projection onto the direction toward *target_point*.
        Default is 0 (accept if moving toward the target).  Use a negative
        value to accept events moving away from the target.
    """

    def __init__(
        self,
        target_point: np.ndarray,
        atom_indices: list[int] | None = None,
        threshold: float = 0.0,
    ) -> None:
        super().__init__()
        self._target = np.asarray(target_point, dtype=float)
        self._atom_set = set(atom_indices) if atom_indices is not None else None
        self._threshold = threshold

    def accept(
        self,
        event: pd.Series,
        system: System,
        reference_table: ReferenceEventTable,
    ) -> bool:
        atom_idx = int(event["atom_index"])
        if self._atom_set is not None and atom_idx not in self._atom_set:
            return True
        current_pos = system.positions[atom_idx]
        to_target = self._target - current_pos
        dist = np.linalg.norm(to_target)
        if dist < 1e-10:
            return True
        local_direction = to_target / dist
        displacement = self._moving_atom_displacement(event, system, reference_table)
        return float(np.dot(displacement, local_direction)) >= self._threshold


class TopoBias(Bias):
    """Accept only events that reduce the distance between two topology defects.

    On each KMC step :meth:`_prepare` locates all atoms carrying
    ``topo_source`` and ``topo_target`` in the current atomic environment.
    :meth:`accept` then accepts a candidate event only if the moving atom
    belongs to the source topology and its displacement brings it closer to the
    nearest target-topology atom.  Events from non-source atoms are always
    accepted.  If either topology is absent the bias is inactive for that step.

    Parameters
    ----------
    topo_source : str | bytes
        Topology ID of the defect to move (e.g. vacancy graph ID).
    topo_target : str | bytes
        Topology ID of the defect to approach (e.g. interstitial graph ID).
    """

    def __init__(self, topo_source: str | bytes, topo_target: str | bytes) -> None:
        super().__init__()
        self._topo_source = topo_source
        self._topo_target = topo_target
        self._source_positions = None
        self._target_positions = None

    def _prepare(self, system, reference_table, atomic_environment) -> None:
        source_atoms = atomic_environment.get_atoms_with_id(self._topo_source)
        target_atoms = atomic_environment.get_atoms_with_id(self._topo_target)
        self._source_positions = system.positions[source_atoms] if source_atoms else None
        self._target_positions = system.positions[target_atoms] if target_atoms else None

    def accept(self, event, system, reference_table) -> bool:
        if self._source_positions is None or self._target_positions is None:
            return True
        current_pos = system.positions[int(event["atom_index"])]
        if not any(np.allclose(current_pos, s) for s in self._source_positions):
            return True
        displacement = self._moving_atom_displacement(event, system, reference_table)
        final_pos = current_pos + displacement
        current_min_dist = np.min(np.linalg.norm(self._target_positions - current_pos, axis=1))
        final_min_dist = np.min(np.linalg.norm(self._target_positions - final_pos, axis=1))
        return final_min_dist < current_min_dist
