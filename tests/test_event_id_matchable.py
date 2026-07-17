"""Guard: accepted reference events must be matchable by the live classifier.

Refinement candidates are selected by ``reference_table.has_id_subset_table(
atomic_environment_list)`` -- a reference event can only ever be applied if its
``event_id`` equals some atom's id in the live ``AtomicEnvironment``
classification. ``_build_event_series`` instead RE-DERIVES ``event_id`` as an
unconditional graph hash of the mover's environment computed from the event's
own (active-volume-relaxed) geometry. Two mismatch channels orphan the event
(stored, but never selectable as a refinement candidate):

1. The live classifier coarse-grains atoms with coordination >= threshold to
   the label ``"crystal"``; a graph hash never equals ``"crystal"``.
2. The event geometry (AV-relaxed / direction-swapped min1) drifts from the
   live full-system geometry, so the recomputed hash differs from the
   classifier's hash for the same atom.

Observed in production (NiCr Ni95Cr05 slab, 1 vacancy, T400/T500, AV+HTST):
~45% of the reference table orphaned; at step ~1047 every current candidate
was an orphan -> "Refining 0 events" -> "All event reconstuctions failed." ->
silent rc=0 death at half the requested steps. The fatal event's true mover
was a defect atom with coordination exactly 8 == threshold -> "crystal".

Invariant pinned here: after ``add_events`` against a live classification,
every ACCEPTED forward event's ``event_id`` is present in the classifier's id
space. An event whose mover the classifier cannot see must be rejected, not
stored as an unmatchable orphan.
"""

from pathlib import Path

import numpy as np
import pytest
from ase.build import fcc100

from pykmc.atomic_environment import AtomicEnvironment
from pykmc.config import Config
from pykmc.event_table import ReferenceEventTable
from pykmc.neighbors_list import NeighborsList
from pykmc.result import EventSearchOutput
from pykmc.system import System

RNEI, RCUT = 3.0, 6.5
_INPUT = Path(__file__).resolve().parent / "data" / "input.in"


def _config() -> Config:
    cfg = Config.from_ini_file(str(_INPUT))
    cfg.rateconstant.style = "constant"
    cfg.rateconstant.k0 = 1.0
    cfg.atomicenvironment.rnei = RNEI
    cfg.atomicenvironment.rcut = RCUT
    cfg.atomicenvironment.atom_coloring_mode = "full"
    return cfg


def _vacancy_slab() -> tuple[System, int, np.ndarray]:
    """Ni fcc100 slab with one mid-slab vacancy; returns (system, mover, vac_site)."""
    atoms = fcc100("Ni", size=(6, 6, 6), a=3.52, vacuum=6.0, periodic=True)
    pos = atoms.get_positions()
    zmid = (pos[:, 2].min() + pos[:, 2].max()) / 2.0
    vac_idx = int(np.argmin(np.abs(pos[:, 2] - zmid)))
    vac_site = pos[vac_idx].copy()
    del atoms[vac_idx]

    system = System()
    system.positions = atoms.get_positions()
    system.cell = np.array(atoms.get_cell())
    system.types = atoms.get_chemical_symbols()
    system.pbc = atoms.get_pbc()
    system.update_positions(system.positions)

    d = np.linalg.norm(system.positions - vac_site, axis=1)
    mover = int(np.argmin(d))  # a first neighbor of the vacancy
    return system, mover, vac_site


def _classify(system: System, thr: int) -> AtomicEnvironment:
    nl = NeighborsList(system, RNEI, RCUT)
    return AtomicEnvironment(
        "coordination/graph",
        neighbors_list=nl.neighbors_list["rnei"],
        environment_list=nl.neighbors_list["rcut"],
        types=list(system.types),
        coordination_threshold=thr,
    )


def _fake_event(system: System, mover: int, vac_site: np.ndarray) -> EventSearchOutput:
    """A physically-shaped vacancy hop: mover slides into the vacancy site."""
    min1 = system.positions.copy()
    min2 = min1.copy()
    min2[mover] = vac_site
    saddle = min1.copy()
    saddle[mover] = 0.5 * (min1[mover] + vac_site)
    return EventSearchOutput(
        central_atom_index=mover,
        dE_forward=0.7,
        dE_backward=0.7,
        min1_positions=min1,
        saddle_positions=saddle,
        min2_positions=min2,
        # production delivers np.int64 (atom_map[index_move]); the plain-int
        # path crashes in _build_event_series (list == int is not broadcast)
        move_atom_index=np.int64(mover),
        cell=np.array(system.cell),
    )


def test_accepted_forward_event_is_matchable() -> None:
    """An accepted event's event_id must exist in the live classifier id space.

    threshold=8 reproduces the production death condition: the fcc vacancy
    neighbor (coordination >= 8) is classified 'crystal', while the event-side
    hash is an unconditional graph hash -> the stored event is an orphan.
    """
    system, mover, vac_site = _vacancy_slab()
    ae = _classify(system, 8)
    assert ae.atomic_environment_list[mover] == "crystal", (
        "test-geometry precondition: the mover must be coordination-crystal "
        "(the production death condition)"
    )
    table = ReferenceEventTable(_config())

    results = table.add_events(
        [_fake_event(system, mover, vac_site)],
        types=list(system.types),
        atomic_environment_list=ae.atomic_environment_list,
    )
    current_ids = set(ae.atomic_environment_list)
    for res in results:
        if res.is_ok():
            eid = res.ok_value().iloc[0]["event_id"]
            assert eid in current_ids, (
                "accepted forward event is ORPHANED: its event_id is not in the "
                "live classifier's id space, so has_id_subset_table can never "
                "select it and it can never be refined/applied (production "
                "death: 'Refining 0 events' -> 'All event reconstuctions "
                "failed'). Events with a classifier-invisible mover must be "
                "rejected, and matchable movers must be keyed by the "
                "classifier's id, not a re-derived hash."
            )


def test_hashed_mover_event_keyed_by_classifier_id() -> None:
    """With threshold=12 the defect mover IS hashed; event_id must equal it.

    Simulates the AV-relaxation drift channel: the event's own geometry is
    slightly perturbed relative to the live system (as AV-relaxed pARTn output
    always is), so a re-derived hash may disagree with the live classifier's
    hash for the same atom. Keying by the classifier id makes the event
    matchable by construction.
    """
    system, mover, vac_site = _vacancy_slab()
    ae = _classify(system, 12)
    mover_id = ae.atomic_environment_list[mover]
    assert mover_id != "crystal", (
        "test-geometry precondition: at threshold=12 the vacancy neighbor "
        "(coordination 11) must be graph-hashed"
    )
    table = ReferenceEventTable(_config())

    ev = _fake_event(system, mover, vac_site)
    # drift: push one of the mover's far neighbors across the rnei boundary in
    # the event's own copy of the geometry (the live system is untouched)
    nl = NeighborsList(system, RNEI, RCUT)
    neigh = list(nl.neighbors_list["rnei"][mover])
    far = max(
        neigh,
        key=lambda j: np.linalg.norm(system.positions[j] - system.positions[mover]),
    )
    direction = system.positions[far] - system.positions[mover]
    direction /= np.linalg.norm(direction)
    # 0.6 A pushes the ~2.49 A fcc first-shell bond past rnei=3.0, so the
    # event-geometry graph genuinely loses an edge relative to the live system
    for arr in (ev.min1_positions, ev.saddle_positions, ev.min2_positions):
        arr[far] = arr[far] + 0.6 * direction

    results = table.add_events(
        [ev],
        types=list(system.types),
        atomic_environment_list=ae.atomic_environment_list,
    )
    accepted = [r for r in results if r.is_ok()]
    assert accepted, "the drifted vacancy-hop event should still be accepted"
    eid = accepted[0].ok_value().iloc[0]["event_id"]
    assert eid == mover_id, (
        "forward event_id was re-derived from the event's own drifted geometry "
        f"({str(eid)[:24]}...) instead of using the live classifier's id of the "
        f"mover ({str(mover_id)[:24]}...); the event is an unmatchable orphan"
    )
