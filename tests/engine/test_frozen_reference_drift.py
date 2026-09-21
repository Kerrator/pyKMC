"""A user-frozen atom that drifted from its initialisation reference (V6).

``resolve_event_constraints`` validates the CURRENT positions of the
user-frozen atoms against the reference captured at initialisation (1e-10 A)
and raises ``ConstraintViolationError`` (a ``ValueError``). That check sat
outside the ``Err`` guard of ``KMC._reconstruction_active_event``, unguarded
in ``Refinement.refine_single`` and in ``EventSearch.execute``, so a drift
left ``KMC.run`` as a bare ``ValueError`` (contracts 7f policy 5 forbids it).

Reconstruction and refinement now return the same
``Err(RECONSTRUCTION_INVALID_EVENT_DATA)`` their overlay guards produce; the
event search, where a drift breaks the invariant of the whole constrained
run rather than one catalogue row, aborts with a named RuntimeError whose
message names the atom, the drifted identity and the cause, after logging it.
"""

from __future__ import annotations

import numpy as np
import pytest

from pykmc.eventsearch import EventSearch
from pykmc.physics import ConstraintViolationError, ResolvedConstraints
from pykmc.result import ErrorType, Ok
from tests import test_av_search_transport as search
from tests.engine import test_policy5_user_constraints as p5

INVALID = ErrorType.RECONSTRUCTION_INVALID_EVENT_DATA
DRIFT = 0.5


class _UnreachedReconstruction:
    """Reconstruction double that must not be reached after a drift."""

    reached = 0

    def __init__(self, config, manager, **kwargs):
        pass

    def reconstruct(self, *args, **kwargs):
        type(self).reached += 1
        return Ok(object())


def test_reconstruction_of_a_drifted_frozen_atom_is_err(monkeypatch):
    sim, table, source = p5._kmc_case(monkeypatch, _UnreachedReconstruction)
    sim.global_constraints = ResolvedConstraints.resolve(
        source,
        sim.system.types,
        sim.config.frozen_atoms,
        sim.system.index,
        cell=sim.system.cell,
        pbc=sim.system.pbc,
    )
    assert sim.global_constraints.user_fixed_ids == (8,)
    # Control: the undrifted source resolves and reaches the reconstruction.
    _UnreachedReconstruction.reached = 0
    assert sim._reconstruction_active_event(0, table).is_ok()
    assert _UnreachedReconstruction.reached == 1
    # The user-frozen atom (row 1, id 8) has moved since initialisation.
    sim.system.positions[1, 0] += DRIFT
    drifted = sim.system.positions.copy()
    result = sim._reconstruction_active_event(0, table)
    assert not result.is_ok()
    assert result.err_value().type is INVALID
    assert "fixed reference" in result.err_value().message
    assert _UnreachedReconstruction.reached == 1, "no reconstruction after a drift"
    np.testing.assert_array_equal(sim.system.positions, drifted)
    assert len(table.table) == 1


@pytest.mark.parametrize("active", [True, False], ids=["av", "non-av"])
def test_refinement_of_a_drifted_frozen_atom_is_err_future(monkeypatch, active):
    cfg = p5.refine_config(active=active, thr=0.1)
    triplet = p5.refine_geometries()
    caller, manager, row, system = p5.refinery(monkeypatch, cfg, triplet)
    warnings: list[str] = []
    caller.loggers.warning = lambda name, msg, *a, **k: warnings.append(str(msg))
    system.positions[p5.R_USER_ROW, 0] += DRIFT
    drifted = system.positions.copy()
    result, context = p5._refine(caller, row)
    assert not result.is_ok()
    assert result.err_value().type is INVALID
    assert "fixed reference" in result.err_value().message
    assert manager.calls == [], "nothing is dispatched for a drifted source"
    assert context["num_reference_event"] == 5
    assert any("user-fixed" in m and "atom 1" in m for m in warnings), warnings
    np.testing.assert_array_equal(system.positions, drifted)


def test_event_search_on_a_drifted_frozen_atom_aborts_with_a_named_error():
    cfg = search.config()
    triplet = search.geometries(False)
    user = search.user_snapshot(cfg)
    assert user.fixed_ids == (8,)
    source = search.system(triplet[0])
    manager = search.SearchManager(triplet, user)
    messages: list[tuple[str, str]] = []
    log = search.logger()
    log.error = lambda name, msg, *a, **k: messages.append((name, str(msg)))
    caller = EventSearch(cfg, source, manager, log, global_constraints=user)
    # Control: the undrifted source searches.
    caller.execute([search.SEARCH_CENTER])
    assert len(caller.get_successes_results()) == 1
    row = list(source.index).index(8)
    source.positions[row, 0] += DRIFT
    with pytest.raises(RuntimeError) as captured:
        caller.execute([search.SEARCH_CENTER])
    text = str(captured.value)
    assert not isinstance(captured.value, ValueError)
    assert isinstance(captured.value.__cause__, ConstraintViolationError)
    assert f"atom {search.SEARCH_CENTER}" in text
    assert "frozen" in text and "initialisation" in text
    assert "fixed reference" in text
    assert len(manager.calls) == 1, "no search is dispatched for a drifted source"
    assert any("frozen" in m for _, m in messages), messages
