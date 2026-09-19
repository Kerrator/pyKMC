"""Basin connectivity consumes current prefactor physics in both directions."""

import math
from types import SimpleNamespace

import numpy as np
import pytest
from pykmc.basins.exploration import BasinGenericEventExplorer

from . import prefactor_catalogue_helpers as helper


class OneApplicableEnvironment:
    """Already-classified state: only the forward topology is present here."""

    atomic_environment_list = ["event-17"]

    def get_atoms_with_id(self, event_id):
        assert event_id == "event-17"
        return [3]


@pytest.mark.parametrize("change_physics", [False, True], ids=["same", "changed"])
def test_basin_explore_guards_unmatched_reverse_before_use(tmp_path, change_physics):
    helper.require_schema2()
    potential = tmp_path / "same-path-physics.json"
    helper.write_potential(potential, a=0.0, scale=1.0, bias=0.3)
    original_service, original_worker = helper.service(potential)
    original = helper.calculate(original_service, "original-source")
    table = helper.table_for(
        original_service,
        [(17, original, "forward", 47), (47, original, "backward", 17)],
    )
    # Consistent forward/reverse topology, with labels unrelated to logical IDs.
    # Topology fields are fixture classifications, not numerical row/provenance edits.
    table.table["id_final"] = ["event-47", "event-17"]
    table.table.index = [9, 103]
    geometry = helper.geometric_rows(table.table)
    old_records = dict(table.prefactor_archive.calculations)
    old_reverse_rate = float(helper.row_at(table, 47).k)
    assert len(original_worker.calls) == 1

    scale = 2.0 if change_physics else 1.0
    mass = helper.MASS_B if change_physics else helper.MASS_A
    if change_physics:
        helper.write_potential(potential, a=0.0, scale=scale, bias=0.3)
    current, worker = helper.service(potential, mass=mass)
    current.config.basin.energy_thr = 2.0
    table.prefactor_service = current

    def before_submit(request):
        # The applicable old forward value must be unavailable before work.
        row = helper.row_at(table, 17)
        assert row.nu0_status != "ok" and np.isnan(row.nu0)
        assert row.k_prefactor == current.config.rateconstant.k0
        np.testing.assert_array_equal(request.min1_positions, helper.FIRST)
        np.testing.assert_array_equal(request.saddle_positions, helper.SADDLE)
        np.testing.assert_array_equal(request.min2_positions, helper.FINAL)

    worker.before_submit = before_submit
    explorer = BasinGenericEventExplorer(current.config, table)
    state = SimpleNamespace(environment=OneApplicableEnvironment())
    # Real explorer, real detector and real connectivity storage. Do not invoke
    # reference_estimate for the reverse first: that would hide the caller gap.
    explorer.explore(state, state_index=5, start_index=11)
    assert len(worker.calls) == int(change_physics)
    connectivity = explorer.get_connectivity_table()
    assert len(connectivity) == 1
    edge = connectivity.iloc[0]
    assert int(edge.state) == 5 and int(edge.state_connexion) == 11
    assert int(edge.event_connexion) == 17 and int(edge.central_atom) == 3
    assert int(edge.sym) == 0

    # Literal independent energy differences of the analytic stationary potential.
    # U(-1)=+.04s, U(0)=1s, U(+1)=-.04s. Uniform force scale and
    # mass scaling give nu_current/nu_original=sqrt(s*m_original/m_current).
    ratio = math.sqrt(scale * helper.MASS_A / mass)
    for direction, barrier in (("forward", 0.96 * scale), ("backward", 1.04 * scale)):
        frequency = getattr(original, direction).nu0_hz * ratio
        expected_rate = frequency * 1e-12 * math.exp(-barrier / (8.6173303e-5 * 300.0))
        assert float(edge[f"dE_{direction}"]) == pytest.approx(barrier, rel=1e-14)
        assert float(edge[f"k_{direction}"]) == pytest.approx(expected_rate, rel=1e-10)
    # The threshold also reads reverse geometry. New forward=1.92<2 while
    # reverse=2.08>2 makes this absorbing; using old reverse=1.04 makes it transient.
    assert bool(edge.transient) is (not change_physics)
    if change_physics:
        assert not math.isclose(float(edge.k_backward), old_reverse_rate, rel_tol=1e-6)
        request, backward, energies, result = worker.calls[0]
        assert backward is True and energies is True
        assert request.masses == (helper.MASS_B,)
        assert result.provenance.energies == pytest.approx((0.08, 2.0, -0.08))
        for idx, direction in ((17, "forward"), (47, "backward")):
            assert helper.linked_calculation(table, idx) == result.calculation(
                direction
            )
    assert all(
        table.prefactor_archive.calculations[key] == record
        for key, record in old_records.items()
    )
    helper.assert_geometry_unchanged(table.table, geometry)
    assert table.table.index.tolist() == [9, 103]

    # Repeating actual consumer work may append an edge but cannot trigger a
    # second calculation or revive the old reverse value.
    explorer.clear()
    explorer.explore(state, state_index=5, start_index=11)
    assert len(worker.calls) == int(change_physics)
    repeated = explorer.get_connectivity_table().iloc[0]
    for name in ("dE_forward", "dE_backward", "k_forward", "k_backward", "transient"):
        assert repeated[name] == edge[name]
