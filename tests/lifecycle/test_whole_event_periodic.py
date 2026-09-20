"""R08 periodic-lattice protocol oracle, not a native MEP or Hessian claim.

Actual validated requests and directional producing records feed the public
catalogue path and calculations_equivalent. The worker declares equal 5 THz;
the series builder supplies topology labels only. No matcher/lattice helper
is patched or called directly. Original arrays and records are immutable.
"""

from concurrent.futures import Future
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

import pykmc
from pykmc.config import Config, RateConstantConfig
from pykmc.event_table import ReferenceEventTable
from pykmc.htst.event_identity import calculations_equivalent
from pykmc.htst.free_region import common_free_indices
from pykmc.htst.provenance import CalculationProvenance, RequestSnapshot
from pykmc.htst.result import DirectionalPrefactor, EventPrefactors
from pykmc.physics import ResolvedConstraints
from pykmc.rate_constant import create_rate_constant
from pykmc.rate_constant.prefactors import PrefactorService
from pykmc.result import EventSearchOutput


TYPES = ("Si", "Ge", "Ge", "C", "C")
SPECIES = ("Si", "Ge", "C")
MASSES = (28.0855, 72.63, 12.011)
IDS = (42, 8, 91, 17, 63)
ROTATION = np.array([[0.0, 1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]])
PERMUTATION = np.array([0, 2, 1, 4, 3])
INITIAL = np.array(
    [[0, 0, 0], [1.6, 0.1, 0.2], [0.2, 1.0, 0.4], [0.3, 2.2, 0.5], [1.3, 0.4, 0.7]],
    dtype=float,
)
SADDLE = np.array(
    [[0, 0, 0], [1, 0, 0.1], [0, 1, 0.1], [2, 0, 0.3], [0, 2, 0.3]],
    dtype=float,
)
FINAL = (INITIAL @ ROTATION.T)[PERMUTATION]


class Worker:
    def __init__(self):
        self.requests = []
        self.results = []

    def submit(self, operation, *, request, compute_backward=True):
        assert operation == "compute_event_prefactors" and compute_backward is True
        request.validate()
        free = tuple(int(i) for i in common_free_indices(request))
        assert free == (0, 1, 2, 3, 4)
        estimate = DirectionalPrefactor(5e12, "ok", None, None, 5, 15, 1)
        result = EventPrefactors(
            request.event_key,
            estimate,
            estimate,
            "periodic-identity-protocol",
            5,
            request.settings,
            provenance=CalculationProvenance.capture(
                request, request, method="periodic-identity-protocol", free_indices=free
            ),
        )
        self.requests.append(request)
        self.results.append(result)
        future = Future()
        future.set_result(result)
        return future


def configuration():
    root = Path(pykmc.__file__).resolve().parent.parent
    original = Config.from_ini_file(str(root / "tests/data/input.in"))
    return original.model_copy(
        update={
            "control": original.control.model_copy(
                update={"reference_table": None, "active_volume": False}
            ),
            "frozen_atoms": None,
            "atomicenvironment": original.atomicenvironment.model_copy(
                update={"atom_coloring_mode": "full"}
            ),
            "lammps": original.lammps.model_copy(
                update={"pair_style": "lj/cut 2.5", "pair_coeff": "* * 0.05 1.0"}
            ),
            "rateconstant": RateConstantConfig(
                style="htst", k0=1.0, free_radius=30.0, premin=False
            ),
        }
    )


def row(initial, saddle, final):
    return pd.Series(
        {
            "idx_ref": -1,
            "idx_backward": -1,
            "event_id": "same-topology",
            "id_final": "same-topology",
            "id_saddle": "declared-saddle",
            "initial_positions": initial.copy(),
            "saddle_positions": saddle.copy(),
            "final_positions": final.copy(),
            "types": list(TYPES),
            "energy_barrier": 0.5,
            "k": 0.0,
            "move_atom_idx": 0,
            "sym_matrix": [np.eye(3)],
            "sym_perm": [np.arange(5)],
            "dra": 1.0,
            "k_prefactor": 1.0,
            "nu0": np.nan,
            "nu0_status": "pending",
            "nu0_reason": "",
        }
    )


def assert_independent_geometry_bounds(tolerance):
    """Coordinate/radius bounds, independent of IRA or lattice implementation."""
    assert tolerance == 0.1  # The documented default; never enlarge it here.
    np.testing.assert_array_equal((INITIAL @ ROTATION.T)[PERMUTATION], FINAL)
    np.testing.assert_array_equal((FINAL @ ROTATION.T)[PERMUTATION], INITIAL)
    np.testing.assert_array_equal((SADDLE @ ROTATION.T)[PERMUTATION], SADDLE)
    assert abs(np.linalg.det(INITIAL[1:4])) > 0.5
    # Unique Si pins translation modulo the lattice. Since every source vector
    # has length <2.3 and the shortest period is 8, a relative image change
    # leaves residual >3.4, much greater than two point errors (2*tolerance).
    radii = np.linalg.norm(INITIAL, axis=1)
    assert max(radii) < 2.3
    assert 8 - 2 * max(radii) > 2 * tolerance
    # Each equal-species pair has separated radial lengths. After allowing
    # translation error at the Si center, any admitted endpoint map must use
    # the declared pair swap; same-index alternative assignments are excluded.
    assert abs(radii[1] - radii[2]) > 2 * tolerance
    assert abs(radii[3] - radii[4]) > 2 * tolerance
    # For either negative cell, a lattice isometry can only change axis signs.
    # Source Ge1=(1.6,.1,.2) must reach target Ge2=(.1,1.6,.2).
    # The best signed-axis map has residual sqrt(1.5^2+1.5^2), before the
    # center's possible translation error. Even after that error it exceeds tau.
    lower_bound = np.sqrt(2 * 1.5**2) - tolerance
    assert lower_bound > tolerance


@pytest.mark.parametrize(
    "lengths,pbc,equivalent",
    [
        ((8.0, 8.0, 12.0), (True, True, True), True),
        ((8.0, 8.0, 12.0), (True, False, True), False),
        ((8.0, 10.0, 12.0), (True, True, True), False),
    ],
    ids=[
        "square-periodic-positive",
        "open-y-negative",
        "rectangular-periodic-negative",
    ],
)
def test_actual_periodic_lattice_controls_whole_event_collapse(
    lengths, pbc, equivalent, monkeypatch
):
    cfg = configuration()
    assert_independent_geometry_bounds(cfg.psr.matching_score_thr)
    cell = np.diag(lengths)
    constraints = ResolvedConstraints.resolve(
        INITIAL, TYPES, None, IDS, cell=cell, pbc=pbc
    )
    worker = Worker()
    service = PrefactorService(
        cfg,
        worker,
        create_rate_constant(cfg.rateconstant),
        species_masses=(SPECIES, MASSES),
        global_constraints=constraints,
        method="periodic-identity-protocol",
    )
    table = ReferenceEventTable(cfg, prefactor_service=service)
    expected_axes = tuple(bool(p) for p in pbc)

    def build_series(
        *,
        min1_positions,
        saddle_positions,
        min2_positions,
        index_move,
        dE_forward,
        dE_backward,
        cell,
        types,
        pbc=None,
    ):
        assert index_move == 0 and tuple(types) == TYPES
        assert dE_forward == dE_backward == 0.5
        assert tuple(bool(p) for p in pbc) == expected_axes
        np.testing.assert_array_equal(cell, np.diag(lengths))
        return (
            row(min1_positions, saddle_positions, min2_positions),
            row(min2_positions, saddle_positions, min1_positions),
        )

    monkeypatch.setattr(table, "_build_event_series", build_series)
    geometry = (INITIAL.copy(), SADDLE.copy(), FINAL.copy())
    originals = tuple(p.copy() for p in geometry)
    event = EventSearchOutput(
        central_atom_index=0,
        min1_positions=geometry[0],
        saddle_positions=geometry[1],
        min2_positions=geometry[2],
        dE_forward=0.5,
        dE_backward=0.5,
        move_atom_index=0,
        cell=cell.copy(),
        types=list(TYPES),
        constraints=constraints,
        prefactor_geometry=tuple(p.copy() for p in geometry),
    )
    admitted = table.add_events([event], pbc=pbc)
    assert len(admitted) == 1 and admitted[0].is_ok()
    assert len(worker.requests) == service.n_submitted == 1
    request, result = worker.requests[0], worker.results[0]
    assert request.constraints == request.user_constraints == constraints
    assert request.constraints.atom_ids == IDS
    assert request.constraints.fixed_ids == ()
    assert request.pbc == pbc
    assert result.provenance.source == RequestSnapshot.capture(request)
    assert result.provenance.source == result.provenance.produced
    assert (
        result.provenance.free_indices
        == result.provenance.zone_indices
        == tuple(range(5))
    )
    forward, backward = result.calculation("forward"), result.calculation("backward")
    assert (
        calculations_equivalent(
            forward,
            backward,
            tolerance=cfg.psr.matching_score_thr,
            kmax_factor=cfg.ira.kmax_factor,
        )
        is equivalent
    )
    assert len(table.table) == (1 if equivalent else 2)
    links = {int(r.idx_ref): int(r.idx_backward) for _, r in table.table.iterrows()}
    assert all(links[back] == own for own, back in links.items())
    assert all((own == back) is equivalent for own, back in links.items())
    assert table.table.nu0.astype(float).tolist() == pytest.approx([5e12] * len(links))
    assert table.table.nu0_status.tolist() == ["ok"] * len(links)
    for calculation in (forward, backward):
        assert (
            table.prefactor_archive.calculations[calculation.calculation_id]
            == calculation
        )
    for _, stored in table.table.iterrows():
        assert (
            table.prefactor_archive.calculation_for(int(stored.idx_ref), stored)
            is not None
        )
    for actual, original in zip(geometry, originals, strict=True):
        np.testing.assert_array_equal(actual, original)
    for actual, original in zip(event.prefactor_geometry, originals, strict=True):
        np.testing.assert_array_equal(actual, original)
