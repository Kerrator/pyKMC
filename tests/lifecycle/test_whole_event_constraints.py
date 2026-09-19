"""Independent R08 geometry/constraint oracle, with actual producing requests.

Only the worker's equal 5 THz directional values are declared protocol inputs.
No native stationary geometry, Hessian, force or numerical-rate claim is made.
Service validation, source authority, catalogue admission/resolution and typed
producing records are real. No matcher or identity predicate is patched.
"""

from concurrent.futures import Future
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

import pykmc
from pykmc.config import Config, RateConstantConfig, RegionConfig
from pykmc.event_table import ReferenceEventTable
from pykmc.htst.free_region import common_free_indices
from pykmc.htst.provenance import CalculationProvenance
from pykmc.htst.result import DirectionalPrefactor, EventPrefactors
from pykmc.physics import ResolvedConstraints
from pykmc.rate_constant import create_rate_constant
from pykmc.rate_constant.prefactors import PrefactorService
from pykmc.result import EventSearchOutput


CELL = 20.0 * np.eye(3)
PBC = (False, False, False)
TYPES = ("Si", "Ge", "C", "Ni", "Ni")
IDS = (42, 8, 91, 17, 63)
SPECIES = ("Si", "Ge", "C", "Ni", "O")
MASSES = (28.0855, 72.63, 12.011, 58.6934, 15.999)
INITIAL = np.array(
    [[0, 0, 0], [1, 0, 0.3], [0, 2, 1.6], [1, 1, 2], [-1, -1, 2]],
    dtype=float,
)
SADDLE = np.array(
    [[0, 0, 0], [0, 0, 0.3], [0, 0, 1.6], [1, 1, 2], [-1, -1, 2]],
    dtype=float,
)
ROTATION = np.diag([-1.0, -1.0, 1.0])
PERMUTATION = np.array([0, 1, 2, 4, 3])
FINAL = (INITIAL @ ROTATION)[PERMUTATION]
METHOD = "identity-protocol"


class EqualEstimateWorker:
    """Declare only spectra outcomes; capture the actual validated input."""

    def __init__(self):
        self.calls = []

    def submit(self, operation, *, request, compute_backward=True):
        assert operation == "compute_event_prefactors"
        assert compute_backward is True
        request.validate()
        free = tuple(int(i) for i in common_free_indices(request))
        estimate = DirectionalPrefactor(
            5e12, "ok", None, None, len(free), 3 * len(free), 1
        )
        provenance = CalculationProvenance.capture(
            request, request, method=METHOD, free_indices=free
        )
        result = EventPrefactors(
            request.event_key,
            estimate,
            estimate,
            METHOD,
            len(free),
            request.settings,
            provenance=provenance,
        )
        self.calls.append((request, result))
        future = Future()
        future.set_result(result)
        return future


def configuration(fixed_rows):
    root = Path(pykmc.__file__).resolve().parent.parent
    cfg = Config.from_ini_file(str(root / "tests/data/input.in"))
    return cfg.model_copy(
        update={
            "rateconstant": RateConstantConfig(
                style="htst", k0=1.0, free_radius=30.0, premin=False
            ),
            "control": cfg.control.model_copy(
                update={"reference_table": None, "active_volume": False}
            ),
            "lammps": cfg.lammps.model_copy(
                update={"pair_style": "lj/cut 2.5", "pair_coeff": "* * 0.05 1.0"}
            ),
            "atomicenvironment": cfg.atomicenvironment.model_copy(
                update={"atom_coloring_mode": "full"}
            ),
            "frozen_atoms": RegionConfig(indices=list(fixed_rows)),
        }
    )


def row(initial, saddle, final):
    """The catalogue crop is the first five atoms of the actual source."""
    return pd.Series(
        {
            "idx_ref": -1,
            "event_id": "same-topology",
            "id_final": "same-topology",
            "id_saddle": "saddle",
            "idx_backward": -1,
            "initial_positions": initial[:5].copy(),
            "saddle_positions": saddle[:5].copy(),
            "final_positions": final[:5].copy(),
            "types": list(TYPES),
            "energy_barrier": 0.5,
            "k": 0.0,
            "move_atom_idx": 1,
            "sym_matrix": [np.eye(3)],
            "sym_perm": [np.arange(5)],
            "dra": 1.0,
            "k_prefactor": 1.0,
            "nu0": np.nan,
            "nu0_status": "pending",
            "nu0_reason": "",
        }
    )


def admit(fixed_rows, *, outside_crop=False):
    geometry = [INITIAL.copy(), SADDLE.copy(), FINAL.copy()]
    types, ids = TYPES, IDS
    if outside_crop:
        geometry = [np.vstack((p, [2.2, 1.2, 2.0])) for p in geometry]
        types, ids = (*types, "O"), (*ids, 105)
    originals = [p.copy() for p in geometry]
    cfg = configuration(fixed_rows)
    constraints = ResolvedConstraints.resolve(
        geometry[0], types, cfg.frozen_atoms, ids, cell=CELL, pbc=PBC
    )
    frozen_references = constraints.fixed_positions
    # These are valid stationary-triplet constraint coordinates even in the
    # negative case: the fixed spectator itself never moved between vertices.
    for positions in geometry:
        constraints.validate_positions(positions)
    worker = EqualEstimateWorker()
    service = PrefactorService(
        cfg,
        worker,
        create_rate_constant(cfg.rateconstant),
        species_masses=(SPECIES, MASSES),
        global_constraints=constraints,
        method=METHOD,
    )
    table = ReferenceEventTable(cfg, prefactor_service=service)
    outcome = table._admit_series(
        row(*geometry), row(geometry[2], geometry[1], geometry[0])
    )
    assert outcome.is_ok()
    admission = outcome.ok_value()
    table.add(admission.frame, reverse_idx_ref=admission.reverse_idx_ref)
    forward = int(admission.frame.iloc[0].idx_ref)
    backward = (
        int(admission.frame.iloc[1].idx_ref) if len(admission.frame) > 1 else None
    )
    event = EventSearchOutput(
        central_atom_index=1,
        min1_positions=geometry[0],
        saddle_positions=geometry[1],
        min2_positions=geometry[2],
        dE_forward=0.5,
        dE_backward=0.5,
        move_atom_index=1,
        cell=CELL.copy(),
        types=list(types),
        constraints=constraints,
    )
    table._resolve_prefactors([(forward, backward, admission, event)], PBC)
    assert len(worker.calls) == service.n_submitted == 1
    request, result = worker.calls[0]
    assert request.constraints == request.user_constraints == constraints
    assert request.constraints.atom_ids == ids
    assert request.constraints.fixed_ids == tuple(ids[i] for i in fixed_rows)
    assert request.species == SPECIES
    assert request.masses == MASSES
    assert result.provenance.free_indices == tuple(
        i for i in range(len(ids)) if i not in fixed_rows
    )
    assert result.provenance.zone_indices == tuple(range(len(ids)))
    assert result.provenance.source.types == types
    assert result.provenance.source == result.provenance.produced
    assert constraints.fixed_positions == frozen_references
    for current, original, field in zip(
        geometry, originals, ("min1_positions", "saddle_positions", "min2_positions")
    ):
        np.testing.assert_array_equal(current, original)
        np.testing.assert_array_equal(
            getattr(result.provenance.source, field), original
        )
        assert not np.shares_memory(getattr(request, field), current)
    return table, result


def assert_reciprocal(table, result):
    assert len(table.table) == 2, "A map must preserve full source and constraints"
    links = dict(
        zip(table.table.idx_ref.astype(int), table.table.idx_backward.astype(int))
    )
    assert len(set(links.values())) == 2
    assert all(other != own and links[other] == own for own, other in links.items())
    calculations = []
    for _, event in table.table.iterrows():
        calculation = table.prefactor_archive.calculation_for(int(event.idx_ref), event)
        assert calculation is not None
        calculation.validate()
        assert calculation.provenance == result.provenance
        calculations.append(calculation)
    assert {calculation.direction for calculation in calculations} == {
        "forward",
        "backward",
    }
    assert table.table.nu0_status.tolist() == ["ok", "ok"]
    assert table.table.nu0.astype(float).tolist() == pytest.approx([5e12, 5e12])


def test_geometric_reversal_swapping_fixed_and_free_spectators_keeps_two_directions():
    np.testing.assert_array_equal((INITIAL @ ROTATION)[PERMUTATION], FINAL)
    np.testing.assert_array_equal((FINAL @ ROTATION)[PERMUTATION], INITIAL)
    np.testing.assert_array_equal((SADDLE @ ROTATION)[PERMUTATION], SADDLE)
    np.testing.assert_array_equal(INITIAL[3:], FINAL[3:])
    np.testing.assert_array_equal(INITIAL[3:], SADDLE[3:])
    fixed = np.array([False, False, False, True, False])
    assert not np.array_equal(fixed[PERMUTATION], fixed)
    table, result = admit((3,))
    assert_reciprocal(table, result)


def test_same_reversal_with_both_spectators_fixed_can_collapse():
    fixed = np.array([False, False, False, True, True])
    np.testing.assert_array_equal(fixed[PERMUTATION], fixed)
    table, result = admit((3, 4))
    assert len(table.table) == 1
    event = table.table.iloc[0]
    assert int(event.idx_backward) == int(event.idx_ref)
    assert event.nu0_status == "ok"
    assert float(event.nu0) == pytest.approx(5e12)
    calculation = table.prefactor_archive.calculation_for(int(event.idx_ref), event)
    assert calculation is not None
    calculation.validate()
    assert calculation.provenance == result.provenance


def test_asymmetric_full_source_atom_outside_catalogue_crop_keeps_two_directions():
    spectator = np.array([2.2, 1.2, 2.0])
    assert np.linalg.norm(spectator @ ROTATION - spectator) > 5.0
    table, result = admit((3, 4, 5), outside_crop=True)
    assert len(result.provenance.source.types) == 6
    assert all(len(positions) == 5 for positions in table.table.saddle_positions)
    assert result.provenance.source.constraints.fixed_ids == (17, 63, 105)
    assert_reciprocal(table, result)
