"""Early catalogue identity gates: geometric/protocol tests, no native MEP claim.

Real add_events, admission, IRA, typed service and row patching are exercised.
The graph/series builder is the only catalogue seam: it returns full four-row
triplets with explicitly shared topology labels, isolating topology collisions.
The worker declares protocol values; it never claims an actual LJ Hessian.
Authoring performed no imports or test/native launches.
"""

from concurrent.futures import Future
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

import pykmc
from pykmc.config import Config, RateConstantConfig
from pykmc.event_table import ReferenceEventTable
from pykmc.htst.free_region import common_free_indices
from pykmc.htst.provenance import CalculationProvenance, RequestSnapshot
from pykmc.htst.result import DirectionalPrefactor, EventPrefactors
from pykmc.physics import ResolvedConstraints
from pykmc.point_set_registration import check_match, simple_ira
from pykmc.rate_constant import create_rate_constant
from pykmc.rate_constant.prefactors import PrefactorService
from pykmc.result import ErrorType, EventSearchOutput


TYPES = ("Si", "Ge", "C", "Ni")
MASSES = (28.0855, 72.63, 12.011, 58.6934)
IDS = (42, 8, 91, 17)
CELL = np.eye(3) * 20.0
PBC = (False, False, False)
I = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 2.0, 0.0], [0.0, 0.0, 3.0]])
S = np.array([[0.0, 0.0, 0.0], [0.5, 0.6, 0.7], [0.0, 2.0, 0.0], [0.0, 0.0, 3.0]])
F = np.array([[0.0, 0.0, 0.0], [1.0, 0.5, 0.0], [0.0, 2.0, 0.0], [0.0, 0.0, 3.0]])
G = np.array([[0.0, 0.0, 0.0], [1.0, 1.5, 0.0], [0.0, 2.0, 0.0], [0.0, 0.0, 3.0]])


def configuration(style):
    root = Path(pykmc.__file__).resolve().parent.parent
    original = Config.from_ini_file(str(root / "tests/data/input.in"))
    return original.model_copy(
        update={
            "frozen_atoms": None,
            "control": original.control.model_copy(
                update={
                    "reference_table": None,
                    "active_volume": False,
                }
            ),
            "atomicenvironment": original.atomicenvironment.model_copy(
                update={
                    "atom_coloring_mode": "full",
                }
            ),
            "lammps": original.lammps.model_copy(
                update={
                    "pair_style": "lj/cut 2.5",
                    "pair_coeff": "* * .05 1.0 2.5",
                }
            ),
            "rateconstant": RateConstantConfig(
                style=style,
                k0=1.0,
                T=300.0,
                free_radius=10.0,
                premin=False,
            ),
        }
    )


def protocol_result(request):
    request.validate()
    free = tuple(int(i) for i in common_free_indices(request))
    assert free == tuple(range(4))
    estimate = DirectionalPrefactor(5e12, "ok", None, None, 4, 12, 1)
    return EventPrefactors(
        event_key=request.event_key,
        forward=estimate,
        backward=estimate,
        method="protocol",
        n_free=4,
        settings=request.settings,
        provenance=CalculationProvenance.capture(
            request,
            request,
            method="protocol",
            free_indices=free,
        ),
    )


class Worker:
    def __init__(self):
        self.requests = []

    def submit(self, operation, *, request, compute_backward=True):
        assert operation == "compute_event_prefactors"
        assert compute_backward is True
        self.requests.append(request)
        future = Future()
        future.set_result(protocol_result(request))
        return future


def row(initial, saddle, final, *, initial_id="A", final_id="B", barrier=0.5):
    return pd.Series(
        {
            "idx_ref": -1,
            "idx_backward": -1,
            "event_id": initial_id,
            "id_final": final_id,
            "id_saddle": "shared-S",
            "initial_positions": np.array(initial, copy=True),
            "saddle_positions": np.array(saddle, copy=True),
            "final_positions": np.array(final, copy=True),
            "types": list(TYPES),
            "energy_barrier": barrier,
            "k": 0.0,
            "move_atom_idx": 0,
            "dra": 0.5,
            "sym_matrix": [np.eye(3)],
            "sym_perm": [np.arange(4)],
            "k_prefactor": 1.0,
            "nu0": np.nan,
            "nu0_status": "pending",
            "nu0_reason": "",
        }
    )


def full_constraints(positions):
    return ResolvedConstraints.resolve(
        positions,
        TYPES,
        None,
        IDS,
        cell=CELL,
        pbc=PBC,
    )


def setup_table(style, *, seed_reverse, monkeypatch):
    cfg = configuration(style)
    worker = Worker()
    authority = full_constraints(I)
    service = PrefactorService(
        cfg,
        worker,
        create_rate_constant(cfg.rateconstant),
        species_masses=(TYPES, MASSES),
        global_constraints=authority,
        method="protocol",
    )
    table = ReferenceEventTable(cfg, prefactor_service=service)
    seed = (F, S, I) if seed_reverse else (I, S, F)
    existing = row(
        *seed,
        initial_id="B" if seed_reverse else "A",
        final_id="A" if seed_reverse else "B",
        barrier=0.7 if seed_reverse else 0.5,
    )
    existing["idx_ref"] = 7
    existing["idx_backward"] = 7
    table.table = existing.to_frame().T
    table.table.index = [503]  # Logical ID 7 is neither position 0 nor label 503.
    seed_request = service.build_request(
        event_key=("explicit-existing-protocol-row", 7),
        min1_positions=seed[0].copy(),
        saddle_positions=seed[1].copy(),
        min2_positions=seed[2].copy(),
        types=TYPES,
        cell=CELL.copy(),
        pbc=PBC,
        center_index=0,
        constraints=full_constraints(seed[0]),
    )
    produced = protocol_result(seed_request)
    table._patch_row(
        7, produced.forward, calculation=produced.calculation("forward"), fresh=True
    )
    calculation = table.prefactor_archive.calculation_for(7, table.table.iloc[0])
    assert calculation is not None and calculation.provenance.source.is_complete
    assert calculation.provenance.source == RequestSnapshot.capture(seed_request)

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
        assert tuple(types) == TYPES and index_move == 0
        assert np.array_equal(cell, CELL)
        assert tuple(pbc) == PBC  # the admitted axes reach the series builder
        return (
            row(min1_positions, saddle_positions, min2_positions, barrier=dE_forward),
            row(
                min2_positions,
                saddle_positions,
                min1_positions,
                initial_id="B",
                final_id="A",
                barrier=dE_backward,
            ),
        )

    monkeypatch.setattr(table, "_build_event_series", build_series)
    return cfg, worker, table, calculation


def event(first, last):
    geometry = (first.copy(), S.copy(), last.copy())
    return EventSearchOutput(
        central_atom_index=0,
        min1_positions=geometry[0].copy(),
        saddle_positions=geometry[1].copy(),
        min2_positions=geometry[2].copy(),
        dE_forward=0.5,
        dE_backward=0.7,
        move_atom_index=0,
        cell=CELL.copy(),
        types=list(TYPES),
        constraints=full_constraints(first),
        prefactor_geometry=geometry,
    )


def assert_unchanged(ev, original, table, old_calculation):
    for actual, expected in zip(
        (ev.min1_positions, ev.saddle_positions, ev.min2_positions),
        original,
        strict=True,
    ):
        assert np.array_equal(actual, expected)
    for actual, expected in zip(ev.prefactor_geometry, original, strict=True):
        assert np.array_equal(actual, expected)
    current = table.prefactor_archive.calculations[old_calculation.calculation_id]
    assert current == old_calculation
    assert ev.constraints.atom_ids == IDS and ev.constraints.fixed_ids == ()


def test_shared_saddle_is_not_a_whole_event_proof():
    # Four distinct labels fix the permutation. Si at zero fixes translation.
    # The three independent remaining saddle vectors then force R=identity,
    # even if improper orthogonal maps are allowed. Thus F cannot map to G.
    assert len(set(TYPES)) == 4
    assert np.linalg.det(S[1:] - S[0]) == pytest.approx(3.0, abs=1e-12, rel=0)
    assert np.array_equal(I, I.copy()) and np.array_equal(S, S.copy())
    assert np.max(np.linalg.norm(F - G, axis=1)) == 1.0
    assert np.max(np.linalg.norm(I - G, axis=1)) == 1.5
    cfg = configuration("htst")
    assert cfg.psr.matching_score_thr < 0.5
    # The negative is separated even at the configured, nonzero threshold.
    # F1 = I1 + .25*I2 - .25*I0; I1 = F1 - .25*F2 + .25*F0.
    # A common affine map preserving the shared initial rows within tau can
    # move either combination by at most 1.5*tau. The target gaps are 1/1.5.
    assert np.array_equal(F[1], I[1] + 0.25 * I[2] - 0.25 * I[0])
    assert np.array_equal(I[1], F[1] - 0.25 * F[2] + 0.25 * F[0])
    tau = cfg.psr.matching_score_thr
    assert 1.0 - 1.5 * tau > tau
    assert 1.5 - 1.5 * tau > tau
    match = simple_ira(
        4, list(TYPES), S.copy(), 4, list(TYPES), S.copy(), cfg.ira.kmax_factor
    )
    assert check_match(match, cfg.psr.matching_score_thr).is_ok()


@pytest.mark.parametrize("style", ["htst", "rpa"])
@pytest.mark.parametrize(
    "seed_reverse", [False, True], ids=["known-forward", "known-reverse"]
)
@pytest.mark.parametrize(
    "different_endpoint", [False, True], ids=["exact-whole-event", "different-endpoint"]
)
def test_early_admission_needs_whole_event(
    style, seed_reverse, different_endpoint, monkeypatch
):
    cfg, worker, table, old_calculation = setup_table(
        style,
        seed_reverse=seed_reverse,
        monkeypatch=monkeypatch,
    )
    if seed_reverse:
        first, last = (G if different_endpoint else I), F
    else:
        first, last = I, (G if different_endpoint else F)
    ev = event(first, last)
    original = (first.copy(), S.copy(), last.copy())
    probe = row(
        last if seed_reverse else first,
        S,
        first if seed_reverse else last,
        initial_id="B" if seed_reverse else "A",
        final_id="A" if seed_reverse else "B",
        barrier=0.7 if seed_reverse else 0.5,
    )
    # Diagnostic only: the geometric query may remain a candidate generator.
    # Its return value must not by itself authorize destructive admission.
    print("saddle_only_candidate_id", table.find_matching_event(probe))
    results = table.add_events([ev], pbc=PBC)
    assert len(results) == 1
    if different_endpoint:
        assert results[0].is_ok(), (
            "A saddle-only forward match discarded a distinct full endpoint channel"
        )
        assert len(table.table) == 3, (
            "A saddle-only reverse match discarded the new backward channel"
        )
        added = table.table[table.table.idx_ref.astype(int) != 7]
        assert len(added) == 2
        new_ids = set(added.idx_ref.astype(int))
        assert set(added.idx_backward.astype(int)) == new_ids
        assert all(int(r.idx_ref) != int(r.idx_backward) for _, r in added.iterrows())
        assert len(worker.requests) == 1
        for _, r in added.iterrows():
            calc = table.prefactor_archive.calculation_for(int(r.idx_ref), r)
            assert calc is not None and calc.provenance.source.is_complete
            assert float(r.nu0) == 5e12 and r.nu0_status == "ok"
    elif seed_reverse:
        assert results[0].is_ok()
        assert len(table.table) == 2
        added = table.table[table.table.idx_ref.astype(int) != 7]
        assert len(added) == 1 and int(added.iloc[0].idx_backward) == 7
        assert len(worker.requests) == 1
    else:
        assert not results[0].is_ok()
        assert results[0].err_value().type == ErrorType.EVENT_NOT_NEW
        assert len(table.table) == 1
        assert worker.requests == []
    assert_unchanged(ev, original, table, old_calculation)
