"""Catalogue identity probes; full event geometry has one common atom order.

The worker supplies equal accepted estimates to isolate identity from spectra.
IRA and catalogue admission/resolution are real. These are geometric/protocol
counterexamples, not claims that the fixture is a native potential's MEP.
"""

from concurrent.futures import Future
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from pykmc.config import Config, RateConstantConfig
from pykmc.event_table import ReferenceEventTable
from pykmc.htst.free_region import common_free_indices
from pykmc.htst.provenance import CalculationProvenance
from pykmc.htst.result import DirectionalPrefactor, EventPrefactors, PrefactorRejection
from pykmc.point_set_registration import check_match, simple_ira
from pykmc.rate_constant import create_rate_constant
from pykmc.rate_constant.prefactors import PrefactorService
from pykmc.result import EventSearchOutput

import pykmc

TYPES = ["Si", "Ge", "C"]
INITIAL = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 2.0, 0.0]])
ROTATION = np.diag([-1.0, -1.0, 1.0])
FINAL = INITIAL @ ROTATION
ASYMMETRIC_SADDLE = np.array([[0.0, 0.0, 0.0], [0.4, 0.9, 0.3], [-0.5, 0.3, 1.6]])
SYMMETRIC_TYPES = ["Si", "Ge", "Ge", "C", "C"]
SYMMETRIC_INITIAL = np.array(
    [
        [0.0, 0.0, 0.0],
        [1.2, 0.1, 0.1],
        [-0.8, 0.2, 0.1],
        [0.2, 2.0, 0.3],
        [0.1, -1.7, 0.2],
    ]
)
SYMMETRIC_SADDLE = np.array(
    [
        [0.0, 0.0, 0.0],
        [1.0, 0.0, 0.1],
        [-1.0, 0.0, 0.1],
        [0.0, 2.0, 0.3],
        [0.0, -2.0, 0.3],
    ]
)
SYMMETRIC_PERM = np.array([0, 2, 1, 4, 3])
SYMMETRIC_FINAL = (SYMMETRIC_INITIAL @ ROTATION)[SYMMETRIC_PERM]


def configuration():
    root = Path(pykmc.__file__).resolve().parent.parent
    cfg = Config.from_ini_file(str(root / "tests/data/input.in"))
    return cfg.model_copy(
        update={
            "rateconstant": RateConstantConfig(style="htst", k0=1.0),
            "control": cfg.control.model_copy(update={"reference_table": None}),
            "atomicenvironment": cfg.atomicenvironment.model_copy(
                update={"atom_coloring_mode": "full"}
            ),
        }
    )


class EqualEstimateWorker:
    def __init__(self, reject_backward=False):
        self.reject_backward = reject_backward

    def submit(self, operation, *, request, compute_backward=True):
        assert operation == "compute_event_prefactors"
        request.validate()
        free = tuple(int(i) for i in common_free_indices(request))
        n = len(free)
        provenance = CalculationProvenance.capture(
            request, request, method="protocol", free_indices=free
        )
        assert provenance.source.is_complete and provenance.produced.is_complete
        forward = DirectionalPrefactor(5e12, "ok", None, None, n, 3 * n, 1)
        backward = (
            DirectionalPrefactor(
                None,
                "rejected",
                PrefactorRejection.UNSTABLE_MINIMUM,
                "independent rejection control",
                n,
                3 * n - 1,
                1,
            )
            if self.reject_backward
            else forward
        )
        future = Future()
        future.set_result(
            EventPrefactors(
                request.event_key,
                forward,
                backward,
                "protocol",
                n,
                request.settings,
                provenance=provenance,
            )
        )
        return future


def row(initial, saddle, final, types=TYPES):
    return pd.Series(
        {
            "idx_ref": -1,
            "event_id": "same-topology",
            "id_final": "same-topology",
            "id_saddle": "saddle",
            "idx_backward": -1,
            "initial_positions": initial,
            "saddle_positions": saddle,
            "final_positions": final,
            "types": types,
            "energy_barrier": 0.5,
            "k": 0.0,
            "move_atom_idx": 0,
            "sym_matrix": [np.eye(3)],
            "sym_perm": [np.arange(len(types))],
            "dra": 2.0,
            "k_prefactor": 1.0,
            "nu0": np.nan,
            "nu0_status": "pending",
            "nu0_reason": "",
        }
    )


def admitted(saddle, reject_backward=False, initial=INITIAL, final=FINAL, types=TYPES):
    cfg = configuration()
    svc = PrefactorService(
        cfg,
        EqualEstimateWorker(reject_backward),
        create_rate_constant(cfg.rateconstant),
        species_masses=(("Si", "Ge", "C"), (28.0855, 72.63, 12.011)),
        method="protocol",
    )
    table = ReferenceEventTable(cfg, prefactor_service=svc)
    outcome = table._admit_series(
        row(initial, saddle, final, types), row(final, saddle, initial, types)
    )
    assert outcome.is_ok()
    admission = outcome.ok_value()
    table.add(admission.frame, reverse_idx_ref=admission.reverse_idx_ref)
    frame = admission.frame
    forward = int(frame.iloc[0].idx_ref)
    backward = int(frame.iloc[1].idx_ref) if len(frame) > 1 else None
    event = EventSearchOutput(
        0, initial, saddle, final, 0.5, 0.5, 0, 10 * np.eye(3), types
    )
    table._resolve_prefactors([(forward, backward, admission, event)], [False] * 3)
    return table


def test_separate_endpoint_and_saddle_matches_are_not_one_event_mapping():
    cfg = configuration()
    for first, second in [(INITIAL, FINAL), (ASYMMETRIC_SADDLE, ASYMMETRIC_SADDLE)]:
        match = simple_ira(
            3, TYPES, first.copy(), 3, TYPES, second.copy(), cfg.ira.kmax_factor
        )
        assert match.is_ok()
        assert check_match(match, cfg.psr.matching_score_thr).is_ok()
    # Unique species fix the permutation; atom 0 fixes translation. Mapping
    # atom1/2 in the endpoints forces x -> -x and y -> -y. Hence any mapping
    # (proper or improper) changes the nonzero saddle x/y coordinates.
    assert np.allclose(INITIAL @ ROTATION, FINAL)
    assert (
        np.max(np.abs((ASYMMETRIC_SADDLE @ ROTATION - ASYMMETRIC_SADDLE)[:, :2])) > 1.0
    )


def test_equal_prefactors_and_separate_matches_do_not_collapse_whole_event():
    table = admitted(ASYMMETRIC_SADDLE)
    print(
        "whole_event_rows",
        table.table[["idx_ref", "idx_backward", "nu0", "nu0_status"]].to_dict(
            "records"
        ),
    )
    assert len(table.table) == 2, (
        "No single species-preserving map reverses endpoints while preserving the saddle; separate crop matches are insufficient"
    )
    assert set(
        zip(table.table.idx_ref.astype(int), table.table.idx_backward.astype(int))
    ) == {(0, 1), (1, 0)}
    assert table.table.nu0.astype(float).tolist() == pytest.approx([5e12, 5e12])


def test_common_mapping_and_equal_prefactors_can_collapse():
    assert np.array_equal(
        (SYMMETRIC_INITIAL @ ROTATION)[SYMMETRIC_PERM], SYMMETRIC_FINAL
    )
    assert np.array_equal(
        (SYMMETRIC_SADDLE @ ROTATION)[SYMMETRIC_PERM], SYMMETRIC_SADDLE
    )
    table = admitted(
        SYMMETRIC_SADDLE,
        initial=SYMMETRIC_INITIAL,
        final=SYMMETRIC_FINAL,
        types=SYMMETRIC_TYPES,
    )
    assert len(table.table) == 1
    assert int(table.table.iloc[0].idx_backward) == int(table.table.iloc[0].idx_ref)
    assert float(table.table.iloc[0].nu0) == pytest.approx(5e12)


def test_rejected_reverse_collapses_to_one_row_and_is_archived():
    # contracts 7f policy 6: a rejected backward never proves the collapse and
    # never becomes a second selectable row sharing the event_id.
    table = admitted(
        SYMMETRIC_SADDLE,
        reject_backward=True,
        initial=SYMMETRIC_INITIAL,
        final=SYMMETRIC_FINAL,
        types=SYMMETRIC_TYPES,
    )
    assert len(table.table) == 1
    row = table.table.iloc[0]
    assert int(row.idx_backward) == int(row.idx_ref)
    assert row.nu0_status == "ok" and float(row.k_prefactor) == pytest.approx(5.0)
    assert "self-reverse unproven: backward prefactor rejected" in row.nu0_reason
    history = table.prefactor_archive.history[1]
    assert any(
        entry["nu0_status"] == "rejected" and entry["k_prefactor"] == 1.0
        for entry in history
    )
    assert table.max_idx_ref() == 2
