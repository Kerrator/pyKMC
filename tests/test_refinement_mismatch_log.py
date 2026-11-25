"""Tests that each refinement mismatch is logged with its reference event and size.

`Refinement.refine_single` is replaced by a stub returning prepared results,
so no PSR, LAMMPS or pARTn runs.
"""

from concurrent.futures import Future
from types import SimpleNamespace

import numpy as np
import pandas as pd

from pykmc.refinement import Refinement
from pykmc.result import Err, ErrorInfo, ErrorType, EventRefinementOutput, Ok

N_ATOMS = 4
TOTAL_ENERGY = -100.0


class StubLoggers:
    def __init__(self) -> None:
        self.lines = []

    def info(self, name: str, message: str) -> None:
        self.lines.append((name, message))

    def progress_bar(self, name: str, done: int, total: int) -> None:
        pass


def _refinement(outcomes: dict[int, object]) -> tuple[Refinement, StubLoggers]:
    """Build a Refinement whose refinements return ``outcomes[reference event]``."""
    config = SimpleNamespace(
        control=SimpleNamespace(active_volume=False),
        eventsearch=SimpleNamespace(refined_energy_thr=0.05, refine_style="partn"),
        partn=SimpleNamespace(r_delr_sad_thr=0.4),
    )
    loggers = StubLoggers()
    atomic_environment = SimpleNamespace(get_atoms_with_id=lambda event_id: [2])
    refinement = Refinement(config, loggers, None, None, atomic_environment, None)
    refinement.get_total_refinements_todo = lambda df: (len(df), 1.0)
    refinement.get_energy_thr_refine = lambda df, k: 10.0

    def refine_single(at_idx, dfevent, total_energy, future_context, e_thr):
        future = Future()
        future.set_result(outcomes[int(dfevent["idx_ref"])])
        future_context[future] = {
            "min2_positions": np.zeros((N_ATOMS, 3)),
            "num_reference_event": dfevent["idx_ref"],
            "reference_energy_barrier": 0.60,
            "neighbors": np.arange(N_ATOMS),
        }
        return future

    refinement.refine_single = refine_single
    return refinement, loggers


def _refined(barrier: float) -> Ok:
    return Ok(
        EventRefinementOutput(
            central_atom_index=2,
            saddle_positions=np.zeros((N_ATOMS, 3)),
            E_saddle=TOTAL_ENERGY + barrier,
            refined="T",
        )
    )


def _events(*idx_ref: int) -> pd.DataFrame:
    return pd.DataFrame({"event_id": ["e"] * len(idx_ref), "idx_ref": list(idx_ref)})


def test_energy_mismatch_is_logged_with_reference_event_and_size() -> None:
    refinement, loggers = _refinement({0: _refined(0.62), 7: _refined(0.70)})

    refinement.execute(_events(0, 7), TOTAL_ENERGY)

    assert refinement.results[0].is_ok()
    assert refinement.results[1].err_value().type == (
        ErrorType.REFINEMENT_INVALID_ENERGY_BARRIER
    )
    mismatches = [m for _, m in loggers.lines if "mismatch" in m]
    assert len(mismatches) == 1
    assert "reference event 7, atom 2" in mismatches[0]
    assert "barrier 0.7000 eV against 0.6000 eV" in mismatches[0]
    assert "|dE| = 0.1000 eV > 0.05 eV" in mismatches[0]


def test_saddle_distance_mismatch_is_logged_with_reference_event_and_size() -> None:
    moved = Err(
        ErrorInfo(
            type=ErrorType.EVENT_NOT_FOUND,
            message="no event found",
            details=(0, ""),
            variables={"central_atom": 2, "delr_sad": 0.55},
        )
    )
    no_saddle = Err(
        ErrorInfo(
            type=ErrorType.EVENT_NOT_FOUND,
            message="no event found",
            details=(4, "pARTn failed"),
            variables={"central_atom": 2, "delr_sad": None},
        )
    )
    refinement, loggers = _refinement({3: moved, 5: no_saddle})

    refinement.execute(_events(3, 5), TOTAL_ENERGY)

    mismatches = [m for _, m in loggers.lines if "mismatch" in m]
    assert mismatches == [
        "\t :=> Refinement mismatch: reference event 3, atom 2: "
        "saddle moved 0.5500 A >= r_delr_sad_thr 0.4 A"
    ]
    assert refinement.results[0].err_value().variables["n_ref_event"] == 3
    assert refinement.results[1].err_value().variables["n_ref_event"] == 5
