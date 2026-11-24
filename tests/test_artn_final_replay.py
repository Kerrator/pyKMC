"""Full-search step and endpoint replay against the eight pARTn recordings."""

import os

import numpy as np
import pytest

from .artn_replay import RECORDINGS_ENV, check_search, holds_search, replay
from .test_artn_replay import recorded, synthetic_recording


@pytest.mark.parametrize("search", range(8))
def test_complete_search_matches_partn(search: int) -> None:
    """Check every callback, the end step and reason, and the raw endpoints."""
    rec = recorded(search)
    if search != 0 and not holds_search(rec) and not os.environ.get(RECORDINGS_ENV):
        pytest.skip(f"set {RECORDINGS_ENV} to the full recordings")
    check = check_search(rec)
    assert not (problems := check.problems()), problems


def test_complete_search_refuses_a_truncated_recording() -> None:
    """Even a search missing only its last numbered evaluation is incomplete."""
    rec = recorded(0)
    last = np.flatnonzero(rec["partn_step"] >= 0)[-1]
    rec["partn_step"][last] = -1
    with pytest.raises(ValueError, match="complete search"):
        check_search(rec)


@pytest.mark.parametrize("keep_going", (False, True))
def test_changed_push_block_fails_even_on_the_same_positions(keep_going: bool) -> None:
    """Swapping SMTH for EIGN is a structural mismatch under both rulings."""
    rec = synthetic_recording(nevalf_max=500)
    start = rec["block_out"].index("SMTH")
    rec["block_out"][start] = "EIGN"
    step = rec["partn_step"][start] + 1
    report = replay(rec, last_step=step, keep_going=keep_going)
    assert report.end == "mismatch"
    assert report.first_mismatch.step == step
    assert "block EIGN against core move smooth" in report.first_mismatch.problem


def test_endpoint_corruption_fails_the_final_check() -> None:
    """Changing only a saved minimum's energy or geometry, including NaN, fails."""
    rec = recorded(0)
    check = check_search(rec)
    assert not check.problems()
    energy, positions = rec["partn"]["etot_min1"], rec["tau_min2"]
    for value in (energy + 2e-9, np.nan):
        rec["partn"]["etot_min1"] = value
        assert any("min1:" in p for p in check.problems())
    rec["partn"]["etot_min1"] = energy
    for delta in (2e-10, np.nan):
        rec["tau_min2"] = positions.copy()
        rec["tau_min2"][0, 0] += delta
        assert any("min2:" in p for p in check.problems())
