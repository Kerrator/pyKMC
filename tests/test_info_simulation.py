"""``info_refinements`` counts every refinement error type it can receive.

``Refinement`` attaches ``variables["n_ref_event"]`` to every ``Err`` it
collects, so each failure bucket can name the references it hit. A bucket
that is incremented under a key the dictionary does not carry is a latent
KeyError on the first such failure; an error type without a case vanishes
from the step statistics.
"""

import numpy as np

from pykmc.info_simulation import info_refinements
from pykmc.result import Err, ErrorInfo, ErrorType, EventRefinementOutput, Ok


def _err(kind: ErrorType, ref: int) -> Err:
    return Err(
        ErrorInfo(type=kind, message=kind.name.lower(), variables={"n_ref_event": ref})
    )


def _ok() -> Ok:
    return Ok(
        EventRefinementOutput(
            central_atom_index=0,
            saddle_positions=np.zeros((1, 3)),
            E_saddle=0.1,
        )
    )


def test_invalid_minima_and_invalid_event_data_are_counted_with_their_references() -> (
    None
):
    """Both buckets exist, are counted and carry the reference ids."""
    results = [
        _ok(),
        _err(ErrorType.REFINEMENT_INVALID_MINIMA, 3),
        _err(ErrorType.RECONSTRUCTION_INVALID_EVENT_DATA, 47),
        _err(ErrorType.RECONSTRUCTION_INVALID_EVENT_DATA, 48),
    ]
    info = info_refinements(results)
    assert info.n_attempts == 4 and info.n_sucesses == 1
    assert info.n_fails["invalid_minima"] == {"n": 1, "ref_event": [3]}
    assert info.n_fails["invalid_event_data"] == {"n": 2, "ref_event": [47, 48]}
    counted = sum(bucket["n"] for bucket in info.n_fails.values())
    assert counted == 3, "every failure is counted under exactly one bucket"
    assert "invalid_min" not in info.n_fails


def test_every_refinement_error_type_lands_in_a_bucket() -> None:
    """No refinement error type is silently excluded from the statistics."""
    kinds = [
        ErrorType.PSR_NO_MATCH_FOUND,
        ErrorType.REFINEMENT_INVALID_ENERGY_BARRIER,
        ErrorType.REFINEMENT_INVALID_MINIMA,
        ErrorType.EVENT_NOT_FOUND,
        ErrorType.RECONSTRUCTION_INVALID_EVENT_DATA,
    ]
    info = info_refinements([_err(kind, 10 + i) for i, kind in enumerate(kinds)])
    assert info.n_attempts == len(kinds) and info.n_sucesses == 0
    assert sum(bucket["n"] for bucket in info.n_fails.values()) == len(kinds)
    seen = [ref for bucket in info.n_fails.values() for ref in bucket["ref_event"]]
    assert sorted(seen) == [10 + i for i in range(len(kinds))]
