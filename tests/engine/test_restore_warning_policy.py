"""An application warning policy must not replace an initiating engine error.

Both public wrappers preserve the original error and permit a later retry.
"""

from types import SimpleNamespace
import warnings

import pytest

from tests.engine.test_restore_retry_physics import (
    CELL,
    POSITIONS,
    Harness,
    InitiatingFailure,
)


@pytest.mark.parametrize("operation", ["partn_search", "partn_refine"])
def test_warning_promoted_to_error_preserves_initiating_failure(monkeypatch, operation):
    h = Harness(monkeypatch)
    initiating = InitiatingFailure(f"original {operation} failure")

    def fail_operation(*args, **kwargs):
        h.mark_pending(closed=True)
        raise initiating

    monkeypatch.setattr(h.engine, "_" + operation + "_impl", fail_operation)
    h.failures["start"] = RuntimeError("secondary restart failure")
    config = SimpleNamespace(
        control=SimpleNamespace(active_volume=True), frozen_atoms=None
    )
    with warnings.catch_warnings():
        warnings.simplefilter("error", RuntimeWarning)
        with pytest.raises(InitiatingFailure) as raised:
            getattr(h.engine, operation)(
                config, 0, positions=POSITIONS, cell=CELL, types=h.original.types
            )
    assert raised.value is initiating
    if hasattr(BaseException, "add_note"):
        assert "secondary restart failure" in initiating.__notes__[0]
    h.assert_pending()
    assert h.engine.ensure_full_system(POSITIONS) is True
    h.assert_restored()
