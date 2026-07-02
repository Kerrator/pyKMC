"""Pure-python tests for the MPI-safety guards on the basin reconstruction path.

Two guards are exercised without launching MPI:

* ``lammps_operations._nonfinite_reconstruct_payload`` -- the pre-collective finite
  check that converts a poisoned (NaN/inf) candidate geometry into the standard
  ``ok: False`` payload *on rank 0, before any collective op is issued*, so every
  engine rank short-circuits together instead of one rank hitting a per-rank LAMMPS
  ``error->one`` and hanging the pool (finding #4).
* ``run.main`` -- the MPI-wide failure boundary that prints the traceback and calls
  ``MPI.COMM_WORLD.Abort(1)`` when an exception escapes on any rank under a
  multi-rank world, while leaving single-rank behaviour (a clean re-raise) and a
  normal ``SystemExit`` shutdown untouched (finding #11).
"""

from typing import Any
from unittest.mock import patch

import numpy as np
import pytest

from pykmc.enginemanager.lmpi import lammps_operations as ops


class TestNonfiniteReconstructPayload:
    """The rank-0 pre-collective finite check for basin reconstruction geometry."""

    def test_finite_positions_return_none(self) -> None:
        """Finite geometry produces no error payload."""
        pos = np.arange(12, dtype=float).reshape(4, 3)
        assert ops._nonfinite_reconstruct_payload(pos, "min1") is None

    @pytest.mark.parametrize("bad", [np.nan, np.inf, -np.inf])
    def test_nonfinite_returns_error_payload(self, bad: float) -> None:
        """NaN/inf coordinates produce a RECONSTRUCTION_MINIMIZE_FAILED payload."""
        pos = np.zeros((4, 3), dtype=float)
        pos[2, 1] = bad
        payload = ops._nonfinite_reconstruct_payload(pos, "min2")
        assert payload is not None
        assert payload["ok"] is False
        assert payload["error_type"] == "RECONSTRUCTION_MINIMIZE_FAILED"

    def test_payload_names_the_tripping_step(self) -> None:
        """The message records which reconstruction step tripped the guard."""
        pos = np.zeros((2, 3), dtype=float)
        pos[0, 0] = np.nan
        for step in ("min1", "min2", "min_global"):
            payload = ops._nonfinite_reconstruct_payload(pos, step)
            assert payload is not None
            assert step in payload["message"]

    def test_payload_reports_the_number_of_bad_values(self) -> None:
        """The message counts every non-finite coordinate found."""
        pos = np.zeros((3, 3), dtype=float)
        pos[0, 0] = np.nan
        pos[1, 2] = np.inf
        payload = ops._nonfinite_reconstruct_payload(pos, "min1")
        assert payload is not None
        assert "2 non-finite" in payload["message"]

    def test_payload_shape_matches_the_other_reconstruct_failures(self) -> None:
        """The keys match the ``ok: False`` shape callers already handle."""
        pos = np.array([[np.nan, 0.0, 0.0]])
        payload = ops._nonfinite_reconstruct_payload(pos, "min1")
        assert payload is not None
        assert set(payload) == {"ok", "error_type", "message"}

    def test_accepts_list_input(self) -> None:
        """A plain nested list is coerced just like an ndarray."""
        assert ops._nonfinite_reconstruct_payload([[0.0, 1.0, 2.0]], "min1") is None
        payload = ops._nonfinite_reconstruct_payload([[float("nan"), 0.0, 0.0]], "min1")
        assert payload is not None and payload["ok"] is False


class _FakeComm:
    """Minimal COMM_WORLD stand-in recording Abort calls."""

    def __init__(self, size: int) -> None:
        self._size = size
        self.abort_codes: list[int] = []

    def Get_size(self) -> int:  # noqa: N802 - mirror the mpi4py API name
        """Return the configured fake world size."""
        return self._size

    def Abort(self, code: int = 0) -> None:  # noqa: N802 - mirror the mpi4py API name
        """Record the abort code instead of tearing the process down."""
        self.abort_codes.append(code)


class TestMainAbortBoundary:
    """``run.main`` converts an uncaught exception into ``MPI.Abort`` only under MPI."""

    def _run_main(
        self, world_size: int, launch_side_effect: "BaseException | None"
    ) -> _FakeComm:
        """Invoke ``run.main`` with a fake COMM_WORLD and a stubbed ``_launch``.

        Parameters
        ----------
        world_size : int
            Value returned by the fake ``COMM_WORLD.Get_size``.
        launch_side_effect : BaseException or None
            Raised by the stubbed ``_launch`` (``None`` = clean return).

        Returns
        -------
        _FakeComm
            The fake communicator, so the test can inspect ``abort_codes``.

        """
        from pykmc import run

        fake_comm = _FakeComm(world_size)
        args = type("Args", (), {"input": "dummy.in"})()

        def _launch(_args: Any) -> None:
            if launch_side_effect is not None:
                raise launch_side_effect

        with patch.object(run.MPI, "COMM_WORLD", fake_comm), \
                patch.object(run, "_launch", side_effect=_launch), \
                patch("argparse.ArgumentParser.parse_args", return_value=args):
            run.main()
        return fake_comm

    def test_single_rank_reraises_and_never_aborts(self) -> None:
        """A single-rank failure re-raises unchanged (no Abort)."""
        with pytest.raises(RuntimeError, match="boom"):
            self._run_main(1, RuntimeError("boom"))

    def test_single_rank_clean_return_no_abort(self) -> None:
        """A single-rank clean finish never calls Abort."""
        fake_comm = self._run_main(1, None)
        assert fake_comm.abort_codes == []

    def test_multi_rank_exception_triggers_abort(self) -> None:
        """A multi-rank uncaught exception calls ``Abort(1)``."""
        fake_comm = self._run_main(8, RuntimeError("boom"))
        assert fake_comm.abort_codes == [1]

    def test_multi_rank_clean_return_no_abort(self) -> None:
        """A multi-rank clean finish never calls Abort."""
        fake_comm = self._run_main(8, None)
        assert fake_comm.abort_codes == []

    def test_multi_rank_systemexit_is_not_aborted(self) -> None:
        """A normal shutdown exits via ``sys.exit`` (SystemExit) -- must not Abort."""
        with pytest.raises(SystemExit):
            self._run_main(8, SystemExit(0))
