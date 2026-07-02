"""Tests for event reconstruction error handling (recycle pool-hang follow-up)."""

from unittest.mock import Mock

import numpy as np

from pykmc.reconstruction import Reconstruction
from pykmc.result import ErrorType


def _config() -> Mock:
    config = Mock()
    config.psr.matching_score_thr = 0.1
    return config


_SADDLE = np.array([[0.0, 0.0, 0.0], [1.5, 0.0, 0.0]])
_MIN1 = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]])
_MIN2 = np.array([[0.0, 0.0, 0.0], [2.0, 0.0, 0.0]])
_CELL = np.diag([10.0, 10.0, 10.0])


def test_reconstruct_returns_err_when_engine_minimize_raises():
    """A LAMMPS error during the reconstruction minimize must become an Err so the
    recycler drops that reconstruction -- it must NOT propagate and crash the whole
    run (the recycle pool-hang trigger).
    """
    manager = Mock()
    manager.global_minimize_with_results.side_effect = RuntimeError(
        "Remote minimize_with_results failed on engine rank 1 (RuntimeError): Lost atoms"
    )
    recon = Reconstruction(_config(), manager, types=["Ni", "Ni"])

    result = recon.reconstruct(
        _MIN1.copy(), _MIN2.copy(), _SADDLE.copy(), _CELL, delr_thr=0.1
    )

    assert not result.is_ok()
    assert result.err_value().type == ErrorType.RECONSTRUCTION_MINIMIZE_FAILED


def test_reconstruct_ok_path_still_works():
    """When both minimizes succeed and match the supposed minima, reconstruct
    returns Ok (guard against the error handling breaking the happy path).
    """
    manager = Mock()
    manager.global_minimize_with_results.side_effect = [
        (_MIN1.copy(), 0.0),
        (_MIN2.copy(), -5.0),
    ]
    recon = Reconstruction(_config(), manager, types=["Ni", "Ni"])

    result = recon.reconstruct(
        _MIN1.copy(), _MIN2.copy(), _SADDLE.copy(), _CELL, delr_thr=0.1
    )

    assert result.is_ok()
    assert result.ok_value().min2_etot == -5.0
