"""Tests for event reconstruction error handling (recycle pool-hang follow-up)."""
from unittest.mock import Mock

import numpy as np

from pykmc.reconstruction import Reconstruction
from pykmc.result import ErrorType


def _config() -> Mock:
    config = Mock()
    config.reconstruction.push_fraction = 0.15
    config.reconstruction.n_movers = 3
    config.reconstruction.containment_margin = 1.0
    config.reconstruction.shell_tolerance = 1.0
    config.atomicenvironment.rcut = 6.5
    config.psr.matching_score_thr = 0.1
    return config


_SADDLE = np.array([[0.0, 0.0, 0.0], [1.5, 0.0, 0.0]])
_MIN1 = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]])
_MIN2 = np.array([[0.0, 0.0, 0.0], [2.0, 0.0, 0.0]])
_CELL = np.diag([10.0, 10.0, 10.0])


def test_reconstruct_returns_err_when_engine_minimize_raises():
    """A LAMMPS error during the reconstruction minimize must become an Err so the
    recycler drops that reconstruction -- it must NOT propagate and crash the whole
    run (the recycle pool-hang trigger; see HANDOFF_recycle_pool_hang.md)."""
    manager = Mock()
    manager.global_minimize_with_results.side_effect = RuntimeError(
        "Remote minimize_with_results failed on engine rank 1 (RuntimeError): Lost atoms"
    )
    recon = Reconstruction(_config(), manager, types=["Ni", "Ni"])

    result = recon.reconstruct(
        _MIN1.copy(), _MIN2.copy(), _SADDLE.copy(), _CELL    )

    assert not result.is_ok()
    assert result.err_value().type == ErrorType.RECONSTRUCTION_MINIMIZE_FAILED


def test_reconstruct_ok_path_still_works():
    """When both minimizes succeed and match the supposed minima, reconstruct
    returns Ok (guard against the error handling breaking the happy path)."""
    manager = Mock()
    manager.global_minimize_with_results.side_effect = [
        (_MIN1.copy(), 0.0),
        (_MIN2.copy(), -5.0),
    ]
    recon = Reconstruction(_config(), manager, types=["Ni", "Ni"])

    result = recon.reconstruct(
        _MIN1.copy(), _MIN2.copy(), _SADDLE.copy(), _CELL    )

    assert result.is_ok()
    assert result.ok_value().min2_etot == -5.0


# Three-atom event: atom 1 is the mover (min1 [1,0,0] -> min2 [2,0,0]); atoms 0
# and 2 are static. atom 2 sits far out so a small reconstruction error on it must
# NOT veto the match once the acceptance focuses on the movers.
_MIN1_3 = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [3.0, 0.0, 0.0]])
_MIN2_3 = np.array([[0.0, 0.0, 0.0], [2.0, 0.0, 0.0], [3.0, 0.0, 0.0]])
_SADDLE_3 = np.array([[0.0, 0.0, 0.0], [1.5, 0.0, 0.0], [3.0, 0.0, 0.0]])


def test_peripheral_atom_offset_does_not_veto_when_movers_match():
    """A peripheral (non-event) atom reconstructed past the threshold must NOT
    reject an otherwise-correct reconstruction; only the top-n event movers gate
    acceptance. This is the delr=0.416 near-miss the layer-2 check fixes."""
    # atom 2 lands 0.5 A off (>> matching_score_thr=0.1), the mover atom 1 is exact.
    min1_ret = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [3.5, 0.0, 0.0]])
    min2_ret = np.array([[0.0, 0.0, 0.0], [2.0, 0.0, 0.0], [3.5, 0.0, 0.0]])
    manager = Mock()
    manager.global_minimize_with_results.side_effect = [
        (min1_ret, 0.0),
        (min2_ret, -5.0),
    ]
    recon = Reconstruction(_config(), manager, types=["Ni", "Ni", "Ni"])

    result = recon.reconstruct(
        _MIN1_3.copy(), _MIN2_3.copy(), _SADDLE_3.copy(), _CELL    )

    assert result.is_ok()


def test_peripheral_gross_misland_rejected_by_shell_bound():
    """A peripheral (non-mover) atom that relaxes into a DISTINCT site (a large
    displacement, > shell_tolerance) must reject the reconstruction even though the
    event movers match -- the whole-shell loose bound catches a wrong overall state
    that the movers-only check would have accepted (review finding #1)."""
    # mover atom 1 exact; peripheral atom 2 lands 1.5 A off (>> shell_tolerance=1.0).
    min1_ret = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [4.5, 0.0, 0.0]])
    manager = Mock()
    manager.global_minimize_with_results.side_effect = [(min1_ret, 0.0)]
    recon = Reconstruction(_config(), manager, types=["Ni", "Ni", "Ni"])

    result = recon.reconstruct(
        _MIN1_3.copy(), _MIN2_3.copy(), _SADDLE_3.copy(), _CELL    )

    assert not result.is_ok()
    assert result.err_value().type == ErrorType.RECONSTRUCTION_INVALID_MIN1
    assert result.err_value().variables["delr_shell1"] > 1.0


def test_mover_offset_rejects_reconstruction():
    """If a top event mover is reconstructed past the threshold, reject (INVALID_MIN1)."""
    # mover atom 1 lands 0.5 A off in min1.
    min1_ret = np.array([[0.0, 0.0, 0.0], [1.5, 0.0, 0.0], [3.0, 0.0, 0.0]])
    manager = Mock()
    manager.global_minimize_with_results.side_effect = [(min1_ret, 0.0)]
    recon = Reconstruction(_config(), manager, types=["Ni", "Ni", "Ni"])

    result = recon.reconstruct(
        _MIN1_3.copy(), _MIN2_3.copy(), _SADDLE_3.copy(), _CELL    )

    assert not result.is_ok()
    assert result.err_value().type == ErrorType.RECONSTRUCTION_INVALID_MIN1


def test_event_not_contained_in_rcut_rejects_before_minimize():
    """If a top mover sits in the outer rcut shell, the event is not contained and
    reconstruction is rejected before the (expensive) minimize ever runs."""
    config = _config()
    config.atomicenvironment.rcut = 3.0  # limit = rcut - margin = 2.0
    # central atom 0 at origin; mover atom 1 at radius 2.5 (> 2.0) and it moves.
    min1 = np.array([[0.0, 0.0, 0.0], [2.5, 0.0, 0.0], [1.0, 0.0, 0.0]])
    min2 = np.array([[0.0, 0.0, 0.0], [3.5, 0.0, 0.0], [1.0, 0.0, 0.0]])
    saddle = np.array([[0.0, 0.0, 0.0], [3.0, 0.0, 0.0], [1.0, 0.0, 0.0]])
    manager = Mock()
    recon = Reconstruction(config, manager, types=["Ni", "Ni", "Ni"])

    result = recon.reconstruct(
        min1, min2, saddle, _CELL,
        neighbors=np.array([0, 1, 2]), central_atom=0,
    )

    assert not result.is_ok()
    assert result.err_value().type == ErrorType.RECONSTRUCTION_EVENT_NOT_CONTAINED
    manager.global_minimize_with_results.assert_not_called()
