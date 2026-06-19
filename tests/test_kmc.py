from unittest.mock import Mock

import numpy as np
import pytest

from pykmc import System
from pykmc.kmc import KMC
from pykmc.result import Ok, ReconstructionOutput


def _toy_system(offset: float) -> System:
    return System(
        positions=np.array([[offset, 0.0, 0.0], [offset + 1.0, 0.0, 0.0]], dtype=float),
        types=np.array(["Ni", "Ni"]),
        cell=np.diag([20.0, 20.0, 20.0]),
        pbc=np.array([True, True, True]),
        index=np.array([0, 1]),
    )


def test_apply_original_migration_event_restores_positions_and_total_energy():
    kmc = KMC(config=Mock())
    kmc.system = _toy_system(0.0)
    reconstructed_system = _toy_system(3.0)

    result_reconstruction = Ok(
        ReconstructionOutput(
            min1_positions=_toy_system(1.0).positions,
            saddle_positions=_toy_system(2.0).positions,
            min2_positions=reconstructed_system.positions,
            min2_etot=-7.5,
        )
    )

    kmc._apply_original_migration_event(result_reconstruction)

    assert np.allclose(kmc.system.positions, reconstructed_system.positions)
    assert kmc.total_energy == -7.5


def test_run_closes_the_engine_pool_when_the_body_raises():
    """If the simulation body raises, run() must still tear down the engine pool.

    Otherwise the engine ranks strand busy-spinning in run_engine_loop waiting for a
    command that never comes -- the production face of the recycle pool-hang (see
    HANDOFF_recycle_pool_hang.md). The original error must still propagate.
    """
    kmc = KMC(config=Mock())
    kmc.manager = Mock()
    kmc._run_impl = Mock(side_effect=RuntimeError("engine blew up"))

    with pytest.raises(RuntimeError, match="engine blew up"):
        kmc.run()

    kmc.manager.close_all.assert_called_once()


def test_run_does_not_double_close_on_a_normal_exit():
    """A normal finish ends via ``_close()`` -> ``sys.exit()`` (SystemExit). The
    teardown guard must NOT catch that, so the pool is closed exactly once (by
    ``_close()``), never twice (a double close would block on the second teardown).
    """
    kmc = KMC(config=Mock())
    kmc.manager = Mock()
    kmc._run_impl = Mock(side_effect=SystemExit(0))

    with pytest.raises(SystemExit):
        kmc.run()

    kmc.manager.close_all.assert_not_called()
