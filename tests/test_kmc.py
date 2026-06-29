"""Tests for KMC.run() engine-pool teardown (recycle pool-hang follow-up)."""

from unittest.mock import Mock

import pytest

from pykmc.kmc import KMC


def test_run_closes_the_engine_pool_when_the_body_raises():
    """If the simulation body raises, run() must still tear down the engine pool.

    Otherwise the engine ranks strand busy-spinning in run_engine_loop waiting for a
    command that never comes -- the production face of the recycle pool-hang. The
    original error must still propagate.
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
