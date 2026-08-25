import logging
import typing

from pykmc.basins import FPTASelector
import numpy as np
import pandas as pd

class TestSelector :

    def test_solver_dispatch(self, test_logger) :
        """`solver` selects bisection vs QSD; 'auto' switches on stiffness > 1e6."""
        # Stiff reduced generator: transient mixing 1e3, absorbing escape 1e-3.
        stiff = np.array([
            [1.0e3 + 1.0e-3, -1.0e3,          0.0],
            [-1.0e3,          1.0e3 + 1.0e-3, 0.0],
            [-1.0e-3,        -1.0e-3,         0.0],
        ])
        # Non-stiff: comparable transient and absorbing rates.
        mild = np.array([
            [0.3, -0.1, 0.0],
            [-0.1, 0.3, 0.0],
            [-0.2, -0.2, 0.0],
        ])

        # auto -> QSD on the stiff matrix, bisection on the mild one
        sel = FPTASelector(solver="auto"); sel.M_abs_reduced = stiff
        assert sel.get_exit_time().is_ok() and sel._use_qsd is True
        sel = FPTASelector(solver="auto"); sel.M_abs_reduced = mild
        assert sel.get_exit_time().is_ok() and sel._use_qsd is False

        # forced modes ignore stiffness
        sel = FPTASelector(solver="bisection"); sel.M_abs_reduced = stiff
        sel.get_exit_time()
        assert sel._use_qsd is False
        sel = FPTASelector(solver="qsd"); sel.M_abs_reduced = mild
        assert sel.get_exit_time().is_ok() and sel._use_qsd is True


    def test_ftpa(self, test_logger, connectivity_table_Cu) : 

        test_logger.debug("FTPA selector for Copper fake") 
        #Get fake connectivity table (Cu 1 sia 1 vac, remove transition sia event)
        connectivity_table = connectivity_table_Cu

        selector = FPTASelector() 
        result = selector.select_from_connectivity(connectivity_table) 
        
        test_logger.debug("For connectivity table : \n {}".format(connectivity_table.df))
        test_logger.debug("FTPASelector build Generator matrix : \n {}".format(selector.M_abs))
        test_logger.debug("And reduced matrix : \n {}".format(selector.M_abs_reduced))
        test_logger.debug("Got exit time = {} and exit state = {}".format(result.ok_value().t_exit, result.ok_value().exit_state))

    def test_auto_fallback_bisection_to_qsd(self, monkeypatch) :
        """solver='auto': when bisection fails below the stiffness heuristic, the
        QSD solver is tried as a backstop; forced 'bisection' stays strict."""
        from pykmc.result import Err, ErrorInfo, ErrorType
        from pykmc.basins import selection as selection_mod

        mild = np.array([
            [0.3, -0.1, 0.0],
            [-0.1, 0.3, 0.0],
            [-0.2, -0.2, 0.0],
        ])
        failing = Err(ErrorInfo(type=ErrorType.BASIN_TEXIT_NOT_FOUND, message="boom"))

        class _FailingBisection:
            def __init__(self, *a, **k):
                pass
            def solve(self):
                return failing

        monkeypatch.setattr(selection_mod, "BisectionSolver", _FailingBisection)

        sel = FPTASelector(solver="auto"); sel.M_abs_reduced = mild
        result = sel.get_exit_time()
        assert result.is_ok()
        assert sel._use_qsd is True

        sel = FPTASelector(solver="bisection"); sel.M_abs_reduced = mild
        result = sel.get_exit_time()
        assert not result.is_ok()

    def test_excluded_states_never_selected(self, connectivity_table_Cu) :
        """Excluded absorbing states get zero probability; the draw avoids them."""
        selector = FPTASelector()
        result = selector.select_from_connectivity(connectivity_table_Cu)
        assert result.is_ok()
        t_exit = result.ok_value().t_exit
        n_transient = len(selector.M_abs_reduced) - 1
        n_states = len(selector.M_abs)
        absorbing = set(range(n_transient, n_states))

        # exclude one absorbing state: 200 draws must never land on it
        excluded = {n_transient}
        for _ in range(200):
            choice = selector.select_absorbing_state(t_exit, excluded_states=excluded)
            assert choice is not None
            assert choice not in excluded
            assert choice in absorbing

    def test_all_excluded_returns_none_and_err(self, connectivity_table_Cu) :
        """With every absorbing exit excluded the draw returns None and
        select_from_connectivity maps it to Err(BASIN_NO_VIABLE_EXIT)."""
        from pykmc.result import ErrorType

        selector = FPTASelector()
        result = selector.select_from_connectivity(connectivity_table_Cu)
        assert result.is_ok()
        t_exit = result.ok_value().t_exit
        n_transient = len(selector.M_abs_reduced) - 1
        all_absorbing = set(range(n_transient, len(selector.M_abs)))

        assert selector.select_absorbing_state(t_exit, excluded_states=all_absorbing) is None

        result = selector.select_from_connectivity(connectivity_table_Cu, excluded_states=all_absorbing)
        assert not result.is_ok()
        assert result.err_value().type == ErrorType.BASIN_NO_VIABLE_EXIT

    def test_auto_recovers_when_direct_mfpt_solve_fails(self, test_logger: logging.Logger) -> None :
        """solver='auto' on a basin whose direct MFPT solve is numerically dead.

        Fast 0<->1 at 1e14 g, slow 1<->2 at 1e-3 g, escape g from state 2 into two
        absorbing states. ``np.linalg.solve`` on Q returns garbage there, so the
        QSD solver must reach its cancellation-free fallback and still hand
        ``select_absorbing_state`` a valid occupation-time weight vector.
        """
        from tests.basins.test_exit_time_solver import _exact_mfpt_fraction

        w, s, g = 1.0e14, 1.0e-6, 1.0e-3
        # 3 transient + 2 absorbing; state 2 escapes to both absorbers at g/2.
        M_abs = np.array([
            [w,    -w,     0.0,    0.0, 0.0],
            [-w,   w + s,  -s,     0.0, 0.0],
            [0.0,  -s,     s + g,  0.0, 0.0],
            [0.0,  0.0,    -g / 2, 0.0, 0.0],
            [0.0,  0.0,    -g / 2, 0.0, 0.0],
        ])
        sel = FPTASelector(solver="auto")
        sel.M_abs = M_abs
        sel.build_reduced_matrix(3)
        exact_mfpt = _exact_mfpt_fraction(sel.M_abs_reduced)

        np.random.seed(20260825)
        r1 = np.random.random()
        np.random.seed(20260825)
        result = sel.get_exit_time()
        assert result.is_ok(), result.err_value() if not result.is_ok() else ""
        t_exit = result.ok_value().t_exit
        assert np.isfinite(t_exit) and t_exit > 0
        np.testing.assert_allclose(t_exit, -np.log(1.0 - r1) * exact_mfpt, rtol=1e-9)
        assert sel._use_qsd is True
        assert sel._qsd is not None
        assert np.all(sel._qsd >= 0)
        np.testing.assert_allclose(np.sum(sel._qsd), 1.0, rtol=1e-12)

        exit_state = sel.select_absorbing_state(t_exit)
        assert exit_state in (3, 4)

    def test_auto_falls_back_when_qsd_fails(self, monkeypatch: typing.Any) -> None :
        """solver='auto': a failed QSD is no longer returned as the basin error.

        The stiff branch is forced to fail; 'auto' must fall back to bisection and
        return Ok with _use_qsd/_qsd reset. Forced 'qsd' stays strict.
        """
        from pykmc.result import Err, ErrorInfo, ErrorType
        from pykmc.basins import selection as selection_mod

        stiff = np.array([
            [1.0e3 + 1.0e-3, -1.0e3,          0.0],
            [-1.0e3,          1.0e3 + 1.0e-3, 0.0],
            [-1.0e-3,        -1.0e-3,         0.0],
        ])

        failing = Err(ErrorInfo(type=ErrorType.BASIN_TEXIT_NOT_FOUND, message="boom"))

        class _FailingQSD:
            def __init__(self, *a: typing.Any, **k: typing.Any) -> None:
                self.qsd = None

            def solve(self) -> typing.Any :
                return failing

        monkeypatch.setattr(selection_mod, "QSDSolver", _FailingQSD)

        sel = FPTASelector(solver="auto")
        sel.M_abs_reduced = stiff
        result = sel.get_exit_time()
        assert result.is_ok()
        assert sel._use_qsd is False
        assert sel._qsd is None

        sel = FPTASelector(solver="qsd")
        sel.M_abs_reduced = stiff
        assert not sel.get_exit_time().is_ok()
        assert sel._use_qsd is False
