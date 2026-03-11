from unittest.mock import Mock

from pykmc import System
from pykmc.basins import BasinsGenericEvents, BasinStatesConnectivity
from pykmc.result import Ok, Err, ErrorInfo, ErrorType
import logging
from mpi4py import MPI
import numpy as np
from pykmc.enginemanager.lmpi.pool import ManagerFactory

logger = logging.getLogger("tests")


def _basin_mock_config() -> Mock:
    """Create a Mock config that routes to COM fingerprint (no atoms-of-interest)."""
    config = Mock()
    config.basin.fingerprint_coordination_thr = None
    config.basin.fingerprint_tolerance = None
    config.atomicenvironment.style = "cna"
    config.atomicenvironment.coordination_threshold = None
    return config


def _toy_system(offset: float) -> System:
    return System(
        positions=np.array([[offset, 0.0, 0.0], [offset + 1.0, 0.0, 0.0]], dtype=float),
        types=np.array(["Ni", "Ni"]),
        cell=np.diag([20.0, 20.0, 20.0]),
        pbc=np.array([True, True, True]),
        index=np.array([0, 1]),
    )


def _skip_without_ranks(n_sessions: int, use_rank_0: bool) -> None:
    required_ranks = n_sessions if use_rank_0 else n_sessions + 1
    if MPI.COMM_WORLD.Get_size() < required_ranks:
        import pytest

        pytest.skip(f"requires mpirun with at least {required_ranks} ranks")


class _CompletedFuture:
    def __init__(self, value):
        self._value = value

    def result(self):
        return self._value


class _FailedFuture:
    def __init__(self, exc):
        self._exc = exc

    def result(self):
        raise self._exc


class TestFingerprint:

    def test_com_fingerprint_permutation_invariance(self):
        """COM fingerprint should be invariant to atom permutation."""
        positions = np.array([[0,0,0],[1,0,0],[0,1,0],[0,0,1]], dtype=float)
        cell = np.diag([10.0, 10.0, 10.0])
        pbc = np.array([True, True, True])
        fp1 = BasinsGenericEvents._com_fingerprint(positions, cell, pbc)
        fp2 = BasinsGenericEvents._com_fingerprint(positions[[2,0,3,1]], cell, pbc)
        assert np.allclose(fp1, fp2)

    def test_com_fingerprint_translation_invariance(self):
        """COM fingerprint should be invariant to uniform translation (no boundary crossing)."""
        positions = np.array([[1,1,1],[2,1,1],[1,2,1],[1,1,2]], dtype=float)
        cell = np.diag([10.0, 10.0, 10.0])
        pbc = np.array([True, True, True])
        fp1 = BasinsGenericEvents._com_fingerprint(positions, cell, pbc)
        fp2 = BasinsGenericEvents._com_fingerprint(positions + [3.0, 3.0, 3.0], cell, pbc)
        assert np.allclose(fp1, fp2)

    def test_com_fingerprint_different_structures(self):
        """Different structures should produce different COM fingerprints."""
        cell = np.diag([10.0, 10.0, 10.0])
        pbc = np.array([True, True, True])
        pos1 = np.array([[0,0,0],[1,0,0],[0,1,0],[0,0,1]], dtype=float)
        pos2 = np.array([[0,0,0],[3,0,0],[0,3,0],[0,0,3]], dtype=float)
        fp1 = BasinsGenericEvents._com_fingerprint(pos1, cell, pbc)
        fp2 = BasinsGenericEvents._com_fingerprint(pos2, cell, pbc)
        assert not np.allclose(fp1, fp2, atol=0.3)

    def test_atoms_of_interest_fingerprint_permutation_invariance(self):
        """Atoms of interest fingerprint should be invariant to atom permutation."""
        positions = np.array([[0,0,0],[1,0,0],[0,1,0],[0,0,1]], dtype=float)
        cell = np.diag([10.0, 10.0, 10.0])
        pbc = np.array([True, True, True])
        fp1 = BasinsGenericEvents._atoms_of_interest_fingerprint(positions, cell, pbc, rnei=1.5, coord_thr=10)
        fp2 = BasinsGenericEvents._atoms_of_interest_fingerprint(positions[[2,0,3,1]], cell, pbc, rnei=1.5, coord_thr=10)
        assert np.allclose(fp1, fp2)

    def test_atoms_of_interest_fingerprint_different_structures(self):
        """Different structures should produce different atoms of interest fingerprints."""
        cell = np.diag([10.0, 10.0, 10.0])
        pbc = np.array([True, True, True])
        pos1 = np.array([[0,0,0],[1,0,0],[0,1,0],[0,0,1]], dtype=float)
        pos2 = np.array([[0,0,0],[3,0,0],[0,3,0],[0,0,3]], dtype=float)
        fp1 = BasinsGenericEvents._atoms_of_interest_fingerprint(pos1, cell, pbc, rnei=1.5, coord_thr=10)
        fp2 = BasinsGenericEvents._atoms_of_interest_fingerprint(pos2, cell, pbc, rnei=1.5, coord_thr=10)
        assert not np.allclose(fp1, fp2)


class TestBasin :

    def test_initialize_resets_cached_state(self):
        basin = BasinsGenericEvents(config=_basin_mock_config(), reference_table=Mock(), known_environments=set(), manager=Mock())
        basin.states[99] = Mock()
        basin._state_fingerprints[99] = np.array([1.0])
        basin.absorbing_saddle_positions[99] = np.array([[0.0, 0.0, 0.0]])
        basin._was_capped = True

        basin._initialize(_toy_system(0.0))

        assert set(basin.states.keys()) == {0}
        assert set(basin._state_fingerprints.keys()) == {0}
        assert basin.absorbing_saddle_positions == {}
        assert basin._was_capped is False

    def test_cap_remaining_states_materializes_frontier(self):
        basin = BasinsGenericEvents(config=_basin_mock_config(), reference_table=Mock(), known_environments=set(), manager=Mock())
        basin.connectivity_table = BasinStatesConnectivity()
        basin._add_state(state_index=0, system=_toy_system(0.0))
        basin.states_to_explore = [1, 2]
        basin.explored_states = [0]

        for state_idx in (1, 2):
            basin.connectivity_table.add_connectivity(
                state=0,
                state_connexion=state_idx,
                event_connexion=state_idx,
                central_atom=0,
                sym=0,
                transient=True,
                dE_forward=0.1,
                k_forward=1.0,
                dE_backward=0.1,
                k_backward=1.0,
            )

        systems = {
            1: _toy_system(2.0),
            2: _toy_system(4.0),
        }
        basin.system_from_state = Mock(
            side_effect=lambda from_state, event_idx, central_atom, sym_idx: Ok(systems[event_idx])
        )

        result = basin._cap_remaining_as_absorbing()

        assert result.is_ok()
        assert basin.states_to_explore == []
        assert set(basin.states.keys()) == {0, 1, 2}
        assert basin.states[1].transient is False
        assert basin.states[2].transient is False
        assert not basin.connectivity_table.df["transient"].any()
        assert basin._was_capped is True

    def test_parallel_reconstruction_failures_are_returned(self):
        manager = Mock()
        manager.use_local = Mock()
        manager.use_global = Mock()
        manager.basin_reconstruct = Mock(
            return_value=_CompletedFuture(
                {
                    "ok": False,
                    "error_type": "RECONSTRUCTION_INVALID_MIN2",
                    "message": "boom",
                }
            )
        )

        config = _basin_mock_config()
        config.basin.strategy = "wavefront"
        config.basin.n_workers = 2
        config.basin.max_states = None

        basin = BasinsGenericEvents(config=config, reference_table=Mock(), known_environments=set(), manager=manager)
        basin.connectivity_table = BasinStatesConnectivity()
        basin.connectivity_table.add_connectivity(
            state=0,
            state_connexion=1,
            event_connexion=1,
            central_atom=0,
            sym=0,
            transient=True,
            dE_forward=0.1,
            k_forward=1.0,
            dE_backward=0.1,
            k_backward=1.0,
        )
        basin._add_state(state_index=0, system=_toy_system(0.0))
        basin.states_to_explore = [1]
        basin.explored_states = []
        basin._prepare_reconstruct_kwargs = Mock(return_value={})

        result = basin.construct_connexion_table_parallel()

        assert not result.is_ok()
        assert result.err_value().message == "boom"
        manager.use_local.assert_called_once()
        manager.use_global.assert_called_once()

    def test_parallel_reconstruction_transport_failures_are_returned(self):
        manager = Mock()
        manager.use_local = Mock()
        manager.use_global = Mock()
        manager.basin_reconstruct = Mock(return_value=_FailedFuture(RuntimeError("remote boom")))

        config = _basin_mock_config()
        config.basin.strategy = "wavefront"
        config.basin.n_workers = 2
        config.basin.max_states = None

        basin = BasinsGenericEvents(config=config, reference_table=Mock(), known_environments=set(), manager=manager)
        basin.connectivity_table = BasinStatesConnectivity()
        basin.connectivity_table.add_connectivity(
            state=0,
            state_connexion=1,
            event_connexion=1,
            central_atom=0,
            sym=0,
            transient=True,
            dE_forward=0.1,
            k_forward=1.0,
            dE_backward=0.1,
            k_backward=1.0,
        )
        basin._add_state(state_index=0, system=_toy_system(0.0))
        basin.states_to_explore = [1]
        basin.explored_states = []
        basin._prepare_reconstruct_kwargs = Mock(return_value={})

        result = basin.construct_connexion_table_parallel()

        assert not result.is_ok()
        assert result.err_value().type == ErrorType.MPI_REMOTE_ERROR
        assert "remote boom" in result.err_value().message
        manager.use_local.assert_called_once()
        manager.use_global.assert_called_once()

    def test_parallel_exploration_transport_failures_are_returned(self):
        manager = Mock()
        manager.basin_explore = Mock(return_value=_FailedFuture(RuntimeError("explore boom")))

        basin = BasinsGenericEvents(config=Mock(), reference_table=Mock(), known_environments=set(), manager=manager)
        basin._next_state_index = 1
        basin._estimate_max_transitions_per_state = Mock(return_value=4)
        basin._prepare_explore_kwargs = Mock(return_value={})

        result = basin._explore_states_parallel([0])

        assert not result.is_ok()
        assert result.err_value().type == ErrorType.MPI_REMOTE_ERROR
        assert "explore boom" in result.err_value().message

    def test_refine_absorbing_restores_global_mode_on_failure(self):
        manager = Mock()
        manager.use_local = Mock()
        manager.use_global = Mock()

        basin = BasinsGenericEvents(config=Mock(), reference_table=Mock(), known_environments=set(), manager=manager)
        basin.refine_absorbing = Mock(
            return_value=Err(ErrorInfo(type=ErrorType.REFINEMENT_INVALID_MINIMA, message="refine failed"))
        )

        result = basin._refine_absorbing_states(_toy_system(0.0))

        assert not result.is_ok()
        assert result.err_value().message == "refine failed"
        manager.use_local.assert_called_once()
        manager.use_global.assert_called_once()

    def test_connectivity_table_construction(self, test_logger, config_Cu, reference_table_Cu_fake, system_Cu, visited_environments_Cu) :
        _skip_without_ranks(
            n_sessions=config_Cu.control.n_sessions,
            use_rank_0=config_Cu.control.engine_use_rank_0,
        )
        
        #Create Manager
        factory = ManagerFactory(
            n_sessions=config_Cu.control.n_sessions,
            use_rank_0=config_Cu.control.engine_use_rank_0,
        )
        manager = factory.launch()

        if manager is not None: #On rank 0
            manager.initialize_sessions(config_Cu, system_Cu)

            self.basin = BasinsGenericEvents(config=config_Cu, reference_table=reference_table_Cu_fake, known_environments=visited_environments_Cu, manager = None)
            self.basin.manager = manager

            result = self.basin.execute(system=system_Cu)
            if result.is_ok() : 
                test_logger.debug("Find Exit State : ")
                test_logger.debug("Exit time t_exit = {}ps".format(result.ok_value().t_exit))
                test_logger.debug("Exit state n : {}".format(result.ok_value().exit_state))
            else : 
                test_logger.debug("Error: {}".format(result.err_value()))
            
            manager.close_all()
