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

    def test_com_fingerprint_boundary_crossing_invariance(self):
        """COM fingerprint must be invariant when translation causes atoms to wrap."""
        positions = np.array([[0.5, 0.5, 0.5], [1.5, 0.5, 0.5],
                              [0.5, 1.5, 0.5], [0.5, 0.5, 1.5]], dtype=float)
        cell = np.diag([10.0, 10.0, 10.0])
        pbc = np.array([True, True, True])
        fp1 = BasinsGenericEvents._com_fingerprint(positions, cell, pbc)
        # Shift so atoms wrap: 0.5 + 9.7 = 10.2 → wraps to 0.2
        shifted = positions + np.array([9.7, 9.7, 9.7])
        fp2 = BasinsGenericEvents._com_fingerprint(shifted, cell, pbc)
        assert np.allclose(fp1, fp2, atol=1e-10)

    def test_aoi_fingerprint_boundary_crossing_invariance(self):
        """Atoms-of-interest fingerprint must be invariant when shift causes wrapping."""
        # Small cluster of 4 atoms (all undercoordinated with coord_thr=10)
        positions = np.array([[0.5, 0.5, 0.5], [1.5, 0.5, 0.5],
                              [0.5, 1.5, 0.5], [0.5, 0.5, 1.5]], dtype=float)
        cell = np.diag([10.0, 10.0, 10.0])
        pbc = np.array([True, True, True])
        fp1 = BasinsGenericEvents._atoms_of_interest_fingerprint(
            positions, cell, pbc, rnei=1.5, coord_thr=10,
        )
        # Shift so atoms straddle the x=0/x=L boundary
        shifted = positions + np.array([9.7, 0.0, 0.0])
        fp2 = BasinsGenericEvents._atoms_of_interest_fingerprint(
            shifted, cell, pbc, rnei=1.5, coord_thr=10,
        )
        assert np.allclose(fp1, fp2, atol=1e-10)

    def test_circular_mean_localized_cluster(self):
        """Circular mean preserves COM-to-atom distances across boundary wrapping."""
        box = np.array([10.0, 10.0, 10.0])
        pbc = np.array([True, True, True])
        # Cluster centered near (1, 1, 1)
        cluster = np.array([[0.5, 0.8, 1.0], [1.5, 1.2, 0.9],
                            [1.0, 0.5, 1.1], [1.0, 1.5, 1.0]], dtype=float)
        com1, r1 = BasinsGenericEvents._circular_mean_position(cluster, box, pbc)
        assert np.all(r1 > 0.9), "Localized cluster should have high resultant"

        # Shift to straddle boundary: add (9.5, 0, 0) → wraps around x
        shifted = (cluster + np.array([9.5, 0.0, 0.0])) % box
        com2, r2 = BasinsGenericEvents._circular_mean_position(shifted, box, pbc)

        # COM-to-atom minimum-image distances must be identical
        def _mic_dists(positions, com):
            diffs = positions - com
            for dim in range(3):
                diffs[:, dim] -= np.round(diffs[:, dim] / box[dim]) * box[dim]
            return np.sort(np.linalg.norm(diffs, axis=1))

        dists1 = _mic_dists(cluster, com1)
        dists2 = _mic_dists(shifted, com2)
        assert np.allclose(dists1, dists2, atol=1e-10)

    def test_circular_mean_fallback_triggers(self):
        """Uniform atom distribution should produce low resultant (ill-conditioned)."""
        box = np.array([10.0, 10.0, 10.0])
        pbc = np.array([True, True, True])
        # Uniformly distributed atoms across the cell
        rng = np.random.default_rng(42)
        uniform_pos = rng.uniform(0, 10, size=(200, 3))
        _, resultant = BasinsGenericEvents._circular_mean_position(uniform_pos, box, pbc)
        # At least one dimension should have low resultant for 200 uniform points
        assert np.any(resultant < 0.2), f"Expected low resultant for uniform dist, got {resultant}"

    def test_two_component_discriminates_position(self):
        """Fingerprint has K+1 elements: K sorted defect distances + 1 bulk-relative scalar.

        Creates a system with a dense bulk cluster (many neighbors, fully
        coordinated above coord_thr) plus a sparse defect cluster
        (undercoordinated). Verifies the fingerprint encodes position info
        via the last element.
        """
        cell = np.diag([30.0, 30.0, 30.0])
        pbc = np.array([True, True, True])
        # Dense bulk: 10 atoms packed within a 0.4 Ang sphere at (15, 15, 15)
        # Each has 9 neighbors within rnei=1.5 → above coord_thr=4
        rng = np.random.default_rng(123)
        bulk = rng.uniform(-0.2, 0.2, size=(10, 3)) + np.array([15.0, 15.0, 15.0])
        # Sparse defect: 3 atoms with spacing ~3 Ang, far from bulk
        # Each has 0 neighbors within rnei=1.5 → below coord_thr=4
        defect = np.array([[0.0, 0.0, 0.0], [3.0, 0.0, 0.0], [0.0, 3.0, 0.0]], dtype=float)
        pos1 = np.vstack([bulk, defect + np.array([3.0, 3.0, 3.0])])
        pos2 = np.vstack([bulk, defect + np.array([25.0, 25.0, 25.0])])
        fp1 = BasinsGenericEvents._atoms_of_interest_fingerprint(
            pos1, cell, pbc, rnei=1.5, coord_thr=4,
        )
        fp2 = BasinsGenericEvents._atoms_of_interest_fingerprint(
            pos2, cell, pbc, rnei=1.5, coord_thr=4,
        )
        # Should have 3 defect atoms + 1 scalar = 4 elements
        assert len(fp1) == 4
        assert len(fp2) == 4
        # Internal distances (first 3 elements) should match (same defect shape)
        assert np.allclose(fp1[:3], fp2[:3], atol=1e-10)
        # Bulk-relative scalar (last element) should differ (different position)
        assert not np.isclose(fp1[-1], fp2[-1], atol=0.1)


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
