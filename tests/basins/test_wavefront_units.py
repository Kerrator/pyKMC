"""Pure-python unit tests for the wavefront building blocks.

Covers the two pieces of the wavefront strategy whose logic does not need the MPI
manager: intra-batch deduplication (``is_new_state_batch``) and the ``max_states``
frontier capping (``_cap_remaining_as_absorbing`` with already-materialized states).
"""

from types import SimpleNamespace

import numpy as np

from pykmc import System
from pykmc.basins import BasinsGenericEvents, BasinStatesConnectivity
from pykmc.basins.basin import StateData
from pykmc.basins import fingerprinting


def _make_system(positions: np.ndarray) -> System:
    cell = np.diag([10.0, 10.0, 10.0])
    pbc = np.array([True, True, True])
    n = len(positions)
    return System(positions=np.array(positions, dtype=float), types=["Cu"] * n,
                  cell=cell, pbc=pbc, index=np.arange(n))


def _make_basin(config) -> BasinsGenericEvents:
    """Build a BasinsGenericEvents without running __init__ (no manager needed)."""
    basin = BasinsGenericEvents.__new__(BasinsGenericEvents)
    basin.config = config
    basin.states = {}
    basin._state_fingerprints = {}
    basin._was_capped = False
    basin.states_to_explore = []
    basin.explored_states = []
    basin.connectivity_table = BasinStatesConnectivity()
    basin._exit_excluded_states = set()
    basin._deferred_states = set()
    basin._deferred_systems = {}
    basin._n_failed = 0
    basin._n_reconstruction_attempts = 0
    basin.absorbing_saddle_positions = {}
    return basin


def _com_config(fingerprint_mode: str = "auto") -> SimpleNamespace:
    """Config stub routing compute_fingerprint to the COM-distance fallback."""
    return SimpleNamespace(
        basin=SimpleNamespace(fingerprint_mode=fingerprint_mode,
                              fingerprint_coordination_thr=None, fingerprint_tolerance=1.0,
                              strategy="wavefront", n_workers=2),
        atomicenvironment=SimpleNamespace(style="graph", coordination_threshold=None, rnei=3.0),
    )


class TestIsNewStateBatch:

    def test_detects_existing_intra_batch_and_new(self):
        """Batch dedup must catch duplicates of existing states AND of batch members."""
        config = _com_config()
        basin = _make_basin(config)

        base = np.array([[1.0, 1.0, 1.0], [2.0, 1.0, 1.0],
                         [1.0, 2.0, 1.0], [1.0, 1.0, 2.0]])
        sys_a = _make_system(base)
        # Same physical state in a different periodic representation (some atoms
        # expressed one box-image over; identical once wrapped)
        reimaged = base.copy()
        reimaged[1] += np.array([10.0, 0.0, 0.0])
        reimaged[3] -= np.array([0.0, 0.0, 10.0])
        sys_a_wrapped = _make_system(reimaged)
        # A genuinely different state (cluster elsewhere with different shape)
        sys_b = _make_system(np.array([[5.0, 5.0, 5.0], [6.5, 5.0, 5.0],
                                       [5.0, 6.5, 5.0], [5.0, 5.0, 6.5]]))
        sys_b_copy = _make_system(sys_b.positions.copy())

        # Existing state 0 = A
        basin.states[0] = StateData(system=sys_a, environment=None, neighbors_list=None)
        basin._state_fingerprints[0] = fingerprinting.compute_fingerprint(
            config, sys_a.positions, sys_a.cell, sys_a.pbc)

        results = basin.is_new_state_batch({5: sys_a_wrapped, 6: sys_b, 7: sys_b_copy})

        assert results[5] == 0      # duplicate of the existing state
        assert results[6] == -1     # genuinely new
        assert results[7] == 6      # intra-batch duplicate of 6

    def test_empty_batch(self):
        basin = _make_basin(_com_config())
        assert basin.is_new_state_batch({}) == {}

    def test_off_mode_still_finds_duplicates(self):
        """With fingerprint_mode='off' the pre-filter is skipped but dedup must still work."""
        config = _com_config(fingerprint_mode="off")
        basin = _make_basin(config)

        base = np.array([[1.0, 1.0, 1.0], [2.0, 1.0, 1.0],
                         [1.0, 2.0, 1.0], [1.0, 1.0, 2.0]])
        sys_a = _make_system(base)
        sys_a_copy = _make_system(base.copy())
        sys_b = _make_system(base + np.array([4.0, 4.0, 4.0]))

        basin.states[0] = StateData(system=sys_a, environment=None, neighbors_list=None)
        # off mode: nothing cached
        assert fingerprinting.compute_fingerprint(config, sys_a.positions, sys_a.cell, sys_a.pbc) is None

        results = basin.is_new_state_batch({5: sys_a_copy, 6: sys_b})
        assert results[5] == 0    # duplicate found via full comparison, no pre-filter
        assert results[6] == -1   # translated cluster = different state

        # serial path too
        assert basin.is_new_state(sys_a_copy) == 0
        assert basin.is_new_state(sys_b) == -1


class TestResultFromMpi:

    def test_engine_positions_sanitized_into_box(self):
        """Engine positions at/over the cell boundary must survive _result_from_mpi.

        Regression (both directions): re-wrapping with ase.geometry.wrap_positions
        shifted boundary atoms (a slab layer at z = 0) to ~-5e-6, and raw LAMMPS
        minimize output can sit slightly OUTSIDE the box (LAMMPS only re-wraps on
        reneighbouring). Either kills the periodic cKDTree in NeighborsList
        ('Negative input data...' / 'Some input data are greater than the size...').
        _result_from_mpi must route through System.update_positions, which wraps and
        clamps exactly like the serial path.
        """
        from scipy.spatial import cKDTree

        config = _com_config()
        basin = _make_basin(config)
        from_positions = np.array([[0.0, 0.0, 0.0], [2.0, 2.0, 0.0],
                                   [1.0, 1.0, 1.5], [3.0, 3.0, 1.5]])
        basin.states[0] = StateData(system=_make_system(from_positions),
                                    environment=None, neighbors_list=None)

        # engine output: one atom slightly negative, one slightly past the box edge
        engine_positions = from_positions.copy()
        engine_positions[0, 2] = -5.0e-6          # ase-style boundary negative
        engine_positions[3, 0] = 10.0 + 3.0e-7    # LAMMPS drift past the box

        mpi_result = {"ok": True, "min2_positions": engine_positions, "min2_etot": -1.0}
        result = basin._result_from_mpi(mpi_result, from_state=0)
        assert result.is_ok()
        pos = result.ok_value().positions
        box = np.diag(result.ok_value().cell)
        assert (pos >= 0.0).all()
        assert (pos < box).all()
        # the exact construction NeighborsList performs must not raise
        cKDTree(pos, boxsize=box.tolist())

    def test_error_payload_maps_to_err(self):
        basin = _make_basin(_com_config())
        result = basin._result_from_mpi({"ok": False, "error_type": "PSR_NO_MATCH_FOUND",
                                         "message": "no match"}, from_state=0)
        assert not result.is_ok()


class TestSerialTimingCheckpoint:

    def test_finalizer_writes_serial_checkpoints(self, tmp_path, monkeypatch):
        """_finalize_exploration_run writes the timing files compare_scaling.py reads."""
        monkeypatch.chdir(tmp_path)
        config = _com_config()
        config.basin.strategy = "serial"
        basin = _make_basin(config)
        basin.connectivity_table.add_connectivity(
            state=0, state_connexion=1, event_connexion=10, central_atom=0, sym=0,
            transient=False, dE_forward=0.1, k_forward=1.0, dE_backward=0.1, k_backward=1.0)

        import time as _time
        prof = {"reconstruct": 0.1, "psr": 0.0, "minimize": 0.0, "dedup": 0.2,
                "ensure_state": 0.0, "explore": 0.05, "merge": 0.01, "other": 0.0}
        result = basin._finalize_exploration_run(_time.perf_counter() - 1.0, prof, 3, 1,
                                                 strategy_label="serial")
        assert result.is_ok()
        assert (tmp_path / "basin_timing_serial.txt").exists()
        assert (tmp_path / "basin_connectivity_0_L0_level_complete.txt").exists()
        content = (tmp_path / "basin_timing_serial.txt").read_text()
        assert "strategy = serial" in content
        assert "wall_time_s" in content


class TestCapRemainingAsAbsorbing:

    def test_materialized_frontier_flips_to_absorbing(self):
        """With all frontier states already materialized, capping flips them to
        absorbing in the connectivity table, clears the queue, and sets _was_capped.
        """
        config = _com_config()
        basin = _make_basin(config)

        pos = np.array([[1.0, 1.0, 1.0], [2.0, 1.0, 1.0]])
        for idx in (0, 1, 2):
            basin.states[idx] = StateData(system=_make_system(pos + idx),
                                          environment=None, neighbors_list=None,
                                          transient=True)
        basin.states_to_explore = [1, 2]

        for target in (1, 2):
            basin.connectivity_table.add_connectivity(
                state=0, state_connexion=target, event_connexion=10 + target,
                central_atom=0, sym=0, transient=True,
                dE_forward=0.1, k_forward=1.0, dE_backward=0.1, k_backward=1.0)

        result = basin._cap_remaining_as_absorbing()

        assert result.is_ok()
        assert basin._was_capped is True
        assert basin.states_to_explore == []
        assert sorted(basin.explored_states) == [1, 2]
        df = basin.connectivity_table.df
        assert not df[df["state_connexion"].isin([1, 2])]["transient"].any()
        assert basin.states[1].transient is False
        assert basin.states[2].transient is False
        assert basin._deferred_states == set()

    def test_unmaterialized_frontier_deferred_without_reconstruction(self):
        """Unmaterialized frontier states are capped as deferred absorbing states with
        NO engine reconstruction or deduplication (basin.manager is not even set here,
        so any manager use would raise AttributeError)."""
        config = _com_config()
        basin = _make_basin(config)

        pos = np.array([[1.0, 1.0, 1.0], [2.0, 1.0, 1.0]])
        basin.states[0] = StateData(system=_make_system(pos),
                                    environment=None, neighbors_list=None,
                                    transient=True)
        basin.states_to_explore = [1, 2]

        for target in (1, 2):
            basin.connectivity_table.add_connectivity(
                state=0, state_connexion=target, event_connexion=10 + target,
                central_atom=0, sym=0, transient=True,
                dE_forward=0.1, k_forward=1.0, dE_backward=0.1, k_backward=1.0)

        result = basin._cap_remaining_as_absorbing(reason="max_basin_walltime_s=1.0")

        assert result.is_ok()
        assert basin._was_capped is True
        assert basin.states_to_explore == []
        assert basin._deferred_states == {1, 2}
        assert 1 not in basin.states and 2 not in basin.states
        df = basin.connectivity_table.df
        assert not df[df["state_connexion"].isin([1, 2])]["transient"].any()
        # rows keep their exploration rates (k_tot / t_exit stay exact)
        assert (df[df["state_connexion"].isin([1, 2])]["k_forward"] == 1.0).all()


class TestMarkFailedAbsorbing:

    def test_row_kept_absorbing_and_excluded_from_exit_draw(self):
        """A failed state keeps its connectivity row (rate mass preserved), flips
        absorbing, leaves the frontier, and is excluded from the exit draw."""
        config = _com_config()
        basin = _make_basin(config)

        basin.states_to_explore = [1]
        basin.connectivity_table.add_connectivity(
            state=0, state_connexion=1, event_connexion=11,
            central_atom=0, sym=0, transient=True,
            dE_forward=0.2, k_forward=2.0, dE_backward=0.2, k_backward=2.0)

        from pykmc.result import ErrorInfo, ErrorType
        err = ErrorInfo(type=ErrorType.RECONSTRUCTION_INVALID_MIN2, message="delr2 = 2.46")
        basin._mark_failed_absorbing(1, err)

        df = basin.connectivity_table.df
        row = df[df["state_connexion"] == 1].iloc[0]
        assert row["transient"] == False  # noqa: E712 (pandas bool)
        assert row["k_forward"] == 2.0
        assert basin._exit_excluded_states == {1}
        assert basin._n_failed == 1
        assert basin.states_to_explore == []
        assert basin.explored_states == [1]


class TestMaterializeExitState:

    def test_deferred_system_is_deduped_and_merged(self):
        """A deferred system that equals a known state merges into it (row remap +
        saddle positions copied)."""
        config = _com_config()
        basin = _make_basin(config)

        pos = np.array([[1.0, 1.0, 1.0], [2.0, 1.0, 1.0]])
        basin.states[0] = StateData(system=_make_system(pos), environment=None,
                                    neighbors_list=None, transient=True)
        basin.states[1] = StateData(system=_make_system(pos + 3.0), environment=None,
                                    neighbors_list=None, transient=False)
        basin._state_fingerprints[1] = fingerprinting.compute_fingerprint(
            config, basin.states[1].system.positions,
            basin.states[1].system.cell, basin.states[1].system.pbc)

        # state 2: deferred, physically identical to state 1
        basin._deferred_states.add(2)
        basin._deferred_systems[2] = _make_system(pos + 3.0)
        basin.absorbing_saddle_positions[2] = np.array([[0.5, 0.5, 0.5]])
        basin.connectivity_table.add_connectivity(
            state=0, state_connexion=2, event_connexion=12,
            central_atom=0, sym=0, transient=False,
            dE_forward=0.1, k_forward=1.0, dE_backward=0.1, k_backward=1.0)

        result = basin._materialize_exit_state(2)

        assert result.is_ok()
        assert result.ok_value() == 1
        assert 2 not in basin._deferred_systems
        df = basin.connectivity_table.df
        assert (df["state_connexion"] == 1).any() and not (df["state_connexion"] == 2).any()
        assert 1 in basin.absorbing_saddle_positions

    def test_deferred_system_new_state_is_added(self):
        """A genuinely new deferred system is added to states as absorbing."""
        config = _com_config()
        basin = _make_basin(config)

        pos = np.array([[1.0, 1.0, 1.0], [2.0, 1.0, 1.0]])
        basin.states[0] = StateData(system=_make_system(pos), environment=None,
                                    neighbors_list=None, transient=True)
        basin._deferred_states.add(1)
        basin._deferred_systems[1] = _make_system(pos + 4.0)

        result = basin._materialize_exit_state(1)

        assert result.is_ok()
        assert result.ok_value() == 1
        assert basin.states[1].system is not None
        assert basin.states[1].transient is False
        assert 1 not in basin._deferred_states
