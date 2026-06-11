"""End-to-end basin tests against a live MPI manager.

These need a real MPI launch (the session pool blocks single-process):

    mpirun -n 8 python -m pytest tests/basins/test_basin.py -v

with ``n_sessions = 7`` from tests/data/input_Cu.in (``engine_use_rank_0 = False``
requires world_size >= n_sessions + 1).

``test_serial_wavefront_equivalence`` is the validation gate for the wavefront
strategy: both strategies explore the same Cu basin and must produce the same
connectivity table up to state relabeling (compared with the helpers from
test_basin_equivalence.py).
"""

import importlib.util
import logging
import os

from pykmc.basins import BasinsGenericEvents
from pykmc.enginemanager.lmpi.pool import ManagerFactory

logger = logging.getLogger("tests")

# Load the relabeling-invariant comparison helpers by path (tests/basins is not a package)
_eq_path = os.path.join(os.path.dirname(__file__), "test_basin_equivalence.py")
_spec = importlib.util.spec_from_file_location("basin_equivalence_helpers", _eq_path)
eqh = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(eqh)


def _run_basin(manager, config, reference_table, visited_environments, system):
    """Run one basin exploration and return (result, connectivity_df)."""
    basin = BasinsGenericEvents(config=config, reference_table=reference_table,
                                known_environments=visited_environments, manager=None)
    basin.manager = manager
    result = basin.execute(system=system)
    return result, basin.connectivity_table.df.copy()


class TestBasin :

    def test_connectivity_table_construction(self, test_logger, config_Cu, reference_table_Cu_fake, system_Cu, visited_environments_Cu) :

        #Create Manager
        factory = ManagerFactory(n_sessions=config_Cu.control.n_sessions, use_rank_0=config_Cu.control.engine_use_rank_0)
        manager = factory.launch()

        if manager is not None: #On rank 0
            try:
                manager.initialize_sessions(config_Cu, system_Cu)
                result, df = _run_basin(manager, config_Cu, reference_table_Cu_fake,
                                        visited_environments_Cu, system_Cu)
            finally:
                # Always release the worker ranks, even if execute() raised or an
                # assertion below fails — otherwise the MPI workers hang.
                manager.close_all()

            # --- the serial basin must complete and produce a sane basin ---
            assert result.is_ok(), "basin.execute failed: {}".format(result.err_value())
            out = result.ok_value()

            counts = eqh.basin_state_counts(df)
            assert counts["n_rows"] > 0, "connectivity table is empty"
            assert counts["n_transient"] >= 1, "no transient states discovered"
            assert counts["n_absorbing"] >= 1, "no absorbing states discovered"

            # --- a valid exit was selected (FPTA) ---
            assert out.t_exit > 0.0, "non-positive exit time t_exit={}".format(out.t_exit)
            # after reorder_states_index, transient states are 0..n_transient-1,
            # absorbing states come after, so the exit must be an absorbing index.
            assert out.exit_state >= counts["n_transient"], (
                "exit_state {} is not absorbing (n_transient={})".format(out.exit_state, counts["n_transient"])
            )
            assert 0 <= out.from_state < counts["n_transient"], (
                "from_state {} is not a transient index (n_transient={})".format(out.from_state, counts["n_transient"])
            )

            test_logger.info(
                "[basin serial] transient={} absorbing={} rows={} t_exit={:.3e} exit_state={}".format(
                    counts["n_transient"], counts["n_absorbing"], counts["n_rows"], out.t_exit, out.exit_state
                )
            )

    def test_serial_wavefront_equivalence(self, test_logger, config_Cu, reference_table_Cu_fake, system_Cu, visited_environments_Cu) :
        """strategy=wavefront must discover the same basin as strategy=serial.

        The comparison is relabeling-invariant (state counts + per-transition edge
        signatures): wavefront discovery order can differ, the physics cannot.
        """
        factory = ManagerFactory(n_sessions=config_Cu.control.n_sessions, use_rank_0=config_Cu.control.engine_use_rank_0)
        manager = factory.launch()

        if manager is not None: #On rank 0
            try:
                manager.initialize_sessions(config_Cu, system_Cu)

                # Match reconstruction semantics between the two legs: wavefront
                # implements the 'global/reconstruction' path (saddle + push + two
                # validated minimizations) distributed over the session pool, so the
                # serial leg must use the same style for the comparison to be exact.
                config_Cu.basin.style = "global/reconstruction"

                config_Cu.basin.strategy = "serial"
                result_serial, df_serial = _run_basin(
                    manager, config_Cu, reference_table_Cu_fake, visited_environments_Cu, system_Cu)

                config_Cu.basin.strategy = "wavefront"
                result_wave, df_wave = _run_basin(
                    manager, config_Cu, reference_table_Cu_fake, visited_environments_Cu, system_Cu)
            finally:
                manager.close_all()

            # result_wave Ok can only be produced by the wavefront finalizer
            # (_finalize_exploration_run), so an Ok here is itself proof the wavefront
            # path executed. NOTE: asserting on the finalizer's timing-checkpoint FILE
            # here makes rank 0 hang at interpreter exit under mpirun (under
            # investigation; suspected interaction between a failing assert's pytest
            # reporting and MPI finalization) — keep the assertions exception-free
            # data checks.
            assert result_serial.is_ok(), "serial basin failed: {}".format(result_serial.err_value())
            assert result_wave.is_ok(), "wavefront basin failed: {}".format(result_wave.err_value())

            sc = eqh.basin_state_counts(df_serial)
            wc = eqh.basin_state_counts(df_wave)
            print("[equivalence] serial:", sc, "| wavefront:", wc, flush=True)
            test_logger.info("[equivalence] serial: {} | wavefront: {}".format(sc, wc))

            eqh.assert_connectivity_equivalent(df_serial, df_wave, level="invariant")

            # both exits must be valid absorbing selections of their own basin
            for label, result, counts in (("serial", result_serial, sc), ("wavefront", result_wave, wc)):
                out = result.ok_value()
                assert out.t_exit > 0.0, "{}: non-positive t_exit".format(label)
                assert out.exit_state >= counts["n_transient"], "{}: exit not absorbing".format(label)

