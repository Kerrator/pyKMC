from pykmc.enginemanager.lmpi.engines import MpiApiEngine
from pykmc.enginemanager.lmpi.sessions import MpiApiSession
from pykmc.enginemanager.lmpi.pool import ManagerFactory, Manager
from pykmc.enginemanager.messenger import MpiMessenger
from pykmc import System, Config
from mpi4py import MPI 
import pytest
from pytest_lazy_fixtures import lf
import os
import time
import threading
import numpy as np

class TestLammpsApiMpiEngine : 

    #def test_send_commands_engine(self) : 
    #    comm = MPI.COMM_WORLD 
    #    rank = comm.Get_rank() 
    #    size = comm.Get_size() 

    #    #test when engine also live on the master session rank or not 
    #    start_rank_engine = 1

    #    if size < 2:
    #        raise RuntimeError("This test requires at least 2 MPI ranks.")  
    
    #    engine_ranks = list(range(start_rank_engine, size)) 

    #    engine_comm = comm.Split(color=1 if rank in engine_ranks else MPI.UNDEFINED, key=rank)

    #    # Start the MPI API Engine only on the specified engine ranks
    #    if rank in engine_ranks:
    #        engine = MpiApiEngine(engine_comm, engine_id=0)
    #        engine.start()

    #    # Master rank sends a message to the engine
    #    if rank == 0:
    #        msg = {"type": "command", "value": "units metal"}
    #        comm.send(msg, dest=engine_ranks[0], tag=1)

    #        msg = {"type": "command", "value": "log flush"}
    #        comm.send(msg, dest=engine_ranks[0], tag=1)

    #        msg = {"type": "close"}
    #        comm.send(msg, dest=engine_ranks[0], tag=1)

    #    time.sleep(4)
    #    # Test if command was sent to lammps : 
    #    if rank == 0 : 
    #        logfile = os.path.join(os.getcwd(), 'lammps.log.0')
    #        with open(logfile) as f : 
    #            log_text = f.read() 
    #        assert 'units metal' in log_text

    def test_send_commends_from_session(self) : 
        comm = MPI.COMM_WORLD 
        rank = comm.Get_rank() 
        size = comm.Get_size() 

        #test when engine also live on the master session rank or not 
        start_rank_engine = 1
        messenger = MpiMessenger(comm=MPI.COMM_WORLD)

        if size < 2:
            raise RuntimeError("This test requires at least 2 MPI ranks.")  
    
        engine_ranks = list(range(start_rank_engine, size)) 

        engine_comm = comm.Split(color=1 if rank in engine_ranks else MPI.UNDEFINED, key=rank)

        # Start the MPI API Engine only on the specified engine ranks
        if rank in engine_ranks:
            engine = MpiApiEngine(messenger=messenger,engine_comm=engine_comm, engine_id=0)
            engine.start()
            return 
        
        # ------------ SESSION CODE (rank 0) ------------
        session = MpiApiSession(messenger=messenger, engine_ranks=engine_ranks, session_id=0)
        session.command("units metal")
        session.command("dimension 3")
        session.command("log flush")

        session.close() 

        time.sleep(4)
        # Test if command was sent to lammps : 
        logfile = os.path.join(os.getcwd(), 'lammps.log.0')
        with open(logfile) as f : 
            log_text = f.read() 
        assert 'units metal' in log_text


    @pytest.mark.parametrize("system, config", [(lf("system_single_type_fcc"), lf("config_system_single_type"))])
    def test_initialize_session(self, system: System, config: Config) : 
        comm = MPI.COMM_WORLD 
        rank = comm.Get_rank() 
        size = comm.Get_size() 

        #test when engine also live on the master session rank or not 
        start_rank_engine = 1
        messenger = MpiMessenger(comm=MPI.COMM_WORLD)

        if size < 2:
            raise RuntimeError("This test requires at least 2 MPI ranks.")  
    
        engine_ranks = list(range(start_rank_engine, size)) 

        engine_comm = comm.Split(color=1 if rank in engine_ranks else MPI.UNDEFINED, key=rank)

        # Start the MPI API Engine only on the specified engine ranks
        if rank in engine_ranks:
            engine = MpiApiEngine(messenger=messenger,engine_comm=engine_comm, engine_id=0)
            engine.start()
            return 
        
        # ------------ SESSION CODE (rank 0) ------------
        session = MpiApiSession(messenger=messenger, engine_ranks=engine_ranks, session_id=0)
        session.initialize_parameters()
        session.initialize_system(system)
        session.initialize_potential(config)
        session.command("log flush")
        session.close() 
        time.sleep(1.5)
        # Test if command was sent to lammps : 
        logfile = os.path.join(os.getcwd(), 'lammps.log.0')
        with open(logfile) as f : 
            log_text = f.read() 
        assert 'units metal' in log_text
        assert 'atom_style atomic' in log_text
        assert 'dimension 3' in log_text
        assert 'boundary p p p' in log_text
        assert 'atom_modify sort 0 0.0' in log_text
        assert 'region box' in log_text 
        assert 'create_box' in log_text 



    @pytest.mark.parametrize("system, config", [(lf("system_single_type_fcc"), lf("config_system_single_type"))])
    def test_minimize(self, system: System, config: Config) : 
        comm = MPI.COMM_WORLD 
        rank = comm.Get_rank() 
        size = comm.Get_size() 

        #test when engine also live on the master session rank or not 
        start_rank_engine = 1
        messenger = MpiMessenger(comm=MPI.COMM_WORLD)

        if size < 2:
            raise RuntimeError("This test requires at least 2 MPI ranks.")  
    
        engine_ranks = list(range(start_rank_engine, size)) 

        engine_comm = comm.Split(color=1 if rank in engine_ranks else MPI.UNDEFINED, key=rank)

        # Start the MPI API Engine only on the specified engine ranks
        if rank in engine_ranks:
            engine = MpiApiEngine(messenger=messenger,engine_comm=engine_comm, engine_id=0)
            engine.start()
            return 
        
        # ------------ SESSION CODE (rank 0) ------------
        session = MpiApiSession(messenger=messenger,engine_ranks=engine_ranks, session_id=0)
        session.initialize_parameters()
        session.initialize_system(system)
        session.initialize_potential(config)
        session.minimize(config)
        session.command("log flush")
        session.close() 
        time.sleep(1.5)
        # Test if command was sent to lammps : 
        logfile = os.path.join(os.getcwd(), 'lammps.log.0')
        with open(logfile) as f : 
            log_text = f.read() 
        assert 'units metal' in log_text
        assert 'atom_style atomic' in log_text
        assert 'dimension 3' in log_text
        assert 'boundary p p p' in log_text
        assert 'atom_modify sort 0 0.0' in log_text
        assert 'region box' in log_text 
        assert 'create_box' in log_text


    @pytest.mark.parametrize("system, config", [(lf("system_single_type_fcc"), lf("config_system_single_type"))])
    def test_get_total_energy(self, system: System, config: Config) : 
        comm = MPI.COMM_WORLD 
        rank = comm.Get_rank() 
        size = comm.Get_size() 

        #test when engine also live on the master session rank or not 
        start_rank_engine = 1
        messenger = MpiMessenger(comm=MPI.COMM_WORLD)

        if size < 2:
            raise RuntimeError("This test requires at least 2 MPI ranks.")  
    
        engine_ranks = list(range(start_rank_engine, size)) 

        engine_comm = comm.Split(color=1 if rank in engine_ranks else MPI.UNDEFINED, key=rank)

        # Start the MPI API Engine only on the specified engine ranks
        if rank in engine_ranks:
            engine = MpiApiEngine(messenger=messenger,engine_comm=engine_comm, engine_id=0)
            engine.start()
            return 
        
        # ------------ SESSION CODE (rank 0) ------------
        session = MpiApiSession(messenger=messenger,engine_ranks=engine_ranks, session_id=0)
        session.initialize_parameters()
        session.initialize_system(system)
        session.initialize_potential(config)
        session.minimize(config)
        e = session.get_total_energy()
        print(e)
        session.command("log flush")
        session.close() 
        assert round(e,3) == round(-1139.1999963495148,3)


    @pytest.mark.parametrize("system, config", [(lf("system_single_type_fcc"), lf("config_system_single_type"))])
    def test_get_positions(self, system: System, config: Config) : 
        comm = MPI.COMM_WORLD 
        rank = comm.Get_rank() 
        size = comm.Get_size() 

        #test when engine also live on the master session rank or not 
        start_rank_engine = 1
        messenger = MpiMessenger(comm=MPI.COMM_WORLD)

        if size < 2:
            raise RuntimeError("This test requires at least 2 MPI ranks.")  
    
        engine_ranks = list(range(start_rank_engine, size)) 

        engine_comm = comm.Split(color=1 if rank in engine_ranks else MPI.UNDEFINED, key=rank)

        # Start the MPI API Engine only on the specified engine ranks
        if rank in engine_ranks:
            engine = MpiApiEngine(messenger=messenger,engine_comm=engine_comm, engine_id=0)
            engine.start()
            return 
        
        # ------------ SESSION CODE (rank 0) ------------
        session = MpiApiSession(messenger=messenger,engine_ranks=engine_ranks, session_id=0)
        session.initialize_parameters()
        session.initialize_system(system)
        session.initialize_potential(config)
        session.minimize(config)
        e = session.get_total_energy()
        print(e)
        e = session.get_total_energy()
        print(e)
        positions = session.get_positions()
        print(positions)
        session.command("log flush")
        session.close() 

    @pytest.mark.parametrize("system, config", [(lf("system_single_type_fcc"), lf("config_system_single_type"))])
    def test_set_positions(self, system: System, config: Config) : 
        comm = MPI.COMM_WORLD 
        rank = comm.Get_rank() 
        size = comm.Get_size() 

        #test when engine also live on the master session rank or not 
        start_rank_engine = 1
        messenger = MpiMessenger(comm=MPI.COMM_WORLD)

        if size < 2:
            raise RuntimeError("This test requires at least 2 MPI ranks.")  
    
        engine_ranks = list(range(start_rank_engine, size)) 

        engine_comm = comm.Split(color=1 if rank in engine_ranks else MPI.UNDEFINED, key=rank)

        # Start the MPI API Engine only on the specified engine ranks
        if rank in engine_ranks:
            engine = MpiApiEngine(messenger=messenger,engine_comm=engine_comm, engine_id=0)
            engine.start()
            return 
        
        # ------------ SESSION CODE (rank 0) ------------
        session = MpiApiSession(messenger=messenger,engine_ranks=engine_ranks, session_id=0)
        session.initialize_parameters()
        session.initialize_system(system)
        session.initialize_potential(config)
        positions = session.get_positions()
        positions[0][0], positions[0][1], positions[0][2] = 0.1, 0.2, 0.3
        session.set_positions(positions)
        positions = session.get_positions()
        session.command("log flush")
        session.close()

        assert positions[0][0] == 0.1
        assert positions[0][1] == 0.2
        assert positions[0][2] == 0.3


    @pytest.mark.parametrize("system, config", [(lf("system_single_type_fcc"), lf("config_system_single_type"))])
    def test_partn_search(self, system: System, config: Config) : 
        comm = MPI.COMM_WORLD 
        rank = comm.Get_rank() 
        size = comm.Get_size() 

        #test when engine also live on the master session rank or not 
        start_rank_engine = 1
        messenger = MpiMessenger(comm=MPI.COMM_WORLD)

        if size < 2:
            raise RuntimeError("This test requires at least 2 MPI ranks.")  
    
        engine_ranks = list(range(start_rank_engine, size)) 

        engine_comm = comm.Split(color=1 if rank in engine_ranks else MPI.UNDEFINED, key=rank)

        # Start the MPI API Engine only on the specified engine ranks
        if rank in engine_ranks:
            engine = MpiApiEngine(messenger=messenger,engine_comm=engine_comm, engine_id=0)
            engine.start()
            return 
        
        # ------------ SESSION CODE (rank 0) ------------
        session = MpiApiSession(messenger=messenger,engine_ranks=engine_ranks, session_id=0)
        session.initialize_parameters()
        session.initialize_system(system)
        session.initialize_potential(config)
        result = session.partn_search(config, 0)
        result = session.partn_refine(config, 0)
        if result.is_ok() : 
            print(result.ok_value())
        else : 
            print(result.err_value())
        session.command("log flush")
        session.close()


    @pytest.mark.parametrize("system, config", [(lf("system_single_type_fcc"), lf("config_system_single_type"))])
    def test_initialize_manager(self, system: System, config: Config)  : 
        factory = ManagerFactory(n_sessions=config.control.n_sessions, use_rank_0=config.control.engine_use_rank_0)
        manager = factory.launch()

        if manager is None:
            return  # Engine processes stop here
        # ------------ SESSION CODE (rank 0) ------------a
        print("HERERER")
        manager.initialize_sessions(config, system)
        manager.close_all()

    @pytest.mark.parametrize("system, config", [(lf("system_single_type_fcc"), lf("config_system_single_type"))])
    def test_minimize_manager(self, system: System, config: Config)  : 
        factory = ManagerFactory(n_sessions=config.control.n_sessions, use_rank_0=config.control.engine_use_rank_0)
        manager = factory.launch()
        if manager is None:
            return  # Engine processes stop here
        # ------------ SESSION CODE (rank 0) ------------
        manager.initialize_sessions(config, system)
        print("done")
        f = manager.minimize(config)
        re = f.result()
        manager.close_all()


    @pytest.mark.parametrize("system, config", [(lf("system_single_type_fcc"), lf("config_system_single_type"))])
    def test_partn_manager(self, system: System, config: Config)  : 
        factory = ManagerFactory(n_sessions=config.control.n_sessions, use_rank_0=config.control.engine_use_rank_0)
        manager = factory.launch()
        if manager is None:
            return  # Engine processes stop here
        # ------------ SESSION CODE (rank 0) ------------
        manager.initialize_sessions(config, system)
        idx = 20*[0]
        futures = manager.partn_refine(config, idx)
        re = [f.result() for f in futures] 
        manager.close_all()


    @pytest.mark.parametrize("system, config", [(lf("system_single_type_fcc"), lf("config_system_single_type"))])
    def test_minimize_with_results_manager(self, system: System, config: Config)  :
        factory = ManagerFactory(n_sessions=config.control.n_sessions, use_rank_0=config.control.engine_use_rank_0)
        manager = factory.launch()
        if manager is None:
            return  # Engine processes stop here
        # ------------ SESSION CODE (rank 0) ------------
        manager.initialize_sessions(config, system)
        futures = manager.minimize_with_results(config)
        positions, total_energy = futures.result()
        print(positions)
        print(total_energy)
        manager.close_all()

    @pytest.mark.parametrize("system, config", [(lf("system_single_type_fcc"), lf("config_system_single_type"))])
    def test_compute_forces_and_dynamical_matrix_manager(self, system: System, config: Config) -> None:
        """Forces + eskm Hessian round-trip through the session pool (local mode).

        Uses the n_sessions=7 / engine_use_rank_0=False layout (mpirun -n 8) --
        the rank-0-as-engine mode of the fixture config has a known deadlock on
        this branch lineage (fixed separately on develop-refactoring, PR #70).
        """
        if MPI.COMM_WORLD.Get_size() < 8:
            pytest.skip("needs mpirun -n 8 (n_sessions=7, engine_use_rank_0=False)")
        config.control.n_sessions = 7
        config.control.engine_use_rank_0 = False
        factory = ManagerFactory(n_sessions=config.control.n_sessions, use_rank_0=config.control.engine_use_rank_0)
        manager = factory.launch()
        if manager is None:
            return  # Engine processes stop here
        # ------------ SESSION CODE (rank 0) ------------
        manager.initialize_sessions(config, system)
        # jobs run on the LOCAL session pool; engines end initialization in global mode
        manager.use_local()

        f = manager.compute_forces(positions=system.positions.copy())
        forces = f.result()
        assert forces.shape == (system.positions.shape[0], 3)
        assert np.isfinite(forces).all()

        free = [0, 1]
        g = manager.compute_dynamical_matrix(
            positions=system.positions.copy(), free_indices=free, dx=0.01
        )
        hessian = g.result()
        assert hessian.shape == (3 * len(free), 3 * len(free))
        assert np.isfinite(hessian).all()
        assert np.allclose(hessian, hessian.T)  # symmetrized by the op
        manager.close_all()

    @pytest.mark.parametrize("system, config", [(lf("system_single_type_fcc"), lf("config_system_single_type"))])
    def test_compute_event_prefactors_multirank_session(self, system: System, config: Config) -> None:
        """HTST prefactor on a MULTI-RANK session must not deadlock.

        Regression for the multi-rank ``dynamical_matrix`` hang: with >1 engine
        rank per session the LAMMPS ``dynamical_matrix`` collective deadlocked, so
        every HTST prefactor job stalled rank 0 on the never-returning future.
        Single-rank sessions (n_sessions == n_engine_ranks, e.g. n_sessions=7 on
        ``mpirun -n 8``) were the ONLY tested layout. Here n_sessions=1 forces the
        whole engine_comm into ONE multi-rank session, the production-hang case.

        Run with a wall guard so the deadlock surfaces as a failure, not a hang:
            timeout 180 mpirun -n 3 python -m pytest \
                tests/test_lammps_engine_api_mpi.py -k multirank_session -s
        """
        if MPI.COMM_WORLD.Get_size() < 3:
            pytest.skip("needs mpirun -n >= 3 for a multi-rank session (n_sessions=1)")
        config.control.n_sessions = 1
        config.control.engine_use_rank_0 = False
        config.rateconstant.style = "htst"  # exercise the Vineyard Hessian path
        factory = ManagerFactory(n_sessions=1, use_rank_0=False)
        manager = factory.launch()
        if manager is None:
            return  # engine ranks block in their service loop
        # ------------ SESSION CODE (rank 0) ------------
        manager.initialize_sessions(config, system)
        manager.use_local()
        pos = system.positions.copy()
        payload = {
            "central_atom_idx": 0,
            "min1_positions": pos,
            "saddle_positions": pos,
            "min2_positions": pos,
            "types": list(system.types),
            "cell": system.cell,
        }
        futures = manager.compute_event_prefactors(config, [payload])
        results = [f.result() for f in futures]  # <-- deadlocks here before the fix
        manager.close_all()
        assert len(results) == 1
        # The prefactor must come back (real nu0 or a graceful k0 fallback), never hang.
        assert hasattr(results[0], "ok_forward")

    @pytest.mark.parametrize("system, config", [(lf("system_single_type_fcc"), lf("config_system_single_type"))])
    def test_engine_error_during_global_minimize_does_not_hang(self, system: System, config: Config) -> None:
        """An engine-side error during a global pool op must fail cleanly, not hang.

        Regression for the recycle-reconstruction pool hang (see the repo-root
        HANDOFF_recycle_pool_hang.md). Root cause, confirmed by stack-sampling the
        hung ranks: a *per-rank* LAMMPS error (``error->one`` -- e.g. "Non-numeric
        atom coords" from a poisoned recycle geometry) fires on only a SUBSET of the
        global engine ranks. The erroring rank unwinds to the Python finally-barrier
        in ``_handle_message`` while the surviving rank stays trapped inside
        liblammps's own collective (an ``MPI_Sendrecv`` inside ``minimize``) -- so the
        two engine ranks deadlock, busy-spin at ~100% CPU, and rank 0 blocks in
        ``receive_status``. Because the survivor never returns from ``lammps_command``,
        no *post-op* Python-level resync can recover it.

        The fix detects the poisoned geometry COLLECTIVELY before LAMMPS sees it:
        every engine rank validates the (identically broadcast) positions and raises
        the SAME error together, so the failure travels the symmetric error path --
        it surfaces as a RuntimeError on rank 0 and the pool stays shut-downable.

        Contract: (a) the engine error surfaces as a RuntimeError on rank 0, and
        (b) ``close_all()`` returns. Reaching the end of this test (rather than
        stalling) is the regression check. Runs under any ``mpirun -n >= 3``; with
        fewer ranks it skips.
        """
        if MPI.COMM_WORLD.Get_size() < 3:
            pytest.skip("needs mpirun -n >= 3 for a multi-rank global engine")
        config.control.n_sessions = 1
        config.control.engine_use_rank_0 = False
        factory = ManagerFactory(n_sessions=1, use_rank_0=False)
        manager = factory.launch()
        if manager is None:
            return  # engine ranks block in their service loop
        # ------------ SESSION CODE (rank 0) ------------
        try:
            manager.initialize_sessions(config, system)
            manager.use_global()

            # Poisoned geometry: a non-finite coordinate. Fed to a multi-rank global
            # minimize this provokes a per-rank LAMMPS error (error->one) that strands
            # the surviving engine rank inside liblammps -- the documented pool hang.
            # The collective pre-validation must turn this into a clean symmetric
            # failure instead.
            bad = system.positions.copy().astype(float)
            bad[1] = np.nan

            raised = False
            try:
                manager.global_minimize_with_results(
                    config, positions=bad, types=list(system.types)
                )
            except RuntimeError:
                raised = True  # the engine error reached rank 0 -- good
            assert raised, (
                "engine error must surface as a RuntimeError on rank 0, not hang"
            )
        finally:
            # The pool must remain shut-downable. close_all() lives in a finally so a
            # failed assertion (or any early rank-0 exit) can never strand the engine
            # ranks busy-spinning in run_engine_loop -- the second defect behind the hang.
            manager.close_all()

    @pytest.mark.parametrize("system, config", [(lf("system_single_type_fcc"), lf("config_system_single_type"))])
    @pytest.mark.parametrize("chunk_kind", ["single_rank", "multi_rank"])
    def test_close_all_after_error_with_multiple_sessions_does_not_hang(
        self, chunk_kind: str, system: System, config: Config
    ) -> None:
        """``close_all()`` must return for a MULTI-session pool torn down in global mode.

        Successor to ``test_engine_error_during_global_minimize_does_not_hang`` (see
        the repo-root HANDOFF_close_all_teardown_hang.md). The per-op recycle hang is
        already fixed: a global-minimize error is handled gracefully and the engines
        return to their service loop. But the *teardown itself* still hangs whenever
        the pool is closed while in GLOBAL mode -- which is exactly the recycle
        "All event reconstructions failed" path (``kmc.py``: ``use_global()`` ->
        reconstruction fails -> ``_close()`` -> ``close_all()``).

        Mechanism: the global ``engine_comm`` spans EVERY engine rank, so
        ``global_session.close()`` broadcasts a shutdown to all engines and exits
        their run loops at once. In global mode only the global *master* rank emits a
        status (consumed by the first local session's close); the next local session
        then blocks in ``receive_status()`` forever, waiting on an engine that has
        already gone. The single-session reproducer cannot catch this -- it needs
        ``n_sessions >= 2`` so there is a second local session to strand.

        Both chunk topologies are exercised: ``single_rank`` (one engine rank per
        session, the ``mpirun -n 3`` minimal case) and ``multi_rank`` (>=2 ranks per
        session, the production benchmark shape -- needs ``mpirun -n >= 5``) so the
        fix's per-session bcast/barrier teardown is covered for multi-rank engines
        too.

        A thread wall-guard turns the hang into a clean assertion failure (RED)
        instead of stalling the whole MPI job; after the fix ``close_all()`` returns
        (GREEN).

            timeout 180 mpirun -n 5 python -m pytest \
                tests/test_lammps_engine_api_mpi.py -k close_all_after_error -s
        """
        n_engine_ranks = MPI.COMM_WORLD.Get_size() - 1  # engines on ranks 1..size-1
        if chunk_kind == "single_rank":
            # One rank per session; need >=2 engine ranks so there are >=2 sessions.
            if n_engine_ranks < 2:
                pytest.skip("needs mpirun -n >= 3 for >=2 single-rank sessions")
            n_sessions = n_engine_ranks
        else:  # multi_rank: >=2 sessions, each owning >=2 engine ranks.
            if n_engine_ranks < 4:
                pytest.skip("needs mpirun -n >= 5 for >=2 multi-rank sessions")
            n_sessions = 2
        config.control.n_sessions = n_sessions
        config.control.engine_use_rank_0 = False
        factory = ManagerFactory(n_sessions=n_sessions, use_rank_0=False)
        manager = factory.launch()
        if manager is None:
            return  # engine ranks block in their service loop
        # ------------ SESSION CODE (rank 0) ------------
        manager.initialize_sessions(config, system)
        # Mirror the recycle reconstruction path: switch to the global pool, take a
        # handled engine error, and STAY in global mode for teardown.
        manager.use_global()
        bad = system.positions.copy().astype(float)
        bad[1] = np.nan
        with pytest.raises(RuntimeError):
            manager.global_minimize_with_results(
                config, positions=bad, types=list(system.types)
            )

        # close_all() under a wall guard: a hang here is the bug under test. Running
        # it in a daemon thread converts the stall into a deterministic assertion
        # failure instead of blocking the whole MPI job until the outer timeout.
        outcome: dict = {}

        def _teardown() -> None:
            try:
                manager.close_all()
                outcome["ok"] = True
            except BaseException as exc:  # surface, never swallow
                outcome["error"] = exc

        guard_s = 60.0
        worker = threading.Thread(target=_teardown, daemon=True)
        worker.start()
        worker.join(timeout=guard_s)
        # Branch on the thread's liveness, not on ``outcome`` alone: a thread that has
        # terminated has fully written ``outcome`` (happens-before), so this is free of
        # the join-deadline read race.
        assert not worker.is_alive(), (
            f"close_all() did not return within {guard_s}s -- the multi-session "
            f"teardown hang ({chunk_kind} chunks; HANDOFF_close_all_teardown_hang.md)"
        )
        if "error" in outcome:
            raise outcome["error"]
        assert outcome.get("ok")

        


