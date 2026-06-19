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
        """An engine-side LAMMPS error during a pool op must fail cleanly, not hang.

        Regression for the recycle-reconstruction pool hang (see the repo-root
        HANDOFF_recycle_pool_hang.md). In a 10-step recycle benchmark run, a recycle
        reconstruction fed ``global_minimize_with_results`` an unstable geometry,
        LAMMPS raised ``Lost atoms``, and although the error surfaced to rank 0 as a
        RuntimeError, the engine ranks were left busy-spinning in their service loop
        (~100% CPU) instead of the pool failing/closing cleanly -- the whole job hung
        for hours until killed.

        This drives the SAME global-minimize path with a deliberately degenerate
        geometry (two atoms collapsed ~1e-4 A apart, so the EAM repulsion blows the
        minimize up into a LAMMPS error). The contract: the error must (a) surface as
        a RuntimeError on rank 0 and (b) leave the pool shut-downable -- reaching the
        end of this test rather than stalling is the regression check.

        Gated behind PYKMC_REPRODUCE_RECYCLE_HANG (it hangs while the bug is open).
        Run with a wall guard so the hang surfaces as a failure, not an infinite stall:
            PYKMC_REPRODUCE_RECYCLE_HANG=1 timeout 130 mpirun \
                -x PYKMC_REPRODUCE_RECYCLE_HANG -n 3 --oversubscribe python -m pytest \
                tests/test_lammps_engine_api_mpi.py \
                -k engine_error_during_global_minimize_does_not_hang -s
        Exit 124 (timeout) == bug reproduced; exit 0 == fixed.
        """
        # This reproduces an OPEN bug that HANGS the pool, so it must NOT run in the
        # normal suite (it would stall any `mpirun -n>=3 pytest` run). Gate it behind
        # an env var until the bug is fixed; then drop this gate so it becomes a live
        # regression guard. See HANDOFF_recycle_pool_hang.md.
        if not os.environ.get("PYKMC_REPRODUCE_RECYCLE_HANG"):
            pytest.skip("open-bug reproducer that hangs the pool; set "
                        "PYKMC_REPRODUCE_RECYCLE_HANG=1 under `timeout ... mpirun -n>=3` "
                        "to run it -- see HANDOFF_recycle_pool_hang.md")
        if MPI.COMM_WORLD.Get_size() < 3:
            pytest.skip("needs mpirun -n >= 3 for a multi-rank global engine")
        config.control.n_sessions = 1
        config.control.engine_use_rank_0 = False
        factory = ManagerFactory(n_sessions=1, use_rank_0=False)
        manager = factory.launch()
        if manager is None:
            return  # engine ranks block in their service loop
        # ------------ SESSION CODE (rank 0) ------------
        manager.initialize_sessions(config, system)
        manager.use_global()

        # Degenerate geometry: collapse atom 1 onto atom 0. The EAM repulsion is
        # enormous, so the global minimize drives atoms out of the box -> LAMMPS
        # "Lost atoms" -> engine-side RuntimeError (the production trigger).
        bad = system.positions.copy()
        bad[1] = bad[0] + 1.0e-4

        raised = False
        try:
            manager.global_minimize_with_results(
                config, positions=bad, types=list(system.types)
            )
        except RuntimeError:
            raised = True  # the engine error reached rank 0 -- good
        assert raised, "engine LAMMPS error must surface as a RuntimeError on rank 0"

        # The pool must remain shut-downable. Before the fix the engine ranks spin
        # forever here (or never let this line be reached) -> the run_engine_loop
        # never sees the close, and the test stalls until the wall guard kills it.
        manager.close_all()

        


