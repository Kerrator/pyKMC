from pykmc.enginemanager.lmpi.engines.mpi_api_engine import MpiApiEngine
from pykmc.enginemanager.lmpi.sessions.mpi_api_sessions import MpiApiSession
from pykmc import System
from mpi4py import MPI 
import pytest
from pytest_lazy_fixtures import lf
import os
import time

class TestLammpsApiMpiEngine : 

    def test_send_commands_engine(self) : 
        comm = MPI.COMM_WORLD 
        rank = comm.Get_rank() 
        size = comm.Get_size() 

        #test when engine also leave on the master session rank or not 
        start_rank_engine = 1

        if size < 2:
            raise RuntimeError("This test requires at least 2 MPI ranks.")  
    
        engine_ranks = list(range(start_rank_engine, size)) 

        engine_comm = comm.Split(color=1 if rank in engine_ranks else MPI.UNDEFINED, key=rank)

        # Start the MPI API Engine only on the specified engine ranks
        if rank in engine_ranks:
            engine = MpiApiEngine(engine_comm, engine_id=0)
            engine.start()

        # Master rank sends a message to the engine
        if rank == 0:
            msg = {"type": "command", "value": "units metal"}
            comm.send(msg, dest=engine_ranks[0], tag=1)

            msg = {"type": "command", "value": "log flush"}
            comm.send(msg, dest=engine_ranks[0], tag=1)

            msg = {"type": "close"}
            comm.send(msg, dest=engine_ranks[0], tag=1)

        # Test if command was sent to lammps : 
        if rank == 0 : 
            logfile = os.path.join(os.getcwd(), 'lammps.log.0')
            with open(logfile) as f : 
                log_text = f.read() 
            assert 'units metal' in log_text

    def test_send_commends_from_session(self) : 
        comm = MPI.COMM_WORLD 
        rank = comm.Get_rank() 
        size = comm.Get_size() 

        #test when engine also leave on the master session rank or not 
        start_rank_engine = 1

        if size < 2:
            raise RuntimeError("This test requires at least 2 MPI ranks.")  
    
        engine_ranks = list(range(start_rank_engine, size)) 

        engine_comm = comm.Split(color=1 if rank in engine_ranks else MPI.UNDEFINED, key=rank)

        # Start the MPI API Engine only on the specified engine ranks
        if rank in engine_ranks:
            engine = MpiApiEngine(engine_comm, engine_id=0)
            engine.start()
            return 
        
        # ------------ SESSION CODE (rank 0) ------------
        session = MpiApiSession(engine_ranks=engine_ranks, session_id=0)
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


    @pytest.mark.parametrize("system", [lf("system_single_type_fcc")])
    def test_initialize_session(self, system: System) : 
        comm = MPI.COMM_WORLD 
        rank = comm.Get_rank() 
        size = comm.Get_size() 

        #test when engine also leave on the master session rank or not 
        start_rank_engine = 1

        if size < 2:
            raise RuntimeError("This test requires at least 2 MPI ranks.")  
    
        engine_ranks = list(range(start_rank_engine, size)) 

        engine_comm = comm.Split(color=1 if rank in engine_ranks else MPI.UNDEFINED, key=rank)

        # Start the MPI API Engine only on the specified engine ranks
        if rank in engine_ranks:
            engine = MpiApiEngine(engine_comm, engine_id=0)
            engine.start()
            return 
        
        # ------------ SESSION CODE (rank 0) ------------
        session = MpiApiSession(engine_ranks=engine_ranks, session_id=0)
        session.initialize_parameters()
        session.initialize_system(system)
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


        


