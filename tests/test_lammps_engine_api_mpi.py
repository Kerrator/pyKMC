from pykmc.enginemanager.lmpi.engines.mpi_api_engine import MpiApiEngine
from mpi4py import MPI 
import os
from pathlib import Path

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

            msg = {"type": "close"}
            comm.send(msg, dest=engine_ranks[0], tag=1)

        # Test if command was sent to lammps : 
        if rank == 0 : 
            logfile = Path(os.getcwd())/'lammps.log.0' 
            log_text = logfile.read_text()
            assert 'units metal' in log_text

        


        


