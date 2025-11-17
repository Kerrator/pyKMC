from mpi4py import MPI 
from ...messenger import MpiMessenger, QueueMessenger
from ...lmpi.pool import Manager
from ...lmpi.sessions import MpiApiSession
from ...lmpi.engines import MpiApiEngine
import numpy as np
import threading

class ManagerFactory:
    """
    Responsible for splitting ranks and instantiating Engines, Sessions,
    and returning a configured PoolSessionManager.
    """
    def __init__(self, n_sessions: int, use_rank_0: bool, has_global: bool = True):
        self.world = MPI.COMM_WORLD
        self.world_rank = self.world.Get_rank()
        self.world_size = self.world.Get_size()
        self.n_sessions = n_sessions
        self.use_rank_0 = use_rank_0
        self.has_global = has_global

        if self.use_rank_0 : 
            start_rank = 0 
        else : 
            start_rank = 1

        if self.world_size < n_sessions + start_rank:
            raise ValueError("Not enough MPI ranks to allocate sessions")

        self.available_ranks = list(range(start_rank, self.world_size))  
        self.chunks = self._split_ranks()

    def _split_ranks(self) -> list[list[int]]:
        split_arrays = np.array_split(self.available_ranks, self.n_sessions)
        chunks = [arr.tolist() for arr in split_arrays]
        
        return chunks

    def launch(self) -> Manager | None:

        my_color = MPI.UNDEFINED
        engine_id = None
        for session_id, chunk in enumerate(self.chunks):
            if self.world_rank in chunk:
                my_color = session_id + 1
                engine_id = session_id
                break
    
        # Split communicator 
        engine_comm = self.world.Split(color=my_color, key=self.world_rank)

        sessions = []

        # messenger for each chunk 
        messengers = []
        for session_id, chunk in enumerate(self.chunks):
            if 0 in chunk:
                messengers.append(QueueMessenger())
            else:
                messengers.append(MpiMessenger(comm=self.world))

        # ---  Engine (all rank) ---
        
        if self.has_global : 
            print("global")
            global_comm = self.world.Dup()  # clone de COMM_WORLD, pour isolation
            #global_messenger = MpiMessenger(comm=global_comm)
            #global_color = 0
            #global_comm = self.world.Split(color=global_color, key=self.world_rank)
            global_messenger = MpiMessenger(comm=global_comm)
            #global_messenger = QueueMessenger()
            global_engine = MpiApiEngine(
                messenger=global_messenger, 
                engine_comm=global_comm, 
                engine_id=0
            )
            # --- Lancer global_engine dans un thread ---
            threading.Thread(target=global_engine.start, daemon=True).start()
            #global_engine.start()


        if engine_id is not None:  #  rank in a chunk
            print("yes")
            messenger = messengers[engine_id]
            engine = MpiApiEngine(
                messenger=messenger,
                engine_comm=engine_comm,
                engine_id=engine_id+1
            )
            engine.start()   # bloque ici

        


        # --- Session (On rank 0) ---
        if self.world_rank == 0:
            print("rank0")
            for session_id, chunk in enumerate(self.chunks):
                messenger = messengers[session_id]
                print(f"[Factory] Creating session {session_id} for ranks: {chunk}")
                session = MpiApiSession(
                    messenger=messenger,
                    engine_ranks=chunk,
                    session_id=session_id+1
                )
                sessions.append(session)
            if self.has_global : 
                global_session = MpiApiSession(messenger=global_messenger, engine_ranks=list(range(0, self.world_size)), session_id=0) 
                return Manager(sessions=sessions, global_session=global_session)
            else :
                return Manager(sessions=sessions)