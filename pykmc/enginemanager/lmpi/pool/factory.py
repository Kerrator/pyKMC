from mpi4py import MPI 
from ...lmpi.pool import Manager
from ...lmpi.sessions import MpiApiSession
from ...lmpi.engines import MpiApiEngine
import numpy as np

class ManagerFactory:
    """
    Responsible for splitting ranks and instantiating Engines, Sessions,
    and returning a configured PoolSessionManager.
    """
    def __init__(self, n_sessions: int):
        self.world = MPI.COMM_WORLD
        self.world_rank = self.world.Get_rank()
        self.world_size = self.world.Get_size()
        self.n_sessions = n_sessions

        if self.world_size < n_sessions + 1:
            raise ValueError("Not enough MPI ranks to allocate sessions")

        self.available_ranks = list(range(1, self.world_size))  # reserve rank 0 for sessions
        self.chunks = self._split_ranks()

    def _split_ranks(self) -> list[list[int]]:
        split_arrays = np.array_split(self.available_ranks, self.n_sessions)
        chunks = [arr.tolist() for arr in split_arrays]
        
        return chunks

    def launch(self) -> Manager | None:
        my_color = MPI.UNDEFINED
        engine_id = 0
        for session_id, chunk in enumerate(self.chunks):
            if self.world_rank in chunk:
                my_color = session_id + 1
                engine_id = session_id
                break

        # Split communicator for engines
        engine_comm = self.world.Split(color=my_color, key=self.world_rank)

        if self.world_rank != 0:
            engine = MpiApiEngine(engine_comm=engine_comm, engine_id=engine_id)
            engine.start()  # blocks
            return None  # Engine processes stop here

        # World rank 0 creates and manages sessions
        sessions = []
        for session_id, chunk in enumerate(self.chunks):
            print(f"[Factory] Creating session {session_id} for ranks: {chunk}")
            session = MpiApiSession(engine_ranks=chunk, session_id=session_id)
            sessions.append(session)

        return Manager(sessions=sessions)