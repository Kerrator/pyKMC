from mpi4py import MPI 
from ...messenger import MpiMessenger, QueueMessenger
from ...lmpi.pool import Manager
from ...lmpi.sessions import MpiApiSession
from ...lmpi.engines import MpiApiEngine
import numpy as np

class ManagerFactory:
    """
    Responsible for splitting ranks and instantiating Engines, Sessions,
    and returning a configured PoolSessionManager.
    """
    def __init__(self, n_sessions: int, use_rank_0: bool):
        self.world = MPI.COMM_WORLD
        self.world_rank = self.world.Get_rank()
        self.world_size = self.world.Get_size()
        self.n_sessions = n_sessions
        self.use_rank_0 = use_rank_0

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
        #my_color = MPI.UNDEFINED
        #engine_id = 0
        #for session_id, chunk in enumerate(self.chunks):
        #    if self.world_rank in chunk:
        #        my_color = session_id + 1
        #        engine_id = session_id
        #        break
        
        ## Split communicator for engines
        #engine_comm = self.world.Split(color=my_color, key=self.world_rank)

        #sessions = []
        ##Create session and corresponding engine
        #for session_id, chunk in enumerate(self.chunks) : 
        #    #Which messenger type 
        #    if 0 in chunk : 
        #        messenger = QueueMessenger()
        #    else : 
        #        messenger = MpiMessenger(comm=self.COMM_WORLD)
        #    
        #    #then Launch engine 

        #    if self.world_rank in chunk : 
        #        engine = MpiApiEngine(messenger=messenger, comm=engine_comm, engine_id=engine_id)
        #        engine.start() 
        #        return 
        #    #then create session 

        #    if self.world_rank == 0 : 
        #        print(f"[Factory] Creating session {session_id} for ranks: {chunk}")
        #        session = MpiApiSession(messenger=messenger,engine_ranks=chunk, session_id=session_id)
        #        sessions.append(session)

        #return Manager(sessions=sessions) 
        #List messenger : 
        #l_messenger = []
        #for chuck in self.chunks : 
        #    if 0 in chuck : 
        #        l_messenger.append(QueueMessenger())
        #    else : 
        #        l_messenger.append(MpiMessenger(comm=MPI.COMM_WORLD))
        #        
            

        ##How session and engine communicate with each other 
        #messenger = MpiMessenger(comm=MPI.COMM_WORLD)

        #if self.world_rank != 0:
        #    engine = MpiApiEngine(messenger=messenger,engine_comm=engine_comm, engine_id=engine_id)
        #    engine.start()  # blocks
        #    return None  # Engine processes stop here

        ## World rank 0 creates and manages sessions
        #sessions = []
        #for session_id, chunk in enumerate(self.chunks):
        #    print(f"[Factory] Creating session {session_id} for ranks: {chunk}")
        #    session = MpiApiSession(messenger=messenger,engine_ranks=chunk, session_id=session_id)
        #    sessions.append(session)

        #return Manager(sessions=sessions)



        # Déterminer si ce rank appartient à un engine
        print("HERE1")
        my_color = MPI.UNDEFINED
        engine_id = None
        for session_id, chunk in enumerate(self.chunks):
            if self.world_rank in chunk:
                my_color = session_id + 1
                engine_id = session_id
                break
    
        # Split communicator (chaque engine reçoit son sous-communicateur)
        engine_comm = self.world.Split(color=my_color, key=self.world_rank)

        # Liste des sessions (rank 0 only)
        sessions = []

        # On crée un messenger par chunk, partagé entre session et engine
        messengers = []
        for session_id, chunk in enumerate(self.chunks):
            if 0 in chunk:
                messengers.append(QueueMessenger())
            else:
                messengers.append(MpiMessenger(comm=self.world))

        print("HEREERERERE2")
        # --- Partie Engine (tous les ranks concernés, y compris 0 si dans chunk) ---
        if engine_id is not None:  # ce rank fait partie d’un chunk
            messenger = messengers[engine_id]
            engine = MpiApiEngine(
                messenger=messenger,
                engine_comm=engine_comm,
                engine_id=engine_id
            )
            print("engine start")
            engine.start()   # bloque ici
            print("engine started", engine_id)

#            return None      # Les ranks engines ne continuent pas

        # --- Partie Session (uniquement rank 0) ---
        if self.world_rank == 0:
            print("houlala")
            for session_id, chunk in enumerate(self.chunks):
                messenger = messengers[session_id]
                print(f"[Factory] Creating session {session_id} for ranks: {chunk}")
                session = MpiApiSession(
                    messenger=messenger,
                    engine_ranks=chunk,
                    session_id=session_id
                )
                sessions.append(session)
            print("HERE3") 
            return Manager(sessions=sessions)