from mpi4py import MPI 
import numpy as np
from ..lammps_operations import initialize_parameters, initialize_system
from ...messenger import MpiMessenger

class MpiApiSession : 
    """A class to manage an MPI API session for LAMMPS.
    This class provides an interface to send messages to the Lammps MPI API engine.
    It should live on the rank 0 of the MPI World communicator.
    It should knows on which ranks the LAMMPS engine is running.
    """
    def __init__(self, messenger, engine_ranks, session_id) -> None:
        self.messenger = messenger
        self.engine_ranks = engine_ranks
        self.engine_master_rank = engine_ranks[0]
        self.session_id = session_id
        self._is_alive = False
        self._is_busy = False

        if MPI.COMM_WORLD.Get_rank() != 0:
            raise RuntimeError("MpiApiSession must be used from rank 0.")
        

    def send_message(self, msg: dict,  expect_status: bool = True) -> None:
        """
        Send a message to the engine's master rank.
        """
        self.messenger.send(msg, dest=self.engine_master_rank, tag=2)
        #NOTE : If a lot of message are sent, it will slow down a lot, it is ok if it's just at the initialization, but if 
        #it became a bottleneck, we will need to implement a more efficient way to get status.
        if expect_status:
            self.receive_status()

    def receive_status(self) -> None:
        """
        Receive the status of the engine.
        """
        msg = self.messenger.recv(source=self.engine_master_rank, tag = 0)
        if msg.get("type") == "status":
            value = msg.get("value", {})
            self._is_alive = value.get("alive", False)
            self._is_busy = value.get("busy", False)
        else:
            raise RuntimeError(f"Unexpected message type received: {msg}, expected 'status' but got '{msg.get('type')}'")

    def command(self, cmd: str) -> None: 
        """
        Send a LAMMPS command to the engine.
        """
        print(f"[Session] Sending command: {cmd}")
        self.send_message({"type": "command", "value": cmd})


    def close(self) -> None:
        """
        Instruct the engine to shut down.
        """
        print(f"[Session] Sending close message to engine at rank {self.engine_master_rank}")
        self.send_message({"type": "close"})
        self._is_alive = False

    def is_alive(self) -> bool:
        """
        Check if the engine is alive.
        """
        return self._is_alive   
    
    def is_busy(self) -> bool:
        """
        Check if the engine is busy.
        """
        return self._is_busy
    
    #ACTIONS 
    def initialize_parameters(self) -> None : 
        """ 
        Initialize LAMMPS engine with default parameters
        """
        print(f"[Session] Initializing Lammps parameters")
        self.send_message({"type": "initialize_parameters"})

    def initialize_system(self, system) -> None : 
        """ 
        Initialize Lammps system
        """
        print(f"[Session] Initializing Lammps System")
        self.send_message({"type": "initialize_system", "value": system})
    
    def initialize_potential(self, config) -> None : 
        """ 
        Initialize Lammps potential
        """
        print(f"[Session] Initializing Lammps Potential")
        self.send_message({"type": "initialize_potential", "value": config})
    
    def minimize(self, config) -> None : 
        """ 
        Minimize the system
        """
        print(f"[Session] Minimizing the system")
        self.send_message({"type": "minimize", "value" : config})

    def get_total_energy(self) -> float : 
        """ 
        """
        self._is_busy = True  # Mark the session as busy
        print(f"[Session] Get total energy")
        try : 
            self.send_message({"type": "get_total_energy"})
            msg = self.messenger.recv(source=self.engine_master_rank, tag=1)
            if msg.get("type") == "result":
                return msg["value"]  
            else:
                raise RuntimeError(f"Unexpected message type: {msg}")
        finally:
            self._is_busy = False
    
    def get_positions(self) -> np.ndarray[float] : 
        self._is_busy = True
        print(f"[Session] Get Positions")
        try : 
            self.send_message({"type": "get_positions"})
            msg = self.messenger.recv(source=self.engine_master_rank, tag=1)
            if msg.get("type") == "result" : 
                return msg["value"]
            else : 
                raise RuntimeError(f"Unexpected message type: {msg}")
        finally : 
            self._is_busy = False

    def set_positions(self, positions: np.ndarray[float]) -> None : 
        self._is_busy = True 
        print(f"[Session] Set new positions")
        try : 
            self.send_message({"type": "set_positions", "value": positions})
        finally : 
            self._is_busy = False

    def partn_search(self, config, central_atom_idx) : 
        self._is_busy = True 
        print(f"[Session] Launching pARTn search")
        try : 
            self.send_message({"type": "partn_search", "value": {"config": config, "central_atom_idx": central_atom_idx}})
            msg = self.messenger.recv(source=self.engine_master_rank, tag=1)
            if msg.get("type") == "result" : 
                return msg["value"]
            else : 
                raise RuntimeError(f"Unexpected message type: {msg}")
        finally : 
            self._is_busy = False

    def partn_refine(self, config, central_atom_idx) : 
        self._is_busy = True 
        print(f"[Session] Launching pARTn search")
        try : 
            self.send_message({"type": "partn_refine", "value": {"config": config, "central_atom_idx": central_atom_idx}})
            msg = self.messenger.recv(source=self.engine_master_rank, tag=1)
            if msg.get("type") == "result" : 
                return msg["value"]
            else : 
                raise RuntimeError(f"Unexpected message type: {msg}")
        finally : 
            self._is_busy = False