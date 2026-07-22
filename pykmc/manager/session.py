from __future__ import annotations
from mpi4py import MPI 
from typing import Any

class Session:
    """Proxy for communicating with a Worker rank from rank 0.

    MPI tag convention:
        tag 2 — command : Session → Worker  (operation request)
        tag 0 — status  : Worker  → Session (completion + error flag)
        tag 1 — result  : Worker  → Session (return value, only if has_result)
    """

    def __init__(self, engine_master_rank: int, session_id: int = 0, world_comm: "MPI.COMM"|None = None) -> None:

        self.engine_master_rank = engine_master_rank
        self.session_id = session_id
        self.world_comm = world_comm or MPI.COMM_WORLD
        self._is_busy = False

        if self.world_comm.Get_rank() != 0 :
            raise RuntimeError("Session must be used from rank 0.")
        
    def use_local(self) -> None:
        """Switch worker to local mode (fire-and-forget)."""
        self.world_comm.send({"type": "use_local"}, dest=self.engine_master_rank, tag=2)

    def use_group(self) -> None:
        """Switch worker to group mode (fire-and-forget). Workers without a group_comm ignore this."""
        self.world_comm.send({"type": "use_group"}, dest=self.engine_master_rank, tag=2)

    def use_global(self) -> None:
        """Switch worker to global mode (fire-and-forget)."""
        self.world_comm.send({"type": "use_global"}, dest=self.engine_master_rank, tag=2)

    def shutdown(self) -> None:
        """Shut down the worker (fire-and-forget, no status reply)."""
        self.world_comm.send({"type": "shutdown"}, dest=self.engine_master_rank, tag=2)

    def call(self, op_name: str, **kwargs) -> Any : 
        """Send an operation to the worker and optionally retrieve a result."""

        self._is_busy = True 

        try :
            msg = {"type": op_name}
            if kwargs:
                msg["value"] = kwargs
            self.world_comm.send(msg, dest = self.engine_master_rank, tag=2)
            has_result = self._recv_status()
            if has_result :
                return self._recv_result()
        finally : 
            self._is_busy = False 

    def _recv_status(self) -> bool:
        msg = self.world_comm.recv(source=self.engine_master_rank, tag=0)
        if msg.get("type") != "status":
            raise RuntimeError(f"Expected 'status', got '{msg.get('type')}'")
        value = msg.get("value", {})
        error = value.get("error")
        if error:
            raise RuntimeError(error)
        return value.get("has_result", False)

    def _recv_result(self) -> Any : 
        msg = self.world_comm.recv(source=self.engine_master_rank, tag = 1)
        if msg.get("type") != "result" : 
            raise RuntimeError(f"Expected 'result' got '{msg.get('type')}'")
        return msg["value"]
    
    def is_busy(self) -> bool :
        return self._is_busy

    
