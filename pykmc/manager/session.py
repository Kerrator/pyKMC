from __future__ import annotations
from mpi4py import MPI
from typing import Any


class Session:
    """Proxy for communicating with a Worker rank from rank 0.

    MPI tag convention:
        _TAG_CMD    (2) — command : Session → Worker  (operation request)
        _TAG_STATUS (0) — status  : Worker  → Session (completion + error flag)
        _TAG_RESULT (1) — result  : Worker  → Session (return value, only if has_result)
    """

    _TAG_CMD = 2
    _TAG_STATUS = 0
    _TAG_RESULT = 1

    def __init__(
        self,
        engine_master_rank: int,
        session_id: int = 0,
        world_comm: "MPI.COMM" | None = None,
    ) -> None:
        world_comm = world_comm or MPI.COMM_WORLD
        if world_comm.Get_rank() != 0:
            raise RuntimeError("Session must be used from rank 0.")

        self.engine_master_rank = engine_master_rank
        self.session_id = session_id
        self.world_comm = world_comm

    def _send_command(self, op_type: str) -> None:
        self.world_comm.send(
            {"type": op_type}, dest=self.engine_master_rank, tag=self._TAG_CMD
        )

    def use_local(self) -> None:
        self._send_command("use_local")

    def use_group(self) -> None:
        self._send_command("use_group")

    def use_global(self) -> None:
        self._send_command("use_global")

    def shutdown(self) -> None:
        self._send_command("shutdown")

    def call(self, op_name: str, **kwargs) -> Any:
        """Send an operation to the worker and optionally retrieve a result."""
        msg = {"type": op_name}
        if kwargs:
            msg["value"] = kwargs
        self.world_comm.send(msg, dest=self.engine_master_rank, tag=self._TAG_CMD)
        has_result = self._recv_status()
        if has_result:
            return self._recv_result()

    def _recv_status(self) -> bool:
        msg = self.world_comm.recv(source=self.engine_master_rank, tag=self._TAG_STATUS)
        if msg.get("type") != "status":
            raise RuntimeError(f"Expected 'status', got '{msg.get('type')}'")
        value = msg.get("value", {})
        error = value.get("error")
        if error:
            raise RuntimeError(error)
        return value.get("has_result", False)

    def _recv_result(self) -> Any:
        msg = self.world_comm.recv(source=self.engine_master_rank, tag=self._TAG_RESULT)
        if msg.get("type") != "result":
            raise RuntimeError(f"Expected 'result' got '{msg.get('type')}'")
        return msg["value"]
