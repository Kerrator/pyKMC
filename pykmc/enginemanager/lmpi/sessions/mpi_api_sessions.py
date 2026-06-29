from mpi4py import MPI 
import numpy as np
from ...messenger import MpiMessenger
from threading import RLock  
from functools import wraps
from collections.abc import Callable
import time

_POLL_INTERVAL_S = 0.05


class EngineOpTimeout(Exception):
    """Rank 0 waited past the engine-op deadline for a reply -- the pool is desynced."""


def _await_reply(
    poll: "Callable[[], tuple[bool, object]]",
    timeout_s: float,
    *,
    monotonic: "Callable[[], float]" = time.monotonic,
    sleep: "Callable[[float], object]" = time.sleep,
) -> object:
    """Poll ``poll() -> (done, value)`` until done; raise EngineOpTimeout past the deadline.

    Backs the opt-in engine-op wall guard: a bounded wait so a desynced pool (an engine
    rank stuck below the Python layer that will never reply -- see
    HANDOFF_recycle_pool_hang.md) fails fast instead of stalling for the full per-run
    timeout. ``monotonic``/``sleep`` are injectable for deterministic testing.
    """
    deadline = monotonic() + timeout_s
    while True:
        done, value = poll()
        if done:
            return value
        if monotonic() >= deadline:
            raise EngineOpTimeout(f"no engine reply within {timeout_s}s")
        sleep(_POLL_INTERVAL_S)


#TODO more general way to deal with operations
#TODO : commented print should be log depending of the verbosity but need to thing of how we modify log before (also loggers are 
#initiated in kmc, after the initialization of manager ...))

def session_locked(method):
    @wraps(method)
    def wrapper(self, *args, **kwargs):
        with self._lock:
            self._is_busy = True
            try:
                return method(self, *args, **kwargs)
            finally:
                self._is_busy = False
    return wrapper

class MpiApiSession :
    """A class to manage an MPI API session for LAMMPS.
    This class provides an interface to send messages to the Lammps MPI API engine.
    It should live on the rank 0 of the MPI World communicator.
    It should knows on which ranks the LAMMPS engine is running.
    """

    def __init__(self, messenger: MpiMessenger, engine_ranks, session_id) -> None:
        self.messenger = messenger
        self.engine_ranks = engine_ranks
        self.engine_master_rank = engine_ranks[0]
        self.session_id = session_id
        self._is_alive = False
        self._is_busy = False
        self._lock = RLock()
        # Opt-in engine-op wall guard (seconds); None -> plain blocking recv.
        # Set by Manager.initialize_sessions from config.control.engine_op_timeout_s.
        self._op_timeout = None

        if MPI.COMM_WORLD.Get_rank() != 0:
            raise RuntimeError("MpiApiSession must be used from rank 0.")
        

    def send_message(self, msg: dict,  expect_status: bool = True) -> None:
        """Send a message to the engine's master rank.
        """
        self.messenger.send(msg, dest=self.engine_master_rank, tag=2)
        #NOTE : If a lot of message are sent, it will slow down a lot, it is ok if it's just at the initialization, but if 
        #it became a bottleneck, we will need to implement a more efficient way to get status.
        if expect_status:
            self.receive_status()

    def _recv(self, source: int, tag: int) -> object:
        """Receive a message, optionally bounded by the engine-op wall guard.

        With ``self._op_timeout`` unset (default) this is a plain blocking recv --
        byte-identical to the previous behaviour. When a timeout is configured (and the
        transport is MPI), rank 0 polls for the reply and, on the deadline, aborts the
        whole MPI job: a desynced pool cannot be recovered (an engine rank is stuck
        below the Python layer and will never reply), so failing fast beats a multi-hour
        stall. See HANDOFF_recycle_pool_hang.md.
        """
        if self._op_timeout is None or not isinstance(self.messenger, MpiMessenger):
            return self.messenger.recv(source=source, tag=tag)
        request = self.messenger.comm.irecv(source=source, tag=tag)
        try:
            return _await_reply(request.test, self._op_timeout)
        except EngineOpTimeout:
            try:
                request.Cancel()
            except Exception:
                pass
            print(
                f"[Session] No engine reply within {self._op_timeout}s "
                f"(source={source}, tag={tag}); the pool is desynced -- aborting the MPI job."
            )
            MPI.COMM_WORLD.Abort(1)

    def receive_status(self) -> None:
        """Receive the status of the engine.
        """
        msg = self._recv(source=self.engine_master_rank, tag=0)
        if msg.get("type") == "status":
            value = msg.get("value", {})
            self._is_alive = value.get("alive", False)
            self._is_busy = value.get("busy", False)
        else:
            raise RuntimeError(f"Unexpected message type received: {msg}, expected 'status' but got '{msg.get('type')}'")

    #@session_locked
    def command(self, cmd: str) -> None:
        """Send a LAMMPS command to the engine.
        """
        #print(f"[Session] Sending command: {cmd}")
        self.send_message({"type": "command", "value": cmd})

    #@session_locked
    def use_local(self) -> None:
        """Instruct the engine to use local pool
        """
        #print(f"[Session {self.session_id}] sending 'use local' to rank {self.engine_master_rank}")
        self.send_message({"type": "use_local"})

    def use_global(self) -> None:
        """Instruct the engine to use global pool
        """
        #print(f"[Session {self.session_id}] sending 'use global' to rank {self.engine_master_rank}")
        self.send_message({"type": "use_global"})

    #@session_locked
    def close(self, wait_status: bool = False) -> None:
        """Instruct the engine to shut down.

        Parameters
        ----------
        wait_status : bool, default False
            If True, wait for the engine to send a status message (for normal sessions).
            If False, just send the close message (for global / long-running engines).

        """
        print(f"[Session] Sending close message to engine at rank {self.engine_master_rank}")
        self.send_message({"type": "close"}, expect_status=wait_status)
        self._is_alive = False

    def is_alive(self) -> bool:
        """Check if the engine is alive.
        """
        return self._is_alive   
    
    def is_busy(self) -> bool:
        """Check if the engine is busy.
        """
        return self._is_busy
    
    #ACTIONS 
    #@session_locked
    def initialize_parameters(self) -> None :
        """
        Initialize LAMMPS engine with default parameters
        """
        #print(f"[Session {self.session_id}] Initializing Lammps parameters")
        self.send_message({"type": "initialize_parameters"})

    #@session_locked
    def initialize_system(self, system) -> None :
        """ 
        Initialize Lammps system
        """
        #print(f"[Session {self.session_id}] Initializing Lammps System")
        self.send_message({"type": "initialize_system", "value": system})
    
    #@session_locked
    def initialize_potential(self, config) -> None :
        """ 
        Initialize Lammps potential
        """
        #print(f"[Session {self.session_id}] Initializing Lammps Potential")
        self.send_message({"type": "initialize_potential", "value": config})
    
    #@session_locked
    def minimize(self, config, positions=None) -> None :
        """ 
        Minimize the system
        """
        #print(f"[Session] Minimizing the system")
        self.send_message({"type": "minimize", "value" : {"config": config, "positions": positions}})

    #@session_locked
    def get_positions(self) -> np.ndarray[float] :
        self._is_busy = True
        #print(f"[Session] Get Positions")
        try :
            self.send_message({"type": "get_positions"})
            return self._receive_result_or_error()
        finally :
            self._is_busy = False

    #@session_locked
    def set_positions(self, positions: np.ndarray[float]) -> None :
        self._is_busy = True 
        #print(f"[Session] Set new positions")
        try : 
            self.send_message({"type": "set_positions", "value": positions})
        finally : 
            self._is_busy = False

    #@session_locked
    def minimize_with_results(self, config, positions=None, types=None) :
        """Minimize and return the minimized positions and the total energy.
        """
        self._is_busy = True
        #print(f"[Session n°{self.session_id}] Minimizing and get positions and total energy")
        try :
            self.send_message({"type": "minimize_with_results", "value": {"config": config, "positions": positions, "types": types}})
            return self._receive_result_or_error()
        finally :
            self._is_busy = False

    def _receive_result_or_error(self) -> object:
        """Receive a tag-1 reply, raising on explicit remote failures."""
        msg = self._recv(source=self.engine_master_rank, tag=1)
        msg_type = msg.get("type")
        if msg_type == "result":
            return msg.get("value")
        if msg_type == "error":
            value = msg.get("value", {})
            operation = value.get("operation", "unknown operation")
            error_type = value.get("error_type", "RuntimeError")
            message = value.get("message", "Unknown remote error")
            raise RuntimeError(
                f"Remote {operation} failed on engine rank {self.engine_master_rank} "
                f"({error_type}): {message}"
            )
        raise RuntimeError(f"Unexpected message type: {msg}")

    def basin_reconstruct(self, **kwargs: object) -> object:
        """Send a basin reconstruction task to the engine and return the result."""
        self._is_busy = True
        try:
            self.send_message({"type": "basin_reconstruct", "value": kwargs})
            return self._receive_result_or_error()
        finally:
            self._is_busy = False

    def basin_explore(self, **kwargs: object) -> object:
        """Send a basin exploration task to the engine and return the result."""
        self._is_busy = True
        try:
            self.send_message({"type": "basin_explore", "value": kwargs})
            return self._receive_result_or_error()
        finally:
            self._is_busy = False

    #@session_locked
    def get_total_energy(self, positions=None) :
        self._is_busy = True
        #print(f"[Session n°{self.session_id}]  get potential energy")
        try :
            self.send_message({"type": "get_total_energy", "value": {"positions": positions}})
            return self._receive_result_or_error()
        finally :
            self._is_busy = False

    #@session_locked
    def get_potential_energy(self, positions=None) :
        self._is_busy = True
        #print(f"[Session n°{self.session_id}]  get potential energy")
        try :
            self.send_message({"type": "get_potential_energy"})
            return self._receive_result_or_error()
        finally :
            self._is_busy = False

    #@session_locked
    def partn_search(self, config, central_atom_idx, positions=None, cell=None, types=None) :
        self._is_busy = True
        #print(f"[Session] Launching pARTn search")
        try :
            self.send_message({"type": "partn_search", "value": {"config": config, "central_atom_idx": central_atom_idx, "positions": positions, "cell": cell, "types": types}})
            return self._receive_result_or_error()
        finally :
            self._is_busy = False

    #@session_locked
    def partn_refine(self, config, central_atom_idx, positions=None, cell=None, types=None, saddle_idx=None, saddle_positions=None) :
        self._is_busy = True
        #print(f"[Session] Launching pARTn search")
        try :
            self.send_message({"type": "partn_refine", "value": {"config": config, "central_atom_idx": central_atom_idx, "positions": positions, "cell":cell, "types":types, "saddle_idx":saddle_idx, "saddle_positions":saddle_positions}})
            return self._receive_result_or_error()
        finally :
            self._is_busy = False

    def get_forces(self, positions: "np.ndarray | None" = None) -> "np.ndarray":
        """Request the (N, 3) forces from the engine (run 0 + gather)."""
        self._is_busy = True
        try:
            self.send_message({"type": "get_forces", "value": {"positions": positions}})
            return self._receive_result_or_error()
        finally:
            self._is_busy = False

    def dynamical_matrix_eskm(
        self,
        positions: "np.ndarray",
        free_indices: "np.ndarray | list[int] | None" = None,
        dx: float = 1e-2,
    ) -> "np.ndarray":
        """Request the mass-weighted partial Hessian (LAMMPS eskm) from the engine."""
        self._is_busy = True
        try:
            self.send_message(
                {
                    "type": "dynamical_matrix_eskm",
                    "value": {
                        "positions": positions,
                        "free_indices": free_indices,
                        "dx": dx,
                    },
                }
            )
            return self._receive_result_or_error()
        finally:
            self._is_busy = False

    def compute_event_prefactors(
        self, config, central_atom_idx, min1_positions, saddle_positions,
        min2_positions, types, cell,
    ):
        """Request per-event Vineyard nu0 from the engine (returns EventPrefactors)."""
        self._is_busy = True
        try:
            self.send_message(
                {
                    "type": "compute_event_prefactors",
                    "value": {
                        "config": config,
                        "central_atom_idx": central_atom_idx,
                        "min1_positions": min1_positions,
                        "saddle_positions": saddle_positions,
                        "min2_positions": min2_positions,
                        "types": types,
                        "cell": cell,
                    },
                }
            )
            return self._receive_result_or_error()
        finally:
            self._is_busy = False
