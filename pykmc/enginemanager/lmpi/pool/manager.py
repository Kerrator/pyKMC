"""Class for managing a pool of LAMMPS instances."""
from ..sessions import MpiApiSession
from dataclasses import dataclass
from concurrent.futures import Future
import queue
import threading

#TODO : commented print should be log depending of the verbosity but need to thing of how we modify log before (also loggers are 
#initiated in kmc, after the initialization of manager ...))

@dataclass 
class Job: 
    operation_name: str 
    params: dict
    future: Future


class Manager:
    """A class to manage a pool of Lammps sessions."""

    def __init__(self, sessions: list[MpiApiSession], global_session: MpiApiSession = None) -> None:
        """
        Initialize the LammpsPoolManager with a specified number of sessions.
        """
        self.sessions = sessions
        self.global_session = global_session
        self.using_global = True
        self.job_queue: queue.Queue[Job] = queue.Queue()
        #Thread that dispatch job to workers
        #self.dispatcher_thread = threading.Thread(target=self._dispatcher, daemon=True)
        #self.dispatcher_thread.start()

        #pool de workers
        self.workers = []
        for session in self.sessions:
            t = threading.Thread(target=self._worker_loop, args=(session,), daemon=True)
            t.start()
            self.workers.append(t)

    def broadcast_command(self, cmd: str):
        """
        Send the same command to all sessions and wait for all to finish.
        """
        #print("[PoolManager] Broadcasting command:", cmd)
        for session in self.sessions:
            session.command(cmd)

    def initialize_sessions(self, config, system) : 
        """ 
        Initialize engines with the same system and config
        """
        # Propagate the opt-in engine-op wall guard to every session on rank 0 so a
        # desynced pool fails fast instead of stalling.
        op_timeout = getattr(config.control, "engine_op_timeout_s", None)
        for session in self.sessions:
            session._op_timeout = op_timeout
        if self.global_session is not None:
            self.global_session._op_timeout = op_timeout
        print("[Manager] use local")
        self.use_local()
        print("[Manager] Initializing all Lammps engines")
        for session in self.sessions : 
            session.initialize_parameters() 
            session.initialize_system(system)
            session.initialize_potential(config)
        print("[Manager] use global")
        self.use_global()
        print("[Manager] Initializing global Lammps engines")
        if self.global_session is not None :
            self.global_initialize_parameters()
            self.global_initialize_system(system)
            self.global_initialize_potential(config)


    def use_local(self):
        """
        Have engines switch from global pool to local pools
        """
        if self.using_global:
            self.global_session.use_local()
            self.using_global = False

    def use_global(self):
        """
        Have engines switch from local pools to global pool
        """
        if not self.using_global:
            for session in self.sessions :
                session.use_global()
            self.using_global = True
    def _worker_loop(self, session: MpiApiSession):
        """Boucle infinie tournant dans un thread dédié à 'session'."""
        while True:
            job = self.job_queue.get()

            if job is None:
                break

            try:
                method = getattr(session, job.operation_name)

                if job.params is None:
                    result = method()
                else:
                    result = method(**job.params)

                job.future.set_result(result)

            except Exception as e:
                job.future.set_exception(e)
            finally:
                self.job_queue.task_done()

    #def _dispatcher(self) :
    #    while True :
    #        job = self.job_queue.get() #block until a job is get
    #        while True :
    #            session = self._get_available_engine()
    #            if session is not None :
    #                #print(f"[PoolManager] Found available session: {session.session_id}")
    #                threading.Thread(target=self._run_job, args=(session, job), daemon=True).start()
    #                threading.Event().wait(0.1) # Wait a bit to allow the job to be processed
    #                break #job is submited
    #            else :
    #                threading.Event().wait(0.1)


    #def _get_available_engine(self) :
    #    """Check if worker is available, if yes return worker, if not, return None"""
    #    for session in self.sessions :
    #        if session._is_busy == False :
    #            return session
    #    return None

    #def _run_job(self, session, job: Job) :
    #    try :
    #        #find method session having job.method_name
    #        method = getattr(session, job.operation_name)
    #        #print(f"[PoolManager] Running job: {job.operation_name}  on session: {session.session_id}")
    #        if job.params is None :
    #            result = method()
    #        else :
    #            result = method(**job.params)
    #        job.future.set_result(result)
    #    except Exception as e :
    #        job.future.set_exception(e)

    def set_all_positions(self, positions) : 
        #print("[Manager] Setting positions to all sessions.")
        for session in self.sessions : 
            session.set_positions(positions=positions)

    def submit_job(self, method_name: str, params: dict = None) -> Future:

        future = Future()
        job = Job(method_name, params, future)
        #print(f"[PoolManager] Submitting job: {job.operation_name}") #with params: {job.params}")
        self.job_queue.put(job)
        return future

    # API

    def minimize(self, config ) : 
        future = self.submit_job("minimize", {"config" : config})
        return future
    
    def minimize_with_results(self, config, positions=None, types=None) :
        future = self.submit_job("minimize_with_results", {"config": config, "positions": positions, "types": types})
        return future
    
    def get_potential_energy(self, positions=None) :
        future = self.submit_job("get_potential_energy", {"positions": positions})
        return future

    def get_total_energy(self, positions=None) :
        future = self.submit_job("get_total_energy", {"positions": positions})
        return future

    def partn_search(self, config, central_atom: list[int], positions=None, cell=None, types=None) -> list[Future] :
        futures = []
        for atom in central_atom :
            f = self.submit_job("partn_search", {"config": config, "central_atom_idx": atom, "positions": positions, "cell":cell, "types":types})
            futures.append(f)
        return futures

    def partn_refine(self, config, central_atom: int, positions=None, cell=None, types=None, saddle_idx=None, saddle_positions=None) -> list[Future] :
        future = self.submit_job("partn_refine", {"config": config, "central_atom_idx": central_atom, "positions": positions, "cell":cell, "types":types, "saddle_idx":saddle_idx, "saddle_positions":saddle_positions})
        return future

    def compute_event_prefactors(self, config, events: list[dict]) -> list[Future]:
        """Fan out one per-event Vineyard nu0 job per event across the session pool.

        Each item in ``events`` is a dict with keys: central_atom_idx,
        min1_positions, saddle_positions, min2_positions, types, cell.
        Mirrors ``partn_search`` (one submit_job per work item).
        """
        futures = []
        for ev in events:
            f = self.submit_job("compute_event_prefactors", {"config": config, **ev})
            futures.append(f)
        return futures

    def compute_forces(self, positions: object = None) -> Future:
        """Compute the (N, 3) forces on one session of the pool."""
        return self.submit_job("get_forces", {"positions": positions})

    def compute_dynamical_matrix(
        self, positions: object, free_indices: object = None, dx: float = 1e-2
    ) -> Future:
        """Compute the mass-weighted partial Hessian (eskm) on one session of the pool."""
        return self.submit_job(
            "dynamical_matrix_eskm",
            {"positions": positions, "free_indices": free_indices, "dx": dx},
        )

    def basin_reconstruct(self, **kwargs: object) -> Future :
        """Submit a basin state reconstruction (PSR + 2x minimize) to the session pool."""
        return self.submit_job("basin_reconstruct", kwargs)

    def basin_explore(self, **kwargs: object) -> Future :
        """Submit a basin state exploration (reference-table lookups) to the session pool."""
        return self.submit_job("basin_explore", kwargs)

    def close_all(self):
        """Close all sessions and their underlying engines.

        Teardown must run with every engine listening on its OWN local engine
        communicator. The global ``engine_comm`` spans every engine rank, so closing
        the global session broadcasts a shutdown to all engines at once and exits
        their run loops together; only the global master rank then emits a status, so
        the first local ``close(wait_status=True)`` consumes it and the next one
        blocks in ``receive_status()`` forever -- the multi-session teardown hang seen
        when a global-mode op path closes the pool after a failure.

        Switching the pool to local mode first makes every engine listen on its own
        comm, so each local ``close`` is handled and acknowledged by exactly one
        engine. ``MpiApiEngine.close()`` shuts down BOTH that rank's local and global
        LAMMPS instance, so closing every local session tears the whole pool down --
        the separate global-session close is redundant as well as unsafe.
        """
        #print("[PoolManager] Closing all sessions.")
        if self.global_session is not None :
            self.use_local()
        for session in self.sessions:
            session.close(wait_status=True)


    def __getattr__(self, name:str) :
        """Check if method start with global_, if yes, then return global_session.method"""
        if name.startswith('global_'):
            method_name = name[7:]  # remove prefixe 'global_'
            if not self.global_session:
                raise RuntimeError("Global session is not available")

            def global_method(*args, **kwargs):
                method = getattr(self.global_session, method_name)
                return method(*args, **kwargs)

            return global_method

        raise AttributeError(f"'{type(self).__name__}' object has no attribute '{name}'")



