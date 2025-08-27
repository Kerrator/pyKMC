"""Class for managing a pool of LAMMPS instances."""
from ..sessions import MpiApiSession
from dataclasses import dataclass
from concurrent.futures import Future
import queue
import threading

@dataclass 
class Job: 
    operation_name: str 
    params: dict
    future: Future


class Manager:
    """A class to manage a pool of Lammps sessions."""

    def __init__(self, sessions: list[MpiApiSession]) -> None:
        """
        Initialize the LammpsPoolManager with a specified number of sessions.
        """
        self.sessions = sessions
        self.job_queue: queue.Queue[Job] = queue.Queue()
        #Thread that dispatch job to workers
        self.dispatcher_thread = threading.Thread(target=self._dispatcher, daemon=True) 
        self.dispatcher_thread.start()

    def broadcast_command(self, cmd: str):
        """
        Send the same command to all sessions and wait for all to finish.
        """
        print("[PoolManager] Broadcasting command:", cmd)
        for session in self.sessions:
            session.command(cmd)

    def initialize_sessions(self, config, system) : 
        """ 
        Initialize engines with the same system and config
        """
        print("[Manager] Initializing all Lammps engines")
        for session in self.sessions : 
            session.initialize_parameters() 
            session.initialize_system(system)
            session.initialize_potential(config)


    def _dispatcher(self) : 
        while True : 
            job = self.job_queue.get() #block until a job is get 
            while True : 
                session = self._get_available_engine() 
                if session is not None : 
                    print(f"[PoolManager] Found available session: {session.session_id}")
                    threading.Thread(target=self._run_job, args=(session, job), daemon=True).start()
                    threading.Event().wait(0.05) # Wait a bit to allow the job to be processed
                    break #job is submited
                else : 
                    threading.Event().wait(0.05)


    def _get_available_engine(self) : 
        """Check if worker is available, if yes return worker, if not, return None"""
        for session in self.sessions : 
            if session._is_busy == False : 
                return session
        return None

    def _run_job(self, session, job: Job) : 
        try : 
            #find method session having job.method_name
            method = getattr(session, job.operation_name)
            print(f"[PoolManager] Running job: {job.operation_name} with params: {job.params} on session: {session.session_id}") 
            if job.params is None : 
                result = method()
            else : 
                result = method(**job.params) 
            job.future.set_result(result)
        except Exception as e : 
            job.future.set_exception(e)


    def submit_job(self, method_name: str, params: dict = None) -> Future:

        future = Future()
        job = Job(method_name, params, future)
        print(f"[PoolManager] Submitting job: {job.operation_name} with params: {job.params}")
        self.job_queue.put(job)
        return future

    def minimize(self, config ) : 
        future = self.submit_job("minimize", {"config" : config})
        return future

    def partn_search(self, config, central_atom: list[int]) -> list[Future] : 
        futures = []
        for atom in central_atom :
            f = self.submit_job("partn_search", {"config": config, "central_atom_idx": atom})
            futures.append(f) 
        return futures

    def partn_refine(self, config, central_atom: list[int]) -> list[Future] : 
        futures = []
        for atom in central_atom :
            f = self.submit_job("partn_refine", {"config": config, "central_atom_idx": atom})
            futures.append(f) 
        return futures

    def close_all(self):
        """
        Close all sessions and their underlying engines.
        """
        print("[PoolManager] Closing all sessions.")
        for session in self.sessions:
            session.close()    