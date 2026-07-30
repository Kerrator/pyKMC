from mpi4py import MPI
import numpy as np
from .manager import Manager
from .worker import Worker
from .session import Session
from typing import Any, Callable


class ManagerFactory:
    """Splits MPI ranks and instantiates Workers, Sessions, and a Manager.

    Object creation is fully delegated to ``obj_factory``, a plain callable
    with signature::

        obj_factory(comm: MPI.Comm, mode: str) -> Any | None

    This keeps the factory independent of any specific class or framework.
    The returned object is registered in the Worker's registry; it is expected
    to be initialised with the provided communicator if it relies on MPI.

    Example
    -------
    ::

        factory = ManagerFactory(
            obj_factory=lambda comm, mode: MyEngine(config=cfg, comm=comm),
            n_workers=4,
            comm=MPI.COMM_WORLD,
        )
        manager = factory.launch()   # returns Manager on rank 0, None elsewhere

    Parameters
    ----------
    obj_factory  : Callable[[MPI.Comm, str], Any | None]
        Called once per (worker × mode) with the communicator and the mode name
        (``"local"``, ``"group"``, or ``"global"``).  Return None to skip object
        creation for that mode — the worker's registry will then only hold
        extra_ops for that mode.
    n_workers    : int
    comm         : MPI.Comm
    has_global   : bool          Enable global communicator (all workers).
    group_size   : int | None    Number of ranks sharing the group communicator.
                                 None disables group mode.
    extra_ops    : dict[str, Callable] | None
                                 MPI-aware callables ``fn(comm, **kwargs)``
                                 registered in every worker registry.
    """

    _MANAGER_RANK = 0  # rank 0 is always the manager

    def __init__(
        self,
        obj_factory: Callable[[MPI.Comm, str], Any | None],
        n_workers: int,
        comm: MPI.Comm,
        has_global: bool = True,
        group_size: int | None = None,
        extra_ops: dict[str, Callable] | None = None,
    ) -> None:
        self.obj_factory = obj_factory
        self.comm = comm
        self.n_workers = n_workers
        self.has_global = has_global
        self.group_size = group_size
        self.extra_ops = extra_ops

        self.size = self.comm.Get_size()
        self.rank = self.comm.Get_rank()

        if self.size < self.n_workers + self._MANAGER_RANK + 1:
            raise ValueError("Not enough MPI ranks to allocate workers.")

        self.available_ranks = list(range(self._MANAGER_RANK + 1, self.size))
        self.chunks = self._split_ranks()

        if self.group_size is not None:
            if self.group_size > len(self.available_ranks):
                raise ValueError(
                    f"group_size ({group_size}) cannot exceed the number of available ranks ({len(self.available_ranks)})."
                )
            acceptable_group_size = np.cumsum([len(c) for c in self.chunks])
            if self.group_size not in acceptable_group_size:
                raise ValueError(
                    f"group_size ({group_size}) must be running on a subset of workers, available group_size=({acceptable_group_size})."
                )
            self.group_ranks: list[int] = self.available_ranks[: self.group_size]
        else:
            self.group_ranks = []

    def _split_ranks(self) -> list[list[int]]:
        return [
            arr.tolist() for arr in np.array_split(self.available_ranks, self.n_workers)
        ]

    def launch(self) -> Manager | None:
        """Split ranks, start workers, and return a Manager on rank 0.

        All ranks must call this collectively. Workers block inside their loop;
        only rank 0 returns a Manager instance. All other ranks return None.
        """
        my_color = MPI.UNDEFINED
        worker_id = None
        for idx, chunk in enumerate(self.chunks):
            if self.rank in chunk:
                my_color = idx + 1
                worker_id = idx
                break

        local_comm = self.comm.Split(color=my_color, key=self.rank)

        global_comm = None
        if self.has_global:
            in_global = self.rank in self.available_ranks
            global_split = self.comm.Split(
                color=1 if in_global else MPI.UNDEFINED, key=self.rank
            )
            if global_split != MPI.COMM_NULL:
                global_comm = global_split

        group_comm = None
        if self.group_ranks:
            group_split = self.comm.Split(
                color=1 if self.rank in self.group_ranks else MPI.UNDEFINED,
                key=self.rank,
            )
            if group_split != MPI.COMM_NULL:
                group_comm = group_split

        if worker_id is not None:
            worker = self._create_worker(local_comm, worker_id, global_comm, group_comm)
            worker.start()
            return None

        # rank 0 — manager
        local_sessions = [
            Session(
                engine_master_rank=self.chunks[i][0],
                world_comm=self.comm,
                session_id=i + 1,
            )
            for i in range(self.n_workers)
        ]
        global_session = (
            Session(
                engine_master_rank=self.available_ranks[0],
                session_id=0,
                world_comm=self.comm,
            )
            if self.has_global
            else None
        )

        group_session = (
            Session(
                engine_master_rank=self.group_ranks[0],
                session_id=-1,
                world_comm=self.comm,
            )
            if self.group_ranks
            else None
        )

        manager = Manager(
            local_sessions=local_sessions,
            global_session=global_session,
            group_session=group_session,
        )
        manager.start()
        return manager

    def _create_worker(
        self,
        local_comm: MPI.Comm,
        worker_id: int,
        global_comm: MPI.Comm | None,
        group_comm: MPI.Comm | None,
    ) -> Worker:
        return Worker(
            local_obj=self.obj_factory(local_comm, "local"),
            local_comm=local_comm,
            worker_id=worker_id,
            global_obj=self.obj_factory(global_comm, "global")
            if global_comm is not None
            else None,
            global_comm=global_comm,
            group_obj=self.obj_factory(group_comm, "group")
            if group_comm is not None
            else None,
            group_comm=group_comm,
            extra_ops=self.extra_ops,
            world_comm=self.comm,
        )
