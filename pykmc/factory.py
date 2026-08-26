from mpi4py import MPI
from typing import Any, Callable
from pykmc.engine import Engine
from pykmc.manager import Manager, ManagerFactory
from pykmc.manager.worker import Worker


class EngineManagerFactory(ManagerFactory):
    """Engine-specific subclass of ManagerFactory.

    Encodes the pyKMC convention: one Engine per worker for local and group
    modes, None for global (global mode is reserved for extra_ops).  The
    engine is constructed but not started — call ``manager.broadcast("start")``
    and the other initialisation methods after ``launch()``.

    Log-file numbering (when ``engine_config.verbosity != 0``):
        - ``lammps.log.0``          — group engine (shared across workers)
        - ``lammps.log.<worker+1>`` — local engine of each worker

    Parameters
    ----------
    engine_style      : str
    engine_config     : Any | None    Forwarded to Engine.create.
    n_workers         : int
    comm              : MPI.Comm
    group_size        : int | None
    engine_extensions : list | None   EngineExtension subclasses to instantiate
                                      on each engine after creation.
    extra_ops         : dict[str, Callable] | None
                                      MPI-aware callables ``fn(comm, **kwargs)``
                                      available in every mode.
    """

    def __init__(
        self,
        engine_style: str,
        n_workers: int,
        comm: MPI.Comm,
        group_size: int | None = None,
        engine_config: Any | None = None,
        engine_extensions: list | None = None,
        extra_ops: dict[str, Callable] | None = None,
    ) -> None:
        self._engine_style = engine_style
        self._engine_config = engine_config
        self._engine_extensions = engine_extensions or []

        super().__init__(
            obj_factory=lambda *_: None,  # unused — overridden by _create_worker
            n_workers=n_workers,
            comm=comm,
            has_global=True,
            group_size=group_size,
            extra_ops=extra_ops,
        )

    def _make_engine(self, engine_comm: MPI.Comm, mode: str, worker_id: int) -> Engine:
        engine_id = 0 if mode == "group" else worker_id + 1
        engine = Engine.create(
            self._engine_style,
            config=self._engine_config,
            comm=engine_comm,
            engine_id=engine_id,
        )
        for ext_cls in self._engine_extensions:
            ext_cls(engine)
        return engine

    def _create_worker(
        self,
        local_comm: MPI.Comm,
        worker_id: int,
        global_comm: MPI.Comm | None,
        group_comm: MPI.Comm | None,
    ) -> Worker:
        return Worker(
            local_obj=self._make_engine(local_comm, "local", worker_id),
            local_comm=local_comm,
            worker_id=worker_id,
            global_obj=None,  # global mode is reserved for extra_ops
            global_comm=global_comm,
            group_obj=self._make_engine(group_comm, "group", worker_id)
            if group_comm is not None
            else None,
            group_comm=group_comm,
            extra_ops=self.extra_ops,
            world_comm=self.comm,
        )
