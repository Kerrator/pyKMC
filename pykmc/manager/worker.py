from __future__ import annotations
from typing import Callable, Any
from dataclasses import dataclass
from enum import Enum
from mpi4py import MPI
import inspect


def build_registry(obj: object | list[object] | None = None) -> dict[str, Callable]:
    """Build an operation registry from one or more objects.

    Only object methods are collected here.  Plain callables (extra_ops) are
    kept separate and merged by the Worker, so that consistency checks can
    distinguish between object-derived methods and extra_ops.

    Parameters
    ----------
    obj : object | list[object] | None
        One object or a list of objects whose public methods are collected.
        Returns an empty dict if None.

    Returns
    -------
    dict[str, Callable]

    Raises
    ------
    ValueError
        If two objects expose a method with the same name.
    """
    registry: dict[str, Callable] = {}
    objs = obj if isinstance(obj, list) else ([obj] if obj is not None else [])
    for o in objs:
        for name, method in inspect.getmembers(o, predicate=callable):
            if name.startswith("_"):
                continue
            if name in registry:
                raise ValueError(
                    f"Operation '{name}' is defined on multiple objects passed to build_registry."
                )
            registry[name] = method
    return registry


class DispatchStatus(Enum):
    SILENT  = "silent"
    SUCCESS = "success"
    ERROR   = "error"


@dataclass
class DispatchResult:
    status: DispatchStatus
    value: Any = None
    error: str | None = None


class Worker:

    def __init__(self,
                 local_obj:   object | list[object] | None,
                 local_comm:  "MPI.COMM",
                 worker_id:   int,
                 global_obj:  object | list[object] | None = None,
                 global_comm: "MPI.COMM" | None = None,
                 group_obj:   object | list[object] | None = None,
                 group_comm:  "MPI.COMM" | None = None,
                 extra_ops:   dict[str, Callable] | None = None,
                 world_comm:  "MPI.COMM" | None = None) -> None:
        """MPI worker. Runs on all ranks in its communicator.

        Rank 0 reads incoming messages from ``world_comm``, broadcasts them to
        all ranks in the active communicator, and dispatches to the registry.
        Other ranks only participate in the collective broadcast and execute.

        Three modes
        -----------
        local  — one registry per worker (master-worker architecture).
        group  — a subset of workers execute collectively via ``group_comm``.
                 Workers without a ``group_comm`` silently stay in local mode
                 when a ``use_group`` message arrives.
        global — all workers execute collectively via ``global_comm``.

        Registries
        ----------
        Each mode's registry = object methods (from ``*_obj``) merged with
        ``extra_ops``.  ``build_registry`` handles only objects; the Worker
        merges extra_ops separately so that consistency checks can distinguish
        between the two sources.

        Modes where ``obj=None`` have a registry that contains only extra_ops.
        Their consistency is not checked against local — the caller accepts
        that object methods are unavailable in those modes.

        Design note on objects
        ----------------------
        ``local_obj``, ``global_obj``, and ``group_obj`` are expected to be
        MPI-aware objects (e.g. Engine subclasses) initialised with their
        respective communicator.

        Design note on extra_ops
        ------------------------
        ``extra_ops`` are MPI-aware callables with signature
        ``fn(comm, **kwargs)``.  The Worker injects the *active* communicator
        at dispatch time, so the same function adapts to every mode.

        Parameters
        ----------
        local_obj   : object | list | None
        local_comm  : MPI.Comm
        worker_id   : int
        global_obj  : object | list | None
        global_comm : MPI.Comm | None
        group_obj   : object | list | None
        group_comm  : MPI.Comm | None
        extra_ops   : dict[str, Callable] | None
            Keys must not clash with object method names, nor with builtin
            names (use_local, use_group, use_global, close).
        world_comm  : MPI.Comm | None  Defaults to MPI.COMM_WORLD.

        Raises
        ------
        ValueError
            If obj-derived registries differ across modes that have objects,
            extra_ops clash with object methods, or a key clashes with a builtin.
        """
        self.local_obj   = local_obj
        self.local_comm  = local_comm
        self.global_obj  = global_obj
        self.global_comm = global_comm
        self.group_obj   = group_obj
        self.group_comm  = group_comm
        self.local_rank  = local_comm.Get_rank()
        self.worker_id   = worker_id
        self._is_alive   = False

        self.world_comm = world_comm or MPI.COMM_WORLD

        self._builtins_op = {
            "use_local":  self.use_local,
            "use_group":  self.use_group,
            "use_global": self.use_global,
            "shutdown":   self.shutdown,
            "list_ops":   self.list_ops,
        }

        # extra_ops names tracked separately so dispatch can inject comm.
        self._extra_op_names: frozenset[str] = frozenset(extra_ops) if extra_ops else frozenset()
        _extra = extra_ops or {}

        # Build obj-only registries (used for consistency check), then merge extra_ops.
        _local_obj_reg  = build_registry(local_obj)
        _global_obj_reg = build_registry(global_obj) if global_comm is not None else {}
        _group_obj_reg  = build_registry(group_obj)  if group_comm  is not None else {}

        self._check_obj_clash(_extra)
        self._check_builtin_clashes(_local_obj_reg,  _extra, "local")
        self._check_obj_registries(_local_obj_reg, _global_obj_reg, _group_obj_reg)

        self.local_registry  = {**_local_obj_reg,  **_extra}
        if global_comm is not None:
            self.global_rank     = global_comm.Get_rank()
            self.global_registry = {**_global_obj_reg, **_extra}
            self._check_builtin_clashes(_global_obj_reg, _extra, "global")
        if group_comm is not None:
            self.group_rank     = group_comm.Get_rank()
            self.group_registry = {**_group_obj_reg, **_extra}
            self._check_builtin_clashes(_group_obj_reg, _extra, "group")

        self.use_local()

    # ------------------------------------------------------------------
    # Registry validation
    # ------------------------------------------------------------------

    def _check_obj_clash(self, extra_ops: dict) -> None:
        """Raise if any extra_ops key clashes with a local object method."""
        clashes = set(extra_ops) & set(build_registry(self.local_obj))
        if clashes:
            raise ValueError(
                f"extra_ops clash with object methods: {clashes}. "
                f"Rename the extra_ops keys or remove the conflicting methods."
            )

    def _check_builtin_clashes(self, obj_reg: dict, extra_ops: dict, mode_name: str) -> None:
        """Raise if any operation name (obj or extra) shadows a builtin."""
        clashes = (set(obj_reg) | set(extra_ops)) & set(self._builtins_op)
        if clashes:
            raise ValueError(
                f"{mode_name} registry clashes with builtin operations: {clashes}. "
                f"Rename the conflicting methods or extra_ops keys."
            )

    def _check_obj_registries(self,
                               local_reg:  dict,
                               global_reg: dict,
                               group_reg:  dict) -> None:
        """Raise if a mode with an object exposes different methods from local.

        Modes where obj=None have an empty obj-registry and are not checked —
        they intentionally only expose extra_ops.
        """
        local_ops = set(local_reg)
        if self.global_obj is not None and set(global_reg) != local_ops:
            diff = local_ops.symmetric_difference(set(global_reg))
            raise ValueError(
                f"local and global obj-registries differ: {diff}. "
                f"Pass equivalent objects to all modes."
            )
        if self.group_obj is not None and set(group_reg) != local_ops:
            diff = local_ops.symmetric_difference(set(group_reg))
            raise ValueError(
                f"local and group obj-registries differ: {diff}. "
                f"Pass equivalent objects to all modes."
            )

    # ------------------------------------------------------------------
    # Mode switching
    # ------------------------------------------------------------------

    def use_local(self) -> None:
        """Switch to local mode."""
        self.mode     = "local"
        self.comm     = self.local_comm
        self.rank     = self.local_comm.Get_rank()
        self.registry = self.local_registry

    def use_global(self) -> None:
        """Switch to global mode. No-op if no global_comm."""
        if self.global_comm is None:
            return
        self.mode     = "global"
        self.comm     = self.global_comm
        self.rank     = self.global_comm.Get_rank()
        self.registry = self.global_registry

    def use_group(self) -> None:
        """Switch to group mode. No-op if worker has no group_comm."""
        if self.group_comm is None:
            return
        self.mode     = "group"
        self.comm     = self.group_comm
        self.rank     = self.group_comm.Get_rank()
        self.registry = self.group_registry

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def start(self) -> None:
        """Enter the message loop. Blocks until ``shutdown`` is dispatched."""
        self._is_alive = True
        self._loop()

    def shutdown(self) -> None:
        """Stop the message loop and call .close() on all objects that support it."""
        self._is_alive = False
        all_objs = (
            (self.local_obj  if isinstance(self.local_obj,  list) else ([self.local_obj]  if self.local_obj  is not None else [])) +
            (self.global_obj if isinstance(self.global_obj, list) else ([self.global_obj] if self.global_obj is not None else [])) +
            (self.group_obj  if isinstance(self.group_obj,  list) else ([self.group_obj]  if self.group_obj  is not None else []))
        )
        for o in all_objs:
            if hasattr(o, "close"):
                o.close()

    # ------------------------------------------------------------------
    # Internal loop / dispatch
    # ------------------------------------------------------------------

    def _loop(self) -> None:
        """Main loop. All ranks run this until ``_is_alive`` is False."""
        while self._is_alive:

            if self.rank == 0:
                msg = self._read_messages()
            else:
                msg = None

            msg = self.comm.bcast(msg, root=0)
            r   = self._dispatch(msg)

            if self.rank == 0:
                if r.status == DispatchStatus.SILENT:
                    pass
                elif r.status == DispatchStatus.SUCCESS:
                    self.world_comm.send({"type": "status", "value": {"has_result": r.value is not None, "error": None}}, dest=0, tag=0)
                    if r.value is not None:
                        self.world_comm.send({"type": "result", "value": r.value}, dest=0, tag=1)
                elif r.status == DispatchStatus.ERROR:
                    self.world_comm.send({"type": "status", "value": {"has_result": False, "error": r.error}}, dest=0, tag=0)

            if not self._is_alive:
                break

    def _read_messages(self) -> dict:
        """Read one message from world_comm on rank 0. Blocks until a message arrives."""
        return self.world_comm.recv(source=MPI.ANY_SOURCE, tag=2)

    def _dispatch(self, msg: dict) -> DispatchResult:
        op_type = msg.get("type")

        value = msg.get("value")
        if value is None:
            kwargs = {}
        elif isinstance(value, dict):
            kwargs = value
        else:
            kwargs = {"value": value}

        if op_type in self._builtins_op:
            result = self._builtins_op[op_type](**kwargs)
            if result is not None:
                return DispatchResult(DispatchStatus.SUCCESS, value=result)
            return DispatchResult(DispatchStatus.SILENT)

        handler = self.registry.get(op_type)
        if handler is None:
            return DispatchResult(DispatchStatus.ERROR, error=f"Unknown operation '{op_type}'. Available: {list(self._builtins_op) + list(self.registry)}")

        self.comm.barrier()
        try:
            # extra_ops receive the active comm as first arg so they can use
            # MPI collectives appropriate to the current mode.
            if op_type in self._extra_op_names:
                result = handler(self.comm, **kwargs)
            else:
                result = handler(**kwargs)
            return DispatchResult(DispatchStatus.SUCCESS, value=result)
        except Exception as e:
            return DispatchResult(DispatchStatus.ERROR, error=str(e))
        finally:
            self.comm.barrier()

    def list_ops(self, mode: str = "local") -> list[str]:
        registries = {
            "local":  self.local_registry,
            "global": getattr(self, "global_registry", {}),
            "group":  getattr(self, "group_registry",  {}),
        }
        if mode not in registries:
            raise ValueError(f"Unknown mode '{mode}'. Expected one of: {list(registries)}")
        return list(registries[mode])

    def __repr__(self) -> str:
        return (
            f"Worker(\n"
            f"  mode     = {self.mode!r},\n"
            f"  ops      = {list(self.registry)},\n"
            f"  builtins = {list(self._builtins_op)}\n"
            f")"
        )
