from __future__ import annotations
from typing import Callable, Any
from dataclasses import dataclass
from enum import Enum
from mpi4py import MPI
import inspect


def build_registry(obj: object | list[object] | None = None) -> dict[str, Callable]:
    """Build an operation registry from one or more objects.

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
            if name.startswith("_"):  # only collect public methods.
                continue
            if name in registry:
                raise ValueError(
                    f"Operation '{name}' is defined on multiple objects passed to build_registry."
                )
            registry[name] = method
    return registry


# Convienient part to deal with worker operation that expect a result, or not, and errors.
class DispatchStatus(Enum):
    SILENT = "silent"
    SUCCESS = "success"
    ERROR = "error"


@dataclass
class DispatchResult:
    status: DispatchStatus
    value: Any = None
    error: str | None = None


class Worker:
    # Must match Session._TAG_*
    _TAG_CMD = 2
    _TAG_STATUS = 0
    _TAG_RESULT = 1

    def __init__(
        self,
        local_obj: object | list[object] | None,
        local_comm: "MPI.COMM",
        worker_id: int,
        global_obj: object | list[object] | None = None,
        global_comm: "MPI.COMM" | None = None,
        group_obj: object | list[object] | None = None,
        group_comm: "MPI.COMM" | None = None,
        extra_ops: dict[str, Callable] | None = None,
        world_comm: "MPI.COMM" | None = None,
    ) -> None:
        """MPI worker. Each instance runs on all ranks in ``local_comm``.

        Rank 0 of the active communicator reads incoming messages from
        ``world_comm``, broadcasts them to all ranks in the active communicator,
        and dispatches to the registry. Other ranks only participate in the
        collective broadcast and execute. In group/global mode the active
        communicator spans multiple Worker instances; all of them must be
        running their loop simultaneously for the collective to complete.

        Designed to be used with Session/Manager.

        Three modes
        -----------
        local  - one registry per worker (master-worker architecture).
        group  - a subset of workers execute collectively via ``group_comm``.
                 Workers without a ``group_comm`` silently stay in local mode
                 when a ``use_group`` message arrives.
        global - all workers execute collectively via ``global_comm``.

        Registries
        ----------
        Each mode's registry = object methods (from ``*_obj``) merged with
        ``extra_ops``.  ``build_registry`` handles only objects, the Worker
        merges extra_ops separately so that consistency checks can distinguish
        between the two sources.

        Modes where ``obj=None`` have a registry that contains only extra_ops.
        Their consistency is not checked against local, the caller accepts
        that object methods are unavailable in those modes.

        Design note on objects
        ----------------------
        ``local_obj``, ``global_obj``, and ``group_obj`` are expected to be
        MPI-aware objects (e.g. Engine subclasses) initialised with their
        respective communicator.

        Design note on extra_ops
        ------------------------
        ``extra_ops`` are MPI-aware callables with signature
        ``fn(comm, *args, **kwargs)``.  The Worker injects the *active*
        communicator at dispatch time, so the same function adapts to every mode.

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
        self.local_obj = local_obj
        self.local_comm = local_comm
        self.global_obj = global_obj
        self.global_comm = global_comm
        self.group_obj = group_obj
        self.group_comm = group_comm
        self.local_rank = local_comm.Get_rank()
        self.worker_id = worker_id
        self._is_alive = False

        self.world_comm = world_comm or MPI.COMM_WORLD

        self._builtins_op = {
            "use_local": self.use_local,
            "use_group": self.use_group,
            "use_global": self.use_global,
            "shutdown": self.shutdown,
            "list_ops": self.list_ops,
        }

        # extra_ops names tracked separately so dispatch can inject comm.
        self._extra_op_names = frozenset(extra_ops) if extra_ops else frozenset()
        _extra = extra_ops or {}

        # Build obj-only registries (used for consistency check), then merge extra_ops.
        _local_obj_reg = build_registry(local_obj)
        _global_obj_reg = build_registry(global_obj) if global_comm is not None else {}
        _group_obj_reg = build_registry(group_obj) if group_comm is not None else {}

        self._check_obj_clash(_extra)
        self._check_builtin_clashes(_local_obj_reg, _extra, "local")
        self._check_obj_registries(_local_obj_reg, _global_obj_reg, _group_obj_reg)

        self.local_registry = {**_local_obj_reg, **_extra}
        self.global_rank = None
        self.global_registry = {}
        self.group_rank = None
        self.group_registry = {}

        if global_comm is not None:
            self.global_rank = global_comm.Get_rank()
            self.global_registry = {**_global_obj_reg, **_extra}
            self._check_builtin_clashes(_global_obj_reg, _extra, "global")
        if group_comm is not None:
            self.group_rank = group_comm.Get_rank()
            self.group_registry = {**_group_obj_reg, **_extra}
            self._check_builtin_clashes(_group_obj_reg, _extra, "group")

        self.use_local()

    # Registry validation
    def _check_obj_clash(self, extra_ops: dict) -> None:
        """Raise if any extra_ops key clashes with a local object method."""
        clashes = set(extra_ops) & set(build_registry(self.local_obj))
        if clashes:
            raise ValueError(
                f"extra_ops clash with object methods: {clashes}. "
                f"Rename the extra_ops keys or remove the conflicting methods."
            )

    def _check_builtin_clashes(
        self, obj_reg: dict, extra_ops: dict, mode_name: str
    ) -> None:
        """Raise if any operation name (obj or extra) shadows a builtin."""
        clashes = (set(obj_reg) | set(extra_ops)) & set(self._builtins_op)
        if clashes:
            raise ValueError(
                f"{mode_name} registry clashes with builtin operations: {clashes}. "
                f"Rename the conflicting methods or extra_ops keys."
            )

    def _check_obj_registries(
        self, local_reg: dict, global_reg: dict, group_reg: dict
    ) -> None:
        """Raise if a mode with an object exposes different methods from local.

        Modes where obj=None have an empty obj-registry and are not checked,
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

    # Mode switching
    def _switch_mode(self, mode: str, comm: "MPI.COMM", registry: dict) -> None:
        self.mode = mode
        self.comm = comm
        self.rank = comm.Get_rank()
        self.registry = registry

    def use_local(self) -> None:
        """Switch to local mode."""
        self._switch_mode("local", self.local_comm, self.local_registry)

    def use_global(self) -> None:
        """Switch to global mode. No-op if no global_comm."""
        if self.global_comm is None:
            return
        self._switch_mode("global", self.global_comm, self.global_registry)

    def use_group(self) -> None:
        """Switch to group mode. No-op if worker has no group_comm."""
        if self.group_comm is None:
            return
        self._switch_mode("group", self.group_comm, self.group_registry)

    # Lifecycle
    def start(self) -> None:
        """Enter the message loop. Blocks until ``shutdown`` is dispatched."""
        self._is_alive = True
        self._loop()

    def _all_objs(self) -> list[object]:
        """Flatten local, global, and group objects into a single list."""

        def _to_list(o):
            if o is None:
                return []
            if isinstance(o, list):
                return o
            return [o]

        return (
            _to_list(self.local_obj)
            + _to_list(self.global_obj)
            + _to_list(self.group_obj)
        )

    def shutdown(self) -> None:
        """Stop the message loop and call .close() on all objects that support it."""
        self._is_alive = False
        for o in self._all_objs():
            if hasattr(o, "close"):
                o.close()

    # Internal loop and dispatch
    def _loop(self) -> None:
        """Main loop. All ranks run this until ``_is_alive`` is False."""
        while self._is_alive:
            if self.rank == 0:
                msg = self._read_messages()
            else:
                msg = None

            msg = self.comm.bcast(msg, root=0)
            r = self._dispatch(msg)

            # Fire-and-forget senders mark their message "reply": False.
            if self.rank == 0 and msg.get("reply", True):
                self._send_result(r)

            if not self._is_alive:
                break

    def _send_result(self, r: DispatchResult) -> None:
        """Send dispatch result back to world_comm rank 0. Called only on local rank 0."""
        if r.status == DispatchStatus.SILENT:
            # A reply was requested: unblock the caller with a void status.
            self.world_comm.send(
                {"type": "status", "value": {"has_result": False, "error": None}},
                dest=0,
                tag=self._TAG_STATUS,
            )
            return
        if r.status == DispatchStatus.SUCCESS:
            self.world_comm.send(
                {
                    "type": "status",
                    "value": {"has_result": r.value is not None, "error": None},
                },
                dest=0,
                tag=self._TAG_STATUS,
            )
            if r.value is not None:
                # The status above already promised a result, so the receiver is
                # committed to a tag-RESULT message: an unserialisable value must
                # still produce one, or the Session blocks forever.
                try:
                    self.world_comm.send(
                        {"type": "result", "value": r.value},
                        dest=0,
                        tag=self._TAG_RESULT,
                    )
                except Exception as e:
                    self.world_comm.send(
                        {
                            "type": "result",
                            "value": None,
                            "error": f"operation succeeded but its result could not "
                            f"be returned: {type(e).__name__}: {e}",
                        },
                        dest=0,
                        tag=self._TAG_RESULT,
                    )
        elif r.status == DispatchStatus.ERROR:
            self.world_comm.send(
                {"type": "status", "value": {"has_result": False, "error": r.error}},
                dest=0,
                tag=self._TAG_STATUS,
            )

    def _read_messages(self) -> dict:
        """Read one message from world_comm on rank 0. Blocks until a message arrives."""
        return self.world_comm.recv(source=MPI.ANY_SOURCE, tag=self._TAG_CMD)

    def _dispatch(self, msg: dict) -> DispatchResult:
        """Dispatch a message to the appropriate handler on all ranks.

        Message format
        --------------
        {"type": <op_name>, "args": <positional payload>, "value": <payload>}

        ``args`` (absent when the caller passed none) is forwarded as positional
        arguments, ahead of the kwargs built from ``value``:
          - absent / None    → no positional arguments
          - tuple / list     → used directly
          - anything else    → wrapped as a single positional argument

        ``value`` is mapped to kwargs as follows:
          - absent / None  → no kwargs
          - dict           → used directly as kwargs
          - scalar         → wrapped as {"value": scalar}

        Builtins (use_local, use_group, use_global, shutdown, list_ops) are
        dispatched without a barrier and return SILENT unless they produce a
        result. Registry operations are bracketed by a comm.barrier() on all
        ranks so that MPI collectives inside handlers are safe. extra_ops
        receive ``self.comm`` as their first positional argument, followed by
        ``args``.
        """
        op_type = msg.get("type")

        raw_args = msg.get("args")
        if raw_args is None:
            args = ()
        elif isinstance(raw_args, (tuple, list)):
            args = tuple(raw_args)
        else:
            args = (raw_args,)

        value = msg.get("value")
        if value is None:
            kwargs = {}
        elif isinstance(value, dict):
            kwargs = value
        else:
            kwargs = {"value": value}

        if op_type in self._builtins_op:
            try:
                result = self._builtins_op[op_type](*args, **kwargs)
            except Exception as e:
                return DispatchResult(DispatchStatus.ERROR, error=str(e))
            if result is not None:
                return DispatchResult(DispatchStatus.SUCCESS, value=result)
            return DispatchResult(DispatchStatus.SILENT)

        handler = self.registry.get(op_type)
        if handler is None:
            return DispatchResult(
                DispatchStatus.ERROR,
                error=f"Unknown operation '{op_type}'. Available: {list(self._builtins_op) + list(self.registry)}",
            )

        self.comm.barrier()
        try:
            if op_type in self._extra_op_names:
                result = handler(self.comm, *args, **kwargs)
            else:
                result = handler(*args, **kwargs)
            return DispatchResult(DispatchStatus.SUCCESS, value=result)
        except Exception as e:
            return DispatchResult(DispatchStatus.ERROR, error=str(e))
        finally:
            self.comm.barrier()

    def list_ops(self, mode: str = "local") -> list[str]:
        registries = {
            "local": self.local_registry,
            "global": self.global_registry,
            "group": self.group_registry,
        }
        if mode not in registries:
            raise ValueError(
                f"Unknown mode '{mode}'. Expected one of: {list(registries)}"
            )
        return list(registries[mode])

    def __repr__(self) -> str:
        return (
            f"Worker(\n"
            f"  mode     = {self.mode!r},\n"
            f"  ops      = {list(self.registry)},\n"
            f"  builtins = {list(self._builtins_op)}\n"
            f")"
        )
