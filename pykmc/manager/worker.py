from __future__ import annotations
from typing import Callable, Any
from dataclasses import dataclass
from enum import Enum
from mpi4py import MPI
import inspect
import logging
import types

logger = logging.getLogger(__name__)

_MISSING = object()

# Worker builtins, in dispatch order. The control builtins are *reserved*: the
# Manager refuses to submit them (see ``RESERVED_OPS``); ``list_ops`` is the only
# builtin that returns a value and stays reachable through ``Manager.list_ops``.
# ``Worker._builtins_op`` is built from this tuple, so the two cannot diverge.
BUILTIN_OPS: tuple[str, ...] = (
    "use_local",
    "use_group",
    "use_global",
    "shutdown",
    "list_ops",
)
RESERVED_OPS: frozenset[str] = frozenset(BUILTIN_OPS) - {"list_ops"}

# Routine types that are safe to bind with ``getattr``: binding applies the
# descriptor protocol but computes nothing. Deliberately narrower than
# ``inspect.isroutine``, which also accepts every ``__get__``-only descriptor
# (``functools.cached_property``, ``partialmethod``, ``singledispatchmethod``,
# custom lazy descriptors) and would let discovery evaluate them.
_ROUTINE_TYPES: tuple[type, ...] = (
    types.FunctionType,
    types.BuiltinFunctionType,
    types.MethodType,
    types.MethodDescriptorType,
    types.ClassMethodDescriptorType,
    types.WrapperDescriptorType,
    types.MethodWrapperType,
)


def is_static_callable(raw: Any) -> bool:
    """Return True if a statically looked-up attribute denotes a callable operation.

    ``raw`` is the object returned by :func:`inspect.getattr_static`, i.e. the
    attribute exactly as stored on the class or instance, with no descriptor
    protocol applied. Only an explicit allow-list qualifies: plain and builtin
    functions, bound methods, the C-level method/wrapper descriptors, and
    ``classmethod``/``staticmethod`` wrappers, plus callable objects whose type
    has no ``__get__``. Every other descriptor (``property``,
    ``functools.cached_property``, ``partialmethod``, custom ``__get__``
    objects) computes a value on access and is rejected, so it is never
    evaluated during discovery. ``inspect.isroutine`` is not used because it
    accepts any ``__get__``-only descriptor.

    Parameters
    ----------
    raw : Any
        Attribute as returned by ``inspect.getattr_static``.

    Returns
    -------
    bool

    """
    if isinstance(raw, (classmethod, staticmethod)):
        return True
    if isinstance(raw, _ROUTINE_TYPES):
        return True
    return callable(raw) and not hasattr(type(raw), "__get__")


def build_registry(obj: object | list[object] | None = None) -> dict[str, Callable]:
    """Build an operation registry from one or more objects.

    Names are taken from ``dir(o)`` so that dynamically exposed methods (e.g.
    ``EngineExtension`` methods advertised by ``Engine.__dir__``) are seen, but
    every name is first inspected statically with :func:`inspect.getattr_static`
    so that ``@property`` getters and other value-computing descriptors are never
    evaluated. Bound methods, classmethods, staticmethods and callables stored on
    the instance are collected; a name that is neither on the class nor on the
    instance is resolved through the object's own ``__getattr__`` (this is how
    engine extension methods are reached).

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
        for name in dir(o):
            if name.startswith("_"):
                continue
            raw = inspect.getattr_static(o, name, _MISSING)
            if raw is _MISSING:
                # Advertised by __dir__ but stored neither on the class nor on the
                # instance: delegated dynamically (Engine.__getattr__ screens the
                # extension class itself, so no property is evaluated here).
                try:
                    attr = getattr(o, name)
                except AttributeError:
                    continue
                if not callable(attr):
                    continue
            elif is_static_callable(raw):
                attr = getattr(o, name)  # binds the method; evaluates nothing
                if not callable(attr):
                    continue  # defensive: a routine always binds to a callable
            else:
                continue
            if name in registry:
                raise ValueError(
                    f"Operation '{name}' is defined on multiple objects passed to build_registry."
                )
            registry[name] = attr
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


def _describe(e: BaseException) -> str:
    """Format an exception as ``Type: message`` so an empty message stays visible."""
    return f"{type(e).__name__}: {e}"


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
            names (use_local, use_group, use_global, shutdown, list_ops).
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
        for comm in filter(
            None, [self.world_comm, local_comm, global_comm, group_comm]
        ):
            comm.Set_errhandler(MPI.ERRORS_RETURN)

        # Derived from BUILTIN_OPS so RESERVED_OPS (imported by the Manager)
        # always matches the control builtins this Worker actually dispatches.
        self._builtins_op: dict[str, Callable] = {
            name: getattr(self, name) for name in BUILTIN_OPS
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
    def _switch_mode(
        self, mode: str, comm: "MPI.COMM", registry: dict, rank: int
    ) -> None:
        self.mode = mode
        self.comm = comm
        self.rank = rank
        self.registry = registry

    def use_local(self) -> None:
        """Switch to local mode."""
        self._switch_mode(
            "local", self.local_comm, self.local_registry, self.local_rank
        )

    def use_global(self) -> None:
        """Switch to global mode. No-op if no global_comm."""
        if self.global_comm is None:
            return
        self._switch_mode(
            "global", self.global_comm, self.global_registry, self.global_rank
        )

    def use_group(self) -> None:
        """Switch to group mode. No-op if worker has no group_comm."""
        if self.group_comm is None:
            return
        self._switch_mode(
            "group", self.group_comm, self.group_registry, self.group_rank
        )

    # Lifecycle
    def start(self) -> None:
        """Enter the message loop. Blocks until ``shutdown`` is dispatched."""
        self._is_alive = True
        try:
            self._loop()
        except BaseException as e:
            if self.local_rank == 0:
                try:
                    self.world_comm.send(
                        {
                            "type": "status",
                            "value": {"has_result": False, "error": str(e)},
                        },
                        dest=0,
                        tag=self._TAG_STATUS,
                    )
                except Exception:
                    pass
            MPI.COMM_WORLD.Abort(1)
            raise

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
        """Stop the message loop and call ``close()`` on every object that has one.

        Every object is attempted even if an earlier ``close()`` raises: the
        failures are collected and re-raised together once, so the dispatch
        boundary reports (or logs, for the fire-and-forget control message)
        every object that stayed open instead of only the first one.

        Raises
        ------
        RuntimeError
            Listing every ``close()`` that raised, after all were attempted.

        """
        self._is_alive = False
        failures: list[str] = []
        for o in self._all_objs():
            if not hasattr(o, "close"):
                continue
            try:
                o.close()
            except Exception as e:
                failures.append(f"{type(o).__name__}.close(): {_describe(e)}")
        if failures:
            raise RuntimeError(
                f"Worker {self.worker_id} shutdown: {len(failures)} close() "
                "call(s) failed: " + "; ".join(failures)
            )

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

            if self.rank == 0:
                # Fire-and-forget senders (Session._send_command) mark their
                # message "reply": False; every other message gets one reply.
                self._send_result(r, msg.get("type"), reply=msg.get("reply", True))

            if not self._is_alive:
                break

    def _send_status(self, has_result: bool, error: str | None) -> None:
        """Send one status message to world_comm rank 0."""
        self.world_comm.send(
            {"type": "status", "value": {"has_result": has_result, "error": error}},
            dest=0,
            tag=self._TAG_STATUS,
        )

    def _send_result(
        self, r: DispatchResult, op_type: str | None, reply: bool = True
    ) -> None:
        """Reply to world_comm rank 0 with exactly one terminal message.

        Called only on rank 0 of the active communicator.

        Parameters
        ----------
        r : DispatchResult
            Outcome computed by ``_dispatch``.
        op_type : str | None
            Operation name, used in error messages.
        reply : bool
            False for fire-and-forget messages: nothing is sent, but an error is
            logged so it is not dropped silently.

        Notes
        -----
        A ``SUCCESS`` result is pickle-checked *before* any status is sent, so a
        value that cannot be serialised turns into an ``ERROR`` status and the
        caller is never left waiting for a result message that cannot follow.
        A ``SILENT`` result (void builtin) still gets a void status when a reply
        was requested.

        """
        if not reply:
            if r.status == DispatchStatus.ERROR:
                logger.error(
                    "Fire-and-forget operation '%s' failed on worker %s: %s",
                    op_type,
                    self.worker_id,
                    r.error,
                )
            return
        if r.status == DispatchStatus.ERROR:
            self._send_status(False, r.error)
            return
        if r.status == DispatchStatus.SILENT or r.value is None:
            self._send_status(False, None)
            return
        try:
            MPI.pickle.dumps(r.value)
        except Exception as e:
            self._send_status(
                False,
                f"Operation '{op_type}' succeeded but its result could not be "
                f"pickled: {type(e).__name__}: {e}",
            )
            return
        self._send_status(True, None)
        self.world_comm.send(
            {"type": "result", "value": r.value}, dest=0, tag=self._TAG_RESULT
        )

    def _read_messages(self) -> dict:
        """Read one message from world_comm on rank 0. Blocks until a message arrives."""
        return self.world_comm.recv(source=MPI.ANY_SOURCE, tag=self._TAG_CMD)

    def _dispatch(self, msg: dict) -> DispatchResult:
        """Dispatch a message to the appropriate handler on all ranks.

        Message format
        --------------
        {"type": <op_name>, "value": <payload>, "reply": <bool>}

        ``reply`` is read by ``_loop`` (not here): ``False`` marks a
        fire-and-forget control message (``Session._send_command``) and no
        reply is sent; a message without the field is treated as
        ``reply: True`` and always receives exactly one terminal reply.

        ``value`` is mapped to kwargs as follows:
          - absent / None  → no kwargs
          - dict           → used directly as kwargs
          - scalar         → wrapped as {"value": scalar}

        Builtins (use_local, use_group, use_global, shutdown, list_ops) are
        dispatched without a barrier, inside the same error boundary as registry
        operations, and return SILENT unless they produce a result. Registry
        operations are bracketed by a comm.barrier() on all ranks so that MPI
        collectives inside handlers are safe; inside that bracket every rank
        gathers its error string on rank 0, so a failure on a non-root rank is
        reported to the caller (the value returned is still rank 0's). extra_ops
        receive ``self.comm`` as their first positional argument.
        """
        op_type = msg.get("type")

        value = msg.get("value")
        if value is None:
            kwargs = {}
        elif isinstance(value, dict):
            kwargs = value
        else:
            kwargs = {"value": value}

        if op_type in self._builtins_op:
            try:
                result = self._builtins_op[op_type](**kwargs)
            except Exception as e:
                return DispatchResult(DispatchStatus.ERROR, error=_describe(e))
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
                result = handler(self.comm, **kwargs)
            else:
                result = handler(**kwargs)
            r = DispatchResult(DispatchStatus.SUCCESS, value=result)
        except Exception as e:
            r = DispatchResult(DispatchStatus.ERROR, error=_describe(e))
        # Collective reached by every rank on both paths: the root learns about
        # failures on the other participating ranks before anyone replies.
        errors = self.comm.gather(r.error, root=0)
        if self.rank == 0:
            failed = [(i, e) for i, e in enumerate(errors) if e is not None]
            if failed:
                ranks = ", ".join(str(i) for i, _ in failed)
                details = "; ".join(f"rank {i}: {e}" for i, e in failed)
                r = DispatchResult(
                    DispatchStatus.ERROR,
                    error=(
                        f"Operation '{op_type}' failed on rank(s) {ranks} of the "
                        f"{self.mode} communicator: {details}"
                    ),
                )
        self.comm.barrier()
        return r

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
