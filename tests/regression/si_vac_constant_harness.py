"""In-process ``examples/Si_vac`` constant-mode run shared by develop and the endpoint.

A real pyKMC run builds its engines through ``EngineManagerFactory``, whose base
``ManagerFactory.__init__`` raises ``ValueError("Not enough MPI ranks ...")``
when ``COMM_WORLD`` has fewer than ``n_sessions + 1`` ranks: rank 0 is the
manager and every worker session needs a rank of its own. Plain ``pytest`` is a
single rank, so the factory cannot be used there. :class:`InProcessEngineManager`
stands in for the manager of a two-rank ``n_sessions = 1`` run. It holds the two
engines that rank 1 hosts in that run, the worker's local engine
(``engine_id = 1``) and the group engine (``engine_id = 0``; ``group_size``
defaults to ``size - 1 = 1``), both built as ``Engine.create(style,
config=config.lammps, comm=MPI.COMM_SELF, engine_id=...)``, and it runs every
operation synchronously, in submission order. That is the order in which the
single worker thread of the real manager dispatches them: ``broadcast`` and
``submit_group`` drain the local queue first.

What crosses the MPI boundary in a real run is reproduced: keyword arguments
and results are pickled and unpickled; an engine exception reaches the caller
as a ``RuntimeError``, raised by ``broadcast``/``submit_group`` or attached to
the ``Future`` returned by ``submit``; and the engines draw from Python/NumPy
global random state kept apart from the manager's, as on a separate rank.

The harness uses only interfaces present on both develop ``c6f2824`` and
``SK_HTST_endpoint`` (``Engine.create``, ``build_registry``,
``Config.from_ini_file`` and ``KMC`` with its ``_initialize``/``run`` entry
points), so the same file records the develop oracle (``PYTHONPATH=<develop
tree>``) and runs the endpoint check. It only observes: the wrapped ``KMC``
methods forward their arguments and return values untouched and draw no random
numbers.

Develop has no ``[Control] seed``; the endpoint's only calls ``random.seed`` and
``np.random.seed`` in ``KMC.__init__``. The INI therefore sets no ``seed`` and
the harness seeds both global generators immediately before constructing
``KMC``, identically on both trees. ``zseed`` goes in ``[pARTn]``.

Usage
-----
::

    PYTHONPATH=<tree> python si_vac_constant_harness.py --root <tree> --out r.json
"""

from __future__ import annotations

import argparse
import concurrent.futures
import contextlib
import hashlib
import inspect
import json
import logging
import os
import pickle
import platform
import random
import shlex
import subprocess
import sys
import tempfile
import time
from collections import Counter
from collections.abc import Callable, Iterator
from pathlib import Path
from typing import Any

import numpy as np

SI_VAC_INPUT = """[Control]
initial_config = initial_config.xyz
n_steps = {n_steps}
engine = lammps
verbosity = 1
n_sessions = 1

[Lammps]
pair_style = sw
pair_coeff = * * Si.sw Si
min_style = cg
minimize = 1e-10 1e-12 10000 10000

[AtomicEnvironment]
style = diamond/graph
rnei = 3.0
rcut = 6.3

[EventSearch]
style = partn
nsearch = {nsearch}
emax_event = 3.5
emin_event = 0.0

[pARTn]
zseed = {zseed}

[RateConstant]
style = constant
k0 = 10
T = 300.0

[PSR]
style = ira

[IRA]
"""
"""Input of the run; ``initial_config.xyz`` and ``Si.sw`` are copied next to it."""

EXAMPLE_FILES = ("initial_config.xyz", "Si.sw")
KMC_LOGGERS = ("log", "output", "info", "events", "progress")
CONTROL_OPS = frozenset({"use_local", "use_group", "use_global", "shutdown"})
PICKLE_PROTOCOL = pickle.HIGHEST_PROTOCOL
ENV_KEYS = (
    "PYTHONHASHSEED",
    "PYTHONPATH",
    "PYTHONNOUSERSITE",
    "PYTHONDONTWRITEBYTECODE",
    "DYLD_LIBRARY_PATH",
    "OMP_NUM_THREADS",
)
PYKMC_OUT_RTOL = 1e-6
"""``pykmc.out`` prints 7 significant digits (``{:e}``): at most 5e-7 relative."""


def _transport(value: Any) -> Any:
    """Return a copy of ``value`` as it arrives on the other side of an MPI send.

    Parameters
    ----------
    value : Any
        Keyword arguments or a result.

    Returns
    -------
    Any
        ``pickle.loads(pickle.dumps(value))``, what ``comm.send``/``comm.recv`` do.

    """
    return pickle.loads(pickle.dumps(value, protocol=PICKLE_PROTOCOL))


def _np_state_equal(a: tuple, b: tuple) -> bool:
    """Return True when two ``np.random.get_state()`` tuples are identical."""
    return a[0] == b[0] and np.array_equal(a[1], b[1]) and tuple(a[2:]) == tuple(b[2:])


class InProcessEngineManager:
    """Synchronous single-process stand-in for the manager of a two-rank run.

    Parameters
    ----------
    engine_style : str
        ``[Control] engine`` (``"lammps"``).
    engine_config : Any
        ``config.lammps``, forwarded to ``Engine.create`` as in
        ``EngineManagerFactory._make_engine``.

    Attributes
    ----------
    local_engine, group_engine : Engine
        The worker's local (``engine_id = 1``) and group (``engine_id = 0``) engines.
    calls : Counter
        Number of dispatches per ``(mode, operation)``.
    engine_rng_used : bool
        True if an engine operation drew from the Python or NumPy global
        generator (on a real worker rank such draws come from an unseeded
        process-local state).

    """

    def __init__(self, engine_style: str, engine_config: Any) -> None:
        from mpi4py import MPI
        from pykmc.engine import Engine
        from pykmc.manager.worker import build_registry

        # Worker(local_obj=..., group_obj=...) builds the local engine first.
        self.local_engine = Engine.create(
            engine_style, config=engine_config, comm=MPI.COMM_SELF, engine_id=1
        )
        self.group_engine = Engine.create(
            engine_style, config=engine_config, comm=MPI.COMM_SELF, engine_id=0
        )
        # EngineManagerFactory passes global_obj=None: no global operations.
        self._registries: dict[str, dict[str, Callable]] = {
            "local": build_registry(self.local_engine),
            "group": build_registry(self.group_engine),
            "global": {},
        }
        self.mode = "local"
        self.calls: Counter = Counter()
        self.engine_rng_used = False
        self._started = False
        self._closed = False
        self._engine_py_state = random.Random(0).getstate()
        self._engine_np_state = np.random.RandomState(0).get_state()

    # -- dispatch ----------------------------------------------------------

    def _check(self, op_name: str) -> None:
        """Refuse worker control builtins and calls after shutdown, synchronously."""
        if op_name in CONTROL_OPS:
            raise ValueError(
                f"'{op_name}' is a Worker control builtin and cannot be submitted."
            )
        if self._closed:
            raise RuntimeError("Manager has been shut down.")

    def _dispatch(self, mode: str, op_name: str, kwargs: dict[str, Any]) -> Any:
        """Run one operation on the engine of ``mode`` as a worker rank would.

        Parameters
        ----------
        mode : str
            ``"local"``, ``"group"`` or ``"global"``.
        op_name : str
            Operation name (``list_ops`` or a registry entry).
        kwargs : dict
            Keyword arguments, transported before the call.

        Returns
        -------
        Any
            The transported result.

        Raises
        ------
        RuntimeError
            For an unknown operation or when the engine raises.

        """
        self.mode = mode
        self.calls[(mode, op_name)] += 1
        payload = _transport(kwargs)
        if op_name == "list_ops":  # Worker builtin, answered before the registry
            return list(self._registries[payload.get("mode", "local")])
        handler = self._registries[mode].get(op_name)
        if handler is None:
            raise RuntimeError(
                f"Unknown operation '{op_name}'. Available: "
                f"{['list_ops', *self._registries[mode]]}"
            )
        manager_py_state, manager_np_state = random.getstate(), np.random.get_state()
        random.setstate(self._engine_py_state)
        np.random.set_state(self._engine_np_state)
        try:
            result = handler(**payload)
        except Exception as exc:
            raise RuntimeError(
                f"Operation '{op_name}' failed on the in-process {mode} engine: "
                f"{type(exc).__name__}: {exc}"
            ) from exc
        finally:
            py_after, np_after = random.getstate(), np.random.get_state()
            if py_after != self._engine_py_state or not _np_state_equal(
                np_after, self._engine_np_state
            ):
                self.engine_rng_used = True
            self._engine_py_state, self._engine_np_state = py_after, np_after
            random.setstate(manager_py_state)
            np.random.set_state(manager_np_state)
        return _transport(result)

    # -- Manager surface ---------------------------------------------------

    def start(self) -> None:
        """Mark the manager started (the real one starts its worker threads)."""
        if self._started:
            raise RuntimeError("Manager is already started.")
        self._started = True

    def broadcast(self, op_name: str, **kwargs: Any) -> None:
        """Run ``op_name`` on the one local session; raise on failure."""
        self._check(op_name)
        self._dispatch("local", op_name, kwargs)

    def submit(self, op_name: str, **kwargs: Any) -> concurrent.futures.Future:
        """Run ``op_name`` on the local engine and return a resolved Future."""
        self._check(op_name)
        if not self._started:
            raise RuntimeError("Manager has not been started. Call start() first.")
        future: concurrent.futures.Future = concurrent.futures.Future()
        future.set_running_or_notify_cancel()
        try:
            value = self._dispatch("local", op_name, kwargs)
        except Exception as exc:
            future.set_exception(exc)
        else:
            future.set_result(value)
        return future

    def submit_group(self, op_name: str, **kwargs: Any) -> Any:
        """Run ``op_name`` on the group engine and return its result."""
        self._check(op_name)
        return self._dispatch("group", op_name, kwargs)

    def submit_global(self, op_name: str, **kwargs: Any) -> Any:
        """Global mode holds no engine (``global_obj=None``): only ``list_ops``."""
        self._check(op_name)
        return self._dispatch("global", op_name, kwargs)

    def list_ops(self, mode: str = "local") -> list[str]:
        """Return the operation names of ``mode``'s registry."""
        return self.submit("list_ops", mode=mode).result()

    def shutdown(self) -> None:
        """Close both engines once, as ``Worker.shutdown`` does; idempotent."""
        if self._closed:
            return
        self._closed = True
        failures = []
        for engine in (self.local_engine, self.group_engine):
            try:
                engine.close()
            except Exception as exc:
                failures.append(
                    f"engine {engine.engine_id}: {type(exc).__name__}: {exc}"
                )
        if failures:
            raise RuntimeError("shutdown: close() failed: " + "; ".join(failures))

    def __getattr__(self, name: str) -> Callable[..., Any]:
        """Return ``submit``/``submit_group``/``submit_global`` wrappers by name."""
        if name.startswith("_"):
            raise AttributeError(
                f"'{type(self).__name__}' object has no attribute '{name}'"
            )
        if name.startswith("global_"):
            op = name[len("global_") :]
            return lambda **kw: self.submit_global(op, **kw)
        if name.startswith("group_"):
            op = name[len("group_") :]
            return lambda **kw: self.submit_group(op, **kw)
        return lambda **kw: self.submit(name, **kw)


class _Observer:
    """Record what a run selects, without touching its inputs, outputs or draws."""

    WRAPPED = (
        "get_new_environments",
        "central_atoms_research",
        "_select_event",
        "_reconstruction_active_event",
        "reconstruction",
        "_save_restart_file",
        "_close",
    )

    def __init__(self) -> None:
        self.steps: list[dict[str, Any]] = []
        self.restart: tuple[int, float] | None = None
        self.reference_rows: list[dict[str, Any]] | None = None
        self.close_calls: list[dict[str, str]] = []

    def _step(self) -> dict[str, Any]:
        if not self.steps:
            self.steps.append({})
        return self.steps[-1]

    @contextlib.contextmanager
    def attached(self, kmc_cls: type) -> Iterator[None]:
        """Wrap the ``KMC`` methods in :attr:`WRAPPED` for the duration of the block."""
        orig = {name: kmc_cls.__dict__[name] for name in self.WRAPPED}
        obs = self

        def get_new_environments(kmc: Any) -> Any:
            result = orig["get_new_environments"](kmc)
            obs.steps.append({"new_environments": [str(e) for e in result]})
            return result

        def central_atoms_research(
            kmc: Any, new_environments: Any, nsearch: Any
        ) -> Any:
            result = orig["central_atoms_research"](kmc, new_environments, nsearch)
            obs._step()["central_atoms"] = [int(a) for a in result]
            return result

        def _select_event(kmc: Any, active_table: Any) -> Any:
            result = orig["_select_event"](kmc, active_table)
            obs._step().setdefault("draws", []).append(
                [int(result[0]), float(result[1]), float(result[2])]
            )
            return result

        def _reconstruction_active_event(kmc: Any, idx: Any, active_table: Any) -> Any:
            row = active_table.table.loc[idx]
            attempt = {
                "label": int(idx),
                "atom_index": int(row["atom_index"]),
                "energy_barrier": float(row["energy_barrier"]),
                "num_reference_event": int(row["num_reference_event"]),
            }
            result = orig["_reconstruction_active_event"](kmc, idx, active_table)
            attempt["ok"] = bool(result.is_ok())
            obs._step().setdefault("attempts", []).append(attempt)
            return result

        def reconstruction(kmc: Any, active_table: Any) -> Any:
            result = orig["reconstruction"](kmc, active_table)
            obs._step().update(
                delta_t_ps=float(result[1]),
                k_tot=float(result[2]),
                selected_label=int(result[3]),
                failed_references=[int(r) for r in result[4]],
                n_active=int(len(active_table.table)),
            )
            return result

        def _save_restart_file(kmc: Any, last_step: Any, last_time: Any) -> Any:
            obs.restart = (int(last_step), float(last_time))
            return orig["_save_restart_file"](kmc, last_step, last_time)

        def _close(kmc: Any, *args: Any, **kwargs: Any) -> Any:
            if obs.reference_rows is None and kmc.reference_table is not None:
                table = kmc.reference_table.table
                obs.reference_rows = [
                    {
                        "idx_ref": int(r.idx_ref),
                        "idx_backward": int(r.idx_backward),
                        "energy_barrier": float(r.energy_barrier),
                        "event_id": str(r.event_id),
                    }
                    for r in table[
                        ["idx_ref", "idx_backward", "energy_barrier", "event_id"]
                    ].itertuples(index=False)
                ]
            obs.close_calls.append(
                {
                    "args": repr(args),
                    "kwargs": repr(kwargs),
                    "after_step": len(obs.steps),
                }
            )
            return orig["_close"](kmc, *args, **kwargs)

        wrappers = {
            "get_new_environments": get_new_environments,
            "central_atoms_research": central_atoms_research,
            "_select_event": _select_event,
            "_reconstruction_active_event": _reconstruction_active_event,
            "reconstruction": reconstruction,
            "_save_restart_file": _save_restart_file,
            "_close": _close,
        }
        try:
            for name, wrapper in wrappers.items():
                setattr(kmc_cls, name, wrapper)
            yield
        finally:
            for name, original in orig.items():
                setattr(kmc_cls, name, original)


def _reference_pairs(rows: list[dict[str, Any]]) -> list[list[float | None]]:
    """Return the sorted ``[dE_forward, dE_backward]`` pair of every catalogue row.

    ``dE_forward`` is the row's ``energy_barrier``; ``dE_backward`` is the
    ``energy_barrier`` of the row its ``idx_backward`` names (``None`` if that
    row is absent). A forward/backward couple therefore contributes both
    orientations, and a self-backward row contributes ``[dE, dE]``.
    """
    barrier = {r["idx_ref"]: r["energy_barrier"] for r in rows}
    pairs = [[r["energy_barrier"], barrier.get(r["idx_backward"])] for r in rows]
    return sorted(pairs, key=lambda p: (p[0], float("-inf") if p[1] is None else p[1]))


def _pykmc_out_last_row(path: Path) -> dict[str, float] | None:
    """Return step, ``T(s)`` and ``k_tot`` of the last full table row of ``pykmc.out``."""
    if not path.is_file():
        return None
    last = None
    for line in path.read_text().splitlines():
        tokens = line.split()
        if len(tokens) >= 8 and tokens[0].isdigit():
            last = tokens
    if last is None:
        return None
    return {"step": int(last[0]), "T": float(last[2]), "k_tot": float(last[6])}


def _git(root: Path, *args: str) -> str | None:
    try:
        proc = subprocess.run(
            ["git", "--no-optional-locks", "-C", str(root), *args],
            capture_output=True,
            text=True,
            check=True,
        )
    except (OSError, subprocess.CalledProcessError):
        return None
    return proc.stdout.rstrip("\n")


def _versions() -> dict[str, str | None]:
    versions: dict[str, str | None] = {"python": sys.version, "numpy": np.__version__}
    for name in ("mpi4py", "pandas", "lammps"):
        try:
            versions[name] = str(__import__(name).__version__)
        except Exception:
            versions[name] = None
    try:
        from mpi4py import MPI

        # First field only ("Open MPI v5.0.7"): the rest names the build host.
        versions["mpi_library"] = MPI.Get_library_version().split(",")[0].strip()
    except Exception:
        versions["mpi_library"] = None
    return versions


def _placeholders(**paths: Path | str) -> list[tuple[str, str]]:
    """Map local paths (as given and resolved) and the home directory to placeholders.

    Parameters
    ----------
    **paths : Path or str
        ``name=path``; every occurrence of ``path`` becomes ``<name>``.

    Returns
    -------
    list of (str, str)
        ``(path, placeholder)`` pairs, longest path first.

    """
    pairs = {}
    for name, path in paths.items():
        for variant in (str(path), str(Path(path).resolve())):
            pairs[variant] = f"<{name}>"
    pairs.setdefault(str(Path.home()), "<home>")
    return sorted(pairs.items(), key=lambda kv: -len(kv[0]))


def _scrub(value: Any, replacements: list[tuple[str, str]]) -> Any:
    """Replace local paths in every string of ``value`` (the oracle is committed).

    Parameters
    ----------
    value : Any
        A JSON-like structure.
    replacements : list of (str, str)
        ``(path, placeholder)`` pairs from :func:`_placeholders`.

    Returns
    -------
    Any
        ``value`` with each path replaced by its placeholder.

    """
    if isinstance(value, str):
        for old, new in replacements:
            value = value.replace(old, new)
        return value
    if isinstance(value, dict):
        return {k: _scrub(v, replacements) for k, v in value.items()}
    if isinstance(value, list):
        return [_scrub(v, replacements) for v in value]
    return value


def _source_sha256(package_dir: Path) -> str:
    """Return a SHA-256 over the relative paths and contents of ``package_dir/**/*.py``.

    Identifies the code that ran even when the tree has uncommitted edits.
    """
    digest = hashlib.sha256()
    for path in sorted(package_dir.rglob("*.py")):
        digest.update(path.relative_to(package_dir).as_posix().encode() + b"\0")
        digest.update(path.read_bytes() + b"\0")
    return digest.hexdigest()


def _close_kmc_log_handlers() -> None:
    """Detach and close the handlers ``LogKMC`` attached to the KMC loggers."""
    for name in KMC_LOGGERS:
        logger = logging.getLogger(name)
        for handler in list(logger.handlers):
            logger.removeHandler(handler)
            try:
                handler.close()
            except Exception:
                pass


def run_si_vac_constant(
    root: Path,
    workdir: Path,
    n_steps: int = 10,
    seed: int = 12345,
    zseed: int = 12345,
    nsearch: int = 2,
) -> dict[str, Any]:
    """Run ``examples/Si_vac`` in constant mode in-process and summarise it.

    Parameters
    ----------
    root : Path
        Repository root whose ``pykmc`` must be the one imported, and whose
        ``examples/Si_vac`` provides ``initial_config.xyz`` and ``Si.sw``.
    workdir : Path
        Fresh (absent or empty) run directory; pyKMC, LAMMPS and pARTn write
        into the current directory, which is switched to it during the run.
    n_steps : int, optional
        KMC steps (default 10).
    seed : int, optional
        Seed of the Python ``random`` and NumPy global generators, applied
        immediately before ``KMC`` is constructed (default 12345).
    zseed : int, optional
        ``[pARTn] zseed`` (default 12345).
    nsearch : int, optional
        ``[EventSearch] nsearch`` (default 2).

    Returns
    -------
    dict
        ``k_tot`` (ps^-1) and ``T`` (s) after the last step, ``reference_pairs``
        (sorted ``[dE_forward, dE_backward]`` in eV of the final catalogue),
        ``selected`` (``[atom_index, dE]`` of the executed event of each step),
        ``n_steps`` (steps executed), ``exit_code``, per-step diagnostics
        ``steps`` and ``meta``.

    Raises
    ------
    RuntimeError
        If the imported ``pykmc`` is not under ``root``, ``workdir`` is not
        empty, or the recorded time disagrees with the run's own outputs.

    """
    import pykmc

    root = Path(root).resolve()
    workdir = Path(workdir).resolve()
    pykmc_file = Path(pykmc.__file__).resolve()
    if not pykmc_file.is_relative_to(root):
        raise RuntimeError(f"imported pykmc {pykmc_file} is not under --root {root}")
    workdir.mkdir(parents=True, exist_ok=True)
    if any(workdir.iterdir()):
        raise RuntimeError(f"run directory {workdir} is not empty")
    example = root / "examples" / "Si_vac"
    sha256 = {}
    for name in EXAMPLE_FILES:
        data = (example / name).read_bytes()
        sha256[name] = hashlib.sha256(data).hexdigest()
        (workdir / name).write_bytes(data)
    ini_text = SI_VAC_INPUT.format(n_steps=n_steps, nsearch=nsearch, zseed=zseed)
    (workdir / "input.in").write_text(ini_text)

    from pykmc.config import Config
    from pykmc.kmc import KMC

    observer = _Observer()
    exit_code: Any = "no SystemExit"
    previous_cwd = os.getcwd()
    started = time.perf_counter()
    os.chdir(workdir)
    try:
        config = Config.from_ini_file("input.in")
        if config.rateconstant.style != "constant":
            raise RuntimeError("the harness supports the constant style only")
        manager = InProcessEngineManager(config.control.engine, config.lammps)
        manager.start()
        takes_manager = "manager" in inspect.signature(KMC.__init__).parameters
        try:
            with observer.attached(KMC):
                random.seed(seed)
                np.random.seed(seed)
                if takes_manager:
                    kmc = KMC(config, manager=manager)
                else:  # develop: KMC(config), manager assigned afterwards
                    kmc = KMC(config)
                    kmc.manager = manager
                try:
                    kmc._initialize()
                    kmc.run()
                except SystemExit as exc:
                    exit_code = exc.code
        finally:
            manager.shutdown()
            _close_kmc_log_handlers()
    finally:
        os.chdir(previous_cwd)
    wall_time = time.perf_counter() - started

    selected = []
    total_time = 0.0
    for step in observer.steps:
        ok = [a for a in step.get("attempts", []) if a["ok"]]
        if not ok:
            break
        selected.append([ok[-1]["atom_index"], ok[-1]["energy_barrier"]])
        total_time += step["delta_t_ps"] * 10**-12  # KMC.run's own accumulation
    executed = observer.steps[: len(selected)]
    k_tot = executed[-1]["k_tot"] if executed else None
    out_row = _pykmc_out_last_row(workdir / "pykmc.out")
    checks: dict[str, Any] = {
        "T_accumulated_s": total_time,
        "pykmc_out_last_row": out_row,
    }
    if observer.restart is not None:
        checks["restart"] = list(observer.restart)
        if observer.restart[1] != total_time:
            raise RuntimeError(
                f"T passed to _save_restart_file {observer.restart[1]!r} != "
                f"accumulated delta_t {total_time!r}"
            )
    if out_row is not None and executed:
        for key, value in (("T", total_time), ("k_tot", k_tot)):
            rel = abs(out_row[key] - value) / abs(value)
            checks[f"{key}_vs_pykmc_out_rel"] = rel
            if out_row["step"] != len(selected) or rel > PYKMC_OUT_RTOL:
                raise RuntimeError(
                    f"pykmc.out last row {out_row} disagrees with the observed "
                    f"{key}={value!r} after step {len(selected)}"
                )
    rows = observer.reference_rows or []
    meta = {
        "pykmc_file": str(pykmc_file),
        "root": str(root),
        "git_head": _git(root, "rev-parse", "HEAD"),
        "git_tree": _git(root, "rev-parse", "HEAD^{tree}"),
        "git_status_short": (_git(root, "status", "--short") or "").splitlines(),
        "pykmc_source_sha256": _source_sha256(pykmc_file.parent),
        "pythonhashseed": os.environ.get("PYTHONHASHSEED"),
        "platform": platform.platform(),
        "versions": _versions(),
        "seeds": {
            "python": seed,
            "zseed": zseed,
            "seeding": "random.seed(python) and np.random.seed(python) immediately "
            "before KMC construction; no [Control] seed (absent on develop); "
            "zseed in [pARTn]",
        },
        "nsearch": nsearch,
        "n_steps_requested": n_steps,
        "ini": ini_text,
        "example_sha256": sha256,
        "kmc_constructor": "KMC(config, manager=manager)"
        if takes_manager
        else "KMC(config); kmc.manager = manager",
        "adapter": "InProcessEngineManager: synchronous stand-in for the manager of "
        "a 2-rank n_sessions=1 run (local engine_id=1, group engine_id=0, "
        "MPI.COMM_SELF, pickled transport, engine-side RNG state isolated)",
        "engine_rng_used": manager.engine_rng_used,
        "manager_calls": {
            f"{m}:{op}": n for (m, op), n in sorted(manager.calls.items())
        },
        "close_calls": observer.close_calls,
        "n_reference_rows": len(rows),
        "wall_time_s": wall_time,
        "workdir": str(workdir),
    }
    meta = _scrub(
        meta, _placeholders(workdir=workdir, root=root, python=sys.executable)
    )
    return {
        "k_tot": k_tot,
        "T": total_time,
        "reference_pairs": _reference_pairs(rows),
        "reference_rows": rows,
        "selected": selected,
        "n_steps": len(selected),
        "exit_code": exit_code,
        "checks": checks,
        "steps": observer.steps,
        "meta": meta,
    }


def main(argv: list[str] | None = None) -> int:
    """Run the harness from the command line and write its JSON result.

    Parameters
    ----------
    argv : list of str, optional
        Arguments (default ``sys.argv[1:]``).

    Returns
    -------
    int
        0 on success.

    """
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--root", type=Path, required=True, help="tree whose pykmc runs"
    )
    parser.add_argument("--out", type=Path, required=True, help="JSON result path")
    parser.add_argument(
        "--workdir",
        type=Path,
        default=None,
        help="new run directory (default: mkdtemp)",
    )
    parser.add_argument("--n-steps", type=int, default=10)
    parser.add_argument("--seed", type=int, default=12345)
    parser.add_argument("--zseed", type=int, default=12345)
    parser.add_argument("--nsearch", type=int, default=2)
    args = parser.parse_args(argv)
    workdir = args.workdir or Path(tempfile.mkdtemp(prefix="si-vac-constant-"))
    result = run_si_vac_constant(
        args.root,
        workdir,
        n_steps=args.n_steps,
        seed=args.seed,
        zseed=args.zseed,
        nsearch=args.nsearch,
    )
    replacements = _placeholders(
        workdir=workdir,
        root=args.root,
        out=args.out,
        harness=Path(__file__),
        python=sys.executable,
        cwd=os.getcwd(),
    )
    command = [sys.executable, __file__, *(sys.argv[1:] if argv is None else argv)]
    result["meta"]["command"] = _scrub(shlex.join(command), replacements)
    result["meta"]["cwd"] = "<cwd>"
    env = {key: os.environ.get(key) for key in ENV_KEYS}
    if env["DYLD_LIBRARY_PATH"] is not None:
        env["DYLD_LIBRARY_PATH"] = "<set: LAMMPS build directory>"
    result["meta"]["env"] = _scrub(env, replacements)
    text = json.dumps(result, indent=1, sort_keys=True) + "\n"
    markers = (str(Path.home()), tempfile.gettempdir(), "/private/", "/tmp/")
    leaked = [m for m in markers if m in text]
    if leaked:
        raise RuntimeError(f"local paths left in the result: {leaked}")
    args.out.write_text(text)
    print(
        f"[si_vac_constant] root={result['meta']['root']} "
        f"head={result['meta']['git_head']} steps={result['n_steps']} "
        f"k_tot={result['k_tot']!r} T={result['T']!r} exit={result['exit_code']!r} "
        f"wall={result['meta']['wall_time_s']:.1f}s -> {args.out}",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
