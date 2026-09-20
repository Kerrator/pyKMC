from __future__ import annotations
from lammps import lammps
import numpy as np
import ctypes
import functools
import os
import warnings
from contextlib import contextmanager
from dataclasses import dataclass, replace
from typing import Any, Protocol
from .base import Engine
from ase.cell import Cell
from ase.data import atomic_masses, atomic_numbers

try:
    from mpi4py import MPI
except ImportError:
    pass
try:
    import pypARTn
except ImportError:
    pypARTn = None

from ..activevolume.active_volume import (
    ActiveVolumeSaddleError,
    partn_search_AV,
    partn_refine_AV,
    position_results_AV,
    require_orthorhombic_cell,
)
from ..atomic_environment import AtomicEnvironment
from ..physics import (
    EnginePhysics,
    ForceModel,
    ResolvedConstraints,
    validate_event_constraints,
    _indices,
)
from ..result import (
    ErrorInfo,
    EventSearchOutput,
    EventRefinementOutput,
    Ok,
    Err,
    ErrorType,
)

try:
    from lammps import LAMMPSException as _LAMMPSException

    _LAMMPS_EXCEPTIONS = (_LAMMPSException,)
except ImportError:
    # The installed `lammps` Python module does not export LAMMPSException
    # (e.g. lammps 20250722 exposes only MPIAbortException and raises a plain
    # Exception from `command`): the decorator is then a no-op and a LAMMPS
    # error propagates as that plain exception without closing the engine.
    _LAMMPS_EXCEPTIONS = ()


def lammps_error_handler(method):
    """Close the engine and re-raise as RuntimeError on a LAMMPSException.

    Only active when the installed ``lammps`` module exports
    ``LAMMPSException`` (``_LAMMPS_EXCEPTIONS`` is non-empty); otherwise the
    wrapped method's exceptions propagate unchanged and the engine stays open.
    A LAMMPS build without exception support calls MPI_Abort instead of
    raising at all. Closing drops the native handle but retains any pending
    restoration and its full-system descriptor for a later retry.
    """

    @functools.wraps(method)
    def wrapper(self, *args, **kwargs):
        try:
            return method(self, *args, **kwargs)
        except _LAMMPS_EXCEPTIONS as e:
            self.close()
            raise RuntimeError(
                f"[LammpsEngine] LAMMPS error in `{method.__name__}`: {e}"
            ) from e

    return wrapper


# ----------------------------------------------------------------------
# Species / type / mass rule (one authoritative implementation)
# ----------------------------------------------------------------------


def species_map(
    types: list[str] | np.ndarray,
) -> tuple[tuple[str, ...], tuple[float, ...]]:
    """Return the potential species order and the masses LAMMPS is given.

    This is the default rule for chemical symbols without an explicit map:
    species are ``sorted(set(types))`` (alphabetical),
    species ``i`` (0-based) is LAMMPS type ``i + 1`` and its mass is the ASE
    standard atomic mass in amu. ``LammpsEngine.initialize_system`` and the
    standalone active-volume helpers both use it. Real active-volume crops
    instead reuse the initialized full-system descriptor, preserving explicit
    species order, absent slots and post-potential masses.

    Consequently the elements of a multi-element ``pair_coeff`` **must be
    listed in this alphabetical order when this default map is used**
    (for example ``* * NiFeCr.eam Cr Ni``
    for a Ni/Cr system), and ``types`` must be the *full* system's symbols:
    a crop that holds only a subset of the species still needs the full map.

    Parameters
    ----------
    types : list[str] | np.ndarray
        Chemical symbols, one per atom (full system).

    Returns
    -------
    tuple[tuple[str, ...], tuple[float, ...]]
        ``(species, masses)``: plain tuples of ``str`` and ``float`` in LAMMPS
        type order, ready to be stored on an ``HTSTEventRequest``.

    Raises
    ------
    ValueError
        If ``types`` is empty or contains an unknown chemical symbol.

    """
    symbols = [str(t) for t in types]
    if not symbols:
        raise ValueError("species_map: `types` must contain at least one atom")
    species = tuple(sorted(set(symbols)))
    try:
        masses = tuple(float(atomic_masses[atomic_numbers[s]]) for s in species)
    except KeyError as exc:
        raise ValueError(f"species_map: unknown chemical symbol {exc}") from exc
    return species, masses


def types_to_int(types: list[str] | np.ndarray, species: tuple[str, ...]) -> np.ndarray:
    """Map chemical symbols to 1-based LAMMPS integer types.

    Parameters
    ----------
    types : list[str] | np.ndarray
        Chemical symbols, one per atom (any subset of the full system).
    species : tuple[str, ...]
        Species order from ``species_map`` (built from the *full* system).

    Returns
    -------
    np.ndarray
        ``int32`` array, ``species.index(symbol) + 1`` for every entry.

    Raises
    ------
    ValueError
        If a symbol in ``types`` is not in ``species``.

    """
    lookup = {s: i + 1 for i, s in enumerate(species)}
    try:
        return np.array([lookup[str(t)] for t in types], dtype=np.int32)
    except KeyError as exc:
        raise ValueError(
            f"types_to_int: symbol {exc} is not in the species map {species}"
        ) from exc


def _validate_species_override(
    species: tuple[str, ...], masses: tuple[float, ...]
) -> tuple[tuple[str, ...], tuple[float, ...]]:
    """Check an explicit ``(species, masses)`` map for ``initialize_system``.

    The override lets a scratch instance that holds only a subset of the atoms
    (an HTST zone crop) keep the *full* system's type numbering and masses, so
    an integer type means the same species in every instance built from the
    same full system and a multi-element ``pair_coeff`` still matches
    ``create_box``.

    Parameters
    ----------
    species : tuple[str, ...]
        Potential species order (``pair_coeff`` order); non-empty, unique.
    masses : tuple[float, ...]
        One finite positive mass in amu per species, in ``species`` order.

    Returns
    -------
    tuple[tuple[str, ...], tuple[float, ...]]
        ``(species, masses)`` as plain tuples of ``str`` and ``float``.

    Raises
    ------
    ValueError
        If either sequence is malformed or their lengths differ.

    """
    if isinstance(species, str) or not species:
        raise ValueError("initialize_system: species must be a non-empty sequence")
    species_t = tuple(str(s) for s in species)
    if len(set(species_t)) != len(species_t):
        raise ValueError(f"initialize_system: species contains duplicates: {species_t}")
    if len(masses) != len(species_t):
        raise ValueError(
            f"initialize_system: masses has {len(masses)} entries but species has "
            f"{len(species_t)}"
        )
    masses_t = []
    for symbol, mass in zip(species_t, masses, strict=True):
        if isinstance(mass, (bool, np.bool_)):
            raise ValueError(f"initialize_system: mass of {symbol!r} must be a number")
        value = float(mass)
        if not np.isfinite(value) or value <= 0.0:
            raise ValueError(
                f"initialize_system: mass of {symbol!r} must be finite and > 0, "
                f"got {mass!r}"
            )
        masses_t.append(value)
    return species_t, tuple(masses_t)


def _require_finite_positions(positions: np.ndarray, op_name: str) -> np.ndarray:
    """Reject non-finite positions before they reach a collective LAMMPS call.

    A NaN/inf coordinate makes LAMMPS raise a *per-rank* error ("Non-numeric
    atom coords") on the rank(s) owning the bad atom only. On a multi-rank
    engine the erroring rank unwinds to Python while the other ranks stay
    inside liblammps's own collectives, so the whole worker hangs. Every rank
    receives the same ``positions``, so raising here is symmetric: either all
    ranks raise together or none does.

    Parameters
    ----------
    positions : np.ndarray
        Candidate positions.
    op_name : str
        Operation name used in the error message.

    Returns
    -------
    np.ndarray
        ``positions`` as a float array.

    Raises
    ------
    ValueError
        If any coordinate is NaN or infinite.

    """
    arr = np.asarray(positions, dtype=np.float64)
    if not np.isfinite(arr).all():
        n_bad = int((~np.isfinite(arr)).sum())
        raise ValueError(
            f"{op_name}: positions contain {n_bad} non-finite value(s) (NaN/inf); "
            "refusing to scatter them to LAMMPS (a per-rank 'Non-numeric atom "
            "coords' error would desynchronise a multi-rank engine and hang it)."
        )
    return arr


def _require_positions(
    positions: np.ndarray, op_name: str, natoms: int | None = None
) -> np.ndarray:
    """Finite-position guard plus an optional ``(natoms, 3)`` shape check.

    ``scatter_atoms`` copies exactly ``3 * natoms`` doubles from the buffer it
    is given: a longer array is silently truncated (only the first ``natoms``
    atoms move) and a shorter one is read past its end (garbage or NaN
    coordinates, no error). ``natoms`` is the live instance's global atom
    count, so the raise is symmetric across ranks.

    Parameters
    ----------
    positions : np.ndarray
        Candidate positions.
    op_name : str
        Operation name used in the error message.
    natoms : int | None
        Expected atom count; ``None`` skips the shape check.

    Returns
    -------
    np.ndarray
        ``positions`` as a float array.

    Raises
    ------
    ValueError
        If any coordinate is non-finite or the shape is not ``(natoms, 3)``.

    """
    arr = _require_finite_positions(positions, op_name)
    if natoms is not None and arr.shape != (natoms, 3):
        raise ValueError(
            f"{op_name}: positions have shape {arr.shape}, expected "
            f"{(natoms, 3)} (the live instance has {natoms} atoms); refusing "
            "to scatter them to LAMMPS (a size mismatch is silently truncated "
            "or read past the buffer)"
        )
    return arr


def _normalize_periodic_positions(
    positions: np.ndarray, cell: np.ndarray, pbc: tuple[bool, bool, bool]
) -> np.ndarray:
    """Remove periodic lattice translations in the source ASE cell frame.

    Only periodic fractional components choose image shifts; nonperiodic
    directions remain physical displacements. Subtracting lattice vectors
    preserves atom ordering and avoids changing an already-normalized array.
    The caller's array is never modified, including read-only views.
    """
    cell = np.asarray(cell, dtype=float)
    periodic = np.asarray(pbc, dtype=bool)
    if cell.shape != (3, 3) or not np.isfinite(cell).all():
        raise ValueError("set_positions: cell must be a finite (3, 3) array")
    if periodic.shape != (3,):
        raise ValueError("set_positions: pbc must have three entries")
    try:
        fractional = np.linalg.solve(cell.T, positions.T).T
    except np.linalg.LinAlgError as exc:
        raise ValueError("set_positions: cell must be nonsingular") from exc
    if not np.isfinite(fractional).all():
        raise ValueError("set_positions: periodic coordinates are not finite")
    shifts = np.zeros_like(fractional)
    shifts[:, periodic] = np.floor(fractional[:, periodic])
    normalized = positions - shifts @ cell
    return _require_finite_positions(normalized, "set_positions (normalized)")


@dataclass(frozen=True, eq=False)
class FullSystem:
    """What ``LammpsEngine`` remembers about the last fully initialised system.

    Positions are deliberately not stored: they change every KMC step and the
    caller that needs a restore always has the current ones. ``cell`` is a
    private copy; ``pbc`` is a plain tuple.

    Attributes
    ----------
    types : tuple[str, ...]
        Chemical symbol of every atom, in engine (LAMMPS id - 1) order.
    species : tuple[str, ...]
        Potential species order (see ``species_map``). Together with
        ``masses`` it is the explicit map ``ensure_full_system`` replays, so
        a system initialised with a map larger than its own symbols is
        rebuilt with the same type numbering.
    masses : tuple[float, ...]
        Mass of each species in amu, in ``species`` order, as the live
        instance holds them: ``initialize_system`` records the ASE values it
        emitted and ``initialize_potential`` refreshes them from
        ``extract_atom("mass")`` after ``pair_coeff``, because a potential
        file that carries masses (``eam/alloy``, ``eam/fs``) overrides the
        emitted ones. Between the two calls they are the ASE values.
    cell : np.ndarray
        (3, 3) cell in the ASE frame.
    pbc : tuple[bool, bool, bool]
        Periodicity per axis.

    """

    types: tuple[str, ...]
    species: tuple[str, ...]
    masses: tuple[float, ...]
    cell: np.ndarray
    pbc: tuple[bool, bool, bool]
    physics: EnginePhysics | None = None

    @property
    def natoms(self) -> int:
        """Number of atoms in the full system."""
        return len(self.types)


class LammpsConfigProtocol(Protocol):
    """
    Protocol defining the configuration interface for LammpsEngine.

    Attributes
    ----------
    pair_style : str
        LAMMPS pair_style command string (e.g. "eam/alloy").
    pair_coeff : str
        LAMMPS pair_coeff command string (e.g. "* * potential.eam Ni").
    min_style : str
        Minimization algorithm (e.g. "cg", "fire").
    minimize : str
        Minimization convergence parameters (e.g. "1e-6 1e-8 1000 10000").
    frz_min : str
        Minimization convergence parameters used when core atoms are frozen.
    verbosity : int
        Log verbosity. 0 disables log file, any other value enables it.
    """

    pair_style: str
    pair_coeff: str
    min_style: str
    minimize: str
    frz_min: str
    verbosity: int


class LammpsEngine(Engine):
    """
    LAMMPS engine wrapper.

    This class is designed to be used in a master-worker MPI pattern.
    All LAMMPS commands are executed collectively across all ranks,
    but data extraction methods (get_positions, get_total_energy, etc.)
    only return values on rank 0, other ranks return None.

    Parameters
    ----------
    config : LammpsConfigProtocol
        Configuration object containing simulation parameters:
        pair_style, pair_coeff, min_style, minimize, verbosity.
    comm : MPI.COMM, optional
        MPI communicator for parallel execution. If None, runs in serial.
    engine_id : int, optional
        Unique identifier for this engine instance, used for log file naming.
        Default is 0.

    Notes
    -----
    Coordinate systems
        LAMMPS requires the simulation cell to be lower triangular
        (see https://docs.lammps.org/Howto_triclinic.html).
        For non-orthorhombic cells, positions are rotated from the ASE
        coordinate system to the LAMMPS coordinate system via a rotation
        matrix Q computed from `Cell.standard_form()`. The inverse rotation
        is applied when retrieving positions from LAMMPS.
        For orthorhombic cells, no rotation is needed and Q is not defined.

        `self.Q` is set during `initialize_system()`. All position data
        returned by `get_positions()` and accepted by `set_positions()` are
        in the ASE coordinate system — the rotation is handled internally.
        If you access LAMMPS positions directly (e.g. via `self.lmp`), you
        must apply the rotation manually using `_positions_to_lammps()` and
        `_positions_from_lammps()`.

    State ownership
        The live LAMMPS instance (`self.lmp`) holds *positions*, a *box*
        (cell, boundary), an *atom count* with integer types, per-type
        *masses* and the *potential*. Which operation mutates what, and who
        restores it:

        - `initialize_system` defines box, atoms, types and masses and records
          them in `self.full_system` (types, species, masses, cell, pbc; not
          positions). It is the only operation that changes the *identity* of
          the full system; nothing restores a previous one. `initialize_potential`
          completes it: it is the step that takes the cropped marker down.
        - `set_positions`, `minimize`, `minimize_with_results`,
          `minimize_freeze_core` and the energy getters with `positions=`
          mutate *positions only*. They never restore anything: the caller
          owns the positions it wants back and passes them to the next call.
        - `partn_search` / `partn_refine` without active volume mutate
          positions only (the search leaves the last pARTn geometry in the
          instance). Groups and fixes they create are removed on the success
          path. `lammps_error_handler` closes the engine on a LAMMPS error
          only when the installed `lammps` module exports `LAMMPSException`;
          otherwise (lammps 20250722 raises a plain `Exception`) the error
          propagates as-is and the engine stays open.
        - `partn_search` / `partn_refine` **with** active volume rebuild the
          instance as a crop (`activevolume.active_volume.reset`): box, atom
          count, types, masses and potential all change and the buffer shell
          is frozen. Both wrappers restore the full system on every exit path
          (Ok, Err, exception) through `ensure_full_system(positions)`, which
          replays `initialize_parameters` / `initialize_system` (with the
          remembered `species` / `masses` as the explicit map, so a
          subset-species system keeps its type numbering and explicit masses)
          / `initialize_potential` from `self.full_system` so the restored
          instance reproduces a fresh engine's energy for the same positions.
          A replay that raises leaves the instance marked cropped with the
          remembered descriptor intact, so a later `ensure_full_system`
          retries from the same map, starting a new native instance if the
          pending handle was closed. Replay verifies the remembered force
          snapshot and masses before reporting success. When the
          search/refine itself failed, a
          failure of the restore is reported as a `RuntimeWarning` and the
          *original* exception is the one raised (`system_is_cropped` then
          tells whether the engine is still a crop). The AV helpers clean up
          the groups/fixes/computes they create on their own failure paths;
          on success the `clear` of the restore removes them.
        - `system_is_cropped` reports whether the live instance still holds
          the remembered full system. Any `clear` issued through
          `self.command` (the AV helpers' path, and the start of a restore)
          marks the instance as replaced; only a completed
          `initialize_potential` clears that marker, and an atom-count
          mismatch is detected independently. Pending state survives handle
          closure; explicitly closing an intact instance remains a no-op for
          restoration.
        - Scratch work that must not disturb this engine (HTST Hessians)
          belongs in a separate `LammpsEngine(comm=MPI.COMM_SELF)`; nothing
          here caches state across instances.

    Examples
    --------
    Serial usage:
        engine = LammpsEngine(config=my_config)
        engine.start()
        engine.initialize_parameters()
        engine.initialize_system(types, positions, cell, pbc)
        engine.initialize_potential()

    MPI usage:
        from mpi4py import MPI
        comm = MPI.COMM_WORLD
        engine = LammpsEngine(config=my_config, comm=comm, engine_id=0)
        engine.start()
        ...
        pe = engine.get_total_energy()
        # ... only rank 0 receives data from extraction methods
    """

    name = "lammps"

    def __init__(
        self,
        config: LammpsConfigProtocol,
        comm: MPI.Intracomm | None = None,
        engine_id: int = 0,
    ) -> None:
        super().__init__()
        self.config = config
        self.comm = comm
        self.engine_id = engine_id
        self._is_orthorhombic = None
        self.lmp = None
        # Remembered full system (set by initialize_system) and whether the live
        # instance has been replaced by a crop since then.
        self.full_system: FullSystem | None = None
        self._cleared_since_init = False

    # Convenience
    @property
    def _is_rank0(self) -> bool:
        return self.comm is None or self.comm.Get_rank() == 0

    # NOTE: rank and command are exposed publicly to support the active volume
    # helpers (partn_search_AV / partn_refine_AV) until #60 is resolved.
    @property
    def rank(self) -> int:
        """MPI rank of this process (0 in serial)."""
        return 0 if self.comm is None else self.comm.Get_rank()

    def command(self, cmd: str) -> None:
        """Run a LAMMPS command string.

        A ``clear`` issued through this method (the active-volume helpers'
        path) marks the live instance as no longer holding the remembered full
        system; the marker stays up until ``initialize_potential`` completes.
        See ``system_is_cropped``.
        """
        if cmd.strip() == "clear":
            self._cleared_since_init = True
        self.lmp.command(cmd)
        if cmd.strip() == "clear":
            self._forget_velocity_callbacks(self.lmp)

    # ------------------------------------------------------------------
    # Full-system memory and restore
    # ------------------------------------------------------------------

    @property
    def system_is_cropped(self) -> bool:
        """True when the live instance does not hold the remembered full system.

        Either the cropped marker is up or the live atom count differs from
        the remembered one. The marker goes up on any ``clear`` issued through
        ``command`` (the active-volume crop path and the start of an
        ``ensure_full_system`` replay) and comes down only when
        ``initialize_potential`` completes, so an instance whose rebuild
        stopped short of the potential is still reported as cropped. The
        marker survives native handle closure, allowing a later restore to
        start a new instance. False before ``initialize_system`` or after
        explicitly closing an intact instance.
        """
        if self.full_system is None:
            return False
        if self._cleared_since_init:
            return True
        if self.lmp is None:
            return False
        return int(self.lmp.get_natoms()) != self.full_system.natoms

    def ensure_full_system(self, positions: np.ndarray | None = None) -> bool:
        """Rebuild the remembered full system if the live instance is cropped.

        Replays ``initialize_parameters`` / ``initialize_system`` /
        ``initialize_potential`` from ``self.full_system`` after a ``clear``,
        so a restored engine reproduces a freshly initialised one (same box,
        boundary, types, masses and potential) for the same positions. The
        remembered ``species`` / ``masses`` are passed to ``initialize_system``
        as the explicit map, so a system initialised with a species map larger
        than the symbols it holds (or with explicit masses) keeps its type
        numbering and per-type masses.

        The rebuild is complete only once ``initialize_potential`` has
        returned. If any replay step raises, the exception propagates, the
        instance stays reported as cropped (``system_is_cropped`` is True even
        though the atom count may already match) and ``self.full_system`` is
        the same descriptor as before, so calling this method again retries
        the rebuild from the same species/mass map, cell, pbc and types.

        A pending restore with a closed handle starts a new instance before
        replay. Start and clear failures obey the same retention rule. The
        remembered force snapshot and actual replayed species/masses must
        agree before restoration is complete; changed physics is an explicit
        failure, never a relabeled successful restore. An explicitly closed
        intact instance remains a no-op.

        Parameters
        ----------
        positions : np.ndarray | None
            Full-system positions (ASE frame) to create the atoms at. Required
            when a rebuild is needed; ignored otherwise (this method never
            moves atoms of an intact full system).

        Returns
        -------
        bool
            True if the system was rebuilt, False if it was already intact.

        Raises
        ------
        RuntimeError
            If ``initialize_system`` has not been called on this engine.
        ValueError
            If a rebuild is needed and ``positions`` is missing or has the
            wrong shape.
        Exception
            Whatever a replay step raises; see above for the state left
            behind.

        """
        if self.full_system is None:
            raise RuntimeError(
                "ensure_full_system: initialize_system has not been called, "
                "there is no remembered full system to restore"
            )
        if not self.system_is_cropped:
            return False
        fs = self.full_system
        if positions is None:
            raise ValueError(
                "ensure_full_system: the live LAMMPS instance is cropped and "
                "full-system positions are required to rebuild it"
            )
        positions = _require_positions(
            positions, "ensure_full_system", natoms=fs.natoms
        )
        try:
            # Latch even an atom-count-only mismatch before any operation can
            # close the native handle. Pending state outlives that handle.
            self._cleared_since_init = True
            if (
                fs.physics is not None
                and ForceModel.capture(self.config) != fs.physics.force_model
            ):
                raise RuntimeError(
                    "ensure_full_system: force-model physics changed since initialization"
                )
            if self.lmp is None:
                self.start()
            else:
                self.command("clear")
            self.initialize_parameters()
            self.initialize_system(
                types=fs.types,
                positions=positions,
                cell=Cell(fs.cell),
                pbc=fs.pbc,
                species=fs.species,
                masses=fs.masses,
            )
            self.initialize_potential()
            restored = self.full_system
            if restored.species != fs.species or restored.masses != fs.masses:
                raise RuntimeError(
                    "ensure_full_system: restored species/masses differ from the descriptor"
                )
            if fs.physics is not None and (
                restored.physics is None
                or restored.physics.force_model != fs.physics.force_model
            ):
                raise RuntimeError(
                    "ensure_full_system: restored force-model physics differs from the descriptor"
                )
        except BaseException:
            # Incomplete rebuild: keep the authoritative descriptor (the replay
            # may have re-recorded it) and the marker so the next call retries.
            self.full_system = fs
            self._cleared_since_init = True
            raise
        return True

    def _positions_to_lammps(self, positions: np.ndarray) -> np.ndarray:
        return positions @ self.Q.T if not self._is_orthorhombic else positions

    def _positions_from_lammps(self, positions: np.ndarray) -> np.ndarray:
        return positions @ self.Q if not self._is_orthorhombic else positions

    def _has_compute(self, compute_id: str) -> bool:
        """Check if a compute with the given id exists in LAMMPS."""
        return self.lmp.has_id("compute", compute_id)

    @lammps_error_handler
    def start(self) -> None:
        """Create a native handle, retaining any pending restoration state."""
        engine_log = (
            "none" if self.config.verbosity == 0 else f"lammps.log.{self.engine_id}"
        )
        self.lmp = lammps(
            comm=self.comm, cmdargs=["-screen", "none", "-log", engine_log]
        )

    def close(self) -> None:
        """Drop the handle while retaining the descriptor and pending marker."""
        if self.lmp is not None:
            self.lmp.close()
            self._forget_velocity_callbacks(self.lmp)
            self.lmp = None

    @lammps_error_handler
    def initialize_parameters(self) -> None:
        self.lmp.command("units metal")
        self.lmp.command("atom_style atomic")
        self.lmp.command("dimension 3")
        self.lmp.command("atom_modify map array")  #! necessary for scatter atoms
        self.lmp.command("atom_modify sort 0 0.0")  #! necessary for partn

    @lammps_error_handler
    def initialize_system(
        self,
        types: list[str] | np.ndarray[str],
        positions: np.ndarray,
        cell: Cell,
        pbc: list[bool] | np.ndarray[bool],
        *,
        species: tuple[str, ...] | None = None,
        masses: tuple[float, ...] | None = None,
    ) -> None:
        """Define the box, atoms, integer types and per-type masses.

        Parameters
        ----------
        types : list[str] | np.ndarray
            Chemical symbol of every atom.
        positions : np.ndarray
            ``(N, 3)`` positions in the ASE frame.
        cell : Cell
            Simulation cell (anything ``Cell.new`` accepts).
        pbc : list[bool] | np.ndarray
            Periodicity per axis.
        species : tuple[str, ...], optional
            Keyword-only. Explicit potential species order to use instead of
            ``species_map(types)``. Pass it together with ``masses`` when the
            instance holds only a subset of a larger system (an HTST zone
            crop) so the full system's type numbering and masses are kept;
            ``ensure_full_system`` passes the remembered map this way. When
            omitted the behaviour is exactly ``species_map(types)``.
        masses : tuple[float, ...], optional
            Keyword-only. Masses in amu in ``species`` order; required with
            ``species``.

        Raises
        ------
        ValueError
            If ``pbc`` is malformed, if only one of ``species``/``masses`` is
            given, if the override is malformed, or if a symbol in ``types``
            is not in the species map.

        Notes
        -----
        This records the descriptor but does not take the cropped marker
        down: after a ``clear`` the live instance counts as restored only
        once ``initialize_potential`` has completed (see ``system_is_cropped``).

        """
        # system parameters
        natoms = len(types)
        cell = Cell.new(cell)
        pbc = tuple(bool(p) for p in pbc)
        if len(pbc) != 3:
            raise ValueError(f"initialize_system: pbc must have 3 entries, got {pbc}")
        # To deal with Lammps convention if non orthonhombic cell
        self._is_orthorhombic = cell.orthorhombic
        if not self._is_orthorhombic:
            cell_lammps, self.Q = cell.standard_form()
        else:
            cell_lammps = np.array(cell)

        positions = self._positions_to_lammps(positions=positions)

        x = positions.flatten()  # Lammps format
        # cell
        xhi = cell_lammps[0, 0]
        yhi = cell_lammps[1, 1]
        zhi = cell_lammps[2, 2]
        # non diagonal terms
        xy = cell_lammps[1, 0]
        xz = cell_lammps[2, 0]
        yz = cell_lammps[2, 1]

        # boundary
        boundary = " ".join("p" if p else "f" for p in pbc)
        self.lmp.command(f"boundary {boundary}")

        ind = np.arange(1, natoms + 1)  # Lammps ids start at 1
        # One rule for species -> 1-based LAMMPS type and mass (see species_map):
        # alphabetical species order, ASE masses. pair_coeff must follow it. An
        # explicit (species, masses) pair carries a full system's map into a
        # scratch instance that holds only some of its atoms.
        if (species is None) != (masses is None):
            raise ValueError(
                "initialize_system: species and masses must be given together"
            )
        if species is None:
            species, masses = species_map(types)
        else:
            species, masses = _validate_species_override(species, masses)
        int_types = types_to_int(types, species).tolist()

        # lammps create system
        # ortho
        if np.allclose([xy, xz, yz], 0):
            self.lmp.command(f"region box block 0.0 {xhi} 0.0 {yhi} 0.0 {zhi}")
        # triclinic
        else:
            self.lmp.command(
                f"region box prism 0.0 {xhi} 0.0 {yhi} 0.0 {zhi} {xy} {xz} {yz}"
            )
        self.lmp.command("create_box {} box".format(len(species)))
        self.lmp.create_atoms(natoms, ind, int_types, x)
        # Set masses
        for i, mass in enumerate(masses):
            self.lmp.command("mass {} {}".format(i + 1, mass))
        # Label atoms name to type :
        self.lmp.command(
            "labelmap atom " + " ".join(f"{i + 1} {s}" for i, s in enumerate(species))
        )
        # Remember the full system so a cropped instance can be rebuilt. The
        # cropped marker is left as it is: only `initialize_potential` clears it.
        self.full_system = FullSystem(
            types=tuple(str(t) for t in types),
            species=species,
            masses=masses,
            cell=np.array(cell, dtype=np.float64, copy=True),
            pbc=(pbc[0], pbc[1], pbc[2]),
        )

    @lammps_error_handler
    def initialize_potential(self) -> None:
        """Replay ``pair_style`` / ``pair_coeff`` and complete the live system.

        Refreshes ``full_system.masses`` from the live instance (a potential
        file may override the emitted masses) and takes the cropped marker
        down, so ``system_is_cropped`` reports False only once the remembered
        full system has its potential back. Nothing before this point (a
        ``clear``, ``initialize_parameters``, ``initialize_system``) counts as
        a completed rebuild.
        """
        force_model = ForceModel.capture(self.config)
        self.lmp.command("pair_style {}".format(self.config.pair_style))
        self.lmp.command("pair_coeff {}".format(self.config.pair_coeff))
        self._refresh_full_system_masses()
        if ForceModel.capture(self.config) != force_model:
            raise RuntimeError("force-model contents changed during initialization")
        if self.full_system is not None:
            fs = self.full_system
            self.full_system = replace(
                fs, physics=EnginePhysics.capture(self.config, fs.species, fs.masses)
            )
        self._cleared_since_init = False

    def _refresh_full_system_masses(self) -> None:
        """Record the per-type masses the live instance holds after ``pair_coeff``.

        ``eam/alloy``-style potentials re-set the masses from the potential
        file, so ``full_system.masses`` (what an HTST request is built from)
        follows the live instance rather than the ASE values emitted earlier.
        Per-type masses are global, so every rank reads the same values.
        """
        fs = self.full_system
        if fs is None or self.lmp is None:
            return
        live = self.lmp.extract_atom("mass")
        if live is None:
            return
        masses = tuple(float(live[i + 1]) for i in range(len(fs.species)))
        if masses != fs.masses:
            self.full_system = replace(fs, masses=masses)

    @lammps_error_handler
    def get_positions(self) -> np.ndarray | None:
        result = self.lmp.gather_atoms("x", 1, 3)
        if self._is_rank0:
            # convert ctype positions into a numpy array
            result = np.ctypeslib.as_array(result)
            result = np.reshape(result, (-1, 3))
            return self._positions_from_lammps(positions=result)
        else:
            return None

    @lammps_error_handler
    def set_positions(self, positions: np.ndarray) -> None:
        """Scatter ASE-frame positions after normalizing periodic images.

        Equivalent images may be arbitrarily many supported cell translations
        away. Normalize in the remembered source cell before native rotation;
        nonperiodic lattice directions are never folded into the box.
        """
        # Symmetric finite + shape guard on every rank before the collective
        # scatter (get_natoms is the global count, so all ranks agree).
        positions = _require_positions(
            positions, "set_positions", natoms=int(self.lmp.get_natoms())
        )
        if self.full_system is None:
            raise RuntimeError("set_positions: initialize_system must be called first")
        positions = _normalize_periodic_positions(
            positions, self.full_system.cell, self.full_system.pbc
        )
        positions = self._positions_to_lammps(positions=positions)
        positions = _require_finite_positions(positions, "set_positions (native frame)")
        positions = positions.flatten().astype(np.float64)
        positions = np.ascontiguousarray(positions)
        c_array = (ctypes.c_double * len(positions))(*positions)
        self.lmp.scatter_atoms("x", 1, 3, c_array)

    @lammps_error_handler
    def get_forces(
        self, positions: np.ndarray | None = None, recompute: bool = True
    ) -> np.ndarray | None:
        """Return the forces on every atom in eV/Å, in the ASE frame.

        Like the energy getters this mutates *positions only* (when
        ``positions`` is given) and restores nothing.

        Parameters
        ----------
        positions : np.ndarray, optional
            ``(N, 3)`` positions to scatter first (ASE frame).
        recompute : bool, optional
            Run ``run 0 post no`` before gathering so the forces match the
            current positions. ``False`` returns the forces of the last run.

        Returns
        -------
        np.ndarray | None
            ``(N, 3)`` float64 copy of the forces on rank 0; ``None`` elsewhere.

        """
        if positions is not None:
            self.set_positions(positions=positions)
        if recompute:
            self.lmp.command("run 0 post no")
        result = self.lmp.gather_atoms("f", 1, 3)
        if self._is_rank0:
            forces = np.ctypeslib.as_array(result).reshape(-1, 3).astype(np.float64)
            return self._positions_from_lammps(positions=forces)
        return None

    @lammps_error_handler
    def get_total_energy(
        self, positions: np.ndarray = None, recompute: bool = True
    ) -> float | None:
        if positions is not None:
            self.set_positions(positions=positions)
        # Get total energy
        if recompute:
            self.lmp.command("run 0 post no")
        result = self.lmp.get_thermo("etotal")
        if self._is_rank0:
            return result
        else:
            return None

    @lammps_error_handler
    def get_potential_energy(
        self, positions: np.ndarray = None, recompute: bool = True
    ) -> float | None:
        if positions is not None:
            self.set_positions(positions=positions)

        # Check if compute exists
        define_compute = self._has_compute("c_pe")

        if not define_compute:
            self.lmp.command("compute c_pe all pe")

        # If run to get up-to-date value
        if recompute:
            self.lmp.command("run 0 post no")
        result = self.lmp.extract_compute("c_pe", 0, 0)

        if self._is_rank0:
            return result
        return None

    @lammps_error_handler
    def minimize(self, positions: np.ndarray = None) -> None:
        if positions is not None:
            self.set_positions(positions=positions)
        self.lmp.command("min_style {}".format(self.config.min_style))
        self.lmp.command("minimize {}".format(self.config.minimize))

    @lammps_error_handler
    def get_types(self) -> list[str]:
        # get_category_keywords does not exist — disabled temporarily
        # int_types = self.lmp.gather_atoms("type", 0, 1)
        # labels = self.lmp.get_category_keywords("typelabel")
        # return [labels[t - 1] for t in int_types]
        raise NotImplementedError("get_types is temporarily disabled")

    # ------------------------------------------------------------------
    # Frozen-atom helpers
    # ------------------------------------------------------------------

    def _make_frozen_group(
        self, config, positions=None, types=None, constraints=None
    ) -> bool:
        """Resolve frozen atoms from config and create g_frozen group. Returns True if any atoms are frozen."""
        if constraints is not None:
            frozen_indices = constraints.local_fixed_indices
            if not frozen_indices:
                return False
            lammps_ids = " ".join(str(i + 1) for i in frozen_indices)
            self.lmp.command(f"group g_frozen id {lammps_ids}")
            return True
        if config.frozen_atoms is None:
            return False
        if positions is None:
            positions = self.get_positions()
        if types is None and config.frozen_atoms.types:
            raise NotImplementedError(
                "frozen_atoms by type requires types — get_types is disabled"
            )
        # Compute frozen indices on rank 0 only (positions is None on other ranks
        # when falling back to get_positions()), then broadcast before the
        # collective lmp.command so all ranks participate.
        if self._is_rank0:
            frozen_ae = AtomicEnvironment(
                style="region",
                region=config.frozen_atoms,
                positions=positions,
                atom_types=types,
            )
            frozen_indices = frozen_ae.get_atoms_with_id("in")
        else:
            frozen_indices = None
        if self.comm is not None:
            frozen_indices = self.comm.bcast(frozen_indices, root=0)
        if not frozen_indices:
            return False
        lammps_ids = " ".join(str(i + 1) for i in frozen_indices)
        self.lmp.command(f"group g_frozen id {lammps_ids}")
        return True

    def _apply_frozen_fix(self, fix_name: str, atoms_frozen: bool) -> None:
        if atoms_frozen:
            self.lmp.command(f"fix {fix_name} g_frozen setforce 0.0 0.0 0.0")

    def _remove_frozen_fix(self, fix_name: str, atoms_frozen: bool) -> None:
        if atoms_frozen:
            self.lmp.command(f"unfix {fix_name}")

    def _delete_frozen_group(self, atoms_frozen: bool) -> None:
        if atoms_frozen:
            self.lmp.command("group g_frozen delete")

    def _forget_velocity_callbacks(self, native):
        """Release owned Python callbacks only after their native fixes are gone."""
        owned = getattr(self, "_velocity_callbacks", {})
        for name, handle in tuple(owned.items()):
            if handle is native:
                handle.callback.pop(name, None)
                del owned[name]

    def _operation_failures(self, error):
        """Agree on a recoverable Python failure before a native collective."""
        failure = repr(error) if error is not None else None
        return self.comm.allgather(failure) if self.comm is not None else [failure]

    def _raise_operation_failure(self, error, failures, phase):
        if error is not None:
            raise error
        if any(failure is not None for failure in failures):
            raise RuntimeError(f"pARTn {phase} failed on a worker: {failures}")

    @contextmanager
    def _fixed_velocity_guard(self, local_fixed_indices):
        """Hold fixed velocities after ARTn, without projecting coordinates.

        ARTn's perpendicular relaxation can replace native velocities even
        when a later setforce fix removes its trial forces. FIRE then integrates
        those velocities. This MIN_POST_FORCE callback closes that second
        channel using current local native tags, including after redistribution.
        """
        native = self.lmp
        fixed = _indices(local_fixed_indices, upper=int(native.get_natoms()))
        if not fixed:
            yield
            return
        native_ids = np.asarray(fixed, dtype=np.int64) + 1
        serial = getattr(self, "_velocity_serial", 0)
        while True:
            serial += 1
            name = f"pykmc_artn_velocity_{serial}"
            collision = native.has_id("fix", name) or name in native.callback
            collisions = (
                self.comm.allgather(collision) if self.comm is not None else [collision]
            )
            if not any(collisions):
                break
        self._velocity_serial = serial
        owned = getattr(self, "_velocity_callbacks", None)
        if owned is None:
            owned = self._velocity_callbacks = {}
        callback_errors = []
        callback_failures = []

        def hold_velocity(_caller, _step, nlocal, tags, _positions, added_force):
            # ctypes cannot propagate an exception out of this callback. Keep
            # the cause and reject the operation after native control returns.
            try:
                # Empty native ranks can receive None instead of empty arrays.
                if nlocal:
                    added_force.fill(0.0)
                    velocity = native.numpy.extract_atom("v", nelem=nlocal, dim=3)
                    if velocity is None or velocity.shape[0] < nlocal:
                        raise RuntimeError("native local velocities are unavailable")
                    selected = np.isin(np.asarray(tags), native_ids)
                    velocity[:nlocal][selected] = 0.0
            except BaseException as exc:
                if not callback_errors:
                    callback_errors.append(exc)
            # Every native rank calls this fix, including ranks with no atoms.
            # Retain errors until the configured finite native evaluation cap
            # returns. A forced timeout would poison subsequent minimizations.
            failures = self._operation_failures(
                callback_errors[0] if callback_errors else None
            )
            if any(failure is not None for failure in failures):
                if not callback_failures:
                    callback_failures.extend(failures)

        original = None
        try:
            failure = None
            if not callable(getattr(native, "set_fix_external_callback", None)):
                failure = RuntimeError(
                    "constrained pARTn requires LAMMPS fix external callbacks"
                )
            self._raise_operation_failure(
                failure, self._operation_failures(failure), "velocity preflight"
            )
            # Record ownership before allocation: a failing native command or
            # callback registration may leave a partially installed resource.
            owned[name] = native
            for phase, action in (
                (
                    "velocity fix allocation",
                    lambda: native.command(f"fix {name} all external pf/callback 1 1"),
                ),
                (
                    "velocity callback registration",
                    lambda: native.set_fix_external_callback(name, hold_velocity),
                ),
            ):
                failure = None
                try:
                    action()
                except BaseException as exc:
                    failure = exc
                self._raise_operation_failure(
                    failure, self._operation_failures(failure), phase
                )
            failure = None
            try:
                yield
            except BaseException as exc:
                failure = exc
            self._raise_operation_failure(
                failure, self._operation_failures(failure), "minimization"
            )
            if callback_errors:
                raise RuntimeError(
                    "pARTn fixed-velocity callback failed"
                ) from callback_errors[0]
            if callback_failures:
                raise RuntimeError(
                    f"pARTn fixed-velocity callback failed on a worker: {callback_failures}"
                )
        except BaseException as exc:
            original = exc
            raise
        finally:
            cleanup = None
            try:
                if self.lmp is native and native.has_id("fix", name):
                    native.command(f"unfix {name}")
                # A closed handle or successful unfix cannot call Python again.
                native.callback.pop(name, None)
                owned.pop(name, None)
            except BaseException as exc:
                cleanup = exc
            failures = self._operation_failures(cleanup)
            if any(failure is not None for failure in failures):
                self._cleared_since_init = True
                # Retain the callable while the native fix may still be live.
                if original is None:
                    self._raise_operation_failure(
                        cleanup, failures, "velocity unfix cleanup"
                    )
                add_note = getattr(BaseException, "add_note", None)
                if add_note is not None:
                    add_note(
                        original, f"Fixed-velocity unfix cleanup failed: {failures}"
                    )

    # ------------------------------------------------------------------
    # Minimization
    # ------------------------------------------------------------------

    @lammps_error_handler
    def minimize_with_results(
        self, positions=None, config=None, types=None, constraints=None
    ) -> tuple[np.ndarray, float] | None:
        """Minimize and return positions/energy.

        Explicit source-resolved constraints make this an endpoint transaction:
        return its result and restore entry coordinates, even after failure.
        The payload is authoritative; never reclassify its mask after a push.
        """
        if constraints is not None:
            return self._minimize_constrained(positions, constraints)
        if positions is not None:
            self.set_positions(positions=positions)
        atoms_frozen = (
            self._make_frozen_group(config, positions, types)
            if config is not None
            else False
        )
        self._apply_frozen_fix("f_frozen_min", atoms_frozen)
        self.minimize()
        self._remove_frozen_fix("f_frozen_min", atoms_frozen)
        self._delete_frozen_group(atoms_frozen)
        new_positions = self.get_positions()
        total_energy = self.get_total_energy(recompute=False)
        if self._is_rank0:
            return new_positions, total_energy
        else:
            return None

    def _minimize_constrained(self, positions, constraints):
        if not isinstance(constraints, ResolvedConstraints):
            raise ValueError("constraints must be a resolved source payload")
        n_atoms = int(self.lmp.get_natoms())
        constraints.validate(n_atoms)
        if constraints.cell is not None and (
            self.full_system is None
            or not np.array_equal(constraints.cell, self.full_system.cell)
            or constraints.pbc != tuple(self.full_system.pbc)
        ):
            raise ValueError("constraint source cell/PBC differs from native system")
        if positions is not None:
            constraints.validate_positions(positions)
        entry = self.get_positions()
        if self.comm is not None:
            entry = self.comm.bcast(entry, root=0)
        entry = np.array(entry, copy=True)
        proposed = entry if positions is None else positions
        constraints.validate_positions(proposed)
        proposed = constraints.protect_positions(proposed)
        # All ranks inspect identical native resource tables and use the same
        # deterministic, collision-checked names. Existing user resources stay.
        serial = getattr(self, "_constraint_serial", 0)
        while True:
            serial += 1
            name = f"pykmc_endpoint_{serial}"
            if not self.lmp.has_id("group", name) and not self.lmp.has_id("fix", name):
                break
        self._constraint_serial = serial
        resources = []
        original = None
        try:
            self.set_positions(proposed)
            if constraints.local_fixed_indices:
                ids = " ".join(str(i + 1) for i in constraints.local_fixed_indices)
                resources.append(("group", name, f"group {name} delete"))
                self.lmp.command(f"group {name} id {ids}")
                resources.append(("fix", name, f"unfix {name}"))
                self.lmp.command(f"fix {name} {name} setforce 0.0 0.0 0.0")
            self.minimize()
            result = self.get_positions()
            if self.comm is not None:
                result = self.comm.bcast(result, root=0)
            constraints.validate_positions(result)
            energy = self.get_total_energy(recompute=False)
            if self._is_rank0:
                return np.array(result, copy=True), energy
            return None
        except BaseException as exc:
            original = exc
            raise
        finally:
            failures = []
            if self.lmp is None:
                self._cleared_since_init = True
                failures.append(
                    RuntimeError(
                        "endpoint handle closed; full-system replay is pending and native user resources were lost"
                    )
                )
            else:
                for kind, resource, command in reversed(resources):
                    try:
                        if self.lmp.has_id(kind, resource):
                            self.lmp.command(command)
                    except Exception as exc:
                        failures.append(exc)
                try:
                    self.set_positions(entry)
                    self.lmp.command("run 0 post no")
                except Exception as exc:
                    self._cleared_since_init = True
                    failures.append(exc)
            if failures:
                self._cleared_since_init = True
                if original is None:
                    raise failures[0]
                add_note = getattr(BaseException, "add_note", None)
                if add_note is not None:
                    add_note(original, f"Endpoint cleanup failures: {failures!r}")

    @lammps_error_handler
    def minimize_freeze_core(self, core_idx) -> None:
        """Freeze directly translated atoms and minimize to relax surrounding atoms."""
        if core_idx is not None:
            core_ids = [idx + 1 for idx in core_idx]
            self.lmp.command(f"group frozen_group id {' '.join(map(str, core_ids))}")
            self.lmp.command("fix freeze frozen_group setforce 0.0 0.0 0.0")
            self.lmp.command(f"min_style {self.config.min_style}")
            self.lmp.command(f"minimize {self.config.frz_min}")
            self.lmp.command("unfix freeze")
            self.lmp.command("group frozen_group delete")

    # ------------------------------------------------------------------
    # pARTn search and refinement
    # ------------------------------------------------------------------

    def _destroy_artn(self, artn, original):
        """Release native run state even when a traceback retains its owner.

        The Python ARTn wrapper normally destroys process-local Fortran state
        in __del__. A retained exception retains the implementation frame and
        that wrapper, so garbage collection cannot define the next run's state.
        All result arrays have been copied before this cleanup runs.
        """
        failure = None
        if artn is not None:
            try:
                artn.destroy()
            except BaseException as exc:
                failure = exc
        failures = self._operation_failures(failure)
        if any(value is not None for value in failures):
            self._cleared_since_init = True
            if original is None:
                self._raise_operation_failure(
                    failure, failures, "native ARTn destruction"
                )
            add_note = getattr(BaseException, "add_note", None)
            if add_note is not None:
                add_note(original, f"Native ARTn destruction failed: {failures}")

    @contextmanager
    def _partn_resource_scope(self, active):
        """Clean full-system ARTn resources after a constrained failure.

        Active-volume calls have their own complete crop/replay transaction.
        Full-system calls retain their native handle and unrelated user fixes.
        """
        if active:
            yield
            return
        native = self.lmp
        resources = (
            ("fix", "10", "unfix 10"),
            ("fix", "f_frozen_post", "unfix f_frozen_post"),
            ("fix", "f_frozen_pre", "unfix f_frozen_pre"),
            ("fix", "freeze", "unfix freeze"),
            ("group", "g_frozen", "group g_frozen delete"),
            ("group", "frozen_group", "group frozen_group delete"),
        )
        occupied = [name for kind, name, _ in resources if native.has_id(kind, name)]
        failure = (
            ValueError(f"pARTn resource names already in use: {occupied}")
            if occupied
            else None
        )
        self._raise_operation_failure(
            failure, self._operation_failures(failure), "resource preflight"
        )
        entry = self.get_positions()
        if self.comm is not None:
            entry = self.comm.bcast(entry, root=0)
        entry = np.array(entry, copy=True)
        original = None
        try:
            try:
                yield
            except BaseException as exc:
                original = exc
            failures = self._operation_failures(original)
            if original is None and any(value is not None for value in failures):
                original = RuntimeError(f"pARTn failed on a worker: {failures}")
        finally:
            cleanup = None
            if self.lmp is not native:
                cleanup = RuntimeError("pARTn native handle closed during operation")
            else:
                for kind, name, command in resources:
                    try:
                        if native.has_id(kind, name):
                            native.command(command)
                    except BaseException as exc:
                        if cleanup is None:
                            cleanup = exc
            failures = self._operation_failures(cleanup)
            if any(value is not None for value in failures):
                self._cleared_since_init = True
                if original is None:
                    self._raise_operation_failure(cleanup, failures, "resource cleanup")
                add_note = getattr(BaseException, "add_note", None)
                if add_note is not None:
                    add_note(original, f"pARTn resource cleanup failed: {failures}")
            elif original is not None and not self._cleared_since_init:
                try:
                    self.set_positions(entry)
                except BaseException as exc:
                    cleanup = exc
                failures = self._operation_failures(cleanup)
                if any(value is not None for value in failures):
                    self._cleared_since_init = True
                    add_note = getattr(BaseException, "add_note", None)
                    if add_note is not None:
                        add_note(
                            original, f"pARTn position restoration failed: {failures}"
                        )
        if original is not None:
            raise original

    def _check_active_volume_inputs(
        self,
        config: Any,
        positions: np.ndarray | None,
        cell: np.ndarray | None,
        types: list[str] | np.ndarray | None,
        op_name: str,
    ) -> None:
        """Reject unsupported active-volume calls before any LAMMPS state changes.

        Raises
        ------
        ValueError
            If positions, cell or types are missing (the crop and the restore
            both need the full system). Resolved execution constraints are
            validated separately before the crop changes native state.

        """
        if positions is None or cell is None or types is None:
            raise ValueError(
                f"{op_name}: active_volume requires full-system positions, cell "
                "and types (needed for the crop and to restore the engine)"
            )
        require_orthorhombic_cell(cell, op_name)

    def _restore_after_failure(
        self, positions: np.ndarray, original: BaseException, op_name: str
    ) -> None:
        """Restore the full system after ``op_name`` raised ``original``.

        A failure of the restore itself must not replace the original
        exception (the first failure is the one to report), so it is turned
        into a ``RuntimeWarning`` naming both; ``system_is_cropped`` then
        tells the caller whether the engine is still a crop. A warning filter
        that promotes warnings to errors must also preserve the first failure.
        """
        try:
            self.ensure_full_system(positions)
        except Exception as restore_exc:  # noqa: BLE001 - secondary failure
            message = (
                f"[LammpsEngine] {op_name} raised {type(original).__name__}: "
                f"{original}; restoring the full system afterwards failed too "
                f"({restore_exc!r}). The original exception is raised; the "
                "engine may still be cropped (see system_is_cropped)."
            )
            try:
                warnings.warn(message, RuntimeWarning, stacklevel=3)
            except RuntimeWarning:
                # Python 3.10 has no exception notes; keep its original error too.
                add_note = getattr(BaseException, "add_note", None)
                if add_note is not None:
                    add_note(original, message)

    @lammps_error_handler
    def partn_search(
        self,
        config,
        central_atom_idx: int,
        positions=None,
        cell=None,
        types=None,
        constraints=None,
        user_constraints=None,
    ):
        """Run a pARTn event search around ``central_atom_idx``.

        Under ``config.control.active_volume`` the engine is rebuilt as the
        cropped active volume for the search and the full system is restored
        on every exit path (Ok, Err, exception); see the class docstring,
        "State ownership".
        """
        active = config.control.active_volume
        if active:
            self._check_active_volume_inputs(
                config, positions, cell, types, "partn_search"
            )
        constraints = self._resolve_search_constraints(
            config,
            central_atom_idx,
            positions,
            cell,
            types,
            constraints,
            user_constraints,
        )
        try:
            with self._partn_resource_scope(active):
                result = self._partn_search_impl(
                    config,
                    central_atom_idx,
                    positions,
                    cell,
                    types,
                    constraints,
                    user_constraints,
                )
                if not active:
                    self._validate_search_result(result, constraints)
        except BaseException as exc:
            if active:
                self._restore_after_failure(positions, exc, "partn_search")
            raise
        if active:
            self.ensure_full_system(positions)
            self._validate_search_result(result, constraints)
        return result

    def _resolve_search_constraints(
        self, config, center, positions, cell, types, constraints, user_constraints
    ):
        full = self.full_system
        if positions is None:
            positions = self.get_positions()
            if self.comm is not None:
                positions = self.comm.bcast(positions, root=0)
        if types is None and full is not None:
            types = full.types
        if cell is None and full is not None:
            cell = full.cell
        pbc = full.pbc if full is not None else (True, True, True)
        return validate_event_constraints(
            config,
            positions,
            types,
            cell,
            pbc,
            center,
            constraints,
            user_constraints=user_constraints,
        )

    def _validate_search_result(self, result, constraints):
        # Only the user-declared fixed atoms are a coordinate contract for a
        # returned event; the AV shell is a crop restriction held by setforce
        # (a refined saddle keeps a placed shell overlay). The output carries
        # that user set, so an HTST request built from it keeps the
        # free_radius Vineyard region and excludes user-frozen atoms only
        # (contracts 7f policy 5). The AV union stays engine-side transport.
        user = constraints.user_view()
        # All ranks take the same failure path; output arrays live on rank zero.
        failure = None
        if self._is_rank0 and result is not None and result.is_ok():
            output = result.ok_value()
            try:
                for field in ("min1_positions", "saddle_positions", "min2_positions"):
                    positions = getattr(output, field, None)
                    if positions is not None:
                        user.validate_positions(positions)
            except ValueError as exc:
                failure = str(exc)
            output.constraints = user
        if self.comm is not None:
            failure = self.comm.bcast(failure, root=0)
        if failure is not None:
            raise ValueError(
                f"pARTn returned an incompatible constrained event: {failure}"
            )

    def _partn_search_impl(
        self,
        config: Any,
        central_atom_idx: int,
        positions: np.ndarray | None = None,
        cell: np.ndarray | None = None,
        types: list[str] | np.ndarray | None = None,
        constraints=None,
        user_constraints=None,
    ) -> Ok | Err | None:
        artn = None
        original = None
        try:
            original_stdout_fd = os.dup(1)
            devnull = os.open(os.devnull, os.O_WRONLY)
            os.dup2(devnull, 1)
            try:
                print("Central Atom", central_atom_idx)
                if config.control.active_volume:
                    atom_map, central_lammps_id = partn_search_AV(
                        self,
                        config,
                        central_atom_idx,
                        positions,
                        cell,
                        types,
                        constraints=constraints,
                        user_constraints=user_constraints,
                        validated=True,
                    )
                else:
                    atom_map = None
                    central_lammps_id = [central_atom_idx + 1]
                    if positions is not None:
                        self.set_positions(positions=positions)

                delr_threshold = config.eventsearch.delr_thr

                artn = pypARTn.artn(engine="lmp")

                self.lmp.command(f"plugin load {artn.lib._name}")
                atoms_frozen = (
                    False
                    if config.control.active_volume
                    else self._make_frozen_group(config, positions, types, constraints)
                )
                self._apply_frozen_fix("f_frozen_pre", atoms_frozen)
                self.lmp.command("fix 10 all artn dmax {}".format(config.partn.dmax))
                self._apply_frozen_fix("f_frozen_post", atoms_frozen)
                if config.control.active_volume:
                    # ARTn replaces forces with trial displacements after f_buffer
                    # runs. Keep the buffer fixed through that second force update.
                    self.lmp.command("fix f_buffer_post buffer setforce 0.0 0.0 0.0")
                self.lmp.command("min_style fire")

                artn.reset_input()
                artn.set("filout", "artn.out." + str(self.engine_id))
                artn.set("engine_units", "lammps/metal")
                artn.set("verbose", config.partn.verbosity)
                artn.set("struc_format_out", "none")
                artn.set("delr_thr", config.partn.delr_thr)
                artn.set("lpush_final", True)
                artn.set("lmove_nextmin", False)
                artn.set("zseed", config.partn.zseed)
                artn.set("push_mode", config.partn.push_mode)
                if config.partn.push_mode == "rad":
                    artn.set("push_dist_thr", config.partn.push_dist_thr)
                artn.set("push_step_size", config.partn.push_step_size)
                artn.set("push_ids", central_lammps_id)
                artn.set("ninit", config.partn.ninit)
                artn.set("lanczos_min_size", config.partn.lanczos_min_size)
                artn.set("lanczos_max_size", config.partn.lanczos_max_size)
                artn.set("lanczos_disp", config.partn.lanczos_disp)
                artn.set("lanczos_eval_conv_thr", config.partn.lanczos_eval_conv_thr)
                artn.set("eigval_thr", config.partn.eigval_thr)
                artn.set("eigen_step_size", config.partn.eigen_step_size)
                artn.set("nsmooth", config.partn.nsmooth)
                artn.set("neigen", config.partn.neigen)
                artn.set("alpha_mix_cr", config.partn.alpha_mix_cr)
                artn.set("nnewchance", config.partn.nnewchance)
                if config.partn.nperp is not None:
                    artn.set("nperp", config.partn.nperp)
                if config.partn.nperp_limitation is not None:
                    artn.set(
                        "nperp_limitation", np.array(config.partn.nperp_limitation)
                    )
                else:
                    artn.set("lnperp_limitation", False)
                artn.set("forc_thr", config.partn.forc_thr)
                artn.set("push_over", config.partn.push_over)

                fixed_rows = (
                    constraints.crop(atom_map).local_fixed_indices
                    if config.control.active_volume
                    else constraints.local_fixed_indices
                )
                with self._fixed_velocity_guard(fixed_rows):
                    self.lmp.command(
                        f"minimize 1e-6 1e-8 10000 {config.partn.nevalf_max}"
                    )
                self.lmp.command("unfix 10")
                if config.control.active_volume:
                    self.lmp.command("unfix f_buffer_post")
                self._remove_frozen_fix("f_frozen_post", atoms_frozen)
                self._remove_frozen_fix("f_frozen_pre", atoms_frozen)
                self._delete_frozen_group(atoms_frozen)
            finally:
                # Always give stdout back, even when pARTn or LAMMPS raised.
                os.dup2(original_stdout_fd, 1)
                os.close(original_stdout_fd)
                os.close(devnull)

            if self._is_rank0:
                err = artn.get_error()
                if err[0] == 0:
                    delr1 = artn.extract("delr_min1")
                    delr2 = artn.extract("delr_min2")
                    if delr1 < delr_threshold or delr2 < delr_threshold:
                        E_sad = artn.extract("etot_sad")
                        E_min1 = artn.extract("etot_min1")
                        E_min2 = artn.extract("etot_min2")
                        dE_forward = E_sad - E_min1
                        dE_backward = E_sad - E_min2

                        if config.control.active_volume:
                            (
                                min1positions,
                                min2positions,
                                saddlepositions,
                                index_move,
                            ) = position_results_AV(config, artn, atom_map, positions)
                        else:
                            min1positions = self._positions_from_lammps(
                                artn.extract("tau_min1")
                            )
                            min2positions = self._positions_from_lammps(
                                artn.extract("tau_min2")
                            )
                            saddlepositions = self._positions_from_lammps(
                                artn.extract("tau_sad")
                            )
                            dist = (min1positions - saddlepositions) ** 2
                            dist = dist.sum(axis=-1)
                            dist = np.sqrt(dist)
                            dist[dist > config.atomicenvironment.rcut] = 0
                            index_move = np.argmax(dist)

                        if delr1 < delr2:
                            return Ok(
                                EventSearchOutput(
                                    central_atom_index=central_atom_idx,
                                    dE_forward=dE_forward,
                                    dE_backward=dE_backward,
                                    min1_positions=min1positions,
                                    saddle_positions=saddlepositions,
                                    min2_positions=min2positions,
                                    move_atom_index=index_move,
                                    types=types,
                                )
                            )
                        else:
                            return Ok(
                                EventSearchOutput(
                                    central_atom_index=central_atom_idx,
                                    dE_forward=dE_backward,
                                    dE_backward=dE_forward,
                                    min1_positions=min2positions,
                                    saddle_positions=saddlepositions,
                                    min2_positions=min1positions,
                                    move_atom_index=index_move,
                                    types=types,
                                )
                            )
                    else:
                        return Err(
                            ErrorInfo(
                                type=ErrorType.EVENT_MINIMA_NOT_MATCH_POSITIONS,
                                message="delr1 and delr2 > at {}".format(
                                    delr_threshold
                                ),
                                variables={"delr1": delr1, "delr2": delr2},
                            )
                        )
                else:
                    return Err(
                        ErrorInfo(
                            type=ErrorType.EVENT_NOT_FOUND,
                            message="No event found",
                            details=err,
                        )
                    )
        except BaseException as exc:
            original = exc
            raise
        finally:
            self._destroy_artn(artn, original)

    @lammps_error_handler
    def partn_refine(
        self,
        config,
        central_atom_idx: int,
        positions=None,
        cell=None,
        types=None,
        saddle_idx=None,
        saddle_positions=None,
        minimize_outer_atoms: bool = True,
        constraints=None,
        user_constraints=None,
    ):
        """Refine a saddle point with pARTn starting from ``saddle_positions``.

        Under ``config.control.active_volume`` the engine is rebuilt as the
        cropped active volume and the full system is restored on every exit
        path (Ok, Err, exception); see the class docstring, "State
        ownership". A saddle atom that is missing from the crop yields
        ``Err(REFINEMENT_INVALID_MINIMA)`` on rank 0 (``None`` elsewhere)
        instead of an exception; a saddle position is otherwise placed where
        the caller put it (no distance check against ``ract``, as in the
        base). Non-finite ``saddle_positions`` raise ``ValueError`` on every
        rank before any LAMMPS call.
        """
        active = config.control.active_volume
        if active:
            self._check_active_volume_inputs(
                config, positions, cell, types, "partn_refine"
            )
        constraints = self._resolve_search_constraints(
            config,
            central_atom_idx,
            positions,
            cell,
            types,
            constraints,
            user_constraints,
        )
        # The helper validates the proposed overlay before it clears the engine.
        try:
            with self._partn_resource_scope(active):
                result = self._partn_refine_impl(
                    config,
                    central_atom_idx,
                    positions,
                    cell,
                    types,
                    saddle_idx,
                    saddle_positions,
                    minimize_outer_atoms,
                    constraints,
                    user_constraints,
                )
                if not active:
                    self._validate_search_result(result, constraints)
        except BaseException as exc:
            if active:
                self._restore_after_failure(positions, exc, "partn_refine")
            raise
        if active:
            self.ensure_full_system(positions)
            self._validate_search_result(result, constraints)
        return result

    def _partn_refine_impl(
        self,
        config: Any,
        central_atom_idx: int,
        positions: np.ndarray | None = None,
        cell: np.ndarray | None = None,
        types: list[str] | np.ndarray | None = None,
        saddle_idx: np.ndarray | None = None,
        saddle_positions: np.ndarray | None = None,
        minimize_outer_atoms: bool = True,
        constraints=None,
        user_constraints=None,
    ) -> Ok | Err | None:
        artn = None
        original = None
        try:
            if config.control.active_volume:
                try:
                    E_init, atom_map, central_lammps_id = partn_refine_AV(
                        self,
                        config,
                        central_atom_idx,
                        positions,
                        cell,
                        types,
                        saddle_idx,
                        saddle_positions,
                        constraints=constraints,
                        user_constraints=user_constraints,
                        validated=True,
                    )
                except ActiveVolumeSaddleError as exc:
                    # A saddle atom missing from the crop: report, do not crash the
                    # worker (the base's `.item()` numpy error).
                    if self._is_rank0:
                        return Err(
                            ErrorInfo(
                                type=ErrorType.REFINEMENT_INVALID_MINIMA,
                                message=str(exc),
                            )
                        )
                    return None
            else:
                central_lammps_id = [central_atom_idx + 1]
                E_init = 0
                atom_map = None
                if positions is not None:
                    self.set_positions(positions=positions)
                    if minimize_outer_atoms:
                        core = set(() if saddle_idx is None else saddle_idx)
                        core.update(constraints.local_fixed_indices)
                        if core:
                            self.minimize_freeze_core(sorted(core))

            artn = pypARTn.artn(engine="lmp")
            self.lmp.command(f"plugin load {artn.lib._name}")

            artn.reset_input()
            artn.set("filout", "artn.out." + str(self.engine_id))
            artn.set("engine_units", "lammps/metal")
            artn.set("verbose", config.partn.verbosity)
            artn.set("struc_format_out", "none")
            artn.set("delr_thr", config.partn.delr_thr)
            artn.set("lpush_final", False)
            artn.set("lmove_nextmin", False)
            artn.set("zseed", config.partn.zseed)
            artn.set("push_mode", config.partn.r_push_mode)
            if config.partn.push_mode == "rad":
                artn.set("push_dist_thr", config.partn.r_push_dist_thr)
            artn.set("push_step_size", config.partn.r_push_step_size)
            artn.set("push_ids", central_lammps_id)
            artn.set("ninit", config.partn.r_ninit)
            artn.set("lanczos_min_size", config.partn.r_lanczos_min_size)
            artn.set("lanczos_max_size", config.partn.r_lanczos_max_size)
            artn.set("lanczos_disp", config.partn.r_lanczos_disp)
            artn.set("lanczos_eval_conv_thr", config.partn.r_lanczos_eval_conv_thr)
            artn.set("eigval_thr", config.partn.r_eigval_thr)
            artn.set("eigen_step_size", config.partn.r_eigen_step_size)
            artn.set("nsmooth", config.partn.r_nsmooth)
            artn.set("neigen", config.partn.r_neigen)
            artn.set("alpha_mix_cr", config.partn.r_alpha_mix_cr)
            artn.set("nnewchance", config.partn.r_nnewchance)
            if config.partn.r_nperp is not None:
                artn.set("nperp", config.partn.r_nperp)
            if config.partn.r_nperp_limitation is not None:
                artn.set("nperp_limitation", np.array(config.partn.r_nperp_limitation))
            else:
                artn.set("lnperp_limitation", False)
            artn.set("forc_thr", config.partn.r_forc_thr)

            max_attempts = config.partn.r_max_attempts
            attempt = 0
            atoms_frozen = (
                False
                if config.control.active_volume
                else self._make_frozen_group(config, positions, types, constraints)
            )
            self._apply_frozen_fix("f_frozen_pre", atoms_frozen)

            while attempt < max_attempts:
                exit_flag = False
                result = None
                self.lmp.command("fix 10 all artn dmax {}".format(config.partn.r_dmax))
                self._apply_frozen_fix("f_frozen_post", atoms_frozen)
                if config.control.active_volume:
                    self.lmp.command("fix f_buffer_post buffer setforce 0.0 0.0 0.0")
                self.lmp.command("min_style fire")
                fixed_rows = (
                    constraints.crop(atom_map).local_fixed_indices
                    if config.control.active_volume
                    else constraints.local_fixed_indices
                )
                with self._fixed_velocity_guard(fixed_rows):
                    self.lmp.command(
                        f"minimize 1e-6 1e-8 10000 {config.partn.r_nevalf_max}"
                    )
                self.lmp.command("unfix 10")
                if config.control.active_volume:
                    self.lmp.command("unfix f_buffer_post")
                self._remove_frozen_fix("f_frozen_post", atoms_frozen)

                if self._is_rank0:
                    err = artn.get_error()
                    if err[0] == 0:
                        delr_sad = artn.extract("delr_sad")
                        if delr_sad < config.partn.r_delr_sad_thr:
                            E_sad = artn.extract("etot_sad")
                            E_result = E_sad - E_init
                            saddlepositions = self._positions_from_lammps(
                                artn.extract("tau_sad")
                            )
                            if config.control.active_volume:
                                saddlepositions_results = positions.copy()
                                for i, atom_idx in enumerate(atom_map):
                                    saddlepositions_results[atom_idx] = saddlepositions[
                                        i
                                    ]
                            else:
                                saddlepositions_results = saddlepositions
                            exit_flag = True
                            result = Ok(
                                EventRefinementOutput(
                                    central_atom_index=central_atom_idx,
                                    saddle_positions=saddlepositions_results,
                                    E_saddle=E_result,
                                    refined="T",
                                )
                            )

                exit_flag = (
                    self.comm.bcast(exit_flag, root=0)
                    if self.comm is not None
                    else exit_flag
                )
                if exit_flag:
                    self._remove_frozen_fix("f_frozen_pre", atoms_frozen)
                    self._delete_frozen_group(atoms_frozen)
                    return result

                attempt += 1
                artn.set("zseed", config.partn.zseed)

            else:
                self._remove_frozen_fix("f_frozen_pre", atoms_frozen)
                self._delete_frozen_group(atoms_frozen)
                if self._is_rank0:
                    err = artn.get_error()
                    return Err(
                        ErrorInfo(
                            type=ErrorType.EVENT_NOT_FOUND,
                            message="no event found",
                            details=err,
                        )
                    )
                return None
        except BaseException as exc:
            original = exc
            raise
        finally:
            self._destroy_artn(artn, original)
