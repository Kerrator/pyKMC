from __future__ import annotations
from lammps import lammps
import numpy as np
import ctypes
import functools
import os
from dataclasses import dataclass, replace
from typing import Protocol
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
    partn_search_AV,
    partn_refine_AV,
    position_results_AV,
)
from ..atomic_environment import AtomicEnvironment
from ..physics import EnginePhysics, ForceModel
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
    _LAMMPS_EXCEPTIONS = ()  # LAMMPS not compiled with -DLAMMPS_EXCEPTIONS=yes — decorator is a no-op


def lammps_error_handler(method):
    """Catch LAMMPSException, close the engine, and re-raise as RuntimeError.

    Requires LAMMPS compiled with -DLAMMPS_EXCEPTIONS=yes.
    Without it, LAMMPS calls MPI_Abort and no exception is raised.
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
        ``masses`` it is the explicit map an HTST scratch instance is built
        from, so a crop holding a subset of the species keeps the full
        system's type numbering.
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
        # Remembered full system (set by initialize_system).
        self.full_system: FullSystem | None = None

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
        """Run a LAMMPS command string."""
        self.lmp.command(cmd)

    def _positions_to_lammps(self, positions: np.ndarray) -> np.ndarray:
        return positions @ self.Q.T if not self._is_orthorhombic else positions

    def _positions_from_lammps(self, positions: np.ndarray) -> np.ndarray:
        return positions @ self.Q if not self._is_orthorhombic else positions

    def _has_compute(self, compute_id: str) -> bool:
        """Check if a compute with the given id exists in LAMMPS."""
        return self.lmp.has_id("compute", compute_id)

    @lammps_error_handler
    def start(self) -> None:
        engine_log = (
            "none" if self.config.verbosity == 0 else f"lammps.log.{self.engine_id}"
        )
        self.lmp = lammps(
            comm=self.comm, cmdargs=["-screen", "none", "-log", engine_log]
        )

    def close(self) -> None:
        if self.lmp is not None:
            self.lmp.close()
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
            crop) so the full system's type numbering and masses are kept.
            When omitted the behaviour is exactly ``species_map(types)``.
        masses : tuple[float, ...], optional
            Keyword-only. Masses in amu in ``species`` order; required with
            ``species``.

        Raises
        ------
        ValueError
            If ``pbc`` is malformed, if only one of ``species``/``masses`` is
            given, if the override is malformed, or if a symbol in ``types``
            is not in the species map.

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
        # Remember the full system: HTST scratch instances are built from it.
        self.full_system = FullSystem(
            types=tuple(str(t) for t in types),
            species=species,
            masses=masses,
            cell=np.array(cell, dtype=np.float64, copy=True),
            pbc=(pbc[0], pbc[1], pbc[2]),
        )

    @lammps_error_handler
    def initialize_potential(self) -> None:
        """Issue ``pair_style`` / ``pair_coeff`` and complete the live system.

        Refreshes ``full_system.masses`` from the live instance (a potential
        file may override the emitted masses) and records the engine physics
        (force model, species and masses) the HTST extension verifies.
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
        positions = self._positions_to_lammps(positions=positions)
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

    def _make_frozen_group(self, config, positions=None, types=None) -> bool:
        """Resolve frozen atoms from config and create g_frozen group. Returns True if any atoms are frozen."""
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

    # ------------------------------------------------------------------
    # Minimization
    # ------------------------------------------------------------------

    @lammps_error_handler
    def minimize_with_results(
        self, positions=None, config=None, types=None
    ) -> tuple[np.ndarray, float] | None:
        """Minimize and return (positions, total_energy). Pass config and types to enable frozen-atom support."""
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

    @lammps_error_handler
    def partn_search(
        self, config, central_atom_idx: int, positions=None, cell=None, types=None
    ):
        original_stdout_fd = os.dup(1)
        devnull = os.open(os.devnull, os.O_WRONLY)
        os.dup2(devnull, 1)

        print("Central Atom", central_atom_idx)
        if config.control.active_volume:
            atom_map, central_lammps_id = partn_search_AV(
                self, config, central_atom_idx, positions, cell, types
            )
        else:
            atom_map = None
            central_lammps_id = [central_atom_idx + 1]
            if positions is not None:
                self.set_positions(positions=positions)

        delr_threshold = config.eventsearch.delr_thr

        artn = pypARTn.artn(engine="lmp")

        self.lmp.command(f"plugin load {artn.lib._name}")
        atoms_frozen = self._make_frozen_group(config, positions, types)
        self._apply_frozen_fix("f_frozen_pre", atoms_frozen)
        self.lmp.command("fix 10 all artn dmax {}".format(config.partn.dmax))
        self._apply_frozen_fix("f_frozen_post", atoms_frozen)
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
            artn.set("nperp_limitation", np.array(config.partn.nperp_limitation))
        else:
            artn.set("lnperp_limitation", False)
        artn.set("forc_thr", config.partn.forc_thr)
        artn.set("push_over", config.partn.push_over)

        self.lmp.command(f"minimize 1e-6 1e-8 10000 {config.partn.nevalf_max}")
        self.lmp.command("unfix 10")
        self._remove_frozen_fix("f_frozen_post", atoms_frozen)
        self._remove_frozen_fix("f_frozen_pre", atoms_frozen)
        self._delete_frozen_group(atoms_frozen)

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
                        min1positions, min2positions, saddlepositions, index_move = (
                            position_results_AV(config, artn, atom_map, positions)
                        )
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
                            message="delr1 and delr2 > at {}".format(delr_threshold),
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
    ):
        if config.control.active_volume:
            E_init, atom_map, central_lammps_id = partn_refine_AV(
                self,
                config,
                central_atom_idx,
                positions,
                cell,
                types,
                saddle_idx,
                saddle_positions,
            )
        else:
            central_lammps_id = [central_atom_idx + 1]
            E_init = 0
            atom_map = None
            if positions is not None:
                self.set_positions(positions=positions)
                if minimize_outer_atoms:
                    self.minimize_freeze_core(saddle_idx)

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
        atoms_frozen = self._make_frozen_group(config, positions, types)
        self._apply_frozen_fix("f_frozen_pre", atoms_frozen)

        while attempt < max_attempts:
            exit_flag = False
            result = None
            self.lmp.command("fix 10 all artn dmax {}".format(config.partn.r_dmax))
            self._apply_frozen_fix("f_frozen_post", atoms_frozen)
            self.lmp.command("min_style fire")
            self.lmp.command(f"minimize 1e-6 1e-8 10000 {config.partn.r_nevalf_max}")
            self.lmp.command("unfix 10")
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
                                saddlepositions_results[atom_idx] = saddlepositions[i]
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
