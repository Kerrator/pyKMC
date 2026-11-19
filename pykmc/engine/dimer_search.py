"""One dimer saddle search as a worker operation, on its own LAMMPS instance.

The manager sends each operation to whichever worker is free, with no session
affinity, so a dimer search cannot be driven force call by force call from
rank 0. :class:`DimerSearchExtension` runs a whole search inside one operation
instead: attach it through the engine factory's ``engine_extensions`` argument
and call ``dimer_search`` through the manager like ``partn_search``.

Forces come from one dedicated serial LAMMPS instance per worker, on the
worker's leader rank (``MPI.COMM_SELF``). It is built on the first search and
reused by every later one, and pARTn never touches it. The worker's own engine
is left as it was. A worker with several ranks therefore gets no speed-up on
dimer force calls: its other ranks wait at the operation's closing barrier
while the leader searches.
"""

from __future__ import annotations

import ctypes
from typing import TYPE_CHECKING, Any, Mapping, Optional

import numpy as np
from ase.cell import Cell
from mpi4py import MPI

from ..config import validate_dimer_neighbor_reuse
from ..dimer import DimerResult, _wrap
from ..dimer import dimer_search as run_dimer
from .base import Engine, EngineExtension

if TYPE_CHECKING:
    from .lammps import LammpsEngine

__all__ = ["DimerSearchExtension"]

# LAMMPS ids of the group and fix holding the frozen atoms during a search.
_FROZEN_GROUP = "g_dimer_frozen"
_FROZEN_FIX = "f_dimer_frozen"
_NEIGHBOR_SKIN = 0.5  # Å


class _EAMForceEvaluator:
    """Evaluate prescribed positions on the private serial engine for one search."""

    def __init__(self, engine: LammpsEngine, cell: Cell, pbc: np.ndarray) -> None:
        self.engine = engine
        self.cell = np.asarray(cell)
        self.pbc = pbc
        self.fixed_axes = None if pbc.all() else np.linalg.inv(self.cell)[:, ~pbc]
        self.previous: np.ndarray | None = None

    def __call__(self, positions: np.ndarray) -> tuple[float, np.ndarray]:
        """Return the fresh EAM potential energy and forces at this geometry."""
        engine = self.engine
        delta = None if self.previous is None else positions - self.previous
        small_step = delta is not None and np.max(np.abs(delta)) <= _NEIGHBOR_SKIN / 2
        if small_step and self.fixed_axes is not None:
            # Let full setup detect a lost atom immediately at a fixed face.
            fixed = positions @ self.fixed_axes
            small_step = np.all((fixed >= 0) & (fixed < 1))
        if small_step:
            # With dt = 1 in metal units, nve/noforce translates every atom by
            # its prescribed increment. Verlet updates ghosts and checks the
            # accumulated displacement against half the skin before computing
            # the EAM energy and forces. Scattering x and skipping setup would
            # bypass this part of the integration loop.
            velocity = np.ascontiguousarray(
                engine._positions_to_lammps(delta), dtype=np.float64
            )
            engine.lmp.scatter_atoms(
                "v", 1, 3, velocity.ctypes.data_as(ctypes.POINTER(ctypes.c_double))
            )
            engine.command("run 1 pre no post no")
            values = engine.lmp.get_thermo("pe"), engine._gather_forces()
        else:
            # A new search may change the frozen group. Use full setup for it
            # and for large jumps, including a change of periodic image.
            values = engine.get_energy_and_forces(_wrap(positions, self.cell, self.pbc))
        # The minimizer can update its positions in place between calls.
        self.previous = positions.copy()
        return values


class DimerSearchExtension(EngineExtension):
    """Run one dimer search per call on a dedicated serial LAMMPS instance.

    The instance takes the potential from the worker engine's configuration
    and the system (types, cell, periodicity) from the request. It is kept for
    later searches while the system and neighbor-list mode stay the same.
    Changing either resets it with the LAMMPS ``clear`` command. Any
    error during a search closes and drops it, so the next search builds a new
    one.

    Parameters
    ----------
    engine : Engine
        The worker's engine, a :class:`~pykmc.engine.lammps.LammpsEngine`.

    """

    def __init__(self, engine: Engine) -> None:
        super().__init__(engine)
        self._force_engine: Optional[LammpsEngine] = None
        self._force_engine_system: Optional[tuple] = None

    def dimer_search(
        self,
        positions: np.ndarray,
        types: list[str] | np.ndarray,
        cell: Cell | np.ndarray,
        pbc: list[bool] | np.ndarray,
        movable: list[int] | np.ndarray,
        central_atom: int,
        seed: int,
        options: Optional[Mapping[str, Any]] = None,
    ) -> Optional[DimerResult]:
        """Run one dimer search from ``positions`` and return what it found.

        Every rank of the worker runs this operation; only the leader rank
        searches, and the other ranks return at once.

        Parameters
        ----------
        positions : np.ndarray
            Starting positions of the full system, (N, 3), Å, pyKMC atom order.
        types : list[str] | np.ndarray
            Element symbol of each atom.
        cell : Cell | np.ndarray
            Cell vectors as rows, (3, 3), Å.
        pbc : list[bool] | np.ndarray
            Periodicity of each cell axis.
        movable : list[int] | np.ndarray
            Integer indices of the atoms the search may move. The others are
            frozen: they never move and LAMMPS reports zero force on them.
        central_atom : int
            Atom the ``local`` start is centred on.
        seed : int
            Seed of the starting direction.
        options : Optional[Mapping[str, Any]]
            Keyword options of :func:`pykmc.dimer.dimer_search`, such as
            ``start``, ``start_radius`` and ``force_tolerance``. An
            interatomic potential is translation-invariant, so
            ``remove_translation`` is True unless set here. The worker option
            ``reuse_neighbor_list`` enables Verlet neighbor-list reuse for
            native static ``eam/alloy`` and ``eam/fs`` (default False).

        Returns
        -------
        Optional[DimerResult]
            The dimer core's result on the leader rank, positions in pyKMC atom
            order and wrapped on the periodic axes only; None on the other
            ranks. A failed search carries a reason and its force-call counts.

        Raises
        ------
        ValueError
            If ``movable`` is not a set of integer indices within the system,
            or an option is malformed (an unknown option raises TypeError). An
            error from LAMMPS is not caught.

        """
        comm = self.engine.comm
        if comm is not None and comm.Get_rank() != 0:
            return None

        positions = np.array(positions, dtype=float)
        n_atoms = len(positions)
        cell = Cell(np.array(cell, dtype=float))
        pbc = np.array(pbc, dtype=bool)
        movable = np.asarray(movable).ravel()
        if movable.size == 0:
            movable = movable.astype(int)
        if not np.issubdtype(movable.dtype, np.integer):
            raise ValueError("movable must hold integer atom indices, not a mask")
        if movable.size and (movable.min() < 0 or movable.max() >= n_atoms):
            raise ValueError("movable holds an index outside the system")
        mask = np.zeros(n_atoms, dtype=bool)
        mask[movable] = True
        frozen = np.flatnonzero(~mask)

        search_options = {"remove_translation": True, **(options or {})}
        reuse = search_options.pop("reuse_neighbor_list", False)
        if not isinstance(reuse, (bool, np.bool_)):
            raise ValueError("reuse_neighbor_list must be a boolean")
        if reuse:
            validate_dimer_neighbor_reuse(self.engine.config.pair_style)
        try:
            force_engine = self._dedicated_engine(types, positions, cell, pbc, reuse)
            if frozen.size:
                ids = " ".join(str(i + 1) for i in frozen)
                force_engine.command(f"group {_FROZEN_GROUP} id {ids}")
                force_engine.command(
                    f"fix {_FROZEN_FIX} {_FROZEN_GROUP} setforce 0 0 0"
                )
            result = run_dimer(
                positions,
                np.array(cell),
                pbc,
                mask,
                central_atom,
                seed,
                _EAMForceEvaluator(force_engine, cell, pbc)
                if reuse
                else force_engine.get_energy_and_forces,
                **search_options,
            )
            if frozen.size:
                force_engine.command(f"unfix {_FROZEN_FIX}")
                force_engine.command(f"group {_FROZEN_GROUP} delete")
        except BaseException:
            # An error from LAMMPS leaves the instance unusable: close and drop
            # it so a later search builds a new one, and let the error abort
            # the run.
            self._drop_dedicated_engine()
            raise
        return result

    def _dedicated_engine(
        self,
        types: list[str] | np.ndarray,
        positions: np.ndarray,
        cell: Cell,
        pbc: np.ndarray,
        reuse: bool,
    ) -> LammpsEngine:
        """Return the serial instance, set up for this system.

        Built on first use and reset when the types, cell, periodicity or
        neighbor-list mode differ from the last search's.
        """
        from .lammps import LammpsEngine

        system = (
            tuple(str(t) for t in types),
            np.array(cell).tobytes(),
            tuple(bool(p) for p in pbc),
            reuse,
        )
        force_engine = self._force_engine
        if force_engine is None:
            force_engine = LammpsEngine(
                config=self.engine.config,
                comm=MPI.COMM_SELF,
                engine_id=f"dimer.{self.engine.engine_id}",
            )
            self._force_engine = force_engine
            self._force_engine_system = None
            force_engine.start()
            force_engine.initialize_parameters()
        elif system != self._force_engine_system:
            self._force_engine_system = None
            force_engine.command("clear")
            force_engine.initialize_parameters()
        if self._force_engine_system is None:
            if reuse:
                positions = _wrap(positions, np.asarray(cell), pbc)
            force_engine.initialize_system(types, positions, cell, pbc)
            force_engine.initialize_potential()
            if reuse:
                force_engine.command(f"neighbor {_NEIGHBOR_SKIN} bin")
                force_engine.command("neigh_modify delay 0 every 1 check yes")
                force_engine.command("thermo_style custom step pe")
                force_engine.command("timestep 1")
                force_engine.command("fix f_dimer_displace all nve/noforce")
            self._force_engine_system = system
        return force_engine

    def _drop_dedicated_engine(self) -> None:
        """Close the serial instance, if any, and forget it."""
        force_engine = self._force_engine
        self._force_engine = None
        self._force_engine_system = None
        if force_engine is not None:
            try:
                force_engine.close()
            except Exception:  # already handling an error; keep that one
                pass
