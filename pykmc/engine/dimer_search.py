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

from contextlib import ExitStack
from typing import TYPE_CHECKING, Any, Callable, Mapping, Optional

import numpy as np
from ase.cell import Cell
from mpi4py import MPI

from ..config import (
    search_pair_style,
    validate_dimer_eam_style,
    validate_dimer_neighbor_reuse,
)
from ..dimer import DimerResult, _wrap
from ..dimer import dimer_search as run_dimer
from .base import Engine, EngineExtension
from .dimer_forces import configure_neighbor_reuse, dimer_force_query

if TYPE_CHECKING:
    from .lammps import LammpsEngine

__all__ = ["DimerSearchExtension"]

# LAMMPS ids of the group and fix holding the frozen atoms during a search.
_FROZEN_GROUP = "g_dimer_frozen"
_FROZEN_FIX = "f_dimer_frozen"


def _movable_mask(movable: list[int] | np.ndarray, n_atoms: int) -> np.ndarray:
    """Return the (N,) mask of the atoms indexed by ``movable``.

    Raises
    ------
    ValueError
        If ``movable`` is not a set of integer indices within the system.

    """
    movable = np.asarray(movable).ravel()
    if movable.size == 0:
        movable = movable.astype(int)
    if not np.issubdtype(movable.dtype, np.integer):
        raise ValueError("movable must hold integer atom indices, not a mask")
    if movable.size and (movable.min() < 0 or movable.max() >= n_atoms):
        raise ValueError("movable holds an index outside the system")
    mask = np.zeros(n_atoms, dtype=bool)
    mask[movable] = True
    return mask


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
            ``eam_force_crop`` limits the private EAM force calculation to a
            guarded neighborhood of the movable atoms (default False).

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
        return self._search(
            run_dimer,
            positions,
            types,
            cell,
            pbc,
            movable,
            central_atom,
            seed,
            {"remove_translation": True, **(options or {})},
        )

    def _search(
        self,
        core: Callable[..., Any],
        positions: np.ndarray,
        types: list[str] | np.ndarray,
        cell: Cell | np.ndarray,
        pbc: list[bool] | np.ndarray,
        movable: list[int] | np.ndarray,
        central_atom: int,
        seed: int,
        options: Mapping[str, Any],
    ) -> Any:
        """Run one search ``core`` on the serial instance, on the leader rank.

        ``core`` is called as :func:`pykmc.dimer.dimer_search` is, with the
        force query in place of the callback and ``options`` as keywords,
        less the worker options ``reuse_neighbor_list`` and
        ``eam_force_crop``. Other ranks return None at once. Arguments,
        errors and the frozen atoms are as for :meth:`dimer_search`.
        """
        comm = self.engine.comm
        if comm is not None and comm.Get_rank() != 0:
            return None

        positions = np.array(positions, dtype=float)
        cell = Cell(np.array(cell, dtype=float))
        pbc = np.array(pbc, dtype=bool)
        mask = _movable_mask(movable, len(positions))
        frozen = np.flatnonzero(~mask)

        search_options = dict(options)
        reuse = search_options.pop("reuse_neighbor_list", False)
        if not isinstance(reuse, (bool, np.bool_)):
            raise ValueError("reuse_neighbor_list must be a boolean")
        if reuse:
            validate_dimer_neighbor_reuse(search_pair_style(self.engine.config))
        crop = search_options.pop("eam_force_crop", False)
        if not isinstance(crop, (bool, np.bool_)):
            raise ValueError("eam_force_crop must be a boolean")
        if crop:
            validate_dimer_eam_style(
                search_pair_style(self.engine.config), "eam_force_crop"
            )
        try:
            force_engine = self._dedicated_engine(types, positions, cell, pbc, reuse)
            if frozen.size:
                ids = " ".join(str(i + 1) for i in frozen)
                force_engine.command(f"group {_FROZEN_GROUP} id {ids}")
                force_engine.command(
                    f"fix {_FROZEN_FIX} {_FROZEN_GROUP} setforce 0 0 0"
                )
            with ExitStack() as resources:
                query = dimer_force_query(
                    force_engine,
                    types,
                    cell,
                    pbc,
                    mask,
                    central_atom,
                    reuse,
                    crop,
                    resources,
                )
                result = core(
                    positions,
                    np.array(cell),
                    pbc,
                    mask,
                    central_atom,
                    seed,
                    query,
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
                search_instance=True,
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
                configure_neighbor_reuse(force_engine)
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
