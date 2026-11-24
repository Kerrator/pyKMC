"""One ARTn saddle search as a worker operation, on the dimer's force path.

:class:`ArtnSearchExtension` adds the ``artn_search`` operation to the dimer's
worker extension: the same dedicated serial LAMMPS instance on the worker's
leader rank, the same frozen-atom fix and the same force query (the full
system with the frozen atoms masked, optional neighbor-list reuse and EAM
crop), driving :func:`pykmc.artn.artn_search` instead of the dimer core.
"""

from __future__ import annotations

from typing import Any, Mapping, Optional

import numpy as np
from ase.cell import Cell
from ase.data import atomic_masses, atomic_numbers

from ..artn import artn_search as run_artn
from ..dimer import DimerResult
from .dimer_search import DimerSearchExtension

__all__ = ["ArtnSearchExtension"]


class ArtnSearchExtension(DimerSearchExtension):
    """Run one ARTn search per call on the dimer's dedicated serial instance."""

    def artn_search(
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
        """Run one ARTn search from ``positions`` and return what it found.

        Arguments, errors and the result are as for
        :meth:`~pykmc.engine.dimer_search.DimerSearchExtension.dimer_search`,
        with the ARTn core in place of the dimer's. ``options`` are keyword
        options of :func:`pykmc.artn.artn_search` (the `[pARTn]` parameters it
        takes) plus the worker options ``reuse_neighbor_list`` and
        ``eam_force_crop`` (default False). FIRE gets the masses LAMMPS has,
        which pyKMC sets from ASE's table.

        Returns
        -------
        Optional[DimerResult]
            The ARTn core's result on the leader rank, None on the other ranks.

        """
        masses = np.array([atomic_masses[atomic_numbers[str(t)]] for t in types])
        return self._search(
            run_artn,
            positions,
            types,
            cell,
            pbc,
            movable,
            central_atom,
            seed,
            {**(options or {}), "masses": masses},
        )
