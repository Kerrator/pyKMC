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
        ``eam_force_crop`` (default False). FIRE gets the masses the serial
        instance uses: pyKMC sets them from ASE's table, and an EAM file read
        by ``pair_coeff`` replaces them with its own.

        Returns
        -------
        Optional[DimerResult]
            The ARTn core's result on the leader rank, None on the other ranks.

        """

        def core(*args: Any, **kwargs: Any) -> DimerResult:
            # The instance exists and holds this system once the core is called.
            return run_artn(*args, masses=self._instance_masses(), **kwargs)

        return self._search(
            core,
            positions,
            types,
            cell,
            pbc,
            movable,
            central_atom,
            seed,
            dict(options or {}),
        )

    def _instance_masses(self) -> np.ndarray:
        """Return each atom's mass on the serial instance, amu, in pyKMC order."""
        lmp = self._force_engine.lmp
        types = np.array(lmp.gather_atoms("type", 0, 1), dtype=int)
        per_type = lmp.numpy.extract_atom("mass")
        return np.asarray(per_type, dtype=float)[types]
