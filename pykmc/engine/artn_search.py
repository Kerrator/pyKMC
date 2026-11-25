"""One ARTn saddle search as a worker operation, on either of two force paths.

:class:`ArtnSearchExtension` adds the ``artn_search`` operation to the dimer's
worker extension. A single-rank worker uses the dimer's serial force path. On a
multi-rank worker, each rank evaluates the forces on its own rows. Both use
:func:`pykmc.artn.artn_search` for the search. With the worker option
``av_radii`` the search runs instead on pARTn's active volume, built as
``partn_search`` builds it, so the forces are pARTn's; it runs on the
worker's leader rank.
"""

from __future__ import annotations

import ctypes
from types import SimpleNamespace
from typing import TYPE_CHECKING, Any, Mapping, Optional, cast

import numpy as np
from ase.cell import Cell
from ase.geometry import find_mic
from mpi4py import MPI

from ..artn import artn_search as run_artn
from ..config import supports_neighbor_reuse
from ..dimer import DimerResult
from ..search_rules import map_to_full_system
from .dimer_search import DimerSearchExtension, _movable_mask

if TYPE_CHECKING:
    from .lammps import LammpsEngine

__all__ = ["ArtnSearchExtension"]


class ArtnSearchExtension(DimerSearchExtension):
    """Run one ARTn search on all ranks of the worker."""

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
        ``eam_force_crop``, and ``av_radii`` (default None). On any worker,
        ``reuse_neighbor_list`` is on by default for ``eam/alloy``,
        ``eam/fs``, ``mtp`` and ``mtp/small``, as for the dimer, and
        ``eam_force_crop`` is off by default. ``av_radii = (rmov, ract)`` runs the search on pARTn's
        active volume
        (:meth:`_av_search`) instead of the full system, on the leader rank
        whatever the worker's size. FIRE gets the masses the force instance
        uses: pyKMC sets them from ASE's table, and an EAM file read by
        ``pair_coeff`` replaces them with its own.

        Returns
        -------
        Optional[DimerResult]
            The ARTn core's result on the leader rank, None on the other ranks.

        """
        search_options = dict(options or {})
        search_options.setdefault(
            "reuse_neighbor_list",
            supports_neighbor_reuse(self.engine.config.pair_style),
        )
        av_radii = search_options.pop("av_radii", None)
        if av_radii is not None:
            return self._av_search(
                av_radii,
                positions,
                types,
                cell,
                movable,
                central_atom,
                seed,
                search_options,
            )
        comm = self.engine.comm
        if comm is not None and comm.Get_size() > 1:
            from .artn_ranks import distributed_search

            return distributed_search(
                cast("LammpsEngine", self.engine),
                positions,
                types,
                cell,
                pbc,
                movable,
                central_atom,
                seed,
                search_options,
            )

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
            search_options,
        )

    def _av_search(
        self,
        av_radii: tuple[float, float],
        positions: np.ndarray,
        types: list[str] | np.ndarray,
        cell: Cell | np.ndarray,
        movable: list[int] | np.ndarray,
        central_atom: int,
        seed: int,
        options: Mapping[str, Any],
    ) -> Optional[DimerResult]:
        """Run one ARTn search on pARTn's active volume, on the leader rank.

        The serial instance is cleared and the active volume built on it as
        ``partn_search`` builds it
        (:func:`~pykmc.activevolume.active_volume.partn_search_AV`): the atoms
        within ``ract`` of the central atom, in a box with every axis periodic
        whose edges are the cell's diagonal, the buffer (beyond ``rmov``) under
        a zero ``setforce``. Each force call places its atoms and runs
        LAMMPS ``run 0``. With the option ``reuse_neighbor_list`` the active volume gets
        the dimer's 0.5 Å skin and a step that moves no atom further than half
        of it is made by ``fix nve/noforce`` and ``run 1 pre no``, so LAMMPS
        rebuilds its neighbor list only when its own check asks.

        As under pARTn, the core sees zero force on the buffer but moves it:
        the buffer is movable to the core. Atoms within ``rmov`` that are not
        in ``movable`` (``[frozen_atoms]``) also see zero force and never move.
        The result is mapped back to the full system: atoms beyond ``ract``
        keep their positions and get zero mode; energies are the active volume's.

        Raises
        ------
        ValueError
            If ``movable`` is malformed, or the cell is not orthogonal with
            its edges along the axes (its box uses only its diagonal).

        """
        comm = self.engine.comm
        if comm is not None and comm.Get_rank() != 0:
            return None
        # Imported here: the module imports pypARTn and LAMMPS.
        from ..activevolume.active_volume import partn_search_AV, set_positions
        from ..config import validate_dimer_neighbor_reuse
        from .dimer_forces import _NEIGHBOR_SKIN, configure_neighbor_reuse
        from .lammps import LammpsEngine

        options = dict(options)
        reuse = bool(options.pop("reuse_neighbor_list", False))
        if reuse:
            validate_dimer_neighbor_reuse(self.engine.config.pair_style)

        positions = np.array(positions, dtype=float)
        cell = np.array(cell, dtype=float)
        if not np.allclose(cell, np.diag(np.diag(cell)), rtol=0.0, atol=1e-12):
            raise ValueError(
                "the active volume needs an orthogonal cell with its edges "
                "along x, y and z"
            )
        mask = _movable_mask(movable, len(positions))
        rmov, ract = av_radii
        force_engine = self._force_engine
        if force_engine is None:
            force_engine = LammpsEngine(
                config=self.engine.config,
                comm=MPI.COMM_SELF,
                engine_id=f"dimer.{self.engine.engine_id}",
                search_instance=True,
            )
            self._force_engine = force_engine
            force_engine.start()
        # The active volume replaces whatever system the instance held.
        self._force_engine_system = None
        try:
            atom_map, _ = partn_search_AV(
                force_engine,
                SimpleNamespace(activevolume=SimpleNamespace(rmov=rmov, ract=ract)),
                central_atom,
                positions,
                cell,
                types,
            )
            # The buffer, by define_AV's rule: beyond rmov, minimum image.
            _, distance = find_mic(
                positions[atom_map] - positions[central_atom], cell, pbc=True
            )
            buffer = np.abs(distance) > rmov
            in_movable = mask[atom_map] & ~buffer
            zero_force = ~in_movable
            if reuse:
                configure_neighbor_reuse(force_engine)
            previous: Optional[np.ndarray] = None

            def query(x: np.ndarray) -> tuple[float, np.ndarray]:
                nonlocal previous
                step = None if previous is None else x - previous
                if step is not None and np.max(np.abs(step)) <= _NEIGHBOR_SKIN / 2:
                    # As the dimer's evaluator: nve/noforce moves each atom by
                    # its step and Verlet checks the list before the forces.
                    force_engine.lmp.scatter_atoms(
                        "v",
                        1,
                        3,
                        np.ascontiguousarray(step).ctypes.data_as(
                            ctypes.POINTER(ctypes.c_double)
                        ),
                    )
                    force_engine.command("run 1 pre no post no")
                else:
                    set_positions(force_engine, x)
                    force_engine.command("run 0 post no")
                if reuse:
                    previous = x.copy()
                energy = force_engine.lmp.get_thermo("pe")
                forces = np.ctypeslib.as_array(
                    force_engine.lmp.gather_atoms("f", 1, 3)
                ).reshape(-1, 3)
                forces = forces.copy()
                forces[zero_force] = 0.0
                return energy, forces

            result = run_artn(
                positions[atom_map],
                cell,
                np.ones(3, dtype=bool),
                in_movable | buffer,
                int(np.flatnonzero(atom_map == central_atom)[0]),
                seed,
                query,
                masses=self._instance_masses(),
                **options,
            )
        except BaseException:
            self._drop_dedicated_engine()
            raise

        def full(array: Optional[np.ndarray], base: np.ndarray) -> Optional[np.ndarray]:
            return None if array is None else map_to_full_system(array, atom_map, base)

        return result._replace(
            saddle_positions=full(result.saddle_positions, positions),
            min1_positions=full(result.min1_positions, positions),
            min2_positions=full(result.min2_positions, positions),
            mode=full(result.mode, np.zeros_like(positions)),
        )

    def _instance_masses(self) -> np.ndarray:
        """Return each atom's mass on the serial instance, amu, in pyKMC order."""
        lmp = self._force_engine.lmp
        types = np.array(lmp.gather_atoms("type", 0, 1), dtype=int)
        per_type = lmp.numpy.extract_atom("mass")
        return np.asarray(per_type, dtype=float)[types]
