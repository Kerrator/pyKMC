"""One ARTn saddle search or refinement as a worker operation.

:class:`ArtnSearchExtension` adds the ``artn_search`` and ``artn_refine``
operations to the dimer's worker extension. A single-rank worker uses the
dimer's serial force path. On a multi-rank worker, each rank evaluates the
forces on its own rows. The search uses :func:`pykmc.artn.artn_search`, the
refinement :func:`pykmc.artn.artn_refine`. With the worker option
``av_radii`` the search runs instead on pARTn's active volume, built as
``partn_search`` builds it, so the forces are pARTn's; it runs on the
worker's leader rank. Refinement runs on the full system only.
"""

from __future__ import annotations

import ctypes
from types import SimpleNamespace
from typing import TYPE_CHECKING, Any, Mapping, Optional, cast

import numpy as np
from ase.cell import Cell
from ase.geometry import find_mic
from mpi4py import MPI

from ..artn import RefineResult
from ..artn import artn_refine as run_refine
from ..artn import artn_search as run_artn
from ..config import search_pair_style, supports_neighbor_reuse
from ..dimer import DimerResult, _minimum_image
from ..search_rules import map_to_full_system
from .base import EngineExtension
from .dimer_search import DimerSearchExtension, _movable_mask

if TYPE_CHECKING:
    from .lammps import LammpsEngine

__all__ = ["ArtnSearchExtension", "with_refiner"]


def with_refiner(
    extensions: list[type[EngineExtension]], refine_style: str
) -> list[type[EngineExtension]]:
    """Return the engine extensions a run needs, the refiner's included.

    With ``refine_style = artn`` the ARTn extension is loaded whatever the
    search style; it extends the dimer's, so it takes the dimer's place.
    """
    extensions = list(extensions)
    if refine_style != "artn" or any(
        issubclass(ext, ArtnSearchExtension) for ext in extensions
    ):
        return extensions
    kept = [ext for ext in extensions if not issubclass(ArtnSearchExtension, ext)]
    return [*kept, ArtnSearchExtension]


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
            supports_neighbor_reuse(search_pair_style(self.engine.config)),
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

    def artn_refine(
        self,
        positions: np.ndarray,
        types: list[str] | np.ndarray,
        cell: Cell | np.ndarray,
        pbc: list[bool] | np.ndarray,
        movable: list[int] | np.ndarray,
        central_atom: int,
        saddle_idx: list[int] | np.ndarray,
        saddle_positions: np.ndarray,
        seeds: list[int],
        delr_sad_thr: float,
        relax_radius: Optional[float] = None,
        options: Optional[Mapping[str, Any]] = None,
    ) -> Optional[RefineResult]:
        """Refine one saddle from the minimum ``positions`` and return what it found.

        First every rank of the worker pre-relaxes on the worker's own engine,
        as pyKMC's ``partn_refine`` does on the full path: the rcut atoms
        ``saddle_idx`` are moved to ``saddle_positions`` and held while LAMMPS
        minimizes the others (``[lammps] min_style`` and ``frz_min``,
        :meth:`~pykmc.engine.lammps.LammpsEngine.minimize_freeze_core`). With
        ``relax_radius`` (``rmov`` with the active volume on), the atoms
        farther than it from the central atom are held too. Then
        :func:`pykmc.artn.artn_refine` runs from the pre-relaxed positions on
        the search's force path, one attempt per seed, and takes the start
        energy at ``positions``, the potential energy (``thermo_pe``) on the
        same instance as the saddle's, so the result's ``barrier`` is the
        barrier itself.

        Parameters
        ----------
        positions : np.ndarray
            The minimum, full system, (N, 3), Å, pyKMC atom order.
        types, cell, pbc, movable, central_atom
            As for :meth:`artn_search`; ``movable`` is the core's movable set.
        saddle_idx : list[int] | np.ndarray
            The rcut atoms, which start at the reference saddle.
        saddle_positions : np.ndarray
            Their reference saddle positions, (len(saddle_idx), 3), Å.
        seeds : list[int]
            One seed per attempt (``r_max_attempts`` of them).
        delr_sad_thr : float
            Acceptance distance, Å (``r_delr_sad_thr``).
        relax_radius : Optional[float]
            Radius of the pre-relaxation around the central atom, Å; None
            relaxes every atom but the rcut atoms.
        options : Optional[Mapping[str, Any]]
            Keyword options of :func:`pykmc.artn.artn_refine` and the worker
            options of :meth:`artn_search`, less ``av_radii``.

        Returns
        -------
        Optional[RefineResult]
            The refiner's result on the leader rank, with the pre-relaxation's
            iterations; None on the other ranks.

        Raises
        ------
        ValueError
            If ``av_radii`` is given (refinement runs on the full system
            only), ``attempt_options`` on a multi-rank worker, or an argument
            is malformed.

        """
        refine_options = dict(options or {})
        if refine_options.pop("av_radii", None) is not None:
            raise ValueError("artn_refine runs on the full system only")
        comm = self.engine.comm
        multi_rank = comm is not None and comm.Get_size() > 1
        if multi_rank and refine_options.get("attempt_options"):
            raise ValueError("attempt_options need a single-rank worker")
        refine_options.setdefault(
            "reuse_neighbor_list",
            supports_neighbor_reuse(search_pair_style(self.engine.config)),
        )
        positions = np.array(positions, dtype=float)
        saddle_idx = np.asarray(saddle_idx, dtype=int).ravel()
        start = positions.copy()
        start[saddle_idx] = np.asarray(saddle_positions, dtype=float)
        held = saddle_idx
        if relax_radius is not None:
            offsets = _minimum_image(
                positions - positions[central_atom],
                np.asarray(cell, dtype=float),
                np.asarray(pbc, dtype=bool),
            )
            far = np.flatnonzero(np.linalg.norm(offsets, axis=1) > relax_radius)
            held = np.union1d(held, far)

        # The pre-relaxation, on every rank of the worker's own engine.
        worker = cast("LammpsEngine", self.engine)
        worker.set_positions(positions=start)
        before = int(worker.lmp.extract_global("ntimestep"))
        worker.minimize_freeze_core(held)
        iterations = int(worker.lmp.extract_global("ntimestep")) - before
        start = worker.get_positions()
        if multi_rank:
            from .artn_ranks import distributed_search

            start = comm.bcast(start, root=0)
            result = distributed_search(
                worker,
                start,
                types,
                cell,
                pbc,
                movable,
                central_atom,
                0,
                {
                    **refine_options,
                    "seeds": seeds,
                    "delr_sad_thr": delr_sad_thr,
                    "minimum": positions,
                },
                core=_refine_core,
            )
        else:

            def core(
                x: np.ndarray,
                cell: np.ndarray,
                pbc: np.ndarray,
                mask: np.ndarray,
                central: int,
                _seed: int,
                query: Any,
                **kwargs: Any,
            ) -> RefineResult:
                return run_refine(
                    x,
                    cell,
                    pbc,
                    mask,
                    central,
                    seeds,
                    query,
                    delr_sad_thr=delr_sad_thr,
                    minimum=positions,
                    masses=self._instance_masses(),
                    **kwargs,
                )

            result = self._search(
                core,
                start,
                types,
                cell,
                pbc,
                movable,
                central_atom,
                0,
                refine_options,
            )
        return (
            None if result is None else result._replace(prerelax_iterations=iterations)
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
            validate_dimer_neighbor_reuse(search_pair_style(self.engine.config))

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


def _refine_core(
    positions: np.ndarray,
    cell: np.ndarray,
    pbc: np.ndarray,
    movable: np.ndarray,
    central_atom: int,
    _seed: int,
    energy_and_forces: Any,
    *,
    seeds: list[int],
    **options: Any,
) -> RefineResult:
    """Call :func:`pykmc.artn.artn_refine` as the rank driver calls a search."""
    return run_refine(
        positions, cell, pbc, movable, central_atom, seeds, energy_and_forces, **options
    )
