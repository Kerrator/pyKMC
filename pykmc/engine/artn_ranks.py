"""Rank-local forces and vector operations for one faithful ARTn search.

Each rank keeps the movable rows it owned at the start, its home rows, for the
whole search. LAMMPS may move an atom to another rank at any force call, so each
row is exchanged with the atom's current LAMMPS owner. Every FIRE and Lanczos
vector, saved endpoints included, keeps the home row order. Only the result is
assembled, on rank zero.
"""

from __future__ import annotations

from contextlib import ExitStack
from typing import TYPE_CHECKING, Any

import numpy as np
from ase.cell import Cell
from mpi4py import MPI

from ..artn import (
    DrawOrder,
    NumpyVectorSpace,
    _draw_rows,
    _wrap_result_positions,
    artn_search,
)
from ..config import validate_dimer_eam_style, validate_dimer_neighbor_reuse
from ..dimer import DimerResult, _wrap
from .dimer_forces import (
    ForceQuery,
    _NEIGHBOR_SKIN,
    _crop_selection,
    _interaction_bound,
    configure_neighbor_reuse,
)

if TYPE_CHECKING:
    from .lammps import LammpsEngine


class DistributedSpace(NumpyVectorSpace):
    """Reductions over all ranks of the worker, and each atom's home row."""

    def __init__(
        self,
        comm: MPI.Intracomm,
        home: np.ndarray,
        mask: np.ndarray,
        start: np.ndarray,
        participants: np.ndarray | None = None,
    ) -> None:
        self.comm = comm
        self.home = np.sort(home)
        self.mask = mask
        self.start = start
        self.participants = mask if participants is None else participants
        # Split the directory from atom ID to home rank over the ranks. Every
        # rank has the worker request, but no rank gathers the owners or forces.
        size = comm.Get_size()
        rank = comm.Get_rank()
        outgoing = [self.home[self.home % size == r] for r in range(size)]
        self.registrations = comm.alltoall(outgoing)
        registered = np.concatenate(self.registrations)
        expected = np.flatnonzero(self.participants[rank::size]) * size + rank
        if not self.all_true(np.array_equal(np.sort(registered), expected)):
            raise ValueError("participating atoms must each have one initial owner")
        self.directory = np.full(len(mask[rank::size]), -1, dtype=int)
        for owner, ids in enumerate(self.registrations):
            self.directory[ids // size] = owner

    def owners(self, ids: np.ndarray) -> np.ndarray:
        """Return each atom's home rank, from the directory for other ranks' atoms."""
        comm = self.comm
        size = comm.Get_size()
        result = np.full(len(ids), comm.Get_rank(), dtype=int)
        unknown = ~np.isin(ids, self.home)
        if not self.all_true(not unknown.any()):
            selectors = [
                np.flatnonzero(unknown & (ids % size == r)) for r in range(size)
            ]
            requests = comm.alltoall([ids[selection] for selection in selectors])
            replies = comm.alltoall([self.directory[req // size] for req in requests])
            for selection, owner in zip(selectors, replies, strict=True):
                result[selection] = owner
        return result

    def atom_count(self, local: int) -> int:
        """Return the number of atoms in the whole system, frozen atoms included."""
        return len(self.mask)

    def count(self, local: int) -> int:
        """Sum a count over all ranks."""
        return self.comm.allreduce(local, op=MPI.SUM)

    def all_true(self, local: bool) -> bool:
        """Return whether a condition holds on every rank."""
        return self.comm.allreduce(bool(local), op=MPI.LAND)

    def dot(self, a: np.ndarray, b: np.ndarray) -> float:
        """Sum the local dot products of all ranks."""
        return self.comm.allreduce(float(np.vdot(a, b)), op=MPI.SUM)

    def max_abs(self, a: np.ndarray) -> float:
        """Return the largest component over all ranks.

        A rank with no rows takes part, and a non-finite value counts as infinite.
        """
        local = super().max_abs(a)
        return self.comm.allreduce(local if np.isfinite(local) else np.inf, op=MPI.MAX)

    def draw_orders(
        self,
        x: np.ndarray,
        central: int,
        cell: np.ndarray,
        pbc: np.ndarray,
        mask: np.ndarray,
        push_mode: str,
        radius: float,
    ) -> tuple[DrawOrder, DrawOrder]:
        """Split the serial draw order, offset ties included, among the home ranks."""
        comm = self.comm
        size = comm.Get_size()
        orders = (
            _draw_rows(self.start, central, cell, pbc, self.mask, push_mode, radius)
            if comm.Get_rank() == 0
            else (None, None)
        )

        def distribute(order: np.ndarray | None) -> DrawOrder:
            # Rank zero already has the input geometry. Send the integer draw
            # ordinals through the directory to the home rank of each atom.
            partitions = None
            if order is not None:
                ordinals = np.full(len(self.mask), -1, dtype=int)
                ordinals[order] = np.arange(len(order))
                partitions = [(ordinals[r::size], len(order)) for r in range(size)]
            shard, total = comm.scatter(partitions, root=0)
            requests = [
                np.column_stack((ids, shard[ids // size])) for ids in self.registrations
            ]
            mapped = np.concatenate(comm.alltoall(requests))
            drawn = mapped[mapped[:, 1] >= 0]
            drawn = drawn[np.argsort(drawn[:, 1])]
            return DrawOrder(
                np.searchsorted(self.home, drawn[:, 0]), drawn[:, 1], total
            )

        return distribute(orders[0]), distribute(orders[1])

    def collect_result(
        self, result: DimerResult, cell: np.ndarray, pbc: np.ndarray
    ) -> DimerResult | None:
        """Assemble on rank zero the result geometries that are set, in pyKMC order."""
        names = [
            k
            for k in ("saddle_positions", "min1_positions", "min2_positions", "mode")
            if getattr(result, k) is not None
        ]
        if not names:
            return result if self.comm.Get_rank() == 0 else None
        parts = self.comm.gather(
            (self.home, [getattr(result, k) for k in names]), root=0
        )
        if parts is None:
            return None
        fields = {
            k: (
                np.zeros_like(self.start)
                if k == "mode"
                else _wrap_result_positions(self.start, cell, pbc)
            )
            for k in names
        }
        for ids, arrays in parts:
            for name, values in zip(names, arrays, strict=True):
                fields[name][ids] = values
        return result._replace(
            saddle_positions=fields.get("saddle_positions"),
            min1_positions=fields.get("min1_positions"),
            min2_positions=fields.get("min2_positions"),
            mode=fields.get("mode"),
        )


class OwnedRows:
    """Send a row between its LAMMPS owner and its home rank only when they differ."""

    def __init__(self, space: DistributedSpace) -> None:
        self.space = space
        self.previous: np.ndarray | None = None

    def update(self, ids: np.ndarray) -> None:
        """Rebuild the routes when the atoms' owners or local order change."""
        space = self.space
        if space.all_true(
            self.previous is not None and np.array_equal(ids, self.previous)
        ):
            return
        self.previous = ids.copy()
        rank = space.comm.Get_rank()
        owner = space.owners(ids)
        self.engine_rows = [
            np.flatnonzero(owner == r) for r in range(space.comm.Get_size())
        ]
        self.remote = not space.all_true(bool(np.all(owner == rank)))
        # For each current engine rank, the home rows requested by that rank.
        requests = [ids[rows] for rows in self.engine_rows]
        if self.remote:
            own = requests[rank]
            requests[rank] = ids[:0]
            requests = space.comm.alltoall(requests)
            requests[rank] = own
        received = np.concatenate(requests)
        if not space.all_true(np.array_equal(np.sort(received), space.home)):
            raise RuntimeError("lost or duplicate movable atom after LAMMPS migration")
        self.home_rows = [np.searchsorted(space.home, req) for req in requests]

    def transfer(self, values: np.ndarray, *, to_home: bool) -> np.ndarray:
        """Copy per-atom values between the LAMMPS owners and the home rows."""
        rank = self.space.comm.Get_rank()
        src, dst = (
            (self.engine_rows, self.home_rows)
            if to_home
            else (self.home_rows, self.engine_rows)
        )
        out = np.empty((sum(map(len, dst)), *values.shape[1:]), dtype=values.dtype)
        out[dst[rank]] = values[src[rank]]
        if self.remote:
            chunks = [
                values[rows] if r != rank else values[:0] for r, rows in enumerate(src)
            ]
            for r, chunk in enumerate(self.space.comm.alltoall(chunks)):
                if r != rank:
                    out[dst[r]] = chunk
        return out


class RankForces:
    """Evaluate one full or cropped LAMMPS instance without gathering forces."""

    def __init__(
        self,
        engine: LammpsEngine,
        system_ids: np.ndarray,
        space: DistributedSpace,
        cell: np.ndarray,
        pbc: np.ndarray,
        reuse: bool,
    ) -> None:
        self.engine = engine
        if engine.lmp is None:
            raise ValueError("the force engine must be started")
        self.lmp = engine.lmp
        self.system_ids = system_ids
        self.space = space
        self.cell = cell
        self.pbc = pbc
        self.reuse = reuse
        self.previous = space.start[space.home]
        self.routes = OwnedRows(space)
        self.refresh()

    def refresh(self) -> None:
        """Read this rank's owned rows again, ghosts excluded, after each LAMMPS run."""
        lmp = self.lmp
        self.nlocal = lmp.extract_global("nlocal")
        ids = (
            self.system_ids[lmp.numpy.extract_atom("id")[: self.nlocal] - 1]
            if self.nlocal
            else self.system_ids[:0]
        )
        self.rows = np.flatnonzero(self.space.participants[ids])
        self.routes.update(ids[self.rows])

    def masses(self) -> np.ndarray:
        """Read the force engine's type masses in the core's row order."""
        atoms = self.lmp.numpy
        types = (
            atoms.extract_atom("type")[self.rows]
            if len(self.rows)
            else np.empty(0, dtype=int)
        )
        return self.routes.transfer(
            np.asarray(atoms.extract_atom("mass"))[types], to_home=True
        )

    def values(self) -> tuple[float, np.ndarray]:
        """Return the evaluated potential energy and local forces in pyKMC axes."""
        engine = self.engine
        energy = float(self.lmp.get_thermo("pe"))
        local = engine._positions_from_lammps(
            self.lmp.numpy.extract_atom("f")[self.rows]
            if len(self.rows)
            else np.empty((0, 3))
        )
        # The serial force instance zeroes the frozen group's force with setforce.
        # A replay injection can give frozen atoms velocities, never a force.
        local[~self.space.mask[self.routes.previous]] = 0.0
        return energy, self.routes.transfer(local, to_home=True)

    def __call__(self, x: np.ndarray) -> tuple[float, np.ndarray]:
        """Evaluate energy and forces at ``x`` through the current LAMMPS owners."""
        engine = self.engine
        delta = x - self.previous
        small = self.reuse and self.space.max_abs(delta) <= _NEIGHBOR_SKIN / 2
        if small and not self.pbc.all():
            fixed = x @ np.linalg.inv(self.cell)[:, ~self.pbc]
            small = self.space.all_true(bool(np.all((fixed >= 0) & (fixed < 1))))
        if small:
            local = self.routes.transfer(delta, to_home=False)
            if self.nlocal:
                velocity = self.lmp.numpy.extract_atom("v")
                velocity[: self.nlocal] = 0.0
                velocity[self.rows] = engine._positions_to_lammps(local)
            engine.command("run 1 pre no post no")
        else:
            target = self.routes.transfer(_wrap(x, self.cell, self.pbc), to_home=False)
            if len(self.rows):
                self.lmp.numpy.extract_atom("x")[self.rows] = (
                    engine._positions_to_lammps(target)
                )
            # A jump back to the saddle may cross more than one subdomain.
            # This command invokes LAMMPS's irregular atom migration.
            engine.command("displace_atoms all move 0 0 0 units box")
            engine.command("run 0 post no")
        self.previous = x.copy()
        self.refresh()
        return self.values()


def distributed_search(
    worker: LammpsEngine,
    positions: np.ndarray,
    types: list[str] | np.ndarray,
    cell: Cell | np.ndarray,
    pbc: list[bool] | np.ndarray,
    movable: list[int] | np.ndarray,
    central: int,
    seed: int,
    options: dict[str, Any],
) -> DimerResult | None:
    """Run the ARTn core on each rank's home rows and return the result on rank zero."""
    from .lammps import LammpsEngine

    comm = worker.comm
    if comm is None:
        raise ValueError("distributed ARTn requires the worker communicator")
    positions = np.asarray(positions, dtype=float)
    types = np.asarray(types)
    cell = Cell(cell)
    pbc = np.asarray(pbc, dtype=bool)
    n = len(positions)
    if positions.shape != (n, 3) or n == 0 or types.shape != (n,) or pbc.shape != (3,):
        raise ValueError("invalid ARTn system shape")
    if not 0 <= central < n:
        raise ValueError("central_atom is not an atom index")
    movable = np.asarray(movable).ravel()
    if movable.size and (
        not np.issubdtype(movable.dtype, np.integer)
        or movable.min() < 0
        or movable.max() >= n
    ):
        raise ValueError("movable must contain integer indices within the system")
    mask = np.zeros(n, dtype=bool)
    mask[movable.astype(int)] = True
    options = dict(options)
    style = worker.config.pair_style
    reuse = options.pop("reuse_neighbor_list", style.strip() in {"eam/alloy", "eam/fs"})
    crop = options.pop("eam_force_crop", False)
    if not all(isinstance(x, (bool, np.bool_)) for x in (crop, reuse)):
        raise ValueError("eam_force_crop and reuse_neighbor_list must be booleans")
    if reuse:
        validate_dimer_neighbor_reuse(style)
    if crop:
        validate_dimer_eam_style(style, "eam_force_crop")
    participants = mask.copy()
    injections = {}
    for key in ("initial_push", "first_lanczos_vector", "new_chance_pushes"):
        value = options.get(key)
        if value is None:
            continue
        arrays = [
            np.asarray(v, dtype=float)
            for v in (value if key == "new_chance_pushes" else [value])
        ]
        if any(v.shape != (n, 3) for v in arrays):
            raise ValueError(f"{key} must supply full-system (N, 3) vectors")
        injections[key] = arrays
        if key != "initial_push":
            for v in arrays:
                participants |= np.any(v != 0, axis=1)
    lattice = np.asarray(cell)
    with ExitStack() as resources:

        def instance(ids: np.ndarray, name: str) -> LammpsEngine:
            engine = LammpsEngine(
                worker.config,
                worker.comm,
                engine_id=f"artn.{name}.{worker.engine_id}",
                search_instance=True,
            )
            resources.callback(engine.close)
            engine.start()
            engine.initialize_parameters()
            engine.initialize_system(
                types[ids],
                _wrap(positions[ids], lattice, pbc),
                cell,
                pbc,
                species=sorted(set(types)),
            )
            if engine.lmp is None or int(engine.lmp.get_natoms()) != len(ids):
                raise ValueError(
                    "LAMMPS did not create every atom in the force instance"
                )
            engine.initialize_potential()
            if reuse:
                configure_neighbor_reuse(engine)
            engine.command("run 0 post no")
            return engine

        ids = np.arange(n)
        engine = instance(ids, "full")
        assert engine.lmp is not None
        nlocal = engine.lmp.extract_global("nlocal")
        local = engine.lmp.numpy.extract_atom("id")[:nlocal] - 1 if nlocal else ids[:0]
        space = DistributedSpace(
            comm, local[participants[local]], mask, positions, participants
        )
        full = RankForces(engine, ids, space, lattice, pbc, reuse)
        initial = full.values()
        masses = full.masses()
        query: ForceQuery = full
        if (
            crop
            and np.array_equal(mask, participants)
            and mask.any()
            and not mask.all()
            and space.all_true(
                bool(np.isfinite(initial[0]) and np.isfinite(initial[1]).all())
            )
        ):
            cutoff = _interaction_bound(engine, reuse)
            if cutoff is not None:
                selection = _crop_selection(
                    positions, mask, positions[central], lattice, pbc, cutoff
                )
                if selection is not None:
                    kept, within_reach = selection
                    cropped = RankForces(
                        instance(kept, "crop"), kept, space, lattice, pbc, reuse
                    )
                    offset = initial[0] - float(cropped.lmp.get_thermo("pe"))
                    if np.isfinite(offset):

                        def crop_query(x: np.ndarray) -> tuple[float, np.ndarray]:
                            nonlocal query
                            if space.all_true(within_reach(x)):
                                e, f = cropped(x)
                                if space.all_true(
                                    bool(np.isfinite(e) and np.isfinite(f).all())
                                ):
                                    return e + offset, f
                            query = full
                            return full(x)

                        query = crop_query

        first: tuple[float, np.ndarray] | None = initial
        del initial

        def evaluate(x: np.ndarray) -> tuple[float, np.ndarray]:
            nonlocal first
            if first is not None:
                values, first = first, None
                return values
            return query(x)

        for key, vectors in injections.items():
            local_vectors = [v[space.home] for v in vectors]
            options[key] = (
                local_vectors if key == "new_chance_pushes" else local_vectors[0]
            )
        result = artn_search(
            positions[space.home],
            lattice,
            pbc,
            mask[space.home],
            central,
            seed,
            evaluate,
            masses=masses,
            vectors=space,
            **options,
        )
        return space.collect_result(result, lattice, pbc)
