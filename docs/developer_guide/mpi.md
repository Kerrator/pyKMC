# MPI & Parallel Execution

How pyKMC distributes work over MPI ranks. The user-facing configuration is
covered in [Parallelization](../user_guide/parallelization.md); this page is
the developer-level outline.

## Rank-0 orchestrator and the session pool

Rank 0 runs the main KMC loop (environment classification, event selection,
bookkeeping). The remaining ranks are grouped into **sessions**, each hosting
one LAMMPS instance. The engine manager (`pykmc/enginemanager/`) dispatches
event searches and refinements to free sessions and collects results.

## World size rule

$$
\text{world\_size} \geq n\_\text{sessions} + 1
\qquad (\texttt{engine\_use\_rank\_0 = False})
$$

Session allocation fails at start-up if the MPI world is too small.

## `engine_use_rank_0`

When `True`, rank 0 hosts a LAMMPS instance *in addition to* the KMC loop.
That instance communicates via threads instead of MPI messages, which can
slightly slow the run — the trade-off is one extra session on a fixed core
budget.

## Core splitting

Ranks are partitioned contiguously across sessions. Example with
`mpirun -n 8`, `n_sessions = 4`, `engine_use_rank_0 = False`: sessions occupy
ranks 1–2, 3–4, 5–6, and 7; rank 0 orchestrates.

## Local vs global pool mode

The manager exposes the same pool in two modes. In **local** mode each session
works independently on its own communicator — this is how concurrent event
searches and refinements are farmed out. In **global** mode the engine ranks
are joined so whole-system operations (e.g. reconstruction of the selected
event) run on one large LAMMPS instance. The KMC loop switches between them
with `manager.use_local()` / `manager.use_global()`; any driver code doing
standalone searches must follow the same sequencing.

## Launching: `mpirun` vs `srun`

- Local: `mpirun -n N python -m pykmc -in input.in`.
- SLURM clusters: `srun --ntasks=$SLURM_NTASKS --distribution=block:block
  --cpu-bind=cores --mem-bind=local python -m pykmc -in input.in`.

**Caveat:** `mpi4py` must be built against the same MPI library that LAMMPS
links to; mismatched pre-built wheels segfault inside collectives. See
[Troubleshooting](../troubleshooting.md).

## See also

- [Parallelization (user guide)](../user_guide/parallelization.md)
- [Architecture](architecture.md) — the engine manager's place in the package
