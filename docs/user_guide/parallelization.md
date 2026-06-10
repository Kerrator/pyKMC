# Parallelization

pyKMC parallelizes the expensive parts of the KMC loop — event searches and
refinements — by running a pool of independent LAMMPS instances over MPI, each
on its own set of cores. This page covers how to configure and launch a
parallel run; the implementation details live in the
[developer guide](../developer_guide/mpi.md).

## Why multiple LAMMPS instances

Event searches at different atoms are independent of one another, so they can
run concurrently. Each LAMMPS instance (a *session*) handles one search or
refinement at a time; with `N` sessions, up to `N` searches proceed in
parallel while the main KMC loop orchestrates from rank 0.

## `n_sessions`

The number of LAMMPS instances is set with the `n_sessions` parameter in the
`[Control]` section of the input file.

## `engine_use_rank_0`

The main KMC loop always runs on rank 0. The `engine_use_rank_0` parameter
(`[Control]`) chooses whether the LAMMPS instances use all *other* ranks only
(`False`) or include rank 0 as well (`True`). Including rank 0 can slightly
slow the simulation, since that instance communicates via threads rather than
MPI messages.

## World size requirement

With `engine_use_rank_0 = False`, rank 0 is reserved for orchestration, so the
MPI world must satisfy:

$$
\text{world\_size} \geq n\_\text{sessions} + 1
$$

If too few ranks are launched, the run aborts at start-up — see
[Troubleshooting](../troubleshooting.md).

## Worked example: 8 cores, 4 sessions

```INI
[Control]
...
n_sessions = 4
engine_use_rank_0 = False
...
```

```bash
mpirun -n 8 python -m pykmc -in <your_input_file>
```

The available cores are split automatically: the LAMMPS instances run on ranks
1–2, 3–4, 5–6, and 7, while the main KMC loop runs on rank 0.

## `mpirun` vs `srun`

- **Local workstation:** launch with `mpirun -n N`.
- **HPC clusters (SLURM):** launch with `srun` and let SLURM size the world,
  e.g.

  ```bash
  srun --ntasks=$SLURM_NTASKS --distribution=block:block \
       --cpu-bind=cores --mem-bind=local \
       python -m pykmc -in input.in
  ```

If the run crashes inside MPI collectives, check that `mpi4py` was built
against the same MPI library as LAMMPS — see
[Troubleshooting](../troubleshooting.md).
