# Troubleshooting

Common problems when installing and running pyKMC, and how to fix them.

## MPI session allocation errors

pyKMC runs a pool of LAMMPS engine sessions over MPI. When
`engine_use_rank_0 = False` (the default), rank 0 acts only as the orchestrator,
so you need **`world_size >= n_sessions + 1`**.

**Symptom:** the run aborts at start-up complaining that there are not enough
ranks for the requested number of sessions.

**Fix:** either launch more MPI ranks or lower `n_sessions`. For example, with
`n_sessions = 7` and `engine_use_rank_0 = False`:

```bash
mpirun -n 8 python -m pykmc -in input.in
```

See [`n_sessions`](parameters.md) and `engine_use_rank_0` on the
[KMC Parameters](parameters.md) page.

## Import errors for LAMMPS, IRA, or pARTn

**Symptom:** `ModuleNotFoundError` / `ImportError` for `lammps`, `ira_mod`, or
`pypARTn`.

**Fix:**

1. Make sure the pyKMC virtual environment is active
   (`source pykmc_env/bin/activate`).
2. Confirm each component built and installed into that environment. Run the
   smoke test:

   ```bash
   python -c "from lammps import lammps; import ase, pykmc, ira_mod, pypARTn; print('All imports OK')"
   ```

3. If only `lammps` is missing, you likely skipped `make install-python`
   (make build) or the cmake `make install-python` step. Rebuild following the
   [installation guide](user_guide/install/installation.md).

## `mpi4py` segfault (crash in `MPI_Allreduce`)

**Symptom:** the run segfaults early, often inside an MPI collective, when
launched with `mpirun`/`srun`.

**Cause:** the pre-built `mpi4py` wheel is linked against a *different* MPI
library than the one LAMMPS uses.

**Fix:** rebuild `mpi4py` from source against your MPI:

```bash
export CC=mpicc CXX=mpicxx FC=mpif90
python -m pip install --no-binary mpi4py mpi4py --force-reinstall
```

On DRAC/Alliance clusters with the `mpi4py` module loaded, skip the pip install
and use the module instead.

## Library not found when loading LAMMPS

**Symptom:** an error loading the LAMMPS shared library (`liblammps`) at import
time.

**Fix:** make sure the shared-library search path includes the LAMMPS build and
your MPI/FFTW libraries:

- **macOS:** `export DYLD_LIBRARY_PATH="$(brew --prefix)/lib:${DYLD_LIBRARY_PATH}"`
- **Linux:** `export LD_LIBRARY_PATH="/path/to/lammps/build:${LD_LIBRARY_PATH}"`

The one-shot installers write an `activate.sh` that sets these for you — just
`source activate.sh`.

## Configuration validation errors

**Symptom:** the run stops immediately with a pydantic validation error naming a
section or field.

**Cause:** pyKMC validates the entire input file against a typed `Config` model
before running. A misspelled section header, an unknown field, or a wrong type
will fail validation.

**Fix:** check the offending name against the [KMC Parameters](parameters.md)
page (every section and field is listed there with its type and default).
Section headers must match exactly, e.g. `[Control]`, `[Engine]`, `[Partn]`,
`[ActiveVolume]`, `[Basin]`.

## Active Volume searches or refinements fail

**Symptom:** with `active_volume = True`, event searches or refinements fail or
give wrong energies.

**Cause:** the active radius `ract` or movable radius `rmov` is smaller than the
potential's cutoff `rcut`. If `rmov < rcut` refinements fail; if `ract < rcut`
event searches fail. The buffer region (`ract − rmov`) must be at least one
cutoff thick so boundary atoms feel correct forces.

**Fix:** increase `ract`/`rmov`, and use `AV_debug = True` in the
`[ActiveVolume]` section to check whether the volume is large enough. See
[Active Volumes](user_guide/active_volumes.md).

## Event search appears to hang

**Symptom:** the simulation seems stuck during an event-search step.

**Cause:** saddle-point searches are genuinely compute-intensive, especially for
large environments or many search attempts (`nsearch`). This is usually slow
progress, not a hang.

**Fix:** monitor CPU usage to confirm work is happening. Consider reducing the
number of searches per environment, or use [Active Volumes](user_guide/active_volumes.md)
to shrink the region searched in large systems.
