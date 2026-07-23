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

See [`n_sessions`](parameters.md#control-n_sessions) and
[`engine_use_rank_0`](parameters.md#control-engine_use_rank_0) on the
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

3. If only `lammps` is missing, the LAMMPS shared library may have been built
   without installing its Python bindings into the active virtual environment.
   From the configured LAMMPS build directory, run `make install-python`, then
   repeat the import check — see the
   [installation guide](user_guide/install/installation.md).

## `mpi4py` segfault (crash in `MPI_Allreduce`)

**Symptom:** the run segfaults early, often inside an MPI collective, when
launched with `mpirun`/`srun`.

**Possible cause:** `mpi4py` and LAMMPS were built against different MPI
implementations. Confirm the loaded libraries before rebuilding, because MPI
collective crashes can have other causes.

**Fix (for the mismatch case):** rebuild `mpi4py` from source against your MPI:

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

- **macOS:** `export DYLD_LIBRARY_PATH="$(brew --prefix)/lib:${DYLD_LIBRARY_PATH:-}"`
- **Linux:** `export LD_LIBRARY_PATH="/path/to/lammps/build:${LD_LIBRARY_PATH:-}"`

The one-shot installers write an `activate.sh` that activates the virtual
environment and cleans stale pyKMC entries from `PYTHONPATH`; it does **not**
set a shared-library search path, so export the variable above yourself if
your LAMMPS installation needs one.

## Configuration validation errors

**Symptom:** the run stops immediately with a pydantic validation error naming a
section or field.

**Cause:** pyKMC validates the input file against a typed `Config` model
before running. Section names are lowercased on parsing, so section
capitalization is case-insensitive; field names remain case-sensitive.
Missing required sections or fields, invalid literals, and invalid types
raise validation errors. Unknown fields and unrecognized optional sections
are currently **ignored**, so a typo'd optional key silently falls back to
its default rather than failing.

**Fix:** check the offending name against the [KMC Parameters](parameters.md)
page (every section and field is listed there with its type and default).
The base required sections are `[Control]`, `[AtomicEnvironment]`,
`[EventSearch]`, `[RateConstant]`, and `[PSR]`; with the currently supported
styles, `[LAMMPS]`, `[pARTn]`, and `[IRA]` are also required.

## Active Volume searches or refinements fail

**Symptom:** with `active_volume = True`, event searches or refinements fail or
give wrong energies.

**Cause:** the Active Volume radii are too small. `ract` must be strictly
greater than the `[AtomicEnvironment]` local-environment radius `rcut`, or
event search raises an error. There is no automatic check on `rmov`, but the
frozen buffer (`ract − rmov`) must be at least as thick as the interatomic
potential's interaction cutoff so boundary atoms feel correct forces, and
`rmov` must contain the reconstructed local environment.

**Fix:** increase `ract`/`rmov`, and use `AV_debug = True` in the
`[ActiveVolume]` section to inspect the printed before/after energies and
their percentage difference (pyKMC applies no automatic threshold). See
[Active Volumes](user_guide/active_volumes.md).

## Event search appears to hang

**Symptom:** the simulation seems stuck during an event-search step.

**Cause:** saddle-point searches are genuinely compute-intensive, especially for
large environments or many search attempts (`nsearch`). This is usually slow
progress, not a hang.

**Fix:** monitor CPU usage to confirm work is happening. Reducing `nsearch`
shortens discovery but also reduces catalog coverage; do so only after
confirming that enough distinct events survive. For large systems,
[Active Volumes](user_guide/active_volumes.md) can reduce the atoms included
in each search.

## No events survive discovery

**Symptom:** pyKMC closes with an empty reference event table.

**Checks:** inspect the pARTn output and the pyKMC event log. Distinguish
searches that found no saddle from events rejected by `emin_event`,
`emax_event`, the backward-barrier test, the asymmetry test, duplicate
detection, or inactive-atom filtering. Increase `nsearch` only when the
searches are completing successfully but sampling too few distinct paths;
otherwise tune the failing pARTn or acceptance setting.

## All active-event reconstructions fail

**Symptom:** candidate events are removed one by one and the run closes with
the logged message `All event reconstuctions failed.`

**Checks:** confirm that `matching_score_thr` and `push_fraction` are
appropriate, that the current `rcut` and neighbor set are compatible with the
stored event geometry, and that the reference table was generated for the
same chemistry, potential, and atomic-environment style.
