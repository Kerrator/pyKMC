# pyKMC HTST versus constant A/B benchmark

This benchmark measures full pyKMC runs with and without online HTST. It uses
one base input and generates paired configurations that differ only in:

```ini
[RateConstant]
style = constant  # A
style = htst      # B
```

The runner copies the initial configuration and potential files into isolated
run directories. It records total launcher wall time, pyKMC per-step wall time,
launcher start/end timestamps, reference-event counts, rate ranges, and HTST
prefactor coverage. Runs are launched sequentially, so the constant and HTST
jobs never overlap.

Every generated input forces `basin = False` and `recycle = False`, removes any
`[EventRecycling]` section, and starts without restart/reference-table inputs.

## Required software

- The `SK_HTST_backend` pyKMC branch installed in a Python environment.
- Python dependencies from `pyproject.toml`.
- LAMMPS built with `PHONON` and `PLUGIN`.
- pARTn's Python interface and `libartn-lmp.so`.
- IRA's Python interface and `libira.so`.
- An MPI launcher compatible with the `mpi4py` and LAMMPS builds.

The runner checks these dependencies before starting. A failure at this stage
usually means the Python environment, MPI implementation, or native library
RPATHs do not match.

## Prepare one base input

Use a system already known to complete a normal pyKMC run. The base input may
use either prefactor style because the runner replaces it. Use relative paths
for the structure and potential where practical:

```ini
[Control]
initial_config = ./initial_config.xyz
n_steps = 1
engine = lammps
n_sessions = 7
engine_use_rank_0 = False
basin = False

[Lammps]
pair_style = eam/alloy
pair_coeff = * * ./NiAlH_jea.eam Ni

[pARTn]
zseed = 19073

[RateConstant]
style = constant
k0 = 1e13
T = 500
```

Keep the remaining event-search, atomic-environment, PSR, and IRA settings from
the validated system. The runner sets the same nonzero pARTn and Python random
seed for each A/B pair.

## Preflight

From the pyKMC worktree:

```bash
PYTHON=/path/to/pykmc_env/bin/python

"$PYTHON" scripts/run_htst_ab.py \
  --input /path/to/system/input.in \
  --output /scratch/htst_ab_preflight \
  --python "$PYTHON" \
  --launcher mpirun \
  --nproc 8 \
  --steps 1 \
  --nsearch 1 \
  --preflight-only
```

For Slurm, replace the launcher arguments with:

```bash
--launcher srun --nproc-flag=--ntasks --nproc 8
```

`nproc` must be at least `n_sessions` when `engine_use_rank_0 = True`, or
`n_sessions + 1` when it is false. Additional ranks are divided among the
configured sessions.

## Recommended run sequence

First isolate the cost of computing HTST on the same initial state:

```bash
"$PYTHON" scripts/run_htst_ab.py \
  --input /path/to/system/input.in \
  --output /scratch/htst_ab_one_step \
  --python "$PYTHON" \
  --launcher mpirun \
  --nproc 8 \
  --steps 1 \
  --nsearch 1 \
  --repeats 5
```

Then run a multi-step characterization to include event reuse and trajectory
effects:

```bash
"$PYTHON" scripts/run_htst_ab.py \
  --input /path/to/system/input.in \
  --output /scratch/htst_ab_ten_steps \
  --python "$PYTHON" \
  --launcher mpirun \
  --nproc 8 \
  --steps 10 \
  --repeats 3
```

The run order alternates between repeats to reduce first-run/cache bias.

## Outputs

- `summary.json`: pass/fail checks, environment metadata, median wall times, and
  the HTST/constant wall-time ratio.
- `runs.csv`: one row per style and repeat, including launcher start/end and
  elapsed wall time.
- `repeat_NNN/{constant,htst}/`: generated input, copied assets, pyKMC outputs,
  and captured launcher stdout/stderr.
- `metadata.json`: host, launcher, rank allocation, input hash, and preflight
  details.

The benchmark fails if either run produces no reference events, if the constant
table unexpectedly contains `nu0`, or if HTST produces no finite `nu0` values.
Partial HTST coverage remains a passing smoke result but is emitted as an
explicit warning with the number of events that used the `k0` fallback.

## Interpretation limits

The one-step benchmark is the cleanest measure of online HTST overhead. In a
multi-step run, HTST changes event probabilities and can lead the simulation
through different states. Its total wall-time ratio therefore includes both
the Hessian cost and legitimate trajectory/event-count differences.

Even with matched Python, NumPy, and pARTn seeds, parallel pARTn scheduling can
produce slightly different accepted-event counts. Compare `n_events` and
`wall_per_event_s` alongside total wall time. Treat a pair with substantially
different event counts as workload characterization, not a controlled timing
ratio.

Do not interpret a low HTST coverage as a performance success. Inspect
`nu0_coverage`, `launcher.stderr`, and `pykmc.log`; fallback to `k0` means the
run was only partially using HTST.
