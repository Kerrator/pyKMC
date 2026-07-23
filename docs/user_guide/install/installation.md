# Installation

pyKMC is a Python package, but a working installation also requires **LAMMPS**
(the energy/force engine), **pARTn** (saddle-point search), and **IRA**
(point-set registration and symmetry detection) — all three are imported by
the current implementation. The fastest way to get a complete, working install
is the one-shot script for your platform.

## Platform support

| Platform | Recommended path | Detailed guide |
|---|---|---|
| macOS (Apple Silicon) | One-shot script `install_pykmc_mac.sh` | [macOS instructions](pykmc_mac_Compile_Instructions.md) |
| Linux (Ubuntu/Debian, RHEL/CentOS/Rocky) | One-shot script `install_pykmc_linux.sh` | [Linux instructions](pykmc_linux_Compile_Instructions.md) |
| DRAC / Alliance HPC cluster | One-shot script on a login node (auto-detects clusters) or manual cmake + module loads — both validated on Trillium (2026-06) | [Linux instructions — cluster notes](pykmc_linux_Compile_Instructions.md#drac-alliance-hpc-clusters) |

## Quick install (recommended)

Each one-shot installer creates a `pykmc/` directory **inside your current
working directory** and builds pyKMC, LAMMPS, pARTn, and IRA into it
(roughly 10–20 minutes, mostly LAMMPS compilation). Pick where you want the
install to live, then run the script from there.

### macOS

Install Xcode Command Line Tools and [Homebrew](https://brew.sh) first. The
script checks for Homebrew and `git`, installs missing Homebrew dependencies,
and verifies `gfortran` plus the MPI compiler wrappers; it does not
independently verify the Xcode Command Line Tools installation. Then run the
installer:

```bash
mkdir -p /path/to/your/install-folder && cd /path/to/your/install-folder
chmod +x /path/to/install_pykmc_mac.sh
/path/to/install_pykmc_mac.sh 2>&1 | tee install.log
```

The script uses Homebrew to install any missing packages (`gcc`, `cmake`,
`fftw`, `open-mpi`), so no `sudo` is required. See the
[macOS instructions](pykmc_mac_Compile_Instructions.md) for the full
walkthrough, prerequisites, and troubleshooting.

### Linux

```bash
mkdir -p /path/to/your/install-folder && cd /path/to/your/install-folder
chmod +x /path/to/install_pykmc_linux.sh
/path/to/install_pykmc_linux.sh 2>&1 | tee install.log
```

The script prompts **once** for your `sudo` password so it can `apt`/`dnf`
install missing system packages, then runs unattended. See the
[Linux instructions](pykmc_linux_Compile_Instructions.md) for the full
walkthrough.

On **DRAC/Alliance HPC clusters** (Trillium, Narval, …) the script detects the
cluster automatically and skips the `sudo` stage — load the toolchain modules
first, run it on a **login node**, and run it from a directory under
`$SCRATCH`:

```bash
module load StdEnv/2023 gcc/12.3 openmpi/4.1.5 python/3.12.4 mpi4py/4.1.0 cmake/3.31.0
cd $SCRATCH
/path/to/install_pykmc_linux.sh 2>&1 | tee install.log
```

Both paths are **validated on Trillium (2026-06)**: the manual steps
end-to-end, and this script run unmodified on a login node (full install +
verification, then an 8-task `srun` smoke simulation from the generated
`activate.sh`). On other Alliance clusters, or if the script misbehaves, fall
back to the
[manual cluster steps](pykmc_linux_Compile_Instructions.md#drac-alliance-hpc-clusters),
which also document the ground rules (login vs compute nodes, filesystems,
sbatch templates).

Both scripts accept `PYTHON_BIN=/path/to/python` to choose a specific
interpreter. The package metadata requires Python 3.10 or newer, but the
one-shot installers currently validate Python 3.10–3.13; for a newer Python
release, use a tested interpreter in that range or update and validate the
installer. Each script produces an `activate.sh` you can `source` before
every run.

## What the scripts install

Under the `pykmc/` folder they create:

- `pykmc_env/` — a Python virtualenv with pyKMC installed editable (`pip install -e .`)
- `lammps/` — LAMMPS (`stable_22Jul2025_update3`) built shared, with Python bindings
- `artn-plugin/` — the pARTn plugin (`pypARTn` Python module)
- `IterativeRotationsAssignments/` — IRA (`ira_mod` Python module)
- `activate.sh` — activates the pyKMC virtual environment and removes stale
  pyKMC interface paths from `PYTHONPATH`. Set a shared-library search path
  separately if your LAMMPS installation requires one; installed wheels or
  embedded runtime paths may make that unnecessary.

## Manual / advanced installation

If you prefer to build by hand, are on an unsupported platform, or are
installing on an HPC cluster, follow the detailed step-by-step guide for your
OS — these are the authoritative manual instructions and stay in sync with the
one-shot scripts:

- [macOS — manual steps](pykmc_mac_Compile_Instructions.md#0-system-prerequisites)
- [Linux / cluster — manual steps](pykmc_linux_Compile_Instructions.md#0-system-prerequisites)

The essential sequence is:

1. Create and activate a virtual environment (Python ≥ 3.10) and run
   `pip install -e .` from the pyKMC repository.
2. Build LAMMPS as a **shared** library with Python bindings (cmake; enable at
   least the `BASIC`, `EXTRA-COMPUTE`, `MANYBODY`, and `PLUGIN` packages).
3. Build the pARTn plugin against that LAMMPS
   (`cmake -B build -DWITH_LAMMPS=ON -DLAMMPS_PATH=.../lammps/build -DARTN_INSTALL_PYTHON=ON`).
4. Build IRA (`python -m pip install .` from the IRA source).

### Minimal Python environment

If you already have LAMMPS, pARTn, and IRA built, installing pyKMC itself is
just:

```bash
python3 -m venv /path_to_environment/pykmc_env
source /path_to_environment/pykmc_env/bin/activate
cd /path_to/pyKMC
pip install -e .
```

### Other codes

- **LAMMPS** — a recent release is required (tested with
  `stable_22Jul2025_update3`). It must be built as a shared library with Python
  bindings; see the platform guides for the exact cmake invocation.
- **pARTn** — required by the current implementation, used for event
  (saddle-point) search. Project page:
  [pARTn](https://mammasmias.gitlab.io/artn-plugin/).
- **IRA** — required by the current implementation, used for point-set
  registration during event reconstruction and for symmetry detection.
  Project page:
  [IRA](https://mammasmias.github.io/IterativeRotationsAssignments/).

## Verify the install

With the environment active, run the full component check — it not only
imports each module but also starts a LAMMPS instance and loads the pARTn
plugin into it, which is what a real run requires:

```bash
python - <<'PY'
import ase
import ira_mod
import pykmc
import pypARTn
from lammps import lammps

lmp = lammps()
artn = pypARTn.artn(engine="lmp")
lmp.command(f"plugin load {artn.lib._name}")
print("Loaded liblammps:", lmp.lib._name)
print("Loaded pARTn plugin:", artn.lib._name)
print("All components OK")
PY
```

A successful run prints `All components OK`. If a step fails, see
[Troubleshooting](../../troubleshooting.md).

## Running

pyKMC runs under MPI. With the default `engine_use_rank_0 = False`, launch at
least `n_sessions + 1` ranks: rank 0 runs the main KMC loop and the remaining
ranks are split among the `n_sessions` LAMMPS instances (`[Control]` section
of the input file). With `engine_use_rank_0 = True`, rank 0 also hosts the
first engine session, so `n_sessions` ranks suffice.

```bash
mpirun -n 8 python -m pykmc -in input.in
```

See [Parallelization](../parallelization.md) for `n_sessions`,
`engine_use_rank_0`, world-size requirements, and rank splitting.

On a cluster, submit through your scheduler as usual — e.g. with Slurm,
activate the venv in the job script and launch with `srun`:

```bash
source /path/to/pykmc_env/bin/activate
srun --ntasks=$SLURM_NTASKS python -m pykmc -in input.in
```
