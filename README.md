# pyKMC

Python framework for adaptive Kinetic Monte Carlo (aKMC) simulations.  
Full documentation: [hugomoison.github.io/pyKMC](https://hugomoison.github.io/pyKMC) — use the version switcher to pick `latest` (main branch) or `develop`.

## Requirements

pyKMC currently requires LAMMPS, pARTn and IRA (no alternative implementations exist yet).

- Python ≥ 3.10
- [LAMMPS](https://github.com/lammps/lammps) : compiled with `PKG_BASIC`, `PKG_EXTRA_COMPUTE`, `PKG_PLUGIN`
- [pARTn](https://gitlab.com/mammasmias/artn-plugin) : LAMMPS plugin for saddle point searches
- [IRA](https://github.com/mammasmias/IterativeRotationsAssignments) : shape matching library

> [!NOTE]
> Detailed install instructions and the one-shot install scripts (`install_pykmc_mac.sh`, `install_pykmc_linux.sh`) are available in `docs/user_guide/install/`.

## Installation

### Python environment

It is recommended to use a dedicated Python environment to prevent package conflicts. Ensure you have Python ≥ 3.10 installed.

```bash
python3 -m venv /path/to/pykmc_env
source /path/to/pykmc_env/bin/activate
cd /path/to/pyKMC
pip install -e .
```

### LAMMPS

A recent version of LAMMPS is recommended (tested with the 24 August 2024 version). Use the cmake method, the traditional make way does not support the plugin package anymore.

You should at least enable these options:

```bash
-D BUILD_SHARED_LIBS=on
-D LAMMPS_EXCEPTIONS=on
-D PKG_BASIC=yes
-D PKG_EXTRA_COMPUTE=yes
-D PKG_PLUGIN=yes
-D Python_EXECUTABLE="$(which python)"
```

More details available in the [LAMMPS cmake guide](https://docs.lammps.org/Build_cmake.html).

### pARTn

The following commands install the python module `pypARTn` into your environment:

```bash
cd path/to/artn-plugin
cmake -B build -DWITH_LAMMPS=ON -DLAMMPS_PATH=path/to/lammps/build -DARTN_INSTALL_PYTHON=ON
cmake --build build && cmake --install build
```

More details available in the [pARTn documentation](https://mammasmias.gitlab.io/artn-plugin/user_guide/Installation.html).

### IRA

```bash
cd /path/to/ira
pip install .
```

More details available in the [IRA documentation](https://mammasmias.github.io/IterativeRotationsAssignments/#compilation).

## Usage

Running pyKMC requires MPI and an input file. See the [documentation](https://hugomoison.github.io/pyKMC) and `/examples` for input file format and examples.

```bash
mpirun -n 8 python -m pykmc -in input.in
```