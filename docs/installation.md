# Installation

## System Prerequisites

### Ubuntu/Debian Systems

Before creating a Python virtual environment, you need to install the `python3-venv` package:

```bash
# For Python 3.12 (replace with your Python version if different)
sudo apt install python3.12-venv

# Or use the generic package which installs for the default Python3 version
sudo apt install python3-venv
```

This package is required for the `venv` module to work properly on Debian-based systems.

**Note**: Other Linux distributions (Fedora, Arch, etc.) and macOS typically include venv support by default.

## Python Environment 

It is recommended to use a dedicated Python environment for pyKMC to prevent package conflicts. Ensure you have Python >= 3.9 installed.

To create a new virtual environment using venv:
```bash
python3 -m venv /path_to_environment/pykmc_env
```
Then, activate the newly created environment:
```bash
source /path_to_environment/pykmc_env/bin/activate
```

**Optional - Shell Alias for Convenient Activation**: To activate the environment from anywhere by simply typing `pykmc_env`, add this alias to your `~/.bashrc` (or `~/.zshrc` for zsh users):
```bash
# pyKMC virtual environment activation shortcut
alias pykmc_env='source /path_to_environment/pykmc_env/bin/activate'
```
After adding the alias, run `source ~/.bashrc` or restart your terminal.

Finally, install pyKMC along with its dependencies:
```bash 
cd /path_to/pyKMC
pip install -e .
```


## Other Codes

Depending on the selected options for running a KMC simulation, pyKMC relies on additional software to handle different parts of the algorithm. Below are the installation steps for each required tool.

### LAMMPS
A recent version of [LAMMPS](https://docs.lammps.org/Manual.html) is recommended (tested with the 24 August 2024 version).

#### Prerequisites for LAMMPS Compilation

LAMMPS requires MPI (Message Passing Interface) development tools and build essentials to compile with parallel support:

**Ubuntu/Debian:**
```bash
sudo apt install build-essential gfortran libopenmpi-dev
```

**Alternative MPI implementation (MPICH):**
```bash
sudo apt install build-essential gfortran mpich libmpich-dev
```

**Verify the installation:**
```bash
mpicxx --version  # Should display the MPI C++ compiler version
```

#### Compiling LAMMPS

To install LAMMPS using the make method from the LAMMPS source directory:

```bash
make yes-basic
make yes-extra-compute  # Required for CNA computation
make yes-plugin         # Required for pARTn
make mode=shared mpi    # Required for pARTn (otherwise use `make mpi`)
make install-python     # Enables Python bindings
```
If LAMMPS is already installed, only the last command (`make install-python`) is necessary.

### pARTn
For event search, [pARTn](https://mammasmias.gitlab.io/artn-plugin/sections/Intro.html) can be used with LAMMPS.
Follow the installation instructions provided [here](https://mammasmias.gitlab.io/artn-plugin/sections/Installation.html):

- Run the configuration script:
```bash
cd /path/to/artn-plugin
./configure --with-lammps LAMMPS_PATH=/path/to/lammps
```

**Example with actual paths:**
```bash
cd /home/kerr/pykmc/artn-plugin
./configure --with-lammps LAMMPS_PATH=/home/kerr/pykmc/lammps
```
**Important:** Note the `LAMMPS_PATH=` prefix before the path - it's required syntax, not a placeholder.

- Compile the plugin:
```bash
make lmplib
```

Add the interface path to the PYTHONPATH environment variable:
```bash
export PYTHONPATH=/your/path/to/artn-plugin/interface:$PYTHONPATH

# Example:
export PYTHONPATH=/home/kerr/pykmc/artn-plugin/interface:$PYTHONPATH
```

### IRA
For point set registration (used during event reconstruction), [IRA](https://mammasmias.github.io/IterativeRotationsAssignments/) can be used.

Follow the installation instructions provided [here](https://mammasmias.github.io/IterativeRotationsAssignments/#compilation):

#### Prerequisites for IRA Compilation

IRA requires LAPACK and a Fortran compiler:

**Ubuntu/Debian:**
```bash
sudo apt install gfortran liblapack-dev
```

#### Compiling IRA

- Compile the shared library:
```bash
cd /path/to/ira/src/
make shlib
```

**Example with actual paths:**
```bash
cd /home/kerr/pykmc/IterativeRotationsAssignments/src/
make shlib
```

**Note:** Use `make shlib` (not `make all`) to build the shared library (`libira.so`) that pyKMC requires via Python's `ctypes`.

- Add the interface path to PYTHONPATH:
```bash
export PYTHONPATH=$PYTHONPATH:/your/path/to/IRA/interface

# Example:
export PYTHONPATH=$PYTHONPATH:/home/kerr/pykmc/IterativeRotationsAssignments/interface
```

**Tip:** To make these PYTHONPATH exports permanent, add them to your `~/.bashrc`:
```bash
# Add to ~/.bashrc for permanent setup
export PYTHONPATH=/home/kerr/pykmc/artn-plugin/interface:$PYTHONPATH
export PYTHONPATH=/home/kerr/pykmc/IterativeRotationsAssignments/interface:$PYTHONPATH
```