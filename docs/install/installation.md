# Installation 

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
Finally, install pyKMC along with its dependencies:
```bash 
cd /path_to/pyKMC
pip install -e .
```


## Other Codes

Depending on the selected options for running a KMC simulation, pyKMC relies on additional software to handle different parts of the algorithm. Below are the installation steps for each required tool.

### LAMMPS
A recent version of [LAMMPS](https://docs.lammps.org/Manual.html) is recommended (tested with the 24 August 2024 version). To install it using the make method from the LAMMPS source directory:

```bash
make yes-basic 
make yes-extra-compute  # Required for CNA computation  
make yes-plugin         # Required for pARTn  
make mode=shared mpi    # Required for pARTn (otherwise use `make mpi`)  
make install-python     # Enables Python bindings  
```
If LAMMPS is already installed, only the last command (`make install-python`) is necessary.

### pARTn
For event search, [pARTn](https://mammasmias.gitlab.io/artn-plugin/) can be used with LAMMPS.
Follow the installation instructions provided [here](https://mammasmias.gitlab.io/artn-plugin/user_guide/Installation.html):

- Run the following commands, this will install the python module `pypARTn` into your python env:
```bash
cd /path/to/artn-plugin
cmake -B build -DWITH_LAMMPS=ON -DLAMMPS_PATH=/path/to/lammps/build -DARTN_INSTALL_PYTHON=ON
cmake --build build
cmake --install build
```

### IRA
For point set registration (used during event reconstruction), [IRA](https://mammasmias.github.io/IterativeRotationsAssignments/) can be used.

Follow the installation instructions provided [here](https://mammasmias.github.io/IterativeRotationsAssignments/#compilation):

- Compile the source and create the python module `ira_mod`:
```bash 
cd /path/to/ira
python -m pip install .
```
