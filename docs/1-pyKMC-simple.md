# Basic KMC simulation 

This tutorial provides a first overview of pyKMC's features and explains how to run a simple simulation. All units used are lammps metal units.

We will use an FCC Ni system with a vacancy (from the `pyKMC/examples/Ni_fcc_2047at_movacancy` example directory) and basic input parameters to see how the algorithm works. 


To run a KMC simulation, an input file in a format compatible with [configparser](https://docs.python.org/3/library/configparser.html) is required.  

The file is structured into different sections, each controlling a specific aspect of the KMC simulation:  

- `[Control]` section  
Defines general KMC parameters and computational resources.  

- `[Potential]` section  
Specifies the potential used by the E/F engine (e.g., LAMMPS).  

- `[Minimization]` section  
Contains parameters related to the system minimization performed at each KMC step.  

- `[AtomicEnvironment]` section  
In pyKMC, events (reactions) are associated with the atomic environment topology of an atom. Atoms sharing the same topology can undergo the same events listed under that topology ID in the event catalog. This section defines parameters for the method used to determine atomic topologies.  

- `[EventSearch]` section  
Configures the method used to identify transition pathways.  

- `[PSR]` section  
When reusing events from previous steps, pyKMC applies a point set registration (shape matching) algorithm to compare the atomic environment of an atom at the current step with previously generated events. This section defines parameters for the shape matching method.  

## Input file 

We will see each basic paramters for each section in the input file. For an overview of all paramters see the [parameters](parameters.md) section.

### `[Control]`  

To start a KMC simulation, you need to provide an initial configuration file describing the system. This file must be compatible with [ASE](https://wiki.fysik.dtu.dk/ase/ase/io/io.html) and include atomic positions, the simulation cell, and atom types. The path to this file is specified using the `config_file` key.  

Additionally, you must define the number of KMC steps to be executed using the `nkmc_steps` key.  

For example, if your initial configuration file is named `initial_config.xyz` and you want to run 100 KMC steps, the `[Control]` section of the input file should be:  

```ini
[Control]
config_file = ./initial_config.xyz  
nkmc_steps = 100
```

### `[Potential]`  

In this simulation, we will use LAMMPS as the E/F engine along with an EAM potential. This requires specifying the LAMMPS commands [pair_style](https://docs.lammps.org/pair_style.html) and [pair_coeff](https://docs.lammps.org/pair_coeff.html).  

Since EAM potentials are defined using an external file, it is also possible to include a `file` key to improve readability.  

For example, when using the EAM potential [Ni_v6_2.0_LKBeland2016.eam](https://www.ctcms.nist.gov/potentials/entry/2016--Stoller-R-E-Tamm-A-Beland-L-K-et-al--Ni/), the `[Potential]` section of the input file would be:  

```ini
[Potential]
file = ./Ni_v6_2.0_LKBeland2016.eam  
pair_style = eam/alloy  
pair_coeff = * * ${file} Ni 
```

### `[Minimization]`  

To perform energy minimization, we need to specify that LAMMPS will be used by setting the `style` key.  

LAMMPS provides different [minimization styles](https://docs.lammps.org/min_style.html) and allows customization of the parameters used during the [minimization process](https://docs.lammps.org/minimize.html).  

In this example, we use the default parameters meaning that this section could technically be left blank.  

```ini
[Minimization]
style = lammps  
min_style = cg  
etol = 1.0e-6  
ftol = 1.0e-8  
maxiter = 1000  
maxeval = 1000  
```

### `[AtomicEnvironment]`  

To assign an ID to the atomic environment of an atom, three methods are available, specified using the `style` key:  

- **`cna`**  
  A **Common Neighbor Analysis (CNA)** is performed to distinguish atoms with a crystalline environment (ID = `crystal`) from those without (ID = `noncrystal`). This option is particularly useful when searching for events at each step of the KMC simulation (see the `[EventSearch]` section).  

- **`graph`**  
  A connectivity graph is built for each atom by considering neighbors within a cutoff radius `rcut`. Two atoms are considered "connected" if their distance is less than `rnei`. The library [pyNauty](https://github.com/pdobsan/pynauty) is then used to assign an ID based on the canonical form of the graph.  

- **`cna/graph`**  
  This method combines both approaches. First, CNA is applied to classify atoms as `crystal` or `noncrystal`. Then, for each `noncrystal` atom, a connectivity graph is constructed, and an ID is assigned.  

  In pyKMC, only atoms with a `noncrystal` ID can undergo events. However, since we attribute an event ID based on the atom that moves the most, it may involves an atom that is a neighbor of a `noncrystal` atom. To account for this, it is possible to compute the connectivity graph for atoms within a distance `radd_cna` from any `noncrystal` atom. 


We will use the `cna/graph` method : 
```ini 
[AtomicEnvironment] 
style = cna/graph 
rcut = 7.0 
rnei = 3.01 
radd_cna = 3.0 
``` 

### `[EventSearch]`  

To find transition paths and generate events, we use the **ARTn LAMMPS plugin**, pARTn (`style = pARTn`).

Next, we define the number of event searches performed for each atomic environment ID using the `nsearch` key.  

When an event is found, it is added to the catalog. Only the atomic positions within a cutoff radius `rcutenv` from the atom that moves the most are stored.  

To run a KMC simulation, we need to compute the reaction rate, $k$, associated with the energy barrier $dE$ of the event. The reaction rate is given by:  

$$
k = k_{0} \frac{k_{b}T}{h} e^{-\frac{dE}{k_{b}T}}
$$  

where:  
- $k_0$ is a prefactor set by the `k0` key,  
- $T$ is the temperature defined by the `T` key,  
- $k_b$ is the Boltzmann constant,  
- $h$ is Planck’s constant.  

For the pARTn parameters, we will use the default values.  

This gives us : 
```ini 
[EventSearch]
style = pARTn 
nsearch = 10 
rcutenv = 7.0 
k0 = 1.0e-12
T = 300.0 
```  

### `[PSR]` 

Finally, we use [IRA](https://github.com/mammasmias/IterativeRotationsAssignments/tree/master) for the shape matching. Using default parameters value, we just need to gives the `style` key : 

```ini 
[PSR]
style = ira
```

## Running the simulation 

First, we copy to the directory the initial configuration file and the potential file to the newly created directory : 


```bash
%%bash 
cp ../Ni_fcc_2047at_monovacancy/initial_config.xyz ../Ni_fcc_2047at_monovacancy/Ni_v6_2.0_LKBeland2016.eam .
```

 
Then, we need to put all together the parameters in the input file, you can create it by yourself or run :


```bash
%%bash 
cat <<EOF > ./input.in
[Control]
config_file = ./initial_config.xyz  
nkmc_steps = 100

[Potential]
pair_style = eam/alloy  
pair_coeff = * * ./Ni_v6_2.0_LKBeland2016.eam Ni

[Minimization]
style = lammps  
min_style = cg  
etol = 1.0e-6  
ftol = 1.0e-8  
maxiter = 1000  
maxeval = 1000  

[AtomicEnvironment] 
style = cna/graph 
rcut = 7.0 
rnei = 3.01 
radd_cna = 3.0 

[EventSearch]
style = pARTn 
nsearch = 10 
rcutenv = 7.0 
k0 = 1.0e-12
T = 300.0 

[PSR]
style = ira

EOF
```

Finally, we launch the simulation.

Supposing that you are working in your python environment and that you add to the `$PYTHONPATH` the paths to `pARTn` and `IRA` library, as explained in the [installation](../../docs/installation.md), you just need to execute : 


```python
from pykmc.system import System 
system = System('./input.in')
system.kmc()
```

The simulation should take a few minutes to complete.  

During the run, several log files are generated, along with a trajectory file containing the atomic positions at each step. By default, this file is named **`trajkmc.xyz`**.  

The **`pykmc.log`** file provides information about the system's evolution in the following format:  

```bash
Step       Time         Ndiff_env  N_event    n_selected_event dE_event    dh           Reconstruction dE   Reconstruction Topo
0          7.435022e+19 5          1          0               1.188492e+00 6.027361e-03 1                  1
1          7.449341e+19 5          1          0               1.188492e+00 4.027686e-03 1                  1
``` 
- Step : Current step of the simulation.
- Time : Simulation time (in ps).
- Ndiff_env : Number of distinct atomic environments at the current step.
- N_event : Number of events in the catalog.
- n_selected_event : Index of the selected event in the catalog.
- dE_event : Energy barrier of the selected event (in eV).
- dh : Hausdorff distance used for shape matching.
- Reconstruction dE : Indicates whether the energy barrier at the saddle point matches the one stored in the catalog after applying the event.
- Reconstruction Topo : Indicates whether the topology of the most mobile atom matches the one in the catalog after applying the event.

It is then possible to open the `trajkmc.xyz` file in your favorite visualization tool (e.g. Ovito, VMD, ASE)

If you don’t want to manually activate your Python environment and set the `$PYTHONPATH` for external libraries every time, or if you are working on a cluster, you can simply create a Python script, **`run.py`**:  

```python
from pykmc.system import System  
system = System('input.in')  
system.kmc()  
``` 

You can then create a Bash (or SLURM sbatch) script to handle the environment setup and execution :

```bash 
#!/bin/bash  

# Activate the Python environment  
source /path_to_your_python_environment/bin/activate  

# Export necessary paths  
export PYTHONPATH=/your/path/to/artn-plugin/interface:$PYTHONPATH  
export PYTHONPATH=$PYTHONPATH:/your/path/to/IRA/interface  

# Run the simulation  
python run.py   
```

Finally, run the Bash script to launch the simulation.


