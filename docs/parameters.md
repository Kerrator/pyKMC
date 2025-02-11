# Inputs Parameters 
 
## Control
 
The following parameters are general parameters that control the KMC simulations and resources used.
 
**nkmc_steps** 

- **Default**: ` **MANDATORY** ` 

- **Description**: number of KMC steps 

**config_file** 

- **Default**: `None` 

- **Description**: Path to the initial configuration file 

**output_file** 

- **Default**: `trajkmc.xyz` 

- **Description**: Path to the file where the trajectory is written, format must be recognized by ase.io.write() 

**catalog** 

- **Default**: `None` 

- **Description**: Path to a catalog to reuse from a previous simulation 

**dimension** 

- **Default**: `3` 

- **Description**: Dimension of the system 

**nprocs** 

- **Default**: `1` 

- **Description**: number of MPI process to use 

**backend** 

- **Default**: `local` 

- **Description**: if running the simulation locally (`'local'`), or on a cluster (`'slurm_allocation'`) 

**reconstruction** 

- **Default**: `True` 

- **Description**: if a new catalog is generated at each step or reused 

## Potential
 
The potential section deals with the description of the potential in the format of the E/F engine used, for lammps :
 
**file** 

- **Default**: `None` 

- **Description**: path to the potential file if needed 

**pair_style** 

- **Default**: ` **MANDATORY** ` 

- **Description**: lammps pair style command 

**pair_coeff** 

- **Default**: ` **MANDATORY** ` 

- **Description**: lammps pair coeff command 

## Minimization
 
Parameters used for the minimization of the system
 
**style** 

- **Default**: `lammps` 

- **Description**: E/F engine used 

**min_style** 

- **Default**: `cg` 

- **Description**: lammps minimization style 

**etol** 

- **Default**: `1e-06` 

- **Description**: lammps stopping tolerance for energy 

**ftol** 

- **Default**: `1e-08` 

- **Description**: lammps stopping tolerance for force 

**maxiter** 

- **Default**: `100` 

- **Description**: lammps max iterations of minimizer 

**maxeval** 

- **Default**: `1000` 

- **Description**: lammps max number of force/energy evaluations 

## AtomicEnvironment
 
Parameters used to attribute an atomic environment to each atoms
 
**style** 

- **Default**: ` **MANDATORY** ` 

- **Description**: style used, `cna`, `graph` or `cna/graph` 

**rnei** 

- **Default**: ` **MANDATORY** ` 

- **Description**: maximal distance to consider that two atoms are connected when `graph` and `cna/graph` is used 

**rcut** 

- **Default**: ` **MANDATORY** ` 

- **Description**: radial cuttoff defining the environment around an atom, used when `graph` and `cna/graph` 

**radd_cna** 

- **Default**: `0` 

- **Description**: when `cna/graph` is used, graph for atoms at a distance inferior to radd_cna to a atom having a non crystalline environment is also computed 

## EventSearch
 
Parameters related to the search of transition paths
 
**style** 

- **Default**: ` **MANDATORY** ` 

- **Description**: which method used, `'pARTn'` for ARTn method 

**rcutenv** 

- **Default**: ` **MANDATORY** ` 

- **Description**: radius of the sphere defining the positions of atoms, around the central atom, that are saved in the catalog 

**nsearch** 

- **Default**: ` **MANDATORY** ` 

- **Description**: number of event search for each different atomic environments 

**path_artnso** 

- **Default**: ` **MANDATORY** ` 

- **Description**: path to the partn library file 

**emax_event** 

- **Default**: `6` 

- **Description**: event found having a higher barrier energy value are not saved to the catalog 

**emin_event** 

- **Default**: `0.2` 

- **Description**: event found having a lower barrier energy value are not saved to the catalog 

**k0** 

- **Default**: `1e-12` 

- **Description**: value of k0 when computing the rate constant with $k0 rac{k_{b}T}{h}e^{-rac{dE}{k_{b}T}} 

**T** 

- **Default**: ` **MANDATORY** ` 

- **Description**: Temperature of the simulation 

## PSR
 
Parameters controlling point set registration (shape matching)
 
**style** 

- **Default**: ` **MANDATORY** ` 

- **Description**: which method is used, `'ira'` for IterativeRotationsAssignments 

**kmax_factor** 

- **Default**: `1.8` 

- **Description**: factor for multiplication of search radius 

