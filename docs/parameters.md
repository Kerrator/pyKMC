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

