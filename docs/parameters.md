# Inputs Parameters 
 
## Control
 
The following parameters are general parameters that control the KMC simulations and resources used.
 
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

## Minimization
 
**style** 

- **Default**: `lammps` 

- **Description**: No description available 

## AtomicEnvironment
 
**radd_cna** 

- **Default**: `0` 

- **Description**: No description available 

## EventSearch
 
**emin_event** 

- **Default**: `0.2` 

- **Description**: No description available 

**emax_event** 

- **Default**: `6` 

- **Description**: No description available 

**partn_dmax** 

- **Default**: `6.0` 

- **Description**: No description available 

**partn_verbose** 

- **Default**: `2` 

- **Description**: No description available 

**partn_ninit** 

- **Default**: `2` 

- **Description**: No description available 

**partn_forc_thr** 

- **Default**: `0.01` 

- **Description**: No description available 

**partn_push_mode** 

- **Default**: `rad` 

- **Description**: No description available 

**partn_push_dist_thr** 

- **Default**: `3.0` 

- **Description**: No description available 

**partn_push_step_size** 

- **Default**: `0.4` 

- **Description**: No description available 

**partn_eigen_step_size** 

- **Default**: `0.2` 

- **Description**: No description available 

**partn_lanczos_disp** 

- **Default**: `0.0005` 

- **Description**: No description available 

**partn_nsmooth** 

- **Default**: `3` 

- **Description**: No description available 

**partn_nperp** 

- **Default**: `5` 

- **Description**: No description available 

**k0** 

- **Default**: `1` 

- **Description**: No description available 

## PSR
 
**kmax_factor** 

- **Default**: `1.8` 

- **Description**: No description available 

