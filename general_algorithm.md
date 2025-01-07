# Overview of the algorithm : 

```system.kmc(...)``` run KMC simulation 

- Append current system Atoms in ```system.kmc_traj```
- ```for step in range(nkmc_step)```: 

    - ```if step == 0``` minimization of the system using lammps 

        open lammps, minimize, get new positions, update ```system.positions```
    - ```system.find_environment('cna/graph', ...)``` 

        find atomic environment ID and update ```system.environment```
        
        open lammps, compute cna, for atom that does not have a cristalline environment, compute graph and graph certificate
    -  
