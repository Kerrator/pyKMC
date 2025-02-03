# Overview of the algorithm : 

**Current problems** 
- Premier step toujours ok, mais ensuite : 
    
    - soit problem dh > threshold 
    - soit dE reconstruction très élevée


```system.kmc(...)``` run KMC simulation 

- Append current system Atoms in ```system.kmc_traj```
- ```for step in range(nkmc_step)```: 

    - ```if step == 0``` minimization of the system using lammps 

        open lammps, minimize, get new positions, update ```system.positions```
    - ```system.find_environment('cna/graph', ...)``` 

        find atomic environment ID and update ```system.environment```
        
        open lammps, compute cna, for atoms that does not have a crystalline environment, compute graph and graph certificate
    - ```system.event_search(...)```

        use pARTn to perform 'nseach' event search for each environment ID that are not in the catalog 

        add backward reaction 

        update catalog  

    - ```ìf len(catalog) > 1``` meaning there is at least one event in the catalog : 
        
        - select an event in the catalog (for the moment random)
        - select an atom on which we gonna perform the event (they have same ID)
        - ```system.point_set_registration(...)``` do a point set registration to find transfomation matrix between the event positions and the selected atom
        - ```update_positions```of the system based on the point set registration results
            
            - check dh distance 
            - check dE consistency 

        - ```system.minimize(...)```
        - append new configuration to ```system.kmc_traj```


# NOTES : 
- Use lammps metal units
- When use lammps, atom types in alphabetic order
