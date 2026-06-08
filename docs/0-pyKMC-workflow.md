# 0-pyKMC Workflow 

This tutorial shows how pyKMC works. 

The code is build on a class, call `System` which is a extension of the ASE `Atoms` object. It is then composed of a classes so that each class performs one step of the kmc simulation and a class `KMC`. 
The class `System` can call each subclasses. 

<img src="pykmcarchi.png" width=400 />


We gonna use a simple example, a Ni fcc box with a monovacancy, to see how each subclass works. 

This could be usefull if you want to implement/modify fonctionnalities or if you want to do some tests (e.g. test the event search parameters) before running a KMC simulation.


## System : 

pyKMC uses a configuration file to initialize the simulation. It could be of any format readable by `ase.io.read` with positions, cell parameters and atoms names. 

We will use `ASE`to generate our simulation box : a Ni fcc cubic box with 2047 atoms and one vacancy.


```python
from ase.build import bulk 
from ase.io import write 

#Parameters : 
output = './initial_config.xyz'
alat = 3.52 

atoms = bulk('Ni', crystalstructure='fcc', a=alat, cubic=True)
atoms = atoms.repeat((8,8,8))
atoms.pop(1166)
write(output, atoms)
```

We can vizualize our system : 


```python
from ase.io import read 
from ase.visualize import view 

#Parameters : 
file_path = './initial_config.xyz'
atoms = read(file_path)


print("Number of atoms = {}".format(atoms.get_global_number_of_atoms()))
print("Cell is : \n {}".format(atoms.get_cell()))
view(atoms, viewer='ngl')

```

    Number of atoms = 2047
    Cell is : 
     Cell([28.16, 28.16, 28.16])



    





    HBox(children=(NGLWidget(), VBox(children=(Dropdown(description='Show', options=('All', 'Ni'), value='All'), D…



## Parts of the KMC algorithm 

pyKMC uses the ```System```class on which each part of the algorithm is build. 
We first need to initialize the system and then we can apply different procédure to it, e.g : 
```python 
system = System(config_file = './initial_config.xyz')
system.minimize(...) 
...
system.kmc(...) 
```



```python
from pykmc.system import System
system = System(config_file = 'initial_config.xyz')
```


It is possible, and preferable, to use a configuration file, to set the parameters of a KMC simulation. But here, we gonna use dictionnaries, one for each part of the KMC algorithm. 


In this example, we will use lammps for all energies and forces calculation, thus we need to define a dictionnary for our potential composed of Lammps commands.



```python
pot_file_path = '../Ni_fcc_499at_monovacancy/Ni_v6_2.0_LKBeland2016.eam'
potential = {'pair_style' : 'eam/alloy', 
             'pair_coeff' : '* * {} Ni'.format(pot_file_path)}
```

### Minimization : 

At the start of a KMC simulation, and after each KMC step, pyKMC perform a minimization. 

The dictionnary for the minimization is composed of all parameters that Lammps use for the minimization.


```python
minimization_parameters = {'min_style' : 'cg', 
                           'etol' : 1.0e-6, 
                           'ftol' : 1.0e-8, 
                           'maxiter' : 100, 
                           'maxeval' : 1000}
```

And then we minimize our system


```python
system.minimize('lammps', minimization_parameters, potential)
```

### Atomic Environment : 

Before searching events, we need to attribute to each atom an atomic environment ID. 
Atoms with a same ID share have a same environment. 

Those IDs will be used during the event search and the reconstruction of event if this option is set to True.

Three options are available : 
- 'cna' : Compute the Common Neigbhor Analysis with Lammps to find atoms that have a cristalline environment (ID = 'crystal') and those who have not (ID = 'noncrystal'). 
- 'graph' : for each atoms, compute a connectivity graph and attribute an ID based on the canonical form of the graph using pynauty. 
- 'cna/graph' : use 'cna' and compute the graph for atoms that do not have a cristalline environment. 

We will use the 'cna/graph' option.

The parameters that we need to set are : 
- rcut : radial cutoff defining the environement of an atom (used by 'graph')
- rnei : if the distance between two atoms is less than rnei, those atoms are connected 
- radd_cna : to compute the graph of atoms at a distance < radd_cna from non cristalline atoms (only with 'cna/graph')  


```python
atomic_environment_parameters = {'rcut' : 6.0, 
                                 'rnei' : 3.0 ,
                                 'radd_cna' : 0}
```

To find atomic environment : 


```python
system.find_environment('cna/graph', atomic_environment_parameters)
```

it will update the ```system.environment```attribute, the result is a list for dictionnaries with the ID of the environment and the list of atoms having that ID. 

We can check the result by savint it to a yaml file :


```python
import yaml 
with open('atomic_environment.yml', 'w') as outfile : 
    yaml.dump_all(system.environment, outfile, default_flow_style=False, sort_keys=False)
```

And visualize it with ASE : 


```python
from ase import Atoms
from ase.visualize import view

#Create new Atoms : 
atoms = Atoms(positions = system.get_positions(), cell=system.get_cell(), pbc=True)

#Read atomic_environment file : 
with open('atomic_environment.yml', 'r') as file : 
    data = yaml.safe_load_all(file)
    l_id = list(data)

#Set element for each ID : 
z = [1]*atoms.get_global_number_of_atoms() 
for i, id in enumerate(l_id) : 
    at_idx = [int(e) for e in id["atom index"]]
    if id['ID'] != 'crystal' : 
        for at in at_idx : 
            z[at] = 30+i
atoms.set_atomic_numbers(z)

#view 
view(atoms, viewer='ngl')
```




    HBox(children=(NGLWidget(), VBox(children=(Dropdown(description='Show', options=('All', 'Ga', 'H'), value='All…



### Event Search 

We will use pARTn to find event and use the option 'reconstruction' = True. 
It will, for each atoms that have not a cristalline environement, perform 'nsearch' event searches. 

The dictionnary is composed of pARTn parameters, parameters needed to compute the constant rate, and rcutenv which is a radial cutoff defining the region around the atom on which the event search is perfrom that we will be save.


```python
search_params = {'nsearch' : 10,
                 'path_artnso' : '/root/programs/artn-plugin/lib/libartn-lmp.so', 
                 'rcutenv' : 7.0, 
                 'emax_event' : 5.0, 
                 'emin_event' : 0.2,
                 'partn_dmax' : 6.0, 
                 'partn_verbose' : 0, 
                 'partn_ninit' : 2, 
                 'partn_forc_thr' : 0.01,
                 'partn_push_mode' : 'rad', 
                 'partn_push_dist_thr' : 3.0, 
                 'partn_push_step_size' : 0.4, 
                 'partn_eigen_step_size' : 0.2, 
                 'partn_lanczos_disp' : 0.0005,
                 'partn_nsmooth' : 3, 
                 'partn_nperp' : 5,
                 'k0' : 1, 
                 'T' : 300.0, 
                 }
```


```python
system.event_search('pARTn', search_params, atomic_environment_parameters, potential, reconstruction=True)
```

    ============================================================
    ::>> Output from err_write() subroutine:
    ::>> ERROR in pARTn, ierr value: -99
    ::>> Message  : STOPPING DUE TO ERROR:EIGENVALUE LOST, try to increase nnewchance or nsmooth
    ::>> Source   : block_finalize.f90 line: 48
    ::>> Last caller : artn.f90 line: 603
    ============================================================
    ============================================================
    ::>> Output from err_write() subroutine:
    ::>> ERROR in pARTn, ierr value: -99
    ::>> Message  : STOPPING DUE TO ERROR:EIGENVALUE LOST, try to increase nnewchance or nsmooth
    ::>> Source   : block_finalize.f90 line: 48
    ::>> Last caller : artn.f90 line: 603
    ============================================================
    ============================================================
    ::>> Output from err_write() subroutine:
    ::>> ERROR in pARTn, ierr value: -99
    ::>> Message  : STOPPING DUE TO ERROR:EIGENVALUE LOST, try to increase nnewchance or nsmooth
    ::>> Source   : block_finalize.f90 line: 48
    ::>> Last caller : artn.f90 line: 603
    ============================================================
    ============================================================
    ::>> Output from err_write() subroutine:
    ::>> ERROR in pARTn, ierr value: -99
    ::>> Message  : STOPPING DUE TO ERROR:EIGENVALUE LOST, try to increase nnewchance or nsmooth
    ::>> Source   : block_finalize.f90 line: 48
    ::>> Last caller : artn.f90 line: 603
    ============================================================
    ============================================================
    ::>> Output from err_write() subroutine:
    ::>> ERROR in pARTn, ierr value: -99
    ::>> Message  : STOPPING DUE TO ERROR:EIGENVALUE LOST, try to increase nnewchance or nsmooth
    ::>> Source   : block_finalize.f90 line: 48
    ::>> Last caller : artn.f90 line: 603
    ============================================================


To see what we obtain, we can save the catalog (also usefull if we want to restart a simulation).


```python
system.catalog.to_pickle('catalog.pickle')
```

The ```system.catalog```is a pandas DataFrame, we can print some informations


```python
import pandas as pd 

catalog = system.catalog[['event_id', 'energy_barrier', 'k']]
catalog
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>event_id</th>
      <th>energy_barrier</th>
      <th>k</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>b'\x02\x00\x00\x03\x00\x03\x00 \x00\x00\x00\x0...</td>
      <td>1.190599</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>b'\x02\x00\x00\x03\x00\x03\x00 \x00\x00\x00\x0...</td>
      <td>1.191314</td>
      <td>0.0</td>
    </tr>
  </tbody>
</table>
</div>



and we can see the event : 


```python
idx_cat = 0
event_traj = [] 
c = ['initial_positions', 'saddle_positions', 'final_positions']
central_atom = system.catalog.loc[idx_cat].at['move_atom_idx'] 
for e in c : 
    atoms = Atoms(positions=system.catalog.loc[idx_cat].at[e])
    z = atoms.get_global_number_of_atoms()*[30]
    z[central_atom] = 35
    atoms.set_atomic_numbers(z)
    event_traj.append(atoms)
view(event_traj, viewer='ngl')
```




    HBox(children=(NGLWidget(max_frame=2), VBox(children=(Dropdown(description='Show', options=('All', 'Zn', 'Br')…



### Point Set Registration : 

When pyKMC uses generic event and try to reconstruct it, it uses a shape matching (point set registration) algorithm. 

Here we use IRA. 




```python
psr_parameters = {'kmax_factor' : 2.0}
```

To reconstruct the event 0 of the catalog on an atom in the system having the same atomic environment ID than the event id : 


```python
#Catalog infos
idx_cat = 0
event_id = system.catalog.loc[idx_cat].at['event_id']
#find atoms in the system that have the event_id
l_atoms = [dic['atom index'] for dic in system.environment if dic['ID'] == event_id ][0]
import random
at_idx = random.choice(l_atoms)

system.point_set_registration('ira', psr_parameters, idx_cat, at_idx, 7.0, save=True)
```




    (array([[ 9.99999999e-01, -2.81417734e-05,  1.96675688e-05],
            [ 2.81413948e-05,  9.99999999e-01,  1.92467965e-05],
            [ 1.96681104e-05,  1.92462430e-05, -1.00000000e+00]]),
     array([ 3.50327971,  1.74205704, 26.39949026]),
     array([  1,   0,   5,   3,   6,   2,   4,  10,   8,  11,   7,   9,  20,
             21,  23,  19,  16,  17,  22,  15,  12,  13,  18,  14,  35,  32,
             33,  36,  31,  29,  34,  28,  25,  26,  30,  24,  27,  49,  46,
             47,  50,  45,  42,  43,  48,  41,  38,  39,  44,  37,  40,  55,
             54,  53,  52,  51,  64,  65,  67,  63,  60,  61,  66,  59,  56,
             57,  62,  58,  80,  77,  78,  81,  76,  73,  74,  79,  72,  69,
             70,  75,  68,  71,  94,  91,  92,  95,  90,  87,  88,  93,  86,
             83,  84,  89,  82,  85, 100,  99,  98,  97,  96, 106, 105, 103,
            107, 102, 101, 104, 119, 116, 117, 115, 112, 113, 118, 111, 109,
            110, 114, 108, 131, 128, 129, 127, 124, 125, 130, 123, 121, 122,
            126, 120, 133, 132]),
     0.00510513880942386)



And finaly we can see the reconstruction 


```python
import numpy as np
psr = pd.read_pickle('psr_event_0.pickle')

#Read event : 
coords2 = system.catalog.loc[idx_cat].at['initial_positions']
nat2 = len(coords2)
typ2 = nat2*['Ni']
#Read PSR : 
rmat = psr.loc[0].at['R']
tr = psr.loc[0].at['T']
perm = psr.loc[0].at['P']

#Environment of atom at at_idx
ind = np.linspace(0, system.get_global_number_of_atoms()-1, system.get_global_number_of_atoms()).astype(int)
dist = system.get_distances(at_idx, ind, mic=True)
neighbor_list = np.where(dist<7.0)[0]

coords1 = system.get_positions()[neighbor_list]
nat1 = len(coords1)
typ1 = nat1*['Cu']

pos = np.concatenate((coords1, coords2), axis=0)
at_name = typ1+typ2

traj = [Atoms(at_name, positions=pos)]

#change coords2 : 
for i in range(nat2) :
    coords2[i] = np.matmul(rmat, coords2[i])+tr
coords2[:] = coords2[perm]

pos = np.concatenate((coords1, coords2), axis=0)
at_name = typ1+typ2

traj.append(Atoms(at_name, positions=pos))

view(traj, viewer='ngl')

```




    HBox(children=(NGLWidget(max_frame=1), VBox(children=(Dropdown(description='Show', options=('All', 'Ni', 'Cu')…


