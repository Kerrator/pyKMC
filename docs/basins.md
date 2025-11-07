# Basins 

The basins parameters are defined with a `[Basin]` section in the input file. What needs to be specified is the energy threshold where we consider to be in the basin. 
You also need to activate the basin mode in the `[Control]` section. 
Those sections will look like : 
```INI 
[Control]
...
basin = True 
... 

.
.
.

[Basin] 
energy_thr = 0.1

.
.
.
``` 
_Note: currently only one way to deal with basin is implemented, the use of a basin's section in the input file is for future implementation, if multiple algorithms will be available (e.g. FTPA, MRT, ..) or local basins._

**Note : In the following part, we explain how the current global basin is implemented, that use generic events to explore the basin**

## Workflow : 

During a KMC step, when an event, having a forward and backward energy barrier lower than the `energy_thr` is selected, we construct a Basin object, that will explore the basin, find the exit time and the exit state. 
Then the selected event is replaced in the KMC loop. 

FIGURE 

When exploring the basin, the goal is to first construct a connectivity table. This is a pandas DataFrame with all needed information to apply the selected algorithm. 
We use a Connectivity object, that store the dataframe, and has utilities method (merge, remove, find a connection between to states, ...). 
The dataframe looks like : 

TABLE 

It reads as follow : 
- `state`: a transient state 
- `state_connexion`: a state connected to the `state`
- `event_connexion`: the number of the event in the events table to apply to go from `state` to `state_connexion`
- `central_atom` : the atom index on which the `event_connexion` should be applied 
- `sym` : the number of the symmetry of the `event_connexion` 
- `transient` : a bool informing if the `state_connexion` is a transient state or not 
- `dE_forward` : the energy barrier to go from `state` to `state_connexion`, taken from the events table
- `k_forward` : the rate constant to go from `state` to `state_connexion`, taken from the events table
- `dE_backward` : the energy barrier to go from `state_connexion` to `state`, taken from the events table
- `k_backward` : the rate constant to go from `state_connexion` to `state`, taken from the events table

_Note: The choice of a pandas Dataframe where made to have fast query/sorting methods. It is also easy to generate a graph from this table for analysis/caracterization_ 

The Basin object also use two objects, an Explorer object, that explore one state and construct a connectivity table for that staten and a Selector object, that takes the connectivity table to find the exit time and exit state. 

_Note: This structure is made for future exploration, selection implementation._ 

