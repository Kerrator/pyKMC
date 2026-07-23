# Basins

Enable basin mode with `basin = True` in the `[Control]` section; this makes a
`[Basin]` section mandatory. The key parameter is `energy_thr`, the energy
threshold (in eV) below which a state is considered part of the basin. It
defaults to `0.0` eV, which effectively disables basin detection for
nonnegative barriers, so set a positive threshold to use the feature.

Example:

```INI
[Control]
...
basin = True
...

[Basin]
energy_thr = 0.1
```

The optional `style` key controls how the geometry of a newly connected state
is built during exploration:

- `style = global` (default): the mapped event's final positions are applied
  and the whole system is re-minimised.
- `style = global/reconstruction`: the full event reconstruction is used
  instead (relaxation from the saddle towards both minima, with matching
  verification) — stricter, at a higher cost per state.

*Note: currently only one exit-selection algorithm (FPTA, below) is implemented.
The `[Basin]` section is intended for future extensions when multiple algorithms (e.g., MRT, local basins, …) will be available.*

---

## General Idea

During a KMC step, a `Basin` object is created when the selected event's forward barrier is below `energy_thr` and at least one catalog event leaving the selected event's final topology also has a barrier below `energy_thr` (the reverse move at minimum).
It explores the basin, computes the exit time, and determines the exit state.
Once finished, the selected event in the KMC loop is replaced with the basin event.

<!-- TODO: add a figure illustrating basin entry and the super-event replacement. -->

While exploring, the main objective is to build a connectivity table containing all information required to apply the exit algorithm.
This table is stored as a `pandas.DataFrame`, managed by a connectivity object that merges rows, removes transitions, and queries state connections.

<!-- TODO: add a figure of the connectivity table / state-transition graph. -->

Column meaning:

* **state**: a transient state
* **state_connexion**: a state reachable from `state`
* **event_connexion**: ID of the event that takes `state → state_connexion`
* **central_atom**: atom index on which the event must be applied
* **sym**: symmetry index of the event
* **transient**: whether `state_connexion` is transient or not
* **dE_forward / k_forward**: barrier and rate from `state` to `state_connexion`
* **dE_backward / k_backward**: barrier and rate from `state_connexion` back to `state`

_Note: a pandas `DataFrame` is used for efficient querying and sorting and can easily be converted to a graph for analysis and characterization._

The `Basin` object uses two additional components:

* **Explorer**: explores a state and builds its connectivity rows
* **Selector**: uses the connectivity table to compute the exit state and exit time

*This structure is designed to support multiple future exploration/selection algorithms.*


## Algorithm

1. **Initialization**: state to explore = current state
2. **While states remain to explore:**

   1. If the state is already known (distance-based check), stop
   2. If the state has an unknown atomic environment → mark as absorbing → stop
   3. If the state is absorbing, stop
   4. Find the connection (event) between a known state and this state
   5. Build the connected state using the configured Basin style (`global` or `global/reconstruction`)
   6. Send the state to the Explorer:

      * detect applicable events
      * build its connectivity table
   7. Merge with the global connectivity table
3. **Refine** all transient → absorbing transitions (update dE and k)
4. **Selector step** (First Passage Time Analysis):

   * build the absorbing Markov-chain generator matrix $M$ from the
     connectivity table, collapsing all absorbing states into one
   * propagate the occupation vector $P(t) = e^{-Mt} P_0$ and use bisection to
     find the time $t_\text{exit}$ at which the absorbing probability reaches
     a random target, then draw the exit state from the distribution over the
     original absorbing states
5. Build the result and return it to the KMC loop
6. Replace the initially chosen KMC event with the basin event


The basin process may fail during PSR, refinement, reconstruction, or exit-time calculation.
If a failure occurs, the basin returns an `Err`, and the originally selected KMC event is applied instead.

