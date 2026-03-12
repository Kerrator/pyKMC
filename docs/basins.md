# Basins

Basin settings are defined in the `[Basin]` section of the input file.
The only required parameter is the energy threshold below which a state is considered part of the basin.
You must also enable basin mode in the `[Control]` section.

Example:

```INI
[Control]
...
basin = True
...

[Basin]
energy_thr = 0.1
strategy = wavefront
n_workers = 4
max_states = 500
```

### Exploration Strategies

The BFS exploration loop can be parallelized using different strategies, configured via `strategy`:

| Strategy | Description |
|----------|-------------|
| `serial` (default) | Sequential BFS — one state at a time |
| `parallel_explore` | Parallel exploration of newly discovered transient states |
| `batch_dedup` | Batched cKDTree deduplication across multiple candidates |
| `parallel_reconstruct` | Parallel reconstruction via session pool (distributes LAMMPS minimizations across all MPI ranks) |
| `wavefront` | Full wavefront BFS combining parallel reconstruction, batch dedup, and parallel exploration in each iteration |

`n_workers` controls the number of threads used by parallel strategies (default: 4).
`max_states` can be used as a safety cap for large basins. When the cap is reached, the remaining frontier states are reconstructed and converted to absorbing exits instead of continuing the BFS indefinitely. The capped basin is then treated as complete for exit selection: the selector chooses an exit rate/state from the capped connectivity graph, and KMC accepts that selected exit directly instead of falling back to the original event.
For MPI-backed basin strategies, keep `[Control] engine_use_rank_0 = False`; rank-0 engine participation remains unsupported in the current manager implementation.

---

## General Idea

During a KMC step, if the selected event has both forward and backward barriers lower than `energy_thr`, a `Basin` object is created.
It explores the basin, computes the exit time, and determines the exit state.
Once finished, the selected event in the KMC loop is replaced with the basin event.

*States with all barriers below `energy_thr` are transient (intra-basin); states reachable via at least one barrier above the threshold are absorbing (basin exits).*

While exploring, the main objective is to build a connectivity table containing all information required to apply the exit algorithm.
This table is stored as a `pandas.DataFrame`, managed by a `Connectivity` object (merge, remove, search for connections, ...).

*Each row in the connectivity table represents a directed transition between two states, with the associated event, rate, and barrier information.*

Column meaning:

* **state**: a transient state
* **state_connexion**: a state reachable from `state`
* **event_connexion**: ID of the event that takes `state → state_connexion`
* **central_atom**: atom index on which the event must be applied
* **sym**: symmetry index of the event
* **transient**: whether `state_connexion` is transient or not
* **dE_forward / k_forward**: barrier and rate from `state` to `state_connexion`
* **dE_backward / k_backward**: barrier and rate from `state_connexion` back to `state`

_Note: Pandas Dataframe were choosen for fast querying and sorting, and it can easily be converted to a graph for analysis and characterization._

The `Basin` object uses two additional components:

* **Explorer**: explores a given state by creating its connectivity table
* **Selector**: use the connectivity table to compute the exit state and exit time

*This structure is designed to support multiple exploration/selection algorithms.*

### Optimizations

Three optimizations reduce basin exploration time, particularly for large systems:

1. **State fingerprinting**: A fast pre-filter for deduplication. Each state's fingerprint is the sorted vector of per-atom distances from the center of mass — rotationally and permutationally invariant, O(N) to compute. Non-matching fingerprints skip the expensive cKDTree comparison entirely.

2. **Batched DataFrame operations**: The connectivity table uses a row buffer (`list[dict]`) with lazy DataFrame materialization. Rows are appended in O(1); the `pandas.DataFrame` is only constructed when first queried. This avoids O(N^2) overhead from repeated `pd.concat` calls.

3. **Session pool reconstruction**: When enabled (via `parallel_reconstruct` or `wavefront` strategies), LAMMPS minimizations during reconstruction are distributed across all available MPI sessions instead of routing through rank 0 only. This is the dominant speedup for systems with expensive minimizations.

*See the [basin optimization report](../../toolkit/docs/basin_optimization_report.md) for detailed profiling and benchmarks.*

## Algorithm

The following describes the serial BFS flow. Parallel strategies (`wavefront`, `parallel_reconstruct`, etc.) restructure the inner loop into batched phases — parallel reconstruction, batch deduplication, and parallel exploration — but the overall structure is the same.

1. **Initialization**: state to explore = current state
2. **While states remain to explore:**

   1. If the state is already known (distance-based check), stop
   2. If the state has an unknown atomic environment → mark as absorbing → stop
   3. If the state is absorbing, stop
   4. Find the connection (event) between a known state and this state
   5. Apply the event (reconstruction using a generic event)
   6. Send the state to the Explorer:

      * detect applicable events
      * build its connectivity table
   7. Merge with the global connectivity table
3. **Refine** all transient → absorbing transitions (update dE and k)
4. **Selector step**:

   * build matrix ( M ), solve ( P = e^{-Mt} P_0 )
   * use bisection to find ( t_{\text{exit}} ) and the exit state
5. Build the result and return it to the KMC loop
6. Replace the initially chosen KMC event with the basin event


The basin process may fail during PSR, refinement, reconstruction, or exit-time calculation.
If a failure occurs, the basin returns an `Err`, and the originally selected KMC event is applied instead.
