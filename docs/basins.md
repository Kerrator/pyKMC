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
```

*Note: currently only one basin-handling strategy is implemented.
The `[Basin]` section is intended for future extensions when multiple algorithms (e.g., FTPA, MRT, local basins, …) will be available.*

---

## General Idea

During a KMC step, if the selected event has both forward and backward barriers lower than `energy_thr`, a `Basin` object is created.
The backward barrier belongs to the catalogue row explicitly linked by the
selected event's logical reference ID. A refined active event retains its own
forward barrier. Equality with the threshold is outside the basin. Missing or
ambiguous logical links raise an error naming the affected IDs; self-links and
directional aliases are supported. Exploration uses the same linked reverse
for its classification and recorded backward barrier and rate.

A self-link is authoritative only for a genuine self-reverse, whose initial
and final topologies coincide (`event_id == id_final`). The constant-mode
writer also self-links a forward whose reverse was already catalogued; that
placeholder is resolved by topology among the reciprocal rows only (the
catalogued rows whose `event_id` is the forward's `id_final` **and** whose
`id_final` is the forward's `event_id`, lowest barrier when several exist),
never by comparing the forward barrier with itself and never by following
another channel that merely leaves the forward's final topology.

A placeholder whose reciprocal row is gone is not a malformed catalogue: in
constant mode `ReferenceEventTable.remove` keeps the base rule (the failed
row and the row its `idx_backward` names, no closure), so purging the
reciprocal after a failed reconstruction leaves the placeholder behind with
the catalogue layout unchanged. Its reverse barrier is then unknown, and an
unknown reverse cannot be shown fast, so the transition is **absorbing**
(the detector answers "not in a basin", the explorer records the edge as
absorbing with `NaN` backward barrier and rate) and one `WARNING` names the
row, the missing reciprocal (`id_final -> event_id`) and the reason. The
KMC step continues with the selected event; nothing propagates.

Missing or duplicated logical identities (two rows with one `idx_ref`, an
explicit `idx_backward` naming no row or several) are catalogue-integrity
errors, not basin failures: they raise a `ValueError` naming the affected
IDs and propagate out of the KMC step instead of taking the `Err` fallback
described below, because a catalogue that cannot resolve its own reverse
links must not silently fall back to executing the selected event.
It explores the basin, computes the exit time, and determines the exit state.
Once finished, the selected event in the KMC loop is replaced with the basin event.

FIGURE

While exploring, the main objective is to build a connectivity table containing all information required to apply the exit algorithm.
This table is stored as a `pandas.DataFrame`, managed by a `Connectivity` object (merge, remove, search for connections, ...).

FIGURE

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

*This structure is designed to support multiple future exploration/selection algorithms.*


## Algorithm

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
4. **Selector step** (first-passage-time analysis):

   * build the absorbing generator matrix ( M ) and its reduced form (all absorbing states merged)
   * draw the exit time by bisection on the reduced system: ( p_{\text{abs}}(t_{\text{exit}}) = r_1 )
   * draw the exit **transition** from the *instantaneous* flux at ( t_{\text{exit}} ): with the
     transient occupation ( q(t) = e^{-M_T t} e_0 ), a transient → absorbing transition
     ( e = (i \to a) ) of rate ( k_e ) has weight ( q_i(t_{\text{exit}}) \, k_e ). Cumulative
     absorption up to ( t_{\text{exit}} ) is a different conditioning and is not used. The selected
     transition keeps its own source state, reference event, symmetry, refined barrier and saddle,
     also when several transitions share a destination or connect the same pair of states.
   * a flux vector that is not finite, real and non-negative with a positive total is an `Err`
     (`BASIN_INVALID_EXIT_FLUX`): no approximate or uniform channel is substituted. `k_tot` in the
     step log is the unweighted sum of the exit rates (diagnostic); the clock advances by
     ( t_{\text{exit}} ).
5. Build the result and return it to the KMC loop
6. Replace the initially chosen KMC event with the basin event


The basin process may fail during PSR, refinement, reconstruction, or exit-time calculation.
If a failure occurs, the basin returns an `Err`, and the originally selected KMC event is applied instead.
