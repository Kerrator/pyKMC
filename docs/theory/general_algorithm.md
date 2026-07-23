# KMC Algorithm Overview

pyKMC is an **off-lattice, on-the-fly** Kinetic Monte Carlo engine. Instead of
enumerating a fixed event list on a predefined lattice, it discovers transition
events from the atomic configuration as the simulation runs, and reuses them
wherever an equivalent local environment appears. This page describes the
overall workflow; the configuration fields that control it are documented on the
[KMC Parameters](../parameters.md) page.

## Components

| Component | Role |
|---|---|
| **LAMMPS** | Energy/force engine. Used for minimisation and as the force evaluator during searches. Run as a pool of MPI sessions. |
| **pARTn** | Saddle-point search (Activation-Relaxation Technique nouveau). Finds transition events and their activation barriers. |
| **IRA / SOFI** | Point-set registration and symmetry detection. Reconstructs known events in new equivalent environments and finds symmetry-equivalent variants. |

## Initialization

Before the main loop, pyKMC sets up the run:

1. Read and validate the input file into a typed configuration.
2. Start the logger and the LAMMPS engine session pool.
3. Build the initial `System` and minimise it, updating atomic positions.
4. Build the neighbour list and classify each atom's **atomic environment**
   (common-neighbor analysis and/or graph topology hashing, selected by the
   `[AtomicEnvironment]` `style`).
5. Initialise the event **catalog** and the set of already-visited
   environments — either empty, or restored from a previous run's saved files
   (`reference_table` / `visited_environments`), which lets a simulation chain
   onto an existing catalog.
6. Set `time = 0` and record the first trajectory snapshot.

## The KMC loop

Each step repeats the following:

1. **Find new environments.** Compare the environments present in the current
   configuration against the set already visited. Any environment not seen
   before needs events.

2. **Search for events.** For each new environment, run `nsearch` saddle-point
   searches (with pARTn) at representative atoms carrying that environment ID.
   Each discovered event records its initial state, saddle point, final state,
   and activation barrier. Events are recentred to handle periodic boundary
   conditions and added to the catalog as **generic** events.

      - **Symmetry expansion.** SOFI (from IRA) returns the symmetry operations
        of the initial configuration. pyKMC keeps only the operations that map
        the event's displacement to a distinct result, storing the unique
        symmetry and permutation matrices alongside the event. See
        [Symmetries](../user_guide/symmetries.md).

3. **Build the specific (active) event list.** Every generic event whose
   environment ID occurs in the current configuration is instantiated for each
   atom carrying that environment, including its stored symmetry-equivalent
   variants. To bound the cost, only the instances whose rate would contribute
   a significant fraction of the total rate (threshold controlled by
   `refine_thr`) are individually **refined** — re-converged to the saddle with
   pARTn at their actual site; the remaining instances inherit the generic
   event's barrier unchanged. Duplicates are removed, and the surviving rows
   form the **active** event table with a rate constant per event.

4. **Select and reconstruct.** An event is chosen with the rejection-free
   KMC algorithm, and the clock increment is drawn from the total rate (see
   [Kinetic Monte Carlo](kmc.md)). The chosen event's stored saddle and final
   geometries are then **reconstructed** onto the current configuration with
   IRA, and relaxations from the saddle must recover both the initial and the
   final minimum within the matching threshold. If reconstruction fails, the
   active event is dropped, its reference event and topology are purged (so
   the environment will be re-searched later), and selection repeats with the
   remaining events.

5. **Basin handling (optional).** If the selected event has both its forward and
   backward barriers below the basin threshold, the system has entered a
   metastable basin. The [basin algorithm](../user_guide/basins.md) explores the connected
   transient states, solves for the mean exit time, selects an exit state, and
   replaces the single hop with a super-event that bridges the whole basin.

6. **Apply the event.** Update the atomic positions to the reconstructed final
   minimum and advance the simulation time.

7. **Update state.** Rebuild the neighbour list and atomic environments, record
   a trajectory snapshot, and return to step 1. If every atom now classifies as
   a crystalline environment (no defects left to move), the run ends. When
   **event recycling** is enabled (`recycle = True`), active events whose
   central atom did not move during the executed event and lies far from it
   are carried over to the next step instead of being refined again.

## Accelerations

- **[Basins](../user_guide/basins.md)** bridge fast, low-barrier hops between metastable
  states with a single super-event, advancing simulated time by many orders of
  magnitude.
- **[Active Volumes](../user_guide/active_volumes.md)** restrict searches and refinements to a
  region around a defect, making large systems tractable.
