# KMC Algorithm Overview

pyKMC is an **off-lattice, on-the-fly** Kinetic Monte Carlo engine. Instead of
enumerating a fixed event list on a predefined lattice, it discovers transition
events from the atomic configuration as the simulation runs, and reuses them
wherever an equivalent local environment appears. This page describes the
overall workflow; the configuration fields that control it are documented on the
[KMC Parameters](parameters.md) page.

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
   (e.g. by coordination/graph hashing).
5. Initialise the event **catalog** and the set of already-visited environments.
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
        [Symmetries](symmetries.md).

3. **Build specific events and select.** Generic events that contribute
   significantly to the total rate are **refined** (and their symmetry-equivalent
   variants generated) for every atom sharing the same environment, producing
   the **specific** event list. An event is then chosen with the rejection-free
   KMC algorithm, and the simulation clock is advanced by the corresponding
   time increment.

4. **Basin handling (optional).** If the selected event has both its forward and
   backward barriers below the basin threshold, the system has entered a
   metastable basin. The [basin algorithm](basins.md) explores the connected
   transient states, solves for the mean exit time, selects an exit state, and
   replaces the single hop with a super-event that bridges the whole basin.

5. **Apply the event.** Update the atomic positions according to the selected
   event's final state.

6. **Update state.** Rebuild the neighbour list and atomic environments, record a
   trajectory snapshot, and return to step 1.

## Accelerations

- **[Basins](basins.md)** bridge fast, low-barrier hops between metastable
  states with a single super-event, advancing simulated time by many orders of
  magnitude.
- **[Active Volumes](active_volumes.md)** restrict searches and refinements to a
  region around a defect, making large systems tractable.
