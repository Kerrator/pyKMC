# pyKMC

**pyKMC** is an off-lattice, on-the-fly Kinetic Monte Carlo (KMC) framework for
atomic-scale rare-event simulations in materials science. Rather than relying on
a predefined lattice and a fixed event list, pyKMC discovers transition events
directly from the atomic configuration during the simulation and reuses them
across equivalent local environments.

It coordinates three computational components:

- **[LAMMPS](https://www.lammps.org/)** — energy and force evaluation for atomic
  configurations (managed as a pool of parallel MPI instances).
- **[pARTn](https://mammasmias.gitlab.io/artn-plugin/)** — saddle-point search
  (parallel Activation-Relaxation Technique nouveau) to discover transition
  events and their activation barriers.
- **[IRA](https://mammasmias.github.io/IterativeRotationsAssignments/)** —
  point-set registration to reconstruct known events in new but equivalent
  local environments, avoiding redundant saddle-point searches.

## Features

- On-the-fly event discovery via saddle-point search (no precomputed catalog needed)
- Event reuse across symmetry-equivalent environments
- Rejection-free KMC event selection
- **[Basin acceleration](user_guide/basins.md)** to bridge metastable regions and advance
  simulation time by orders of magnitude
- **[Active Volumes](user_guide/active_volumes.md)** to restrict expensive searches to a
  region around a defect in large systems
- Multi-element alloy support and MPI-parallel execution

## Get started

1. **[Install pyKMC](user_guide/install/installation.md)** — one-shot scripts for macOS and
   Linux, plus manual build instructions.
2. **[User Guide](user_guide/index.md)** — how to set up and run a simulation.
3. **[KMC Parameters](parameters.md)** — the full configuration reference.

## Documentation map

| Section | What's there |
|---|---|
| [User Guide](user_guide/index.md) | Installation, running simulations, and feature guides |
| [Theory & Concepts](theory/general_algorithm.md) | The KMC algorithm, transition state theory, and KMC background |
| [KMC Parameters](parameters.md) | Every configuration field, auto-generated from the code |
| [Troubleshooting](troubleshooting.md) | Common errors and fixes |
| [Developer Guide](developer_guide/index.md) | Contributing, architecture, testing, and building the docs |
| [API Reference](api/index.md) | Module-by-module API, generated from docstrings |
