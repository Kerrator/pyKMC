# API Reference

Auto-generated reference for the public `pykmc` modules, rendered from the
code's NumPy-style docstrings by mkdocstrings. Pick a module from the
navigation, or start from the groups below. For how the pieces fit together,
see the [Architecture](../developer_guide/architecture.md) page.

## Core loop and entry points

- [KMC](kmc.md) — the main per-step loop and orchestration
- [Algorithms](algorithms.md) — rejection-free (BKL) selection and time advance
- [System](system.md) — the atomic configuration
- [Initializer](initializer.md) / [Run](run.md) — simulation setup and the
  `python -m pykmc` entry point
- [Config](config.md) — the typed (pydantic) configuration model

## Environments and events

- [NeighborsList](neighbors_list.md) — periodic neighbour lists
- [AtomicEnvironment](atomic_environment.md) / [Environments](environments.md)
  — environment classification (CNA, graph, coordination, region)
- [EventSearch](eventsearch.md) — saddle-point searches via pARTn
- [EventTable](event_table.md) — reference and active event tables
- [EventRecycling](event_recycling.md) — carry-over of still-valid events
- [PointSetRegistration](point_set_registration.md) /
  [Symmetries](symmetries.md) — IRA matching and SOFI symmetry expansion
- [Refinement](refinement.md) / [Reconstruction](reconstruction.md) — adapting
  and validating reused events
- [RateConstant](rate_constant.md) — rates from barriers
- [Bias](bias.md) — event-selection bias

## Acceleration and engines

- [Basins](basins.md) — basin detection and exit-time solution
- [ActiveVolume](active_volume.md) — active-volume restriction
- [Engine](engine.md) / [LammpsEngine](lammpsengine.md) — the engine
  abstraction and its LAMMPS implementation

## Output and utilities

- [Log](log.md) — logging
- [Result](result.md) — result and error types
- [Info](info_simulation.md) — simulation diagnostics
- [Utils](utils.md) — geometry helpers

> **Note:** `pykmc.enginemanager` (the MPI session pool) is a namespace
> package that the documentation tooling cannot introspect; it is documented
> in prose in the
> [developer guide](../developer_guide/architecture.md#engine-manager-lammps-session-pool)
> and [MPI & Parallel Execution](../developer_guide/mpi.md).
