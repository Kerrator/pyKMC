# Architecture

This page sketches how the pyKMC package is organized and how the pieces fit
together — see the [API Reference](../api/system.md) for the per-module details
rendered from docstrings, and the [Strategy pattern](strategy_pattern.md) and
[Engine](engine.md) pages for deep dives into the two extension mechanisms.

## Package map

| Module / package | Role |
|---|---|
| `pykmc/kmc.py` | Main KMC loop and orchestration |
| `pykmc/algorithms.py` | Rejection-free (BKL) event selection and time increment |
| `pykmc/initializer.py`, `pykmc/run.py` | Simulation setup and the `python -m pykmc` entry point |
| `pykmc/system.py` | Atomic configuration (positions, cell, types) |
| `pykmc/config.py` | Typed (pydantic) configuration model — self-documents the [parameters page](../parameters.md) |
| `pykmc/neighbors_list.py` | Periodic neighbour lists (the `rnei` / `rcut` shells) |
| `pykmc/atomic_environment.py`, `pykmc/environments/` | Environment classification (CNA / graph / coordination / region) |
| `pykmc/eventsearch.py` | Saddle-point searches via pARTn |
| `pykmc/event_table.py` | Reference (generic) and active (specific) event tables |
| `pykmc/point_set_registration.py` | Point-set registration between local environments (IRA) |
| `pykmc/reconstruction.py` | Event reconstruction in equivalent environments (IRA) |
| `pykmc/refinement.py` | Refinement of reused events to the current configuration |
| `pykmc/event_recycling.py` | Carry-over of still-valid active events between steps |
| `pykmc/rate_constant.py` | Rate constants from barriers (see [TST](../theory/tst.md)) |
| `pykmc/symmetries.py` | Symmetry-equivalent event expansion (SOFI) |
| `pykmc/basins/` | Basin detection, exploration, and exit-time solution |
| `pykmc/activevolume/` | Active-volume restriction for large systems |
| `pykmc/engine/` | Engine abstraction (`Engine` ABC + extensions) and the LAMMPS implementation — see [Engine](engine.md) |
| `pykmc/enginemanager/` | LAMMPS engine session pool over MPI (see below) |
| `pykmc/_core/` | `Registrable` registry and autodiscovery shared by all pluggable hierarchies |
| `pykmc/bias.py` | Event-selection bias |
| `pykmc/log.py`, `pykmc/result.py`, `pykmc/info_simulation.py` | Logging and output |
| `pykmc/utils/` | Geometry helpers and generic-event application |

## KMC loop orchestration

`kmc.py` drives the cycle described in the
[Algorithm Overview](../theory/general_algorithm.md): classify environments →
search for events at new environments → reconstruct/refine known events →
select with rejection-free KMC → apply → update. Each stage delegates to the
modules above.

## Configuration and validation

All input is validated up front against the pydantic `Config` model in
`config.py`. Every field carries a description that is rendered automatically
into the [KMC Parameters](../parameters.md) page — adding a config field
documents itself.

## Pluggable components (`_core`)

`pykmc/_core/` holds the plumbing shared by every pluggable hierarchy:
`Registrable` gives a base class declared with `root=True` its own registry,
into which concrete subclasses insert themselves at class-definition time
under a declared `name`; `autodiscover()` imports every submodule of a package
so implementations register simply by being defined. Two kinds of component
are meant to build on this:

- **Engines** — computational backends for energy/force operations, which are
  fixed backends rather than interchangeable algorithms. The engine hierarchy
  is currently the only production user of `Registrable`/autodiscovery. See
  [Engine](engine.md).
- **Strategies** — interchangeable algorithm implementations behind a stable
  facade (environment classification, event search, …). This is a *proposed*
  architecture: no production KMC component has migrated to it yet. See
  [Strategy pattern](strategy_pattern.md).

## Engine layer

`pykmc/engine/` defines the abstract `Engine` (a `Registrable` root) plus the
`EngineExtension` mechanism, and `LammpsEngine` as the concrete LAMMPS
implementation — usable standalone (serial or on an MPI communicator). It is
designed as the future backend for the session pool below, but that
integration has not happened yet: the current `python -m pykmc` production
path still drives LAMMPS through `enginemanager.lmpi.engines.MpiApiEngine`
and the `lammps_operations` functions. The [Engine](engine.md) page documents
the interface, the rank-0 result convention, and how to add a new engine.

## Engine manager (LAMMPS session pool)

`pykmc/enginemanager/` manages the pool of LAMMPS instances that a
production run works through (`run.py` builds it via `ManagerFactory`):

- `lmpi/pool/` — `ManagerFactory` splits `MPI.COMM_WORLD` into contiguous
  per-session communicators and wires the pool; `Manager` runs one worker
  thread per session on rank 0 and dispatches jobs to free sessions.
- `lmpi/sessions/` — the rank-0-side session handles through which the
  Manager sends commands and receives results.
- `lmpi/engines/` — `MpiApiEngine`, the worker-side command loop executing on
  each session's ranks.
- `lmpi/lammps_operations.py` — the LAMMPS/pARTn operations the workers run
  (minimisation, event search, refinement, …).
- `messenger.py` — the rank-0 ↔ session transport: `MpiMessenger` over MPI
  messages, or `QueueMessenger` (in-process queues) when a session shares
  rank 0.

The manager exposes the pool in two modes: **local** (each session works
independently on its own communicator — used for concurrent event searches and
refinements) and **global** (all engine ranks act together for whole-system
operations such as reconstruction). At launch every worker joins both a local
session communicator and one global communicator and creates one LAMMPS
instance for each; `use_local()` / `use_global()` switch which pre-created
instance receives subsequent operations. See
[MPI & Parallel Execution](mpi.md) for rank layout and sizing rules.

> **Note:** `enginemanager` is currently a namespace package (no
> `__init__.py`), which the documentation tooling cannot introspect — so it has
> no auto-generated API page and is documented in prose here and in
> [MPI & Parallel Execution](mpi.md).

## Event catalog and reuse

Discovered events are stored as **generic** events in the reference table,
keyed by the environment ID of the central atom. When an environment recurs,
the generic event is reconstructed and refined into a **specific** event for
the current geometry, avoiding a new saddle-point search.

## Extending pyKMC

- **New engine:** subclass `Engine` in `pykmc/engine/` and declare a `name` —
  see [Adding a new engine](engine.md#adding-a-new-engine).
- **New strategy** for a pluggable operation: the facade/strategy migration is
  still a design proposal — see
  [Strategy pattern](strategy_pattern.md) for the intended shape.
- **New rate-constant style:** requires a new implementation, a new allowed
  configuration literal, and dispatch at every rate-construction call site —
  the call sites currently invoke `compute_rate_Eyring` directly, so there is
  no style registry to hook into yet.
- **New basin algorithm:** the explorer and selector are currently
  hard-coded in `pykmc/basins/basin.py`; a pluggable builder is planned but
  adding an algorithm today means editing that construction directly.

See [CONTRIBUTING](https://github.com/hugomoison/pyKMC/blob/develop/CONTRIBUTING.md)
for the docstring and documentation requirements that keep the API reference
complete.
