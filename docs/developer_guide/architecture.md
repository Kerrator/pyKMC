# Architecture

This page sketches how the pyKMC package is organized and how the pieces fit
together. It is an outline intended to grow with the code — see the
[API Reference](../api/system.md) for the per-module details rendered from
docstrings.

## Package map

| Module / package | Role |
|---|---|
| `pykmc/kmc.py` | Main KMC loop and orchestration |
| `pykmc/system.py` | Atomic configuration (positions, cell, types) |
| `pykmc/config.py` | Typed (pydantic) configuration model — self-documents the [parameters page](../parameters.md) |
| `pykmc/eventsearch.py` | Saddle-point searches via pARTn |
| `pykmc/reconstruction.py` | Event reconstruction in equivalent environments (IRA) |
| `pykmc/refinement.py` | Refinement of reused events to the current configuration |
| `pykmc/event_table.py` | Reference (generic) and active (specific) event tables |
| `pykmc/atomic_environment.py`, `pykmc/environments/` | Environment classification (CNA / graph / hybrid) |
| `pykmc/rate_constant.py` | Rate constants from barriers (see [TST](../theory/tst.md)) |
| `pykmc/symmetries.py` | Symmetry-equivalent event expansion (SOFI) |
| `pykmc/basins/` | Basin detection, exploration, and exit-time solution |
| `pykmc/activevolume.py` | Active-volume restriction for large systems |
| `pykmc/enginemanager/` | LAMMPS engine session pool over MPI (see below) |
| `pykmc/bias.py` | Event-selection bias |
| `pykmc/log.py`, `pykmc/result.py`, `pykmc/info_simulation.py` | Logging and output |

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

## Engine manager (LAMMPS session pool)

`pykmc/enginemanager/` manages the pool of LAMMPS instances: session
allocation across MPI ranks, message passing between the rank-0 orchestrator
and the sessions, and the LAMMPS operations themselves
(`lmpi/`: `pool/`, `sessions/`, `engines/`).

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

- **New engine:** implement the engine interface under `enginemanager/`.
- **New rate-constant style:** extend `rate_constant.py` and register the
  style in `config.py`.
- **New basin algorithm:** add an explorer/selector under `pykmc/basins/`.

See [CONTRIBUTING](https://github.com/hugomoison/pyKMC/blob/develop/CONTRIBUTING.md)
for the docstring and documentation requirements that keep the API reference
complete.
