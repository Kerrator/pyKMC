# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

pyKMC is an on-the-fly kinetic Monte Carlo (KMC) simulation program for atomistic systems. It dynamically discovers and catalogs atomic transition events during simulation by performing event searches when new atomic environments are encountered, then refines and reuses these events when similar environments recur.

## Development Commands

### Installation
```bash
# Install in editable mode
pip install -e .

# Install with development dependencies
pip install -e ".[dev]"
```

### Running Simulations
```bash
# Basic usage (serial)
python -m pykmc -in <input_file.in>

# Parallel execution with MPI (recommended, minimum 8 cores)
mpirun -n 8 python -m pykmc -in <input_file.in>
```

Entry point: `pykmc/__main__.py` → `run.py:main()` → parses INI config, launches MPI via `ManagerFactory`, runs `KMC._initialize()` then `KMC.run()`. Only rank 0 runs the KMC loop; other ranks act as LAMMPS worker engines.

### Testing
```bash
# Run all tests
pytest

# Run specific test file
pytest tests/test_atomic_environment.py

# Run tests with MPI (for engine manager tests)
mpirun -n 4 pytest tests/test_lammps_engine_api_mpi.py

# Run tests in a specific directory
pytest tests/basins/
```

### Code Quality
```bash
# Format code with ruff
ruff format .

# Lint code
ruff check .

# Auto-fix linting issues
ruff check --fix .

# Type checking with mypy
mypy pykmc/
```

### Documentation
```bash
# Build documentation
mkdocs build

# Serve documentation locally
mkdocs serve
```

## Architecture

### Core Simulation Flow

The KMC simulation follows this workflow:
1. **Initialize** system from configuration file (readable by ASE)
2. **Minimize** atomic structure
3. **Classify** atomic environments (assign IDs to atoms based on local structure)
4. **Search** for new events in unexplored environments (via pARTn/LAMMPS)
5. **Refine** reference events to current configuration (via point set registration)
6. **Select** event from active table (using KMC algorithm)
7. **Apply** event and update system
8. **Repeat** steps 2-7 until completion

### Module Structure

- **`kmc.py`**: Central controller coordinating all simulation phases
- **`system.py`**: Extends ASE `Atoms` to represent the atomic system
- **`config.py`**: Configuration management (parses INI files with Pydantic validation)
- **`event_table.py`**: Manages reference and active event catalogs (pandas DataFrames)
- **`atomic_environment.py`**: Assigns environment IDs to atoms
- **`environments/`**: Environment classification methods
  - `common_neighbor_analysis.py`: CNA-based crystal/non-crystal classification
  - `graph_nauty.py`: Graph-based fingerprinting using pyNauty canonical forms
- **`eventsearch.py`**: Coordinates event search operations
- **`refinement.py`**: Adapts reference events to current atomic configurations
- **`point_set_registration.py`**: Shape matching (IRA) to align event with local environment
- **`basins/`**: Advanced basin detection and exploration
  - `basin.py`: Basin state representation and management
  - `detection.py`: Detects when system enters metastable basins
  - `exploration.py`: Basin exploration strategies
  - `selection.py`: Basin-aware event selection
  - `connectivity.py`: Tracks transitions between basin states (DataFrame with lazy materialization via `_rows` buffer)
  - `exit_time_solver.py`: Computes mean exit times (FPTA algorithm)

  Basin exploration supports multiple parallelization strategies via `config.basin.strategy`: `serial` (default), `parallel_explore`, `batch_dedup`, `parallel_reconstruct`, `wavefront`. Key bottlenecks are deduplication (~58% of serial time) and reconstruction (~37%); exploration is negligible (~0.2%).
- **`enginemanager/lmpi/`**: MPI-based engine parallelization
  - `pool/`: Manages pool of LAMMPS instances across MPI ranks
  - `engines/`: LAMMPS MPI API wrapper
  - `sessions/`: Session management for parallel event searches
  - `lammps_operations.py`: High-level LAMMPS operations (minimize, compute energy/forces)
- **`algorithms.py`**: KMC algorithms (e.g., rejection-free)
- **`rate_constant.py`**: Event rate calculations (Arrhenius)
- **`symmetries.py`**: Symmetry operations for events
- **`reconstruction.py`**: Event reconstruction from reference
- **`result.py`**: Result types (using Rust-style Result[T, E] pattern with Ok/Err)
- **`log.py`**: Structured logging with colors. Named loggers: `"log"` (main simulation), `"output"` (KMC step table), `"info"` (statistics), `"events"` (event details), `"progress"` (step progress). **Important**: always use these exact names with `logging.getLogger()` — using `__name__` will silently drop messages.
- **`initializer.py`**: Simulation initialization
- **`neighbors_list.py`**: Neighbor list management

### Engine Manager (LMPI)

The engine manager enables parallel event searches by maintaining a pool of LAMMPS instances distributed across MPI ranks:
- **Master rank (0)**: Runs main KMC loop, delegates work to LAMMPS sessions
- **Worker ranks**: Each runs a LAMMPS instance, processes search/refinement requests
- Communication via MPI (or threading for rank 0 if `engine_use_rank_0=True`)
- Sessions are allocated from pool, perform operations, then return to pool

### Configuration System

Simulations use INI files with sections:
- **`[Control]`**: General parameters (input file, steps, engine choice, output files)
- **`[Lammps]`**: LAMMPS potential parameters (pair_style, pair_coeff, minimization settings)
- **`[AtomicEnvironment]`**: Environment ID parameters (style: cna/graph/cna_graph, rnei, rcut, neighbors_add)
- **`[EventSearch]`**: Event search settings (style: partn, nsearch, energy thresholds)
- **`[pARTn]`**: pARTn-specific parameters (path_artnso, push_dist_thr, etc.)
- **`[RateConstant]`**: Rate calculation (style: constant, k0, T)
- **`[PSR]`**: Point Set Registration (style: ira, matching_score_thr)
- **`[IRA]`**: IRA-specific parameters
- **`[Basin]`**: Basin exploration (energy_thr, strategy, n_workers, fingerprint settings)

See `docs/user_guide.md` for detailed parameter explanations.

## External Dependencies

### LAMMPS
- Requires Python bindings: `make install-python`
- Must be compiled with: `make mode=shared mpi` for pARTn support
- Packages needed: BASIC, EXTRA-COMPUTE (for CNA), PLUGIN (for pARTn)
- Installation tested with August 24, 2024 version

### pARTn (Activation-Relaxation Technique nouveau)
- LAMMPS plugin for saddle point searches
- Located in `../artn-plugin/` (sibling directory)
- Compiled as shared library: `libartn-lmp.so`
- Path must be specified in configuration: `path_artnso = /path/to/artn-plugin/lib/libartn-lmp.so`
- Python interface must be in PYTHONPATH: `export PYTHONPATH=/path/to/artn-plugin/interface:$PYTHONPATH`

### IRA (Iterative Rotations Assignments)
- Point set registration library for event reconstruction
- Located in `../IterativeRotationsAssignments/` (sibling directory)
- Python interface must be in PYTHONPATH: `export PYTHONPATH=/path/to/IRA/interface:$PYTHONPATH`

## Code Style and Type Safety

### Type Checking
- Strict mypy configuration (`mypy.ini`) enforces:
  - All function signatures must be typed
  - No implicit `Any` types allowed
  - Strict handling of `Optional` types
  - All instance attributes must be explicitly typed
- Run `mypy pykmc/` before committing

### Formatting and Linting
- Follow ruff configuration (`ruff.toml`):
  - Line length: 88 characters (Black-compatible)
  - Python 3.10 target
  - Enabled checks: E4, E7, E9, F, B, Q, ANN (annotations), D (docstrings)
  - Docstrings: Google/NumPy style (ignore D107, D213, D203)
- All public functions/classes require docstrings

### Result Types
- Use `Result[T, E]` pattern from `result.py` for operations that can fail
- Return `Ok(value)` for success, `Err(error_info)` for failure
- Error types defined in `ErrorType` enum (e.g., `EVENT_SEARCH_FAILED`, `REFINEMENT_FAILED`)

## Common Development Patterns

### Adding a New Atomic Environment Style
1. Create classifier in `pykmc/environments/`
2. Add to `AtomicEnvironment` class in `atomic_environment.py`
3. Update `Config` validation in `config.py`
4. Add tests in `tests/test_atomic_environment.py`

### Adding a New Event Search Method
1. Implement interface in `eventsearch.py`
2. Add engine-specific operations in `enginemanager/lmpi/lammps_operations.py`
3. Update configuration parsing in `config.py`
4. Add corresponding config section validator

### Modifying LAMMPS Operations
- All LAMMPS operations go through `enginemanager/lmpi/`
- Session-based interface: request session, perform operations, release
- Use `messenger.py` for communication between master and workers
- Operations are serialized/deserialized via pickle

### Working with Events
- Events stored as pandas DataFrame rows in `ReferenceEventTable` / `ActiveEventTable`
- Key columns: `event_id`, `energy_barrier`, `k` (rate), `initial_positions`, `saddle_positions`, `final_positions`
- Event ID is the atomic environment ID of the moving atom
- Positions are stored as numpy arrays in the DataFrame

## Testing Notes

- Use fixtures from `tests/conftest.py` for common test setups
- Basin tests have separate conftest: `tests/basins/conftest.py`
- MPI tests require `mpirun` and may need special pytest plugins
- Test data files in `tests/data/`
- Example configurations in `examples/` directory

## Related Repositories

This repository is part of a multi-project workspace:
- **pyKMC** (this project): Main KMC simulation code
- **artn-plugin**: Event search engine (saddle point finding)
- **IterativeRotationsAssignments**: Point set registration library
- **lammps**: Molecular dynamics simulator (customized version)
- **lammps-mtp-kokkos**: LAMMPS MTP integration with Kokkos

When making changes that affect interfaces with pARTn or IRA, coordinate across repositories.
