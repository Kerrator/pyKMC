# Developer Guide

This page covers working on pyKMC itself — repository layout, tests, code
quality, and building the documentation. For the full contribution workflow and
the documentation/docstring standard, see
[`CONTRIBUTING.md`](https://github.com/hugomoison/pyKMC/blob/develop/CONTRIBUTING.md).

## Repository layout

```
pyKMC/
├── pykmc/                # the package
│   ├── kmc.py            # main KMC loop / orchestration
│   ├── system.py         # atomic configuration
│   ├── eventsearch.py    # saddle-point search (pARTn)
│   ├── reconstruction.py # event reconstruction (IRA)
│   ├── engine/           # engine abstraction + LAMMPS implementation
│   ├── enginemanager/    # LAMMPS engine session pool (MPI)
│   ├── basins/           # basin acceleration
│   ├── activevolume/     # active-volume restriction
│   ├── config.py         # typed (pydantic) configuration model
│   └── ...
├── docs/                 # mkdocs documentation sources
├── scripts/              # helper scripts (e.g. parameters doc generator)
└── tests/                # pytest test suite
```

## Development install

```bash
python3 -m venv pykmc_env
source pykmc_env/bin/activate
pip install -e ".[dev]"   # ruff, mkdocs, mkdocstrings, mike, ...
pip install pytest pytest-lazy-fixtures   # not in the project dependencies; install separately
```

(See the [installation guide](../user_guide/install/installation.md) for the LAMMPS / pARTn /
IRA components needed to actually run simulations.)

## Running the tests

Some test modules exercise the MPI session pool and cannot run in a plain
single-process `pytest` invocation — they need a real `mpirun` launch with
enough ranks. Run the serial subset and the MPI subset separately:

```bash
# serial subset (excludes the MPI-pool test modules)
pytest --ignore=tests/manager/lmpi \
       --ignore=tests/basins/test_basin.py \
       --ignore=tests/test_lammps_engine_api_mpi.py

# MPI-pool tests (n_sessions = 7 plus the rank-0 driver)
mpirun -n 8 python -m pytest tests/basins/test_basin.py
mpirun -n 8 python -m pytest tests/manager/lmpi/test_manager.py

# a single test / pattern match
pytest tests/test_system.py::TestSystem::test_create_from_file_xyz -v
pytest -k "vacancy"
```

## Code quality

pyKMC enforces strict linting and typing. Run all three before opening a PR:

```bash
ruff check . --fix    # lint (auto-fix)
ruff format .         # format
mypy pykmc/           # strict type checking
```

The ruff configuration (`ruff.toml`) requires **docstrings** (`D` rules) and
**type annotations** (`ANN` rules), uses double quotes, and an 88-character line
length, targeting Python 3.10.

> **Note:** CI enforces **formatting only** — every push and pull request runs
> `ruff format --check` (see [Code formatting](code_formatting.md)). The lint
> rules (`ruff check`), `mypy`, and the test suite are not gated by CI and must
> be run locally.

## Building the documentation

The docs are built with [MkDocs](https://www.mkdocs.org/) + Material, with the
API reference auto-generated from docstrings by
[mkdocstrings](https://mkdocstrings.github.io/) and versioned with
[mike](https://github.com/jimporter/mike).

```bash
pip install -e ".[doc]"                     # docs toolchain
python scripts/generate_parameters_doc.py    # regenerate the parameters reference
mkdocs serve                                 # live preview at http://127.0.0.1:8000
mkdocs build                                 # site pass gate
mkdocs build --strict                        # inspect the warning delta
```

Run `mkdocs build` before committing documentation changes — a clean
non-strict build is the site pass gate. Also run `mkdocs build --strict` to
inspect warnings: the repository has a known baseline of pre-existing
mkdocstrings warnings, so the standard is a clean non-strict build plus no
*new* strict warnings from the files you changed.

The published site is deployed automatically by the
`.github/workflows/deploy_docs.yml` GitHub Action: pushes to `main` publish the
`latest` version and pushes to `develop` publish the `develop` version, both
selectable from the version switcher on the site.

For the documentation/docstring standard that keeps the API reference complete,
see [`CONTRIBUTING.md`](https://github.com/hugomoison/pyKMC/blob/develop/CONTRIBUTING.md).
