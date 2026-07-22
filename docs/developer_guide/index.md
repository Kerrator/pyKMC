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
pip install pytest         # not in the project dependencies; install separately
```

(See the [installation guide](../user_guide/install/installation.md) for the LAMMPS / pARTn /
IRA components needed to actually run simulations.)

## Running the tests

```bash
pytest                                   # whole suite
pytest tests/test_system.py::test_fn -v  # a single test
pytest -k "vacancy"                      # pattern match
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

> **Note:** there is currently no CI workflow that gates these checks on pull
> requests, so they must be run locally.

## Building the documentation

The docs are built with [MkDocs](https://www.mkdocs.org/) + Material, with the
API reference auto-generated from docstrings by
[mkdocstrings](https://mkdocstrings.github.io/) and versioned with
[mike](https://github.com/jimporter/mike).

```bash
pip install -e ".[doc]"                     # docs toolchain
python scripts/generate_parameters_doc.py    # regenerate the parameters reference
mkdocs serve                                 # live preview at http://127.0.0.1:8000
mkdocs build --strict                        # fail on any broken nav entry or link
```

Run `mkdocs build --strict` before committing documentation changes — it catches
broken navigation entries and dead internal links.

The published site is deployed automatically by the
`.github/workflows/deploy_docs.yml` GitHub Action: pushes to `main` publish the
`latest` version and pushes to `develop` publish the `develop` version, both
selectable from the version switcher on the site.

For the documentation/docstring standard that keeps the API reference complete,
see [`CONTRIBUTING.md`](https://github.com/hugomoison/pyKMC/blob/develop/CONTRIBUTING.md).
