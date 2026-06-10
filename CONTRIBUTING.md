# Contributing to pyKMC

Thanks for your interest in pyKMC. Bug reports, focused pull requests, and
documentation improvements are all welcome.

This guide also defines the **documentation standard** for the project. The API
reference on the [documentation site](https://hugomoison.github.io/pyKMC/) is
generated automatically from the code's docstrings, so following the standard
below keeps the published docs complete and correct with no extra manual work.

## Development setup

```bash
python3 -m venv pykmc_env
source pykmc_env/bin/activate
pip install -e ".[dev]"   # ruff, mkdocs, mkdocstrings, mike, ...
pip install pytest         # not in the project dependencies; install separately
```

To actually run simulations you also need LAMMPS, and optionally pARTn and IRA —
see the [installation guide](docs/user_guide/install/installation.md).

## Running the tests

```bash
pytest                                   # whole suite
pytest tests/test_system.py::test_fn -v  # a single test
pytest -k "vacancy"                      # pattern match
```

## Linting and type checking

pyKMC enforces strict linting and typing via [ruff](https://docs.astral.sh/ruff/)
and [mypy](https://mypy-lang.org/). Run all three before opening a PR:

```bash
ruff check . --fix    # lint + simple bug-finding (auto-fix)
ruff format .         # auto-format (double quotes, 88-col)
mypy pykmc/           # strict type checking
```

The ruff configuration (`ruff.toml`) selects the `D` (pydocstyle) and `ANN`
(flake8-annotations) rule sets, so **docstrings and full type annotations are
required** on public code, targeting Python 3.10.

> **Note:** there is currently no CI workflow that gates these checks on pull
> requests. Please run them locally — the PR checklist below is the enforcement
> mechanism until a CI lint job is added.

## Docstring standard

Every public module, class, and function must carry a **NumPy-style** docstring.
This matches the `docstring_style: numpy` setting in `mkdocs.yml` and is what the
API reference is rendered from.

Example (from `pykmc/bias.py`):

```python
def _moving_atom_displacement(
    event: pd.Series,
    system: System,
    reference_table: ReferenceEventTable,
) -> np.ndarray:
    """Return the displacement of the moving atom for a candidate event.

    Parameters
    ----------
    event : pd.Series
        One row of the active event table.
    system : System
        Current atomic configuration.
    reference_table : ReferenceEventTable
        Reference table used to look up ``move_atom_idx``.

    Returns
    -------
    np.ndarray, shape (3,)
        Displacement vector (final − initial) of the moving atom.
    """
```

Guidelines:

- A one-line summary, then `Parameters`, `Returns`, and (where relevant)
  `Raises` and `Examples` sections.
- Use double backticks for inline code/identifiers in the NumPy convention.
- `__init__` docstrings are not required (ruff `D107` is ignored) — document the
  class instead.

## How the API reference auto-populates

The docs site uses [mkdocstrings](https://mkdocstrings.github.io/): each page
under `docs/api/` is a three-line stub that pulls a module's rendered docstrings
into the site. As long as your code has good docstrings, the API page is
generated for you — no hand-written API prose.

### Rule: when you add a new public module

If you add a new module or subpackage to `pykmc/`, you **must** also:

1. Create `docs/api/<module>.md` with this exact pattern:

   ```markdown
   # `pykmc.<module>` Module

   ::: pykmc.<module>
       options:
           show_source: true
   ```

2. Add it to the `API Reference` section of the `nav:` in `mkdocs.yml`.

A PR that adds a public module without its API page and nav entry will be asked
to add them. Run `mkdocs build` to confirm there are no missing pages or broken
links, and `mkdocs build --strict` to confirm the new module's docstrings render
without warnings.

## Configuration fields document themselves

The [KMC Parameters](docs/parameters.md) page is generated from the pydantic
`Config` model in `pykmc/config.py` by `scripts/generate_parameters_doc.py`
(into `docs/parameters_details.md`, which is included into `parameters.md`).

- When you add or change a `Config` field, give it a clear description in the
  model; it will appear on the parameters page automatically.
- **Do not hand-edit `docs/parameters_details.md`** — it is generated and will be
  overwritten. Regenerate it with:

  ```bash
  python scripts/generate_parameters_doc.py
  ```

## Building and previewing the docs

```bash
pip install -e ".[doc]"
python scripts/generate_parameters_doc.py   # refresh the parameters reference
mkdocs serve                                 # live preview at http://127.0.0.1:8000
mkdocs build                                 # build the site (what the deploy workflow runs)
mkdocs build --strict                        # also fail on warnings (broken links, docstring issues)
```

`mkdocs build --strict` also flags malformed docstrings (e.g. a documented
parameter that is not in the signature, or wrong continuation-line indentation).
The codebase currently has a few **pre-existing** docstring warnings — please do
not add new ones: any module you add or change must build warning-free under
`--strict`.

The site is versioned with [mike](https://github.com/jimporter/mike) and
published by `.github/workflows/deploy_docs.yml`: pushes to `main` publish the
`latest` version, pushes to `develop` publish the `develop` version. Contributors
do not normally run `mike` — the GitHub Action does the deploy.

## Pull request checklist

Before opening a PR, confirm:

- [ ] `ruff check .` and `ruff format .` pass
- [ ] `mypy pykmc/` passes
- [ ] `pytest` passes
- [ ] New public functions/classes/modules have **NumPy-style docstrings**
- [ ] New public module → added `docs/api/<module>.md` **and** a `mkdocs.yml` nav entry
- [ ] New/changed `Config` fields verified via `python scripts/generate_parameters_doc.py`
- [ ] `mkdocs build` succeeds, and `mkdocs build --strict` raises no new warnings for files you touched
- [ ] User-facing behavior changes are reflected in the relevant `docs/` page

## Commit style

- Imperative subject line ("add", "fix", "rename"), under 72 characters.
- Wrap the body at 72 columns.
- Keep changes focused; unrelated cleanups belong in their own PR.

## License

By contributing, you agree that your contributions will be licensed under the
project's MIT License.
