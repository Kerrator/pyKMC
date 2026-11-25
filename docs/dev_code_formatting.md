# Developer Guide

## Code Formatting

pyKMC uses [Ruff](https://docs.astral.sh/ruff/) for Python formatting. Formatting is enforced by CI on every push and pull request — the check runs `ruff format --check` and fails if any file is not formatted.

### Installation

```bash
pip install ruff
# or, if using uv:
uv sync --extra dev
```

### Formatting

```bash
# Format all files
ruff format .

# Check without modifying (same as CI)
ruff format --check .
```

Configuration lives in `ruff.toml` at the repo root. Key settings: line length 88, Python 3.10 target, double quotes.

### Editor Setup

=== "VSCode"

    Install the [Ruff extension](https://marketplace.visualstudio.com/items?itemName=charliermarsh.ruff) (`charliermarsh.ruff`).

    The repo includes `.vscode/settings.json` which sets Ruff as the default Python formatter and enables format-on-save automatically.

=== "PyCharm / JetBrains"

    Install the [Ruff plugin](https://plugins.jetbrains.com/plugin/20574-ruff) from the JetBrains Marketplace.

    Then enable format-on-save: **Settings → Tools → Ruff → Format on save**.

    The plugin automatically picks up `ruff.toml` from the project root.

=== "Neovim / other editors"

    Any editor with an LSP client can use `ruff server`. See the [Ruff editor integrations docs](https://docs.astral.sh/ruff/editors/) for setup instructions.

## Linting

CI also runs `ruff check`, in two parts.

The bug-class rules (`E4`, `E7`, `E9`, `F`, `B`, `Q`) must report nothing:

```bash
ruff check --select E4,E7,E9,F,B,Q
```

The annotation (`ANN`) and docstring (`D`) rules are ratcheted. `scripts/lint_baseline.json` records how many findings each file has for each rule. CI fails if a count grows or if a new file has any finding, so new code must be annotated and documented while the existing findings are fixed over time.

```bash
# Same check as CI
python scripts/lint_ratchet.py

# After fixing findings, or after moving or renaming a file
python scripts/lint_ratchet.py --update
```

`ANN` and `D` apply to `pykmc/` only: `tests/`, `benchmarks/`, `scripts/` and `utils/` are exempt in `ruff.toml`. The lint job pins the Ruff version, because counts can change between Ruff releases. When you raise the pin in `.github/workflows/format.yml`, run `--update` with the same version.
