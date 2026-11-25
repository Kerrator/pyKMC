"""Static checks on the inputs shipped in ``examples/``.

No simulation is run. For every example the tests check that

- the shipped input loads,
- the parser does not silently ignore one of its sections or keys,
- the files it points to exist,
- ``examples/README.md`` mentions the example,
- the reference run in ``expected_outputs/`` was made with the shipped input.
"""

from __future__ import annotations

import configparser
from pathlib import Path

import pytest

from pykmc import Config

EXAMPLES = Path(__file__).resolve().parents[1] / "examples"
EXAMPLE_DIRS = sorted(p for p in EXAMPLES.glob("*") if p.is_dir())
REFERENCE_DIRS = [
    p for p in EXAMPLE_DIRS if (p / "expected_outputs" / "pykmc.log").is_file()
]

# A value no parameter accepts unchanged: a key that the parser reads either
# fails validation with it or ends up with a different value.
SENTINEL = "__not_a_value__"


def _input_file(example: Path) -> Path:
    """Return the tracked input of an example: the template, else ``input.in``."""
    template = example / "input.in.template"
    return template if template.is_file() else example / "input.in"


def _read_ini(path: Path) -> configparser.ConfigParser:
    """Read an input the way ``Config.from_ini_file`` does, keeping key case."""
    parser = configparser.ConfigParser()
    parser.optionxform = str
    parser.read(path)
    return parser


def _ignored_entries(path: Path, tmp_path: Path) -> list[str]:
    """List the sections and keys of an input that have no effect on the Config.

    Each key is set to ``SENTINEL`` in turn and the input is loaded again. If the
    load succeeds and the Config is unchanged, the parser never read that key.
    """
    reference = Config.from_ini_file(str(path))
    parser = _read_ini(path)
    ignored = []
    for section in parser.sections():
        if section.lower() not in Config.model_fields:
            ignored.append(f"[{section}]")
            continue
        for key in parser[section]:
            poked = _read_ini(path)
            poked[section][key] = SENTINEL
            poked_path = tmp_path / "poked.in"
            with poked_path.open("w") as handle:
                poked.write(handle)
            try:
                changed = Config.from_ini_file(str(poked_path)) != reference
            except ValueError:
                changed = True
            if not changed:
                ignored.append(f"[{section}] {key}")
    return ignored


def _is_file_token(token: str) -> bool:
    """Tell whether a ``pair_coeff`` token names a file rather than a number."""
    if "." not in token and "/" not in token:
        return False
    try:
        float(token)
    except ValueError:
        return True
    return False


def _logged_parameters(log: Path) -> dict[str, dict[str, str]]:
    """Read the ``SIMULATION PARAMETERS`` block that a run writes to its log."""
    lines = iter(log.read_text().splitlines())
    for line in lines:
        if line.strip() == "SIMULATION PARAMETERS":
            next(lines)  # closing separator
            break
    else:
        raise AssertionError(f"no SIMULATION PARAMETERS block in {log}")

    logged: dict[str, dict[str, str]] = {}
    section = ""
    for line in lines:
        if not line.strip():
            break
        if line.startswith(" "):
            key, _, value = line.partition(" : ")
            logged[section][key.strip()] = value
        else:
            section = line.strip()
            logged[section] = {}
    return logged


def _same_as_logged(value: object, logged: str) -> bool:
    """Compare a parameter with the text a run logged for it.

    A parameter that became a list after the run was made (``pair_coeff``)
    still matches the single value logged then.
    """
    if str(value) == logged:
        return True
    return isinstance(value, list) and len(value) == 1 and str(value[0]) == logged


@pytest.mark.parametrize("example", EXAMPLE_DIRS, ids=lambda p: p.name)
def test_example_has_one_input(example: Path) -> None:
    """Each example ships a template or an ``input.in``."""
    assert _input_file(example).is_file(), f"{example.name} has no input file"


@pytest.mark.parametrize("example", EXAMPLE_DIRS, ids=lambda p: p.name)
def test_example_input_loads(example: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """The shipped input passes ``Config.from_ini_file``."""
    monkeypatch.chdir(example)
    Config.from_ini_file(str(_input_file(example)))


@pytest.mark.parametrize("example", EXAMPLE_DIRS, ids=lambda p: p.name)
def test_example_input_has_no_ignored_key(
    example: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Every section and key of the shipped input is read by the parser."""
    monkeypatch.chdir(example)
    ignored = _ignored_entries(_input_file(example), tmp_path)
    assert not ignored, f"the parser ignores {ignored} in {example.name}"


@pytest.mark.parametrize("example", EXAMPLE_DIRS, ids=lambda p: p.name)
def test_example_files_exist(example: Path) -> None:
    """The initial configuration and the potential files are in the example."""
    parser = _read_ini(_input_file(example))
    wanted = [parser["Control"]["initial_config"].strip()]
    if parser.has_option("Lammps", "pair_coeff"):
        wanted += [
            token
            for token in parser["Lammps"]["pair_coeff"].split()
            if _is_file_token(token)
        ]
    missing = [name for name in wanted if not (example / name).is_file()]
    assert not missing, f"{example.name} points to missing files: {missing}"


@pytest.mark.parametrize("example", EXAMPLE_DIRS, ids=lambda p: p.name)
def test_example_is_in_readme(example: Path) -> None:
    """``examples/README.md`` mentions every example directory."""
    assert example.name in (EXAMPLES / "README.md").read_text()


@pytest.mark.parametrize("example", REFERENCE_DIRS, ids=lambda p: p.name)
def test_reference_run_matches_input(
    example: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The reference run in ``expected_outputs/`` used the shipped parameters.

    A parameter that the reference log does not list was added to pyKMC after
    the run: it is accepted only at its default value.
    """
    monkeypatch.chdir(example)
    config = Config.from_ini_file(str(_input_file(example)))
    logged = _logged_parameters(example / "expected_outputs" / "pykmc.log")

    stale = []
    for section, model in config:
        if model is None:
            continue
        for key, value in model:
            if key in logged.get(section, {}):
                if not _same_as_logged(value, logged[section][key]):
                    stale.append(
                        f"{section}.{key}: input {value}, "
                        f"reference run {logged[section][key]}"
                    )
            else:
                default = (
                    type(model).model_fields[key].get_default(call_default_factory=True)
                )
                if value != default:
                    stale.append(
                        f"{section}.{key}: input {value}, not in the reference log"
                    )
    assert not stale, (
        f"expected_outputs/ of {example.name} was not made with the shipped input: "
        f"{stale}"
    )
