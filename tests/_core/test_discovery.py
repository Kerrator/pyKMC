"""autodiscover (pykmc._core.discovery): failed submodules are recorded, not fatal."""

from __future__ import annotations

import importlib
import sys
from abc import abstractmethod
from collections.abc import Iterator
from pathlib import Path

import pytest

from pykmc._core import Registrable, autodiscover

PKG = "autodiscover_probe_pkg"

MODULES = {
    "a_native_broken": "raise OSError('dlopen failed: libprobe.so not found')\n",
    "b_runtime_broken": "raise RuntimeError('backend refused to initialise')\n",
    "c_missing_dep": "import module_that_does_not_exist_for_autodiscover\n",
    "z_good": "LOADED = True\n",
}


@pytest.fixture()
def probe_package(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Iterator[Path]:
    """Create a throwaway package on sys.path and drop its modules afterwards."""
    pkg = tmp_path / PKG
    pkg.mkdir()
    (pkg / "__init__.py").write_text("")
    for name, body in MODULES.items():
        (pkg / f"{name}.py").write_text(body)
    monkeypatch.syspath_prepend(str(tmp_path))
    importlib.invalidate_caches()
    try:
        yield pkg
    finally:
        for name in [m for m in sys.modules if m == PKG or m.startswith(PKG + ".")]:
            del sys.modules[name]


def _discover(pkg: Path) -> dict[str, ImportError]:
    return autodiscover(PKG, [str(pkg)])


def test_good_modules_import_after_a_broken_one(probe_package: Path) -> None:
    """Assert a failing module earlier in the scan does not stop later imports."""
    failed = _discover(probe_package)
    assert f"{PKG}.z_good" in sys.modules
    assert sys.modules[f"{PKG}.z_good"].LOADED is True
    assert "z_good" not in failed


def test_import_error_recorded_as_is(probe_package: Path) -> None:
    """Assert a genuine ImportError is stored unwrapped."""
    failed = _discover(probe_package)
    err = failed["c_missing_dep"]
    assert isinstance(err, ModuleNotFoundError)
    assert err.__cause__ is None
    assert "module_that_does_not_exist_for_autodiscover" in str(err)


@pytest.mark.parametrize(
    "module,exc_type,text",
    [
        ("a_native_broken", OSError, "dlopen failed"),
        ("b_runtime_broken", RuntimeError, "backend refused"),
    ],
)
def test_non_import_errors_wrapped_with_cause(
    probe_package: Path, module: str, exc_type: type[BaseException], text: str
) -> None:
    """Assert other exceptions become ImportError with the original as __cause__."""
    failed = _discover(probe_package)
    err = failed[module]
    assert type(err) is ImportError
    assert isinstance(err.__cause__, exc_type)
    assert text in str(err.__cause__)
    assert err.name == f"{PKG}.{module}"
    assert exc_type.__name__ in str(err)
    assert text in str(err)


def test_only_failed_modules_are_recorded(probe_package: Path) -> None:
    """Assert the mapping holds exactly the three broken modules."""
    failed = _discover(probe_package)
    assert set(failed) == {"a_native_broken", "b_runtime_broken", "c_missing_dep"}
    assert all(isinstance(e, ImportError) for e in failed.values())


def test_base_exceptions_are_not_swallowed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Assert SystemExit from a submodule still propagates."""
    name = "autodiscover_exit_probe_pkg"
    pkg = tmp_path / name
    pkg.mkdir()
    (pkg / "__init__.py").write_text("")
    (pkg / "quit.py").write_text("raise SystemExit(3)\n")
    monkeypatch.syspath_prepend(str(tmp_path))
    importlib.invalidate_caches()
    try:
        with pytest.raises(SystemExit):
            autodiscover(name, [str(pkg)])
    finally:
        for key in [m for m in sys.modules if m == name or m.startswith(name + ".")]:
            del sys.modules[key]


def test_registrable_create_surfaces_wrapped_error(probe_package: Path) -> None:
    """Assert a wrapped error reaches create() with its native cause intact."""

    class ProbeRoot(Registrable, root=True):
        @abstractmethod
        def run(self) -> int: ...

    ProbeRoot._import_errors = _discover(probe_package)
    with pytest.raises(ImportError, match="a_native_broken") as excinfo:
        ProbeRoot.create("a_native_broken")
    wrapped = excinfo.value.__cause__
    assert isinstance(wrapped, ImportError)
    assert isinstance(wrapped.__cause__, OSError)
