"""Prove ``pykmc.htst`` depends only on NumPy and the standard library at import."""

from __future__ import annotations

import ast
import pathlib
import subprocess
import sys

import pykmc.htst

PACKAGE_DIR = pathlib.Path(pykmc.htst.__file__).resolve().parent
HEAVY = (
    "pandas",
    "mpi4py",
    "lammps",
    "ase",
    "pykmc.config",
    "pykmc.manager",
    "pykmc.engine",
)


def _module_level_imports(path: pathlib.Path) -> set[tuple[str, int]]:
    """Return ``(module, level)`` for every import statement at module scope."""
    tree = ast.parse(path.read_text(encoding="utf-8"))
    found: set[tuple[str, int]] = set()
    for node in tree.body:
        if isinstance(node, ast.Import):
            for alias in node.names:
                found.add((alias.name, 0))
        elif isinstance(node, ast.ImportFrom):
            found.add((node.module or "", node.level))
    return found


def test_only_numpy_and_stdlib_imported_at_module_level() -> None:
    """An AST scan of every module in the package finds numpy, stdlib and relative imports."""
    stdlib = set(sys.stdlib_module_names)
    offenders: list[str] = []
    for path in sorted(PACKAGE_DIR.glob("*.py")):
        for module, level in _module_level_imports(path):
            if level > 0:
                continue  # relative import inside pykmc.htst
            top = module.split(".")[0]
            if top == "numpy" or top in stdlib:
                continue
            offenders.append(f"{path.name}: {module}")
    assert offenders == []


def test_ase_only_inside_the_optional_request_helper() -> None:
    """``ase`` appears only inside ``default_masses_for_species``, never at module scope."""
    tree = ast.parse((PACKAGE_DIR / "request.py").read_text(encoding="utf-8"))
    ase_sites = []
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom) and (node.module or "").startswith("ase"):
            ase_sites.append(node.lineno)
    helper = next(
        n
        for n in tree.body
        if isinstance(n, ast.FunctionDef) and n.name == "default_masses_for_species"
    )
    assert ase_sites, "the lazy ASE import moved or vanished"
    assert all(helper.lineno < line <= helper.end_lineno for line in ase_sites)


def test_htst_modules_load_without_heavy_dependencies_in_isolation() -> None:
    """With a stub ``pykmc`` package, importing the subpackage loads no heavy module."""
    code = (
        "import sys, types\n"
        "pkg = types.ModuleType('pykmc'); pkg.__path__ = [sys.argv[1]]\n"
        "sys.modules['pykmc'] = pkg\n"
        "import pykmc.htst\n"
        "heavy = [m for m in sys.argv[2:] if m in sys.modules]\n"
        "print('loaded=' + ','.join(heavy))\n"
    )
    proc = subprocess.run(
        [sys.executable, "-c", code, str(PACKAGE_DIR.parent), *HEAVY],
        capture_output=True,
        text=True,
        check=False,
    )
    assert proc.returncode == 0, proc.stderr
    assert "loaded=" in proc.stdout
    assert proc.stdout.strip().endswith("loaded=")


def test_report_heavy_modules_loaded_by_real_import() -> None:
    """Observe (report only) which heavy modules ``import pykmc.htst`` pulls via ``pykmc``."""
    code = (
        "import sys\n"
        "import pykmc.htst\n"
        "print('loaded=' + ','.join(m for m in sys.argv[1:] if m in sys.modules))\n"
    )
    proc = subprocess.run(
        [sys.executable, "-c", code, *HEAVY],
        capture_output=True,
        text=True,
        check=False,
        cwd=str(PACKAGE_DIR.parent.parent),
    )
    assert proc.returncode == 0, proc.stderr
    line = next(ln for ln in proc.stdout.splitlines() if ln.startswith("loaded="))
    print(f"real 'import pykmc.htst' via pykmc/__init__.py: {line}")
