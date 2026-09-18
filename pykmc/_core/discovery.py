"""Submodule discovery for the pluggable registries."""

from __future__ import annotations

import importlib
import pkgutil
from collections.abc import Iterable


def autodiscover(
    package_name: str, package_path: Iterable[str]
) -> dict[str, ImportError]:
    """Import every submodule so components register themselves.

    Modules that fail to import are skipped and recorded rather than breaking
    the import of the whole package: the returned mapping lets callers surface
    a precise error only when the unavailable component is explicitly
    requested. An ``ImportError`` is recorded as is. Any other exception raised
    while importing a submodule (for example an ``OSError`` from a native
    ``dlopen``) is wrapped in an ``ImportError`` whose ``__cause__`` is the
    original exception, so unrelated components keep working and the real
    failure stays reachable.

    Parameters
    ----------
    package_name : str
        Fully qualified name of the package whose submodules are imported.
    package_path : Iterable[str]
        The package ``__path__``.

    Returns
    -------
    dict[str, ImportError]
        Module names that could not be imported, keyed by module basename.

    """
    failed: dict[str, ImportError] = {}
    for _, module_name, _ in pkgutil.iter_modules(package_path):
        full_name = f"{package_name}.{module_name}"
        try:
            importlib.import_module(full_name)
        except ImportError as e:
            failed[module_name] = e
        except Exception as e:
            wrapped = ImportError(
                f"{full_name} failed to import ({type(e).__name__}: {e})",
                name=full_name,
            )
            wrapped.__cause__ = e
            failed[module_name] = wrapped
    return failed
