import importlib
import pkgutil


def autodiscover(package_name: str, package_path) -> dict[str, ImportError]:
    """Import every submodule so components register themselves.

    Modules that fail to import are skipped silently. The returned mapping
    lets callers surface a precise error only when the unavailable component
    is explicitly requested.

    Returns
    -------
    dict[str, ImportError]
        Module names that could not be imported, keyed by module name.
    """
    failed: dict[str, ImportError] = {}
    for _, module_name, _ in pkgutil.iter_modules(package_path):
        try:
            importlib.import_module(f"{package_name}.{module_name}")
        except ImportError as e:
            failed[module_name] = e
    return failed
