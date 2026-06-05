import importlib
import pkgutil

from .base import PrefactorBackend

# Auto-import every module in this package so subclasses register themselves
for _, _module_name, _ in pkgutil.iter_modules(__path__):
    importlib.import_module(f"{__name__}.{_module_name}")

__all__ = ["PrefactorBackend"]