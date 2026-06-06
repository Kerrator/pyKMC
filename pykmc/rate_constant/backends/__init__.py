"""Prefactor backend implementations for rate constant computation.

Exposes [`PrefactorBackend`][pykmc.rate_constant.backends.PrefactorBackend],
the abstract base class all backends must inherit from.

All modules in this package are imported automatically at load time via
``pkgutil``, so any backend defined in a new file here is discovered without
modifying existing code.
"""
import importlib
import pkgutil

from .base import PrefactorBackend

# Auto-import every module in this package so subclasses register themselves
for _, _module_name, _ in pkgutil.iter_modules(__path__):
    importlib.import_module(f"{__name__}.{_module_name}")

__all__ = ["PrefactorBackend"]