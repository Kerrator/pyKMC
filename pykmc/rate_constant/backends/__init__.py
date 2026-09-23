"""Prefactor backends for rate constant computation.

Every module in this package is imported by :func:`pykmc._core.autodiscover`
so that each :class:`PrefactorBackend` subclass registers itself under its
``name``. A module that fails to import is recorded in
``PrefactorBackend._import_errors`` and only surfaces when that backend is
requested through ``PrefactorBackend.create``; the other backends stay usable.
"""

from pykmc._core import autodiscover

from .base import PrefactorBackend

PrefactorBackend._import_errors = autodiscover(__name__, __path__)

__all__ = ["PrefactorBackend"]
