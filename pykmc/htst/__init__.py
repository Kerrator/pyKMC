"""Harmonic Transition State Theory (HTST) Vineyard prefactor subsystem.

Pure-NumPy mode analysis is vendored from
``apps/PyKMC_Analysis/Analysis/htst/kappa_rpa.py`` (see module headers for
provenance). Engine binding lives in
``pykmc.enginemanager.lmpi.lammps_operations``.
"""

from .free_region import select_free_indices
from .hessian import mass_weighted_partial_hessian
from .normal_modes import normal_modes_from_hessian
from .vineyard import vineyard_prefactor

# Note: the per-event orchestrator now lives in ``pykmc.rate_constant.prefactor``
# (EventPrefactors / compute_event_prefactors); this package is the numerics layer.
__all__ = [
    "select_free_indices",
    "mass_weighted_partial_hessian",
    "normal_modes_from_hessian",
    "vineyard_prefactor",
]
