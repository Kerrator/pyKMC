"""Harmonic Transition State Theory (HTST) Vineyard prefactor plugin.

Pure-NumPy mode analysis is vendored from
``apps/PyKMC_Analysis/Analysis/htst/kappa_rpa.py`` (see module headers for
provenance). The engine binding is the :class:`HtstLammpsExtension`
``EngineExtension`` in :mod:`pykmc.htst.lammps_extension`; the per-event
orchestrator lives in :mod:`pykmc.rate_constant.prefactor`.
"""

from .free_region import select_free_indices
from .hessian import mass_weighted_partial_hessian
from .normal_modes import normal_modes_from_hessian
from .vineyard import vineyard_prefactor

__all__ = [
    "select_free_indices",
    "mass_weighted_partial_hessian",
    "normal_modes_from_hessian",
    "vineyard_prefactor",
]
