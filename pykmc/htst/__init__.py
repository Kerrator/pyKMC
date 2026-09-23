"""Harmonic transition state theory (HTST) numerical layer.

Pure NumPy: free-region selection, finite-difference mass-weighted partial
Hessians, normal-mode classification, the Vineyard prefactor, and the per-event
orchestrator that returns typed directional results. This subpackage imports only
NumPy and the standard library at module level; engine binding, MPI dispatch and
the Hz -> ps⁻¹ rate conversion live elsewhere.

Note that ``import pykmc.htst`` executes the parent ``pykmc/__init__.py`` first,
which loads pandas, mpi4py and ase (and ``pykmc.config`` / ``pykmc.manager``), so
a real import is not install-independent. The isolation test proves the
NumPy-only property of the subpackage itself with a stub parent package.
"""

from .constants import (
    AMU_KG,
    ANGSTROM_M,
    ESKM_METAL_CONVERSION,
    EV_J,
    H_EV_S,
    HBAR_EV_S,
    HBAR_J_S,
    HBAR_OMEGA_EV,
    OMEGA_SI_PER_SQRT_EIGVAL,
    eigval_to_hbar_omega_ev,
    eigval_to_hz,
    hbar_omega_ev_to_hz,
)
from .free_region import select_free_indices
from .hessian import ForcesFn, HessianFn, fd_hessian_fn, mass_weighted_partial_hessian
from .normal_modes import ModeSpectrum, normal_modes_from_hessian
from .prefactor import N_ZERO_MODES_PARTIAL_HESSIAN, compute_event_prefactors
from .request import (
    HTSTEventRequest,
    HTSTGeometryError,
    HTSTRequestError,
    default_masses_for_species,
    require_orthorhombic,
)
from .result import (
    DirectionalPrefactor,
    EventPrefactors,
    PrefactorRejected,
    PrefactorRejection,
)
from .settings import FREE_REGION_CENTERS, HTSTSettings
from .vineyard import (
    VineyardEstimate,
    vineyard_from_spectra,
    vineyard_prefactor,
    vineyard_prefactor_detailed,
)

__all__ = [
    "AMU_KG",
    "ANGSTROM_M",
    "ESKM_METAL_CONVERSION",
    "EV_J",
    "FREE_REGION_CENTERS",
    "H_EV_S",
    "HBAR_EV_S",
    "HBAR_J_S",
    "HBAR_OMEGA_EV",
    "N_ZERO_MODES_PARTIAL_HESSIAN",
    "OMEGA_SI_PER_SQRT_EIGVAL",
    "DirectionalPrefactor",
    "EventPrefactors",
    "ForcesFn",
    "HTSTEventRequest",
    "HTSTGeometryError",
    "HTSTRequestError",
    "HTSTSettings",
    "HessianFn",
    "ModeSpectrum",
    "PrefactorRejected",
    "PrefactorRejection",
    "VineyardEstimate",
    "compute_event_prefactors",
    "default_masses_for_species",
    "eigval_to_hbar_omega_ev",
    "eigval_to_hz",
    "fd_hessian_fn",
    "hbar_omega_ev_to_hz",
    "mass_weighted_partial_hessian",
    "normal_modes_from_hessian",
    "require_orthorhombic",
    "select_free_indices",
    "vineyard_from_spectra",
    "vineyard_prefactor",
    "vineyard_prefactor_detailed",
]
