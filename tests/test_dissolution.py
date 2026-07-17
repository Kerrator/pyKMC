"""Contract tests for the dealloying dissolution event (feature: delete
under-coordinated less-noble atoms).

Physics (user-specified): when an atom of a configured dissolvable element
(Cr in NiCr, Fe in NiFe -- never the noble matrix) becomes sufficiently
under-coordinated (coordination <= coord_max, default 6: kinks and worse;
vacancy-ring atoms at 7 rearrange but do not dissolve), a synthetic
dissolution event competes in the BKL selection with rate

    k_diss(n) = nu_d * exp((phi - n * E_b) / (kb * T))  [Erlebacher bond-counting]

with n the atom's current first-shell (rnei) coordination, E_b the effective
bond energy (eV), phi the electrochemical driving force (eV, default 0 = pure
bond counting), nu_d the attempt frequency in ps^-1 (same unit contract as
k0/nu0 -- a Hz-scale value must be rejected, cf. the k0 clock-freeze bug).
Executing the event deletes the atom (dealloying: the surface recedes as
less-noble atoms dissolve).

These tests pin the config surface, the rate formula, and the eligibility
scan. BKL integration and deletion execution are covered by the live smoke
(run separately -- they need an engine).
"""

from pathlib import Path

import numpy as np
import pytest
from pydantic import ValidationError

from pykmc.config import Config, DissolutionConfig, PhysicalConstants

KB = PhysicalConstants().kb
_INPUT = Path(__file__).resolve().parent / "data" / "input.in"


def _base_config(**dissolution_fields) -> Config:
    cfg = Config.from_ini_file(str(_INPUT))
    cfg.control.dissolution = True
    cfg.dissolution = DissolutionConfig(**dissolution_fields)
    return cfg


# ---------------------------------------------------------------- config ----


def test_dissolution_disabled_by_default() -> None:
    """A config without [Dissolution] parses and has the feature off."""
    cfg = Config.from_ini_file(str(_INPUT))
    assert not cfg.control.dissolution


def test_dissolution_config_fields_and_defaults() -> None:
    """[Dissolution] carries elements, nu_d (ps^-1), E_b (eV), coord_max=6."""
    d = DissolutionConfig(elements="Cr", nu_d=10.0, E_b=0.15)
    assert d.elements == ["Cr"]
    assert d.nu_d == 10.0
    assert d.E_b == 0.15
    assert d.coord_max == 6  # user decision: kinks and worse dissolve


def test_dissolution_elements_parse_comma_list() -> None:
    """elements accepts a comma list (e.g. 'Cr,Fe')."""
    d = DissolutionConfig(elements="Cr,Fe", nu_d=10.0, E_b=0.15)
    assert d.elements == ["Cr", "Fe"]


def test_nu_d_rejects_hz_scale() -> None:
    """nu_d is in ps^-1; a Hz-scale value (1e13) must be rejected loudly.

    Same units contract and guard rationale as RateConstantConfig.k0: a Hz
    attempt frequency silently makes every dissolution ~1e12x too fast.
    """
    with pytest.raises(ValidationError):
        DissolutionConfig(elements="Cr", nu_d=1.0e13, E_b=0.15)


def test_e_b_must_be_positive() -> None:
    """A zero/negative bond energy is unphysical for bond-counting."""
    with pytest.raises(ValidationError):
        DissolutionConfig(elements="Cr", nu_d=10.0, E_b=0.0)


# ------------------------------------------------------------------ rate ----


def test_bond_counting_rate_formula() -> None:
    """k_diss = nu_d * exp(-n*E_b/kT), elementwise over coordinations."""
    from pykmc.dissolution import dissolution_rates

    nu_d, E_b, T = 10.0, 0.15, 500.0
    coords = np.array([3, 5, 6])
    k = dissolution_rates(coords, nu_d=nu_d, E_b=E_b, T=T)
    expected = nu_d * np.exp(-coords * E_b / (KB * T))
    assert np.allclose(k, expected, rtol=1e-12)
    # fewer bonds -> faster dissolution (monotone decreasing in n)
    assert k[0] > k[1] > k[2]


def test_overpotential_enters_exponent_additively() -> None:
    """A positive phi lowers the effective barrier: k(phi) = k(0) * exp(phi/kT).

    Canonical Erlebacher form (Nature 410, 450 (2001)):
    k_E,N = nu_E * exp(-(N*eps - phi)/kBT). phi=0 must reproduce the pure
    bond-counting rate exactly (backward compatibility).
    """
    from pykmc.dissolution import dissolution_rates

    nu_d, E_b, T, phi = 10.0, 0.15, 500.0, 0.45
    coords = np.array([3, 5, 6])
    k0 = dissolution_rates(coords, nu_d=nu_d, E_b=E_b, T=T)
    k_phi = dissolution_rates(coords, nu_d=nu_d, E_b=E_b, T=T, phi=phi)
    assert np.allclose(k_phi, k0 * np.exp(phi / (KB * T)), rtol=1e-12)
    # default phi=0.0 keeps the original values
    assert np.allclose(
        dissolution_rates(coords, nu_d=nu_d, E_b=E_b, T=T, phi=0.0), k0, rtol=0.0
    )


def test_phi_config_default_and_sign() -> None:
    """The phi field defaults to 0.0 (pure bond counting) and rejects < 0."""
    d = DissolutionConfig(elements="Cr", nu_d=10.0, E_b=0.15)
    assert d.phi == 0.0
    d2 = DissolutionConfig(elements="Cr", nu_d=10.0, E_b=0.15, phi=1.5)
    assert d2.phi == 1.5
    with pytest.raises(ValidationError):
        DissolutionConfig(elements="Cr", nu_d=10.0, E_b=0.15, phi=-0.1)


# ----------------------------------------------------------- eligibility ----


def test_eligibility_scan_element_and_coordination_gated() -> None:
    """Eligible = (element in dissolvable set) AND (coordination <= coord_max).

    A Cr kink atom (coord 6) is eligible; a Cr vacancy-ring atom (coord 7) is
    not; a Ni atom is never eligible regardless of coordination.
    """
    from pykmc.dissolution import eligible_atoms

    types = ["Ni", "Cr", "Cr", "Ni", "Cr"]
    coordination = np.array([5, 6, 7, 12, 4])
    idx = eligible_atoms(
        types=types, coordination=coordination, elements=["Cr"], coord_max=6
    )
    assert list(idx) == [1, 4]  # the coord-6 and coord-4 Cr only


def test_eligibility_empty_when_no_candidates() -> None:
    """A pristine surface (everything above coord_max) yields no events."""
    from pykmc.dissolution import eligible_atoms

    idx = eligible_atoms(
        types=["Cr", "Ni"], coordination=np.array([8, 12]),
        elements=["Cr"], coord_max=6,
    )
    assert len(idx) == 0
