"""Unit oracles for ``pykmc.htst.constants`` against independently restated CODATA."""

from __future__ import annotations

import math

import pytest

from pykmc.config import PhysicalConstants
from pykmc.htst import constants as c

# CODATA 2018 primaries restated independently of the module under test.
EV_J = 1.602176634e-19
AMU_KG = 1.66053906660e-27
HBAR_EV_S = 6.582119569e-16
H_EV_S = 4.135667696e-15
AVOGADRO = 6.02214076e23


def test_primaries_match_codata_2018() -> None:
    """The module's primaries are the CODATA 2018 values, not rounded copies."""
    assert c.EV_J == EV_J
    assert c.AMU_KG == AMU_KG
    assert c.HBAR_EV_S == HBAR_EV_S
    assert c.H_EV_S == H_EV_S
    assert c.HBAR_J_S == 1.054571817e-34
    assert c.ANGSTROM_M == 1.0e-10


def test_h_and_hbar_are_independently_rounded() -> None:
    """``h = 2 pi hbar`` holds only to ~1.5e-10; oracles must compare at 1e-9."""
    rel = c.H_EV_S / (2.0 * math.pi * c.HBAR_EV_S) - 1.0
    assert abs(rel) < 1.0e-9
    assert abs(rel) > 1.0e-11  # documents why bit-exact comparisons are wrong


def test_hbar_omega_ev_is_derived_from_primaries() -> None:
    """``HBAR_OMEGA_EV`` equals ``hbar * sqrt(eV / (amu Å^2))`` from the primaries."""
    expected = HBAR_EV_S * math.sqrt(EV_J / (AMU_KG * 1.0e-20))
    assert c.HBAR_OMEGA_EV == pytest.approx(expected, rel=1.0e-12)
    assert 0.0646 < c.HBAR_OMEGA_EV < 0.0647


def test_planck_matches_base_physical_constants_in_ev_ps() -> None:
    """``h_eV_s * 1e12`` agrees with the base ``PhysicalConstants.h`` (eV ps) to 1e-6."""
    assert c.H_EV_S * 1.0e12 == pytest.approx(PhysicalConstants.h, rel=1.0e-6)


def test_eigval_to_hz_is_linear_frequency_in_si() -> None:
    """``nu = sqrt(lambda eV/(amu Å^2)) / (2 pi)`` in Hz, no stray 2 pi or THz factor."""
    lam = 0.37
    omega_si = math.sqrt(lam * EV_J / (AMU_KG * 1.0e-20))
    expected_hz = omega_si / (2.0 * math.pi)
    assert c.eigval_to_hz(lam) == pytest.approx(expected_hz, rel=1.0e-9)
    # Typical metal eigenvalues (~0.1-1 eV/(amu Å^2)) land in the THz band.
    assert 1.0e12 < c.eigval_to_hz(lam) < 1.0e14


def test_eigval_to_hbar_omega_ev_round_trip() -> None:
    """``hbar_omega_ev_to_hz(eigval_to_hbar_omega_ev(l)) == eigval_to_hz(l)``."""
    lam = 0.25
    e = c.eigval_to_hbar_omega_ev(lam)
    assert e == pytest.approx(c.HBAR_OMEGA_EV * 0.5, rel=1.0e-14)
    assert c.hbar_omega_ev_to_hz(e) == pytest.approx(c.eigval_to_hz(lam), rel=1.0e-14)
    assert c.hbar_omega_ev_to_hz(e) == pytest.approx(e / H_EV_S, rel=1.0e-14)


@pytest.mark.parametrize("bad", [-1.0, float("nan"), float("inf")])
def test_eigval_to_hbar_omega_ev_rejects_invalid(bad: float) -> None:
    """Negative or non-finite eigenvalues raise ``ValueError``."""
    with pytest.raises(ValueError):
        c.eigval_to_hbar_omega_ev(bad)


def test_eskm_metal_conversion_is_lammps_literal() -> None:
    """9648.5 is LAMMPS's rounded eV -> 10 J/mol factor, within 1e-4 of the exact value."""
    exact = EV_J * AVOGADRO / 10.0
    assert c.ESKM_METAL_CONVERSION == 9648.5
    assert c.ESKM_METAL_CONVERSION == pytest.approx(exact, rel=1.0e-4)
