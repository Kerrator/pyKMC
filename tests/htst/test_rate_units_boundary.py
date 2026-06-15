"""Regression: the HTST prefactor must reach the KMC clock in ps^-1, not Hz.

``vineyard_prefactor`` produces the attempt frequency in **Hz**, but the rate
layer (``rate_from_prefactor`` / ``RateComponents``) and the KMC clock
(``kmc.py``: ``delta_t = -ln(u)/k_tot``; ``total_time += delta_t * 1e-12`` to get
seconds) both interpret the prefactor as **ps^-1**. Feeding a Hz value straight
through therefore makes every HTST rate 1e12 too large and the simulated time
1e12 too small.

These tests pin the unit boundary on a real sampled event (the committed
Ni(100) surface hop, dE ~ 0.597 eV): a known nu0 in Hz must yield the rate in
ps^-1 and the residence time in seconds that the physics demands, and an HTST
run must match a constant run at the same *physical* prefactor.
"""

import math
from pathlib import Path

import numpy as np

from pykmc.config import PhysicalConstants, RateConstantConfig
from pykmc.rate_constant import create_rate_constant

KB = PhysicalConstants().kb
_FIXTURE = Path(__file__).resolve().parent.parent / "data" / "htst_ni100_surface_hop.npz"

# 1 ps^-1 == 1 THz == 1e12 Hz, so the Hz -> ps^-1 conversion factor is 1e-12.
HZ_TO_PER_PS = 1e-12
# Mirror the ps -> s conversion applied to delta_t in pykmc/kmc.py.
PS_TO_S = 1e-12


def _dE() -> float:
    """Energy barrier (eV) of the committed Ni(100) surface-hop event."""
    return float(np.load(_FIXTURE)["energy_barrier"])


def _htst_rate_constant(T: float) -> object:
    cfg = RateConstantConfig(style="htst", k0=10.0, T=T)
    return create_rate_constant(T=T, prefactor_backend_name="htst", config=cfg)


def test_htst_rate_is_per_ps_not_hz() -> None:
    """A nu0 given in Hz must produce a rate in ps^-1 (Eyring on nu0*1e-12)."""
    T = 500.0
    dE = _dE()
    nu0_hz = 5.0e12  # a 5 THz mode
    rc = _htst_rate_constant(T)

    rate = rc.compute_rate(dE, nu0=nu0_hz).rate
    expected_per_ps = (nu0_hz * HZ_TO_PER_PS) * math.exp(-dE / (KB * T))

    assert math.isclose(rate, expected_per_ps, rel_tol=1e-9), (
        f"HTST rate is off by ~{rate / expected_per_ps:.3e}x: the Hz prefactor "
        "is being consumed as ps^-1 (expected a 1e-12 Hz->ps^-1 conversion)."
    )


def test_htst_residence_time_in_seconds_is_physical() -> None:
    """nu0 in Hz must yield the physically correct residence time in seconds.

    Drives the same arithmetic the KMC clock uses: for a single event
    ``k_tot = k``, the mean ``delta_t`` (ps) is ``1/k_tot``, and the elapsed
    seconds is ``delta_t * 1e-12`` (kmc.py). That must equal the inverse of the
    physical rate in s^-1, ``1 / (nu0_Hz * exp(-dE/kT))``.
    """
    T = 500.0
    dE = _dE()
    nu0_hz = 5.0e12
    rc = _htst_rate_constant(T)

    k_per_ps = rc.compute_rate(dE, nu0=nu0_hz).rate
    delta_t_ps = 1.0 / k_per_ps
    elapsed_s = delta_t_ps * PS_TO_S

    physical_rate_per_s = nu0_hz * math.exp(-dE / (KB * T))
    expected_residence_s = 1.0 / physical_rate_per_s

    assert math.isclose(elapsed_s, expected_residence_s, rel_tol=1e-9), (
        f"residence time is off by ~{elapsed_s / expected_residence_s:.3e}x: "
        "the Hz->ps^-1 boundary is unconverted, so simulated seconds are 1e12 too small."
    )


def test_constant_equals_htst_at_equal_physical_prefactor() -> None:
    """Same physical prefactor must give the same rate in both styles.

    A constant run with ``k0 = 5.0`` ps^-1 and an HTST run with
    ``nu0 = 5e12`` Hz describe the identical 5 THz attempt frequency, so their
    rates must agree.
    """
    T = 500.0
    dE = _dE()

    const_cfg = RateConstantConfig(style="constant", k0=5.0, T=T)
    const_rc = create_rate_constant(T=T, prefactor_backend_name="constant", config=const_cfg)
    const_rate = const_rc.compute_rate(dE).rate

    htst_rate = _htst_rate_constant(T).compute_rate(dE, nu0=5.0e12).rate

    assert math.isclose(const_rate, htst_rate, rel_tol=1e-9), (
        f"constant (k0=5.0 ps^-1) and HTST (nu0=5e12 Hz) disagree by "
        f"~{htst_rate / const_rate:.3e}x at the same physical 5 THz prefactor."
    )
