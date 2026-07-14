"""Guard against a Hz-scale ``k0`` reaching the ps^-1 rate layer.

``RateConstantConfig.k0`` is the per-event fallback *prefactor in ps^-1*
(config default ``1.0``; 1 THz = 1.0 ps^-1). Every shipped input instead sets
``k0 = 1e12`` -- an attempt frequency in *Hz* -- which is 1e12x too large under
the ps^-1 contract.

For the ``htst``/``rpa`` backends this is catastrophic: an event whose Vineyard
``nu0`` succeeds gets a real ~1e1 ps^-1 prefactor (``nu0`` Hz converted to
ps^-1), but an event whose ``nu0`` computation fails falls back to this raw
``k0``. Mixing a correct ~19 ps^-1 with a bogus 1e12 ps^-1 makes the failed
events ~1e12x too fast, so they dominate BKL selection and freeze the KMC clock.
Observed on the NiCr 3-vac AV+HTST run: k ~ 1e4 ps^-1 at Ea ~ 0.8 eV,
dt ~ 1e-16 s, total time frozen at 6.97e-7 s for 47 consecutive steps.

The config must reject such a ``k0`` at load time with a clear message, so the
mistake can never silently reach the rate layer again.
"""

import pytest
from pydantic import ValidationError

from pykmc.config import RateConstantConfig


def test_htst_rejects_hz_scale_k0() -> None:
    """A Hz-scale k0 (1e12) is unphysical as a ps^-1 prefactor -> must be rejected."""
    with pytest.raises(ValidationError):
        RateConstantConfig(style="htst", k0=1e12, T=500.0)


def test_rpa_rejects_hz_scale_k0() -> None:
    """The rpa backend shares the same k0 fallback path and guard."""
    with pytest.raises(ValidationError):
        RateConstantConfig(style="rpa", k0=1e12, T=500.0)


def test_physical_k0_is_accepted() -> None:
    """A sane ps^-1 fallback prefactor (~1-100 THz) loads without error."""
    assert RateConstantConfig(style="htst", k0=1.0, T=500.0).k0 == 1.0
    # a few-tens-of-THz attempt frequency is also physical
    RateConstantConfig(style="htst", k0=50.0, T=500.0)
    # the config default (1.0 ps^-1) must remain valid
    RateConstantConfig(style="htst", T=500.0)
