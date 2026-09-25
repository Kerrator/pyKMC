"""Regression: style=constant rate equals k0 * the Eyring exponential.

Key safety guarantee for the HTST work: adopting the pluggable-backend
dispatcher must not change any existing constant-style run.
"""

import math
from typing import Any

from pykmc.config import PhysicalConstants
from pykmc.rate_constant import create_rate_constant


def test_constant_rate_is_k0_times_eyring(
    config_Ni_4000at_monovacancy_sia: Any,
) -> None:
    """Constant style: rate == k0 * exp(-dE/(kb T)) and prefactor == k0."""
    cfg = config_Ni_4000at_monovacancy_sia  # style=constant from input.in
    rc = create_rate_constant(
        T=cfg.rateconstant.T,
        prefactor_backend_name=cfg.rateconstant.style,
        config=cfg.rateconstant,
    )
    kb = PhysicalConstants().kb
    for d_e in (0.1, 0.5, 1.2):
        out = rc.compute_rate(d_e)
        assert out.prefactor == cfg.rateconstant.k0
        assert math.isclose(
            out.rate, cfg.rateconstant.k0 * math.exp(-d_e / (kb * cfg.rateconstant.T))
        )
