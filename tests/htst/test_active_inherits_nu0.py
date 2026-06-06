"""Active events rebuild k from the reference event's inherited k_prefactor.

The active rate uses the resolved ``k_prefactor`` (which already encodes the
nu0-or-k0 fallback decided at reference time), combined with the refined dE; the
raw ``nu0`` is carried through only as a diagnostic.
"""

import math
from typing import Any

import numpy as np

from pykmc.event_table import ActiveEventTable
from pykmc.rate_constant import rate_from_prefactor
from pykmc.result import EventRefinementOutput


def _refinement_output(
    k_prefactor: float | None, nu0: float | None
) -> EventRefinementOutput:
    return EventRefinementOutput(
        central_atom_index=0,
        saddle_positions=np.zeros((1, 3)),
        E_saddle=0.0,
        min2_positions=np.zeros((1, 3)),
        dE_forward=0.5,
        num_reference_event=0,
        k_prefactor=k_prefactor,
        nu0=nu0,
        refined="T",
    )


def test_active_event_k_uses_inherited_k_prefactor(
    config_Ni_4000at_monovacancy_sia: Any,
) -> None:
    """The active rate is rebuilt from the inherited k_prefactor and refined dE."""
    config = config_Ni_4000at_monovacancy_sia
    config.rateconstant.style = "htst"
    table = ActiveEventTable(config)
    series = table.build_event_series(_refinement_output(k_prefactor=5.0e12, nu0=5.0e12))
    assert math.isclose(series["k"], rate_from_prefactor(5.0e12, 0.5, config.rateconstant.T))
    assert series["nu0"] == 5.0e12  # diagnostic carried through refinement


def test_active_event_k_uses_k0_prefactor_on_fallback(
    config_Ni_4000at_monovacancy_sia: Any,
) -> None:
    """On fallback the reference k_prefactor was resolved to k0; the active reuses it."""
    config = config_Ni_4000at_monovacancy_sia  # style=constant from input.in
    k0 = config.rateconstant.k0
    table = ActiveEventTable(config)
    series = table.build_event_series(_refinement_output(k_prefactor=k0, nu0=None))
    assert math.isclose(series["k"], rate_from_prefactor(k0, 0.5, config.rateconstant.T))
