"""ReferenceEventTable stores directional nu0 and resolves k via the htst backend."""

import math
from typing import Any

import numpy as np
from pytest import MonkeyPatch

from pykmc.event_table import ReferenceEventTable


def test_reference_table_has_nu0_and_k_prefactor_columns(
    config_Ni_4000at_monovacancy_sia: Any,
) -> None:
    """Fresh reference table carries both the nu0 diagnostic and k_prefactor columns."""
    table = ReferenceEventTable(config_Ni_4000at_monovacancy_sia)
    assert "nu0" in table.table.columns
    assert "k_prefactor" in table.table.columns


def test_build_event_series_stores_directional_nu0(
    config_Ni_4000at_monovacancy_sia: Any,
    system_single_type_fcc: Any,
    monkeypatch: MonkeyPatch,
) -> None:
    """Forward row keeps nu0_forward, backward row keeps nu0_backward; k uses them.

    Notes
    -----
    ``_build_event_series`` internally builds a ``NeighborsList`` and calls
    ``np.where(neighbor_list == index_move)`` on the result.  The production
    path receives numpy arrays from LAMMPS; the unit-test path gets plain
    Python lists from ``NeighborsList``, which triggers a numpy 0d-array error
    (``list == scalar`` is a scalar bool).  We monkeypatch
    ``pykmc.event_table.NeighborsList`` so that each per-atom list is
    returned as a numpy array, keeping the test focused on nu0/k bookkeeping
    without altering production code.

    """
    import pykmc.event_table as _et
    from pykmc.neighbors_list import NeighborsList as _RealNL

    class _NumpyNL(_RealNL):
        """Thin subclass that converts per-atom lists to numpy int64 arrays."""

        def __init__(self, *args: Any, **kwargs: Any) -> None:
            super().__init__(*args, **kwargs)
            for key in ("rnei", "rcut"):
                if key in self.neighbors_list:
                    self.neighbors_list[key] = [
                        np.array(lst, dtype=np.int64)
                        for lst in self.neighbors_list[key]
                    ]

    monkeypatch.setattr(_et, "NeighborsList", _NumpyNL)

    config = config_Ni_4000at_monovacancy_sia
    config.rateconstant.style = "htst"
    table = ReferenceEventTable(config)
    sys_ = system_single_type_fcc
    positions = sys_.positions
    cell = sys_.cell

    # A trivial "event" where min1 == saddle == min2 is enough to exercise the
    # series builder's nu0/k bookkeeping (we are not testing the physics here).
    fwd, bwd = table._build_event_series(
        min1_positions=positions,
        saddle_positions=positions,
        min2_positions=positions,
        index_move=0,
        dE_forward=0.5,
        dE_backward=0.7,
        cell=cell,
        nu0_forward=5.0e12,
        nu0_backward=3.0e12,
    )
    assert fwd["nu0"] == 5.0e12
    assert bwd["nu0"] == 3.0e12
    # Load-bearing wiring: the nu0 kwarg reaches the htst backend, so the resolved
    # prefactor is nu0 (NOT the k0 fallback). This single assertion guards the
    # whole per-event nu0 path (active events freeze this prefactor).
    assert fwd["k_prefactor"] == 5.0e12
    assert bwd["k_prefactor"] == 3.0e12
    assert math.isclose(fwd["k"], table.rate_constant.compute_rate(0.5, nu0=5.0e12).rate)
    assert math.isclose(bwd["k"], table.rate_constant.compute_rate(0.7, nu0=3.0e12).rate)
