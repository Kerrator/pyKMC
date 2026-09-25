"""ReferenceEventTable stores directional nu0 and resolves k via the htst backend."""

import math
from typing import Any

import numpy as np
from pytest import MonkeyPatch

from pykmc.event_table import ActiveEventTable, ReferenceEventTable


def test_constant_reference_table_has_no_nu0_column(
    config_Ni_4000at_monovacancy_sia: Any,
) -> None:
    """Gating: a constant-style table has k_prefactor but NOT nu0 (base schema)."""
    config = config_Ni_4000at_monovacancy_sia  # style=constant from input.in
    ref = ReferenceEventTable(config)
    act = ActiveEventTable(config)
    assert "k_prefactor" in ref.table.columns
    assert "nu0" not in ref.table.columns
    assert "nu0" not in act.table.columns


def test_htst_reference_table_has_nu0_and_k_prefactor_columns(
    config_Ni_4000at_monovacancy_sia: Any,
) -> None:
    """Gating: an htst-style table carries both nu0 and k_prefactor."""
    config = config_Ni_4000at_monovacancy_sia
    config.rateconstant.style = "htst"
    ref = ReferenceEventTable(config)
    act = ActiveEventTable(config)
    assert "nu0" in ref.table.columns
    assert "k_prefactor" in ref.table.columns
    assert "nu0" in act.table.columns


def test_build_event_series_is_k0_placeholder(
    config_Ni_4000at_monovacancy_sia: Any,
    system_single_type_fcc: Any,
    monkeypatch: MonkeyPatch,
) -> None:
    """The series builder writes k0-placeholder rates and a None nu0 column.

    Per-event nu0 is patched AFTERWARDS by the accepted-events backfill
    (see test_reference_backfill.py for the directional patch assertions).

    Notes
    -----
    ``_build_event_series`` internally builds a ``NeighborsList`` and calls
    ``np.where(neighbor_list == index_move)`` on the result.  The production
    path receives numpy arrays from LAMMPS; the unit-test path gets plain
    Python lists from ``NeighborsList``, which triggers a numpy 0d-array error
    (``list == scalar`` is a scalar bool).  We monkeypatch
    ``pykmc.event_table.NeighborsList`` so that each per-atom list is
    returned as a numpy array, keeping the test focused on the rate
    bookkeeping without altering production code.

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
    # series builder's rate bookkeeping (we are not testing the physics here).
    fwd, bwd = table._build_event_series(
        min1_positions=positions,
        saddle_positions=positions,
        min2_positions=positions,
        index_move=0,
        dE_forward=0.5,
        dE_backward=0.7,
        cell=cell,
    )
    k0 = config.rateconstant.k0
    assert fwd["nu0"] is None
    assert bwd["nu0"] is None
    # Placeholder semantics: no per-event nu0 yet -> htst backend resolves to k0.
    assert fwd["k_prefactor"] == k0
    assert bwd["k_prefactor"] == k0
    assert math.isclose(fwd["k"], table.rate_constant.compute_rate(0.5).rate)
    assert math.isclose(bwd["k"], table.rate_constant.compute_rate(0.7).rate)
