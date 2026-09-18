"""Basin rates under htst/rpa (architecture rule 5).

``refine_absorbing`` re-rates a transient -> absorbing transition through the
reference row's resolved prefactor at the refined barrier and stores a
scalar; the temporary active event built for the exit carries the reference
metadata; a tiny two-state basin with unequal prefactors is solved
analytically. No LAMMPS, no MPI.
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np
import pandas as pd
import pytest

from pykmc.basins import BasinsGenericEvents, BasinStatesConnectivity, FPTASelector
from pykmc.config import Config, RateConstantConfig
from pykmc.event_table import ActiveEventTable, ReferenceEventTable
from pykmc.rate_constant import compute_rate_Eyring, rate_from_prefactor
from pykmc.result import EventRefinementOutput
from tests.lifecycle.conftest import DATA_INPUT, accepted, rejected


def _config(style: str, k0: float = 1.0, T: float = 300.0) -> Config:
    config = Config.from_ini_file(DATA_INPUT)
    return config.model_copy(
        update={"rateconstant": RateConstantConfig(style=style, k0=k0, T=T)}
    )


def _reference(
    config: Config, system: Any, rows: dict[int, Any]
) -> ReferenceEventTable:
    """Build a reference table with one trivial row per id, patched with ``rows[id]``."""
    table = ReferenceEventTable(config)
    pos = system.positions
    for idx in rows:
        fwd, _ = table._build_event_series(
            min1_positions=pos,
            saddle_positions=pos,
            min2_positions=pos,
            index_move=0,
            dE_forward=0.5,
            dE_backward=0.5,
            cell=system.cell,
            types=list(system.types),
        )
        fwd["idx_ref"] = idx
        fwd["idx_backward"] = idx
        table.table = pd.concat([table.table, fwd.to_frame().T], ignore_index=True)
    if table.uses_prefactors:
        for idx, estimate in rows.items():
            table._patch_row(idx, estimate)
    return table


class TestAbsorbingRate:
    """``_absorbing_rate`` combines the refined barrier with the reference prefactor."""

    def test_uses_the_reference_prefactor_and_returns_a_scalar(
        self, system_single_type_fcc: Any
    ) -> None:
        """Accepted reference: nu0 -> ps^-1 prefactor; rejected: k0; float result."""
        config = _config("htst", k0=1.5, T=350.0)
        ref = _reference(
            config, system_single_type_fcc, {4: accepted(5.0e12), 9: rejected("x")}
        )
        basin = BasinsGenericEvents(config, ref, set(), manager=None)

        k_ok = basin._absorbing_rate(0.42, 4)
        k_rej = basin._absorbing_rate(0.42, 9)
        assert type(k_ok) is float and type(k_rej) is float
        assert k_ok == rate_from_prefactor(5.0, 0.42, 350.0)
        assert k_rej == rate_from_prefactor(1.5, 0.42, 350.0)
        assert k_ok != k_rej
        with pytest.raises(ValueError, match="not in the reference table"):
            basin._absorbing_rate(0.42, 77)

    def test_constant_mode_matches_the_base_expression(
        self, system_single_type_fcc: Any
    ) -> None:
        """Constant style: exactly compute_rate_Eyring(dE, config)."""
        config = _config("constant", k0=1e12, T=300.0)
        ref = _reference(config, system_single_type_fcc, {4: None})
        basin = BasinsGenericEvents(config, ref, set(), manager=None)
        assert basin.uses_prefactors is False
        assert basin._absorbing_rate(0.42, 4) == compute_rate_Eyring(0.42, config)


class TestTemporaryActiveEvent:
    """The exit event reconstructed in kmc.py carries the reference estimate."""

    def test_reference_estimate_seeds_the_temporary_row(
        self, system_single_type_fcc: Any
    ) -> None:
        """The temporary row inherits nu0, status and source from the reference."""
        config = _config("htst", k0=1.0, T=300.0)
        ref = _reference(config, system_single_type_fcc, {4: accepted(5.0e12)})
        tmp_table = ActiveEventTable(config)
        tmp_table.add_events(
            EventRefinementOutput(
                central_atom_index=0,
                saddle_positions=np.zeros((3, 3)),
                E_saddle=-1,
                min2_positions=np.zeros((3, 3)),
                dE_forward=0.3,
                num_reference_event=4,
                **ref.reference_estimate(4),
            )
        )
        row = tmp_table.table.iloc[0]
        assert row["nu0"] == 5.0e12
        assert row["k_prefactor"] == 5.0
        assert row["nu0_status"] == "ok" and row["nu0_source"] == "reference"
        assert row["k"] == rate_from_prefactor(5.0, 0.3, 300.0)
        assert bool(row["nu0_site_attempted"]) is False


class TestTwoStateBasin:
    """One transient state, two absorbing exits with unequal prefactors."""

    def _connectivity(self, k1: float, k2: float) -> BasinStatesConnectivity:
        table = BasinStatesConnectivity()
        table.add_connectivity(
            state=0,
            state_connexion=1,
            event_connexion=4,
            central_atom=0,
            sym=0,
            transient=False,
            dE_forward=0.3,
            k_forward=k1,
            dE_backward=0.3,
            k_backward=k1,
        )
        table.add_connectivity(
            state=0,
            state_connexion=2,
            event_connexion=9,
            central_atom=1,
            sym=0,
            transient=False,
            dE_forward=0.3,
            k_forward=k2,
            dE_backward=0.3,
            k_backward=k2,
        )
        return table

    @pytest.mark.parametrize("r2,expected_state", [(0.2, 1), (0.9, 2)])
    def test_exit_time_and_state_are_analytic(
        self, r2: float, expected_state: int, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """t_exit = -ln(1 - r1) / (k1 + k2); the exit state follows k_i / k_tot."""
        T = 300.0
        k1 = rate_from_prefactor(5.0, 0.3, T)  # nu0 = 5e12 Hz
        k2 = rate_from_prefactor(2.0, 0.3, T)  # nu0 = 2e12 Hz
        assert k1 != k2
        table = self._connectivity(k1, k2)
        assert table.df["k_forward"].dtype == float
        k_tot = float(
            table.df.loc[~table.df["transient"].astype(bool), "k_forward"].sum()
        )
        assert k_tot == pytest.approx(k1 + k2)

        draws = iter([0.5, r2])
        monkeypatch.setattr(np.random, "random", lambda *a, **k: next(draws))
        selector = FPTASelector()
        result = selector.select_from_connectivity(table)
        assert result.is_ok()

        assert selector.M_abs.dtype == float
        assert selector.M_abs[1, 0] == pytest.approx(-k1)
        assert selector.M_abs[2, 0] == pytest.approx(-k2)
        assert selector.M_abs[0, 0] == pytest.approx(k1 + k2)

        t_exit = result.ok_value().t_exit
        assert t_exit == pytest.approx(-math.log(1.0 - 0.5) / (k1 + k2), rel=5e-3)
        assert result.ok_value().exit_state == expected_state
        # the exit probability of state 1 is k1 / (k1 + k2) = 5/7
        assert k1 / (k1 + k2) == pytest.approx(5.0 / 7.0)
