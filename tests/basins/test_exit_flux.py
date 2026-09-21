"""Exit-channel selection at the sampled exit time (contracts 7f, R11 / F03).

Rates are in ps^-1. The reversible model 0->1=1, 1->0=2, 0->A=1, 1->B=1 entered
in state 0 has transient occupation q0=(2e^-t+e^-4t)/3, q1=(e^-t-e^-4t)/3,
survival e^-t and the conditional channel law P(A|T=t)=(2+e^-3t)/3: 17/24 at
t=ln2, while cumulative absorption up to ln2 would give 79/96. The frozen
19-case CTMC oracle keeps the statistical claim; these tests pin the law, the
transition identity and the numerical failure policy on the repository API.
"""

import math

import numpy as np
import pytest

from pykmc.basins import BasinStatesConnectivity, FPTASelector
from pykmc.basins.selection import (
    FLUX_ROUNDOFF_RTOL,
    BasinExitFluxError,
    draw_channel,
    normalized_weights,
    validate_flux,
)
from pykmc.result import ErrorType


def connectivity(edges, labels=None):
    """Rows (source, destination, rate); ``transient`` follows the destination."""
    table = BasinStatesConnectivity()
    sources = {source for source, _, _ in edges}
    for index, (source, destination, rate) in enumerate(edges):
        table.add_connectivity(
            state=source,
            state_connexion=destination,
            event_connexion=17 + index * 7,
            central_atom=index,
            sym=0,
            transient=destination in sources,
            dE_forward=0.0,
            k_forward=float(rate),
            dE_backward=0.0,
            k_backward=0.0,
        )
    if labels is not None:
        table.df.index = list(labels)
    return table


def reversible_table():
    return connectivity([(0, 1, 1.0), (1, 0, 2.0), (0, 2, 1.0), (1, 3, 1.0)])


def occupation_closed_form(t):
    return np.array(
        [
            (2 * math.exp(-t) + math.exp(-4 * t)) / 3,
            (math.exp(-t) - math.exp(-4 * t)) / 3,
        ]
    )


def built(table):
    selector = FPTASelector()
    selector.build_absorbing_matrix_from_connectivity(table)
    selector.build_reduced_matrix(len(set(table.df["state"])))
    return selector


def fixed_draws(monkeypatch, *draws):
    iterator = iter(draws)
    consumed = []

    def random():
        value = next(iterator)
        consumed.append(value)
        return value

    monkeypatch.setattr(np.random, "random", random)
    return consumed


class TestInstantaneousFluxLaw:
    @pytest.mark.parametrize("t", [0.01, math.log(2), 1.0, 3.0])
    def test_occupation_and_flux_match_the_closed_form(self, t):
        selector = built(reversible_table())
        q = selector.transient_occupation(t)
        np.testing.assert_allclose(q, occupation_closed_form(t), rtol=0, atol=1e-12)
        flux = selector.absorbing_flux(t)
        # unit exit rates: the flux into A and B is the occupation of 0 and 1
        np.testing.assert_allclose(flux, q, rtol=0, atol=1e-12)
        # total flux = exit-time density = survival here (constant hazard 1)
        assert flux.sum() == pytest.approx(math.exp(-t), rel=1e-12)
        assert flux.sum() / q.sum() == pytest.approx(1.0, rel=1e-12)

    def test_channel_conditions_on_instantaneous_flux_not_cumulative_absorption(
        self, monkeypatch
    ):
        consumed = fixed_draws(monkeypatch, 0.5, 0.76)
        result = FPTASelector().select_from_connectivity(reversible_table())
        assert result.is_ok()
        actual = result.ok_value()
        assert actual.t_exit == pytest.approx(math.log(2), rel=5e-4)
        assert 17 / 24 < 0.76 < 79 / 96
        assert actual.exit_state == 3
        assert actual.from_state == 1
        assert actual.exit_row == 3  # the (1 -> 3) row
        assert consumed == [0.5, 0.76]

    def test_state_draw_agrees_with_the_direct_call(self, monkeypatch):
        selector = built(reversible_table())
        for draw in [0.0, 0.3, 0.7, 0.71, 0.9]:
            fixed_draws(monkeypatch, draw)
            state = selector.select_absorbing_state(math.log(2))
            fixed_draws(monkeypatch, draw)
            row, source, destination = selector.select_exit_transition(
                reversible_table(), math.log(2)
            )
            assert destination == state
            assert (state == 2) is (draw < 17 / 24)

    def test_transition_order_by_destination_matches_state_selection(self, monkeypatch):
        """Both transient states exit 2:1 to states 2 and 3; the table lists the
        transitions by source, the draw orders them by destination."""
        table = connectivity(
            [
                (0, 1, 1.0),
                (1, 0, 2.0),
                (0, 2, 2.0),
                (0, 3, 1.0),
                (1, 2, 2.0),
                (1, 3, 1.0),
            ]
        )
        selector = built(table)
        t = math.log(2) / 3
        exits = table.absorbing_transitions()
        assert list(exits["state_connexion"]) == [2, 2, 3, 3]
        assert list(exits["state"]) == [0, 1, 0, 1]
        assert list(exits.index) == [2, 4, 3, 5]
        for draw in [0.0, 0.1, 0.55, 0.6, 0.66, 0.67, 0.9, 0.95]:
            fixed_draws(monkeypatch, draw)
            state = selector.select_absorbing_state(t)
            fixed_draws(monkeypatch, draw)
            _, _, destination = selector.select_exit_transition(table, t)
            assert destination == state
            assert (state == 2) is (draw < 2 / 3)


class TestTransitionIdentity:
    def test_shared_destination_selects_the_source_by_occupation_flux(
        self, monkeypatch
    ):
        table = connectivity(
            [(0, 1, 1.0), (1, 0, 2.0), (0, 2, 1.0), (1, 2, 1.0)], [91, 3, 104, 8]
        )
        consumed = fixed_draws(monkeypatch, 0.5, 0.76)
        result = FPTASelector().select_from_connectivity(table)
        assert result.is_ok()
        actual = result.ok_value()
        assert consumed == [0.5, 0.76]
        assert actual.exit_state == 2
        # at the median, source 0 carries 17/24 of the flux into state 2
        assert actual.from_state == 1
        assert actual.exit_row == 8
        fixed_draws(monkeypatch, 0.5, 0.7)
        actual = FPTASelector().select_from_connectivity(table).ok_value()
        assert (actual.from_state, actual.exit_row) == (0, 104)

    @pytest.mark.parametrize(
        "draw,label", [(0.1, 104), (0.24, 104), (0.26, 8), (0.76, 8)]
    )
    def test_parallel_transitions_keep_their_row_identity(
        self, monkeypatch, draw, label
    ):
        table = connectivity([(0, 1, 1.0), (0, 1, 3.0)], [104, 8])
        fixed_draws(monkeypatch, 0.5, draw)
        actual = FPTASelector().select_from_connectivity(table).ok_value()
        assert actual.t_exit == pytest.approx(math.log(2) / 4, rel=5e-4)
        assert (actual.exit_state, actual.from_state, actual.exit_row) == (1, 0, label)

    def test_zero_rate_leading_transition_is_never_selected(self, monkeypatch):
        table = connectivity([(0, 1, 0.0), (0, 2, 1.0)])
        fixed_draws(monkeypatch, 0.5, 0.0)
        actual = FPTASelector().select_from_connectivity(table).ok_value()
        assert (actual.exit_state, actual.exit_row) == (2, 1)
        selector = built(table)
        fixed_draws(monkeypatch, 0.0)
        assert selector.select_absorbing_state(1.0) == 2

    def test_absorbing_transitions_are_structural_not_the_flag(self):
        """A row flagged transient whose destination is never a source is an
        exit of the generator; the frozen oracle's single-transient control
        relies on it."""
        table = connectivity([(0, 1, 2.0), (0, 2, 1.0)])
        table.df.loc[0, "transient"] = True
        assert list(table.absorbing_transitions().index) == [0, 1]
        selector = built(table)
        np.testing.assert_array_equal(
            selector.M_abs, [[3, 0, 0], [-2, 0, 0], [-1, 0, 0]]
        )


class TestNumericalPolicy:
    def test_roundoff_within_tolerance_is_discarded(self):
        scale = 0.5
        noise = 0.5 * FLUX_ROUNDOFF_RTOL * scale
        values = np.array([scale + 0j, -noise + 1j * noise, 0.25 + 0j])
        cleaned = validate_flux(values, "flux", 1.0)
        assert cleaned.dtype == float
        np.testing.assert_array_equal(cleaned, [scale, 0.0, 0.25])

    def test_tiny_positive_flux_is_usable(self):
        cleaned = validate_flux(np.array([1e-300, 2e-300]), "flux", 1.0)
        np.testing.assert_allclose(
            normalized_weights(cleaned), [1 / 3, 2 / 3], rtol=1e-12
        )

    @pytest.mark.parametrize(
        "values,fragment",
        [
            (np.array([1.0, np.nan]), "not finite"),
            (np.array([np.inf, 1.0]), "not finite"),
            (np.array([1.0, -2 * FLUX_ROUNDOFF_RTOL]), "materially negative"),
            (np.array([1.0 + 0j, 0.5 + 1j * 2 * FLUX_ROUNDOFF_RTOL]), "not real"),
            (np.array([0.0, 0.0]), "no positive entry"),
            (np.array([-1.0, -2.0]), "no positive entry"),
            (np.array([]), "empty"),
        ],
    )
    def test_invalid_vectors_are_rejected_explicitly(self, values, fragment):
        with pytest.raises(BasinExitFluxError, match=fragment) as info:
            validate_flux(values, "flux", 2.5)
        assert info.value.variables["t_exit"] == 2.5

    def test_draw_never_selects_a_zero_weight_channel(self, monkeypatch):
        weights = normalized_weights(np.array([0.0, 1.0, 0.0, 3.0]))
        for draw, expected in [(0.0, 1), (0.2, 1), (0.25, 3), (0.999999, 3)]:
            fixed_draws(monkeypatch, draw)
            assert draw_channel(weights) == expected

    def test_invalid_flux_is_an_explicit_err_through_the_result_contract(
        self, monkeypatch
    ):
        garbage = np.array([1.0, -0.5])
        monkeypatch.setattr(
            "pykmc.basins.selection.solve_master_equation", lambda M, t, p0: garbage
        )
        fixed_draws(monkeypatch, 0.5, 0.76)
        result = FPTASelector().select_from_connectivity(reversible_table())
        assert not result.is_ok()
        error = result.err_value()
        assert error.type is ErrorType.BASIN_INVALID_EXIT_FLUX
        assert "materially negative" in error.message
        assert error.variables["t_exit"] == pytest.approx(math.log(2), rel=5e-4)
        selector = built(reversible_table())
        with pytest.raises(BasinExitFluxError, match="materially negative"):
            selector.select_absorbing_state(1.0)

    def test_singular_solver_output_is_an_explicit_err(self, monkeypatch):
        def singular(M, t, p0):
            raise np.linalg.LinAlgError("Singular matrix")

        monkeypatch.setattr("pykmc.basins.selection.solve_master_equation", singular)
        fixed_draws(monkeypatch, 0.5)
        result = FPTASelector().select_from_connectivity(reversible_table())
        assert not result.is_ok()
        assert result.err_value().type is ErrorType.BASIN_INVALID_EXIT_FLUX
        assert "Singular" in result.err_value().message
