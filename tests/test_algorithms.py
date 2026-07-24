"""Tests for the rejection-free KMC selection in pykmc.algorithms."""

import math

import pytest

from pykmc import algorithms
from pykmc.algorithms import rejection_free

RATES = [1.0, 2.0, 3.0]


def test_zero_draw_gives_finite_delta_t(monkeypatch: pytest.MonkeyPatch) -> None:
    """An exact 0.0 draw must not raise: u = 1 - 0 = 1 gives delta_t = 0."""
    monkeypatch.setattr(algorithms.random, "random", lambda: 0.0)
    idx, delta_t, ktot = rejection_free(RATES)
    assert math.isfinite(delta_t)
    assert delta_t >= 0.0
    assert 0 <= idx < len(RATES)
    assert ktot == pytest.approx(sum(RATES))


def test_near_one_draw_gives_finite_positive_delta_t(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The largest float below 1 gives a large but finite, positive delta_t."""
    largest_below_one = math.nextafter(1.0, 0.0)
    monkeypatch.setattr(algorithms.random, "random", lambda: largest_below_one)
    idx, delta_t, _ = rejection_free(RATES)
    assert math.isfinite(delta_t)
    assert delta_t > 0.0
    assert 0 <= idx < len(RATES)


def test_typical_draw_gives_positive_delta_t() -> None:
    """Unmocked draws give a valid index and a strictly positive time step."""
    algorithms.random.seed(1234)
    idx, delta_t, ktot = rejection_free(RATES)
    assert math.isfinite(delta_t)
    assert delta_t > 0.0
    assert 0 <= idx < len(RATES)
    assert ktot == pytest.approx(sum(RATES))
