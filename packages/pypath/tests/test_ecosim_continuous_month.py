"""Month integrator with quadrature (continuous path)."""

import logging

import numpy as np
import pytest

from pypath.core.ecosim_continuous import (
    METHOD_NAMES,
    MonthResult,
    default_atol,
    integrate_month,
)

K = 0.7
Y0 = np.array([2.0, 5.0])


class Decay:
    """dy = -k y on states 1..n; rates are full length, index 0 = Outside."""

    def __init__(self, k=K, nan_after=None):
        self.k = k
        self.nan_after = nan_after
        self.stats = {}

    def __call__(self, t, y):
        if self.nan_after is not None and t > self.nan_after:
            return np.full_like(y, np.nan)
        return -self.k * y

    def rates(self, y):
        full = np.concatenate([[0.0], y])
        return {"FoodGain": full.copy(), "LossPropToB": self.k * full}


def test_the_month_mean_is_the_exact_integral():
    dt = 1.0
    res = integrate_month(Decay(), Y0, dt, "dop853", 1e-12, np.full(2, 1e-14))
    assert isinstance(res, MonthResult)
    expected = Y0 * (1 - np.exp(-K * dt)) / (K * dt)
    np.testing.assert_allclose(res.y_mean, expected, rtol=1e-9)
    np.testing.assert_allclose(res.food_gain_mean[1:], expected, rtol=1e-9)
    assert res.food_gain_mean[0] == 0.0
    np.testing.assert_allclose(res.loss_rate_mean[1:], K, rtol=1e-9)
    assert res.loss_rate_mean[0] == 0.0
    assert res.nfev > 0 and res.njev == 0 and res.status == 0


@pytest.mark.parametrize("method", sorted(METHOD_NAMES))
def test_every_method_solves_a_month(method):
    res = integrate_month(Decay(), Y0, 1.0, method, 1e-8, np.full(2, 1e-10))
    np.testing.assert_allclose(res.y_end, Y0 * np.exp(-K), rtol=1e-5)


def test_a_failed_month_raises_or_warns(caplog):
    rhs = Decay(nan_after=0.5)
    atol = np.full(2, 1e-10)
    with pytest.raises(RuntimeError):
        integrate_month(rhs, Y0, 1.0, "dop853", 1e-8, atol)
    with caplog.at_level(logging.WARNING, logger="pypath.core.ecosim_continuous"):
        res = integrate_month(rhs, Y0, 1.0, "dop853", 1e-8, atol, on_failure="warn")
    assert np.all(np.isfinite(res.y_end)) and np.all(res.y_end > 0)
    assert np.all(np.isfinite(res.y_mean))
    assert res.status < 0
    assert 0.0 < res.t_reached < 1.0
    tr = res.t_reached
    np.testing.assert_allclose(res.y_end, Y0 * np.exp(-K * tr), rtol=1e-6)
    np.testing.assert_allclose(
        res.y_mean, Y0 * (1 - np.exp(-K * tr)) / (K * tr), rtol=1e-6
    )
    assert any(r.levelname == "WARNING" for r in caplog.records)


@pytest.mark.parametrize("on_failure", ["raise", "warn"])
def test_a_nan_start_derivative_raises_without_solving(monkeypatch, on_failure):
    import pypath.core.ecosim_continuous as mod

    def boom(*a, **k):
        raise AssertionError("solve_ivp must not be called")

    monkeypatch.setattr(mod, "solve_ivp", boom)
    with pytest.raises(RuntimeError, match="non-finite start derivative"):
        integrate_month(
            Decay(nan_after=-1.0),
            Y0,
            1.0,
            "lsoda",
            1e-8,
            np.full(2, 1e-10),
            on_failure=on_failure,
        )


def test_loss_rate_is_zero_where_biomass_is_zero():
    res = integrate_month(
        Decay(), np.array([0.0, 3.0]), 1.0, "lsoda", 1e-8, np.full(2, 1e-10)
    )
    assert res.loss_rate_mean[1] == 0.0
    assert res.y_end[0] == 1e-20
    np.testing.assert_allclose(res.loss_rate_mean[2], K, rtol=1e-6)


def test_default_atol_scales_with_biomass():
    out = default_atol(np.array([1e-6, 1.0, 100.0]))
    np.testing.assert_allclose(out, [1e-12, 1e-8, 1e-6], rtol=1e-12)
