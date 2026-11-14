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
    assert res.steps >= 1 and res.degraded is False


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
    # averaged over the interval reached: a failure, but no fallback
    assert res.degraded is False
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


class _Sol:
    """A solve_ivp result stand-in for branches a natural ODE cannot reach."""

    def __init__(self, y_end, dense, status=0):
        self.status = status
        self.message = "stub"
        self.t = np.array([0.0, 1.0])
        self.y = np.column_stack([Y0, y_end])
        self.sol = dense
        self.nfev = 1


def _patched(monkeypatch, y_end, dense):
    import pypath.core.ecosim_continuous as mod

    monkeypatch.setattr(mod, "solve_ivp", lambda *a, **k: _Sol(y_end, dense))


def _run(on_failure):
    return integrate_month(
        Decay(), Y0, 1.0, "lsoda", 1e-8, np.full(2, 1e-10), on_failure=on_failure
    )


def _good(t):
    return Y0 * np.exp(-K * t)


def test_a_non_finite_final_state_is_a_failure(monkeypatch, caplog):
    _patched(monkeypatch, np.array([np.nan, 1.0]), _good)
    with pytest.raises(RuntimeError, match="non-finite final state"):
        _run("raise")
    with caplog.at_level(logging.WARNING, logger="pypath.core.ecosim_continuous"):
        res = _run("warn")
    assert np.all(np.isfinite(res.y_end))
    assert res.t_reached == 0.0
    assert res.degraded is True  # no interval reached: the start state's means
    assert any("non-finite final state" in r.getMessage() for r in caplog.records)


def test_non_finite_dense_states_are_a_failure(monkeypatch, caplog):
    _patched(monkeypatch, _good(1.0), lambda t: np.array([np.nan, 1.0]))
    with pytest.raises(RuntimeError, match="non-finite dense output"):
        _run("raise")
    with caplog.at_level(logging.WARNING, logger="pypath.core.ecosim_continuous"):
        res = _run("warn")
    np.testing.assert_allclose(res.y_mean, Y0)
    assert res.status == 0 and res.t_reached == 1.0 and res.degraded is True
    assert any("non-finite dense output" in r.getMessage() for r in caplog.records)


class _NanRates(Decay):
    def rates(self, y):
        out = super().rates(y)
        if y[0] != Y0[0]:
            out["FoodGain"][1] = np.nan
        return out


def test_non_finite_rates_are_a_failure(caplog):
    args = (Y0, 1.0, "dop853", 1e-8, np.full(2, 1e-10))
    with pytest.raises(RuntimeError, match="non-finite rates"):
        integrate_month(_NanRates(), *args)
    with caplog.at_level(logging.WARNING, logger="pypath.core.ecosim_continuous"):
        res = integrate_month(_NanRates(), *args, on_failure="warn")
    assert np.all(np.isfinite(res.food_gain_mean))
    assert np.all(np.isfinite(res.loss_rate_mean))
    assert res.status == 0 and res.t_reached == 1.0 and res.degraded is True
    assert any("non-finite rates" in r.getMessage() for r in caplog.records)


def test_steps_count_the_solver_steps(monkeypatch):
    _patched(monkeypatch, _good(1.0), _good)
    res = _run("raise")
    assert res.steps == 1  # the stub's t is [0, 1]
    assert res.degraded is False


def test_the_month_mean_is_clipped_at_zero(monkeypatch):
    """A dense output dipping below 0 gives a mean of exactly 0, not 1e-20."""
    _patched(monkeypatch, np.array([1e-20, 1.0]), lambda t: np.array([-1e-14, 1.0]))
    res = _run("raise")
    assert res.y_mean[0] == 0.0
    assert res.y_mean[1] == pytest.approx(1.0, rel=1e-12)
    assert res.degraded is False


def test_a_label_names_the_failure_in_the_warning(caplog):
    rhs = Decay(nan_after=0.5)
    with caplog.at_level(logging.WARNING, logger="pypath.core.ecosim_continuous"):
        integrate_month(
            rhs,
            Y0,
            1.0,
            "dop853",
            1e-8,
            np.full(2, 1e-10),
            on_failure="warn",
            label="month 3 (year 1)",
        )
    warnings_ = [r.getMessage() for r in caplog.records if r.levelname == "WARNING"]
    assert warnings_ and all(m.startswith("month 3 (year 1): ") for m in warnings_)
