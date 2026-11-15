"""Log-transformed biomass states on the continuous path (opt-in)."""

import copy
import warnings

import numpy as np
import pytest

import pypath.core.ecosim_continuous as cont
from pypath.core.ecosim import rsim_run
from pypath.core.ecosim_continuous import JacobianReuse, integrate_month
from tests.continuous_support import needs_reco, reco_plain, reco_stanzas, scenario


def _run_years(scen, years, positivity, **opts):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return rsim_run(
            copy.deepcopy(scen),
            method="LSODA",
            years=range(1, years + 1),
            solver_options={"positivity": positivity, **opts},
        )


class _Decay:
    """dy = -0.7 y on states 1..n, with the rates convention of the path."""

    stats: dict = {}

    def __call__(self, t, y):
        return -0.7 * np.asarray(y, dtype=float)

    def rates(self, y, t=0.0):
        full = np.concatenate([[0.0], y])
        return {"FoodGain": full.copy(), "LossPropToB": 0.7 * full}


@needs_reco
@pytest.mark.integration
def test_log_states_never_clip():
    scen = scenario(reco_plain(), effort=100.0, years=range(1, 6))
    clip = _run_years(scen, 5, "clip")
    log = _run_years(scen, 5, "log")
    n = int(scen.params.NUM_LIVING + scen.params.NUM_DEAD)
    assert sum(m["clips"] for m in clip.solver_info["months"]) > 0
    assert sum(m["clips"] for m in log.solver_info["months"]) == 0
    assert min(m["min_state"] for m in log.solver_info["months"]) >= 1e-20
    assert log.solver_info["positivity"] == "log"
    assert clip.solver_info["positivity"] == "clip"
    b_log = log.out_Biomass[1:, 1 : n + 1]
    b_clip = clip.out_Biomass[1:, 1 : n + 1]
    assert np.all(b_log >= 1e-20)
    b0 = np.asarray(scen.params.B_BaseRef, dtype=float)[1 : n + 1]
    big = b_clip > 1e-2 * b0[None, :]
    rel = np.abs(b_log / b_clip - 1)[big]
    assert rel.max() <= 1e-4


@needs_reco
def test_log_states_agree():
    scen = scenario(reco_stanzas(), effort=2.0, years=range(1, 3))
    clip = _run_years(scen, 2, "clip")
    log = _run_years(scen, 2, "log")
    n = int(scen.params.NUM_LIVING + scen.params.NUM_DEAD)
    rel = np.abs(log.out_Biomass[1:, 1 : n + 1] / clip.out_Biomass[1:, 1 : n + 1] - 1)
    assert rel.max() <= 1e-5


def test_log_tolerance(monkeypatch):
    seen = {}
    real = cont.solve_ivp

    def spy(fun, span, y0, **kw):
        seen["atol"] = np.array(kw["atol"])
        seen["y0"] = np.array(y0)
        return real(fun, span, y0, **kw)

    monkeypatch.setattr(cont, "solve_ivp", spy)
    y0 = np.array([2.0, 1e-3, 50.0])
    atol = np.array([1e-8, 1e-8, 1e-3])
    integrate_month(_Decay(), y0, 1 / 12, "lsoda", 1e-6, atol, positivity="log")
    want = np.minimum(np.maximum(atol / y0, 1e-6), 0.1)
    assert np.allclose(seen["atol"], want, rtol=0, atol=0)
    assert seen["atol"][2] == 2e-5  # an uncapped value
    assert np.allclose(seen["y0"], np.log(y0))
    # a group at the floor is capped at 0.1, not 1e12
    seen.clear()
    integrate_month(
        _Decay(),
        np.array([1e-20, 1.0]),
        1 / 12,
        "lsoda",
        1e-6,
        1e-8,
        positivity="log",
    )
    assert seen["atol"][0] == 0.1


def test_log_feeding_states_stay_linear(monkeypatch):
    seen = {}
    real = cont.solve_ivp

    def spy(fun, span, y0, **kw):
        seen["atol"] = np.array(kw["atol"])
        seen["y0"] = np.array(y0)
        return real(fun, span, y0, **kw)

    monkeypatch.setattr(cont, "solve_ivp", spy)

    class _Ft(_Decay):
        def __call__(self, t, y):
            return np.concatenate([-0.7 * y[:2], -0.5 * (y[2:] - 1.0)])

        def rates(self, y, t=0.0):
            return super().rates(y[:2], t)

    def factory(fun, step_floor):
        seen["floor"] = np.array(step_floor)
        return JacobianReuse("lsoda", atol, 1e-6).for_month(fun, step_floor)

    y0 = np.array([2.0, 1e-3, 0.9])
    atol = np.array([1e-8, 1e-8, 1e-8])
    res = integrate_month(
        _Ft(),
        y0,
        1 / 12,
        "lsoda",
        1e-6,
        atol,
        n_bio=2,
        positivity="log",
        jac_factory=factory,
    )
    assert seen["y0"][2] == 0.9
    assert seen["atol"][2] == 1e-8
    # the difference step's floor: 1 for the log states, atol / rtol for
    # the linear feeding time
    np.testing.assert_array_equal(seen["floor"], [1.0, 1.0, 1e-8 / 1e-6])
    assert res.y_end[0] == pytest.approx(2.0 * np.exp(-0.7 / 12), rel=1e-5)
    assert res.y_mean[0] == pytest.approx(
        2.0 * (1 - np.exp(-0.7 / 12)) / (0.7 / 12), rel=1e-5
    )


class _Spy:
    """Records the z the solver saw (``solve_ivp``'s solution)."""

    def __init__(self, monkeypatch):
        self.sols = []
        real = cont.solve_ivp

        def spy(*a, **k):
            sol = real(*a, **k)
            self.sols.append(sol)
            return sol

        monkeypatch.setattr(cont, "solve_ivp", spy)


def test_log_floor_absorbs(monkeypatch):
    """A group at the floor with dy < 0 cannot fall: its z stays constant
    (the dz = max(dz, 0) rule; the back-transform's clamp alone would hide a
    falling z), and its biomass is exactly 1e-20."""
    spy = _Spy(monkeypatch)

    class _Shrink(_Decay):
        def __call__(self, t, y):
            return -5.0 * np.asarray(y, dtype=float) - 1e-3 * np.array([0.0, 1.0])

    res = integrate_month(
        _Shrink(),
        np.array([1.0, 1e-20]),
        1 / 12,
        "lsoda",
        1e-6,
        1e-8,
        positivity="log",
    )
    z = spy.sols[0].y[1]
    assert np.all(z == np.log(1e-20))
    assert res.y_end[1] == 1e-20
    assert res.y_end[0] == pytest.approx(np.exp(-5.0 / 12), rel=1e-5)


class _Regrow:
    """dB = r(t) B with r(t) = A (1 + sin(w t)): B(t) = B0 exp(R(t))."""

    A = 300.0
    W = 2 * np.pi * 36.0  # three cycles in a month
    stats: dict = {}

    def __call__(self, t, y):
        return self.A * (1 + np.sin(self.W * t)) * np.asarray(y, dtype=float)

    def rates(self, y, t=0.0):
        full = np.concatenate([[0.0], y])
        return {"FoodGain": full.copy(), "LossPropToB": 0.0 * full}

    def exact(self, b0, t):
        return b0 * np.exp(self.A * (t + (1 - np.cos(self.W * t)) / self.W))


def _regrow_month(**kw):
    dt = 1 / 12
    rhs = _Regrow()
    res = integrate_month(
        rhs,
        np.array([1e-20]),
        dt,
        "lsoda",
        1e-6,
        np.array([1e-8]),
        positivity="log",
        **kw,
    )
    return res, rhs.exact(1e-20, dt)


def test_log_regrowth_from_the_floor():
    """A group starting at the floor regrows at the right rate (exact B =
    1e-20 exp(R(t)), 11 orders of magnitude). The capped z tolerance (0.1)
    keeps error control, a relative error of a few percent at most; the
    measured one is 6e-3, bounded here by 2e-2."""
    res, want = _regrow_month()
    assert want > 1e-10
    assert res.y_end[0] == pytest.approx(want, rel=2e-2)


def test_log_regrowth_needs_the_cap(monkeypatch):
    """Without the cap atol_z = 1e12 at the floor: no control (7 evaluations,
    a value 60 % off)."""
    monkeypatch.setattr(cont, "_ATOL_Z_MAX", 1e30)
    with np.errstate(all="ignore"):
        res, want = _regrow_month()
    assert abs(res.y_end[0] / want - 1) > 0.3


def test_log_jacobian_in_z():
    seen = {}

    def factory(fun, step_floor):
        seen["fun"], seen["floor"] = fun, np.array(step_floor)
        jr = JacobianReuse("lsoda", np.full(2, 7.0), 1e-6)  # the run's: not used
        return jr.for_month(fun, step_floor=step_floor)

    # atol_z 1e-6, 1e-3 and (at the floor) the 0.1 cap: atol_z / rtol 1,
    # 1e3 and 1e5
    y0 = np.array([2.0, 1e-5, 1e-20])
    integrate_month(
        _Decay(),
        y0,
        1 / 12,
        "lsoda",
        1e-6,
        np.full(3, 1e-8),
        positivity="log",
        jac_factory=factory,
    )
    # a log state's difference-step floor is 1, h = 1.49e-8 max(|z|, 1):
    # min(atol_z / rtol, 1), not atol_z / rtol (which gave h = 1.49e-5 in z
    # for the second group and 1.49e-3 for the third)
    assert np.array_equal(seen["floor"], np.ones(3))
    # ... and the floor sets the finite-difference step
    calls = []

    def probe(t, z):
        calls.append(np.array(z))
        return -0.7 * np.ones_like(z)

    jr = JacobianReuse("lsoda", np.full(2, 7.0), 1e-6)  # the run's: not used
    z = np.array([0.1, np.log(1e-5)])  # |z| 0.1 and 11.5
    jr.for_month(probe, step_floor=seen["floor"][:2])(0.0, z)
    assert calls[1][0] - z[0] == pytest.approx(1.49e-8 * 1.0, rel=1e-6)
    assert calls[2][1] - z[1] == pytest.approx(1.49e-8 * abs(z[1]), rel=1e-6)
    # the wrapped function in z: dz = dy / B = -0.7
    assert np.allclose(seen["fun"](0.0, np.log(y0[:2])), -0.7)
    with pytest.raises(ValueError, match="jac_factory"):
        integrate_month(
            _Decay(),
            y0,
            1 / 12,
            "lsoda",
            1e-6,
            1e-8,
            positivity="log",
            jac=lambda t, y: np.eye(2),
        )


def test_positivity_option():
    with pytest.raises(ValueError, match="positivity"):
        integrate_month(
            _Decay(), np.ones(2), 1 / 12, "lsoda", 1e-6, 1e-8, positivity="x"
        )
    from pypath.core.ecosim_continuous import continuous_options

    with pytest.raises(ValueError, match="positivity"):
        continuous_options("lsoda", {"positivity": "x"}, 3)
    assert continuous_options("lsoda", None, 3)["positivity"] == "clip"
    assert continuous_options("lsoda", {"positivity": "log"}, 3)["positivity"] == "log"


@needs_reco
def test_log_forced_group_held():
    scen = scenario(reco_plain(), effort=2.0, years=range(1, 3))
    b0 = np.asarray(scen.params.B_BaseRef, dtype=float)
    g = 3
    scen.forcing.ForcedBio[:, g] = 0.8 * b0[g]
    clip = _run_years(scen, 1, "clip")
    log = _run_years(scen, 1, "log")
    assert np.allclose(log.out_Biomass[1:, g], 0.8 * b0[g], rtol=1e-12)
    assert sum(m["clips"] for m in log.solver_info["months"]) == 0
    other = [i for i in range(1, 8) if i != g]
    assert np.allclose(
        log.out_Biomass[1:, other], clip.out_Biomass[1:, other], rtol=1e-4
    )
