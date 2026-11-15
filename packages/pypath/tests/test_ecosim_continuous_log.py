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
@pytest.mark.integration
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
    want = np.maximum(atol / y0, 1e-6)
    assert np.allclose(seen["atol"], want, rtol=0, atol=0)
    assert np.allclose(seen["y0"], np.log(y0))


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

    y0 = np.array([2.0, 1e-3, 0.9])
    atol = np.array([1e-8, 1e-8, 1e-8])
    res = integrate_month(
        _Ft(), y0, 1 / 12, "lsoda", 1e-6, atol, n_bio=2, positivity="log"
    )
    assert seen["y0"][2] == 0.9
    assert seen["atol"][2] == 1e-8
    assert res.y_end[0] == pytest.approx(2.0 * np.exp(-0.7 / 12), rel=1e-5)
    assert res.y_mean[0] == pytest.approx(
        2.0 * (1 - np.exp(-0.7 / 12)) / (0.7 / 12), rel=1e-5
    )


def test_log_floor_absorbs():
    class _Shrink(_Decay):
        def __call__(self, t, y):
            return -5.0 * np.asarray(y, dtype=float) - 1e-3

    res = integrate_month(
        _Shrink(),
        np.array([1.0, 1e-20]),
        1 / 12,
        "lsoda",
        1e-6,
        1e-8,
        positivity="log",
    )
    assert np.all(np.isfinite(res.y_end))
    assert np.all(res.y_end >= 1e-20)


def test_log_jacobian_in_z():
    seen = {}

    def factory(fun, atol):
        seen["fun"], seen["atol"] = fun, atol
        return JacobianReuse("lsoda", atol, 1e-6).for_month(fun, atol=atol)

    y0 = np.array([2.0, 3.0])
    integrate_month(
        _Decay(),
        y0,
        1 / 12,
        "lsoda",
        1e-6,
        np.array([1e-8, 1e-8]),
        positivity="log",
        jac_factory=factory,
    )
    # the wrapped function in z: dz = dy / B = -0.7
    assert np.allclose(seen["fun"](0.0, np.log(y0)), -0.7)
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
@pytest.mark.integration
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
