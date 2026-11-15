"""Interpolated mediation and the applicable-corrections list (continuous path)."""

import warnings

import numpy as np
import pytest

from pypath.core.ecosim import rsim_run
from pypath.core.mediation import EweMediation
from tests.continuous_support import needs_reco, scenario
from tests.continuous_support import reco_plain as _reco_plain
from tests.test_ewe_mediation import (
    BENTHOS,
    FISH,
    SEARCH,
    SHAPE_A,
    ZOO,
    _mediation,
    _scenario,
)

pytestmark = needs_reco


@pytest.fixture(scope="module")
def reco_plain():
    return _reco_plain()


def _base_mediation():
    return _mediation(SHAPE_A, 300, [0.0, 1.0], [], [1.0, 1.0])


def test_the_interpolated_lookup():
    med = _base_mediation()
    x64 = float(med.xbase64[0])
    pts = np.asarray(SHAPE_A, dtype=np.float64)
    assert med.lookup(0, x64, interpolate=True) == 1.0
    mid = med.lookup(0, x64 * 300.5 / 300, interpolate=True)
    assert mid == pytest.approx((pts[299] + pts[300]) / 2 / pts[299], rel=1e-15)
    for f in (1 + 1e-9, 1 - 1e-9):
        assert med.lookup(0, x64 * f, interpolate=True) != 1.0
        assert med.lookup(0, x64 * f) == 1.0  # EwE's truncating lookup
    assert med.lookup(0, 0.0, interpolate=True) == pts[0] / pts[299]
    assert med.lookup(0, 10 * x64, interpolate=True) == pts[1199] / pts[299]


def test_xbase64_is_the_float64_base_sum():
    med = _mediation(SHAPE_A, 300, [0.0, 0.3], [], [1.0, 0.1])
    assert med.xbase64[0] == 0.3 * 0.1  # not rounded to single precision
    # float64 values at the base are exactly 1 whatever the weights
    assert med.values(np.array([1.0, 0.1]), interpolate=True)[0] == 1.0


def _run(sc, method):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return rsim_run(sc, method=method, years=range(1, 2))


def _mediated_scenario():
    sc = _scenario()
    n = sc.params.NUM_GROUPS + 1
    weights = np.zeros(n)
    weights[BENTHOS] = 1.0
    sc.params.EweMediation = _mediation(
        SHAPE_A, 300, weights, [(ZOO, FISH, 0, SEARCH)], sc.params.B_BaseRef
    )
    return sc


def test_the_continuous_path_interpolates_mediation(monkeypatch):
    seen = []
    real = EweMediation.link_factors

    def spy(self, *args, **kwargs):
        seen.append(kwargs.get("interpolate", False))
        return real(self, *args, **kwargs)

    monkeypatch.setattr(EweMediation, "link_factors", spy)
    out = _run(_mediated_scenario(), "lsoda")
    assert seen and all(seen)
    assert "mediation interpolated" in out.solver_info["corrections"]
    seen.clear()
    _run(_mediated_scenario(), "RK4")
    assert seen and not any(seen)


def test_corrections_false_keeps_mediation_interpolated():
    out = rsim_run(
        _mediated_scenario(),
        method="lsoda",
        years=range(1, 2),
        solver_options={"corrections": False},
    )
    assert "mediation interpolated" in out.solver_info["corrections"]


def test_the_rhs_does_not_flag_the_callers_params():
    sc = _mediated_scenario()
    _run(sc, "lsoda")
    assert not getattr(sc.params, "MEDIATION_INTERPOLATE", False)


def test_corrections_list_what_applies(reco_plain):
    rpath_out = rsim_run(
        scenario(reco_plain, rule="rpath"), method="lsoda", years=range(1, 2)
    )
    corr = rpath_out.solver_info["corrections"]
    assert "DF once" not in corr and "R2" not in corr and "R5" not in corr
    scen = scenario(reco_plain, rule="ewe")
    scen.params.EmigRate[:] = 0.0
    ewe = rsim_run(scen, method="lsoda", years=range(1, 2))
    assert "R4" not in ewe.solver_info["corrections"]
    scen.params.EmigRate[12] = 0.1
    ewe = rsim_run(scen, method="lsoda", years=range(1, 2))
    assert "R4" in ewe.solver_info["corrections"]
