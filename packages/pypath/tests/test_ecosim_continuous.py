"""Continuous (solve_ivp) integration pathway of rsim_run."""

import pytest

from pypath.core.ecosim import rsim_run
from tests.continuous_support import needs_reco, run_dicts, scenario
from tests.continuous_support import reco_plain as _reco_plain

pytestmark = needs_reco


@pytest.fixture(scope="module")
def reco_plain():
    return _reco_plain()


def test_an_unknown_method_raises(reco_plain):
    with pytest.raises(ValueError, match="lsoda"):
        rsim_run(scenario(reco_plain), method="rk45", years=range(1, 2))


def test_the_parity_paths_have_no_solver_info(reco_plain):
    out = rsim_run(scenario(reco_plain), method="RK4", years=range(1, 2))
    assert out.solver_info is None and out.out_Catch_month_end is None


def test_run_dicts_returns_the_per_run_dicts(reco_plain):
    state, params_dict, forcing_dict, fishing_dict = run_dicts(scenario(reco_plain))
    assert "QQbase" in params_dict and "NUM_LIVING" in params_dict
    assert state.ndim == 1
