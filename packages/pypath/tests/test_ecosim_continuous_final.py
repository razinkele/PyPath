"""Final-review fixes of the continuous path: exported Hden residual, start
feeding time clipped, no budget on a degraded month, guarded pool residual,
and the feeding-time atol."""

import warnings
from types import SimpleNamespace

import numpy as np
import pytest

from pypath.core.ecosim import rsim_run
from pypath.core.ecosim_continuous import month_budget
from tests.continuous_support import needs_reco, reco_plain, scenario

pytestmark = needs_reco


def _adj_scen(adj=0.5):
    scen = scenario(reco_plain(), rule="ewe")
    scen.params.FtimeAdj[1 : scen.params.NUM_LIVING + 1] = adj
    return scen


def test_hden_residual_max_is_exported():
    scen = scenario(reco_plain(), rule="ewe", effort=2.0)
    scen.params.CmCo[1 : scen.params.NUM_LIVING + 1] = 2.0
    out = rsim_run(scen, method="lsoda", years=range(1, 2))
    for m in out.solver_info["months"]:
        r = m["hden_residual_max"]
        assert np.isfinite(r) and 0.0 <= r <= 1e-11, (m["month"], r)


@pytest.mark.parametrize("start", [5.0, 0.001])
def test_start_feeding_time_is_clipped_into_its_bounds(start):
    scen = _adj_scen()
    scen.start_state.Ftime[:] = start
    out = rsim_run(scen, method="lsoda", years=range(1, 2))
    lower = scen.params.ForagingTimeLowerLimit
    lo, hi = out.solver_info["months"][0]["ftime_range"]
    assert lower - 1e-6 <= lo <= hi <= 2.0 + 1e-6, (lo, hi)


def test_atol_ftime_records_the_feeding_time_tail():
    out = rsim_run(_adj_scen(), method="lsoda", years=range(1, 2))
    info = out.solver_info
    assert info["atol_ftime"] == [1e-2 * info["rtol"]] * len(info["ftime_groups"])
    plain = rsim_run(scenario(reco_plain()), method="lsoda", years=range(1, 2))
    assert plain.solver_info["atol_ftime"] == []


def _res(degraded=False, pool_net=None, y_end=None):
    return SimpleNamespace(
        t_reached=1.0 / 12,
        exchange_mean=0.0,
        pool_net_mean=pool_net,
        y_end=np.array([1.0, 2.0, 3.0] if y_end is None else y_end),
        degraded=degraded,
    )


def test_a_degraded_month_has_no_budget():
    net = np.zeros(4)
    res = _res(degraded=True, pool_net=net)
    assert month_budget(np.array([1.0, 2.0, 3.0]), res, 2, 3, False) == (None, None)
    ok = month_budget(np.array([1.0, 2.0, 3.0]), _res(pool_net=net), 2, 3, False)
    assert ok[0] is not None and ok[1] is not None


def test_a_pool_starting_at_zero_is_skipped():
    y0 = np.array([1.0, 2.0, 0.0, 3.0])
    y_end = np.array([1.0, 2.0, 0.0, 3.0])
    net = np.zeros(5)
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        # groups: 2 living, 2 pools (second pool at index 3 is healthy)
        total, pool = month_budget(y0, _res(pool_net=net, y_end=y_end), 2, 4, False)
        assert np.isfinite(total) and pool == 0.0
        y0z = np.array([1.0, 2.0, 0.0, 0.0])
        total, pool = month_budget(
            y0z, _res(pool_net=net, y_end=np.array([1.0, 2.0, 0.0, 0.0])), 2, 4, False
        )
        assert pool is None
