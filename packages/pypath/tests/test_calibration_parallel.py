"""fit_to_timeseries(workers=...): serial default unchanged, parallel runs
reproducible whatever the number of workers."""

import numpy as np
import pytest

from pypath.core.calibration import _resolve_workers, fit_to_timeseries
from pypath.core.ecosim import rsim_run, rsim_scenario
from pypath.core.timeseries import (
    DATTYPE_REL_BIOMASS,
    EweTimeSeries,
    EweTimeSeriesCollection,
)
from tests.test_calibration import (
    TestFitToTimeseries as _FitCase,  # alias: not re-collected
)


def _problem():
    balanced, params = _FitCase.simple_model.__wrapped__(None)
    scenario = rsim_scenario(balanced, params, years=range(1, 6))
    out = rsim_run(scenario)
    obs = np.array([out.out_Biomass[y * 12 : y * 12 + 12, 2].mean() for y in range(5)])
    obs = obs * (1 + 0.1 * np.random.default_rng(1).standard_normal(5))
    ts = EweTimeSeriesCollection(
        [EweTimeSeries(1, "Consumer", DATTYPE_REL_BIOMASS, 1, None, obs)]
    )
    return balanced, params, ts


def _fit(workers):
    balanced, params, ts = _problem()
    return fit_to_timeseries(
        balanced, params, ts, fit_vv=True, max_iterations=2, workers=workers
    )


@pytest.mark.slow
def test_parallel_is_reproducible_across_worker_counts():
    a, b = _fit(2), _fit(3)
    np.testing.assert_array_equal(a.best_vv, b.best_vv)
    assert a.ss == b.ss
    assert a.n_iterations == b.n_iterations > 0


def test_resolve_workers():
    assert _resolve_workers(1) == 1
    assert _resolve_workers(None) == 1
    assert _resolve_workers(-1) >= 1
    assert _resolve_workers(4) == 4
    with pytest.raises(ValueError):
        _resolve_workers(0)
