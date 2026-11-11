"""EwE6's sum of squares against reference time series (C1-C3).

EwE accumulates one log prediction error per observation and series
(cEcoSimModel.AccumulateDataInfo, vb 2575-2735) and sums them in
PlotDataInfo (vb 3849-3966):

- annual data are compared once a year, at the start of month 6, before that
  month's integration: the biomass at the end of month (year - 1) * 12 + 5;
- Z = ln(obs / pred); catch is predicted as B x FishTime, FishTime being the
  F of the month before (setFishTime runs after AccumulateDataInfo);
- relative series take q = mean(Z) (the conditional estimate), absolute ones
  q = 0; SS = sum over series of WtType x sum (Z - q)^2;
- every series counts, observations <= 0 are skipped.
"""

import logging
from types import SimpleNamespace

import numpy as np
import pytest

from pypath.core.calibration import ewe_sum_of_squares
from pypath.core.timeseries import (
    DATTYPE_ABS_BIOMASS,
    DATTYPE_CATCH,
    DATTYPE_CATCH_REL,
    DATTYPE_EFFORT,
    DATTYPE_REL_BIOMASS,
    DATTYPE_TOTAL_MORTALITY,
    EweTimeSeries,
    EweTimeSeriesCollection,
)

N_MONTHS = 24


def _output(n_months=N_MONTHS, n_groups=2):
    """Biomass and F that differ in every row, so a wrong row shows."""
    rows = np.arange(n_months + 1, dtype=float)[:, None]
    cols = np.arange(n_groups + 1, dtype=float)[None, :]
    biomass = 1.0 + 0.1 * rows + cols
    fish_mort = 0.2 + 0.01 * rows + 0.1 * cols
    return SimpleNamespace(out_Biomass=biomass, out_FishMort=fish_mort)


def _series(values, dat_type=DATTYPE_REL_BIOMASS, group=0, weight=1.0, sid=1):
    return EweTimeSeries(
        sid, f"s{sid}", dat_type, group, None, np.asarray(values, float), weight
    )


def _ss(z, relative):
    z = np.asarray(z)
    q = z.mean() if relative else 0.0
    return float(np.sum((z - q) ** 2)), q


def test_relative_biomass_at_the_start_of_month_six_with_log_q():
    out = _output()
    b = out.out_Biomass[:, 1]
    res = ewe_sum_of_squares(out, EweTimeSeriesCollection([_series([2.0, 3.5])]))
    expected, q = _ss([np.log(2.0 / b[5]), np.log(3.5 / b[17])], relative=True)
    assert res.total == pytest.approx(expected, rel=1e-12)
    assert res.log_q[1] == pytest.approx(q, rel=1e-12)
    assert res.n_obs[1] == 2
    assert res.by_group == pytest.approx({0: expected})


def test_absolute_biomass_has_no_q():
    out = _output()
    b = out.out_Biomass[:, 1]
    ts = EweTimeSeriesCollection([_series([2.0, 3.5], DATTYPE_ABS_BIOMASS)])
    expected, _ = _ss([np.log(2.0 / b[5]), np.log(3.5 / b[17])], relative=False)
    assert ewe_sum_of_squares(out, ts).total == pytest.approx(expected, rel=1e-12)


def test_weights_scale_each_series_and_zero_weight_drops_it():
    out = _output()
    one = ewe_sum_of_squares(out, EweTimeSeriesCollection([_series([2.0, 3.5])]))
    ts = EweTimeSeriesCollection(
        [
            _series([2.0, 3.5], weight=100.0, sid=1),
            _series([9.0, 1.0], group=1, weight=0.0, sid=2),
        ]
    )
    res = ewe_sum_of_squares(out, ts)
    assert res.total == pytest.approx(100.0 * one.total, rel=1e-12)
    assert res.by_series[2] == 0.0
    assert res.by_group[1] == 0.0


def test_every_series_counts_also_on_one_group():
    """Two biomass series and a catch series on one group all add up (PyPath
    kept one series per group and dropped the catch)."""
    out = _output()
    a = _series([2.0, 3.5], sid=1)
    b = _series([1.0, 4.0], DATTYPE_ABS_BIOMASS, sid=2)
    c = _series([0.5, 0.9], DATTYPE_CATCH, sid=3)
    parts = [
        ewe_sum_of_squares(out, EweTimeSeriesCollection([s])).total for s in (a, b, c)
    ]
    res = ewe_sum_of_squares(out, EweTimeSeriesCollection([a, b, c]))
    assert res.total == pytest.approx(sum(parts), rel=1e-12)
    assert res.by_group[0] == pytest.approx(sum(parts), rel=1e-12)
    assert all(p > 0 for p in parts)


def test_catch_is_biomass_times_the_previous_months_f():
    """EwE: Zstat = ln(obs / (BB x FishTime)), FishTime set in month 5."""
    out = _output()
    b, f = out.out_Biomass[:, 1], out.out_FishMort[:, 1]
    ts = EweTimeSeriesCollection([_series([0.5, 0.9], DATTYPE_CATCH)])
    expected, _ = _ss(
        [np.log(0.5 / (b[5] * f[5])), np.log(0.9 / (b[17] * f[17]))],
        relative=False,
    )
    assert ewe_sum_of_squares(out, ts).total == pytest.approx(expected, rel=1e-12)


def test_unfished_catch_observation_counts_as_zero_error():
    """With FishTime 0 EwE keeps the observation with Zstat = 0, so for a
    relative catch series it still enters q."""
    out = _output()
    out.out_FishMort[17, 1] = 0.0
    b, f = out.out_Biomass[:, 1], out.out_FishMort[:, 1]
    ts = EweTimeSeriesCollection([_series([0.5, 0.9], DATTYPE_CATCH_REL)])
    res = ewe_sum_of_squares(out, ts)
    expected, _ = _ss([np.log(0.5 / (b[5] * f[5])), 0.0], relative=True)
    assert res.n_obs[1] == 2
    assert res.total == pytest.approx(expected, rel=1e-12)


def test_missing_and_non_positive_values_and_years_past_the_run_are_skipped():
    out = _output()  # two years
    vals = [np.nan, 0.0, -1.0, 2.0, 5.0]
    out3 = _output(n_months=48)
    b3 = out3.out_Biomass[:, 1]
    ts = EweTimeSeriesCollection([_series(vals)])
    assert ewe_sum_of_squares(out, ts).n_obs[1] == 0
    res = ewe_sum_of_squares(out3, ts)  # four years: only year 4 is usable
    assert res.n_obs[1] == 1
    assert res.total == 0.0  # one relative observation: Z - q = 0
    ts_abs = EweTimeSeriesCollection([_series(vals, DATTYPE_ABS_BIOMASS)])
    assert ewe_sum_of_squares(out3, ts_abs).total == pytest.approx(
        np.log(2.0 / b3[41]) ** 2, rel=1e-12
    )


def test_time_step_data_compare_at_the_start_of_each_month():
    """eTSDataSetInterval.TimeStep: data point k at model step k, before its
    integration (the end of month k - 1)."""
    out = _output()
    b = out.out_Biomass[:, 1]
    ts = EweTimeSeriesCollection(
        [_series([2.0, 3.0, 4.0], DATTYPE_ABS_BIOMASS)], interval="timestep"
    )
    expected, _ = _ss([np.log(v / b[k]) for k, v in enumerate([2.0, 3.0, 4.0])], False)
    assert ewe_sum_of_squares(out, ts).total == pytest.approx(expected, rel=1e-12)


def test_unsupported_reference_types_are_reported_and_left_out(caplog):
    out = _output()
    ts = EweTimeSeriesCollection(
        [
            _series([2.0, 3.5], sid=1),
            _series([0.4, 0.5], DATTYPE_TOTAL_MORTALITY, sid=2),
            _series([1.0, 1.2], DATTYPE_EFFORT, group=None, sid=3),  # a driver
        ]
    )
    only = ewe_sum_of_squares(out, EweTimeSeriesCollection([ts.series[0]]))
    with caplog.at_level(logging.WARNING, logger="pypath.core.calibration"):
        res = ewe_sum_of_squares(out, ts)
    assert res.total == pytest.approx(only.total, rel=1e-12)
    assert "s2" in caplog.text and "total mortality" in caplog.text.lower()
    assert "s3" not in caplog.text


def test_catch_series_need_the_monthly_fishing_mortality():
    out = SimpleNamespace(out_Biomass=_output().out_Biomass, out_FishMort=None)
    ts = EweTimeSeriesCollection([_series([0.5, 0.9], DATTYPE_CATCH)])
    with pytest.raises(ValueError, match="out_FishMort"):
        ewe_sum_of_squares(out, ts)
