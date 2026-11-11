"""Ecosim calibration via sum-of-squares fitting to time series data.

Provides fit_to_timeseries() for optimizing Ecosim vulnerability (VV)
and primary production (PP) parameters against observed time series.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from pypath.core.ecosim import RsimScenario

logger = logging.getLogger(__name__)


@dataclass
class CalibrationResult:
    """Result of time series calibration.

    Parameters
    ----------
    best_vv : np.ndarray
        Fitted vulnerability values, one per pred-prey link.
    best_pp : np.ndarray or None
        Fitted primary production anomaly (if fit_pp was True).
    ss : float
        Final sum-of-squares value.
    ss_by_group : dict[int, float]
        SS contribution per group index (0-based).
    n_iterations : int
        Number of optimizer iterations performed.
    converged : bool
        Whether the optimizer converged.
    fitted_scenario : RsimScenario or None
        Scenario with best-fit parameters applied.
    link_map : list[tuple[int, int]]
        (prey_idx, pred_idx) 0-based for each entry in best_vv.
    """

    best_vv: np.ndarray
    best_pp: np.ndarray | None
    ss: float
    ss_by_group: dict[int, float]
    n_iterations: int
    converged: bool
    fitted_scenario: RsimScenario | None
    link_map: list[tuple[int, int]]


# EwE6 eTimeSeriesType codes that AccumulateDataInfo compares with the model
# (cEcoSimModel.vb 2604-2615), and those of them whose catchability q
# PlotDataInfo estimates (relative series, vb 3891-3903). PyPath predicts the
# biomass and catch types; the others are reported and left out.
_EWE_REFERENCE_TYPES = {
    0: "relative biomass",
    1: "absolute biomass",
    5: "total mortality",
    6: "catch",
    -6: "catch (forcing)",
    61: "relative catch",
    7: "average weight",
    12: "landings",
    13: "discards",
    19: "total discards",
    20: "relative total discards",
    104: "fishing mortality (reference)",
}
_EWE_RELATIVE_TYPES = {0, 5, 7, 20, 61, 104}
_EWE_BIOMASS_TYPES = {0, 1}
_EWE_CATCH_TYPES = {6, -6, 61}
# EwE's annual data are compared at the start of this month (1-12), before the
# month's integration (cTimeSeriesDataStructures.ANNUAL_DATA_MONTH).
_EWE_ANNUAL_DATA_MONTH = 6


@dataclass
class EweSSResult:
    """EwE6's sum of squares of a run against reference time series.

    Parameters
    ----------
    total : float
        Sum over series of WtType x sum (Z - q)^2.
    by_series : dict[int, float]
        Each series' weighted contribution, by ``series_id``.
    by_group : dict[int, float]
        Summed by the series' 0-based group (EwE's SS by group).
    log_q : dict[int, float]
        ln q of each series: mean(Z) for relative series, 0 otherwise.
    n_obs : dict[int, int]
        Observations compared, by ``series_id``.
    """

    total: float
    by_series: dict[int, float]
    by_group: dict[int, float]
    log_q: dict[int, float]
    n_obs: dict[int, int]


def _compared_series(timeseries, warn: bool = True) -> list:
    """The reference series EwE compares with the run that PyPath can predict.

    Drivers are skipped silently (EwE does not compare them). With ``warn``,
    weighted series of reference types PyPath does not predict, and series
    linked to no model group, are reported.
    """
    compared = []
    for s in timeseries.series:
        if s.dat_type not in _EWE_REFERENCE_TYPES:
            continue  # a driver
        if s.dat_type not in _EWE_BIOMASS_TYPES | _EWE_CATCH_TYPES:
            if warn and s.weight > 0:
                logger.warning(
                    "Time series %r (%s) is not compared: PyPath does not "
                    "predict this EwE reference type yet, so it is left out of "
                    "the sum of squares",
                    s.name,
                    _EWE_REFERENCE_TYPES[s.dat_type],
                )
            continue
        if s.group_idx is None:
            if warn and s.weight > 0:
                logger.warning(
                    "Time series %r is linked to no model group; left out of "
                    "the sum of squares",
                    s.name,
                )
            continue
        compared.append(s)
    return compared


def ewe_sum_of_squares(output, timeseries, *, warn: bool = True) -> EweSSResult:
    """EwE6's sum of squares of an Ecosim run against reference time series.

    Follows ``cEcoSimModel.AccumulateDataInfo`` and ``PlotDataInfo`` (vb
    2575-2735, 3849-3966):

    - Annual data (``timeseries.interval == "annual"``) are compared once a
      year, at the start of month 6 before its integration: the biomass at the
      end of month (year - 1) x 12 + 5, i.e. ``out_Biomass`` row
      (year - 1) x 12 + 5. Time-step data compare value k with row k - 1 (the
      start of month k). Data years past the run are ignored.
    - Each observation gives Z = ln(obs / pred). Biomass series predict B;
      catch series predict B x F, F being the fishing mortality of the month
      before the sample (EwE sets FishTime after the comparison), from
      ``out_FishMort``. An observation of an unfished group (F = 0) counts
      with Z = 0, as in EwE. Observations <= 0 or missing are skipped.
    - Relative series (types 0, 61) subtract q = mean(Z); absolute ones do
      not. Each series adds WtType x sum (Z - q)^2, and every series counts,
      several per group included.

    Other reference types (total mortality, average weight, landings,
    discards, fishing mortality) are reported and left out; drivers (forced
    biomass, effort, ...) are not compared, as in EwE.

    Parameters
    ----------
    output : RsimOutput
        The run; needs ``out_Biomass`` and, for catch series,
        ``out_FishMort``.
    timeseries : EweTimeSeriesCollection
        The dataset, with each series' ``weight`` (EwE's WtType).
    warn : bool
        Report the weighted series left out (unsupported types, no group).
        ``fit_to_timeseries`` reports them once, not at every evaluation.

    Returns
    -------
    EweSSResult
    """
    biomass = np.asarray(output.out_Biomass, dtype=float)
    fish_mort = getattr(output, "out_FishMort", None)
    n_months = biomass.shape[0] - 1
    annual = getattr(timeseries, "interval", "annual") != "timestep"
    n_points = n_months // 12 if annual else n_months

    result = EweSSResult(0.0, {}, {}, {}, {})
    for s in _compared_series(timeseries, warn):
        col = s.group_idx + 1
        is_catch = s.dat_type in _EWE_CATCH_TYPES
        if is_catch and fish_mort is None:
            raise ValueError(
                f"Catch series {s.name!r} needs the run's monthly fishing "
                "mortality (out_FishMort)"
            )

        z = []
        for k, obs in enumerate(np.asarray(s.values, dtype=float)[:n_points], 1):
            if not obs > 0:  # also skips NaN
                continue
            row = (k - 1) * 12 + _EWE_ANNUAL_DATA_MONTH - 1 if annual else k - 1
            pred = biomass[row, col]
            if is_catch:
                f = float(fish_mort[row, col])
                if not f > 0:
                    z.append(0.0)
                    continue
                pred = pred * f
            # A crashed group (B = 0) gives a large, finite error.
            z.append(np.log(obs / max(pred, np.finfo(float).tiny)))

        z_arr = np.asarray(z, dtype=float)
        log_q = (
            float(z_arr.mean())
            if len(z_arr) and s.dat_type in _EWE_RELATIVE_TYPES
            else 0.0
        )
        ss = float(s.weight * np.sum((z_arr - log_q) ** 2))
        result.by_series[s.series_id] = ss
        result.log_q[s.series_id] = log_q
        result.n_obs[s.series_id] = len(z_arr)
        result.by_group[s.group_idx] = result.by_group.get(s.group_idx, 0.0) + ss
        result.total += ss
    return result


def fit_to_timeseries(
    rpath_model,
    params,
    timeseries,
    *,
    fit_vv: bool = True,
    fit_pp: bool = False,
    fit_groups: list[int] | None = None,
    vv_bounds: tuple = (1.0, 100.0),
    pp_bounds: tuple = (0.0, 2.0),
    method: str = "differential_evolution",
    max_iterations: int = 1000,
    verbose: bool = False,
) -> CalibrationResult:
    """Fit Ecosim parameters to observed time series data.

    Parameters
    ----------
    rpath_model : Rpath
        Balanced Ecopath model.
    params : RpathParams
        Model parameters.
    timeseries : EweTimeSeriesCollection or dict
        Observed time series. If dict, treated as {group_idx: np.array}
        of relative biomass (0-based group indices).
    fit_vv : bool
        Fit vulnerability (VV) parameters (default True).
    fit_pp : bool
        Fit primary production anomaly. Not yet implemented.
    fit_groups : list[int] or None
        0-based group indices to fit. None = all groups with observed data.
    vv_bounds : tuple
        (min, max) bounds for VV parameters.
    pp_bounds : tuple
        (min, max) bounds for PP parameters (reserved).
    method : str
        "differential_evolution" or "minimize" (L-BFGS-B).
    max_iterations : int
        Maximum optimizer iterations.
    verbose : bool
        Print progress.

    Returns
    -------
    CalibrationResult
    """
    from pypath.core.ecosim import rsim_run, rsim_scenario
    from pypath.core.timeseries import (
        DATTYPE_REL_BIOMASS,
        EweTimeSeries,
        EweTimeSeriesCollection,
        apply_timeseries_drivers,
    )

    if fit_pp:
        raise NotImplementedError(
            "Primary production fitting (fit_pp=True) is not yet implemented. "
            "It will be added in a future release."
        )

    # Convert dict input to EweTimeSeriesCollection
    if isinstance(timeseries, dict):
        series_list = []
        for grp_idx, values in timeseries.items():
            series_list.append(
                EweTimeSeries(
                    series_id=grp_idx,
                    name=f"Group_{grp_idx}",
                    dat_type=DATTYPE_REL_BIOMASS,
                    group_idx=grp_idx,
                    fleet_idx=None,
                    values=values,
                )
            )
        timeseries = EweTimeSeriesCollection(series_list)

    n_points = timeseries.n_timesteps
    if n_points < 2:
        raise ValueError("Need at least 2 observed timesteps for calibration.")
    annual = getattr(timeseries, "interval", "annual") != "timestep"
    n_years = n_points if annual else -(-n_points // 12)
    # rsim_scenario needs two years; only the supplied data are compared
    years = range(1, max(n_years, 2) + 1)

    # Build base scenario
    scenario = rsim_scenario(rpath_model, params, years=years)

    # Apply driver series
    apply_timeseries_drivers(scenario, timeseries)

    # The objective is EwE6's sum of squares (ewe_sum_of_squares): every
    # reference series counts, catch is compared with B x F, relative series
    # take q = mean ln(obs / pred), annual data are taken at the start of
    # month 6, each series weighted by its WtType.
    reference_groups = {
        s.group_idx for s in _compared_series(timeseries) if s.weight > 0
    }
    if not reference_groups:
        raise ValueError("No observed series found in timeseries collection.")

    if fit_groups is None:
        fit_groups = sorted(reference_groups)

    # Build link map: PreyFrom/PreyTo are 1-based
    link_indices = []
    link_map = []
    n_links = len(scenario.params.PreyFrom)
    for i in range(1, n_links):
        prey = scenario.params.PreyFrom[i] - 1  # to 0-based
        pred = scenario.params.PreyTo[i] - 1
        if prey in fit_groups or pred in fit_groups:
            link_indices.append(i)
            link_map.append((prey, pred))

    if not link_indices and fit_vv:
        raise ValueError("No pred-prey links found for the specified fit_groups.")

    bounds = []
    if fit_vv:
        bounds.extend([vv_bounds] * len(link_indices))

    n_vv = len(link_indices) if fit_vv else 0
    iteration_count = [0]

    def objective(param_vector):
        if fit_vv:
            for j, link_idx in enumerate(link_indices):
                scenario.params.VV[link_idx] = param_vector[j]

        try:
            output = rsim_run(scenario)
        except (RuntimeError, ValueError, FloatingPointError) as e:
            logger.warning("Simulation failed during optimization: %s", e)
            return 1e10

        ss = ewe_sum_of_squares(output, timeseries, warn=False).total
        if not np.isfinite(ss):
            return 1e10

        iteration_count[0] += 1
        if verbose and iteration_count[0] % 50 == 0:
            logger.info("Iteration %d: SS = %.6f", iteration_count[0], ss)

        return ss

    if method == "differential_evolution":
        from scipy.optimize import differential_evolution

        result = differential_evolution(
            objective,
            bounds=bounds,
            maxiter=max_iterations,
            seed=42,
            tol=1e-6,
            polish=True,
        )
        best_params = result.x
        converged = result.success
        final_ss = result.fun

    elif method == "minimize":
        from scipy.optimize import minimize

        x0 = np.array([scenario.params.VV[i] for i in link_indices])
        result = minimize(
            objective,
            x0,
            method="L-BFGS-B",
            bounds=bounds,
            options={"maxiter": max_iterations},
        )
        best_params = result.x
        converged = result.success
        final_ss = result.fun

    else:
        raise ValueError(
            f"Unknown method: {method!r}. Use 'differential_evolution' or 'minimize'."
        )

    best_vv = best_params[:n_vv] if fit_vv else np.array([])

    if fit_vv:
        for j, link_idx in enumerate(link_indices):
            scenario.params.VV[link_idx] = best_vv[j]

    # SS by group of the fitted parameters
    try:
        ss_by_group = ewe_sum_of_squares(
            rsim_run(scenario), timeseries, warn=False
        ).by_group
    except (RuntimeError, ValueError, FloatingPointError) as e:
        logger.warning("Final run of the fitted parameters failed: %s", e)
        ss_by_group = {}

    return CalibrationResult(
        best_vv=best_vv,
        best_pp=None,
        ss=final_ss,
        ss_by_group=ss_by_group,
        n_iterations=iteration_count[0],
        converged=converged,
        fitted_scenario=scenario,
        link_map=link_map,
    )
