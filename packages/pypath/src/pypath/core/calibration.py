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


def _compute_ss(
    observed: dict[int, np.ndarray],
    predicted: dict[int, np.ndarray],
    weights: dict[int, float],
    relative: dict[int, bool],
) -> tuple[float, dict[int, float]]:
    """Compute EwE-standard log-ratio sum-of-squares.

    SS = sum_groups( weight_i * sum_t( (log(pred/obs))^2 ) )
    """
    total_ss = 0.0
    ss_by_group: dict[int, float] = {}

    for grp_idx, obs in observed.items():
        if grp_idx not in predicted:
            continue
        pred = predicted[grp_idx]
        w = weights.get(grp_idx, 1.0)

        n = min(len(obs), len(pred))
        obs_n = obs[:n]
        pred_n = pred[:n]

        valid = ~np.isnan(obs_n) & ~np.isnan(pred_n) & (obs_n > 0) & (pred_n > 0)

        if not np.any(valid):
            ss_by_group[grp_idx] = 0.0
            continue

        obs_v = obs_n[valid]
        pred_v = pred_n[valid]

        if relative.get(grp_idx, False):
            scale = np.mean(obs_v) / np.mean(pred_v)
            pred_v = pred_v * scale

        log_ratios = np.log(pred_v / obs_v)
        group_ss = w * np.sum(log_ratios**2)
        ss_by_group[grp_idx] = group_ss
        total_ss += group_ss

    return total_ss, ss_by_group


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

    n_obs_years = timeseries.n_timesteps
    if n_obs_years < 2:
        raise ValueError("Need at least 2 observed timesteps for calibration.")
    years = range(1, n_obs_years + 1)

    # Build base scenario
    scenario = rsim_scenario(rpath_model, params, years=years)

    # Apply driver series
    apply_timeseries_drivers(scenario, timeseries)

    # Build observed data dict
    observed: dict[int, np.ndarray] = {}
    weights: dict[int, float] = {}
    relative: dict[int, bool] = {}

    for s in timeseries.observed_biomass:
        if s.group_idx is not None:
            observed[s.group_idx] = s.values[:n_obs_years]
            weights[s.group_idx] = s.weight
            relative[s.group_idx] = s.dat_type == DATTYPE_REL_BIOMASS

    for s in timeseries.observed_catch:
        if s.group_idx is not None and s.group_idx not in observed:
            observed[s.group_idx] = s.values[:n_obs_years]
            weights[s.group_idx] = s.weight
            relative[s.group_idx] = False

    if not observed:
        raise ValueError("No observed series found in timeseries collection.")

    if fit_groups is None:
        fit_groups = list(observed.keys())

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

        predicted: dict[int, np.ndarray] = {}
        n_months = output.out_Biomass.shape[0]
        months_per_year = n_months // n_obs_years if n_obs_years > 0 else 12

        for grp_idx in observed:
            col = grp_idx + 1
            if col >= output.out_Biomass.shape[1]:
                continue
            annual = np.zeros(n_obs_years)
            for yr in range(n_obs_years):
                start = yr * months_per_year
                end = min(start + months_per_year, n_months)
                if start < n_months:
                    annual[yr] = np.mean(output.out_Biomass[start:end, col])
            predicted[grp_idx] = annual

        ss, _ = _compute_ss(observed, predicted, weights, relative)

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

    # Compute final SS breakdown
    try:
        output = rsim_run(scenario)
        predicted_final: dict[int, np.ndarray] = {}
        n_months = output.out_Biomass.shape[0]
        months_per_year = n_months // n_obs_years if n_obs_years > 0 else 12
        for grp_idx in observed:
            col = grp_idx + 1
            if col >= output.out_Biomass.shape[1]:
                continue
            annual = np.zeros(n_obs_years)
            for yr in range(n_obs_years):
                start = yr * months_per_year
                end = min(start + months_per_year, n_months)
                if start < n_months:
                    annual[yr] = np.mean(output.out_Biomass[start:end, col])
            predicted_final[grp_idx] = annual
        _, ss_by_group = _compute_ss(observed, predicted_final, weights, relative)
    except Exception:
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
