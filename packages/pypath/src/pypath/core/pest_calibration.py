"""PEST Calibration Framework for Ecosim Model.

Provides tools for Parameter ESTimation (PEST) calibration of Ecosim models
using automated parameter adjustment to match observed time series data.

PEST is a parameter estimation and uncertainty analysis tool. This module
implements PEST-style workflows in Python for Ecosim calibration:

1. Define parameters (VV links, PP driving factors, fishing effort)
2. Define observations (biomass time series)
3. Run PEST algorithm (gradient-based or stochastic)
4. Evaluate parameter sensitivity and uncertainty

Reference:
    Doherty, J. (2015). PEST, Model-Independent Parameter Estimation,
    User Manual: 6th Edition.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict

import numpy as np
import pandas as pd

from pypath.core.ecosim import RsimScenario, rsim_run, rsim_scenario

logger = logging.getLogger(__name__)


@dataclass
class PestParameter:
    """PEST-style parameter definition.

    Attributes
    ----------
    name : str
        Parameter name (e.g., "VV_link_0_1" or "PP_fish")
    initial : float
        Initial parameter value
    lower : float
        Lower bound for parameter
    upper : float
        Upper bound for parameter
    relative_change : float
        Relative change limit for Gauss-Levenberg-Marquardt (0.01 = 1%)
    p_type : str
        Parameter type: "vv" (vulnerability), "pp" (primary production),
        "fish" (fishing effort), etc.
    group_indices : tuple[int, int] | int | None
        Associated group indices (predator, prey) for VV, or single index for PP
    """

    name: str
    initial: float
    lower: float
    upper: float
    relative_change: float = 0.01
    p_type: str = "vv"
    group_indices: tuple[int, int] | int | None = None


@dataclass
class PestObservation:
    """PEST-style observation definition.

    Attributes
    ----------
    name : str
        Observation name (e.g., "Biomass_Fish_Year_5")
    value : float
        Observed value
    weight : float
        Observation weight (inverse of expected error variance)
    o_type : str
        Observation type: "biomass", "catch", "catch_rate", etc.
    group_index : int
        0-based group index
    time_year : float
        Time of observation in simulation years
    """

    name: str
    value: float
    weight: float = 1.0
    o_type: str = "biomass"
    group_index: int = 0
    time_year: float = 0.0


@dataclass
class PestCalibrationResult:
    """Result of PEST calibration run."""

    best_params: Dict[str, float]
    best_parmvec: np.ndarray
    residuals: Dict[str, float]
    sum_of_squared_residuals: float
    n_iterations: int
    converged: bool
    jacobian: np.ndarray | None = None
    covariance: np.ndarray | None = None
    uncertainty: Dict[str, float] | None = None
    correlation: pd.DataFrame | None = None
    fitted_scenario: RsimScenario | None = None
    calibration_time: float = 0.0


class PestCalibrator:
    """PEST-based parameter calibration for Ecosim models."""

    def __init__(self, rpath_model, params):
        """Initialize PEST calibrator.

        Parameters
        ----------
        rpath_model : Rpath
            Balanced Ecopath model from rpath()
        params : RpathParams
            Model parameters from create_rpath_params() or read_ewemdb()
        """
        self.rpath_model = rpath_model
        self.params = params
        self.pest_parameters = []
        self.pest_observations = []
        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")

    def add_parameter(self, param):
        """Add a PEST parameter."""
        self.pest_parameters.append(param)
        self.logger.debug(f"Added parameter: {param.name}")

    def add_observation(self, obs):
        """Add a PEST observation."""
        self.pest_observations.append(obs)
        self.logger.debug(f"Added observation: {obs.name}")

    def add_parameters_from_dict(self, params_dict, p_type="vv"):
        """Add multiple parameters from a dictionary."""
        for name, (initial, lower, upper) in params_dict.items():
            param = PestParameter(
                name=name,
                initial=initial,
                lower=lower,
                upper=upper,
                p_type=p_type,
            )
            self.add_parameter(param)

    def add_observations_from_dataframe(self, df, value_col="value", weight_col="weight", group_col="group_index", time_col="time_year", o_type="biomass"):
        """Add observations from a pandas DataFrame."""
        for _, row in df.iterrows():
            obs = PestObservation(
                name=row.get("name", f"Obs_{len(self.pest_observations)}"),
                value=row[value_col],
                weight=row.get(weight_col, 1.0),
                o_type=o_type,
                group_index=int(row[group_col]),
                time_year=float(row[time_col]),
            )
            self.add_observation(obs)

    def calibrate(self, scenario_factory, method="glm", max_iterations=20, tolerance=0.01, lambdas=None, verbose=True):
        """Run PEST calibration."""
        if method == "glm":
            return self._calibrate_glm(
                scenario_factory,
                max_iterations=max_iterations,
                tolerance=tolerance,
                lambdas=lambdas,
                verbose=verbose,
            )
        else:
            raise ValueError(f"Unknown calibration method: {method}")

    def _calibrate_glm(self, scenario_factory, max_iterations=20, tolerance=0.01, lambdas=None, verbose=True):
        """Gauss-Levenberg-Marquardt calibration."""
        start_time = time.time()

        parmvec = np.array([p.initial for p in self.pest_parameters])
        n_parms = len(self.pest_parameters)
        n_obs = len(self.pest_observations)

        if verbose:
            self.logger.info("PEST Calibration (GLM)")
            self.logger.info(f"  Parameters: {n_parms}")
            self.logger.info(f"  Observations: {n_obs}")

        if lambdas is None:
            lambdas = np.ones(max_iterations) * 10.0
            for i in range(1, max_iterations):
                lambdas[i] = lambdas[i - 1] * 0.9

        ssr_history = []

        for iteration in range(max_iterations):
            base_scenario = rsim_scenario(self.rpath_model, self.params, years=np.arange(1, 21))
            ssr, residuals, jacobian = self._evaluate_residuals_and_jacobian(scenario_factory, base_scenario, parmvec)
            ssr_history.append(ssr)

            if verbose:
                self.logger.info(f"  Iteration {iteration + 1}: SSR = {ssr:.6e}")

            if iteration > 0:
                delta_ssr = abs(ssr_history[-2] - ssr_history[-1]) / (ssr_history[-2] + 1e-10)
                if delta_ssr < tolerance:
                    if verbose:
                        self.logger.info(f"  Converged after {iteration + 1} iterations")
                    break

            lamb = lambdas[iteration]
            try:
                hessian = jacobian.T @ jacobian
                damped = hessian + lamb * np.eye(n_parms)
                gradient = -jacobian.T @ residuals
                delta_parm = np.linalg.solve(damped, gradient)
                parmvec = self._apply_bounds(parmvec + delta_parm)
            except np.linalg.LinAlgError as e:
                self.logger.warning(f"  Singular matrix at iteration {iteration}: {e}")
                break

        base_scenario = rsim_scenario(self.rpath_model, self.params, years=np.arange(1, 21))
        final_ssr, final_residuals, final_jacobian = self._evaluate_residuals_and_jacobian(scenario_factory, base_scenario, parmvec)

        covariance, correlation, uncertainty = self._compute_uncertainty(final_jacobian, final_ssr, n_obs, n_parms)

        best_params = {p.name: v for p, v in zip(self.pest_parameters, parmvec)}
        fitted_scenario = scenario_factory(base_scenario, parmvec)

        result = PestCalibrationResult(
            best_params=best_params,
            best_parmvec=parmvec,
            residuals={self.pest_observations[i].name: final_residuals[i] for i in range(min(len(final_residuals), n_obs))},
            sum_of_squared_residuals=final_ssr,
            n_iterations=iteration + 1,
            converged=(delta_ssr < tolerance) if iteration > 0 else False,
            jacobian=final_jacobian,
            covariance=covariance,
            uncertainty=uncertainty,
            correlation=correlation,
            fitted_scenario=fitted_scenario,
            calibration_time=time.time() - start_time,
        )

        if verbose:
            self.logger.info(f"Calibration Complete - Final SSR: {final_ssr:.6e}, Time: {result.calibration_time:.2f}s")

        return result

    def _evaluate_residuals_and_jacobian(self, scenario_factory, base_scenario, parmvec):
        """Evaluate residuals and Jacobian."""
        n_obs = len(self.pest_observations)
        n_parms = len(self.pest_parameters)
        eps = 1e-6
        jacobian = np.zeros((n_obs, n_parms))

        for i_param in range(n_parms):
            parmvec_plus = parmvec.copy()
            parmvec_plus[i_param] += eps * max(abs(parmvec[i_param]), 1.0)
            scenario_plus = scenario_factory(base_scenario, parmvec_plus)
            output_plus = rsim_run(scenario_plus, method="RK4")

            parmvec_minus = parmvec.copy()
            parmvec_minus[i_param] -= eps * max(abs(parmvec[i_param]), 1.0)
            scenario_minus = scenario_factory(base_scenario, parmvec_minus)
            output_minus = rsim_run(scenario_minus, method="RK4")

            for j_obs, obs in enumerate(self.pest_observations):
                pred_plus = self._extract_prediction(output_plus, obs)
                pred_minus = self._extract_prediction(output_minus, obs)
                if pred_plus is not None and pred_minus is not None:
                    jacobian[j_obs, i_param] = (pred_plus - pred_minus) / (parmvec_plus[i_param] - parmvec_minus[i_param])

        scenario_best = scenario_factory(base_scenario, parmvec)
        output_best = rsim_run(scenario_best, method="RK4")
        residuals = np.zeros(n_obs)
        ssr = 0.0

        for i_obs, obs in enumerate(self.pest_observations):
            pred = self._extract_prediction(output_best, obs)
            if pred is not None:
                residual = (obs.value - pred) * np.sqrt(obs.weight)
                residuals[i_obs] = residual
                ssr += residual**2

        return ssr, residuals, jacobian

    def _extract_prediction(self, output, obs):
        """Extract predicted value from output."""
        if not hasattr(output, "out_Biomass"):
            return None
        month_idx = int(obs.time_year * 12)
        biomass_ts = output.out_Biomass[:, :-1]
        if month_idx >= len(biomass_ts) or obs.group_index >= biomass_ts.shape[1]:
            return None
        return float(biomass_ts[month_idx, obs.group_index])

    def _apply_bounds(self, parmvec):
        """Apply parameter bounds."""
        bounded = parmvec.copy()
        for i, param in enumerate(self.pest_parameters):
            bounded[i] = np.clip(bounded[i], param.lower, param.upper)
        return bounded

    def _compute_uncertainty(self, jacobian, ssr, n_obs, n_parms):
        """Compute parameter uncertainty."""
        try:
            hessian = jacobian.T @ jacobian
            dof = max(n_obs - n_parms, 1)
            sigma_sq = ssr / dof
            cov = np.linalg.inv(hessian) * sigma_sq
            std_dev = np.sqrt(np.diag(cov))
            uncertainty = {p.name: float(std_dev[i]) for i, p in enumerate(self.pest_parameters)}
            corr = cov / np.outer(std_dev, std_dev)
            corr_df = pd.DataFrame(corr, index=[p.name for p in self.pest_parameters], columns=[p.name for p in self.pest_parameters]) if len(self.pest_parameters) > 0 else None
            return cov, corr_df, uncertainty
        except np.linalg.LinAlgError:
            return None, None, None

    def report(self, result, filepath=None):
        """Generate PEST calibration report."""
        report = [
            "=" * 70,
            "PEST CALIBRATION REPORT - Ecosim Model",
            "=" * 70,
            "",
            "CALIBRATION SUMMARY",
            f"  Converged: {result.converged}",
            f"  Iterations: {result.n_iterations}",
            f"  Time (s): {result.calibration_time:.2f}",
            f"  Final SSR: {result.sum_of_squared_residuals:.6e}",
            "",
            "BEST-FIT PARAMETERS",
        ]

        for name, value in result.best_params.items():
            unc = result.uncertainty.get(name, np.nan) if result.uncertainty else np.nan
            report.append(f"  {name:30s} = {value:12.6f} ± {unc:10.6f}")

        report_text = "\n".join(report)
        if filepath:
            Path(filepath).write_text(report_text)
            self.logger.info(f"Report written to {filepath}")
        return report_text
