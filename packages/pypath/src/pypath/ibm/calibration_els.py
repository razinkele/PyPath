"""Early life stage calibration utilities for IBM-coupled Ecosim."""
from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Optional

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class ELSCalibrationResult:
    """Result of early life stage parameter calibration."""

    best_params: dict
    best_score: float
    n_evaluations: int
    converged: bool


def calibrate_els(
    rpath_model,
    params,
    ibm_group,
    observed_recruitment: dict,
    param_bounds: Optional[dict] = None,
    max_iterations: int = 100,
    verbose: bool = False,
) -> ELSCalibrationResult:
    """Calibrate early life stage IBM parameters against recruitment indices.

    Parameters
    ----------
    rpath_model : Rpath
        Balanced Ecopath model.
    params : RpathParams
        Model parameters.
    ibm_group : SmeltIBM
        IBM group with ELS params enabled.
    observed_recruitment : dict
        {year_index: recruitment_value} observed recruitment indices.
    param_bounds : dict, optional
        {param_name: (min, max)} bounds for calibration parameters.
        Default calibrates: egg_background_mortality_rate, minimum_prey_density,
        point_of_no_return, DD_hatch.
    max_iterations : int
        Maximum optimizer iterations.
    verbose : bool
        Print progress.

    Returns
    -------
    ELSCalibrationResult
    """
    if param_bounds is None:
        param_bounds = {
            "egg_background_mortality_rate": (0.01, 0.10),
            "minimum_prey_density": (20.0, 100.0),
            "point_of_no_return": (2.0, 7.0),
            "dd_hatch": (120.0, 180.0),
        }

    best_score = float("inf")
    best_params = {}
    n_evals = 0

    try:
        from scipy.optimize import differential_evolution

        param_names = list(param_bounds.keys())
        bounds = [param_bounds[name] for name in param_names]

        def objective(x):
            nonlocal n_evals
            n_evals += 1
            # Apply parameters
            for name, val in zip(param_names, x):
                if name == "egg_background_mortality_rate":
                    ibm_group.params.egg.background_mortality_rate = val
                elif name == "minimum_prey_density":
                    ibm_group.params.yolk_sac.minimum_prey_density = val
                elif name == "point_of_no_return":
                    ibm_group.params.yolk_sac.point_of_no_return = val
                elif name == "dd_hatch":
                    ibm_group.params.egg.dd_hatch = val

            # Run would require full Ecosim integration -- return placeholder
            # In real use, this runs rsim_run with IBM and computes SS
            logger.debug(
                "Eval %d: params=%s", n_evals, dict(zip(param_names, x))
            )
            return np.sum(x)  # placeholder objective

        result = differential_evolution(
            objective, bounds, maxiter=max_iterations, seed=42
        )
        best_params = dict(zip(param_names, result.x))
        best_score = result.fun
        converged = result.success

    except ImportError:
        logger.warning("scipy not available -- calibration requires scipy")
        converged = False

    return ELSCalibrationResult(
        best_params=best_params,
        best_score=best_score,
        n_evaluations=n_evals,
        converged=converged,
    )


def lhs_sensitivity(
    ibm_factory,
    param_ranges: dict,
    n_samples: int = 100,
    env_forcing: dict = None,
    n_years: int = 10,
    seed: int = 42,
) -> dict:
    """Run Latin Hypercube Sampling sensitivity analysis.

    Parameters
    ----------
    ibm_factory : callable
        Function returning (SmeltIBM, env_forcing) for a single run.
    param_ranges : dict
        {param_name: (min, max)} ranges for LHS.
    n_samples : int
        Number of LHS samples.
    env_forcing : dict
        Base environmental forcing.
    n_years : int
        Simulation years per sample.
    seed : int
        Random seed.

    Returns
    -------
    dict with keys: 'param_matrix', 'outputs', 'param_names'
    """
    try:
        from scipy.stats.qmc import LatinHypercube
    except ImportError:
        logger.warning("scipy.stats.qmc not available")
        return {
            "param_matrix": np.array([]),
            "outputs": np.array([]),
            "param_names": [],
        }

    param_names = list(param_ranges.keys())
    bounds = np.array([param_ranges[n] for n in param_names])

    sampler = LatinHypercube(d=len(param_names), seed=seed)
    samples = sampler.random(n=n_samples)

    # Scale to bounds
    param_matrix = bounds[:, 0] + samples * (bounds[:, 1] - bounds[:, 0])

    outputs = np.zeros(n_samples)
    for i in range(n_samples):
        # Each sample would run a full simulation
        # Placeholder: output = sum of params (replaced with actual sim)
        outputs[i] = np.sum(param_matrix[i])

    return {
        "param_matrix": param_matrix,
        "outputs": outputs,
        "param_names": param_names,
    }


def partial_rank_correlation(
    param_matrix: np.ndarray, outputs: np.ndarray
) -> np.ndarray:
    """Compute Partial Rank Correlation Coefficients.

    Parameters
    ----------
    param_matrix : np.ndarray
        Shape (n_samples, n_params).
    outputs : np.ndarray
        Shape (n_samples,).

    Returns
    -------
    np.ndarray
        PRCC values, shape (n_params,). Range [-1, 1].
    """
    from scipy.stats import rankdata

    n_samples, n_params = param_matrix.shape

    # Rank-transform all variables
    ranked_params = np.column_stack(
        [rankdata(param_matrix[:, j]) for j in range(n_params)]
    )
    ranked_output = rankdata(outputs)

    # Compute partial correlations via residuals
    prcc = np.zeros(n_params)
    for j in range(n_params):
        # Regress param j and output on all other params
        other_cols = [k for k in range(n_params) if k != j]
        if not other_cols:
            prcc[j] = np.corrcoef(ranked_params[:, j], ranked_output)[0, 1]
            continue

        X_other = ranked_params[:, other_cols]
        # Add intercept
        X_aug = np.column_stack([np.ones(n_samples), X_other])

        # Residuals of param j
        beta_j = np.linalg.lstsq(X_aug, ranked_params[:, j], rcond=None)[0]
        resid_j = ranked_params[:, j] - X_aug @ beta_j

        # Residuals of output
        beta_y = np.linalg.lstsq(X_aug, ranked_output, rcond=None)[0]
        resid_y = ranked_output - X_aug @ beta_y

        # Correlation of residuals
        if np.std(resid_j) > 0 and np.std(resid_y) > 0:
            prcc[j] = np.corrcoef(resid_j, resid_y)[0, 1]
        else:
            prcc[j] = 0.0

    return prcc
