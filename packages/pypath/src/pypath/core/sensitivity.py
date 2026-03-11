"""Sensitivity analysis for Ecopath/Ecosim models.

Morris elementary effects screening and optional Sobol variance-based
sensitivity analysis (requires SALib).
"""
from __future__ import annotations

import logging
import math
import warnings
from dataclasses import dataclass
from typing import Callable

import numpy as np

logger = logging.getLogger(__name__)

try:
    import SALib
    from SALib.analyze import sobol as salib_sobol
    from SALib.sample import saltelli

    HAS_SALIB = True
except ImportError:
    HAS_SALIB = False


@dataclass
class MorrisResult:
    """Morris elementary effects screening results."""

    parameter_names: list[str]
    mu_star: np.ndarray
    sigma: np.ndarray
    mu: np.ndarray
    output_name: str = "Biomass"


@dataclass
class SobolResult:
    """Sobol variance-based sensitivity indices."""

    parameter_names: list[str]
    S1: np.ndarray
    ST: np.ndarray
    S1_conf: np.ndarray
    ST_conf: np.ndarray
    output_name: str = "Biomass"


@dataclass
class SensitivityConfig:
    """Sensitivity analysis configuration."""

    method: str = "morris"
    n_trajectories: int = 10
    n_levels: int = 4
    n_samples: int = 1024
    seed: int | None = None
    n_jobs: int = 1
    output_variable: str = "Biomass"
    output_group_idx: int | None = None
    ecopath_only: bool = False
    ecosim_years: range | None = None


def _generate_morris_trajectories(
    k: int, n_trajectories: int, n_levels: int = 4,
    rng: np.random.Generator | None = None,
) -> np.ndarray:
    """Generate Morris OAT trajectories in the unit hypercube.

    Uses the standard Morris (1991) design: base points are chosen from
    the lower half of the grid ``{0, 1/(p-1), ..., (p-2)/(p-1)}`` and
    delta is always added (never subtracted) to guarantee a valid move.

    Returns array of shape (n_trajectories * (k+1), k).
    """
    if rng is None:
        rng = np.random.default_rng()

    delta = n_levels / (2 * (n_levels - 1))
    # Base points from the lower (p-1) grid values so base + delta <= 1
    base_grid = np.linspace(0, 1 - delta, n_levels - 1)
    trajectories = []

    for _ in range(n_trajectories):
        # Random base point — one per dimension independently
        base = rng.choice(base_grid, size=k)

        trajectory = [base.copy()]
        order = rng.permutation(k)
        current = base.copy()

        for param_idx in order:
            current = current.copy()
            current[param_idx] = current[param_idx] + delta
            trajectory.append(current.copy())

        trajectories.extend(trajectory)

    return np.array(trajectories)


def _compute_elementary_effects(
    trajectories: np.ndarray,
    y_values: np.ndarray,
    k: int,
    n_trajectories: int,
    n_levels: int = 4,
) -> MorrisResult:
    """Compute Morris elementary effects from trajectories and outputs."""
    delta = n_levels / (2 * (n_levels - 1))
    elementary_effects = [[] for _ in range(k)]

    for t in range(n_trajectories):
        start = t * (k + 1)
        for step in range(k):
            diff = trajectories[start + step + 1] - trajectories[start + step]
            changed_idx = np.argmax(np.abs(diff))
            ee = (y_values[start + step + 1] - y_values[start + step]) / delta
            elementary_effects[changed_idx].append(ee)

    mu_star = np.array([np.mean(np.abs(ee)) if ee else 0.0 for ee in elementary_effects])
    sigma = np.array([np.std(ee) if len(ee) > 1 else 0.0 for ee in elementary_effects])
    mu = np.array([np.mean(ee) if ee else 0.0 for ee in elementary_effects])

    return MorrisResult(
        parameter_names=[f"param_{i}" for i in range(k)],
        mu_star=mu_star,
        sigma=sigma,
        mu=mu,
    )


def run_sensitivity(
    params: "RpathParams",
    config: SensitivityConfig | None = None,
    *,
    pedigree_config: "PedigreeConfig | None" = None,
    progress_callback: Callable[[int, int], None] | None = None,
) -> MorrisResult | SobolResult:
    """Run sensitivity analysis on Ecopath/Ecosim model.

    Parameters
    ----------
    params : RpathParams
        Base Ecopath parameters.
    config : SensitivityConfig, optional
        Sensitivity configuration.
    pedigree_config : PedigreeConfig, optional
        EwE pedigree mapping.
    progress_callback : callable, optional
        Called with (current, total) after each model evaluation.

    Returns
    -------
    MorrisResult or SobolResult
    """
    from pypath.core.pedigree import (
        ScalarDistribution,
        apply_sample,
        build_distributions,
    )
    from pypath.core.ecopath import rpath as run_rpath

    if config is None:
        config = SensitivityConfig()

    if config.method == "sobol" and not HAS_SALIB:
        raise ImportError(
            "Install SALib for Sobol analysis: pip install SALib"
        )

    rng = np.random.default_rng(config.seed)
    distributions = build_distributions(params, pedigree_config)
    scalars = [d for d in distributions if isinstance(d, ScalarDistribution)]

    if len(scalars) == 0:
        raise ValueError("No scalar distributions to analyze (all CVs are 0).")

    k = len(scalars)
    param_names = [f"{d.param_name}_{d.group_idx}" for d in scalars]

    def _evaluate(x_unit: np.ndarray) -> float:
        """Map unit hypercube point to params, run model, extract output."""
        sample = {}
        for j, dist in enumerate(scalars):
            sigma = math.sqrt(math.log(1 + dist.cv**2))
            mu = math.log(dist.base_value) - sigma**2 / 2
            from scipy.stats import lognorm
            val = float(lognorm.ppf(x_unit[j], s=sigma, scale=math.exp(mu)))
            sample[(dist.param_name, dist.group_idx)] = val

        sampled_params = apply_sample(params, sample)
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                rpath_result = run_rpath(sampled_params)
            if config.output_group_idx is not None:
                return float(rpath_result.Biomass[config.output_group_idx])
            # Use nansum to skip fleet groups (which have NaN biomass)
            n_bio = int((params.model["Type"] != 3).sum())
            return float(np.nansum(rpath_result.Biomass[:n_bio]))
        except Exception:
            return np.nan

    if config.method == "morris":
        trajectories = _generate_morris_trajectories(
            k, config.n_trajectories, config.n_levels, rng,
        )
        n_evals = len(trajectories)
        y_values = np.empty(n_evals)
        for i in range(n_evals):
            y_values[i] = _evaluate(trajectories[i])
            if progress_callback:
                progress_callback(i + 1, n_evals)

        result = _compute_elementary_effects(
            trajectories, y_values, k, config.n_trajectories, config.n_levels,
        )
        result.parameter_names = param_names
        result.output_name = config.output_variable
        return result

    elif config.method == "sobol":
        n_runs = config.n_samples * (2 * k + 2)
        if n_runs > 10000:
            warnings.warn(
                f"Sobol analysis requires {n_runs} model evaluations.",
                UserWarning,
            )

        problem = {
            "num_vars": k,
            "names": param_names,
            "bounds": [[0.0, 1.0]] * k,
        }
        X = saltelli.sample(problem, config.n_samples, seed=config.seed)
        Y = np.empty(len(X))
        for i in range(len(X)):
            Y[i] = _evaluate(X[i])
            if progress_callback:
                progress_callback(i + 1, len(X))

        # Impute NaN with mean (SALib requires exact N*(2k+2) rows)
        nan_mask = np.isnan(Y)
        n_nan = np.sum(nan_mask)
        if n_nan > len(Y) * 0.5:
            raise RuntimeError("More than 50% of model evaluations failed.")
        if n_nan > 0:
            Y[nan_mask] = np.nanmean(Y)
            logger.warning("Imputed %d NaN evaluations with mean for Sobol.", n_nan)

        Si = salib_sobol.analyze(problem, Y)
        return SobolResult(
            parameter_names=param_names,
            S1=Si["S1"],
            ST=Si["ST"],
            S1_conf=Si["S1_conf"],
            ST_conf=Si["ST_conf"],
            output_name=config.output_variable,
        )

    raise ValueError(f"Unknown method: {config.method}")
