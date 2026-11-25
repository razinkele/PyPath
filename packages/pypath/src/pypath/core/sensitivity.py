"""Sensitivity analysis for Ecopath/Ecosim models.

Morris elementary effects screening and optional Sobol variance-based
sensitivity analysis (requires SALib). ``SensitivityConfig.n_jobs`` runs the
evaluations in worker processes; the results are the serial run's, bit for
bit.
"""

from __future__ import annotations

import logging
import math
import warnings
from dataclasses import dataclass
from typing import TYPE_CHECKING, Callable

import numpy as np

from pypath.core._lazy import LazyModule, is_installed
from pypath.core._parallel import (
    NUMPY_DEFAULT_ERRSTATE,
    mp_context,
    outputs_in_order,
    resolve_n_jobs,
)

if TYPE_CHECKING:
    from pypath.core.params import RpathParams
    from pypath.core.pedigree import PedigreeConfig

logger = logging.getLogger(__name__)

# SALib (which loads scipy.stats) is imported when a Sobol analysis runs
HAS_SALIB = is_installed("SALib")
salib_sobol = LazyModule("SALib.analyze.sobol")
sobol_sampler = LazyModule("SALib.sample.sobol")


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
    """Sensitivity analysis configuration.

    Every evaluation balances the model with Ecopath and reads its biomass.

    Attributes
    ----------
    method : str
        "morris" (elementary effects screening) or "sobol" (variance-based
        indices, needs SALib).
    n_trajectories : int
        Morris trajectories; each costs one evaluation per parameter, plus
        one.
    n_levels : int
        Morris grid levels ``p``. The unit grid maps linearly onto each
        parameter's quantiles ``1/(2p)`` to ``1 - 1/(2p)``, the centres of
        the outermost of ``p`` equally likely bins.
    n_samples : int
        Sobol base sample size ``N``, best a power of 2. The analysis costs
        ``N * (2k + 2)`` evaluations for ``k`` parameters.
    seed : int, optional
        Seed for the design and for Sobol's bootstrap confidence intervals;
        None draws fresh ones.
    n_jobs : int
        Worker processes for the evaluations. 1 (the default) runs them
        serially in this process. A larger number runs that many processes,
        never more than there are evaluations; -1 uses one per CPU this
        process may use. The design is drawn up front from ``seed``, each
        point is evaluated with warnings ignored and numpy's default
        floating-point error handling, and the outputs come back in order,
        so a parallel run gives the serial run's results, bit for bit.
        Starting workers takes from a fraction of a second to about 2 s, so
        they pay off on long designs (Sobol, or Morris on a large model).
    output_variable : str
        Name stored as the result's ``output_name``.
    output_group_idx : int, optional
        Group whose biomass is the output; None sums the biomass of every
        group that is not a fleet.
    ecopath_only : bool
        Not used: every evaluation is an Ecopath balance.
    ecosim_years : range, optional
        Not used: every evaluation is an Ecopath balance.
    mp_context : str, optional
        The multiprocessing start method for the workers: "forkserver",
        "spawn" or "fork". None uses forkserver on Linux and spawn on macOS
        and Windows, as CPython's own defaults do from Python 3.14.

        forkserver and spawn start clean worker processes:

        - a script that calls ``run_sensitivity`` must do so under
          ``if __name__ == "__main__":``;
        - ``params`` must be picklable;
        - workers do not see the caller's warning filters, logging
          configuration or later changes to ``os.environ``;
        - starting the workers takes about 2 s per call when the caller is
          not a script that imports pypath (a notebook, for example).

        "fork" starts workers fastest but is unsafe in a multi-threaded
        process (Python warns, and numpy's BLAS threads count), so ask for it
        only from a single-threaded script.
    """

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
    mp_context: str | None = None


def _generate_morris_trajectories(
    k: int,
    n_trajectories: int,
    n_levels: int = 4,
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


def _lognormal_shape(cv: float, base_value: float) -> tuple[float, float]:
    """Shape ``s`` and ``scale`` of the lognormal with mean ``base_value``
    and coefficient of variation ``cv``, as scipy's ``lognorm`` takes them."""
    sigma = math.sqrt(math.log(1 + cv**2))
    mu = math.log(base_value) - sigma**2 / 2
    return sigma, math.exp(mu)


def _morris_quantiles(trajectories: np.ndarray, n_levels: int) -> np.ndarray:
    """Quantiles that Morris unit-grid points stand for.

    The unit interval maps linearly onto ``[1/(2p), 1 - 1/(2p)]`` for ``p``
    levels: the grid's ends become the centres of the outermost of ``p``
    equally likely bins, so every point is a finite parameter value (the
    lognormal quantile function sends 0 and 1 to 0 and infinity).
    """
    edge = 1.0 / (2 * n_levels)
    return edge + trajectories * (1.0 - 2 * edge)


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

    with np.errstate(invalid="ignore"):
        for t in range(n_trajectories):
            start = t * (k + 1)
            for step in range(k):
                diff = trajectories[start + step + 1] - trajectories[start + step]
                changed_idx = np.argmax(np.abs(diff))
                y1 = y_values[start + step + 1]
                y0 = y_values[start + step]
                if np.isnan(y1) or np.isnan(y0):
                    continue
                ee = (y1 - y0) / delta
                elementary_effects[changed_idx].append(ee)

    mu_star = np.array(
        [np.mean(np.abs(ee)) if ee else 0.0 for ee in elementary_effects]
    )
    sigma = np.array([np.std(ee) if len(ee) > 1 else 0.0 for ee in elementary_effects])
    mu = np.array([np.mean(ee) if ee else 0.0 for ee in elementary_effects])

    return MorrisResult(
        parameter_names=[f"param_{i}" for i in range(k)],
        mu_star=mu_star,
        sigma=sigma,
        mu=mu,
    )


def _evaluate_point(params, keys, values, output_group_idx, bio_idx) -> float:
    """One design point's output: the model's biomass, or NaN when the
    balance fails.

    Runs with warnings ignored and numpy's default floating-point error
    handling, so the output does not depend on the process it runs in.
    """
    from pypath.core.ecopath import rpath
    from pypath.core.pedigree import apply_sample

    with warnings.catch_warnings(), np.errstate(**NUMPY_DEFAULT_ERRSTATE):
        warnings.simplefilter("ignore")
        sample = {key: float(v) for key, v in zip(keys, values)}
        sampled_params = apply_sample(params, sample)
        try:
            rpath_result = rpath(sampled_params)
            if output_group_idx is not None:
                return float(rpath_result.Biomass[output_group_idx])
            # Every group but the fleets (nansum skips missing biomass)
            return float(np.nansum(rpath_result.Biomass[bio_idx]))
        except Exception:
            return np.nan


# A worker process's model and output settings, set once when the worker
# starts so that each task carries only its point's parameter values
_WORKER: dict = {}


def _init_worker(params, keys, output_group_idx, bio_idx):
    _WORKER.update(
        params=params, keys=keys, output_group_idx=output_group_idx, bio_idx=bio_idx
    )


def _evaluate_in_worker(values):
    return _evaluate_point(
        _WORKER["params"],
        _WORKER["keys"],
        values,
        _WORKER["output_group_idx"],
        _WORKER["bio_idx"],
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
        Called with (current, total) after each model evaluation, in order,
        in this process (also when ``config.n_jobs`` runs workers).

    Returns
    -------
    MorrisResult or SobolResult
    """
    from pypath.core.pedigree import ScalarDistribution, build_distributions

    if config is None:
        config = SensitivityConfig()

    if config.method not in ("morris", "sobol"):
        raise ValueError(f"Unknown method: {config.method}")
    if config.method == "sobol" and not HAS_SALIB:
        raise ImportError("Install SALib for Sobol analysis: pip install SALib")
    context = mp_context(config.mp_context)  # refuses an unknown method

    rng = np.random.default_rng(config.seed)
    distributions = build_distributions(params, pedigree_config)
    scalars = [d for d in distributions if isinstance(d, ScalarDistribution)]

    if len(scalars) == 0:
        raise ValueError("No scalar distributions to analyze (all CVs are 0).")

    k = len(scalars)
    param_names = [f"{d.param_name}_{d.group_idx}" for d in scalars]
    keys = [(d.param_name, d.group_idx) for d in scalars]
    shapes = [_lognormal_shape(d.cv, d.base_value) for d in scalars]
    sigmas = np.array([sigma for sigma, _ in shapes])
    scales = np.array([scale for _, scale in shapes])
    # Every group but the fleets, by type: Ecopath keeps the model's order
    bio_idx = np.flatnonzero(
        params.model["Type"].to_numpy(dtype=float, na_value=np.nan) != 3
    )

    if config.method == "morris":
        trajectories = _generate_morris_trajectories(
            k,
            config.n_trajectories,
            config.n_levels,
            rng,
        )
        points = _morris_quantiles(trajectories, config.n_levels)
    else:
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
        # Scrambled Sobol' points, reproducible from the seed
        points = sobol_sampler.sample(problem, config.n_samples, seed=config.seed)

    # Every point's parameter values in one vectorised call; it gives the
    # same values, bit for bit, as the quantile function point by point
    from scipy.stats import lognorm  # here: a sixth of the package's import time

    values = lognorm.ppf(points, s=sigmas, scale=scales)
    n_evals = len(values)
    n_workers = resolve_n_jobs(config.n_jobs, n_evals)

    y_values = np.empty(n_evals)
    with outputs_in_order(
        _evaluate_in_worker,
        values,
        n_workers=n_workers,
        context=context,
        serial=lambda row: _evaluate_point(
            params, keys, row, config.output_group_idx, bio_idx
        ),
        # Workers get the model once; each task carries a point's values
        initializer=_init_worker,
        initargs=(params, keys, config.output_group_idx, bio_idx),
        broken_message=(
            "A sensitivity worker process died. It may have run out of "
            "memory, or failed to unpickle the model. Try n_jobs=1, or "
            "mp_context='fork' from a single-threaded script."
        ),
    ) as outputs:
        for i, y in enumerate(outputs):
            y_values[i] = y
            if progress_callback:
                progress_callback(i + 1, n_evals)

    if config.method == "morris":
        result = _compute_elementary_effects(
            trajectories,
            y_values,
            k,
            config.n_trajectories,
            config.n_levels,
        )
        result.parameter_names = param_names
        result.output_name = config.output_variable
        return result

    # Impute NaN with mean (SALib requires exact N*(2k+2) rows)
    nan_mask = np.isnan(y_values)
    n_nan = np.sum(nan_mask)
    if n_nan > len(y_values) * 0.5:
        raise RuntimeError("More than 50% of model evaluations failed.")
    if n_nan > 0:
        y_values[nan_mask] = np.nanmean(y_values)
        logger.warning("Imputed %d NaN evaluations with mean for Sobol.", n_nan)

    # The seed also fixes the bootstrap behind the confidence intervals. SALib
    # reads a seed of 0 as no seed, so hand it a generator: for any other
    # seed it draws the same numbers as SALib's own default_rng(seed)
    bootstrap_rng = None if config.seed is None else np.random.default_rng(config.seed)
    Si = salib_sobol.analyze(problem, y_values, seed=bootstrap_rng)
    return SobolResult(
        parameter_names=param_names,
        S1=Si["S1"],
        ST=Si["ST"],
        S1_conf=Si["S1_conf"],
        ST_conf=Si["ST_conf"],
        output_name=config.output_variable,
    )
