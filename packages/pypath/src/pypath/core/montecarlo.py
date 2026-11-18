"""Monte Carlo uncertainty analysis for Ecopath/Ecosim.

Runs ensemble simulations with parameter sampling from pedigree-defined
distributions. ``MCConfig.n_jobs`` runs the samples in worker processes; the
results are the serial run's, sample for sample.
"""

from __future__ import annotations

import logging
import multiprocessing
import numbers
import os
import warnings
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Callable

import numpy as np
import pandas as pd

if TYPE_CHECKING:
    from pypath.core.params import RpathParams
    from pypath.core.pedigree import PedigreeConfig

logger = logging.getLogger(__name__)

# Whether joblib is installed. Kept for compatibility (pypath.core exports
# it); parallel runs use concurrent.futures and do not need joblib.
try:
    from joblib import Parallel, delayed  # noqa: F401  (availability probe)

    HAS_JOBLIB = True
except ImportError:
    HAS_JOBLIB = False


@dataclass
class MCConfig:
    """Monte Carlo run configuration.

    n_jobs : int
        Worker processes for the samples. 1 (the default) runs them serially
        in this process; a larger number runs that many processes, never more
        than there are samples; -1 uses one per CPU. The samples are drawn up
        front from ``seed`` and come back in order, so the results are the
        serial run's, bit for bit.
    mp_context : str, optional
        The multiprocessing start method for the workers: "forkserver",
        "spawn" or "fork". None uses forkserver where the platform has it
        (Linux, macOS) and spawn elsewhere (Windows). Both start clean worker
        processes: a script that calls ``run_montecarlo`` must do so under
        ``if __name__ == "__main__":``, ``params`` and ``mediation`` must be
        picklable, and warnings raised in a worker do not pass through the
        caller's warning filters. "fork" starts workers fastest but is unsafe
        in a multi-threaded process (Python warns, and numpy's BLAS threads
        count), so ask for it only from a single-threaded script.
    """

    n_samples: int = 1000
    method: str = "lhs"
    seed: int | None = None
    ecopath_only: bool = False
    ecosim_years: range | None = None
    store_runs: bool = False
    n_jobs: int = 1
    mediation: Any = None
    ecosim_method: str = "RK4"
    eco_area: float = 1.0
    mp_context: str | None = None


@dataclass
class MCResult:
    """Monte Carlo ensemble results."""

    n_total: int
    n_feasible: int
    n_ecosim: int
    ecopath_stats: dict[str, pd.DataFrame]
    ecosim_stats: dict[str, np.ndarray] | None
    ecopath_runs: list[dict] | None
    ecosim_runs: list[np.ndarray] | None
    feasibility_rate: float
    parameter_samples: pd.DataFrame | None

    def to_dataframe(self) -> pd.DataFrame:
        """Return ecopath_stats as a single flat DataFrame."""
        frames = []
        for param_name, df in self.ecopath_stats.items():
            df_copy = df.copy()
            df_copy.insert(0, "parameter", param_name)
            frames.append(df_copy)
        if frames:
            return pd.concat(frames, ignore_index=True)
        return pd.DataFrame()

    def to_dict(self) -> dict:
        """Return JSON-serializable summary dict."""
        return {
            "n_total": self.n_total,
            "n_feasible": self.n_feasible,
            "n_ecosim": self.n_ecosim,
            "feasibility_rate": self.feasibility_rate,
            "ecopath_stats": {
                k: v.to_dict(orient="list") for k, v in self.ecopath_stats.items()
            },
        }


def _run_single_ecopath(sampled_params, eco_area):
    """Run a single Ecopath mass balance, return result or None."""
    from pypath.core.ecopath import rpath

    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            return rpath(sampled_params, eco_area=eco_area)
    except Exception:
        return None


def _run_single_ecosim(rpath_result, sampled_params, config):
    """Run a single Ecosim simulation, return out_Biomass or None."""
    from pypath.core.ecosim import rsim_run, rsim_scenario

    try:
        years = config.ecosim_years if config.ecosim_years is not None else range(1, 11)
        scenario = rsim_scenario(rpath_result, sampled_params, years=years)
        result = rsim_run(
            scenario, method=config.ecosim_method, mediation=config.mediation
        )
        return result.out_Biomass
    except Exception:
        return None


def _resolve_n_jobs(n_jobs, n_samples: int) -> int:
    """Number of worker processes for ``n_jobs``: 1 runs serially.

    ``n_jobs`` >= 1 asks for that many workers and -1 for one per CPU. The
    count never exceeds the number of samples, and is at least 1.
    """
    if isinstance(n_jobs, bool) or not isinstance(n_jobs, numbers.Integral):
        raise ValueError(f"n_jobs must be an integer >= 1 or -1, not {n_jobs!r}")
    if n_jobs == -1:
        wanted = os.cpu_count() or 1
    elif n_jobs >= 1:
        wanted = int(n_jobs)
    else:
        raise ValueError(f"n_jobs must be an integer >= 1 or -1, not {n_jobs!r}")
    return max(1, min(wanted, n_samples))


def _mp_context(name: str | None):
    """The workers' start method: forkserver where available, else spawn."""
    if name is None:
        methods = multiprocessing.get_all_start_methods()
        name = "forkserver" if "forkserver" in methods else "spawn"
    return multiprocessing.get_context(name)


def _evaluate_sample(params, sample, config, n_groups):
    """One sample: (feasible, Ecopath biomass, Ecosim out_Biomass).

    The biomass is None when the balance has none, and the Ecosim biomass is
    None for an infeasible balance, an Ecopath-only run or a failed Ecosim.
    """
    from pypath.core.pedigree import apply_sample

    sampled_params = apply_sample(params, sample)
    rpath_result = _run_single_ecopath(sampled_params, config.eco_area)
    if rpath_result is None:
        return False, None, None
    bio = (
        rpath_result.Biomass[:n_groups].copy()
        if hasattr(rpath_result, "Biomass")
        else None
    )
    ecosim_bio = None
    if not config.ecopath_only:
        ecosim_bio = _run_single_ecosim(rpath_result, sampled_params, config)
    return True, bio, ecosim_bio


# A worker process's model and config, set once when the worker starts so
# that each task carries only its sample
_WORKER: dict = {}


def _init_worker(params, config, n_groups):
    _WORKER.update(params=params, config=config, n_groups=n_groups)


def _evaluate_in_worker(sample):
    return _evaluate_sample(
        _WORKER["params"], sample, _WORKER["config"], _WORKER["n_groups"]
    )


def run_montecarlo(
    params: "RpathParams",
    config: MCConfig | None = None,
    *,
    pedigree_config: "PedigreeConfig | None" = None,
    progress_callback: Callable[[int, int], None] | None = None,
) -> MCResult:
    """Run Monte Carlo uncertainty analysis.

    Parameters
    ----------
    params : RpathParams
        Base Ecopath parameters.
    config : MCConfig, optional
        MC configuration. Defaults to MCConfig().
    pedigree_config : PedigreeConfig, optional
        EwE pedigree mapping.
    progress_callback : callable, optional
        Called with (current_sample, total_samples) after each run, in sample
        order, in this process (also when ``config.n_jobs`` runs workers).

    Returns
    -------
    MCResult
    """
    from pypath.core.pedigree import (
        build_distributions,
        sample_parameters,
    )

    if config is None:
        config = MCConfig()
    n_workers = _resolve_n_jobs(config.n_jobs, config.n_samples)

    rng = np.random.default_rng(config.seed)
    distributions = build_distributions(params, pedigree_config)

    if len(distributions) == 0:
        warnings.warn("No distributions to sample (all CVs are 0).", UserWarning)

    samples = sample_parameters(distributions, config.n_samples, config.method, rng)

    # Collect results — exclude fleet groups (type=3) from biomass stats
    n_groups = int((params.model["Type"] != 3).sum())
    ecopath_biomass = []
    ecopath_runs_list = [] if config.store_runs else None
    ecosim_biomass_list = []
    ecosim_runs_list = [] if config.store_runs else None
    n_feasible = 0
    n_ecosim = 0

    pool = None
    if n_workers == 1:
        outcomes = (_evaluate_sample(params, s, config, n_groups) for s in samples)
    else:
        # Workers get the model and config once; map keeps sample order
        pool = ProcessPoolExecutor(
            max_workers=n_workers,
            mp_context=_mp_context(config.mp_context),
            initializer=_init_worker,
            initargs=(params, config, n_groups),
        )
        outcomes = pool.map(
            _evaluate_in_worker,
            samples,
            chunksize=max(1, len(samples) // (4 * n_workers)),
        )

    try:
        for i, (feasible, bio, ecosim_bio) in enumerate(outcomes):
            if feasible:
                n_feasible += 1
                if bio is not None:
                    ecopath_biomass.append(bio)
                if config.store_runs:
                    ecopath_runs_list.append(
                        {
                            "Biomass": bio.copy() if bio is not None else None,
                        }
                    )
                if ecosim_bio is not None:
                    n_ecosim += 1
                    ecosim_biomass_list.append(ecosim_bio)
                    if config.store_runs:
                        ecosim_runs_list.append(ecosim_bio.copy())

            if progress_callback is not None:
                progress_callback(i + 1, config.n_samples)
    finally:
        if pool is not None:
            pool.shutdown(wait=True, cancel_futures=True)

    # Compute ecopath statistics
    ecopath_stats = {}
    if ecopath_biomass:
        bio_array = np.array(ecopath_biomass)  # (n_feasible, n_groups)
        ecopath_stats["Biomass"] = pd.DataFrame(
            {
                "mean": np.mean(bio_array, axis=0),
                "std": np.std(bio_array, axis=0),
                "p5": np.percentile(bio_array, 5, axis=0),
                "p25": np.percentile(bio_array, 25, axis=0),
                "p50": np.percentile(bio_array, 50, axis=0),
                "p75": np.percentile(bio_array, 75, axis=0),
                "p95": np.percentile(bio_array, 95, axis=0),
            }
        )

    # Compute ecosim statistics
    ecosim_stats = None
    if ecosim_biomass_list:
        # All arrays should have same shape; exclude padding col 0
        min_t = min(b.shape[0] for b in ecosim_biomass_list)
        stacked = np.array([b[:min_t, 1 : n_groups + 1] for b in ecosim_biomass_list])
        # stacked: (n_ecosim, timesteps, n_groups)
        ecosim_stats = {
            "Biomass": np.stack(
                [
                    np.mean(stacked, axis=0),
                    np.std(stacked, axis=0),
                    np.percentile(stacked, 5, axis=0),
                    np.percentile(stacked, 25, axis=0),
                    np.percentile(stacked, 50, axis=0),
                    np.percentile(stacked, 75, axis=0),
                    np.percentile(stacked, 95, axis=0),
                ],
                axis=-1,
            ),  # (timesteps, n_groups, 7)
        }

    feasibility_rate = n_feasible / config.n_samples if config.n_samples > 0 else 0.0

    return MCResult(
        n_total=config.n_samples,
        n_feasible=n_feasible,
        n_ecosim=n_ecosim,
        ecopath_stats=ecopath_stats,
        ecosim_stats=ecosim_stats,
        ecopath_runs=ecopath_runs_list,
        ecosim_runs=ecosim_runs_list,
        feasibility_rate=feasibility_rate,
        parameter_samples=None,
    )
