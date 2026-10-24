"""Ensemble modeling with Ecopath Monte Carlo sampling.

This module implements ensemble runs of the Ecopath mass-balance model
using bootstrapped parameter samples, enabling uncertainty quantification
and confidence interval estimation.

**Phase 1A.6 Task:** rpath() Ensemble Integration

Provides:
- Ensemble Monte Carlo runs with sampled Ecopath parameters
- Confidence interval calculation (95% CI) for outputs
- Batch processing of multiple replicates
- Summary statistics and visualization-ready output
"""

from __future__ import annotations

import logging
from copy import deepcopy
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from pypath.core.ecopath import Rpath, rpath
from pypath.core.params import RpathParams
from pypath.core.sampling import SampleContainer

logger = logging.getLogger(__name__)

__all__ = [
    "EnsembleResult",
    "run_ensemble_rpath",
    "get_ensemble_confidence_intervals",
    "ensemble_statistics",
]


@dataclass
class EnsembleResult:
    """Results from an ensemble Ecopath run.

    Stores outputs from all replicates and provides methods to compute
    confidence intervals and summary statistics.

    Attributes
    ----------
    n_replicates : int
        Number of bootstrap replicates run.
    base_model : Rpath
        The base balanced model (first replicate or mean estimates).
    replicate_models : List[Rpath]
        Balanced models for each replicate.
    replicate_params : List[RpathParams]
        Parameter sets used for each replicate.
    random_seed : Optional[int]
        Seed used for reproducible sampling.
    sampling_info : Dict[str, Any]
        Metadata about the sampling process.
    """

    n_replicates: int
    base_model: Rpath
    replicate_models: List[Rpath] = field(default_factory=list)
    replicate_params: List[RpathParams] = field(default_factory=list)
    random_seed: Optional[int] = None
    sampling_info: Dict[str, Any] = field(default_factory=dict)

    def biomass_percentiles(
        self, group: str, percentiles: Tuple[float, ...] = (2.5, 50.0, 97.5)
    ) -> Dict[float, float]:
        """Get percentiles of biomass for a group across replicates.

        Parameters
        ----------
        group : str
            Name of the group.
        percentiles : tuple of float
            Percentiles to compute (default: 2.5, 50, 97.5 for 95% CI).

        Returns
        -------
        dict[float, float]
            Mapping of percentile to biomass value.
        """
        group_idx = self._find_group_index(group)
        if group_idx is None:
            raise ValueError(f"Group '{group}' not found in model")

        biomasses = np.array(
            [model.Biomass[group_idx] for model in self.replicate_models]
        )
        return {p: float(np.percentile(biomasses, p)) for p in percentiles}

    def pb_percentiles(
        self, group: str, percentiles: Tuple[float, ...] = (2.5, 50.0, 97.5)
    ) -> Dict[float, float]:
        """Get percentiles of P/B ratio for a group."""
        group_idx = self._find_group_index(group)
        if group_idx is None:
            raise ValueError(f"Group '{group}' not found in model")

        pbs = np.array([model.PB[group_idx] for model in self.replicate_models])
        return {p: float(np.percentile(pbs, p)) for p in percentiles}

    def trophic_level_percentiles(
        self, group: str, percentiles: Tuple[float, ...] = (2.5, 50.0, 97.5)
    ) -> Dict[float, float]:
        """Get percentiles of trophic level for a group."""
        group_idx = self._find_group_index(group)
        if group_idx is None:
            raise ValueError(f"Group '{group}' not found in model")

        tls = np.array([model.TL[group_idx] for model in self.replicate_models])
        return {p: float(np.percentile(tls, p)) for p in percentiles}

    def confidence_interval(
        self, group: str, param: str, alpha: float = 0.05
    ) -> Tuple[float, float]:
        """Get (lower, upper) confidence bounds for a parameter.

        Parameters
        ----------
        group : str
            Name of the group.
        param : str
            Parameter name: 'biomass', 'pb', 'qb', 'ee', 'tl', 'ge'
        alpha : float
            Significance level (default: 0.05 for 95% CI).

        Returns
        -------
        tuple[float, float]
            (lower_bound, upper_bound) of confidence interval.
        """
        lower_p = (alpha / 2) * 100
        upper_p = (1 - alpha / 2) * 100

        percs = self._get_percentiles_for_param(group, param)
        return (
            float(np.percentile(percs, lower_p)),
            float(np.percentile(percs, upper_p)),
        )

    def summary_table(self) -> pd.DataFrame:
        """Generate summary statistics table for all groups.

        Returns
        -------
        pd.DataFrame
            Table with columns: Group, Biomass_Mean, Biomass_CI_95,
            PB_Mean, PB_CI_95, TL_Mean, TL_CI_95, etc.
        """
        records = []

        for group_idx, group_name in enumerate(self.base_model.Group):
            biomasses = np.array(
                [model.Biomass[group_idx] for model in self.replicate_models]
            )
            pbs = np.array(
                [model.PB[group_idx] for model in self.replicate_models]
            )
            tls = np.array(
                [model.TL[group_idx] for model in self.replicate_models]
            )

            bio_ci = (
                float(np.percentile(biomasses, 2.5)),
                float(np.percentile(biomasses, 97.5)),
            )
            pb_ci = (
                float(np.percentile(pbs, 2.5)),
                float(np.percentile(pbs, 97.5)),
            )
            tl_ci = (
                float(np.percentile(tls, 2.5)),
                float(np.percentile(tls, 97.5)),
            )

            records.append({
                "Group": group_name,
                "Biomass_Mean": float(biomasses.mean()),
                "Biomass_Std": float(biomasses.std()),
                "Biomass_LCI_95": bio_ci[0],
                "Biomass_UCI_95": bio_ci[1],
                "PB_Mean": float(pbs.mean()),
                "PB_Std": float(pbs.std()),
                "PB_LCI_95": pb_ci[0],
                "PB_UCI_95": pb_ci[1],
                "TL_Mean": float(tls.mean()),
                "TL_Std": float(tls.std()),
                "TL_LCI_95": tl_ci[0],
                "TL_UCI_95": tl_ci[1],
            })

        return pd.DataFrame(records)

    def _find_group_index(self, group_name: str) -> Optional[int]:
        """Find index of group by name."""
        try:
            return list(self.base_model.Group).index(group_name)
        except ValueError:
            return None

    def _get_percentiles_for_param(self, group: str, param: str) -> np.ndarray:
        """Extract parameter values across replicates."""
        group_idx = self._find_group_index(group)
        if group_idx is None:
            raise ValueError(f"Group '{group}' not found in model")

        param_map = {
            "biomass": "Biomass",
            "pb": "PB",
            "qb": "QB",
            "ee": "EE",
            "ge": "GE",
            "tl": "TL",
            "m0": "M0",
        }

        attr_name = param_map.get(param.lower())
        if attr_name is None:
            raise ValueError(f"Unknown parameter: {param}")

        return np.array(
            [getattr(model, attr_name)[group_idx] for model in self.replicate_models]
        )


def run_ensemble_rpath(
    base_params: RpathParams,
    sample_container: SampleContainer,
    eco_name: str = "",
    eco_area: float = 1.0,
    random_seed: Optional[int] = None,
    verbose: bool = False,
) -> EnsembleResult:
    """Run ensemble of Ecopath models with sampled parameters.

    Executes rpath() once for each replicate in the sample container,
    each time overwriting specific parameters with samples from the
    sampling data. Returns results from all replicates plus the base model.

    Parameters
    ----------
    base_params : RpathParams
        Base parameter set to use as template.
    sample_container : SampleContainer
        Container with bootstrap samples for groups and optionally diet.
    eco_name : str, optional
        Name of ecosystem (default: empty string).
    eco_area : float, optional
        Area of ecosystem in km² (default: 1.0).
    random_seed : int, optional
        Random seed for reproducibility. If provided, stored in result
        but doesn't affect deterministic replicate application.
    verbose : bool, optional
        If True, print progress for each replicate (default: False).

    Returns
    -------
    EnsembleResult
        Object containing all replicate models and convenience methods
        for confidence intervals and summary statistics.

    Raises
    ------
    ValueError
        If sample_container has inconsistent replicate counts or
        if any replicate fails to balance.

    Notes
    -----
    The ensemble is run sequentially. Each replicate applies samples to
    a deep copy of base_params before calling rpath(). Groups with samples
    have their Biomass, PB, QB, and EE values overwritten; groups without
    samples retain their base_params values.

    Example
    -------
    >>> from pypath.io.ewemdb import read_ewe_samples
    >>> params = create_rpath_params(...)
    >>> sample_container = read_ewe_samples("model.eweaccdb")
    >>> ensemble_result = run_ensemble_rpath(
    ...     params, sample_container, eco_name="Baltic Sea"
    ... )
    >>> ci = ensemble_result.confidence_interval("Cod", "biomass")
    >>> print(f"95% CI for Cod biomass: {ci[0]:.3f} - {ci[1]:.3f} t/km²")
    """
    if not sample_container.group_samples:
        raise ValueError("sample_container has no group samples")

    if random_seed is not None:
        np.random.seed(random_seed)

    replicate_models: List[Rpath] = []
    replicate_params_list: List[RpathParams] = []

    # First, run base model (replicate 0 with base parameters)
    base_model = rpath(base_params, eco_name=eco_name, eco_area=eco_area)
    replicate_models.append(base_model)
    replicate_params_list.append(base_params)

    if verbose:
        logger.info(
            f"Running ensemble with {sample_container.n_replicates} replicates"
        )

    # Run each replicate by applying samples
    for rep_idx in range(sample_container.n_replicates):
        # Deep copy base params for this replicate
        params_copy = deepcopy(base_params)

        # Apply samples to this replicate
        _apply_replicate_to_params(params_copy, sample_container, rep_idx)

        # Run rpath with sampled parameters
        try:
            model = rpath(params_copy, eco_name=eco_name, eco_area=eco_area)
            replicate_models.append(model)
            replicate_params_list.append(params_copy)

            if verbose:
                logger.debug(
                    f"Completed replicate {rep_idx + 1}/{sample_container.n_replicates}"
                )

        except Exception as e:
            logger.error(f"Replicate {rep_idx} failed: {e}")
            raise ValueError(f"Replicate {rep_idx} failed to balance: {e}") from e

    info = sample_container.summary_stats()
    info["n_models_run"] = len(replicate_models)

    result = EnsembleResult(
        n_replicates=sample_container.n_replicates,
        base_model=base_model,
        replicate_models=replicate_models,
        replicate_params=replicate_params_list,
        random_seed=random_seed,
        sampling_info=info,
    )

    if verbose:
        logger.info(f"Ensemble complete: {result.n_replicates} replicates + base model")

    return result


def get_ensemble_confidence_intervals(
    ensemble_result: EnsembleResult, alpha: float = 0.05
) -> pd.DataFrame:
    """Extract 95% (or alpha-level) confidence intervals for all groups.

    Parameters
    ----------
    ensemble_result : EnsembleResult
        Result object from run_ensemble_rpath().
    alpha : float
        Significance level (default: 0.05 for 95% CI).

    Returns
    -------
    pd.DataFrame
        Table with columns: Group, Param, LCI, Mean, UCI
        for each parameter (Biomass, PB, QB, EE, TL).
    """
    records = []
    lower_p = (alpha / 2) * 100
    upper_p = (1 - alpha / 2) * 100

    for group_idx, group_name in enumerate(ensemble_result.base_model.Group):
        biomasses = np.array(
            [model.Biomass[group_idx] for model in ensemble_result.replicate_models]
        )
        pbs = np.array(
            [model.PB[group_idx] for model in ensemble_result.replicate_models]
        )
        tls = np.array(
            [model.TL[group_idx] for model in ensemble_result.replicate_models]
        )

        for param, values in [
            ("Biomass", biomasses),
            ("PB", pbs),
            ("TL", tls),
        ]:
            records.append({
                "Group": group_name,
                "Parameter": param,
                "LCI": float(np.percentile(values, lower_p)),
                "Mean": float(values.mean()),
                "UCI": float(np.percentile(values, upper_p)),
            })

    return pd.DataFrame(records)


def ensemble_statistics(ensemble_result: EnsembleResult) -> Dict[str, Any]:
    """Compute summary statistics for ensemble run.

    Parameters
    ----------
    ensemble_result : EnsembleResult
        Result object from run_ensemble_rpath().

    Returns
    -------
    dict
        Keys: 'n_groups', 'n_replicates', 'groups_list',
        'total_system_biomass_mean', 'total_system_biomass_cv',
        'mean_trophic_level', etc.
    """
    groups = ensemble_result.base_model.Group
    n_groups = len(groups)
    n_replicates = ensemble_result.n_replicates

    # Total system biomass
    total_biomasses = np.array(
        [model.Biomass.sum() for model in ensemble_result.replicate_models]
    )

    # Mean trophic level (weighted by biomass)
    mean_tls = []
    for model in ensemble_result.replicate_models:
        wt_tl = np.sum(model.Biomass * model.TL) / np.sum(model.Biomass)
        mean_tls.append(wt_tl)
    mean_tls = np.array(mean_tls)

    return {
        "n_groups": n_groups,
        "n_replicates": n_replicates,
        "groups": list(groups),
        "total_system_biomass_mean": float(total_biomasses.mean()),
        "total_system_biomass_std": float(total_biomasses.std()),
        "total_system_biomass_cv": float(
            total_biomasses.std() / total_biomasses.mean()
        ),
        "mean_trophic_level_mean": float(mean_tls.mean()),
        "mean_trophic_level_std": float(mean_tls.std()),
    }


def _apply_replicate_to_params(
    params: RpathParams, sample_container: SampleContainer, rep_idx: int
) -> None:
    """Apply samples from a specific replicate to RpathParams in-place.

    Overwrites Biomass, PB, QB, EE for groups that have samples.
    """
    for group_name, sample in sample_container.group_samples.items():
        # Find group in model DataFrame
        mask = params.model["Group"] == group_name
        if not mask.any():
            logger.warning(f"Group '{group_name}' not found in params")
            continue

        idx = params.model.index[mask][0]

        # Overwrite with sample values
        params.model.loc[idx, "Biomass"] = float(sample.biomass_samples[rep_idx])
        params.model.loc[idx, "PB"] = float(sample.pb_samples[rep_idx])
        params.model.loc[idx, "QB"] = float(sample.qb_samples[rep_idx])
        params.model.loc[idx, "EE"] = float(sample.ee_samples[rep_idx])