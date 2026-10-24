"""Ecopath sampling and uncertainty quantification module.

This module implements the EcopathSample framework for Monte Carlo sampling
and uncertainty propagation through Ecopath mass-balance calculations.

**Phase 1A Task 1.A.1:** Full EcopathSample core classes implementation

This module enables users to:
- Represent parameter uncertainty as probability distributions
- Run ensemble Ecopath models with sampled parameters
- Analyze uncertainty propagation through the food web
- Export sampling data to EwE databases
- Calculate confidence intervals and sensitivity metrics
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

__all__ = [
    "SampleContainer",
    "EcopathGroupSample",
    "EcopathDietCompSample",
    "EcopathGroupCatchSample",
    "add_sampling_to_params",
    "apply_samples_to_derivatives",
    "validate_samples",
]


@dataclass
class EcopathGroupSample:
    """Sample uncertainty data for a single group.

    Holds bootstrap replicates of Ecopath parameters (biomass, PB, QB, EE)
    for uncertainty analysis.

    Attributes
    ----------
    group_name : str
        Name of the group being sampled.
    biomass_samples : np.ndarray, shape (n_replicates,)
        Bootstrap samples of group biomass (t/km²).
    pb_samples : np.ndarray, shape (n_replicates,)
        Bootstrap samples of Production/Biomass ratio (1/year).
    qb_samples : np.ndarray, shape (n_replicates,)
        Bootstrap samples of Consumption/Biomass ratio (1/year).
    ee_samples : np.ndarray, shape (n_replicates,)
        Bootstrap samples of Ecotrophic Efficiency (fraction, 0-1).
    biomass_se : float
        Coefficient of variation (CV) of biomass estimate (CV = SE / mean).
    pb_se : float
        CV of Production/Biomass estimate.
    qb_se : float
        CV of Consumption/Biomass estimate.
    ee_se : float
        CV of Ecotrophic Efficiency estimate.
    """

    group_name: str
    biomass_samples: np.ndarray
    pb_samples: np.ndarray
    qb_samples: np.ndarray
    ee_samples: np.ndarray
    biomass_se: float = 0.0
    pb_se: float = 0.0
    qb_se: float = 0.0
    ee_se: float = 0.0

    def __post_init__(self):
        """Validate sample shapes are consistent."""
        n = len(self.biomass_samples)
        if not all(
            len(s) == n for s in [self.pb_samples, self.qb_samples, self.ee_samples]
        ):
            raise ValueError("All sample arrays must have same length")

    @property
    def n_replicates(self) -> int:
        """Number of bootstrap replicates."""
        return len(self.biomass_samples)

    def get_percentile(self, param: str, p: float) -> float:
        """Get percentile of parameter samples (0-100)."""
        samples = getattr(self, f"{param}_samples", None)
        if samples is None:
            raise ValueError(f"Unknown parameter: {param}")
        return float(np.percentile(samples, p))

    def confidence_interval(
        self, param: str, alpha: float = 0.05
    ) -> Tuple[float, float]:
        """Get (lower, upper) confidence bounds for parameter."""
        lower_p = (alpha / 2) * 100
        upper_p = (1 - alpha / 2) * 100
        return (
            self.get_percentile(param, lower_p),
            self.get_percentile(param, upper_p),
        )


@dataclass
class EcopathDietCompSample:
    """Sample uncertainty data for diet composition.

    Attributes
    ----------
    predator_name : str
        Name of predator group.
    prey_name : str
        Name of prey group.
    diet_fraction_samples : np.ndarray, shape (n_replicates,)
        Bootstrap samples of diet fraction (0-1).
    diet_fraction_se : float
        Coefficient of variation (CV) of diet fraction.
    """

    predator_name: str
    prey_name: str
    diet_fraction_samples: np.ndarray
    diet_fraction_se: float = 0.0

    @property
    def n_replicates(self) -> int:
        """Number of bootstrap replicates."""
        return len(self.diet_fraction_samples)


@dataclass
class EcopathGroupCatchSample:
    """Sample uncertainty data for catch/landings.

    Attributes
    ----------
    group_name : str
        Name of group being caught.
    fleet_name : str
        Name of catching fleet.
    catch_samples : np.ndarray, shape (n_replicates,)
        Bootstrap samples of catch (t/km²/year).
    catch_se : float
        Coefficient of variation (CV) of catch estimate.
    """

    group_name: str
    fleet_name: str
    catch_samples: np.ndarray
    catch_se: float = 0.0

    @property
    def n_replicates(self) -> int:
        """Number of bootstrap replicates."""
        return len(self.catch_samples)


@dataclass
class SampleContainer:
    """Container for all Ecopath sampling data.

    Manages sampling information across groups, diet composition, and catches.
    All samples must have consistent n_replicates.

    Attributes
    ----------
    n_replicates : int
        Number of Monte Carlo bootstrap replicates.
    group_samples : dict[str, EcopathGroupSample]
        Group-level samples indexed by group name.
    diet_samples : dict[Tuple[str, str], EcopathDietCompSample]
        Diet samples indexed by (predator, prey) tuple.
    catch_samples : dict[Tuple[str, str], EcopathGroupCatchSample]
        Catch samples indexed by (group, fleet) tuple.
    random_seed : Optional[int]
        Seed used for reproducible sampling.
    """

    n_replicates: int
    group_samples: Dict[str, EcopathGroupSample] = field(default_factory=dict)
    diet_samples: Dict[Tuple[str, str], EcopathDietCompSample] = field(
        default_factory=dict
    )
    catch_samples: Dict[Tuple[str, str], EcopathGroupCatchSample] = field(
        default_factory=dict
    )
    random_seed: Optional[int] = None

    def add_group_sample(self, sample: EcopathGroupSample) -> None:
        """Add a group sample to the container.

        Raises ValueError if replicate counts don't match.
        """
        if sample.n_replicates != self.n_replicates:
            raise ValueError(
                f"Sample has {sample.n_replicates} replicates, "
                f"container expects {self.n_replicates}"
            )
        self.group_samples[sample.group_name] = sample

    def add_diet_sample(self, sample: EcopathDietCompSample) -> None:
        """Add a diet composition sample."""
        if sample.n_replicates != self.n_replicates:
            raise ValueError(
                f"Sample has {sample.n_replicates} replicates, "
                f"container expects {self.n_replicates}"
            )
        key = (sample.predator_name, sample.prey_name)
        self.diet_samples[key] = sample

    def add_catch_sample(self, sample: EcopathGroupCatchSample) -> None:
        """Add a catch sample."""
        if sample.n_replicates != self.n_replicates:
            raise ValueError(
                f"Sample has {sample.n_replicates} replicates, "
                f"container expects {self.n_replicates}"
            )
        key = (sample.group_name, sample.fleet_name)
        self.catch_samples[key] = sample

    def uncertainty_dataframe(self, param_type: str = "group") -> pd.DataFrame:
        """Export uncertainty summary as DataFrame.

        Parameters
        ----------
        param_type : str
            "group", "diet", or "catch"

        Returns
        -------
        pd.DataFrame
            Summary statistics (mean, CV, 95% CI)
        """
        if param_type == "group":
            data = []
            for group_name, sample in self.group_samples.items():
                lb, ub = sample.confidence_interval("biomass")
                data.append(
                    {
                        "Group": group_name,
                        "Biomass_Mean": sample.biomass_samples.mean(),
                        "Biomass_CV": sample.biomass_se,
                        "Biomass_LCI_95": lb,
                        "Biomass_UCI_95": ub,
                    }
                )
            return pd.DataFrame(data)
        else:
            raise ValueError(f"Unknown param_type: {param_type}")

    def summary_stats(self) -> Dict[str, int]:
        """Get summary statistics of container contents."""
        return {
            "n_replicates": self.n_replicates,
            "n_groups": len(self.group_samples),
            "n_diet_pairs": len(self.diet_samples),
            "n_catch_pairs": len(self.catch_samples),
        }


def add_sampling_to_params(params: Any, sample_container: SampleContainer) -> None:
    """Attach sampling data to RpathParams."""
    params.samples = sample_container  # type: ignore
    logger.info(f"Added sampling to params: {sample_container.summary_stats()}")


def apply_samples_to_derivatives(
    replicate_index: int,
    sample_container: SampleContainer,
    params: Dict[str, Any],
) -> Dict[str, Any]:
    """Apply a specific sample replicate to parameter dictionary."""
    if replicate_index >= sample_container.n_replicates:
        raise ValueError(
            f"Replicate index {replicate_index} >= {sample_container.n_replicates}"
        )

    for group_name, sample in sample_container.group_samples.items():
        if group_name not in params:
            logger.warning(f"Group {group_name} not in params")
            continue
        if isinstance(params[group_name], dict):
            params[group_name]["biomass"] = float(
                sample.biomass_samples[replicate_index]
            )
            params[group_name]["pb"] = float(sample.pb_samples[replicate_index])
            params[group_name]["qb"] = float(sample.qb_samples[replicate_index])
            params[group_name]["ee"] = float(sample.ee_samples[replicate_index])

    return params


def validate_samples(sample_container: SampleContainer, params: Any) -> List[str]:
    """Validate sampling data against RpathParams."""
    issues = []

    # Check replicate consistency
    for group_name, sample in sample_container.group_samples.items():
        if sample.n_replicates != sample_container.n_replicates:
            issues.append(
                f"Group {group_name}: replicate count "
                f"({sample.n_replicates}) != container ({sample_container.n_replicates})"
            )

    # Check value ranges
    for group_name, sample in sample_container.group_samples.items():
        if np.any(sample.biomass_samples < 0):
            issues.append(f"Group {group_name}: biomass < 0")
        if np.any(sample.ee_samples < 0) or np.any(sample.ee_samples > 1):
            issues.append(f"Group {group_name}: EE outside [0, 1]")

    return issues
