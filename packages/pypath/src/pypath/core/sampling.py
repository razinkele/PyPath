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
    """Sample uncertainty data for a single functional group.
    
    Holds bootstrap replicates of Ecopath parameters (biomass, P/B, Q/B, EE)
    for a single group, enabling uncertainty analysis across ensemble runs.
    
    Used internally by SampleContainer to manage per-group uncertainty
    distributions. Users typically do not create these directly, but instead
    work with read_ewe_samples() or create_parameter_samples().

    Attributes
    ----------
    group_name : str
        Name of the functional group being sampled.
    biomass_samples : np.ndarray, shape (n_replicates,)
        Bootstrap samples of group biomass (t/km²).
        Should be positive values representing group standing stock.
    pb_samples : np.ndarray, shape (n_replicates,)
        Bootstrap samples of Production/Biomass ratio (1/year).
        Values typically in range [0.01, 10] depending on group type.
    qb_samples : np.ndarray, shape (n_replicates,)
        Bootstrap samples of Consumption/Biomass ratio (1/year).
        Generally QB > PB for consumers.
    ee_samples : np.ndarray, shape (n_replicates,)
        Bootstrap samples of Ecotrophic Efficiency (fraction, 0-1).
        Must be in [0, 1]; represents proportion of production consumed.
    biomass_se : float, optional
        Coefficient of variation (CV) of biomass estimate (default: 0).
        CV = SE / mean; useful for diagnostics and documentation.
    pb_se : float, optional
        CV of Production/Biomass estimate (default: 0).
    qb_se : float, optional
        CV of Consumption/Biomass estimate (default: 0).
    ee_se : float, optional
        CV of Ecotrophic Efficiency estimate (default: 0).

    Methods
    -------
    n_replicates : int property
        Number of bootstrap replicates (length of sample arrays).
    get_percentile : Get specific percentile of parameter distribution
    confidence_interval : Get confidence interval for parameter

    Examples
    --------
    >>> import numpy as np
    >>> biomass = np.random.lognormal(mean=0, sigma=0.5, size=100)
    >>> pb = np.random.gamma(shape=2, scale=0.5, size=100)
    >>> qb = np.random.gamma(shape=2, scale=1.0, size=100)
    >>> ee = np.random.beta(a=5, b=2, size=100)
    >>> sample = EcopathGroupSample(
    ...     group_name="Cod",
    ...     biomass_samples=biomass,
    ...     pb_samples=pb,
    ...     qb_samples=qb,
    ...     ee_samples=ee,
    ...     biomass_se=0.3
    ... )
    >>> ci = sample.confidence_interval("biomass", alpha=0.05)
    >>> print(f"95%% CI: {ci[0]:.2f} - {ci[1]:.2f}")

    Raises
    ------
    ValueError
        If sample array lengths are not equal in __post_init__.

    See Also
    --------
    SampleContainer : Container for all sampling data
    EcopathDietCompSample : Sample uncertainty for diet composition
    run_ensemble_rpath : Use samples in ensemble run
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
        """Validate sample shapes are consistent.
        
        Called automatically after __init__ to ensure all sample arrays
        have the same length (number of replicates).
        
        Raises
        ------
        ValueError
            If sample arrays have inconsistent lengths.
        """
        n = len(self.biomass_samples)
        if not all(
            len(s) == n
            for s in [self.pb_samples, self.qb_samples, self.ee_samples]
        ):
            raise ValueError("All sample arrays must have same length")

    @property
    def n_replicates(self) -> int:
        """Number of bootstrap replicates.
        
        Returns
        -------
        int
            Length of all sample arrays (e.g., len(biomass_samples)).
        """
        return len(self.biomass_samples)

    def get_percentile(self, param: str, p: float) -> float:
        """Get percentile of parameter samples (0-100).
        
        Computes a single percentile from the distribution of a parameter
        across all replicates.
        
        Parameters
        ----------
        param : str
            Parameter name: 'biomass', 'pb', 'qb', or 'ee'.
        p : float
            Percentile value (0-100).
            
        Returns
        -------
        float
            The p-th percentile value.
            
        Raises
        ------
        ValueError
            If param is not recognized.
        """
        samples = getattr(self, f"{param}_samples", None)
        if samples is None:
            raise ValueError(f"Unknown parameter: {param}")
        return float(np.percentile(samples, p))

    def confidence_interval(self, param: str, alpha: float = 0.05) -> Tuple[float, float]:
        """Get (lower, upper) confidence bounds for parameter.
        
        Parameters
        ----------
        param : str
            Parameter name: 'biomass', 'pb', 'qb', or 'ee'.
        alpha : float, optional
            Significance level (default: 0.05 for 95%% CI).
            
        Returns
        -------
        tuple[float, float]
            (lower_bound, upper_bound) of confidence interval.
            
        Examples
        --------
        >>> sample = EcopathGroupSample(...)
        >>> ci = sample.confidence_interval("biomass", alpha=0.05)
        >>> print(f"95%% CI: {ci[0]:.2f} - {ci[1]:.2f}")
        """
        lower_p = (alpha / 2) * 100
        upper_p = (1 - alpha / 2) * 100
        return (
            self.get_percentile(param, lower_p),
            self.get_percentile(param, upper_p),
        )


@dataclass
class EcopathDietCompSample:
    """Sample uncertainty data for diet composition link.
    
    Represents the uncertainty in how much of one group (prey) appears
    in the diet of another group (predator), across bootstrap replicates.
    
    Used to propagate diet uncertainty through ensemble model runs.

    Attributes
    ----------
    predator_name : str
        Name of the predator (consuming) group.
    prey_name : str
        Name of the prey (consumed) group.
    diet_fraction_samples : np.ndarray, shape (n_replicates,)
        Bootstrap samples of diet fraction (0-1).
        Represents the proportion of predator's diet from this prey.
    diet_fraction_se : float, optional
        Coefficient of variation (CV) of diet fraction (default: 0).
        Useful for documenting source of uncertainty.

    Properties
    ----------
    n_replicates : Number of bootstrap replicates

    See Also
    --------
    SampleContainer : Container for all sampling data
    EcopathGroupSample : Sample uncertainty for group parameters
    """

    predator_name: str
    prey_name: str
    diet_fraction_samples: np.ndarray
    diet_fraction_se: float = 0.0

    @property
    def n_replicates(self) -> int:
        """Number of bootstrap replicates.
        
        Returns
        -------
        int
            Length of all sample arrays (e.g., len(biomass_samples)).
        """
        return len(self.diet_fraction_samples)


@dataclass
class EcopathGroupCatchSample:
    """Sample uncertainty data for catch/landings by fleet.
    
    Represents uncertainty in landings/catch data, typically from
    fisheries statistics or scientific surveys, across replicates.

    Attributes
    ----------
    group_name : str
        Name of the group being caught.
    fleet_name : str
        Name of catching fleet or fishing sector.
    catch_samples : np.ndarray, shape (n_replicates,)
        Bootstrap samples of catch (t/km²/year).
        Should be non-negative values.
    catch_se : float, optional
        Coefficient of variation (CV) of catch estimate (default: 0).
        Documents source of uncertainty (e.g., reporting errors).

    Properties
    ----------
    n_replicates : Number of bootstrap replicates

    See Also
    --------
    SampleContainer : Container for all sampling data
    """

    group_name: str
    fleet_name: str
    catch_samples: np.ndarray
    catch_se: float = 0.0

    @property
    def n_replicates(self) -> int:
        """Number of bootstrap replicates.
        
        Returns
        -------
        int
            Length of all sample arrays (e.g., len(biomass_samples)).
        """
        return len(self.catch_samples)


@dataclass
class SampleContainer:
    """Container for all Ecopath sampling data.
    
    Manages bootstrap replicates of Ecopath parameters, diet composition,
    and catch data across multiple groups, diets, and fleets. All samples
    must have consistent n_replicates to enable ensemble simulation.
    
    This is the primary data structure passed to run_ensemble_rpath()
    and other ensemble functions.

    Attributes
    ----------
    n_replicates : int
        Number of Monte Carlo bootstrap replicates. All samples added
        must have exactly this many replicates.
    group_samples : dict[str, EcopathGroupSample]
        Per-group uncertainty samples indexed by group name.
        Example keys: "Cod", "Herring", "Zooplankton".
    diet_samples : dict[Tuple[str, str], EcopathDietCompSample]
        Diet composition samples indexed by (predator, prey) tuple.
        Example key: ("Cod", "Herring") for Herring in Cod's diet.
    catch_samples : dict[Tuple[str, str], EcopathGroupCatchSample]
        Catch samples indexed by (group, fleet) tuple.
        Example key: ("Cod", "Industrial Demersal") for industrial fishing.
    random_seed : Optional[int]
        Seed used for reproducible sampling (if applicable).
        None if samples were generated non-deterministically.

    Methods
    -------
    add_group_sample : Add EcopathGroupSample to container
    add_diet_sample : Add EcopathDietCompSample to container
    add_catch_sample : Add EcopathGroupCatchSample to container
    uncertainty_dataframe : Export uncertainty summary as DataFrame
    summary_stats : Get summary statistics

    Examples
    --------
    >>> from pypath.core.sampling import SampleContainer, EcopathGroupSample
    >>> import numpy as np
    >>> 
    >>> container = SampleContainer(n_replicates=100)
    >>> 
    >>> # Add samples for Cod
    >>> cod_sample = EcopathGroupSample(
    ...     group_name="Cod",
    ...     biomass_samples=np.random.lognormal(0, 0.3, 100),
    ...     pb_samples=np.random.gamma(2, 0.2, 100),
    ...     qb_samples=np.random.gamma(2, 0.5, 100),
    ...     ee_samples=np.random.beta(5, 2, 100)
    ... )
    >>> container.add_group_sample(cod_sample)
    >>> print(f"Groups sampled: {len(container.group_samples)}")

    Raises
    ------
    ValueError
        When adding a sample with n_replicates != self.n_replicates.

    See Also
    --------
    EcopathGroupSample : Per-group parameter samples
    run_ensemble_rpath : Use container in ensemble run
    read_ewe_samples : Read samples from EwE database
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
        
        Parameters
        ----------
        sample : EcopathGroupSample
            Group sample to add. Must have n_replicates == self.n_replicates.
        
        Raises
        ------
        ValueError
            If sample.n_replicates != self.n_replicates.
            
        Examples
        --------
        >>> container = SampleContainer(n_replicates=100)
        >>> sample = EcopathGroupSample(...)
        >>> container.add_group_sample(sample)
        """
        if sample.n_replicates != self.n_replicates:
            raise ValueError(
                f"Sample has {sample.n_replicates} replicates, "
                f"container expects {self.n_replicates}"
            )
        self.group_samples[sample.group_name] = sample

    def add_diet_sample(self, sample: EcopathDietCompSample) -> None:
        """Add a diet composition sample to the container.
        
        Parameters
        ----------
        sample : EcopathDietCompSample
            Diet sample to add.
            
        Raises
        ------
        ValueError
            If sample.n_replicates != self.n_replicates.
        """
        if sample.n_replicates != self.n_replicates:
            raise ValueError(
                f"Sample has {sample.n_replicates} replicates, "
                f"container expects {self.n_replicates}"
            )
        key = (sample.predator_name, sample.prey_name)
        self.diet_samples[key] = sample

    def add_catch_sample(self, sample: EcopathGroupCatchSample) -> None:
        """Add a catch sample to the container.
        
        Parameters
        ----------
        sample : EcopathGroupCatchSample
            Catch sample to add.
            
        Raises
        ------
        ValueError
            If sample.n_replicates != self.n_replicates.
        """
        if sample.n_replicates != self.n_replicates:
            raise ValueError(
                f"Sample has {sample.n_replicates} replicates, "
                f"container expects {self.n_replicates}"
            )
        key = (sample.group_name, sample.fleet_name)
        self.catch_samples[key] = sample

    def uncertainty_dataframe(self, param_type: str = "group") -> pd.DataFrame:
        """Export uncertainty summary as DataFrame.
        
        Creates a summary table of uncertainty metrics for easier viewing,
        export, or plotting.
        
        Parameters
        ----------
        param_type : str, optional
            Type of data to export (default: "group").
            Options: "group" (parameters), "diet", or "catch".
        
        Returns
        -------
        pd.DataFrame
            Summary statistics table with columns:
            - Group : Group or fleet name
            - Parameter_Mean : Mean value
            - Parameter_CV : Coefficient of variation
            - Parameter_LCI_95 : Lower 95% CI
            - Parameter_UCI_95 : Upper 95% CI
            
        Raises
        ------
        ValueError
            If param_type is not recognized.
            
        Examples
        --------
        >>> container = SampleContainer(...)
        >>> df = container.uncertainty_dataframe("group")
        >>> print(df[["Group", "Biomass_Mean", "Biomass_CV"]])
        """
        if param_type == "group":
            data = []
            for group_name, sample in self.group_samples.items():
                lb, ub = sample.confidence_interval("biomass")
                data.append({
                    "Group": group_name,
                    "Biomass_Mean": sample.biomass_samples.mean(),
                    "Biomass_CV": sample.biomass_se,
                    "Biomass_LCI_95": lb,
                    "Biomass_UCI_95": ub,
                })
            return pd.DataFrame(data)
        else:
            raise ValueError(f"Unknown param_type: {param_type}")

    def summary_stats(self) -> Dict[str, int]:
        """Get summary statistics of container contents.
        
        Returns
        -------
        dict[str, int]
            Dictionary with keys:
            - n_replicates : Number of replicates
            - n_groups : Number of groups with samples
            - n_diet_pairs : Number of diet links with samples
            - n_catch_pairs : Number of catch records with samples
            
        Examples
        --------
        >>> container = SampleContainer(...)
        >>> stats = container.summary_stats()
        >>> print(f"Samples: {stats['n_groups']} groups, "
        ...       f"{stats['n_diet_pairs']} diet links")
        """
        return {
            "n_replicates": self.n_replicates,
            "n_groups": len(self.group_samples),
            "n_diet_pairs": len(self.diet_samples),
            "n_catch_pairs": len(self.catch_samples),
        }


def add_sampling_to_params(params: Any, sample_container: SampleContainer) -> None:
    """Attach sampling data to RpathParams.
    
    Attaches a SampleContainer object to an RpathParams instance for
    later use or reference.
    
    Parameters
    ----------
    params : RpathParams
        Parameter object to attach sampling to.
    sample_container : SampleContainer
        Container with all sampling data.
        
    Notes
    -----
    This function adds a .samples attribute to params. Does not modify
    the actual parameter values; to apply samples, use run_ensemble_rpath().
    
    Examples
    --------
    >>> from pypath.core.params import create_rpath_params
    >>> from pypath.core.sampling import SampleContainer, add_sampling_to_params
    >>> params = create_rpath_params("model.csv")
    >>> samples = SampleContainer(n_replicates=100)
    >>> add_sampling_to_params(params, samples)
    >>> print(hasattr(params, "samples"))
    True
    
    See Also
    --------
    run_ensemble_rpath : Apply samples in ensemble run
    apply_samples_to_derivatives : Apply specific replicate samples
    """
    params.samples = sample_container  # type: ignore
    logger.info(f"Added sampling to params: {sample_container.summary_stats()}")


def apply_samples_to_derivatives(
    replicate_index: int,
    sample_container: SampleContainer,
    params: Dict[str, Any],
) -> Dict[str, Any]:
    """Apply a specific sample replicate to parameter dictionary.
    
    Overwrites parameter values with samples from a specific replicate.
    Typically used in optimization or sensitivity analysis workflows.
    
    Parameters
    ----------
    replicate_index : int
        Index of replicate to apply (0 to n_replicates - 1).
    sample_container : SampleContainer
        Container with sampling data.
    params : dict[str, Any]
        Parameter dictionary to modify. Groups should be keys; values
        should be dicts with 'biomass', 'pb', 'qb', 'ee' keys.
    
    Returns
    -------
    dict[str, Any]
        Modified params dictionary with sample values applied.
    
    Raises
    ------
    ValueError
        If replicate_index >= sample_container.n_replicates.
    
    Notes
    -----
    This is a lower-level utility. Most users should use
    run_ensemble_rpath() instead.
    
    See Also
    --------
    run_ensemble_rpath : Main ensemble function
    add_sampling_to_params : Attach samples to RpathParams
    """
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


def validate_samples(
    sample_container: SampleContainer, params: Any, tolerance: float = 1e-6
) -> List[str]:
    """Validate sampling data against RpathParams.
    
    Performs basic validation checks on sample data:
    - Replicate counts are consistent
    - Biomass values are non-negative
    - EE values are in [0, 1]
    - Other constraints
    
    Parameters
    ----------
    sample_container : SampleContainer
        Container with sampling data to validate.
    params : RpathParams
        Base parameters (used for reference; checks samples are consistent
        with model structure).
    tolerance : float, optional
        Tolerance for floating-point comparisons (default: 1e-6).
    
    Returns
    -------
    list[str]
        List of validation issues (empty if all checks pass).
        Each string describes a problem found.

    Raises
    ------
    ValueError
        Not directly, but issues list documents validation failures that
        may cause problems in ensemble runs.
    
    Examples
    --------
    >>> container = SampleContainer(...)
    >>> issues = validate_samples(container, params)
    >>> if issues:
    ...     for issue in issues:
    ...         print(f"Warning: {issue}")
    >>> else:
    ...     print("All validation checks passed")
    
    See Also
    --------
    run_ensemble_rpath : Runs validation before ensemble
    SampleContainer.uncertainty_dataframe : Export validation results
    """
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
