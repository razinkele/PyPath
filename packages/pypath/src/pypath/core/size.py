"""Size-structured population analysis using EwE PSD tables.

Phase 1B.3: TG3.3 Size Distribution System

This module implements size-structured population analysis for aquatic groups,
including size class definitions, population size distributions (PSD), and
von Bertalanffy growth function integration for age-size conversions.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


@dataclass
class SizeClass:
    """A single size interval for population structure analysis.

    Parameters
    ----------
    size_min : float
        Minimum size of the interval.
    size_max : float
        Maximum size of the interval.
    size_unit : str
        Unit of size measurement ('cm' or 'mm').

    Attributes
    ----------
    size_min, size_max : float
        Interval boundaries (inclusive).
    size_unit : str
        Unit (default 'cm').
    """

    size_min: float
    size_max: float
    size_unit: str = "cm"

    def __post_init__(self) -> None:
        """Validate size_min, size_max, and size_unit after initialization."""
        if self.size_min >= self.size_max:
            raise ValueError(
                f"size_min ({self.size_min}) must be less than "
                f"size_max ({self.size_max})"
            )
        if self.size_unit not in ('cm', 'mm'):
            raise ValueError(
                f"size_unit must be 'cm' or 'mm', got {self.size_unit}"
            )

    def midpoint(self) -> float:
        """Return the midpoint of the size interval.

        Returns
        -------
        float
            Midpoint: (size_min + size_max) / 2.
        """
        return (self.size_min + self.size_max) / 2.0

    def width(self) -> float:
        """Return the width of the size interval.

        Returns
        -------
        float
            Width: size_max - size_min.
        """
        return self.size_max - self.size_min

    def contains(self, size: float) -> bool:
        """Check if a size value falls within this interval.

        Parameters
        ----------
        size : float
            Size value to check.

        Returns
        -------
        bool
            True if size_min <= size < size_max.
        """
        return self.size_min <= size < self.size_max


@dataclass
class SizeDistribution:
    """Population size distribution (PSD) for a group.

    Represents the count or biomass distribution across size classes,
    typically imported from EwE PSD (Population Size Distribution) tables.

    Parameters
    ----------
    group_name : str
        Name of the group (e.g., "Herring").
    size_classes : list[SizeClass]
        Ordered list of size intervals.
    counts : np.ndarray
        Individual counts per size class (1D array, same length as size_classes).

    Attributes
    ----------
    group_name : str
        Unique identifier for the group.
    size_classes : list[SizeClass]
        Size intervals (assumed to be non-overlapping and ordered).
    counts : np.ndarray
        Count array (float64).
    """

    group_name: str
    size_classes: list[SizeClass] = field(default_factory=list)
    counts: np.ndarray = field(default_factory=lambda: np.array([], dtype=np.float64))

    def __post_init__(self) -> None:
        """Validate that size_classes and counts have matching lengths."""
        if len(self.size_classes) != len(self.counts):
            raise ValueError(
                f"Number of size classes ({len(self.size_classes)}) must "
                f"match number of counts ({len(self.counts)})"
            )

    def is_valid(self) -> bool:
        """Validate the size distribution for consistency.

        Checks that:
        - size_classes and counts have matching lengths
        - all counts are non-negative
        - size intervals are ordered and non-overlapping

        Returns
        -------
        bool
            True if valid.
        """
        if len(self.size_classes) == 0:
            logger.warning(f"Empty size distribution for {self.group_name}")
            return False

        if len(self.size_classes) != len(self.counts):
            logger.error(
                f"{self.group_name}: size_classes ({len(self.size_classes)}) "
                f"!= counts ({len(self.counts)})"
            )
            return False

        if np.any(self.counts < 0):
            logger.error(f"{self.group_name}: negative counts detected")
            return False

        for i, sc in enumerate(self.size_classes):
            if sc.size_min >= sc.size_max:
                logger.error(
                    f"{self.group_name} class {i}: size_min >= size_max"
                )
                return False

        for i in range(len(self.size_classes) - 1):
            if self.size_classes[i].size_max > self.size_classes[i + 1].size_min:
                logger.error(
                    f"{self.group_name}: overlapping size classes at {i}"
                )
                return False

        return True

    def total_count(self) -> float:
        """Return total individuals across all size classes.

        Returns
        -------
        float
            Sum of all counts.
        """
        return float(np.sum(self.counts))

    def mean_size(self) -> float:
        """Compute mean size (weighted by counts).

        Returns
        -------
        float
            Weighted mean midpoint across size classes.
        """
        if self.total_count() == 0:
            logger.warning(f"{self.group_name}: total_count is 0")
            return 0.0

        midpoints = np.array([sc.midpoint() for sc in self.size_classes])
        return float(np.average(midpoints, weights=self.counts))

    def to_dataframe(self) -> pd.DataFrame:
        """Export size distribution to DataFrame.

        Returns
        -------
        pd.DataFrame
            Columns: ['group_name', 'size_min', 'size_max', 'size_unit',
                     'midpoint', 'width', 'count'].
        """
        data = []
        for sc, count in zip(self.size_classes, self.counts):
            data.append(
                {
                    "group_name": self.group_name,
                    "size_min": sc.size_min,
                    "size_max": sc.size_max,
                    "size_unit": sc.size_unit,
                    "midpoint": sc.midpoint(),
                    "width": sc.width(),
                    "count": count,
                }
            )
        return pd.DataFrame(data)

    @classmethod
    def from_dataframe(cls, df: pd.DataFrame, group_name: str) -> SizeDistribution:
        """Load size distribution from DataFrame.

        Parameters
        ----------
        df : pd.DataFrame
            DataFrame with columns: size_min, size_max, size_unit, count.
            May also contain: midpoint, width (ignored).
        group_name : str
            Group identifier.

        Returns
        -------
        SizeDistribution
            Reconstructed distribution object.
        """
        required_cols = {"size_min", "size_max", "count"}
        if not required_cols.issubset(df.columns):
            raise ValueError(f"DataFrame missing required columns: {required_cols}")

        size_classes = []
        counts = []

        for _, row in df.iterrows():
            size_unit = row.get("size_unit", "cm")
            sc = SizeClass(
                size_min=float(row["size_min"]),
                size_max=float(row["size_max"]),
                size_unit=size_unit,
            )
            size_classes.append(sc)
            counts.append(float(row["count"]))

        return cls(
            group_name=group_name,
            size_classes=size_classes,
            counts=np.array(counts, dtype=np.float64),
        )


@dataclass
class SizeStructure:
    """Container for size distributions across multiple groups.

    Manages a collection of SizeDistribution objects and provides
    validation and export methods.

    Parameters
    ----------
    distributions : dict[str, SizeDistribution]
        Mapping from group_name to SizeDistribution.

    Attributes
    ----------
    distributions : dict[str, SizeDistribution]
        Collection of PSD data by group.
    """

    distributions: dict[str, SizeDistribution] = field(default_factory=dict)

    def add_distribution(self, distribution: SizeDistribution) -> None:
        """Add or replace a size distribution.

        Parameters
        ----------
        distribution : SizeDistribution
            Distribution object to add.
        """
        self.distributions[distribution.group_name] = distribution
        logger.debug(f"Added size distribution for {distribution.group_name}")

    def get_distribution(self, group_name: str) -> Optional[SizeDistribution]:
        """Retrieve a size distribution by group name.

        Parameters
        ----------
        group_name : str
            Name of the group.

        Returns
        -------
        SizeDistribution | None
            Distribution if found, else None.
        """
        return self.distributions.get(group_name)

    def validate(self) -> tuple[bool, list[str]]:
        """Validate all distributions for consistency.

        Returns
        -------
        tuple[bool, list[str]]
            (is_valid, error_messages). If is_valid is False, error_messages
            contains diagnostic details.
        """
        errors = []

        if len(self.distributions) == 0:
            errors.append("No distributions defined")
            return False, errors

        for group_name, dist in self.distributions.items():
            if not dist.is_valid():
                errors.append(f"{group_name}: distribution is invalid")

        return len(errors) == 0, errors

    def to_dataframe(self) -> pd.DataFrame:
        """Export all distributions to a single DataFrame.

        Returns
        -------
        pd.DataFrame
            Combined DataFrame with rows for all size classes across all groups.
            Columns: ['group_name', 'size_min', 'size_max', 'size_unit',
                     'midpoint', 'width', 'count'].
        """
        dfs = [dist.to_dataframe() for dist in self.distributions.values()]
        if not dfs:
            return pd.DataFrame()
        return pd.concat(dfs, ignore_index=True)

    @classmethod
    def from_dataframe(cls, df: pd.DataFrame) -> SizeStructure:
        """Load size distributions from DataFrame.

        Parameters
        ----------
        df : pd.DataFrame
            Combined DataFrame with a 'group_name' column grouping rows.
            Expected columns: group_name, size_min, size_max, size_unit, count.

        Returns
        -------
        SizeStructure
            Reconstructed structure with distributions for each unique group.
        """
        ss = cls()

        if "group_name" not in df.columns:
            raise ValueError("DataFrame must contain 'group_name' column")

        for group_name, group_df in df.groupby("group_name"):
            dist = SizeDistribution.from_dataframe(
                group_df.reset_index(drop=True), str(group_name)
            )
            ss.add_distribution(dist)

        return ss


def vb_length(age: float, loo: float, k: float, t0: float = 0.0) -> float:
    """Von Bertalanffy length at age.

    Computes length given age using the von Bertalanffy growth function:
    L(t) = L∞ * (1 - exp(-k * (t - t₀)))

    Parameters
    ----------
    age : float
        Age in years.
    loo : float
        Asymptotic (maximum) length (same units as returned value).
    k : float
        Growth constant (year⁻¹).
    t0 : float, optional
        Theoretical age at length 0 (years, default 0.0).

    Returns
    -------
    float
        Length at the given age.
    """
    if k <= 0:
        raise ValueError("k must be positive")
    return loo * (1.0 - np.exp(-k * (age - t0)))


def vb_age(length: float, loo: float, k: float, t0: float = 0.0) -> float:
    """Age from von Bertalanffy length.

    Inverts the von Bertalanffy equation to compute age given length:
    t = t₀ - ln(1 - L / L∞) / k

    Parameters
    ----------
    length : float
        Length in same units as loo.
    loo : float
        Asymptotic length.
    k : float
        Growth constant (year⁻¹).
    t0 : float, optional
        Theoretical age at length 0 (years, default 0.0).

    Returns
    -------
    float
        Age in years corresponding to the given length.

    Raises
    ------
    ValueError
        If length >= loo or k <= 0.
    """
    if k <= 0:
        raise ValueError("k must be positive")
    if length >= loo:
        raise ValueError("length must be less than loo (L∞)")

    return t0 - np.log(1.0 - length / loo) / k


def age_at_size_class(
    size_class: SizeClass, loo: float, k: float, t0: float = 0.0
) -> tuple[float, float]:
    """Compute age range for a size class using von Bertalanffy.

    Given a size interval, computes the corresponding age interval using
    the inverse von Bertalanffy function.

    Parameters
    ----------
    size_class : SizeClass
        Size interval (min, max).
    loo : float
        Asymptotic length in same units as size_class.
    k : float
        Growth constant (year⁻¹).
    t0 : float, optional
        Theoretical age at length 0 (default 0.0).

    Returns
    -------
    tuple[float, float]
        (age_at_min, age_at_max) in years. age_at_max corresponds to
        size_min, age_at_min corresponds to size_max (inverse relationship).
    """
    age_at_min = vb_age(size_class.size_min, loo, k, t0)
    age_at_max = vb_age(size_class.size_max, loo, k, t0)
    return (age_at_min, age_at_max)


def validate_size_structure(
    structure: SizeStructure,
) -> tuple[bool, list[str]]:
    """Comprehensive validation of a SizeStructure.

    Checks for:
    - Missing or empty distributions
    - Overlapping or gapped size ranges
    - Negative or inconsistent counts
    - Size classes not ordered

    Parameters
    ----------
    structure : SizeStructure
        Size structure to validate.

    Returns
    -------
    tuple[bool, list[str]]
        (is_valid, error_messages).
    """
    is_valid, errors = structure.validate()

    if not is_valid:
        return False, errors

    for group_name, dist in structure.distributions.items():
        if len(dist.size_classes) < 2:
            errors.append(f"{group_name}: fewer than 2 size classes")

        for i in range(len(dist.size_classes) - 1):
            curr = dist.size_classes[i]
            next_ = dist.size_classes[i + 1]

            if curr.size_max != next_.size_min:
                errors.append(
                    f"{group_name}: gap/overlap between classes {i} and {i + 1} "
                    f"({curr.size_max} vs {next_.size_min})"
                )

    return len(errors) == 0, errors


def estimate_psd_from_vb(
    loo: float,
    k: float,
    t0: float,
    recruitment_rate: float,
    time_steps: int,
    n_size_classes: int,
    size_min: float = 0.1,
    size_max: float = 100.0,
    size_unit: str = "cm",
    group_name: str = "Group",
) -> SizeDistribution:
    """Estimate PSD from von Bertalanffy growth parameters.

    Generates a theoretical population size distribution by simulating
    cohorts of individuals aging under the von Bertalanffy growth function
    and applying recruitment. This is a simplified approach intended for
    initialization or sensitivity analysis.

    Parameters
    ----------
    loo : float
        Asymptotic length.
    k : float
        Growth constant (year⁻¹).
    t0 : float
        Theoretical age at length 0 (years).
    recruitment_rate : float
        Annual recruitment (new individuals per year).
    time_steps : int
        Number of annual time steps to simulate.
    n_size_classes : int
        Number of size classes to create.
    size_min : float, optional
        Minimum size (default 0.1, same units as loo).
    size_max : float, optional
        Maximum size (default 100.0).
    size_unit : str, optional
        Unit of measurement ('cm' or 'mm', default 'cm').
    group_name : str, optional
        Group identifier (default 'Group').

    Returns
    -------
    SizeDistribution
        Theoretical PSD with uniform bin widths.

    Notes
    -----
    This function uses a simple discrete-age cohort model with no natural
    mortality. The distribution is for illustrative and initialization use;
    a full population model should include mortality and recruitment dynamics.
    """
    if k <= 0:
        raise ValueError("k must be positive")
    if n_size_classes < 1:
        raise ValueError("n_size_classes must be >= 1")

    bin_width = (size_max - size_min) / float(n_size_classes)
    size_classes = []
    for i in range(n_size_classes):
        sc = SizeClass(
            size_min=size_min + i * bin_width,
            size_max=size_min + (i + 1) * bin_width,
            size_unit=size_unit,
        )
        size_classes.append(sc)

    counts = np.zeros(n_size_classes, dtype=np.float64)

    for ts in range(time_steps):
        cohort_age = ts + 1
        cohort_size = vb_length(float(cohort_age), loo, k, t0)

        if cohort_size > size_max:
            cohort_size = size_max - 0.001

        for i, sc in enumerate(size_classes):
            if sc.contains(cohort_size):
                counts[i] += recruitment_rate
                break

    return SizeDistribution(
        group_name=group_name,
        size_classes=size_classes,
        counts=counts,
    )



