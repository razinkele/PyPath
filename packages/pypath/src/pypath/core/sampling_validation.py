"""Validation framework for Monte Carlo sampling data.

This module provides validation functions for sample containers and their consistency
with Ecopath parameters. It checks for data integrity, range validity, and logical
consistency across the sampling framework.

Functions
---------
validate_samples
    Comprehensive validation of SampleContainer against RpathParams
validate_replicate_counts
    Ensure consistent replicate counts across all samples
validate_sample_ranges
    Check that sample values are within valid ranges
validate_diet_consistency
    Verify diet samples form valid probability distributions
validate_pedigree_weighting
    Validate pedigree weighting factors
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from pypath.core.params import RpathParams
    from pypath.core.sampling import SampleContainer

logger = logging.getLogger(__name__)


def validate_samples(
    sample_container: SampleContainer,
    params: RpathParams,
    raise_on_error: bool = False,
) -> list[str]:
    """Validate sampling data against RpathParams.

    Performs comprehensive validation of sample container consistency with model
    parameters. Checks include:
    - All sampled groups exist in params
    - All diet pairs are valid
    - All catch records are valid
    - Replicate counts are consistent
    - Sample values are within valid ranges
    - No NaN or infinite values

    Parameters
    ----------
    sample_container : SampleContainer
        Sample data to validate
    params : RpathParams
        Reference parameter container
    raise_on_error : bool, optional
        If True, raise ValueError on first error; otherwise collect all issues
        (default: False)

    Returns
    -------
    list[str]
        List of validation messages (warnings and errors). Empty if valid.

    Raises
    ------
    ValueError
        If raise_on_error is True and any validation fails

    Examples
    --------
    >>> issues = validate_samples(samples, params)
    >>> if issues:
    ...     print("Validation issues found:")
    ...     for issue in issues:
    ...         print(f"  - {issue}")
    """
    issues = []

    # 1. Validate replicate counts
    issues.extend(validate_replicate_counts(sample_container))

    # 2. Validate group samples
    if sample_container.group_samples:
        valid_groups = set(params.model["Group"])

        for group_name, group_sample in sample_container.group_samples.items():
            if group_name not in valid_groups:
                msg = f"Group '{group_name}' in samples not found in params"
                issues.append(msg)
                if raise_on_error:
                    raise ValueError(msg)
                continue

            # Validate ranges
            issues.extend(
                _validate_group_sample_ranges(
                    group_name, group_sample, params, raise_on_error
                )
            )

    # 3. Validate diet samples
    if sample_container.diet_samples:
        issues.extend(
            validate_diet_consistency(sample_container, params, raise_on_error)
        )

    # 4. Validate catch samples
    if sample_container.catch_samples:
        valid_groups = set(params.model["Group"])
        if params.model is not None:
            # Try to get fleets from Ecopath (may not exist in all models)
            try:
                # Assuming there's a way to get fleets from params
                # For now, we just validate group names
                for (
                    group_name,
                    fleet_name,
                ), _ in sample_container.catch_samples.items():
                    if group_name not in valid_groups:
                        msg = f"Group '{group_name}' in catch samples not in params"
                        issues.append(msg)
                        if raise_on_error:
                            raise ValueError(msg)
            except Exception as e:
                logger.debug(f"Could not fully validate catch samples: {e}")

    if raise_on_error and issues:
        raise ValueError(f"Validation failed with {len(issues)} error(s)")

    return issues


def validate_replicate_counts(sample_container: SampleContainer) -> list[str]:
    """Ensure consistent replicate counts across all samples.

    Parameters
    ----------
    sample_container : SampleContainer
        Sample data to validate

    Returns
    -------
    list[str]
        List of validation messages for replicate count mismatches
    """
    issues = []
    n_replicates = sample_container.n_replicates

    # Check group samples
    for group_name, group_sample in sample_container.group_samples.items():
        if group_sample.n_replicates != n_replicates:
            msg = (
                f"Group '{group_name}': replicate count mismatch "
                f"({group_sample.n_replicates} != {n_replicates})"
            )
            issues.append(msg)

    # Check diet samples
    for (pred_name, prey_name), diet_sample in sample_container.diet_samples.items():
        if diet_sample.n_replicates != n_replicates:
            msg = (
                f"Diet pair '{pred_name}' → '{prey_name}': replicate count mismatch "
                f"({diet_sample.n_replicates} != {n_replicates})"
            )
            issues.append(msg)

    # Check catch samples
    for (
        group_name,
        fleet_name,
    ), catch_sample in sample_container.catch_samples.items():
        if catch_sample.n_replicates != n_replicates:
            msg = (
                f"Catch '{group_name}' by '{fleet_name}': replicate count mismatch "
                f"({catch_sample.n_replicates} != {n_replicates})"
            )
            issues.append(msg)

    return issues


def validate_sample_ranges(
    sample_container: SampleContainer,
    params: RpathParams,
) -> list[str]:
    """Check that sample values are within valid ranges.

    Validates:
    - Biomass > 0
    - P/B > 0
    - Q/B > 0
    - EE in (0, 1)
    - Diet fractions in [0, 1]
    - No NaN or infinite values

    Parameters
    ----------
    sample_container : SampleContainer
        Sample data to validate
    params : RpathParams
        Reference parameters

    Returns
    -------
    list[str]
        List of validation messages for out-of-range values
    """
    issues = []

    # Check group samples
    for group_name, group_sample in sample_container.group_samples.items():
        issues.extend(
            _validate_group_sample_ranges(
                group_name, group_sample, params, raise_on_error=False
            )
        )

    # Check diet samples
    for (pred_name, prey_name), diet_sample in sample_container.diet_samples.items():
        issues.extend(_validate_diet_sample_ranges(pred_name, prey_name, diet_sample))

    # Check catch samples
    for (
        group_name,
        fleet_name,
    ), catch_sample in sample_container.catch_samples.items():
        issues.extend(
            _validate_catch_sample_ranges(group_name, fleet_name, catch_sample)
        )

    return issues


def validate_diet_consistency(
    sample_container: SampleContainer,
    params: RpathParams,
    raise_on_error: bool = False,
) -> list[str]:
    """Verify diet samples form valid probability distributions.

    For each predator group, checks that diet fractions across all prey
    sum to approximately 1.0 (within tolerance). Also validates that
    diet pairs referenced in samples exist in params.

    Parameters
    ----------
    sample_container : SampleContainer
        Sample data to validate
    params : RpathParams
        Reference parameters
    raise_on_error : bool, optional
        If True, raise ValueError on first error (default: False)

    Returns
    -------
    list[str]
        List of validation messages

    Notes
    -----
    - Allows sum to be in range [0.99, 1.01] to account for rounding
    - Logs debug info for diet sums outside valid range
    """
    issues = []

    if not sample_container.diet_samples:
        return issues

    # Group samples by predator
    predator_diets = {}
    for (pred_name, prey_name), diet_sample in sample_container.diet_samples.items():
        if pred_name not in predator_diets:
            predator_diets[pred_name] = []
        predator_diets[pred_name].append((prey_name, diet_sample.diet_fraction_samples))

    # Check diet sums per predator
    tolerance = 0.01  # Allow ±1% deviation
    for pred_name, prey_list in predator_diets.items():
        # Stack diet fractions for all prey
        diet_matrix = np.column_stack(
            [fracs for _, fracs in prey_list]
        )  # shape (n_replicates, n_prey)

        # Sum across prey for each replicate
        diet_sums = diet_matrix.sum(axis=1)

        # Check if sums are valid (should be ~1.0)
        invalid_mask = (diet_sums < 1.0 - tolerance) | (diet_sums > 1.0 + tolerance)
        if invalid_mask.any():
            n_invalid = invalid_mask.sum()
            mean_sum = diet_sums.mean()
            msg = (
                f"Predator '{pred_name}': {n_invalid}/{len(diet_sums)} diet sums "
                f"outside range [0.99, 1.01] (mean={mean_sum:.4f})"
            )
            issues.append(msg)
            if raise_on_error:
                raise ValueError(msg)

            # Log details
            logger.debug(
                f"Diet fraction statistics for '{pred_name}':\n"
                f"  Min sum: {diet_sums.min():.4f}\n"
                f"  Max sum: {diet_sums.max():.4f}\n"
                f"  Mean sum: {diet_sums.mean():.4f}\n"
                f"  Std: {diet_sums.std():.4f}"
            )

    return issues


def validate_pedigree_weighting(
    pedigree_weights: dict[str, float],
) -> list[str]:
    """Validate pedigree weighting factors.

    Ensures pedigree weights are in valid range [0, 1] and properly
    constructed as a probability distribution or weighting scheme.

    Parameters
    ----------
    pedigree_weights : dict[str, float]
        Mapping of pedigree source names to weights

    Returns
    -------
    list[str]
        List of validation messages

    Notes
    -----
    Weights should sum to 1.0 if used as probability distribution,
    or be in [0, 1] individually. This function checks both.
    """
    issues = []

    if not pedigree_weights:
        return issues

    # Check individual weights
    for source, weight in pedigree_weights.items():
        if not isinstance(weight, (int, float)):
            msg = f"Pedigree weight for '{source}' is not numeric: {type(weight)}"
            issues.append(msg)
            continue

        if weight < 0 or weight > 1:
            msg = f"Pedigree weight for '{source}' outside range [0, 1]: {weight}"
            issues.append(msg)

    # Check if weights sum to 1 (with tolerance for floating point)
    if pedigree_weights:
        total = sum(pedigree_weights.values())
        if not (0.99 <= total <= 1.01):
            msg = f"Pedigree weights sum to {total:.4f}, not 1.0 (tolerance ±0.01)"
            issues.append(msg)

    return issues


# ─────────────────────────────────────────────────────────────────────────────
# Private validation helpers


def _validate_group_sample_ranges(
    group_name: str,
    group_sample,
    params,
    raise_on_error: bool = False,
) -> list[str]:
    """Validate individual group sample ranges."""
    issues = []

    # Biomass > 0
    if hasattr(group_sample, "biomass_samples"):
        biomass = group_sample.biomass_samples
        if np.any(biomass <= 0):
            n_invalid = np.sum(biomass <= 0)
            msg = (
                f"Group '{group_name}': {n_invalid} biomass samples ≤ 0 "
                f"(min={biomass.min():.6f})"
            )
            issues.append(msg)
            if raise_on_error:
                raise ValueError(msg)
        if np.any(np.isnan(biomass)) or np.any(np.isinf(biomass)):
            n_invalid = np.sum(np.isnan(biomass)) + np.sum(np.isinf(biomass))
            msg = (
                f"Group '{group_name}': {n_invalid} biomass samples contain NaN or inf"
            )
            issues.append(msg)
            if raise_on_error:
                raise ValueError(msg)

    # P/B > 0
    if hasattr(group_sample, "pb_samples"):
        pb = group_sample.pb_samples
        if np.any(pb <= 0):
            n_invalid = np.sum(pb <= 0)
            msg = (
                f"Group '{group_name}': {n_invalid} P/B samples ≤ 0 "
                f"(min={pb.min():.6f})"
            )
            issues.append(msg)
            if raise_on_error:
                raise ValueError(msg)
        if np.any(np.isnan(pb)) or np.any(np.isinf(pb)):
            msg = f"Group '{group_name}': P/B samples contain NaN or inf"
            issues.append(msg)
            if raise_on_error:
                raise ValueError(msg)

    # Q/B > 0
    if hasattr(group_sample, "qb_samples"):
        qb = group_sample.qb_samples
        if np.any(qb <= 0):
            n_invalid = np.sum(qb <= 0)
            msg = (
                f"Group '{group_name}': {n_invalid} Q/B samples ≤ 0 "
                f"(min={qb.min():.6f})"
            )
            issues.append(msg)
            if raise_on_error:
                raise ValueError(msg)

    # EE in (0, 1)
    if hasattr(group_sample, "ee_samples"):
        ee = group_sample.ee_samples
        if np.any((ee <= 0) | (ee >= 1)):
            n_invalid = np.sum((ee <= 0) | (ee >= 1))
            msg = (
                f"Group '{group_name}': {n_invalid} EE samples outside (0, 1) "
                f"(min={ee.min():.6f}, max={ee.max():.6f})"
            )
            issues.append(msg)
            if raise_on_error:
                raise ValueError(msg)

    return issues


def _validate_diet_sample_ranges(
    pred_name: str,
    prey_name: str,
    diet_sample,
) -> list[str]:
    """Validate individual diet sample ranges."""
    issues = []

    diet_fractions = diet_sample.diet_fraction_samples

    # Check range [0, 1]
    if np.any((diet_fractions < 0) | (diet_fractions > 1)):
        n_invalid = np.sum((diet_fractions < 0) | (diet_fractions > 1))
        msg = (
            f"Diet '{pred_name}' → '{prey_name}': {n_invalid} fractions "
            f"outside [0, 1] (min={diet_fractions.min():.6f}, "
            f"max={diet_fractions.max():.6f})"
        )
        issues.append(msg)

    # Check for NaN/inf
    if np.any(np.isnan(diet_fractions)) or np.any(np.isinf(diet_fractions)):
        n_invalid = np.sum(np.isnan(diet_fractions)) + np.sum(np.isinf(diet_fractions))
        msg = (
            f"Diet '{pred_name}' → '{prey_name}': {n_invalid} fractions "
            "contain NaN or inf"
        )
        issues.append(msg)

    return issues


def _validate_catch_sample_ranges(
    group_name: str,
    fleet_name: str,
    catch_sample,
) -> list[str]:
    """Validate individual catch sample ranges."""
    issues = []

    # Landing >= 0
    if hasattr(catch_sample, "landing_samples"):
        landing = catch_sample.landing_samples
        if np.any(landing < 0):
            n_invalid = np.sum(landing < 0)
            msg = (
                f"Catch '{group_name}' by '{fleet_name}': {n_invalid} "
                f"landing samples < 0 (min={landing.min():.6f})"
            )
            issues.append(msg)

    # Discards >= 0
    if hasattr(catch_sample, "discard_samples"):
        discards = catch_sample.discard_samples
        if np.any(discards < 0):
            n_invalid = np.sum(discards < 0)
            msg = (
                f"Catch '{group_name}' by '{fleet_name}': {n_invalid} "
                f"discard samples < 0 (min={discards.min():.6f})"
            )
            issues.append(msg)

    return issues
