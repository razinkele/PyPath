"""Input validation utilities for PyPath Shiny app.

This module provides validation functions that use the centralized ValidationConfig
to ensure parameters are within acceptable ranges and provide helpful error messages.
"""

from typing import Optional, List, Tuple, Union
import pandas as pd
import numpy as np

try:
    from app.config import VALIDATION, VALID_GROUP_TYPES, NO_DATA_VALUE
except ModuleNotFoundError:
    # When running from app directory, use relative import
    import sys
    from pathlib import Path
    app_dir = Path(__file__).parent.parent
    if str(app_dir) not in sys.path:
        sys.path.insert(0, str(app_dir))
    from config import VALIDATION, VALID_GROUP_TYPES, NO_DATA_VALUE


def validate_group_types(types: Union[List[int], np.ndarray, pd.Series]) -> Tuple[bool, Optional[str]]:
    """Validate that all group types are valid.

    Parameters
    ----------
    types : Union[List[int], np.ndarray, pd.Series]
        Group type codes to validate

    Returns
    -------
    Tuple[bool, Optional[str]]
        (is_valid, error_message)
        - is_valid: True if all types are valid
        - error_message: None if valid, otherwise helpful error message

    Examples
    --------
    >>> is_valid, error = validate_group_types([0, 1, 2, 3])
    >>> is_valid
    True
    >>> is_valid, error = validate_group_types([0, 1, 99])
    >>> is_valid
    False
    >>> "99" in error
    True
    """
    types_array = np.array(types)
    invalid_types = [t for t in types_array if t not in VALIDATION.valid_group_types]

    if invalid_types:
        unique_invalid = list(set(invalid_types))
        error_msg = (
            f"Invalid group types found: {unique_invalid}\n\n"
            f"Valid group types are:\n"
            f"  0 = Consumer (fish, invertebrates)\n"
            f"  1 = Producer (phytoplankton, plants)\n"
            f"  2 = Detritus (organic matter)\n"
            f"  3 = Fleet (fishing gear)\n\n"
            f"Please check your model definition."
        )
        return False, error_msg

    return True, None


def validate_biomass(biomass: Union[float, np.ndarray, pd.Series],
                     group_name: Optional[str] = None) -> Tuple[bool, Optional[str]]:
    """Validate biomass values are within acceptable range.

    Parameters
    ----------
    biomass : Union[float, np.ndarray, pd.Series]
        Biomass value(s) to validate (t/km²)
    group_name : Optional[str]
        Name of group for error message context

    Returns
    -------
    Tuple[bool, Optional[str]]
        (is_valid, error_message)

    Examples
    --------
    >>> is_valid, error = validate_biomass(10.5, "Fish")
    >>> is_valid
    True
    >>> is_valid, error = validate_biomass(-5.0, "Fish")
    >>> is_valid
    False
    """
    biomass_array = np.atleast_1d(biomass)

    # Check for negative values
    if np.any(biomass_array < VALIDATION.min_biomass):
        group_str = f" for group '{group_name}'" if group_name else ""
        error_msg = (
            f"Negative biomass values found{group_str}.\n\n"
            f"Biomass must be ≥ {VALIDATION.min_biomass} t/km².\n"
            f"Found minimum: {biomass_array.min():.6f}\n\n"
            f"Solutions:\n"
            f"  1. Check for data entry errors\n"
            f"  2. Use {NO_DATA_VALUE} for unknown biomass (will be estimated)\n"
            f"  3. Remove groups with zero biomass"
        )
        return False, error_msg

    # Check for extremely high values (likely errors)
    if np.any(biomass_array > VALIDATION.max_biomass):
        group_str = f" for group '{group_name}'" if group_name else ""
        error_msg = (
            f"Extremely high biomass values found{group_str}.\n\n"
            f"Biomass should be < {VALIDATION.max_biomass:,.0f} t/km².\n"
            f"Found maximum: {biomass_array.max():,.2f}\n\n"
            f"This is likely a data entry error. "
            f"Check your biomass units (should be t/km², not kg or tons)."
        )
        return False, error_msg

    return True, None


def validate_pb(pb: Union[float, np.ndarray, pd.Series],
                group_name: Optional[str] = None,
                group_type: Optional[int] = None) -> Tuple[bool, Optional[str]]:
    """Validate Production/Biomass ratio.

    Parameters
    ----------
    pb : Union[float, np.ndarray, pd.Series]
        P/B ratio(s) to validate (year⁻¹)
    group_name : Optional[str]
        Name of group for error message context
    group_type : Optional[int]
        Group type (0=Consumer, 1=Producer, 2=Detritus, 3=Fleet)
        Used to apply type-specific thresholds

    Returns
    -------
    Tuple[bool, Optional[str]]
        (is_valid, error_message)
    """
    pb_array = np.atleast_1d(pb)

    if np.any(pb_array < VALIDATION.min_pb):
        group_str = f" for group '{group_name}'" if group_name else ""
        error_msg = (
            f"Negative P/B values found{group_str}.\n\n"
            f"P/B must be ≥ {VALIDATION.min_pb}.\n"
            f"Found minimum: {pb_array.min():.6f}\n\n"
            f"P/B represents production per unit biomass per year.\n"
            f"Use {NO_DATA_VALUE} for unknown values."
        )
        return False, error_msg

    # Use type-specific threshold: producers can have higher P/B
    max_pb_threshold = VALIDATION.max_pb_producer if group_type == 1 else VALIDATION.max_pb

    if np.any(pb_array > max_pb_threshold):
        group_str = f" for group '{group_name}'" if group_name else ""
        group_type_str = f" (type={group_type})" if group_type is not None else ""

        error_msg = (
            f"Extremely high P/B values found{group_str}{group_type_str}.\n\n"
            f"P/B should be < {max_pb_threshold} year⁻¹.\n"
            f"Found maximum: {pb_array.max():.2f}\n\n"
            f"Typical ranges:\n"
            f"  - Small fish, invertebrates: 1-10\n"
            f"  - Large fish: 0.1-1\n"
            f"  - Phytoplankton/Producers: 20-250\n\n"
            f"Check if you're using the correct time units (per year)."
        )
        return False, error_msg

    return True, None


def validate_ee(ee: Union[float, np.ndarray, pd.Series],
                group_name: Optional[str] = None) -> Tuple[bool, Optional[str]]:
    """Validate Ecotrophic Efficiency.

    Parameters
    ----------
    ee : Union[float, np.ndarray, pd.Series]
        EE value(s) to validate (0-1)
    group_name : Optional[str]
        Name of group for error message context

    Returns
    -------
    Tuple[bool, Optional[str]]
        (is_valid, error_message)
    """
    ee_array = np.atleast_1d(ee)

    if np.any(ee_array < VALIDATION.min_ee):
        group_str = f" for group '{group_name}'" if group_name else ""
        error_msg = (
            f"Negative EE values found{group_str}.\n\n"
            f"EE must be between 0 and 1.\n"
            f"Found minimum: {ee_array.min():.6f}\n\n"
            f"EE represents the fraction of production that is used in the system."
        )
        return False, error_msg

    if np.any(ee_array > VALIDATION.max_ee):
        group_str = f" for group '{group_name}'" if group_name else ""
        error_msg = (
            f"EE exceeds 1.0{group_str} - model is unbalanced!\n\n"
            f"Found maximum: {ee_array.max():.4f}\n\n"
            f"EE > 1 means more production is consumed than produced.\n\n"
            f"Solutions:\n"
            f"  1. Reduce predation on this group (lower diet fractions)\n"
            f"  2. Increase production (higher P/B)\n"
            f"  3. Increase biomass\n"
            f"  4. Reduce fishing mortality\n\n"
            f"The model must be rebalanced before running Ecosim."
        )
        return False, error_msg

    return True, None


def validate_model_parameters(
    model_df: pd.DataFrame,
    check_groups: bool = True,
    check_biomass: bool = True,
    check_pb: bool = True,
    check_ee: bool = True
) -> Tuple[bool, List[str]]:
    """Validate all parameters in a model DataFrame.

    Parameters
    ----------
    model_df : pd.DataFrame
        Model parameters DataFrame with columns: Group, Type, Biomass, PB, EE, etc.
    check_groups : bool, default True
        Whether to validate group types
    check_biomass : bool, default True
        Whether to validate biomass values
    check_pb : bool, default True
        Whether to validate P/B ratios
    check_ee : bool, default True
        Whether to validate EE values

    Returns
    -------
    Tuple[bool, List[str]]
        (all_valid, error_messages)
        - all_valid: True if all validations pass
        - error_messages: List of error messages (empty if all_valid)

    Examples
    --------
    >>> df = pd.DataFrame({
    ...     'Group': ['Fish', 'Plankton'],
    ...     'Type': [0, 1],
    ...     'Biomass': [10.5, 5.0],
    ...     'PB': [0.8, 50.0],
    ...     'EE': [0.9, 0.8]
    ... })
    >>> is_valid, errors = validate_model_parameters(df)
    >>> is_valid
    True
    """
    errors = []

    # Validate group types
    if check_groups and 'Type' in model_df.columns:
        is_valid, error = validate_group_types(model_df['Type'])
        if not is_valid:
            errors.append(error)

    # Validate each group's parameters
    for idx, row in model_df.iterrows():
        group_name = row.get('Group', f'Group {idx}')

        # Skip validation for detritus and fleets (type 2, 3)
        group_type = row.get('Type', 0)
        if group_type in [2, 3]:
            continue

        # Validate biomass
        if check_biomass and 'Biomass' in row:
            biomass = row['Biomass']
            if biomass != NO_DATA_VALUE:  # Skip no-data values
                is_valid, error = validate_biomass(biomass, group_name)
                if not is_valid:
                    errors.append(error)

        # Validate P/B
        if check_pb and 'PB' in row:
            pb = row['PB']
            if pb != NO_DATA_VALUE:
                is_valid, error = validate_pb(pb, group_name, group_type)
                if not is_valid:
                    errors.append(error)

        # Validate EE
        if check_ee and 'EE' in row:
            ee = row['EE']
            if ee != NO_DATA_VALUE:
                is_valid, error = validate_ee(ee, group_name)
                if not is_valid:
                    errors.append(error)

    return len(errors) == 0, errors
