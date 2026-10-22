"""Import/export validation utilities for PyPath Shiny app.

Provides comprehensive validation for CSV file uploads and data format,
including schema validation, data type checking, and completeness verification.
"""

from typing import List, Optional, Tuple

import pandas as pd


# Required columns for model.csv
REQUIRED_MODEL_COLUMNS = {
    "Group",  # Group name
    "Type",  # Group type (0-3)
    "Biomass",  # Biomass (t/km²)
    "PB",  # Production/Biomass ratio (per year)
}

# Optional but important model columns
IMPORTANT_MODEL_COLUMNS = {
    "QB",  # Consumption/Biomass ratio
    "EE",  # Ecotrophic efficiency
    "ProdCons",  # Production/Consumption ratio
    "BioAcc",  # Biomass accumulation
    "Unassim",  # Unassimilated fraction
    "DetInput",  # Detritus input
}

# All valid model columns
ALL_MODEL_COLUMNS = REQUIRED_MODEL_COLUMNS | IMPORTANT_MODEL_COLUMNS | {
    "Detritus",  # Detritus fate
    "Discards",  # Discard fate
}


def validate_model_csv_structure(
    df: pd.DataFrame,
) -> Tuple[bool, List[str]]:
    """Validate model.csv file structure and required columns.

    Parameters
    ----------
    df : pd.DataFrame
        Model DataFrame loaded from CSV

    Returns
    -------
    Tuple[bool, List[str]]
        (is_valid, error_messages)
        - is_valid: True if structure is valid
        - error_messages: List of validation errors (empty if valid)

    Examples
    --------
    >>> import pandas as pd
    >>> df = pd.DataFrame({
    ...     "Group": ["Fish", "Plankton"],
    ...     "Type": [0, 1],
    ...     "Biomass": [10.0, 5.0],
    ...     "PB": [0.8, 50.0],
    ... })
    >>> is_valid, errors = validate_model_csv_structure(df)
    >>> is_valid
    True
    """
    errors = []

    if df is None or df.empty:
        return False, ["Model CSV is empty or could not be loaded."]

    # Check for required columns
    missing_cols = REQUIRED_MODEL_COLUMNS - set(df.columns)
    if missing_cols:
        missing_str = ", ".join(sorted(missing_cols))
        errors.append(
            f"Model CSV is missing required columns: {missing_str}.\n\n"
            f"Required columns are: Group, Type, Biomass, PB.\n\n"
            f"Current columns: {', '.join(df.columns.tolist())}"
        )
        return False, errors

    # Note about missing important columns (warning only, not blocking)
    missing_important = IMPORTANT_MODEL_COLUMNS - set(df.columns)
    if missing_important:
        missing_str = ", ".join(sorted(missing_important))
        # This is just informational - important columns are optional
        # Only add if there are missing critical columns
        if missing_str:  # Only warn if actual important columns missing
            pass  # Could add as warning but dont block loading

    # Check for data types
    if "Type" in df.columns:
        try:
            types = pd.to_numeric(df["Type"], errors="coerce")
            if types.isna().any():
                errors.append(
                    f"Type column contains non-numeric values.\n\n"
                    f"Type must be numeric: 0 (consumer), 1 (producer), "
                    f"2 (detritus), 3 (fleet).\n\n"
                    f"Found: {df.loc[types.isna(), 'Type'].unique().tolist()}"
                )
        except Exception as e:
            errors.append(f"Could not validate Type column: {e}")

    if "Biomass" in df.columns:
        try:
            biomass = pd.to_numeric(df["Biomass"], errors="coerce")
            if biomass.isna().any():
                invalid_count = biomass.isna().sum()
                errors.append(
                    f"Biomass column contains {invalid_count} non-numeric values.\n\n"
                    f"Biomass must be numeric (t/km²).\n\n"
                    f"Use 999999 for unknown/estimated values."
                )
        except Exception as e:
            errors.append(f"Could not validate Biomass column: {e}")

    if "PB" in df.columns:
        try:
            pb = pd.to_numeric(df["PB"], errors="coerce")
            if pb.isna().any():
                invalid_count = pb.isna().sum()
                errors.append(
                    f"PB column contains {invalid_count} non-numeric values.\n\n"
                    f"P/B must be numeric (per year).\n\n"
                    f"Use 999999 for unknown/estimated values."
                )
        except Exception as e:
            errors.append(f"Could not validate PB column: {e}")

    return len(errors) == 0, errors


def validate_diet_csv_structure(
    df: pd.DataFrame,
    model_df: Optional[pd.DataFrame] = None,
) -> Tuple[bool, List[str]]:
    """Validate diet.csv file structure and consistency with model.csv.

    Parameters
    ----------
    df : pd.DataFrame
        Diet DataFrame loaded from CSV
    model_df : pd.DataFrame, optional
        Model DataFrame for cross-validation

    Returns
    -------
    Tuple[bool, List[str]]
        (is_valid, error_messages)

    Examples
    --------
    >>> diet = pd.DataFrame({
    ...     "Group": ["Fish", "Plankton"],
    ...     "Fish": [0.0, 0.5],
    ...     "Plankton": [1.0, 0.5],
    ... })
    >>> is_valid, errors = validate_diet_csv_structure(diet)
    >>> is_valid
    True
    """
    errors = []

    if df is None or df.empty:
        return False, ["Diet CSV is empty or could not be loaded."]

    # Check for Group column
    if "Group" not in df.columns:
        errors.append(
            f"Diet CSV must have a 'Group' column.\n\n"
            f"Current columns: {', '.join(df.columns.tolist())}"
        )
        return False, errors

    # Get group names and predator columns
    groups = df["Group"].tolist()
    predator_cols = [c for c in df.columns if c != "Group"]

    # Check that all groups in diet match model groups (if provided)
    if model_df is not None and "Group" in model_df.columns:
        model_groups = set(model_df["Group"])
        diet_groups = set(groups)
        
        if not diet_groups.issubset(model_groups):
            extra_groups = diet_groups - model_groups
            errors.append(
                f"Diet CSV has groups not in model: {', '.join(extra_groups)}.\n\n"
                f"Diet matrix rows (prey) should match model groups.\n\n"
                f"Remove extra groups from diet or add them to model."
            )

        if not model_groups.issubset(diet_groups):
            missing_groups = model_groups - diet_groups
            errors.append(
                f"Diet CSV is missing model groups: {', '.join(missing_groups)}.\n\n"
                f"Diet matrix should have rows for all model groups."
            )

    # Check that predator columns are reasonable
    if not predator_cols:
        errors.append("Diet CSV must have predator columns (columns other than 'Group').")
        return False, errors

    # Validate diet data types
    for pred_col in predator_cols:
        if pred_col not in df.columns:
            continue
        try:
            diet_vals = pd.to_numeric(df[pred_col], errors="coerce")
            if diet_vals.isna().any():
                invalid_count = diet_vals.isna().sum()
                errors.append(
                    f"Diet column {repr(pred_col)} has {invalid_count} non-numeric values.\n\n"
                    f"All diet fractions must be numeric (0.0-1.0)."
                )
        except Exception as e:
            errors.append(f"Could not validate diet column {repr(pred_col)}: {e}")

    return len(errors) == 0, errors


def validate_csv_file_pair(
    model_df: pd.DataFrame,
    diet_df: pd.DataFrame,
) -> Tuple[bool, List[str]]:
    """Validate model and diet CSV files together for consistency.

    Parameters
    ----------
    model_df : pd.DataFrame
        Model DataFrame
    diet_df : pd.DataFrame
        Diet DataFrame

    Returns
    -------
    Tuple[bool, List[str]]
        (is_valid, error_messages)
    """
    errors = []

    # Validate each file individually
    model_valid, model_errors = validate_model_csv_structure(model_df)
    diet_valid, diet_errors = validate_diet_csv_structure(diet_df, model_df)

    errors.extend(model_errors)
    errors.extend(diet_errors)

    # Cross-validate only if both files have basic structure
    if model_valid and diet_valid:
        model_groups = set(model_df["Group"])
        diet_groups = set(diet_df["Group"])
        
        # Check that diet rows match model groups
        if model_groups != diet_groups:
            missing_in_diet = model_groups - diet_groups
            extra_in_diet = diet_groups - model_groups
            
            if missing_in_diet:
                errors.append(
                    f"Model has groups not in diet: {', '.join(missing_in_diet)}.\n\n"
                    f"Add rows to diet matrix for these groups."
                )
            
            if extra_in_diet:
                errors.append(
                    f"Diet has groups not in model: {', '.join(extra_in_diet)}.\n\n"
                    f"Remove rows from diet or add groups to model."
                )

    return len(errors) == 0, errors


def get_import_error_suggestion(error_message: str) -> str:
    """Generate import error recovery suggestion.

    Parameters
    ----------
    error_message : str
        The validation error message

    Returns
    -------
    str
        A suggestion for fixing the error
    """
    msg_lower = error_message.lower()
    
    if "missing" in msg_lower and "column" in msg_lower:
        return "Quick fix: Ensure your CSV file has all required columns: Group, Type, Biomass, PB"
    elif "non-numeric" in msg_lower and "type" in msg_lower:
        return "Quick fix: Type must be 0 (consumer), 1 (producer), 2 (detritus), or 3 (fleet)"
    elif "non-numeric" in msg_lower and "biomass" in msg_lower:
        return "Quick fix: Use 999999 for unknown biomass; numeric values required (e.g., 10.5)"
    elif "non-numeric" in msg_lower and "pb" in msg_lower:
        return "Quick fix: P/B must be per-year rate (e.g., 0.8 for fish, 50 for phytoplankton)"
    elif "non-numeric" in msg_lower and "diet" in msg_lower:
        return "Quick fix: Diet fractions must be numeric (0.0 to 1.0), use 999999 for unknown"
    elif "missing.*diet" in msg_lower:
        return "Quick fix: Diet CSV must have 'Group' column and predator columns"
    elif "groups not in" in msg_lower or "missing.*group" in msg_lower:
        return "Quick fix: Ensure diet and model CSV have matching group names"
    
    return ""
