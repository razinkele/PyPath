"""Import/export validation utilities for PyPath Shiny app.

Provides comprehensive validation for CSV file uploads and data format,
including schema validation, data type checking, and completeness verification.
"""

from typing import List, Optional, Tuple

import numpy as np
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
ALL_MODEL_COLUMNS = (
    REQUIRED_MODEL_COLUMNS
    | IMPORTANT_MODEL_COLUMNS
    | {
        "Detritus",  # Detritus fate
        "Discards",  # Discard fate
    }
)


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
            invalid_count = _count_non_numeric(df["Biomass"])
            if invalid_count:
                errors.append(
                    f"Biomass column contains {invalid_count} non-numeric values.\n\n"
                    f"Biomass must be numeric (t/km²).\n\n"
                    f"Leave the cell blank for a value Ecopath should estimate."
                )
        except Exception as e:
            errors.append(f"Could not validate Biomass column: {e}")

    if "PB" in df.columns:
        try:
            invalid_count = _count_non_numeric(df["PB"])
            if invalid_count:
                errors.append(
                    f"PB column contains {invalid_count} non-numeric values.\n\n"
                    f"P/B must be numeric (per year).\n\n"
                    f"Leave the cell blank for a value Ecopath should estimate."
                )
        except Exception as e:
            errors.append(f"Could not validate PB column: {e}")

    return len(errors) == 0, errors


def _count_non_numeric(col: pd.Series) -> int:
    """Cells that hold something other than a number. Blank cells are fine:
    in Ecopath input a blank marks a value to be estimated."""
    blank = col.isna() | (col.astype(str).str.strip() == "")
    return int((pd.to_numeric(col, errors="coerce").isna() & ~blank).sum())


def _diet_row_groups(diet_df: pd.DataFrame) -> set:
    """Diet rows (prey) without Rpath's Import row, which is not a model group."""
    return {g for g in diet_df["Group"] if str(g).strip().lower() != "import"}


def _model_prey_groups(model_df: pd.DataFrame) -> Tuple[set, set]:
    """``(prey, fleets)``: every non-fleet group needs a diet row; fleets
    (Type 3) never eat, so a fleet row is neither required nor an error."""
    groups = model_df["Group"]
    if "Type" in model_df.columns:
        is_fleet = pd.to_numeric(model_df["Type"], errors="coerce") == 3
        return set(groups[~is_fleet]), set(groups[is_fleet])
    return set(groups), set()


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

    # Predator columns
    predator_cols = [c for c in df.columns if c != "Group"]

    # Check that all groups in diet match model groups (if provided)
    if model_df is not None and "Group" in model_df.columns:
        prey_groups, fleet_groups = _model_prey_groups(model_df)
        model_groups = prey_groups | fleet_groups
        diet_groups = _diet_row_groups(df)

        if not diet_groups.issubset(model_groups):
            extra_groups = diet_groups - model_groups
            errors.append(
                f"Diet CSV has groups not in model: {', '.join(extra_groups)}.\n\n"
                f"Diet matrix rows (prey) should match model groups.\n\n"
                f"Remove extra groups from diet or add them to model."
            )

        if not prey_groups.issubset(diet_groups):
            missing_groups = prey_groups - diet_groups
            errors.append(
                f"Diet CSV is missing model groups: {', '.join(missing_groups)}.\n\n"
                f"Diet matrix should have rows for all model groups."
            )

    # Check that predator columns are reasonable
    if not predator_cols:
        errors.append(
            "Diet CSV must have predator columns (columns other than 'Group')."
        )
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
        prey_groups, fleet_groups = _model_prey_groups(model_df)
        diet_groups = _diet_row_groups(diet_df)

        # Diet rows are the prey: every non-fleet group (plus Import, ignored)
        missing_in_diet = prey_groups - diet_groups
        extra_in_diet = diet_groups - prey_groups - fleet_groups
        if missing_in_diet or extra_in_diet:
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
        return "Quick fix: Biomass must be a number (e.g., 10.5); leave the cell blank for Ecopath to estimate it"
    elif "non-numeric" in msg_lower and "pb" in msg_lower:
        return "Quick fix: P/B must be per-year rate (e.g., 0.8 for fish, 50 for phytoplankton)"
    elif "non-numeric" in msg_lower and "diet" in msg_lower:
        return "Quick fix: Diet fractions must be numeric (0.0 to 1.0); use 0 where a prey is not eaten"
    elif "missing.*diet" in msg_lower:
        return "Quick fix: Diet CSV must have 'Group' column and predator columns"
    elif "groups not in" in msg_lower or "missing.*group" in msg_lower:
        return "Quick fix: Ensure diet and model CSV have matching group names"

    return ""


def validate_model_export_structure(df: pd.DataFrame) -> Tuple[bool, List[str]]:
    """Validate model.csv before export.

    Ensures the exported model CSV will be valid for re-import.

    Parameters
    ----------
    df : pd.DataFrame
        Model DataFrame to be exported

    Returns
    -------
    Tuple[bool, List[str]]
        (is_valid, error_messages)
    """
    errors = []

    if df is None or df.empty:
        return False, ["Cannot export empty model."]

    # Check required columns
    missing_cols = REQUIRED_MODEL_COLUMNS - set(df.columns)
    if missing_cols:
        missing_str = ", ".join(sorted(missing_cols))
        errors.append(f"Model missing required columns for export: {missing_str}")

    # Check for non-null values in required columns
    for col in REQUIRED_MODEL_COLUMNS & set(df.columns):
        null_count = df[col].isna().sum()
        if null_count > 0:
            errors.append(
                f"Column {col} has {null_count} missing values. All required columns must be complete."
            )

    # Check data types
    if "Type" in df.columns:
        non_int_types = df[~df["Type"].isna()][
            ~df["Type"].apply(
                lambda x: isinstance(x, (int, float, np.integer, np.floating))
            )
        ]
        if len(non_int_types) > 0:
            errors.append("Type column contains non-numeric values.")

    if "Biomass" in df.columns:
        non_numeric = df[~df["Biomass"].isna()][
            ~df["Biomass"].apply(
                lambda x: isinstance(x, (int, float, np.integer, np.floating))
            )
        ]
        if len(non_numeric) > 0:
            errors.append("Biomass column contains non-numeric values.")

    return len(errors) == 0, errors


def validate_diet_export_structure(df: pd.DataFrame) -> Tuple[bool, List[str]]:
    """Validate diet.csv before export.

    Ensures the exported diet CSV will be valid for re-import.

    Parameters
    ----------
    df : pd.DataFrame
        Diet DataFrame to be exported

    Returns
    -------
    Tuple[bool, List[str]]
        (is_valid, error_messages)
    """
    errors = []

    if df is None or df.empty:
        return False, ["Cannot export empty diet matrix."]

    # Check for Group column
    if "Group" not in df.columns:
        errors.append("Diet CSV must have a Group column identifying prey species.")
        return False, errors

    # Ensure at least one predator column (any column besides Group)
    predator_cols = [c for c in df.columns if c != "Group"]
    if not predator_cols:
        errors.append(
            "Diet CSV must have at least one predator column (besides Group)."
        )

    # Validate diet fractions are numeric where present
    for pred_col in predator_cols:
        if pred_col in df.columns:
            non_numeric = df[~df[pred_col].isna()][
                ~df[pred_col].apply(
                    lambda x: isinstance(x, (int, float, np.integer, np.floating))
                )
            ]
            if len(non_numeric) > 0:
                errors.append(f"Diet column {pred_col} contains non-numeric values.")

    return len(errors) == 0, errors
