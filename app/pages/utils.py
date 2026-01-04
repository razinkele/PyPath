"""Shared utilities for PyPath Shiny app pages.

This module contains common formatting functions used across multiple pages
to avoid code duplication.
"""

import pandas as pd
import numpy as np
from typing import Optional, Dict, List, Any, Tuple

# Import centralized configuration
try:
    from app.config import DISPLAY, TYPE_LABELS, NO_DATA_VALUE, THRESHOLDS
except ModuleNotFoundError:
    from config import DISPLAY, TYPE_LABELS, NO_DATA_VALUE, THRESHOLDS


# =============================================================================
# CONSTANTS
# =============================================================================

# Style constants (UI-specific, not in config)
NO_DATA_STYLE = {"background-color": "#f0f0f0", "color": "#999"}  # Light gray for no data cells
REMARK_STYLE = {"background-color": "#fff9e6", "border-bottom": "2px dashed #f0ad4e"}  # Yellow tint for cells with remarks
STANZA_STYLE = {"background-color": "#e6f3ff", "border-left": "3px solid #0066cc"}  # Light blue for stanza groups

# Column tooltips for parameter documentation
COLUMN_TOOLTIPS: Dict[str, str] = {
    # Basic Model Parameters
    'Group': 'Name of the functional group (species or group of species)',
    'Type': 'Group type: 0=Consumer, 1=Producer, 2=Detritus, 3=Fleet',
    'Biomass': 'Biomass (t/km²) - standing stock of the group',
    'PB': 'Production/Biomass ratio (1/year) - turnover rate',
    'QB': 'Consumption/Biomass ratio (1/year) - feeding rate',
    'EE': 'Ecotrophic Efficiency (0-1) - fraction of production used in the system',
    'ProdCons': 'Production/Consumption ratio (P/Q or GE) - gross food conversion efficiency',
    'Unassim': 'Unassimilated consumption (0-1) - fraction of food not assimilated',
    'BioAcc': 'Biomass accumulation rate (t/km²/year) - change in biomass over time',
    'DetInput': 'Detrital input from outside the system (t/km²/year)',
    
    # Balanced Model Results
    'TL': 'Trophic Level - position in the food web (1=primary producer/detritus, 2+=consumers)',
    'GE': 'Gross Efficiency (P/Q) - production divided by consumption',
    'Removals': 'Total removals by fishing (t/km²/year) - landings plus discards',
    
    # Diet Matrix
    'Import': 'Fraction of diet imported from outside the model area',
    
    # Stanza Parameters - stgroups
    'StGroupNum': 'Unique identifier for the multi-stanza group',
    'StanzaName': 'Name of the multi-stanza group (e.g., species name)',
    'nstanzas': 'Number of life stages in this multi-stanza group',
    'VBGF_Ksp': 'von Bertalanffy growth coefficient K (1/year)',
    'VBGF_d': 'Exponent relating consumption to body weight (typically ~0.67)',
    'Wmat': 'Weight at maturity (fraction of Winf)',
    'RecPower': 'Recruitment power parameter for stock-recruitment relationship',
    'Wmat001': 'Age at which 0.1% maturity is reached',
    'Wmat50': 'Age at which 50% maturity is reached',
    'Amax': 'Maximum age (months)',
    'First_age': 'Age of first stanza (months)',
    
    # Stanza Parameters - stindiv
    'StanzaNum': 'Stanza number within the multi-stanza group',
    'GroupNum': 'Reference to the Ecopath group number',
    'First': 'First month of this life stage',
    'Last': 'Last month of this life stage', 
    'Z': 'Total mortality rate (1/year)',
    'Leading': 'Whether this stanza leads the group (1=yes, 0=no)',
}


# =============================================================================
# MODEL TYPE HELPERS
# =============================================================================

def is_balanced_model(model) -> bool:
    """Check if model is a balanced Rpath model.

    Parameters
    ----------
    model : object
        Model to check

    Returns
    -------
    bool
        True if model is balanced (has NUM_LIVING attribute)

    Examples
    --------
    >>> from pypath.core.ecopath import rpath
    >>> from pypath.core.params import create_rpath_params
    >>> params = create_rpath_params(...)
    >>> balanced = rpath(params)
    >>> is_balanced_model(balanced)
    True
    >>> is_balanced_model(params)
    False
    """
    return hasattr(model, 'NUM_LIVING')


def is_rpath_params(model) -> bool:
    """Check if model is RpathParams (unbalanced).

    Parameters
    ----------
    model : object
        Model to check

    Returns
    -------
    bool
        True if model is RpathParams

    Examples
    --------
    >>> from pypath.core.params import create_rpath_params
    >>> params = create_rpath_params(...)
    >>> is_rpath_params(params)
    True
    """
    return (hasattr(model, 'model') and
            hasattr(model.model, 'columns') and
            'Group' in model.model.columns)


def get_model_type(model) -> str:
    """Get model type as string.

    Parameters
    ----------
    model : object
        Model to identify

    Returns
    -------
    str
        'balanced', 'params', or 'unknown'

    Examples
    --------
    >>> from pypath.core.ecopath import rpath
    >>> from pypath.core.params import create_rpath_params
    >>> params = create_rpath_params(...)
    >>> get_model_type(params)
    'params'
    >>> balanced = rpath(params)
    >>> get_model_type(balanced)
    'balanced'
    """
    if is_balanced_model(model):
        return 'balanced'
    elif is_rpath_params(model):
        return 'params'
    else:
        return 'unknown'


# =============================================================================
# DATAFRAME FORMATTING
# =============================================================================

def format_dataframe_for_display(
    df: pd.DataFrame,
    decimal_places: Optional[int] = None,
    remarks_df: Optional[pd.DataFrame] = None,
    stanza_groups: Optional[List[str]] = None
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Format a DataFrame for display with number formatting and cell styling.

    This function processes a DataFrame to prepare it for display in the Shiny app by:
    - Replacing 9999 (no data) sentinel values with NaN
    - Rounding numeric values to specified decimal places
    - Converting Type column from numeric codes to category labels
    - Creating boolean masks for special cell highlighting (no data, remarks, stanza groups)

    OPTIMIZED VERSION: Uses vectorized operations and single-pass processing for better
    performance with large DataFrames (100+ rows).

    Parameters
    ----------
    df : pd.DataFrame
        The DataFrame to format for display
    decimal_places : Optional[int], default None
        Number of decimal places for rounding numeric values.
        If None, uses DISPLAY.decimal_places from config (default: 3)
    remarks_df : Optional[pd.DataFrame], default None
        DataFrame with same structure as df containing remark text.
        Cells with non-empty remarks will be marked in the remarks mask
    stanza_groups : Optional[List[str]], default None
        List of group names that are part of multi-stanza configurations.
        These groups will be highlighted in the output

    Returns
    -------
    Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]
        A 4-tuple containing:
        - formatted_df : DataFrame with formatted values (9999→NaN, rounded decimals)
        - no_data_mask_df : Boolean DataFrame, True where original value was 9999
        - remarks_mask_df : Boolean DataFrame, True where cell has a remark
        - stanza_mask_df : Boolean DataFrame, True for stanza group rows

    Examples
    --------
    >>> import pandas as pd
    >>> df = pd.DataFrame({
    ...     'Group': ['Fish', 'Plankton'],
    ...     'Type': [0, 1],
    ...     'Biomass': [10.12345, 9999]
    ... })
    >>> formatted, no_data, remarks, stanza = format_dataframe_for_display(df, decimal_places=2)
    >>> formatted['Biomass'].tolist()
    [10.12, nan]
    >>> no_data['Biomass'].tolist()
    [False, True]
    """
    # Use config default if not specified
    if decimal_places is None:
        decimal_places = DISPLAY.decimal_places

    # Create output DataFrames
    formatted = df.copy()
    no_data_mask = pd.DataFrame(False, index=df.index, columns=df.columns)
    remarks_mask = pd.DataFrame(False, index=df.index, columns=df.columns)
    stanza_mask = pd.DataFrame(False, index=df.index, columns=df.columns)

    # OPTIMIZATION 1: Vectorized Type column conversion
    if 'Type' in formatted.columns:
        # Use vectorized map instead of apply for better performance
        type_col = pd.to_numeric(formatted['Type'], errors='coerce')
        formatted['Type'] = type_col.map(TYPE_LABELS).fillna(formatted['Type'])

    # OPTIMIZATION 2: Vectorized stanza group marking
    if stanza_groups and 'Group' in formatted.columns:
        # Create boolean mask for stanza rows in one operation
        is_stanza_row = formatted['Group'].isin(stanza_groups)
        # Broadcast mask across all columns
        stanza_mask.loc[:, :] = is_stanza_row.values[:, np.newaxis]

    # OPTIMIZATION 3: Single-pass numeric column processing
    # Identify special columns that don't need numeric processing
    skip_cols = {'Group', 'Type'}

    # Process all columns in a single pass
    for col in formatted.columns:
        if col in skip_cols:
            continue

        # Convert to numeric (works for both numeric and object dtypes)
        numeric_col = pd.to_numeric(formatted[col], errors='coerce')

        # VECTORIZED: Mark no-data values
        is_no_data = (numeric_col == NO_DATA_VALUE) | (numeric_col == THRESHOLDS.negative_no_data_value)
        no_data_mask[col] = is_no_data

        # VECTORIZED: Replace sentinel values with NaN and round
        numeric_col = numeric_col.replace([NO_DATA_VALUE, THRESHOLDS.negative_no_data_value], np.nan)
        numeric_col = numeric_col.round(decimal_places)

        formatted[col] = numeric_col

    # OPTIMIZATION 4: Vectorized NaN filling for object columns
    # Only fill NaN in object/string columns
    object_cols = formatted.select_dtypes(include=['object']).columns
    formatted[object_cols] = formatted[object_cols].fillna('')

    # OPTIMIZATION 5: Vectorized remarks mask creation
    if remarks_df is not None:
        # Find common columns between data and remarks
        common_cols = formatted.columns.intersection(remarks_df.columns)

        for col in common_cols:
            # VECTORIZED: Check for non-empty remarks
            # Use pandas vectorized string operations
            if len(remarks_df) > 0:
                has_remark = remarks_df[col].astype(str).str.strip().ne('')
                # Only set mask for rows that exist in both DataFrames
                max_rows = min(len(formatted), len(has_remark))
                remarks_mask.loc[:max_rows-1, col] = has_remark.iloc[:max_rows].values

    return formatted, no_data_mask, remarks_mask, stanza_mask


def create_cell_styles(
    df: pd.DataFrame,
    no_data_mask: pd.DataFrame,
    remarks_mask: Optional[pd.DataFrame] = None,
    stanza_mask: Optional[pd.DataFrame] = None
) -> List[Dict[str, Any]]:
    """Create cell style rules for Shiny DataGrid component.

    Generates style dictionaries for highlighting special cells in the DataGrid:
    - No data cells (9999 values) → gray background
    - Non-applicable parameters by group type → italicized gray
    - Cells with remarks → yellow tint with dashed border
    - Stanza group rows → light blue background with left border

    OPTIMIZED VERSION: Uses numpy boolean indexing and pre-computed lookups
    for significantly faster performance with large DataFrames.

    Parameters
    ----------
    df : pd.DataFrame
        The formatted DataFrame to generate styles for
    no_data_mask : pd.DataFrame
        Boolean DataFrame where True indicates cell originally contained 9999 (no data)
    remarks_mask : Optional[pd.DataFrame], default None
        Boolean DataFrame where True indicates cell has an associated remark
    stanza_mask : Optional[pd.DataFrame], default None
        Boolean DataFrame where True indicates row is part of a multi-stanza group

    Returns
    -------
    List[Dict[str, Any]]
        List of style dictionaries for DataGrid. Each dict has:
        - 'location': str - Always 'body'
        - 'rows': int - Row index to style
        - 'cols': int - Column index to style
        - 'style': Dict[str, str] - CSS style properties

    Notes
    -----
    Style priority (highest to lowest):
    1. No data cells (gray)
    2. Non-applicable parameters (gray italic)
    3. Cells with remarks (yellow)
    4. Stanza group rows (blue)

    Non-applicable parameters by group type:
    - QB (Consumption): Not applicable to producers (type=1) and detritus (type=2)
    - Unassim: Not applicable to producers (type=1) and detritus (type=2)

    Examples
    --------
    >>> import pandas as pd
    >>> df = pd.DataFrame({'Group': ['Fish'], 'Type': ['Consumer'], 'Biomass': [10.5]})
    >>> no_data = pd.DataFrame({'Group': [False], 'Type': [False], 'Biomass': [False]})
    >>> styles = create_cell_styles(df, no_data)
    >>> len(styles)
    0  # No special styling needed
    """
    styles = []

    # Define parameters that don't apply to certain group types
    NON_APPLICABLE_PARAMS = {
        'QB': [1, 2],      # QB doesn't apply to producers (1) and detritus (2)
        'Unassim': [1, 2], # Unassim doesn't apply to producers (1) and detritus (2)
    }

    # Grey style for non-applicable parameters
    GREY_STYLE = {"background-color": "#f8f9fa", "color": "#6c757d", "font-style": "italic"}

    # OPTIMIZATION 1: Pre-compute type mapping and group types array
    type_map = {v: k for k, v in TYPE_LABELS.items()}
    group_types = None
    if 'Type' in df.columns:
        # Vectorized conversion of all type labels to codes
        group_types = df['Type'].map(type_map).values

    # OPTIMIZATION 2: Convert masks to numpy arrays for faster indexing
    no_data_array = no_data_mask.values
    remarks_array = remarks_mask.values if remarks_mask is not None else None
    stanza_array = stanza_mask.values if stanza_mask is not None else None

    # OPTIMIZATION 3: Pre-compute column indices for non-applicable params
    non_app_col_indices = {}
    for param, _ in NON_APPLICABLE_PARAMS.items():
        if param in df.columns:
            non_app_col_indices[param] = df.columns.get_loc(param)

    # OPTIMIZATION 4: Create column index lookup
    col_list = df.columns.tolist()

    # OPTIMIZATION 5: Batch process cells by mask type using numpy where
    n_rows, n_cols = df.shape

    # Process no_data cells (highest priority)
    if no_data_array is not None:
        no_data_coords = np.argwhere(no_data_array)
        for row_idx, col_idx in no_data_coords:
            styles.append({
                "location": "body",
                "rows": int(row_idx),
                "cols": int(col_idx),
                "style": NO_DATA_STYLE
            })

    # Process non-applicable params
    if group_types is not None:
        for param, types in NON_APPLICABLE_PARAMS.items():
            if param in non_app_col_indices:
                col_idx = non_app_col_indices[param]
                # Find rows where this parameter doesn't apply
                for group_type in types:
                    row_indices = np.where(group_types == group_type)[0]
                    for row_idx in row_indices:
                        # Skip if already styled as no_data
                        if not (no_data_array is not None and no_data_array[row_idx, col_idx]):
                            styles.append({
                                "location": "body",
                                "rows": int(row_idx),
                                "cols": int(col_idx),
                                "style": GREY_STYLE
                            })

    # Process remark cells
    if remarks_array is not None:
        # Get coordinates of cells with remarks
        remark_coords = np.argwhere(remarks_array)
        for row_idx, col_idx in remark_coords:
            # Skip if already styled as no_data or non-applicable
            if not (no_data_array is not None and no_data_array[row_idx, col_idx]):
                col_name = col_list[col_idx]
                group_type = group_types[row_idx] if group_types is not None else None
                is_non_app = (col_name in NON_APPLICABLE_PARAMS and
                             group_type is not None and
                             group_type in NON_APPLICABLE_PARAMS[col_name])

                if not is_non_app:
                    styles.append({
                        "location": "body",
                        "rows": int(row_idx),
                        "cols": int(col_idx),
                        "style": REMARK_STYLE
                    })

    # Process stanza group cells (lowest priority)
    if stanza_array is not None:
        stanza_coords = np.argwhere(stanza_array)
        for row_idx, col_idx in stanza_coords:
            # Skip if already styled
            has_higher_priority = (
                (no_data_array is not None and no_data_array[row_idx, col_idx]) or
                (remarks_array is not None and remarks_array[row_idx, col_idx])
            )

            if not has_higher_priority:
                col_name = col_list[col_idx]
                group_type = group_types[row_idx] if group_types is not None else None
                is_non_app = (col_name in NON_APPLICABLE_PARAMS and
                             group_type is not None and
                             group_type in NON_APPLICABLE_PARAMS[col_name])

                if not is_non_app:
                    styles.append({
                        "location": "body",
                        "rows": int(row_idx),
                        "cols": int(col_idx),
                        "style": STANZA_STYLE
                    })

    return styles


# =============================================================================
# MODEL INFO EXTRACTION
# =============================================================================

def get_model_info(model: Any) -> Optional[Dict[str, Any]]:
    """Extract comprehensive model information from Rpath or RpathParams object.

    This utility function provides a unified interface for accessing model properties
    regardless of whether the model is a balanced Rpath object or unbalanced RpathParams.
    It handles the different attribute structures of these two object types.

    Parameters
    ----------
    model : Any
        Either an Rpath object (balanced model) or RpathParams object (unbalanced model).
        Can also be None, in which case None is returned.

    Returns
    -------
    Optional[Dict[str, Any]]
        Dictionary containing model information with the following keys:

        - 'groups' : List[str]
            Names of all functional groups in the model
        - 'num_living' : int
            Number of living groups (consumers and producers)
        - 'num_dead' : int
            Number of detritus groups
        - 'num_groups' : int
            Total number of groups
        - 'trophic_level' : Optional[np.ndarray]
            Trophic levels (only for balanced models, None otherwise)
        - 'biomass' : Optional[np.ndarray]
            Biomass values for each group
        - 'type_codes' : np.ndarray
            Group type codes (0=consumer, 1=producer, 2=detritus, 3=fleet)
        - 'eco_name' : str
            Model name/identifier
        - 'is_balanced' : bool
            True if Rpath object (balanced), False if RpathParams (unbalanced)
        - 'params' : Any
            Reference to the parameter object

        Returns None if model is None.

    Notes
    -----
    **Rpath vs RpathParams:**
    - **Rpath** (balanced): Has attributes like Group, NUM_LIVING, TL directly
    - **RpathParams** (unbalanced): Properties are in params.model DataFrame

    Type Codes:
    - 0: Consumer (fish, invertebrates)
    - 1: Producer (phytoplankton, macroalgae)
    - 2: Detritus (organic matter)
    - 3: Fleet (fishing gear)

    Examples
    --------
    >>> from pypath.core.params import create_params
    >>> params = create_params(n_groups=3, n_living=2)
    >>> info = get_model_info(params)
    >>> info['is_balanced']
    False
    >>> info['num_groups']
    3

    >>> # After balancing
    >>> model = rpath(params)
    >>> info = get_model_info(model)
    >>> info['is_balanced']
    True
    >>> 'trophic_level' in info and info['trophic_level'] is not None
    True
    """
    if model is None:
        return None
    
    # Check if it's an Rpath (balanced model) or RpathParams
    if hasattr(model, 'NUM_LIVING'):
        # It's an Rpath object
        return {
            'groups': list(model.Group),
            'num_living': int(model.NUM_LIVING),
            'num_dead': int(model.NUM_DEAD),
            'num_groups': int(model.NUM_GROUPS),
            'trophic_level': model.TL if hasattr(model, 'TL') else None,
            'biomass': model.Biomass if hasattr(model, 'Biomass') else None,
            'type_codes': model.Type if hasattr(model, 'Type') else None,
            'eco_name': model.eco_name if hasattr(model, 'eco_name') else 'Model',
            'is_balanced': True,
            'params': model.params if hasattr(model, 'params') else None,
        }
    elif hasattr(model, 'model') and hasattr(model.model, 'columns'):
        # It's an RpathParams object
        groups = list(model.model['Group'].values)
        types = model.model['Type'].values
        num_living = int(np.sum(types == 0))  # Type 0 = consumer
        num_dead = int(np.sum(types == 2))    # Type 2 = detritus
        num_groups = len(groups)
        
        return {
            'groups': groups,
            'num_living': num_living,
            'num_dead': num_dead,
            'num_groups': num_groups,
            'trophic_level': None,  # Not calculated until balanced
            'biomass': model.model['Biomass'].values if 'Biomass' in model.model.columns else None,
            'type_codes': types,
            'eco_name': 'Unbalanced Model',
            'is_balanced': False,
            'params': model,
        }
    
    return None
