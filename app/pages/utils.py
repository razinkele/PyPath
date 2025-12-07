"""Shared utilities for PyPath Shiny app pages.

This module contains common formatting functions used across multiple pages
to avoid code duplication.
"""

import pandas as pd
import numpy as np
from typing import Optional, Dict, List, Any


# =============================================================================
# CONSTANTS
# =============================================================================

# Constants for "no data" handling
NO_DATA_VALUE = 9999
NO_DATA_STYLE = {"background-color": "#f0f0f0", "color": "#999"}  # Light gray for no data cells
REMARK_STYLE = {"background-color": "#fff9e6", "border-bottom": "2px dashed #f0ad4e"}  # Yellow tint for cells with remarks
STANZA_STYLE = {"background-color": "#e6f3ff", "border-left": "3px solid #0066cc"}  # Light blue for stanza groups

# Type code to category name mapping
TYPE_LABELS: Dict[int, str] = {
    0: 'Consumer',
    1: 'Producer',
    2: 'Detritus',
    3: 'Fleet'
}

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
# DATAFRAME FORMATTING
# =============================================================================

def format_dataframe_for_display(
    df: pd.DataFrame, 
    decimal_places: int = 3,
    remarks_df: Optional[pd.DataFrame] = None,
    stanza_groups: Optional[list] = None
) -> tuple:
    """
    Format a DataFrame for display by:
    - Replacing 9999 (no data) values with NaN
    - Rounding numbers to specified decimal places
    - Adding remark indicators to cells with comments
    - Converting Type column from numeric codes to category names
    - Optionally marking groups that are part of multi-stanza configurations
    
    Args:
        df: DataFrame to format
        decimal_places: Number of decimal places for rounding
        remarks_df: Optional DataFrame with remarks (same structure as df)
        stanza_groups: Optional list of group names that are part of multi-stanza configurations
    
    Returns:
        tuple: (formatted_df, no_data_mask_df, remarks_mask_df, stanza_mask_df)
        - formatted_df: DataFrame with formatted values
        - no_data_mask_df: Boolean DataFrame where True indicates original 9999 value
        - remarks_mask_df: Boolean DataFrame where True indicates cell has a remark
        - stanza_mask_df: Boolean DataFrame where True indicates group is part of multi-stanza
    """
    formatted = df.copy()
    no_data_mask = pd.DataFrame(False, index=df.index, columns=df.columns)
    remarks_mask = pd.DataFrame(False, index=df.index, columns=df.columns)
    stanza_mask = pd.DataFrame(False, index=df.index, columns=df.columns)
    
    # Convert Type column to category labels
    if 'Type' in formatted.columns:
        formatted['Type'] = formatted['Type'].apply(
            lambda x: TYPE_LABELS.get(int(x), str(x)) if pd.notna(x) and x != '' else x
        )
    
    # Mark stanza groups (entire row)
    if stanza_groups and 'Group' in formatted.columns:
        for row_idx, group_name in enumerate(formatted['Group']):
            if group_name in stanza_groups:
                for col in formatted.columns:
                    stanza_mask.iloc[row_idx, list(formatted.columns).index(col)] = True
    
    for col in formatted.columns:
        if col == 'Type':
            # Type column already converted to labels, skip numeric processing
            continue
        if formatted[col].dtype in ['float64', 'float32', 'int64', 'int32'] or col not in ['Group', 'Type']:
            # Convert to numeric where possible
            numeric_col = pd.to_numeric(formatted[col], errors='coerce')
            
            # Mark 9999 values as no data
            is_no_data = (numeric_col == NO_DATA_VALUE) | (numeric_col == -9999)
            no_data_mask[col] = is_no_data
            
            # Replace 9999 with NaN, then round
            numeric_col = numeric_col.replace([NO_DATA_VALUE, -9999], np.nan)
            
            # Round non-NaN values
            if col not in ['Group', 'Type']:
                numeric_col = numeric_col.round(decimal_places)
            
            formatted[col] = numeric_col
    
    # Keep NaN values in numeric columns (DataGrid handles them properly)
    # Only fill NaN in string columns if needed
    for col in formatted.columns:
        if formatted[col].dtype == 'object':
            formatted[col] = formatted[col].fillna('')
    
    # Check for remarks
    if remarks_df is not None:
        for col in formatted.columns:
            if col in remarks_df.columns:
                for row_idx in range(len(formatted)):
                    if row_idx < len(remarks_df):
                        remark = remarks_df.iloc[row_idx].get(col, '')
                        if isinstance(remark, str) and remark.strip():
                            remarks_mask.iloc[row_idx, list(formatted.columns).index(col)] = True
    
    return formatted, no_data_mask, remarks_mask, stanza_mask


def create_cell_styles(
    df: pd.DataFrame, 
    no_data_mask: pd.DataFrame,
    remarks_mask: Optional[pd.DataFrame] = None,
    stanza_mask: Optional[pd.DataFrame] = None
) -> list:
    """
    Create cell style rules for DataGrid based on no-data mask, remarks mask, and stanza mask.
    
    Args:
        df: The formatted DataFrame
        no_data_mask: Boolean DataFrame where True indicates no data (9999 values)
        remarks_mask: Optional Boolean DataFrame where True indicates cell has a remark
        stanza_mask: Optional Boolean DataFrame where True indicates stanza group row
    
    Returns:
        list: Style dictionaries for DataGrid
    """
    styles = []
    for row_idx in range(len(df)):
        for col_idx, col in enumerate(df.columns):
            # Check for no-data cells (highest priority)
            if col in no_data_mask.columns and no_data_mask.iloc[row_idx][col]:
                styles.append({
                    "location": "body",
                    "rows": row_idx,
                    "cols": col_idx,
                    "style": NO_DATA_STYLE
                })
            # Check for cells with remarks (second priority)
            elif remarks_mask is not None and col in remarks_mask.columns and remarks_mask.iloc[row_idx][col]:
                styles.append({
                    "location": "body",
                    "rows": row_idx,
                    "cols": col_idx,
                    "style": REMARK_STYLE
                })
            # Check for stanza group rows (lowest priority for styling)
            elif stanza_mask is not None and col in stanza_mask.columns and stanza_mask.iloc[row_idx][col]:
                styles.append({
                    "location": "body",
                    "rows": row_idx,
                    "cols": col_idx,
                    "style": STANZA_STYLE
                })
    return styles


# =============================================================================
# MODEL INFO EXTRACTION
# =============================================================================

def get_model_info(model) -> Optional[Dict[str, Any]]:
    """Extract model info from either Rpath or RpathParams object.
    
    Args:
        model: Rpath (balanced model) or RpathParams object
    
    Returns:
        dict with: groups, num_living, num_dead, trophic_level, biomass, type_codes, etc.
        Returns None if model is None
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
