"""
EcoBase database connector for PyPath.

This module provides functions to connect to the EcoBase database
(http://ecobase.ecopath.org/) and download Ecopath model data.

EcoBase is a global repository of Ecopath models maintained by
AGROCAMPUS OUEST (France).

Functions:
- list_ecobase_models(): Get list of all available public models
- get_ecobase_model(model_id): Download a specific model's data
- ecobase_to_rpath(model_data): Convert EcoBase data to RpathParams

Example:
    >>> from pypath.io.ecobase import list_ecobase_models, get_ecobase_model
    >>> models = list_ecobase_models()
    >>> print(f"Found {len(models)} models")
    >>> model_data = get_ecobase_model(403)  # Get specific model
    >>> rpath_params = ecobase_to_rpath(model_data)
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass, field
from typing import Optional, Dict, List, Any, Union
import xml.etree.ElementTree as ET

import numpy as np
import pandas as pd

# Try to import requests, fall back to urllib if not available
try:
    import requests
    HAS_REQUESTS = True
except ImportError:
    HAS_REQUESTS = False
    import urllib.request
    import urllib.error

from pypath.core.params import RpathParams, create_rpath_params
from pypath.io.utils import safe_float, fetch_url


# EcoBase API endpoints
ECOBASE_LIST_URL = "http://sirs.agrocampus-ouest.fr/EcoBase/php/webser/soap-client_3.php"
ECOBASE_MODEL_URL = "http://sirs.agrocampus-ouest.fr/EcoBase/php/webser/soap-client.php?no_model="


# Note: safe_float and fetch_url are now imported from pypath.io.utils
# to avoid code duplication


@dataclass
class EcoBaseModel:
    """Container for EcoBase model metadata.
    
    Attributes
    ----------
    model_number : int
        Unique model identifier in EcoBase
    model_name : str
        Name of the model
    country : str
        Country/region of the ecosystem
    ecosystem_type : str
        Type of ecosystem (marine, freshwater, etc.)
    num_groups : int
        Number of functional groups
    author : str
        Model author(s)
    year : int
        Year of model creation
    reference : str
        Publication reference
    description : str
        Model description
    dissemination_allow : bool
        Whether public access is allowed
    """
    model_number: int
    model_name: str = ""
    country: str = ""
    ecosystem_type: str = ""
    num_groups: int = 0
    author: str = ""
    year: int = 0
    reference: str = ""
    description: str = ""
    dissemination_allow: bool = True


@dataclass 
class EcoBaseGroupData:
    """Data for a single functional group from EcoBase.
    
    Attributes
    ----------
    group_seq : int
        Group sequence number (1-based)
    group_name : str
        Name of the group
    trophic_level : float
        Calculated trophic level
    biomass : float
        Biomass (t/km²)
    biomass_hab : float
        Biomass in habitat area
    prod_biom : float
        Production/Biomass ratio (/year)
    cons_biom : float
        Consumption/Biomass ratio (/year)
    ecotrophic_eff : float
        Ecotrophic efficiency
    prod_cons : float
        Production/Consumption ratio
    unassim_cons : float
        Unassimilated consumption fraction
    habitat_area : float
        Habitat area fraction
    """
    group_seq: int
    group_name: str = ""
    trophic_level: float = 0.0
    biomass: float = 0.0
    biomass_hab: float = 0.0
    prod_biom: float = 0.0
    cons_biom: float = 0.0
    ecotrophic_eff: float = 0.0
    prod_cons: float = 0.0
    unassim_cons: float = 0.2
    habitat_area: float = 1.0
    group_type: int = 0  # 0=consumer, 1=producer, 2=detritus, 3=fleet


def list_ecobase_models(
    filter_public: bool = True,
    timeout: int = 60
) -> pd.DataFrame:
    """Get list of available Ecopath models from EcoBase.
    
    Connects to the EcoBase SOAP API and retrieves metadata for
    all available models.
    
    Parameters
    ----------
    filter_public : bool
        If True, only return models with public access allowed
    timeout : int
        Request timeout in seconds
    
    Returns
    -------
    pd.DataFrame
        DataFrame with model metadata including:
        - model_number: Unique ID
        - model_name: Name
        - country: Location
        - ecosystem_type: Type
        - num_groups: Number of groups
        - author: Author(s)
        - year: Year
        - reference: Publication
    
    Example
    -------
    >>> models = list_ecobase_models()
    >>> print(f"Found {len(models)} public models")
    >>> # Filter by ecosystem type
    >>> marine = models[models['ecosystem_type'].str.contains('marine', case=False)]
    """
    try:
        xml_content = fetch_url(ECOBASE_LIST_URL, timeout=timeout, parse_json=False)
    except Exception as e:
        raise ConnectionError(f"Failed to connect to EcoBase: {e}")
    
    # Parse XML response
    try:
        root = ET.fromstring(xml_content)
    except ET.ParseError as e:
        raise ValueError(f"Failed to parse EcoBase response: {e}")
    
    # Extract model data
    models = []
    
    # Navigate through SOAP envelope to find model data
    # The structure varies, so we try multiple paths
    for model_elem in root.iter('model'):
        model_data = {}
        for child in model_elem:
            tag = child.tag.replace('{http://schemas.xmlsoap.org/soap/envelope/}', '')
            model_data[tag] = child.text
        
        if model_data:
            try:
                model = {
                    'model_number': int(model_data.get('model_number', model_data.get('no_model', 0))),
                    'model_name': model_data.get('model_name', model_data.get('name', '')),
                    'country': model_data.get('country', model_data.get('location', '')),
                    'ecosystem_type': model_data.get('ecosystem_type', model_data.get('type', '')),
                    'num_groups': int(model_data.get('number_group', model_data.get('nb_group', 0)) or 0),
                    'author': model_data.get('author', ''),
                    'year': int(model_data.get('year', 0) or 0),
                    'reference': model_data.get('reference', ''),
                    'dissemination_allow': model_data.get('dissemination_allow', 'true').lower() == 'true',
                }
                models.append(model)
            except (ValueError, TypeError):
                continue
    
    # Also try alternative XML structure
    if not models:
        for item in root.iter():
            if 'model' in item.tag.lower() or item.tag == 'item':
                model_data = {child.tag: child.text for child in item}
                if model_data and any(k in model_data for k in ['model_number', 'no_model', 'model_name']):
                    try:
                        model = {
                            'model_number': int(model_data.get('model_number', model_data.get('no_model', 0)) or 0),
                            'model_name': str(model_data.get('model_name', model_data.get('name', ''))),
                            'country': str(model_data.get('country', model_data.get('location', ''))),
                            'ecosystem_type': str(model_data.get('ecosystem_type', model_data.get('type', ''))),
                            'num_groups': int(model_data.get('number_group', model_data.get('nb_group', 0)) or 0),
                            'author': str(model_data.get('author', '')),
                            'year': int(model_data.get('year', 0) or 0),
                            'reference': str(model_data.get('reference', '')),
                            'dissemination_allow': str(model_data.get('dissemination_allow', 'true')).lower() == 'true',
                        }
                        if model['model_number'] > 0:
                            models.append(model)
                    except (ValueError, TypeError):
                        continue
    
    df = pd.DataFrame(models)
    
    if filter_public and 'dissemination_allow' in df.columns:
        df = df[df['dissemination_allow'] == True].copy()
    
    return df


def get_ecobase_model(
    model_id: int,
    timeout: int = 60
) -> Dict[str, Any]:
    """Download a specific model from EcoBase.
    
    Parameters
    ----------
    model_id : int
        Model number (from list_ecobase_models())
    timeout : int
        Request timeout in seconds
    
    Returns
    -------
    dict
        Dictionary containing:
        - 'metadata': Model metadata
        - 'groups': List of group data dictionaries
        - 'diet': Diet matrix as nested dict
        - 'raw_xml': Raw XML string for debugging
    
    Example
    -------
    >>> model_data = get_ecobase_model(403)
    >>> print(f"Model has {len(model_data['groups'])} groups")
    """
    url = f"{ECOBASE_MODEL_URL}{model_id}"
    
    try:
        xml_content = fetch_url(url, timeout=timeout, parse_json=False)
    except Exception as e:
        raise ConnectionError(f"Failed to download model {model_id}: {e}")
    
    # Parse XML
    try:
        root = ET.fromstring(xml_content)
    except ET.ParseError as e:
        raise ValueError(f"Failed to parse model data: {e}")
    
    result = {
        'model_id': model_id,
        'metadata': {},
        'groups': [],
        'diet': {},
        'fleets': [],
        'catches': {},
        'raw_xml': xml_content,
    }
    
    # First pass: Build group_seq to group_name mapping
    group_seq_to_name = {}
    for group_elem in root.iter('group'):
        group_name = None
        group_seq = None
        for child in group_elem:
            if child.tag == 'group_name':
                group_name = child.text
            elif child.tag == 'group_seq':
                try:
                    group_seq = int(child.text) if child.text else None
                except ValueError:
                    group_seq = None
        if group_name and group_seq is not None:
            group_seq_to_name[group_seq] = group_name
    
    # Extract groups and diet data
    for group_elem in root.iter('group'):
        group_data = {}
        pred_name = None
        
        for child in group_elem:
            tag = child.tag
            text = child.text
            
            # Store group name for diet processing
            if tag == 'group_name':
                pred_name = text
            
            # Handle diet_descr specially - extract nested diet elements
            if tag == 'diet_descr':
                # Process nested diet elements
                for diet_elem in child.iter('diet'):
                    prey_seq = None
                    proportion = 0.0
                    
                    for diet_child in diet_elem:
                        if diet_child.tag == 'prey_seq':
                            try:
                                prey_seq = int(diet_child.text) if diet_child.text else None
                            except ValueError:
                                prey_seq = None
                        elif diet_child.tag == 'proportion':
                            try:
                                proportion = float(diet_child.text) if diet_child.text else 0.0
                            except ValueError:
                                proportion = 0.0
                    
                    # Map prey_seq to prey_name and store diet
                    if prey_seq is not None and proportion > 0 and pred_name:
                        prey_name = group_seq_to_name.get(prey_seq, f"Group_{prey_seq}")
                        if pred_name not in result['diet']:
                            result['diet'][pred_name] = {}
                        result['diet'][pred_name][prey_name] = proportion
                continue
            
            # Try to convert values appropriately
            if text:
                text_lower = text.lower().strip()
                # Handle boolean strings first
                if text_lower in ('true', 'false', 'yes', 'no'):
                    group_data[tag] = text_lower in ('true', 'yes')
                else:
                    # Try numeric conversion
                    try:
                        if '.' in text or ('e' in text_lower and text_lower not in ('true', 'false')):
                            group_data[tag] = float(text)
                        else:
                            group_data[tag] = int(text)
                    except ValueError:
                        group_data[tag] = text
            else:
                group_data[tag] = None
        
        if group_data:
            result['groups'].append(group_data)
    
    # Build group_id to group_name mapping for diet matrix
    group_id_to_name = {}
    for g in result['groups']:
        gid = g.get('group_seq', g.get('group_id', g.get('sequence', g.get('no', None))))
        gname = g.get('group_name', g.get('name', None))
        if gid is not None and gname is not None:
            group_id_to_name[int(gid)] = gname
    
    # Extract diet from dc (diet composition) fields in groups
    # Format: dc fields contain "prey_id proportion" pairs
    for g in result['groups']:
        pred_name = g.get('group_name', g.get('name', None))
        if not pred_name:
            continue
        
        # Look for dc fields (dc1, dc2, ... or dc_1, dc_2, ...)
        for key, value in g.items():
            if key.lower().startswith('dc') and value is not None:
                # Try to parse as "prey_id proportion" or just get prey_id
                try:
                    if isinstance(value, str) and ' ' in value:
                        parts = value.strip().split()
                        if len(parts) >= 2:
                            prey_id = int(parts[0])
                            proportion = float(parts[1])
                        else:
                            prey_id = int(parts[0])
                            proportion = 1.0
                    elif isinstance(value, (int, float)):
                        # Could be just a proportion or an ID
                        continue
                    else:
                        continue
                    
                    # Map prey_id to name
                    prey_name = group_id_to_name.get(prey_id, f"Group_{prey_id}")
                    
                    if proportion > 0:
                        if pred_name not in result['diet']:
                            result['diet'][pred_name] = {}
                        result['diet'][pred_name][prey_name] = proportion
                except (ValueError, TypeError):
                    continue
    
    # Also try DietComp fields (another common format)
    for g in result['groups']:
        pred_name = g.get('group_name', g.get('name', None))
        if not pred_name:
            continue
        
        # Look for DietComp, dietcomp fields
        for key, value in g.items():
            key_lower = key.lower()
            if ('dietcomp' in key_lower or 'diet_comp' in key_lower) and value is not None:
                try:
                    if isinstance(value, str) and ' ' in value:
                        parts = value.strip().split()
                        if len(parts) >= 2:
                            prey_id = int(parts[0])
                            proportion = float(parts[1])
                            prey_name = group_id_to_name.get(prey_id, f"Group_{prey_id}")
                            
                            if proportion > 0:
                                if pred_name not in result['diet']:
                                    result['diet'][pred_name] = {}
                                result['diet'][pred_name][prey_name] = proportion
                except (ValueError, TypeError):
                    continue
    
    # Extract diet matrix from dedicated diet elements (alternative format)
    for diet_elem in root.iter('diet'):
        prey_name = None
        pred_name = None
        value = 0.0
        
        for child in diet_elem:
            if child.tag in ['prey', 'prey_name', 'from']:
                prey_name = child.text
            elif child.tag in ['predator', 'pred_name', 'to']:
                pred_name = child.text
            elif child.tag in ['diet', 'value', 'proportion']:
                try:
                    value = float(child.text) if child.text else 0.0
                except ValueError:
                    value = 0.0
        
        if prey_name and pred_name and value > 0:
            if pred_name not in result['diet']:
                result['diet'][pred_name] = {}
            result['diet'][pred_name][prey_name] = value
    
    # Alternative diet structure (nested in groups)
    for group_elem in root.iter('group'):
        group_name = None
        for child in group_elem:
            if child.tag in ['group_name', 'name']:
                group_name = child.text
                break
        
        if group_name:
            for diet_elem in group_elem.iter('diet_item'):
                prey_name = None
                value = 0.0
                for child in diet_elem:
                    if child.tag in ['prey', 'prey_name']:
                        prey_name = child.text
                    elif child.tag in ['proportion', 'value', 'diet']:
                        try:
                            value = float(child.text) if child.text else 0.0
                        except ValueError:
                            value = 0.0
                
                if prey_name and value > 0:
                    if group_name not in result['diet']:
                        result['diet'][group_name] = {}
                    result['diet'][group_name][prey_name] = value
    
    # Extract fleet/fishery data with catches from catch_descr
    for fleet_elem in root.iter('fleet'):
        fleet_data = {}
        fleet_name = None
        
        for child in fleet_elem:
            if child.tag == 'fleet_name':
                fleet_name = child.text
            elif child.tag == 'catch_descr':
                # Parse catch entries within fleet
                for catch_elem in child.findall('catch'):
                    group_seq = None
                    catch_value = 0.0
                    catch_type = None
                    
                    for catch_child in catch_elem:
                        if catch_child.tag == 'group_seq':
                            try:
                                group_seq = int(catch_child.text) if catch_child.text else None
                            except ValueError:
                                group_seq = None
                        elif catch_child.tag == 'catch_value':
                            try:
                                catch_value = float(catch_child.text) if catch_child.text else 0.0
                            except ValueError:
                                catch_value = 0.0
                        elif catch_child.tag == 'catch_type':
                            catch_type = catch_child.text.strip() if catch_child.text else None
                    
                    # Store catches by fleet and group
                    if fleet_name and group_seq is not None and catch_type:
                        group_name = group_seq_to_name.get(group_seq, f"Group_{group_seq}")
                        catch_key = (fleet_name, group_name, catch_type)
                        
                        if group_name not in result['catches']:
                            result['catches'][group_name] = {}
                        if fleet_name not in result['catches'][group_name]:
                            result['catches'][group_name][fleet_name] = {
                                'landings': 0.0, 
                                'discards': 0.0,
                                'discard_mort': 0.0,
                                'market': 0.0,
                                'prop_mort': 0.0
                            }
                        
                        # Map catch types to our structure
                        if catch_type == 'total landings':
                            result['catches'][group_name][fleet_name]['landings'] = catch_value
                        elif catch_type == 'discards':
                            result['catches'][group_name][fleet_name]['discards'] = catch_value
                        elif catch_type == 'market':
                            result['catches'][group_name][fleet_name]['market'] = catch_value
                        elif catch_type == 'prop mort':
                            result['catches'][group_name][fleet_name]['prop_mort'] = catch_value
            else:
                fleet_data[child.tag] = child.text
        
        if fleet_name:
            fleet_data['fleet_name'] = fleet_name
            result['fleets'].append(fleet_data)
    
    # Extract catch data
    for catch_elem in root.iter('catch'):
        group_name = None
        fleet_name = None
        landings = 0.0
        discards = 0.0
        
        for child in catch_elem:
            if child.tag in ['group', 'group_name']:
                group_name = child.text
            elif child.tag in ['fleet', 'fleet_name']:
                fleet_name = child.text
            elif child.tag == 'landings':
                try:
                    landings = float(child.text) if child.text else 0.0
                except ValueError:
                    landings = 0.0
            elif child.tag == 'discards':
                try:
                    discards = float(child.text) if child.text else 0.0
                except ValueError:
                    discards = 0.0
        
        if group_name and fleet_name:
            if group_name not in result['catches']:
                result['catches'][group_name] = {}
            # Only add if not already present from fleet/catch_descr parsing
            if fleet_name not in result['catches'][group_name]:
                result['catches'][group_name][fleet_name] = {
                    'landings': landings,
                    'discards': discards
                }
            else:
                # Update only if values are provided
                if landings > 0:
                    result['catches'][group_name][fleet_name]['landings'] = landings
                if discards > 0:
                    result['catches'][group_name][fleet_name]['discards'] = discards
    
    return result


def ecobase_to_rpath(
    model_data: Dict[str, Any],
    include_fleets: bool = True,
    use_input_values: bool = True
) -> RpathParams:
    """Convert EcoBase model data to RpathParams.
    
    Parameters
    ----------
    model_data : dict
        Model data from get_ecobase_model()
    include_fleets : bool
        Whether to include fishing fleets
    use_input_values : bool
        If True, prefer input values (before balancing) over output values.
        EcoBase stores both input (original) and output (balanced) parameters.
    
    Returns
    -------
    RpathParams
        PyPath parameter structure ready for balancing
    
    Example
    -------
    >>> model_data = get_ecobase_model(403)
    >>> params = ecobase_to_rpath(model_data)
    >>> from pypath.core.ecopath import rpath
    >>> balanced = rpath(params)
    """
    groups_data = model_data.get('groups', [])
    diet_data = model_data.get('diet', {})
    fleets_data = model_data.get('fleets', [])
    catches_data = model_data.get('catches', {})
    
    if not groups_data:
        raise ValueError("No group data found in model")
    
    # Classify groups
    group_names = []
    group_types = []  # 0=consumer, 1=producer, 2=detritus, 3=fleet
    
    for g in groups_data:
        name = g.get('group_name', g.get('name', f"Group_{len(group_names)+1}"))
        group_names.append(name)
        
        # Determine type from various possible fields
        gtype = g.get('group_type', g.get('type', 0))
        if isinstance(gtype, str):
            gtype_lower = gtype.lower()
            if 'producer' in gtype_lower or 'primary' in gtype_lower:
                gtype = 1
            elif 'detritus' in gtype_lower or 'det' in gtype_lower:
                gtype = 2
            elif 'fleet' in gtype_lower or 'fish' in gtype_lower:
                gtype = 3
            else:
                gtype = 0
        
        # Also check if PB > 0 but QB = 0 for producers
        pb = g.get('prod_biom', g.get('pb', 0)) or 0
        qb = g.get('cons_biom', g.get('qb', 0)) or 0
        if pb > 0 and (qb == 0 or qb is None):
            gtype = 1
        
        group_types.append(int(gtype))
    
    # Add fleets if present and requested
    if include_fleets and fleets_data:
        for f in fleets_data:
            fleet_name = f.get('fleet_name', f.get('name', f"Fleet_{len(group_names)+1}"))
            group_names.append(fleet_name)
            group_types.append(3)
    
    # Create RpathParams
    params = create_rpath_params(
        groups=group_names,
        types=group_types
    )
    
    # Fill in group parameters
    # EcoBase field names:
    # - Numeric values are stored in: biomass, pb, qb, ee, gs, etc.
    # - Boolean flags (*_input) indicate if user entered the value or it was calculated
    # The actual values are ALWAYS in pb, qb, ee, biomass - the _input suffix is a boolean flag!
    for i, g in enumerate(groups_data):
        # Biomass - the numeric value is in 'biomass', not 'biomass_input'
        biomass = g.get('biomass', g.get('b', None))
        biomass_val = safe_float(biomass)
        if biomass_val is not None:
            params.model.loc[i, 'Biomass'] = biomass_val
        
        # PB (P/B ratio) - the numeric value is in 'pb', not 'pb_input'  
        pb = g.get('pb', g.get('prod_biom', None))
        pb_val = safe_float(pb)
        if pb_val is not None:
            params.model.loc[i, 'PB'] = pb_val
        
        # QB (Q/B ratio) - the numeric value is in 'qb', not 'qb_input'
        qb = g.get('qb', g.get('cons_biom', None))
        qb_val = safe_float(qb)
        if qb_val is not None and group_types[i] != 1:  # Not for producers
            params.model.loc[i, 'QB'] = qb_val
        
        # EE (Ecotrophic efficiency) - the numeric value is in 'ee', not 'ee_input'
        ee = g.get('ee', g.get('ecotrophic_eff', None))
        ee_val = safe_float(ee)
        if ee_val is not None:
            params.model.loc[i, 'EE'] = ee_val
        
        # Unassimilated fraction (GS in EcoBase)
        unassim = g.get('gs', g.get('unassim_cons', 0.2))
        unassim_val = safe_float(unassim, default=0.2)
        if unassim_val is not None:
            params.model.loc[i, 'Unassim'] = unassim_val
        
        # Biomass accumulation
        ba = g.get('biomass_accum', g.get('biomass_acc', g.get('ba', 0.0)))
        ba_val = safe_float(ba, default=0.0)
        if ba_val is not None:
            params.model.loc[i, 'BioAcc'] = ba_val
    
    # Fill diet matrix
    # Note: params.diet has 'Group' as a column with prey names, not as index
    # We need to find the row by matching the Group column
    diet_groups = params.diet['Group'].tolist()
    
    for pred_name, prey_dict in diet_data.items():
        if pred_name in params.diet.columns:
            for prey_name, proportion in prey_dict.items():
                # Find the row index for this prey
                if prey_name in diet_groups:
                    row_idx = diet_groups.index(prey_name)
                    prop_val = safe_float(proportion, default=0.0)
                    if prop_val is not None and prop_val > 0:
                        params.diet.iloc[row_idx, params.diet.columns.get_loc(pred_name)] = prop_val
    
    # Fill catch data
    if include_fleets and catches_data:
        for group_name, fleet_catches in catches_data.items():
            if group_name in params.model['Group'].values:
                group_idx = params.model[params.model['Group'] == group_name].index[0]
                for fleet_name, catch_data in fleet_catches.items():
                    if fleet_name in params.model.columns:
                        landings = safe_float(catch_data.get('landings', 0), default=0.0)
                        if landings is not None:
                            params.model.loc[group_idx, fleet_name] = landings
    
    # Store model name
    params.model_name = f"EcoBase Model {model_data.get('model_id', 'Unknown')}"
    
    return params


def search_ecobase_models(
    query: str,
    field: str = 'all',
    models_df: Optional[pd.DataFrame] = None
) -> pd.DataFrame:
    """Search EcoBase models by keyword.
    
    Parameters
    ----------
    query : str
        Search term
    field : str
        Field to search: 'all', 'model_name', 'country', 'ecosystem_type', 'author'
    models_df : pd.DataFrame, optional
        Pre-fetched models DataFrame. If None, will fetch from EcoBase.
    
    Returns
    -------
    pd.DataFrame
        Matching models
    
    Example
    -------
    >>> results = search_ecobase_models("Baltic")
    >>> results = search_ecobase_models("coral", field="ecosystem_type")
    """
    if models_df is None:
        models_df = list_ecobase_models()
    
    query_lower = query.lower()
    
    # Reset index to avoid alignment issues
    models_df = models_df.reset_index(drop=True)
    
    if field == 'all':
        # Search across all text fields
        mask = pd.Series([False] * len(models_df), index=models_df.index)
        for col in ['model_name', 'country', 'ecosystem_type', 'author', 'reference']:
            if col in models_df.columns:
                col_mask = models_df[col].astype(str).str.lower().str.contains(query_lower, na=False)
                mask = mask | col_mask
        return models_df[mask].copy().reset_index(drop=True)
    else:
        if field not in models_df.columns:
            raise ValueError(f"Unknown field: {field}")
        mask = models_df[field].astype(str).str.lower().str.contains(query_lower, na=False)
        return models_df[mask].copy().reset_index(drop=True)


def download_ecobase_model_to_file(
    model_id: int,
    output_path: str,
    format: str = 'csv'
) -> None:
    """Download EcoBase model and save to file(s).
    
    Parameters
    ----------
    model_id : int
        Model ID from EcoBase
    output_path : str
        Base path for output files (without extension)
    format : str
        Output format: 'csv', 'excel', 'json'
    
    Example
    -------
    >>> download_ecobase_model_to_file(403, "baltic_model", format="csv")
    # Creates: baltic_model_groups.csv, baltic_model_diet.csv
    """
    model_data = get_ecobase_model(model_id)
    params = ecobase_to_rpath(model_data)
    
    if format == 'csv':
        params.model.to_csv(f"{output_path}_groups.csv", index=False)
        params.diet.to_csv(f"{output_path}_diet.csv")
    elif format == 'excel':
        with pd.ExcelWriter(f"{output_path}.xlsx") as writer:
            params.model.to_excel(writer, sheet_name='Groups', index=False)
            params.diet.to_excel(writer, sheet_name='Diet')
    elif format == 'json':
        import json
        result = {
            'model': params.model.to_dict(orient='records'),
            'diet': params.diet.to_dict(),
        }
        with open(f"{output_path}.json", 'w') as f:
            json.dump(result, f, indent=2)
    else:
        raise ValueError(f"Unknown format: {format}")
