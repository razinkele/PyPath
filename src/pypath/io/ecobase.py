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


# EcoBase API endpoints
ECOBASE_LIST_URL = "http://sirs.agrocampus-ouest.fr/EcoBase/php/webser/soap-client_3.php"
ECOBASE_MODEL_URL = "http://sirs.agrocampus-ouest.fr/EcoBase/php/webser/soap-client.php?no_model="


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


def _fetch_url(url: str, timeout: int = 30) -> str:
    """Fetch content from URL.
    
    Parameters
    ----------
    url : str
        URL to fetch
    timeout : int
        Request timeout in seconds
    
    Returns
    -------
    str
        Response content as string
    """
    if HAS_REQUESTS:
        response = requests.get(url, timeout=timeout)
        response.raise_for_status()
        return response.text
    else:
        with urllib.request.urlopen(url, timeout=timeout) as response:
            return response.read().decode('utf-8')


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
        xml_content = _fetch_url(ECOBASE_LIST_URL, timeout=timeout)
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
        xml_content = _fetch_url(url, timeout=timeout)
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
    
    # Extract groups
    for group_elem in root.iter('group'):
        group_data = {}
        for child in group_elem:
            tag = child.tag
            text = child.text
            
            # Try to convert numeric values
            if text:
                try:
                    if '.' in text or 'e' in text.lower():
                        group_data[tag] = float(text)
                    else:
                        group_data[tag] = int(text)
                except ValueError:
                    group_data[tag] = text
            else:
                group_data[tag] = None
        
        if group_data:
            result['groups'].append(group_data)
    
    # Extract diet matrix
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
    
    # Extract fleet/fishery data
    for fleet_elem in root.iter('fleet'):
        fleet_data = {}
        for child in fleet_elem:
            fleet_data[child.tag] = child.text
        if fleet_data:
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
            result['catches'][group_name][fleet_name] = {
                'landings': landings,
                'discards': discards
            }
    
    return result


def ecobase_to_rpath(
    model_data: Dict[str, Any],
    include_fleets: bool = True
) -> RpathParams:
    """Convert EcoBase model data to RpathParams.
    
    Parameters
    ----------
    model_data : dict
        Model data from get_ecobase_model()
    include_fleets : bool
        Whether to include fishing fleets
    
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
    for i, g in enumerate(groups_data):
        # Biomass
        biomass = g.get('biomass', g.get('b', None))
        if biomass is not None:
            params.model.loc[i, 'Biomass'] = float(biomass)
        
        # PB
        pb = g.get('prod_biom', g.get('pb', None))
        if pb is not None:
            params.model.loc[i, 'PB'] = float(pb)
        
        # QB
        qb = g.get('cons_biom', g.get('qb', None))
        if qb is not None and group_types[i] != 1:  # Not for producers
            params.model.loc[i, 'QB'] = float(qb)
        
        # EE
        ee = g.get('ecotrophic_eff', g.get('ee', None))
        if ee is not None:
            params.model.loc[i, 'EE'] = float(ee)
        
        # Unassimilated fraction
        unassim = g.get('unassim_cons', g.get('gs', 0.2))
        if unassim is not None:
            params.model.loc[i, 'Unassim'] = float(unassim)
        
        # Biomass accumulation
        ba = g.get('biomass_acc', g.get('ba', 0.0))
        if ba is not None:
            params.model.loc[i, 'BioAcc'] = float(ba)
    
    # Fill diet matrix
    for pred_name, prey_dict in diet_data.items():
        if pred_name in params.diet.columns:
            for prey_name, proportion in prey_dict.items():
                if prey_name in params.diet.index:
                    params.diet.loc[prey_name, pred_name] = float(proportion)
    
    # Fill catch data
    if include_fleets and catches_data:
        for group_name, fleet_catches in catches_data.items():
            if group_name in params.model['Group'].values:
                group_idx = params.model[params.model['Group'] == group_name].index[0]
                for fleet_name, catch_data in fleet_catches.items():
                    if fleet_name in params.model.columns:
                        landings = catch_data.get('landings', 0)
                        params.model.loc[group_idx, fleet_name] = landings
    
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
    
    if field == 'all':
        # Search across all text fields
        mask = pd.Series([False] * len(models_df))
        for col in ['model_name', 'country', 'ecosystem_type', 'author', 'reference']:
            if col in models_df.columns:
                mask |= models_df[col].astype(str).str.lower().str.contains(query_lower, na=False)
        return models_df[mask].copy()
    else:
        if field not in models_df.columns:
            raise ValueError(f"Unknown field: {field}")
        mask = models_df[field].astype(str).str.lower().str.contains(query_lower, na=False)
        return models_df[mask].copy()


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
