# -*- coding: utf-8 -*-
"""Species mapping utilities for Curonian Lagoon model.

Provides helper functions to map between CPUE/landings species codes
and Ecopath food web group indices.
"""

import pandas as pd
from pathlib import Path
from typing import Optional, Dict, List

# Load mapping table globally
_MAPPING_FILE = Path(__file__).parent / 'curonian_species_to_group_mapping.csv'
if _MAPPING_FILE.exists():
    SPECIES_MAPPING = pd.read_csv(_MAPPING_FILE)
else:
    SPECIES_MAPPING = None


def get_group_index(species_code: str) -> Optional[int]:
    """Return 0-based group index for species code.
    
    Parameters
    ----------
    species_code : str
        Species code (e.g., 'BRAM', 'PERC', 'ROAC', 'VIMB', 'ZAND')
    
    Returns
    -------
    int or None
        Group index (0-32) or None if not found
        
    Examples
    --------
    >>> get_group_index('BRAM')
    1
    >>> get_group_index('ROAC')
    0
    """
    if SPECIES_MAPPING is None:
        raise RuntimeError("Mapping table not found at:", _MAPPING_FILE)
    
    match = SPECIES_MAPPING[SPECIES_MAPPING['code'] == species_code.upper()]
    if len(match) > 0:
        return int(match['group_idx'].values[0])
    return None


def get_species_code(group_index: int) -> Optional[str]:
    """Return species code for group index.
    
    Parameters
    ----------
    group_index : int
        Group index (0-32)
    
    Returns
    -------
    str or None
        Species code or None if group not mapped
        
    Examples
    --------
    >>> get_species_code(0)
    'ROAC'
    >>> get_species_code(1)
    'BRAM'
    """
    if SPECIES_MAPPING is None:
        raise RuntimeError("Mapping table not found at:", _MAPPING_FILE)
    
    match = SPECIES_MAPPING[SPECIES_MAPPING['group_idx'] == group_index]
    if len(match) > 0:
        return match['code'].values[0]
    return None


def get_group_name(species_code: str) -> Optional[str]:
    """Return Ecopath group name for species code.
    
    Parameters
    ----------
    species_code : str
        Species code
    
    Returns
    -------
    str or None
        Group name from Ecopath model
    """
    if SPECIES_MAPPING is None:
        raise RuntimeError("Mapping table not found at:", _MAPPING_FILE)
    
    match = SPECIES_MAPPING[SPECIES_MAPPING['code'] == species_code.upper()]
    if len(match) > 0:
        return match['group_name'].values[0]
    return None


def validate_timeseries_mapping(collection) -> Dict[str, any]:
    """Verify all series have valid group indices.
    
    Parameters
    ----------
    collection : EweTimeSeriesCollection
        Timeseries collection with group_idx attributes
    
    Returns
    -------
    dict
        Validation results with keys:
        - 'valid': bool, all indices valid?
        - 'total_series': int, number of series
        - 'mapped_series': int, series with group_idx >= 0
        - 'unmapped_series': list, series without mapping
        - 'invalid_indices': list, invalid group indices found
    """
    results = {
        'valid': True,
        'total_series': len(collection.series),
        'mapped_series': 0,
        'unmapped_series': [],
        'invalid_indices': []
    }
    
    for s in collection.series:
        if hasattr(s, 'group_idx'):
            idx = s.group_idx
            if idx >= 0:
                results['mapped_series'] += 1
                if not (0 <= idx <= 32):
                    results['invalid_indices'].append((s.name, idx))
                    results['valid'] = False
            else:
                results['unmapped_series'].append(s.name)
        else:
            results['unmapped_series'].append(f"{s.name} (no group_idx attr)")
            results['valid'] = False
    
    return results


def apply_mapping_to_collection(collection, mapping_table=None):
    """Apply species->group mapping to timeseries collection.
    
    Modifies collection in-place by adding group_idx to each series.
    
    Parameters
    ----------
    collection : EweTimeSeriesCollection
        Timeseries collection
    mapping_table : pd.DataFrame, optional
        Custom mapping table. If None, uses global SPECIES_MAPPING
    
    Returns
    -------
    dict
        Mapping results
    """
    if mapping_table is None:
        if SPECIES_MAPPING is None:
            raise RuntimeError("No mapping table provided and global mapping not loaded")
        mapping_table = SPECIES_MAPPING
    
    results = {'success': 0, 'failed': 0, 'series': []}
    
    for i, s in enumerate(collection.series):
        if i == 0:  # Environmental forcing (discharge)
            s.group_idx = -1
            results['success'] += 1
            results['series'].append({'name': s.name, 'group_idx': -1, 'status': 'environmental'})
        else:
            # Try to match series name to species
            found = False
            for _, row in mapping_table.iterrows():
                if row['common'].lower() in s.name.lower():
                    s.group_idx = int(row['group_idx'])
                    results['success'] += 1
                    results['series'].append({
                        'name': s.name,
                        'group_idx': s.group_idx,
                        'status': 'mapped',
                        'species_code': row['code']
                    })
                    found = True
                    break
            
            if not found:
                results['failed'] += 1
                results['series'].append({'name': s.name, 'status': 'unmapped'})
    
    return results


def get_cpue_species_list() -> List[str]:
    """Return list of CPUE species codes.
    
    Returns
    -------
    list of str
        CPUE species codes: ['ROAC', 'BRAM', 'ZAND', 'PERC', 'VIMB']
    """
    if SPECIES_MAPPING is None:
        return []
    
    cpue_species = SPECIES_MAPPING[SPECIES_MAPPING['data_type'] == 'CPUE/Landings']
    return cpue_species['code'].tolist()


def get_all_mappings() -> pd.DataFrame:
    """Return full species mapping table.
    
    Returns
    -------
    pd.DataFrame
        Complete mapping with columns:
        code, common, scientific, group_idx, group_name, data_type, time_period, notes
    """
    if SPECIES_MAPPING is None:
        raise RuntimeError("Mapping table not found")
    return SPECIES_MAPPING.copy()


if __name__ == '__main__':
    # Test the module
    print("Testing species_mapping module...")
    print()
    
    try:
        # Test lookups
        idx = get_group_index('BRAM')
        print(f"BRAM -> group index {idx}")
        
        code = get_species_code(0)
        print(f"Group 0 -> species code {code}")
        
        name = get_group_name('ROAC')
        print(f"ROAC -> group name '{name}'")
        
        cpue = get_cpue_species_list()
        print(f"CPUE species: {cpue}")
        
        print("\nAll tests passed!")
    except Exception as e:
        print(f"Error: {e}")
