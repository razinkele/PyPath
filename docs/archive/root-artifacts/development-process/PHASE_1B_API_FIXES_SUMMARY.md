# Phase 1B API Fixes Summary

## Overview
Fixed all API usage issues in `packages/pypath/tests/test_phase1b_ewe_export.py` by:
1. Correcting imports to use actual module classes
2. Removing invalid `groups` kwarg from container __init__() calls
3. Replacing incorrect method calls with actual container methods
4. Adding required fields to meet validation constraints

## Key API Fixes

### 1. PedigreeContainer
**Issue**: Test passed `groups` kwarg to `__init__()`; tried to use non-existent `set_group_index()` method
**Fix**: 
- Initialize with `PedigreeContainer()` (no args)
- Use `add_pedigree(group_name, EwePedigreeConfig)` method
- Changed `EwePedigreeConfig` import (from `PedigreeConfig`)

```python
# Before
pedigree = PedigreeContainer(groups=["Phyto", "Zoo", "Fish"])
pedigree.set_group_index("Phyto", 1)

# After
pedigree = PedigreeContainer()
pedigree.add_pedigree("Phyto", EwePedigreeConfig(pedigree_index=1))
```

### 2. TaxonomyLinkage
**Issue**: Test passed `groups` kwarg; tried to use non-existent `set_group_taxon()` method
**Fix**:
- Initialize with `TaxonomyLinkage()` (no args)
- Use `add_taxon(TaxonLink)` to register taxon
- Use `assign_group(group_name, scientific_name)` to link group to taxon
- Added required `worms_aphia_id` to TaxonLink for phyto (validation requires external ID)

```python
# Before
linkage = TaxonomyLinkage(groups=[...])
linkage.set_group_taxon("Phyto", phyto_taxon)

# After
linkage = TaxonomyLinkage()
linkage.add_taxon(phyto_taxon)
linkage.assign_group("Phyto", phyto_taxon.scientific_name)
```

### 3. SizeStructure
**Issue**: Test passed `groups` kwarg; tried to use non-existent `set_group_distribution()` method
**Fix**:
- Initialize with `SizeStructure()` (no args)
- Use `add_distribution(SizeDistribution)` method

```python
# Before
size_struct = SizeStructure(groups=[...])
size_struct.set_group_distribution("Zoo", zoo_dist)

# After
size_struct = SizeStructure()
size_struct.add_distribution(zoo_dist)
```

### 4. FleetDynamics
**Issue**: Test passed `groups` kwarg
**Fix**:
- Initialize with `FleetDynamics()` (no args)
- Already using correct `add_fleet()` method (no change needed)

### 5. LandingsData
**Issue**: Test passed `groups` kwarg
**Fix**:
- Initialize with `LandingsData()` (no args)
- Already using correct `add_record()` method (no change needed)

### 6. NutrientCycle
**Issue**: Test passed `groups` kwarg; tried to use non-existent `set_group_content()` method
**Fix**:
- Initialize with `NutrientCycle()` (no args)
- Use `add_content(NutrientContent)` method

## Import Changes
Changed from:
```python
from pypath.core.pedigree import PedigreeContainer, PedigreeConfig
```

To:
```python
from pypath.core.pedigree import PedigreeContainer, EwePedigreeConfig
from pypath.core.fleet import FleetGroup, GearType, FleetDynamics
```

## Validation Fixes
- Added `worms_aphia_id=181` to phyto TaxonLink (Diatomeae) because TaxonLink validation requires at least one external ID (WoRMS or ITIS)

## Test Results
✅ All 18 tests pass:
- 6 instantiation smoke tests
- 2 pedigree tests
- 2 taxonomy tests
- 2 size structure tests
- 2 fleet dynamics tests
- 2 landings data tests
- 2 nutrient cycle tests

## Files Changed
- `packages/pypath/tests/test_phase1b_ewe_export.py` - Completely rewritten with all API fixes applied