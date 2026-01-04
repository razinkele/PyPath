# Biodiversity Data Module - Quick Start Guide

## Installation

```bash
pip install pypath-ecopath[biodata]
```

This installs PyPath with optional biodiversity data support (pyworms, pyobis, requests).

## Quick Examples

### 1. Get Species Information

```python
from pypath.io.biodata import get_species_info

# Simple query
info = get_species_info("Atlantic cod")

print(f"Scientific name: {info.scientific_name}")
print(f"WoRMS AphiaID: {info.aphia_id}")
print(f"Trophic level: {info.trophic_level}")
print(f"Max length: {info.max_length} cm")
print(f"Occurrences: {info.occurrence_count}")
print(f"Depth range: {info.depth_range}")
```

### 2. Batch Process Multiple Species

```python
from pypath.io.biodata import batch_get_species_info

species = ["Cod", "Herring", "Sprat", "Mackerel"]
df = batch_get_species_info(species)

print(df[['common_name', 'scientific_name', 'trophic_level']])
```

### 3. Create Ecopath Model from Biodiversity Data

```python
from pypath.io.biodata import batch_get_species_info, biodata_to_rpath
from pypath.core.ecopath import rpath

# 1. Get species data
species = ["Atlantic cod", "Herring", "Sprat"]
df = batch_get_species_info(species)

# 2. Provide biomass estimates (required for good model)
biomass = {
    'Gadus morhua': 2.0,      # t/km²
    'Clupea harengus': 5.0,
    'Sprattus sprattus': 8.0
}

# 3. Convert to Ecopath parameters
params = biodata_to_rpath(df, biomass_estimates=biomass)

# 4. Balance the model
balanced = rpath(params)

# 5. View results
print(balanced.model[['Group', 'Biomass', 'PB', 'QB', 'TL']])
```

## Workflow

```
Common Name → WoRMS → AphiaID → Scientific Name → OBIS + FishBase → Ecopath
```

## Data Sources

| Database | Data Type | Package |
|----------|-----------|---------|
| **WoRMS** | Taxonomy, nomenclature | pyworms |
| **OBIS** | Occurrence records, spatial | pyobis |
| **FishBase** | Traits, diet, growth | Custom REST API |

## Main Functions

### get_species_info()
Get comprehensive data for a single species.

```python
get_species_info(
    common_name: str,
    include_occurrences: bool = True,
    include_traits: bool = True,
    strict: bool = False,
    cache: bool = True,
    timeout: int = 30
) -> SpeciesInfo
```

### batch_get_species_info()
Process multiple species in parallel.

```python
batch_get_species_info(
    common_names: List[str],
    max_workers: int = 5,
    ...
) -> pd.DataFrame
```

### biodata_to_rpath()
Convert biodiversity data to Ecopath parameters.

```python
biodata_to_rpath(
    species_data: Union[SpeciesInfo, pd.DataFrame],
    biomass_estimates: Optional[Dict[str, float]] = None,
    area_km2: float = 1000.0
) -> RpathParams
```

## Parameter Estimation

The module automatically estimates Ecopath parameters:

- **P/B**: Estimated from von Bertalanffy growth parameter K
  - Formula: `P/B ≈ K × 2.5`

- **Q/B**: Estimated from trophic level and P/B
  - Uses Palomares & Pauly (1998) relationship
  - Accounts for trophic efficiency

- **Biomass**: From user estimates (recommended) or occurrence density proxy

- **Diet**: From FishBase diet composition data

- **Trophic Level**: Directly from FishBase

## Caching

The module uses intelligent caching to reduce API load:

```python
from pypath.io.biodata import get_cache_stats, clear_cache

# Check cache performance
stats = get_cache_stats()
print(f"Hit rate: {stats['hit_rate']:.2%}")
print(f"Size: {stats['size']} entries")

# Clear if needed
clear_cache()
```

- Default TTL: 1 hour
- LRU eviction when full
- Separate cache keys per data source

## Error Handling

### Strict Mode (raises exceptions)
```python
try:
    info = get_species_info("Unknown species", strict=True)
except SpeciesNotFoundError:
    print("Species not found in WoRMS")
except APIConnectionError:
    print("API connection failed")
```

### Non-Strict Mode (graceful degradation)
```python
# Returns partial data if some APIs fail
info = get_species_info("Species name", strict=False)
# Warning logged, but continues with available data
```

## Tips & Best Practices

### 1. Always Provide Biomass Estimates
The occurrence-based proxy is rough. Provide your own biomass estimates:

```python
# Good: User-provided biomass
biomass = {'Species A': 5.0, 'Species B': 3.0}
params = biodata_to_rpath(df, biomass_estimates=biomass)

# Not ideal: Occurrence proxy (warning issued)
params = biodata_to_rpath(df)  # Uses occurrence density
```

### 2. Use Batch Processing for Multiple Species
More efficient than sequential queries:

```python
# Good: Parallel batch processing
df = batch_get_species_info(["Sp1", "Sp2", "Sp3"], max_workers=5)

# Less efficient: Sequential
results = [get_species_info(sp) for sp in species]
```

### 3. Cache Improves Performance
First query is slow (~2-3 sec), subsequent queries are instant:

```python
# First call: 2-3 seconds
info1 = get_species_info("Atlantic cod")

# Second call: < 0.001 seconds (cached)
info2 = get_species_info("Atlantic cod")
```

### 4. Handle Ambiguous Names
Some common names match multiple species:

```python
try:
    info = get_species_info("Cod")  # Multiple species
except AmbiguousSpeciesError as e:
    print(f"Multiple matches:")
    for match in e.matches:
        print(f"  - {match['scientificname']}")
    # Use more specific name or select manually
```

### 5. Check Data Availability
Not all species have complete data:

```python
info = get_species_info("Species name", strict=False)

if info.trophic_level is None:
    print("No FishBase data available")
if info.occurrence_count is None:
    print("No OBIS records")
```

## Common Issues

### ImportError: pyworms/pyobis not found
```bash
# Install biodiversity dependencies
pip install pyworms pyobis requests
# Or install with extra
pip install pypath-ecopath[biodata]
```

### SpeciesNotFoundError
- Check spelling of common name
- Try scientific name instead
- Species may not be in WoRMS database

### APIConnectionError
- Check internet connection
- APIs may be temporarily down
- Use `strict=False` for graceful degradation

### Incomplete Data
- Not all fish in FishBase (use strict=False)
- Some species lack trait data
- Provide manual estimates where needed

## Advanced Usage

### Custom Cache Configuration
```python
from pypath.io.biodata import _biodata_cache

# Configure cache
_biodata_cache = BiodiversityCache(
    maxsize=500,      # Fewer entries
    ttl_seconds=7200  # 2 hours
)
```

### Selective Data Fetching
```python
# Skip OBIS (faster, no occurrence data)
info = get_species_info(
    "Atlantic cod",
    include_occurrences=False,
    include_traits=True
)

# Skip FishBase (faster, no trait data)
info = get_species_info(
    "Atlantic cod",
    include_occurrences=True,
    include_traits=False
)
```

### Export to Different Formats
```python
# Get data as DataFrame
df = batch_get_species_info(species)

# Export to CSV
df.to_csv("species_data.csv", index=False)

# Export to Excel
df.to_excel("species_data.xlsx", index=False)

# Convert to Ecopath and export
params = biodata_to_rpath(df, biomass_estimates=biomass)
params.model.to_csv("ecopath_model.csv", index=False)
```

## Example Workflow: North Sea Model

```python
from pypath.io.biodata import batch_get_species_info, biodata_to_rpath
from pypath.core.ecopath import rpath

# 1. Define species
north_sea_species = [
    "Atlantic cod",
    "Haddock",
    "Whiting",
    "Herring",
    "Sprat",
    "Norway pout",
    "Plaice",
    "Sole"
]

# 2. Fetch biodiversity data
print("Fetching species data from WoRMS, OBIS, and FishBase...")
df = batch_get_species_info(north_sea_species, max_workers=8)

# 3. Define biomass (t/km²) - from surveys or literature
biomass_estimates = {
    'Gadus morhua': 0.8,      # Cod
    'Melanogrammus aeglefinus': 1.2,  # Haddock
    'Merlangius merlangus': 0.9,     # Whiting
    'Clupea harengus': 6.0,          # Herring
    'Sprattus sprattus': 10.0,       # Sprat
    'Trisopterus esmarkii': 2.0,     # Norway pout
    'Pleuronectes platessa': 1.5,    # Plaice
    'Solea solea': 0.4               # Sole
}

# 4. Create Ecopath model
print("Converting to Ecopath parameters...")
params = biodata_to_rpath(
    df,
    biomass_estimates=biomass_estimates,
    area_km2=750000  # North Sea area
)

# 5. Balance the model
print("Balancing model...")
balanced = rpath(params)

# 6. Analyze results
print("\nNorth Sea Ecopath Model:")
print(balanced.model[['Group', 'Type', 'Biomass', 'PB', 'QB', 'TL', 'EE']])

# 7. Check diagnostics
from pypath.core.ecopath import check_rpath_params
diagnostics = check_rpath_params(balanced)
print("\nModel Diagnostics:")
print(diagnostics)

# 8. Export
balanced.model.to_csv("north_sea_model.csv", index=False)
print("\nModel exported to north_sea_model.csv")
```

## Further Reading

- Full implementation docs: `BIODATA_MODULE_IMPLEMENTATION.md`
- API documentation: Use `help(function_name)` in Python
- Test suite: `tests/test_biodata.py` for usage examples
- WoRMS: https://www.marinespecies.org/
- OBIS: https://obis.org/
- FishBase: https://www.fishbase.org/

## Support

For issues or questions:
1. Check the test suite for examples
2. Read the full docstrings: `help(get_species_info)`
3. See `BIODATA_MODULE_IMPLEMENTATION.md` for details
