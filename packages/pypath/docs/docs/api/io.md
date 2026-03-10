# I/O API Reference

Import and export functions for loading Ecopath models from various
sources: EcoBase online repository, EwE Access databases, CSV files,
and external biological/environmental data services.

## EcoBase

Search, download, and convert models from the EcoBase online repository
of published Ecopath models.

::: pypath.io.ecobase
    options:
      show_root_heading: true

## EwE Database (.eweaccdb)

Read Ecopath with Ecosim models stored in Microsoft Access database
format (`.eweaccdb` / `.ewemdb`).

Key functions:

- `read_ewemdb()` — load Ecopath parameters (biomass, diet, stanzas)
- `read_ewemdb_table()` — read any raw table from the database
- `list_ewemdb_tables()` — list all tables in an EwE database
- `get_ewemdb_metadata()` — retrieve model metadata
- `ecosim_scenario_from_ewemdb()` — load a complete Ecosim scenario with
  vulnerability overrides, foraging time adjustments, forced biomass,
  fishing effort, and environmental forcing from a specific EwE scenario ID

See the [EwE Database Loading example](../examples/ewe-database.md) for
a complete walkthrough.

::: pypath.io.ewemdb
    options:
      show_root_heading: true

### EwE Database Export

Export PyPath models back to native EwE format:

- `write_ewemdb(params, path)` — Auto-detects best backend
- `write_ewemdb(params, path, backend="csv")` — Force CSV bundle
- `write_ewemdb(params, path, scenarios=[scen1])` — Include Ecosim
- `write_ewemdb(params, path, ecospace=ecospace)` — Include Ecospace

**Example:**

```python
from pypath.io.ewe_writer import write_ewemdb

# Export Ecopath model
write_ewemdb(params, "my_model.eweaccdb")

# With Ecosim scenarios
write_ewemdb(params, "my_model.eweaccdb", scenarios=[scenario1])

# Cross-platform CSV fallback
write_ewemdb(params, "my_model.ewecsv.zip", backend="csv")
```

## Biological Data (WoRMS/OBIS/FishBase)

Integration with marine biological databases: WoRMS for taxonomy,
OBIS for occurrence records, and FishBase/SeaLifeBase for species
parameters (growth, mortality, diet).

::: pypath.io.biodata
    options:
      show_root_heading: true

## Marine Environmental Data (EMODnet)

Download and process marine environmental layers (bathymetry,
temperature, salinity, substrate) from the EMODnet Web Coverage
Service for use as Ecospace habitat and environmental inputs.

::: pypath.io.marine_data
    options:
      show_root_heading: true

## Utilities

Shared I/O helper functions: CSV reading, unit conversions, and data
validation.

::: pypath.io.utils
    options:
      show_root_heading: true
