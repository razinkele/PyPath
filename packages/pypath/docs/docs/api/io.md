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
- `ecosim_scenario_from_ewemdb()` — load a complete Ecosim scenario with
  vulnerability overrides, foraging time adjustments, forced biomass,
  fishing effort, and environmental forcing from a specific EwE scenario ID

::: pypath.io.ewemdb
    options:
      show_root_heading: true

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
