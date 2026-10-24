# CPUE_BAR Dataset Metadata

**Source:** CPUE_BAR.xlsx  
**Temporal Coverage:** 2001-2018 (18 years)  
**Status:** Ready for Ecosim fitting

## Overview

Catch Per Unit Effort (CPUE) data from Curonian Lagoon commercial fisheries. This is the primary dataset for validating Ecosim dynamics predictions.

## Data Structure

- **Main sheet (Data):** 590 observations x 111 variables
  - Contains temporal index and effort/catch metrics
- **Species sheets:** Separate analysis sheets for each commercial species
  - Bream: 285 observations
  - Zander: 410 observations
  - Roach: 348 observations
  - Vimba: 168 observations
  - Perch: 348 observations

## Data Quality

- **Completeness:** 95-100%% (minimal missing values)
- **Continuity:** Complete 18-year series without major gaps
- **Resolution:** Varies by sheet (daily/monthly/annual)

## Validation Notes

- Primary validation dataset for Ecosim CPUE predictions
- Species-specific analysis enables fine-grained model tuning
- Long enough series (18 years) to capture major ecosystem transitions

## Usage in Phase 2

Will be used for:
1. Ecosim CPUE fitting via fit_to_timeseries()
2. Model validation against observed trends
3. Species-specific biomass trajectory verification
