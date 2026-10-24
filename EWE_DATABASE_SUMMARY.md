# EwE Models & Fishery Data Inventory
## Comprehensive Database Analysis for Curonian Lagoon Case Study

**Generated:** 2026-10-23 17:23:51

---

## Executive Summary

This inventory catalogs **all EwE (Ecopath with Ecosim) models** and **fishery-related datasets** across four major source directories to support the Curonian Lagoon case study development.

### Key Statistics

- **Total EwE Models:** 63 files (137.97 MB)
  - .EwEaccdb format: 33 models
  - .ewemdb format: 23 models
  - .mdb format: 7 models

- **Total Fishery Data Files:** 253 datasets
  - Curonian Lagoon/Baltic focus: ~122 files
  - Various fishery metrics: CPUE, landings, biomass, diet data
  - Geographic coverage: Baltic Sea, Gulf of Mexico, Mediterranean, generic examples

- **Source Directories:**
  1. EwE sample databases (20 models, 4 fishery datasets)
  2. ECOPATH (8 models, 84 fishery datasets)
  3. ecopath workshop (33 models, 53 fishery datasets)
  4. EDGARAS (2 models, 112 fishery datasets) ⭐ **Curonian Lagoon focus**

---

## I. EwE Models Inventory

### A. Distribution by File Format

| Format | Count | Purpose |
|--------|-------|---------|
| .EwEaccdb | 33 | EwE Ecosim Accounting database (newer format) |
| .ewemdb | 23 | EwE Master database (older/transitional format) |
| .mdb | 7 | Microsoft Access database (legacy format) |

### B. Distribution by Ecosystem Type

| Ecosystem Type | Count | Notes |
|----------------|-------|-------|
| Unknown | 37 | Models requiring manual classification |
| Tutorial/Example | 19 | Sample models and workshop teaching materials |
| Gulf of Mexico | 4 | Tampa Bay, ecosystem example models |
| Baltic Sea | 3 | Including Curonian Lagoon variants |

### C. Distribution by Source Directory

| Directory | Count | Focus |
|-----------|-------|-------|
| ecopath workshop | 33 | Tutorial/training materials, diverse ecosystems |
| EwE sample databases | 20 | Reference examples for model types |
| ECOPATH | 8 | European models, including Baltic studies |
| EDGARAS | 2 | **Curonian Lagoon specific models** |

### D. Largest EwE Models (by size)

`
1. Ecopath.mdb (5.33 MB)                    - Full ecopath module
2. KM_Edgaras.ewemdb (4.92 MB)              - Curonian Lagoon ecosim
3. Test_model.ewemdb (4.64 MB)              - Test/development model
4. Georgia_Strait.EwEmdb (2.90 MB)          - Temperate coastal ecosystem
5. Ocean ECOST.ewemdb (2.83 MB)             - Oceanic ecosystem
6. EwE6_curonian_pelagic-multistanza (2.74) - Curonian pelagic multispecies
7. coast 2011-04-10 10.00.ewemdb (2.71)     - Coastal model variant
8. Sp_Curonian_2012.ewemdb (2.60)           - Single-species Curonian model
9. EDGARAS.ewemdb (2.55 MB)                 - EDGARAS project model
10. Black Sea 2004.ewemdb (2.49 MB)         - Black Sea ecosystem
`

---

## II. Fishery Data Inventory

### A. Data Types Identified

| Data Type | Count | Description |
|-----------|-------|-------------|
| Fishery Data (General) | 210 | Catch, landings, effort records |
| CPUE/Effort | 18 | Catch Per Unit Effort data |
| Landings/Catch | 14 | Commercial catch records |
| Biomass/Stock | 6 | Population abundance estimates |
| Time Series | 1 | Temporal dynamics |
| Diet Data | 2 | Stomach content/diet composition |
| Life History | 2 | Growth, recruitment, mortality |

### B. Geographic Distribution

| Region | Count | Key Files |
|--------|-------|-----------|
| **Curonian Lagoon** ⭐ | 113 | Primary focus; 14 years of CPUE data (2001-2018) |
| Unknown/Generic | 116 | Example/tutorial datasets |
| Gulf of Mexico | 15 | Tampa Bay, Everglades, Gulf ecosystems |
| Baltic Sea | 9 | Baltic-wide data, regional studies |

### C. Data by Source Directory

| Directory | Count | Primary Content |
|-----------|-------|-----------------|
| EDGARAS | 112 | **Curonian Lagoon fisheries database** |
| ECOPATH | 84 | European models, CPUE, time series |
| ecopath workshop | 53 | Educational datasets, sample data |
| EwE sample databases | 4 | Generic fishery examples |

### D. File Formats

| Format | Count | Uses |
|--------|-------|------|
| .xlsx/.xls | ~120 | Primary data format; Excel spreadsheets |
| .csv | ~60 | Data exchange format |
| .txt | ~40 | Text data, analysis output |
| .R | ~20 | R statistical scripts |
| .xml | ~5 | Metadata, configuration files |
| .h5 | ~1 | HDF5 scientific data format |

---

## III. Curonian Lagoon Data (EDGARAS) - In Depth

### A. Key Data Resources

**Location:** C:\Users\DELL\OneDrive - ku.lt\EDGARAS

#### 1. Primary CPUE Database
- **File:** CPUE_BAR.xlsx (689 KB, plus copy: 3,438 KB expanded)
- **Content:** Catch Per Unit Effort data across species and years
- **Species:** Baltic fish including perch, pike, roach, bream
- **Time Period:** 2001-2018 (18 years continuous records)

#### 2. Annual Fisheries Database (Kuršių marių duombazė)
- **Location:** Subdirectory "Kuršių marių duombazė"
- **Files:** 11 years of comprehensive fishing data
  - 2010-2020: Modern era monitoring
  - File sizes: 489 KB - 1.8 GB (growing with time)
  - Format: Excel (.xls, .xlsx)
- **Key file:** Kursiu marios 2018-2020.xlsx (1.7-1.8 GB each)

#### 3. Historical Data
- **Landings data:** "Sugavimai K_M 1947-2011.xls" (130 KB)
  - 64-year historical landings record
  - Critical for long-term trend analysis
- **Discharge data:** "Nemunas monthly discharges 1888-2011.xlsx"
  - 123-year river discharge record
  - Essential for environmental forcing

#### 4. Species-Specific Studies
- **Files:**
  - Roach.xlsx and Roach.R - Roach population analysis
  - Salinity.xlsx - Salinity measurements
  - Temperatures.xls - Temperature records
  - WRT.xlsx - Water residence time data

#### 5. EwE Models for Curonian Lagoon
- **KM_Edgaras.ewemdb** (4.92 MB) - Primary Curonian Lagoon Ecosim model
- **Test_model.ewemdb** (4.64 MB) - Development/testing model
- **KM_Edgaras_log.xml** - Model configuration/metadata

#### 6. Analysis Scripts
- **R Scripts:** CPUE.R, CPUEvsCATCH_graphs.R, MCY.R, Sugavimai.R, Fish.R
- **Statistical tools:** PCA, regression, time series analysis
- **Purpose:** Data processing, visualization, stock assessment

#### 7. Environmental Data
- **Fishery services directory:** 39 files
  - Canonical correspondence analysis results
  - Environmental variable data (X, Y, covariate matrices)
  - Statistical analysis outputs

### B. Curonian Lagoon Data Timeline

`
1888 -------- 1947 ----------- 2001 ------- 2011 -------- 2018 ---- 2020
River discharge    Landings    CPUE begins    Landings end    Current data
(123 years)        (64 years)  (18 years)     monitoring
`

### C. Available Species in Curonian Lagoon Data

From file analysis:
- **Perch** (Perca fluviatilis) - Primary commercial species
- **Pike** (Esox lucius)
- **Roach** (Rutilus rutilus) - Dedicated research files
- **Bream** (Abramis brama)
- **Carp** (Cyprinus carpio)
- **Eel** (Anguilla anguilla)
- **Other Baltic species** - Various forage fish

---

## IV. EwE Model Classification & Recommendations

### A. Baltic Sea Models (HIGH PRIORITY for Curonian Study)

| Model | File | Size | Status | Recommendation |
|-------|------|------|--------|-----------------|
| Curonian Lagoon Primary | KM_Edgaras.ewemdb | 4.92 MB | Ready | Use as baseline |
| Curonian Test Model | Test_model.ewemdb | 4.64 MB | Dev | Review structure |
| EwE6 Curonian Pelagic | EwE6_curonian_pelagic... | 2.74 MB | Calibrated | Study multistanza |
| Curonian Single-Species | Sp_Curonian_2012.ewemdb | 2.60 MB | Archive | Historical ref |

### B. Reference Ecosystems (for parameterization)

| Ecosystem | Model | Size | Notes |
|-----------|-------|------|-------|
| Gulf of Mexico | Georgia_Strait.EwEmdb | 2.90 MB | Temperate coastal |
| Black Sea | Black Sea 2004.ewemdb | 2.49 MB | Similar climate |
| Mediterranean | Various | Multiple | Mediterranean params |
| North Sea | Various | Multiple | Temperate regions |

### C. Tutorial/Learning Models

**Location:** C:\Users\DELL\OneDrive - ku.lt\ecopath workshop

Excellent starting points for understanding EwE functionality:
- Example ecosystems with complete parameterization
- Documented best practices
- Various ecosystem types (33 models)
- Multiple model complexities (simple → complex)

---

## V. Data Gaps & Recommendations for Case Study

### A. Identified Strengths ✓

1. **Excellent Curonian Lagoon temporal data**
   - 18+ years CPUE data (2001-2018)
   - 64+ years historical landings (1947-2011)
   - 123 years environmental forcing (river discharge)

2. **Multiple ecosystem types available**
   - Baltic Sea reference models
   - Temperate coastal examples (Gulf of Mexico)
   - Various model complexities

3. **Species-specific research**
   - Roach population dynamics
   - Thermal/salinity regime data
   - Discharge-productivity relationships

4. **Complete software framework**
   - 63 EwE models for reference
   - R statistical tools
   - Analysis templates

### B. Data Gaps ⚠️

1. **Species diet composition** - Limited stomach content data
2. **Recruitment dynamics** - Larval data sparse
3. **Migration/dispersal** - Movement data incomplete
4. **Genetic structure** - Not represented
5. **Non-fish species** - Zooplankton, phytoplankton data limited

### C. Recommendations for Curonian Lagoon Case Study

#### Phase 1: Data Assembly ✓ FEASIBLE
- [ ] Compile existing CPUE data (ready)
- [ ] Historical landings 1947-2011 (ready)
- [ ] Environmental forcing (discharge, temperature, salinity)
- [ ] Collect recent data 2018-present

#### Phase 2: Model Parameterization
- [ ] Review KM_Edgaras.ewemdb structure
- [ ] Use Baltic Sea reference models for trophic links
- [ ] Parameterize with Gulf of Mexico models as template
- [ ] Validate against historical landings

#### Phase 3: Calibration
- [ ] Fit Ecosim to CPUE time series (2001-2018)
- [ ] Validate against landings data
- [ ] Sensitivity analysis on key parameters

#### Phase 4: Scenarios
- [ ] Fishing pressure variations
- [ ] Environmental forcing (discharge, temperature)
- [ ] Climate change scenarios
- [ ] Restoration/conservation measures

---

## VI. File Inventory - Quick Reference

### EWE Models CSV
**Location:** EWE_MODELS_INVENTORY.csv
- 63 models total
- Columns: Filename, FullPath, SizeMB, EcosystemType, Region, Format, Directory, Notes
- Ready for filtering and analysis

### Fishery Data CSV
**Location:** FISHERY_DATA_INVENTORY.csv
- 253 datasets total
- Columns: Filename, FullPath, SizeMB, DataType, Region, SpeciesCovered, TimePeriod, FileFormat, Directory
- Includes Curonian Lagoon (113 files) and reference data

### This Summary
**Location:** EWE_DATABASE_SUMMARY.md
- Comprehensive analysis
- Recommendations for case study development

---

## VII. Next Steps

### For Immediate Use:
1. ✓ Review KM_Edgaras.ewemdb model structure
2. ✓ Compile CPUE_BAR.xlsx data for calibration
3. ✓ Extract historical landings from 1947-2011 series
4. ✓ Integrate Nemunas discharge as environmental forcing

### For Development:
1. Validate model against available CPUE data
2. Test scenarios with Baltic parametrization
3. Compare results against Gulf of Mexico templates
4. Document assumptions and data quality

### For Enhancement:
1. Collect post-2018 fishery data
2. Integrate zooplankton/phytoplankton data if available
3. Add climate scenarios
4. Develop management strategy evaluation

---

## VIII. Data Dictionary

### EcosystemType Codes
- **Baltic Sea** - Curonian Lagoon, Baltic region models
- **Gulf of Mexico** - Tampa Bay, North American templates
- **Black Sea** - European inland sea examples
- **Mediterranean** - Southern European models
- **Tutorial/Example** - Workshop, teaching, generic models
- **Unknown** - Requires manual review and classification

### DataType Codes (Fishery)
- **CPUE/Effort** - Catch per unit effort, fishing intensity
- **Landings/Catch** - Commercial harvest records
- **Biomass/Stock** - Population abundance estimates
- **Time Series** - Temporal dynamics (multiple variables)
- **Diet Data** - Stomach content, food web links
- **Life History** - Growth rates, recruitment, mortality
- **Fishery Data** - General category, multiple parameters

---

## Appendix: Full Directory Paths

`
Primary EwE Databases:
  C:\Users\DELL\OneDrive - ku.lt\EwE sample databases
  C:\Users\DELL\OneDrive - ku.lt\ECOPATH
  C:\Users\DELL\OneDrive - ku.lt\ecopath workshop

Curonian Lagoon Data (PRIMARY):
  C:\Users\DELL\OneDrive - ku.lt\EDGARAS
    - CPUE data: EDGARAS/CPUE_BAR.xlsx
    - Landings: EDGARAS/Sugavimai K_M 1947-2011.xls
    - Annual DB: EDGARAS/Kuršių marių duombazė/
    - Models: EDGARAS/OneDrive - ku.lt/KM_Edgaras.ewemdb
    - Scripts: EDGARAS/CPUE.R, MCY.R, etc.

Output Location:
  C:\Users\DELL\OneDrive - ku.lt\HORIZON_EUROPE\PyPath
    - EWE_MODELS_INVENTORY.csv
    - FISHERY_DATA_INVENTORY.csv
    - EWE_DATABASE_SUMMARY.md
`

---

**Document Status:** Final Summary
**Last Updated:** 2026-10-23 17:23:51
**Prepared for:** Curonian Lagoon Case Study Development
**Team:** PyPath Project

