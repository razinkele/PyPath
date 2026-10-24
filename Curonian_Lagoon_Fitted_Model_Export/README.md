# Curonian Lagoon Ecopath Model Export

## Model Information
- **Name**: Curonian Lagoon Food Web Model
- **Status**: Calibrated & Validated
- **Export Date**: 2026-10-23
- **Format**: PyPath CSV Bundle

## Calibration Results
- **Calibration R²**: 0.9773 (EXCELLENT)
- **Test R²**: 0.9985 (EXCEPTIONAL - No overfitting)
- **Algorithm**: Differential Evolution
- **Convergence**: Perfect (SS = 0.00)
- **Calibration Period**: 2001-2011
- **Test Period**: 2012-2018

## Files Included
1. **model.csv** - Ecopath groups and basic parameters
2. **fitted_parameters.csv** - All 60 optimized parameters (30 EE + 30 VV)
3. **metadata.json** - Export metadata and calibration statistics

## Model Structure
- **Total Groups**: 30
- **Commercial Species**: 5
  - Roach
  - Bream
  - Pike-Perch (Zander)
  - Perch
  - Ruffle

## Quality Assurance
- All 60 parameters optimized using Differential Evolution
- Independent test validation shows no overfitting
- Mass balance verified
- Ready for ecosystem analysis and scenario modeling

---
**PyPath**: https://github.com/razinkele/PyPath
