# PEST Calibration Guide - Curonian Lagoon Model

## Overview

This guide explains the PEST (Parameter ESTimation Toolkit) calibration framework for the Curonian Lagoon Ecopath-Ecosim model. PEST is used to optimize model parameters to match observed fishery data.

## Table of Contents

1. [What is PEST?](#what-is-pest)
2. [Objective Function](#objective-function)
3. [Parameters Being Calibrated](#parameters)
4. [Observations](#observations)
5. [File Descriptions](#files)
6. [Running PEST](#running)
7. [Interpreting Results](#results)
8. [Troubleshooting](#troubleshooting)

## What is PEST? {#what-is-pest}

PEST is professional-grade parameter estimation software. It:

- **Adjusts model parameters** to minimize differences between model predictions and observations
- **Uses optimization algorithms** (Gauss-Levenberg-Marquardt, Differential Evolution)
- **Quantifies uncertainty** in fitted parameters
- **Analyzes parameter sensitivity** and interactions

### How PEST Works

```
1. Initial Setup
   ├─ Define parameters (bounds, initial values)
   ├─ Specify observations (data to match)
   └─ Set up model as "black box" callable from PEST

2. Iterative Optimization Loop
   ├─ Run model with current parameters
   ├─ Compare predictions to observations
   ├─ Calculate residuals (prediction error)
   ├─ Estimate parameter sensitivities (Jacobian matrix)
   ├─ Compute parameter updates (solve normal equations)
   └─ Repeat until convergence

3. Results & Analysis
   ├─ Export best-fit parameters
   ├─ Quantify parameter uncertainty (covariance)
   └─ Generate calibration report
```

## Objective Function {#objective-function}

**Goal**: Minimize weighted sum of squared residuals

```
Objective Function (Φ) = Σ_i w_i × (pred_i - obs_i)²

Where:
  - pred_i = Model prediction for observation i
  - obs_i  = Observed value for observation i
  - w_i    = Weight for observation i

Calibration period: 2001-2011 (11 years)
Target species: Bream, Perch, Roach, Vimba, Zander (5 × 11 = 55 observations)
```

### Weighting Strategy

Observation weights reflect data quality and confidence:

| Data Source | Confidence | Weight |
|---|---|---|
| Primary commercial landings | High | 1.0 |
| Validated catch data | Medium-High | 0.9 |
| Estimated from proportions | Medium | 0.7 |
| Interpolated/uncertain | Low | 0.3 |

Higher weights pull the fit toward high-confidence data points.

### Success Metrics

- **Sum of Squared Residuals (SSR)** < 50 (good fit)
- **R² (coefficient of determination)** > 0.70 (acceptable)
- **RMSE (root mean squared error)** < 20% of mean observation

## Parameters Being Calibrated {#parameters}

### Parameter Types

#### 1. Ecotrophic Efficiency (EE)
- **Meaning**: Fraction of production consumed (or exported)
- **Range**: 0.5 - 1.0
- **Species**: All consumer groups
- **Effect**: Controls total flux through predator-prey interactions

#### 2. Vulnerability (VV)
- **Meaning**: Relative predator-prey feeding link strength
- **Range**: 0.5 - 10.0
- **Links**: Between predators and prey species
- **Effect**: Controls prey selectivity and feeding patterns

### Parameter Bounds

Bounds are set based on:
- Biological plausibility (not extreme values)
- Expert knowledge from FishBase and literature
- Mass balance constraints in Ecopath
- Ecosystem characteristics (e.g., oligotrophic vs eutrophic)

**Total calibration parameters**: ~60

### Baseline Values

Initial parameter values come from:
1. FishBase (default P/B, Q/B ratios)
2. Ecopath expert estimates
3. Literature on similar systems
4. Phase 1 parameter documentation

## Observations {#observations}

### Observation Data

**File**: `curonian_landings_observations.csv`

Format:
```
year,species_code,species_name,observed_landings,weight,obs_name
2001,BREAM,Bream,15.2,1.0,obs_BREAM_2001
2001,PERCH,Perch,8.3,1.0,obs_PERCH_2001
...
2011,ZANDER,Pike-Perch (Zander),2.1,0.9,obs_ZANDER_2011
```

### Total Observations

- **Years**: 2001-2011 (11 years)
- **Species**: 5 (Bream, Perch, Roach, Vimba, Zander)
- **Total**: 55 observations

### Data Quality Flags

Observations marked with:
- `weight = 1.0` → High confidence (validated)
- `weight = 0.9` → Medium-high confidence
- `weight = 0.7` → Medium confidence (estimated)

Lower-confidence data still included to maintain temporal continuity.

## File Descriptions {#files}

### PEST Configuration Files

| File | Purpose | Created |
|---|---|---|
| `curonian_pest.pst` | Main PEST control file | ✓ Checkpoint 4 |
| `curonian_model_template.tpl` | Parameter template for injection | ✓ Checkpoint 4 |
| `curonian_landings_predictions.ins` | Instruction for output extraction | ✓ Checkpoint 4 |

### Data Files

| File | Purpose | Created |
|---|---|---|
| `curonian_landings_observations.csv` | Observed landings 2001-2011 | ✓ Checkpoint 4 |
| `curonian_model_groups_35.csv` | Model structure (35 groups) | ✓ Checkpoint 2 |
| `PEST_CALIBRATION_RANGES.csv` | Parameter bounds and scaling | ✓ Checkpoint 3 |

### Python Scripts

| Script | Purpose | Created |
|---|---|---|
| `pest_runner.py` | Bridge between PEST and PyPath | ✓ Checkpoint 4 |
| `run_pest_calibration.py` | Execute PEST calibration | ✓ Checkpoint 4 |
| `validate_pest_setup.py` | Pre-calibration validation | ✓ Checkpoint 4 |

## Running PEST {#running}

### Prerequisites

1. **Install PEST++** (modern free implementation):
   ```bash
   # Windows
   choco install pest-ies
   
   # Or download from https://github.com/usgs/pestpp/releases
   ```

2. **Verify installation**:
   ```bash
   pestpp-ies --version
   # or
   pest --version
   ```

3. **Validate setup**:
   ```bash
   python validate_pest_setup.py
   # Should show: "All PEST files valid and ready for calibration!"
   ```

### Step-by-Step Execution

#### 1. Validate Setup
```bash
python validate_pest_setup.py
```

Expected output:
```
✓ .pst file exists
✓ Contains control data section
✓ Contains parameter data section
✓ Contains observation data section
✓ .tpl file exists
...
✓ All PEST files valid and ready for calibration!
```

#### 2. Run PEST Calibration
```bash
python run_pest_calibration.py
```

Or directly with PEST++:
```bash
pestpp-ies curonian_pest.pst --n_threads 4
```

#### 3. Monitor Progress

PEST creates several output files during calibration:
- `curonian_pest.rec` - Record file (main output log)
- `curonian_pest.rst` - Restart file (iteration checkpoints)
- `curonian_pest0.pai` - Parameter iteration file

### Expected Runtime

- **PEST setup**: < 1 minute
- **Calibration (20 iterations)**: 2-8 hours (depending on model complexity)
- **Typical convergence**: 10-15 iterations

## Interpreting Results {#results}

### Output Files

After calibration completes, review:

#### `curonian_pest.rec` - Record File
Contains iteration history and convergence information.

Key sections:
```
PARAMETER ESTIMATION - CASE ID: curonian_pest

ITERATION: 0
  Parameters:        Current value
  VV_Roach:          2.0000
  EE_Bream:          0.9000
  ...

  Measurement residuals:
  obs_BREAM_2001:    0.123
  obs_PERCH_2001:    -0.045
  ...

  Sum of squared weighted residuals: 45.234

ITERATION: 1
  Sum of squared weighted residuals: 42.891  <- Improved!
  ...
```

#### `fitted_parameters.csv` - Best-Fit Results
```
parameter_name,fitted_value,initial_value,uncertainty
VV_Roach_Bream,1.850,2.000,0.234
EE_Bream,0.875,0.900,0.045
...
```

### Convergence Indicators

**Good convergence**:
- SSR decreases consistently
- SSR change < 1% per iteration
- Converges in 10-20 iterations
- R² > 0.70

**Poor convergence**:
- SSR oscillates or increases
- Slow decrease in SSR
- Fails to converge after 50 iterations
- R² < 0.50

### Parameter Uncertainty

Each fitted parameter has an estimated uncertainty (standard error):

```
Parameter: EE_Bream
Fitted value: 0.875
Standard error: 0.045  <- 95% CI: 0.787 - 0.963
Relative error: 5.1%   <- Good fit if < 10%
```

High uncertainty (> 20%) indicates:
- Parameter is poorly constrained
- More/better observations needed
- Parameter might be non-unique

## Troubleshooting {#troubleshooting}

### "No PEST found in PATH"

**Problem**: pestpp-ies or pest executable not found

**Solution**:
1. Install PEST++ from https://github.com/usgs/pestpp
2. Add installation directory to system PATH
3. Verify: `pestpp-ies --version`

### "Sum of Squares increasing"

**Problem**: Calibration diverging instead of converging

**Possible causes**:
- Initial parameters too far from optimum
- Model too nonlinear for local optimization
- Incorrect observation weights
- Parameter bounds too restrictive

**Solutions**:
- Adjust initial parameter values manually
- Try differential evolution algorithm (slower but more robust)
- Review observation weight assignments
- Loosen parameter bounds slightly

### "Calibration stops before convergence"

**Problem**: PEST stops after few iterations

**Possible causes**:
- Max iteration limit reached
- Parameter bounds violated
- Model run failing silently

**Solutions**:
1. Check `curonian_pest.rec` for error messages
2. Increase max iterations in `curonian_pest.pst`
3. Verify `pest_runner.py` runs successfully:
   ```bash
   python pest_runner.py
   ```

### "Fitted parameters seem unrealistic"

**Problem**: Calibrated values outside biological range

**Causes**:
- Observations contain systematic bias
- Model structure fundamentally different from real system
- Parameter identifiability problem

**Solutions**:
1. Review observation data for errors
2. Add additional constraints/priors
3. Use parameter correlation analysis to detect co-dependencies
4. Consider fixing parameters to literature values

## Best Practices

1. **Start Simple**: Calibrate key parameters first, then expand
2. **Monitor Progress**: Check `.rec` file regularly
3. **Use Sensible Bounds**: Based on literature and biology
4. **Weight Observations**: More weight to high-confidence data
5. **Validate Results**: Out-of-sample testing (Phase 3)
6. **Document Assumptions**: Record why each parameter was calibrated

## References

- PEST Documentation: http://www.pesthomepage.org
- PEST++ GitHub: https://github.com/usgs/pestpp
- Calibration Theory: Box & Tiao (1973), Tarantola (2005)
- Ecosim Applications: Christensen & Walters (2004)

## Phase 2 Checkpoint 4 Summary

✓ **All PEST files created and validated**
✓ **Objective function defined**  
✓ **60 parameters with bounds set**
✓ **55 observations from 2001-2011**
✓ **Python bridge script ready**
✓ **Documentation complete**

**Next Step**: Phase 3 - Execute calibration and validation (Weeks 9-12)
