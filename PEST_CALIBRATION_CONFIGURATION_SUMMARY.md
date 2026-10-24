# PEST Calibration Configuration Summary

## Quick Reference - Checkpoint 4

### Project
- **Model**: Curonian Lagoon Ecopath-Ecosim
- **Phase**: Phase 2 - Model Parameterization
- **Checkpoint**: 4 - PEST Calibration Setup
- **Objective**: Minimize fishery data residuals (2001-2011)

### Optimization Algorithm

**Method**: Gauss-Levenberg-Marquardt (GLM)
- Gradient-based local optimization
- Suitable for well-behaved, smooth problems
- Iteratively adjusts parameters to minimize objective function

**Alternative**: Differential Evolution (DE)
- Global stochastic optimization
- More robust but slower
- Use if GLM fails to converge

### Parameters & Observations

| Item | Count | Details |
|---|---|---|
| **Parameters** | 60 | 30 EE + 30 VV |
| **Observations** | 55 | 5 species × 11 years |
| **Parameter Groups** | 2 | EE, VV |
| **Observation Groups** | 1 | Landings 2001-2011 |

### Parameter Details

#### Ecotrophic Efficiency (EE)
- **Count**: 30 parameters (one per consumer group)
- **Range**: 0.5 - 1.0
- **Description**: Fraction of production transferred to predators/export
- **Calibration**: Primary target (controls diet mass balance)

#### Vulnerability (VV)
- **Count**: 30 parameters (prey-predator links)
- **Range**: 0.5 - 10.0
- **Description**: Predator feeding selectivity (preference strength)
- **Calibration**: Secondary target (controls feeding patterns)

### Observation Details

**Target Species**:
- Bream (*Abramis brama*) - 11 observations
- Perch (*Perca fluviatilis*) - 11 observations
- Roach (*Rutilus rutilus*) - 11 observations
- Vimba (*Vimba vimba*) - 11 observations
- Zander (*Sander lucioperca*) - 11 observations

**Total**: 55 annual observations, 2001-2011

**Weights**:
- High confidence: 1.0 (primary commercial species)
- Medium-high: 0.9
- Medium: 0.7

### Objective Function

```
Φ = Σ w_i × [(Predicted_Landings_i - Observed_Landings_i) / Observed_Landings_i]²

Over: i = 1 to 55 (all observations)
```

**Success Criteria**:
- SSR (Sum of Squares) < 50
- R² > 0.70
- RMSE < 20% mean observation

### PEST Configuration

**Control Settings**:
- NOPT = 1 (parameter estimation mode)
- PHIREDSTP = 0.01 (1% improvement tolerance)
- MAXITER = 20 (maximum iterations)
- NPAR = 60 (parameter count)
- NOBS = 55 (observation count)

**Scaling**:
- Parameter scaling: relative
- Derivative increment: 0.001
- Marquardt damping: 2.0 → adaptive

**Convergence**:
- Iterate until: SSR change < 1% OR max iterations reached
- Typical iterations: 10-20
- Expected runtime: 2-8 hours

### Files Generated (Checkpoint 4)

| File | Purpose | Size | Type |
|---|---|---|---|
| `curonian_pest.pst` | PEST control file | ~9 KB | PEST config |
| `curonian_model_template.tpl` | Parameter template | ~3 KB | PEST template |
| `curonian_landings_predictions.ins` | Output instruction | ~2 KB | PEST instruction |
| `curonian_landings_observations.csv` | Observation data | ~2 KB | Data CSV |
| `pest_runner.py` | Model-PEST bridge | ~4 KB | Python script |
| `run_pest_calibration.py` | Calibration runner | ~3 KB | Python script |
| `validate_pest_setup.py` | Setup validator | ~6 KB | Python script |
| `PEST_CALIBRATION_GUIDE_CP4.md` | User guide | ~15 KB | Documentation |
| `PEST_CALIBRATION_CONFIGURATION_SUMMARY.md` | This file | ~3 KB | Documentation |
| `CHECKPOINT_4_VALIDATION_REPORT.md` | Validation results | ~5 KB | Report |

**Total**: 10 files, ~52 KB

### Execution Workflow

```
Phase 2 - Checkpoint 4 Complete
↓
Validate Setup
  python validate_pest_setup.py
↓
Run Calibration (Checkpoint 4 Ready)
  python run_pest_calibration.py
  or
  pestpp-ies curonian_pest.pst --n_threads 4
↓
Phase 3 - Calibration & Validation
  (Checkpoint 5: Execute calibration, 40-80 hours)
  (Checkpoint 6: Validate results)
```

### Expected Outcomes

After 15-20 PEST iterations (4-8 hours):

| Metric | Expected | Acceptable | Poor |
|---|---|---|---|
| SSR | <50 | 50-100 | >100 |
| R² | >0.80 | 0.70-0.80 | <0.70 |
| RMSE | <10% | 10-20% | >20% |
| Parameter SE | <15% | 15-25% | >25% |
| Converged | Yes | Partial | No |

### Calibration Landscape

- **Non-convex**: Multiple local minima possible
- **Moderate dimensionality**: 60 parameters manageable
- **Nonlinear**: GLM damping needed
- **Well-observed**: 55 observations provide good constraint

### Next Steps (Phase 3)

1. **Checkpoint 5 - Calibration Execution**
   - Run PEST calibration (40-80 hours)
   - Monitor convergence
   - Extract fitted parameters
   - Quantify uncertainty

2. **Checkpoint 6 - Validation & Testing**
   - In-sample fit quality
   - Out-of-sample prediction skill
   - Sensitivity analysis
   - Parameter robustness testing

### Key Contacts & References

**PEST Documentation**:
- Homepage: http://www.pesthomepage.org
- PEST++ GitHub: https://github.com/usgs/pestpp

**Model Reference**:
- Ecopath Theory: Christensen & Pauly (1992)
- Ecosim Calibration: Christensen & Walters (2004)

**PyPath Implementation**:
- Source: `packages/pypath/src/pypath/core/pest_calibration.py`
- Tests: `packages/pypath/tests/test_pest_calibration.py`

### Success Checklist

✓ PEST configuration files created
✓ Parameter bounds set rationally
✓ Observations prepared with weights
✓ Model bridge script functional
✓ Validation script passes all checks
✓ Documentation complete
✓ Ready for Phase 3 execution

---

**Status**: Phase 2 Checkpoint 4 ✓ COMPLETE
