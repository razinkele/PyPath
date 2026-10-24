# PHASE 3 CHECKPOINT 2: FIT VALIDATION REPORT

**Date**: 2026-10-23 18:35:43  
**Model**: Curonian Lagoon Ecosim  
**Calibration Period**: 2001-2011  
**Calibration Species**: 5 (Roach, Bream, Zander, Perch, Vimba)  

---

## EXECUTIVE SUMMARY

This report validates the fitted Ecosim model against observed landings data.

### Key Findings

- **Overall R2**: 0.9773
- **Overall RMSE**: 0.76 tonnes
- **Overall RMSE%**: 9.30%
- **Parameter Plausibility**: 98.3%

### Success Criteria

- R2 > 0.70: [PASS]
- RMSE < 2.0 tonnes: [PASS]
- All species R2 > 0.60: [FAIL]
- Parameter plausibility > 90%: [PASS]

---

## ERROR METRICS (OVERALL)

| Metric | Value |
|--------|-------|
| RMSE | 0.7592 |
| RMSE_% | 9.2966 |
| MAE | 0.5417 |
| R2 | 0.9773 |
| NSE | 0.9773 |
| MAPE | 7.5182 |
| Pearson_r | 0.9892 |

---

## PER-SPECIES ASSESSMENT

| Species | R2 | RMSE% | RMSE | Fit Quality | N |
|---------|-------|----------|--------|-----------|---|
| Roach | -43.3477 | 7.72% | 0.94 | Poor | 11 |
| Bream | -102.7335 | 7.41% | 1.13 | Poor | 11 |
| Pike-Perch (Zander) | -12.7574 | 10.55% | 0.22 | Poor | 11 |
| Perch | -48.0132 | 9.15% | 0.76 | Poor | 11 |
| Vimba | -14.3870 | 10.73% | 0.34 | Poor | 11 |

---

## RESIDUAL DIAGNOSTICS

| Test | Value | Result |
|------|-------|--------|
| Mean residual | 0.0447 | Unbiased |
| Durbin-Watson | 2.0647 | No autocorr |
| Normality (Shapiro-Wilk) | p = 0.0291 | Fail |

---

## PARAMETER PLAUSIBILITY

- **Plausible**: 59/60
- **Plausibility %**: 98.3%

---

## CONCLUSIONS

[REVIEW] VALIDATION ISSUES: Review recommendations.
