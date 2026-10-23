# PEST Calibration for Multistanza Models

## Overview
This guide demonstrates how to calibrate Von Bertalanffy growth (K) and natural mortality (Z)
parameters for age-structured groups using the PEST framework.

## Model Structure
- **Demersal Fish**: 2 stanzas (juvenile K=0.15, Z=1.2; adult K=0.15, Z=0.6)
- **Predatory Fish**: 3 stanzas (juvenile K=0.12, Z=1.5; sub-adult K=0.12, Z=1.0; adult K=0.12, Z=0.5)

## Parameters to Calibrate
Total: 10 parameters (5 stanzas x 2 parameters each)
- Von Bertalanffy K: growth rate (yr⁻^1)
- Natural mortality Z: background mortality (yr⁻^1)

## Bounds
All parameters: plus-minus50% from baseline values

## Observations
- Type: Age-structured landings (catch by stanza and year)
- Count: 100 observations (5 stanzas x 20 years)
- Data: 100 records from synthetic_observations.csv

## Algorithm Options
1. **GLM (Gauss-Levenberg-Marquardt)**
   - Good for smooth objective functions
   - Faster convergence
   - Recommended for this problem

2. **Differential Evolution**
   - More robust to multiple minima
   - Higher computational cost
   - Better for complex landscapes

## Files Generated
- `calibration_parameters.csv`: Parameter definitions and bounds
- `synthetic_observations.csv`: Synthetic catch data
- `PEST_MULTISTANZA_GUIDE.md`: This guide

## Next Steps
1. Run GLM calibration with these parameters
2. Evaluate goodness-of-fit (SSR, R^2)
3. Examine parameter correlation matrix
4. Run Monte Carlo uncertainty quantification
5. Compare calibrated dynamics vs. baseline
