# Recruitment Forcing for Multistanza Models

## Overview
This guide demonstrates how to incorporate environmental drivers (temperature, 
primary productivity) into Ecosim simulations to modulate juvenile recruitment success.

## Environmental Forcing Functions

### Temperature Response
- Optimal temperature: 18 C (Curonian Lagoon summer average)
- Functional form: Gaussian curve exp(-0.2 * (T - T_opt)^2)
- Peak recruitment at optimal temperature
- Recruitment declines at cold (<10 C) and warm (>25 C) extremes

### Productivity Response
- Threshold: 0.3 (normalized 0-1 scale)
- Below threshold: baseline recruitment
- Above threshold: linear increase up to 50% enhancement
- Represents food availability for juvenile fish

### Combined Forcing
- Temperature weight: 40% (direct physiological effect)
- Productivity weight: 60% (indirect effect via food)
- Monthly timestep: 12 months/year

## Implementation

### Step 1: Create Time Series
- Temperature: seasonal cycle + inter-annual variability + trend
- Productivity: spring bloom pattern + seasonal signal + year-to-year variation

### Step 2: Define Response Functions
- Each function maps environmental variable to recruitment response (0-2)
- Functions are species/stanza-specific

### Step 3: Modify Juvenile Mortality
- Recruitment factor inverted to Z (natural mortality) multiplier
- Low recruitment factor (< 1.0) -> high Z -> low juvenile survival
- High recruitment factor (> 1.0) -> low Z -> high juvenile survival

## Application to Multistanza Groups

### Affected Groups
- Demersal fish juvenile (Z_baseline = 1.2 yr-1)
- Predatory fish juvenile (Z_baseline = 1.5 yr-1)
- Predatory fish sub-adult (Z_baseline = 1.0 yr-1)

### Ecosystem Effects
1. Warm years with high productivity: enhanced juvenile recruitment
2. Cold years with low productivity: suppressed juvenile recruitment
3. Cascading effects through predator-prey interactions

## Time Series Statistics

- Temperature range: 0-30 C (clipped at ecosystem extremes)
- Productivity range: 0.1-1.0 (normalized)
- Recruitment factor range: typically 0.5-1.5

## File Outputs

- `environmental_forcing.csv`: Monthly time series data
- `environmental_forcing.png`: Visualization of forcing and response
- `RECRUITMENT_FORCING_GUIDE.md`: This guide

## Future Enhancements

1. Couple to oxygen dynamics (hypoxic stress on juveniles)
2. Add nutrient/upwelling forcing for productivity
3. Include fishing pressure feedback on recruitment
4. Species-specific response curves (calibrate to data)
5. Spatial heterogeneity of forcing across habitat patches
