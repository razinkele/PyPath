# Nutrient System Quick Reference - PyPath 1B.5

## Import
`python
from pypath.core.nutrients import (
    NutrientElement, NutrientContent, NutrientExcretion,
    NutrientUptake, NutrientCycle, estimate_excretion_from_pb
)
`

## Quick Examples

### 1. Create Nutrient Content
`python
content = NutrientContent(
    group_name="Fish",
    nitrogen_percent=10.0,      # 0-20%
    phosphorus_percent=1.5,     # 0-5%
    carbon_percent=47.0,        # 40-55%
    ash_percent=8.0             # 0-20%
)
assert content.is_valid()  # Validate constraints
`

### 2. Set Excretion Rates
`python
# Option A: Direct specification
excretion = NutrientExcretion(
    group_name="Fish",
    n_excretion_rate=12.0,      # % consumption/day (0-50%)
    p_excretion_rate=2.0,       # % consumption/day (0-20%)
    fecal_n_fraction=0.4,       # 0-1
    fecal_p_fraction=0.6        # 0-1
)

# Option B: Estimate from P/B ratio
pb_ratio = 0.5
estimates = estimate_excretion_from_pb(pb_ratio, "consumer")
excretion = NutrientExcretion("Fish", **estimates)

# Calculate excretion
daily_n_excretion = excretion.calculate_excretion(100.0, "nitrogen")  # 12.0
`

### 3. Primary Producer Uptake
`python
uptake = NutrientUptake(
    group_name="Phytoplankton",
    n_uptake_rate=0.8,          # mg N / mg biomass / day
    p_uptake_rate=0.15,         # mg P / mg biomass / day
    half_saturation_n=8.0,      # mg/L (Km for N)
    half_saturation_p=1.0       # mg/L (Km for P)
)

# Monod kinetics: v = (Vmax * [S]) / (Km + [S])
uptake_rate = uptake.calculate_uptake(
    nutrient_level=10.0,
    rate=uptake.n_uptake_rate   # Returns 0.7407
)
`

### 4. Manage Nutrient Cycles
`python
cycle = NutrientCycle()

# Add parameters
cycle.add_content(content)
cycle.add_excretion(excretion)
cycle.add_uptake(uptake)

# Validate
is_valid, messages = cycle.validate()
if not is_valid:
    print(f"Errors: {messages}")

# Export to DataFrame
df = cycle.to_dataframe()
df.to_csv("nutrients.csv", index=False)

# Import from DataFrame
cycle2 = NutrientCycle.from_dataframe(df)

# Calculate mass balance
from pypath.core.params import create_rpath_params
rpath_params = create_rpath_params(...)
balance = cycle.calculate_mass_balance(rpath_params)
print(f"Total N biomass: {balance['total_n_biomass']} t")
print(f"Total P biomass: {balance['total_p_biomass']} t")
`

### 5. Nutrient Elements
`python
elem = NutrientElement.NITROGEN
print(elem.value)              # "nitrogen"
print(elem.molecular_weight)   # 14.01

for elem in NutrientElement:
    print(f"{elem.name}: {elem.molecular_weight} g/mol")
`

## DataFrame Format
`
      Group  N_percent  P_percent  C_percent  Ash_percent  N_excretion_rate  ...
0      Fish       11.0        1.5       47.0          8.0              15.0  ...
1 Phytoplankton 8.5        1.2       48.0          5.0               NaN  ...
`

## Validation Checks
- Nitrogen: 0-20% dry weight
- Phosphorus: 0-5% dry weight
- Carbon: 40-55% dry weight
- Ash: 0-20% dry weight
- Excretion rates: Realistic per-day percentages
- Fecal fractions: 0-1 (proportions)

## Error Handling
`python
try:
    content = NutrientContent("Bad", 50.0, 10.0, 20.0, 10.0)
except ValueError as e:
    print(e)  # "Bad: nitrogen_percent must be 0-20%, got 50.0"
`

## Constants
- N molecular weight: 14.01 g/mol
- P molecular weight: 30.97 g/mol
- C molecular weight: 12.01 g/mol

## Integration Points
- Use with RpathParams for biomass calculations
- Connect to Ecosim derivatives for nutrient fluxes
- Wire uptake into primary producer growth
- Link excretion outputs to detritus pools
