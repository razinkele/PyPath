# Example Model Advanced Features - Fixed

## Issue
When loading the example model from the Home page, the advanced features (multi-stanza, detritus fate, remarks) were not available because the example model only had basic Ecopath parameters.

## Solution
Updated `app/pages/home.py` to include complete advanced feature data structures in the example model.

## Changes Made

### 1. Multi-Stanza Groups (Age-Structured Populations)
The example model now includes a **Roundfish** multi-stanza group with two life stages:

- **JuvRoundfish1** (Juvenile stage)
  - Age range: 0-24 months
  - Higher mortality (Z=0.8)
  - High EE due to predation and maturation

- **AduRoundfish1** (Adult stage)
  - Age range: 24-120 months (10 years)
  - Lower mortality (Z=0.35)
  - Leading stanza (plus group)

**von Bertalanffy Growth Parameters:**
- K (growth rate): 0.4
- d (allometric parameter): 0.66667
- Wmat (maturity weight): 50.0 g

### 2. Remarks/Tooltips
Added sample remarks to demonstrate the tooltips feature:

| Group | Parameter | Remark |
|-------|-----------|--------|
| Seals | Biomass | Low biomass typical for top predator |
| Seals | EE | Low EE - top predator, little predation |
| JuvRoundfish1 | Biomass | Part of Roundfish multi-stanza group |
| JuvRoundfish1 | EE | High EE due to predation and growth to adult stage |
| AduRoundfish1 | Biomass | Leading stanza of Roundfish group |
| AduRoundfish1 | PB | Lower P/B for adult stage |
| Phytoplankton | Type | Primary producer (Type=1) |
| Phytoplankton | Biomass | Autotroph - no QB value needed |
| Detritus | Type | Detritus pool (Type=2) |
| Detritus | DetInput | Import of detritus from outside system |

### 3. Detritus Fate Import/Export
The example model includes:
- **Detritus** column in model DataFrame (for detritus fate routing)
- **DetInput** parameter (for tracking detritus imports from outside the system)

### 4. Pedigree Data
Empty pedigree DataFrame initialized (data quality tracking - can be populated later)

## Testing
All advanced features have been verified:
- ✅ Multi-stanza groups properly configured with 2 life stages
- ✅ von Bertalanffy growth parameters set
- ✅ 10 sample remarks added for tooltips
- ✅ Detritus fate columns present
- ✅ Pedigree structure initialized
- ✅ Diet matrix includes Import row

## Usage
When users click **"Load Example Model"** on the Home page, they can now:

1. **Navigate to "Multi-Stanza Groups"** under Advanced Features to see:
   - Roundfish stanza group with parameters
   - Growth curves visualization
   - Age-structured population dynamics

2. **View remarks/tooltips** in any data grid by hovering over cells with remarks

3. **Explore detritus fate** routing between groups

4. **Use the model for all advanced features** demonstrations and testing

## Files Modified
- `app/pages/home.py` - Updated `_load_example_model()` function

## Backward Compatibility
✅ The changes are fully backward compatible:
- Existing models without these features continue to work
- Data sync between pages works correctly
- Advanced feature pages handle both empty and populated data structures
