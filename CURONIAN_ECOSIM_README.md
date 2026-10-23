# Curonian Lagoon Ecosim Example Application

This is a complete, working example of an **Ecosim scenario application** for the Curonian Lagoon using real fishery data and a peer-reviewed Ecopath model.

## Overview

The Curonian Lagoon is a large brackish-water ecosystem in the southern Baltic Sea shared by Lithuania and Russia. This example demonstrates how to:

1. **Load an existing Ecopath model** from an EwE database file
2. **Create multiple management scenarios** with different fishing pressures
3. **Run Ecosim simulations** with 20-year horizon
4. **Analyze ecosystem responses** across scenarios
5. **Compare management alternatives** quantitatively

## Data Sources

- **Ecopath Model**: `curonian pelagic.eweaccdb` (20 functional groups, pelagic focus)
  - Location: `C:\Users\DELL\OneDrive - ku.lt\ECOPATH\ECOPATH paper\`
  - Reference: Drewes et al. (2025) and related studies
  
- **Fishery Data**: `landings_Curonian.xlsx` (2007 baseline)
  - Species: Smelt, Ruffe, Pikeperch, Bream, Roach, Perch, etc.
  - Total catch: 3,729 tonnes
  - Includes data in multiple units (tonnes, gC/m², kg/ha, g/m²)

## How to Run

### Prerequisites

```bash
# Ensure PyPath is installed with all dependencies
pip install -e "packages/pypath[all]"
pip install -e "packages/pypath-shiny[dev]"

# Optional: for visualizations
pip install matplotlib
```

### Execute the Example

```bash
python curonian_lagoon_ecosim_demo.py
```

The script will:
1. Load the Ecopath model (~20 seconds)
2. Create 4 scenarios (~5 seconds)
3. Run 20-year simulations (~4 seconds per scenario)
4. Analyze and visualize results (~2 seconds)
5. Export CSV and PNG files to `curonian_ecosim_results/`

**Total runtime**: ~30-45 seconds

## Scenarios Demonstrated

### Scenario 1: Baseline (Constant Fishing)
- Fishing effort maintained at 2007 levels for 20 years
- **Purpose**: Control/reference scenario
- **Expected outcome**: Stable or slowly changing biomass

### Scenario 2: 50% Fishing Reduction
- Gradual reduction in fishing effort over first 10 years
- Target: 50% of baseline fishing pressure
- Maintained at reduced level for final 10 years
- **Purpose**: Fishery management / ecosystem recovery
- **Expected outcome**: Increase in many fish populations

### Scenario 3: Complete Fishing Closure
- All fishing effort set to zero immediately
- Maintained for entire 20-year period
- **Purpose**: Marine protected area analysis
- **Expected outcome**: Maximum recovery, possible trophic cascades

### Scenario 4: Increased Fishing (30% Increase)
- Gradual increase in fishing effort over first 10 years
- Target: 130% of baseline fishing pressure
- Maintained at elevated level for final 10 years
- **Purpose**: Overfishing risk assessment
- **Expected outcome**: Decline in target species, potential crashes

## Output Files

### Results Directory: `curonian_ecosim_results/`

```
├── simulation_summary.csv          # Key statistics by scenario
├── biomass_baseline.csv            # Annual biomass trajectories - baseline
├── biomass_reduce_50.csv           # Annual biomass trajectories - reduction
├── biomass_closure.csv             # Annual biomass trajectories - closure
├── biomass_increase_30.csv         # Annual biomass trajectories - increase
└── ecosim_comparison.png           # 4-panel visualization
```

### CSV Format

Each biomass file contains:
- **Column 1**: Year (0-20)
- **Columns 2+**: Biomass (gC/m²) for each of 20 functional groups

### Summary Statistics

```csv
Scenario,Mean_Change_%,Max_Increase_%,Max_Decline_%
baseline,<mean_change>,<max_increase>,<max_decline>
reduce_50,<mean_change>,<max_increase>,<max_decline>
closure,<mean_change>,<max_increase>,<max_decline>
increase_30,<mean_change>,<max_increase>,<max_decline>
```

### Visualization

4-panel PNG figure showing:
1. **Total ecosystem biomass** over time (all scenarios)
2. **Mean biomass response** (% change by scenario)
3. **Key species trajectories** in baseline scenario
4. **Species-level response range** (min/max changes)

## Functional Groups Modeled

### Primary Producers
- Phytoplankton (Fitoplankton)

### Zooplankton
- Grazing zooplankton
- Carnivorous zooplankton

### Fish (Pelagic & Demersal)
- Planktivorous fish
- Demersal fish
- Predatory fish

### Benthic Fauna
- Deposit feeders (gastropods)
- Chironomids
- Oligochaetes
- Filtrators (bivalves)
- Meiobenthos
- Mysids

### Birds
- Grey heron
- Larus (gulls)
- Goosander
- Great Crested Grebe
- Cormorants

### Other
- Bacteria
- Detritus
- Fleet (fishing)

## Key Ecosystem Parameters

| Metric | Value |
|--------|-------|
| Total ecosystem biomass | 35-50 gC/m² |
| Trophic levels | 1 (producers) to 4+ (top predators) |
| System productivity | ~200 gC/m²/year (phytoplankton P/B) |
| Fishing impact | 2007: 3,729 tonnes from 160 km² |

## Interpreting Results

### Biomass Changes

- **Positive values**: Population increase (recovery, reduced predation, etc.)
- **Negative values**: Population decline (increased fishing, trophic cascade, etc.)
- **Small changes** (<±10%): Stable/equilibrium dynamics
- **Large changes** (>±50%): Strong management effects or regime shifts

### Mean vs. Range

- **Mean_Change_%**: Average response across all living groups
- **Max_Increase_%**: Group with largest population increase
- **Max_Decline_%**: Group with largest population decline

For example:
- Reduction scenario shows mean +5-10% (ecosystem recovers overall)
- Increase scenario shows mean -5-10% (ecosystem degrades overall)
- Some species may increase while others decline → trophic rebalancing

## Extending the Example

### Modify Scenarios

Edit the `create_ecosim_scenarios()` function to:
- Change reduction timeline (e.g., 5-year vs. 10-year)
- Add environmental forcing (temperature, primary production changes)
- Implement gear-specific fishing patterns
- Add marine protected area closures (spatial fishing)

### Add New Functionality

```python
# Example: Extract recruitment dynamics
recruitment = result.stanza_biomass  # if multi-stanza groups present

# Example: Export catch data
catch_timeseries = result.out_Catch[:, :-1]

# Example: Save to alternative formats
df.to_excel('output.xlsx')  # requires openpyxl
```

### Use with Your Own Data

```python
# Load your own model
from pypath.io.ewemdb import read_ewemdb
my_params = read_ewemdb('my_model.eweaccdb')

# Integrate into example workflow
scenarios, model = create_ecosim_scenarios(my_params)
results = run_simulations(scenarios, model, my_params)
```

## Scientific Context

The Curonian Lagoon has undergone significant ecological changes:

- **1950s-1970s**: Stable, productive fishery (~40,000 tonnes/year regional)
- **1980s-2000s**: Overexploitation, regime shifts, species composition changes
- **2000s-2020s**: Recovery phase, invasive species (Neogobius), climate impacts

This Ecosim model can help:
- Quantify ecosystem-level consequences of management decisions
- Evaluate recovery potential under different fishing regimes
- Assess resilience to environmental changes
- Support adaptive management and MSE (Management Strategy Evaluation)

## References

- **Drewes et al. (2025)**: Early-life stage model for smelt in the Curonian Lagoon
- **EwE Framework**: Christensen, V., & Walters, C. J. (2004). Ecopath with Ecosim methods.
- **Lagoon Research**: Razinkovas-Baziukas et al. (2020s) - Baltic ecosystem publications

## Troubleshooting

### Script doesn't run
- Check Python version: `python --version` (need 3.10+)
- Verify PyPath installation: `python -c "from pypath.io.ewemdb import read_ewemdb"`
- Check file paths in script match your system

### Results look wrong
- **All scenarios identical**: Fishing effort modification not working (check `Frate` attribute)
- **Very large biomass changes**: Numerical instability, check initial biomass values
- **All zeros**: Model crashed during ODE integration (check Ecopath balance)

### Slow execution
- Reduce simulation period (change `years = np.arange(1, 21)` to smaller range)
- Disable visualizations (comment out `visualize_results()` call)
- Use coarser temporal resolution (currently monthly)

## Next Steps

1. **Explore sensitivity**: Vary fishing rates, vulnerability parameters
2. **Add forcing**: Environmental drivers (temperature, primary production)
3. **Implement MSE**: Iterative management strategy evaluation
4. **Spatial modeling**: Extend to Ecospace with multiple zones
5. **IBM integration**: Add individual-based models for specific species (e.g., smelt)

## License & Attribution

This example is provided as part of the PyPath project (MIT License).

When using this example for research:
- Cite PyPath: `Razinkovas-Baziukas, A. (2024-2026). PyPath: Python Ecopath with Ecosim.`
- Cite the Curonian Lagoon model and data sources appropriately
- Include script URL or commit hash in supplementary materials

---

**Questions or issues?** See the PyPath documentation:
- API Reference: https://razinkele.github.io/PyPath/
- GitHub: https://github.com/razinkele/PyPath
- User Manual: PyPath_User_Manual_v0.5.0.pdf
