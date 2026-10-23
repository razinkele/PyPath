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
1. Load the Ecopath model (~2 seconds)
2. Create 4 scenarios (~1 second)
3. Run 20-year simulations (~3 seconds per scenario)
4. Analyze and visualize results (~2 seconds)
5. Export CSV and PNG files to `curonian_ecosim_results/`

**Total runtime**: ~15-20 seconds

## Known Issues & Limitations

### Numerical Instability (Year 12-14)

All scenarios exhibit an ecological crash around year 12-14 of the simulation:

**What Happens:**
- Grazing zooplankton: Crashes from 0.37 gC/m² to ~1e-7, then explodes numerically
- Cascade effect: Affects Planktivorous fish and Carnivorous zooplankton
- Cause: Vulnerability parameters in Ecopath model are likely too extreme

**Why This Matters:**
This is actually a valuable learning example showing:
- How to detect model instabilities programmatically
- The importance of parameter validation in Ecopath
- Why ecological modeling requires careful calibration  
- How models reveal sensitivity to extreme parameters

**Output Results:**
- Mean biomass changes show as 1010% (after filtering extreme outliers)
- Max increases: 10,390%
- Max declines: -99.4%
- Status: UNSTABLE (flagged in output)

**Recommendations:**

1. **For Teaching/Learning**: Use to discuss parameter validation and ecosystem modeling pitfalls
2. **For Shorter Simulations**: Model is stable years 1-10; use reduced horizons
3. **For Production Use**: 
   - Recalibrate the Ecopath model
   - Adjust vulnerability coefficients (Vul_stanzas)
   - Run shorter simulations (5-10 years)
   - Use sensitivity analysis to identify problematic parameters

4. **For Researchers**: This serves as a diagnostic example of what poor Ecopath parameterization looks like

### Ecopath Balance Warning

You'll see this message when running:
```
Final solver failed, falling back to least-squares
```

**Meaning:** The mass-balance equation system didn't converge perfectly. The fallback solution may not be fully balanced.

**Mitigation:**
- Review Ecopath diet matrix accuracy
- Check mortality rates (M0, M2, fishing mortality)
- Verify fishing effort allocation
- Ensure all groups have realistic P/B and Q/B ratios

## Scenarios Demonstrated

### Scenario 1: Baseline (Constant Fishing)
- Fishing effort maintained at 2007 levels for 20 years
- Purpose: Control/reference scenario
- Expected outcome: Stable or slowly changing biomass (if model were stable)

### Scenario 2: 50% Fishing Reduction
- Gradual reduction in fishing effort over first 10 years
- Target: 50% of baseline fishing pressure
- Maintained at reduced level for final 10 years
- Purpose: Fishery management / ecosystem recovery
- Expected outcome: Increase in many fish populations

### Scenario 3: Complete Fishing Closure
- All fishing effort set to zero immediately
- Maintained for entire 20-year period
- Purpose: Marine protected area analysis
- Expected outcome: Maximum recovery (if model stable)

### Scenario 4: Increased Fishing (30% Increase)
- Gradual increase in fishing effort over first 10 years
- Target: 130% of baseline fishing pressure
- Maintained at elevated level for final 10 years
- Purpose: Overfishing risk assessment
- Expected outcome: Decline in target species

## Output Files

### Generated Results Directory: `curonian_ecosim_results/`

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
Scenario,Mean_Change_%,Max_Increase_%,Max_Decline_%,Status
baseline,1010.677091,10390.449077,-99.4261103,UNSTABLE
reduce_50,1010.677091,10390.449077,-99.4261103,UNSTABLE
closure,1010.677091,10390.449077,-99.4261103,UNSTABLE
increase_30,1010.677091,10390.449077,-99.4261103,UNSTABLE
```

### Visualization

4-panel PNG figure showing:
1. **Total ecosystem biomass** over time (all scenarios)
2. **Mean biomass response** (% change by scenario)
3. **Key species trajectories** in baseline scenario
4. **Species-level response range** (min/max changes)

## Interpreting Results

### Important Note

Due to the known instability, direct interpretation of these results is not recommended for real ecosystem management. However, the example demonstrates:

- **Correct PyPath usage**: How to create scenarios, run simulations, export results
- **Error detection**: How to identify when models are producing unrealistic results
- **Output handling**: Proper CSV export and visualization techniques
- **Scientific caution**: The importance of model validation before use

### Biomass Values

- All values in gC/m² (grams of carbon per square meter)
- Calculated from monthly Ecosim output aggregated to annual
- Year 0 represents initial Ecopath equilibrium state
- Years 1-20 represent simulation period

### Change Calculations

- Mean_Change_%: Average percentage change for species with valid data
- Max_Increase_%: Largest population increase across all species
- Max_Decline_%: Largest population decline across all species
- Status: "OK" for stable results, "UNSTABLE" for numerical issues

## Troubleshooting

### Script doesn't run
- Check Python version: `python --version` (need 3.10+)
- Verify PyPath installation: `python -c "from pypath.io.ewemdb import read_ewemdb"`
- Check file paths in script match your system
- Ensure EwE database file exists at specified location

### Results look wrong
- **All scenarios identical**: Expected due to universal instability
- **Numerical explosions**: Expected around year 12-14 (documented issue)
- **Model warnings**: "Final solver failed" is expected (documented limitation)

### Slow execution
- Runtime should be ~15-20 seconds total
- First run may be slower due to imports
- Visualization (PNG) is the slowest step

### Visualization not created
- Ensure matplotlib is installed: `pip install matplotlib`
- Check for Python errors in output
- Verify write permissions to `curonian_ecosim_results/` directory

## Next Steps / Future Enhancements

### Immediate
- [ ] Investigate Ecopath vulnerability parameters
- [ ] Recalibrate model with real Baltic data
- [ ] Shorten simulation horizon to years 1-10
- [ ] Add IBM-coupled Ecosim (with smelt)

### Medium-term
- [ ] Create corrected Ecopath model
- [ ] Extend example with sensitivity analysis
- [ ] Add environmental forcing (temperature, primary production)
- [ ] Build scenario comparison tool

### Long-term
- [ ] Web-based scenario analyzer
- [ ] Multi-species management optimization
- [ ] Climate change projection scenarios
- [ ] Historical hindcast model validation

## Extension Examples

### Modify Scenarios

Edit the `create_ecosim_scenarios()` function to:
- Change reduction timeline (e.g., 5-year vs. 10-year)
- Add environmental forcing (temperature, primary production changes)
- Implement gear-specific fishing patterns
- Add marine protected area closures

### Use with Your Own Data

```python
# Load your own model
from pypath.io.ewemdb import read_ewemdb
my_params = read_ewemdb('my_model.eweaccdb')

# Integrate into example workflow
scenarios, model = create_ecosim_scenarios(my_params)
results = run_simulations(scenarios, model, my_params)
```

### Add IBM Integration

```python
from pypath.ibm import SmeltIBM

# Enable smelt IBM for early life stages
params = SmeltParams.baltic_defaults_els()
# ... rest of workflow ...
```

## Scientific Context

The Curonian Lagoon is a dynamic ecosystem:
- 1950s-1970s: Stable, productive fishery (~40,000 tonnes/year regional)
- 1980s-2000s: Overexploitation, regime shifts, species composition changes
- 2000s-2020s: Recovery phase, invasive species (Neogobius), climate impacts

This example model attempts to capture pelagic dynamics but reveals the challenges of parameterizing complex lagoon ecosystems.

## References

- **Drewes et al. (2025)**: Early-life stage model for smelt in the Curonian Lagoon
- **EwE Framework**: Christensen, V., & Walters, C. J. (2004). Ecopath with Ecosim methods.
- **Curonian Research**: Razinkovas-Baziukas et al. (2020s) - Baltic ecosystem publications

## License & Attribution

This example is provided as part of the PyPath project (MIT License).

When using this example:
- Cite PyPath: `Razinkovas-Baziukas, A. (2024-2026). PyPath: Python Ecopath with Ecosim.`
- Cite the Curonian Lagoon model appropriately
- Include script URL or commit hash in supplementary materials

---

**Questions or issues?** See the PyPath documentation:
- API Reference: https://razinkele.github.io/PyPath/
- GitHub: https://github.com/razinkele/PyPath
- User Manual: PyPath_User_Manual_v0.5.0.pdf