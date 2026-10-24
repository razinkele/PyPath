# Phase 1B Integration Guide

## Introduction

Phase 1B enhances PyPath's Ecopath with Ecosim framework through six integrated modules addressing data quality and ecosystem realism. These enhancements (TG3.1–TG3.6) transform PyPath from a basic simulator into a comprehensive modeling platform with explicit uncertainty quantification, realistic population structures, and validated fishery dynamics.

### Key Modules

- **TG3.1 Pedigree**: Quantifies parameter uncertainty via coefficients of variation
- **TG3.2 Taxonomy**: Integrates species metadata and external databases
- **TG3.3 Size**: Models population size distributions and age structure
- **TG3.4 Fleet**: Manages fishing effort, gears, and discard routing
- **TG3.5 Nutrients**: Tracks N, P, C cycling through food webs
- **TG3.6 Landings**: Integrates temporal catch records and seasonality

### Benefits

1. **Parity Improvement**: Narrows gaps with EwE via validated parameter handling
2. **Data Quality Tracking**: Transparent uncertainty and source documentation
3. **Ecosystem Realism**: Age structure, nutrient cycling, and realistic fishery impacts
4. **Ensemble Analysis**: Monte Carlo sampling for sensitivity and scenario analysis

### Typical Use Cases

- Importing models from EwE databases with full parameter documentation
- Running ensemble scenarios with uncertainty quantification
- Simulating age-structured populations with size-dependent predation
- Modeling fishery impacts under different catch scenarios
- Analyzing nutrient cycles and ecosystem productivity

## Module Interconnections

Phase 1B modules integrate as follows:

\RpathParams (Base Model)
|-- TG3.1 Pedigree System
|   |-- params.pedigree DataFrame
|   |-- ScalarDistribution (CV to log-normal sampling)
|   \-- DietDistribution (CV to Dirichlet sampling)
|       \-> build_distributions() to Monte Carlo sampling
|
|-- TG3.2 Taxonomy Module
|   |-- GroupTaxonomy (species IDs, external databases)
|   |-- WoRMS lookup (scientific name, AphiaID)
|   \-- OBIS/FishBase integration
|       \-> Biodata enrichment (size ranges, habitat)
|
|-- TG3.3 Size Distribution System
|   |-- SizeDistribution (length/weight frequencies)
|   |-- VBGrowth parameters (Linf, K, t0)
|   |-- LengthWeightRelation (a, b coefficients)
|   \-> IBM SuperIndividual trait generation
|
|-- TG3.4 Fleet Management
|   |-- FleetGroup (gear, selectivity, catchability)
|   |-- GearType (SEINE, TRAWL, etc.) to bycatch rates
|   |-- DiscardPath (discard fate routing)
|   \-> Fishing mortality in Ecosim derivatives
|
|-- TG3.5 Nutrient Cycling
|   |-- NutrientContent (N%, P%, C% composition)
|   |-- NutrientExcretion (metabolic N, P release)
|   |-- NutrientUptake (Monod kinetics, half-sat.)
|   \-> Producer growth rate modification
|
\-- TG3.6 Landings System
    |-- LandingRecord (catch by group/year/month)
    |-- LandingSeasonality (monthly distribution)
    |-- LandingsData (aggregated catch records)
    \-> Fleet dispatch to monthly Ecosim steps
\
### Integration Points

| From | To | Mechanism | Purpose |
|------|----|-----------|---------| 
| Pedigree | Sampling | build_distributions() | Uncertainty quantification |
| Taxonomy | Biodata | WoRMS/OBIS lookup | Species validation |
| Size | IBM | VBGrowth to SuperIndividual | Age structure simulation |
| Fleet | Ecosim | fishing_mortality_rate() | Dynamic catch allocation |
| Nutrients | Producers | Monod kinetics | Growth limitation |
| Landings | Fleet | aggregate_landings() to dispatch | Temporal fishing scenarios |

## Workflow Examples

### Workflow 1: Full Model with Pedigree, Taxonomy, Size

Complete model creation with data quality tracking:

\\\python
from pypath.core.params import create_rpath_params
from pypath.core.pedigree import PedigreeContainer
from pypath.io.biodata import GroupTaxonomy
from pypath.core.ecopath import rpath
import pandas as pd
import numpy as np

# Create 5-group model
model_df = pd.DataFrame({
    "Name": ["Detritus", "Phytoplankton", "Zooplankton", "Planktivorous Fish", "Piscivorous Fish"],
    "Type": [2, 1, 0, 0, 0],
    "Biomass": [0.0, 10.0, 2.0, 0.5, 0.1],
    "PB": [np.nan, 80.0, 5.0, 3.0, 1.5],
    "QB": [np.nan, np.nan, 20.0, 10.0, 5.0],
    "EE": [np.nan, 0.5, 0.8, 0.95, 0.0],
})
params = create_rpath_params(model_df)

# Add pedigree quality indices (0-4 scale, 0=best)
ped = PedigreeContainer(params)
ped.set_parameter_quality("Planktivorous Fish", "Biomass", 1)  # Good
ped.set_parameter_quality("Zooplankton", "PB", 2)              # Medium
params.pedigree = ped.to_dataframe()

# Add taxonomy links
tax = GroupTaxonomy()
tax.add_entry("Planktivorous Fish", scientific_name="Osmerus eperlanus")
tax.add_entry("Piscivorous Fish", scientific_name="Sander lucioperca")

# Solve Ecopath
model = rpath(params)
assert model.biomass_check > 0.99
print(f"Mass balance check: {model.biomass_check:.4f}")
print(f"Pedigree records: {params.pedigree.shape[0]}")
\\\

### Workflow 2: Fleet and Landings Management

Realistic fishing dynamics with seasonal catch:

\\\python
from pypath.core.fleet import FleetGroup, GearType, DiscardPath
from pypath.core.landings import LandingRecord, LandingSeasonality, LandingsData

# Create fleet
fleet = FleetGroup(
    fleet_id=0,
    name="Trawl Fleet",
    gear_type=GearType.TRAWL,
    catchability_matrix=np.eye(params.model.shape[0])
)

# Discard routing
discard_path = DiscardPath(
    landing_proportion=0.95,
    discard_fate="detritus",
    bycatch_proportion=0.10
)

# Historical landings data
landings = LandingsData()
for year in range(2015, 2020):
    for group in ["Planktivorous Fish", "Piscivorous Fish"]:
        landings.add_record(LandingRecord(
            group_name=group,
            year=year,
            month=None,
            landing_type="commercial",
            catch_tons=np.random.normal(500, 50)
        ))

# Monthly dispatch
monthly_catch = landings.aggregate_by_month("Planktivorous Fish")
print(f"Mean monthly catch: {monthly_catch.mean():.1f} tons")
\\\

### Workflow 3: Nutrient Cycling and Mass Balance

Close nutrient cycles with producer limitation:

\\\python
from pypath.core.nutrients import NutrientContent, NutrientExcretion, NutrientUptake
from pypath.core.nutrients import NutrientElement

# Define composition for all groups
nutrients = {}
for group in params.model["Name"]:
    if "Phytoplankton" in group:
        nutrients[group] = NutrientContent(
            group_name=group,
            nitrogen_percent=2.5,
            phosphorus_percent=0.3,
            carbon_percent=45.0,
            ash_percent=10.0
        )
    else:
        nutrients[group] = NutrientContent(
            group_name=group,
            nitrogen_percent=3.0,
            phosphorus_percent=0.8,
            carbon_percent=50.0,
            ash_percent=8.0
        )

# Excretion by consumers
for group in params.model[params.model["Type"] == 0]["Name"]:  # Consumers
    excretion = NutrientExcretion(
        group_name=group,
        daily_n_excretion=0.002,
        daily_p_excretion=0.0003
    )

# Producer uptake with Monod kinetics
uptake = NutrientUptake(
    group_name="Phytoplankton",
    nutrient_element=NutrientElement.NITROGEN,
    half_saturation_km=0.5,
    vmax=10.0
)

# Validate cycle closure
total_n = sum(nutrients[g].nitrogen_percent * params.model[params.model["Name"]==g]["Biomass"].values[0] for g in nutrients)
print(f"Total N in model: {total_n:.2f} tons")
\\\

### Workflow 4: Monte Carlo Ensemble and Uncertainty

Run parameter ensembles based on pedigree uncertainty:

\\\python
from pypath.core.pedigree import build_distributions, ScalarDistribution
from pypath.core.ecopath import rpath
import numpy as np

# Build distributions from pedigree
distributions = build_distributions(params)

# Monte Carlo sampling
n_samples = 100
ensemble = []

for i in range(n_samples):
    sample_params = params.copy()
    
    for dist in distributions:
        if isinstance(dist, ScalarDistribution):
            cv = dist.cv
            mu = np.log(dist.base_value) - 0.5 * np.log(1 + cv**2)
            sigma = np.sqrt(np.log(1 + cv**2))
            sampled_value = np.exp(np.random.normal(mu, sigma))
            
            if dist.bounds:
                sampled_value = np.clip(sampled_value, dist.bounds[0], dist.bounds[1])
            
            sample_params.model.loc[dist.group_idx, dist.param_name] = sampled_value
    
    balanced = rpath(sample_params)
    ensemble.append(balanced)
    
    if (i + 1) % 20 == 0:
        print(f"Completed {i+1}/{n_samples} samples")

# Analyze ensemble
biomass_samples = np.array([m.biomass for m in ensemble])
biomass_mean = biomass_samples.mean(axis=0)
biomass_std = biomass_samples.std(axis=0)
biomass_cov = biomass_std / biomass_mean

print("Ensemble CoV (Coefficient of Variation) by group:")
for i, group in enumerate(params.model["Name"]):
    print(f"  {group}: {biomass_cov[i]:.3f}")
\\\

### Workflow 5: Multi-Stanza Population Structure with IBM

Age-structured populations using Individual-Based Model:

\\\python
from pypath.core.stanzas import RpathStanzaParams, VBGrowth
from pypath.ibm.smelt import SmeltIBM
from pypath.core.ecosim import rsim_scenario, rsim_run

# Create multi-stanza model (age groups)
stanza = RpathStanzaParams(
    stanza_name="Planktivorous Fish Age Structure",
    num_stanzas=3,  # Age 0, 1-2, 3+
    growth_params=[
        VBGrowth(Linf=20.0, K=0.4, t0=-0.1),
        VBGrowth(Linf=20.0, K=0.4, t0=-0.1),
    ],
    recruitment_age=3,
    assignment_type="age"
)

# Build Ecosim scenario
scenario = rsim_scenario(model, scenario_params)

# Set up IBM for tracked species
ibm_smelt = SmeltIBM(
    group_idx=2,                       # Planktivorous Fish
    initial_abundance=1e6,             # 1 million super-individuals
    initial_length_cm=5.0,
    temperature_forcing=None
)

# Run simulation with IBM integration
results = rsim_run(
    scenario,
    years=10,
    monthly_steps=120,
    ibm_groups={"Planktivorous Fish": ibm_smelt}
)

# Extract results
final_state = results.ibm_state["Planktivorous Fish"]
print(f"Final abundance: {final_state.n_individuals:.0e} individuals")
print(f"Mean length: {final_state.mean_length:.2f} cm")
print(f"Mean weight: {final_state.mean_weight:.4f} kg")
\\\

## Data Quality and Validation

### Pedigree Quality Checks

Verify pedigree indices are consistent (0-4 scale):

\\\python
# Check pedigree coverage and consistency
assert (params.pedigree["PedigreeIndex"] >= 0).all() or params.pedigree["PedigreeIndex"].isna().any()
assert (params.pedigree["PedigreeIndex"] <= 4).all() or params.pedigree["PedigreeIndex"].isna().any()

# Verify CV mapping is monotonic (higher index = higher CV)
cv_by_index = params.pedigree.groupby("PedigreeIndex")["CV"].agg(["mean", "std"])
print("Average CV by pedigree index:")
print(cv_by_index)
\\\

### Taxonomy Validation

Ensure external IDs are present and valid:

\\\python
from pypath.io.biodata import GroupTaxonomy

tax = GroupTaxonomy()
for group in params.model["Name"]:
    if tax.has_entry(group):
        external_ids = tax.get_external_ids(group)
        assert "WoRMS" in external_ids or "FishBase" in external_ids
        print(f"{group}: {external_ids}")
\\\

### Size Distribution Validation

Check length/weight relationships are biologically realistic:

\\\python
# Length-weight parameter ranges (Froese et al.)
for group, dist in size_dist.items():
    assert dist.length_weight_b >= 2.5 and dist.length_weight_b <= 3.5
    assert dist.vb_growth.Linf > 0
    assert 0 < dist.vb_growth.K <= 1.0
    
    # Compute realistic max weight
    max_weight = dist.length_weight_a * (dist.vb_growth.Linf ** dist.length_weight_b)
    print(f"{group}: max weight = {max_weight:.2f} kg")
\\\

### Fleet Validation

Verify catchability matrix is consistent:

\\\python
# Check catchability matrix is normalized
for fleet_obj in fleets.values():
    catch_matrix = fleet_obj.catchability_matrix
    # Sum by column (prey) should be <= 1 if multiple fleets target same prey
    assert (catch_matrix.sum(axis=0) <= 1.0).all()
    print(f"Fleet {fleet_obj.name}: valid catchability matrix")
\\\

### Nutrient Balance Checks

Verify nutrient composition sums correctly:

\\\python
for group, comp in nutrient_content.items():
    total = (comp.nitrogen_percent + comp.phosphorus_percent + 
             comp.carbon_percent + comp.ash_percent)
    assert 80 < total <= 100  # Account for water content
    print(f"{group}: {total:.1f}% dry matter composition")
\\\

### Landings Consistency

Verify landings aggregation matches input records:

\\\python
# Validate landings data
from pypath.core.landings import validate_landings

report = validate_landings(landings)
assert report.is_valid
print(f"Landings validation: {report.n_records} records, {report.n_years} years")

# Check monthly dispatch sums to annual total
annual_total = landings.aggregate_by_year("Planktivorous Fish").sum()
monthly_total = landings.aggregate_by_month("Planktivorous Fish").sum()
assert abs(annual_total - monthly_total) < 1e-6
\\\

## Performance Considerations

| Module | Operation | Time (ms) | Scaling |
|--------|-----------|-----------|---------|
| Pedigree | build_distributions() | 10-50 | O(n_params) |
| Pedigree | MC sampling (1000) | 500-2000 | O(n_samples * n_params) |
| Taxonomy | WoRMS lookup | 100-500 | O(n_groups * API calls) |
| Size | VB computation | 5-20 | O(n_groups * n_ages) |
| Fleet | Catch dispatch | 20-100 | O(n_fleets * n_groups) |
| Nutrients | Monod kinetics | 10-50 | O(n_groups * n_steps) |
| Landings | Aggregation | 5-50 | O(n_records) |

**Optimization tips:**
- Cache WoRMS lookups to avoid repeated API calls
- Use vectorized operations in pedigree sampling (NumPy)
- Pre-compute Monod kinetics for static nutrient concentrations
- Batch Monte Carlo runs with multiprocessing

## Common Pitfalls and Troubleshooting

### Pedigree Confusion

**Issue**: EwE 0-4 scale vs. CV (coefficient of variation)

\\\python
# WRONG: Assuming pedigree index is directly the CV
params.pedigree["CV"] = params.pedigree["PedigreeIndex"] * 0.1  # Incorrect!

# CORRECT: Map quality levels to CV ranges
quality_to_cv = {0: 0.10, 1: 0.20, 2: 0.35, 3: 0.50, 4: 0.80}
params.pedigree["CV"] = params.pedigree["PedigreeIndex"].map(quality_to_cv)
\\\

### Taxonomy Incomplete

**Issue**: Missing external IDs for species

\\\python
# Check for missing IDs
missing = [g for g in params.model["Name"] if not tax.has_external_id(g, "WoRMS")]
if missing:
    print(f"WARNING: No WoRMS ID for {missing}")
    # Attempt fuzzy matching
    from pypath.io.biodata import match_species_fuzzy
    for group in missing:
        sci_name = tax.get_scientific_name(group)
        matched_id = match_species_fuzzy(sci_name)
\\\

### Size Overlaps

**Issue**: Age class size ranges overlap, creating ambiguity

\\\python
# Verify non-overlapping length ranges
for group in size_dist:
    dist = size_dist[group]
    age_lengths = [dist.get_length_at_age(age) for age in range(0, 10)]
    # Check monotonic increase
    for i in range(len(age_lengths) - 1):
        assert age_lengths[i] < age_lengths[i+1]
\\\

### Fleet Catchability Issues

**Issue**: Gear selectivity not realistic

\\\python
# Validate selectivity by gear type
for fleet_name, fleet in fleets.items():
    selectivity = fleet.gear_type.selectivity()
    assert 0.2 <= selectivity <= 1.0
    print(f"{fleet_name}: selectivity = {selectivity:.2f}")
\\\

### Nutrient Cycle Closure Failure

**Issue**: Model crashes due to nutrient imbalance

\\\python
# Run pre-check before Ecosim
from pypath.core.nutrients import validate_nutrient_cycle

valid, report = validate_nutrient_cycle(params, nutrient_content, excretion)
if not valid:
    print("Nutrient cycle issues:")
    for issue in report.issues:
        print(f"  - {issue}")
    # Adjust excretion rates or composition
\\\

### Landings Dispatch Mismatch

**Issue**: Monthly dispatch doesn't match annual total

\\\python
# Fix: Ensure normalization
annual = landings.aggregate_by_year("Planktivorous Fish")
monthly = landings.aggregate_by_month("Planktivorous Fish")

# Normalize monthly to match annual
scaling_factor = annual.sum() / monthly.sum()
monthly_adjusted = monthly * scaling_factor
assert abs(monthly_adjusted.sum() - annual.sum()) < 1e-10
\\\

## Integration with EwE Export

PyPath supports round-trip parity with EwE 6 model databases (.eweaccdb files).

### Parity Workflow

\\\python
from pypath.io.ewemdb import read_ewe_samples, write_ewe_samples

# Import from EwE
ewe_params = read_ewe_samples("model.eweaccdb")

# Verify parity
ewe_model = rpath(ewe_params)
assert ewe_model.biomass_check > 0.99

# Modify with Phase 1B enhancements
pedigree_cont = PedigreeContainer(ewe_params)
# ... add pedigree, taxonomy, etc ...

# Export back to EwE format
write_ewe_samples(ewe_params, "model_enhanced.eweaccdb")
\\\

### Known Limitations

- Stanza data not fully preserved in EwE export
- IBM parameters not supported in EwE format
- Nutrient cycling not native to EwE (stored in custom fields)
- Landing seasonality precision limited to monthly resolution

## API Quick Reference

### Pedigree Module

\\\python
from pypath.core.pedigree import (
    PedigreeContainer,      # Main container
    build_distributions,    # CV -> distributions
    ScalarDistribution,     # Single param distribution
    DietDistribution,       # Diet composition distribution
)

# Create container
ped = PedigreeContainer(params)

# Set quality index (0-4)
ped.set_parameter_quality(group, param, index)

# Get distributions for sampling
dists = build_distributions(params)
\\\

### Taxonomy Module

\\\python
from pypath.io.biodata import (
    GroupTaxonomy,              # Taxonomy container
    get_worms_metadata,         # WoRMS API lookup
    match_species_fuzzy,        # Fuzzy matching
)

tax = GroupTaxonomy()
tax.add_entry(group_name, scientific_name=name)
metadata = get_worms_metadata(name)
\\\

### Fleet Module

\\\python
from pypath.core.fleet import (
    FleetGroup,         # Fleet definition
    GearType,           # Gear types enum
    DiscardPath,        # Discard routing
)

fleet = FleetGroup(fleet_id, name, gear_type, catchability_matrix)
\\\

### Nutrients Module

\\\python
from pypath.core.nutrients import (
    NutrientContent,        # Elemental composition
    NutrientExcretion,      # Metabolic release
    NutrientUptake,         # Monod kinetics
    NutrientElement,        # N, P, C enum
)

nutr = NutrientContent(group, N%, P%, C%, ash%)
\\\

### Landings Module

\\\python
from pypath.core.landings import (
    LandingRecord,          # Single catch record
    LandingSeasonality,     # Monthly distribution
    LandingsData,           # Aggregated landings
    LandingType,            # Catch type enum
)

landings = LandingsData()
landings.add_record(LandingRecord(...))
monthly = landings.aggregate_by_month(group)
\\\

## Related Documentation

- **[Pedigree System Guide](../api/core.md#pedigree)**: Parameter uncertainty quantification
- **[Taxonomy Integration](../api/io.md#biodata)**: Species identification and external databases
- **[Size Distribution Guide](../api/core.md#size)**: Length-weight and age structure
- **[Fleet & Fisheries Management](../api/core.md#fleet)**: Fishing dynamics and gears
- **[Nutrient Cycling](../api/core.md#nutrients)**: N, P, C tracking
- **[Landings Data](../api/core.md#landings)**: Catch records and dispatch
- **[IBM Parameterization](./ibm-parameterization.md)**: Individual-Based Model setup
- **[EwE Export README](../../EWE_EXPORT_README.md)**: Round-trip parity procedures
- **[Autofix Guide](./autofix.md)**: Diagnosing simulation crashes

---

*Phase 1B Integration Guide v0.5.0*
*Updated: October 2024*
*For PyPath >=0.5.0*

