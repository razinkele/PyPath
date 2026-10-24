# Ensemble Usage Guide

## Overview

An **ensemble** in PyPath represents multiple runs of the Ecopath mass-balance model, each parameterized with different (sampled) values for key biological parameters. Rather than running a single "best estimate" model, an ensemble explores the space of plausible parameter combinations to understand how parameter uncertainty propagates through to model outputs (e.g., biomass, production rate, trophic level).

**Why Monte Carlo Sampling?**
Marine ecosystem parameters are rarely measured with perfect precision. Biomass estimates from surveys carry uncertainty; literature values for Production/Biomass ratios vary by population; diet composition can shift seasonally. Monte Carlo sampling (random draws from probability distributions) allows us to quantify this parameter uncertainty and ask: "Given what we know and don't know, what confidence intervals should we place on model predictions?"

**When to Use Ensembles:**
- Conducting uncertainty analysis before presenting model results to stakeholders
- Comparing models to observations with realistic confidence bands
- Identifying which parameters most influence ecosystem predictions (sensitivity analysis)
- Stress-testing model conclusions under plausible parameter variation
- Publishing scientific results with transparent uncertainty bounds

## Installation & Setup

**Prerequisites:**
PyPath ensemble functionality requires:
- numpy (≥ 1.20)
- pandas (≥ 1.2)
- scipy (for statistical distributions)

These are included in the core `pypath-ewe` installation.

**Quick Install:**
```bash
pip install pypath-ewe
```

**Imports:**
```python
# Core ensemble functions
from pypath.core.ensemble import (
    run_ensemble_rpath,
    get_ensemble_confidence_intervals,
    ensemble_statistics,
    EnsembleResult
)

# Create and manage samples
from pypath.core.params import create_rpath_params, RpathParams
from pypath.core.sampling import (
    SampleContainer,
    EcopathGroupSample,
    validate_samples
)

# Data loading and manipulation
import numpy as np
import pandas as pd
from pathlib import Path
```

**Quick Verification:**
```python
# Test that ensemble functions are available
from pypath.core.ensemble import run_ensemble_rpath
print("✓ PyPath ensemble module loaded successfully")
```

## Step-by-Step Example

This section walks through a complete ensemble run using the 12-group example coastal model included with PyPath.

### Step 1: Load Base Model

Start by loading the example model from CSV:

```python
from pypath.core.params import create_rpath_params
from pathlib import Path

# Path to the example model (included with pypath installation)
example_dir = Path(__file__).parent.parent / "example_model_data"
model_path = example_dir / "model.csv"
diet_path = example_dir / "diet.csv"

# Create base Ecopath parameters
params = create_rpath_params(
    model_csv=str(model_path),
    diet_csv=str(diet_path),
)

print(f"Loaded model with {len(params.model)} groups:")
print(params.model[["Group", "Type", "Biomass", "PB"]].to_string())
```

**Output:**
```
Loaded model with 12 groups:
                    Group Type  Biomass    PB
0          Phytoplankton    1     20.0 150.0
1             Macroalgae    1      5.0  12.0
2             Zooplankton    0      8.0  35.0
3             Meiobenthos    0      2.0   8.0
...
```

### Step 2: Define Sampling Parameters

Create uncertainty samples for key groups. This example applies 10% uncertainty (1 standard deviation) to three important groups:

```python
from pypath.core.sampling import SampleContainer, EcopathGroupSample
import numpy as np

# Set random seed for reproducibility
np.random.seed(42)

# Create container for 50 replicates
n_replicates = 50
container = SampleContainer(n_replicates=n_replicates, random_seed=42)

# Define groups to sample
groups_to_sample = ["Zooplankton", "Benthic invertebrates", "Demersal fish"]

# Sampling parameters: base value ± CV (coefficient of variation)
# CV = 0.1 means we sample with sigma = 0.1 * mean
sampling_cv = 0.1

for group_name in groups_to_sample:
    # Find group in model
    group_row = params.model[params.model["Group"] == group_name].iloc[0]
    
    # Extract base parameters
    biomass_base = group_row["Biomass"]
    pb_base = group_row["PB"]
    qb_base = group_row["QB"]
    ee_base = group_row["EE"]
    
    # Generate lognormal samples (typical for positive-valued ecological parameters)
    biomass_samples = np.random.lognormal(
        mean=np.log(biomass_base),
        sigma=sampling_cv,
        size=n_replicates
    )
    pb_samples = np.random.lognormal(
        mean=np.log(pb_base),
        sigma=sampling_cv,
        size=n_replicates
    )
    qb_samples = np.random.lognormal(
        mean=np.log(qb_base),
        sigma=sampling_cv,
        size=n_replicates
    )
    ee_samples = np.random.beta(
        a=ee_base * (1 - ee_base) / (sampling_cv**2) - 1,
        b=(1 - ee_base) * (1 - ee_base) / (sampling_cv**2) - 1,
        size=n_replicates
    ).clip(0, 1)
    
    # Create and add sample to container
    sample = EcopathGroupSample(
        group_name=group_name,
        biomass_samples=biomass_samples,
        pb_samples=pb_samples,
        qb_samples=qb_samples,
        ee_samples=ee_samples,
        biomass_se=sampling_cv,
        pb_se=sampling_cv,
    )
    container.add_group_sample(sample)

print(f"Created samples for {len(container.group_samples)} groups")
```

### Step 3: Create Sample Container

The container holds all sampling data and is passed directly to the ensemble runner:

```python
# Container is already created above with:
# container = SampleContainer(n_replicates=n_replicates, random_seed=42)

# Verify container contents
print(f"Replicates: {container.n_replicates}")
print(f"Groups sampled: {list(container.group_samples.keys())}")
print(f"  - Zooplankton: {container.group_samples['Zooplankton'].n_replicates} replicates")
```

### Step 4: Run Ensemble

Execute the ensemble run with 50 replicates:

```python
from pypath.core.ensemble import run_ensemble_rpath

# Run ensemble
result = run_ensemble_rpath(
    base_params=params,
    sample_container=container,
    eco_name="Example Coastal Ecosystem",
    eco_area=1.0,
    random_seed=42,
    verbose=True
)

print(f"✓ Ensemble complete: {result.n_replicates} replicates + base model")
print(f"  Total models run: {len(result.replicate_models)}")
```

### Step 5: Extract and Interpret Results

**Option A: Confidence Intervals as DataFrame**
```python
from pypath.core.ensemble import get_ensemble_confidence_intervals

# Get confidence intervals for all groups and parameters
ci_df = get_ensemble_confidence_intervals(result, alpha=0.05)
print("95% Confidence Intervals:")
print(ci_df.to_string())

# Save to CSV
ci_df.to_csv("ensemble_ci.csv", index=False)
```

**Output:**
```
        Group Parameter       LCI       Mean       UCI
0   Zooplankton   Biomass    6.54     8.12     9.89
1   Zooplankton        PB   31.50    34.80    38.10
2   Zooplankton        TL    2.15     2.30     2.40
3 Benthic inver Biomass    4.10     5.00     6.20
...
```

**Option B: Query Specific Groups**
```python
# Get confidence interval for a single group parameter
fish_biomass_ci = result.confidence_interval(
    group="Demersal fish",
    param="biomass",
    alpha=0.05
)
print(f"Demersal fish 95% CI: {fish_biomass_ci[0]:.3f} - {fish_biomass_ci[1]:.3f} t/km²")

# Get percentile distribution
percentiles = result.biomass_percentiles(
    group="Demersal fish",
    percentiles=(5, 25, 50, 75, 95)
)
print("Biomass percentiles:", percentiles)
```

**Option C: System-Level Statistics**
```python
from pypath.core.ensemble import ensemble_statistics

# Get system-level summary statistics
stats = ensemble_statistics(result)
print(f"Total system biomass: {stats['total_system_biomass_mean']:.2f} ± "
      f"{stats['total_system_biomass_std']:.2f} t/km²")
print(f"Biomass CV: {stats['total_system_biomass_cv']:.3f}")
print(f"Mean trophic level: {stats['mean_trophic_level_mean']:.2f}")
```

## Result Interpretation

### Understanding Confidence Intervals

A **95% confidence interval** (LCI, UCI) is an interval that contains the true parameter value with 95% probability, given our sampling assumptions. In an ensemble context:

- **LCI (Lower Confidence Interval)**: The 2.5th percentile of model outputs across all replicates
- **Mean**: The median or average of model outputs
- **UCI (Upper Confidence Interval)**: The 97.5th percentile of model outputs

**Example:**
If Demersal fish biomass has a 95% CI of (0.8, 1.5 t/km²), we interpret this as: "Given our parameter uncertainty, we are 95% confident the true Demersal fish biomass lies between 0.8 and 1.5 t/km²."

### Interpreting Percentiles

Percentiles describe where a value falls in the distribution:
- **2.5th percentile**: Lower bound of 95% CI (only 2.5% of values are smaller)
- **25th percentile**: Lower quartile (one quarter of values are smaller)
- **50th percentile**: Median (half the values are smaller, half larger)
- **75th percentile**: Upper quartile (three quarters of values are smaller)
- **97.5th percentile**: Upper bound of 95% CI

### What the Statistics Mean

**Coefficient of Variation (CV):**
- CV = (Standard Deviation) / (Mean)
- Dimensionless metric for comparing uncertainty across systems
- CV < 0.1: Low uncertainty
- CV 0.1–0.3: Moderate uncertainty
- CV > 0.3: High uncertainty

**Standard Error (reported in samples):**
- Documents the original data uncertainty (e.g., from survey sampling)
- Helpful for auditing data quality and traceability
- See `EcopathGroupSample.biomass_se`, `pb_se`, etc.

For more details, see the [Ensemble Integration Tutorial](ENSEMBLE_INTEGRATION_TUTORIAL.md) and API documentation.
