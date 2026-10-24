# Ensemble Integration Tutorial

## Full Workflow Description

This tutorial shows how to integrate Ecopath ensemble runs into a complete data analysis pipeline, from loading an EwE database through exporting results for publication.

### The Complete Pipeline

**Phase 1: Data Preparation**
Load an Ecopath with Ecosim (EwE) model from an `.eweaccdb` database file. Extract parameter uncertainty information (if available) or define parametric uncertainty distributions based on literature. Create bootstrap samples representing plausible parameter combinations.

**Phase 2: Sampling Definition**
Convert base parameters and uncertainty estimates into a `SampleContainer` with `n_replicates` bootstrap samples for each group parameter (Biomass, P/B, Q/B, Ecotrophic Efficiency). Optionally sample diet composition and catch/landing data if uncertainty exists in these dimensions.

**Phase 3: Ensemble Execution**
Run `run_ensemble_rpath()` to execute the mass-balance solver for the base model plus each replicate, generating a full ensemble of balanced Rpath models. The function handles parameter substitution and validation automatically.

**Phase 4: Results Analysis**
Extract confidence intervals, percentiles, and summary statistics using convenience functions (`get_ensemble_confidence_intervals()`, `ensemble_statistics()`). Compare base model predictions to ensemble confidence bands to assess parameter sensitivity.

**Phase 5: Export & Visualization**
Save results to CSV for downstream analysis. Generate publication-quality plots showing uncertainty bands, percentile distributions, and group-level comparisons.

## Complete Code Example

This working example demonstrates the full pipeline with error handling and common patterns:

```python
import numpy as np
import pandas as pd
import logging
from pathlib import Path
from pypath.core.params import create_rpath_params
from pypath.core.sampling import (
    SampleContainer,
    EcopathGroupSample,
)
from pypath.core.ensemble import (
    run_ensemble_rpath,
    get_ensemble_confidence_intervals,
    ensemble_statistics,
)

# Setup logging for diagnostic output
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# --- Phase 1: Load Base Model ---
def load_model(model_csv, diet_csv):
    """Load Ecopath model from CSV files."""
    try:
        params = create_rpath_params(
            model_csv=model_csv,
            diet_csv=diet_csv,
        )
        logger.info(f"Loaded model with {len(params.model)} groups")
        return params
    except Exception as e:
        logger.error(f"Failed to load model: {e}")
        raise

# --- Phase 2: Create Sampling ---
def create_sample_container(params, n_replicates=100, uncertainty_cv=0.15):
    """Create bootstrap samples with specified uncertainty."""
    container = SampleContainer(n_replicates=n_replicates, random_seed=42)
    
    for idx, row in params.model.iterrows():
        group_name = row["Group"]
        
        # Only sample consumers (Type 0)
        if row["Type"] != 0:
            continue
        
        try:
            # Create lognormal samples
            biomass = np.random.lognormal(
                mean=np.log(row["Biomass"]),
                sigma=uncertainty_cv,
                size=n_replicates
            )
            pb = np.random.lognormal(
                mean=np.log(row["PB"]),
                sigma=uncertainty_cv,
                size=n_replicates
            )
            qb = np.random.lognormal(
                mean=np.log(row["QB"]),
                sigma=uncertainty_cv,
                size=n_replicates
            )
            ee = np.random.beta(
                a=row["EE"] * (1 - row["EE"]) / (uncertainty_cv**2) - 1,
                b=(1 - row["EE"]) * (1 - row["EE"]) / (uncertainty_cv**2) - 1,
                size=n_replicates
            ).clip(0, 1)
            
            sample = EcopathGroupSample(
                group_name=group_name,
                biomass_samples=biomass,
                pb_samples=pb,
                qb_samples=qb,
                ee_samples=ee,
                biomass_se=uncertainty_cv,
                pb_se=uncertainty_cv,
            )
            container.add_group_sample(sample)
            
        except Exception as e:
            logger.warning(f"Skipped sampling for {group_name}: {e}")
            continue
    
    logger.info(f"Created samples for {len(container.group_samples)} groups")
    return container

# --- Phase 3: Run Ensemble ---
def run_ensemble(params, container, name="Test Ecosystem"):
    """Execute ensemble run with error handling."""
    try:
        result = run_ensemble_rpath(
            base_params=params,
            sample_container=container,
            eco_name=name,
            eco_area=1.0,
            random_seed=42,
            verbose=True
        )
        logger.info(f"✓ Ensemble complete: {result.n_replicates} replicates")
        return result
    except ValueError as e:
        logger.error(f"Ensemble failed: {e}")
        raise

# --- Phase 4: Extract Results ---
def export_results(result, output_dir="."):
    """Extract and export ensemble results."""
    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True)
    
    # Confidence intervals
    ci_df = get_ensemble_confidence_intervals(result, alpha=0.05)
    ci_file = output_dir / "ensemble_ci.csv"
    ci_df.to_csv(ci_file, index=False)
    logger.info(f"Saved confidence intervals to {ci_file}")
    
    # Summary table
    summary = result.summary_table()
    summary_file = output_dir / "ensemble_summary.csv"
    summary.to_csv(summary_file, index=False)
    logger.info(f"Saved summary to {summary_file}")
    
    # System statistics
    stats = ensemble_statistics(result)
    stats_file = output_dir / "ensemble_stats.txt"
    with open(stats_file, "w") as f:
        f.write("System-Level Statistics\n")
        f.write("========================\n")
        f.write(f"Total Biomass: {stats['total_system_biomass_mean']:.2f} ± "
                f"{stats['total_system_biomass_std']:.2f} t/km²\n")
        f.write(f"Biomass CV: {stats['total_system_biomass_cv']:.3f}\n")
        f.write(f"Mean Trophic Level: {stats['mean_trophic_level_mean']:.2f}\n")
    logger.info(f"Saved stats to {stats_file}")
    
    return ci_file, summary_file, stats_file

# --- Phase 5: Visualization (Optional) ---
def plot_results(result, output_dir="."):
    """Create visualization plots (requires matplotlib)."""
    import matplotlib.pyplot as plt
    
    output_dir = Path(output_dir)
    
    # Plot confidence intervals
    ci_df = get_ensemble_confidence_intervals(result)
    biomass_ci = ci_df[ci_df["Parameter"] == "Biomass"].copy()
    biomass_ci = biomass_ci.sort_values("Mean", ascending=False)
    
    fig, ax = plt.subplots(figsize=(12, 6))
    groups = range(len(biomass_ci))
    ax.errorbar(
        groups,
        biomass_ci["Mean"],
        yerr=[
            biomass_ci["Mean"] - biomass_ci["LCI"],
            biomass_ci["UCI"] - biomass_ci["Mean"]
        ],
        fmt="o", capsize=5, markersize=8, label="95% CI"
    )
    ax.set_xticks(groups)
    ax.set_xticklabels(biomass_ci["Group"], rotation=45, ha="right")
    ax.set_ylabel("Biomass (t/km²)")
    ax.set_title("Ensemble Biomass Confidence Intervals")
    ax.legend()
    ax.grid(alpha=0.3)
    plt.tight_layout()
    
    output_file = output_dir / "ensemble_ci_plot.png"
    plt.savefig(output_file, dpi=300)
    logger.info(f"Saved plot to {output_file}")
    plt.close()

# --- Main Execution ---
if __name__ == "__main__":
    # Set paths (adjust as needed)
    model_file = "model.csv"
    diet_file = "diet.csv"
    output_dir = "./ensemble_results"
    
    # Run full pipeline
    params = load_model(model_file, diet_file)
    container = create_sample_container(params, n_replicates=50)
    result = run_ensemble(params, container)
    export_results(result, output_dir)
    plot_results(result, output_dir)
    
    logger.info("✓ Full ensemble workflow complete!")
```

### Error Handling Patterns

**Problem: Model doesn't balance for some replicates**

```python
try:
    result = run_ensemble_rpath(params, container)
except ValueError as e:
    logger.error(f"Replicate failed to balance: {e}")
    # Reduce uncertainty or check input data
    container_reduced = create_sample_container(params, uncertainty_cv=0.10)
    result = run_ensemble_rpath(params, container_reduced)
```

**Problem: Inconsistent replicate counts**

```python
# Ensure all samples have matching n_replicates
if not all(s.n_replicates == container.n_replicates 
           for s in container.group_samples.values()):
    raise ValueError("Sample replicate counts don't match!")
```

**Problem: Memory pressure with large ensembles**

```python
# Use smaller batches and aggregate results
batch_size = 25
all_results = []
for batch_start in range(0, n_replicates, batch_size):
    batch_end = min(batch_start + batch_size, n_replicates)
    batch_container = SampleContainer(n_replicates=batch_end - batch_start)
    # ... add samples for this batch ...
    batch_result = run_ensemble_rpath(params, batch_container)
    all_results.append(batch_result)
```

## Common Pitfalls & Troubleshooting

### Sampling Parameter Validation Errors

**Error:** "Sample has X replicates, container expects Y"

**Cause:** All samples added to a `SampleContainer` must have identical `n_replicates`. This ensures ensemble runs complete.

**Solution:** Check that all `EcopathGroupSample` and `EcopathDietCompSample` objects have matching lengths:
```python
n_reps = container.n_replicates
for sample in container.group_samples.values():
    assert sample.n_replicates == n_reps, f"Mismatch: {sample.n_replicates} vs {n_reps}"
```

### Ensemble Failure Modes

**Problem:** "Replicate N failed to balance"

This occurs when the mass-balance equations (e.g., diet intake must equal Q/B × Biomass) cannot be satisfied with a given parameter set. Common causes:
- Sampling parameters outside biologically plausible ranges
- Invalid diet matrix (rows don't sum to 1.0)
- Ecotrophic Efficiency > 1.0 (consumption exceeds production)

**Solution:** Reduce sampling uncertainty, inspect base model for structural issues, or validate diet matrix:
```python
# Check diet matrix sums
diet_sums = params.diet.groupby("Predator").size()
if (diet_sums > 1.0).any():
    logger.warning("Some diet rows exceed 1.0 - normalizing")
    params.diet = params.diet / params.diet.groupby("Predator").transform("sum")
```

### Performance Considerations

**Slow ensemble runs:**
- Each replicate calls `rpath()` (the full Ecopath solver)
- Typical time per replicate: 10-100 ms depending on group count
- 100 replicates on a 50-group model: ~1-10 seconds

**Optimization:** Use fewer replicates for testing (e.g., 25-50), then scale to 500-1000 for publication.

### Memory Management

- Base model: ~1 MB for a 50-group model
- Per replicate: ~100 KB additional
- 100 replicates on 50 groups: ~15 MB total

For very large models (>200 groups), consider:
1. Sampling only key groups
2. Processing in batches
3. Storing only statistics, not full models

## Related Functions

- `pypath.io.ewemdb.read_ewe_samples()` - Read samples from EwE database
- `pypath.core.sampling.validate_samples()` - Validate sample data before ensemble run
- `pypath.core.ensemble.EnsembleResult.summary_table()` - Wide-format results table
- See [API Documentation](../api/core.md#ensemble) for full reference
