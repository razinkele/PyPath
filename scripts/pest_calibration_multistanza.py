"""
PEST Calibration for Curonian Lagoon Multistanza Model
======================================================

Simplified demonstration using existing multistanza example.
"""

import logging
import numpy as np
import pandas as pd
from pathlib import Path

logger = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)

OUTPUT_DIR = Path("./pest_multistanza_calibration_results")
OUTPUT_DIR.mkdir(exist_ok=True)


def run_calibration_demo():
    """Demonstrate PEST calibration workflow with multistanza model."""
    logger.info("=" * 80)
    logger.info("PEST CALIBRATION FOR CURONIAN LAGOON MULTISTANZA MODEL")
    logger.info("=" * 80)
    
    # Step 1: Load multistanza model
    logger.info("\nStep 1: Creating multistanza Ecopath model")
    try:
        from pypath.core.ecopath import rpath
        from pypath.core.params import create_rpath_params
        import pandas as pd
        
        # Load Curonian model data
        data_dir = Path(__file__).parent.parent / "example_model_data"
        model_df = pd.read_csv(data_dir / "CurLag_Ecopath_Groups.csv", index_col=0)
        diet_df = pd.read_csv(data_dir / "CurLag_Ecopath_Diet.csv", index_col=0)
        
        groups = list(model_df.index)
        types = model_df["Type"].values
        params = create_rpath_params(groups=groups, types=types)
        
        # Fill in values
        for col in ["Biomass", "PB", "QB", "EE"]:
            if col in model_df.columns:
                params.model[col] = model_df[col].values
        
        params = rpath(params)
        logger.info(f"  Model created: {params.model.shape[0]} groups")
    except Exception as e:
        logger.error(f"Failed to load model: {e}")
        logger.info("  Using simplified demo mode")
        params = None
    
    # Step 2: Define parameter bounds for calibration
    logger.info("\nStep 2: Defining calibration parameters")
    param_ranges = {
        "demersal_juvenile_K": (0.075, 0.225),
        "demersal_juvenile_Z": (0.6, 1.8),
        "demersal_adult_K": (0.075, 0.225),
        "demersal_adult_Z": (0.3, 0.9),
        "predatory_juvenile_K": (0.06, 0.18),
        "predatory_juvenile_Z": (0.75, 2.25),
        "predatory_subadult_K": (0.06, 0.18),
        "predatory_subadult_Z": (0.5, 1.5),
        "predatory_adult_K": (0.06, 0.18),
        "predatory_adult_Z": (0.25, 0.75),
    }
    
    logger.info(f"  Defined {len(param_ranges)} parameters for calibration")
    
    # Step 3: Create synthetic observations
    logger.info("\nStep 3: Generating synthetic age-structured catch data")
    np.random.seed(42)
    
    catch_records = []
    stanza_names = ["demersal_juvenile", "demersal_adult", 
                    "predatory_juvenile", "predatory_subadult", "predatory_adult"]
    
    for stanza_name in stanza_names:
        for year in range(1, 21):
            # Synthetic catch with realistic pattern
            base_catch = np.random.uniform(10, 50)
            seasonal_var = 1 + 0.2 * np.sin(2 * np.pi * year / 5)
            catch = base_catch * seasonal_var
            
            catch_records.append({
                "stanza": stanza_name,
                "year": year,
                "age_group": stanza_name.split("_")[1],
                "catch_tons": catch,
            })
    
    obs_df = pd.DataFrame(catch_records)
    obs_path = OUTPUT_DIR / "synthetic_observations.csv"
    obs_df.to_csv(obs_path, index=False)
    logger.info(f"  Created {len(obs_df)} synthetic observations")
    logger.info(f"  Total synthetic catch: {obs_df['catch_tons'].sum():.1f} tons")
    
    # Step 4: Create parameter comparison table
    logger.info("\nStep 4: Creating calibration parameter summary")
    
    baseline_params = {
        "demersal_juvenile_K": 0.15,
        "demersal_juvenile_Z": 1.2,
        "demersal_adult_K": 0.15,
        "demersal_adult_Z": 0.6,
        "predatory_juvenile_K": 0.12,
        "predatory_juvenile_Z": 1.5,
        "predatory_subadult_K": 0.12,
        "predatory_subadult_Z": 1.0,
        "predatory_adult_K": 0.12,
        "predatory_adult_Z": 0.5,
    }
    
    param_rows = []
    for param_name, baseline_val in baseline_params.items():
        lower, upper = param_ranges[param_name]
        param_type = "K" if "K" in param_name else "Z"
        stanza = param_name.split("_K")[0].split("_Z")[0]
        
        param_rows.append({
            "Stanza": stanza,
            "Parameter": param_type,
            "Baseline": baseline_val,
            "Lower_Bound": lower,
            "Upper_Bound": upper,
            "Range_Fraction": f"plus-minus50%",
        })
    
    param_df = pd.DataFrame(param_rows)
    param_path = OUTPUT_DIR / "calibration_parameters.csv"
    param_df.to_csv(param_path, index=False)
    logger.info(f"  Saved {len(param_df)} parameter definitions")
    
    # Step 5: Optimization strategy
    logger.info("\nStep 5: PEST Calibration Workflow Summary")
    logger.info("  * 5 multistanza groups defined (demersal 2-stanza, predatory 3-stanza)")
    logger.info("  * 10 Von Bertalanffy K and natural mortality Z parameters")
    logger.info("  * 100 synthetic age-structured catch observations (5 stanzas x 20 years)")
    logger.info("  * Parameter bounds: plus-minus50% from baseline")
    logger.info("  * Recommended algorithm: GLM (Gauss-Levenberg-Marquardt)")
    logger.info("  * Alternative algorithm: Differential Evolution")
    
    # Step 6: Export documentation
    logger.info("\nStep 6: Creating calibration guide")
    guide_path = OUTPUT_DIR / "PEST_MULTISTANZA_GUIDE.md"
    guide_text = f"""# PEST Calibration for Multistanza Models

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
- Data: {len(obs_df)} records from {obs_path.name}

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
"""
    
    with open(guide_path, "w", encoding="utf-8") as f:
        f.write(guide_text)
    logger.info(f"  Saved calibration guide to {guide_path}")
    
    logger.info("\n" + "=" * 80)
    logger.info("PEST CALIBRATION SETUP COMPLETE")
    logger.info("=" * 80)
    logger.info(f"\nResults saved to: {OUTPUT_DIR}")
    logger.info("Files:")
    logger.info("  - calibration_parameters.csv")
    logger.info("  - synthetic_observations.csv")
    logger.info("  - PEST_MULTISTANZA_GUIDE.md")
    

if __name__ == "__main__":
    run_calibration_demo()

