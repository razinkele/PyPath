"""PEST Calibration Example: Curonian Lagoon Ecosim Model.

This script demonstrates PEST-based calibration for the Curonian Lagoon
Ecosim model, using synthetic biomass observations generated from a
reference scenario.

Workflow:
1. Load Ecopath model and create baseline scenario
2. Generate synthetic "observed" biomass time series (from perturbed run)
3. Define parameters for calibration (vulnerability, PP, fishing)
4. Run PEST calibration (Gauss-Levenberg-Marquardt)
5. Evaluate parameter recovery and uncertainty
6. Export results
"""

import logging
from pathlib import Path

import numpy as np
import pandas as pd

from pypath.core.ecopath import rpath
from pypath.core.ecosim import rsim_run, rsim_scenario
from pypath.core.params import create_rpath_params
from pypath.io.ewemdb import read_ewemdb
from pypath.core.pest_calibration import (
    PestCalibrator,
    PestParameter,
    PestObservation,
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Paths
ECOPATH_MODEL = Path(
    "C:/Users/DELL/OneDrive - ku.lt/ECOPATH/curonian pelagic.eweaccdb"
)
OUTPUT_DIR = Path("curonian_pest_calibration_results")
OUTPUT_DIR.mkdir(exist_ok=True)

def generate_synthetic_observations(
    rpath_model,
    params,
    perturb_factor=0.15,
    years=(1, 6, 11, 16, 20),
):
    """Generate synthetic observations from a perturbed scenario.

    Simulates "observed" data by running Ecosim with slightly perturbed
    parameters and sampling at specific timepoints.

    Parameters
    ----------
    rpath_model : Rpath
        Balanced Ecopath model
    params : RpathParams
        Model parameters
    perturb_factor : float
        Fraction of parameter variation (0.15 = ±15%)
    years : tuple
        Years at which to sample biomass

    Returns
    -------
    pd.DataFrame
        Observations with columns: name, group_index, time_year, value, weight
    """
    logger.info(f"Generating synthetic observations...")

    # Create scenario with perturbed vulnerability
    scenario = rsim_scenario(rpath_model, params, years=np.arange(1, 21))

    # Perturb vulnerability slightly (simulate "true" system)
    n_groups = len(params.model)
    perturb_factor_array = np.random.RandomState(42).normal(
        1.0, perturb_factor, size=(n_groups, n_groups)
    )
    perturb_factor_array = np.clip(perturb_factor_array, 0.5, 2.0)

    if hasattr(scenario, "vv"):
        scenario.vv = scenario.vv * perturb_factor_array

    # Run simulation
    output = rsim_run(scenario, method="RK4")

    # Extract observations
    observations = []
    biomass_ts = output.out_Biomass[:, :-1]

    group_names = params.model.index.tolist()

    for year in years:
        month_idx = int(year * 12)
        if month_idx < len(biomass_ts):
            for group_idx in range(min(5, biomass_ts.shape[1])):
                # Sample only selected groups to avoid redundancy
                biomass = biomass_ts[month_idx, group_idx]
                if biomass > 0:
                    obs_dict = {
                        "name": f"Biomass_{group_names[group_idx]}_Y{year}",
                        "group_index": group_idx,
                        "time_year": float(year),
                        "value": float(biomass),
                        "weight": 1.0 / (0.1 * biomass) ** 2,  # Relative error weight
                    }
                    observations.append(obs_dict)

    obs_df = pd.DataFrame(observations)
    logger.info(f"  Generated {len(obs_df)} observations from {len(group_names)} groups")
    return obs_df

def create_pest_parameters(params):
    """Create PEST parameters for calibration.

    Calibrates vulnerability (VV) for selected predator-prey pairs.

    Parameters
    ----------
    params : RpathParams
        Model parameters

    Returns
    -------
    list of PestParameter
    """
    pest_params = []

    # Add vulnerability parameters for key predator-prey pairs
    n_groups = len(params.model)
    key_pairs = [
        (4, 0),  # Example: group 4 eating group 0
        (3, 1),  # group 3 eating group 1
        (2, 5),  # group 2 eating group 5
    ]

    for pred_idx, prey_idx in key_pairs:
        if pred_idx < n_groups and prey_idx < n_groups:
            param = PestParameter(
                name=f"VV_{pred_idx}_{prey_idx}",
                initial=1.0,
                lower=0.1,
                upper=10.0,
                relative_change=0.05,
                p_type="vv",
                group_indices=(pred_idx, prey_idx),
            )
            pest_params.append(param)

    logger.info(f"Created {len(pest_params)} PEST parameters")
    return pest_params

def scenario_factory_vv(base_scenario, parmvec):
    """Apply vulnerability parameters to scenario.

    Parameters
    ----------
    base_scenario : RsimScenario
        Base scenario
    parmvec : np.ndarray
        Parameter vector (vulnerability multipliers)

    Returns
    -------
    RsimScenario
        Modified scenario
    """
    scenario = base_scenario.__class__.__new__(base_scenario.__class__)
    for attr in dir(base_scenario):
        if not attr.startswith("_"):
            try:
                setattr(scenario, attr, getattr(base_scenario, attr))
            except (AttributeError, TypeError):
                pass

    # Apply VV multipliers
    if hasattr(scenario, "vv"):
        vv_modified = scenario.vv.copy()
        for i, parm_val in enumerate(parmvec):
            # This is simplified; in practice, map parmvec to specific VV elements
            vv_modified[i % vv_modified.shape[0], i % vv_modified.shape[1]] *= parm_val
        scenario.vv = np.clip(vv_modified, 0.01, 100.0)

    return scenario

def main():
    """Run PEST calibration on Curonian Lagoon model."""

    logger.info("=" * 70)
    logger.info("PEST Calibration: Curonian Lagoon Ecosim Model")
    logger.info("=" * 70)

    # Load model
    logger.info("\n1. Loading Ecopath model...")
    if not ECOPATH_MODEL.exists():
        logger.error(f"Model file not found: {ECOPATH_MODEL}")
        return

    try:
        params = read_ewemdb(str(ECOPATH_MODEL))
    except Exception as e:
        logger.error(f"Failed to load model: {e}")
        return

    logger.info(f"  Loaded model with {len(params.model)} groups")

    # Run Ecopath
    logger.info("\n2. Running Ecopath mass balance...")
    rpath_model = rpath(params, quiet=True)
    if not rpath_model:
        logger.error("Ecopath solver failed")
        return
    logger.info("  Ecopath solved successfully")

    # Generate synthetic observations
    logger.info("\n3. Generating synthetic observations...")
    obs_df = generate_synthetic_observations(rpath_model, params)
    obs_df.to_csv(OUTPUT_DIR / "synthetic_observations.csv", index=False)
    logger.info(f"  Saved {len(obs_df)} observations to CSV")

    # Create PEST calibrator
    logger.info("\n4. Setting up PEST calibrator...")
    calibrator = PestCalibrator(rpath_model, params)

    # Add parameters
    pest_params = create_pest_parameters(params)
    for param in pest_params:
        calibrator.add_parameter(param)

    # Add observations
    for _, row in obs_df.iterrows():
        obs = PestObservation(
            name=row["name"],
            value=row["value"],
            weight=row["weight"],
            group_index=int(row["group_index"]),
            time_year=float(row["time_year"]),
        )
        calibrator.add_observation(obs)

    logger.info(f"  Added {len(calibrator.pest_parameters)} parameters")
    logger.info(f"  Added {len(calibrator.pest_observations)} observations")

    # Run calibration
    logger.info("\n5. Running PEST calibration (Gauss-Levenberg-Marquardt)...")
    result = calibrator.calibrate(
        scenario_factory=scenario_factory_vv,
        method="glm",
        max_iterations=15,
        tolerance=0.001,
        verbose=True,
    )

    # Report results
    logger.info("\n6. Calibration Results")
    logger.info(f"  Converged: {result.converged}")
    logger.info(f"  Iterations: {result.n_iterations}")
    logger.info(f"  Final SSR: {result.sum_of_squared_residuals:.6e}")
    logger.info(f"  Time: {result.calibration_time:.2f}s")

    logger.info("\n  Best-fit parameters:")
    for name, value in result.best_params.items():
        unc = result.uncertainty.get(name, np.nan) if result.uncertainty else np.nan
        logger.info(f"    {name:20s} = {value:8.4f} +/- {unc:8.4f}")

    # Save report
    logger.info("\n7. Exporting results...")
    report = calibrator.report(result, filepath=OUTPUT_DIR / "pest_report.txt")

    # Save residuals
    residuals_df = pd.DataFrame(
        [{"observation": k, "residual": v} for k, v in result.residuals.items()]
    )
    residuals_df.to_csv(OUTPUT_DIR / "residuals.csv", index=False)
    logger.info(f"  Saved report and residuals")

    # Save parameter estimates
    params_df = pd.DataFrame(
        [
            {
                "parameter": name,
                "value": value,
                "uncertainty": result.uncertainty.get(name, np.nan)
                if result.uncertainty
                else np.nan,
            }
            for name, value in result.best_params.items()
        ]
    )
    params_df.to_csv(OUTPUT_DIR / "fitted_parameters.csv", index=False)

    logger.info(f"\n" + "=" * 70)
    logger.info(f"Calibration complete! Results saved to {OUTPUT_DIR.absolute()}")
    logger.info(f"=" * 70)

if __name__ == "__main__":
    main()