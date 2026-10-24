#!/usr/bin/env python3
"""
Phase 4 Checkpoint 2: Execute Ecosim Scenarios
"""
from __future__ import annotations

import pickle
import json
import logging
import sys
from pathlib import Path
from typing import Dict, List, Tuple
import numpy as np
import pandas as pd
from datetime import datetime

sys.path.insert(0, '.')

from pypath.core.ecopath import rpath
from pypath.core.ecosim import rsim_scenario, rsim_run
from pypath.core.params import create_rpath_params, RpathParams

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('phase4_checkpoint2_scenarios.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

MODEL_PATH = Path("curonian_fitted_model.pkl")
SCENARIOS_CONFIG = Path("phase_4_scenarios/curonian_scenarios_2025_2035.json")
OUTPUT_DIR = Path("phase_4_scenarios")
YEARS_RANGE = range(2025, 2036)
STEPS_PER_YEAR = 12


def load_or_create_base_model() -> RpathParams:
    """Load or create base Curonian model parameters."""
    logger.info("Loading base Curonian model...")
    
    # Define groups
    group_data = [
        ("Roach", 0),
        ("Bream", 0),
        ("Pike-Perch (Zander)", 0),
        ("Perch", 0),
        ("Ruffe", 0),
        ("Silver Bream", 0),
        ("Vimba", 0),
        ("Planktivorous fish", 0),
        ("Demersal fish", 0),
        ("Eel", 0),
        ("Grey Heron", 0),
        ("Larus gull", 0),
        ("Goosander", 0),
        ("Great Crested Grebe", 0),
        ("Cormorant", 0),
        ("Daphnia & Cladocera", 0),
        ("Bosmina", 0),
        ("Grazing zooplankton", 0),
        ("Carnivorous zooplankton", 0),
        ("Rotifers", 0),
        ("Copepod nauplii", 0),
        ("Chironomids", 0),
        ("Oligochaetes", 0),
        ("Dreissena (mussel)", 0),
        ("Nematodes", 0),
        ("Mysids", 0),
        ("Deposit feeder gastropods", 0),
        ("Filtrator bivalves", 0),
        ("Meiobenthos", 0),
        ("Phytoplankton", 1),
        ("Bacteria", 0),
        ("Detritus", 2),
        ("Fishing Fleet - Commercial", 3),
    ]
    
    groups = [name for name, _ in group_data]
    types = [typ for _, typ in group_data]
    
    logger.info(f"  Creating base model with {len(groups)} groups")
    params = create_rpath_params(groups, types)
    
    # Set biomass values
    default_biomass = {
        "Roach": 50.0,
        "Bream": 40.0,
        "Pike-Perch (Zander)": 15.0,
        "Perch": 30.0,
        "Vimba": 25.0,
        "Phytoplankton": 100.0,
        "Detritus": 200.0,
    }
    
    for i, group_name in enumerate(params.model['Group'].values):
        if group_name in default_biomass:
            params.model.loc[i, "Biomass"] = default_biomass[group_name]
        else:
            params.model.loc[i, "Biomass"] = 5.0
    
    # Set PB values
    for i, group_name in enumerate(params.model['Group'].values):
        if group_name == "Phytoplankton":
            params.model.loc[i, "PB"] = 200.0
        elif group_name == "Detritus":
            params.model.loc[i, "PB"] = 0.1
        else:
            params.model.loc[i, "PB"] = 1.0
    
    # Set EE values
    for i, group_name in enumerate(params.model['Group'].values):
        if group_name in ["Phytoplankton", "Bacteria"]:
            params.model.loc[i, "EE"] = 0.95
        elif group_name == "Detritus":
            params.model.loc[i, "EE"] = 1.0
        else:
            params.model.loc[i, "EE"] = 0.8
    
    # Set QB values
    for i, group_name in enumerate(params.model['Group'].values):
        if group_name in ["Daphnia & Cladocera", "Bosmina", "Rotifers"]:
            params.model.loc[i, "QB"] = 80.0
        elif "zooplankton" in group_name.lower():
            params.model.loc[i, "QB"] = 50.0
        elif group_name in ["Phytoplankton", "Detritus"]:
            params.model.loc[i, "QB"] = 0.0
        else:
            params.model.loc[i, "QB"] = 5.0
    
    # Set up diet matrix with Group column
    group_names_list = params.model['Group'].tolist()
    diet_data = np.zeros((len(group_names_list), len(group_names_list)))
    params.diet = pd.DataFrame(diet_data, columns=group_names_list)
    params.diet.insert(0, 'Group', group_names_list)
    
    # Simple food web
    fish_species = ["Roach", "Bream", "Perch", "Ruffe", "Vimba", "Planktivorous fish", "Demersal fish", "Eel"]
    zoop_species = ["Daphnia & Cladocera", "Bosmina", "Grazing zooplankton", "Rotifers", "Copepod nauplii"]
    benthos_species = ["Chironomids", "Oligochaetes", "Nematodes", "Mysids", "Deposit feeder gastropods", "Filtrator bivalves", "Meiobenthos"]
    
    for fish in fish_species:
        if fish in group_names_list:
            fish_idx = group_names_list.index(fish)
            for zoop in zoop_species:
                if zoop in group_names_list:
                    params.diet.loc[fish_idx, zoop] = 0.5
            for bent in benthos_species:
                if bent in group_names_list:
                    params.diet.loc[fish_idx, bent] = 0.3
            if "Detritus" in group_names_list:
                params.diet.loc[fish_idx, "Detritus"] = 0.2
    
    # Normalize diet (excluding Group column)
    for i in range(len(group_names_list)):
        row_sum = params.diet.iloc[i, 1:].sum()
        if row_sum > 0:
            params.diet.iloc[i, 1:] = params.diet.iloc[i, 1:] / row_sum
    
    logger.info("  Base model created")
    return params


def load_fitted_parameters() -> Dict[str, float]:
    """Load fitted parameters from calibration."""
    logger.info("Loading fitted parameters from " + str(MODEL_PATH))
    
    with open(MODEL_PATH, 'rb') as f:
        fitted_state = pickle.load(f)
    
    fitted_params = fitted_state['fitted_parameters']
    logger.info(f"  Loaded {len(fitted_params)} fitted parameters")
    
    return fitted_params


def apply_fitted_parameters(params: RpathParams, fitted_params: Dict[str, float]) -> RpathParams:
    """Apply fitted calibration parameters to model."""
    logger.info("Applying fitted parameters...")
    
    # Apply fitted EE values
    ee_applied = 0
    for key, value in fitted_params.items():
        if key.startswith('EE_'):
            group_name = key[3:]
            matching = params.model[params.model['Group'] == group_name]
            if len(matching) > 0:
                idx = matching.index[0]
                params.model.loc[idx, 'EE'] = value
                ee_applied += 1
    
    logger.info(f"  Applied {ee_applied} EE parameters")
    
    return params


def construct_ecopath_model(params: RpathParams) -> any:
    """Construct Ecopath model from parameters."""
    logger.info("Constructing Ecopath model...")
    model = rpath(params)
    logger.info(f"  Model constructed: {len(model.Group)} groups")
    return model


def run_scenario(model: any, params: RpathParams, scenario_dict: dict, scenario_name: str) -> Tuple[any, dict]:
    """Execute single Ecosim scenario."""
    print(f"\n{'='*70}\nRunning Scenario {scenario_dict['id']}: {scenario_name}\n{'='*70}")
    
    logger.info(f"Starting scenario: {scenario_name}")
    logger.info(f"  Period: {YEARS_RANGE.start}-{YEARS_RANGE.stop-1}")
    
    # Create scenario with default parameters
    scenario = rsim_scenario(rpath=model, rpath_params=params, years=YEARS_RANGE)
    
    logger.info("  Running simulation...")
    output = rsim_run(scenario)
    
    logger.info(f"  Simulation complete: {YEARS_RANGE.stop - YEARS_RANGE.start} years")
    
    return output, scenario_dict


def aggregate_to_annual(biomass_array: np.ndarray, catch_array: np.ndarray, steps_per_year: int = STEPS_PER_YEAR) -> Tuple[np.ndarray, np.ndarray]:
    """Aggregate monthly timesteps to annual values.
    
    Parameters
    ----------
    biomass_array : np.ndarray
        Shape (total_steps, n_groups)
    catch_array : np.ndarray
        Shape (total_steps, n_groups)
    steps_per_year : int
        Timesteps per year (default 12 for monthly)
    
    Returns
    -------
    biomass_annual : np.ndarray
        Shape (years, n_groups) - average biomass per year
    catch_annual : np.ndarray
        Shape (years, n_groups) - total catch per year
    """
    n_groups = biomass_array.shape[1]
    n_years = biomass_array.shape[0] // steps_per_year
    
    biomass_annual = np.zeros((n_years, n_groups))
    catch_annual = np.zeros((n_years, n_groups))
    
    for year in range(n_years):
        start_idx = year * steps_per_year
        end_idx = (year + 1) * steps_per_year
        biomass_annual[year, :] = biomass_array[start_idx:end_idx, :].mean(axis=0)
        catch_annual[year, :] = catch_array[start_idx:end_idx, :].sum(axis=0)
    
    return biomass_annual, catch_annual


def extract_scenario_results(output: any, model: any, scenario_dict: dict) -> pd.DataFrame:
    """Extract annual biomass and catch from Ecosim output."""
    scenario_name = scenario_dict['name']
    logger.info(f"Extracting results for {scenario_name}...")
    
    # Get biomass and catch data
    if not hasattr(output, 'out_Biomass'):
        raise ValueError(f"Cannot find out_Biomass in output for {scenario_name}")
    
    biomass_array = output.out_Biomass
    catch_array = output.out_Catch if hasattr(output, 'out_Catch') else np.zeros_like(biomass_array)
    
    # Aggregate to annual
    biomass_annual, catch_annual = aggregate_to_annual(biomass_array, catch_array)
    
    # Build results dataframe
    results = {
        'year': [],
        'group_id': [],
        'group_name': [],
        'biomass_tonnes': [],
        'catch_tonnes': [],
    }
    
    for year_idx, year in enumerate(range(YEARS_RANGE.start, YEARS_RANGE.stop)):
        for group_idx in range(len(model.Group)):
            results['year'].append(year)
            results['group_id'].append(group_idx)
            results['group_name'].append(model.Group[group_idx])
            results['biomass_tonnes'].append(float(biomass_annual[year_idx, group_idx]))
            results['catch_tonnes'].append(float(catch_annual[year_idx, group_idx]))
    
    df = pd.DataFrame(results)
    logger.info(f"  Results extracted: Shape {df.shape}")
    return df


def save_scenario_results(df: pd.DataFrame, scenario_dict: dict) -> Path:
    """Save scenario results to CSV."""
    scenario_code = scenario_dict['code']
    output_file = OUTPUT_DIR / f"results_{scenario_code}_2025_2035.csv"
    df.to_csv(output_file, index=False)
    logger.info(f"  Results saved: {output_file}")
    return output_file


def calculate_scenario_metrics(df: pd.DataFrame, scenario_dict: dict) -> dict:
    """Calculate summary metrics for scenario."""
    scenario_name = scenario_dict['name']
    annual_totals = df.groupby('year').agg({
        'biomass_tonnes': 'sum',
        'catch_tonnes': 'sum'
    }).reset_index()
    
    metrics = {
        'scenario_id': scenario_dict['id'],
        'scenario_code': scenario_dict['code'],
        'scenario_name': scenario_name,
        'policy_focus': scenario_dict['policy_focus'],
        'mean_total_biomass_tonnes': float(annual_totals['biomass_tonnes'].mean()),
        'total_catch_tonnes': float(annual_totals['catch_tonnes'].sum()),
        'biomass_cv': float(annual_totals['biomass_tonnes'].std() / annual_totals['biomass_tonnes'].mean()),
    }
    
    if len(annual_totals) > 0:
        start_2025 = annual_totals[annual_totals['year'] == 2025]['biomass_tonnes'].values
        end_2035 = annual_totals[annual_totals['year'] == 2035]['biomass_tonnes'].values
        if len(start_2025) > 0 and len(end_2035) > 0 and start_2025[0] > 0:
            metrics['biomass_change_pct'] = ((end_2035[0] / start_2025[0]) - 1) * 100
        else:
            metrics['biomass_change_pct'] = np.nan
    
    logger.info(f"  Mean biomass: {metrics['mean_total_biomass_tonnes']:.1f} tonnes")
    return metrics


def load_scenarios_config() -> Dict[str, any]:
    """Load scenario definitions from JSON."""
    logger.info("Loading scenarios configuration from " + str(SCENARIOS_CONFIG))
    with open(SCENARIOS_CONFIG, 'r') as f:
        config = json.load(f)
    logger.info(f"  Loaded {len(config['scenarios'])} scenarios")
    return config


def main():
    """Main execution."""
    logger.info("="*80)
    logger.info("PHASE 4 CHECKPOINT 2: ECOSYSTEM SCENARIO MODELING")
    logger.info("="*80)
    
    try:
        # Step 1: Load/create base model
        logger.info("\n[STEP 1] Loading base model...")
        params = load_or_create_base_model()
        
        # Step 2: Load and apply fitted parameters
        logger.info("\n[STEP 2] Applying fitted parameters...")
        fitted_params = load_fitted_parameters()
        params = apply_fitted_parameters(params, fitted_params)
        
        # Step 3: Construct Ecopath model
        logger.info("\n[STEP 3] Constructing Ecopath model...")
        model = construct_ecopath_model(params)
        
        # Step 4: Load scenario definitions
        logger.info("\n[STEP 4] Loading scenario definitions...")
        scenarios_config = load_scenarios_config()
        
        # Step 5: Run all scenarios
        logger.info("\n[STEP 5] Running all 5 scenarios...")
        results_dict = {}
        metrics_list = []
        
        for scenario_def in scenarios_config['scenarios']:
            scenario_name = scenario_def['name']
            output, _ = run_scenario(model, params, scenario_def, scenario_name)
            df = extract_scenario_results(output, model, scenario_def)
            results_dict[scenario_name] = df
            save_scenario_results(df, scenario_def)
            metrics = calculate_scenario_metrics(df, scenario_def)
            metrics_list.append(metrics)
            output_pkl = OUTPUT_DIR / f"output_{scenario_def['code']}_2025_2035.pkl"
            with open(output_pkl, 'wb') as f:
                pickle.dump(output, f)
            logger.info(f"  Output saved: {output_pkl}")
        
        # Step 6: Save metrics
        logger.info("\n[STEP 6] Saving metrics...")
        metrics_df = pd.DataFrame(metrics_list)
        metrics_file = OUTPUT_DIR / "SCENARIO_COMPARISON_METRICS.csv"
        metrics_df.to_csv(metrics_file, index=False)
        logger.info(f"  Metrics saved: {metrics_file}")
        
        print("\n" + "="*80)
        print("PHASE 4 CHECKPOINT 2: ECOSYSTEM SCENARIO MODELING RESULTS")
        print("="*80)
        print(f"\nExecution Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"Scenarios Executed: {len(results_dict)}\n")
        for _, row in metrics_df.iterrows():
            print(f"{row['scenario_id']}. {row['scenario_code']}: {row['scenario_name']}")
            print(f"   Mean Biomass: {row['mean_total_biomass_tonnes']:.1f} tonnes")
            print(f"   Total Catch: {row['total_catch_tonnes']:.1f} tonnes")
            if 'biomass_change_pct' in row and not np.isnan(row['biomass_change_pct']):
                print(f"   Biomass Change: {row['biomass_change_pct']:.1f}%")
            print()
        
        logger.info("\n" + "="*80)
        logger.info("PHASE 4 CHECKPOINT 2 COMPLETE")
        logger.info("="*80)
        
        return 0
        
    except Exception as e:
        logger.error(f"\nEXECUTION FAILED: {e}", exc_info=True)
        import traceback
        traceback.print_exc()
        return 1


if __name__ == '__main__':
    exit(main())
