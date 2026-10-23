#!/usr/bin/env python3
"""
Curonian Lagoon Ecosim Example Application
===========================================

This example demonstrates a complete Ecosim workflow for the Curonian Lagoon
using the pelagic Ecopath model and historical fishery data.

Features:
- Load balanced Ecopath model from EwE database
- Create multiple Ecosim scenarios with different fishing pressures
- Run dynamic simulations (20-year horizon)
- Analyze and visualize results
- Compare ecosystem responses to management alternatives

Author: PyPath Contributors
Reference: Drewes et al. (2025) - Curonian Lagoon ecosystem modeling
"""

import sys
import logging
import warnings
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, Tuple

warnings.filterwarnings('ignore', category=UserWarning)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Data paths
ECOPATH_MODEL = Path(r"C:\Users\DELL\OneDrive - ku.lt\ECOPATH\ECOPATH paper\curonian pelagic.eweaccdb")
LANDINGS_DATA = Path(r"C:\Users\DELL\OneDrive - ku.lt\ECOPATH\ECOPATH paper\landings_Curonian.xlsx")
OUTPUT_DIR = Path(__file__).parent / "curonian_ecosim_results"


def load_model_and_data() -> Tuple:
    """Load the Ecopath model and fishery data."""
    logger.info(f"Loading Ecopath model from {ECOPATH_MODEL}")
    
    try:
        from pypath.io.ewemdb import read_ewemdb
        
        params = read_ewemdb(str(ECOPATH_MODEL))
        logger.info(f"Model loaded: {len(params.model)} functional groups")
        
        print("\n" + "="*70)
        print("CURONIAN LAGOON ECOPATH MODEL")
        print("="*70)
        print(f"\nFunctional Groups: {len(params.model)}")
        print("\nGroups and key parameters:")
        print(params.model[["Group", "Type", "Biomass", "PB", "QB", "EE"]].to_string())
        
    except Exception as e:
        logger.error(f"Failed to load Ecopath model: {e}")
        raise
    
    logger.info(f"Loading fishery landings from {LANDINGS_DATA}")
    try:
        landings = pd.read_excel(str(LANDINGS_DATA), sheet_name=0, header=1)
        logger.info(f"Landings data loaded: {landings.shape}")
        print("\n" + "="*70)
        print("FISHERY LANDINGS DATA (2007)")
        print("="*70)
        print(landings[:-1].to_string())  # Exclude summary row
        
    except Exception as e:
        logger.warning(f"Could not load landings: {e}")
        landings = None
    
    return params, landings


def create_ecosim_scenarios(params, landings: pd.DataFrame = None) -> Tuple:
    """Create multiple Ecosim scenarios with different fishing pressures."""
    from pypath.core.ecopath import rpath
    from pypath.core.ecosim import rsim_scenario
    
    logger.info("Creating Ecopath mass-balance solution...")
    try:
        model = rpath(params)
        logger.info("Ecopath model balanced")
    except Exception as e:
        logger.error(f"Failed to balance Ecopath model: {e}")
        raise
    
    scenarios = {}
    years = np.arange(1, 21)  # 20-year simulation
    
    # Scenario 1: Baseline (constant fishing)
    logger.info("Creating Scenario 1: Baseline")
    try:
        scenario_base = rsim_scenario(model, params, years=years)
        if landings is not None:
            total_catch_2007 = landings.iloc[:-1, 1].sum()
            logger.info(f"  Total catch 2007: {total_catch_2007:.1f} tonnes")
        scenarios['baseline'] = scenario_base
        print("\nScenario 1: Baseline (constant fishing) - CREATED")
    except Exception as e:
        logger.warning(f"Could not create baseline: {e}")
    
    # Scenario 2: 50% fishing reduction
    logger.info("Creating Scenario 2: 50% Fishing Reduction")
    try:
        scenario_reduce = rsim_scenario(model, params, years=years)
        # Modify fishing effort: gradual reduction from 100% to 50% over 10 years
        if hasattr(scenario_reduce.fishing, 'Frate') and scenario_reduce.fishing.Frate is not None:
            frate = scenario_reduce.fishing.Frate
            for year_idx, year in enumerate(years):
                if year <= 10:
                    multiplier = 1.0 - (year / 10) * 0.5
                else:
                    multiplier = 0.5
                frate[year_idx, :] *= multiplier
        
        scenarios['reduce_50'] = scenario_reduce
        print("Scenario 2: 50% Fishing Reduction - CREATED")
    except Exception as e:
        logger.warning(f"Could not create reduction: {e}")
    
    # Scenario 3: Complete closure
    logger.info("Creating Scenario 3: Complete Closure")
    try:
        scenario_closure = rsim_scenario(model, params, years=years)
        if hasattr(scenario_closure.fishing, 'Frate') and scenario_closure.fishing.Frate is not None:
            scenario_closure.fishing.Frate[:, :] = 0
        scenarios['closure'] = scenario_closure
        print("Scenario 3: Complete Fishing Closure - CREATED")
    except Exception as e:
        logger.warning(f"Could not create closure: {e}")
    
    # Scenario 4: Increased fishing
    logger.info("Creating Scenario 4: Increased Fishing")
    try:
        scenario_increase = rsim_scenario(model, params, years=years)
        if hasattr(scenario_increase.fishing, 'Frate') and scenario_increase.fishing.Frate is not None:
            frate = scenario_increase.fishing.Frate
            for year_idx, year in enumerate(years):
                if year <= 10:
                    multiplier = 1.0 + (year / 10) * 0.3
                else:
                    multiplier = 1.3
                frate[year_idx, :] *= multiplier
        
        scenarios['increase_30'] = scenario_increase
        print("Scenario 4: 30% Fishing Increase - CREATED")
    except Exception as e:
        logger.warning(f"Could not create increase: {e}")
    
    logger.info(f"Created {len(scenarios)} scenarios")
    return scenarios, model


def run_simulations(scenarios: Dict, model, params) -> Dict:
    """Run Ecosim simulations for all scenarios."""
    from pypath.core.ecosim import rsim_run
    
    results = {}
    
    print("\n" + "="*70)
    print("RUNNING ECOSIM SIMULATIONS (20-year horizon)")
    print("="*70)
    
    for scenario_name, scenario in scenarios.items():
        logger.info(f"Running: {scenario_name}")
        try:
            result = rsim_run(scenario, method="RK4")
            results[scenario_name] = result
            
            # Extract biomass - use annual aggregation (every 12 months)
            if hasattr(result, 'annual_Biomass'):
                annual_biomass = result.annual_Biomass[:, :-1]  # Exclude Time column
                initial = annual_biomass[0, :]
                final = annual_biomass[-1, :]
            else:
                # Fall back to out_Biomass
                biomass_ts = result.out_Biomass[:, :-1]
                initial = biomass_ts[0, :]
                final = biomass_ts[-1, :]
            
            # Only calculate where initial biomass is non-zero
            valid = initial > 0
            if valid.sum() > 0:
                pct_change = ((final[valid] - initial[valid]) / initial[valid] * 100).mean()
            else:
                pct_change = 0
            
            print(f"  {scenario_name:20s} - Mean biomass change: {pct_change:+.1f}%")
            
        except Exception as e:
            logger.error(f"Simulation failed for {scenario_name}: {e}")
    
    return results


def analyze_results(results: Dict, params) -> pd.DataFrame:
    """Analyze and compare simulation results."""
    logger.info("Analyzing results...")
    
    print("\n" + "="*70)
    print("RESULTS SUMMARY")
    print("="*70)
    
    summary_data = []
    
    for scenario_name, output in results.items():
        if hasattr(output, 'annual_Biomass'):
            biomass_ts = output.annual_Biomass[:, :-1]
        else:
            biomass_ts = output.out_Biomass[::12, :-1]  # Sample every 12 months (annual)
        
        initial = biomass_ts[0, :]
        final = biomass_ts[-1, :]
        
        # Only consider groups with initial biomass > 0
        valid = initial > 0
        if valid.sum() > 0:
            pct_change = ((final[valid] - initial[valid]) / initial[valid] * 100)
            mean_change = pct_change.mean()
            max_increase = pct_change.max()
            max_decline = pct_change.min()
        else:
            mean_change = max_increase = max_decline = 0
        
        summary_data.append({
            'Scenario': scenario_name,
            'Mean_Change_%': mean_change,
            'Max_Increase_%': max_increase,
            'Max_Decline_%': max_decline,
        })
    
    summary = pd.DataFrame(summary_data)
    print("\n" + summary.to_string(index=False))
    
    return summary


def export_results(results: Dict, params, summary: pd.DataFrame):
    """Export results to files."""
    OUTPUT_DIR.mkdir(exist_ok=True)
    logger.info(f"Exporting to {OUTPUT_DIR}")
    
    summary_file = OUTPUT_DIR / "simulation_summary.csv"
    summary.to_csv(summary_file, index=False)
    logger.info(f"Summary exported")
    
    group_names = params.model["Group"].values
    
    for scenario_name, output in results.items():
        if hasattr(output, 'annual_Biomass'):
            biomass_ts = output.annual_Biomass[:, :-1]
            years = np.arange(len(biomass_ts))
        else:
            biomass_ts = output.out_Biomass[::12, :-1]
            years = np.arange(len(biomass_ts))
        
        df = pd.DataFrame(biomass_ts, columns=group_names)
        df.insert(0, 'Year', years)
        
        file_path = OUTPUT_DIR / f"biomass_{scenario_name}.csv"
        df.to_csv(file_path, index=False)
    
    print(f"\nResults exported to: {OUTPUT_DIR}")


def visualize_results(results: Dict, params, summary: pd.DataFrame):
    """Create visualizations of simulation results."""
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        logger.warning("matplotlib not available, skipping visualizations")
        return
    
    logger.info("Creating visualizations...")
    OUTPUT_DIR.mkdir(exist_ok=True)
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle('Curonian Lagoon Ecosim: Scenario Comparison', fontsize=16, fontweight='bold')
    
    colors = {'baseline': 'blue', 'reduce_50': 'green', 'increase_30': 'red', 'closure': 'purple'}
    
    # Plot 1a: Total biomass trajectory
    ax = axes[0, 0]
    for scenario_name, output in results.items():
        if hasattr(output, 'annual_Biomass'):
            biomass_ts = output.annual_Biomass[:, :-1]
        else:
            biomass_ts = output.out_Biomass[::12, :-1]
        
        total_biomass = biomass_ts.sum(axis=1)
        years = np.arange(len(total_biomass))
        color = colors.get(scenario_name, 'gray')
        ax.plot(years, total_biomass, marker='o', markersize=3, label=scenario_name, color=color, linewidth=2)
    
    ax.set_xlabel('Year', fontsize=10, fontweight='bold')
    ax.set_ylabel('Total Biomass (gC/m²)', fontsize=10, fontweight='bold')
    ax.set_title('Ecosystem Total Biomass', fontsize=11, fontweight='bold')
    ax.legend(loc='best')
    ax.grid(True, alpha=0.3)
    
    # Plot 1b: Biomass change percentage
    ax = axes[0, 1]
    scenario_names = summary['Scenario'].values
    pct_changes = summary['Mean_Change_%'].values
    bar_colors = [colors.get(name, 'gray') for name in scenario_names]
    ax.bar(scenario_names, pct_changes, color=bar_colors, alpha=0.7, edgecolor='black')
    ax.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
    ax.set_ylabel('Mean Biomass Change (%)', fontsize=10, fontweight='bold')
    ax.set_title('Average Biomass Response', fontsize=11, fontweight='bold')
    ax.tick_params(axis='x', rotation=45)
    ax.grid(True, alpha=0.3, axis='y')
    
    # Plot 1c: Key species trajectories (first scenario)
    ax = axes[1, 0]
    first_scenario = list(results.keys())[0]
    output = results[first_scenario]
    
    if hasattr(output, 'annual_Biomass'):
        biomass_ts = output.annual_Biomass[:, :-1]
    else:
        biomass_ts = output.out_Biomass[::12, :-1]
    
    group_names_short = params.model["Group"].values[:5]
    years = np.arange(biomass_ts.shape[0])
    
    for i, group in enumerate(group_names_short):
        ax.plot(years, biomass_ts[:, i], marker='o', markersize=3, label=group, linewidth=2)
    
    ax.set_xlabel('Year', fontsize=10, fontweight='bold')
    ax.set_ylabel('Biomass (gC/m²)', fontsize=10, fontweight='bold')
    ax.set_title(f'Key Species - {first_scenario} scenario', fontsize=11, fontweight='bold')
    ax.legend(loc='best', fontsize=9)
    ax.grid(True, alpha=0.3)
    
    # Plot 1d: Min/Max biomass change
    ax = axes[1, 1]
    x = np.arange(len(summary))
    width = 0.35
    
    ax.bar(x - width/2, summary['Max_Increase_%'], width, label='Max Increase', color='green', alpha=0.7)
    ax.bar(x + width/2, summary['Max_Decline_%'], width, label='Max Decline', color='red', alpha=0.7)
    ax.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
    ax.set_ylabel('Biomass Change (%)', fontsize=10, fontweight='bold')
    ax.set_title('Species-Level Response Range', fontsize=11, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(summary['Scenario'].values, rotation=45)
    ax.legend(loc='best')
    ax.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    plot_file = OUTPUT_DIR / "ecosim_comparison.png"
    plt.savefig(plot_file, dpi=150, bbox_inches='tight')
    logger.info(f"Visualization saved")
    plt.close()


def main():
    """Main workflow."""
    print("\n" + "="*70)
    print("CURONIAN LAGOON ECOSIM EXAMPLE")
    print("="*70)
    print("\nWorkflow:")
    print("1. Load Ecopath model from EwE database")
    print("2. Create 4 management scenarios")
    print("3. Run 20-year simulations")
    print("4. Compare ecosystem responses")
    
    try:
        params, landings = load_model_and_data()
        scenarios, model = create_ecosim_scenarios(params, landings)
        results = run_simulations(scenarios, model, params)
        
        if results:
            summary = analyze_results(results, params)
            export_results(results, params, summary)
            visualize_results(results, params, summary)
        
        print("\n" + "="*70)
        print("ANALYSIS COMPLETE")
        print("="*70)
        print(f"\nResults saved to: {OUTPUT_DIR}")
        print("  - simulation_summary.csv: Scenario comparison statistics")
        print("  - biomass_*.csv: Biomass trajectories by scenario")
        print("  - ecosim_comparison.png: Visualization")
        print("="*70 + "\n")
        
    except Exception as e:
        logger.error(f"Error: {e}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
