"""
Curonian Lagoon Ecosim Example Application

Demonstrates a complete workflow for ecosystem modeling using PyPath:
1. Load Ecopath model from EwE database
2. Create management scenarios
3. Run Ecosim dynamic simulations
4. Analyze and compare ecosystem responses

Note: This model exhibits numerical instabilities around year 12-14,
indicating potential structural issues. See analysis warnings for details.
"""

import logging
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Dict, Tuple
from pypath.io.ewemdb import read_ewemdb
from pypath.core.ecopath import rpath
from pypath.core.ecosim import rsim_scenario, rsim_run

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

OUTPUT_DIR = Path("curonian_ecosim_results")

def load_model_and_data() -> Tuple:
    """Load Ecopath model and fishery data."""
    logger.info(f"Loading Ecopath model")
    
    model_path = r"C:\Users\DELL\OneDrive - ku.lt\ECOPATH\ECOPATH paper\curonian pelagic.eweaccdb"
    params = read_ewemdb(model_path)
    logger.info(f"Model loaded: {params.model.shape[0]} functional groups")
    
    logger.info(r"Loading fishery landings")
    landings = pd.read_excel(r"C:\Users\DELL\OneDrive - ku.lt\ECOPATH\ECOPATH paper\landings_Curonian.xlsx")
    logger.info(f"Landings data loaded: {landings.shape}")
    
    return params, landings


def create_ecosim_scenarios(params) -> Dict:
    """Create 4 management scenarios."""
    model = rpath(params)
    scenarios = {}
    
    print("\n" + "="*70)
    print("CURONIAN LAGOON ECOPATH MODEL")
    print("="*70)
    print(f"\nFunctional Groups: {model.NUM_GROUPS}")
    print("\nGroups and key parameters:")
    
    display_data = params.model[['Group', 'Type', 'Biomass', 'PB', 'QB', 'EE']].copy()
    print(display_data.to_string(index=True))
    
    print("\n" + "="*70)
    print("FISHERY LANDINGS DATA (2007)")
    print("="*70)
    # Filter to data rows only (skip header and summary rows)
    landings_clean = pd.read_excel(r"C:\Users\DELL\OneDrive - ku.lt\ECOPATH\ECOPATH paper\landings_Curonian.xlsx", skiprows=1, nrows=8)
    print(landings_clean.to_string(index=False))
    
    # Scenario 1: Baseline (no fishing modifications)
    logger.info("Creating Scenario 1: Baseline")
    scenario1 = rsim_scenario(model, params, years=np.arange(1, 21))
    scenarios['baseline'] = scenario1
    print("\nScenario 1: Baseline (constant fishing) - CREATED")
    
    # Scenario 2: 50% Reduction
    logger.info("Creating Scenario 2: 50% Fishing Reduction")
    scenario2 = rsim_scenario(model, params, years=np.arange(1, 21))
    # Modify fishing rates gradually over 10 years
    # ForcedFRate shape is (20 years, 20 groups)
    for year_idx in range(scenario2.fishing.ForcedFRate.shape[0]):
        year_decimal = year_idx / 1.0  # Direct year index
        if year_decimal < 10:
            # Gradually reduce to 50%
            factor = 1.0 - 0.5 * (year_decimal / 10.0)
        else:
            # Maintain at 50%
            factor = 0.5
        scenario2.fishing.ForcedFRate[year_idx, :] *= factor
    scenarios['reduce_50'] = scenario2
    print("Scenario 2: 50% Fishing Reduction - CREATED")
    
    # Scenario 3: Complete Closure
    logger.info("Creating Scenario 3: Complete Closure")
    scenario3 = rsim_scenario(model, params, years=np.arange(1, 21))
    scenario3.fishing.ForcedFRate[:, :] = 0  # No fishing
    scenarios['closure'] = scenario3
    print("Scenario 3: Complete Fishing Closure - CREATED")
    
    # Scenario 4: 30% Increase
    logger.info("Creating Scenario 4: Increased Fishing")
    scenario4 = rsim_scenario(model, params, years=np.arange(1, 21))
    # Modify fishing rates gradually over 10 years
    for year_idx in range(scenario4.fishing.ForcedFRate.shape[0]):
        year_decimal = year_idx / 1.0
        if year_decimal < 10:
            # Gradually increase to 130%
            factor = 1.0 + 0.3 * (year_decimal / 10.0)
        else:
            # Maintain at 130%
            factor = 1.3
        scenario4.fishing.ForcedFRate[year_idx, :] *= factor
    scenarios['increase_30'] = scenario4
    print("Scenario 4: 30% Fishing Increase - CREATED")
    
    logger.info(f"Created {len(scenarios)} scenarios")
    
    return scenarios, model, params


def run_simulations(scenarios: Dict, model, params) -> Dict:
    """Run Ecosim simulations for all scenarios."""
    print("\n" + "="*70)
    print("RUNNING ECOSIM SIMULATIONS (20-year horizon)")
    print("="*70)
    
    results = {}
    for scenario_name, scenario in scenarios.items():
        logger.info(f"Running: {scenario_name}")
        try:
            result = rsim_run(scenario, method="RK4")
            results[scenario_name] = result
            
            # Print diagnostic info
            if hasattr(result, 'out_Biomass'):
                biomass_ts = result.out_Biomass[::12, :-1]  # Annual (every 12 months)
                initial = biomass_ts[0, :]
                final = biomass_ts[-1, :]
                valid = initial > 0
                if valid.sum() > 0:
                    pct_change = ((final[valid] - initial[valid]) / initial[valid] * 100).mean()
                else:
                    pct_change = 0
                print(f"  {scenario_name:20s} - Mean biomass change: {pct_change:+.1f}%")
        except Exception as e:
            logger.error(f"Simulation failed for {scenario_name}: {e}")
            import traceback
            traceback.print_exc()
    
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
        
        # Check for numerical instabilities
        has_instability = np.any(np.abs(final) > 1e10) or np.any((final > 0) & (final < 1e-6))
        
        # Only consider groups with valid values
        valid = (initial > 0) & np.isfinite(final) & (final > -1e10) & (final < 1e10)
        
        if valid.sum() > 0:
            pct_change = ((final[valid] - initial[valid]) / initial[valid] * 100)
            # Filter extreme values
            pct_change_filtered = pct_change[np.isfinite(pct_change) & (np.abs(pct_change) < 1e6)]
            
            if len(pct_change_filtered) > 0:
                mean_change = pct_change_filtered.mean()
                max_increase = pct_change_filtered.max()
                max_decline = pct_change_filtered.min()
            else:
                mean_change = max_increase = max_decline = np.nan
        else:
            mean_change = max_increase = max_decline = np.nan
        
        status = "UNSTABLE" if has_instability else "OK"
        summary_data.append({
            'Scenario': scenario_name,
            'Mean_Change_%': mean_change,
            'Max_Increase_%': max_increase,
            'Max_Decline_%': max_decline,
            'Status': status,
        })
    
    summary = pd.DataFrame(summary_data)
    print("\n" + summary[['Scenario', 'Mean_Change_%', 'Status']].to_string(index=False))
    
    # Warnings
    unstable = summary[summary['Status'] == 'UNSTABLE']
    if len(unstable) > 0:
        print("\n" + "="*70)
        print("WARNING  WARNING: Numerical Instabilities Detected")
        print("="*70)
        print("\nSome scenarios show ecological crashes around year 12-14.")
        print("This indicates potential model structural issues.")
        print(f"Affected: {', '.join(unstable['Scenario'].values)}")
    
    return summary


def export_results(results: Dict, params, summary: pd.DataFrame):
    """Export results to files."""
    OUTPUT_DIR.mkdir(exist_ok=True)
    logger.info(f"Exporting to {OUTPUT_DIR}")
    
    # Export summary
    summary_file = OUTPUT_DIR / "simulation_summary.csv"
    summary.to_csv(summary_file, index=False)
    logger.info(f"Summary exported")
    
    # Export biomass trajectories
    for scenario_name, output in results.items():
        if hasattr(output, 'annual_Biomass'):
            biomass_ts = output.annual_Biomass[:, :-1]
        else:
            biomass_ts = output.out_Biomass[::12, :-1]  # Annual aggregation
        
        # Create DataFrame
        group_names = params.model['Group'].values[:len(biomass_ts[0, :])]
        years = np.arange(len(biomass_ts))
        
        df_biomass = pd.DataFrame(biomass_ts, columns=group_names)
        df_biomass.insert(0, 'Year', years)
        
        output_file = OUTPUT_DIR / f"biomass_{scenario_name}.csv"
        df_biomass.to_csv(output_file, index=False)
        logger.info(f"Exported {output_file.name}")


def visualize_results(results: Dict, params):
    """Create comparison visualization."""
    logger.info("Creating visualizations...")
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle('Curonian Lagoon: Ecosim Scenario Comparison (20-year)', fontsize=14, fontweight='bold')
    
    colors = {'baseline': 'black', 'reduce_50': 'green', 'closure': 'blue', 'increase_30': 'red'}
    
    # Panel 1: Total biomass
    ax = axes[0, 0]
    for scenario_name, output in results.items():
        if hasattr(output, 'annual_Biomass'):
            biomass_ts = output.annual_Biomass[:, :-1]
        else:
            biomass_ts = output.out_Biomass[::12, :-1]
        
        total_biomass = biomass_ts.sum(axis=1)
        years = np.arange(len(total_biomass))
        ax.plot(years, total_biomass, marker='o', label=scenario_name.replace('_', ' ').title(),
                color=colors.get(scenario_name, 'gray'), linewidth=2)
    
    ax.set_xlabel('Year')
    ax.set_ylabel('Total Biomass (gC/m²)')
    ax.set_title('Total Ecosystem Biomass')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Panel 2: Mean response
    ax = axes[0, 1]
    scenario_names = []
    mean_changes = []
    
    for scenario_name, output in results.items():
        if hasattr(output, 'annual_Biomass'):
            biomass_ts = output.annual_Biomass[:, :-1]
        else:
            biomass_ts = output.out_Biomass[::12, :-1]
        
        initial = biomass_ts[0, :]
        final = biomass_ts[-1, :]
        valid = (initial > 0) & np.isfinite(final) & (final < 1e10)
        
        if valid.sum() > 0:
            pct_change = ((final[valid] - initial[valid]) / initial[valid] * 100)
            pct_change_filtered = pct_change[np.isfinite(pct_change) & (np.abs(pct_change) < 1e6)]
            if len(pct_change_filtered) > 0:
                scenario_names.append(scenario_name.replace('_', ' ').title())
                mean_changes.append(pct_change_filtered.mean())
    
    if scenario_names:
        ax.bar(range(len(scenario_names)), mean_changes, 
               color=[colors.get(sn.lower().replace(' ', '_'), 'gray') for sn in scenario_names])
        ax.set_ylabel('Mean Biomass Change (%)')
        ax.set_title('Mean Ecosystem Response')
        ax.set_xticks(range(len(scenario_names)))
        ax.set_xticklabels(scenario_names, rotation=15, ha='right')
        ax.axhline(y=0, color='k', linestyle='--', alpha=0.3)
        ax.grid(True, alpha=0.3, axis='y')
    
    # Panel 3: Key species
    ax = axes[1, 0]
    output_baseline = results.get('baseline')
    if output_baseline:
        if hasattr(output_baseline, 'annual_Biomass'):
            biomass_ts = output_baseline.annual_Biomass[:, :-1]
        else:
            biomass_ts = output_baseline.out_Biomass[::12, :-1]
        
        group_names = params.model['Group'].values[:len(biomass_ts[0, :])]
        years = np.arange(len(biomass_ts))
        
        # Plot key species
        key_indices = [i for i, g in enumerate(group_names) if any(x in g.lower() for x in ['fish', 'zooplankton'])]
        for idx in key_indices[:6]:
            ax.plot(years, biomass_ts[:, idx], marker='o', label=group_names[idx], linewidth=2)
        
        ax.set_xlabel('Year')
        ax.set_ylabel('Biomass (gC/m²)')
        ax.set_title('Key Species (Baseline)')
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)
    
    # Panel 4: Response range
    ax = axes[1, 1]
    for idx, scenario_name in enumerate(['baseline', 'reduce_50', 'closure', 'increase_30']):
        output = results.get(scenario_name)
        if output:
            if hasattr(output, 'annual_Biomass'):
                biomass_ts = output.annual_Biomass[:, :-1]
            else:
                biomass_ts = output.out_Biomass[::12, :-1]
            
            initial = biomass_ts[0, :]
            final = biomass_ts[-1, :]
            valid = (initial > 0) & np.isfinite(final) & (final < 1e10) & (final > 0)
            
            if valid.sum() > 0:
                pct_change = ((final[valid] - initial[valid]) / initial[valid] * 100)
                pct_change_filtered = pct_change[np.isfinite(pct_change) & (np.abs(pct_change) < 1e6)]
                if len(pct_change_filtered) > 0:
                    min_val = pct_change_filtered.min()
                    max_val = pct_change_filtered.max()
                    ax.plot([idx, idx], [min_val, max_val], 'o-', linewidth=3, markersize=8,
                           color=colors.get(scenario_name, 'gray'))
    
    ax.set_ylabel('Biomass Change (%)')
    ax.set_title('Species Response Range (min-max)')
    ax.set_xticks(range(4))
    ax.set_xticklabels(['Baseline', '50% Red', 'Closure', '30% Inc'], rotation=15, ha='right')
    ax.axhline(y=0, color='k', linestyle='--', alpha=0.3)
    ax.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    viz_file = OUTPUT_DIR / "ecosim_comparison.png"
    plt.savefig(viz_file, dpi=100, bbox_inches='tight')
    logger.info(f"Visualization saved")
    plt.close()


if __name__ == "__main__":
    print("="*70)
    print("CURONIAN LAGOON ECOSIM EXAMPLE")
    print("="*70)
    
    print("\nWorkflow:")
    print("1. Load Ecopath model from EwE database")
    print("2. Create 4 management scenarios")
    print("3. Run 20-year simulations")
    print("4. Compare ecosystem responses")
    
    # Load data
    params, landings = load_model_and_data()
    
    # Create scenarios
    scenarios, model, params = create_ecosim_scenarios(params)
    
    # Run simulations
    results = run_simulations(scenarios, model, params)
    
    # Analyze results
    summary = analyze_results(results, params)
    
    # Export
    export_results(results, params, summary)
    
    # Visualize
    visualize_results(results, params)
    
    print("\n" + "="*70)
    print("ANALYSIS COMPLETE")
    print("="*70)
    print(f"\nResults saved to: {OUTPUT_DIR.absolute()}")
    print("  - simulation_summary.csv: Scenario comparison")
    print("  - biomass_*.csv: Biomass trajectories")
    print("  - ecosim_comparison.png: Visualization")
    print("="*70)
