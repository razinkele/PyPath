"""
Curonian Lagoon Multistanza Ecosim Scenarios

Implements complete Ecosim workflow with multi-stanza age-structured groups:
1. Load Ecopath model
2. Setup multi-stanza configuration
3. Create management scenarios (4 types)
4. Run 20-year Ecosim simulations
5. Analyze ecosystem responses
6. Visualize and export results

Management Scenarios:
  - Baseline: Current fishing effort maintained
  - 50% Reduction: Gradual fishing reduction over 10 years
  - Complete Closure: Phase-out fishing over 10 years
  - 30% Increase: Gradual fishing increase over 10 years
"""

import logging
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Dict, Tuple
import sys

from pypath.io.ewemdb import read_ewemdb
from pypath.core.ecopath import rpath
from pypath.core.ecosim import rsim_scenario, rsim_run
from pypath.core.stanzas import EcosimStanzaParams, StanzaGroup, StanzaIndividual, von_bertalanffy_weight

# Setup logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

OUTPUT_DIR = Path("curonian_multistanza_ecosim_results")
OUTPUT_DIR.mkdir(exist_ok=True)

# Import from multistanza example
sys.path.insert(0, str(Path(__file__).parent))
try:
    from curonian_multistanza_example import (
        create_stanza_definitions,
        setup_multistanza_groups,
        distribute_biomass_across_stanzas
    )
except ImportError:
    logger.error("Could not import multistanza functions. Ensure curonian_multistanza_example.py is in scripts/")
    sys.exit(1)


def create_ecosim_scenarios(model, params) -> Dict:
    """
    Create 4 fishing management scenarios.
    
    Parameters
    ----------
    model : Rpath
        Balanced Ecopath model
    params : RpathParams
        Ecopath parameters
        
    Returns
    -------
    Dict
        Dictionary of scenarios by name
    """
    logger.info("Creating Ecosim scenarios")
    scenarios = {}
    
    # Scenario 1: Baseline (no change)
    logger.info("Scenario 1: Baseline - fishing effort maintained")
    scenario1 = rsim_scenario(model, params, years=np.arange(1, 21))
    scenarios["baseline"] = scenario1
    
    # Scenario 2: 50% Reduction
    logger.info("Scenario 2: 50% Fishing Reduction - gradual over 10 years")
    scenario2 = rsim_scenario(model, params, years=np.arange(1, 21))
    
    # Apply gradual reduction
    for year_idx in range(min(10, scenario2.fishing.ForcedFRate.shape[0])):
        reduction_fraction = (year_idx + 1) / 10.0  # 0 to 1 over 10 years
        reduction_amount = reduction_fraction * 0.5  # 0 to 50%
        scenario2.fishing.ForcedFRate[year_idx, :] *= (1.0 - reduction_amount)
    
    scenarios["reduce_50"] = scenario2
    
    # Scenario 3: Complete Closure
    logger.info("Scenario 3: Complete Fishing Closure - phased over 10 years")
    scenario3 = rsim_scenario(model, params, years=np.arange(1, 21))
    
    # Phase out fishing over 10 years
    for year_idx in range(min(10, scenario3.fishing.ForcedFRate.shape[0])):
        closure_fraction = (year_idx + 1) / 10.0  # 0 to 1 over 10 years
        scenario3.fishing.ForcedFRate[year_idx, :] *= (1.0 - closure_fraction)
    
    # After year 10, no fishing
    if scenario3.fishing.ForcedFRate.shape[0] > 10:
        scenario3.fishing.ForcedFRate[10:, :] = 0.0
    
    scenarios["closure"] = scenario3
    
    # Scenario 4: 30% Increase
    logger.info("Scenario 4: 30% Fishing Increase - gradual over 10 years")
    scenario4 = rsim_scenario(model, params, years=np.arange(1, 21))
    
    # Apply gradual increase
    for year_idx in range(min(10, scenario4.fishing.ForcedFRate.shape[0])):
        increase_fraction = (year_idx + 1) / 10.0  # 0 to 1 over 10 years
        increase_amount = increase_fraction * 0.30  # 0 to 30%
        scenario4.fishing.ForcedFRate[year_idx, :] *= (1.0 + increase_amount)
    
    scenarios["increase_30"] = scenario4
    
    return scenarios


def run_scenarios(scenarios: Dict, scenario_names=None, years: int = 20) -> Dict:
    """
    Run all scenarios and collect results.
    
    Parameters
    ----------
    scenarios : Dict
        Scenarios to run
    scenario_names : list, optional
        Specific scenarios to run (default: all)
    years : int
        Number of years to simulate
        
    Returns
    -------
    Dict
        Results {scenario_name: biomass_timeseries}
    """
    if scenario_names is None:
        scenario_names = list(scenarios.keys())
    
    results = {}
    
    for scenario_name in scenario_names:
        if scenario_name not in scenarios:
            logger.warning(f"Scenario {scenario_name} not found")
            continue
        
        logger.info(f"Running {scenario_name} scenario...")
        try:
            scenario = scenarios[scenario_name]
            sim_result = rsim_run(scenario)
            results[scenario_name] = sim_result.out_Biomass
            logger.info(f"  Complete: {sim_result.out_Biomass.shape[0]} timesteps x {sim_result.out_Biomass.shape[1]} groups")
        except Exception as e:
            logger.error(f"  Error in {scenario_name}: {e}")
    
    return results


def create_scenario_comparison_plots(results: Dict, params, output_dir: Path = OUTPUT_DIR):
    """Create comprehensive scenario comparison plots."""
    logger.info("Creating scenario comparison plots")
    
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle("Curonian Lagoon Multistanza: Ecosim Scenario Comparison (20-year)", 
                 fontsize=16, fontweight="bold")
    
    colors = {"baseline": "black", "reduce_50": "green", "closure": "red", "increase_30": "orange"}
    
    # Plot 1: Total ecosystem biomass
    ax = axes[0, 0]
    for scenario_name, biomass_ts in results.items():
        total_biomass = biomass_ts.sum(axis=1)
        ax.plot(total_biomass, label=scenario_name, linewidth=2.5, 
                color=colors.get(scenario_name, "gray"))
    
    ax.set_xlabel("Year", fontsize=11)
    ax.set_ylabel("Total Biomass (t·km⁻²)", fontsize=11)
    ax.set_title("Total Ecosystem Biomass Over Time")
    ax.legend(loc="best")
    ax.grid(True, alpha=0.3)
    
    # Plot 2: Demersal fish (Group 8)
    ax = axes[0, 1]
    for scenario_name, biomass_ts in results.items():
        demersal = biomass_ts[:, 8]
        ax.plot(demersal, label=scenario_name, linewidth=2.5,
                color=colors.get(scenario_name, "gray"))
    
    ax.set_xlabel("Year", fontsize=11)
    ax.set_ylabel("Demersal Fish Biomass (t·km⁻²)", fontsize=11)
    ax.set_title("Demersal Fish (Age-Structured) Dynamics")
    ax.legend(loc="best")
    ax.grid(True, alpha=0.3)
    
    # Plot 3: Predatory fish (Group 14)
    ax = axes[1, 0]
    for scenario_name, biomass_ts in results.items():
        predatory = biomass_ts[:, 14]
        ax.plot(predatory, label=scenario_name, linewidth=2.5,
                color=colors.get(scenario_name, "gray"))
    
    ax.set_xlabel("Year", fontsize=11)
    ax.set_ylabel("Predatory Fish Biomass (t·km⁻²)", fontsize=11)
    ax.set_title("Predatory Fish (Age-Structured) Dynamics")
    ax.legend(loc="best")
    ax.grid(True, alpha=0.3)
    
    # Plot 4: Percent change from baseline
    ax = axes[1, 1]
    baseline_final = results["baseline"][-1, :].sum()
    
    scenario_names_list = [s for s in results.keys() if s != "baseline"]
    scenario_names_list = ["baseline"] + scenario_names_list
    
    changes = []
    for scenario_name in scenario_names_list:
        if scenario_name in results:
            final_biomass = results[scenario_name][-1, :].sum()
            pct_change = ((final_biomass - baseline_final) / baseline_final) * 100 if scenario_name != "baseline" else 0
            changes.append(pct_change)
    
    bars = ax.bar(scenario_names_list, changes, 
                  color=[colors.get(s, "gray") for s in scenario_names_list],
                  alpha=0.7, edgecolor="black", linewidth=1.5)
    
    ax.set_ylabel("Percent Change from Baseline (%)", fontsize=11)
    ax.set_title("Final Ecosystem Biomass Change by Scenario (Year 20)")
    ax.axhline(y=0, color="black", linestyle="--", linewidth=1.5)
    ax.grid(True, alpha=0.3, axis="y")
    
    # Add value labels on bars
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.1f}%', ha='center', va='bottom' if height >= 0 else 'top',
                fontsize=10, fontweight="bold")
    
    plt.tight_layout()
    plot_path = output_dir / "multistanza_ecosim_scenarios.png"
    plt.savefig(plot_path, dpi=300, bbox_inches="tight")
    logger.info(f"Saved plot: {plot_path}")
    plt.close()


def export_scenario_results(results: Dict, params, output_dir: Path = OUTPUT_DIR):
    """Export scenario results to CSV files."""
    logger.info("Exporting scenario results")
    
    group_names = list(params.model["Group"])
    
    for scenario_name, biomass_ts in results.items():
        # Create DataFrame - biomass_ts has 241 rows x n_cols columns
        # First 20 columns are groups, remaining are fleet/other
        n_groups = len(group_names)
        
        if biomass_ts.shape[1] > n_groups:
            # Include all columns
            all_cols = group_names + [f"Fleet_{i}" for i in range(biomass_ts.shape[1] - n_groups)]
        else:
            all_cols = group_names[:biomass_ts.shape[1]]
        
        df = pd.DataFrame(biomass_ts, columns=all_cols)
        df.insert(0, "Year", range(1, len(df) + 1))
        
        # Export CSV
        filepath = output_dir / f"ecosim_multistanza_{scenario_name}.csv"
        df.to_csv(filepath, index=False)
        logger.info(f"  Saved: {filepath}")
        
        # Export summary statistics
        summary_data = []
        for year in range(1, len(df) + 1):
            total_biomass = df.iloc[year-1, 1:1+n_groups].sum()
            summary_data.append({
                "Year": year,
                "Scenario": scenario_name,
                "Total_Biomass": total_biomass,
                "Demersal_Fish": df.iloc[year-1, 9] if 9 < len(df.columns) else np.nan,
                "Predatory_Fish": df.iloc[year-1, 15] if 15 < len(df.columns) else np.nan,
            })
        
        summary_df = pd.DataFrame(summary_data)
        summary_path = output_dir / f"summary_{scenario_name}.csv"
        summary_df.to_csv(summary_path, index=False)


def print_scenario_summary(results: Dict, params):
    """Print summary of scenario results."""
    print("\n" + "=" * 90)
    print("ECOSIM SCENARIO RESULTS SUMMARY")
    print("=" * 90)
    
    print(f"\n{'Scenario':<20} {'Year 1 Biomass':<20} {'Year 20 Biomass':<20} {'Change (%)':<15}")
    print("-" * 75)
    
    baseline_y1 = results["baseline"][0, :].sum()
    baseline_y20 = results["baseline"][-1, :].sum()
    
    for scenario_name in sorted(results.keys()):
        biomass_ts = results[scenario_name]
        y1_total = biomass_ts[0, :].sum()
        y20_total = biomass_ts[-1, :].sum()
        pct_change = ((y20_total - baseline_y20) / baseline_y20) * 100 if scenario_name != "baseline" else 0
        
        print(f"{scenario_name:<20} {y1_total:<20.2f} {y20_total:<20.2f} {pct_change:>14.1f}")
    
    print("\n" + "=" * 90)


def main():
    """Main execution."""
    print("\n" + "=" * 90)
    print("CURONIAN LAGOON MULTISTANZA ECOSIM - SCENARIO ANALYSIS")
    print("=" * 90 + "\n")
    
    # Step 1: Load model
    logger.info("Step 1: Loading Ecopath model")
    model_path = r"C:\Users\DELL\OneDrive - ku.lt\ECOPATH\ECOPATH paper\curonian pelagic.eweaccdb"
    params = read_ewemdb(model_path)
    logger.info(f"Loaded model with {len(params.model)} functional groups")
    
    # Step 2: Setup multi-stanza
    logger.info("Step 2: Setting up multi-stanza configuration")
    stanza_defs = create_stanza_definitions()
    stanzas = setup_multistanza_groups(params, stanza_defs)
    logger.info(f"Created {len(stanzas.stanza_individuals)} stanza individuals")
    
    # Step 3: Create Ecopath model
    logger.info("Step 3: Balancing Ecopath model")
    model = rpath(params)
    logger.info(f"Ecopath balanced with {model.NUM_GROUPS} groups")
    
    # Step 4: Create scenarios
    logger.info("Step 4: Creating management scenarios")
    scenarios = create_ecosim_scenarios(model, params)
    logger.info(f"Created {len(scenarios)} scenarios")
    
    # Step 5: Run simulations
    logger.info("Step 5: Running Ecosim simulations")
    results = run_scenarios(scenarios)
    
    if not results:
        logger.error("No scenario results generated. Cannot continue.")
        return
    
    # Step 6: Visualize
    logger.info("Step 6: Creating visualizations")
    create_scenario_comparison_plots(results, params)
    
    # Step 7: Export
    logger.info("Step 7: Exporting results")
    export_scenario_results(results, params)
    
    # Step 8: Summary
    print_scenario_summary(results, params)
    
    print("\nResults saved to:", OUTPUT_DIR.absolute())
    logger.info("Execution complete")


if __name__ == "__main__":
    main()


