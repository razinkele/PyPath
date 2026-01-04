"""
Demonstration of Advanced Ecosim Features

This script demonstrates how to use:
1. State-variable forcing (forcing biomass to observations)
2. Dynamic diet rewiring (adaptive foraging)
3. Combined usage of both features

Run this script to see practical examples of the new functionality.
"""

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from pypath.core.forcing import (
    StateForcing,
    create_biomass_forcing,
    create_diet_rewiring,
    create_recruitment_forcing,
)


def demo_biomass_forcing():
    """Demonstrate biomass forcing with seasonal pattern."""
    print("\n" + "=" * 70)
    print("DEMO 1: Biomass Forcing - Seasonal Phytoplankton Pattern")
    print("=" * 70)

    # Create seasonal phytoplankton biomass data
    years = np.linspace(2000, 2005, 61)  # Monthly data for 5 years
    seasonal_biomass = 15.0 + 5.0 * np.sin(2 * np.pi * years)  # Seasonal cycle

    # Create forcing
    forcing = create_biomass_forcing(
        group_idx=0,  # Phytoplankton
        observed_biomass=seasonal_biomass,
        years=years,
        mode="replace",
        interpolate=True,
    )

    print("Created biomass forcing for group 0 (Phytoplankton)")
    print(f"  Time range: {years[0]} - {years[-1]}")
    print(f"  Data points: {len(years)}")
    print(
        f"  Biomass range: {seasonal_biomass.min():.2f} - {seasonal_biomass.max():.2f} t/km²"
    )

    # Test interpolation at arbitrary times
    test_years = [2000.5, 2001.0, 2002.5, 2003.0]
    print("\nInterpolated values:")
    for year in test_years:
        value = forcing.functions[0].get_value(year)
        print(f"  Year {year}: {value:.2f} t/km²")

    # Plot if matplotlib available
    try:
        fig, ax = plt.subplots(figsize=(10, 5))
        ax.plot(years, seasonal_biomass, "b-", linewidth=2, label="Forced Biomass")
        ax.scatter(
            [2000.5, 2001.0, 2002.5, 2003.0],
            [forcing.functions[0].get_value(y) for y in test_years],
            color="red",
            s=100,
            zorder=5,
            label="Interpolated Values",
        )
        ax.set_xlabel("Year")
        ax.set_ylabel("Biomass (t/km²)")
        ax.set_title("Phytoplankton Biomass Forcing - Seasonal Pattern")
        ax.legend()
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig("demo_biomass_forcing.png", dpi=150)
        print("\n[OK] Plot saved to: demo_biomass_forcing.png")
        plt.close()
    except Exception as e:
        print(f"\n(Plot skipped: {e})")


def demo_recruitment_forcing():
    """Demonstrate recruitment forcing with pulses."""
    print("\n" + "=" * 70)
    print("DEMO 2: Recruitment Forcing - Strong Year-Class Events")
    print("=" * 70)

    # Strong recruitment in specific years
    recruitment_data = {
        2000: 1.0,  # Normal
        2002: 3.0,  # Strong year-class
        2004: 0.5,  # Weak year-class
        2006: 1.0,  # Normal
        2008: 2.5,  # Strong year-class
        2010: 1.0,  # Normal
    }

    forcing = create_recruitment_forcing(
        group_idx=3,  # Example: Herring
        recruitment_multiplier=recruitment_data,
        interpolate=False,  # Discrete events
    )

    print("Created recruitment forcing for group 3 (Herring)")
    print("  Recruitment multipliers:")
    for year, mult in sorted(recruitment_data.items()):
        strength = "STRONG" if mult > 1.5 else "weak" if mult < 1.0 else "normal"
        print(f"    {year}: {mult}x ({strength})")

    # Plot
    try:
        years = np.array(sorted(recruitment_data.keys()))
        multipliers = np.array([recruitment_data[y] for y in years])

        fig, ax = plt.subplots(figsize=(10, 5))
        ax.bar(years, multipliers, width=0.8, alpha=0.7, edgecolor="black")
        ax.axhline(
            y=1.0, color="r", linestyle="--", linewidth=2, label="Normal Recruitment"
        )
        ax.set_xlabel("Year")
        ax.set_ylabel("Recruitment Multiplier")
        ax.set_title("Recruitment Forcing - Strong and Weak Year-Classes")
        ax.legend()
        ax.grid(True, alpha=0.3, axis="y")
        plt.tight_layout()
        plt.savefig("demo_recruitment_forcing.png", dpi=150)
        print("\n[OK] Plot saved to: demo_recruitment_forcing.png")
        plt.close()
    except Exception as e:
        print(f"\n(Plot skipped: {e})")


def demo_diet_rewiring():
    """Demonstrate dynamic diet rewiring."""
    print("\n" + "=" * 70)
    print("DEMO 3: Dynamic Diet Rewiring - Prey Switching")
    print("=" * 70)

    # Create diet rewiring with moderate switching
    diet_rewiring = create_diet_rewiring(
        switching_power=2.5,
        min_proportion=0.001,
        update_interval=12,  # Annual updates
    )

    print("Created diet rewiring configuration:")
    print(f"  Switching power: {diet_rewiring.switching_power}")
    print(f"  Minimum proportion: {diet_rewiring.min_proportion}")
    print(f"  Update interval: {diet_rewiring.update_interval} months")

    # Set up example diet matrix (3 prey, 1 predator)
    base_diet = np.array(
        [
            [0.5],  # Prey 0: Herring (50%)
            [0.3],  # Prey 1: Sprat (30%)
            [0.2],  # Prey 2: Zooplankton (20%)
        ]
    )

    diet_rewiring.initialize(base_diet)

    print("\nBase diet composition:")
    prey_names = ["Herring", "Sprat", "Zooplankton"]
    for i, name in enumerate(prey_names):
        print(f"  {name}: {base_diet[i, 0] * 100:.1f}%")

    # Simulate different biomass scenarios
    scenarios = {
        "Normal": np.array([10.0, 10.0, 10.0, 0.0]),
        "Herring Collapse": np.array([2.0, 10.0, 10.0, 0.0]),
        "Sprat Bloom": np.array([10.0, 30.0, 10.0, 0.0]),
        "Zoo Dominant": np.array([10.0, 10.0, 50.0, 0.0]),
    }

    print("\nDiet adjustments under different scenarios:")
    print(f"{'Scenario':<20} {'Herring':<12} {'Sprat':<12} {'Zooplankton':<12}")
    print("-" * 60)

    results = {}
    for scenario_name, biomass in scenarios.items():
        diet_rewiring.current_diet = base_diet.copy()  # Reset
        new_diet = diet_rewiring.update_diet(biomass)
        results[scenario_name] = new_diet.copy()

        print(f"{scenario_name:<20} ", end="")
        for i in range(3):
            change = (new_diet[i, 0] - base_diet[i, 0]) * 100
            arrow = "^" if change > 0.5 else "v" if change < -0.5 else "-"
            print(f"{new_diet[i, 0] * 100:5.1f}% {arrow:<5} ", end="")
        print()

    # Plot
    try:
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        axes = axes.flatten()

        for idx, (scenario_name, biomass) in enumerate(scenarios.items()):
            ax = axes[idx]

            diet = results[scenario_name][:, 0]
            base = base_diet[:, 0]

            x = np.arange(len(prey_names))
            width = 0.35

            ax.bar(x - width / 2, base * 100, width, label="Base Diet", alpha=0.7)
            ax.bar(x + width / 2, diet * 100, width, label="New Diet", alpha=0.7)

            ax.set_ylabel("Diet Proportion (%)")
            ax.set_title(f"{scenario_name}\nBiomass: {biomass[:3]}")
            ax.set_xticks(x)
            ax.set_xticklabels(prey_names, rotation=45, ha="right")
            ax.legend()
            ax.grid(True, alpha=0.3, axis="y")
            ax.set_ylim(0, 80)

        plt.tight_layout()
        plt.savefig("demo_diet_rewiring.png", dpi=150)
        print("\n[OK] Plot saved to: demo_diet_rewiring.png")
        plt.close()
    except Exception as e:
        print(f"\n(Plot skipped: {e})")


def demo_combined_usage():
    """Demonstrate using forcing and diet rewiring together."""
    print("\n" + "=" * 70)
    print("DEMO 4: Combined Usage - Climate Change Scenario")
    print("=" * 70)

    # Climate change scenario: increasing primary production
    pp_forcing = StateForcing()
    pp_forcing.add_forcing(
        group_idx=0,  # Phytoplankton
        variable="primary_production",
        time_series={2000: 1.0, 2020: 1.2, 2040: 1.4, 2060: 1.6, 2080: 1.8, 2100: 2.0},
        mode="multiply",
        interpolate=True,
    )

    print("Climate Change Scenario:")
    print("  Primary production forcing:")
    test_years = [2000, 2020, 2040, 2060, 2080, 2100]
    for year in test_years:
        value = pp_forcing.functions[0].get_value(year)
        increase = (value - 1.0) * 100
        print(f"    {year}: {value:.2f}x baseline (+{increase:.0f}%)")

    # Strong prey switching (climate stress)
    diet_rewiring = create_diet_rewiring(
        switching_power=3.5,  # Strong adaptive response
        update_interval=12,
    )

    print("\n  Diet rewiring:")
    print(f"    Switching power: {diet_rewiring.switching_power} (STRONG)")
    print("    Adaptive foraging enabled")

    print("\nThis scenario simulates:")
    print("  - Increasing primary production due to climate warming")
    print("  - Strong prey switching as species shift distributions")
    print("  - Potential for regime shifts and alternative stable states")

    print("\nUsage in simulation:")
    print("  result = rsim_run_advanced(")
    print("      scenario,")
    print("      state_forcing=pp_forcing,")
    print("      diet_rewiring=diet_rewiring,")
    print("      verbose=True")
    print("  )")


def demo_fishing_moratorium():
    """Demonstrate fishing moratorium scenario."""
    print("\n" + "=" * 70)
    print("DEMO 5: Fishing Moratorium - Recovery Period")
    print("=" * 70)

    # Fishing ban from 2010-2015
    forcing = StateForcing()
    forcing.add_forcing(
        group_idx=5,  # Target species (e.g., Cod)
        variable="fishing_mortality",
        time_series={
            2000: 0.3,  # Pre-ban fishing
            2010: 0.0,  # Ban starts
            2015: 0.0,  # Ban ends
            2020: 0.15,  # Reduced fishing resumes
        },
        mode="replace",
        interpolate=True,
    )

    print("Fishing Moratorium Scenario:")
    print("  Target: Group 5 (Cod)")
    print("  Timeline:")
    print("    2000-2009: Normal fishing (F = 0.30)")
    print("    2010-2015: Complete ban (F = 0.00)")
    print("    2016-2020: Reduced fishing (F = 0.15)")

    # Plot
    try:
        years = np.linspace(2000, 2020, 241)
        f_values = [forcing.functions[0].get_value(y) for y in years]

        fig, ax = plt.subplots(figsize=(10, 5))
        ax.plot(years, f_values, "b-", linewidth=2)
        ax.fill_between(
            [2010, 2015], 0, 0.35, alpha=0.3, color="green", label="Moratorium Period"
        )
        ax.set_xlabel("Year")
        ax.set_ylabel("Fishing Mortality (F)")
        ax.set_title("Fishing Moratorium - 5-Year Recovery Period")
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.set_ylim(0, 0.35)
        plt.tight_layout()
        plt.savefig("demo_fishing_moratorium.png", dpi=150)
        print("\n[OK] Plot saved to: demo_fishing_moratorium.png")
        plt.close()
    except Exception as e:
        print(f"\n(Plot skipped: {e})")


def main():
    """Run all demonstrations."""
    print("\n" + "=" * 70)
    print("PyPath Advanced Ecosim Features - Interactive Demonstrations")
    print("=" * 70)
    print("\nThis script demonstrates the new advanced features:")
    print("  1. State-variable forcing (biomass, recruitment, fishing)")
    print("  2. Dynamic diet rewiring (adaptive foraging)")
    print("  3. Multiple forcing modes (replace, add, multiply)")
    print("  4. Temporal interpolation")
    print("  5. Realistic ecological scenarios")

    # Run all demos
    demo_biomass_forcing()
    demo_recruitment_forcing()
    demo_diet_rewiring()
    demo_combined_usage()
    demo_fishing_moratorium()

    print("\n" + "=" * 70)
    print("All demonstrations complete!")
    print("=" * 70)
    print("\nGenerated files:")
    for fname in [
        "demo_biomass_forcing.png",
        "demo_recruitment_forcing.png",
        "demo_diet_rewiring.png",
        "demo_fishing_moratorium.png",
    ]:
        if Path(fname).exists():
            print(f"  [OK] {fname}")

    print("\nFor detailed documentation, see:")
    print("  - ADVANCED_ECOSIM_FEATURES.md")
    print("  - FORCING_IMPLEMENTATION_SUMMARY.md")

    print("\nTo run tests:")
    print("  pytest tests/test_forcing.py tests/test_diet_rewiring.py -v")


if __name__ == "__main__":
    main()
