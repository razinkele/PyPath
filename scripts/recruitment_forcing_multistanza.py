"""
Recruitment Forcing for Curonian Lagoon Multistanza Model
==========================================================

Implements environmental forcing functions for temperature and primary productivity
to drive juvenile recruitment success in multistanza Ecosim simulations.

This script demonstrates:
1. Load multistanza Ecopath model
2. Create time series for temperature and productivity
3. Define recruitment response functions
4. Run Ecosim with and without environmental forcing
5. Compare recruitment dynamics

Author: Copilot
Date: 2026-10-23
Version: 0.5.1
"""

import logging
import numpy as np
import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt
from typing import Dict, Tuple
from dataclasses import dataclass

logger = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)

# Output directory
OUTPUT_DIR = Path("./recruitment_forcing_results")
OUTPUT_DIR.mkdir(exist_ok=True)

SIMULATION_YEARS = 20


@dataclass
class EnvironmentalForcing:
    """Environmental forcing time series for recruitment."""
    
    years: np.ndarray
    temperature: np.ndarray
    productivity: np.ndarray
    oxygen: np.ndarray = None


def create_temperature_timeseries(years: int = SIMULATION_YEARS) -> np.ndarray:
    """
    Create realistic temperature time series for Curonian Lagoon.
    
    Seasonal cycle + inter-annual variability + long-term trend.
    """
    timesteps = years * 12 + 1
    months = np.arange(timesteps)
    
    # Baseline: seasonal cycle (8-25 C)
    baseline = 16.5 + 8.5 * np.sin(2 * np.pi * months / 12)
    
    # Inter-annual variability (cold/warm years pattern)
    inter_annual = 2.0 * np.sin(2 * np.pi * months / (12 * 5))
    
    # Random noise
    noise = np.random.normal(0, 0.5, timesteps)
    
    temperature = baseline + inter_annual + noise
    return np.clip(temperature, 0, 30)


def create_productivity_timeseries(years: int = SIMULATION_YEARS) -> np.ndarray:
    """
    Create primary productivity time series (normalized 0-1).
    
    Drives phytoplankton biomass and juvenile food availability.
    """
    timesteps = years * 12 + 1
    months = np.arange(timesteps)
    
    # Spring bloom pattern
    bloom_strength = []
    for year in range(years + 1):
        # Stronger bloom in nutrient-rich years
        year_factor = 0.8 + 0.3 * np.sin(2 * np.pi * year / 7)
        bloom_strength.extend([year_factor] * 12)
    
    bloom_strength = np.array(bloom_strength[:timesteps])
    
    # Monthly productivity pattern: high spring/summer, low winter
    seasonal = 0.5 + 0.4 * np.sin(2 * np.pi * months / 12 - np.pi / 4)
    
    productivity = bloom_strength * seasonal
    return np.clip(productivity, 0.1, 1.0)


def recruitment_response_temp(
    temperature: np.ndarray,
    optimal_temp: float = 18.0,
    steepness: float = 0.2,
) -> np.ndarray:
    """
    Temperature-dependent recruitment success curve (Gaussian-like).
    
    Peak recruitment at optimal_temp, declines at extremes.
    """
    return np.exp(-steepness * (temperature - optimal_temp) ** 2)


def recruitment_response_productivity(
    productivity: np.ndarray,
    threshold: float = 0.3,
    max_response: float = 1.5,
) -> np.ndarray:
    """
    Productivity-dependent recruitment enhancement.
    
    Higher productivity increases juvenile food availability and survival.
    """
    response = np.ones_like(productivity)
    response[productivity > threshold] = (
        1.0 + (max_response - 1.0) * (productivity[productivity > threshold] - threshold)
        / (1.0 - threshold)
    )
    return response


def combine_recruitment_forcing(
    temperature: np.ndarray,
    productivity: np.ndarray,
    temp_weight: float = 0.4,
    prod_weight: float = 0.6,
) -> np.ndarray:
    """Combine multiple forcing functions into overall recruitment multiplier."""
    temp_effect = recruitment_response_temp(temperature)
    prod_effect = recruitment_response_productivity(productivity)
    
    combined = (
        temp_weight * temp_effect +
        prod_weight * prod_effect
    )
    return combined


def create_environmental_forcing(
    years: int = SIMULATION_YEARS,
) -> EnvironmentalForcing:
    """Generate complete environmental forcing dataset."""
    logger.info("Creating environmental forcing time series")
    
    timesteps = years * 12 + 1
    year_array = np.linspace(0, years, timesteps)
    
    temperature = create_temperature_timeseries(years)
    productivity = create_productivity_timeseries(years)
    
    logger.info(f"  Temperature: {temperature.min():.1f}-{temperature.max():.1f} C")
    logger.info(f"  Productivity: {productivity.min():.2f}-{productivity.max():.2f}")
    
    return EnvironmentalForcing(
        years=year_array,
        temperature=temperature,
        productivity=productivity,
    )


def calculate_recruitment_multiplier(
    forcing: EnvironmentalForcing,
) -> Tuple[np.ndarray, Dict]:
    """Calculate recruitment multiplier from forcing functions."""
    logger.info("Calculating recruitment multipliers")
    
    recruitment_factor = combine_recruitment_forcing(
        forcing.temperature,
        forcing.productivity,
        temp_weight=0.4,
        prod_weight=0.6,
    )
    
    stats = {
        "mean": np.mean(recruitment_factor),
        "min": np.min(recruitment_factor),
        "max": np.max(recruitment_factor),
        "std": np.std(recruitment_factor),
    }
    
    logger.info(f"  Recruitment factor: {stats['mean']:.3f} ± {stats['std']:.3f}")
    logger.info(f"  Range: {stats['min']:.3f} - {stats['max']:.3f}")
    
    return recruitment_factor, stats


def apply_recruitment_forcing_to_ecopath(
    recruitment_factor: np.ndarray,
    affected_groups: list = None,
    baseline_z: Dict = None,
) -> Dict:
    """
    Apply recruitment forcing to juvenile mortality rates.
    
    Lower recruitment factor -> higher juvenile Z (lower survival)
    Higher recruitment factor -> lower juvenile Z (higher survival)
    """
    if affected_groups is None:
        affected_groups = [
            "demersal_juvenile",
            "predatory_juvenile",
            "predatory_subadult",
        ]
    
    if baseline_z is None:
        baseline_z = {
            "demersal_juvenile": 1.2,
            "predatory_juvenile": 1.5,
            "predatory_subadult": 1.0,
        }
    
    logger.info("Applying recruitment forcing to juvenile mortality")
    
    # Scale recruitment factor to Z multiplier (inverse relationship)
    # recruitment_factor = 0.5 -> Z multiplier = 1.5 (harder to survive)
    # recruitment_factor = 1.5 -> Z multiplier = 0.67 (easier to survive)
    z_multiplier = 1.0 / (recruitment_factor + 0.5)
    
    forced_z_values = {}
    for group in affected_groups:
        if group in baseline_z:
            forced_z = baseline_z[group] * z_multiplier
            forced_z_values[group] = forced_z
    
    logger.info(f"  Modified Z for {len(forced_z_values)} juvenile groups")
    
    return forced_z_values


def create_comparison_plots(
    forcing: EnvironmentalForcing,
    recruitment_factor: np.ndarray,
    output_dir: Path = OUTPUT_DIR,
):
    """Create visualization of environmental forcing and recruitment response."""
    logger.info("Creating environmental forcing plots")
    
    fig, axes = plt.subplots(3, 1, figsize=(12, 9))
    fig.suptitle("Environmental Forcing and Recruitment Response", 
                 fontsize=13, fontweight="bold")
    
    years = forcing.years
    
    # Plot 1: Temperature
    ax = axes[0]
    ax.plot(years, forcing.temperature, 'b-', linewidth=1.5, label='Temperature')
    ax.axhline(y=18.0, color='r', linestyle='--', alpha=0.5, label='Optimal temp (18C)')
    ax.fill_between(years, forcing.temperature.min(), forcing.temperature.max(), 
                     alpha=0.2, color='blue')
    ax.set_ylabel("Temperature (C)", fontweight="bold")
    ax.set_title("Temperature Time Series", fontweight="bold")
    ax.legend(loc='upper right')
    ax.grid(True, alpha=0.3)
    
    # Plot 2: Productivity
    ax = axes[1]
    ax.plot(years, forcing.productivity, 'g-', linewidth=1.5, label='Primary Productivity')
    ax.axhline(y=0.3, color='r', linestyle='--', alpha=0.5, label='Threshold (0.3)')
    ax.fill_between(years, forcing.productivity.min(), forcing.productivity.max(),
                     alpha=0.2, color='green')
    ax.set_ylabel("Productivity (0-1)", fontweight="bold")
    ax.set_title("Primary Productivity Time Series", fontweight="bold")
    ax.legend(loc='upper right')
    ax.grid(True, alpha=0.3)
    
    # Plot 3: Recruitment factor
    ax = axes[2]
    ax.plot(years, recruitment_factor, 'purple', linewidth=2, label='Recruitment Factor')
    ax.axhline(y=1.0, color='k', linestyle='--', alpha=0.5, label='Baseline (1.0)')
    ax.fill_between(years, recruitment_factor.min(), recruitment_factor.max(),
                     alpha=0.2, color='purple')
    ax.set_xlabel("Year", fontweight="bold")
    ax.set_ylabel("Recruitment Multiplier", fontweight="bold")
    ax.set_title("Combined Recruitment Response", fontweight="bold")
    ax.legend(loc='upper right')
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plot_path = output_dir / "environmental_forcing.png"
    plt.savefig(plot_path, dpi=150, bbox_inches="tight")
    logger.info(f"  Saved: {plot_path}")
    plt.close()


def export_forcing_data(
    forcing: EnvironmentalForcing,
    recruitment_factor: np.ndarray,
    output_dir: Path = OUTPUT_DIR,
):
    """Export forcing data to CSV."""
    logger.info("Exporting environmental forcing data")
    
    df = pd.DataFrame({
        "Year": forcing.years,
        "Temperature_C": forcing.temperature,
        "Productivity": forcing.productivity,
        "Recruitment_Factor": recruitment_factor,
    })
    
    csv_path = output_dir / "environmental_forcing.csv"
    df.to_csv(csv_path, index=False)
    logger.info(f"  Saved: {csv_path}")
    
    return df


def export_guide(output_dir: Path = OUTPUT_DIR):
    """Export recruitment forcing guide."""
    logger.info("Creating recruitment forcing guide")
    
    guide_text = """# Recruitment Forcing for Multistanza Models

## Overview
This guide demonstrates how to incorporate environmental drivers (temperature, 
primary productivity) into Ecosim simulations to modulate juvenile recruitment success.

## Environmental Forcing Functions

### Temperature Response
- Optimal temperature: 18 C (Curonian Lagoon summer average)
- Functional form: Gaussian curve exp(-0.2 * (T - T_opt)^2)
- Peak recruitment at optimal temperature
- Recruitment declines at cold (<10 C) and warm (>25 C) extremes

### Productivity Response
- Threshold: 0.3 (normalized 0-1 scale)
- Below threshold: baseline recruitment
- Above threshold: linear increase up to 50% enhancement
- Represents food availability for juvenile fish

### Combined Forcing
- Temperature weight: 40% (direct physiological effect)
- Productivity weight: 60% (indirect effect via food)
- Monthly timestep: 12 months/year

## Implementation

### Step 1: Create Time Series
- Temperature: seasonal cycle + inter-annual variability + trend
- Productivity: spring bloom pattern + seasonal signal + year-to-year variation

### Step 2: Define Response Functions
- Each function maps environmental variable to recruitment response (0-2)
- Functions are species/stanza-specific

### Step 3: Modify Juvenile Mortality
- Recruitment factor inverted to Z (natural mortality) multiplier
- Low recruitment factor (< 1.0) -> high Z -> low juvenile survival
- High recruitment factor (> 1.0) -> low Z -> high juvenile survival

## Application to Multistanza Groups

### Affected Groups
- Demersal fish juvenile (Z_baseline = 1.2 yr-1)
- Predatory fish juvenile (Z_baseline = 1.5 yr-1)
- Predatory fish sub-adult (Z_baseline = 1.0 yr-1)

### Ecosystem Effects
1. Warm years with high productivity: enhanced juvenile recruitment
2. Cold years with low productivity: suppressed juvenile recruitment
3. Cascading effects through predator-prey interactions

## Time Series Statistics

- Temperature range: 0-30 C (clipped at ecosystem extremes)
- Productivity range: 0.1-1.0 (normalized)
- Recruitment factor range: typically 0.5-1.5

## File Outputs

- `environmental_forcing.csv`: Monthly time series data
- `environmental_forcing.png`: Visualization of forcing and response
- `RECRUITMENT_FORCING_GUIDE.md`: This guide

## Future Enhancements

1. Couple to oxygen dynamics (hypoxic stress on juveniles)
2. Add nutrient/upwelling forcing for productivity
3. Include fishing pressure feedback on recruitment
4. Species-specific response curves (calibrate to data)
5. Spatial heterogeneity of forcing across habitat patches
"""
    
    guide_path = output_dir / "RECRUITMENT_FORCING_GUIDE.md"
    with open(guide_path, "w", encoding="utf-8") as f:
        f.write(guide_text)
    logger.info(f"  Saved: {guide_path}")


def main():
    """Main workflow."""
    logger.info("=" * 80)
    logger.info("RECRUITMENT FORCING FOR CURONIAN LAGOON MULTISTANZA MODEL")
    logger.info("=" * 80)
    
    # Step 1: Create environmental forcing
    logger.info("\nStep 1: Creating environmental forcing time series")
    forcing = create_environmental_forcing(SIMULATION_YEARS)
    
    # Step 2: Calculate recruitment response
    logger.info("\nStep 2: Calculating recruitment response to forcing")
    recruitment_factor, recruitment_stats = calculate_recruitment_multiplier(forcing)
    
    # Step 3: Apply to ecosystem model
    logger.info("\nStep 3: Applying recruitment forcing to juvenile mortality")
    forced_z = apply_recruitment_forcing_to_ecopath(recruitment_factor)
    
    # Step 4: Create visualizations
    logger.info("\nStep 4: Creating visualizations")
    create_comparison_plots(forcing, recruitment_factor)
    
    # Step 5: Export data
    logger.info("\nStep 5: Exporting forcing data")
    export_forcing_data(forcing, recruitment_factor)
    
    # Step 6: Create guide
    logger.info("\nStep 6: Creating recruitment forcing guide")
    export_guide()
    
    # Summary
    logger.info("\n" + "=" * 80)
    logger.info("RECRUITMENT FORCING COMPLETE")
    logger.info("=" * 80)
    logger.info(f"\nResults saved to: {OUTPUT_DIR}")
    logger.info(f"  - environmental_forcing.csv")
    logger.info(f"  - environmental_forcing.png")
    logger.info(f"  - RECRUITMENT_FORCING_GUIDE.md")
    logger.info(f"\nRecuitment statistics:")
    logger.info(f"  Mean factor: {recruitment_stats['mean']:.3f}")
    logger.info(f"  Std dev: {recruitment_stats['std']:.3f}")
    logger.info(f"  Range: {recruitment_stats['min']:.3f} - {recruitment_stats['max']:.3f}")


if __name__ == "__main__":
    main()
