"""
Spatial Ecospace for Curonian Lagoon Multistanza Model
=====================================================

Implements spatially-explicit ecosystem simulation with habitat preferences
and ontogenetic migration of age-structured fish groups.

Features:
1. Create 1D spatial grid (littoral to profundal zones)
2. Define habitat preferences for juveniles and adults
3. Implement ontogenetic migration (size-dependent movement)
4. Run multi-patch Ecospace simulation
5. Visualize spatial dynamics

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
OUTPUT_DIR = Path("./ecospace_multistanza_results")
OUTPUT_DIR.mkdir(exist_ok=True)

SIMULATION_YEARS = 20


@dataclass
class HabitatPatch:
    """Represents a spatial patch in the ecosystem."""
    
    name: str
    zone_type: str  # 'littoral', 'sublittoral', 'profundal'
    area_km2: float
    depth_m: float
    connectivity: Dict[str, float] = None  # Neighboring patches and dispersal rates
    
    def __post_init__(self):
        if self.connectivity is None:
            self.connectivity = {}


def create_ecospace_grid() -> list:
    """
    Create 1D spatial grid for Curonian Lagoon.
    
    Habitat zones from shallow littoral to deep profundal.
    """
    logger.info("Creating spatial Ecospace grid")
    
    patches = [
        HabitatPatch(
            name="Littoral_Shallow",
            zone_type="littoral",
            area_km2=150.0,
            depth_m=0.5,
        ),
        HabitatPatch(
            name="Littoral_Deep",
            zone_type="littoral",
            area_km2=200.0,
            depth_m=2.0,
        ),
        HabitatPatch(
            name="Sublittoral",
            zone_type="sublittoral",
            area_km2=300.0,
            depth_m=4.5,
        ),
        HabitatPatch(
            name="Profundal_Shallow",
            zone_type="profundal",
            area_km2=250.0,
            depth_m=7.0,
        ),
        HabitatPatch(
            name="Profundal_Deep",
            zone_type="profundal",
            area_km2=200.0,
            depth_m=9.5,
        ),
    ]
    
    # Define inter-patch connectivity (dispersal rates)
    patches[0].connectivity = {"Littoral_Deep": 0.3, "Sublittoral": 0.1}
    patches[1].connectivity = {"Littoral_Shallow": 0.3, "Sublittoral": 0.4}
    patches[2].connectivity = {
        "Littoral_Deep": 0.1,
        "Profundal_Shallow": 0.4,
        "Sublittoral": 0.1,
    }
    patches[3].connectivity = {
        "Sublittoral": 0.3,
        "Profundal_Deep": 0.5,
    }
    patches[4].connectivity = {"Profundal_Shallow": 0.5}
    
    total_area = sum(p.area_km2 for p in patches)
    logger.info(f"  Created {len(patches)} patches")
    logger.info(f"  Total area: {total_area:.0f} km^2")
    logger.info(f"  Depth range: 0.5-9.5 m")
    
    return patches


def define_habitat_preferences() -> Dict[str, Dict]:
    """
    Define habitat suitability for each stanza group.
    
    Suitability ranges from 0 (unsuitable) to 1 (optimal).
    """
    logger.info("Defining habitat preferences")
    
    preferences = {
        # Demersal fish
        "demersal_juvenile": {
            "Littoral_Shallow": 0.8,
            "Littoral_Deep": 0.7,
            "Sublittoral": 0.4,
            "Profundal_Shallow": 0.1,
            "Profundal_Deep": 0.0,
        },
        "demersal_adult": {
            "Littoral_Shallow": 0.3,
            "Littoral_Deep": 0.5,
            "Sublittoral": 0.8,
            "Profundal_Shallow": 0.6,
            "Profundal_Deep": 0.2,
        },
        # Predatory fish
        "predatory_juvenile": {
            "Littoral_Shallow": 0.9,
            "Littoral_Deep": 0.8,
            "Sublittoral": 0.5,
            "Profundal_Shallow": 0.2,
            "Profundal_Deep": 0.0,
        },
        "predatory_subadult": {
            "Littoral_Shallow": 0.5,
            "Littoral_Deep": 0.7,
            "Sublittoral": 0.9,
            "Profundal_Shallow": 0.6,
            "Profundal_Deep": 0.2,
        },
        "predatory_adult": {
            "Littoral_Shallow": 0.2,
            "Littoral_Deep": 0.4,
            "Sublittoral": 0.7,
            "Profundal_Shallow": 0.8,
            "Profundal_Deep": 0.6,
        },
    }
    
    logger.info(f"  Defined preferences for {len(preferences)} stanza groups")
    
    return preferences


def distribute_biomass_across_patches(
    patches: list,
    habitat_prefs: Dict[str, Dict],
) -> Dict[str, np.ndarray]:
    """
    Distribute group biomass across patches based on habitat preferences.
    
    Returns
    -------
    Dict[str, np.ndarray]
        Initial spatial distribution for each stanza group.
    """
    logger.info("Distributing initial biomass across patches")
    
    patch_names = [p.name for p in patches]
    distributions = {}
    
    for stanza, prefs in habitat_prefs.items():
        # Normalize preferences to probabilities
        pref_values = np.array([prefs[pn] for pn in patch_names])
        total_pref = pref_values.sum()
        
        if total_pref > 0:
            distribution = pref_values / total_pref
        else:
            distribution = np.ones_like(pref_values) / len(pref_values)
        
        distributions[stanza] = distribution
    
    logger.info(f"  Distributed {len(distributions)} stanza groups")
    
    return distributions


def define_ontogenetic_migration() -> Dict[str, Dict]:
    """
    Define migration rates between patches as fish grow.
    
    Juveniles in littoral -> subadults in sublittoral -> adults in profundal.
    """
    logger.info("Defining ontogenetic migration parameters")
    
    migration = {
        "demersal_juvenile": {
            "source": "Littoral_Shallow",
            "destination": "Sublittoral",
            "rate_per_year": 0.15,  # 15% per year move to deeper habitat
            "trigger": "age",  # Migration triggered by age/size
        },
        "demersal_adult": {
            "source": "Sublittoral",
            "destination": "Profundal_Shallow",
            "rate_per_year": 0.05,
        },
        "predatory_juvenile": {
            "source": "Littoral_Shallow",
            "destination": "Sublittoral",
            "rate_per_year": 0.20,
        },
        "predatory_subadult": {
            "source": "Sublittoral",
            "destination": "Profundal_Shallow",
            "rate_per_year": 0.12,
        },
        "predatory_adult": {
            "source": "Profundal_Shallow",
            "destination": "Profundal_Deep",
            "rate_per_year": 0.08,
        },
    }
    
    logger.info(f"  Defined migrations for {len(migration)} stanza transitions")
    
    return migration


def run_ecospace_simulation(
    patches: list,
    habitat_prefs: Dict,
    migrations: Dict,
    years: int = SIMULATION_YEARS,
) -> Dict[str, np.ndarray]:
    """
    Run spatial Ecospace simulation.
    
    Returns
    -------
    Dict[str, np.ndarray]
        Biomass time series for each stanza in each patch.
    """
    logger.info("Running Ecospace simulation")
    
    timesteps = years * 12 + 1
    patch_names = [p.name for p in patches]
    n_patches = len(patches)
    
    # Initialize biomass distributions
    initial_dist = distribute_biomass_across_patches(patches, habitat_prefs)
    
    # Time series storage
    biomass_timeseries = {}
    for stanza in initial_dist.keys():
        biomass_timeseries[stanza] = np.zeros((timesteps, n_patches))
        # Set initial distribution
        biomass_timeseries[stanza][0, :] = initial_dist[stanza] * 1.0
    
    # Simulate movement (simplified: deterministic migration + diffusion)
    for t in range(1, timesteps):
        for stanza, timeseries in biomass_timeseries.items():
            prev_biomass = timeseries[t-1, :].copy()
            new_biomass = prev_biomass.copy()
            
            # Apply migration if defined for this stanza
            if stanza in migrations:
                mig = migrations[stanza]
                source_idx = patch_names.index(mig["source"])
                dest_idx = patch_names.index(mig["destination"])
                
                # Simplified migration: move fraction each timestep
                monthly_rate = mig["rate_per_year"] / 12
                migrant_biomass = prev_biomass[source_idx] * monthly_rate
                
                new_biomass[source_idx] -= migrant_biomass
                new_biomass[dest_idx] += migrant_biomass
            
            # Apply dispersal (local mixing between patches)
            dispersal_rate = 0.01  # 1% monthly dispersal to neighbors
            for i, patch in enumerate(patches):
                for neighbor_name, conn_rate in patch.connectivity.items():
                    j = patch_names.index(neighbor_name)
                    dispersal = new_biomass[i] * dispersal_rate * conn_rate
                    new_biomass[i] -= dispersal
                    new_biomass[j] += dispersal
            
            # Ensure positive biomass
            new_biomass = np.maximum(new_biomass, 0)
            timeseries[t, :] = new_biomass
    
    logger.info(f"  Simulation complete: {timesteps} timesteps x {n_patches} patches")
    
    return biomass_timeseries


def create_spatial_plots(
    biomass_ts: Dict[str, np.ndarray],
    patches: list,
    output_dir: Path = OUTPUT_DIR,
):
    """Create spatial visualization of Ecospace dynamics."""
    logger.info("Creating spatial plots")
    
    patch_names = [p.name for p in patches]
    
    fig, axes = plt.subplots(2, 3, figsize=(15, 8))
    fig.suptitle("Spatial Ecospace Dynamics - Habitat Utilization", 
                 fontsize=13, fontweight="bold")
    
    stanzas = list(biomass_ts.keys())
    for idx, stanza in enumerate(stanzas):
        ax = axes[idx // 3, idx % 3]
        
        # Plot biomass in each patch over time
        timeseries = biomass_ts[stanza]
        timesteps = timeseries.shape[0]
        years = np.linspace(0, SIMULATION_YEARS, timesteps)
        
        # Stack area plot
        ax.stackplot(years, timeseries.T, labels=patch_names, alpha=0.7)
        
        ax.set_title(f"{stanza}", fontweight="bold", fontsize=10)
        ax.set_xlabel("Year")
        ax.set_ylabel("Biomass Distribution")
        ax.grid(True, alpha=0.3)
        
        if idx == 0:
            ax.legend(fontsize=8, loc="upper left")
    
    # Hide empty subplot
    axes[1, 2].set_visible(False)
    
    plt.tight_layout()
    plot_path = output_dir / "ecospace_spatial_dynamics.png"
    plt.savefig(plot_path, dpi=150, bbox_inches="tight")
    logger.info(f"  Saved: {plot_path}")
    plt.close()


def export_ecospace_summary(
    patches: list,
    biomass_ts: Dict[str, np.ndarray],
    output_dir: Path = OUTPUT_DIR,
):
    """Export Ecospace summary to CSV."""
    logger.info("Exporting Ecospace summary")
    
    patch_names = [p.name for p in patches]
    
    # Summary statistics for each stanza across all patches
    summary_rows = []
    for stanza, timeseries in biomass_ts.items():
        total_biomass = timeseries.sum(axis=1)
        
        for patch_idx, patch_name in enumerate(patch_names):
            summary_rows.append({
                "Stanza": stanza,
                "Patch": patch_name,
                "Initial_Biomass_Fraction": timeseries[0, patch_idx] / (timeseries[0, :].sum() + 1e-10),
                "Final_Biomass_Fraction": timeseries[-1, patch_idx] / (timeseries[-1, :].sum() + 1e-10),
                "Peak_Fraction": timeseries[:, patch_idx].max() / (timeseries[:, patch_idx].sum() + 1e-10),
            })
    
    df = pd.DataFrame(summary_rows)
    csv_path = output_dir / "ecospace_summary.csv"
    df.to_csv(csv_path, index=False)
    logger.info(f"  Saved: {csv_path}")


def export_ecospace_guide(output_dir: Path = OUTPUT_DIR):
    """Export Ecospace guide."""
    logger.info("Creating Ecospace guide")
    
    guide_text = """# Spatial Ecospace for Multistanza Models

## Overview
This guide demonstrates how to implement spatially-explicit simulations of 
age-structured fish populations with habitat preferences and ontogenetic migration.

## Spatial Grid

### Habitat Zones
1. **Littoral Shallow**: 0-1 m (nursery habitat for juveniles)
2. **Littoral Deep**: 1-3 m (shallow foraging habitat)
3. **Sublittoral**: 3-6 m (transition zone)
4. **Profundal Shallow**: 6-8 m (adult habitat)
5. **Profundal Deep**: 8-10 m (deep foraging)

### Connectivity
- Adjacent patches connected by dispersal
- Dispersal rates: 1% per month to neighboring zones
- Variable connectivity based on habitat proximity

## Habitat Preferences

### Demersal Fish
- **Juvenile**: Prefers littoral (0.8), avoids profundal (0.0)
- **Adult**: Prefers sublittoral-profundal (0.8), less littoral (0.3)

### Predatory Fish
- **Juvenile**: Prefers littoral (0.9), strong nursery habitat preference
- **Sub-adult**: Transitions to sublittoral (0.9), leaving littoral (0.5)
- **Adult**: Prefers profundal (0.8), avoids shallow (0.2)

## Ontogenetic Migration

### Process
1. Juveniles spawn in shallow littoral habitat
2. As juveniles grow, movement to deeper water increases
3. Adults occupy deeper profundal zones
4. Migration rates vary by stanza (5-20% per year)

### Drivers
- Size-dependent habitat preference
- Predation avoidance (deeper water reduces predation)
- Forage size selectivity (larger fish need larger prey)

## Simulation Process

### Initialization
- Distribute each stanza group across patches based on habitat preference
- Initial condition: all groups in preferred habitat

### Dynamics
- **Migration**: Stanza-specific movement from source to destination patch
- **Dispersal**: Local diffusion (1% monthly) between connected patches
- **Growth**: Implicit in stanza transitions (modeled in Ecopath/Ecosim)

### Timestep
- Monthly resolution (12 timesteps per year)
- 20-year simulation = 241 timesteps

## Ecosystem Effects

### Spatial Heterogeneity
- Juveniles concentrated in high-productivity littoral zones
- Adults distributed across diverse habitats
- Reduced overlap -> reduced inter-stanza predation

### Recovery Potential
- Juvenile nurseries act as population bottlenecks
- Protection of littoral habitat = protection of recruitment
- Loss of littoral habitat cascades to adult populations

## File Outputs

- `ecospace_spatial_dynamics.png`: Stacked area plots by patch
- `ecospace_summary.csv`: Initial, final, peak biomass fractions

## Future Enhancements

1. **Fishing Pressure**: Spatial targeting of specific zones
2. **Habitat Loss**: Remove littoral patches and observe recruitment collapse
3. **Marine Protected Areas**: Restrict fishing in nursery zones
4. **Temperature Coupling**: Link spatial distribution to thermal preferences
5. **Predator-Prey Overlap**: Calculate predation risk by patch
"""
    
    guide_path = output_dir / "ECOSPACE_GUIDE.md"
    with open(guide_path, "w", encoding="utf-8") as f:
        f.write(guide_text)
    logger.info(f"  Saved: {guide_path}")


def main():
    """Main workflow."""
    logger.info("=" * 80)
    logger.info("SPATIAL ECOSPACE FOR CURONIAN LAGOON MULTISTANZA MODEL")
    logger.info("=" * 80)
    
    # Step 1: Create spatial grid
    logger.info("\nStep 1: Creating spatial grid")
    patches = create_ecospace_grid()
    
    # Step 2: Define habitat preferences
    logger.info("\nStep 2: Defining habitat preferences")
    habitat_prefs = define_habitat_preferences()
    
    # Step 3: Define migration
    logger.info("\nStep 3: Defining ontogenetic migration")
    migrations = define_ontogenetic_migration()
    
    # Step 4: Run Ecospace
    logger.info("\nStep 4: Running Ecospace simulation")
    biomass_ts = run_ecospace_simulation(patches, habitat_prefs, migrations)
    
    # Step 5: Create plots
    logger.info("\nStep 5: Creating spatial visualizations")
    create_spatial_plots(biomass_ts, patches)
    
    # Step 6: Export summary
    logger.info("\nStep 6: Exporting Ecospace summary")
    export_ecospace_summary(patches, biomass_ts)
    
    # Step 7: Export guide
    logger.info("\nStep 7: Creating Ecospace guide")
    export_ecospace_guide()
    
    logger.info("\n" + "=" * 80)
    logger.info("ECOSPACE SIMULATION COMPLETE")
    logger.info("=" * 80)
    logger.info(f"\nResults saved to: {OUTPUT_DIR}")
    logger.info(f"  - ecospace_spatial_dynamics.png")
    logger.info(f"  - ecospace_summary.csv")
    logger.info(f"  - ECOSPACE_GUIDE.md")


if __name__ == "__main__":
    main()
