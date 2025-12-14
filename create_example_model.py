"""Create a comprehensive example Ecopath model.

This model demonstrates all key features:
- Multi-stanza groups (age-structured populations)
- Multiple fishing fleets with different selectivities
- Import/export flows
- Detritus fate pathways
- Realistic coastal ecosystem structure

Ecosystem: Temperate Coastal Shelf
Groups: 12 functional groups
Fleets: 3 fishing fleets
Structure: 4 trophic levels
"""

import numpy as np
import pandas as pd
from pathlib import Path
import sys

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from pypath.core.params import create_rpath_params
from pypath.core.ecopath import rpath
from pypath.core.stanzas import create_stanza_params


def create_coastal_ecosystem_model():
    """Create a comprehensive coastal ecosystem model.

    Functional Groups (12):
    1. Phytoplankton - Primary producers
    2. Macroalgae - Benthic primary producers
    3. Zooplankton - Herbivorous zooplankton
    4. Meiobenthos - Small benthic invertebrates
    5. Benthic invertebrates - Large benthic fauna
    6. Small pelagics (juvenile) - Age 0-1 (STANZA)
    7. Small pelagics (adult) - Age 1+ (STANZA)
    8. Demersal fish - Bottom-dwelling fish
    9. Large pelagics - Top predatory fish
    10. Seabirds - Marine birds
    11. Detritus - Dead organic matter
    12. Discards - Fishing discards

    Fleets (3):
    1. Trawl fleet - Targets demersal fish, impacts benthos
    2. Purse seine - Targets small pelagics
    3. Longline - Targets large pelagics, bycatch seabirds
    """

    print("=" * 70)
    print("CREATING COMPREHENSIVE COASTAL ECOSYSTEM MODEL")
    print("=" * 70)

    # Define groups
    groups = [
        'Phytoplankton',
        'Macroalgae',
        'Zooplankton',
        'Meiobenthos',
        'Benthic invertebrates',
        'Small pelagics (juv)',
        'Small pelagics (adult)',
        'Demersal fish',
        'Large pelagics',
        'Seabirds',
        'Detritus',
        'Discards'
    ]

    # Define types
    # 0 = consumer, 1 = producer, 2 = detritus, 3 = fleet
    types = [1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 2, 2]

    print(f"\n1. Initializing model with {len(groups)} groups")

    # Create base parameters
    params = create_rpath_params(groups, types)

    # ==========================================
    # 2. SET BASIC PARAMETERS
    # ==========================================
    print("\n2. Setting basic parameters")

    # Biomass (t/km²)
    params.model['Biomass'] = [
        20.0,   # Phytoplankton - high turnover
        5.0,    # Macroalgae
        8.0,    # Zooplankton
        2.0,    # Meiobenthos
        5.0,    # Benthic invertebrates
        0.5,    # Small pelagics (juv) - will be calculated by stanza
        2.0,    # Small pelagics (adult) - will be calculated by stanza
        1.5,    # Demersal fish
        0.8,    # Large pelagics
        0.05,   # Seabirds - top predator
        10.0,   # Detritus
        0.5     # Discards
    ]

    # Production/Biomass (per year)
    params.model['PB'] = [
        150.0,  # Phytoplankton - very high turnover
        12.0,   # Macroalgae
        35.0,   # Zooplankton
        8.0,    # Meiobenthos
        2.5,    # Benthic invertebrates
        1.8,    # Small pelagics (juv) - will be adjusted by stanza
        0.6,    # Small pelagics (adult) - will be adjusted by stanza
        0.5,    # Demersal fish
        0.4,    # Large pelagics
        0.1,    # Seabirds
        0.0,    # Detritus
        0.0     # Discards
    ]

    # Consumption/Biomass (per year)
    params.model['QB'] = [
        0.0,    # Phytoplankton - producer
        0.0,    # Macroalgae - producer
        80.0,   # Zooplankton
        20.0,   # Meiobenthos
        8.0,    # Benthic invertebrates
        6.0,    # Small pelagics (juv)
        4.0,    # Small pelagics (adult)
        3.0,    # Demersal fish
        3.5,    # Large pelagics
        50.0,   # Seabirds - high metabolism
        0.0,    # Detritus
        0.0     # Discards
    ]

    # Ecotrophic Efficiency (estimated, will be calculated)
    params.model['EE'] = [
        0.90,   # Phytoplankton
        0.50,   # Macroalgae
        0.85,   # Zooplankton
        0.75,   # Meiobenthos
        0.70,   # Benthic invertebrates
        0.80,   # Small pelagics (juv)
        0.75,   # Small pelagics (adult)
        0.60,   # Demersal fish
        0.50,   # Large pelagics
        0.01,   # Seabirds - top predator
        0.90,   # Detritus
        0.95    # Discards
    ]

    # Biomass accumulation (usually 0)
    params.model['BioAcc'] = [0.0] * 12

    # Unassimilated consumption (fraction)
    params.model['Unassim'] = [
        0.0,    # Phytoplankton
        0.0,    # Macroalgae
        0.3,    # Zooplankton
        0.2,    # Meiobenthos
        0.2,    # Benthic invertebrates
        0.2,    # Small pelagics (juv)
        0.2,    # Small pelagics (adult)
        0.2,    # Demersal fish
        0.15,   # Large pelagics
        0.1,    # Seabirds
        0.0,    # Detritus
        0.0     # Discards
    ]

    print(f"   Set biomass for {len(groups)} groups")
    print(f"   Primary production: {params.model['Biomass'][0] * params.model['PB'][0]:.1f} t/km²/year")

    # ==========================================
    # 3. DEFINE DIET MATRIX
    # ==========================================
    print("\n3. Defining predator-prey relationships")

    # Diet matrix: rows = predators, columns = prey
    # Columns: Outside, Phyto, Macro, Zoo, Meio, Bent, SmallJuv, SmallAdult, Demersal, LargePel, Birds, Det, Disc

    diet_data = {
        'Outside': [0.0] * 12,
        'Phytoplankton': [
            0.0,  # Phytoplankton
            0.0,  # Macroalgae
            0.90, # Zooplankton - mainly phytoplankton
            0.10, # Meiobenthos - some phytoplankton
            0.05, # Benthic invertebrates
            0.30, # Small pelagics (juv) - planktivores
            0.20, # Small pelagics (adult)
            0.0,  # Demersal fish
            0.0,  # Large pelagics
            0.0,  # Seabirds
            0.0,  # Detritus
            0.0   # Discards
        ],
        'Macroalgae': [
            0.0,  # Phytoplankton
            0.0,  # Macroalgae
            0.0,  # Zooplankton
            0.0,  # Meiobenthos
            0.20, # Benthic invertebrates - grazers
            0.0,  # Small pelagics (juv)
            0.0,  # Small pelagics (adult)
            0.0,  # Demersal fish
            0.0,  # Large pelagics
            0.0,  # Seabirds
            0.0,  # Detritus
            0.0   # Discards
        ],
        'Zooplankton': [
            0.0,  # Phytoplankton
            0.0,  # Macroalgae
            0.0,  # Zooplankton
            0.0,  # Meiobenthos
            0.0,  # Benthic invertebrates
            0.50, # Small pelagics (juv) - zooplanktivores
            0.40, # Small pelagics (adult)
            0.10, # Demersal fish - some zooplankton
            0.20, # Large pelagics
            0.0,  # Seabirds
            0.0,  # Detritus
            0.0   # Discards
        ],
        'Meiobenthos': [
            0.0,  # Phytoplankton
            0.0,  # Macroalgae
            0.0,  # Zooplankton
            0.0,  # Meiobenthos
            0.15, # Benthic invertebrates
            0.0,  # Small pelagics (juv)
            0.05, # Small pelagics (adult)
            0.20, # Demersal fish - benthic feeders
            0.0,  # Large pelagics
            0.0,  # Seabirds
            0.0,  # Detritus
            0.0   # Discards
        ],
        'Benthic invertebrates': [
            0.0,  # Phytoplankton
            0.0,  # Macroalgae
            0.0,  # Zooplankton
            0.0,  # Meiobenthos
            0.0,  # Benthic invertebrates
            0.0,  # Small pelagics (juv)
            0.0,  # Small pelagics (adult)
            0.30, # Demersal fish - major prey
            0.10, # Large pelagics
            0.10, # Seabirds - coastal feeders
            0.0,  # Detritus
            0.0   # Discards
        ],
        'Small pelagics (juv)': [
            0.0,  # Phytoplankton
            0.0,  # Macroalgae
            0.0,  # Zooplankton
            0.0,  # Meiobenthos
            0.0,  # Benthic invertebrates
            0.0,  # Small pelagics (juv)
            0.05, # Small pelagics (adult) - cannibalism
            0.10, # Demersal fish
            0.20, # Large pelagics - prey on juveniles
            0.30, # Seabirds - important prey
            0.0,  # Detritus
            0.0   # Discards
        ],
        'Small pelagics (adult)': [
            0.0,  # Phytoplankton
            0.0,  # Macroalgae
            0.0,  # Zooplankton
            0.0,  # Meiobenthos
            0.0,  # Benthic invertebrates
            0.0,  # Small pelagics (juv)
            0.0,  # Small pelagics (adult)
            0.10, # Demersal fish
            0.30, # Large pelagics - main prey
            0.50, # Seabirds - important prey
            0.0,  # Detritus
            0.0   # Discards
        ],
        'Demersal fish': [
            0.0,  # Phytoplankton
            0.0,  # Macroalgae
            0.0,  # Zooplankton
            0.0,  # Meiobenthos
            0.0,  # Benthic invertebrates
            0.0,  # Small pelagics (juv)
            0.0,  # Small pelagics (adult)
            0.0,  # Demersal fish
            0.10, # Large pelagics
            0.10, # Seabirds
            0.0,  # Detritus
            0.0   # Discards
        ],
        'Large pelagics': [
            0.0,  # Phytoplankton
            0.0,  # Macroalgae
            0.0,  # Zooplankton
            0.0,  # Meiobenthos
            0.0,  # Benthic invertebrates
            0.0,  # Small pelagics (juv)
            0.0,  # Small pelagics (adult)
            0.0,  # Demersal fish
            0.0,  # Large pelagics
            0.0,  # Seabirds
            0.0,  # Detritus
            0.0   # Discards
        ],
        'Seabirds': [
            0.0,  # Phytoplankton
            0.0,  # Macroalgae
            0.0,  # Zooplankton
            0.0,  # Meiobenthos
            0.0,  # Benthic invertebrates
            0.0,  # Small pelagics (juv)
            0.0,  # Small pelagics (adult)
            0.0,  # Demersal fish
            0.0,  # Large pelagics
            0.0,  # Seabirds
            0.0,  # Detritus
            0.0   # Discards
        ],
        'Detritus': [
            0.0,  # Phytoplankton
            0.0,  # Macroalgae
            0.10, # Zooplankton - some detritivory
            0.80, # Meiobenthos - mainly detritivores
            0.60, # Benthic invertebrates - deposit feeders
            0.20, # Small pelagics (juv)
            0.30, # Small pelagics (adult)
            0.20, # Demersal fish
            0.10, # Large pelagics
            0.0,  # Seabirds
            0.0,  # Detritus
            0.0   # Discards
        ],
        'Discards': [
            0.0,  # Phytoplankton
            0.0,  # Macroalgae
            0.0,  # Zooplankton
            0.10, # Meiobenthos
            0.0,  # Benthic invertebrates
            0.0,  # Small pelagics (juv)
            0.05, # Small pelagics (adult)
            0.10, # Demersal fish - scavengers
            0.0,  # Large pelagics
            0.0,  # Seabirds
            0.0,  # Detritus
            0.0   # Discards
        ]
    }

    # Convert diet_data to proper format with 'Group' column
    # diet_data has predators as keys, need to transpose to have prey as rows
    diet_df_dict = {'Group': groups}
    for predator, prey_list in diet_data.items():
        diet_df_dict[predator] = prey_list

    params.diet = pd.DataFrame(diet_df_dict)

    # Normalize diet to sum to 1 for each predator (skip 'Group' column)
    for col in params.diet.columns[1:]:  # Skip 'Group' column
        col_sum = params.diet[col].sum()
        if col_sum > 0:
            params.diet[col] = params.diet[col] / col_sum

    # Count trophic links (exclude 'Group' column)
    diet_numeric = params.diet.iloc[:, 1:].values  # Skip 'Group' column
    print(f"   Created diet matrix with {np.sum(diet_numeric > 0)} trophic links")

    # ==========================================
    # 4. DEFINE MULTI-STANZA GROUP
    # ==========================================
    print("\n4. Setting up multi-stanza group (Small pelagics)")

    # Small pelagics: 2 stanzas (juvenile 0-1 year, adult 1+ years)
    # Define stanza groups
    stanza_groups = [
        {
            'stanza_group_num': 1,
            'n_stanzas': 2,
            'vbgf_ksp': 0.5,      # von Bertalanffy K (growth rate)
            'vbgf_d': 0.66667,    # Allometric exponent
            'wmat': 15.0,         # Weight at maturity (g)
            'rec_power': 1.0      # Recruitment power
        }
    ]

    # Define individual stanzas
    stanza_individuals = [
        {
            'stanza_group_num': 1,
            'stanza_num': 1,
            'group_num': 6,  # Small pelagics (juv) - index in groups list
            'group_name': 'Small pelagics (juv)',
            'first': 0,      # Age in months
            'last': 11,      # Age in months
            'z': 1.8,        # Total mortality (will be calculated)
            'leading': False
        },
        {
            'stanza_group_num': 1,
            'stanza_num': 2,
            'group_num': 7,  # Small pelagics (adult) - index in groups list
            'group_name': 'Small pelagics (adult)',
            'first': 12,     # Age in months
            'last': 60,      # Age in months (5 years max)
            'z': 0.6,        # Total mortality (will be calculated)
            'leading': True  # Adult is leading stanza
        }
    ]

    stanza_data = create_stanza_params(stanza_groups, stanza_individuals)
    params.stanzas = stanza_data

    print(f"   Configured {stanza_data.stanza_groups[0].n_stanzas} stanzas for Small pelagics")
    print(f"   Juvenile: 0-11 months")
    print(f"   Adult: 12-60 months (leading stanza)")

    # ==========================================
    # 5. DEFINE FISHING FLEETS
    # ==========================================
    print("\n5. Setting up fishing fleets")

    # Landing (what is caught and kept)
    landing_data = {
        'Group': groups,
        'Trawl': [0.0, 0.0, 0.0, 0.0, 0.1, 0.0, 0.2, 0.8, 0.1, 0.0, 0.0, 0.0],
        'Purse_seine': [0.0, 0.0, 0.0, 0.0, 0.0, 0.3, 0.9, 0.0, 0.0, 0.0, 0.0, 0.0],
        'Longline': [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.7, 0.0, 0.0, 0.0]
    }
    params.landing = pd.DataFrame(landing_data)

    # Discard (what is caught but discarded)
    discard_data = {
        'Group': groups,
        'Trawl': [0.0, 0.0, 0.0, 0.0, 0.0, 0.1, 0.05, 0.2, 0.0, 0.0, 0.0, 0.0],
        'Purse_seine': [0.0, 0.0, 0.0, 0.0, 0.0, 0.05, 0.02, 0.0, 0.0, 0.0, 0.0, 0.0],
        'Longline': [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.1, 0.05, 0.0, 0.0]  # Seabird bycatch
    }
    params.discard = pd.DataFrame(discard_data)

    # Discard fate (what happens to discards)
    # 0 = dies and goes to detritus, 1 = survives
    discard_fate_data = {
        'Group': groups,
        'Trawl': [0.0] * 12,  # All trawl discards die
        'Purse_seine': [0.0] * 12,  # All seine discards die
        'Longline': [0.0] * 12  # All longline discards die
    }
    params.discards = pd.DataFrame(discard_fate_data)

    # Set seabird bycatch survival (some survive)
    params.discards.loc[params.discards['Group'] == 'Seabirds', 'Longline'] = 0.3  # 30% survive

    print(f"   Created 3 fishing fleets:")
    print(f"   - Trawl: Targets demersal fish, benthic invertebrates")
    print(f"   - Purse seine: Targets small pelagics")
    print(f"   - Longline: Targets large pelagics, seabird bycatch")

    # ==========================================
    # 6. DEFINE DETRITUS FATE
    # ==========================================
    print("\n6. Setting detritus fate pathways")

    # Detritus fate: where does detritus go?
    # Import: material from outside system
    # Export: material leaving system
    # Flow to detritus groups

    detritus_fate_data = {
        'Group': groups,
        'Detritus': [
            0.0,   # Phytoplankton
            0.0,   # Macroalgae
            0.0,   # Zooplankton
            0.0,   # Meiobenthos
            0.0,   # Benthic invertebrates
            0.0,   # Small pelagics (juv)
            0.0,   # Small pelagics (adult)
            0.0,   # Demersal fish
            0.0,   # Large pelagics
            0.0,   # Seabirds
            0.0,   # Detritus
            0.0    # Discards
        ],
        'Discards': [
            0.0,   # Phytoplankton
            0.0,   # Macroalgae
            0.0,   # Zooplankton
            0.0,   # Meiobenthos
            0.0,   # Benthic invertebrates
            0.0,   # Small pelagics (juv)
            0.0,   # Small pelagics (adult)
            0.0,   # Demersal fish
            0.0,   # Large pelagics
            0.0,   # Seabirds
            0.0,   # Detritus
            0.0    # Discards
        ],
        'Export': [
            0.15,  # Phytoplankton - some exported
            0.10,  # Macroalgae - drift export
            0.05,  # Zooplankton
            0.02,  # Meiobenthos
            0.02,  # Benthic invertebrates
            0.01,  # Small pelagics (juv)
            0.01,  # Small pelagics (adult)
            0.01,  # Demersal fish
            0.01,  # Large pelagics
            0.0,   # Seabirds
            0.20,  # Detritus - major export
            0.05   # Discards
        ]
    }

    params.detritus_fate = pd.DataFrame(detritus_fate_data)

    # Detritus flows to detritus pool
    params.detritus_fate.loc[params.detritus_fate['Group'] != 'Detritus', 'Detritus'] = 0.85
    params.detritus_fate.loc[params.detritus_fate['Group'] != 'Discards', 'Discards'] = 0.0

    # Normalize detritus fate (Detritus + Discards + Export should sum to 1)
    for i, group in enumerate(groups):
        if group not in ['Detritus', 'Discards']:
            total = params.detritus_fate.iloc[i, 1:].sum()
            if total > 0:
                params.detritus_fate.iloc[i, 1:] = params.detritus_fate.iloc[i, 1:] / total

    export_rate = params.detritus_fate['Export'].iloc[10]  # Detritus export
    print(f"   Detritus export rate: {export_rate*100:.1f}%")
    print(f"   Phytoplankton export rate: {params.detritus_fate['Export'].iloc[0]*100:.1f}%")

    # ==========================================
    # 7. IMPORTS AND EXPORTS
    # ==========================================
    print("\n7. Setting import flows")

    # Immigration/recruitment from outside
    params.model.loc[0, 'Biomass'] = 20.0  # Phytoplankton biomass maintained by nutrients from outside

    # Add import to diet (nutrient input for phytoplankton)
    # Phytoplankton gets nutrients from "Outside" (upwelling, rivers, etc.)
    params.diet.loc['Phytoplankton', 'Outside'] = 0.0  # Handled implicitly by P/B

    print(f"   Nutrient import supports primary production")
    print(f"   Organic export: {export_rate*100:.1f}% of detritus")

    return params


def save_model(params, filename="example_coastal_model.csv"):
    """Save model to CSV files."""
    print(f"\n8. Saving model to {filename}")

    # Create output directory
    output_dir = Path("example_model_data")
    output_dir.mkdir(exist_ok=True)

    # Save basic parameters
    params.model.to_csv(output_dir / "model.csv", index=False)
    print(f"   Saved: model.csv")

    # Save diet
    params.diet.to_csv(output_dir / "diet.csv")
    print(f"   Saved: diet.csv")

    # Save fisheries
    params.landing.to_csv(output_dir / "landing.csv", index=False)
    params.discard.to_csv(output_dir / "discard.csv", index=False)
    params.discards.to_csv(output_dir / "discard_fate.csv", index=False)
    print(f"   Saved: landing.csv, discard.csv, discard_fate.csv")

    # Save detritus fate
    params.detritus_fate.to_csv(output_dir / "detritus_fate.csv", index=False)
    print(f"   Saved: detritus_fate.csv")

    # Save stanza parameters
    if hasattr(params, 'stanzas') and params.stanzas is not None:
        # Convert stanza_groups list to DataFrame
        if params.stanzas.stanza_groups:
            stgroups_df = pd.DataFrame([vars(sg) for sg in params.stanzas.stanza_groups])
            stgroups_df.to_csv(output_dir / "stanza_groups.csv", index=False)

        # Convert stanza_individuals list to DataFrame
        if params.stanzas.stanza_individuals:
            stindiv_df = pd.DataFrame([vars(si) for si in params.stanzas.stanza_individuals])
            stindiv_df.to_csv(output_dir / "stanza_individual.csv", index=False)

        print(f"   Saved: stanza_groups.csv, stanza_individual.csv")

    return output_dir


def balance_and_validate(params):
    """Balance the Ecopath model and show diagnostics."""
    print("\n" + "=" * 70)
    print("BALANCING AND VALIDATING MODEL")
    print("=" * 70)

    try:
        model = rpath(params)

        print("\n[OK] MODEL BALANCED SUCCESSFULLY")
        print(f"\nModel summary:")
        print(f"  Groups: {model.NUM_GROUPS}")
        print(f"  Living groups: {model.NUM_LIVING}")
        print(f"  Detritus groups: {model.NUM_DEAD}")

        # Check EE values
        print(f"\n Ecotrophic Efficiency (EE):")
        for i in range(model.NUM_LIVING):
            ee = model.EE[i]
            status = "[OK]" if 0 <= ee <= 1 else "[!]"
            warning = " (WARNING: >1)" if ee > 1 else ""
            print(f"  {status} {model.Group[i]}: {ee:.3f}{warning}")

        # System statistics
        total_biomass = np.sum(model.Biomass[0:model.NUM_LIVING])
        total_production = np.sum(model.Biomass[0:model.NUM_LIVING] * model.PB[0:model.NUM_LIVING])
        total_consumption = np.sum([
            model.Biomass[i] * model.QB[i]
            for i in range(model.NUM_LIVING)
            if model.QB[i] > 0
        ])

        print(f"\nSystem statistics:")
        print(f"  Total biomass: {total_biomass:.2f} t/km²")
        print(f"  Total production: {total_production:.2f} t/km²/year")
        print(f"  Total consumption: {total_consumption:.2f} t/km²/year")
        print(f"  P/C ratio: {total_production/total_consumption:.3f}")

        # Trophic levels
        print(f"\nTrophic levels:")
        for i in range(model.NUM_LIVING):
            tl = model.TL[i]
            print(f"  {model.Group[i]}: {tl:.2f}")

        return model

    except Exception as e:
        print(f"\n[ERROR] MODEL BALANCING FAILED")
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
        return None


if __name__ == '__main__':
    print("\n" + "=" * 70)
    print("COMPREHENSIVE ECOPATH MODEL GENERATOR")
    print("=" * 70)
    print("\nThis script creates a realistic coastal ecosystem model with:")
    print("- 12 functional groups across 4 trophic levels")
    print("- Multi-stanza age-structured population (Small pelagics)")
    print("- 3 fishing fleets with different selectivities")
    print("- Import/export flows")
    print("- Detritus fate pathways")
    print("- Realistic parameter values based on coastal shelf ecosystems")

    # Create model
    params = create_coastal_ecosystem_model()

    # Save to files
    output_dir = save_model(params)

    # Balance and validate
    model = balance_and_validate(params)

    if model is not None:
        print("\n" + "=" * 70)
        print("SUCCESS!")
        print("=" * 70)
        print(f"\nModel files saved to: {output_dir}/")
        print("\nYou can now use this model for:")
        print("1. Ecosim simulations")
        print("2. Bayesian optimization testing")
        print("3. Teaching and demonstrations")
        print("4. Developing new features")

        print("\nNext steps:")
        print("1. Load the model: params = read_rpath_params('example_model_data/model.csv', ...)")
        print("2. Run Ecosim: rsim_run(rsim_scenario(model, params))")
        print("3. Try optimization: See test_bayesian_optimization.py")

    else:
        print("\n" + "=" * 70)
        print("MODEL CREATION FAILED")
        print("=" * 70)
        print("Check error messages above and adjust parameters")
