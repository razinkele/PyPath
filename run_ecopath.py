from pypath.core.params import create_rpath_params
from pypath.core.ecopath import rpath
import pandas as pd
import numpy as np

print("Building and Balancing Curonian Lagoon Ecopath Model")
print("=" * 60)
print("")

# Load group data
model_df = pd.read_csv("curonian_model_groups.csv")
groups_list = model_df["GroupName"].tolist()
types_list = model_df["Type"].tolist()

# Diet rules
diet_rules = {
    "Cormorants": {"Pike": 0.3, "Pikeperch": 0.2, "Perch (large)": 0.2, "Roach": 0.1, "Sprat": 0.1, "Smelt": 0.1},
    "Great crested grebe": {"Sprat": 0.4, "Roach": 0.3, "Bleak": 0.3},
    "Pike": {"Perch (small)": 0.3, "Roach": 0.2, "Bream": 0.15, "Sprat": 0.15, "Amphipods": 0.1, "Mysids": 0.1},
    "Pikeperch": {"Roach": 0.3, "Bream": 0.25, "Sprat": 0.2, "Smelt": 0.15, "Amphipods": 0.1},
    "Perch (large)": {"Roach": 0.25, "Sprat": 0.2, "Amphipods": 0.2, "Mysids": 0.15, "Copepod nauplii": 0.2},
    "Perch (small)": {"Amphipods": 0.25, "Mysids": 0.2, "Copepod nauplii": 0.25, "Daphnia & cladocera": 0.2, "Rotifers": 0.1},
    "Roach": {"Daphnia & cladocera": 0.3, "Copepod nauplii": 0.2, "Bosmina & small cladocera": 0.2, "Rotifers": 0.15, "Chironomid larvae": 0.1, "Phytoplankton": 0.05},
    "Bream": {"Amphipods": 0.25, "Chironomid larvae": 0.25, "Oligochaetes": 0.2, "Mysids": 0.15, "Mollusks": 0.1, "Detritus": 0.05},
    "Asp": {"Sprat": 0.35, "Roach": 0.25, "Bleak": 0.25, "Mysids": 0.15},
    "Tench": {"Oligochaetes": 0.3, "Chironomid larvae": 0.2, "Mollusks": 0.2, "Amphipods": 0.15, "Phytoplankton": 0.15},
    "Rudd": {"Daphnia & cladocera": 0.25, "Chironomid larvae": 0.25, "Copepod nauplii": 0.2, "Phytoplankton": 0.2, "Macrophytes": 0.1},
    "Flounder": {"Amphipods": 0.3, "Oligochaetes": 0.25, "Chironomid larvae": 0.2, "Mollusks": 0.15, "Nematodes": 0.1},
    "Smelt": {"Copepod nauplii": 0.35, "Acartia & copepods": 0.25, "Daphnia & cladocera": 0.2, "Bosmina & small cladocera": 0.2},
    "Weatherfish": {"Oligochaetes": 0.35, "Chironomid larvae": 0.3, "Amphipods": 0.2, "Nematodes": 0.15},
    "Sprat": {"Copepod nauplii": 0.4, "Acartia & copepods": 0.3, "Daphnia & cladocera": 0.2, "Bosmina & small cladocera": 0.1},
    "Bleak": {"Daphnia & cladocera": 0.3, "Bosmina & small cladocera": 0.3, "Copepod nauplii": 0.25, "Rotifers": 0.15},
    "Carp": {"Oligochaetes": 0.2, "Chironomid larvae": 0.2, "Amphipods": 0.15, "Mollusks": 0.15, "Macrophytes": 0.2, "Phytoplankton": 0.1},
    "Mysids": {"Copepod nauplii": 0.3, "Daphnia & cladocera": 0.2, "Nematodes": 0.15, "Detritus": 0.35},
    "Amphipods": {"Detritus": 0.4, "Filamentous algae": 0.3, "Nematodes": 0.15, "Rotifers": 0.15},
    "Chironomid larvae": {"Detritus": 0.5, "Phytoplankton": 0.25, "Nematodes": 0.25},
    "Oligochaetes": {"Detritus": 0.6, "Phytoplankton": 0.2, "Nematodes": 0.2},
    "Mollusks": {"Phytoplankton": 0.4, "Filamentous algae": 0.4, "Detritus": 0.2},
    "Asellus (isopods)": {"Detritus": 0.5, "Filamentous algae": 0.3, "Phytoplankton": 0.2},
    "Daphnia & cladocera": {"Phytoplankton": 0.7, "Filamentous algae": 0.2, "Detritus": 0.1},
    "Bosmina & small cladocera": {"Phytoplankton": 0.75, "Detritus": 0.25},
    "Acartia & copepods": {"Phytoplankton": 0.6, "Detritus": 0.4},
    "Eurytemora & harpacticoids": {"Phytoplankton": 0.65, "Detritus": 0.35},
    "Copepod nauplii": {"Phytoplankton": 0.8, "Detritus": 0.2},
    "Nematodes": {"Detritus": 0.7, "Phytoplankton": 0.3},
    "Rotifers": {"Phytoplankton": 0.85, "Detritus": 0.15},
}

# Build diet dataframe with "Group" column
diet_rows = [{"Group": g} for g in groups_list]
for g in groups_list:
    for row in diet_rows:
        row[g] = 0.0

diet_df = pd.DataFrame(diet_rows)
diet_df = diet_df.set_index("Group")

# Fill diet values
for pred, preys in diet_rules.items():
    if pred in diet_df.columns:
        for prey, prop in preys.items():
            if prey in diet_df.index:
                diet_df.loc[prey, pred] = prop

# Normalize columns
for col in diet_df.columns:
    total = diet_df[col].sum()
    if total > 0:
        diet_df[col] = diet_df[col] / total

# Rename for rpath compatibility
diet_df.index.name = "Group"
diet_df_output = diet_df.reset_index()

print("Created model and diet matrix")
print(f"  Groups: {len(groups_list)}")
print(f"  Diet matrix shape: {diet_df.shape}")
print("")

# Create RpathParams
try:
    params = create_rpath_params(groups_list, types_list)
    
    # Fill parameters
    params.model["Biomass"] = model_df["Biomass"].values
    params.model["PB"] = model_df["PB"].values
    params.model["QB"] = model_df["QB"].values
    params.model["EE"] = model_df["EE"].values
    params.model["Unassim"] = model_df["Unassim"].values
    
    # Set diet (need to transpose to match expected format)
    params.diet = diet_df_output.copy()
    
    print("Running Ecopath solver...")
    balanced = rpath(params)
    
    print("SUCCESS! Model balanced.")
    print("")
    
    # Get results
    results_df = balanced.model.copy()
    print(f"Total Biomass: {results_df['Biomass'].sum():.1f} t/km2")
    print("")
    
    # Save
    results_df.to_csv("curonian_lagoon_balanced.csv")
    print("Saved: curonian_lagoon_balanced.csv")
    
except Exception as e:
    import traceback
    print("Error:")
    traceback.print_exc()

