from pypath.core.params import create_rpath_params
from pypath.core.ecopath import rpath
import pandas as pd
import warnings
import numpy as np

warnings.filterwarnings("ignore")

print("Curonian Lagoon Ecopath Model - Final Results")
print("=" * 70)
print("")

# Load and run model
model_df = pd.read_csv("curonian_model_groups.csv")
groups_list = model_df["GroupName"].tolist()
types_list = model_df["Type"].tolist()

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

# Build diet
diet_rows = [{"Group": g} for g in groups_list]
for g in groups_list:
    for row in diet_rows:
        row[g] = 0.0
diet_df = pd.DataFrame(diet_rows).set_index("Group")

for pred, preys in diet_rules.items():
    if pred in diet_df.columns:
        for prey, prop in preys.items():
            if prey in diet_df.index:
                diet_df.loc[prey, pred] = prop

for col in diet_df.columns:
    total = diet_df[col].sum()
    if total > 0:
        diet_df[col] = diet_df[col] / total

diet_df.index.name = "Group"
diet_df_output = diet_df.reset_index()

# Run model
params = create_rpath_params(groups_list, types_list)
params.model["Biomass"] = model_df["Biomass"].values
params.model["PB"] = model_df["PB"].values
params.model["QB"] = model_df["QB"].values
params.model["EE"] = model_df["EE"].values
params.model["Unassim"] = model_df["Unassim"].values
params.diet = diet_df_output.copy()

balanced = rpath(params)

# Extract results into DataFrame
results = pd.DataFrame({
    "Group": balanced.Group,
    "Biomass": balanced.Biomass,
    "PB": balanced.PB,
    "QB": balanced.QB,
    "EE": balanced.EE,
    "TL": balanced.TL,
    "BA": balanced.BA,
})

print("Model successfully balanced!")
print("")
print(f"Total groups: {len(balanced.Group)}")
print(f"Total Biomass: {balanced.Biomass.sum():.1f} t/km2")
print(f"Total Production: {(balanced.Biomass * balanced.PB).sum():.1f} t/km2/yr")
print("")

print("Top 15 groups by Biomass:")
print("")
top = results.nlargest(15, "Biomass")[["Group", "Biomass", "PB", "QB", "TL"]]
for idx, row in top.iterrows():
    print(f"{row['Group']:30s}  B={row['Biomass']:7.2f}  PB={row['PB']:6.2f}  QB={row['QB']:6.2f}  TL={row['TL']:5.2f}")

print("")
print("Groups by Trophic Level:")
print("")
for tl_range in [(4.0, 5.0), (3.5, 4.0), (3.0, 3.5), (2.5, 3.0), (2.0, 2.5), (1.0, 2.0)]:
    mask = (results["TL"] >= tl_range[0]) & (results["TL"] < tl_range[1])
    subset = results[mask]
    if len(subset) > 0:
        print(f"TL {tl_range[0]:.1f}-{tl_range[1]:.1f}: {len(subset)} groups, Total B={subset['Biomass'].sum():.1f} t/km2")

# Save results
results.to_csv("curonian_lagoon_corrected_results.csv", index=False)
print("")
print("Results saved to: curonian_lagoon_corrected_results.csv")

