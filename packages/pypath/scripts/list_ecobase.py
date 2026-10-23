"""List EcoBase models with summary statistics."""

from pypath.io.ecobase import list_ecobase_models

df = list_ecobase_models()
print(f"Total public models: {len(df)}")
print(f"Models with group counts: {(df.num_groups > 0).sum()}")
print()
print("=== Ecosystem Types ===")
types = df["ecosystem_type"].value_counts()
for t, c in types.items():
    if c > 3:
        print(f"  {t}: {c}")
print()
print("=== Top 20 Countries ===")
countries = df["country"].value_counts().head(20)
for c, n in countries.items():
    print(f"  {c}: {n}")
print()
print("=== Largest Models (by num_groups) ===")
big = df[df.num_groups > 0].nlargest(15, "num_groups")
for _, row in big.iterrows():
    print(
        f"  #{row.model_number} {row.model_name} ({row.country}) - {row.num_groups} groups"
    )
print()
print("=== Full Model List ===")
for _, row in df.sort_values("model_name").iterrows():
    print(
        f"  #{row.model_number:>4d}  {row.model_name[:50]:<50s}  {row.country[:30]:<30s}  {row.ecosystem_type[:20]:<20s}  {row.num_groups:>3d} groups"
    )
