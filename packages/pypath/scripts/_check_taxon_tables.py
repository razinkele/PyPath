"""Quick script to check taxonomy tables in real EwE database."""

from pypath.io.ewemdb import list_ewemdb_tables, read_ewemdb_table

db = "Data/LT2022_0.5ST_final7.eweaccdb"
tables = list_ewemdb_tables(db)
taxon_tables = [t for t in tables if "taxon" in t.lower() or "Taxon" in t]
print("Taxonomy tables:", taxon_tables)
for t in taxon_tables:
    try:
        df = read_ewemdb_table(db, t)
        print(f"\n{t}: {len(df)} rows, columns: {list(df.columns)}")
        if len(df) > 0:
            print(df.head(3).to_string())
    except Exception as e:
        print(f"{t}: ERROR - {e}")
