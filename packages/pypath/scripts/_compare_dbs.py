"""Compare original and exported EwE database structure."""

from pathlib import Path

from pypath.io.ewemdb import list_ewemdb_tables, read_ewemdb_table

orig = str(Path(__file__).parents[3] / "Data" / "LT2022_0.5ST_final7.eweaccdb")
export = str(Path(__file__).parents[3] / "Data" / "LT2022_calibrated.eweaccdb")

print("=" * 70)
print("ORIGINAL database tables:")
print("=" * 70)
orig_tables = sorted(list_ewemdb_tables(orig))
print(f"  Total: {len(orig_tables)}")
for t in orig_tables:
    try:
        df = read_ewemdb_table(orig, t)
        print(f"  {t:45s} {len(df):>5d} rows  cols: {list(df.columns)}")
    except Exception as e:
        print(f"  {t:45s} ERROR: {e}")

print()
print("=" * 70)
print("EXPORTED database tables:")
print("=" * 70)
exp_tables = sorted(list_ewemdb_tables(export))
print(f"  Total: {len(exp_tables)}")
for t in exp_tables:
    try:
        df = read_ewemdb_table(export, t)
        print(f"  {t:45s} {len(df):>5d} rows  cols: {list(df.columns)}")
    except Exception as e:
        print(f"  {t:45s} ERROR: {e}")

print()
print("=" * 70)
print("MISSING from exported (in original but not exported):")
print("=" * 70)
missing = set(orig_tables) - set(exp_tables)
for t in sorted(missing):
    print(f"  {t}")

print()
print("=" * 70)
print("EXTRA in exported (not in original):")
print("=" * 70)
extra = set(exp_tables) - set(orig_tables)
for t in sorted(extra):
    print(f"  {t}")
