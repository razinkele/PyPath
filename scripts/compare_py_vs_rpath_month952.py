from pathlib import Path

import pandas as pd

BASE = Path(__file__).resolve().parents[1]
py_fp = BASE / "build" / "seabirds_stage_per_prey_month952.csv"
r_fp = BASE / "build" / "seabirds_top_links_biomass_month952.csv"
out_fp = BASE / "build" / "seabirds_py_vs_rpath_month952.csv"

py = pd.read_csv(py_fp)
r = pd.read_csv(r_fp)

# merge on prey index where possible, else on prey name
merged = pd.merge(py, r[['prey','name','Q_r']], left_on=['prey_idx','prey_name'], right_on=['prey','name'], how='left')
merged = merged.rename(columns={'Q_final': 'Q_py'})
merged['Q_r'] = merged['Q_r'].fillna(pd.NA)
merged['abs_diff'] = (merged['Q_py'] - merged['Q_r']).abs()
merged['rel_diff'] = merged['abs_diff'] / merged['Q_r'].replace({0: pd.NA})

# select and order cols
out = merged[['prey_idx','prey_name','qbase','B_pre','Q_stage4','Q_py','Q_r','abs_diff','rel_diff']]
out.to_csv(out_fp, index=False)

print(f"Wrote comparison to {out_fp}")
print('\nTop diffs:')
print(out.sort_values('abs_diff', ascending=False).head(10).to_string(index=False))
