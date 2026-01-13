"""Compute parity metrics with adjusted rules to avoid false failures for low-variance groups.

Rules:
- A group is "low-variance" if the reference series stddev < max(absolute reference values) * 1e-8 or stddev < 1e-12.
- For low-variance groups we set `corr = NaN` and only consider `max_rel` for failure.
- Failure condition: (not low_variance and corr < 0.99) OR (max_rel > 0.05)

Writes: tmp_parity_final_report_adjusted.csv
"""
import numpy as np
import pandas as pd
from pathlib import Path

ACCUM = Path('tmp_parity_results_accum.csv')
RPATH = Path('tests/data/rpath_reference/ecosim/biomass_trajectory_rk4.csv')
OUT = Path('tmp_parity_final_report_adjusted.csv')

assert ACCUM.exists(), f"Missing {ACCUM}"
assert RPATH.exists(), f"Missing {RPATH}"

res_df = pd.read_csv(ACCUM)
model_df = pd.read_csv('tests/data/rpath_reference/ecopath/model_params.csv')
rpath_df = pd.read_csv(RPATH)

completed_years = sorted(res_df['year'].unique())
ny = len(completed_years)

# build r_yearly trimmed to available years
r_months = rpath_df.shape[0]
min_months = min(ny*12 if (ny:=ny if (ny:=ny) else ny) else 0, r_months) if False else min(ny*12, r_months)
# simpler approach
min_months = min(ny * 12, rpath_df.shape[0])
r_df_trim = rpath_df.iloc[:min_months]
r_yearly = r_df_trim.groupby(r_df_trim.index // 12).mean().iloc[:ny]

results = []
for gname in model_df['Group'].tolist():
    if gname not in r_yearly:
        continue
    r_series = r_yearly[gname].values[:ny]
    p_series = res_df.sort_values('year')[[gname]].values.flatten()[:ny]

    # define low-variance threshold
    std_r = np.nanstd(r_series)
    max_abs_r = np.nanmax(np.abs(r_series)) if r_series.size else 0.0
    low_var = (std_r < max(1e-12, max_abs_r * 1e-8))

    # compute correlation robustly; set to NaN when std is near-zero
    if low_var or np.all(np.isnan(r_series)) or np.all(np.isnan(p_series)):
        corr = np.nan
    else:
        # handle cases where p_series is constant as well
        if np.std(p_series) < 1e-14:
            corr = np.nan
        else:
            corr = float(np.corrcoef(r_series, p_series)[0, 1])

    denom = np.maximum(np.abs(r_series), 1e-12)
    rel_errs = np.abs(p_series - r_series) / denom
    max_rel = float(np.nanmax(rel_errs)) if rel_errs.size else np.nan
    mean_rel = float(np.nanmean(rel_errs)) if rel_errs.size else np.nan

    # failure rule: if not low_var check corr threshold; in all cases check max_rel threshold
    fail = ((not low_var and (np.isfinite(corr) and corr < 0.99)) or (max_rel > 0.05))

    results.append((gname, corr, max_rel, mean_rel, bool(low_var), bool(fail)))

par_df = pd.DataFrame(results, columns=['group', 'corr', 'max_rel', 'mean_rel', 'low_variance', 'fails_adjusted']).sort_values(['fails_adjusted', 'group'], ascending=[False, True])
par_df.to_csv(OUT, index=False)
print(f"Wrote {OUT} with {len(par_df)} groups. Failing groups: {par_df['fails_adjusted'].sum()} / {len(par_df)}")
print(par_df.head(20).to_string(index=False))
