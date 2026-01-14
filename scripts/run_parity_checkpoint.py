"""Parity runner with checkpointing.

Runs the PyPath RK4 trajectory in year-sized chunks, saves per-year metrics
after each chunk, and supports resuming from a checkpoint if interrupted.

Usage:
  python scripts/run_parity_checkpoint.py --years 100 --chunk-years 1 --report-top 10
"""
import argparse
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd

# Silence verbose debug prints in core modules
os.environ['PYPATH_SILENCE_DEBUG'] = os.environ.get('PYPATH_SILENCE_DEBUG', '1')

from pypath.core.ecopath import rpath
from pypath.core.ecosim import RsimScenario, rsim_run, rsim_scenario
from pypath.core.params import create_rpath_params

# Output files
CHECKPOINT_FILE = Path('tmp_parity_checkpoint.json')
RESULTS_FILE = Path('tmp_parity_results_accum.csv')

parser = argparse.ArgumentParser()
parser.add_argument('--years', type=int, default=100)
parser.add_argument('--chunk-years', type=int, default=1, help='Years per checkpoint (default 1)')
parser.add_argument('--report-top', type=int, default=10)
parser.add_argument('--force', action='store_true', help='overwrite existing checkpoint')
args = parser.parse_args()
NY = args.years
CHUNK = max(1, args.chunk_years)

RE = Path('tests/data/rpath_reference')
ECOPATH_DIR = RE / 'ecopath'
ECOSIM_DIR = RE / 'ecosim'

# Load reference
rpath_df = pd.read_csv(ECOSIM_DIR / 'biomass_trajectory_rk4.csv')
model_df = pd.read_csv(ECOPATH_DIR / 'model_params.csv')

# Prepare base params, rpath model and full scenario
params = create_rpath_params(model_df['Group'].tolist(), [int(t) for t in model_df['Type'].tolist()])
params.model = model_df
params.diet = pd.read_csv(ECOPATH_DIR / 'diet_matrix.csv')
py_rpath = rpath(params)
full_scenario = rsim_scenario(py_rpath, params, years=range(1, NY + 1))

# Helper: slice forcing/fishing for a chunk
def slice_forcing(scenario, start_year, n_years):
    start_idx = (start_year - scenario.start_year) * 12
    end_idx = start_idx + n_years * 12
    fs = scenario.forcing
    # Build a new forcing dataclass with sliced arrays
    class FS:
        pass
    newf = FS()
    for attr in ['ForcedPrey','ForcedMort','ForcedRecs','ForcedSearch','ForcedActresp','ForcedMigrate','ForcedBio','ForcedEffort']:
        arr = getattr(fs, attr)
        newarr = arr[start_idx:end_idx].copy()
        setattr(newf, attr, newarr)
    return newf

# Helper: slice fishing
def slice_fishing(scenario, start_year, n_years):
    start_idx = (start_year - scenario.start_year) * 12
    end_idx = start_idx + n_years * 12
    fk = scenario.fishing
    class FK:
        pass
    newfk = FK()
    newfk.ForcedEffort = fk.ForcedEffort[start_idx:end_idx].copy()
    newfk.ForcedFRate = fk.ForcedFRate[(start_year - scenario.start_year):(start_year - scenario.start_year + n_years)].copy()
    newfk.ForcedCatch = fk.ForcedCatch[(start_year - scenario.start_year):(start_year - scenario.start_year + n_years)].copy()
    return newfk

# Resume or initialize checkpoint
if CHECKPOINT_FILE.exists() and not args.force:
    chk = json.loads(CHECKPOINT_FILE.read_text())
    current_year = chk.get('last_completed_year', 1)
    print(f"Resuming from checkpoint year {current_year}")
    if RESULTS_FILE.exists():
        accum_df = pd.read_csv(RESULTS_FILE)
    else:
        accum_df = pd.DataFrame()
else:
    current_year = 1
    accum_df = pd.DataFrame()
    if CHECKPOINT_FILE.exists():
        CHECKPOINT_FILE.unlink()
    if RESULTS_FILE.exists():
        RESULTS_FILE.unlink()

# Start state to be updated after each chunk
start_state = full_scenario.start_state
scenario_start_year = full_scenario.start_year

# Main loop over chunks
while current_year <= NY:
    remaining = NY - current_year + 1
    chunk_years = min(CHUNK, remaining)
    print(f"Running years {current_year}..{current_year+chunk_years-1} ({chunk_years} years)")
    # Build a chunk scenario with sliced forcing/fishing and current start state
    chunk_forcing = slice_forcing(full_scenario, current_year, chunk_years)
    chunk_fishing = slice_fishing(full_scenario, current_year, chunk_years)
    chunk_scenario = RsimScenario(
        params=full_scenario.params,
        start_state=start_state,
        forcing=chunk_forcing,
        fishing=chunk_fishing,
        stanzas=full_scenario.stanzas,
        eco_name=full_scenario.eco_name,
        start_year=current_year,
    )

    try:
        out = rsim_run(chunk_scenario, method='rk4', years=range(current_year, current_year + chunk_years))
    except Exception as e:
        # Save checkpoint and re-raise
        print(f"ERROR: simulation failed during years {current_year}..{current_year+chunk_years-1}: {e}")
        CHECKPOINT_FILE.write_text(json.dumps({'last_completed_year': current_year}))
        accum_df.to_csv(RESULTS_FILE, index=False)
        raise

    # Convert monthly to annual for the chunk
    py_monthly = out.out_Biomass
    months = py_monthly.shape[0]
    n_years_chunk = months // 12
    py_yearly = py_monthly[: n_years_chunk * 12].reshape((n_years_chunk, 12, -1)).mean(axis=1)

    # Build DataFrame with yearly rows for this chunk
    groups = model_df['Group'].tolist()
    for yi in range(n_years_chunk):
        yindex = current_year + yi
        row = {'year': yindex}
        for gi, gname in enumerate(groups, start=1):
            row[gname] = float(py_yearly[yi, gi])
        accum_df = pd.concat([accum_df, pd.DataFrame([row])], ignore_index=True)

    # Persist partial results and checkpoint
    accum_df.to_csv(RESULTS_FILE, index=False)
    last_done = current_year + chunk_years
    CHECKPOINT_FILE.write_text(json.dumps({'last_completed_year': last_done}))

    # Update start_state for next chunk: take last monthly state
    final_state = out.out_Biomass[-1].copy()
    start_state = start_state
    # Update the start_state object's Biomass in-place if possible
    try:
        start_state.Biomass = final_state
    except Exception:
        # Try to build a new state object using same type
        class S:
            pass
        new_s = S()
        new_s.Biomass = final_state
        new_s.Ftime = out.Ftime if hasattr(out, 'Ftime') else start_state.Ftime
        start_state = new_s

    current_year += chunk_years

print('\nCheckpoint run complete. Computing parity metrics...')
# Aggregate years and compute correlations against rpath
# Build annual Rpath trimmed to completed years
res_df = pd.read_csv(RESULTS_FILE)
completed_years = sorted(res_df['year'].unique())
ny_done = len(completed_years)
rpath_months = rpath_df.shape[0]
min_months = min(ny_done*12, rpath_months)
r_df_trim = rpath_df.iloc[:min_months]
r_yearly = r_df_trim.groupby(r_df_trim.index // 12).mean().iloc[:ny_done]

results = []
for gname in model_df['Group'].tolist():
    if gname not in r_yearly:
        continue
    r_series = r_yearly[gname].values[:ny_done]
    p_series = res_df.sort_values('year')[[gname]].values.flatten()[:ny_done]
    if np.all(np.isnan(r_series)) or np.all(np.isnan(p_series)):
        corr = np.nan
    else:
        corr = np.corrcoef(r_series, p_series)[0, 1]
    denom = np.maximum(np.abs(r_series), 1e-12)
    rel_errs = np.abs(p_series - r_series) / denom
    max_rel = np.nanmax(rel_errs) if rel_errs.size else np.nan
    mean_rel = np.nanmean(rel_errs) if rel_errs.size else np.nan
    results.append((gname, corr, max_rel, mean_rel))

par_df = pd.DataFrame(results, columns=['group', 'corr', 'max_rel', 'mean_rel']).sort_values('corr')
# Save final parity report
par_df.to_csv('tmp_parity_final_report.csv', index=False)

# Print concise summary
sname = 'Seabirds'
row = par_df[par_df['group'] == sname]
print('\n=== Seabirds parity summary ===')
if not row.empty:
    print(row.to_string(index=False))
else:
    print('Seabirds not found in results')

print(f"\nTop {args.report_top} worst-correlated groups:")
print(par_df.head(args.report_top).to_string(index=False))

n_bad = ((par_df['corr'] < 0.99) | (par_df['max_rel'] > 0.05)).sum()
print(f"\nGroups failing parity (corr<0.99 or max_rel>0.05): {n_bad} / {len(par_df)}")
print('\nDone')
