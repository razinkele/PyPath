"""Compare per-month QQ matrices computed from Rpath reference biomass vs PyPath RK4 output.

Produces CSV summary showing max absolute differences overall and for Seabirds
row/column (consumption and predation-on) per month.
"""
import numpy as np
import pandas as pd
from pathlib import Path

from pypath.core.params import create_rpath_params
from pypath.core.ecopath import rpath
from pypath.core.ecosim import rsim_scenario, rsim_run, _compute_Q_matrix, _build_active_link_matrix, _build_link_matrix

REFERENCE_DIR = Path("tests/data/rpath_reference")
ECOPATH_DIR = REFERENCE_DIR / "ecopath"
ECOSIM_DIR = REFERENCE_DIR / "ecosim"

if not REFERENCE_DIR.exists():
    print("Reference data not available")
    raise SystemExit(1)

model_df = pd.read_csv(ECOPATH_DIR / "model_params.csv")
diet_df = pd.read_csv(ECOPATH_DIR / "diet_matrix.csv")

groups = model_df["Group"].tolist()
types = model_df["Type"].tolist()

params = create_rpath_params(groups, types)
params.model = model_df
params.diet = diet_df

pypath_model = rpath(params)
scenario = rsim_scenario(pypath_model, params, years=range(1, 101))

# Run PyPath RK4 to obtain monthly biomass (if not already present)
print("Running PyPath RK4 run (this may take a short while)...")
out = rsim_run(scenario, method="rk4", years=range(1, 101))
py_biom = out.out_Biomass  # (n_months+1, n_groups+1)

# Load Rpath reference biomass trajectory
rpath_df = pd.read_csv(ECOSIM_DIR / "biomass_trajectory_rk4.csv")
# Ensure we have the same order & include outside column as first
spnames = ["Outside"] + groups

# Build mapping from column names to index
# rpath CSV likely has columns matching group names (no Outside); pad to NUM_GROUPS+1
NUM_GROUPS = scenario.params.NUM_GROUPS
NUM_LIVING = scenario.params.NUM_LIVING

n_py_months = py_biom.shape[0] - 1
n_rpath_months = rpath_df.shape[0]
max_months = min(n_py_months, n_rpath_months)
print(f"Comparing {max_months} months (PyPath={n_py_months}, Rpath={n_rpath_months})")

# Build params_dict for compute_Q
params_dict = {
    'NUM_GROUPS': NUM_GROUPS,
    'NUM_LIVING': NUM_LIVING,
    'NUM_DEAD': scenario.params.NUM_DEAD,
    'NUM_GEARS': scenario.params.NUM_GEARS,
    'PB': scenario.params.PBopt,
    'QB': scenario.params.FtimeQBOpt,
    'M0': scenario.params.MzeroMort.copy(),
    'Unassim': scenario.params.UnassimRespFrac,
    'ActiveLink': _build_active_link_matrix(scenario.params),
    'VV': _build_link_matrix(scenario.params, scenario.params.VV),
    'DD': _build_link_matrix(scenario.params, scenario.params.DD),
    'QQbase': _build_link_matrix(scenario.params, scenario.params.QQ),
    'Bbase': scenario.params.B_BaseRef,
    'PP_type': scenario.params.PP_type,
}

results = []

for month in range(0, max_months):
    # PyPath state at month
    state_py = py_biom[month].copy()
    forcing = {
        'Ftime': scenario.start_state.Ftime.copy(),
        'ForcedBio': np.where(scenario.forcing.ForcedBio[month] > 0, scenario.forcing.ForcedBio[month], 0),
        'ForcedMigrate': scenario.forcing.ForcedMigrate[month],
        'ForcedEffort': (scenario.fishing.ForcedEffort[month] if month < len(scenario.fishing.ForcedEffort) else np.ones(scenario.params.NUM_GEARS+1)),
    }
    QQ_py = _compute_Q_matrix(params_dict, state_py, forcing)

    # Build Rpath state vector (with leading Outside=1.0)
    rrow = rpath_df.iloc[month]
    state_r = np.zeros(NUM_GROUPS + 1)
    state_r[0] = 1.0
    for i, g in enumerate(groups, start=1):
        if g in rrow:
            state_r[i] = float(rrow[g])
        else:
            # If column missing, fallback to PyPath state for safety
            state_r[i] = float(state_py[i])

    QQ_r = _compute_Q_matrix(params_dict, state_r, forcing)

    diff = np.abs(QQ_py - QQ_r)
    max_abs = float(np.nanmax(diff))

    # Seabirds-specific indices
    if 'Seabirds' in scenario.params.spname:
        sidx = scenario.params.spname.index('Seabirds')
        seab_col_max = float(np.nanmax(np.abs(QQ_py[:, sidx] - QQ_r[:, sidx])))
        seab_row_max = float(np.nanmax(np.abs(QQ_py[sidx, :] - QQ_r[sidx, :])))
    else:
        sidx = None
        seab_col_max = np.nan
        seab_row_max = np.nan

    # Count significant link diffs > tol
    tol = 1e-9
    count_diff = int(np.sum(diff > tol))

    results.append({
        'month': month,
        'max_abs': max_abs,
        'seab_col_max': seab_col_max,
        'seab_row_max': seab_row_max,
        'n_links_diff': count_diff,
    })

# Save results
out = Path('build')
out.mkdir(exist_ok=True)
out_csv = out / 'qq_comparison_monthly.csv'
pd.DataFrame(results).to_csv(out_csv, index=False)

# Print summary worst months
df = pd.DataFrame(results)
print('Top months by seab_col_max:')
print(df.sort_values('seab_col_max', ascending=False).head(10))
print(f"Saved QQ comparison to {out_csv}")
