import numpy as np
import pandas as pd
from pathlib import Path
from pypath.core.params import create_rpath_params
from pypath.core.ecopath import rpath
from pypath.core.ecosim import rsim_scenario
from pypath.core.ecosim import _build_link_matrix

REPO = Path(__file__).parent.parent
ECOPATH_DIR = REPO / 'tests' / 'data' / 'rpath_reference' / 'ecopath'
DIAG_DIR = REPO / 'tests' / 'data' / 'rpath_reference' / 'ecosim' / 'diagnostics'

model_df = pd.read_csv(ECOPATH_DIR / 'model_params.csv')
diet_df = pd.read_csv(ECOPATH_DIR / 'diet_matrix.csv')
params = create_rpath_params(model_df['Group'].tolist(), model_df['Type'].tolist())
params.model = model_df
params.diet = diet_df

r = rpath(params)
scenario = rsim_scenario(r, params)
sp = scenario.params

groups = model_df['Group'].tolist()
seabirds_idx = groups.index('Seabirds') # 0-based
pred_idx = seabirds_idx + 1  # 1-based for ecosim

QQbase_mat = _build_link_matrix(sp, sp.QQ)
col = QQbase_mat[:, pred_idx]
nonzero_prey = [i for i, v in enumerate(col) if v != 0.0]
print('Seabirds predator index (ecosim 1-based):', pred_idx)
print('Non-zero QQbase prey indices (ecosim 1-based):', nonzero_prey)
print('Corresponding group names:')
for i in nonzero_prey:
    name = groups[i-1] if i-1 < len(groups) else f'idx{i-1}'
    print(' ', i, name, 'QQbase=', QQbase_mat[i, pred_idx])

# Choose month to inspect
month = 1200
# Load saved state: we'll recompute state by running RK4 up to month
from pypath.core.ecosim_deriv import integrate_rk4, integrate_ab, deriv_vector

integrator_params = {
    "NUM_GROUPS": sp.NUM_GROUPS,
    "NUM_LIVING": sp.NUM_LIVING,
    "NUM_DEAD": sp.NUM_DEAD,
    "NUM_GEARS": sp.NUM_GEARS,
    "PB": sp.PBopt,
    "QB": sp.FtimeQBOpt,
    "M0": sp.MzeroMort,
    "Unassim": sp.UnassimRespFrac,
    "ActiveLink": _build_link_matrix(sp, sp.ActiveLink) if hasattr(sp, 'ActiveLink') else _build_link_matrix(sp, sp.QQ),
    "VV": _build_link_matrix(sp, sp.VV),
    "DD": _build_link_matrix(sp, sp.DD),
    "QQbase": QQbase_mat,
    "Bbase": sp.B_BaseRef,
    "PP_type": sp.PP_type,
}

# Simulate to month using RK4 to get final state
state = scenario.start_state.Biomass.copy()
for m in range(1, month + 1):
    t_idx = m - 1
    pp_forcing = (
        scenario.forcing.PP_forcing[t_idx]
        if hasattr(scenario.forcing, 'PP_forcing') and t_idx < len(scenario.forcing.PP_forcing)
        else np.ones(sp.NUM_GROUPS + 1)
    )
    forced_prey = (
        scenario.forcing.ForcedPrey[t_idx]
        if hasattr(scenario.forcing, 'ForcedPrey') and t_idx < len(scenario.forcing.ForcedPrey)
        else np.ones(sp.NUM_GROUPS + 1)
    )
    forcing_dict = {
        'Ftime': scenario.start_state.Ftime.copy(),
        'ForcedBio': np.where(scenario.forcing.ForcedBio[t_idx] > 0, scenario.forcing.ForcedBio[t_idx], 0),
        'ForcedMigrate': scenario.forcing.ForcedMigrate[t_idx],
        'ForcedEffort': (scenario.fishing.ForcedEffort[t_idx] if t_idx < len(scenario.fishing.ForcedEffort) else np.ones(sp.NUM_GEARS + 1)),
        'PP_forcing': pp_forcing,
        'ForcedPrey': forced_prey,
    }
    state = integrate_rk4(state, integrator_params, forcing_dict, {'FishingMort': np.zeros(sp.NUM_GROUPS + 1)}, 1.0/12.0)

print('\nState at month', month, 'Seabirds biomass =', state[pred_idx])

# For each nonzero prey, compute PYY, PDY, dd_term, vv_term, qbase, Q_calc, and value from saved CSV
BB = state.copy()
Bbase = integrator_params['Bbase']
Ftime = forcing_dict['Ftime']

for prey in nonzero_prey:
    prey_name = groups[prey-1] if prey-1 < len(groups) else f'idx{prey-1}'
    qbase = QQbase_mat[prey, pred_idx]
    PYY = BB[prey] / Bbase[prey] * forcing_dict.get('ForcedPrey', np.ones(sp.NUM_GROUPS + 1))[prey] if Bbase[prey] > 0 else 0.0
    PDY = Ftime[pred_idx] * BB[pred_idx] / Bbase[pred_idx] if Bbase[pred_idx] > 0 else 0.0
    dd = integrator_params['DD'][prey, pred_idx]
    vv = integrator_params['VV'][prey, pred_idx]
    dd_term = dd / (dd - 1.0 + max(PYY, 1e-10)) if dd > 1.0 else 1.0
    vv_term = vv / (vv - 1.0 + max(PDY, 1e-10)) if vv > 1.0 else 1.0
    Q_calc = qbase * PDY * PYY * dd_term * vv_term
    print(f"Prey={prey_name}: qbase={qbase:.6g}, BB_prey={BB[prey]:.6g}, Bbase_pre={Bbase[prey]:.6g}, PYY={PYY:.6g}, PDY={PDY:.6g}, dd={dd:.6g}, vv={vv:.6g}, dd_term={dd_term:.6g}, vv_term={vv_term:.6g}, Q_calc={Q_calc:.6g}")

# Print QQ saved value for that month
q_rk = pd.read_csv(DIAG_DIR / 'seabirds_qq_rk4.csv')
row = q_rk[q_rk['month'] == month]
print('\nSaved QQ values (RK4) at month', month, ':')
print(row.iloc[0].to_dict())
