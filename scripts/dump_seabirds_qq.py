from pathlib import Path

import numpy as np
import pandas as pd

from pypath.core.ecopath import rpath
from pypath.core.ecosim import _build_active_link_matrix, rsim_scenario
from pypath.core.ecosim_deriv import deriv_vector, integrate_ab, integrate_rk4
from pypath.core.params import create_rpath_params

REPO = Path(__file__).parent.parent
ECOPATH_DIR = REPO / "tests" / "data" / "rpath_reference" / "ecopath"
OUT_DIR = REPO / "tests" / "data" / "rpath_reference" / "ecosim" / "diagnostics"
OUT_DIR.mkdir(parents=True, exist_ok=True)

model_df = pd.read_csv(ECOPATH_DIR / "model_params.csv")
diet_df = pd.read_csv(ECOPATH_DIR / "diet_matrix.csv")
params = create_rpath_params(model_df['Group'].tolist(), model_df['Type'].tolist())
params.model = model_df
params.diet = diet_df

r = rpath(params)
scenario = rsim_scenario(r, params)

group_names = model_df['Group'].tolist()
seabirds_idx = group_names.index('Seabirds')  # 0-based

# Prepare integrator params
params_obj = scenario.params

def _ensure_mat_global(mat):
    mat = np.asarray(mat)
    if mat.ndim == 1:
        try:
            return mat.reshape((params_obj.NUM_GROUPS + 1, params_obj.NUM_GROUPS + 1))
        except Exception:
            return np.zeros((params_obj.NUM_GROUPS + 1, params_obj.NUM_GROUPS + 1))
    return mat

integrator_params = {
    "NUM_GROUPS": params_obj.NUM_GROUPS,
    "NUM_LIVING": params_obj.NUM_LIVING,
    "NUM_DEAD": params_obj.NUM_DEAD,
    "NUM_GEARS": params_obj.NUM_GEARS,
    "PB": params_obj.PBopt,
    "QB": params_obj.FtimeQBOpt,
    "M0": params_obj.MzeroMort,
    "Unassim": params_obj.UnassimRespFrac,
    "ActiveLink": _ensure_mat_global(_build_active_link_matrix(params_obj)).astype(bool),
    "VV": _ensure_mat_global(params_obj.VV),
    "DD": _ensure_mat_global(params_obj.DD),
    "QQbase": _ensure_mat_global(params_obj.QQ),
    "Bbase": params_obj.B_BaseRef,
    "PP_type": params_obj.PP_type,
}

NUM_GROUPS = integrator_params['NUM_GROUPS']
NUM_LIVING = integrator_params['NUM_LIVING']

methods = ['RK4', 'AB']
for method in methods:
    print(f"Dumping QQ for method={method}")
    nyrs = 100
    n_months = nyrs * 12
    state = scenario.start_state.Biomass.copy()
    deriv_history = []

    # Prepare DataFrame columns: month + prey group names
    columns = ['month'] + group_names
    rows = []

    for month in range(1, n_months + 1):
        t_idx = month - 1
        # Build per-month forcing
        pp_forcing = (
            scenario.forcing.PP_forcing[t_idx]
            if hasattr(scenario.forcing, 'PP_forcing') and t_idx < len(scenario.forcing.PP_forcing)
            else np.ones(params_obj.NUM_GROUPS + 1)
        )
        forced_prey = (
            scenario.forcing.ForcedPrey[t_idx]
            if hasattr(scenario.forcing, 'ForcedPrey') and t_idx < len(scenario.forcing.ForcedPrey)
            else np.ones(params_obj.NUM_GROUPS + 1)
        )
        forcing_dict = {
            'Ftime': scenario.start_state.Ftime.copy(),
            'ForcedBio': np.where(scenario.forcing.ForcedBio[t_idx] > 0, scenario.forcing.ForcedBio[t_idx], 0),
            'ForcedMigrate': scenario.forcing.ForcedMigrate[t_idx],
            'ForcedEffort': (scenario.fishing.ForcedEffort[t_idx] if t_idx < len(scenario.fishing.ForcedEffort) else np.ones(params_obj.NUM_GEARS + 1)),
            'PP_forcing': pp_forcing,
            'ForcedPrey': forced_prey,
        }

        # Compute QQ matrix like deriv_vector does
        BB = state.copy()
        QQ = np.zeros((NUM_GROUPS + 1, NUM_GROUPS + 1))
        QQbase = integrator_params['QQbase']
        ActiveLink = integrator_params['ActiveLink']
        VV = integrator_params['VV']
        DD = integrator_params['DD']
        Bbase = integrator_params['Bbase']
        Ftime = forcing_dict['Ftime']

        preyYY = np.zeros(NUM_GROUPS + 1)
        for i in range(1, NUM_GROUPS + 1):
            if Bbase[i] > 0:
                preyYY[i] = BB[i] / Bbase[i] * forcing_dict.get('ForcedPrey', np.ones(NUM_GROUPS + 1))[i]
        predYY = np.zeros(NUM_GROUPS + 1)
        for i in range(1, NUM_LIVING + 1):
            if Bbase[i] > 0:
                predYY[i] = Ftime[i] * BB[i] / Bbase[i]
        for pred in range(1, NUM_LIVING + 1):
            if BB[pred] <= 0:
                continue
            for prey in range(1, NUM_GROUPS + 1):
                if not ActiveLink[prey, pred]:
                    continue
                if BB[prey] <= 0:
                    continue
                vv = VV[prey, pred]
                dd = DD[prey, pred]
                qbase = QQbase[prey, pred]
                if qbase <= 0:
                    continue
                PYY = preyYY[prey]
                PDY = predYY[pred]
                dd_term = dd / (dd - 1.0 + max(PYY, 1e-10)) if dd > 1.0 else 1.0
                vv_term = vv / (vv - 1.0 + max(PDY, 1e-10)) if vv > 1.0 else 1.0
                Q_calc = qbase * PDY * PYY * dd_term * vv_term
                QQ[prey, pred] = max(Q_calc, 0.0)

        # Save column for Seabirds predator (pred index seabirds_idx+1)
        pred_col = QQ[1:, seabirds_idx + 1]  # exclude index 0 (Outside)
        row = [month] + pred_col.tolist()
        rows.append(row)

        # Step integrator
        dt = 1.0 / 12.0
        if method == 'RK4':
            state = integrate_rk4(state, integrator_params, forcing_dict, {'FishingMort': np.zeros(params_obj.NUM_GROUPS + 1)}, dt)
        else:
            if len(deriv_history) < 3:
                new_state = integrate_rk4(state, integrator_params, forcing_dict, {'FishingMort': np.zeros(params_obj.NUM_GROUPS + 1)}, dt)
                deriv_now = deriv_vector(new_state, integrator_params, forcing_dict, {'FishingMort': np.zeros(params_obj.NUM_GROUPS + 1)})
                deriv_history.insert(0, deriv_now)
                state = new_state
            else:
                state, deriv_now = integrate_ab(state, deriv_history, integrator_params, forcing_dict, {'FishingMort': np.zeros(params_obj.NUM_GROUPS + 1)}, dt)
                deriv_history.insert(0, deriv_now)
                if len(deriv_history) > 3:
                    deriv_history.pop()

    df = pd.DataFrame(rows, columns=columns)
    out_path = OUT_DIR / f'seabirds_qq_{method.lower()}.csv'
    df.to_csv(out_path, index=False)
    print(f"Saved {out_path} (rows={len(df)})")

print('Done')
