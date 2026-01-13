import numpy as np
import pandas as pd
from pathlib import Path

from pypath.core.params import create_rpath_params
from pypath.core.ecopath import rpath
from pypath.core.ecosim import rsim_scenario, rsim_run, _build_active_link_matrix
from pypath.core.ecosim_deriv import deriv_vector, primary_production_forcing

REPO = Path(__file__).parent.parent
ECOPATH_DIR = REPO / "tests" / "data" / "rpath_reference" / "ecopath"
OUT_DIR = REPO / "tests" / "data" / "rpath_reference" / "ecosim" / "diagnostics"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# Load model params
model_df = pd.read_csv(ECOPATH_DIR / "model_params.csv")
diet_df = pd.read_csv(ECOPATH_DIR / "diet_matrix.csv")
params = create_rpath_params(model_df['Group'].tolist(), model_df['Type'].tolist())
params.model = model_df
params.diet = diet_df

# Build balanced rpath model and scenario
r = rpath(params)
scenario = rsim_scenario(r, params)

# Identify Seabirds index in Rref ordering
group_names = model_df['Group'].tolist()
seabirds_idx = group_names.index('Seabirds') + 0  # 0-based index matches deriv_vector indexing

# Helper to compute QQ and component breakdown for a given state and forcing
from copy import deepcopy

def compute_components(state, params_obj, forcing_dict, fishing_dict):
    # Build params dict as used by deriv_vector
    pr = params_obj
    params_dict = {
        "NUM_GROUPS": pr.NUM_GROUPS,
        "NUM_LIVING": pr.NUM_LIVING,
        "NUM_DEAD": pr.NUM_DEAD,
        "NUM_GEARS": pr.NUM_GEARS,
        "PB": pr.PBopt,
        "QB": pr.FtimeQBOpt,
        "M0": pr.MzeroMort,
        "Unassim": pr.UnassimRespFrac,
        "ActiveLink": _build_active_link_matrix(pr),
        "VV": pr.VV,
        "DD": pr.DD,
        "QQbase": pr.QQ,
        "Bbase": pr.B_BaseRef,
        "PP_type": pr.PP_type,
    }

    # Compute QQ exactly as deriv_vector does to inspect consumption/predation
    NUM_GROUPS = params_dict['NUM_GROUPS']
    NUM_LIVING = params_dict['NUM_LIVING']
    QQ = np.zeros((NUM_GROUPS + 1, NUM_GROUPS + 1))

    PB = params_dict['PB']
    QQbase = np.asarray(params_dict.get('QQbase', np.zeros((NUM_GROUPS + 1, NUM_GROUPS + 1))))
    ActiveLink = np.asarray(params_dict['ActiveLink'])
    VV = np.asarray(params_dict['VV'])
    DD = np.asarray(params_dict['DD'])
    Bbase = params_dict['Bbase']

    # Ensure 2D matrices are shaped correctly
    def _ensure_mat(mat):
        mat = np.asarray(mat)
        if mat.ndim == 1:
            try:
                return mat.reshape((NUM_GROUPS + 1, NUM_GROUPS + 1))
            except Exception:
                return np.zeros((NUM_GROUPS + 1, NUM_GROUPS + 1))
        return mat

    QQbase = _ensure_mat(QQbase)
    ActiveLink = _ensure_mat(ActiveLink).astype(bool)
    VV = _ensure_mat(VV)
    DD = _ensure_mat(DD)

    # Update params_dict to pass sanitized matrices into deriv_vector
    params_dict['QQbase'] = QQbase
    params_dict['VV'] = VV
    params_dict['DD'] = DD
    params_dict['ActiveLink'] = ActiveLink

    Ftime = forcing_dict.get('Ftime', np.ones(NUM_GROUPS + 1))
    ForcedPrey = forcing_dict.get('ForcedPrey', np.ones(NUM_GROUPS + 1))

    preyYY = np.zeros(NUM_GROUPS + 1)
    for i in range(1, NUM_GROUPS + 1):
        if Bbase[i] > 0:
            preyYY[i] = state[i] / Bbase[i] * ForcedPrey[i]

    predYY = np.zeros(NUM_GROUPS + 1)
    for i in range(1, NUM_LIVING + 1):
        if Bbase[i] > 0:
            predYY[i] = Ftime[i] * state[i] / Bbase[i]

    for pred in range(1, NUM_LIVING + 1):
        if state[pred] <= 0:
            continue
        for prey in range(1, NUM_GROUPS + 1):
            if not ActiveLink[prey, pred]:
                continue
            if state[prey] <= 0:
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

    # For each group i compute:
    # consumption_by_group = sum(QQ[:, i])  (consumption BY group i)
    # predation_loss = sum(QQ[i, :]) (predation on group i)

    consumption_by_group = np.sum(QQ[1:, 1:NUM_LIVING + 1], axis=0)
    predation_loss = np.sum(QQ[seabirds_idx + 1, 1:NUM_LIVING + 1]) if (seabirds_idx + 1) <= NUM_GROUPS else 0.0

    # Compute production for the Seabirds group
    i = seabirds_idx + 1  # 1-based for the formula
    PB_arr = params_dict['PB']
    QB_arr = params_dict.get('QB', np.zeros_like(PB_arr))

    if params_dict['PP_type'][i] > 0:
        # use primary production forcing
        PP_forcing = forcing_dict.get('PP_forcing', np.ones(NUM_GROUPS + 1))
        prod = primary_production_forcing(state, params_dict['Bbase'], PB_arr, PP_forcing, params_dict['PP_type'], params_dict['NUM_LIVING'])[i]
    elif QB_arr[i] > 0:
        GE = PB_arr[i] / QB_arr[i]
        prod = GE * consumption_by_group[i-1]  # consumption_by_group aligned to predators 1..NUM_LIVING
    else:
        prod = PB_arr[i] * state[i]

    # Fishing mortality
    FishMort = np.zeros(NUM_GROUPS + 1)
    FishFrom = fishing_dict.get('FishFrom', np.array([0]))
    FishThrough = fishing_dict.get('FishThrough', np.array([0]))
    FishQ = fishing_dict.get('FishQ', np.array([0.0]))
    for k in range(1, len(FishFrom)):
        grp = int(FishFrom[k])
        gear_group_idx = int(FishThrough[k])
        gear_idx = gear_group_idx - params_dict['NUM_LIVING'] - params_dict['NUM_DEAD']
        effort_mult = (forcing_dict.get('ForcedEffort')[gear_idx] if 0 < gear_idx < len(forcing_dict.get('ForcedEffort')) else 1.0)
        FishMort[grp] += FishQ[k] * effort_mult
    fish_loss = FishMort[i] * state[i]

    # M0 loss
    M0_arr = params_dict.get('M0', np.zeros(NUM_GROUPS + 1))
    m0_loss = M0_arr[i] * state[i]

    # Derivative (call deriv_vector for consistency)
    deriv = deriv_vector(state, params_dict, forcing_dict, {"FishingMort": FishMort})

    components = {
        'biomass': state[i],
        'production': prod,
        'predation_loss': predation_loss,
        'consumption_by_predator': consumption_by_group[i-1],
        'fish_loss': fish_loss,
        'm0_loss': m0_loss,
        'derivative': deriv[i],
    }
    return components


# Run both RK4 and AB step-by-step, recording components at each month
methods = ['RK4', 'AB']
for method in methods:
    print(f"Running diagnostic for method={method}")
    out_rows = []

    # Initialize state and history (for AB we will still call integrate_ab through rsim_run but
    # we need a step-by-step run to capture forcing at each month)
    from pypath.core.ecosim_deriv import integrate_rk4, integrate_ab

    params_obj = scenario.params
    # Prepare sanitized params dict for integrators
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
        "NoIntegrate": params_obj.NoIntegrate,
    }

    nyrs = 100
    n_months = nyrs * 12
    state = scenario.start_state.Biomass.copy()
    deriv_history = []

    for month in range(1, n_months + 1):
        t_idx = month - 1
        # Build forcing dict for timestep
        # Per-month forcing vectors (fall back to ones if not present)
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
        # Build fishing dict (we will let deriv_vector compute FishMort internally but provide base)
        fishing_dict = {'FishingMort': np.zeros(params_obj.NUM_GROUPS + 1)}

        # Record components BEFORE step
        comps_before = compute_components(state.copy(), params_obj, forcing_dict, fishing_dict)
        row = {'month': month, 'time': month / 12.0}
        row.update(comps_before)
        row['method'] = method
        out_rows.append(row)

        # Integrate one monthly step
        dt = 1.0 / 12.0
        if method == 'RK4':
            state = integrate_rk4(state, integrator_params, forcing_dict, {'FishingMort': np.zeros(params_obj.NUM_GROUPS + 1)}, dt)
        else:
            # AB: we use RK4 warmup to populate history
            if len(deriv_history) < 3:
                # use RK4 to populate history
                new_state = integrate_rk4(state, integrator_params, forcing_dict, {'FishingMort': np.zeros(params_obj.NUM_GROUPS + 1)}, dt)
                # store derivative
                deriv_now = deriv_vector(new_state, integrator_params, forcing_dict, {'FishingMort': np.zeros(params_obj.NUM_GROUPS + 1)})
                deriv_history.insert(0, deriv_now)
                state = new_state
            else:
                state, deriv_now = integrate_ab(state, deriv_history, integrator_params, forcing_dict, {'FishingMort': np.zeros(params_obj.NUM_GROUPS + 1)}, dt)

    # Save CSV
    df = pd.DataFrame(out_rows)
    out_path = OUT_DIR / f'seabirds_components_{method.lower()}.csv'
    df.to_csv(out_path, index=False)
    print(f"Saved {out_path} (rows={len(df)})")

    # Compare biomass series to R reference
    rref = pd.read_csv(REPO / 'tests' / 'data' / 'rpath_reference' / 'ecosim' / f'biomass_trajectory_{method.lower()}.csv')
    rvals = rref['Seabirds'].values[:len(df)]
    pyvals = df['biomass'].values
    # Correlation (safe check)
    if np.std(rvals) == 0 or np.std(pyvals) == 0:
        corr = 1.0 if np.allclose(rvals, pyvals, atol=1e-12) else np.nan
    else:
        corr = np.corrcoef(rvals, pyvals)[0,1]
    print(f"Method={method} Seabirds biomass correlation vs R: {corr}")

print('Done.')
