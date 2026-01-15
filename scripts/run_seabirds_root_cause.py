from io import TextIOWrapper
import json
from typing import Dict, Tuple
import numpy as np
import pandas as pd
from pathlib import Path
from pypath.core.params import RpathParams, create_rpath_params
from pypath.core.ecopath import Rpath, rpath
from pypath.core.ecosim import RsimScenario, RsimParams, rsim_scenario, _build_link_matrix
from pypath.core.ecosim_deriv import integrate_rk4, deriv_vector

# Placeholder for legacy debug variable usages (some scripts reference `p` in debug expressions)
p = None

REPO: Path = Path(__file__).parent.parent
ECOPATH_DIR: Path = REPO / "tests" / "data" / "rpath_reference" / "ecopath"
DIAG_DIR: Path = REPO / "tests" / "data" / "rpath_reference" / "ecosim" / "diagnostics"
DIAG_DIR.mkdir(parents=True, exist_ok=True)

# Load model
model_df: pd.DataFrame = pd.read_csv(ECOPATH_DIR / "model_params.csv")
diet_df: pd.DataFrame = pd.read_csv(ECOPATH_DIR / "diet_matrix.csv")
params: RpathParams = create_rpath_params(model_df['Group'].tolist(), model_df['Type'].tolist())
params.model = model_df
params.diet = diet_df
r: Rpath | Tuple[Rpath, Dict[str, object]] = rpath(params)
scenario: RsimScenario = rsim_scenario(r, params)
sp: RsimParams = scenario.params
groups: list[Any] = model_df['Group'].tolist()
seabirds_idx: int = groups.index('Seabirds')  # 0-based
pred_idx: int = seabirds_idx + 1  # 1-based indexing used in ecosim arrays

# Build QQbase matrix
QQbase = _build_link_matrix(sp, sp.QQ)
VV = _build_link_matrix(sp, sp.VV)
DD = _build_link_matrix(sp, sp.DD)
Bbase = sp.B_BaseRef

# Helper to build forcing dictionary for month index t_idx
def forcing_for_month(t_idx):
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
    return forcing_dict

# Function to compute Q_calc per prey for predator pred_idx from a given state and forcing
def compute_qcalc_for_pred(state, forcing, pred_idx):
    NUM_GROUPS: int = sp.NUM_GROUPS
    NUM_LIVING: int = sp.NUM_LIVING
    BB = state
    Ftime = forcing.get('Ftime', np.ones(NUM_GROUPS + 1))
    ForcedPrey = forcing.get('ForcedPrey', np.ones(NUM_GROUPS + 1))

    preyYY: np.ndarray[tuple[int], np.dtype[np.float64]] = np.zeros(NUM_GROUPS + 1)
    for i: int in range(1, NUM_GROUPS + 1):
        if Bbase[i] > 0:
            preyYY[i] = BB[i] / Bbase[i] * ForcedPrey[i]
    predYY: np.ndarray[tuple[int], np.dtype[np.float64]] = np.zeros(NUM_GROUPS + 1)
    for i: int in range(1, NUM_LIVING + 1):
        if Bbase[i] > 0:
            predYY[i] = Ftime[i] * BB[i] / Bbase[i]

    qvals = {}
    for prey: int in range(1, NUM_GROUPS + 1):
        qbase = QQbase[prey, pred_idx]
        if qbase <= 0:
            qvals[prey] = 0.0
            continue
        PYY: np.ndarray[Tuple[int], np.dtype[np.float64]] = preyYY[prey]
        PDY: np.ndarray[tuple[int, ...], np.dtype[np.float64]] = predYY[pred_idx]
        dd = DD[prey, pred_idx]
        vv = VV[prey, pred_idx]
        dd_term: json.Any | float = dd / (dd - 1.0 + max(PYY, 1e-10)) if dd > 1.0 else 1.0
        vv_term: json.Any | float = vv / (vv - 1.0 + max(PDY, 1e-10)) if vv > 1.0 else 1.0
        Q_calc = qbase * PDY * PYY * dd_term * vv_term
        qvals[prey] = float(Q_calc)
    return qvals

# 1) Check month-1 Q_calc vs QQbase
state0 = scenario.start_state.Biomass.copy()
forcing0 = forcing_for_month(0)
q_month1 = compute_qcalc_for_pred(state0, forcing0, pred_idx)
month1_comparison = []
for prey: int in range(1, sp.NUM_GROUPS + 1):
    qcalc = q_month1[prey]
    qbase = float(QQbase[prey, pred_idx])
    ratio = None
    if qbase != 0:
        ratio = qcalc / qbase
    month1_comparison.append({'prey': prey, 'group': groups[prey-1] if prey-1 < len(groups) else str(prey-1), 'QQbase': qbase, 'Qcalc': qcalc, 'ratio': ratio})

# 2) Collect components for months 1..24 and 1200
months_to_check: list[int] = list(range(1, 25)) + [1200]
components_rows = []
state = scenario.start_state.Biomass.copy()
# integrate step-by-step and capture
for month: int in range(1, 1200 + 1):
    t_idx: int = month - 1
    forcing = forcing_for_month(t_idx)
    # compute components at start of month (before stepping)
    i: int = pred_idx
    # compute QQ matrix and predation loss on seabirds
    qvals = compute_qcalc_for_pred(state, forcing, pred_idx)
    predation_loss: int = sum(qvals.values())
    # consumption by seabirds is sum over prey QQ[:,pred_idx]
    consumption_by_pred = sum(qvals.values())
    # We'll compute consumption_by_pred directly
    NUM_GROUPS: int = sp.NUM_GROUPS
    BB = state
    Ftime = forcing['Ftime']
    ForcedPrey = forcing.get('ForcedPrey', np.ones(NUM_GROUPS + 1))
    preyYY: np.ndarray[tuple[int], np.dtype[np.float64]] = np.zeros(NUM_GROUPS + 1)
    for j: int in range(1, NUM_GROUPS + 1):
        if Bbase[j] > 0:
            preyYY[j] = BB[j] / Bbase[j] * ForcedPrey[j]
    predYY: np.ndarray[tuple[int], np.dtype[np.float64]] = np.zeros(NUM_GROUPS + 1)
    for j: int in range(1, sp.NUM_LIVING + 1):
        if Bbase[j] > 0:
            predYY[j] = Ftime[j] * BB[j] / Bbase[j]
    cons_by_pred = 0.0
    for prey: int in range(1, NUM_GROUPS + 1):
        qbase = QQbase[prey, pred_idx]
        if qbase <= 0:
            continue
        PYY: np.ndarray[Tuple[int], np.dtype[np.float64]] = preyYY[prey]
        PDY = predYY[pred_idx]
        dd = DD[prey, pred_idx]
        vv = VV[prey, pred_idx]
        dd_term: json.Any | float = dd / (dd - 1.0 + max(PYY, 1e-10)) if dd > 1.0 else 1.0
        vv_term: json.Any | float = vv / (vv - 1.0 + max(PDY, 1e-10)) if vv > 1.0 else 1.0
        Q_calc = qbase * PDY * PYY * dd_term * vv_term
        cons_by_pred += Q_calc

    # production
    PB = integrator_params_get = None
    PB = sp.PBopt
    QB = sp.FtimeQBOpt
    pp_type = sp.PP_type
    if pp_type[pred_idx] > 0:
        prod = 0.0
    elif QB[pred_idx] > 0:
        GE = PB[pred_idx] / QB[pred_idx]
        prod = GE * cons_by_pred
    else:
        prod = PB[pred_idx] * BB[pred_idx]

    # fishing loss (approx) - using FishingMort 0 unless fleets apply
    FishMort = 0.0
    # M0 loss
    M0_arr = sp.MzeroMort
    m0_loss: float = float(M0_arr[pred_idx] * BB[pred_idx]) if len(M0_arr) > pred_idx else 0.0

    # derivative
    deriv = deriv_vector(state, {'NUM_GROUPS': sp.NUM_GROUPS, 'NUM_LIVING': sp.NUM_LIVING, 'NUM_DEAD': sp.NUM_DEAD, 'NUM_GEARS': sp.NUM_GEARS, 'PB': sp.PBopt, 'QB': sp.FtimeQBOpt, 'M0': sp.MzeroMort, 'Unassim': sp.UnassimRespFrac, 'ActiveLink': _build_link_matrix(sp, sp.QQ), 'VV': _build_link_matrix(sp, sp.VV), 'DD': _build_link_matrix(sp, sp.DD), 'QQbase': QQbase, 'Bbase': sp.B_BaseRef, 'PP_type': sp.PP_type}, forcing, {'FishingMort': np.zeros(sp.NUM_GROUPS + 1)})

    if month in months_to_check:
        components_rows.append({
            'month': month,
            'biomass': float(BB[pred_idx]),
            'production': float(prod),
            'predation_loss': float(predation_loss),
            'consumption_by_predator': float(cons_by_pred),
            'fish_loss': float(FishMort * BB[pred_idx]),
            'm0_loss': float(m0_loss),
            'derivative': float(deriv[pred_idx]),
        })

    # step one month using RK4 to advance state for simplicity
    state = integrate_rk4(state, {'NUM_GROUPS': sp.NUM_GROUPS, 'NUM_LIVING': sp.NUM_LIVING, 'NUM_DEAD': sp.NUM_DEAD, 'NUM_GEARS': sp.NUM_GEARS, 'PB': sp.PBopt, 'QB': sp.FtimeQBOpt, 'M0': sp.MzeroMort, 'Unassim': sp.UnassimRespFrac, 'ActiveLink': _build_link_matrix(sp, sp.QQ), 'VV': _build_link_matrix(sp, sp.VV), 'DD': _build_link_matrix(sp, sp.DD), 'QQbase': QQbase, 'Bbase': sp.B_BaseRef, 'PP_type': sp.PP_type}, forcing, {'FishingMort': np.zeros(sp.NUM_GROUPS + 1)}, 1.0/12.0)

# Save outputs
pd.DataFrame(month1_comparison).to_csv(DIAG_DIR / 'seabirds_month1_qcalc_vs_qqbase.csv', index=False)
pd.DataFrame(components_rows).to_csv(DIAG_DIR / 'seabirds_components_key_months.csv', index=False)
summary = {
    'month1_q_mismatch': [{ 'group': r['group'], 'QQbase': r['QQbase'], 'Qcalc': r['Qcalc'], 'ratio': r['ratio'] } for r in month1_comparison if (r['QQbase'] != 0 and (abs(r['ratio'] - 1.0) > 1e-6))],
    'components_sample_months': [r['month'] for r in components_rows],
}
with open(DIAG_DIR / 'seabirds_root_cause_summary.json', 'w') as fh: TextIOWrapper[_WrappedBuffer]: TextIOWrapper[_WrappedBuffer]:
    json.dump(summary, fh, indent=2)
print('Saved outputs to', DIAG_DIR)
print('Month-1 Qcalc vs QQbase (non-1 ratios):')
for r in month1_comparison:
    if r['QQbase'] != 0 and (abs((r['Qcalc'] / r['QQbase']) - 1.0) > 1e-6):
        print(r)

print('\nKey months written:', [r['month'] for r in components_rows])
