from pathlib import Path
import numpy as np
import os
# silence verbose debug output from ecosim
os.environ['PYPATH_SILENCE_DEBUG'] = '1'
from pypath.core.ecopath import rpath
from pypath.core.params import create_rpath_params
from pypath.core.ecosim import rsim_scenario, rsim_run, _compute_Q_matrix, _build_link_matrix, _build_active_link_matrix

ECOPATH_DIR = Path("tests/data/rpath_reference/ecopath")
ECOSIM_DIR = Path("tests/data/rpath_reference/ecosim")

# load model
import pandas as pd
model_df = pd.read_csv(ECOPATH_DIR / "model_params.csv")
diet_df = pd.read_csv(ECOPATH_DIR / "diet_matrix.csv")

groups = model_df['Group'].tolist()
types = model_df['Type'].tolist()
params = create_rpath_params(groups, types)
params.model = model_df
params.diet = diet_df
pypath_model = rpath(params)
pypath_ecosim = rsim_scenario(pypath_model, params, years=range(1,101))
if 'Seabirds' not in pypath_ecosim.params.spname:
    print('Seabirds not present')
    raise SystemExit(0)
sidx = pypath_ecosim.params.spname.index('Seabirds')
# Check start_state forcing arrays
print('start_state.Ftime sample =', pypath_ecosim.start_state.Ftime[:10])
print('start_state.Ftime[Seabirds] =', pypath_ecosim.start_state.Ftime[sidx])
# run RK4 for 1 month to get state snapshot
out = rsim_run(pypath_ecosim, method='RK4', years=range(1,2))
state = out.out_Biomass[1]
# build params_dict same as test
from pypath.core.ecosim import _build_link_matrix
params_dict = {
    "NUM_GROUPS": pypath_ecosim.params.NUM_GROUPS,
    "NUM_LIVING": pypath_ecosim.params.NUM_LIVING,
    "NUM_DEAD": pypath_ecosim.params.NUM_DEAD,
    "NUM_GEARS": pypath_ecosim.params.NUM_GEARS,
    "PB": pypath_ecosim.params.PBopt,
    "QB": pypath_ecosim.params.FtimeQBOpt,
    "M0": pypath_ecosim.params.MzeroMort,
    "Unassim": pypath_ecosim.params.UnassimRespFrac,
    "ActiveLink": _build_active_link_matrix(pypath_ecosim.params),
    "VV": _build_link_matrix(pypath_ecosim.params, pypath_ecosim.params.VV),
    "DD": _build_link_matrix(pypath_ecosim.params, pypath_ecosim.params.DD),
    "QQbase": _build_link_matrix(pypath_ecosim.params, pypath_ecosim.params.QQ),
    "Bbase": pypath_ecosim.params.B_BaseRef,
    "PP_type": getattr(pypath_ecosim.params, 'PP_type', np.zeros(pypath_ecosim.params.NUM_GROUPS + 1)),
    "spname": pypath_ecosim.params.spname,
}
QQ = _compute_Q_matrix(params_dict, state, {"Ftime": np.ones_like(state)})
# print per-link breakdown for Seabirds consumption (at initial state)
n = len(QQ)
print('Seabirds sidx (1-based) =', sidx)
print('Per-link breakdown for Seabirds consumption (prey->Seabirds) at initial state:')
Bbase = params_dict['Bbase']
VV = params_dict['VV']
DD = params_dict['DD']
QQbase = params_dict['QQbase']
for prey in range(1, n):
    qbase = QQbase[prey, sidx]
    if qbase == 0:
        continue
    BBprey = float(state[prey])
    BBpred = float(state[sidx])
    PYY = BBprey / Bbase[prey] if Bbase[prey] > 0 else 0.0
    PDY = BBpred / Bbase[sidx] if Bbase[sidx] > 0 else 0.0
    dd = DD[prey, sidx]
    vv = VV[prey, sidx]
    dd_term = dd / (dd - 1.0 + max(PYY, 1e-10)) if dd > 1.0 else 1.0
    vv_term = vv / (vv - 1.0 + max(PDY, 1e-10)) if vv > 1.0 else 1.0
    Q_calc = qbase * PDY * PYY * dd_term * vv_term
    print(f"prey={prey} name={params_dict.get('spname',[None]*n)[prey]} BBprey={BBprey:.6e} Bbase={Bbase[prey]:.6e} PYY={PYY:.6e} PDY={PDY:.6e} qbase={qbase:.6e} dd_term={dd_term:.6e} vv_term={vv_term:.6e} Q_calc={Q_calc:.12e}")

print('Total consumption by Seabirds (col sum) =', QQ[:, sidx].sum())
print('Total predation on Seabirds (row sum) =', QQ[sidx, :].sum())

# Now compute RK4 stage states and QQ at each stage to see where Rpath may sample
from pypath.core.ecosim_deriv import deriv_vector

dt = 1.0 / 12.0
k1 = deriv_vector(state.copy(), params_dict, {"Ftime": np.ones_like(state)}, {"FishingMort": np.zeros_like(state)})
state_k2 = state + 0.5 * dt * k1
k2 = deriv_vector(state_k2, params_dict, {"Ftime": np.ones_like(state_k2)}, {"FishingMort": np.zeros_like(state_k2)})
state_k3 = state + 0.5 * dt * k2
k3 = deriv_vector(state_k3, params_dict, {"Ftime": np.ones_like(state_k3)}, {"FishingMort": np.zeros_like(state_k3)})
state_k4 = state + dt * k3
k4 = deriv_vector(state_k4, params_dict, {"Ftime": np.ones_like(state_k4)}, {"FishingMort": np.zeros_like(state_k4)})

for label, st in [('k1', state), ('k2', state_k2), ('k3', state_k3), ('k4', state_k4)]:
    QQs = _compute_Q_matrix(params_dict, st, {"Ftime": np.ones_like(st)})
    print(f"Stage {label}: total consumption by Seabirds =", float(np.nansum(QQs[:, sidx])))

