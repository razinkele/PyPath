import pandas as pd
import numpy as np
from pathlib import Path
from pypath.core.params import create_rpath_params
from pypath.core.ecopath import rpath
from pypath.core.ecosim import rsim_scenario, rsim_run, _build_active_link_matrix
from pypath.core.ecosim_deriv import deriv_vector, primary_production_forcing

REPO = Path(__file__).parent.parent
ECOPATH_DIR = REPO / "tests" / "data" / "rpath_reference" / "ecopath"
ECOSIM_DIR = REPO / "tests" / "data" / "rpath_reference" / "ecosim"
DIAG_DIR = ECOSIM_DIR / "diagnostics"

# Load model params
model_df = pd.read_csv(ECOPATH_DIR / "model_params.csv")
diet_df = pd.read_csv(ECOPATH_DIR / "diet_matrix.csv")
params = create_rpath_params(model_df["Group"].tolist(), model_df["Type"].tolist())
params.model = model_df
params.diet = diet_df

# Build balanced rpath model and scenario
r = rpath(params)
scenario = rsim_scenario(r, params)
params_obj = scenario.params

# Helper to compute per-group components (same approach as diagnostics)
from copy import deepcopy

def compute_components_for_state(state, forcing_dict, fishing_dict):
    pr = params_obj
    # Build params dict consistent with deriv_vector
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
        "NoIntegrate": pr.NoIntegrate,
    }

    # Use the same 'compute QQ' approach as diagnostics
    NUM_GROUPS = params_dict["NUM_GROUPS"]
    NUM_LIVING = params_dict["NUM_LIVING"]
    QQ = np.zeros((NUM_GROUPS + 1, NUM_GROUPS + 1))

    PB = params_dict["PB"]
    # Ensure 2D matrices
    def _ensure_mat(mat):
        mat = np.asarray(mat)
        if mat.ndim == 1:
            try:
                return mat.reshape((NUM_GROUPS + 1, NUM_GROUPS + 1))
            except Exception:
                return np.zeros((NUM_GROUPS + 1, NUM_GROUPS + 1))
        return mat

    QQbase = _ensure_mat(params_dict.get("QQbase", np.zeros((NUM_GROUPS + 1, NUM_GROUPS + 1))))
    ActiveLink = _ensure_mat(params_dict.get("ActiveLink", np.zeros((NUM_GROUPS + 1, NUM_GROUPS + 1)))).astype(bool)
    VV = _ensure_mat(params_dict.get("VV", np.zeros((NUM_GROUPS + 1, NUM_GROUPS + 1))))
    DD = _ensure_mat(params_dict.get("DD", np.zeros((NUM_GROUPS + 1, NUM_GROUPS + 1))))
    Bbase = params_dict["Bbase"]

    # Prey/pred relative arrays
    Ftime = forcing_dict.get("Ftime", np.ones(NUM_GROUPS + 1))
    ForcedPrey = forcing_dict.get("ForcedPrey", np.ones(NUM_GROUPS + 1))

    preyYY = np.zeros(NUM_GROUPS + 1)
    for i in range(1, NUM_GROUPS + 1):
        if Bbase[i] > 0:
            preyYY[i] = state[i] / Bbase[i] * ForcedPrey[i]

    predYY = np.zeros(NUM_GROUPS + 1)
    for i in range(1, NUM_LIVING + 1):
        if Bbase[i] > 0:
            predYY[i] = Ftime[i] * state[i] / Bbase[i]

    QQbase = np.asarray(QQbase)
    # compute QQ
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
            qbase = QQbase[prey, pred] if QQbase.shape == (NUM_GROUPS + 1, NUM_GROUPS + 1) else 0.0
            if qbase <= 0:
                continue
            PYY = preyYY[prey]
            PDY = predYY[pred]
            dd_term = dd / (dd - 1.0 + max(PYY, 1e-10)) if dd > 1.0 else 1.0
            vv_term = vv / (vv - 1.0 + max(PDY, 1e-10)) if vv > 1.0 else 1.0
            Q_calc = qbase * PDY * PYY * dd_term * vv_term
            QQ[prey, pred] = max(Q_calc, 0.0)

    # For a given focal index (Seabirds) compute components
    # Ensure params passed to deriv_vector have 2D matrices
    safe_params = params_dict.copy()
    safe_params['QQbase'] = QQbase
    safe_params['VV'] = VV
    safe_params['DD'] = DD
    safe_params['ActiveLink'] = ActiveLink
    return QQ, deriv_vector(state, safe_params, forcing_dict, fishing_dict)


def build_state_from_r_row(row):
    # row is pandas Series with first column Year then group columns
    # Map by group name to ensure correct indices
    cols = list(row.index)[1:]
    state = np.zeros(params_obj.NUM_GROUPS + 1)
    for g in cols:
        try:
            idx = params_obj.spname.index(g)
            state[idx] = row[g]
        except ValueError:
            # ignore unknown column names
            pass
    return state


# Run comparisons for RK4 and AB
methods = ["rk4", "ab"]
results = []
for method in methods:
    print(f"Comparing components for method={method}")
    py_csv = DIAG_DIR / f"seabirds_components_{method}.csv"
    r_csv = ECOSIM_DIR / f"biomass_trajectory_{method}.csv"
    if not py_csv.exists() or not r_csv.exists():
        print(f"Missing files for method={method}")
        continue
    py_df = pd.read_csv(py_csv)
    r_df = pd.read_csv(r_csv)

    months_to_check = list(range(1, 25)) + [1200]

    for month in months_to_check:
        idx = month - 1
        if idx >= len(py_df) or idx >= len(r_df):
            continue
        py_row = py_df.iloc[idx]
        r_row = r_df.iloc[idx]
        # Build R state vector
        r_state = build_state_from_r_row(r_row)
        # Forcing dict for that month
        forcing_dict = {
            "Ftime": scenario.start_state.Ftime.copy(),
            "ForcedPrey": scenario.forcing.ForcedPrey[idx] if idx < len(scenario.forcing.ForcedPrey) else np.ones(params_obj.NUM_GROUPS + 1),
            "ForcedBio": np.where(scenario.forcing.ForcedBio[idx] > 0, scenario.forcing.ForcedBio[idx], 0),
            "ForcedEffort": scenario.fishing.ForcedEffort[idx] if idx < len(scenario.fishing.ForcedEffort) else np.ones(params_obj.NUM_GEARS + 1),
            "PP_forcing": scenario.forcing.PP_forcing[idx] if hasattr(scenario.forcing, 'PP_forcing') and idx < len(scenario.forcing.PP_forcing) else np.ones(params_obj.NUM_GROUPS + 1),
        }
        fishing_dict = {"FishingMort": np.zeros(params_obj.NUM_GROUPS + 1)}

        QQ_r, deriv_r = compute_components_for_state(r_state, forcing_dict, fishing_dict)
        QQ_py, deriv_py = compute_components_for_state(np.insert(np.array(py_row["biomass"]), 0, 1.0) if False else r_state * 0 + 0, forcing_dict, fishing_dict)
        # Note: we will instead compute R components directly and compare with Py recorded values in py_row

        # Extract R components of interest using formulas (production and M0 loss)
        seab_i = params_obj.spname.index("Seabirds")
        i = seab_i
        B_r = r_state[i]
        # Production: if QB>0 then GE*consumption else PB*B
        PB = params_obj.PBopt[i]
        QB = params_obj.FtimeQBOpt[i]
        QQcol = QQ_r[:, i]
        consumption = float(np.nansum(QQcol))
        if QB > 0:
            GE = PB / QB
            prod_r = GE * consumption
        else:
            prod_r = PB * B_r
        predation_loss_r = float(np.nansum(QQ_r[i, 1 : params_obj.NUM_LIVING + 1])) if i <= params_obj.NUM_GROUPS else 0.0
        fish_loss_r = 0.0  # fishing dictionary uses zeros here
        m0_r = params_obj.MzeroMort[i]
        m0_loss_r = m0_r * B_r
        deriv_r_val = deriv_r[i]

        # Pull Py recorded components
        py_biomass = float(py_row['biomass'])
        py_production = float(py_row['production'])
        py_predation_loss = float(py_row['predation_loss'])
        py_fish_loss = float(py_row['fish_loss'])
        py_m0_loss = float(py_row['m0_loss'])
        py_derivative = float(py_row['derivative'])

        results.append({
            'method': method,
            'month': month,
            'r_biomass': B_r,
            'py_biomass': py_biomass,
            'prod_r': prod_r,
            'prod_py': py_production,
            'diff_prod': py_production - prod_r,
            'm0_r': m0_r,
            'm0_loss_r': m0_loss_r,
            'm0_loss_py': py_m0_loss,
            'diff_m0_loss': py_m0_loss - m0_loss_r,
            'pred_loss_r': predation_loss_r,
            'pred_loss_py': py_predation_loss,
            'diff_pred_loss': py_predation_loss - predation_loss_r,
            'fish_loss_r': fish_loss_r,
            'fish_loss_py': py_fish_loss,
            'diff_fish_loss': py_fish_loss - fish_loss_r,
            'deriv_r': deriv_r_val,
            'deriv_py': py_derivative,
            'diff_deriv': py_derivative - deriv_r_val,
        })

# Save results as CSV and print summary
out_df = pd.DataFrame(results)
out_path = DIAG_DIR / 'seabirds_parity_report.csv'
out_df.to_csv(out_path, index=False)
print('Saved parity report to', out_path)
print(out_df.groupby('method')[['diff_prod','diff_m0_loss','diff_pred_loss','diff_deriv']].agg(['mean','max','min']))
