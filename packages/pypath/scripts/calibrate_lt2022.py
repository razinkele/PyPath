"""
Calibrate the LT2022 Lithuanian coastal Ecopath/Ecosim model.

Enhanced calibration with three improvements:
  1. Apply FtimeAdjust from scenario 16 to fix baseline stability
     (prevents Zooplankton/Sprat crash from overconsumption)
  2. Apply forcing matrix VV overrides + forced biomass (Round goby invasion)
  3. Calibrate per-predator vulnerability using differential evolution

Indexing convention:
  model.Group[g]     0-based, 24 elements (0=Detritus, 1=Phyto, ...)
  model.Biomass[g]   0-based, 24 elements
  annual_Biomass[t, col]  col 0=Outside, col g+1 = model.Group[g]
  PreyFrom/PreyTo    0=Outside, 1=Detritus, 2=Phyto, ...  (=g+1)
  spname[s]          0=Outside, 1=Detritus, 2=Phyto, ...  (=g+1)

Usage:
    conda run -n shiny python packages/pypath/scripts/calibrate_lt2022.py
"""

import logging
import sys
import time
import warnings
from pathlib import Path

import numpy as np

warnings.filterwarnings("ignore", category=UserWarning)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s: %(message)s",
    stream=sys.stdout,
)
logger = logging.getLogger("calibrate")

PROJECT_ROOT = Path(__file__).resolve().parents[3]
DB_PATH = str(PROJECT_ROOT / "Data" / "LT2022_0.5ST_final7.eweaccdb")

# ── Step 1: Load and balance ───────────────────────────────────────────
logger.info("Loading model from %s", DB_PATH)
from pypath import read_ewemdb, read_ewemdb_table, rpath, rsim_scenario, rsim_run

params = read_ewemdb(DB_PATH)
model = rpath(params)
N_GROUPS = model.NUM_GROUPS  # 24

logger.info("Balanced: %d groups (%d living)", N_GROUPS, model.NUM_LIVING)
print("\n=== Groups (0-based) ===")
for g in range(N_GROUPS):
    gname = str(model.Group[g]).strip()
    print(
        f"  g={g:2d}  col={g + 1:2d}  {gname:<25s}  B={model.Biomass[g]:8.4f}  EE={model.EE[g]:.4f}"
    )

# ── Step 2: Parse observed time series ─────────────────────────────────
logger.info("Extracting time series (dataset #35)...")
ts_all = read_ewemdb_table(DB_PATH, "EcosimTimeSeries")
ds35 = ts_all[ts_all["DatasetID"] == 35].copy()

tsg = read_ewemdb_table(DB_PATH, "EcosimTimeSeriesGroup")
groups_db = read_ewemdb_table(DB_PATH, "EcopathGroup")
gid_to_name = dict(zip(groups_db["GroupID"], groups_db["GroupName"]))

ts_to_group = {}
for _, r in tsg.iterrows():
    db_gname = gid_to_name.get(r["GroupID"], "").strip()
    for g in range(N_GROUPS):
        if str(model.Group[g]).strip() == db_gname:
            ts_to_group[int(r["TimeSeriesID"])] = g
            break


def parse_ts(row):
    return np.array([float(x) for x in str(row["TimeValues"]).split()])


observed_rel = {}
observed_catch = {}
forced_bio = {}
effort_ts = None

for _, row in ds35.iterrows():
    vals = parse_ts(row)
    dtype = int(row["DatType"])
    ts_id = int(row["TimeSeriesID"])
    g = ts_to_group.get(ts_id)
    if dtype == 0 and g is not None:
        observed_rel[g] = vals
    elif dtype == 6 and g is not None:
        observed_catch[g] = vals
    elif dtype == -1 and g is not None:
        forced_bio[g] = vals
    elif dtype == 3:
        effort_ts = vals

n_years = len(effort_ts) if effort_ts is not None else 12
YEARS = range(1, n_years + 1)
logger.info(
    "%d years, %d rel.biomass, %d catch, %d forced bio, effort=%s",
    n_years,
    len(observed_rel),
    len(observed_catch),
    len(forced_bio),
    "yes" if effort_ts is not None else "no",
)

# ── Step 2b: Load scenario 16 settings ────────────────────────────────
logger.info("Loading scenario 16 settings...")

# FtimeAdjust per group from EcosimScenarioGroup
esg = read_ewemdb_table(DB_PATH, "EcosimScenarioGroup")
esg16 = esg[esg["ScenarioID"] == 16]

# Map EcopathGroupID -> 0-based model index
egid_to_model = {}
for _, row in groups_db.iterrows():
    gname = str(row["GroupName"]).strip()
    for g in range(N_GROUPS):
        if str(model.Group[g]).strip() == gname:
            egid_to_model[int(row["GroupID"])] = g
            break

ftime_adjust = np.zeros(N_GROUPS)  # 0-based per group
for _, row in esg16.iterrows():
    g = egid_to_model.get(int(row["EcopathGroupID"]), -1)
    if g >= 0:
        ftime_adjust[g] = float(row["FtimeAdjust"])

print("\n=== FtimeAdjust from scenario 16 ===")
for g in range(N_GROUPS):
    if ftime_adjust[g] != 0.5:
        print(
            f"  g={g:2d} {str(model.Group[g]).strip():<25s}  FtimeAdjust={ftime_adjust[g]:.1f}"
        )

# VV overrides from forcing matrix
fm = read_ewemdb_table(DB_PATH, "EcosimScenarioForcingMatrix")
fm16 = fm[fm["ScenarioID"] == 16]

ecosim_gid_to_model = {}
for _, row in esg16.iterrows():
    ecosim_gid_to_model[int(row["GroupID"])] = egid_to_model.get(
        int(row["EcopathGroupID"]), -1
    )

vv_overrides = {}
for _, row in fm16.iterrows():
    pg = ecosim_gid_to_model.get(int(row["PreyID"]), -1)
    dg = ecosim_gid_to_model.get(int(row["PredID"]), -1)
    if pg >= 0 and dg >= 0:
        vv_overrides[(pg, dg)] = float(row["vulnerability"])

n_nondefault = sum(1 for v in vv_overrides.values() if v != 2.0)
logger.info("VV overrides: %d total, %d non-default", len(vv_overrides), n_nondefault)


# ── Step 3: Scenario builder with all enhancements ────────────────────
def make_scenario(apply_overrides=True, apply_ftime=True, apply_forced_bio=True):
    """Create enhanced scenario."""
    sc = rsim_scenario(model, params, years=YEARS)

    # Apply FtimeAdjust: reduce Ftime for groups where FtimeAdjust < 0.5
    # This prevents overconsumption crashes by reducing predator foraging intensity
    # FtimeAdjust=0.5 is default; 0.0 means predator adjusts very slowly
    if apply_ftime:
        for g in range(N_GROUPS):
            ecosim_idx = g + 1
            # Groups with FtimeAdj=0.0 should have reduced foraging time
            # to prevent initial overconsumption crash
            if ftime_adjust[g] == 0.0:
                # Set Ftime < 1.0 to reduce predation pressure
                sc.start_state.Ftime[ecosim_idx] = 0.5

    # Apply fishing effort
    if effort_ts is not None:
        for yr_idx in range(min(n_years, len(effort_ts))):
            for month in range(12):
                t = yr_idx * 12 + month
                if t < sc.fishing.ForcedEffort.shape[0]:
                    sc.fishing.ForcedEffort[t, :] = effort_ts[yr_idx]

    # Apply VV overrides
    if apply_overrides:
        pf = sc.params.PreyFrom
        pt = sc.params.PreyTo
        for lk in range(len(pf)):
            key = (pf[lk] - 1, pt[lk] - 1)
            if key in vv_overrides:
                sc.params.VV[lk] = vv_overrides[key]

    # Apply forced biomass (Round goby invasion)
    if apply_forced_bio:
        for g, bio_ts in forced_bio.items():
            col = g + 1
            for yr_idx in range(min(n_years, len(bio_ts))):
                for month in range(12):
                    t = yr_idx * 12 + month
                    if (
                        t < sc.forcing.ForcedBio.shape[0]
                        and col < sc.forcing.ForcedBio.shape[1]
                    ):
                        sc.forcing.ForcedBio[t, col] = bio_ts[yr_idx]

    return sc


def compute_ss(output):
    """Compute total SS against observed relative biomass."""
    total_ss = 0.0
    per_group = {}
    for g, obs in observed_rel.items():
        col = g + 1
        sim_abs = output.annual_Biomass[:, col]
        initial_b = model.Biomass[g]
        sim_rel = sim_abs / initial_b if initial_b > 0 else sim_abs
        n = min(len(obs), len(sim_rel))
        ss = np.sum((sim_rel[:n] - obs[:n]) ** 2)
        per_group[g] = ss
        total_ss += ss
    return total_ss, per_group


def run_and_report(label, sc):
    out = rsim_run(sc, method="AB")
    crashed = out.crashed_groups
    if crashed:
        cnames = [
            str(sc.params.spname[c]) for c in crashed if c < len(sc.params.spname)
        ]
        logger.warning("%s crashes: %s", label, cnames)
    else:
        logger.info("%s: no crashes", label)
    ss, grps = compute_ss(out)
    return out, ss, grps


# ── Step 3b: Progressive baseline comparison ──────────────────────────
print("\n" + "=" * 60)
print("PROGRESSIVE BASELINE COMPARISON")
print("=" * 60)

# Plain
sc0 = make_scenario(apply_overrides=False, apply_ftime=False, apply_forced_bio=False)
out_plain, ss_plain, ss_plain_g = run_and_report("Plain", sc0)

# + Ftime
sc1 = make_scenario(apply_overrides=False, apply_ftime=True, apply_forced_bio=False)
out_ftime, ss_ftime, ss_ftime_g = run_and_report("+ Ftime", sc1)

# + Ftime + VV overrides
sc2 = make_scenario(apply_overrides=True, apply_ftime=True, apply_forced_bio=False)
out_vv, ss_vv, ss_vv_g = run_and_report("+ Ftime + VV", sc2)

# + Ftime + VV + forced bio
sc3 = make_scenario(apply_overrides=True, apply_ftime=True, apply_forced_bio=True)
out_full, ss_full, ss_full_g = run_and_report("+ Ftime + VV + ForcedBio", sc3)

print(f"\n{'Configuration':<35s}  {'SS':>10s}  {'vs Plain':>10s}")
print(f"  {'Plain baseline':<35s}  {ss_plain:10.2f}  {'---':>10s}")
print(
    f"  {'+ Ftime fix':<35s}  {ss_ftime:10.2f}  {(1 - ss_ftime / ss_plain) * 100:+9.1f}%"
)
print(
    f"  {'+ VV overrides':<35s}  {ss_vv:10.2f}  {(1 - ss_vv / ss_plain) * 100:+9.1f}%"
)
print(
    f"  {'+ Forced biomass':<35s}  {ss_full:10.2f}  {(1 - ss_full / ss_plain) * 100:+9.1f}%"
)

# Per-group comparison
print(
    f"\n{'Group':<25s}  {'SS_plain':>10s}  {'SS_enh':>10s}  {'R2_plain':>8s}  {'R2_enh':>8s}"
)
for g in sorted(observed_rel.keys()):
    gname = str(model.Group[g]).strip()
    obs = observed_rel[g]
    ss_tot = np.sum((obs - obs.mean()) ** 2)
    sp, se = ss_plain_g[g], ss_full_g[g]
    r2p = 1 - sp / ss_tot if ss_tot > 0 else float("nan")
    r2e = 1 - se / ss_tot if ss_tot > 0 else float("nan")
    print(f"  {gname:<25s}  {sp:10.4f}  {se:10.4f}  {r2p:8.3f}  {r2e:8.3f}")

# ── Step 4: Calibrate VV on enhanced baseline ─────────────────────────
prey_from = sc3.params.PreyFrom
prey_to = sc3.params.PreyTo

obs_ecosim = {g + 1 for g in observed_rel}

# Group calibratable links by predator
pred_to_links = {}
for lk in range(len(prey_from)):
    pg, dg = prey_from[lk] - 1, prey_to[lk] - 1
    # Skip links with non-default VV overrides (already set by modeler)
    if (pg, dg) in vv_overrides and vv_overrides[(pg, dg)] != 2.0:
        continue
    if prey_from[lk] in obs_ecosim or prey_to[lk] in obs_ecosim:
        eidx = prey_to[lk]
        if eidx not in pred_to_links:
            pred_to_links[eidx] = []
        pred_to_links[eidx].append(lk)

calib_preds = sorted(pred_to_links.keys())
initial_vv = sc3.params.VV.copy()
logger.info(
    "Calibrating %d predator-level VV (%d links)",
    len(calib_preds),
    sum(len(v) for v in pred_to_links.values()),
)

# ── Step 5: Differential evolution ────────────────────────────────────
from scipy.optimize import differential_evolution

eval_count = [0]
best_ss = [ss_full]


def objective(vv_mults):
    eval_count[0] += 1
    sc = make_scenario()
    for i, eidx in enumerate(calib_preds):
        for lk in pred_to_links[eidx]:
            sc.params.VV[lk] = vv_mults[i]
    try:
        out = rsim_run(sc, method="AB")
    except Exception:
        return 1e10
    ss, _ = compute_ss(out)
    return ss


def callback(xk, convergence=None):
    val = objective(xk)
    if val < best_ss[0]:
        best_ss[0] = val
        pct = (1 - val / ss_full) * 100
        logger.info(
            "  New best SS=%.4f (%.1f%% improvement), evals=%d", val, pct, eval_count[0]
        )
    return False


n_params = len(calib_preds)
bounds = [(1.0, 100.0)] * n_params

logger.info("Differential evolution: %d params, enhanced SS=%.4f", n_params, ss_full)
t0 = time.time()

result = differential_evolution(
    objective,
    bounds=bounds,
    maxiter=20,
    popsize=8,
    tol=0.01,
    seed=42,
    callback=callback,
    disp=True,
    workers=1,
)
elapsed = time.time() - t0

# ── Step 6: Final results ─────────────────────────────────────────────
sc_opt = make_scenario()
for i, eidx in enumerate(calib_preds):
    for lk in pred_to_links[eidx]:
        sc_opt.params.VV[lk] = result.x[i]
out_opt = rsim_run(sc_opt, method="AB")
_, ss_opt_g = compute_ss(out_opt)

print("\n" + "=" * 70)
print("FINAL CALIBRATION RESULTS")
print("=" * 70)
print(f"Time: {elapsed:.1f}s  |  Evals: {result.nfev}  |  Converged: {result.success}")
print(f"Plain baseline SS:     {ss_plain:.4f}")
print(f"Enhanced baseline SS:  {ss_full:.4f}  ({(1 - ss_full / ss_plain) * 100:+.1f}%)")
print(
    f"Calibrated SS:         {result.fun:.4f}  ({(1 - result.fun / ss_plain) * 100:+.1f}% vs plain)"
)

print(
    f"\n{'Group':<25s}  {'SS_plain':>10s}  {'SS_enh':>10s}  {'SS_cal':>10s}  {'R2_plain':>8s}  {'R2_cal':>8s}"
)
for g in sorted(observed_rel.keys()):
    gname = str(model.Group[g]).strip()
    obs = observed_rel[g]
    ss_tot = np.sum((obs - obs.mean()) ** 2)
    sp, se, so = ss_plain_g[g], ss_full_g[g], ss_opt_g[g]
    r2p = 1 - sp / ss_tot if ss_tot > 0 else float("nan")
    r2o = 1 - so / ss_tot if ss_tot > 0 else float("nan")
    print(f"  {gname:<25s}  {sp:10.4f}  {se:10.4f}  {so:10.4f}  {r2p:8.3f}  {r2o:8.3f}")

print("\n=== Optimized Predator VV ===")
print(f"{'Predator':<25s}  {'VV_enh':>8s}  {'VV_cal':>8s}  {'Links':>6s}")
spname = sc3.params.spname
for i, eidx in enumerate(calib_preds):
    gname = str(spname[eidx]).strip() if eidx < len(spname) else f"G{eidx}"
    enh_vv = initial_vv[pred_to_links[eidx][0]]
    print(
        f"  {gname:<25s}  {enh_vv:8.2f}  {result.x[i]:8.2f}  {len(pred_to_links[eidx]):6d}"
    )

# ── Step 7: Plot ──────────────────────────────────────────────────────
try:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    n_obs = len(observed_rel)
    n_cols = min(4, n_obs)
    n_rows = (n_obs + n_cols - 1) // n_cols
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(4 * n_cols, 3 * n_rows))
    axes = np.array(axes).flatten()
    years_plot = np.arange(2004, 2004 + n_years)

    for i, g in enumerate(sorted(observed_rel.keys())):
        gname = str(model.Group[g]).strip()
        obs = observed_rel[g]
        col = g + 1
        sim_p = out_plain.annual_Biomass[:, col] / model.Biomass[g]
        sim_e = out_full.annual_Biomass[:, col] / model.Biomass[g]
        sim_o = out_opt.annual_Biomass[:, col] / model.Biomass[g]
        n = min(len(obs), len(sim_p))

        axes[i].plot(years_plot[:n], obs[:n], "ko-", ms=3, label="Observed")
        axes[i].plot(years_plot[:n], sim_p[:n], "b--", alpha=0.4, lw=0.8, label="Plain")
        axes[i].plot(
            years_plot[:n], sim_e[:n], "g-.", alpha=0.6, lw=1, label="Enhanced"
        )
        axes[i].plot(years_plot[:n], sim_o[:n], "r-", lw=1.5, label="Calibrated")
        axes[i].set_title(gname, fontsize=8)
        axes[i].tick_params(labelsize=7)
        axes[i].grid(True, alpha=0.3)
        if i == 0:
            axes[i].legend(fontsize=6)

    for j in range(i + 1, len(axes)):
        axes[j].axis("off")

    plt.suptitle(
        f"LT2022 Calibration (SS: {ss_plain:.1f} → {ss_full:.1f} → {result.fun:.1f})",
        fontsize=11,
    )
    plt.tight_layout()
    outpath = PROJECT_ROOT / "Data" / "lt2022_calibration.png"
    plt.savefig(str(outpath), dpi=150, bbox_inches="tight")
    logger.info("Plot saved to %s", outpath)
except Exception as e:
    logger.warning("Could not save plot: %s", e)

print("\nDone.")
