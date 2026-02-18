from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from pypath.core.ecopath import rpath
from pypath.core.ecosim import rsim_run, rsim_scenario
from pypath.core.ecosim_deriv import deriv_vector, integrate_rk4
from pypath.core.params import create_rpath_params

REFERENCE_DIR = Path("tests/data/rpath_reference")
ECOPATH_DIR = REFERENCE_DIR / "ecopath"


def _build_pdict_from_scenario(scenario):
    from pypath.core.ecosim import _build_active_link_matrix, _build_link_matrix

    pdict = {
        "NUM_GROUPS": scenario.params.NUM_GROUPS,
        "NUM_LIVING": scenario.params.NUM_LIVING,
        "NUM_DEAD": scenario.params.NUM_DEAD,
        "NUM_GEARS": scenario.params.NUM_GEARS,
        "PB": scenario.params.PBopt,
        "QB": scenario.params.FtimeQBOpt,
        "M0": scenario.params.MzeroMort.copy(),
        "Unassim": scenario.params.UnassimRespFrac,
        "ActiveLink": _build_active_link_matrix(scenario.params),
        "VV": _build_link_matrix(scenario.params, scenario.params.VV),
        "DD": _build_link_matrix(scenario.params, scenario.params.DD),
        "QQbase": _build_link_matrix(scenario.params, scenario.params.QQ),
        "Bbase": scenario.params.B_BaseRef,
        "PP_type": scenario.params.PP_type,
        "spname": scenario.params.spname,
        # fish mappings intentionally included from scenario.params
        "FishFrom": getattr(scenario.params, "FishFrom", None),
        "FishTo": getattr(scenario.params, "FishTo", None),
        "FishQ": getattr(scenario.params, "FishQ", None),
    }
    return pdict


def test_short_rk4_step_changes_discards_with_fish_discard_mappings():
    """Integration-level test: one RK4 step (or derivative) should show a
    measurable increase in the Discards derivative when fish-derived discard
    mappings are enabled in the params.

    The test runs quickly using the scenario start state and a single RK4 step.
    """
    model_df = pd.read_csv(ECOPATH_DIR / "model_params.csv")
    diet_df = pd.read_csv(ECOPATH_DIR / "diet_matrix.csv")

    params = create_rpath_params(model_df["Group"].tolist(), [int(t) for t in model_df["Type"].tolist()])
    params.model = model_df
    params.diet = diet_df

    pypath_model = rpath(params)
    scenario = rsim_scenario(pypath_model, params, years=range(1, 3))

    # If the reference params do not include fish discard mappings skip the test
    if not (hasattr(scenario.params, "FishFrom") and getattr(scenario.params, "FishFrom") is not None):
        pytest.skip("Reference params have no FishFrom/FishTo/FishQ mappings to test")

    state0 = scenario.start_state.Biomass.copy()

    pdict_with = _build_pdict_from_scenario(scenario)
    pdict_no_fish = pdict_with.copy()
    pdict_no_fish["FishFrom"] = None
    pdict_no_fish["FishTo"] = None
    pdict_no_fish["FishQ"] = None

    # Forcing and fishing: minimal objects (Ftime etc.)
    forcing = {"Ftime": scenario.start_state.Ftime.copy(), "PP_forcing": np.ones(pdict_with["NUM_GROUPS"] + 1)}
    fishing = {"FishFrom": getattr(scenario.params, "FishFrom", None), "FishThrough": getattr(scenario.params, "FishThrough", None), "FishQ": getattr(scenario.params, "FishQ", None)}

    # Index for Discards
    if "Discards" not in pdict_with["spname"]:
        pytest.skip("Reference model has no Discards group; cannot run test")
    idx_discards = pdict_with["spname"].index("Discards")

    # Compute derivatives
    deriv_with = deriv_vector(state0, pdict_with, forcing, fishing)
    deriv_no = deriv_vector(state0, pdict_no_fish, forcing, fishing)

    # Expect that fish-derived mappings increase the Discards derivative
    diff = deriv_with[idx_discards] - deriv_no[idx_discards]
    assert diff >= 0.0, f"Discards derivative not increased by fish-derived discards (diff={diff})"
    assert abs(diff) > 1e-12, "Measured effect is too small to be meaningful"

    # Also check a single RK4 step results in a larger short-term increase
    # in Discards biomass when fish-derived mappings are present.
    dt = 1.0
    new_state_with = integrate_rk4(state0, pdict_with, forcing, fishing, dt)
    new_state_no = integrate_rk4(state0, pdict_no_fish, forcing, fishing, dt)

    delta_with = new_state_with[idx_discards] - state0[idx_discards]
    delta_no = new_state_no[idx_discards] - state0[idx_discards]

    assert delta_with - delta_no >= 0.0, "RK4 step did not increase Discards more when fish-derived discards were enabled"
    assert abs(delta_with - delta_no) > 1e-12, "RK4 short-term effect is too small"


@pytest.fixture
def synthetic_fish_scenario():
    """Create a small synthetic scenario (fast) with fish-derived discard mappings.

    Structure:
      - 3 living groups: 1=FishSource, 2=Forage, 3=Seabirds
      - 1 detritus group: 4=Discards
    """
    NUM_LIVING = 3
    NUM_DEAD = 1
    NUM_GROUPS = NUM_LIVING + NUM_DEAD

    spname = [None, "FishSource", "Forage", "Seabirds", "Discards"]

    # initial biomass
    state0 = np.zeros(NUM_GROUPS + 1)
    state0[1] = 10.0
    state0[2] = 8.0
    state0[3] = 1.5
    state0[4] = 0.5

    PB = np.zeros(NUM_GROUPS + 1)
    QB = np.zeros(NUM_GROUPS + 1)
    M0 = np.zeros(NUM_GROUPS + 1)
    Unassim = np.zeros(NUM_GROUPS + 1)

    M0[1] = 0.05
    M0[2] = 0.02
    M0[3] = 0.02
    Unassim[1] = 0.2
    Unassim[2] = 0.15
    Unassim[3] = 0.2

    QQbase = np.zeros((NUM_GROUPS + 1, NUM_GROUPS + 1))
    ActiveLink = np.zeros((NUM_GROUPS + 1, NUM_GROUPS + 1), dtype=bool)

    # Seabirds (3) consume FishSource (1) and Forage (2)
    ActiveLink[1, 3] = True
    ActiveLink[2, 3] = True
    QQbase[1, 3] = 0.3
    QQbase[2, 3] = 0.2

    # Forage (2) consumes Discards (4) to create a detritus -> forage -> seabirds pathway
    ActiveLink[4, 2] = True
    QQbase[4, 2] = 0.5

    # Ensure consumer production responds to consumption for Forage and Seabirds
    QB[2] = 0.5
    PB[2] = 0.5
    QB[3] = 0.3
    PB[3] = 0.3

    VV = np.ones((NUM_GROUPS + 1, NUM_GROUPS + 1)) * 2.0
    DD = np.ones((NUM_GROUPS + 1, NUM_GROUPS + 1)) * 2.0

    pdict = {
        "NUM_GROUPS": NUM_GROUPS,
        "NUM_LIVING": NUM_LIVING,
        "NUM_DEAD": NUM_DEAD,
        "NUM_GEARS": 0,
        "PB": PB,
        "QB": QB,
        "M0": M0,
        "Unassim": Unassim,
        "ActiveLink": ActiveLink,
        "VV": VV,
        "DD": DD,
        "QQbase": QQbase,
        "Bbase": state0.copy(),
        "PP_type": np.zeros(NUM_GROUPS + 1, dtype=int),
        "spname": spname,
        # fish-derived mapping: both FishSource (1) and Seabirds (3) produce discards
        "FishFrom": np.array([0, 1, 3], dtype=int),
        "FishTo": np.array([0, NUM_LIVING + 1, NUM_LIVING + 1], dtype=int),
        "FishQ": np.array([0.0, 0.05, 0.1]),
    }

    forcing = {"Ftime": np.ones(NUM_GROUPS + 1), "PP_forcing": np.ones(NUM_GROUPS + 1), "ForcedPrey": np.ones(NUM_GROUPS + 1)}
    fishing = {"FishFrom": pdict["FishFrom"], "FishThrough": np.array([0, NUM_LIVING + 1, NUM_LIVING + 1], dtype=int), "FishQ": pdict["FishQ"]}

    return pdict, forcing, fishing, state0, spname


def test_multi_month_synthetic_scenario_effect(synthetic_fish_scenario):
    pdict, forcing, fishing, state0, spname = synthetic_fish_scenario

    # Copy and disable fish mappings for control
    import copy

    pdict_no = copy.deepcopy(pdict)
    pdict_no["FishFrom"] = None
    pdict_no["FishTo"] = None
    pdict_no["FishQ"] = None

    # run short RK4 simulation for 36 months
    months = 36
    NUM_GROUPS = pdict["NUM_GROUPS"]

    biom_with = np.zeros((months + 1, NUM_GROUPS + 1))
    biom_no = np.zeros_like(biom_with)
    biom_with[0] = state0.copy()
    biom_no[0] = state0.copy()

    for m in range(1, months + 1):
        biom_with[m] = integrate_rk4(biom_with[m - 1], pdict, forcing, fishing, 1.0)
        biom_no[m] = integrate_rk4(biom_no[m - 1], pdict_no, forcing, fishing, 1.0)

    spnames = pdict["spname"]
    i_disc = spnames.index("Discards")
    i_seab = spnames.index("Seabirds")

    final_with = biom_with[-12:].mean(axis=0)
    final_no = biom_no[-12:].mean(axis=0)

    disc_diff = final_with[i_disc] - final_no[i_disc]
    seab_diff = final_with[i_seab] - final_no[i_seab]

    assert disc_diff > 1e-4, f"Synthetic scenario: Discards not sufficiently larger: {disc_diff}"
    assert abs(seab_diff) > 1e-5, f"Synthetic scenario: Seabirds effect too small: {seab_diff}"


@pytest.mark.parametrize("variant", [
    {"name": "weak-lowQB", "fish_q_scale": 0.4, "det_consume": 0.2, "unassim_scale": 0.8, "qb_val": 0.3, "m0_scale": 1.0, "extra_predators": [], "det_decay": 0.0},
    {"name": "weak-highUnassim", "fish_q_scale": 0.4, "det_consume": 0.2, "unassim_scale": 1.2, "qb_val": 0.5, "m0_scale": 1.0, "extra_predators": [], "det_decay": 0.0},
    {"name": "m0_high", "fish_q_scale": 1.0, "det_consume": 0.5, "unassim_scale": 1.0, "qb_val": 0.5, "m0_scale": 2.0, "extra_predators": [], "det_decay": 0.0},
    {"name": "multi_pred_detfrac", "fish_q_scale": 1.0, "det_consume": 0.5, "unassim_scale": 1.0, "qb_val": 0.5, "m0_scale": 1.0, "extra_predators": [2,3], "det_decay": 0.0},
    {"name": "strong", "fish_q_scale": 2.0, "det_consume": 1.0, "unassim_scale": 1.0, "qb_val": 0.7, "m0_scale": 1.0, "extra_predators": [], "det_decay": 0.0},
    {"name": "strong-highUnassim", "fish_q_scale": 2.0, "det_consume": 1.0, "unassim_scale": 1.3, "qb_val": 0.8, "m0_scale": 1.0, "extra_predators": [2], "det_decay": 0.0},
    {"name": "zero_detdecay", "fish_q_scale": 1.0, "det_consume": 0.5, "unassim_scale": 1.0, "qb_val": 0.5, "m0_scale": 1.0, "extra_predators": [], "det_decay": 0.0},
    {"name": "high_detdecay", "fish_q_scale": 1.0, "det_consume": 0.5, "unassim_scale": 1.0, "qb_val": 0.5, "m0_scale": 1.0, "extra_predators": [], "det_decay": 1.0},
    {"name": "extreme_unassim", "fish_q_scale": 1.0, "det_consume": 0.5, "unassim_scale": 5.0, "qb_val": 0.5, "m0_scale": 1.0, "extra_predators": [], "det_decay": 0.0},
    {"name": "zero_unassim", "fish_q_scale": 1.0, "det_consume": 0.5, "unassim_scale": 0.0, "qb_val": 0.5, "m0_scale": 1.0, "extra_predators": [], "det_decay": 0.0},
])
def test_multi_month_synthetic_scenario_variants(synthetic_fish_scenario, variant):
    pdict, forcing, fishing, state0, spname = synthetic_fish_scenario

    # Apply variant scaling
    # scale FishQ (skip index 0), set detritus consumption QQbase[4,2],
    # scale Unassim for consumers and set QB for forage and seabirds
    orig_fishq = pdict.get("FishQ").copy()
    pdict_variant = {**pdict}
    pdict_variant["FishQ"] = orig_fishq * variant["fish_q_scale"]
    pdict_variant["QQbase"] = pdict_variant["QQbase"].copy()
    pdict_variant["QQbase"][4, 2] = variant["det_consume"]
    # scale unassimilated fractions for consumer groups (Forage idx=2, Seabirds idx=3)
    pdict_variant["Unassim"] = pdict_variant["Unassim"].copy()
    pdict_variant["Unassim"][2] *= variant["unassim_scale"]
    pdict_variant["Unassim"][3] *= variant["unassim_scale"]

    # scale M0 if requested
    pdict_variant["M0"] = pdict_variant["M0"].copy()
    pdict_variant["M0"][1] *= variant.get("m0_scale", 1.0)
    pdict_variant["M0"][2] *= variant.get("m0_scale", 1.0)
    pdict_variant["M0"][3] *= variant.get("m0_scale", 1.0)

    # set QB for consumers so production responds (Forage and Seabirds)
    pdict_variant["QB"] = pdict_variant["QB"].copy()
    pdict_variant["QB"][2] = variant["qb_val"]
    pdict_variant["QB"][3] = max(variant["qb_val"] * 0.6, 0.1)

    # Optionally add extra predator(s) as fish-derived discard sources (multiple predators)
    if variant.get("extra_predators"):
        ff = pdict_variant.get("FishFrom")
        ft = pdict_variant.get("FishTo")
        fq = pdict_variant.get("FishQ")
        for pred_idx in variant["extra_predators"]:
            # append predator mapping to same Discards column
            ff = np.concatenate((ff, np.array([pred_idx], dtype=int)))
            ft = np.concatenate((ft, np.array([pdict_variant["NUM_LIVING"] + 1], dtype=int)))
            fq = np.concatenate((fq, np.array([0.05 * variant.get("fish_q_scale", 1.0)])))
        pdict_variant["FishFrom"] = ff
        pdict_variant["FishTo"] = ft
        pdict_variant["FishQ"] = fq

    # Set DetDecay if provided (index 1 is the only detritus column)
    pdict_variant["DetDecay"] = np.array([0.0, variant.get("det_decay", 0.0)])

    # control (no fish)
    import copy
    pdict_no = copy.deepcopy(pdict_variant)
    pdict_no["FishFrom"] = None
    pdict_no["FishTo"] = None
    pdict_no["FishQ"] = None

    months = 36
    NUM_GROUPS = pdict_variant["NUM_GROUPS"]

    biom_with = np.zeros((months + 1, NUM_GROUPS + 1))
    biom_no = np.zeros_like(biom_with)
    biom_with[0] = state0.copy()
    biom_no[0] = state0.copy()

    for m in range(1, months + 1):
        biom_with[m] = integrate_rk4(biom_with[m - 1], pdict_variant, forcing, fishing, 1.0)
        biom_no[m] = integrate_rk4(biom_no[m - 1], pdict_no, forcing, fishing, 1.0)

    i_disc = pdict_variant["spname"].index("Discards")
    i_seab = pdict_variant["spname"].index("Seabirds")

    final_with = biom_with[-12:].mean(axis=0)
    final_no = biom_no[-12:].mean(axis=0)

    disc_diff = final_with[i_disc] - final_no[i_disc]
    seab_diff = final_with[i_seab] - final_no[i_seab]

    assert disc_diff > 1e-4, f"Variant {variant['name']}: Discards not sufficiently larger: {disc_diff}"
    assert abs(seab_diff) > 1e-5, f"Variant {variant['name']}: Seabirds effect too small: {seab_diff}"

    # Sanity checks: detritus biomass should never go negative and monthly
    # detritus budget (inputs - sinks) should not be strongly negative
    def det_balance(state, pd):
        """Compute the detritus derivative using the production code path.

        Use the real `deriv_vector` implementation so the test reflects exactly
        what the model computes, including fish-derived DetFrac additions.
        """
        deriv = deriv_vector(state, pd, forcing, fishing)
        # return derivative for the absolute Discards group index
        return deriv[i_disc]

    # check each month for the variant
    for m in range(0, months + 1):
        net = det_balance(biom_with[m], pdict_variant)
        # Allow small transient negative detritus derivatives; bound by baseline detritus
        threshold = -pdict_variant.get("Bbase", np.zeros(NUM_GROUPS + 1))[i_disc]
        assert net >= threshold - 1e-8, f"Variant {variant['name']} month {m} negative det balance {net} (threshold {threshold})"
        assert biom_with[m][i_disc] >= -1e-12, f"Variant {variant['name']} month {m} negative det biomass {biom_with[m][i_disc]}"
def test_multi_month_rk4_persistent_discards_seabirds_effect():
    """Run a short (few-year) RK4 run with and without fish-derived discards
    and assert a persistent difference in Discards (and a measurable effect on
    Seabirds) over the final year of the short run.
    """
    model_df = pd.read_csv(ECOPATH_DIR / "model_params.csv")
    diet_df = pd.read_csv(ECOPATH_DIR / "diet_matrix.csv")

    params = create_rpath_params(model_df["Group"].tolist(), [int(t) for t in model_df["Type"].tolist()])
    params.model = model_df
    params.diet = diet_df

    pypath_model = rpath(params)

    # If no fish mappings present, skip
    if not (hasattr(params, "FishFrom") and getattr(params, "FishFrom") is not None):
        pytest.skip("Reference params have no FishFrom/FishTo/FishQ mappings to test")

    # Build a copy of params with fish-derived discards disabled
    import copy

    params_no_fish = copy.deepcopy(params)
    if hasattr(params_no_fish, "FishFrom"):
        params_no_fish.FishFrom = None
    if hasattr(params_no_fish, "FishTo"):
        params_no_fish.FishTo = None
    if hasattr(params_no_fish, "FishQ"):
        params_no_fish.FishQ = None

    # Short RK4 runs: 3 years (36 months) is enough to show a persistent effect
    years = range(1, 4)
    scen_with = rsim_scenario(pypath_model, params, years=years)
    out_with = rsim_run(scen_with, method="rk4", years=years)

    scen_no = rsim_scenario(pypath_model, params_no_fish, years=years)
    out_no = rsim_run(scen_no, method="rk4", years=years)

    biom_with = out_with.out_Biomass
    biom_no = out_no.out_Biomass

    # Find indices
    spnames = params.spname
    if "Discards" not in spnames or "Seabirds" not in spnames:
        pytest.skip("Reference model missing Discards or Seabirds groups")
    i_disc = spnames.index("Discards")
    i_seab = spnames.index("Seabirds")

    # Compare mean biomasses over the final year (last 12 months)
    final_with = biom_with[-12:].mean(axis=0)
    final_no = biom_no[-12:].mean(axis=0)

    disc_diff = final_with[i_disc] - final_no[i_disc]
    seab_diff = final_with[i_seab] - final_no[i_seab]

    # Expect Discards to be larger when fish-derived discards exist (positive diff)
    assert disc_diff > 1e-4, f"Discards mean over final year not sufficiently larger: diff={disc_diff}"

    # Seabirds should exhibit a measurable effect (positive or negative) above tiny noise
    assert abs(seab_diff) > 1e-5, f"Seabirds mean change over final year too small: diff={seab_diff}"
