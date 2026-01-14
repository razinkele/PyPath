
import numpy as np
import pytest

from pypath.core.ecosim_deriv import deriv_vector


@pytest.mark.parametrize("fish_q,m0_1,qb_1,unassim1,unassim2,include_qq,include_detfrac", [
    (0.1, 0.05, 0.0, 0.2, 0.1, False, False),
    (0.2, 0.1, 0.5, 0.25, 0.15, True, False),
    (0.05, 0.01, 0.2, 0.15, 0.05, True, True),
])
def test_fish_discard_contributions_increase_detritus_inputs_and_deriv(fish_q, m0_1, qb_1, unassim1, unassim2, include_qq, include_detfrac):
    """Reproducer: at a known month/state, adding fish-derived discard links
    should increase Detritus inputs (unas/mort) and increase the Discards derivative.

    This reproduces the diagnostic scenario used interactively in
    scripts/compute_deriv_discards.py (month 1199 from the reference run).
    """
    # Build a small synthetic model to keep the test fast and deterministic.
    # Structure: 2 living groups (1..2) and 1 detritus group (3 = Discards).
    NUM_LIVING = 2
    NUM_DEAD = 1
    NUM_GROUPS = NUM_LIVING + NUM_DEAD

    # indices 0..NUM_GROUPS
    spname = [None, "FishSource", "Other", "Discards"]

    # State: biomass for groups (index 0 unused)
    state = np.zeros(NUM_GROUPS + 1)
    state[1] = 10.0  # fish source (will be fished)
    state[2] = 5.0  # other living group
    state[3] = 1.0  # detritus initial

    # Basic parameters
    PB = np.zeros(NUM_GROUPS + 1)
    QB = np.zeros(NUM_GROUPS + 1)
    M0 = np.zeros(NUM_GROUPS + 1)
    Unassim = np.zeros(NUM_GROUPS + 1)

    # Configure per-variant parameters
    M0[1] = m0_1
    M0[2] = 0.02  # keep second group's M0 small constant
    QB[1] = qb_1
    Unassim[1] = unassim1
    Unassim[2] = unassim2

    # QQbase: will be populated if include_qq True so predation/unas can be tested
    QQbase = np.zeros((NUM_GROUPS + 1, NUM_GROUPS + 1))

    # ActiveLink matrix - set link if include_qq True
    ActiveLink = np.zeros((NUM_GROUPS + 1, NUM_GROUPS + 1), dtype=bool)
    if include_qq:
        # Setup predation: group 2 'Other' consumes FishSource (1)
        ActiveLink[1, 2] = True
        QQbase[1, 2] = 0.5

    # VV / DD default arrays (not used heavily here)
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
        "Bbase": state.copy(),
        "PP_type": np.zeros(NUM_GROUPS + 1, dtype=int),
        "spname": spname,
        # fish mappings (FishFrom/FishTo/FishQ) placed directly in params
        "FishFrom": np.array([0, 1], dtype=int),  # index 1 holds source group 1
        "FishTo": np.array([0, NUM_LIVING + 1], dtype=int),  # index 1 points to detritus group (absolute index)
        "FishQ": np.array([0.0, 0.1]),
    }

    # If requested, also add a fish-derived mapping from the predator group to detritus
    if include_detfrac:
        ff = pdict.get("FishFrom")
        ft = pdict.get("FishTo")
        fq = pdict.get("FishQ")
        pdict["FishFrom"] = np.concatenate((ff, np.array([2], dtype=int)))
        pdict["FishTo"] = np.concatenate((ft, np.array([NUM_LIVING + 1], dtype=int)))
        pdict["FishQ"] = np.concatenate((fq, np.array([float(fish_q)])))


    forcing = {
        "Ftime": np.ones(NUM_GROUPS + 1),
        "ForcedBio": np.zeros(NUM_GROUPS + 1),
        "ForcedMigrate": np.zeros(NUM_GROUPS + 1),
        "ForcedEffort": np.ones(1),
        "PP_forcing": np.ones(NUM_GROUPS + 1),
        "ForcedPrey": np.ones(NUM_GROUPS + 1),
    }

    # fishing maps: fish source group 1's discards go to detritus (absolute detritus group index = NUM_LIVING+1 = 3)
    fishing = {
        "FishFrom": np.array([0, 1], dtype=int),  # 1 entry at index 1
        "FishThrough": np.array([0, NUM_LIVING + 1], dtype=int),
        "FishQ": np.array([0.0, fish_q]),  # fish_q for the mapping: parametric
    }

    # indices
    idx_discards = pdict["spname"].index("Discards")
    # early compute detritus column index for optional DetFrac setup
    det_idx_early = idx_discards - NUM_LIVING

    # Derivative with fish arrays present (original)
    deriv_with = deriv_vector(state, pdict, forcing, fishing)

    # Derivative with fish arrays removed
    pdict_no_fish = pdict.copy()
    pdict_no_fish["FishFrom"] = None
    pdict_no_fish["FishTo"] = None
    pdict_no_fish["FishQ"] = None
    deriv_no_fish = deriv_vector(state, pdict_no_fish, forcing, fishing)

    # Expect that adding fish-derived discards increases Discards derivative
    # Use a small threshold, differences in diagnostics are small but nonzero
    assert deriv_with[idx_discards] >= deriv_no_fish[idx_discards] - 1e-12
    assert deriv_with[idx_discards] - deriv_no_fish[idx_discards] > 1e-9

    # Now compute DetFrac baseline (empty) and DetFrac augmented with fish-derived additions
    NUM_GROUPS = pdict["NUM_GROUPS"]
    NUM_LIVING = pdict["NUM_LIVING"]
    NUM_DEAD = pdict["NUM_DEAD"]

    # Start with zero DetFrac (no baseline detritus allocation)
    DetFrac = np.zeros((NUM_GROUPS + 1, NUM_DEAD + 1))

    # Optionally set a predator DetFrac so unassimilated consumption can be
    # allocated to detritus (exercises unas path).
    if include_detfrac:
        # predator group is index 2 ('Other') in the synthetic model
        DetFrac[2, det_idx_early] = 0.5

    # copy and augment DetFrac with fish-derived contributions (diagnostic logic from deriv_vector)
    DetFrac_aug = DetFrac.copy()
    fish_from = pdict.get("FishFrom")
    fish_to = pdict.get("FishTo")
    fish_q = pdict.get("FishQ")
    if fish_from is not None and fish_to is not None and fish_q is not None:
        for k in range(len(fish_from)):
            f = int(fish_from[k])
            t = int(fish_to[k])
            if t >= (NUM_LIVING + 1) and t <= (NUM_LIVING + NUM_DEAD) and f >= 0 and f <= NUM_GROUPS:
                det_col = t - NUM_LIVING
                src_idx = f
                fish_input = float(fish_q[k]) * float(state[src_idx])
                m0_arr = pdict.get("M0", np.zeros(NUM_GROUPS + 1))
                qb_arr = pdict.get("QB", np.zeros(NUM_GROUPS + 1))
                unassim_arr = pdict.get("Unassim", np.zeros(NUM_GROUPS + 1))
                m0_pos = max(0.0, float(m0_arr[src_idx]))
                qb_loss = 0.0 if np.isnan(qb_arr[src_idx]) else float(qb_arr[src_idx])
                source_loss = m0_pos * float(state[src_idx]) + float(state[src_idx]) * qb_loss * float(unassim_arr[src_idx])
                frac = fish_input / (source_loss + 1e-30)
                if frac > 0:
                    DetFrac_aug[src_idx, det_col] += frac

    # compute unas_input and mort_input for Discards det_idx
    d = idx_discards
    det_idx = d - NUM_LIVING

    # baseline
    _unas_input_base = 0.0
    for pred in range(1, NUM_LIVING + 1):
        total_consump = np.sum(pdict["QQbase"][1:, pred]) if False else np.sum(np.zeros((NUM_GROUPS + 1,)))
        # we should use QQ computed from current state; replicate deriv_vector's QQ computation (simpler)
    # Recompute QQ as deriv_vector does
    QQ = np.zeros((NUM_GROUPS + 1, NUM_GROUPS + 1))
    BB = state.copy()
    Bbase = pdict["Bbase"]
    Ftime = forcing.get("Ftime", np.ones(NUM_GROUPS + 1))
    ForcedPrey = forcing.get("ForcedPrey", np.ones(NUM_GROUPS + 1))
    preyYY = np.zeros(NUM_GROUPS + 1)
    for i in range(1, NUM_GROUPS + 1):
        if Bbase[i] > 0:
            preyYY[i] = BB[i] / Bbase[i] * ForcedPrey[i]
    predYY = np.zeros(NUM_GROUPS + 1)
    for i in range(1, NUM_LIVING + 1):
        if Bbase[i] > 0:
            predYY[i] = Ftime[i] * BB[i] / Bbase[i]
    QQbase = pdict["QQbase"]
    VV = pdict["VV"]
    DD = pdict["DD"]
    ActiveLink = pdict["ActiveLink"]
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

    # compute unas and mort for baseline and augmented DetFrac
    unas_base = 0.0
    mort_base = 0.0
    unas_aug = 0.0
    mort_aug = 0.0
    for pred in range(1, NUM_LIVING + 1):
        total_consump = np.sum(QQ[1:, pred])
        unas_base += total_consump * pdict["Unassim"][pred] * (DetFrac[pred, det_idx] if DetFrac.shape[1] > det_idx else 0)
        unas_aug += total_consump * pdict["Unassim"][pred] * (DetFrac_aug[pred, det_idx] if DetFrac_aug.shape[1] > det_idx else 0)
    for grp in range(1, NUM_LIVING + 1):
        mort_base += pdict.get("M0", np.zeros(NUM_GROUPS + 1))[grp] * state[grp] * (DetFrac[grp, det_idx] if DetFrac.shape[1] > det_idx else 0)
        mort_aug += pdict.get("M0", np.zeros(NUM_GROUPS + 1))[grp] * state[grp] * (DetFrac_aug[grp, det_idx] if DetFrac_aug.shape[1] > det_idx else 0)

    # If predator DetFrac was set and predation links exist, expect unas to increase
    if include_detfrac and include_qq:
        assert unas_aug - unas_base > 1e-9

    # Important check: fish-derived contributions increase mort_input
    assert mort_aug - mort_base > 1e-6

    # ensure the combined inputs increase
    assert (unas_aug + mort_aug) - (unas_base + mort_base) > 1e-6
