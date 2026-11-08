"""EwE6 Ecosim options that calibrated models rely on.

- MoPred: other mortality follows feeding time,
  M0_i(t) = M0_i * (1 - MoPred_i + MoPred_i * Ftime_i) (cEcoSimModel.Derivt).
  EwE's default is 1 for every group; Rpath has no such term, so PyPath's
  default stays 0 and EwE databases set it.
- UseVarPQ: gross efficiency follows total mortality,
  GE_i(t) = AssimEff_i * Z_i / (Z_i + 3 K_i) with Z_i = loss_i / B_i and
  AssimEff_i = GE0_i * (PB_i + 3 K_i) / PB_i, so GE returns to GE0 at
  Z = PB (CalculateAssimilationEfficiencies).
- Feeding time: Ftime = L + (1 - L) * Ftime * (1 - a + a * Qopt / Q),
  L = ForagingTimeLowerLimit (EwE 0.01; Rpath 0.1), capped at FtimeMax(i)
  (EwE per group; Rpath 2).
"""

import warnings

import numpy as np
import pytest

from pypath.core.ecopath import rpath
from pypath.core.ecosim import (
    _build_active_link_matrix,
    _build_link_matrix,
    _compute_Q_matrix,
    _ftime_update_rpath,
    rsim_params,
    rsim_run,
    rsim_scenario,
)
from pypath.core.ecosim_deriv import deriv_vector
from pypath.core.link_array import ActiveLinkArray
from pypath.core.params import create_rpath_params

FISH, ZOO, PHYTO, DET = 1, 2, 3, 4


def _model():
    params = create_rpath_params(
        groups=["Fish", "Zoo", "Phyto", "Det"], types=[0, 0, 1, 2]
    )
    params.model.loc[0, ["Biomass", "PB", "QB", "EE"]] = [1.0, 0.5, 3.0, 0.0]
    params.model.loc[1, ["Biomass", "PB", "QB", "EE"]] = [2.0, 1.5, 5.0, 1.0]
    params.model.loc[2, ["Biomass", "PB", "EE"]] = [10.0, 2.0, 0.5]
    params.model.loc[3, "Biomass"] = 10.0
    params.model["Det"] = 1.0
    params.model.loc[3, "Det"] = 0.0
    params.diet["Fish"] = [0.0, 1.0, 0.0, 0.0, 0.0]
    params.diet["Zoo"] = [0.0, 0.0, 1.0, 0.0, 0.0]
    params.diet["Phyto"] = [0.0] * 5
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        model = rpath(params)
    return model, params


def _params_dict(sim):
    d = {
        "NUM_GROUPS": sim.NUM_GROUPS,
        "NUM_LIVING": sim.NUM_LIVING,
        "NUM_DEAD": sim.NUM_DEAD,
        "NUM_GEARS": sim.NUM_GEARS,
        "PB": sim.PBopt,
        "QB": sim.FtimeQBOpt,
        "M0": sim.MzeroMort,
        "Unassim": sim.UnassimRespFrac,
        "ActiveLink": _build_active_link_matrix(sim),
        "VV": _build_link_matrix(sim, sim.VV),
        "DD": _build_link_matrix(sim, sim.DD),
        "QQbase": _build_link_matrix(sim, sim.QQ),
        "Bbase": sim.B_BaseRef,
        "PP_type": sim.PP_type,
        "NoIntegrate": np.zeros_like(sim.NoIntegrate),
        "FishFrom": sim.FishFrom,
        "FishTo": getattr(sim, "FishTo", np.array([])),
        "FishQ": sim.FishQ,
        "fleet_idx": getattr(sim, "fleet_idx", None),
        "DetFrac": sim.DetFrac,
        "DetFrom": sim.DetFrom,
        "DetTo": sim.DetTo,
        "DetLinkFrom": sim.DetFrom,
        "DetLinkTo": sim.DetTo,
        "DetLinkFrac": sim.DetFrac,
        "MoPred": sim.MoPred,
        "UseVarPQ": sim.UseVarPQ,
        "VbK": sim.VbK,
        "AssimEff": sim.AssimEff,
    }
    links = ActiveLinkArray.from_bool_matrix(d["ActiveLink"])
    d["_link_prey"] = links.prey
    d["_link_pred"] = links.pred
    return d


def _forcing(n, ftime=None):
    return {
        "Ftime": np.ones(n + 1) if ftime is None else ftime,
        "ForcedPrey": np.ones(n + 1),
        "PP_forcing": np.ones(n + 1),
        "ForcedBio": np.zeros(n + 1),
        "ForcedMigrate": np.zeros(n + 1),
        "ForcedEffort": np.ones(2),
    }


def _fishing(sim):
    return {
        "FishFrom": sim.FishFrom,
        "FishThrough": sim.FishThrough,
        "FishQ": sim.FishQ,
        "FishingMort": np.zeros(sim.NUM_GROUPS + 1),
    }


def _deriv(sim, state, ftime=None):
    return deriv_vector(
        state, _params_dict(sim), _forcing(sim.NUM_GROUPS, ftime), _fishing(sim)
    )


def test_defaults_keep_rpath_behaviour():
    model, _ = _model()
    sim = rsim_params(model)
    assert np.all(sim.MoPred == 0.0)
    assert np.all(sim.FtimeMax == 2.0)
    assert sim.ForagingTimeLowerLimit == 0.1
    assert sim.UseVarPQ is False
    assert np.all(sim.VbK == 0.0)
    assert np.all(sim.AssimEff == 0.0)


def test_other_mortality_follows_feeding_time_with_mopred():
    model, _ = _model()
    sim = rsim_params(model)
    base = sim.B_BaseRef.copy()
    ftime = np.ones(sim.NUM_GROUPS + 1)
    ftime[ZOO] = 1.5
    off = _deriv(sim, base, ftime)
    sim.MoPred[ZOO] = 1.0
    on = _deriv(sim, base, ftime)
    extra = 0.5 * sim.MzeroMort[ZOO] * base[ZOO]
    assert on[ZOO] - off[ZOO] == pytest.approx(-extra, rel=1e-9)
    # the extra dead matter reaches the detritus (mass balance)
    assert on[DET] - off[DET] == pytest.approx(extra, rel=1e-9)
    # with Ftime = 1 MoPred changes nothing
    plain = rsim_params(model)
    assert _deriv(sim, base)[ZOO] == pytest.approx(_deriv(plain, base)[ZOO], rel=1e-12)


def test_variable_pq_follows_total_mortality():
    model, _ = _model()
    sim = rsim_params(model)
    k = 0.4
    ge0 = sim.PBopt[ZOO] / sim.FtimeQBOpt[ZOO]
    sim.VbK[ZOO] = k
    sim.AssimEff[ZOO] = ge0 * (sim.PBopt[ZOO] + 3 * k) / sim.PBopt[ZOO]
    base = sim.B_BaseRef.copy()
    more_fish = base.copy()
    more_fish[FISH] *= 2.0

    # at the Ecopath base Z = PB and GE = GE0: nothing changes
    off_base = _deriv(sim, base)
    sim.UseVarPQ = True
    on_base = _deriv(sim, base)
    assert on_base[ZOO] == pytest.approx(off_base[ZOO], rel=1e-9)

    # with twice the predator Zoo's Z rises and so does its GE
    sim.UseVarPQ = False
    off = _deriv(sim, more_fish)
    sim.UseVarPQ = True
    on = _deriv(sim, more_fish)
    qq = _compute_Q_matrix(_params_dict(sim), more_fish, _forcing(sim.NUM_GROUPS))
    predation = qq[ZOO, 1 : sim.NUM_LIVING + 1].sum()
    consumption = qq[:, ZOO].sum()
    z = predation / more_fish[ZOO] + sim.MzeroMort[ZOO]
    ge_var = sim.AssimEff[ZOO] * z / (z + 3 * k)
    assert ge_var > ge0
    assert on[ZOO] - off[ZOO] == pytest.approx((ge_var - ge0) * consumption, rel=1e-9)


def test_scenario_overrides_set_vbk_and_assimilation_efficiency():
    model, params = _model()
    sc = rsim_scenario(
        model,
        params,
        years=range(1, 3),
        scenario_overrides={"vbk": {ZOO - 1: 0.4}, "use_var_pq": True},
    )
    ge0 = sc.params.PBopt[ZOO] / sc.params.FtimeQBOpt[ZOO]
    assert sc.params.VbK[ZOO] == pytest.approx(0.4)
    assert sc.params.AssimEff[ZOO] == pytest.approx(
        ge0 * (sc.params.PBopt[ZOO] + 1.2) / sc.params.PBopt[ZOO]
    )
    assert sc.params.UseVarPQ is True


def test_feeding_time_update_lower_limit_and_cap():
    # old 1.0, adjust 0.5, Qopt 10, actual Q/B 4 -> factor 1.75
    rpath_value = _ftime_update_rpath(1.0, 0.5, 10.0, 8.0, 2.0)
    assert rpath_value == pytest.approx(0.1 + 0.9 * 1.75)
    ewe_value = _ftime_update_rpath(1.0, 0.5, 10.0, 8.0, 2.0, lower=0.01, cap=5.0)
    assert ewe_value == pytest.approx(0.01 + 0.99 * 1.75)
    assert _ftime_update_rpath(1.0, 0.5, 10.0, 8.0, 2.0, lower=0.01, cap=1.5) == 1.5


def test_run_caps_feeding_time_per_group():
    model, params = _model()
    sc = rsim_scenario(model, params, years=range(1, 4))
    sc.params.FtimeAdj[ZOO] = 0.5
    sc.params.FtimeMax[ZOO] = 1.2
    sc.params.ForagingTimeLowerLimit = 0.01
    sc.forcing.ForcedPrey[:, PHYTO] = 0.3  # Zoo's food scarce: Ftime rises
    out = rsim_run(sc, method="RK4")
    ftime = out.end_state.Ftime[ZOO]
    assert 1.0 < ftime <= 1.2 + 1e-9


def test_pbmaxs_sets_the_producer_link_vulnerability():
    """EwE6 producer production PBmaxs*PB*B / (1 + (PBmaxs-1)*B/B0) is Rpath's
    producer link with VV = PBmaxs / (PBmaxs - 1); EwE's default 2 is VV 2."""
    model, params = _model()
    sc = rsim_scenario(
        model,
        params,
        years=range(1, 3),
        scenario_overrides={"pb_maxs": {PHYTO - 1: 3.0}},
    )
    p = sc.params
    link = [
        k
        for k in range(1, len(p.PreyFrom))
        if p.PreyFrom[k] == 0 and p.PreyTo[k] == PHYTO
    ]
    assert len(link) == 1
    assert p.VV[link[0]] == pytest.approx(1.5)


def test_pbmaxs_production_curve_matches_ewe():
    model, params = _model()
    sc = rsim_scenario(
        model,
        params,
        years=range(1, 3),
        scenario_overrides={"pb_maxs": {PHYTO - 1: 3.0}},
    )
    sim = sc.params
    d = _params_dict(sim)
    d["RpathPrimaryProduction"] = True
    pb, b0, pbmax = sim.PBopt[PHYTO], sim.B_BaseRef[PHYTO], 3.0
    for rel in (0.25, 1.0, 3.0):
        state = sim.B_BaseRef.copy()
        state[PHYTO] = rel * b0
        rates = {}
        deriv_vector(state, d, _forcing(sim.NUM_GROUPS), _fishing(sim), rates=rates)
        expected = pbmax * pb * state[PHYTO] / (1.0 + (pbmax - 1.0) * rel)
        assert rates["TotGain"][PHYTO] == pytest.approx(expected, rel=1e-9)


def test_pbmaxs_of_one_is_linear_production():
    model, params = _model()
    sc = rsim_scenario(
        model,
        params,
        years=range(1, 3),
        scenario_overrides={"pb_maxs": {PHYTO - 1: 1.0}},
    )
    p = sc.params
    link = [
        k
        for k in range(1, len(p.PreyFrom))
        if p.PreyFrom[k] == 0 and p.PreyTo[k] == PHYTO
    ]
    assert p.VV[link[0]] > 1e6
