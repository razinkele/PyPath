"""Production forcing and prey-availability forcing are separate channels.

EwE applies a production-rate shape to a producer's P/B only; a vulnerability
shape scales a prey's availability only. ``ForcedPrey`` is availability.
Production is forced through ``ForcedSearch`` (the producer's search rate on
its primary-production link, Rpath's convention), which ``rsim_run`` hands to
the derivative as ``PP_forcing``.
"""

import warnings

import numpy as np

import pypath.core.ecosim as ecosim_mod
import pypath.core.ecosim_deriv as _ecosim_deriv_mod
from pypath.core.ecopath import rpath
from pypath.core.ecosim import rsim_run, rsim_scenario
from pypath.core.params import create_rpath_params

PHYTO, ZOO = 1, 2  # rsim indices (0 = Outside)


def _simple_model(detritus_fate=False):
    # Slow enough that both living groups are integrated (M0*B < 24), so the
    # producer's response to forcing is visible under RK4.
    params = create_rpath_params(groups=["Phyto", "Zoo", "Det"], types=[1, 0, 2])
    if detritus_fate:
        params.model["Det"] = [1.0, 1.0, 0.0]  # dead matter goes to Det
    params.model.loc[0, ["Biomass", "PB", "QB", "EE"]] = [10.0, 2.0, 0.0, 0.5]
    params.model.loc[1, ["Biomass", "PB", "QB", "EE"]] = [2.0, 1.0, 5.0, 0.8]
    params.model.loc[2, ["Biomass", "PB", "QB", "EE"]] = [10.0, 0.0, 0.0, 0.0]
    params.diet["Zoo"] = [1.0, 0.0, 0.0, 0.0]
    params.diet["Phyto"] = [0.0, 0.0, 0.0, 0.0]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        model = rpath(params)
    return model, params


def _first_forcing_dict(scenario):
    captured = {}
    original = ecosim_mod.deriv_vector

    def spy(state, params_dict, forcing, fishing, *args, **kwargs):
        # The first monthly forcing dict (the start-of-run adjustment calls
        # the derivative with a reduced dict).
        if "forcing" not in captured and "ForcedPrey" in forcing:
            captured["forcing"] = {
                k: np.array(v, copy=True) for k, v in forcing.items()
            }
        return original(state, params_dict, forcing, fishing, *args, **kwargs)

    ecosim_mod.deriv_vector = spy

    _ecosim_deriv_mod.deriv_vector = spy
    try:
        rsim_run(scenario, method="RK4")
    finally:
        ecosim_mod.deriv_vector = original
        _ecosim_deriv_mod.deriv_vector = original
    return captured["forcing"]


def test_forced_search_on_a_producer_is_the_production_forcing():
    model, params = _simple_model()
    sc = rsim_scenario(model, params, years=range(1, 3))
    sc.forcing.ForcedSearch[:, PHYTO] = 1.3
    forcing = _first_forcing_dict(sc)
    assert forcing["PP_forcing"][PHYTO] == 1.3
    assert forcing["ForcedPrey"][PHYTO] == 1.0


def test_forced_prey_does_not_force_production():
    model, params = _simple_model()
    sc = rsim_scenario(model, params, years=range(1, 3))
    sc.forcing.ForcedPrey[:, PHYTO] = 0.5
    forcing = _first_forcing_dict(sc)
    assert forcing["ForcedPrey"][PHYTO] == 0.5
    assert forcing["PP_forcing"][PHYTO] == 1.0


def test_production_forcing_changes_the_producer():
    model, params = _simple_model()
    base = rsim_run(rsim_scenario(model, params, years=range(1, 4)), method="RK4")
    sc = rsim_scenario(model, params, years=range(1, 4))
    sc.forcing.ForcedSearch[:, PHYTO] = 1.5
    forced = rsim_run(sc, method="RK4")
    assert forced.annual_Biomass[2, PHYTO] > 1.05 * base.annual_Biomass[2, PHYTO]


# ---------------------------------------------------------------------------
# Other-mortality and consumer search-rate forcing (review F9)
# ---------------------------------------------------------------------------


def test_forced_mort_scales_other_mortality_and_feeds_detritus():
    """EwE6 MortOther forcing scales a group's other mortality (Mo) only; the
    extra dead matter reaches the detritus (Rpath also scales the predation
    loss, without the predator's gain)."""
    model, params = _simple_model(detritus_fate=True)
    sc = rsim_scenario(model, params, years=range(1, 3))
    base = _first_forcing_dict(sc)
    sc.forcing.ForcedMort[:, ZOO] = 2.0
    forced = _first_forcing_dict(sc)
    assert forced["ForcedMort"][ZOO] == 2.0

    from pypath.core.ecosim_deriv import deriv_vector

    p = sc.params
    state = p.B_BaseRef.copy()
    fishing = {
        "FishFrom": p.FishFrom,
        "FishThrough": p.FishThrough,
        "FishQ": p.FishQ,
        "FishingMort": np.zeros(p.NUM_GROUPS + 1),
    }
    pdict = _params_dict(sc)
    d0 = deriv_vector(state.copy(), pdict, base, fishing)
    d1 = deriv_vector(state.copy(), pdict, forced, fishing)
    extra = p.MzeroMort[ZOO] * state[ZOO]
    assert extra > 0
    np.testing.assert_allclose(d1[ZOO] - d0[ZOO], -extra, rtol=1e-9)
    det = p.NUM_LIVING + 1
    np.testing.assert_allclose(d1[det] - d0[det], extra, rtol=1e-9)


def test_forced_search_on_a_consumer_scales_its_feeding():
    """Rpath force_bysearch: the predator's effective biomass in the arena is
    multiplied, so its consumption follows Q0 * PDY * VV / (VV - 1 + PDY)."""
    model, params = _simple_model()
    sc = rsim_scenario(model, params, years=range(1, 3))
    base = _first_forcing_dict(sc)
    sc.forcing.ForcedSearch[:, ZOO] = 2.0
    forced = _first_forcing_dict(sc)
    assert forced["ForcedSearch"][ZOO] == 2.0
    assert forced["PP_forcing"][PHYTO] == 1.0  # producer production untouched

    from pypath.core.ecosim import _compute_Q_matrix

    pdict = _params_dict(sc)
    state = sc.params.B_BaseRef.copy()
    q0 = _compute_Q_matrix(pdict, state, base)[PHYTO, ZOO]
    q1 = _compute_Q_matrix(pdict, state, forced)[PHYTO, ZOO]
    vv = 2.0
    np.testing.assert_allclose(q1 / q0, 2.0 * vv / (vv - 1.0 + 2.0), rtol=1e-9)


def test_ecospace_applies_prey_and_mortality_forcing():
    """One Ecospace patch reproduces rsim_run, forcing included."""
    from pypath.spatial import EcospaceParams, create_1d_grid, rsim_run_spatial

    model, params = _simple_model()
    sc = rsim_scenario(model, params, years=range(1, 3))
    sc.forcing.ForcedPrey[:, PHYTO] = 0.6
    sc.forcing.ForcedMort[:, ZOO] = 1.5
    expected = rsim_run(sc, method="RK4", years=range(1, 3)).out_Biomass
    n = sc.params.NUM_GROUPS + 1
    eco = EcospaceParams(
        grid=create_1d_grid(n_patches=1, spacing=1.0),
        habitat_preference=np.ones((n, 1)),
        habitat_capacity=np.ones((n, 1)),
        dispersal_rate=np.zeros(n),
        advection_enabled=np.zeros(n, dtype=bool),
        gravity_strength=np.zeros(n),
    )
    got = rsim_run_spatial(sc, ecospace=eco, years=range(1, 3)).out_Biomass
    np.testing.assert_allclose(got, expected, rtol=1e-6, atol=1e-10)


def _params_dict(sc):
    from pypath.core.ecosim import _build_active_link_matrix, _build_link_matrix
    from pypath.core.link_array import ActiveLinkArray

    p = sc.params
    d = {
        "NUM_GROUPS": p.NUM_GROUPS,
        "NUM_LIVING": p.NUM_LIVING,
        "NUM_DEAD": p.NUM_DEAD,
        "NUM_GEARS": p.NUM_GEARS,
        "PB": p.PBopt,
        "QB": p.FtimeQBOpt,
        "M0": p.MzeroMort,
        "Unassim": p.UnassimRespFrac,
        "ActiveLink": _build_active_link_matrix(p),
        "VV": _build_link_matrix(p, p.VV),
        "DD": _build_link_matrix(p, p.DD),
        "QQbase": _build_link_matrix(p, p.QQ),
        "Bbase": p.B_BaseRef,
        "PP_type": p.PP_type,
        "NoIntegrate": np.zeros_like(p.NoIntegrate),
        "FishFrom": p.FishFrom,
        "FishTo": getattr(p, "FishTo", np.array([])),
        "FishQ": p.FishQ,
        "fleet_idx": getattr(p, "fleet_idx", None),
        "DetLinkFrom": p.DetFrom,
        "DetLinkTo": p.DetTo,
        "DetLinkFrac": p.DetFrac,
    }
    links = ActiveLinkArray.from_bool_matrix(d["ActiveLink"])
    d["_link_prey"] = links.prey
    d["_link_pred"] = links.pred
    return d


def test_importer_routes_a_consumer_search_rate_shape(monkeypatch):
    """EwE application type 1 (search rate) on a consumer's link becomes that
    consumer's ForcedSearch (PyPath has one search-rate channel per predator)."""
    import pandas as pd

    import pypath.io.ewemdb as ew

    model, params = _simple_model()
    sc = rsim_scenario(model, params, years=range(1, 3))
    tables = {
        "EcosimScenarioPredPreyShape": pd.DataFrame(
            {
                "ScenarioID": [7],
                "PredID": [902],
                "PreyID": [901],
                "ShapeID": [55],
                "FunctionType": [1],
            }
        ),
        "EcosimShapeTime": pd.DataFrame(
            {"ShapeID": [55], "zScale": [" ".join(["1.5"] * 24)], "Title": ["search"]}
        ),
    }
    monkeypatch.setattr(
        ew, "_try_read_table_variants", lambda fp, names: tables.get(names[0])
    )
    selected = {
        "id": 7,
        "scenario_group_df": pd.DataFrame(
            {
                "ScenarioID": [7, 7, 7],
                "EcopathGroupID": [11, 12, 13],
                "GroupID": [901, 902, 903],
            }
        ),
        "group_id_to_name": {11: "Phyto", 12: "Zoo", 13: "Det"},
    }
    ew._apply_forcing_shapes("unused.eweaccdb", selected, sc, params)
    np.testing.assert_allclose(sc.forcing.ForcedSearch[:, ZOO], 1.5)
    np.testing.assert_allclose(sc.forcing.ForcedSearch[:, PHYTO], 1.0)
    assert np.all(sc.forcing.ForcedPrey == 1.0)
