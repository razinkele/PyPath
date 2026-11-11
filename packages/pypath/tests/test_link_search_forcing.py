"""Search-rate and vulnerability forcing on single pred-prey links (EwE rule).

EwE6 applies a SearchRate shape to the search rate of the link it is assigned
to, and a Vulnerability shape to that link's vulnerable-pool inflow
(ApplyAVmodifiers per link; cEcoSimModel.vb:2886-2889). PyPath's importer
folded every link shape into the predator's (or prey's) group-level forcing,
so a shape on one link acted on all of the predator's prey, and several
shaped links of one predator compounded (LT2022 scenario 7: Cypriniformes'
search rate 0.848**7 = 0.315 instead of 0.848 on seven links).

For a private arena, a search multiplier s on a link replaces the predator
term y by s*y: Q = Q0 x f K s y / ((K-1)(1+f)/2 + s y), where f is the prey
Ftime times prey forcing times the link's vulnerability multiplier.
"""

import copy
import warnings
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest

import pypath.core.ecosim_deriv as D
from pypath.core.ecopath import rpath
from pypath.core.ecosim import rsim_run, rsim_scenario
from pypath.core.params import create_rpath_params

# rsim index: 0 Outside, 1 Fish, 2 Zoo, 3 Benthos, 4 Phyto, 5 Det
FISH, ZOO, BENTHOS, PHYTO, DET = 1, 2, 3, 4, 5
LT2022 = Path(__file__).parents[3] / "Data" / "LT2022_0.5ST_final7.eweaccdb"


def _model():
    """Fish eats Zoo and Benthos, so it has two links to force."""
    params = create_rpath_params(
        groups=["Fish", "Zoo", "Benthos", "Phyto", "Det"], types=[0, 0, 0, 1, 2]
    )
    params.model.loc[0, ["Biomass", "PB", "QB"]] = [2.0, 1.0, 5.0]
    params.model.loc[1, ["Biomass", "PB", "QB"]] = [10.0, 30.0, 100.0]
    params.model.loc[2, ["Biomass", "PB", "QB"]] = [20.0, 2.0, 8.0]
    params.model.loc[3, ["Biomass", "PB"]] = [50.0, 100.0]
    params.model.loc[4, "Biomass"] = 1000.0
    params.model["EE"] = np.nan
    params.diet["Fish"] = [0.0, 0.6, 0.4, 0.0, 0.0, 0.0]
    params.diet["Zoo"] = [0.0, 0.0, 0.0, 1.0, 0.0, 0.0]
    params.diet["Benthos"] = [0.0, 0.0, 0.0, 0.0, 1.0, 0.0]
    params.diet["Phyto"] = [0.0] * 6
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return rpath(params), params


def _captured():
    model, params = _model()
    sc = rsim_scenario(model, params, years=range(1, 3))
    # isolates the mechanism under test; EwE's handling time (D5)
    # has its own tests (test_handling_time.py)
    sc.params.EWE_HANDLING_TIME = False
    cap = {}
    real = D.deriv_vector

    def spy(state, p, forcing, fishing, *a, **k):
        if "P" not in cap:
            cap.update(
                P=p,
                Fo=copy.deepcopy(forcing),
                Fi=copy.deepcopy(fishing),
                s=np.array(state, float),
            )
        return real(state, p, forcing, fishing, *a, **k)

    D.deriv_vector = spy
    try:
        rsim_run(sc, method="RK4")
    finally:
        D.deriv_vector = real
    return cap


def _zoo_to_fish(cap, x=1.0, y=1.0, s=1.0, v=1.0, ftime_zoo=1.0):
    """Zoo -> Fish flow at a state with prey ratio x and predator ratio y."""
    st = cap["s"].copy()
    st[ZOO] *= x
    st[FISH] *= y
    fo = copy.deepcopy(cap["Fo"])
    n = st.shape[0]
    fo["Ftime"][ZOO] = ftime_zoo
    fo["ForcedSearchLink"] = np.ones((n, n))
    fo["ForcedSearchLink"][ZOO, FISH] = s
    fo["ForcedVulLink"] = np.ones((n, n))
    fo["ForcedVulLink"][ZOO, FISH] = v
    # isolate the Zoo link: take Benthos out of Fish's diet at this state
    P = dict(cap["P"])
    qqb = np.array(P["QQbase"], float)
    qqb[BENTHOS, FISH] = 0.0
    P["QQbase"] = qqb
    r = {}
    D.deriv_vector(st, P, fo, cap["Fi"], rates=r)
    return float(r["FoodGain"][FISH]), P


@pytest.mark.parametrize(
    "x, y, s, v, ftime_zoo",
    [
        (1.0, 1.0, 1.2, 1.0, 1.0),  # search multiplier alone
        (1.3, 0.8, 0.85, 1.0, 1.0),  # off base
        (1.0, 1.0, 1.0, 0.7, 1.0),  # vulnerability multiplier alone
        (1.1, 1.2, 1.15, 1.25, 1.3),  # all three with prey Ftime
    ],
)
def test_link_multipliers_follow_ewes_arena(x, y, s, v, ftime_zoo):
    cap = _captured()
    got, P = _zoo_to_fish(cap, x, y, s, v, ftime_zoo)
    q0 = float(P["QQbase"][ZOO, FISH])
    k = float(P["VV"][ZOO, FISH])
    dd = float(P["DD"][ZOO, FISH])
    f = ftime_zoo * v
    expected = (
        q0 * x * f * k * s * y / ((k - 1) * (1 + f) / 2 + s * y) * dd / (dd - 1 + x)
    )
    assert got == pytest.approx(expected, rel=1e-9)


# --- importer -------------------------------------------------------------------

_SHAPES = pd.DataFrame(
    {
        "ShapeID": [10, 11],
        "zScale": [
            " ".join(["1"] * 12 + ["0.9"] * 12),
            " ".join(["1"] * 12 + ["0.6"] * 12),
        ],
    }
)
_ASSIGN = pd.DataFrame(
    {
        "ScenarioID": [1, 1, 1],
        "PredID": [501, 501, 501],  # Fish
        "PreyID": [502, 503, 502],  # Zoo, Benthos, Zoo
        "ShapeID": [10, 10, 11],
        "FunctionType": [1, 1, 2],  # search rate, search rate, vulnerability
    }
)


def _imported(rule):
    from pypath.io.ewemdb import _apply_forcing_shapes

    model, params = _model()
    sc = rsim_scenario(model, params, years=range(1, 3), fast_group_rule=rule)
    # isolates the mechanism under test; EwE's handling time (D5)
    # has its own tests (test_handling_time.py)
    sc.params.EWE_HANDLING_TIME = False
    selected = {
        "id": 1,
        "scenario_group_df": pd.DataFrame(
            {"GroupID": [501, 502, 503, 504, 505], "EcopathGroupID": [1, 2, 3, 4, 5]}
        ),
        "group_id_to_name": {1: "Fish", 2: "Zoo", 3: "Benthos", 4: "Phyto", 5: "Det"},
    }
    tables = {"EcosimScenarioPredPreyShape": _ASSIGN, "EcosimShapeTime": _SHAPES}

    def fake(filepath, names):
        return tables.get(names[0])

    with patch("pypath.io.ewemdb._try_read_table_variants", side_effect=fake):
        _apply_forcing_shapes("model.eweaccdb", selected, sc, params)
    return sc.forcing


def test_ewe_rule_keeps_link_shapes_on_their_links():
    fo = _imported("ewe")
    year2 = 12
    assert fo.ForcedSearchLink[year2, ZOO, FISH] == pytest.approx(0.9)
    assert fo.ForcedSearchLink[year2, BENTHOS, FISH] == pytest.approx(0.9)
    assert fo.ForcedVulLink[year2, ZOO, FISH] == pytest.approx(0.6)
    # no predator-wide or prey-wide effect, and no compounding (0.9**2)
    assert np.all(fo.ForcedSearch[:, FISH] == 1.0)
    assert np.all(fo.ForcedPrey[:, ZOO] == 1.0)
    assert fo.ForcedSearchLink[0, ZOO, FISH] == 1.0


def test_rpath_rule_keeps_the_group_level_mapping():
    fo = _imported("rpath")
    assert getattr(fo, "ForcedSearchLink", None) is None
    assert fo.ForcedSearch[12, FISH] == pytest.approx(0.81)  # unchanged legacy mapping


def test_ecospace_uniform_patches_match_rsim_run_with_link_forcing():
    from pypath.spatial import EcospaceParams, create_1d_grid, rsim_run_spatial

    model, params = _model()
    sc = rsim_scenario(model, params, years=range(1, 3))
    # isolates the mechanism under test; EwE's handling time (D5)
    # has its own tests (test_handling_time.py)
    sc.params.EWE_HANDLING_TIME = False
    n = sc.params.NUM_GROUPS + 1
    months = sc.forcing.ForcedSearch.shape[0]
    sc.forcing.ForcedSearchLink = np.ones((months, n, n))
    sc.forcing.ForcedSearchLink[:, ZOO, FISH] = 1.3
    grid = create_1d_grid(n_patches=2, spacing=1.0)
    eco = EcospaceParams(
        grid=grid,
        habitat_preference=np.ones((n, 2)),
        habitat_capacity=np.ones((n, 2)),
        dispersal_rate=np.zeros(n),
        advection_enabled=np.zeros(n, dtype=bool),
        gravity_strength=np.zeros(n),
    )
    expected = rsim_run(sc, method="RK4", years=range(1, 3))
    got = rsim_run_spatial(sc, ecospace=eco, years=range(1, 3))
    assert not np.allclose(expected.out_Biomass[-1], expected.out_Biomass[0])
    np.testing.assert_allclose(
        got.out_Biomass, expected.out_Biomass, rtol=1e-6, atol=1e-10
    )


@pytest.mark.integration
def test_lt2022_scenario7_link_shapes_are_not_compounded():
    if not LT2022.exists():
        pytest.skip("LT2022 database not found")
    from pypath.io.ewemdb import ecosim_scenario_from_ewemdb

    sc = ecosim_scenario_from_ewemdb(str(LT2022), scenario=7, timeseries_dataset=35)
    names = [str(n).strip() for n in sc.params.spname]
    cyp, her, zoo = (
        names.index(g) for g in ("Cypriniformes", "Herring", "Zooplankton")
    )
    fo = sc.forcing
    assert np.all(fo.ForcedSearch[:, cyp] == 1.0)
    assert np.all(fo.ForcedSearch[:, her] == 1.0)
    # shape 72 (year 3: 0.848) on each of Cypriniformes' seven shaped links
    for prey in ("Sand goby", "Mobile crustaceans", "Saduria entomon", "Worms"):
        assert fo.ForcedSearchLink[24, names.index(prey), cyp] == pytest.approx(
            0.8480268, rel=1e-6
        )
    # shape 251 (year 2: 1.11) on Herring's Zooplankton link only
    assert fo.ForcedSearchLink[12, zoo, her] == pytest.approx(1.110533, rel=1e-6)
    assert np.all(np.delete(fo.ForcedSearchLink[12, :, her], zoo) == 1.0)
