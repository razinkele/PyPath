"""Prey feeding time and vulnerability forcing in the foraging arena (D1).

EwE6 scales only the flow into the vulnerable pool: Veff = VulArena * f with
f = prey Ftime times vulnerability forcing, and
Vbiom = Veff * B / (VulArena + Veff + Vdenom) (cEcoSimModel.vb:2886-2905).
For a private arena with K = vulnerability, y the predator term and x = B/B0
this gives (re-derived from the arena equations; audit 2026-11-10, D1)

    Q = Q0 * x * f * K * y / ((K - 1) * (1 + f) / 2 + y)

where PyPath's Rpath form multiplied the prey term by f instead:

    Q = Q0 * (f x) * K * y / (K - 1 + y).

Both carry the same handling factor DD / (DD - 1 + prey term).
"""

import copy
from pathlib import Path

import numpy as np
import pytest

import pypath.core.ecosim_deriv as D
from pypath.core.ecosim import rsim_run, rsim_scenario

from .test_fast_group_rule import FISH, ZOO, _stiff_model


def _captured(rule="ewe"):
    """params / forcing / fishing dicts rsim_run passes to the derivative."""
    model, params = _stiff_model()
    sc = rsim_scenario(model, params, years=range(1, 3), fast_group_rule=rule)
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


def _fish_intake(cap, x=1.0, y=1.0, ftime_zoo=1.0, forced_prey_zoo=1.0):
    """Fish eats only Zoo: its FoodGain is the Zoo -> Fish flow."""
    s = cap["s"].copy()
    s[ZOO] *= x
    s[FISH] *= y
    fo = copy.deepcopy(cap["Fo"])
    fo["Ftime"][ZOO] = ftime_zoo
    fo["ForcedPrey"][ZOO] = forced_prey_zoo
    r = {}
    D.deriv_vector(s, cap["P"], fo, cap["Fi"], rates=r)
    return float(r["FoodGain"][FISH])


def _link(cap):
    P = cap["P"]
    return (
        float(P["QQbase"][ZOO, FISH]),
        float(P["VV"][ZOO, FISH]),
        float(P["DD"][ZOO, FISH]),
    )


@pytest.mark.parametrize(
    "x, y, ftime_zoo, forced_prey_zoo",
    [
        (1.0, 1.0, 1.5, 1.0),  # prey forages more
        (1.0, 1.0, 0.5, 1.0),  # prey forages less
        (1.3, 0.8, 1.4, 1.0),  # off base, prey and predator
        (0.7, 1.2, 1.0, 0.6),  # vulnerability forcing alone
        (1.1, 1.1, 1.2, 1.25),  # both multipliers
    ],
)
def test_ewe_rule_saturates_prey_ftime_in_the_arena(x, y, ftime_zoo, forced_prey_zoo):
    cap = _captured("ewe")
    q0, k, dd = _link(cap)
    f = ftime_zoo * forced_prey_zoo
    expected = q0 * x * f * k * y / ((k - 1) * (1 + f) / 2 + y) * dd / (dd - 1 + x)
    got = _fish_intake(cap, x, y, ftime_zoo, forced_prey_zoo)
    assert got == pytest.approx(expected, rel=1e-9)


def test_rpath_rule_keeps_the_linear_prey_term():
    cap = _captured("rpath")
    q0, k, dd = _link(cap)
    x, y, f = 1.2, 0.9, 1.4
    expected = q0 * (f * x) * k * y / (k - 1 + y) * dd / (dd - 1 + f * x)
    assert _fish_intake(cap, x, y, ftime_zoo=f) == pytest.approx(expected, rel=1e-9)


def test_unforced_links_are_bit_identical_to_the_rpath_form():
    """With every prey at f = 1 the arena change touches nothing."""
    ewe, rpath = _captured("ewe"), _captured("rpath")
    for x, y in ((1.0, 1.0), (1.3, 0.8)):
        assert _fish_intake(ewe, x, y) == _fish_intake(rpath, x, y)


def test_ecospace_uniform_patches_match_rsim_run_with_prey_forcing():
    """Ecospace builds its own parameter dict: it must carry the arena rule."""
    from pypath.spatial import EcospaceParams, create_1d_grid, rsim_run_spatial

    model, params = _stiff_model()
    sc = rsim_scenario(model, params, years=range(1, 3))
    sc.forcing.ForcedPrey[:, ZOO] = 0.7  # f != 1 on Fish's prey
    n = sc.params.NUM_GROUPS + 1
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
    np.testing.assert_allclose(
        got.out_Biomass, expected.out_Biomass, rtol=1e-6, atol=1e-10
    )


ORACLE = Path(__file__).parent / "data" / "ewe_oracle" / "lt2022"
LT2022 = Path(__file__).parents[3] / "Data" / "LT2022_0.5ST_final7.eweaccdb"


@pytest.mark.integration
@pytest.mark.parametrize("group", ["Cod", "Smelt"])
def test_lt2022_feeding_time_ratio_matches_ewe_at_ewe_states(group):
    """At EwE's own state and Ftime, PyPath's Cbase/CBlast equals EwE's.

    Cod and Smelt eat prey whose Ftime rises to 1.16-1.35 after the year-2
    forcing step; with the linear prey term they were 3-4 % off by month 20.
    EwE's ratio is implied by its update, Ftime(t) = L + (1 - L) Ftime(t-1)
    (1 - adj + adj q).
    """
    import pandas as pd

    if not LT2022.exists():
        pytest.skip("LT2022 database not found")
    from pypath.io.ewemdb import ecosim_scenario_from_ewemdb

    biomass = pd.read_csv(ORACLE / "ecosim_biomass_monthly.csv", index_col=0)
    ftime = pd.read_csv(ORACLE / "ecosim_feeding_time_monthly.csv", index_col=0)
    biomass.columns = [c.strip() for c in biomass.columns]
    ftime.columns = [c.strip() for c in ftime.columns]
    sc = ecosim_scenario_from_ewemdb(str(LT2022), scenario=16, timeseries_dataset=35)
    names = [str(n).strip() for n in sc.params.spname]
    i = names.index(group)
    cap = {}
    real = D.deriv_vector

    phyto = names.index("Phytoplankton")

    def spy(state, p, forcing, fishing, *a, **k):
        # year-2 forcing (after the production step), as the states compared
        if "P" not in cap and float(forcing["ForcedSearch"][phyto]) < 0.9:
            cap.update(
                P=p,
                Fo=copy.deepcopy(forcing),
                Fi=copy.deepcopy(fishing),
                s=np.array(state, float),
            )
        return real(state, p, forcing, fishing, *a, **k)

    D.deriv_vector = spy
    try:
        rsim_run(sc, method="RK4", years=range(1, 3))
    finally:
        D.deriv_vector = real
    lower = float(sc.params.ForagingTimeLowerLimit)
    adj = float(sc.params.FtimeAdj[i])
    qbopt = float(sc.params.FtimeQBOpt[i])
    cols = list(biomass.columns)
    for t in (16, 18, 20):
        q_ewe = (
            (ftime.loc[t, group] - lower) / ((1 - lower) * ftime.loc[t - 1, group])
            - (1 - adj)
        ) / adj
        s = cap["s"].copy()
        s[1 : len(cols) + 1] = biomass.loc[t - 1].values  # start of month t
        fo = copy.deepcopy(cap["Fo"])
        fo["Ftime"][1 : len(cols) + 1] = ftime.loc[t - 1].values  # used in month t
        r = {}
        real(s, cap["P"], fo, cap["Fi"], rates=r)
        q_py = qbopt / (r["FoodGain"][i] / s[i])
        assert q_py == pytest.approx(q_ewe, rel=1e-3), (group, t)
