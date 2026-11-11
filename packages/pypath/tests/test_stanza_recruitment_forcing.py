"""Recruitment forcing in the stanza update, as EwE6 applies it.

EwE6's SplitUpdate (cEcoSimModel.vb 1895-1902) sets the youngest month to
    RscaleSplit x egg(t) x RzeroS x hatch(t)
and, only when the stanza has no hatch-code shape (HatchCode = 0), times
(Eggs / BaseEggs)^RecPower. egg(t) is the stanza's egg-production shape
and hatch(t) its hatch-code shape (EcosimStanzaShape; unset shapes read 1,
tval(0) = 1). Eggs are the spawning biomass, sum N (W - Wmat), with no
saturation. RscaleSplit is 12 / (sum of the first 12 months) for a seasonal
egg-production shape (SplitInitialize, vb 3487-3498), else 1.

PyPath (Rpath) multiplied the forcing into the eggs, inside the RecPower
exponent, after a Beverton-Holt saturation (spawn_x), had no hatch code, and
did not read EcosimStanzaShape. No database here attaches a shape to a
stanza (LT2022 defines four egg-production shapes but links none).
"""

import warnings

import numpy as np
import pandas as pd
import pytest

from pypath.core.ecopath import rpath
from pypath.core.ecosim import rsim_scenario
from pypath.core.stanzas import (
    RsimStanzaState,
    rpath_stanzas,
    split_set_pred,
    split_update,
)

from .test_stanza_plus_group import Z_AD, Z_JUV, _mussel_params


def _scenario(rule):
    params = _mussel_params()
    rpath_stanzas(params, rule=rule)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        model = rpath(params)
    sc = rsim_scenario(model, params, years=range(1, 3), fast_group_rule=rule)
    assert sc.stanzas is not None
    return model, sc


def _one_month(model, sc, forced_recs=None, numbers_scale=1.0, month_index=0):
    """One SplitUpdate month from the base at the Ecopath rates; returns the
    recruits and the eggs the update counted."""
    st, p = sc.stanzas, sc.params
    juv, ad = p.spname.index("Juv"), p.spname.index("Adult")
    state = RsimStanzaState.from_base(st, p.NUM_GROUPS)
    state.nage_s[:, 1] *= numbers_scale
    biomass = np.array(p.B_BaseRef, dtype=float)
    split_set_pred(st, state, biomass)
    food = np.zeros_like(biomass)
    z = np.zeros_like(biomass)
    for i, zi in ((juv, Z_JUV), (ad, Z_AD)):
        food[i] = model.QB[i - 1] * model.Biomass[i - 1]
        z[i] = zi
    split_update(
        st,
        state,
        biomass,
        food,
        z * biomass,
        forced_recs=forced_recs,
        month_index=month_index,
    )
    return float(state.nage_s[0, 1]), float(state.eggs_stanza[1])


def _recs(sc, value):
    """A ForcedRecs row with ``value`` at the stanza group's last stage."""
    row = np.ones(sc.params.NUM_GROUPS + 1)
    row[int(sc.stanzas.ecopath_code[1, 2])] = value
    return row


@pytest.mark.parametrize("rule", ["ewe", "rpath"])
def test_recruitment_forcing_and_rec_power(rule):
    """RecPower 0.5, forcing x 2: EwE multiplies the recruits by 2, outside
    the exponent; Rpath multiplies the eggs, inside it."""
    model, sc = _scenario(rule)
    st = sc.stanzas
    st.rec_power[1] = 0.5
    n0, eggs = _one_month(model, sc, forced_recs=_recs(sc, 2.0))
    base = st.base_eggs_stanza[1]
    rz = st.r_scale_split[1] * st.r_zero_s[1]
    if rule == "ewe":
        assert n0 == pytest.approx(rz * 2.0 * (eggs / base) ** 0.5, rel=1e-12)
    else:
        assert n0 == pytest.approx(rz * (eggs / base) ** 0.5, rel=1e-12)
        unforced = _one_month(model, sc)[1]
        assert eggs == pytest.approx(2.0 * unforced, rel=1e-12)


@pytest.mark.parametrize("rule", ["ewe", "rpath"])
def test_eggs_are_the_spawning_biomass_under_the_ewe_rule(rule):
    """EwE has no Beverton-Holt term: with spawn_x = 2 Rpath's eggs saturate,
    EwE's equal the spawning biomass."""
    model, sc = _scenario(rule)
    st = sc.stanzas
    st.spawn_x[1] = 2.0
    _, eggs = _one_month(model, sc, numbers_scale=1.5)
    st2 = _scenario(rule)[1]
    spawn = _spawn_after_one_month(model, st2, 1.5)
    if rule == "ewe":
        assert eggs == pytest.approx(spawn, rel=1e-12)
    else:
        base = st.base_spawn_bio[1]
        assert eggs == pytest.approx(spawn * 2.0 / (1.0 + spawn / base), rel=1e-12)


def _spawn_after_one_month(model, sc, numbers_scale):
    st, p = sc.stanzas, sc.params
    juv, ad = p.spname.index("Juv"), p.spname.index("Adult")
    state = RsimStanzaState.from_base(st, p.NUM_GROUPS)
    state.nage_s[:, 1] *= numbers_scale
    biomass = np.array(p.B_BaseRef, dtype=float)
    split_set_pred(st, state, biomass)
    food = np.zeros_like(biomass)
    z = np.zeros_like(biomass)
    for i, zi in ((juv, Z_JUV), (ad, Z_AD)):
        food[i] = model.QB[i - 1] * model.Biomass[i - 1]
        z[i] = zi
    split_update(st, state, biomass, food, z * biomass)
    return float(state.spawn_bio[1])


def test_hatch_shape_replaces_the_stock_recruitment():
    """With a hatch-code shape the recruits are RscaleSplit x egg x RzeroS x
    hatch, whatever the eggs."""
    model, sc = _scenario("ewe")
    st = sc.stanzas
    st.hatch_forcing = np.full((24, st.n_split + 1), np.nan)
    st.hatch_forcing[:, 1] = 0.5
    st.hatch_forcing[3, 1] = 0.25
    rz = st.r_scale_split[1] * st.r_zero_s[1]
    for scale in (1.0, 1.5):
        n0, _ = _one_month(model, sc, forced_recs=_recs(sc, 2.0), numbers_scale=scale)
        assert n0 == pytest.approx(rz * 2.0 * 0.5, rel=1e-12)
    n0, _ = _one_month(model, sc, month_index=3)
    assert n0 == pytest.approx(rz * 0.25, rel=1e-12)


def _bridge_params():
    """The mussel model with DataFrame stanza tables, as the EwE reader
    fills them (stanza ID 7)."""
    from pypath.core.params import RpathStanzaParams

    params = _mussel_params()
    stgroups = pd.DataFrame(
        {"StGroupNum": [7], "StanzaGroup": ["Mussel"], "VBGF_Ksp": [1.175]}
    )
    stindiv = pd.DataFrame(
        {
            "StGroupNum": [7, 7],
            "StanzaNum": [1, 2],
            "Group": ["Juv", "Adult"],
            "First": [0, 12],
            "Last": [11, 999],
            "Z": [Z_JUV, Z_AD],
            "Leading": [False, True],
        }
    )
    params.stanzas = RpathStanzaParams(
        n_stanza_groups=1, stgroups=stgroups, stindiv=stindiv
    )
    return params


def test_reader_applies_stanza_egg_and_hatch_shapes(monkeypatch):
    """EcosimStanzaShape links stanza 7 to a seasonal egg-production shape
    (12 months, repeated over the run) and a hatch-code shape (held at its
    last value)."""
    import pypath.io.ewemdb as ew

    params = _bridge_params()
    rpath_stanzas(params, rule="ewe")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        model = rpath(params)
    sc = rsim_scenario(model, params, years=range(1, 3))
    season = [0.5, 1.0, 1.5, 1.0, 0.5, 1.0, 1.5, 1.0, 0.5, 1.0, 1.5, 1.0]
    tables = {
        "EcosimStanzaShape": pd.DataFrame(
            {"StanzaID": [7], "EggprodShapeID": [73], "HatchCodeShapeID": [74]}
        ),
        "EcosimShapeEggProd": pd.DataFrame(
            {"ShapeID": [73], "zScale": [" ".join(str(v) for v in season)]}
        ),
        "EcosimShapeTime": pd.DataFrame({"ShapeID": [74], "zScale": ["0.8 0.6"]}),
        "EcosimShape": pd.DataFrame(
            {"ShapeID": [73, 74], "ShapeType": [9, 8], "IsSeasonal": [1, 0]}
        ),
    }
    monkeypatch.setattr(
        ew, "_try_read_table_variants", lambda fp, names: tables.get(names[0])
    )
    ew._apply_stanza_shapes("unused.eweaccdb", sc, params)
    adult = sc.params.spname.index("Adult")
    np.testing.assert_allclose(sc.forcing.ForcedRecs[:, adult], season * 2)
    assert sc.stanzas.r_scale_split[1] == pytest.approx(12.0 / sum(season))
    hatch = sc.stanzas.hatch_forcing[:, 1]
    np.testing.assert_allclose(hatch, [0.8] + [0.6] * 23)
