"""SpawnProp, FixedFecundity and RecStanza, as EwE6's stanza update uses them.

- SpawnProp (StanzaLifeStage, default 1): each life stage's share of the
  spawning. Eggs are sum N (W - Wmat) SpawnProp over mature ages, at the
  base (SplitInitialize, vb 3458-3466) and every month (SplitUpdate,
  vb 1852-1861).
- FixedFecundity (Stanza): eggs per individual stay at their base value,
  EggsSplit = (W - Wmat) SpawnProp; the monthly eggs are
  sum N EggsSplit SpawnProp (SpawnProp twice, the base sum once).
- RecStanza (Stanza, a stanza group's 1-based position, 0 = none): the
  group's recruits follow that group's, times the ratio of the two at the
  first update (vb 1905-1912).
EggAtSpawn has no effect on the model in EwE. No database here sets any
of them away from the default.
"""

import warnings

import numpy as np
import pytest

from pypath.core.ecopath import rpath
from pypath.core.ecosim import rsim_scenario
from pypath.core.params import create_rpath_params
from pypath.core.stanzas import (
    RsimStanzaState,
    create_stanza_params,
    rpath_stanzas,
    split_set_pred,
    split_update,
)

from .test_stanza_plus_group import Z_AD, Z_JUV, _mussel_params

WMAT = 0.1


def _scenario(params):
    sp = params.stanzas
    for sg in sp.stanza_groups:
        sg.wmat = WMAT
    rpath_stanzas(params, rule="ewe")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        model = rpath(params)
    sc = rsim_scenario(model, params, years=range(1, 3), fast_group_rule="ewe")
    assert sc.stanzas is not None
    return model, sc


def _with(spawn_prop=(1.0, 1.0), fixed=False):
    params = _mussel_params()
    for st, p in zip(params.stanzas.stanza_individuals, spawn_prop):
        st.spawn_prop = p
    params.stanzas.stanza_groups[0].fixed_fecundity = fixed
    return _scenario(params)


def _month(model, sc, food_scale=1.0, sstate=None):
    st, p = sc.stanzas, sc.params
    if sstate is None:
        sstate = RsimStanzaState.from_base(st, p.NUM_GROUPS)
    biomass = np.array(p.B_BaseRef, dtype=float)
    split_set_pred(st, sstate, biomass)
    food = np.zeros_like(biomass)
    z = np.zeros_like(biomass)
    for name, zi in (("Juv", Z_JUV), ("Adult", Z_AD)):
        i = p.spname.index(name)
        food[i] = food_scale * model.QB[i - 1] * model.Biomass[i - 1]
        z[i] = zi
    split_update(st, sstate, biomass, food, z * biomass, month_index=0)
    return sstate


def test_spawn_prop_weights_the_base_eggs():
    _, sc = _with(spawn_prop=(0.0, 0.5))
    st = sc.stanzas
    n, w = st.base_nage_s[:, 1], st.base_wage_s[:, 1]
    adult = slice(int(st.age1[1, 2]), int(st.age2[1, 2]) + 1)
    mature = w[adult] > WMAT
    expected = 0.5 * float(np.sum(n[adult][mature] * (w[adult][mature] - WMAT)))
    assert st.base_eggs_stanza[1] == pytest.approx(expected, rel=1e-12)


def test_uniform_spawn_prop_halves_the_eggs_not_the_recruits():
    model, full = _with()
    _, half = _with(spawn_prop=(0.5, 0.5))
    a, b = _month(model, full), _month(model, half)
    assert half.stanzas.base_eggs_stanza[1] == pytest.approx(
        0.5 * full.stanzas.base_eggs_stanza[1], rel=1e-12
    )
    assert b.eggs_stanza[1] == pytest.approx(0.5 * a.eggs_stanza[1], rel=1e-12)
    assert b.nage_s[0, 1] == pytest.approx(a.nage_s[0, 1], rel=1e-12)


def test_fixed_fecundity_eggs_ignore_weight():
    """Better fed, the stanzas grow more: the eggs follow the weights unless
    the fecundity is fixed."""
    model, free = _with()
    _, fixed = _with(fixed=True)
    assert _month(model, free, 2.0).eggs_stanza[1] > _month(model, free).eggs_stanza[1]
    assert _month(model, fixed, 2.0).eggs_stanza[1] == pytest.approx(
        _month(model, fixed).eggs_stanza[1], rel=1e-12
    )


@pytest.mark.parametrize("fixed, ratio", [(False, 0.5), (True, 0.25)])
def test_fixed_fecundity_counts_spawn_prop_twice(fixed, ratio):
    """EwE's monthly sum with fixed fecundity is N x EggsSplit x SpawnProp,
    and EggsSplit already holds SpawnProp."""
    model, one = _with(fixed=fixed)
    _, half = _with(spawn_prop=(0.5, 0.5), fixed=fixed)
    assert _month(model, half).eggs_stanza[1] == pytest.approx(
        ratio * _month(model, one).eggs_stanza[1], rel=1e-12
    )


def _two_groups():
    """Two mussel-like stanza groups; the second recruits as the first."""
    params = create_rpath_params(
        ["Phyto", "Juv", "Adult", "Juv2", "Adult2", "Det"],
        [1, 0, 0, 0, 0, 2],
        stgroups=[None, "A", "A", "B", "B", None],
    )
    params.model["Biomass"] = [100.0, 15.0, 27.0, 7.5, 13.5, 10.0]
    params.model["PB"] = [50.0, Z_JUV, Z_AD, Z_JUV, Z_AD, 0.0]
    params.model["QB"] = [0.0, 13.0, 8.0, 13.0, 8.0, 0.0]
    params.model["EE"] = np.nan
    params.model["Unassim"] = [0.0, 0.2, 0.2, 0.2, 0.2, 0.0]
    params.model["Det"] = [1.0, 1.0, 1.0, 1.0, 1.0, 0.0]
    diet = params.diet
    diet.loc[diet["Group"] == "Phyto", ["Juv", "Adult", "Juv2", "Adult2"]] = 1.0

    def stage(sg, num, gnum, name, first, z, leading):
        return {
            "stanza_group_num": sg,
            "stanza_num": num,
            "group_num": gnum,
            "group_name": name,
            "first": first,
            "last": first + (11 if first == 0 else 48),
            "z": z,
            "leading": leading,
        }

    params.stanzas = create_stanza_params(
        [
            {"stanza_group_num": 1, "n_stanzas": 2, "vbgf_ksp": 1.175},
            {"stanza_group_num": 2, "n_stanzas": 2, "vbgf_ksp": 1.175, "rec_stanza": 1},
        ],
        [
            stage(1, 1, 2, "Juv", 0, Z_JUV, False),
            stage(1, 2, 3, "Adult", 12, Z_AD, True),
            stage(2, 1, 4, "Juv2", 0, Z_JUV, False),
            stage(2, 2, 5, "Adult2", 12, Z_AD, True),
        ],
    )
    return params


def test_rec_stanza_links_the_recruitment():
    model, sc = _scenario(_two_groups())
    st, p = sc.stanzas, sc.params
    sstate = RsimStanzaState.from_base(st, p.NUM_GROUPS)
    biomass = np.array(p.B_BaseRef, dtype=float)
    split_set_pred(st, sstate, biomass)
    food = np.zeros_like(biomass)
    z = np.zeros_like(biomass)
    for name, zi in (
        ("Juv", Z_JUV),
        ("Adult", Z_AD),
        ("Juv2", Z_JUV),
        ("Adult2", Z_AD),
    ):
        i = p.spname.index(name)
        food[i] = model.QB[i - 1] * model.Biomass[i - 1]
        z[i] = zi
    split_update(st, sstate, biomass, food, z * biomass, month_index=0)
    first = sstate.nage_s[0, 2] / sstate.nage_s[0, 1]
    # Group 1 better fed in month 2: its eggs and recruits rise, and group 2's
    # recruits follow at the first month's ratio, whatever its own eggs.
    food[p.spname.index("Adult")] *= 1.5
    split_update(st, sstate, biomass, food, z * biomass, month_index=1)
    assert sstate.nage_s[0, 2] / sstate.nage_s[0, 1] == pytest.approx(first, rel=1e-12)


def test_reader_reads_the_spawning_settings():
    import pandas as pd

    from .test_ewemdb_ba_detritus_import import (
        _STANZA,
        _STANZA_LIFE,
        _WIRING_DIET,
        _WIRING_GROUPS,
        _read_mocked,
    )

    stanza = _STANZA.assign(FixedFecundity=[1], RecStanza=[0])
    life = _STANZA_LIFE.assign(SpawnProp=[0.0, 0.6])
    params = _read_mocked(
        {
            "EcopathGroup": _WIRING_GROUPS,
            "EcopathDietComp": _WIRING_DIET,
            "Stanza": stanza,
            "StanzaLifeStage": life,
        }
    )
    sg = params.stanzas.stgroups.iloc[0]
    assert bool(sg["FixedFecundity"]) is True
    assert int(sg["RecStanza"]) == 0
    si = params.stanzas.stindiv.sort_values("StanzaNum")
    np.testing.assert_allclose(si["SpawnProp"].to_numpy(dtype=float), [0.0, 0.6])
    blank = _STANZA_LIFE.assign(SpawnProp=[np.nan, np.nan])  # EwE default 1
    params = _read_mocked(
        {
            "EcopathGroup": _WIRING_GROUPS,
            "EcopathDietComp": _WIRING_DIET,
            "Stanza": _STANZA,
            "StanzaLifeStage": blank,
        }
    )
    assert pd.to_numeric(params.stanzas.stindiv["SpawnProp"]).tolist() == [1.0, 1.0]
