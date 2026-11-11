"""A stanza group's biomass accumulation rate, EwE's Stanza.BABsplit.

EwE6 (cEcoSimModel.CalculateStanzaParameters) adds BaB to every stage's
mortality in the age table, Surv = exp(-(Z + BaB) / 12) (vb 4311, the plus
group included), and scales Ecosim's base recruits by it,
RzeroS = Recruits x exp(BaB / 12) (vb 4362). With the stages losing Z in
Ecosim, the numbers then grow by exp(BaB / 12) a month: the group
accumulates biomass at rate BaB. (EwE also computes Bat = BaB x B per stage
but ignores it; the stage BA it writes to EcopathGroup is left out of
stanza EE, see _drop_stanza_stage_bioacc.) The reader ignored BABsplit with
a warning, and RzeroS had no exp(BaB / 12). No database here sets it.
"""

import logging
import warnings

import numpy as np
import pytest

from pypath.core.ecopath import rpath
from pypath.core.ecosim import rsim_scenario
from pypath.core.stanzas import (
    RsimStanzaState,
    rpath_stanzas,
    split_set_pred,
    split_update,
)

from .test_ewemdb_ba_detritus_import import (
    _STANZA,
    _STANZA_LIFE,
    _WIRING_DIET,
    _WIRING_GROUPS,
    _read_mocked,
)
from .test_stanza_plus_group import Z_AD, Z_JUV, _mussel_params

BAB = 0.2


def test_reader_reads_babsplit(caplog):
    with caplog.at_level(logging.WARNING, logger="pypath.io.ewemdb"):
        params = _read_mocked(
            {
                "EcopathGroup": _WIRING_GROUPS,
                "EcopathDietComp": _WIRING_DIET,
                "Stanza": _STANZA,  # BABsplit -0.1
                "StanzaLifeStage": _STANZA_LIFE,
            }
        )
    assert float(params.stanzas.stgroups["BAB"].iloc[0]) == pytest.approx(-0.1)
    # It is used now, so the reader no longer reports it as ignored.
    assert not any("BABsplit" in r.getMessage() for r in caplog.records)


def _bab_params(bab=BAB):
    params = _mussel_params()
    params.stanzas.stanza_groups[0].bab = bab
    return params


def test_ewe_table_survival_includes_babsplit():
    params = _bab_params()
    rpath_stanzas(params, stanza_params=params.stanzas, rule="ewe")
    surv = params.stanzas.st_groups[1]["Survive"].to_numpy()
    s_juv = np.exp(-(Z_JUV + BAB) / 12.0)
    s_ad = np.exp(-(Z_AD + BAB) / 12.0)
    np.testing.assert_allclose(surv[1:12] / surv[:11], s_juv, rtol=1e-12)
    np.testing.assert_allclose(surv[13:-1] / surv[12:-2], s_ad, rtol=1e-12)
    assert surv[-1] / surv[-2] == pytest.approx(s_ad / (1.0 - s_ad), rel=1e-12)


def _scenario(rule, bab=BAB):
    params = _bab_params(bab)
    rpath_stanzas(params, rule=rule)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        model = rpath(params)
    sc = rsim_scenario(model, params, years=range(1, 3), fast_group_rule=rule)
    assert sc.stanzas is not None
    return model, sc


@pytest.mark.parametrize("rule, factor", [("ewe", np.exp(BAB / 12.0)), ("rpath", 1.0)])
def test_base_recruits_carry_babsplit_under_the_ewe_rule(rule, factor):
    _, sc = _scenario(rule)
    st = sc.stanzas
    assert st.r_zero_s[1] == pytest.approx(st.recruits[1] * factor, rel=1e-12)


def test_ewe_rule_numbers_grow_at_babsplit():
    """One SplitUpdate month at the Ecopath rates (stages lose Z): every age
    from month 1 up, the plus group included, holds exp(BaB / 12) times its
    base numbers."""
    model, sc = _scenario("ewe")
    st, p = sc.stanzas, sc.params
    juv, ad = p.spname.index("Juv"), p.spname.index("Adult")
    state = RsimStanzaState.from_base(st, p.NUM_GROUPS)
    biomass = np.array(p.B_BaseRef, dtype=float)
    split_set_pred(st, state, biomass)
    n0 = state.nage_s[:, 1].copy()
    food = np.zeros_like(biomass)
    z = np.zeros_like(biomass)
    for i, zi in ((juv, Z_JUV), (ad, Z_AD)):
        food[i] = model.QB[i - 1] * model.Biomass[i - 1]
        z[i] = zi
    split_update(st, state, biomass, food, z * biomass)
    last = int(st.age2[1, 2])
    ratio = state.nage_s[1 : last + 1, 1] / n0[1 : last + 1]
    np.testing.assert_allclose(ratio, np.exp(BAB / 12.0), rtol=1e-10)
