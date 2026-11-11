"""EwE's multi-stanza age table and plus group, under the EwE rule.

EwE6 (cEcoSimModel.CalculateStanzaParameters, vb 4288-4335) ends the age
table at the month a cohort reaches 90 % of W_inf, ``CInt(40.3978 / K)``,
at least 12 months into the last stanza and at most ``cCore.MAX_AGE`` (400),
and closes it with a plus group. The survivors accumulate in the last month
(``Survive / (1 - Surv)``), and its weight is the mean over the tail of the
Ford-Walford line ``w' = Ahat + Rhat * w`` fitted to the last three months.
Each month Ecosim re-closes the plus group with the same line
(SplitUpdate, vb 1866) instead of a numbers-weighted mean. Rpath runs the
table out to 99.999 % of the cumulative biomass, with no plus group; the
'rpath' rule keeps that.

The numbers below use LT2022's Blue mussel: K = 1.175, Z = 1.94 (juveniles,
months 0-11) and 1.75 (adults, from month 12), leading adult B = 27, Q/B = 8.
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

K, Z_JUV, Z_AD = 1.175, 1.94, 1.75


def _mussel_params(k=K):
    params = create_rpath_params(
        ["Phyto", "Juv", "Adult", "Det"],
        [1, 0, 0, 2],
        stgroups=[None, "Mussel", "Mussel", None],
    )
    params.model["Biomass"] = [100.0, 15.0, 27.0, 10.0]
    params.model["PB"] = [50.0, Z_JUV, Z_AD, 0.0]
    params.model["QB"] = [0.0, 13.0, 8.0, 0.0]
    params.model["EE"] = np.nan  # Ecopath solves EE from B and P/B
    params.model["Unassim"] = [0.0, 0.2, 0.2, 0.0]
    params.model["Det"] = [1.0, 1.0, 1.0, 0.0]
    diet = params.diet
    diet.loc[diet["Group"] == "Phyto", ["Juv", "Adult"]] = 1.0
    params.stanzas = create_stanza_params(
        [{"stanza_group_num": 1, "n_stanzas": 2, "vbgf_ksp": k}],
        [
            {
                "stanza_group_num": 1,
                "stanza_num": 1,
                "group_num": 2,
                "group_name": "Juv",
                "first": 0,
                "last": 11,
                "z": Z_JUV,
            },
            {
                "stanza_group_num": 1,
                "stanza_num": 2,
                "group_num": 3,
                "group_name": "Adult",
                "first": 12,
                "last": 60,
                "z": Z_AD,
                "leading": True,
            },
        ],
    )
    return params


def _ewe_table(k=K):
    params = _mussel_params(k)
    sp = params.stanzas
    rpath_stanzas(params, stanza_params=sp, rule="ewe")
    return params, sp, sp.st_groups[1]


@pytest.mark.parametrize(
    "k, last_month",
    [
        (K, 34),  # CInt(40.3978 / 1.175) = CInt(34.38)
        (3.0, 24),  # CInt(13.47) = 13, raised to first + 12
        (0.05, 400),  # CInt(807.96), capped at cCore.MAX_AGE
    ],
)
def test_ewe_age_table_ends_at_90_percent_of_winf(k, last_month):
    _, sp, table = _ewe_table(k)
    assert sp.stanza_groups[0].last_month == last_month
    assert sp.stanza_individuals[-1].last == last_month
    assert table["age"].iloc[-1] == last_month


def test_rule_must_be_ewe_or_rpath():
    params = _mussel_params()
    with pytest.raises(ValueError, match="rule"):
        rpath_stanzas(params, stanza_params=params.stanzas, rule="EwE6")


def test_plus_group_weight_is_the_tail_mean_of_the_walford_line():
    """The plus group's weight is the survival-weighted mean of the weights
    the survivors reach along w' = Ahat + Rhat * w, summed explicitly."""
    _, _, table = _ewe_table()
    n = int(table["age"].iloc[-1])
    w = (1.0 - np.exp(-K * np.arange(n + 1) / 12.0)) ** 3
    rhat = (w[n] - w[n - 1]) / (w[n - 1] - w[n - 2])
    ahat = w[n] - rhat * w[n - 1]
    s = np.exp(-Z_AD / 12.0)
    tail, wj = [], w[n]
    for _ in range(5000):
        tail.append(wj)
        wj = ahat + rhat * wj
    weights = s ** np.arange(len(tail))
    expected = float(np.sum(weights * tail) / np.sum(weights))
    assert table["WageS"].iloc[-1] == pytest.approx(expected, rel=1e-12)
    # Below the plus group the weights are EwE's (1 - exp(-K a / 12))^3,
    # with the exact exponent (not VBGF_d = 0.66667).
    np.testing.assert_allclose(table["WageS"].iloc[:-1], w[:-1], rtol=1e-14)


def test_plus_group_numbers_accumulate_the_survivors():
    _, _, table = _ewe_table()
    s = np.exp(-Z_AD / 12.0)
    surv = table["Survive"].to_numpy()
    assert surv[-1] / surv[-2] == pytest.approx(s / (1.0 - s), rel=1e-12)


def test_plus_group_consumption_weight_keeps_ewes_rhat():
    """EwE closes W^(2/3) with its own slope RhatC but the intercept from the
    weight's Rhat (AhatC = WWa(n) - Rhat * WWa(n-1), cEcoSimModel.vb:4334):
    0.94205 here, against 0.95582 with RhatC in both places."""
    _, _, table = _ewe_table()
    assert table["QageS"].iloc[-1] == pytest.approx(0.9420537349699875, rel=1e-12)


def test_ewe_table_reproduces_the_ecopath_stanza():
    """EwE's derived juvenile biomass and Q/B for this stanza (LT2022 stores
    15.005259 and 13.620459, in single precision)."""
    params, sp, _ = _ewe_table()
    model = params.model.set_index("Group")
    assert model.loc["Juv", "Biomass"] == pytest.approx(15.005259, rel=1e-6)
    assert model.loc["Juv", "QB"] == pytest.approx(13.620459, rel=1e-6)
    assert model.loc["Adult", "Biomass"] == pytest.approx(27.0, rel=1e-12)


def _ewe_scenario():
    params = _mussel_params()
    rpath_stanzas(params, rule="ewe")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        model = rpath(params)
    sc = rsim_scenario(model, params, years=range(1, 3), fast_group_rule="ewe")
    assert sc.stanzas is not None, "stanza setup failed under the EwE rule"
    return model, sc


def test_ewe_rule_scenario_uses_the_ewe_table():
    _, sc = _ewe_scenario()
    st = sc.stanzas
    assert int(st.age2[1, 2]) == 34


def test_rpath_rule_scenario_keeps_rpaths_table():
    params = _mussel_params()
    rpath_stanzas(params)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        model = rpath(params)
    sc = rsim_scenario(model, params, years=range(1, 3), fast_group_rule="rpath")
    assert int(sc.stanzas.age2[1, 2]) > 34


def test_plus_group_steps_once_then_holds_at_equilibrium():
    """One SplitUpdate month at the Ecopath rates: EwE's growth coefficient at
    month n - 1 was set from the corrected plus-group weight, so the plus
    group's weight steps once (adult biomass x 1.0018577, as the EwE 6.7.144
    oracle's 27.000 -> 27.050 in LT2022 scenario 16) and then holds. The
    juveniles do not move."""
    model, sc = _ewe_scenario()
    st, p = sc.stanzas, sc.params
    juv, ad = p.spname.index("Juv"), p.spname.index("Adult")
    state = RsimStanzaState.from_base(st, p.NUM_GROUPS)
    biomass = np.array(p.B_BaseRef, dtype=float)
    split_set_pred(st, state, biomass)
    b0 = biomass.copy()
    food = np.zeros_like(biomass)
    z = np.zeros_like(biomass)
    for i, zi in ((juv, Z_JUV), (ad, Z_AD)):
        food[i] = model.QB[i - 1] * model.Biomass[i - 1]
        z[i] = zi
    n = int(st.age2[1, 2])
    w_plus0 = float(st.base_wage_s[n, 1])
    series, w_plus = [], []
    for _ in range(6):
        # Losses proportional to biomass: survival exp(-Z / 12) every month.
        split_update(st, state, biomass, food, z * biomass)
        split_set_pred(st, state, biomass)
        series.append(biomass[[juv, ad]] / b0[[juv, ad]])
        w_plus.append(float(state.wage_s[n, 1]))
    series = np.array(series)
    assert series[0, 1] == pytest.approx(1.0018577, rel=1e-7)
    assert series[0, 0] == pytest.approx(1.0, abs=1e-12)
    # The plus group's weight steps from 0.93486 to 0.95975 and holds.
    assert w_plus0 == pytest.approx(0.934858050834849, rel=1e-12)
    assert w_plus[0] == pytest.approx(0.9597547706362555, rel=1e-12)
    np.testing.assert_allclose(w_plus[1:], w_plus[0], rtol=1e-12)
    # Adult biomass holds too, but for recruits: eggs are counted after the
    # month's growth (as EwE's SplitUpdate), so recruitment moves slightly
    # and reaches the adults through the juveniles' growth (2e-5 by month 6).
    np.testing.assert_allclose(series[1:, 1], series[0, 1], rtol=1e-4)


def test_stanza_feeding_time_follows_the_predator_index(monkeypatch):
    """EwE's feeding-time ratio is consumption per predator index (CBlast =
    EatenByAvg / PredAvg against Cbase = StartEatenBy / pred; vb 1762, 3310).
    For a stanza group the index is sum(N * WWa), which the plus-group step
    leaves alone, so the adults' feeding time stays at 1 for month 3. Per
    unit biomass (x 1.00186 after month 1) it would rise to
    0.1 + 0.9 * (1 - 0.5 + 0.5 * 1.00186) = 1.00084."""
    import pypath.core.ecosim_deriv as D
    from pypath.core.ecosim import rsim_run

    _, sc = _ewe_scenario()
    ad = sc.params.spname.index("Adult")
    sc.params.FtimeAdj[ad] = 0.5
    seen = []
    real = D.deriv_vector

    def spy(state, params, forcing, fishing, *a, **k):
        seen.append(float(forcing["Ftime"][ad]))
        return real(state, params, forcing, fishing, *a, **k)

    monkeypatch.setattr(D, "deriv_vector", spy)
    out = rsim_run(sc, method="RK4")
    months = sc.forcing.ForcedBio.shape[0]
    per_month = len(seen) // months
    ftime = [seen[m * per_month] for m in range(months)]
    # The plus-group step happened (adult biomass up 0.19 % after month 1).
    assert out.out_Biomass[1, ad] / out.out_Biomass[0, ad] > 1.0018
    assert ftime[2] == pytest.approx(1.0, abs=1e-6)
