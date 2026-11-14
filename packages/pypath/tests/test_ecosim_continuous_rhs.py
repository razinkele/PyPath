"""The pure right-hand side of the continuous pathway."""

import copy

import numpy as np
import pytest

from pypath.core.ecosim_continuous import ContinuousRHS, continuous_emig
from tests.continuous_support import needs_reco, reco_plain, run_dicts, scenario

pytestmark = needs_reco

SWITCHES = (
    "RK4_FAST_EQUILIBRIUM",
    "STANZA_STAGE_RATES",
    "FTIME_START_OF_MONTH",
    "RK4_STEPS",
)


def build(scen, forcing_edit=None, **kwargs):
    """Return ``(rhs, pd, fd, fi, y0)`` for month 1 of ``scen``; ``y0`` is the
    base biomass of groups 1..n_bio. ``kwargs`` go to ``ContinuousRHS``."""
    state, pd, fd, fi = run_dicts(scen)
    if forcing_edit:
        forcing_edit(fd)
    n_bio = pd["NUM_LIVING"] + pd["NUM_DEAD"]
    switches = {k: getattr(scen.params, k, None) for k in SWITCHES}
    b0 = np.array(pd["Bbase"], dtype=float)
    rhs = ContinuousRHS(pd, fi, n_bio, b0, switches=switches, **kwargs)
    n1 = pd["NUM_GROUPS"] + 1
    rhs.set_month(fd, state, np.zeros(n1, dtype=bool), np.zeros(n1))
    return rhs, pd, fd, fi, b0[1 : n_bio + 1].copy()


def neutral(fd):
    fd["Ftime"][:] = 1.0


def snapshot(d):
    return {k: copy.deepcopy(v) for k, v in d.items()}


def same(a, b):
    if set(a) != set(b):
        return False
    for k in a:
        x, y = a[k], b[k]
        if isinstance(x, np.ndarray):
            if not np.array_equal(x, y, equal_nan=True):
                return False
        elif isinstance(x, dict):
            if not same(x, y):
                return False
    return True


def test_the_rhs_is_pure():
    scen = scenario(reco_plain())
    scen.params.CmCo[1 : scen.params.NUM_LIVING + 1] = 2.0
    state, pd, fd, fi = run_dicts(scen)
    n_bio = pd["NUM_LIVING"] + pd["NUM_DEAD"]
    switches = {k: getattr(scen.params, k, None) for k in SWITCHES}
    b0 = np.array(pd["Bbase"], dtype=float)
    pd_before, fd_before, fi_before = snapshot(pd), snapshot(fd), snapshot(fi)
    rhs = ContinuousRHS(pd, fi, n_bio, b0, switches=switches)
    n1 = pd["NUM_GROUPS"] + 1
    rhs.set_month(fd, state, np.zeros(n1, dtype=bool), np.zeros(n1))
    a = b0[1 : n_bio + 1].copy()
    b = a * 1.3
    first = rhs(0.0, a)
    rhs(0.1, b)
    rhs.rates(b)
    second = rhs(0.2, a)
    assert np.array_equal(first, second)
    assert same(pd, pd_before)
    assert same(fd, fd_before)
    assert same(fi, fi_before)
    # nothing is left for deriv_vector to write on the private copy either
    dead = slice(pd["NUM_LIVING"] + 1, n_bio + 1)
    assert not np.isnan(rhs._pd["_det_out"][dead]).any()
    det_before = rhs._pd["_det_out"].copy()
    rhs(0.3, b)
    assert np.array_equal(rhs._pd["_det_out"], det_before, equal_nan=True)


@pytest.mark.parametrize("rule", ["ewe", "rpath"])
@pytest.mark.parametrize("cmco2", [False, True])
def test_a_balanced_model_starts_balanced(rule, cmco2):
    if cmco2 and rule != "ewe":
        pytest.skip("handling time is the EwE rule")
    scen = scenario(reco_plain(), rule=rule, effort=1.0)
    if cmco2:
        scen.params.CmCo[1 : scen.params.NUM_LIVING + 1] = 2.0
    rhs, pd, fd, fi, y0 = build(scen, neutral)
    dy = rhs(0.0, y0)
    rel = np.abs(dy) / y0
    assert np.all(np.isfinite(dy))
    assert rel.max() < 1e-9, (rule, cmco2, rel.max(), int(rel.argmax()) + 1)


def test_a_zero_group_stays_finite():
    scen = scenario(reco_plain(), effort=1.0)
    rhs, pd, fd, fi, y0 = build(scen, neutral)
    y = y0.copy()
    y[4] = 0.0  # a prey group
    dy = rhs(0.0, y)
    assert np.all(np.isfinite(dy))
    assert dy[4] >= 0.0


def test_fast_groups_are_states():
    scen = scenario(reco_plain(), rule="ewe", effort=1.0)
    rhs, pd, fd, fi, y0 = build(scen, neutral)
    fast = np.flatnonzero(np.asarray(scen.params.NoIntegrate)[1 : len(y0) + 1] != 0)
    assert fast.size > 0
    dy = rhs(0.0, y0 * 1.5)
    assert np.all(np.abs(dy[fast]) > 0)


def test_the_rhs_closes_the_detritus_budget():
    """Form used: other-mortality loss alone, with UnassimRespFrac = 0 for the
    living groups (the rates carry no unassimilated-flow key); the inflow from
    living groups is the detritus pools' DetGain, effort 0 (no discards)."""
    scen = scenario(reco_plain(), rule="ewe", effort=0.0)
    nl = scen.params.NUM_LIVING
    scen.params.PaddP[1 : nl + 1] = 0.5
    scen.params.MoPred[1 : nl + 1] = 0.5
    scen.params.UnassimRespFrac[1 : nl + 1] = 0.0

    def edit(fd):
        fd["Ftime"][:] = 0.8

    rhs, pd, fd, fi, y0 = build(scen, edit)
    assert np.all(rhs._pd["Unassim"][1 : nl + 1] == 0.0)
    r = rhs.rates(y0 * 1.2)
    dead = slice(nl + 1, nl + scen.params.NUM_DEAD + 1)
    expected = r["MzeroLoss"][1 : nl + 1].sum()
    assert expected > 0
    assert r["DetGain"][dead].sum() == pytest.approx(expected, rel=1e-10)


def test_the_rhs_closes_the_detritus_budget_with_unassimilated_food():
    """Same closure with unassimilated consumption (Unassim x FoodGain)."""
    scen = scenario(reco_plain(), rule="ewe", effort=0.0)
    nl = scen.params.NUM_LIVING
    scen.params.PaddP[1 : nl + 1] = 0.5
    scen.params.MoPred[1 : nl + 1] = 0.5

    def edit(fd):
        fd["Ftime"][:] = 0.8

    rhs, pd, fd, fi, y0 = build(scen, edit)
    r = rhs.rates(y0 * 1.2)
    unassim = rhs._pd["Unassim"]
    expected = (
        r["MzeroLoss"][1 : nl + 1] + unassim[1 : nl + 1] * r["FoodGain"][1 : nl + 1]
    ).sum()
    dead = slice(nl + 1, nl + scen.params.NUM_DEAD + 1)
    assert expected > 0
    assert r["DetGain"][dead].sum() == pytest.approx(expected, rel=1e-10)


def test_living_emigration_uses_the_rate():
    emig = np.array([0.0, 0.0, 0.3, 0.0, 0.5])
    rate = np.array([0.0, 0.2, 0.9, 0.0, 0.1])
    out = continuous_emig(emig, rate, 3)
    assert np.allclose(out, [0.0, 0.2, 0.3, 0.0, 0.5])
    assert out is not emig and emig[1] == 0.0
    assert continuous_emig(None, rate, 3)[1] == 0.2
    assert continuous_emig(None, None, 3) is None
    assert continuous_emig(np.zeros(5), np.zeros(5), 3) is None
    assert continuous_emig(None, np.array([0.0, 0, 0, 0, 0.4]), 3) is None


def test_the_corrections_and_ignored_switches_are_recorded():
    scen = scenario(reco_plain(), rule="ewe", effort=1.0)
    rhs, *_ = build(scen, neutral)
    for name in ("R2", "R4", "R5", "DF once", "Hden0 recalibrated"):
        assert name in rhs.corrections
    assert "RK4_FAST_EQUILIBRIUM" in rhs.ignored_switches
    assert "STANZA_STAGE_RATES" in rhs.ignored_switches


def test_corrections_can_be_turned_off():
    """``corrections=False`` leaves R2, R4, R5 and DF once out; handling time
    keeps the recalibrated Hden0 and the exact solve."""
    scen = scenario(reco_plain(), rule="ewe", effort=1.0)
    rhs, pd, *_ = build(scen, neutral, corrections=False)
    assert pd["DETRITUS_MO_UNSCALED"]  # the EwE rule's value
    assert rhs._pd["DETRITUS_MO_UNSCALED"] == pd["DETRITUS_MO_UNSCALED"]
    assert "DETRITUS_MO_TOT" not in rhs._pd
    assert "DETRITUS_DF_ONCE" not in rhs._pd
    assert rhs._pd["_hden_exact"] is True
    assert rhs.corrections == ["Hden0 recalibrated"]
    on, *_ = build(scen, neutral)
    assert on._pd["DETRITUS_MO_UNSCALED"] is False
    assert on._pd["DETRITUS_MO_TOT"] is True and on._pd["DETRITUS_DF_ONCE"] is True


def test_emig_rate_reaches_the_continuous_rhs_only():
    """R4: a living group's EmigRate is a loss on the continuous path alone."""
    from pypath.core.ecosim import rsim_run

    g = 12  # OtherGroundfish, no Emigration entered
    scen = scenario(reco_plain(), rule="ewe", effort=1.0)
    _, pd, _, _ = run_dicts(scen)
    assert pd.get("Emig") is None or pd["Emig"][g] == 0.0
    scen.params.EmigRate[g] = 0.05
    rhs, pd, fd, fi, y0 = build(scen, neutral)
    assert pd["EmigRate"][g] == 0.05
    dy = rhs(0.0, y0)
    rel = dy / y0
    assert rel[g - 1] == pytest.approx(-0.05, rel=1e-6)
    others = np.delete(np.abs(rel), g - 1)
    assert others.max() < 1e-9
    off, *_ = build(scen, neutral, corrections=False)
    assert abs(off(0.0, y0)[g - 1] / y0[g - 1]) < 1e-9
    # the parity path ignores it
    plain = scenario(reco_plain(), rule="ewe", effort=1.0)
    with_rate = rsim_run(scen, method="RK4", years=range(1, 2))
    without = rsim_run(plain, method="RK4", years=range(1, 2))
    assert np.array_equal(with_rate.out_Biomass, without.out_Biomass)


def test_stats_and_forced_groups():
    scen = scenario(reco_plain(), effort=1.0)
    rhs, pd, fd, fi, y0 = build(scen, neutral)
    y = y0.copy()
    y[2] = -1e-3
    rhs(0.0, y)
    assert rhs.stats["clips"] == 1
    assert rhs.stats["min_state"] == pytest.approx(-1e-3)
    assert rhs.stats["hden_evals_max"] >= 1
    n1 = pd["NUM_GROUPS"] + 1
    mask = np.zeros(n1, dtype=bool)
    mask[3] = True
    vals = np.zeros(n1)
    vals[3] = y0[2] * 2
    state = np.array(pd["Bbase"], dtype=float)
    rhs.set_month(fd, state, mask, vals)
    assert rhs.stats["clips"] == 0
    assert rhs(0.0, y0)[2] == 0.0
