"""Vectorised fishing-mortality and detritus kernels must match their loop
forms (the reference implementations) bit-for-bit, for every option."""

import itertools

import numpy as np
import pytest

from pypath.core.ecosim_deriv import (
    _fishing_mortality_loop,
    _rpath_detritus_derivs,
    _rpath_detritus_derivs_loop,
    fishing_mortality,
)


def _fishing_case(seed, legacy, forced, med, n_links=30):
    rng = np.random.default_rng(seed)
    NL, ND, NG_ = 15, 2, 3
    NG = NL + ND + NG_
    fish_from = np.r_[0, rng.integers(1, NL + ND + 1, n_links)]
    fish_through = np.r_[0, rng.integers(NL + ND + 1, NG + 1, n_links)]
    params = dict(NUM_GROUPS=NG, NUM_LIVING=NL, NUM_DEAD=ND, NUM_GEARS=NG_)
    if not legacy:
        params["fleet_idx"] = np.arange(NL + ND, NG)
    forcing = {"ForcedEffort": np.r_[1.0, rng.random(NG_) * 2]}
    if forced:
        forcing["ForcedF"] = rng.random(NG + 1) * (rng.random(NG + 1) < 0.4)
        forcing["ForcedFReplace"] = rng.random(NG + 1) < 0.3
    fishing = {
        "FishFrom": fish_from,
        "FishThrough": fish_through,
        "FishQ": np.r_[0, rng.random(n_links)],
    }
    fleet_med = rng.random(NG_) * 2 if med else None
    return params, forcing, fishing, fleet_med


@pytest.mark.parametrize("seed", range(3))
@pytest.mark.parametrize(
    "legacy,forced,med", list(itertools.product([False, True], repeat=3))
)
def test_fishing_mortality_matches_loop(seed, legacy, forced, med):
    case = _fishing_case(seed, legacy, forced, med)
    ref = _fishing_mortality_loop(*case)
    new = fishing_mortality(*case)
    np.testing.assert_array_equal(new[0], ref[0])
    np.testing.assert_array_equal(new[1], ref[1])


def _det_case(seed, ewe, n_det=40, n_fish=20):
    rng = np.random.default_rng(seed)
    NL, ND = 15, 3
    NG = NL + ND + 2
    n = NG + 1
    QQ = rng.random((n, n)) * (rng.random((n, n)) < 0.3)
    args = (
        QQ,
        rng.random(n) * 3,
        rng.random(n),
        rng.random(n) * 0.3,
        np.r_[0, rng.integers(0, NG + 2, n_det)],
        np.r_[0, rng.integers(0, NG + 2, n_det)],
        np.r_[0, rng.random(n_det)],
        np.r_[0, rng.integers(1, NL + ND + 1, n_fish)],
        np.r_[0, rng.integers(0, NG + 1, n_fish)],
        np.r_[0, rng.random(n_fish)],
        NL,
        ND,
    )
    kw = dict(
        import_q=rng.random(n) * 0.1,
        det_input=rng.random(n),
        fish_mort=rng.random(n) * (rng.random(n) < 0.5),
    )
    if ewe:
        det_out = np.full(n, np.nan)
        det_out[NL + 1] = 0.05
        kw.update(
            det_passed_prop=rng.random(n),
            det_ba=rng.random(n) * 0.1,
            det_out=det_out,
            det_emig=rng.random(n) * 0.1,
        )
    return n, args, kw


@pytest.mark.parametrize("seed", range(4))
@pytest.mark.parametrize(
    "ewe,pass_on,df_once,flows", list(itertools.product([False, True], repeat=4))
)
def test_detritus_derivs_match_loop(seed, ewe, pass_on, df_once, flows):
    n, args, kw = _det_case(seed, ewe)
    kw.update(pass_on=pass_on, df_once=df_once)
    kw_ref = dict(kw)
    if "det_out" in kw:
        kw_ref["det_out"] = kw["det_out"].copy()
    fl_ref, fl_new = ({}, {}) if flows else (None, None)
    d_ref, d_new = np.full(n, -3.0), np.full(n, -3.0)
    g_ref, l_ref = _rpath_detritus_derivs_loop(d_ref, *args, flows=fl_ref, **kw_ref)
    g_new, l_new = _rpath_detritus_derivs(d_new, *args, flows=fl_new, **kw)
    np.testing.assert_array_equal(d_new, d_ref)
    np.testing.assert_array_equal(g_new, g_ref)
    np.testing.assert_array_equal(l_new, l_ref)
    if "det_out" in kw:
        np.testing.assert_array_equal(kw["det_out"], kw_ref["det_out"])
    if flows:
        for key in fl_ref:
            np.testing.assert_array_equal(fl_new[key], fl_ref[key], err_msg=key)
