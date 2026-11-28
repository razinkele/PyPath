"""Batched (all-patch) Ecosim derivative for Ecospace.

``deriv_vector_spatial`` evaluates the local Ecosim dynamics of every patch.
Calling :func:`pypath.core.ecosim_deriv.deriv_vector` once per patch spends
most of its time in per-call Python overhead, so this module evaluates all
patches at once on arrays laid out patch-major, ``(n_patches, n_groups + 1)``.

The arithmetic mirrors ``deriv_vector`` operation for operation (same
operand order, same sequential/pairwise summations), so results are
bit-for-bit identical to the per-patch path. Only the commonly used feature
set is covered, including EwE's Ecospace base calibration (each patch's
feeding efficiency EatEff, ``eat_eff``), HandleSwitch and EwE's relative
primary production map (``nut_forcing``). :func:`batch_supported` returns
``False`` for anything else (IBM groups, mediation, prey switching, suite
pooling, variable P/Q, metabolism models, link-level forcing,
instrumentation, the legacy detritus matrix, a single ``_eat_eff``, exact
handling time, the MoTot rule, ...), and the caller then falls back to the
per-patch loop.

A HandleSwitch exponent of 0 or 1 (the values EwE databases hold) gives the
loop's bits; another positive exponent goes through numpy's array power,
which can differ from the loop's scalar power in the last bit.
"""

from __future__ import annotations

import numpy as np

from pypath.core.ecosim_deriv import (
    _det_inflow,
    _ewe_nutrient_factor,
    _fishing_links,
    fishing_mortality,
)

#: rates keys deriv_batched produces (the ones Ecospace's RK4 asks for)
BATCHED_RATE_KEYS = frozenset(
    ("FoodGain", "LossPropToB", "TotGain", "TotLoss", "NetGain", "NetLoss")
)


def _any_positive(x) -> bool:
    return x is not None and bool(np.any(np.asarray(x) > 0))


def batch_supported(params: dict, forcing: dict, rate_keys=()) -> bool:
    """True when :func:`deriv_batched` reproduces ``deriv_vector`` exactly."""
    if not isinstance(params, dict) or not isinstance(forcing, dict):
        return False
    if not set(rate_keys) <= BATCHED_RATE_KEYS:
        return False
    # _space_cal / _eat_eff_patches (EwE's base calibration) are supported:
    # the calibrated arrays are shared by every patch and deriv_batched takes
    # each patch's EatEff. A single _eat_eff (one patch's) is not.
    for key in ("_eat_eff", "_mo_tot"):
        if params.get(key) is not None:
            return False
    if params.get("_hden_exact", False) or params.get("_KERNEL_VECTOR", False):
        return False
    if params.get("DetPassedProp") is not None and params.get(
        "DETRITUS_DF_ONCE", False
    ):
        return False
    if params.get("ibm_groups"):
        return False
    for key in (
        "_mediation",
        "EweMediation",
        "INSTRUMENT_GROUPS",
        "TRACE_DEBUG_GROUPS",
        "ScrambleSelf",
        "HandleSelf",
        "PredPredWeight",
        "PreyPreyWeight",
    ):
        if params.get(key) is not None:
            return False
    hs = params.get("HandleSwitch")
    if hs is not None:
        hs = np.asarray(hs)
        G = params["NUM_GROUPS"] + 1
        if hs.ndim != 2 or hs.shape[0] < G or hs.shape[1] < G:
            return False
    if _any_positive(params.get("SwitchPower")):
        return False
    if params.get("UseVarPQ", False):
        return False
    if (
        params.get("MetabolismModel") is not None
        and params.get("MetabolicTemperature") is not None
        and params.get("MetabolicBodyMass") is not None
    ):
        return False
    if params.get("_link_prey") is None or params.get("_link_pred") is None:
        return False
    if params.get("DetLinkFrom") is None:
        return False
    if params.get("M0") is None or not isinstance(params.get("M0"), np.ndarray):
        return False
    for key in ("ForcedVulLink", "ForcedSearchLink"):
        if forcing.get(key) is not None:
            return False
    ng = params["NUM_GROUPS"] + 1
    for key in ("ForcedRecs", "RecruitmentAdj", "ForcedActresp"):
        v = forcing.get(key)
        if v is not None and np.asarray(v).size != ng:
            return False
    # EwE detritus export must already be fixed (NaN entries are set by the
    # first call, patch by patch).
    if params.get("DetPassedProp") is not None:
        out = params.get("_det_out")
        if out is None:
            return False
        nl, nd = params["NUM_LIVING"], params["NUM_DEAD"]
        if np.isnan(np.asarray(out, dtype=float)[nl + 1 : nl + nd + 1]).any():
            return False
    return True


def _hden_update(hden_old, q, index, cmco_qb):
    """Batched ``_ewe_hden_update`` (same arithmetic, element-wise)."""
    out = np.array(hden_old, dtype=float, copy=True)
    cm = np.broadcast_to(cmco_qb, hden_old.shape)
    ok = (cm > 0) & (index > 0)
    htime_hdent = hden_old[ok] * q[ok] / (cm[ok] * index[ok])
    out[ok] = 0.5 * (1.0 + htime_hdent) + 0.5 * hden_old[ok]
    return out


def _prey_multiplier_arena(VV, QQbase, f):
    """``ewe_prey_multiplier_arena`` for a 1-D f; QQbase may be (P, n, n)."""
    f = np.asarray(f, dtype=float)
    rows = np.flatnonzero(f != 1.0)
    rows = rows[(rows > 0) & (rows < VV.shape[0])]
    if rows.size == 0:
        return VV, QQbase
    VV = np.array(VV, dtype=float)
    QQbase = np.array(QQbase, dtype=float)
    fr = f[rows][:, None]
    vv_old = VV[rows, :]
    vv_new = 1.0 + (vv_old - 1.0) * (1.0 + fr) / 2.0
    pos = vv_new > 0
    denom = np.where(pos, vv_new, 1.0)
    if QQbase.ndim == 3:
        QQbase[:, rows, :] = np.where(
            pos, QQbase[:, rows, :] * vv_old * fr / denom, 0.0
        )
    else:
        QQbase[rows, :] = np.where(pos, QQbase[rows, :] * vv_old * fr / denom, 0.0)
    VV[rows, :] = vv_new
    return VV, QQbase


def _consumption(
    QQ, BB, VV, DD, QQbase, preyYY, predYY, prey, pred, hs_exp=None, eat_eff=None
):
    """Batched sparse consumption kernel (no pooling / prey switching /
    mediation).

    Same formula as ``_compute_consumption_sparse_python``; links skipped
    there (non-positive biomass or base flow) leave ``QQ`` untouched.
    ``hs_exp`` is each link's HandleSwitch * COUPLED (None: no HandleSwitch);
    ``eat_eff`` (P, G) each patch's kernel EatEff, dividing the predator
    term of the vulnerability denominator (None: no EatEff).
    """
    P = BB.shape[0]
    vv = VV[prey, pred]
    dd = DD[prey, pred]
    if QQbase.ndim == 3:
        qbase = QQbase[:, prey, pred]
    else:
        qbase = np.broadcast_to(QQbase[prey, pred], (P, prey.size))
    keep = ~((BB[:, prey] <= 0.0) | (BB[:, pred] <= 0.0) | (qbase <= 0.0))
    PYY = preyYY[:, prey]
    PDY = predYY[:, pred]
    dd_prey_safe = np.maximum(PYY, 1e-10)
    PYY_term = PYY
    dd_denom = dd_prey_safe
    if hs_exp is not None:
        switch = hs_exp > 0.0
        if switch.any():
            # the loop: hs > 0 -> max(PYY, 1e-10) ** hs in both terms; x ** 1
            # is x, so only other exponents take the power
            PYY_term = np.where(switch, dd_prey_safe, PYY)
            power = switch & (hs_exp != 1.0)
            if power.any():
                e = hs_exp[power]
                PYY_term[:, power] = PYY_term[:, power] ** e
                dd_denom = dd_prey_safe.copy()
                dd_denom[:, power] = dd_denom[:, power] ** e
    dd_term = dd / np.maximum(dd - 1.0 + dd_denom, 1e-10)
    pdy = PDY if eat_eff is None else PDY / eat_eff[:, pred]
    vv_pred_safe = np.maximum(pdy, 1e-10)
    vv_term = vv / np.maximum(vv - 1.0 + vv_pred_safe, 1e-10)
    Q_calc = qbase * PDY * PYY_term * dd_term * vv_term
    val = np.where(Q_calc > 0.0, Q_calc, 0.0)
    if keep.all():
        QQ[:, prey, pred] = val
    else:
        p_idx, l_idx = np.nonzero(keep)
        QQ[p_idx, prey[l_idx], pred[l_idx]] = val[p_idx, l_idx]


def deriv_batched(
    S,
    params,
    forcing,
    fishing,
    *,
    bbase=None,
    qqbase=None,
    hden=None,
    patch_effort=None,
    stanza_ratio=None,
    eat_eff=None,
    nut_forcing=None,
    want_rates=False,
):
    """Ecosim derivative of every patch at once.

    Parameters
    ----------
    S : (P, G) array
        Per-patch state (densities, as ``deriv_vector`` receives them).
    bbase : (P, G) array, optional
        Per-patch base biomass (habitat capacity); default ``params["Bbase"]``.
    qqbase : (P, G, G) array, optional
        Per-patch base consumption; default ``params["QQbase"]``.
    hden : (P, G) array, optional
        Per-patch handling-time state; updated in place unless frozen.
    patch_effort : (P, n_effort) array, optional
        Per-patch ``ForcedEffort``.
    stanza_ratio : (P, G) array, optional
        Per-patch ``StanzaPredRatio``.
    eat_eff : (P, G) array, optional
        Per-patch feeding efficiency (``params["_eat_eff"]`` of each patch,
        EwE's EatEff): scales producers' production and divides consumers'
        predator term in the vulnerable pool.
    nut_forcing : (P,) array, optional
        Per-patch ``NutForcing`` (EwE's relative primary production map);
        default ``forcing["NutForcing"]`` for every patch.

    Returns ``(deriv, rates)``: (P, G) derivative and, when ``want_rates``, a
    dict of (P, G) arrays keyed like ``deriv_vector``'s ``rates``.
    """
    NG = params["NUM_GROUPS"]
    NL = params["NUM_LIVING"]
    ND = params["NUM_DEAD"]
    G = NG + 1
    P = S.shape[0]

    PB = params["PB"]
    QB = params.get("QB", np.zeros(G))
    VV = params["VV"]
    DD = params["DD"]
    Unassim = params.get("Unassim", np.zeros(G))
    Bbase = params.get("Bbase") if bbase is None else bbase
    if Bbase is None:
        Bbase = S.copy()
    PP_type = params.get("PP_type", np.zeros(G, dtype=int))
    M0_arr = params["M0"]
    deriv = np.zeros((P, G))
    BB = S.copy()
    pp_t = np.asarray(PP_type, dtype=np.int64)  # as deriv_vector's kernels

    Ftime = forcing.get("Ftime", np.ones(G))
    ForcedBio = forcing.get("ForcedBio", np.zeros(G))
    PP_forcing = forcing.get("PP_forcing", np.ones(G))
    PPF = np.broadcast_to(np.asarray(PP_forcing, dtype=float), (P, G))

    # EwE6 nutrient limitation of primary production, per patch
    _nut_prop = params.get("NutBaseFreeProp")
    if _nut_prop is not None:
        nb = slice(1, NL + ND + 1)
        _producer = np.asarray(PP_type, dtype=float)[:G] > 0
        _producer[0] = False
        _nut_base = params.get("_nut_base")
        nut_b = Bbase if _nut_base is None else _nut_base
        b_sum_shared = None if np.ndim(nut_b) == 2 else float(np.sum(nut_b[nb]))
        nut_f_all = float(forcing.get("NutForcing", 1.0))
        rows = []
        facts = []
        for p in range(P):
            nut_f = nut_f_all if nut_forcing is None else float(nut_forcing[p])
            b_sum = (
                b_sum_shared
                if b_sum_shared is not None
                else float(np.sum(nut_b[p, nb]))
            )
            f = _ewe_nutrient_factor(float(np.sum(S[p, nb])), b_sum, _nut_prop, nut_f)
            if f != 1.0:
                rows.append(p)
                facts.append(f)
        if rows:
            PPF = np.array(PPF, dtype=float, copy=True)
            for p, f in zip(rows, facts):
                PPF[p, _producer] *= f

    # Ecospace feeding efficiency (deriv_vector's _eat_eff, per patch):
    # producers' production x EatEff (after the nutrient factor, as there),
    # and the kernel's EatEff with producers at 1
    _kernel_eat = None
    if eat_eff is not None:
        _eat = np.asarray(eat_eff, dtype=float)
        _prod_e = np.asarray(PP_type, dtype=float)[: _eat.shape[1]] > 0
        _prod_e[0] = False
        PPF = np.array(PPF, dtype=float, copy=True)
        PPF[:, : _eat.shape[1]][:, _prod_e] *= _eat[:, _prod_e]
        _kernel_eat = _eat.copy()
        _kernel_eat[:, _prod_e] = 1.0

    ForcedPrey = forcing.get("ForcedPrey", np.ones(G))
    ForcedSearch = forcing.get("ForcedSearch", None)
    ForcedMort = forcing.get("ForcedMort", None)
    ForcedMigrate = forcing.get("ForcedMigrate", np.zeros(G))
    ForcedActresp = forcing.get("ForcedActresp", np.ones(G))
    ForcedRecs = forcing.get("ForcedRecs", np.ones(G))
    RecruitmentAdj = forcing.get("RecruitmentAdj", np.ones(G))

    Bb = np.broadcast_to(Bbase, (P, G))
    safe_bbase = np.where(Bb > 0, Bb, 1.0)
    preyYY = np.zeros((P, G))
    preyYY[:, 1:] = np.where(
        Bb[:, 1:] > 0,
        Ftime[1:] * BB[:, 1:] / safe_bbase[:, 1:] * ForcedPrey[1:],
        0.0,
    )
    _ewe_prey_f = None
    if params.get("PREY_FTIME_ARENA", False):
        _ewe_prey_f = np.ones(G)
        _ewe_prey_f[1:] = Ftime[1:] * ForcedPrey[1:]
        preyYY[:, 1:] = np.where(Bb[:, 1:] > 0, BB[:, 1:] / safe_bbase[:, 1:], 0.0)
    rpath_pp = params.get("RpathPrimaryProduction", False)
    if rpath_pp:
        ok0 = Bb[:, 0] > 0
        if ok0.any():
            preyYY[ok0, 0] = Ftime[0] * BB[ok0, 0] / safe_bbase[ok0, 0] * ForcedPrey[0]

    predYY = np.zeros((P, G))
    sl = slice(1, NL + 1)
    predYY[:, sl] = np.where(
        Bb[:, sl] > 0, Ftime[sl] * BB[:, sl] / safe_bbase[:, sl], 0.0
    )
    _has_ratio = None
    if stanza_ratio is not None:
        _has_ratio = np.isfinite(stanza_ratio)
        Ft2 = np.broadcast_to(Ftime, (P, G))
        predYY[_has_ratio] = Ft2[_has_ratio] * stanza_ratio[_has_ratio]
    if ForcedSearch is not None:
        _fs = np.asarray(ForcedSearch, dtype=float)
        if _fs.shape[0] == G:
            _consumer = np.asarray(PP_type, dtype=float)[:G] == 0
            _consumer[0] = False
            predYY[:, _consumer] = predYY[:, _consumer] * _fs[_consumer]

    QQbase = params.get("QQbase", np.zeros((G, G))) if qqbase is None else qqbase
    if _ewe_prey_f is not None:
        VV, QQbase = _prey_multiplier_arena(VV, QQbase, _ewe_prey_f)

    prey = np.asarray(params["_link_prey"], dtype=np.intp)
    pred = np.asarray(params["_link_pred"], dtype=np.intp)
    QQ = np.zeros((P, G, G))
    _hs = params.get("HandleSwitch")
    hs_exp = None
    if _hs is not None:
        hs_exp = np.asarray(_hs, dtype=float)[prey, pred] * params.get("COUPLED", 1)
    kw = {"hs_exp": hs_exp, "eat_eff": _kernel_eat}

    _hden0 = params.get("_hden0")
    if hden is not None:
        _cmco_qb = params["_cmco_qb"]
        _consumption(
            QQ, BB, VV, DD, QQbase, preyYY, predYY * (_hden0 / hden), prey, pred, **kw
        )
        _index = np.array(BB, dtype=float, copy=True)
        if _has_ratio is not None:
            _index[_has_ratio] = Bb[_has_ratio] * stanza_ratio[_has_ratio]
        _hden_new = _hden_update(hden, QQ[:, 1:, :].sum(axis=1), _index, _cmco_qb)
        if not params.get("_hden_frozen", False):
            hden[:] = _hden_new
        _consumption(
            QQ,
            BB,
            VV,
            DD,
            QQbase,
            preyYY,
            predYY * (_hden0 / _hden_new),
            prey,
            pred,
            **kw,
        )
    else:
        _consumption(QQ, BB, VV, DD, QQbase, preyYY, predYY, prey, pred, **kw)

    fb = np.asarray(ForcedBio)
    for i in range(1, G):
        if fb[i] > 0:
            BB[:, i] = fb[i]

    FishFrom = _fishing_links(fishing)[0]
    if patch_effort is None:
        FishMort, link_rate = fishing_mortality(params, forcing, fishing)
        FishMort = np.broadcast_to(FishMort, (P, G))
        link_rate = np.broadcast_to(link_rate, (P, len(link_rate)))
    else:
        fm_rows, lr_rows = [], []
        for p in range(P):
            pf = dict(forcing)
            pf["ForcedEffort"] = patch_effort[p]
            fm, lr = fishing_mortality(params, pf, fishing)
            fm_rows.append(fm)
            lr_rows.append(lr)
        FishMort = np.array(fm_rows)
        link_rate = np.array(lr_rows)

    # Primary production (primary_production_forcing, per patch)
    pp_rates = np.zeros((P, G))
    pp_raw = np.asarray(PP_type)
    for i in range(1, min(NL + 1, G)):
        if pp_raw[i] == 1:
            bbi = Bb[:, i]
            has = bbi > 0
            rel_bio = BB[:, i] / np.where(has, bbi, 1.0)
            x = 2.0 - rel_bio
            dd_factor = np.where(x > 0, x, 0.0)  # Python max(0, x), NaN -> 0
            with_dd = PB[i] * BB[:, i] * PPF[:, i] * dd_factor
            no_dd = PB[i] * BB[:, i] * PPF[:, i]
            pp_rates[:, i] = np.where(has, with_dd, no_dd)
    if rpath_pp:
        for i in range(1, NL + 1):
            if PP_type[i] > 0:
                q0 = QQbase[:, 0, i] if QQbase.ndim == 3 else np.full(P, QQbase[0, i])
                m = q0 > 0
                if m.any():
                    pp_rates[m, i] = QQ[m, 0, i] * PPF[m, i]

    _GE_arr = np.zeros(G)
    for gi in range(1, NL + 1):
        if QB[gi] > 0.0:
            _GE_arr[gi] = PB[gi] / QB[gi]
    _GE_base = _GE_arr

    _fr = np.asarray(ForcedRecs, dtype=float).reshape(-1)
    _ra = np.asarray(RecruitmentAdj, dtype=float).reshape(-1)
    _RecruitmentMult = _fr * _ra
    _RecruitmentMult = np.where(
        np.isfinite(_RecruitmentMult) & (_RecruitmentMult > 0.0),
        _RecruitmentMult,
        1.0,
    )
    _far = np.asarray(ForcedActresp, dtype=float).reshape(-1)
    _far = np.where(np.isfinite(_far) & (_far > 0.0), _far, 1.0)
    _M0_effective = M0_arr * _far
    _M0_detritus = _M0_effective
    _mo_pred = params.get("MoPred", None)
    if _mo_pred is not None:
        _mo_pred = np.asarray(_mo_pred, dtype=float)
        if _mo_pred.shape[0] == G and np.any(_mo_pred != 0.0):
            _M0_effective = _M0_effective * (
                1.0 - _mo_pred + _mo_pred * np.asarray(Ftime, dtype=float)
            )
    if ForcedMort is not None:
        _fm = np.asarray(ForcedMort, dtype=float)
        if _fm.shape[0] == G:
            _M0_effective = _M0_effective * _fm
            _M0_detritus = _M0_detritus * _fm
    if not params.get("DETRITUS_MO_UNSCALED", False):
        _M0_detritus = _M0_effective

    # Imported diet
    import_q = np.zeros((P, G))
    q0all = QQbase[:, 0, :] if QQbase.ndim == 3 else np.broadcast_to(QQbase[0], (P, G))
    for i in range(1, NL + 1):
        if PP_type[i] <= 0:
            m = (Bb[:, i] > 0) & (q0all[:, i] > 0)
            if m.any():
                import_q[m, i] = q0all[m, i] / Bb[m, i] * BB[m, i]
                QQ[m, 0, i] = import_q[m, i]

    # Living groups (cumsum: sequential, as the per-patch kernels)
    liv = slice(1, NL + 1)
    if NL > 0:
        consumption = np.cumsum(QQ[:, : NG + 1, liv], axis=1)[:, -1, :]
        predation_loss = np.cumsum(QQ[:, liv, liv], axis=2)[:, :, -1]
        bb = BB[:, liv]
        with np.errstate(all="ignore"):
            production = np.where(
                pp_t[liv] > 0,
                pp_rates[:, liv],
                np.where(QB[liv] > 0.0, _GE_arr[liv] * consumption, PB[liv] * bb),
            )
            production = production * _RecruitmentMult[liv]
            deriv[:, liv] = (
                production
                - predation_loss
                - FishMort[:, liv] * bb
                - _M0_effective[liv] * bb
                + np.asarray(ForcedMigrate, dtype=float)[liv]
            )

    _immigration = params.get("Immigration")
    _emig = params.get("Emig")
    emig_loss = np.zeros((P, G))
    immig_gain = np.zeros((P, G))
    if _emig is not None:
        emig_loss[:, liv] = np.asarray(_emig, dtype=float)[liv] * BB[:, liv]
        deriv[:, liv] -= emig_loss[:, liv]
    if _immigration is not None:
        immig_gain[:, liv] = np.asarray(_immigration, dtype=float)[liv]
        deriv[:, liv] += immig_gain[:, liv]

    rates = None
    if want_rates:
        rates = {}
        food_gain = np.zeros((P, G))
        food_loss = np.zeros((P, G))
        food_gain[:, 1:] = QQ[:, 1:, 1:].sum(axis=1) + import_q[:, 1:]
        food_loss[:, 1:] = QQ[:, 1:, 1 : NL + 1].sum(axis=2)
        mzero_loss = _M0_effective * BB
        fishing_loss = FishMort * BB
        rates["FoodGain"] = food_gain
        rates["FoodLoss"] = food_loss
        rates["MzeroLoss"] = mzero_loss
        rates["FishingLoss"] = fishing_loss
        rates["EmigLoss"] = emig_loss
        rates["LossPropToB"] = food_loss + mzero_loss + fishing_loss + emig_loss
        rates["LossPropToB"][:, 0] = 0.0

    # Detritus (_rpath_detritus_derivs, per patch, links in order)
    n_bio = NL + ND
    det_from = params["DetLinkFrom"]
    det_to = params["DetLinkTo"]
    det_frac = params["DetLinkFrac"]
    fish_to = params.get("FishTo", np.zeros(len(FishFrom)))
    det_passed_prop = params.get("DetPassedProp")
    pass_on = params.get("_detritus_pass_on", True)
    ewe = det_passed_prop is not None
    mzero_l = np.zeros((P, n_bio + 1))
    mzero_l[:, 1:] = _M0_detritus[1 : n_bio + 1] * BB[:, 1 : n_bio + 1]
    cons_d = QQ[:, 1:, 1 : NL + 1].sum(axis=1)
    cons_d = cons_d + import_q[:, 1 : NL + 1]
    unassim_l = np.zeros((P, n_bio + 1))
    unassim_l[:, 1 : NL + 1] = cons_d * Unassim[1 : NL + 1]
    det_gain = np.zeros((P, n_bio + 1))
    det_loss = np.zeros((P, n_bio + 1))
    for k in range(1, len(det_from)):
        src = int(det_from[k])
        dest = int(det_to[k])
        if src < 1 or src > n_bio:
            continue
        received = pass_on or src <= NL or dest == src
        if ewe and src > NL:
            if dest != src and NL < dest <= n_bio and received:
                det_gain[:, dest] += det_passed_prop[src] * det_frac[k] * BB[:, src]
            continue
        flow = float(det_frac[k]) * (mzero_l[:, src] + unassim_l[:, src])
        if 0 < dest <= n_bio and received:
            det_gain[:, dest] += flow
        if src > NL:
            det_loss[:, src] += flow
    fish_gain = np.zeros((P, n_bio + 1))
    for k in range(1, min(len(FishFrom), len(fish_to), link_rate.shape[1])):
        dest = int(fish_to[k])
        src = int(FishFrom[k])
        if ewe and NL < src <= n_bio:
            continue  # EwE6 discards from living groups only
        if NL < dest <= n_bio:
            fish_gain[:, dest] += link_rate[:, k] * BB[:, src]
    gain = det_gain + fish_gain
    dead = slice(NL + 1, n_bio + 1)
    det_input = _det_inflow(params.get("DetInput"), _immigration, NL, ND)
    if det_input is not None:
        gain[:, dead] += np.asarray(det_input, dtype=float)[dead]
    loss = det_loss.copy()
    for d in range(NL + 1, n_bio + 1):
        loss[:, d] += QQ[:, d, 1 : NL + 1].sum(axis=1)
    if not ewe:
        f_dead = np.asarray(FishMort, dtype=float)[:, dead]
        rows = np.any(f_dead != 0.0, axis=1)
        if rows.any():
            loss[rows, dead] += f_dead[rows] * BB[rows, dead]
    if ewe:
        out = np.asarray(params["_det_out"], dtype=float)[: n_bio + 1]
        emig = np.zeros(n_bio + 1)
        if _emig is not None:
            emig[dead] = np.asarray(_emig, dtype=float)[dead]
        loss[:, dead] += (emig[dead] + out[dead]) * BB[:, dead]
    for d in range(NL + 1, n_bio + 1):
        deriv[:, d] = gain[:, d] - loss[:, d]

    if want_rates:
        rates["DetGain"] = np.zeros((P, G))
        rates["DetLoss"] = np.zeros((P, G))
        rates["DetGain"][:, dead] = gain[:, dead]
        rates["DetLoss"][:, dead] = loss[:, dead]
        tot_gain = np.zeros((P, G))
        tot_gain[:, 1 : NL + 1] = (
            QQ[:, 1:, 1 : NL + 1].sum(axis=1) + import_q[:, 1 : NL + 1]
        )
        producers = np.asarray(PP_type[: NL + 1]) > 0
        producers[0] = False
        tg = tot_gain[:, : NL + 1]
        tg[:, producers] = pp_rates[:, : NL + 1][:, producers]
        tot_gain[:, dead] = gain[:, dead]
        food_or_pp = tot_gain.copy()
        tot_gain = tot_gain + immig_gain
        rates["TotGain"] = tot_gain
        rates["TotLoss"] = tot_gain - deriv
        rates["TotLoss"][:, 0] = 0.0
        net_gain = tot_gain.copy()
        net_loss = rates["TotLoss"].copy()
        cons = np.zeros(G, dtype=bool)
        cons[1 : NL + 1] = (~producers[1:]) & (QB[1 : NL + 1] > 0)
        net_gain[:, cons] = _GE_base[cons] * food_or_pp[:, cons] + immig_gain[:, cons]
        net_loss[:, cons] = (
            _GE_arr[cons] * food_or_pp[:, cons] * _RecruitmentMult[cons]
            + immig_gain[:, cons]
            - deriv[:, cons]
        )
        rates["NetGain"] = net_gain
        rates["NetLoss"] = net_loss
    return deriv, rates
