"""Compiled monthly RK4 step for Ecosim (numba).

``rsim_run(..., method="RK4")`` spends most of its time in the four
``deriv_vector`` calls of each month's RK4 step: about 0.4 ms each, nearly
all of it Python and NumPy call overhead on arrays of a few dozen groups.
This module runs the whole step, the four derivatives and the stages, as
one numba-compiled function.

Within a month nothing but the biomass changes, so everything the
derivative reads from the month's forcing (Ftime, prey and search forcing,
the arena remapping of VV and QQbase, fishing mortality, the M0 scaling,
recruitment multipliers, forced biomass) is prepared once per month in
Python (:meth:`FastRK4.month`), with the same NumPy operations as
``deriv_vector``. The compiled part (:func:`_rk4_month`) repeats
``deriv_vector``'s per-call arithmetic in its order of operations,
NumPy's pairwise summation included, so its results agree with the Python
step to round-off (bit for bit in the cases tested).

The compiled step covers the options of the EwE and Rpath rules that
``rsim_run`` sets up: foraging arena with suite pooling and HandleSwitch,
EwE prey Ftime in the arena, link vulnerability and search forcing, EwE
handling time (Hden), nutrient limitation, Rpath primary production,
multi-stanza predator ratios, search-rate forcing, MoPred, other-mortality
forcing and moTot, imported diet, migration, link-based detritus under both
rules (DetritusOut, DetPassedProp, DF once, pass-on) and fast-equilibrium
groups. :func:`unsupported_reason` names anything else (mediation, prey
switching, IBM groups, Ecospace keys, UseVarPQ, metabolic models,
instrumentation, ...); ``rsim_run`` then keeps the Python step.

Setting the environment variable ``PYPATH_FAST_RK4=0`` (or ``false``,
``no``, ``off``) before import keeps the Python step everywhere.
"""

from __future__ import annotations

import os
from typing import Optional

import numpy as np

from pypath.core.ecosim_deriv import (
    HAS_NUMBA,
    _det_inflow,
    _rk4_fast_mask,
    ewe_link_search_arena,
    ewe_prey_multiplier_arena,
)

FAST_RK4_ENV = "PYPATH_FAST_RK4"


def _requested(environ=os.environ) -> bool:
    value = environ.get(FAST_RK4_ENV, "").strip().lower()
    return value not in ("0", "false", "no", "off")


#: Whether rsim_run uses the compiled RK4 step (read at each run).
ENABLED = bool(HAS_NUMBA and _requested())


def python_hooks_intact(*modules) -> bool:
    """Whether every function of ecosim_deriv, ecosim and stanzas is the one
    each module defined (``_DEFINED_HOOKS``), also where one module binds
    another's (ecosim's deriv_vector, split_update, ...). A replaced
    function (a test's spy, a caller's own derivative) keeps the Python
    path, which calls it; the compiled code would not."""
    import importlib

    mods = [
        importlib.import_module(name)
        for name in (
            "pypath.core.ecosim_deriv",
            "pypath.core.ecosim",
            "pypath.core.stanzas",
        )
    ]
    defined = {}
    for m in mods:
        hooks = getattr(m, "_DEFINED_HOOKS", None)
        if hooks is None:
            return False
        for name, f in hooks.items():
            if m.__dict__.get(name) is not f:
                return False
            defined.setdefault(name, f)
    for m in mods + list(modules):
        own = getattr(m, "_DEFINED_HOOKS", {})
        for name, f in defined.items():
            if name in own:
                continue
            bound = m.__dict__.get(name)
            if bound is not None and bound is not f:
                return False
    return True


if HAS_NUMBA:
    import numba

    _njit = numba.njit(cache=True, nogil=True, error_model="numpy")
else:  # pragma: no cover - the module is not used without numba

    def _njit(f):
        return f


# ---------------------------------------------------------------------------
# Packed inputs: index constants into the tuples handed to the kernel
# ---------------------------------------------------------------------------

# int64 scalars and flags (IV)
(
    I_NG,
    I_NL,
    I_ND,
    I_ARENA,
    I_RPP0,
    I_NLINKS,
    I_POOL,
    I_HS,
    I_HDEN,
    I_HDEN_FROZEN,
    I_NUT,
    I_MOTOT,
    I_MOTOT_GATE,
    I_EMIG,
    I_IMMIG,
    I_EWE,
    I_DETOUT_FRESH,
    I_PASS_ON,
    I_DF_ONCE,
    I_DET_INPUT,
    I_FISH_DEAD,
    I_FAST,
    I_NSTEPS,
    I_PREDSEARCH,
    I_SWITCH,
    I_N,
) = range(26)

# float64 scalars (FV)
(F_COUPLED, F_NUT_BASE, F_NUT_PROP, F_NUT_SHAPE, F_SORWT, F_DT, F_N) = range(7)

# 1-D float64 arrays (A1)
(
    A_BBASE,
    A_SAFE,
    A_FTIME,
    A_FPREY,
    A_STANZA_HAS,
    A_STANZA_PRED,
    A_SEARCH,
    A_HDEN,
    A_HDEN0,
    A_CMCO,
    A_STANZA_INDEX,
    A_FBIO,
    A_FMORT,
    A_PB,
    A_QB,
    A_PPTYPE,
    A_PPF,
    A_PRODUCER,
    A_SUNS,
    A_M0E,
    A_M0D,
    A_MOMAX,
    A_QH,
    A_START_EATEN,
    A_MO_SAFE,
    A_GE,
    A_RECMULT,
    A_IMP_MASK,
    A_IMP_COEF,
    A_FMIGR,
    A_EMIG,
    A_IMMIG,
    A_UNASSIM,
    A_DET_FRAC,
    A_DET_PASSED,
    A_DET_BA,
    A_DET_OUT,
    A_DET_INPUT,
    A_LFR,
    A_FASTMASK,
    A_SWPOW,
    A_SWB0,
    A_N,
) = range(43)

# 2-D float64 arrays (A2)
(
    B_VV,
    B_DD,
    B_QQBASE,
    B_SS,
    B_HSELF,
    B_PPW,
    B_PRPW,
    B_HSW,
    B_VV0,
    B_QQB0,
    B_SWA,
    B_N,
) = range(12)

# 1-D int64 arrays (AI)
(J_LPREY, J_LPRED, J_DFROM, J_DTO, J_FFROM, J_FTO, J_N) = range(7)

_D1 = np.zeros(1)
_D2 = np.zeros((1, 1))


# ---------------------------------------------------------------------------
# Compiled kernels
# ---------------------------------------------------------------------------


@_njit
def _pw_leaf(a, lo, n):
    """NumPy's pairwise sum of a block of at most 128 values."""
    if n < 8:
        res = 0.0
        for i in range(n):
            res += a[lo + i]
        return res
    r0 = a[lo]
    r1 = a[lo + 1]
    r2 = a[lo + 2]
    r3 = a[lo + 3]
    r4 = a[lo + 4]
    r5 = a[lo + 5]
    r6 = a[lo + 6]
    r7 = a[lo + 7]
    m = n - n % 8
    for i in range(8, m, 8):
        j = lo + i
        r0 += a[j]
        r1 += a[j + 1]
        r2 += a[j + 2]
        r3 += a[j + 3]
        r4 += a[j + 4]
        r5 += a[j + 5]
        r6 += a[j + 6]
        r7 += a[j + 7]
    res = ((r0 + r1) + (r2 + r3)) + ((r4 + r5) + (r6 + r7))
    for i in range(m, n):
        res += a[lo + i]
    return res


@_njit
def _pw_sum(a, lo, n):
    """NumPy's pairwise sum of a[lo:lo+n] (contiguous float64), bit for bit.

    Above 128 values NumPy splits the block in two (the first part a
    multiple of 8) and adds the halves' sums. The same tree is walked here
    with an explicit stack: numba's on-disk cache does not reload recursive
    functions reliably.
    """
    if n <= 128:
        return _pw_leaf(a, lo, n)
    f_lo = np.empty(64, np.int64)
    f_n = np.empty(64, np.int64)
    f_stage = np.zeros(64, np.int64)
    f_left = np.zeros(64)
    top = 0
    f_lo[0] = lo
    f_n[0] = n
    f_stage[0] = 0
    ret = 0.0
    have_ret = False
    while True:
        if have_ret:
            if top < 0:
                return ret
            if f_stage[top] == 1:
                f_left[top] = ret
                f_stage[top] = 2
                n2 = f_n[top] // 2
                n2 -= n2 % 8
                have_ret = False
                top += 1
                f_lo[top] = f_lo[top - 1] + n2
                f_n[top] = f_n[top - 1] - n2
                f_stage[top] = 0
            else:
                ret = f_left[top] + ret
                top -= 1
            continue
        m = f_n[top]
        if m <= 128:
            ret = _pw_leaf(a, f_lo[top], m)
            have_ret = True
            top -= 1
            continue
        f_stage[top] = 1
        n2 = m // 2
        n2 -= n2 % 8
        top += 1
        f_lo[top] = f_lo[top - 1]
        f_n[top] = n2
        f_stage[top] = 0


@_njit
def _row_pw_sum(M, r, c0, n):
    """Pairwise sum of M[r, c0:c0+n] (a C-contiguous row)."""
    return _pw_sum(M[r], c0, n)


# A copy of ecosim_deriv._compute_consumption_sparse_flat, compiled here so
# that this module's cached kernels hold no call into another module's
# cache (a cross-module cached call crashed on reload). Keep the two equal:
# tests/test_ecosim_fast.py checks it.
@_njit
def _consumption_flat(
    QQ,
    BB,
    VV,
    DD,
    QQbase,
    preyYY,
    predYY,
    link_prey,
    link_pred,
    n_links,
    use_pooling,
    ScrambleSelf,
    HandleSelf,
    PredPredWeight,
    PreyPreyWeight,
    use_hs,
    HandleSwitch,
    COUPLED,
    use_med,
    med_multipliers,
    use_eat,
    eat_eff,
):
    """:func:`_compute_consumption_sparse_python` in a form numba compiles.

    The same arithmetic in the same order; each optional array comes as a
    flag (``use_pooling``, ``use_hs``, ``use_med``, ``use_eat``) and an
    array, a dummy when the flag is False. Called through
    :func:`_compute_consumption_sparse_numba`.
    """
    n_g = QQ.shape[0]
    PredSuite = np.zeros(n_g)
    HandleSuite = np.zeros(n_g)
    if use_pooling:
        for idx in range(n_links):
            prey = link_prey[idx]
            pred = link_pred[idx]
            pdy = predYY[pred] / eat_eff[pred] if use_eat else predYY[pred]
            PredSuite[prey] += pdy * PredPredWeight[prey, pred]
            HandleSuite[pred] += preyYY[prey] * PreyPreyWeight[prey, pred]

    for idx in range(n_links):
        prey = link_prey[idx]
        pred = link_pred[idx]

        if BB[prey] <= 0.0 or BB[pred] <= 0.0:
            continue

        qbase = QQbase[prey, pred]
        if qbase <= 0.0:
            continue

        PYY = preyYY[prey]
        PDY = predYY[pred]

        vv = VV[prey, pred]
        dd = DD[prey, pred]

        hs_exp = 0.0
        if use_hs:
            hs_exp = HandleSwitch[prey, pred] * COUPLED

        if hs_exp > 0.0:
            pyy_safe = max(PYY, 1e-10)
            PYY_term = pyy_safe**hs_exp
        else:
            PYY_term = PYY

        if use_pooling:
            Hself = HandleSelf[prey, pred]
            dd_prey_eff = (1.0 - Hself) * PYY + Hself * HandleSuite[pred]
        else:
            dd_prey_eff = PYY
        dd_prey_safe = max(dd_prey_eff, 1e-10)
        if hs_exp > 0.0:
            dd_denom = dd_prey_safe**hs_exp
        else:
            dd_denom = dd_prey_safe
        dd_term = dd / max(dd - 1.0 + dd_denom, 1e-10)

        pdy = predYY[pred] / eat_eff[pred] if use_eat else predYY[pred]
        if use_pooling:
            Sself = ScrambleSelf[prey, pred]
            vv_pred_eff = (1.0 - Sself) * pdy + Sself * PredSuite[prey]
        else:
            vv_pred_eff = pdy
        vv_pred_safe = max(vv_pred_eff, 1e-10)
        vv_term = vv / max(vv - 1.0 + vv_pred_safe, 1e-10)

        Q_calc = qbase * PDY * PYY_term * dd_term * vv_term

        if use_med:
            Q_calc *= med_multipliers[prey, pred]

        if Q_calc > 0.0:
            QQ[prey, pred] = Q_calc
        else:
            QQ[prey, pred] = 0.0


@_njit
def _switch(BB, VV, Q0, W, s_all, b0ref):
    """apply_prey_switching (EwE6 SetRelaSwitch as VV' and QQbase'), compiled."""
    n = Q0.shape[0]
    vv_out = VV.copy()
    q_out = Q0.copy()
    b = np.zeros(n)
    for i in range(min(n, len(BB))):
        b[i] = _npmax(BB[i], 0.0)
    b0 = np.zeros(n)
    for i in range(min(n, len(b0ref))):
        b0[i] = b0ref[i]
    now = np.empty(n)
    base = np.empty(n)
    rows = np.empty(n, np.int64)
    for j in range(n):
        if j >= len(s_all):
            break
        sp = s_all[j]
        if not (sp > 0):
            continue
        k = 0
        for i in range(n):
            if W[i, j] > 0:
                rows[k] = i
                k += 1
        if k == 0:
            continue
        for t in range(k):
            i = rows[t]
            now[t] = W[i, j] * b[i] ** sp
            base[t] = W[i, j] * b0[i] ** sp
        sn = _pw_sum(now, 0, k) + 1e-20
        sb = _pw_sum(base, 0, k) + 1e-20
        for t in range(k):
            i = rows[t]
            shn = now[t] / sn
            shb = base[t] / sb
            r = shn / _npmax(shb, 1e-300) if shb > 0 else 1.0
            r = _npmax(r, 1e-20)
            vv = vv_out[i, j]
            vn = 1.0 + (vv - 1.0) / r
            q_out[i, j] = q_out[i, j] * vv / vn
            vv_out[i, j] = vn
    return vv_out, q_out


@_njit
def _consume(QQ, BB, VVc, QQc, A2, AI, IV, FV, preyYY, pdy):
    _consumption_flat(
        QQ,
        BB,
        VVc,
        A2[B_DD],
        QQc,
        preyYY,
        pdy,
        AI[J_LPREY],
        AI[J_LPRED],
        IV[I_NLINKS],
        IV[I_POOL] != 0,
        A2[B_SS],
        A2[B_HSELF],
        A2[B_PPW],
        A2[B_PRPW],
        IV[I_HS] != 0,
        A2[B_HSW],
        FV[F_COUPLED],
        False,
        A2[B_SS],
        False,
        preyYY,
    )


@_njit
def _deriv(y, IV, FV, A1, A2, AI, out_rates):
    """deriv_vector for one state, with the month's inputs packed.

    Returns the derivative; fills ``out_rates`` (5, n) with FoodGain,
    FoodLoss, LossPropToB, NetGain and NetLoss, as deriv_vector's rates.
    """
    NG = IV[I_NG]
    NL = IV[I_NL]
    ND = IV[I_ND]
    n = NG + 1
    nb = NL + ND
    Bbase = A1[A_BBASE]
    safe = A1[A_SAFE]
    Ftime = A1[A_FTIME]
    FPrey = A1[A_FPREY]

    deriv = np.zeros(n)
    BB = y.copy()
    QQ = np.zeros((n, n))

    # relative prey and predator abundance
    preyYY = np.zeros(n)
    arena = IV[I_ARENA] != 0
    for i in range(1, n):
        if Bbase[i] > 0:
            if arena:
                preyYY[i] = BB[i] / safe[i]
            else:
                preyYY[i] = Ftime[i] * BB[i] / safe[i] * FPrey[i]
    if IV[I_RPP0] != 0:
        preyYY[0] = Ftime[0] * BB[0] / safe[0] * FPrey[0]
    predYY = np.zeros(n)
    for i in range(1, NL + 1):
        if Bbase[i] > 0:
            predYY[i] = Ftime[i] * BB[i] / safe[i]
    s_has = A1[A_STANZA_HAS]
    s_pred = A1[A_STANZA_PRED]
    for i in range(n):
        if s_has[i] != 0.0:
            predYY[i] = s_pred[i]
    if IV[I_PREDSEARCH] != 0:
        search = A1[A_SEARCH]
        for i in range(n):
            if search[i] != 1.0:
                predYY[i] = predYY[i] * search[i]

    # the link matrices: prey switching at this biomass, then the arena
    VVc = A2[B_VV]
    QQc = A2[B_QQBASE]
    if IV[I_SWITCH] != 0:
        VVs, QQs = _switch(y, A2[B_VV0], A2[B_QQB0], A2[B_SWA], A1[A_SWPOW], A1[A_SWB0])
        if arena:
            VVc = np.empty_like(VVs)
            QQc = np.empty_like(QQs)
            _arena_remap(VVs, QQs, Ftime, FPrey, VVc, QQc)
        else:
            VVc = VVs
            QQc = QQs

    # consumption (with EwE handling time: two passes around one Hden step)
    if IV[I_HDEN] != 0:
        hden = A1[A_HDEN]
        hden0 = A1[A_HDEN0]
        pdy = np.empty(n)
        for i in range(n):
            pdy[i] = predYY[i] * (hden0[i] / hden[i])
        _consume(QQ, BB, VVc, QQc, A2, AI, IV, FV, preyYY, pdy)
        q = np.zeros(n)
        for j in range(n):
            q[j] = QQ[1, j]
        for r in range(2, n):
            for j in range(n):
                q[j] += QQ[r, j]
        cmco = A1[A_CMCO]
        s_idx = A1[A_STANZA_INDEX]
        hnew = hden.copy()
        for j in range(n):
            idx = s_idx[j] if s_has[j] != 0.0 else BB[j]
            if cmco[j] > 0 and idx > 0:
                ht = hden[j] * q[j] / (cmco[j] * idx)
                hnew[j] = 0.5 * (1.0 + ht) + 0.5 * hden[j]
        if IV[I_HDEN_FROZEN] == 0:
            for j in range(n):
                hden[j] = hnew[j]
        for i in range(n):
            pdy[i] = predYY[i] * (hden0[i] / hnew[i])
        _consume(QQ, BB, VVc, QQc, A2, AI, IV, FV, preyYY, pdy)
    else:
        _consume(QQ, BB, VVc, QQc, A2, AI, IV, FV, preyYY, predYY)

    # forced biomass
    fb = A1[A_FBIO]
    for i in range(1, n):
        if fb[i] > 0:
            BB[i] = fb[i]

    # primary production (nutrient limitation scales the forcing)
    ppf = A1[A_PPF]
    PP_type = A1[A_PPTYPE]
    if IV[I_NUT] != 0:
        bsum = _pw_sum(y, 1, nb)
        base_sum = FV[F_NUT_BASE]
        prop = FV[F_NUT_PROP]
        if prop < 0.1:
            prop = 0.1
        nut_tot = base_sum / (1.0 - prop)
        nut_free0 = nut_tot - base_sum
        nf = 1.0
        if nut_free0 > 0:
            a = nut_tot * FV[F_NUT_SHAPE] - bsum
            b = 0.00101 * nut_free0
            nut_free = b if b > a else a
            nf = 2.0 * nut_free / (nut_free + nut_free0)
        if nf != 1.0:
            ppf = ppf.copy()
            prod_m = A1[A_PRODUCER]
            for i in range(len(ppf)):
                if prod_m[i] != 0.0:
                    ppf[i] *= nf
    PB = A1[A_PB]
    QB = A1[A_QB]
    pp = np.zeros(n)
    top = NL + 1 if NL + 1 < n else n
    for i in range(1, top):
        if PP_type[i] == 1:
            if Bbase[i] > 0:
                rel = BB[i] / Bbase[i]
                ddf = 2.0 - rel
                if not (ddf > 0):
                    ddf = 0.0
                pp[i] = PB[i] * BB[i] * ppf[i] * ddf
            else:
                pp[i] = PB[i] * BB[i] * ppf[i]
    suns = A1[A_SUNS]
    for i in range(1, NL + 1):
        if suns[i] != 0.0:
            pp[i] = QQ[0, i] * ppf[i]

    # other mortality, moTot
    M0e = A1[A_M0E]
    M0d = A1[A_M0D]
    if IV[I_MOTOT] != 0:
        eaten = np.zeros(n)
        for i in range(1, NL + 1):
            eaten[i] = _row_pw_sum(QQ, i, 1, NL)
        start = A1[A_START_EATEN]
        mo = A1[A_MO_SAFE]
        momax = A1[A_MOMAX]
        qh = A1[A_QH]
        fac = np.ones(n)
        for j in range(n):
            if start[j] > 0 and mo[j] != 0:
                fac[j] = (momax[j] / (1.0 + qh[j] * eaten[j] / start[j])) / mo[j]
        M0e = M0e * fac
        if IV[I_MOTOT_GATE] != 0:
            M0d = M0d * fac

    # imported diet
    imp = np.zeros(n)
    imp_m = A1[A_IMP_MASK]
    imp_c = A1[A_IMP_COEF]
    for i in range(1, NL + 1):
        if imp_m[i] != 0.0:
            imp[i] = imp_c[i] * BB[i]
            QQ[0, i] = imp[i]

    # living groups
    FM = A1[A_FMORT]
    GE = A1[A_GE]
    RM = A1[A_RECMULT]
    FMig = A1[A_FMIGR]
    for i in range(1, NL + 1):
        cons = 0.0
        for prey in range(0, NG + 1):
            cons += QQ[prey, i]
        ploss = 0.0
        for pred in range(1, NL + 1):
            ploss += QQ[i, pred]
        if PP_type[i] > 0:
            prod = pp[i]
        elif QB[i] > 0.0:
            prod = GE[i] * cons
        else:
            prod = PB[i] * BB[i]
        prod *= RM[i]
        deriv[i] = prod - ploss - FM[i] * BB[i] - M0e[i] * BB[i] + FMig[i]
    emig = A1[A_EMIG]
    immig = A1[A_IMMIG]
    emig_loss = np.zeros(n)
    if IV[I_EMIG] != 0:
        for i in range(1, NL + 1):
            emig_loss[i] = emig[i] * BB[i]
            deriv[i] -= emig_loss[i]
    immig_gain = np.zeros(n)
    if IV[I_IMMIG] != 0:
        for i in range(1, NL + 1):
            immig_gain[i] = immig[i]
            deriv[i] += immig_gain[i]

    # detritus (Rpath link budget; EwE rule with DetPassedProp)
    det_from = AI[J_DFROM]
    det_to = AI[J_DTO]
    det_frac = A1[A_DET_FRAC]
    unassim = A1[A_UNASSIM]
    mz = np.zeros(nb + 1)
    for i in range(1, nb + 1):
        mz[i] = M0d[i] * BB[i]
    colsum = np.zeros(n)  # QQ[1:, :].sum(axis=0), row by row
    for j in range(n):
        colsum[j] = QQ[1, j]
    for r in range(2, n):
        for j in range(n):
            colsum[j] += QQ[r, j]
    un = np.zeros(nb + 1)
    for j in range(1, NL + 1):
        un[j] = (colsum[j] + imp[j]) * unassim[j]
    ewe = IV[I_EWE] != 0
    df_once = IV[I_DF_ONCE] != 0
    pass_on = IV[I_PASS_ON] != 0
    det_passed = A1[A_DET_PASSED]
    det_gain = np.zeros(nb + 1)
    det_loss = np.zeros(nb + 1)
    other_df = np.zeros(nb + 1)
    if ewe and df_once:
        for k in range(1, len(det_from)):
            src = det_from[k]
            dest = det_to[k]
            if NL < src <= nb and NL < dest <= nb and dest != src:
                other_df[src] += det_frac[k]
    for k in range(1, len(det_from)):
        src = det_from[k]
        dest = det_to[k]
        if src < 1 or src > nb:
            continue
        received = pass_on or src <= NL or dest == src
        if ewe and src > NL:
            if dest != src and NL < dest <= nb and received:
                share = det_frac[k]
                if df_once:
                    share = share / other_df[src] if other_df[src] > 0 else 0.0
                det_gain[dest] += det_passed[src] * share * BB[src]
            continue
        flow = det_frac[k] * (mz[src] + un[src])
        if 0 < dest <= nb and received:
            det_gain[dest] += flow
        if src > NL:
            det_loss[src] += flow
    fish_from = AI[J_FFROM]
    fish_to = AI[J_FTO]
    lfr = A1[A_LFR]
    nfk = len(fish_from)
    if len(fish_to) < nfk:
        nfk = len(fish_to)
    if len(lfr) < nfk:
        nfk = len(lfr)
    fish_gain = np.zeros(nb + 1)
    for k in range(1, nfk):
        dest = fish_to[k]
        src = fish_from[k]
        if ewe and NL < src <= nb:
            continue
        if NL < dest <= nb:
            fish_gain[dest] += lfr[k] * BB[src]
    gain = det_gain + fish_gain
    if IV[I_DET_INPUT] != 0:
        dinp = A1[A_DET_INPUT]
        for d in range(NL + 1, nb + 1):
            gain[d] += dinp[d]
    loss = det_loss.copy()
    for d in range(NL + 1, nb + 1):
        loss[d] += _row_pw_sum(QQ, d, 1, NL)
    if (not ewe) and IV[I_FISH_DEAD] != 0:
        for d in range(NL + 1, nb + 1):
            loss[d] += FM[d] * BB[d]
    if ewe:
        if IV[I_DETOUT_FRESH] != 0:
            det_out = np.full(nb + 1, np.nan)
        else:
            det_out = A1[A_DET_OUT]
        ba = A1[A_DET_BA]
        for d in range(NL + 1, nb + 1):
            em = emig[d] if IV[I_EMIG] != 0 else 0.0
            if np.isnan(det_out[d]):
                if BB[d] > 0:
                    rate = (gain[d] - ba[d] - loss[d]) / BB[d]
                else:
                    rate = 0.0
                v = rate - em
                if v < 1e-20:
                    v = 1e-20
                det_out[d] = v
            loss[d] += (em + det_out[d]) * BB[d]
    for d in range(NL + 1, nb + 1):
        deriv[d] = gain[d] - loss[d]

    # rates: FoodGain, FoodLoss, LossPropToB, NetGain, NetLoss
    fg = out_rates[0]
    fl = out_rates[1]
    lp = out_rates[2]
    ng = out_rates[3]
    nl = out_rates[4]
    for i in range(n):
        fg[i] = 0.0
        fl[i] = 0.0
    for i in range(1, n):
        fg[i] = colsum[i] + imp[i]
        fl[i] = _row_pw_sum(QQ, i, 1, NL)
    for i in range(n):
        lp[i] = fl[i] + M0e[i] * BB[i] + FM[i] * BB[i] + emig_loss[i]
    lp[0] = 0.0
    prod_m = A1[A_PRODUCER]
    tot = np.zeros(n)
    for i in range(1, NL + 1):
        tot[i] = colsum[i] + imp[i]
        if PP_type[i] > 0:
            tot[i] = pp[i]
    for d in range(NL + 1, nb + 1):
        tot[d] = gain[d]
    for i in range(n):
        fop = tot[i]
        t = fop + immig_gain[i]
        tl = t - deriv[i]
        if i == 0:
            tl = 0.0
        if 1 <= i <= NL and not (PP_type[i] > 0) and QB[i] > 0:
            ng[i] = GE[i] * fop + immig_gain[i]
            nl[i] = GE[i] * fop * RM[i] + immig_gain[i] - deriv[i]
        else:
            ng[i] = t
            nl[i] = tl
    return deriv


@_njit
def _fast_eq(y, rates, y_eval, y_base, mask, sorwt):
    """EwE6 fast equilibrium of the masked groups (NetGain / NetLoss)."""
    gain = rates[3]
    loss = rates[4]
    for i in range(len(y)):
        if mask[i] != 0.0:
            out = y_base[i]
            g = gain[i]
            lo = loss[i]
            be = y_eval[i]
            if lo > 0 and be > 0 and np.isfinite(g) and np.isfinite(lo):
                out = (1.0 - sorwt) * (g / (lo / be)) + sorwt * out
            y[i] = out


@_njit
def _rk4_month(state, IV, FV, A1, A2, AI, first, last):
    """One month of integrate_rk4: n_steps RK4 sub-steps of dt / n_steps.

    ``first`` and ``last`` (5, n) receive the rates of the first stage of the
    first sub-step and of the last stage of the last sub-step.
    """
    n = len(state)
    n_steps = IV[I_NSTEPS]
    hh = FV[F_DT] / n_steps
    use_fast = IV[I_FAST] != 0
    mask = A1[A_FASTMASK]
    sorwt = FV[F_SORWT]
    new_state = state.astype(np.float64)
    r = np.zeros((5, n))
    for sub in range(n_steps):
        y0 = new_state
        k1 = _deriv(y0, IV, FV, A1, A2, AI, r)
        if sub == 0:
            first[:, :] = r
        f = 0.5 * hh
        y1 = y0 + f * k1
        if use_fast:
            _fast_eq(y1, r, y0, y0, mask, sorwt)
        k2 = _deriv(y1, IV, FV, A1, A2, AI, r)
        y2 = y0 + f * k2
        if use_fast:
            _fast_eq(y2, r, y1, y0, mask, sorwt)
        k3 = _deriv(y2, IV, FV, A1, A2, AI, r)
        y3 = y0 + hh * k3
        if use_fast:
            _fast_eq(y3, r, y2, y0, mask, sorwt)
        k4 = _deriv(y3, IV, FV, A1, A2, AI, r)
        last[:, :] = r
        new_state = y0 + (hh / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)
        if use_fast:
            _fast_eq(new_state, r, y3, y0, mask, sorwt)
        for i in range(n):
            v = new_state[i]
            if v < 0.0:
                new_state[i] = 0.0
    return new_state


# ---------------------------------------------------------------------------
# Python side: what the compiled step supports, and its inputs
# ---------------------------------------------------------------------------

_FALSY_PARAMS = (
    "_mediation",
    "EweMediation",
    "_eat_eff",
    "_nut_base",
    "_KERNEL_VECTOR",
    "_hden_exact",
    "_hden_solver",
    "INSTRUMENT_GROUPS",
    "instrument_callback",
    "TRACE_DEBUG_GROUPS",
    "UseVarPQ",
    "ibm_groups",
    "VERBOSE_INSTRUMENTATION",
)


def _switching(params) -> bool:
    sp = params.get("SwitchPower")
    return sp is not None and bool(np.any(np.asarray(sp) > 0))


def unsupported_reason(
    params: dict, fishing: dict, scenario_forcing=None
) -> Optional[str]:
    """Why the compiled step cannot run these parameters, or None.

    ``scenario_forcing`` (the scenario's forcing object), when given, is
    checked for link forcing, which the compiled step does not combine with
    prey switching."""
    if not HAS_NUMBA:
        return "numba is not installed"
    if not isinstance(params, dict) or not isinstance(fishing, dict):
        return "params or fishing is not a dict"
    for key in _FALSY_PARAMS:
        if params.get(key):
            return f"params[{key!r}] is set"
    if params.get("MetabolismModel") is not None and (
        params.get("MetabolicTemperature") is not None
        and params.get("MetabolicBodyMass") is not None
    ):
        return "a metabolic model is set"
    if _switching(params) and scenario_forcing is not None:
        for key in ("ForcedSearchLink", "ForcedVulLink"):
            if getattr(scenario_forcing, key, None) is not None:
                return f"prey switching with {key}"
    if params.get("_link_prey") is None or params.get("_link_pred") is None:
        return "no sparse link arrays"
    if params.get("DetLinkFrom") is None:
        return "no detritus links (DetLinkFrom)"
    if params.get("Bbase") is None:
        return "no Bbase"
    for key in ("_hden", "_det_out"):
        arr = params.get(key)
        if arr is not None and not (
            isinstance(arr, np.ndarray)
            and arr.dtype == np.float64
            and arr.flags.c_contiguous
            and arr.ndim == 1
        ):
            return f"params[{key!r}] is not a contiguous float64 vector"
    return None


def _f64(a, n=None, fill=0.0):
    if a is None:
        return np.full(n, fill)
    return np.ascontiguousarray(np.asarray(a, dtype=np.float64))


def _vec(a, n, fill):
    """deriv_vector's handling of an optional per-group forcing vector."""
    if a is None:
        return np.full(n, fill)
    a = np.asarray(a, dtype=float).reshape(-1)
    if a.size != n:
        return np.full(n, fill)
    return a


class FastRK4:
    """The compiled RK4 step for one run: run constants, month inputs."""

    def __init__(self, params: dict, fishing: dict):
        p = params
        NG = int(p["NUM_GROUPS"])
        NL = int(p["NUM_LIVING"])
        ND = int(p["NUM_DEAD"])
        n = NG + 1
        self.params = p
        self.n = n
        self.NL = NL
        self.ND = ND
        Bbase = _f64(p["Bbase"])
        self.Bbase = Bbase
        self.safe = np.where(Bbase > 0, Bbase, 1.0)
        self.PP_type = np.asarray(p.get("PP_type", np.zeros(n, dtype=int)))
        self.PB = _f64(p["PB"])
        self.QB = _f64(p.get("QB", np.zeros(n)))
        self.VV0 = p["VV"]
        self.QQbase0 = p.get("QQbase", np.zeros((n, n)))

        iv = np.zeros(I_N, dtype=np.int64)
        iv[I_NG] = NG
        iv[I_NL] = NL
        iv[I_ND] = ND
        iv[I_ARENA] = int(bool(p.get("PREY_FTIME_ARENA", False)))
        iv[I_RPP0] = int(bool(p.get("RpathPrimaryProduction", False)) and Bbase[0] > 0)
        lp = np.ascontiguousarray(np.asarray(p["_link_prey"], dtype=np.int64))
        lq = np.ascontiguousarray(np.asarray(p["_link_pred"], dtype=np.int64))
        iv[I_NLINKS] = len(lp)
        ss, hs = p.get("ScrambleSelf"), p.get("HandleSelf")
        ppw, prpw = p.get("PredPredWeight"), p.get("PreyPreyWeight")
        pool = (
            ss is not None and hs is not None and ppw is not None and prpw is not None
        )
        iv[I_POOL] = int(pool)
        hsw = p.get("HandleSwitch")
        iv[I_HS] = int(hsw is not None)
        hden = p.get("_hden")
        iv[I_HDEN] = int(hden is not None)
        iv[I_NUT] = int(p.get("NutBaseFreeProp") is not None)
        mo_tot = p.get("_mo_tot")
        iv[I_MOTOT] = int(mo_tot is not None)
        iv[I_MOTOT_GATE] = int(bool(p.get("DETRITUS_MO_TOT", False)))
        emig = p.get("Emig")
        immig = p.get("Immigration")
        iv[I_EMIG] = int(emig is not None)
        iv[I_IMMIG] = int(immig is not None)
        dpp = p.get("DetPassedProp")
        iv[I_EWE] = int(dpp is not None)
        iv[I_DETOUT_FRESH] = int(dpp is not None and p.get("_det_out") is None)
        iv[I_PASS_ON] = int(bool(p.get("_detritus_pass_on", True)))
        iv[I_DF_ONCE] = int(bool(p.get("DETRITUS_DF_ONCE", False)))
        det_input = _det_inflow(p.get("DetInput"), immig, NL, ND)
        iv[I_DET_INPUT] = int(det_input is not None)
        fast_mask = _rk4_fast_mask(p, n)
        iv[I_FAST] = int(bool(np.any(fast_mask)))
        iv[I_SWITCH] = int(_switching(p))
        self.iv = iv

        fv = np.zeros(F_N)
        fv[F_COUPLED] = float(p.get("COUPLED", 1))
        n_bio = NL + ND
        fv[F_NUT_BASE] = float(np.sum(Bbase[1 : n_bio + 1]))
        if p.get("NutBaseFreeProp") is not None:
            fv[F_NUT_PROP] = float(p["NutBaseFreeProp"])
        fv[F_SORWT] = float(p.get("SorWt", 0.5))
        self.fv = fv

        a1 = [None] * A_N
        a1[A_BBASE] = Bbase
        a1[A_SAFE] = self.safe
        a1[A_PB] = self.PB
        a1[A_QB] = self.QB
        a1[A_PPTYPE] = np.asarray(self.PP_type, dtype=np.float64)
        a1[A_HDEN] = hden if hden is not None else _D1
        a1[A_HDEN0] = _f64(p["_hden0"]) if hden is not None else _D1
        a1[A_CMCO] = _f64(p["_cmco_qb"]) if hden is not None else _D1
        if mo_tot is not None:
            mo_max, qh, start = mo_tot
            a1[A_MOMAX] = _f64(mo_max)
            a1[A_QH] = _f64(qh)
            a1[A_START_EATEN] = _f64(start)
        else:
            a1[A_MOMAX] = a1[A_QH] = a1[A_START_EATEN] = _D1
        m0 = p.get("M0")
        self.M0_safe = m0 if isinstance(m0, np.ndarray) else np.zeros(n)
        a1[A_MO_SAFE] = _f64(self.M0_safe)
        GE = np.zeros(n)
        qbl = np.asarray(self.QB)[1 : NL + 1]
        eats = qbl > 0.0
        GE[1 : NL + 1][eats] = np.asarray(self.PB)[1 : NL + 1][eats] / qbl[eats]
        a1[A_GE] = GE
        a1[A_EMIG] = _f64(emig) if emig is not None else np.zeros(n)
        a1[A_IMMIG] = _f64(immig) if immig is not None else np.zeros(n)
        a1[A_UNASSIM] = _f64(p.get("Unassim", np.zeros(n)))
        a1[A_DET_FRAC] = _f64(p["DetLinkFrac"])
        a1[A_DET_PASSED] = _f64(dpp) if dpp is not None else _D1
        a1[A_DET_BA] = (
            _f64(np.asarray(p.get("DetBA"), dtype=float))
            if dpp is not None and p.get("DetBA") is not None
            else np.zeros(n)
        )
        det_out = p.get("_det_out")
        a1[A_DET_OUT] = det_out if det_out is not None else _D1
        a1[A_DET_INPUT] = _f64(det_input) if det_input is not None else _D1
        producer = np.asarray(self.PP_type, dtype=float) > 0
        producer[0] = False
        a1[A_PRODUCER] = producer.astype(np.float64)
        a1[A_FASTMASK] = fast_mask.astype(np.float64)
        sw_a = _D2
        if iv[I_SWITCH]:
            from pypath.core.ecosim_deriv import prey_switch_weights

            sp = np.asarray(p["SwitchPower"], dtype=float)
            s_all = np.full(n, float(sp)) if sp.ndim == 0 else np.zeros(n)
            if sp.ndim:
                m = min(n, len(sp))
                s_all[:m] = sp[:m]
            sw_b = p.get("SwitchBbase", p["Bbase"])
            sw_a = p.get("SwitchA")
            if sw_a is None:
                sw_a = prey_switch_weights(self.VV0, self.QQbase0, sw_b)
            a1[A_SWPOW] = s_all
            a1[A_SWB0] = _f64(sw_b)
        else:
            a1[A_SWPOW] = _D1
            a1[A_SWB0] = _D1
        self.a1 = a1
        self.sw_a = _f64(sw_a)

        dummy2 = _D2
        self.a2_fixed = (
            _f64(p["DD"]),
            _f64(ss) if pool else dummy2,
            _f64(hs) if pool else dummy2,
            _f64(ppw) if pool else dummy2,
            _f64(prpw) if pool else dummy2,
            _f64(hsw) if hsw is not None else dummy2,
            _f64(self.VV0),
            _f64(self.QQbase0),
            self.sw_a,
        )
        fish_from = np.asarray(fishing.get("FishFrom", np.array([0])))
        fish_to = np.asarray(p.get("FishTo", np.zeros(len(fish_from))))
        self.ai = (
            lp,
            lq,
            np.ascontiguousarray(np.asarray(p["DetLinkFrom"]).astype(np.int64)),
            np.ascontiguousarray(np.asarray(p["DetLinkTo"]).astype(np.int64)),
            np.ascontiguousarray(fish_from.astype(np.int64)),
            np.ascontiguousarray(fish_to.astype(np.int64)),
        )
        self.det_input = det_input
        self.first = np.zeros((5, n))
        self.last = np.zeros((5, n))

    def month(self, forcing: dict, fish_mort, link_fish_rate, n_steps, dt):
        """Pack the month's inputs (the same arithmetic as deriv_vector)."""
        p = self.params
        n = self.n
        NL = self.NL
        iv = self.iv.copy()
        fv = self.fv.copy()
        a1 = list(self.a1)
        Ftime = _f64(forcing.get("Ftime", np.ones(n)))
        FPrey = _f64(forcing.get("ForcedPrey", np.ones(n)))
        a1[A_FTIME] = Ftime
        a1[A_FPREY] = FPrey
        sr = forcing.get("StanzaPredRatio")
        s_has = np.zeros(n)
        s_pred = np.zeros(n)
        s_idx = np.zeros(n)
        if sr is not None:
            sr = np.asarray(sr, dtype=float)
            has = np.isfinite(sr)
            s_has[has] = 1.0
            s_pred[has] = Ftime[has] * sr[has]
            s_idx[has] = self.Bbase[has] * sr[has]
        a1[A_STANZA_HAS] = s_has
        a1[A_STANZA_PRED] = s_pred
        a1[A_STANZA_INDEX] = s_idx
        search = np.ones(n)
        fs = forcing.get("ForcedSearch")
        if fs is not None:
            fs = np.asarray(fs, dtype=float)
            if fs.shape[0] == n:
                consumer = np.asarray(self.PP_type, dtype=float)[:n] == 0
                consumer[0] = False
                search[consumer] = fs[consumer]
                iv[I_PREDSEARCH] = 1
        a1[A_SEARCH] = search

        VV, QQbase = self.VV0, self.QQbase0
        if p.get("PREY_FTIME_ARENA", False):
            f_arena = np.ones(n)
            f_arena[1:] = Ftime[1:] * FPrey[1:]
            vul_link = forcing.get("ForcedVulLink")
            if vul_link is not None:
                f_arena = f_arena[:, None] * np.asarray(vul_link, dtype=float)
            VV, QQbase = ewe_prey_multiplier_arena(VV, QQbase, f_arena)
        search_link = forcing.get("ForcedSearchLink")
        if search_link is not None:
            VV, QQbase = ewe_link_search_arena(VV, QQbase, search_link)
        VV = _f64(VV)
        QQbase = _f64(QQbase)
        dd, ss, hs, ppw, prpw, hsw, vv0, qqb0, swa = self.a2_fixed
        a2 = (VV, dd, QQbase, ss, hs, ppw, prpw, hsw, vv0, qqb0, swa)

        a1[A_FBIO] = _f64(forcing.get("ForcedBio", np.zeros(n)))
        a1[A_FMORT] = _f64(fish_mort)
        a1[A_LFR] = _f64(link_fish_rate)
        a1[A_PPF] = _f64(forcing.get("PP_forcing", np.ones(n)))
        if p.get("NutBaseFreeProp") is not None:
            fv[F_NUT_SHAPE] = float(forcing.get("NutForcing", 1.0))
        lv = slice(1, NL + 1)
        pt = np.asarray(self.PP_type)
        suns = np.zeros(n)
        if p.get("RpathPrimaryProduction", False):
            s = (pt[lv] > 0) & (QQbase[0, lv] > 0)
            suns[lv][s] = 1.0
        a1[A_SUNS] = suns

        # other mortality (deriv_vector's sequence of array operations)
        act = _vec(forcing.get("ForcedActresp"), n, 1.0)
        act = np.where(np.isfinite(act) & (act > 0.0), act, 1.0)
        m0e = self.M0_safe * act.copy()
        m0d = m0e
        mo_pred = p.get("MoPred", None)
        if mo_pred is not None:
            mo_pred = np.asarray(mo_pred, dtype=float)
            if mo_pred.shape[0] == n and np.any(mo_pred != 0.0):
                m0e = m0e * (1.0 - mo_pred + mo_pred * Ftime)
        fm = forcing.get("ForcedMort")
        if fm is not None:
            fm = np.asarray(fm, dtype=float)
            if fm.shape[0] == n:
                m0e = m0e * fm
                m0d = m0d * fm
        if not p.get("DETRITUS_MO_UNSCALED", False):
            m0d = m0e
        a1[A_M0E] = _f64(m0e)
        a1[A_M0D] = _f64(m0d)

        rec = _vec(forcing.get("ForcedRecs"), n, 1.0) * _vec(
            forcing.get("RecruitmentAdj"), n, 1.0
        )
        a1[A_RECMULT] = np.where(np.isfinite(rec) & (rec > 0.0), rec, 1.0)

        imp_mask = np.zeros(n)
        imp_coef = np.zeros(n)
        imports = (pt[lv] <= 0) & (self.Bbase[lv] > 0) & (QQbase[0, lv] > 0)
        imp_mask[lv][imports] = 1.0
        imp_coef[lv][imports] = QQbase[0, lv][imports] / self.Bbase[lv][imports]
        a1[A_IMP_MASK] = imp_mask
        a1[A_IMP_COEF] = imp_coef
        a1[A_FMIGR] = _f64(forcing.get("ForcedMigrate", np.zeros(n)))

        fmort = a1[A_FMORT]
        dead = slice(NL + 1, NL + self.ND + 1)
        iv[I_FISH_DEAD] = int(bool(np.any(fmort[dead] != 0.0)))
        iv[I_HDEN_FROZEN] = int(bool(p.get("_hden_frozen", False)))
        iv[I_NSTEPS] = max(1, int(n_steps))
        fv[F_DT] = float(dt)
        # the run's Hden and DetritusOut, updated in place by the kernel
        if iv[I_HDEN]:
            a1[A_HDEN] = p["_hden"]
        if p.get("_det_out") is not None:
            a1[A_DET_OUT] = p["_det_out"]
        a1 = tuple(np.ascontiguousarray(a, dtype=np.float64) for a in a1)
        if iv[I_HDEN] and a1[A_HDEN] is not p["_hden"]:
            raise TypeError("params['_hden'] must be a contiguous float64 vector")
        a2 = tuple(np.ascontiguousarray(a, dtype=np.float64) for a in a2)
        return iv, fv, a1, a2, self.ai

    def step(
        self, state, forcing, fish_mort, link_fish_rate, dt, n_steps=1, stage_rates=None
    ):
        """integrate_rk4 for one month; fills ``stage_rates`` as it does."""
        iv, fv, a1, a2, ai = self.month(forcing, fish_mort, link_fish_rate, n_steps, dt)
        new_state = _rk4_month(
            np.asarray(state, dtype=np.float64),
            iv,
            fv,
            a1,
            a2,
            ai,
            self.first,
            self.last,
        )
        if stage_rates is not None:
            for key, arr in (("first", self.first), ("last", self.last)):
                stage_rates[key] = {
                    "FoodGain": arr[0].copy(),
                    "FoodLoss": arr[1].copy(),
                    "LossPropToB": arr[2].copy(),
                    "NetGain": arr[3].copy(),
                    "NetLoss": arr[4].copy(),
                }
        return new_state


# ===========================================================================
# The whole monthly loop of rsim_run (RK4), compiled
# ===========================================================================
#
# rsim_run's month is the RK4 step plus bookkeeping: the month's forcing,
# forced fishing, the stanza age structure, the feeding-time update, the
# consumption matrix for Qlinks and the outputs. Once the step is compiled,
# that Python bookkeeping is most of the run. ``run_months`` below runs the
# months as one compiled loop, with the same arithmetic as rsim_run's loop
# (``CompiledRun`` prepares the forcing tables in Python, from the scenario
# arrays, as ``_month_forcing`` and ``_forced_fishing_rate`` read them).

EPSILON = 1e-10
CRASH_THRESHOLD = 1e-4

# month tables (MT, 2-D float64, months x n)
(
    T_FB,
    T_FMIG,
    T_FRECS,
    T_FACT,
    T_FPREY,
    T_FSEARCH,
    T_FMORT,
    T_FMGEAR,
    T_LFRGEAR,
    T_FRATE,
    T_FCATCH,
    T_FREPL,
    T_N,
) = range(13)

# run flags (RF, int64)
(
    R_NMONTHS,
    R_HAS_FMORT,
    R_HAS_MOPRED,
    R_HAS_ACT,
    R_NUT,
    R_HAS_FRATE,
    R_HAS_FCATCH,
    R_HAS_FREPL,
    R_NYEARS_FRATE,
    R_NYEARS_FCATCH,
    R_NYEARS_FREPL,
    R_HDEN_START,
    R_DETOUT_RESET,
    R_FTIME_ADJ,
    R_FTIME_START,
    R_RISK,
    R_STANZA,
    R_STANZA_STAGES,
    R_START_YEAR,
    R_FS_Q,
    R_MO_UNSCALED,
    R_N,
) = range(22)

# run 1-D float arrays (RA)
(
    Q_MOPRED,
    Q_CONSUMER,
    Q_BBASEREF,
    Q_FTADJ,
    Q_QBOPT,
    Q_FTMAX,
    Q_NOINT,
    Q_RISKTIME,
    Q_RISK0,
    Q_RISKMO,
    Q_NUT,
    Q_LOWER,
    Q_N,
) = range(13)

# stanza ints (SI1: n_stanzas, rec_stanza, fixed, idx) and 2-D (SI2)
(S_NST, S_REC, S_FIXED, S_IDX, S_I1N) = range(5)
(S_CODE, S_AGE1, S_AGE2, S_I2N) = range(4)
# stanza floats 1-D (SF1)
(
    S_WMAT,
    S_VBM,
    S_SPX,
    S_BSPAWN,
    S_SPEN,
    S_AHAT,
    S_RHAT,
    S_BEGGS,
    S_RSCALE,
    S_RZERO,
    S_RECPOW,
    S_VBGFD,
    S_BPRED,
    S_F1N,
) = range(14)
# stanza floats 2-D (SF2)
(S_ALPHA, S_SPROP, S_EGGS, S_HATCH, S_F2N) = range(5)
# stanza flags (SG)
(G_NSPLIT, G_EWE, G_SPROP, G_FIXED, G_HATCH, G_REC, G_N) = range(7)


@_njit
def _npmax(a, b):
    """np.maximum of two scalars (NaN propagates)."""
    if a != a:
        return a
    if b != b:
        return b
    return a if a >= b else b


@_njit
def _sum_col(M, col, r0, r1, tmp):
    """np.nansum(M[r0:r1, col]): NaN as 0, then NumPy's pairwise sum."""
    k = 0
    for r in range(r0, r1):
        v = M[r, col]
        tmp[k] = 0.0 if v != v else v
        k += 1
    return _pw_sum(tmp, 0, k)


@_njit
def _stanza_update(
    SS2, SS1, SI1, SI2, SF1, SF2, SG, biomass, food_gain, loss, forced_recs, month_index
):
    """stanzas.split_update, compiled (the same arithmetic)."""
    nage = SS2[0]
    wage = SS2[1]
    qage = SS2[2]
    stanza_pred = SS1[0]
    spawn_bio = SS1[2]
    eggs_stanza = SS1[3]
    rec_scalar = SS1[4]
    n_split = SG[G_NSPLIT]
    ewe_plus = SG[G_EWE] != 0
    nst_a = SI1[S_NST]
    code = SI2[S_CODE]
    age1 = SI2[S_AGE1]
    age2 = SI2[S_AGE2]
    tmp = np.empty(nage.shape[0])
    for isp in range(1, n_split + 1):
        nst = nst_a[isp]
        if nst < 1:
            continue
        wmat = SF1[S_WMAT][isp]
        spawn = 0.0
        ieco = 0
        su = 0.0
        for ist in range(1, nst + 1):
            ieco = code[isp, ist]
            a1 = age1[isp, ist]
            a2 = age2[isp, ist]
            b = biomass[ieco]
            su = np.exp(-loss[ieco] / 12.0 / b) if b > 0 else 0.0
            pred = stanza_pred[ieco]
            gf = food_gain[ieco] / pred if pred > 0 else 0.0
            vbm = SF1[S_VBM][isp]
            alpha = SF2[S_ALPHA]
            for a in range(a1, a2 + 1):
                nage[a, isp] *= su
                wage[a, isp] = vbm * wage[a, isp] + gf * alpha[a, isp]
            k = 0
            if SG[G_FIXED] != 0 and SI1[S_FIXED][isp] != 0:
                for a in range(a1, a2 + 1):
                    v = nage[a, isp] * SF2[S_EGGS][a, isp]
                    if SG[G_SPROP] != 0:
                        v = v * SF2[S_SPROP][a, isp]
                    tmp[k] = v
                    k += 1
            else:
                for a in range(a1, a2 + 1):
                    w = wage[a, isp]
                    if w > wmat:
                        v = nage[a, isp] * (w - wmat)
                        if SG[G_SPROP] != 0:
                            v = v * SF2[S_SPROP][a, isp]
                        tmp[k] = v
                        k += 1
            spawn += _pw_sum(tmp, 0, k)
        spawn_bio[isp] = spawn
        spawn_x = SF1[S_SPX][isp]
        base_spawn = SF1[S_BSPAWN][isp]
        egg_mult = 1.0
        if forced_recs.shape[0] > 0 and ieco < forced_recs.shape[0]:
            egg_mult = forced_recs[ieco]
        if ewe_plus:
            eggs = spawn
        elif base_spawn > 0:
            eggs = (
                spawn
                * SF1[S_SPEN][isp]
                * spawn_x
                / (spawn_x - 1.0 + spawn / base_spawn)
            )
        else:
            eggs = 0.0
        if not ewe_plus:
            eggs *= egg_mult
        eggs_stanza[isp] = eggs
        last = age2[isp, nst]
        first = age1[isp, 1]
        nt = nage[last, isp] + nage[last - 1, isp]
        if nt == 0:
            nt = 1e-30
        if ewe_plus:
            wage[last, isp] = (
                su * SF1[S_AHAT][isp] + (1.0 - su) * wage[last - 1, isp]
            ) / (1.0 - SF1[S_RHAT][isp] * su)
        else:
            wage[last, isp] = (
                wage[last, isp] * nage[last, isp]
                + wage[last - 1, isp] * nage[last - 1, isp]
            ) / nt
        nage[last, isp] = nt
        for a in range(last - 1, first, -1):
            nage[a, isp] = nage[a - 1, isp]
            wage[a, isp] = wage[a - 1, isp]
        base_eggs = SF1[S_BEGGS][isp]
        if base_eggs > 0:
            recruits = SF1[S_RSCALE][isp] * SF1[S_RZERO][isp]
            if ewe_plus:
                has_hatch = False
                hv = 0.0
                if SG[G_HATCH] != 0:
                    hatch = SF2[S_HATCH]
                    if 0 <= month_index < hatch.shape[0] and isp < hatch.shape[1]:
                        hv = hatch[month_index, isp]
                        has_hatch = np.isfinite(hv)
                recruits *= egg_mult
                if not has_hatch:
                    recruits *= (eggs / base_eggs) ** SF1[S_RECPOW][isp]
                else:
                    recruits *= hv
            else:
                recruits *= (eggs / base_eggs) ** SF1[S_RECPOW][isp]
            nage[first, isp] = recruits
        wage[first, isp] = 0.0
        if not ewe_plus:
            d = SF1[S_VBGFD][isp]
            for a in range(0, last + 1):
                qage[a, isp] = wage[a, isp] ** d
    if SG[G_REC] != 0:
        rec = SI1[S_REC]
        for isp in range(1, n_split + 1):
            src = rec[isp]
            if not (1 <= src <= n_split):
                continue
            first_src = age1[src, 1]
            first = age1[isp, 1]
            s_from = nage[first_src, src]
            if rec_scalar[isp] == 0 and s_from > 0:
                rec_scalar[isp] = nage[first, isp] / s_from
            nage[first, isp] = s_from * rec_scalar[isp]


@_njit
def _stanza_set_pred(SS2, SS1, SI1, SI2, SG, biomass):
    """stanzas.split_set_pred, compiled."""
    nage = SS2[0]
    wage = SS2[1]
    qage = SS2[2]
    stanza_pred = SS1[0]
    nn = SS1[1]
    code = SI2[S_CODE]
    age1 = SI2[S_AGE1]
    age2 = SI2[S_AGE2]
    tmp = np.empty(nage.shape[0])
    for isp in range(1, SG[G_NSPLIT] + 1):
        for ist in range(1, SI1[S_NST][isp] + 1):
            ieco = code[isp, ist]
            a1 = age1[isp, ist]
            a2 = age2[isp, ist]
            k = 0
            for a in range(a1, a2 + 1):
                tmp[k] = nage[a, isp] * wage[a, isp]
                k += 1
            biomass[ieco] = 1e-30 + _pw_sum(tmp, 0, k)
            k = 0
            for a in range(a1, a2 + 1):
                tmp[k] = nage[a, isp] * qage[a, isp]
                k += 1
            stanza_pred[ieco] = 1e-30 + _pw_sum(tmp, 0, k)
            k = 0
            for a in range(a1, a2 + 1):
                tmp[k] = nage[a, isp]
                k += 1
            nn[ieco] = 1e-30 + _pw_sum(tmp, 0, k)


@_njit
def _q_matrix(
    state,
    IV,
    A1,
    A2,
    ftime,
    fprey,
    fsearch,
    use_fs,
    consumer_q,
    s_has,
    s_pred,
    VVq,
    DDq,
    QQbq,
    qprey,
    qpred,
    QQ,
):
    """rsim_run's _compute_Q_matrix (Qlinks), compiled; QQ is overwritten."""
    if IV[I_SWITCH] != 0:
        VVq, QQbq = _switch(state, VVq, QQbq, A2[B_SWA], A1[A_SWPOW], A1[A_SWB0])
    NG = IV[I_NG]
    NL = IV[I_NL]
    n = NG + 1
    Bbase = A1[A_BBASE]
    safe = A1[A_SAFE]
    PP_type = A1[A_PPTYPE]
    BB = state
    preyYY = np.zeros(n)
    for i in range(1, n):
        if Bbase[i] > 0:
            preyYY[i] = BB[i] / safe[i] * fprey[i]
    predYY = np.zeros(n)
    for i in range(1, NL + 1):
        if Bbase[i] > 0:
            predYY[i] = ftime[i] * BB[i] / safe[i]
    if use_fs:
        for i in range(1, NL + 1):
            if consumer_q[i] != 0.0:
                predYY[i] *= fsearch[i]
    for i in range(n):
        if s_has[i] != 0.0:
            predYY[i] = s_pred[i]
    for i in range(n):
        for j in range(n):
            QQ[i, j] = 0.0
    for k in range(len(qprey)):
        prey = qprey[k]
        pred = qpred[k]
        qbase = QQbq[prey, pred]
        if not (BB[prey] > 0 and BB[pred] > 0 and qbase > 0):
            continue
        vv = VVq[prey, pred]
        dd = DDq[prey, pred]
        PYY = preyYY[prey]
        PDY = predYY[pred]
        dd_term = dd / _npmax(dd - 1.0 + _npmax(PYY, 1e-10), 1e-10)
        vv_term = vv / _npmax(vv - 1.0 + _npmax(PDY, 1e-10), 1e-10)
        q = qbase * PDY * PYY * dd_term * vv_term
        QQ[prey, pred] = _npmax(q, 0.0)
    for pred in range(1, NL + 1):
        if PP_type[pred] <= 0 and Bbase[pred] > 0 and QQbq[0, pred] > 0:
            QQ[0, pred] = QQbq[0, pred] / Bbase[pred] * BB[pred]


@_njit
def _arena_remap(VV0, QQb0, ftime, fprey, VV, QQb):
    """ewe_prey_multiplier_arena for a per-prey f = Ftime x ForcedPrey."""
    n = VV0.shape[0]
    for i in range(n):
        for j in range(VV0.shape[1]):
            VV[i, j] = VV0[i, j]
            QQb[i, j] = QQb0[i, j]
    for r in range(1, n):
        if r >= len(ftime):
            break
        f = ftime[r] * fprey[r]
        if f == 1.0:
            continue
        for j in range(VV0.shape[1]):
            vv_old = VV0[r, j]
            vv_new = 1.0 + (vv_old - 1.0) * (1.0 + f) / 2.0
            if vv_new > 0:
                QQb[r, j] = QQb0[r, j] * vv_old * f / vv_new
            else:
                QQb[r, j] = 0.0
            VV[r, j] = vv_new


@_njit
def _run_months(
    m_first,
    m_last,
    state,
    IV,
    FV,
    A1,
    A2,
    AI,
    MT,
    RF,
    RA,
    A2RUN,
    SS2,
    SS1,
    SI1,
    SI2,
    SF1,
    SF2,
    SG,
    OUT2,
    QL,
    PF,
    crash,
    crashed,
    first,
    last,
):
    """rsim_run's monthly loop (RK4) for months m_first..m_last (1-based).

    ``state`` is the run's biomass at the start of m_first; the month's
    end state is returned. OUT2: out_biomass, out_catch, out_gear_catch,
    out_fishmort, stanza_biomass, out_hden (rows 0..n_months). QL is the
    annual Qlink array; PF: (prey_from, prey_to, fish_from_params, the Q
    matrix's link prey and predator).
    """
    NG = IV[I_NG]
    NL = IV[I_NL]
    ND = IV[I_ND]
    n = NG + 1
    nb = NL + ND
    Bbase = A1[A_BBASE]
    ftime = A1[A_FTIME]
    fprey = A1[A_FPREY]
    fbio = A1[A_FBIO]
    ppf = A1[A_PPF]
    search = A1[A_SEARCH]
    fmig = A1[A_FMIGR]
    recm = A1[A_RECMULT]
    m0e = A1[A_M0E]
    m0d = A1[A_M0D]
    s_has = A1[A_STANZA_HAS]
    s_pred = A1[A_STANZA_PRED]
    s_idx = A1[A_STANZA_INDEX]
    fmort = A1[A_FMORT]
    lfr = A1[A_LFR]
    hden = A1[A_HDEN]
    det_out = A1[A_DET_OUT]
    mo_safe = A1[A_MO_SAFE]
    consumer = RA[Q_CONSUMER]
    mopred = RA[Q_MOPRED]
    VV0 = A2RUN[0]
    QQb0 = A2RUN[1]
    VVq = A2RUN[2]
    DDq = A2RUN[3]
    QQbq = A2RUN[4]
    VVw = A2[B_VV]
    QQbw = A2[B_QQBASE]
    out_b = OUT2[0]
    out_c = OUT2[1]
    out_gc = OUT2[2]
    out_f = OUT2[3]
    out_sb = OUT2[4]
    out_h = OUT2[5]
    prey_from = PF[0]
    prey_to = PF[1]
    pfish_from = PF[2]
    qprey = PF[3]
    qpred = PF[4]
    ffrom = AI[J_FFROM]
    QQm = np.zeros((n, n))
    rates = np.zeros((5, n))
    ratio = np.full(n, np.nan)
    stanza_idx = SI1[S_IDX]
    base_pred = SF1[S_BPRED]
    bref = RA[Q_BBASEREF]
    has_arena = IV[I_ARENA] != 0
    ff_rate = np.zeros(n)
    ff_repl = np.zeros(n)
    tmpc = np.empty(n)
    for month in range(m_first, m_last + 1):
        mi = month - 1
        year_idx = mi // 12
        # ---- the month's forcing (_month_forcing) -------------------------
        for i in range(n):
            fprey[i] = MT[T_FPREY][mi, i]
            fbio[i] = MT[T_FB][mi, i]
            ppf[i] = MT[T_FSEARCH][mi, i]
            fmig[i] = MT[T_FMIG][mi, i]
            search[i] = MT[T_FSEARCH][mi, i] if consumer[i] != 0.0 else 1.0
            r = MT[T_FRECS][mi, i]
            recm[i] = r if (np.isfinite(r) and r > 0.0) else 1.0
        if RF[R_NUT] != 0:
            FV[F_NUT_SHAPE] = RA[Q_NUT][mi]
        # other mortality: M0 x ForcedActresp, MoPred, ForcedMort
        for i in range(n):
            act = 1.0
            if RF[R_HAS_ACT] != 0:
                act = MT[T_FACT][mi, i]
                if not (np.isfinite(act) and act > 0.0):
                    act = 1.0
            base = mo_safe[i] * act
            e = base
            if RF[R_HAS_MOPRED] != 0:
                e = e * (1.0 - mopred[i] + mopred[i] * ftime[i])
            d = base
            if RF[R_HAS_FMORT] != 0:
                fm = MT[T_FMORT][mi, i]
                e = e * fm
                d = d * fm
            m0e[i] = e
            m0d[i] = d if RF[R_MO_UNSCALED] != 0 else e
        # stanza predator ratio, read from the age structure at month start
        if RF[R_STANZA] != 0:
            for i in range(n):
                ratio[i] = np.nan
            for k in range(len(stanza_idx)):
                i = stanza_idx[k]
                if base_pred[i] > 0:
                    ratio[i] = SS1[0][i] / base_pred[i]
            for i in range(n):
                if np.isfinite(ratio[i]):
                    s_has[i] = 1.0
                    s_pred[i] = ftime[i] * ratio[i]
                    s_idx[i] = Bbase[i] * ratio[i]
                else:
                    s_has[i] = 0.0
        # the arena remapping of VV and QQbase by the prey's Ftime x forcing
        if has_arena:
            _arena_remap(VV0, QQb0, ftime, fprey, VVw, QQbw)
        # forced biomass replaces the state
        for i in range(n):
            if fbio[i] > 0:
                state[i] = fbio[i]
        # forced F and catch of the year (_forced_fishing_rate)
        any_ff = False
        if RF[R_HAS_FRATE] + RF[R_HAS_FCATCH] + RF[R_HAS_FREPL] > 0:
            for i in range(n):
                ff_rate[i] = 0.0
                ff_repl[i] = 0.0
            if RF[R_HAS_FRATE] != 0 and year_idx < RF[R_NYEARS_FRATE]:
                for i in range(n):
                    ff_rate[i] += MT[T_FRATE][year_idx, i]
            if RF[R_HAS_FCATCH] != 0 and year_idx < RF[R_NYEARS_FCATCH]:
                for i in range(n):
                    c = MT[T_FCATCH][year_idx, i]
                    b = state[i]
                    cr = 0.0
                    if c > 0 and b > 0:
                        cr = c / b
                        if cr > 3.0:
                            cr = 3.0
                    ff_rate[i] += cr
            if RF[R_HAS_FREPL] != 0 and year_idx < RF[R_NYEARS_FREPL]:
                for i in range(n):
                    ff_repl[i] = MT[T_FREPL][year_idx, i]
            ff_rate[0] = 0.0
            ff_repl[0] = 0.0
            for i in range(n):
                if ff_rate[i] > 0 or ff_repl[i] != 0.0:
                    any_ff = True
        # the month's F (fishing_mortality)
        for i in range(n):
            fmort[i] = MT[T_FMGEAR][mi, i]
        for k in range(len(lfr)):
            lfr[k] = MT[T_LFRGEAR][mi, k]
        if any_ff:
            hi = n if n < nb + 1 else nb + 1
            for grp in range(1, hi):
                if ff_repl[grp] != 0.0:
                    fmort[grp] = 0.0
                    for k in range(1, len(ffrom)):
                        if ffrom[k] == grp:
                            lfr[k] = 0.0
                fmort[grp] += ff_rate[grp]
        IV[I_FISH_DEAD] = 0
        for d in range(NL + 1, nb + 1):
            if fmort[d] != 0.0:
                IV[I_FISH_DEAD] = 1
        month_start = state.copy()
        if month == 1 and RF[R_HDEN_START] != 0:
            IV[I_HDEN_FROZEN] = 0
            _deriv(month_start.copy(), IV, FV, A1, A2, AI, rates)
        if month == 1 and RF[R_DETOUT_RESET] != 0:
            for i in range(len(det_out)):
                det_out[i] = np.nan
        for i in range(n):
            out_f[month, i] = fmort[i]
        ft_start = RF[R_FTIME_ADJ] != 0 and RF[R_FTIME_START] != 0
        ft_index = month_start.copy()
        if ft_start and RF[R_STANZA] != 0:
            for i in range(n):
                if s_has[i] != 0.0:
                    ft_index[i] = bref[i] * ratio[i]
        # ---- the RK4 step -------------------------------------------------
        IV[I_HDEN_FROZEN] = 0
        state = _rk4_month(state, IV, FV, A1, A2, AI, first, last)
        for i in range(n):
            if fbio[i] > 0:
                state[i] = fbio[i]
        for i in range(n):
            v = state[i]
            if not np.isfinite(v):
                v = EPSILON
            if v < EPSILON:
                v = EPSILON
            state[i] = v
        b_integrated = state.copy()
        # ---- multi-stanza update --------------------------------------------
        if RF[R_STANZA] != 0:
            frecs = MT[T_FRECS][mi]
            if RF[R_STANZA_STAGES] != 0:
                _stanza_update(
                    SS2,
                    SS1,
                    SI1,
                    SI2,
                    SF1,
                    SF2,
                    SG,
                    month_start,
                    last[0],
                    first[2],
                    frecs,
                    mi,
                )
            else:
                IV[I_HDEN_FROZEN] = 1
                rs = state.copy()
                _deriv(rs.copy(), IV, FV, A1, A2, AI, rates)
                IV[I_HDEN_FROZEN] = 0
                _stanza_update(
                    SS2,
                    SS1,
                    SI1,
                    SI2,
                    SF1,
                    SF2,
                    SG,
                    rs,
                    rates[0].copy(),
                    rates[2].copy(),
                    frecs,
                    mi,
                )
            _stanza_set_pred(SS2, SS1, SI1, SI2, SG, state)
            for k in range(len(stanza_idx)):
                i = stanza_idx[k]
                out_sb[month, i] = state[i]
        # ---- crash check, outputs -------------------------------------------
        if crash[0] < 0:
            low = False
            for i in range(1, NL + 1):
                if state[i] < CRASH_THRESHOLD:
                    low = True
                    crashed[i] = 1
            if low:
                crash[0] = year_idx + RF[R_START_YEAR]
        for i in range(n):
            out_b[month, i] = state[i]
        if IV[I_HDEN] != 0:
            for i in range(n):
                out_h[month, i] = hden[i]
        _q_matrix(
            state,
            IV,
            A1,
            A2,
            ftime,
            fprey,
            MT[T_FSEARCH][mi],
            RF[R_FS_Q] != 0,
            consumer,
            s_has,
            s_pred,
            VVq,
            DDq,
            QQbq,
            qprey,
            qpred,
            QQm,
        )
        # ---- feeding-time update ------------------------------------------------
        if RF[R_FTIME_ADJ] != 0:
            lower = RA[Q_LOWER][0]
            adj = RA[Q_FTADJ]
            qbopt = RA[Q_QBOPT]
            noint = RA[Q_NOINT]
            rt = RA[Q_RISKTIME]
            r0 = RA[Q_RISK0]
            rmo = RA[Q_RISKMO]
            cap_a = RA[Q_FTMAX]
            end_done = False
            for i in range(1, NL + 1):
                if ft_start and noint[i] != 0:
                    continue
                b_src = ft_index[i] if ft_start else state[i]
                if adj[i] > 0 and b_src > 0 and qbopt[i] > 0:
                    if ft_start:
                        food_gain = first[0][i]
                    else:
                        food_gain = _sum_col(QQm, i, 0, n, tmpc)
                    qopt = qbopt[i]
                    if RF[R_RISK] != 0 and rt[i] > 0:
                        if ft_start:
                            eaten_of = first[1][i]
                            b_risk = b_integrated[i]
                        else:
                            if not end_done:
                                IV[I_HDEN_FROZEN] = 1
                                _deriv(state.copy(), IV, FV, A1, A2, AI, rates)
                                IV[I_HDEN_FROZEN] = 0
                                end_done = True
                            eaten_of = rates[1][i]
                            b_risk = state[i]
                        if rt[i] > 0:
                            if b_risk > 0:
                                rr = eaten_of / b_risk + rmo[i] + 1e-10
                            else:
                                rr = np.inf
                            qopt = (1.0 - rt[i]) * qopt + rt[i] * qopt * r0[i] / rr
                    old = ftime[i]
                    if not (food_gain <= 0 or b_src <= 0 or qopt <= 0):
                        aq = food_gain / b_src
                        new = lower + (1.0 - lower) * old * (
                            (1.0 - adj[i]) + adj[i] * qopt / aq
                        )
                        cap = cap_a[i]
                        ftime[i] = cap if cap < new else new
        # ---- Qlinks and catch -----------------------------------------------
        for li in range(len(prey_from)):
            prey = prey_from[li]
            pred = prey_to[li]
            if prey < n and pred < n:
                QL[year_idx, li] += QQm[prey, pred] / 12.0
        for k in range(1, len(pfish_from)):
            out_gc[month, k] = lfr[k] * state[pfish_from[k]] / 12.0
        for i in range(1, nb + 1):
            out_c[month, i] = fmort[i] * state[i] / 12.0
    return state


# ---------------------------------------------------------------------------
# Python side of the compiled loop
# ---------------------------------------------------------------------------


def _month_rows(arr, n_months, n, fill):
    """Rows 0..n_months-1 of a (months, n) forcing table, ``fill`` past it
    (``_month_forcing``'s defaults), or None when its width is not n."""
    out = np.full((n_months, n), float(fill))
    if arr is None:
        return out
    a = np.asarray(arr, dtype=float)
    if a.ndim != 2 or a.shape[1] != n:
        return None
    k = min(len(a), n_months)
    out[:k] = a[:k]
    return out


def _year_rows(arr, n, transform):
    """A per-year table padded or cut to width n (``_forced_fishing_rate``)."""
    if arr is None:
        return np.zeros((1, n)), 0
    a = np.asarray(arr)
    rows = len(a)
    out = np.zeros((max(rows, 1), n))
    for y in range(rows):
        row = transform(np.asarray(a[y]))
        m = min(len(row), n)
        out[y, :m] = row[:m]
    return out, rows


class CompiledRun:
    """rsim_run's monthly RK4 loop as one compiled call (``_run_months``).

    :meth:`build` returns None when the run needs something the compiled
    loop does not do (rsim_run then keeps its own loop, with the compiled
    step).
    """

    @classmethod
    def build(cls, scenario, setup, fast: FastRK4, rk4_steps, dt, outputs):
        try:
            return cls(scenario, setup, fast, rk4_steps, dt, outputs)
        except _Unsupported:
            return None

    def __init__(self, scenario, setup, fast: FastRK4, rk4_steps, dt, outputs):
        from pypath.core.ecosim_deriv import fishing_mortality

        params = scenario.params
        forcing = scenario.forcing
        fishing_obj = scenario.fishing
        p = setup.params_dict
        fd = setup.fishing_dict
        n = setup.n_groups
        n_months = setup.n_months
        NL = int(params.NUM_LIVING)
        if n != fast.n:
            raise _Unsupported("group count")
        for key in ("ForcedSearchLink", "ForcedVulLink"):
            if getattr(forcing, key, None) is not None:
                raise _Unsupported(key)
        if not isinstance(setup.ftime, np.ndarray) or setup.ftime.dtype != np.float64:
            raise _Unsupported("Ftime")

        # ---- month tables ----------------------------------------------------
        fb = _month_rows(forcing.ForcedBio, n_months, n, 0.0)
        fmig = _month_rows(forcing.ForcedMigrate, n_months, n, 0.0)
        frecs = _month_rows(forcing.ForcedRecs, n_months, n, 1.0)
        fact = _month_rows(forcing.ForcedActresp, n_months, n, 1.0)
        fprey = _month_rows(getattr(forcing, "ForcedPrey", None), n_months, n, 1.0)
        fsearch = _month_rows(getattr(forcing, "ForcedSearch", None), n_months, n, 1.0)
        fmort = _month_rows(getattr(forcing, "ForcedMort", None), n_months, n, 1.0)
        for t in (fb, fmig, frecs, fact, fprey, fsearch, fmort):
            if t is None:
                raise _Unsupported("forcing table width")
        for name, arr in (
            ("ForcedBio", forcing.ForcedBio),
            ("ForcedMigrate", forcing.ForcedMigrate),
            ("ForcedRecs", forcing.ForcedRecs),
            ("ForcedActresp", forcing.ForcedActresp),
        ):
            if len(arr) < n_months:
                raise _Unsupported(f"{name} shorter than the run")
        fb = np.where(fb > 0, fb, 0.0)
        # gear F of each month (fishing_mortality without forced F)
        n_gears = int(params.NUM_GEARS)
        eff = np.asarray(fishing_obj.ForcedEffort, dtype=float)
        fm_gear = np.zeros((n_months, n))
        n_fl = len(fd.get("FishFrom", [0]))
        lfr_gear = np.zeros((n_months, n_fl))
        cache = {}
        for mi in range(n_months):
            e = eff[mi] if mi < len(eff) else np.ones(n_gears + 1)
            key = np.asarray(e, dtype=float).tobytes()
            if key not in cache:
                cache[key] = fishing_mortality(p, {"ForcedEffort": e}, fd)
            fmg, lr = cache[key]
            fm_gear[mi] = fmg
            lfr_gear[mi] = lr
        # forced F and catch, by year
        frate = getattr(fishing_obj, "ForcedFRate", None)
        fcatch = getattr(fishing_obj, "ForcedCatch", None)
        frepl = getattr(fishing_obj, "ForcedFReplace", None)
        if frate is None and fcatch is None:
            frepl = None
        t_frate, ny_frate = _year_rows(
            frate,
            n,
            lambda r: np.maximum(np.nan_to_num(np.asarray(r, dtype=float)), 0.0),
        )
        t_fcatch, ny_fcatch = _year_rows(
            fcatch, n, lambda r: np.nan_to_num(np.asarray(r, dtype=float))
        )
        t_frepl, ny_frepl = _year_rows(
            frepl, n, lambda r: np.asarray(r, dtype=bool).astype(float)
        )
        mt = [None] * T_N
        mt[T_FB] = fb
        mt[T_FMIG] = fmig
        mt[T_FRECS] = frecs
        mt[T_FACT] = fact
        mt[T_FPREY] = fprey
        mt[T_FSEARCH] = fsearch
        mt[T_FMORT] = fmort
        mt[T_FMGEAR] = fm_gear
        mt[T_LFRGEAR] = lfr_gear
        mt[T_FRATE] = t_frate
        mt[T_FCATCH] = t_fcatch
        mt[T_FREPL] = t_frepl
        self.mt = tuple(np.ascontiguousarray(a, dtype=np.float64) for a in mt)

        # ---- run flags and arrays -----------------------------------------------
        rf = np.zeros(R_N, dtype=np.int64)
        rf[R_NMONTHS] = n_months
        rf[R_HAS_FMORT] = 1
        rf[R_HAS_ACT] = 1
        mo_pred = p.get("MoPred", None)
        mo_arr = np.zeros(n)
        if mo_pred is not None:
            mp = np.asarray(mo_pred, dtype=float)
            if mp.shape[0] == n and np.any(mp != 0.0):
                rf[R_HAS_MOPRED] = 1
                mo_arr = mp
        rf[R_NUT] = int(p.get("NutBaseFreeProp") is not None)
        rf[R_HAS_FRATE] = int(frate is not None)
        rf[R_HAS_FCATCH] = int(fcatch is not None)
        rf[R_HAS_FREPL] = int(frepl is not None)
        rf[R_NYEARS_FRATE] = ny_frate
        rf[R_NYEARS_FCATCH] = ny_fcatch
        rf[R_NYEARS_FREPL] = ny_frepl
        rf[R_HDEN_START] = int(setup.handling is not None)
        det_out = p.get("_det_out")
        rf[R_DETOUT_RESET] = int(
            p.get("DetPassedProp") is not None and det_out is not None
        )
        rf[R_FTIME_ADJ] = int(bool(np.any(params.FtimeAdj > 0)))
        rf[R_FTIME_START] = int(bool(getattr(params, "FTIME_START_OF_MONTH", False)))
        rf[R_RISK] = int(setup.risk0 is not None)
        stz = setup.stanza_state is not None
        rf[R_STANZA] = int(stz)
        rf[R_STANZA_STAGES] = int(
            stz and bool(getattr(params, "STANZA_STAGE_RATES", False))
        )
        rf[R_START_YEAR] = int(scenario.start_year)
        rf[R_FS_Q] = 1
        rf[R_MO_UNSCALED] = int(bool(p.get("DETRITUS_MO_UNSCALED", False)))
        self.rf = rf

        pt = np.asarray(fast.PP_type, dtype=float)[:n]
        consumer = (pt == 0).astype(float)
        consumer[0] = 0.0
        ra = [None] * Q_N
        ra[Q_MOPRED] = mo_arr
        ra[Q_CONSUMER] = consumer
        ra[Q_BBASEREF] = np.asarray(params.B_BaseRef, dtype=float)
        ra[Q_FTADJ] = np.asarray(params.FtimeAdj, dtype=float)
        ra[Q_QBOPT] = np.asarray(params.FtimeQBOpt, dtype=float)
        ftmax = getattr(params, "FtimeMax", None)
        ra[Q_FTMAX] = (
            np.full(n, 2.0) if ftmax is None else np.asarray(ftmax, dtype=float)
        )
        ra[Q_NOINT] = np.asarray(params.NoIntegrate, dtype=float)
        rt = getattr(params, "RiskTime", None)
        ra[Q_RISKTIME] = np.zeros(n) if rt is None else np.asarray(rt, dtype=float)
        ra[Q_RISK0] = (
            np.zeros(n) if setup.risk0 is None else np.asarray(setup.risk0, dtype=float)
        )
        ra[Q_RISKMO] = np.asarray(params.MzeroMort, dtype=float)
        nut = np.ones(n_months)
        nf = getattr(forcing, "NutForcing", None)
        if nf is not None:
            k = min(len(nf), n_months)
            nut[:k] = np.asarray(nf, dtype=float)[:k]
        ra[Q_NUT] = nut
        ra[Q_LOWER] = np.array([float(getattr(params, "ForagingTimeLowerLimit", 0.1))])
        for k in (
            Q_BBASEREF,
            Q_FTADJ,
            Q_QBOPT,
            Q_FTMAX,
            Q_NOINT,
            Q_RISKTIME,
            Q_RISK0,
            Q_RISKMO,
        ):
            if ra[k].shape[0] < n:
                raise _Unsupported("parameter vector length")
        self.ra = tuple(np.ascontiguousarray(a, dtype=np.float64) for a in ra)

        # ---- the step's inputs, with this loop's work arrays ---------------------
        iv = fast.iv.copy()
        iv[I_NSTEPS] = max(1, int(rk4_steps))
        iv[I_PREDSEARCH] = 1
        iv[I_HDEN_FROZEN] = 0
        fv = fast.fv.copy()
        fv[F_DT] = float(dt)
        fv[F_NUT_SHAPE] = 1.0
        a1 = list(fast.a1)
        a1[A_FTIME] = setup.ftime
        for k in (
            A_FPREY,
            A_FBIO,
            A_PPF,
            A_SEARCH,
            A_FMIGR,
            A_RECMULT,
            A_M0E,
            A_M0D,
            A_STANZA_HAS,
            A_STANZA_PRED,
            A_STANZA_INDEX,
            A_FMORT,
        ):
            a1[k] = np.zeros(n)
        a1[A_LFR] = np.zeros(n_fl)
        if iv[I_HDEN]:
            a1[A_HDEN] = p["_hden"]
        if det_out is not None:
            a1[A_DET_OUT] = det_out
        QQb0 = _f64(fast.QQbase0)
        lv = slice(1, NL + 1)
        suns = np.zeros(n)
        if p.get("RpathPrimaryProduction", False):
            s = (pt[lv] > 0) & (QQb0[0, lv] > 0)
            suns[lv][s] = 1.0
        a1[A_SUNS] = suns
        imp_mask = np.zeros(n)
        imp_coef = np.zeros(n)
        imports = (pt[lv] <= 0) & (fast.Bbase[lv] > 0) & (QQb0[0, lv] > 0)
        imp_mask[lv][imports] = 1.0
        imp_coef[lv][imports] = QQb0[0, lv][imports] / fast.Bbase[lv][imports]
        a1[A_IMP_MASK] = imp_mask
        a1[A_IMP_COEF] = imp_coef
        a1 = tuple(np.ascontiguousarray(a, dtype=np.float64) for a in a1)
        if iv[I_HDEN] and a1[A_HDEN] is not p["_hden"]:
            raise _Unsupported("_hden layout")
        if det_out is not None and a1[A_DET_OUT] is not det_out:
            raise _Unsupported("_det_out layout")
        if a1[A_FTIME] is not setup.ftime:
            raise _Unsupported("Ftime layout")
        VV0 = _f64(fast.VV0)
        dd, ss, hs, ppw, prpw, hsw, vv0, qqb0, swa = fast.a2_fixed
        a2 = (VV0.copy(), dd, QQb0.copy(), ss, hs, ppw, prpw, hsw, vv0, qqb0, swa)
        self.iv, self.fv, self.a1, self.a2 = iv, fv, a1, a2
        self.ai = fast.ai
        self.fast = fast

        # ---- the Q matrix of Qlinks (_compute_Q_matrix's inputs) ---------------
        active = p.get("ActiveLink")
        if (
            not isinstance(active, np.ndarray)
            or active.shape != (n, n)
            or not active.any()
        ):
            raise _Unsupported("ActiveLink")
        for key in ("VV", "DD", "QQbase"):
            a = p.get(key)
            if not (isinstance(a, np.ndarray) and a.shape == (n, n)):
                raise _Unsupported(key)
        qp, qq = np.nonzero(np.asarray(active[1:, 1 : NL + 1], dtype=bool))
        self.a2run = (
            VV0,
            QQb0,
            _f64(p["VV"]),
            _f64(p["DD"]),
            _f64(p["QQbase"]),
        )
        prey_from = np.asarray(params.PreyFrom, dtype=np.int64)
        prey_to = np.asarray(params.PreyTo, dtype=np.int64)
        self.pf = (
            np.ascontiguousarray(prey_from),
            np.ascontiguousarray(prey_to),
            np.ascontiguousarray(np.asarray(params.FishFrom, dtype=np.int64)),
            np.ascontiguousarray((qp + 1).astype(np.int64)),
            np.ascontiguousarray((qq + 1).astype(np.int64)),
        )

        # ---- stanzas -------------------------------------------------------------
        d2 = np.zeros((1, 1))
        d1 = np.zeros(1)
        di1 = np.zeros(1, dtype=np.int64)
        di2 = np.zeros((1, 1), dtype=np.int64)
        sg = np.zeros(G_N, dtype=np.int64)
        if stz:
            st = setup.stanzas
            ss_ = setup.stanza_state
            for key in ("nage_s", "wage_s", "qage_s"):
                a = getattr(ss_, key)
                if not (a.dtype == np.float64 and a.flags.c_contiguous):
                    setattr(ss_, key, np.ascontiguousarray(a, dtype=np.float64))
            if ss_.rec_scalar is None:
                ss_.rec_scalar = np.zeros(int(st.n_split) + 1)
            for key in ("stanza_pred", "n", "spawn_bio", "eggs_stanza", "rec_scalar"):
                a = getattr(ss_, key)
                if not (a.dtype == np.float64 and a.flags.c_contiguous):
                    setattr(ss_, key, np.ascontiguousarray(a, dtype=np.float64))
            self.ss2 = (ss_.nage_s, ss_.wage_s, ss_.qage_s)
            self.ss1 = (
                ss_.stanza_pred,
                ss_.n,
                ss_.spawn_bio,
                ss_.eggs_stanza,
                ss_.rec_scalar,
            )
            ns = int(st.n_split)
            sg[G_NSPLIT] = ns
            sg[G_EWE] = int(getattr(st, "plus_group_rule", "rpath") == "ewe")
            sprop = getattr(st, "spawn_prop", None)
            sg[G_SPROP] = int(sprop is not None)
            fixed = getattr(st, "fixed_fecundity", None)
            sg[G_FIXED] = int(fixed is not None)
            hatch = getattr(st, "hatch_forcing", None)
            sg[G_HATCH] = int(hatch is not None)
            rec = getattr(st, "rec_stanza", None)
            sg[G_REC] = int(rec is not None)

            def _i(a, fill=-1):
                a = np.asarray(a, dtype=float)
                return np.ascontiguousarray(
                    np.where(np.isfinite(a), a, fill).astype(np.int64)
                )

            self.si1 = (
                _i(st.n_stanzas, 0),
                _i(rec, 0) if rec is not None else di1,
                _i(np.asarray(fixed, dtype=bool).astype(float), 0)
                if fixed is not None
                else di1,
                np.ascontiguousarray(np.asarray(setup.stanza_idx, dtype=np.int64)),
            )
            self.si2 = (_i(st.ecopath_code), _i(st.age1), _i(st.age2))
            bpred = np.zeros(n)
            bp = np.asarray(st.base_stanza_pred, dtype=float)
            k = min(len(bp), n)
            bpred[:k] = bp[:k]
            sf1 = [
                st.wmat,
                st.vbm,
                st.spawn_x,
                st.base_spawn_bio,
                st.spawn_energy,
                st.ahat,
                st.rhat,
                st.base_eggs_stanza,
                st.r_scale_split,
                st.r_zero_s,
                st.rec_power,
                st.vbgf_d,
                bpred,
            ]
            self.sf1 = tuple(
                np.ascontiguousarray(np.asarray(a, dtype=np.float64)) for a in sf1
            )
            self.sf2 = (
                _f64(st.split_alpha),
                _f64(sprop) if sprop is not None else d2,
                _f64(st.eggs_split)
                if (fixed is not None and getattr(st, "eggs_split", None) is not None)
                else d2,
                _f64(hatch) if hatch is not None else d2,
            )
            if sg[G_FIXED] and self.sf2[2] is d2:
                raise _Unsupported("fixed fecundity without eggs_split")
            self.stanza_idx = np.asarray(setup.stanza_idx, dtype=np.int64)
        else:
            self.ss2 = (d2.copy(), d2.copy(), d2.copy())
            self.ss1 = (np.zeros(n), np.zeros(n), d1.copy(), d1.copy(), d1.copy())
            self.si1 = (di1, di1, di1, np.zeros(0, dtype=np.int64))
            self.si2 = (di2, di2, di2)
            self.sf1 = tuple(np.zeros(n) for _ in range(S_F1N))
            self.sf2 = (d2, d2, d2, d2)
        self.sg = sg

        # ---- outputs -------------------------------------------------------------
        (out_b, out_c, out_gc, out_f, out_sb, out_h, qlink) = outputs
        self.out2 = (
            out_b,
            out_c,
            out_gc,
            out_f,
            out_sb if out_sb is not None else d2.copy(),
            out_h if out_h is not None else d2.copy(),
        )
        for a in self.out2:
            if not (a.dtype == np.float64 and a.flags.c_contiguous):
                raise _Unsupported("output layout")
        self.qlink = (
            qlink if qlink is not None else np.zeros((max(setup.n_years, 1), 0))
        )
        self.crash = np.array([-1], dtype=np.int64)
        self.crashed = np.zeros(n, dtype=np.int64)

    def run(self, state, m_first, m_last):
        """Months m_first..m_last; returns the state at the end of m_last."""
        return _run_months(
            m_first,
            m_last,
            np.array(state, dtype=np.float64, copy=True),
            self.iv,
            self.fv,
            self.a1,
            self.a2,
            self.ai,
            self.mt,
            self.rf,
            self.ra,
            self.a2run,
            self.ss2,
            self.ss1,
            self.si1,
            self.si2,
            self.sf1,
            self.sf2,
            self.sg,
            self.out2,
            self.qlink,
            self.pf,
            self.crash,
            self.crashed,
            self.fast.first,
            self.fast.last,
        )

    def crash_info(self):
        year = int(self.crash[0])
        groups = {int(i) for i in np.flatnonzero(self.crashed)}
        return year, groups


class _Unsupported(Exception):
    pass
