"""Compiled all-patch Ecospace derivative (numba).

:func:`pypath.spatial.batched.deriv_batched` evaluates every patch at once
with NumPy; on a map of a few hundred patches its time goes to NumPy calls
on (patches, groups, groups) arrays and to a per-patch nutrient loop. This
module computes the same derivative patch by patch in one numba-compiled
loop, the patches spread over threads (``prange``).

What changes over a call but not between patches (forcing vectors, the
arena remapping of VV and QQbase, fishing mortality, the M0 scaling,
recruitment multipliers) is prepared in Python with the NumPy operations of
``deriv_batched``. The compiled loop repeats ``deriv_vector``'s per-patch
arithmetic in its order of operations, through the compiled kernels of
:mod:`pypath.core.ecosim_fast` (``_consumption_flat``, the consumption loop
with HandleSwitch and EatEff, and ``_pw_sum``, NumPy's pairwise
summation), which this module calls and does not change. The result is the
per-patch loop's bit for bit, as tests/test_spatial_batched_parity.py
checks, HandleSwitch exponents other than 0 and 1 included (the loop's
scalar power, not NumPy's array power).

It covers what :func:`pypath.spatial.batched.batch_supported` accepts.
Setting ``PYPATH_FAST_ECOSPACE=0`` (or ``false``, ``no``, ``off``) before
import keeps NumPy's ``deriv_batched``.
"""

from __future__ import annotations

import os

import numpy as np

from pypath.core.ecosim_deriv import (
    HAS_NUMBA,
    _det_inflow,
    _fishing_links,
    fishing_mortality,
)
from pypath.spatial.batched import _prey_multiplier_arena

FAST_ECOSPACE_ENV = "PYPATH_FAST_ECOSPACE"


def _requested(environ=os.environ) -> bool:
    value = environ.get(FAST_ECOSPACE_ENV, "").strip().lower()
    return value not in ("0", "false", "no", "off")


#: Whether deriv_vector_spatial uses the compiled batched derivative.
ENABLED = bool(HAS_NUMBA and _requested())

FAST_MONTH_ENV = "PYPATH_FAST_ECOSPACE_MONTH"


#: Whether rsim_run_spatial runs its RK4 months on the compiled derivative
#: directly, with the compiled diffusion (pypath.spatial.diffusion_fast):
#: results agree with the other path to round-off, not bit for bit.
#: ``PYPATH_FAST_ECOSPACE_MONTH=0`` (or false, no, off) keeps that path.
def _month_requested(environ=os.environ) -> bool:
    value = environ.get(FAST_MONTH_ENV, "").strip().lower()
    return value not in ("0", "false", "no", "off")


MONTH_ENABLED = bool(ENABLED and _month_requested())

#: Rate keys the compiled derivative fills, in the order of its rate array.
RATE_KEYS = (
    "FoodGain",
    "FoodLoss",
    "LossPropToB",
    "TotGain",
    "TotLoss",
    "NetGain",
    "NetLoss",
)

if HAS_NUMBA:
    import numba

    from pypath.core.ecosim_fast import (
        _consumption_flat,
        _pw_leaf,
        _pw_sum,
        _row_pw_sum,
    )

    _njit = numba.njit(cache=True, nogil=True, error_model="numpy")
    # ecosim_fast's consumption loop and pairwise leaf, unchanged, compiled
    # into _cells itself (numba inlines them): called as separate functions,
    # once per patch with their many array arguments, they cost a quarter
    # of the time
    _consumption_inl = numba.njit(nogil=True, error_model="numpy", inline="always")(
        _consumption_flat.py_func
    )
    _pw_leaf_inl = numba.njit(nogil=True, error_model="numpy", inline="always")(
        _pw_leaf.py_func
    )
    _njit_par = numba.njit(cache=True, nogil=True, error_model="numpy", parallel=True)
    _prange = numba.prange
else:  # pragma: no cover - the module is not used without numba

    def _njit(f):
        return f

    _consumption_inl = _pw_leaf_inl = None

    _njit_par = _njit
    _prange = range

_D1 = np.zeros(1)
_D2 = np.zeros((1, 1))
_D3 = np.zeros((1, 1, 1))

# int64 scalars and flags (IV)
(
    I_NG,
    I_NL,
    I_ND,
    I_ARENA,
    I_RPP,
    I_NLINKS,
    I_HS,
    I_HDEN,
    I_HDEN_FROZEN,
    I_STANZA,
    I_SEARCH,
    I_NUT,
    I_NUT_SHARED,
    I_EAT,
    I_EMIG,
    I_IMMIG,
    I_EWE,
    I_PASS_ON,
    I_DET_INPUT,
    I_RATES,
    I_BB_CELL,
    I_QQ_CELL,
    I_FISH_CELL,
    I_NUT_CELL,
    I_RATES_LOSS,
    I_N,
) = range(26)

# rows of a thread's workspace (W)
(
    W_BB,
    W_PREY,
    W_PRED,
    W_PDY,
    W_INDEX,
    W_Q,
    W_HNEW,
    W_PPF,
    W_PP,
    W_IMP,
    W_EMIG,
    W_IMMIG,
    W_COLSUM,
    W_MZ,
    W_UN,
    W_DGAIN,
    W_DLOSS,
    W_FGAIN,
    W_GAIN,
    W_LOSS,
    W_TOT,
    W_Y,
    W_D,
    W_KEAT,
    W_H,
    W_R0,
) = range(26)

#: rates in the kernel's rate array, in RATE_KEYS' order (rows W_R0 ...)
N_RATES = 7
N_WORK = W_R0 + N_RATES

# float64 scalars (FV)
F_COUPLED, F_NUT_PROP, F_NUT_BSUM, F_N = range(4)


@_njit
def _nutrient_factor(biomass_sum, base_sum, base_free_prop, shape):
    """ecosim_deriv._ewe_nutrient_factor (Python's max: the first of equals,
    and NaN where it would give NaN)."""
    prop = base_free_prop
    if prop < 0.1:
        prop = 0.1
    nut_tot = base_sum / (1.0 - prop)
    nut_free0 = nut_tot - base_sum
    if nut_free0 <= 0:
        return 1.0
    a = nut_tot * shape - biomass_sum
    b = 0.00101 * nut_free0
    nut_free = b if b > a else a
    return 2.0 * nut_free / (nut_free + nut_free0)


@_njit_par
def _cells(
    IV,
    FV,
    S,
    w,
    Bb,
    safe,
    Ftime,
    FPrey,
    fsearch,
    consumer,
    ratio,
    VV,
    DD,
    QQb,
    lprey,
    lpred,
    HS,
    hden,
    hden0,
    cmco,
    keat,
    eat,
    prod_e,
    fbio,
    FishMort,
    link_rate,
    ppf0,
    producer,
    nut_b,
    nut_f,
    PP_type,
    pp_t,
    PB,
    QB,
    GE,
    RM,
    M0e,
    M0d,
    FMig,
    emig,
    immig,
    unassim,
    det_from,
    det_to,
    det_frac,
    det_passed,
    fish_from,
    fish_to,
    det_out,
    det_input,
    emig_dead,
    deriv_out,
    rates_out,
    n_chunks,
):
    """deriv_batched for every patch p, in deriv_vector's order of
    operations, on group-major arrays (groups x patches) as the spatial run
    keeps them: the patch's density is ``S[:, p] / w[p]`` (``w`` its share
    of the area), and ``deriv_out[:, p]`` and ``rates_out[k, :, p]`` get
    ``w[p]`` times its derivative and rates. ``hden[:, p]`` is updated in
    place (unless frozen); ``ratio``, ``keat`` and ``eat`` are read by
    column too.

    The patches are split into ``n_chunks`` chunks, one per thread (given
    by the caller: numba.get_num_threads here would keep numba from caching
    this function). Each thread has its own
    workspace, ``W`` (N_WORK, G) and ``QQ`` (G, G): every vector of a patch
    is a row of ``W``, set before it is read. The patch's code is written
    out in the loop, not called: a call per patch with this many array
    arguments cost about a third of the time.
    """
    G = S.shape[0]
    P = S.shape[1]
    # the stand-in for the kernel's optional arrays that are not given (a
    # global array would keep numba from caching this function)
    D2 = np.zeros((1, 1))
    if n_chunks < 1:
        n_chunks = 1
    if n_chunks > P:
        n_chunks = P
    for c in _prange(n_chunks):
        W = np.empty((N_WORK, G))
        QQ = np.empty((G, G))
        ci = np.int64(c)
        for p in range(ci * P // n_chunks, (ci + 1) * P // n_chunks):
            NG = IV[I_NG]
            NL = IV[I_NL]
            ND = IV[I_ND]
            G = NG + 1
            nb = NL + ND
            y = W[W_Y]
            wp = w[p]
            for i in range(G):
                y[i] = S[i, p] / wp
            zero = p - p  # 0 in p's type (prange's index can be unsigned)
            pb = p if IV[I_BB_CELL] != 0 else zero
            pq = p if IV[I_QQ_CELL] != 0 else zero
            pf = p if IV[I_FISH_CELL] != 0 else zero
            pn = p if IV[I_NUT_CELL] != 0 else zero
            bb = Bb[pb]
            sf = safe[pb]
            qq = QQb[pq]
            FM = FishMort[pf]
            lr = link_rate[pf]

            BB = W[W_BB]
            for i in range(G):
                BB[i] = y[i]

            # relative prey and predator abundance
            preyYY = W[W_PREY]
            for i in range(G):
                preyYY[i] = 0.0
            arena = IV[I_ARENA] != 0
            for i in range(1, G):
                if bb[i] > 0:
                    if arena:
                        preyYY[i] = BB[i] / sf[i]
                    else:
                        preyYY[i] = Ftime[i] * BB[i] / sf[i] * FPrey[i]
            if IV[I_RPP] != 0 and bb[0] > 0:
                preyYY[0] = Ftime[0] * BB[0] / sf[0] * FPrey[0]
            predYY = W[W_PRED]
            for i in range(G):
                predYY[i] = 0.0
            for i in range(1, NL + 1):
                if bb[i] > 0:
                    predYY[i] = Ftime[i] * BB[i] / sf[i]
            stanza = IV[I_STANZA] != 0
            if stanza:
                rr = ratio[:, p]
                for i in range(G):
                    if np.isfinite(rr[i]):
                        predYY[i] = Ftime[i] * rr[i]
            if IV[I_SEARCH] != 0:
                for i in range(G):
                    if consumer[i] != 0:
                        predYY[i] = predYY[i] * fsearch[i]

            # consumption (EwE handling time: two passes around one Hden step)
            for i in range(G):
                for j in range(G):
                    QQ[i, j] = 0.0
            use_hs = IV[I_HS] != 0
            use_eat = IV[I_EAT] != 0
            ke = W[W_KEAT]  # the patch's column, contiguous for the link loop
            if use_eat:
                for i in range(G):
                    ke[i] = keat[i, p]
            nlinks = IV[I_NLINKS]
            if IV[I_HDEN] != 0:
                h = W[W_H]  # the patch's Hden column; written back below
                for i in range(G):
                    h[i] = hden[i, p]
                pdy = W[W_PDY]
                for i in range(G):
                    pdy[i] = predYY[i] * (hden0[i] / h[i])
                _consumption_inl(
                    QQ,
                    BB,
                    VV,
                    DD,
                    qq,
                    preyYY,
                    pdy,
                    lprey,
                    lpred,
                    nlinks,
                    False,
                    D2,
                    D2,
                    D2,
                    D2,
                    use_hs,
                    HS,
                    FV[F_COUPLED],
                    False,
                    D2,
                    use_eat,
                    ke,
                )
                index = W[W_INDEX]
                for i in range(G):
                    index[i] = BB[i]
                if stanza:
                    rr = ratio[:, p]
                    for i in range(G):
                        if np.isfinite(rr[i]):
                            index[i] = bb[i] * rr[i]
                q = W[W_Q]
                for j in range(G):
                    q[j] = QQ[1, j]
                for r in range(2, G):
                    for j in range(G):
                        q[j] += QQ[r, j]
                hnew = W[W_HNEW]
                for j in range(G):
                    hnew[j] = h[j]
                for j in range(G):
                    if cmco[j] > 0 and index[j] > 0:
                        ht = h[j] * q[j] / (cmco[j] * index[j])
                        hnew[j] = 0.5 * (1.0 + ht) + 0.5 * h[j]
                if IV[I_HDEN_FROZEN] == 0:
                    for j in range(G):
                        h[j] = hnew[j]
                        hden[j, p] = hnew[j]
                for i in range(G):
                    pdy[i] = predYY[i] * (hden0[i] / hnew[i])
                _consumption_inl(
                    QQ,
                    BB,
                    VV,
                    DD,
                    qq,
                    preyYY,
                    pdy,
                    lprey,
                    lpred,
                    nlinks,
                    False,
                    D2,
                    D2,
                    D2,
                    D2,
                    use_hs,
                    HS,
                    FV[F_COUPLED],
                    False,
                    D2,
                    use_eat,
                    ke,
                )
            else:
                _consumption_inl(
                    QQ,
                    BB,
                    VV,
                    DD,
                    qq,
                    preyYY,
                    predYY,
                    lprey,
                    lpred,
                    nlinks,
                    False,
                    D2,
                    D2,
                    D2,
                    D2,
                    use_hs,
                    HS,
                    FV[F_COUPLED],
                    False,
                    D2,
                    use_eat,
                    ke,
                )

            # forced biomass
            for i in range(1, G):
                if fbio[i] > 0:
                    BB[i] = fbio[i]

            # production forcing: nutrient limitation, then EatEff
            ppf = W[W_PPF]
            for i in range(G):
                ppf[i] = ppf0[i]
            if IV[I_NUT] != 0:
                bsum = _pw_sum(y, 1, nb)
                if IV[I_NUT_SHARED] != 0:
                    base_sum = FV[F_NUT_BSUM]
                else:
                    base_sum = _pw_sum(nut_b[pn], 1, nb)
                f = _nutrient_factor(bsum, base_sum, FV[F_NUT_PROP], nut_f[p])
                if f != 1.0:
                    for i in range(G):
                        if producer[i] != 0:
                            ppf[i] *= f
            if use_eat:
                er = eat[:, p]
                for i in range(len(prod_e)):
                    if prod_e[i] != 0:
                        ppf[i] *= er[i]

            # primary production
            pp = W[W_PP]
            for i in range(G):
                pp[i] = 0.0
            top = NL + 1 if NL + 1 < G else G
            for i in range(1, top):
                if PP_type[i] == 1:
                    if bb[i] > 0:
                        x = 2.0 - BB[i] / bb[i]
                        ddf = x if x > 0 else 0.0
                        pp[i] = PB[i] * BB[i] * ppf[i] * ddf
                    else:
                        pp[i] = PB[i] * BB[i] * ppf[i]
            if IV[I_RPP] != 0:
                for i in range(1, NL + 1):
                    if PP_type[i] > 0 and qq[0, i] > 0:
                        pp[i] = QQ[0, i] * ppf[i]

            # imported diet
            imp = W[W_IMP]
            for i in range(G):
                imp[i] = 0.0
            for i in range(1, NL + 1):
                if PP_type[i] <= 0 and bb[i] > 0 and qq[0, i] > 0:
                    imp[i] = qq[0, i] / bb[i] * BB[i]
                    QQ[0, i] = imp[i]

            # living groups (sequential sums, as cumsum)
            deriv = W[W_D]
            for i in range(G):
                deriv[i] = 0.0
            for i in range(1, NL + 1):
                cons = QQ[0, i]
                for prey in range(1, NG + 1):
                    cons += QQ[prey, i]
                ploss = QQ[i, 1]
                for pred in range(2, NL + 1):
                    ploss += QQ[i, pred]
                if pp_t[i] > 0:
                    prod = pp[i]
                elif QB[i] > 0.0:
                    prod = GE[i] * cons
                else:
                    prod = PB[i] * BB[i]
                prod = prod * RM[i]
                deriv[i] = prod - ploss - FM[i] * BB[i] - M0e[i] * BB[i] + FMig[i]
            emig_loss = W[W_EMIG]
            for i in range(G):
                emig_loss[i] = 0.0
            if IV[I_EMIG] != 0:
                for i in range(1, NL + 1):
                    emig_loss[i] = emig[i] * BB[i]
                    deriv[i] -= emig_loss[i]
            immig_gain = W[W_IMMIG]
            for i in range(G):
                immig_gain[i] = 0.0
            if IV[I_IMMIG] != 0:
                for i in range(1, NL + 1):
                    immig_gain[i] = immig[i]
                    deriv[i] += immig_gain[i]

            # detritus (links in order)
            colsum = W[W_COLSUM]  # QQ[1:, :].sum(axis=0), row by row
            for j in range(G):
                colsum[j] = QQ[1, j]
            for r in range(2, G):
                for j in range(G):
                    colsum[j] += QQ[r, j]
            mz = W[W_MZ]
            for i in range(nb + 1):
                mz[i] = 0.0
            for i in range(1, nb + 1):
                mz[i] = M0d[i] * BB[i]
            un = W[W_UN]
            for i in range(nb + 1):
                un[i] = 0.0
            for j in range(1, NL + 1):
                un[j] = (colsum[j] + imp[j]) * unassim[j]
            ewe = IV[I_EWE] != 0
            pass_on = IV[I_PASS_ON] != 0
            det_gain = W[W_DGAIN]
            det_loss = W[W_DLOSS]
            for i in range(nb + 1):
                det_gain[i] = 0.0
                det_loss[i] = 0.0
            for k in range(1, len(det_from)):
                src = det_from[k]
                dest = det_to[k]
                if src < 1 or src > nb:
                    continue
                received = pass_on or src <= NL or dest == src
                if ewe and src > NL:
                    if dest != src and NL < dest <= nb and received:
                        det_gain[dest] += det_passed[src] * det_frac[k] * BB[src]
                    continue
                flow = det_frac[k] * (mz[src] + un[src])
                if 0 < dest <= nb and received:
                    det_gain[dest] += flow
                if src > NL:
                    det_loss[src] += flow
            nfk = len(fish_from)
            if len(fish_to) < nfk:
                nfk = len(fish_to)
            if len(lr) < nfk:
                nfk = len(lr)
            fish_gain = W[W_FGAIN]
            for i in range(nb + 1):
                fish_gain[i] = 0.0
            for k in range(1, nfk):
                dest = fish_to[k]
                src = fish_from[k]
                if ewe and NL < src <= nb:
                    continue
                if NL < dest <= nb:
                    fish_gain[dest] += lr[k] * BB[src]
            gain = W[W_GAIN]
            for i in range(nb + 1):
                gain[i] = det_gain[i] + fish_gain[i]
            if IV[I_DET_INPUT] != 0:
                for d in range(NL + 1, nb + 1):
                    gain[d] += det_input[d]
            loss = W[W_LOSS]
            for i in range(nb + 1):
                loss[i] = det_loss[i]
            for d in range(NL + 1, nb + 1):
                # QQ[d, 1:NL + 1].sum(): ecosim_fast's pairwise leaf, called
                # straight away when it is one block
                loss[d] += (
                    _pw_leaf_inl(QQ[d], 1, NL)
                    if NL <= 128
                    else _row_pw_sum(QQ, d, 1, NL)
                )
            if not ewe:
                any_f = False
                for d in range(NL + 1, nb + 1):
                    if FM[d] != 0.0:
                        any_f = True
                if any_f:
                    for d in range(NL + 1, nb + 1):
                        loss[d] += FM[d] * BB[d]
            else:
                for d in range(NL + 1, nb + 1):
                    loss[d] += (emig_dead[d] + det_out[d]) * BB[d]
            for d in range(NL + 1, nb + 1):
                deriv[d] = gain[d] - loss[d]
            for i in range(G):
                deriv_out[i, p] = wp * deriv[i]

            if IV[I_RATES] == 0:
                continue
            fg = W[W_R0]
            fl = W[W_R0 + 1]
            lp = W[W_R0 + 2]
            tg = W[W_R0 + 3]
            tl = W[W_R0 + 4]
            ng = W[W_R0 + 5]
            nl = W[W_R0 + 6]
            fg[0] = 0.0
            for i in range(1, G):
                fg[i] = colsum[i] + imp[i]
            if IV[I_RATES_LOSS] != 0:  # FoodLoss / LossPropToB asked for
                fl[0] = 0.0
                for i in range(1, G):
                    # QQ[i, 1:NL + 1].sum(): ecosim_fast's pairwise leaf, called
                    # straight away when it is one block
                    fl[i] = (
                        _pw_leaf_inl(QQ[i], 1, NL)
                        if NL <= 128
                        else _row_pw_sum(QQ, i, 1, NL)
                    )
                for i in range(G):
                    lp[i] = fl[i] + M0e[i] * BB[i] + FM[i] * BB[i] + emig_loss[i]
                lp[0] = 0.0
            tot = W[W_TOT]
            for i in range(G):
                tot[i] = 0.0
            for i in range(1, NL + 1):
                tot[i] = colsum[i] + imp[i]
                if PP_type[i] > 0:
                    tot[i] = pp[i]
            for d in range(NL + 1, nb + 1):
                tot[d] = gain[d]
            for i in range(G):
                fop = tot[i]
                t = fop + immig_gain[i]
                tg[i] = t
                tli = t - deriv[i]
                if i == 0:
                    tli = 0.0
                tl[i] = tli
                if 1 <= i <= NL and not (PP_type[i] > 0) and QB[i] > 0:
                    ng[i] = GE[i] * fop + immig_gain[i]
                    nl[i] = GE[i] * fop * RM[i] + immig_gain[i] - deriv[i]
                else:
                    ng[i] = t
                    nl[i] = tli
            loss_rows = IV[I_RATES_LOSS] != 0
            for r in range(N_RATES):
                if (r == 1 or r == 2) and not loss_rows:
                    continue
                row = W[W_R0 + r]
                for i in range(G):
                    rates_out[r, i, p] = row[i] * wp


def _f64(a):
    return np.ascontiguousarray(np.asarray(a, dtype=np.float64))


def _i64(a):
    return np.ascontiguousarray(np.asarray(a, dtype=np.int64))


#: The params keys _prepare reads (fishing_mortality and _det_inflow
#: included): the cache key holds their values, so a change to any of them,
#: in place or not, prepares again. tests/test_spatial_batched_fast.py
#: checks that _prepare reads no other key.
PREP_PARAM_KEYS = (
    "NUM_GROUPS",
    "NUM_LIVING",
    "NUM_DEAD",
    "NUM_GEARS",
    "fleet_idx",
    "PB",
    "QB",
    "VV",
    "DD",
    "Unassim",
    "Bbase",
    "PP_type",
    "NutBaseFreeProp",
    "_nut_base",
    "QQbase",
    "PREY_FTIME_ARENA",
    "RpathPrimaryProduction",
    "_link_prey",
    "_link_pred",
    "HandleSwitch",
    "COUPLED",
    "_hden0",
    "_cmco_qb",
    "M0",
    "MoPred",
    "DETRITUS_MO_UNSCALED",
    "Immigration",
    "Emig",
    "DetLinkFrom",
    "DetLinkTo",
    "DetLinkFrac",
    "FishTo",
    "DetPassedProp",
    "_detritus_pass_on",
    "_det_out",
    "DetInput",
)


class _Unkeyable(Exception):
    """A value the cache key cannot hold: no caching for that call."""


def _key(v):
    """A hashable, exact image of ``v``: arrays by dtype, shape and bytes,
    floats by their bits (NaN equals NaN)."""
    if type(v) is np.ndarray:
        if v.dtype.hasobject:
            raise _Unkeyable
        return (v.dtype.char, v.shape, v.tobytes())
    if v is None or isinstance(v, (bool, int, str)):
        return (type(v).__name__, v)
    if isinstance(v, float):
        return ("f", v.hex())
    if isinstance(v, np.ndarray):  # a subclass
        if v.dtype.hasobject:
            raise _Unkeyable
        return (type(v).__name__, v.dtype.char, v.shape, v.tobytes())
    if isinstance(v, np.generic):
        return (v.dtype.char, v.tobytes())
    if isinstance(v, (list, tuple)):
        return (type(v).__name__,) + tuple(_key(x) for x in v)
    if isinstance(v, dict):
        return ("d",) + tuple((str(k), _key(v[k])) for k in sorted(v, key=str))
    raise _Unkeyable


def _inputs_key(params, forcing, fishing):
    """The cache key of _prepare's inputs: the values of PREP_PARAM_KEYS,
    the whole forcing and the fishing links."""
    get = params.get
    return (
        tuple([_key(get(k)) for k in PREP_PARAM_KEYS]),
        _key(forcing),
        _key(fishing if isinstance(fishing, dict) else _fishing_links(fishing)),
    )


#: The last few preparations, by their inputs' key (most recent last)
_PREP_CACHE: list = []
_PREP_CACHE_SIZE = 4


def _prepare(params, forcing, fishing, P, bbase=None, qqbase=None, patch_effort=None):
    """What the kernel reads besides the state, the Hden, the stanza ratios,
    the EatEff, the nutrient forcing and the rates asked for: the same in
    every call of a month. Returns a dict; ``iv`` is the base of the
    flags, copied by each call."""
    NG = params["NUM_GROUPS"]
    NL = params["NUM_LIVING"]
    ND = params["NUM_DEAD"]
    G = NG + 1
    n_bio = NL + ND

    iv = np.zeros(I_N, dtype=np.int64)
    fv = np.zeros(F_N)
    iv[I_NG], iv[I_NL], iv[I_ND] = NG, NL, ND

    PB = _f64(params["PB"])
    QB = _f64(params.get("QB", np.zeros(G)))
    VV = params["VV"]
    DD = _f64(params["DD"])
    Unassim = _f64(params.get("Unassim", np.zeros(G)))
    Bbase = params.get("Bbase") if bbase is None else bbase
    Bbase = np.asarray(Bbase, dtype=float)
    iv[I_BB_CELL] = int(Bbase.ndim == 2)
    Bb = _f64(Bbase if Bbase.ndim == 2 else Bbase[None, :])
    safe = np.where(Bb > 0, Bb, 1.0)
    PP_type = np.asarray(params.get("PP_type", np.zeros(G, dtype=int)))

    Ftime = _f64(forcing.get("Ftime", np.ones(G)))
    ForcedBio = _f64(forcing.get("ForcedBio", np.zeros(G)))
    PP_forcing = forcing.get("PP_forcing", np.ones(G))
    ppf0 = _f64(np.broadcast_to(np.asarray(PP_forcing, dtype=float), (G,))).copy()

    # nutrient limitation, per patch in the kernel
    producer = (np.asarray(PP_type, dtype=float)[:G] > 0).astype(np.float64)
    producer[0] = 0.0
    nut_b2 = _D2
    _nut_prop = params.get("NutBaseFreeProp")
    if _nut_prop is not None:
        iv[I_NUT] = 1
        fv[F_NUT_PROP] = float(_nut_prop)
        _nut_base = params.get("_nut_base")
        nut_b = Bbase if _nut_base is None else np.asarray(_nut_base, dtype=float)
        if np.ndim(nut_b) == 2:
            nut_b2 = _f64(nut_b)
            iv[I_NUT_CELL] = int(nut_b2.shape[0] > 1)
        else:
            iv[I_NUT_SHARED] = 1
            fv[F_NUT_BSUM] = float(np.sum(nut_b[1 : n_bio + 1]))
    nut_forcing_all = float(forcing.get("NutForcing", 1.0))

    ForcedPrey = _f64(forcing.get("ForcedPrey", np.ones(G)))
    ForcedSearch = forcing.get("ForcedSearch", None)
    ForcedMort = forcing.get("ForcedMort", None)
    ForcedMigrate = forcing.get("ForcedMigrate", np.zeros(G))
    ForcedActresp = forcing.get("ForcedActresp", np.ones(G))
    ForcedRecs = forcing.get("ForcedRecs", np.ones(G))
    RecruitmentAdj = forcing.get("RecruitmentAdj", np.ones(G))

    QQbase = params.get("QQbase", np.zeros((G, G))) if qqbase is None else qqbase
    if params.get("PREY_FTIME_ARENA", False):
        iv[I_ARENA] = 1
        f = np.ones(G)
        f[1:] = Ftime[1:] * ForcedPrey[1:]
        VV, QQbase = _prey_multiplier_arena(VV, QQbase, f)
    VV = _f64(VV)
    QQbase = np.asarray(QQbase, dtype=float)
    iv[I_QQ_CELL] = int(QQbase.ndim == 3)
    QQb = _f64(QQbase if QQbase.ndim == 3 else QQbase[None, :, :])
    iv[I_RPP] = int(bool(params.get("RpathPrimaryProduction", False)))

    fsearch = _D1
    consumer = _D1
    if ForcedSearch is not None:
        _fs = np.asarray(ForcedSearch, dtype=float)
        if _fs.shape[0] == G:
            iv[I_SEARCH] = 1
            fsearch = _f64(_fs)
            c = np.asarray(PP_type, dtype=float)[:G] == 0
            c[0] = False
            consumer = c.astype(np.float64)

    lprey = _i64(params["_link_prey"])
    lpred = _i64(params["_link_pred"])
    iv[I_NLINKS] = len(lprey)
    HS = _D2
    _hs = params.get("HandleSwitch")
    if _hs is not None:
        iv[I_HS] = 1
        HS = _f64(_hs)
    fv[F_COUPLED] = float(params.get("COUPLED", 1))

    hden0 = _D1
    cmco = _D1
    if params.get("_hden0") is not None and params.get("_cmco_qb") is not None:
        hden0 = _f64(params["_hden0"])
        cmco = _f64(np.broadcast_to(params["_cmco_qb"], (G,)))

    FishFrom = _fishing_links(fishing)[0]
    if patch_effort is None:
        FishMort, link_rate = fishing_mortality(params, forcing, fishing)
        FishMort = _f64(FishMort)[None, :]
        link_rate = _f64(link_rate)[None, :]
    else:
        iv[I_FISH_CELL] = 1
        fm_rows, lr_rows = [], []
        for p in range(P):
            pf = dict(forcing)
            pf["ForcedEffort"] = patch_effort[p]
            fm, lr = fishing_mortality(params, pf, fishing)
            fm_rows.append(fm)
            lr_rows.append(lr)
        FishMort = _f64(np.array(fm_rows))
        link_rate = _f64(np.array(lr_rows))

    GE = np.zeros(G)
    for gi in range(1, NL + 1):
        if QB[gi] > 0.0:
            GE[gi] = PB[gi] / QB[gi]
    _fr = np.asarray(ForcedRecs, dtype=float).reshape(-1)
    _ra = np.asarray(RecruitmentAdj, dtype=float).reshape(-1)
    RM = _fr * _ra
    RM = _f64(np.where(np.isfinite(RM) & (RM > 0.0), RM, 1.0))
    _far = np.asarray(ForcedActresp, dtype=float).reshape(-1)
    _far = np.where(np.isfinite(_far) & (_far > 0.0), _far, 1.0)
    M0e = params["M0"] * _far
    M0d = M0e
    _mo_pred = params.get("MoPred", None)
    if _mo_pred is not None:
        _mo_pred = np.asarray(_mo_pred, dtype=float)
        if _mo_pred.shape[0] == G and np.any(_mo_pred != 0.0):
            M0e = M0e * (1.0 - _mo_pred + _mo_pred * np.asarray(Ftime, dtype=float))
    if ForcedMort is not None:
        _fm = np.asarray(ForcedMort, dtype=float)
        if _fm.shape[0] == G:
            M0e = M0e * _fm
            M0d = M0d * _fm
    if not params.get("DETRITUS_MO_UNSCALED", False):
        M0d = M0e
    M0e = _f64(M0e)
    M0d = _f64(M0d)
    FMig = _f64(np.broadcast_to(np.asarray(ForcedMigrate, dtype=float), (G,)))

    emig = _D1
    immig = _D1
    emig_dead = np.zeros(n_bio + 1)
    _immigration = params.get("Immigration")
    _emig = params.get("Emig")
    if _emig is not None:
        iv[I_EMIG] = 1
        emig = _f64(_emig)
        emig_dead[NL + 1 : n_bio + 1] = emig[NL + 1 : n_bio + 1]
    if _immigration is not None:
        iv[I_IMMIG] = 1
        immig = _f64(_immigration)

    det_from = _i64(params["DetLinkFrom"])
    det_to = _i64(params["DetLinkTo"])
    det_frac = _f64(params["DetLinkFrac"])
    fish_from = _i64(FishFrom)
    fish_to = _i64(params.get("FishTo", np.zeros(len(FishFrom))))
    det_passed_prop = params.get("DetPassedProp")
    iv[I_PASS_ON] = int(bool(params.get("_detritus_pass_on", True)))
    det_passed = _D1
    det_out = _D1
    if det_passed_prop is not None:
        iv[I_EWE] = 1
        det_passed = _f64(det_passed_prop)
        det_out = _f64(np.asarray(params["_det_out"], dtype=float)[: n_bio + 1])
    det_input = _det_inflow(params.get("DetInput"), _immigration, NL, ND)
    det_in = _D1
    if det_input is not None:
        iv[I_DET_INPUT] = 1
        det_in = _f64(det_input)

    pe = np.asarray(PP_type, dtype=float)[:G] > 0
    pe[0] = False
    return {
        "iv": iv,
        "fv": fv,
        "G": G,
        "Bb": Bb,
        "safe": safe,
        "Ftime": Ftime,
        "ForcedPrey": ForcedPrey,
        "fsearch": fsearch,
        "consumer": consumer,
        "VV": VV,
        "DD": DD,
        "QQb": QQb,
        "lprey": lprey,
        "lpred": lpred,
        "HS": HS,
        "hden0": hden0,
        "cmco": cmco,
        "ForcedBio": ForcedBio,
        "FishMort": FishMort,
        "link_rate": link_rate,
        "ppf0": ppf0,
        "producer": producer,
        "nut_b2": nut_b2,
        "nut_forcing_all": nut_forcing_all,
        "PP_f64": _f64(PP_type),
        "PP_i64": _i64(PP_type),
        "prod_eat": pe,
        "PB": PB,
        "QB": QB,
        "GE": GE,
        "RM": RM,
        "M0e": M0e,
        "M0d": M0d,
        "FMig": FMig,
        "emig": emig,
        "immig": immig,
        "Unassim": Unassim,
        "det_from": det_from,
        "det_to": det_to,
        "det_frac": det_frac,
        "det_passed": det_passed,
        "fish_from": fish_from,
        "fish_to": fish_to,
        "det_out": det_out,
        "det_in": det_in,
        "emig_dead": emig_dead,
    }


def _prepared(params, forcing, fishing, P, bbase, qqbase, patch_effort):
    """_prepare's result, from the cache when its inputs are the same. Per
    patch base maps and efforts change with every call: not cached."""
    if bbase is not None or qqbase is not None or patch_effort is not None:
        return _prepare(params, forcing, fishing, P, bbase, qqbase, patch_effort)
    try:
        key = _inputs_key(params, forcing, fishing)
    except _Unkeyable:
        return _prepare(params, forcing, fishing, P)
    for entry in _PREP_CACHE:
        if entry[0] == key:
            return entry[1]
    prep = _prepare(params, forcing, fishing, P)
    _PREP_CACHE.append((key, prep))
    del _PREP_CACHE[:-_PREP_CACHE_SIZE]
    return prep


def _rate_keys(want_rates):
    if want_rates is True or want_rates is False or want_rates is None:
        return RATE_KEYS if want_rates else ()
    rate_keys = tuple(want_rates)
    unknown = set(rate_keys) - set(RATE_KEYS)
    if unknown:
        raise KeyError(f"rates not computed here: {sorted(unknown)}")
    return rate_keys


def _c64(a):
    """``a`` as a C-contiguous float64 array, copied only if it is not."""
    if isinstance(a, np.ndarray) and a.dtype == np.float64 and a.flags.c_contiguous:
        return a
    return np.ascontiguousarray(np.asarray(a, dtype=np.float64))


class CellCall:
    """One prepared kernel call: everything but the state and the outputs.

    Made by :func:`spatial_call` (or ``_cell_call``); :meth:`run` evaluates
    the derivative of a state, so a caller that evaluates many states with
    the same inputs (an RK4 month) prepares once. ``hden`` is the caller's
    array, updated in place by each run unless frozen.
    """

    __slots__ = ("iv", "fv", "mid", "rate_keys", "shape", "threads")

    def __init__(self, iv, fv, mid, rate_keys, shape, threads):
        self.iv = iv
        self.fv = fv
        self.mid = mid
        self.rate_keys = rate_keys
        self.shape = shape
        self.threads = threads

    def run(self, S, deriv_out):
        """Fill ``deriv_out`` (G, P) with share x the derivative at ``S``
        (G, P, C-contiguous float64); returns the rate array (N_RATES, G, P),
        rows in RATE_KEYS' order (only the asked-for rows are set)."""
        G, P = self.shape
        rates_arr = np.empty((N_RATES, G, P)) if self.rate_keys else _D3
        k = self.threads
        prev = numba.get_num_threads()
        if k != prev:
            numba.set_num_threads(k)
        try:
            _cells(self.iv, self.fv, S, *self.mid, deriv_out, rates_arr, k)
        finally:
            if k != prev:
                numba.set_num_threads(prev)
        return rates_arr

    def rates(self, rates_arr) -> dict:
        """The asked-for rates of a run's rate array, by key (views)."""
        return {key: rates_arr[RATE_KEYS.index(key)] for key in self.rate_keys}


def _cell_call(
    prep,
    params,
    G,
    P,
    w,
    hden,
    stanza_ratio,
    eat_eff,
    nut_forcing,
    rate_keys,
    n_threads,
):
    """The :class:`CellCall` of group-major (G, P) inputs: ``hden``
    (C-contiguous float64) is updated in place by its runs."""
    iv = prep["iv"].copy()

    nut_f = _D1
    if iv[I_NUT] != 0:
        if nut_forcing is None:
            nut_f = np.full(P, prep["nut_forcing_all"])
        else:
            nut_f = _f64([float(v) for v in nut_forcing])

    keat = _D2
    eat = _D2
    prod_e = _D1
    if eat_eff is not None:
        iv[I_EAT] = 1
        eat = _c64(eat_eff)
        pe = prep["prod_eat"][: eat.shape[0]]
        prod_e = pe.astype(np.float64)
        keat = eat.copy()
        keat[pe, :] = 1.0

    ratio = _D2
    if stanza_ratio is not None:
        iv[I_STANZA] = 1
        ratio = _c64(stanza_ratio)

    hden_arr = _D2
    if hden is not None:
        if prep["hden0"] is _D1 or prep["cmco"] is _D1:
            raise KeyError("handling time (hden) needs params _hden0 and _cmco_qb")
        iv[I_HDEN] = 1
        iv[I_HDEN_FROZEN] = int(bool(params.get("_hden_frozen", False)))
        hden_arr = hden

    iv[I_RATES] = int(bool(rate_keys))
    iv[I_RATES_LOSS] = int("FoodLoss" in rate_keys or "LossPropToB" in rate_keys)
    q = prep
    mid = (
        _c64(w),
        q["Bb"],
        q["safe"],
        q["Ftime"],
        q["ForcedPrey"],
        q["fsearch"],
        q["consumer"],
        ratio,
        q["VV"],
        q["DD"],
        q["QQb"],
        q["lprey"],
        q["lpred"],
        q["HS"],
        hden_arr,
        q["hden0"],
        q["cmco"],
        keat,
        eat,
        prod_e,
        q["ForcedBio"],
        q["FishMort"],
        q["link_rate"],
        q["ppf0"],
        q["producer"],
        q["nut_b2"],
        nut_f,
        q["PP_f64"],
        q["PP_i64"],
        q["PB"],
        q["QB"],
        q["GE"],
        q["RM"],
        q["M0e"],
        q["M0d"],
        q["FMig"],
        q["emig"],
        q["immig"],
        q["Unassim"],
        q["det_from"],
        q["det_to"],
        q["det_frac"],
        q["det_passed"],
        q["fish_from"],
        q["fish_to"],
        q["det_out"],
        q["det_in"],
        q["emig_dead"],
    )
    return CellCall(
        iv, prep["fv"], mid, tuple(rate_keys), (G, P), threads_for(n_threads, P)
    )


def _run_cells(
    prep,
    params,
    S,
    w,
    hden,
    stanza_ratio,
    eat_eff,
    nut_forcing,
    rate_keys,
    deriv_out,
    n_threads,
):
    """Run the kernel on group-major (G, P) arrays (see :class:`CellCall`)."""
    G, P = S.shape
    call = _cell_call(
        prep,
        params,
        G,
        P,
        w,
        hden,
        stanza_ratio,
        eat_eff,
        nut_forcing,
        rate_keys,
        n_threads,
    )
    return call.run(S, deriv_out)


def deriv_batched_fast(
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
    n_threads=1,
):
    """:func:`pypath.spatial.batched.deriv_batched`, compiled.

    Same arguments (patch-major, (P, G)) and results; ``want_rates`` is True
    (``rates`` holds :data:`RATE_KEYS`) or the rate keys to return
    (FoodLoss and LossPropToB, which take most of the rates' time, are
    computed only when asked for); ``hden`` is updated in place.
    ``n_threads`` threads share the patches. The spatial run calls
    :func:`deriv_spatial_fast`, on its own group-major arrays.
    """
    P = S.shape[0]
    if bbase is None and params.get("Bbase") is None:
        bbase = np.array(S, dtype=float)  # deriv_vector's default: the state
    prep = _prepared(params, forcing, fishing, P, bbase, qqbase, patch_effort)
    G = prep["G"]
    rate_keys = _rate_keys(want_rates)
    hden_gp = (
        None if hden is None else np.ascontiguousarray(np.asarray(hden, dtype=float).T)
    )
    deriv = np.empty((G, P))
    rates_arr = _run_cells(
        prep,
        params,
        np.ascontiguousarray(np.asarray(S, dtype=float).T),
        np.ones(P),  # densities as given: x / 1 and 1 * x are x
        hden_gp,
        None if stanza_ratio is None else np.asarray(stanza_ratio, dtype=float).T,
        None if eat_eff is None else np.asarray(eat_eff, dtype=float).T,
        nut_forcing,
        rate_keys,
        deriv,
        n_threads,
    )
    if hden is not None:
        hden[:] = hden_gp.T
    rates = None
    if rate_keys:
        rates = {
            key: np.ascontiguousarray(rates_arr[RATE_KEYS.index(key)].T)
            for key in rate_keys
        }
    return np.ascontiguousarray(deriv.T), rates


def spatial_call(
    state_shape,
    share,
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
    rate_keys=(),
    n_threads=1,
) -> CellCall:
    """The prepared call of :func:`deriv_spatial_fast` (same arguments, the
    state's (G, P) shape for the state): ``call.run(state, deriv_out)``
    evaluates it. ``hden`` must be C-contiguous float64 (it is updated in
    place)."""
    G, P = state_shape
    prep = _prepared(params, forcing, fishing, P, bbase, qqbase, patch_effort)
    rate_keys = _rate_keys(tuple(rate_keys)) if rate_keys else ()
    return _cell_call(
        prep,
        params,
        G,
        P,
        share,
        hden,
        stanza_ratio,
        eat_eff,
        nut_forcing,
        rate_keys,
        n_threads,
    )


def deriv_spatial_fast(
    state,
    share,
    params,
    forcing,
    fishing,
    *,
    deriv_out,
    bbase=None,
    qqbase=None,
    hden=None,
    patch_effort=None,
    stanza_ratio=None,
    eat_eff=None,
    nut_forcing=None,
    rates_out=None,
    n_threads=1,
):
    """The local dynamics of a spatial run's patches, on its own arrays.

    ``state`` (G, P) is the biomass of each group in each patch and
    ``share`` (P,) each patch's share of the area: each patch's derivative
    is evaluated at its density ``state[:, p] / share[p]`` and written,
    times ``share[p]``, into ``deriv_out[:, p]``; ``rates_out`` (a dict of
    (G, P) arrays keyed by :data:`RATE_KEYS`) gets the rates the same way.
    ``hden``, ``stanza_ratio`` and ``eat_eff`` are (G, P) too; ``hden`` is
    updated in place unless it is frozen. ``bbase`` (P, G), ``qqbase``
    (P, G, G) and ``patch_effort`` (P, n) as for :func:`deriv_batched_fast`.
    The arithmetic is :func:`deriv_batched_fast`'s on ``state.T / share``,
    followed by ``share x`` its results, bit for bit.
    """
    hden_arr = None if hden is None else _c64(hden)
    call = spatial_call(
        state.shape,
        share,
        params,
        forcing,
        fishing,
        bbase=bbase,
        qqbase=qqbase,
        hden=hden_arr,
        patch_effort=patch_effort,
        stanza_ratio=stanza_ratio,
        eat_eff=eat_eff,
        nut_forcing=nut_forcing,
        rate_keys=() if not rates_out else tuple(rates_out),
        n_threads=n_threads,
    )
    rates_arr = call.run(_c64(state), deriv_out)
    if hden is not None and hden_arr is not hden:
        hden[:] = hden_arr
    for key in call.rate_keys:
        np.copyto(rates_out[key], rates_arr[RATE_KEYS.index(key)])


def fast_supported(params: dict, eat_eff=None) -> bool:
    """Whether the compiled derivative can run (on top of
    ``batch_supported``): numba present and enabled, and EatEff, when
    given (G, P), one row per group."""
    if not ENABLED:
        return False
    G = params["NUM_GROUPS"] + 1
    if eat_eff is not None and np.shape(eat_eff)[0] != G:
        return False
    return True


#: Patches per thread below which more threads do not pay: a derivative of
#: a few hundred patches takes well under a millisecond, and splitting it
#: finer costs more in thread hand-off than it saves (Black Sea, 304 cells:
#: 4 threads beat 16).
MIN_PATCHES_PER_THREAD = 64


def threads_for(n_threads, n_patches) -> int:
    """Threads to use for ``n_patches``: ``n_threads``, at most one per
    :data:`MIN_PATCHES_PER_THREAD` patches and numba's thread count, at
    least 1."""
    k = min(int(n_threads), int(numba.config.NUMBA_NUM_THREADS))
    k = min(k, max(1, n_patches // MIN_PATCHES_PER_THREAD))
    return max(1, k)
