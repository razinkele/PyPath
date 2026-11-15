"""Feeding time as a state of the continuous pathway.

:class:`FeedingTime` (built by :func:`feeding_time`) holds the feeding-time
groups and their parameters; :func:`ftime_target` is the monthly rule's
target that the feeding-time derivative relaxes to.
"""

from dataclasses import dataclass
from typing import Optional

import numpy as np


@dataclass(frozen=True)
class FeedingTime:
    """The groups whose feeding time is a state of the continuous path.

    ``groups`` are the living groups with ``FtimeAdj > 0``. The other arrays
    are full length (index 0 = Outside): ``adj`` (FtimeAdj), ``qbopt``
    (FtimeQBOpt), ``cap`` (FtimeMax, 2.0 when the scenario has none),
    ``risk_time`` and ``risk0`` (RiskTime and its base predation risk; zeros
    when RiskTime is off) and ``mo`` (MzeroMort). ``lower`` is
    ForagingTimeLowerLimit.
    """

    groups: np.ndarray
    adj: np.ndarray
    qbopt: np.ndarray
    cap: np.ndarray
    risk_time: np.ndarray
    risk0: np.ndarray
    mo: np.ndarray
    lower: float


def feeding_time(params, risk0: Optional[np.ndarray]) -> Optional[FeedingTime]:
    """The feeding-time states of a scenario's ``params``, or None.

    ``risk0`` is the setup's base predation risk (None when RiskTime is
    off). None is returned when no living group has ``FtimeAdj > 0``.
    """
    adj = np.asarray(params.FtimeAdj, dtype=float)
    n_full = len(adj)
    living = np.zeros(n_full, dtype=bool)
    living[1 : int(params.NUM_LIVING) + 1] = True
    groups = np.flatnonzero(living & (adj > 0))
    if groups.size == 0:
        return None
    cap_src = getattr(params, "FtimeMax", None)
    cap = (
        np.full(n_full, 2.0)
        if cap_src is None
        else np.array(cap_src, dtype=float, copy=True)
    )
    if risk0 is None:
        risk_time = np.zeros(n_full)
        risk = np.zeros(n_full)
    else:
        risk_time = np.array(params.RiskTime, dtype=float, copy=True)
        risk = np.array(risk0, dtype=float, copy=True)
    return FeedingTime(
        groups=groups,
        adj=adj.copy(),
        qbopt=np.array(params.FtimeQBOpt, dtype=float, copy=True),
        cap=cap,
        risk_time=risk_time,
        risk0=risk,
        mo=np.array(params.MzeroMort, dtype=float, copy=True),
        lower=float(getattr(params, "ForagingTimeLowerLimit", 0.1)),
    )


def ftime_target(ft, adj, qopt, food_gain, index, lower, cap) -> np.ndarray:
    """``_ftime_update_rpath`` over arrays, clipped to ``[lower, cap]``.

    ``lower + (1 - lower) * ft * ((1 - adj) + adj * qopt / (food_gain /
    index))``, in the scalar rule's order of operations. Where ``food_gain``,
    ``index`` or ``qopt`` is <= 0 the feeding time is kept (no division at a
    zero index).
    """
    ft = np.asarray(ft, dtype=float)
    ok = (food_gain > 0) & (index > 0) & (qopt > 0)
    safe_index = np.where(ok, index, 1.0)
    actual_qb = np.where(ok, food_gain, 1.0) / safe_index
    new = lower + (1.0 - lower) * ft * ((1.0 - adj) + adj * qopt / actual_qb)
    return np.clip(np.where(ok, new, ft), lower, cap)
