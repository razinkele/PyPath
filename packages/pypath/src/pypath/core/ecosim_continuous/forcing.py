"""The forcing the continuous right-hand side sees.

:func:`neutral_forcing` is the unforced month, and :class:`ForcingInterpolant`
interpolates the monthly forcing inside each month, keeping the monthly
means (``solver_options["forcing_interpolation"]``).
"""

from typing import Callable, Optional

import numpy as np


def neutral_forcing(params_dict: dict) -> dict:
    """Unforced monthly forcing: ones for time/prey/search/mortality/effort,
    zeros for forced biomass and migration."""
    n1 = int(params_dict["NUM_GROUPS"]) + 1
    ones = np.ones(n1)
    return {
        "Ftime": ones.copy(),
        "ForcedBio": np.zeros(n1),
        "ForcedMigrate": np.zeros(n1),
        "ForcedPrey": ones.copy(),
        "PP_forcing": ones.copy(),
        "ForcedSearch": ones.copy(),
        "ForcedMort": ones.copy(),
        "ForcedEffort": np.ones(int(params_dict.get("NUM_GEARS", 0)) + 1),
    }


#: The forcing keys interpolated inside a month. The others (ForcedBio,
#: ForcedRecs, ForcedEffort, forced F and catch, fleet dynamics, quotas,
#: StanzaPredRatio) stay monthly-held, so the catch integral F x mean(B)
#: stays exact.
INTERPOLATED_KEYS = (
    "ForcedPrey",
    "ForcedMort",
    "ForcedSearch",
    "PP_forcing",
    "ForcedMigrate",
    "ForcedActresp",
    "NutForcing",
    "ForcedSearchLink",
    "ForcedVulLink",
)

#: The interpolated keys that may be negative (a migration flux); every
#: other one is a multiplier, >= 0.
SIGNED_KEYS = frozenset({"ForcedMigrate"})


class ForcingInterpolant:
    """Mean-preserving interpolation of monthly forcing inside each month.

    ``monthly`` maps a forcing key to its monthly values, an array of shape
    ``(n_months, ...)`` (the same ``n_months`` for every key, else
    ``ValueError``); the arrays are copied, never written. Months are
    0-based: month ``k`` is row ``k``, and ``s`` in ``[0, 1]`` (clipped) is
    the fraction of the month.

    The boundary values: inside the series ``b_k = min((m_{k-1} + m_k) / 2,
    2 m_{k-1}, 2 m_k)`` for a non-negative key and ``(m_{k-1} + m_k) / 2``
    for a key in ``signed``; at the ends ``b_0 = m_0`` and ``b_N =
    m_{N-1}``, so nothing past the series is read. With ``kind="linear"``
    month ``k`` is a continuous tent through ``(0, b_k)``, ``(1/2, c_k)``
    and ``(1, b_{k+1})``, ``c_k = 2 m_k - (b_k + b_{k+1}) / 2``; it has a
    kink at mid-month. With ``kind="smooth"`` it is one quadratic, ``b_k (1
    - s) + b_{k+1} s + 6 (m_k - (b_k + b_{k+1}) / 2) s (1 - s)``. Either way
    the month's mean is exactly ``m_k``, the values are continuous at the
    month boundaries, a non-negative key (every key not in ``signed``,
    whose monthly values are taken to be >= 0) stays >= 0 (``0 <= b <= 2
    m`` on both sides of every month), and a series constant in time gives
    exactly that constant.
    """

    def __init__(
        self,
        monthly: dict,
        signed: frozenset = SIGNED_KEYS,
        kind: str = "linear",
    ):
        if kind not in ("linear", "smooth"):
            raise ValueError(
                f"ForcingInterpolant kind must be 'linear' or 'smooth', got {kind!r}"
            )
        self.kind = kind
        self.signed = frozenset(signed)
        self._m: dict = {}
        self._b: dict = {}
        self._inner: dict = {}  # c (linear) or 6 (m - (b_k + b_{k+1}) / 2)
        n_months: Optional[int] = None
        for key, series in monthly.items():
            m = np.array(series, dtype=float, copy=True)
            if m.ndim == 0 or len(m) == 0:
                raise ValueError(f"ForcingInterpolant: {key!r} needs one row per month")
            if n_months is None:
                n_months = len(m)
            elif len(m) != n_months:
                raise ValueError(
                    f"ForcingInterpolant: {key!r} has {len(m)} months, "
                    f"another key {n_months}"
                )
            b = np.empty((len(m) + 1,) + m.shape[1:])
            b[0], b[-1] = m[0], m[-1]
            avg = 0.5 * (m[:-1] + m[1:])
            if key in self.signed:
                b[1:-1] = avg
            else:
                b[1:-1] = np.minimum(avg, np.minimum(2.0 * m[:-1], 2.0 * m[1:]))
            mid = 0.5 * (b[:-1] + b[1:])
            self._m[key] = m
            self._b[key] = b
            self._inner[key] = 2.0 * m - mid if kind == "linear" else 6.0 * (m - mid)
        self.n_months = 0 if n_months is None else n_months

    @property
    def keys(self) -> tuple:
        """The interpolated keys."""
        return tuple(self._m)

    def at(self, month: int, s: float) -> dict:
        """Every key's value at the fraction ``s`` of month ``month``.

        Each half of the month is written from its own end (the start for
        ``s <= 1/2``, the end beyond), so a month's ends are exactly its
        boundary values and a constant series is exactly constant.
        """
        k = int(month)
        if not 0 <= k < self.n_months:
            raise IndexError(f"month {k} outside 0..{self.n_months - 1}")
        s = min(max(float(s), 0.0), 1.0)
        out = {}
        for key, b in self._b.items():
            b0, b1, inner = b[k], b[k + 1], self._inner[key][k]
            if self.kind == "linear":  # inner = c, the mid-month value
                if s <= 0.5:
                    out[key] = b0 + (inner - b0) * (2.0 * s)
                else:
                    out[key] = b1 + (inner - b1) * (2.0 - 2.0 * s)
            else:
                if s <= 0.5:
                    line = b0 + (b1 - b0) * s
                else:
                    line = b1 - (b1 - b0) * (1.0 - s)
                out[key] = line + inner * (s * (1.0 - s))
        return out

    def mean(self, month: int, key: str) -> np.ndarray:
        """Month ``month``'s mean of ``key``: its monthly value, exactly."""
        return np.array(self._m[key][int(month)], copy=True)

    def for_month(self, month: int, dt: float) -> Callable[[float], dict]:
        """``forcing_at(t)`` for :meth:`ContinuousRHS.set_month`: the values at
        time ``t`` of a month of length ``dt`` (``s = t / dt``)."""
        k = int(month)
        if not 0 <= k < self.n_months:
            raise IndexError(f"month {k} outside 0..{self.n_months - 1}")
        dt = float(dt)

        def forcing_at(t: float) -> dict:
            return self.at(k, t / dt)

        return forcing_at
