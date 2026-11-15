"""Options of the continuous methods.

:func:`continuous_options` checks ``rsim_run``'s method and
``solver_options`` against the vocabularies defined here
(``METHOD_NAMES`` and the option modes), and :func:`default_atol` is the
default absolute tolerance. ``_Y_FLOOR`` is the biomass floor shared by the
quasi-steady solve and the month integrator.
"""

import logging
import numbers
from typing import Optional

import numpy as np

logger = logging.getLogger(__name__)


#: ``solver_options["forcing_interpolation"]``: each month's forcing held
#: (the default), or varying inside the month with the monthly means kept
#: (:class:`ForcingInterpolant`'s two kinds).
FORCING_INTERPOLATIONS = ("constant", "linear", "smooth")


#: The continuous methods (lower case) and their ``solve_ivp`` names; the
#: one source of ``ecosim.CONTINUOUS_METHODS``.
METHOD_NAMES = {
    "lsoda": "LSODA",
    "dop853": "DOP853",
    "radau": "Radau",
    "bdf": "BDF",
}

#: ``solver_options["jacobian"]``: finite-difference Jacobians reused across
#: months (:class:`JacobianReuse`, the default) or the solver's own.
JACOBIAN_MODES = ("reuse", "solver")

#: ``solver_options["positivity"]``: biomass states clipped at 0 in the
#: right-hand side (the default) or integrated as ``z = ln B``.
POSITIVITY_MODES = ("clip", "log")

#: ``solver_options["fast_groups"]``: every group a state (the default), or
#: the fast groups solved quasi-steady (:class:`QuasiSteady`).
FAST_GROUP_MODES = ("states", "quasi-steady")

#: ``solver_options["qs_rate"]``'s default, per year: a flagged group is
#: quasi-steady when its base loss rate exceeds it.
QS_RATE_DEFAULT = 365.0


def continuous_options(method, solver_options: Optional[dict], n_bio: int) -> dict:
    """``rsim_run``'s checks of a continuous method and its ``solver_options``.

    ``method`` is a name in ``METHOD_NAMES``, in any case; anything else, a
    non-string included, raises ``ValueError`` listing the methods
    (``rsim_run`` calls this for every method but RK4 and AB).
    ``solver_options`` takes ``rtol`` (> 0, default 1e-6), ``atol`` (a scalar
    > 0 or ``n_bio`` values > 0), ``corrections`` (True or False, default
    True), ``on_failure`` (``"raise"``, the default, or ``"warn"``),
    ``hden_tol`` (the handling-time tolerance, a number in (0, 1e-8],
    default 1e-12), ``jacobian`` (``"reuse"``, the default:
    :class:`JacobianReuse`; or ``"solver"``: the solver's own Jacobians) and
    ``forcing_interpolation`` (``"constant"``, the default, ``"linear"`` or
    ``"smooth"``: a model option, :class:`ForcingInterpolant`) and
    ``positivity`` (``"clip"``, the default, or ``"log"``: a numerical
    option, the same equations with the biomass states integrated as
    ``ln B``, see :func:`integrate_month`),
    ``fast_groups`` (``"states"``, the default, or ``"quasi-steady"``: a
    model option, :class:`QuasiSteady`) and ``qs_rate`` (a number > 0 per
    year, default 365: the base loss rate above which a flagged group is
    quasi-steady; given with ``fast_groups="states"`` it is unused, which
    logs a WARNING); other keys and invalid values raise ``ValueError``.

    Returns a dict with keys ``method`` (lower case), ``rtol`` (float),
    ``atol`` (None when not given, else an ndarray, 0-d for a scalar),
    ``on_failure``, ``corrections`` (bool), ``hden_tol`` (float),
    ``jacobian``, ``forcing_interpolation``, ``positivity``, ``fast_groups``
    and ``qs_rate`` (float).
    """
    if not isinstance(method, str) or method.lower() not in METHOD_NAMES:
        raise ValueError(
            f"Unknown integration method {method!r}: use "
            f"{', '.join(repr(m) for m in ('RK4', 'AB') + tuple(METHOD_NAMES))}"
        )
    _cont_opts = dict(solver_options or {})
    _known_opts = (
        "rtol",
        "atol",
        "corrections",
        "on_failure",
        "hden_tol",
        "jacobian",
        "forcing_interpolation",
        "positivity",
        "fast_groups",
        "qs_rate",
    )
    _unknown_opts = sorted(set(_cont_opts) - set(_known_opts))
    if _unknown_opts:
        raise ValueError(
            f"Unknown solver_options {_unknown_opts}: use "
            f"{', '.join(repr(k) for k in _known_opts)}"
        )
    _interp = _cont_opts.get("forcing_interpolation", "constant")
    if not isinstance(_interp, str) or _interp not in FORCING_INTERPOLATIONS:
        raise ValueError(
            "solver_options forcing_interpolation must be 'constant', 'linear' "
            f"or 'smooth', got {_interp!r}"
        )
    _positivity = _cont_opts.get("positivity", "clip")
    if not isinstance(_positivity, str) or _positivity not in POSITIVITY_MODES:
        raise ValueError(
            f"solver_options positivity must be 'clip' or 'log', got {_positivity!r}"
        )
    _fast_groups = _cont_opts.get("fast_groups", "states")
    if not isinstance(_fast_groups, str) or _fast_groups not in FAST_GROUP_MODES:
        raise ValueError(
            "solver_options fast_groups must be 'states' or 'quasi-steady', "
            f"got {_fast_groups!r}"
        )
    _qs_rate = _cont_opts.get("qs_rate", QS_RATE_DEFAULT)
    if not (
        isinstance(_qs_rate, numbers.Real)
        and not isinstance(_qs_rate, (bool, np.bool_))
        and np.isfinite(float(_qs_rate))
        and float(_qs_rate) > 0.0
    ):
        raise ValueError(
            f"solver_options qs_rate must be a number > 0 (per year), got {_qs_rate!r}"
        )
    if "qs_rate" in _cont_opts and _fast_groups == "states":
        logger.warning(
            "solver_options qs_rate=%r is unused: it selects quasi-steady "
            "groups, and fast_groups is 'states'",
            _qs_rate,
        )
    _jacobian = _cont_opts.get("jacobian", "reuse")
    if not isinstance(_jacobian, str) or _jacobian not in JACOBIAN_MODES:
        raise ValueError(
            f"solver_options jacobian must be 'reuse' or 'solver', got {_jacobian!r}"
        )
    _on_failure = _cont_opts.get("on_failure", "raise")
    if _on_failure not in ("raise", "warn"):
        raise ValueError(
            f"solver_options on_failure must be 'raise' or 'warn', got {_on_failure!r}"
        )
    _corrections = _cont_opts.get("corrections", True)
    if not isinstance(_corrections, (bool, np.bool_)):
        raise ValueError(
            f"solver_options corrections must be True or False, got {_corrections!r}"
        )
    _rtol_opt = _cont_opts.get("rtol", 1e-6)
    try:
        _rtol_ok = np.ndim(_rtol_opt) == 0 and float(_rtol_opt) > 0
    except (TypeError, ValueError):
        _rtol_ok = False
    if not _rtol_ok:  # NaN fails too
        raise ValueError(f"solver_options rtol must be > 0, got {_rtol_opt!r}")
    _atol_opt = _cont_opts.get("atol")
    _atol_arr: Optional[np.ndarray] = None
    if _atol_opt is not None:
        try:
            _atol_arr = np.asarray(_atol_opt, dtype=float)
        except (TypeError, ValueError):
            _atol_arr = np.array([np.nan])  # fails the check below
        if not (_atol_arr.ndim == 0 or _atol_arr.shape == (n_bio,)) or not np.all(
            _atol_arr > 0
        ):
            raise ValueError(
                "solver_options atol must be a scalar > 0 or "
                f"{n_bio} values > 0 (groups 1..NUM_LIVING + "
                f"NUM_DEAD), got {_atol_opt!r}"
            )
    # LSODA's finite-difference Jacobian perturbs states by about 1.5e-8
    # relative: a looser Hden would show in its columns
    _hden_tol = _cont_opts.get("hden_tol", 1e-12)
    if not (
        isinstance(_hden_tol, numbers.Real)
        and not isinstance(_hden_tol, (bool, np.bool_))
        and 0.0 < float(_hden_tol) <= 1e-8
    ):
        raise ValueError(
            f"solver_options hden_tol must be in (0, 1e-8], got {_hden_tol!r}"
        )
    return {
        "method": method.lower(),
        "rtol": float(_rtol_opt),
        "atol": _atol_arr,
        "on_failure": _on_failure,
        "corrections": bool(_corrections),
        "hden_tol": float(_hden_tol),
        "jacobian": _jacobian,
        "forcing_interpolation": _interp,
        "positivity": _positivity,
        "fast_groups": _fast_groups,
        "qs_rate": float(_qs_rate),
    }


_Y_FLOOR = 1e-20


def default_atol(b0_bio: np.ndarray, rtol: float = 1e-6) -> np.ndarray:
    """Absolute tolerance per state, scaled to the base biomass and, so that
    a tighter ``rtol`` is not capped by ``atol``, to ``rtol / 1e-6``."""
    b0_bio = np.asarray(b0_bio, dtype=float)
    return 1e-8 * (rtol / 1e-6) * np.maximum(b0_bio, 1e-6 * b0_bio.max())
