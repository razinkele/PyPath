"""Solver request service for the Ecosim page.

Turns the page's solver inputs into one validated ``SolverRequest`` and runs
it. The module has no Shiny import, so the page imports it and never the
reverse: the core library (``pypath``) stays the single owner of method names
and option rules, and the app only builds requests the core accepts.
"""

from __future__ import annotations

import json
import logging
import math
from dataclasses import dataclass
from numbers import Real

from pypath.core.ecosim import CONTINUOUS_METHODS, RsimOutput, RsimScenario, rsim_run
from pypath.core.ecosim_advanced import rsim_run_advanced
from pypath.core.ecosim_continuous import QS_RATE_DEFAULT
from pypath.core.forcing import DietRewiring

logger = logging.getLogger(__name__)

PARITY_METHODS = ("RK4", "AB")

METHOD_CHOICES: dict[str, dict[str, str]] = {
    "EwE parity": {"RK4": "Runge-Kutta 4", "AB": "Adams-Bashforth"},
    "Continuous (not EwE parity)": {
        "lsoda": "LSODA (automatic stiff/non-stiff)",
        "bdf": "BDF (recommended for stiff models)",
        "dop853": "DOP853 (non-stiff, high order)",
        "radau": "Radau (stiff, implicit)",
    },
}

# Presets pass only rtol: the core's default atol already follows rtol.
TOLERANCE_PRESETS: dict[str, dict] = {
    "fast": {"rtol": 1e-4},
    "default": {},
    "reference": {"rtol": 1e-10},
}

PRESET_CHOICES: dict[str, str] = {
    "fast": "Fast (rtol 1e-4)",
    "default": "Default (rtol 1e-6)",
    "reference": "Reference (rtol 1e-10)",
}

INTERPOLATION_CHOICES: dict[str, str] = {
    "constant": "Held each month (as EwE)",
    "linear": "Linear (keeps monthly means)",
    "smooth": "Smooth (keeps monthly means)",
}

CONTINUOUS_JS = (
    json.dumps(list(CONTINUOUS_METHODS)) + ".includes(input.integration_method)"
)

_QS_NOTE = (
    "Quasi-steady groups suit DOP853; on LSODA, BDF and Radau they usually "
    "cost more than integrating the fast groups."
)


def is_continuous(method: str) -> bool:
    """Return True when ``method`` is one of the continuous solvers."""
    return method in CONTINUOUS_METHODS


@dataclass(frozen=True)
class SolverRequest:
    """A validated solver choice, ready for :func:`run_ecosim`.

    Parameters
    ----------
    method : str
        ``"RK4"``, ``"AB"`` or a name in ``CONTINUOUS_METHODS``.
    solver_options : dict or None
        ``rsim_run`` options; None for the parity methods.
    notes : tuple of str
        Advice to show the user; never errors.
    """

    method: str
    solver_options: dict | None
    notes: tuple[str, ...] = ()


def _valid_qs_rate(value) -> float:
    ok = (
        isinstance(value, Real)
        and not isinstance(value, bool)
        and math.isfinite(value)
        and value > 0
    )
    if not ok:
        raise ValueError("Quasi-steady rate must be a number above 0 (per year).")
    return float(value)


def build_solver_request(
    method: str,
    *,
    preset: str = "default",
    log_states: bool = False,
    forcing_interpolation: str = "constant",
    quasi_steady: bool = False,
    qs_rate=QS_RATE_DEFAULT,
    diet_rewiring: bool = False,
) -> SolverRequest:
    """Build a solver request from the page's inputs.

    Parameters
    ----------
    method : str
        Integration method.
    preset : str
        Key of ``TOLERANCE_PRESETS``; continuous methods only.
    log_states : bool
        Integrate ``ln B`` (``positivity="log"``).
    forcing_interpolation : str
        ``"constant"``, ``"linear"`` or ``"smooth"``.
    quasi_steady : bool
        Treat the fast groups as quasi-steady.
    qs_rate : float
        Base loss rate (per year) above which a group is quasi-steady; read
        only with ``quasi_steady``.
    diet_rewiring : bool
        Whether diet rewiring is ticked; allowed on parity methods only.

    Returns
    -------
    SolverRequest

    Raises
    ------
    ValueError
        With user-facing text, for an unknown method or preset, a bad
        quasi-steady rate, or diet rewiring on a continuous method.
    """
    if method in PARITY_METHODS:
        return SolverRequest(method, None)
    if not is_continuous(method):
        raise ValueError(f"Unknown integration method {method!r}.")
    if diet_rewiring:
        raise ValueError(
            "Diet rewiring runs on RK4 and AB only. "
            "Turn it off, or choose an EwE parity method."
        )
    if preset not in TOLERANCE_PRESETS:
        raise ValueError(f"Unknown tolerance preset {preset!r}.")
    opts = dict(TOLERANCE_PRESETS[preset])
    if log_states:
        opts["positivity"] = "log"
    if forcing_interpolation in ("linear", "smooth"):
        opts["forcing_interpolation"] = forcing_interpolation
    notes: tuple[str, ...] = ()
    if quasi_steady:
        opts["fast_groups"] = "quasi-steady"
        opts["qs_rate"] = _valid_qs_rate(qs_rate)
        if method != "dop853":
            notes = (_QS_NOTE,)
    return SolverRequest(method, opts, notes)


def run_ecosim(
    scenario: RsimScenario,
    request: SolverRequest,
    *,
    diet_rewiring: DietRewiring | None = None,
) -> RsimOutput:
    """Run the scenario's whole length with the requested solver.

    Parameters
    ----------
    scenario : RsimScenario
        The scenario to run.
    request : SolverRequest
        From :func:`build_solver_request`.
    diet_rewiring : DietRewiring, optional
        Used on the parity methods only.

    Returns
    -------
    RsimOutput

    Raises
    ------
    ValueError
        Diet rewiring with a continuous method (it would run through
        ``rsim_run_advanced``, which has no continuous solver).
    """
    continuous = is_continuous(request.method)
    if diet_rewiring is not None and continuous:
        raise ValueError(
            "Diet rewiring runs on RK4 and AB only. "
            "Turn it off, or choose an EwE parity method."
        )
    if diet_rewiring is not None:
        return rsim_run_advanced(
            scenario,
            state_forcing=None,
            diet_rewiring=diet_rewiring,
            method=request.method,
        )
    if continuous:
        return rsim_run(
            scenario,
            method=request.method,
            solver_options=request.solver_options,
        )
    return rsim_run(scenario, method=request.method)
