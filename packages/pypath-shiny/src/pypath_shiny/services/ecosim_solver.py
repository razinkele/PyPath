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
import threading
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass
from numbers import Real

from pypath.core.ecosim import CONTINUOUS_METHODS, RsimOutput, RsimScenario, rsim_run
from pypath.core.ecosim_advanced import rsim_run_advanced
from pypath.core.ecosim_continuous import QS_RATE_DEFAULT
from pypath.core.forcing import DietRewiring

from pypath_shiny.services.tasks import TaskState, TaskStatus

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


# At most this many Ecosim integrations run at once in the app's process, over
# all sessions. A run cannot be stopped once its thread starts (Cancel only
# discards the result), and each is CPU-bound: an unbounded pool would let
# Run-then-Cancel pile up threads that starve every session. Further runs queue.
MAX_CONCURRENT_RUNS = 2
RUN_EXECUTOR = ThreadPoolExecutor(
    max_workers=MAX_CONCURRENT_RUNS, thread_name_prefix="ecosim-run"
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


# Words of an error about the run's options (a quasi-steady or forced group,
# a bad solver option, an unsupported model), which neither BDF nor a tighter
# tolerance fixes.
_OPTION_ERROR_WORDS = ("quasi-steady", "forced", "solver_options", "not supported")


def continuous_failure_hint(
    method: str, preset: str = "default", error: str = ""
) -> str:
    """Advice to append to a failed run's message, or "" when none can help.

    Parameters
    ----------
    method : str
        The run's method; parity methods get no advice.
    preset : str
        The run's tolerance preset: Reference is not suggested again.
    error : str
        The error's text: an error about the run's options gets no advice.
    """
    if not is_continuous(method):
        return ""
    text = str(error).lower()
    if any(word in text for word in _OPTION_ERROR_WORDS):
        return ""
    try_bdf = method != "bdf"
    try_reference = preset != "reference"
    if try_bdf and try_reference:
        return "Try BDF for stiff models, or the Reference tolerance."
    if try_bdf:
        return "Try BDF for stiff models."
    if try_reference:
        return "Try the Reference tolerance."
    return ""


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


@dataclass(frozen=True, eq=False)
class RunOutcome:
    """A finished background run, as the page's extended task returns it.

    Parameters
    ----------
    run_id : str
        The ``TaskState.run_id`` the run was started under.
    source : RsimScenario
        The scenario object the run was started from (the run itself used a
        deep copy of it).
    output : RsimOutput
        The run's output.
    request : SolverRequest
        The request the run used.
    diet_enabled : bool
        Whether diet rewiring was passed to the run.
    """

    run_id: str
    source: object
    output: RsimOutput
    request: SolverRequest
    diet_enabled: bool


def result_is_current(state: TaskState, outcome: RunOutcome, current_scenario) -> bool:
    """Whether a finished run may still land on the page.

    True only while ``state`` is RUNNING under the outcome's ``run_id`` and the
    page still holds the very scenario object the run started from. A run that
    was superseded, cancelled or retired, or whose scenario was replaced (even
    by an equal copy), is stale.
    """
    return (
        state.status is TaskStatus.RUNNING
        and state.run_id == outcome.run_id
        and current_scenario is outcome.source
    )


class RunSlot:
    """One session's in-flight background run.

    Holds the run's ``concurrent.futures.Future`` from submission until the
    worker finishes. Cancelling the asyncio side does not free the slot while
    the thread still computes (a thread cannot be stopped); a run cancelled
    while still queued for a worker is done at once, and frees it.
    """

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._future: Future | None = None

    @property
    def busy(self) -> bool:
        """True until the held run's future is done (finished or cancelled)."""
        with self._lock:
            return self._future is not None and not self._future.done()

    @property
    def running(self) -> bool:
        """True while a worker thread is computing the held run."""
        with self._lock:
            future = self._future
        return future is not None and future.running()

    @property
    def waiting(self) -> bool:
        """True while the held run is queued for a free worker."""
        with self._lock:
            future = self._future
        return future is not None and not future.running() and not future.done()

    def acquire(self, future: Future) -> None:
        """Hold ``future``; it is released when it completes.

        Raises
        ------
        RuntimeError
            When the slot still holds a run that is not done.
        """
        with self._lock:
            if self._future is not None and not self._future.done():
                raise RuntimeError("This session already has a run in flight.")
            self._future = future
        # Outside the lock: a future that is already done calls back at once.
        future.add_done_callback(self.release)

    def release(self, future: Future) -> None:
        """Forget ``future`` if it is the one held (its done callback)."""
        with self._lock:
            if self._future is future:
                self._future = None
