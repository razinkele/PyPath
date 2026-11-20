"""Worker functions for the Ecospace and IBM pages' background runs.

Both pages integrate on a worker thread of the shared ``RUN_EXECUTOR``
(``services.ecosim_solver``), off the event loop, so a long run does not
freeze the other sessions. This module holds what the worker thread runs and
the checks that decide whether a finished run may still land on the page. It
imports no Shiny, so it can be tested without a session.

The worker functions read no input and send no notification: the page's
launch effect gathers everything on the event loop first, and its result
effect lands the outcome there.
"""

from __future__ import annotations

import copy
import hashlib
import logging
import os
from concurrent.futures.process import BrokenProcessPool
from dataclasses import dataclass
from typing import Any, Optional

import numpy as np
from pypath.core._parallel import usable_cpus
from pypath.core.ecosim import rsim_run
from pypath.ibm import SmeltIBM
from pypath.spatial import rsim_run_spatial

from pypath_shiny.config import SPATIAL
from pypath_shiny.services.ecosim_solver import RunControl
from pypath_shiny.services.tasks import TaskState, TaskStatus

logger = logging.getLogger(__name__)

__all__ = [
    "IbmOutcome",
    "IbmRunSpec",
    "SpatialOutcome",
    "ibm_result_is_current",
    "run_ibm_comparison",
    "run_spatial",
    "scenario_changed_since",
    "scenario_fingerprint",
    "spatial_result_is_current",
]


# --- In-place scenario edits -------------------------------------------

# The parts of a scenario the pages edit in place: forcing and fishing series,
# and parameters such as vulnerabilities.
_FINGERPRINT_PARTS = ("forcing", "fishing", "params")


def scenario_fingerprint(scenario) -> str:
    """A digest of the scenario's forcing, fishing and parameter values.

    The pages keep one scenario object and edit it in place (the Ecosim
    page's forcing card rewrites ``ForcedPrey`` when an Ecosim run starts), so
    the object's identity cannot tell whether a run's inputs still hold. Two
    scenarios with equal arrays and scalars in ``forcing``, ``fishing`` and
    ``params`` give the same digest.

    Parameters
    ----------
    scenario : RsimScenario
        The scenario to digest; it is only read.

    Returns
    -------
    str
        A hex digest.
    """
    digest = hashlib.blake2b(digest_size=16)
    for part in _FINGERPRINT_PARTS:
        obj = getattr(scenario, part, None)
        if obj is None:
            continue
        for name, value in sorted(vars(obj).items()):
            if isinstance(value, (bool, int, float, str)):
                data = repr(value).encode()
            elif isinstance(value, (np.ndarray, list, tuple)):
                try:
                    array = np.ascontiguousarray(value)
                except (TypeError, ValueError):
                    continue
                if array.dtype == object:
                    continue
                data = str((array.dtype, array.shape)).encode() + array.tobytes()
            else:
                continue  # nested objects (IBM groups, stanzas) are not edited here
            digest.update(f"{part}.{name}=".encode())
            digest.update(data)
    return digest.hexdigest()


def scenario_changed_since(outcome, current_scenario) -> bool:
    """Whether the scenario was edited in place after ``outcome``'s launch.

    False for an outcome that carries no fingerprint.
    """
    launch = getattr(outcome, "fingerprint", "")
    return bool(launch) and scenario_fingerprint(current_scenario) != launch


# --- Ecospace ----------------------------------------------------------


@dataclass(frozen=True)
class SpatialOutcome:
    """A finished spatial run, as the Ecospace page's task returns it.

    Parameters
    ----------
    run_id : str
        The ``TaskState.run_id`` the run was started under.
    source : RsimScenario
        The scenario object the run was started from (the run itself used a
        deep copy of it).
    grid : object
        The page's grid object at launch.
    output : RsimOutput
        The run's output.
    switching : object
        The prey-switching summary of ``source`` at launch.
    fingerprint : str
        ``scenario_fingerprint(source)`` at launch; empty when not taken.
    """

    run_id: str
    source: Any
    grid: Any
    output: Any
    switching: Any
    fingerprint: str = ""


#: Environment variable: worker processes for a spatial run (off unless set).
ECOSPACE_JOBS_ENV = "PYPATH_ECOSPACE_JOBS"


def spatial_jobs(n_patches: int) -> int:
    """Worker processes for a spatial run on a grid of *n_patches*; 1 is serial.

    Off unless the environment variable ``PYPATH_ECOSPACE_JOBS`` asks for a
    number above 1, and only for grids of at least
    ``SPATIAL.min_patches_for_workers`` patches. Never more than the CPUs
    this process may use. Anything unreadable in the variable means "off".

    It is opt-in because not every way of starting the app can start worker
    processes: under shiny-server its launcher script is re-run in each
    worker and fails (:func:`run_spatial` then falls back to one process),
    and every run with workers takes several CPU cores on a shared server.
    """
    try:
        wanted = int(os.environ.get(ECOSPACE_JOBS_ENV, "1"))
    except ValueError:
        return 1
    if wanted <= 1 or n_patches < SPATIAL.min_patches_for_workers:
        return 1
    return max(1, min(wanted, usable_cpus()))


def run_spatial(scenario, ecospace_params, spatial_fishing, control=None, n_jobs=1):
    """Run the spatial simulation; called on a worker thread.

    Parameters
    ----------
    scenario : RsimScenario
        A copy the worker owns.
    ecospace_params : EcospaceParams
        The run's spatial configuration.
    spatial_fishing : SpatialFishing or None
        The run's effort allocation.
    control : RunControl, optional
        Its ``on_month`` is passed to the run, which records each month's
        progress on it and stops at the next month boundary after a
        requested stop (``RunCancelled``).
    n_jobs : int
        Worker processes for the patches (:func:`spatial_jobs`). If they
        cannot be started, or die, the run is done again in this process
        alone: the numbers are the same either way.

    Returns
    -------
    RsimOutput
        The spatial run's output.

    Raises
    ------
    RunCancelled
        When ``control`` was asked to stop.
    """
    hook = {} if control is None else {"on_month": control.on_month}

    def run(jobs):
        return rsim_run_spatial(
            scenario,
            ecospace=ecospace_params,
            spatial_fishing=spatial_fishing,
            n_jobs=jobs,
            **hook,
        )

    if n_jobs <= 1:
        return run(1)
    try:
        return run(n_jobs)
    except (BrokenProcessPool, EOFError, OSError) as exc:
        # The workers, not the model: start over with none.
        logger.warning(
            "Ecospace worker processes failed (%s: %s); running in one process. "
            "Unset %s where the app cannot start worker processes.",
            type(exc).__name__,
            exc,
            ECOSPACE_JOBS_ENV,
        )
        return run(1)


def spatial_result_is_current(
    state: TaskState, outcome: SpatialOutcome, current_scenario, current_grid
) -> bool:
    """Whether a finished spatial run may still land on the page.

    True only while ``state`` is RUNNING under the outcome's ``run_id`` and the
    page still holds the very scenario and grid objects the run started from.
    The page stays usable during a run, so either can be replaced under it; a
    result for another grid has the wrong number of patches for the maps.
    """
    return (
        state.status is TaskStatus.RUNNING
        and state.run_id == outcome.run_id
        and current_scenario is outcome.source
        and current_grid is outcome.grid
    )


# --- IBM comparison ----------------------------------------------------


@dataclass(frozen=True)
class IbmRunSpec:
    """What the worker needs to rebuild the page's IBM group for a run.

    Parameters
    ----------
    group_index : int
        The Ecosim group the IBM replaces.
    n_groups : int
        The IBM's group count.
    params : object
        The IBM's parameter object.
    n_super : int
        Number of super-individuals to start with.
    """

    group_index: int
    n_groups: int
    params: Any
    n_super: int


@dataclass(frozen=True)
class IbmOutcome:
    """A finished comparison, as the IBM page's task returns it.

    Parameters
    ----------
    run_id : int
        The launch counter the comparison was started under.
    source : RsimScenario
        The scenario object the comparison was started from.
    ibm : object
        The page's IBM instance at launch.
    standard : RsimOutput
        The standard Ecosim run.
    enhanced : RsimOutput
        The IBM-enhanced run.
    switching : object
        The prey-switching summary of ``source`` at launch.
    fingerprint : str
        ``scenario_fingerprint(source)`` at launch; empty when not taken.
    """

    run_id: int
    source: Any
    ibm: Any
    standard: Any
    enhanced: Any
    switching: Any
    fingerprint: str = ""


def run_ibm_comparison(
    scenario, spec: IbmRunSpec, control: Optional[RunControl] = None
):
    """Run standard Ecosim, then the IBM-enhanced run; on a worker thread.

    Each run uses its own deep copy of ``scenario``. The IBM run gets a fresh
    ``SmeltIBM`` built from ``spec`` and started from the group's base
    biomass.

    Parameters
    ----------
    scenario : RsimScenario
        The scenario to compare on; it is not modified.
    spec : IbmRunSpec
        How to rebuild the IBM group.
    control : RunControl, optional
        Its ``on_month`` is passed to both runs, so a requested stop ends the
        comparison at the next month boundary (``RunCancelled``).

    Returns
    -------
    tuple
        ``(standard_output, ibm_output)``.
    """
    hook = {} if control is None else {"on_month": control.on_month}

    standard = rsim_run(copy.deepcopy(scenario), **hook)

    ibm_scenario = copy.deepcopy(scenario)
    if not getattr(ibm_scenario.params, "ibm_groups", None):
        ibm_scenario.params.ibm_groups = {}
    fresh = SmeltIBM(
        group_index=spec.group_index,
        n_groups=spec.n_groups,
        params=spec.params,
    )
    fresh.initialize_from_ecosim(
        biomass=float(ibm_scenario.params.B_BaseRef[spec.group_index]),
        params={},
        n_super_individuals=spec.n_super,
    )
    ibm_scenario.params.ibm_groups[spec.group_index] = fresh
    enhanced = rsim_run(ibm_scenario, **hook)

    return standard, enhanced


def ibm_result_is_current(
    launched_id: int, outcome: IbmOutcome, current_scenario, current_ibm
) -> bool:
    """Whether a finished comparison may still land on the page.

    True only for the latest launch, while the page still holds the very
    scenario and IBM objects the comparison started from.
    """
    return (
        launched_id == outcome.run_id
        and current_scenario is outcome.source
        and current_ibm is outcome.ibm
    )
