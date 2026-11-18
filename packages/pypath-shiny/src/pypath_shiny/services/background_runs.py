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
from dataclasses import dataclass
from typing import Any, Optional

from pypath.core.ecosim import rsim_run
from pypath.ibm import SmeltIBM
from pypath.spatial import rsim_run_spatial

from pypath_shiny.services.ecosim_solver import RunControl
from pypath_shiny.services.tasks import TaskState, TaskStatus

__all__ = [
    "IbmOutcome",
    "IbmRunSpec",
    "SpatialOutcome",
    "ibm_result_is_current",
    "run_ibm_comparison",
    "run_spatial",
    "spatial_result_is_current",
]


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
    """

    run_id: str
    source: Any
    grid: Any
    output: Any
    switching: Any


def run_spatial(scenario, ecospace_params, spatial_fishing):
    """Run the spatial simulation; called on a worker thread.

    Parameters
    ----------
    scenario : RsimScenario
        A copy the worker owns.
    ecospace_params : EcospaceParams
        The run's spatial configuration.
    spatial_fishing : SpatialFishing or None
        The run's effort allocation.

    Returns
    -------
    RsimOutput
        The spatial run's output.
    """
    return rsim_run_spatial(
        scenario,
        ecospace=ecospace_params,
        spatial_fishing=spatial_fishing,
    )


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
    """

    run_id: int
    source: Any
    ibm: Any
    standard: Any
    enhanced: Any
    switching: Any


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
