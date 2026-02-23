"""
IBM-Ecosim derivative override integration.

Provides the bridge functions that allow IBM-managed functional groups to
override the standard Ecosim derivative calculation. When an IBM group is
active, these functions extract the relevant food-web data from the Ecosim
consumption matrix, delegate the dynamics to the IBM engine, and write the
results back into the derivative vector.

Functions
---------
extract_prey_availability
    Extract non-zero prey consumption rates for a given predator.
extract_predation_pressure
    Sum total predation on a given prey from all living predators.
check_ibm_mass_balance
    Validate that an IBM step result is physically plausible.
apply_ibm_to_derivative
    Override the Ecosim derivative for an IBM group in-place.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Dict, Optional, Tuple

import numpy as np

if TYPE_CHECKING:
    from pypath.ibm.base import IBMGroup, IBMStepResult, SpatialContext

logger = logging.getLogger(__name__)


def extract_prey_availability(
    QQ: np.ndarray, predator_idx: int, n_groups: int
) -> Dict[int, float]:
    """Extract non-zero prey consumption rates for a predator.

    Reads column ``predator_idx`` of the consumption matrix ``QQ`` and
    returns a dictionary mapping each prey index to its consumption rate,
    excluding entries that are exactly zero.

    Parameters
    ----------
    QQ : np.ndarray
        Consumption matrix of shape ``(n_groups+1, n_groups+1)`` where
        ``QQ[prey, predator]`` is the consumption rate.
    predator_idx : int
        1-based index of the predator group.
    n_groups : int
        Total number of functional groups (excluding the 0-index padding).

    Returns
    -------
    Dict[int, float]
        Mapping ``{prey_idx: consumption_rate}`` for all non-zero entries.
    """
    result: Dict[int, float] = {}
    for prey in range(1, n_groups + 1):
        rate = float(QQ[prey, predator_idx])
        if rate != 0.0:
            result[prey] = rate
    return result


def extract_predation_pressure(QQ: np.ndarray, prey_idx: int, n_living: int) -> float:
    """Sum total predation on a prey group from all living predators.

    Parameters
    ----------
    QQ : np.ndarray
        Consumption matrix of shape ``(n_groups+1, n_groups+1)``.
    prey_idx : int
        1-based index of the prey group.
    n_living : int
        Number of living groups in the model.

    Returns
    -------
    float
        Sum of ``QQ[prey_idx, 1:n_living+1]``.
    """
    return float(np.sum(QQ[prey_idx, 1 : n_living + 1]))


def check_ibm_mass_balance(
    result: "IBMStepResult", tolerance: float = 0.05
) -> Tuple[bool, float]:
    """Validate that an IBM step result is physically plausible.

    Checks that biomass is non-negative and that no consumption entry
    is negative. The relative error returned is the magnitude of the
    largest violation (or 0.0 if none).

    Parameters
    ----------
    result : IBMStepResult
        The result from an IBM integration step.
    tolerance : float, optional
        Maximum acceptable relative error (default 0.05).

    Returns
    -------
    Tuple[bool, float]
        ``(is_balanced, relative_error)`` where ``is_balanced`` is True
        when all checks pass within the tolerance.
    """
    error = 0.0

    if result.biomass < 0.0:
        error = max(error, abs(result.biomass))
        return False, error

    if np.any(result.consumption_by_prey < 0.0):
        min_consumption = float(np.min(result.consumption_by_prey))
        error = max(error, abs(min_consumption))
        return False, error

    return True, error


def apply_ibm_to_derivative(
    deriv: np.ndarray,
    QQ: np.ndarray,
    BB: np.ndarray,
    ibm_group: "IBMGroup",
    forcing: dict,
    dt: float,
    spatial_context: Optional["SpatialContext"] = None,
) -> None:
    """Override the Ecosim derivative for an IBM-managed group in-place.

    Calls the IBM group's ``compute_step`` method with prey availability
    and predation pressure extracted from the current Ecosim state, then
    writes the resulting biomass change into the derivative vector and
    subtracts IBM consumption from prey derivatives.

    Parameters
    ----------
    deriv : np.ndarray
        Derivative vector (modified in-place). Shape ``(n_groups+1,)``.
    QQ : np.ndarray
        Consumption matrix ``(n_groups+1, n_groups+1)``.
    BB : np.ndarray
        Current biomass vector ``(n_groups+1,)``.
    ibm_group : IBMGroup
        The IBM group instance that will compute the step.
    forcing : dict
        Environmental forcing dictionary passed through to ``compute_step``.
    dt : float
        Time step size in years.
    spatial_context : SpatialContext or None, optional
        Spatial context for multi-patch IBM movement. When provided, it is
        forwarded to ``ibm_group.compute_step()`` so the IBM engine can
        distribute individuals across patches. Default is ``None``.
    """
    group_idx = ibm_group.group_index
    n_groups = ibm_group.n_groups

    # Extract prey availability as a dict, then convert to array
    prey_dict = extract_prey_availability(QQ, group_idx, n_groups)
    prey_array = np.zeros(n_groups)
    for prey_idx, rate in prey_dict.items():
        if prey_idx < n_groups:
            prey_array[prey_idx] = rate

    # Extract predation pressure from all living predators
    # Use n_groups as upper bound for n_living (safe default)
    predation = extract_predation_pressure(QQ, group_idx, n_groups)

    # Delegate dynamics to the IBM engine
    result = ibm_group.compute_step(
        prey_available=prey_array,
        predation_pressure=predation,
        env_forcing=forcing,
        dt=dt,
        spatial_context=spatial_context,
    )

    # Override derivative: dB/dt = (new_biomass - current_biomass) / dt
    deriv[group_idx] = (result.biomass - BB[group_idx]) / dt

    # Subtract IBM consumption from prey derivatives
    for prey_idx in range(len(result.consumption_by_prey)):
        consumed = result.consumption_by_prey[prey_idx]
        if consumed != 0.0 and prey_idx < len(deriv):
            deriv[prey_idx] -= consumed / dt
