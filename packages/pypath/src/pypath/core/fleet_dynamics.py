"""Fleet Dynamics: profit-responsive effort and TAC quota management.

Models fleet capacity dynamics where effort responds to economic profit signals,
and total allowable catch (TAC) quotas can cap fishing activity.

Each fleet has capacity that grows when profitable and depreciates otherwise.
Effort is derived from capacity via an efficiency power relationship.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field

import numpy as np

logger = logging.getLogger(__name__)

_CAPACITY_FLOOR = 0.01
_EPSILON = 1e-10


@dataclass
class FleetEconParams:
    """Economic parameters for fleet dynamics.

    All cost/price arrays are indexed by fishing link (1-based externally,
    but stored as 0-based arrays here).

    Parameters
    ----------
    fixed_cost : np.ndarray
        Fixed annual cost per fleet (n_fleets,). Does not depend on effort.
    variable_cost : np.ndarray
        Variable cost per fleet per unit capacity (n_fleets,).
    sailing_cost : np.ndarray
        Sailing cost per fleet per unit capacity (n_fleets,).
    price : np.ndarray
        Price per unit catch per fishing link (n_links,).
    cap_depreciate : np.ndarray
        Capacity depreciation rate per fleet (n_fleets,).
    cap_base_growth : np.ndarray
        Base capacity growth rate per fleet when profitable (n_fleets,).
    eff_power : np.ndarray
        Efficiency power: effort = capacity ** eff_power (n_fleets,).
    tac : np.ndarray or None
        Total allowable catch (n_fleets, n_groups). None means no TAC limits.
    """

    fixed_cost: np.ndarray
    variable_cost: np.ndarray
    sailing_cost: np.ndarray
    price: np.ndarray
    cap_depreciate: np.ndarray
    cap_base_growth: np.ndarray
    eff_power: np.ndarray
    tac: np.ndarray | None = None


@dataclass
class FleetDynamicsResult:
    """Output time series from fleet dynamics simulation.

    Parameters
    ----------
    out_Effort : np.ndarray
        Monthly effort per fleet (n_months, n_fleets).
    out_Revenue : np.ndarray
        Monthly revenue per fleet (n_months, n_fleets).
    out_Cost : np.ndarray
        Monthly cost per fleet (n_months, n_fleets).
    out_Profit : np.ndarray
        Monthly profit per fleet (n_months, n_fleets).
    annual_Effort : np.ndarray
        Annual average effort per fleet (n_years, n_fleets).
    annual_Profit : np.ndarray
        Annual total profit per fleet (n_years, n_fleets).
    fleet_names : list[str]
        Fleet name labels.
    """

    out_Effort: np.ndarray
    out_Revenue: np.ndarray
    out_Cost: np.ndarray
    out_Profit: np.ndarray
    annual_Effort: np.ndarray
    annual_Profit: np.ndarray
    fleet_names: list[str]


def create_fleet_econ_params(n_fleets: int, n_links: int) -> FleetEconParams:
    """Create FleetEconParams with sensible defaults.

    Defaults: all cost/depreciation rates zero, eff_power=1.0, tac=None.

    Parameters
    ----------
    n_fleets : int
        Number of fleets.
    n_links : int
        Number of fishing links (fleet-group combinations).

    Returns
    -------
    FleetEconParams
        Parameter object with zero arrays and eff_power=1.0.
    """
    return FleetEconParams(
        fixed_cost=np.zeros(n_fleets),
        variable_cost=np.zeros(n_fleets),
        sailing_cost=np.zeros(n_fleets),
        price=np.zeros(n_links),
        cap_depreciate=np.zeros(n_fleets),
        cap_base_growth=np.zeros(n_fleets),
        eff_power=np.ones(n_fleets),
        tac=None,
    )


def fleet_dynamics_step(
    capacity: np.ndarray,
    monthly_catch: np.ndarray,
    cumulative_catch: np.ndarray,
    params: FleetEconParams,
    fish_through: np.ndarray,
    fish_from: np.ndarray,
    fleet_lookup: dict,
    n_fleets: int,
    dt: float = 1.0 / 12,
) -> tuple[np.ndarray, np.ndarray]:
    """Advance fleet capacity and effort by one time step.

    Computes revenue from catches, costs from capacity, a profit signal,
    then updates capacity using a logistic-style ODE.

    Parameters
    ----------
    capacity : np.ndarray
        Current fleet capacities (n_fleets,). Modified in place via copy.
    monthly_catch : np.ndarray
        Catch from this month's fishing links (n_links,). 1-based index 0 unused.
    cumulative_catch : np.ndarray
        Cumulative annual catch per fleet per group (n_fleets, n_groups).
        Not modified here; used externally.
    params : FleetEconParams
        Economic parameters.
    fish_through : np.ndarray
        Fleet group index per fishing link (1-based, length n_links+1,
        index 0 unused).
    fish_from : np.ndarray
        Prey group index per fishing link (1-based, length n_links+1,
        index 0 unused).
    fleet_lookup : dict
        Maps gear_0based (int) -> 1-based gear array index (int).
    n_fleets : int
        Number of fleets.
    dt : float
        Time step in years (default 1/12 for monthly).

    Returns
    -------
    new_capacity : np.ndarray
        Updated fleet capacities (n_fleets,).
    effort : np.ndarray
        Fleet effort this step (n_fleets,), effort = capacity ** eff_power.
    """
    new_capacity = capacity.copy()
    revenue = np.zeros(n_fleets)
    cost = np.zeros(n_fleets)

    # Step 1: Compute revenue per fleet from fishing links
    # price array has same length as FishFrom (index 0 unused, like FishFrom[0])
    for i in range(1, len(monthly_catch)):
        gear_0based = int(fish_through[i]) - 1
        gear_idx = fleet_lookup.get(gear_0based, 0)
        if gear_idx == 0:
            continue
        fleet_0 = gear_idx - 1
        if fleet_0 < 0 or fleet_0 >= n_fleets:
            continue
        price_i = params.price[i] if i < len(params.price) else 0.0
        revenue[fleet_0] += monthly_catch[i] * price_i

    # Step 2: Compute cost per fleet
    for g in range(n_fleets):
        cost[g] = (
            params.fixed_cost[g] / 12.0
            + (params.variable_cost[g] + params.sailing_cost[g]) * new_capacity[g]
        )

    # Step 3: Compute profit
    profit = revenue - cost

    # Step 4–6: Update capacity
    for g in range(n_fleets):
        denom = max(revenue[g], cost[g])
        if denom <= _EPSILON:
            denom = _EPSILON
        profit_signal = profit[g] / denom

        # Capacity ODE: dC/dt = (cap_base_growth * max(signal, 0) - cap_depreciate) * C
        growth_term = params.cap_base_growth[g] * max(profit_signal, 0.0)
        decay_term = params.cap_depreciate[g]
        new_capacity[g] += (growth_term - decay_term) * new_capacity[g] * dt
        new_capacity[g] = max(new_capacity[g], _CAPACITY_FLOOR)

    # Step 7: Effort from capacity
    effort = new_capacity ** params.eff_power

    return new_capacity, effort


def apply_quota_caps(
    fish_q: np.ndarray,
    cumulative_catch: np.ndarray,
    tac: np.ndarray,
    fish_through: np.ndarray,
    fish_from: np.ndarray,
    fleet_lookup: dict,
) -> np.ndarray:
    """Zero out fishing catchability for links that have reached their TAC.

    Parameters
    ----------
    fish_q : np.ndarray
        Current FishQ values per fishing link (n_links+1,), index 0 unused.
    cumulative_catch : np.ndarray
        Cumulative annual catch (n_fleets, n_groups), 0-based indices.
    tac : np.ndarray
        Total allowable catch allocation (n_fleets, n_groups), 0-based.
    fish_through : np.ndarray
        Fleet group index per fishing link (1-based, length n_links+1).
    fish_from : np.ndarray
        Prey group index per fishing link (1-based, length n_links+1).
    fleet_lookup : dict
        Maps gear_0based (int) -> 1-based gear array index (int).

    Returns
    -------
    np.ndarray
        Copy of fish_q with zeroed entries for TAC-exhausted links.
    """
    new_fish_q = fish_q.copy()
    n_links = len(fish_q) - 1  # index 0 is unused padding

    for i in range(1, n_links + 1):
        gear_0based = int(fish_through[i]) - 1
        gear_idx = fleet_lookup.get(gear_0based, 0)
        if gear_idx == 0:
            continue
        fleet_0 = gear_idx - 1

        group_0 = int(fish_from[i]) - 1
        if group_0 < 0:
            continue

        n_fleets, n_groups = tac.shape
        if fleet_0 < 0 or fleet_0 >= n_fleets:
            continue
        if group_0 >= n_groups:
            continue

        if cumulative_catch[fleet_0, group_0] >= tac[fleet_0, group_0]:
            new_fish_q[i] = 0.0

    return new_fish_q
