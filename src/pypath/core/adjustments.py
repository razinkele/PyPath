"""
Adjustment functions for Ecosim scenarios.

This module provides functions to modify fishing rates,
forcing functions, and other scenario parameters over time.

Based on Rpath's adjust.fishing(), adjust.forcing(), and adjust.scenario() functions.
"""

from typing import Union, List, Sequence, Optional
import numpy as np


def adjust_fishing(
    scenario,
    parameter: str,
    group: Union[str, int, List[Union[str, int]]],
    sim_year: Union[int, range, List[int]],
    value: Union[float, np.ndarray],
    sim_month: Optional[Union[int, range, List[int]]] = None,
):
    """Adjust fishing parameters in an Ecosim scenario.

    Modifies fishing-related forcing matrices (ForcedEffort, ForcedFRate,
    or ForcedCatch) for specified groups and time periods.

    Args:
        scenario: RsimScenario object to modify
        parameter: One of 'ForcedEffort', 'ForcedFRate', or 'ForcedCatch'
        group: Group name(s) or index(es) to modify. Can be:
            - Single group name (str) or index (int)
            - List of group names or indices
        sim_year: Year(s) to modify. Can be:
            - Single year (int)
            - Range of years (range object)
            - List of years
        value: New value(s) to set. Can be:
            - Single value applied to all specified cells
            - Array matching the shape of selected cells
        sim_month: Optional month(s) to modify (1-12). Only used for
            ForcedEffort which is monthly. If None, modifies all months.

    Returns:
        Modified scenario object

    Example:
        >>> # Double fishing mortality for 'Fish' group in years 10-20
        >>> scenario = adjust_fishing(
        ...     scenario,
        ...     parameter='ForcedFRate',
        ...     group='Fish',
        ...     sim_year=range(10, 21),
        ...     value=0.5
        ... )

        >>> # Set catch quota
        >>> scenario = adjust_fishing(
        ...     scenario,
        ...     parameter='ForcedCatch',
        ...     group=['Cod', 'Haddock'],
        ...     sim_year=2025,
        ...     value=100.0
        ... )
    """
    valid_params = ["ForcedEffort", "ForcedFRate", "ForcedCatch"]
    if parameter not in valid_params:
        raise ValueError(f"parameter must be one of {valid_params}")

    # Get the fishing matrix
    fishing_matrix = getattr(scenario.fishing, parameter)

    # Convert group to indices
    group_indices = _resolve_group_indices(scenario, group, parameter)

    # Convert years to row indices
    year_indices = _resolve_year_indices(scenario, sim_year, parameter)

    # For ForcedEffort (monthly), handle month selection
    if parameter == "ForcedEffort" and sim_month is not None:
        row_indices = _resolve_month_indices(
            year_indices, sim_month, fishing_matrix.shape[0]
        )
    else:
        row_indices = year_indices

    # Set values
    if np.isscalar(value):
        for gi in group_indices:
            for ri in row_indices:
                fishing_matrix[ri, gi] = value
    else:
        # Value is an array - must match shape
        value = np.asarray(value)
        if value.shape == (len(row_indices), len(group_indices)):
            for i, ri in enumerate(row_indices):
                for j, gi in enumerate(group_indices):
                    fishing_matrix[ri, gi] = value[i, j]
        elif value.shape == (len(row_indices),):
            # Broadcast across groups
            for gi in group_indices:
                for i, ri in enumerate(row_indices):
                    fishing_matrix[ri, gi] = value[i]
        else:
            raise ValueError(f"value shape {value.shape} doesn't match selection")

    return scenario


def adjust_forcing(
    scenario,
    parameter: str,
    group: Union[str, int, List[Union[str, int]]],
    sim_year: Union[int, range, List[int]],
    sim_month: Union[int, range, List[int]],
    value: Union[float, np.ndarray],
):
    """Adjust forcing parameters in an Ecosim scenario.

    Modifies environmental forcing matrices (ForcedPrey, ForcedMort,
    ForcedRecs, ForcedSearch, ForcedActresp, ForcedMigrate, ForcedBio)
    for specified groups and time periods.

    Args:
        scenario: RsimScenario object to modify
        parameter: One of:
            - 'ForcedPrey': Prey availability multiplier
            - 'ForcedMort': Additional mortality multiplier
            - 'ForcedRecs': Recruitment multiplier
            - 'ForcedSearch': Search rate multiplier
            - 'ForcedActresp': Active respiration multiplier
            - 'ForcedMigrate': Migration rate (additive)
            - 'ForcedBio': Biomass forcing (-1 = off)
        group: Group name(s) or index(es) to modify
        sim_year: Year(s) to modify
        sim_month: Month(s) to modify (1-12)
        value: New value(s) to set

    Returns:
        Modified scenario object

    Example:
        >>> # Reduce prey availability in summer
        >>> scenario = adjust_forcing(
        ...     scenario,
        ...     parameter='ForcedPrey',
        ...     group='Zooplankton',
        ...     sim_year=range(1, 51),
        ...     sim_month=[6, 7, 8],
        ...     value=0.8
        ... )

        >>> # Add pulse recruitment
        >>> scenario = adjust_forcing(
        ...     scenario,
        ...     parameter='ForcedRecs',
        ...     group='Fish',
        ...     sim_year=15,
        ...     sim_month=3,
        ...     value=2.0
        ... )
    """
    valid_params = [
        "ForcedPrey",
        "ForcedMort",
        "ForcedRecs",
        "ForcedSearch",
        "ForcedActresp",
        "ForcedMigrate",
        "ForcedBio",
    ]
    if parameter not in valid_params:
        raise ValueError(f"parameter must be one of {valid_params}")

    # Get the forcing matrix
    forcing_matrix = getattr(scenario.forcing, parameter)

    # Convert group to indices
    group_indices = _resolve_group_indices(scenario, group, parameter, is_forcing=True)

    # Convert years to row indices
    year_indices = _resolve_year_indices(scenario, sim_year, parameter)

    # Convert to monthly row indices
    row_indices = _resolve_month_indices(
        year_indices, sim_month, forcing_matrix.shape[0]
    )

    # Set values
    if np.isscalar(value):
        for gi in group_indices:
            for ri in row_indices:
                forcing_matrix[ri, gi] = value
    else:
        value = np.asarray(value)
        if value.shape == (len(row_indices), len(group_indices)):
            for i, ri in enumerate(row_indices):
                for j, gi in enumerate(group_indices):
                    forcing_matrix[ri, gi] = value[i, j]
        elif value.shape == (len(row_indices),):
            for gi in group_indices:
                for i, ri in enumerate(row_indices):
                    forcing_matrix[ri, gi] = value[i]
        else:
            raise ValueError(f"value shape {value.shape} doesn't match selection")

    return scenario


def adjust_scenario(scenario, parameter: str, value: Union[float, int, np.ndarray]):
    """Adjust global scenario parameters.

    Modifies simulation-wide parameters in the scenario's params object.

    Args:
        scenario: RsimScenario object to modify
        parameter: Parameter name to modify. Common options:
            - 'BURN_YEARS': Number of burn-in years (-1 = off)
            - 'COUPLED': Coupling flag (0 = uncoupled, 1 = coupled)
            - 'RK4_STEPS': Integration steps per month
            - 'SENSE_LIMIT': Sensitivity limits [min, max]
        value: New value to set

    Returns:
        Modified scenario object

    Example:
        >>> # Enable burn-in period
        >>> scenario = adjust_scenario(scenario, 'BURN_YEARS', 10)

        >>> # Change integration precision
        >>> scenario = adjust_scenario(scenario, 'RK4_STEPS', 8)
    """
    if hasattr(scenario.params, parameter):
        setattr(scenario.params, parameter, value)
    else:
        raise AttributeError(f"Parameter '{parameter}' not found in scenario.params")

    return scenario


def set_vulnerability(
    scenario, predator: Union[str, int], prey: Union[str, int], value: float
):
    """Set vulnerability (v) for a predator-prey link.

    Vulnerability controls the functional response shape:
    - v = 1: Linear (Type I)
    - v = 2: Holling Type II (default)
    - v > 2: Approaches Type III

    Args:
        scenario: RsimScenario object to modify
        predator: Predator group name or index
        prey: Prey group name or index
        value: New vulnerability value

    Returns:
        Modified scenario object
    """
    pred_idx = _get_group_index(scenario, predator)
    prey_idx = _get_group_index(scenario, prey)

    # Find the link
    params = scenario.params
    for i in range(1, params.NumPredPreyLinks + 1):
        if params.PreyTo[i] == pred_idx and params.PreyFrom[i] == prey_idx:
            params.VV[i] = value
            return scenario

    raise ValueError(f"No predator-prey link found between {predator} and {prey}")


def set_handling_time(
    scenario, predator: Union[str, int], prey: Union[str, int], value: float
):
    """Set handling time (d) for a predator-prey link.

    Handling time controls predator satiation:
    - d = 1000: Off (default)
    - d = 0: Maximum satiation effect

    Args:
        scenario: RsimScenario object to modify
        predator: Predator group name or index
        prey: Prey group name or index
        value: New handling time value

    Returns:
        Modified scenario object
    """
    pred_idx = _get_group_index(scenario, predator)
    prey_idx = _get_group_index(scenario, prey)

    # Find the link
    params = scenario.params
    for i in range(1, params.NumPredPreyLinks + 1):
        if params.PreyTo[i] == pred_idx and params.PreyFrom[i] == prey_idx:
            params.DD[i] = value
            return scenario

    raise ValueError(f"No predator-prey link found between {predator} and {prey}")


def adjust_group_parameter(
    scenario, group: Union[str, int], parameter: str, value: float
):
    """Adjust a parameter for a specific group.

    Modifies group-level parameters in the scenario's params object.

    Args:
        scenario: RsimScenario object to modify
        group: Group name or index
        parameter: Parameter name. Options include:
            - 'MzeroMort': Background mortality
            - 'UnassimRespFrac': Unassimilated fraction
            - 'ActiveRespFrac': Active respiration fraction
            - 'FtimeAdj': Feeding time adjustment
            - 'PBopt': Optimal P/B
        value: New value to set

    Returns:
        Modified scenario object
    """
    group_idx = _get_group_index(scenario, group) + 1  # +1 for "Outside"

    if hasattr(scenario.params, parameter):
        param_array = getattr(scenario.params, parameter)
        if isinstance(param_array, np.ndarray) and len(param_array) > group_idx:
            param_array[group_idx] = value
        else:
            raise ValueError(
                f"Parameter '{parameter}' not accessible at index {group_idx}"
            )
    else:
        raise AttributeError(f"Parameter '{parameter}' not found in scenario.params")

    return scenario


# Helper functions


def _get_group_index(scenario, group: Union[str, int]) -> int:
    """Get group index from name or index."""
    if isinstance(group, int):
        return group

    # Look up by name
    spname = scenario.params.spname
    for i, name in enumerate(spname):
        if name == group:
            return i - 1  # Subtract 1 because spname has "Outside" at index 0

    raise ValueError(f"Group '{group}' not found in scenario")


def _resolve_group_indices(
    scenario, group: Union[str, int, List], parameter: str, is_forcing: bool = False
) -> List[int]:
    """Resolve group specification to list of column indices."""
    if isinstance(group, (str, int)):
        groups = [group]
    else:
        groups = list(group)

    indices = []
    for g in groups:
        idx = _get_group_index(scenario, g)
        # Adjust for matrix structure
        if is_forcing:
            indices.append(idx + 1)  # Forcing matrices include "Outside"
        elif parameter == "ForcedEffort":
            # Effort is indexed by gear, starts after biomass groups
            if isinstance(g, str):
                # Find gear index
                spname = scenario.params.spname
                gear_start = scenario.params.NUM_LIVING + scenario.params.NUM_DEAD + 1
                for i, name in enumerate(spname[gear_start:], gear_start):
                    if name == g:
                        indices.append(i - gear_start + 1)
                        break
                else:
                    raise ValueError(f"Gear '{g}' not found")
            else:
                indices.append(g + 1)
        else:
            indices.append(idx + 1)

    return indices


def _resolve_year_indices(
    scenario, sim_year: Union[int, range, List[int]], parameter: str
) -> List[int]:
    """Resolve year specification to list of row indices."""
    if isinstance(sim_year, int):
        years = [sim_year]
    elif isinstance(sim_year, range):
        years = list(sim_year)
    else:
        years = list(sim_year)

    # Get year labels from fishing matrix row names
    if parameter in ["ForcedEffort"]:
        # Monthly matrix - get base years
        n_years = scenario.fishing.ForcedEffort.shape[0] // 12
        start_year = 1  # Assume 1-based years
        return [y - start_year for y in years]
    else:
        # Annual matrix
        return [y - 1 for y in years]  # Convert to 0-based


def _resolve_month_indices(
    year_indices: List[int], sim_month: Union[int, range, List[int], None], n_rows: int
) -> List[int]:
    """Convert year and month to row indices for monthly matrices."""
    if sim_month is None:
        # All months
        months = list(range(1, 13))
    elif isinstance(sim_month, int):
        months = [sim_month]
    elif isinstance(sim_month, range):
        months = list(sim_month)
    else:
        months = list(sim_month)

    row_indices = []
    for y in year_indices:
        for m in months:
            row_idx = y * 12 + (m - 1)  # Convert to 0-based month
            if 0 <= row_idx < n_rows:
                row_indices.append(row_idx)

    return row_indices


def create_fishing_ramp(
    scenario,
    group: Union[str, int],
    start_year: int,
    end_year: int,
    start_value: float,
    end_value: float,
    parameter: str = "ForcedFRate",
):
    """Create a linear ramp in fishing pressure.

    Convenience function to linearly interpolate fishing between
    two values over a range of years.

    Args:
        scenario: RsimScenario object to modify
        group: Group to modify
        start_year: First year of ramp
        end_year: Last year of ramp
        start_value: Value at start_year
        end_value: Value at end_year
        parameter: Fishing parameter to modify

    Returns:
        Modified scenario object
    """
    years = list(range(start_year, end_year + 1))
    values = np.linspace(start_value, end_value, len(years))

    return adjust_fishing(
        scenario, parameter=parameter, group=group, sim_year=years, value=values
    )


def create_pulse_forcing(
    scenario,
    group: Union[str, int],
    pulse_years: List[int],
    pulse_months: Union[int, List[int]],
    magnitude: float,
    parameter: str = "ForcedRecs",
):
    """Create pulse forcing events.

    Convenience function to add periodic pulse events
    (e.g., recruitment pulses, mortality events).

    Args:
        scenario: RsimScenario object to modify
        group: Group to modify
        pulse_years: List of years with pulse events
        pulse_months: Month(s) when pulse occurs
        magnitude: Multiplier for pulse (>1 = increase, <1 = decrease)
        parameter: Forcing parameter to modify

    Returns:
        Modified scenario object
    """
    for year in pulse_years:
        scenario = adjust_forcing(
            scenario,
            parameter=parameter,
            group=group,
            sim_year=year,
            sim_month=pulse_months,
            value=magnitude,
        )

    return scenario


def create_seasonal_forcing(
    scenario,
    group: Union[str, int],
    years: Union[range, List[int]],
    monthly_values: List[float],
    parameter: str = "ForcedPrey",
):
    """Create seasonal forcing pattern.

    Applies a repeating 12-month pattern of forcing values
    across multiple years.

    Args:
        scenario: RsimScenario object to modify
        group: Group to modify
        years: Years to apply pattern
        monthly_values: List of 12 values, one per month
        parameter: Forcing parameter to modify

    Returns:
        Modified scenario object

    Example:
        >>> # Higher prey availability in summer
        >>> seasonal = [0.8, 0.9, 1.0, 1.1, 1.2, 1.3,
        ...             1.3, 1.2, 1.1, 1.0, 0.9, 0.8]
        >>> scenario = create_seasonal_forcing(
        ...     scenario, 'Zooplankton', range(1, 51), seasonal
        ... )
    """
    if len(monthly_values) != 12:
        raise ValueError("monthly_values must have exactly 12 elements")

    if isinstance(years, range):
        years = list(years)

    for month in range(1, 13):
        scenario = adjust_forcing(
            scenario,
            parameter=parameter,
            group=group,
            sim_year=years,
            sim_month=month,
            value=monthly_values[month - 1],
        )

    return scenario
