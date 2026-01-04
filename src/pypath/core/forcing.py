"""
Advanced forcing mechanisms for Ecosim simulations.

This module provides:
1. State-variable forcing (biomass, catch, etc.)
2. Dynamic diet rewiring based on prey availability
3. Flexible forcing modes (replace, add, multiply)
4. Temporal interpolation for sub-annual time steps
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional, Union

import numpy as np
import pandas as pd


class ForcingMode(Enum):
    """Mode for applying forced values."""

    REPLACE = "replace"  # Replace state variable with forced value
    ADD = "add"  # Add forced value to computed value
    MULTIPLY = "multiply"  # Multiply computed value by forced value
    RESCALE = "rescale"  # Rescale to match forced value


class StateVariable(Enum):
    """State variables that can be forced."""

    BIOMASS = "biomass"
    CATCH = "catch"
    FISHING_MORTALITY = "fishing_mortality"
    RECRUITMENT = "recruitment"
    MORTALITY = "mortality"
    MIGRATION = "migration"
    PRIMARY_PRODUCTION = "primary_production"


@dataclass
class ForcingFunction:
    """Single forcing function for a state variable.

    Attributes
    ----------
    group_idx : int
        Index of group to force (-1 for all groups)
    variable : StateVariable
        Which state variable to force
    mode : ForcingMode
        How to apply the forcing
    time_series : np.ndarray
        Time series of forced values (years)
    years : np.ndarray
        Year indices corresponding to time_series
    interpolate : bool
        Whether to interpolate between annual values
    active : bool
        Whether this forcing is currently active
    """

    group_idx: int
    variable: StateVariable
    mode: ForcingMode
    time_series: np.ndarray
    years: np.ndarray
    interpolate: bool = True
    active: bool = True

    def get_value(self, year: float) -> float:
        """Get forced value at given year (with interpolation).

        Parameters
        ----------
        year : float
            Simulation year (can be fractional for monthly time steps)

        Returns
        -------
        float
            Forced value at this time
        """
        if not self.active:
            return np.nan

        # Check if year is in range
        if year < self.years[0] or year > self.years[-1]:
            # Outside range - return NaN (no forcing)
            return np.nan

        if self.interpolate:
            # Linear interpolation
            return np.interp(year, self.years, self.time_series)
        else:
            # Use nearest year
            idx = np.argmin(np.abs(self.years - year))
            return self.time_series[idx]


@dataclass
class StateForcing:
    """Collection of forcing functions for state variables.

    Attributes
    ----------
    functions : list[ForcingFunction]
        List of individual forcing functions
    """

    functions: List[ForcingFunction] = field(default_factory=list)

    def add_forcing(
        self,
        group_idx: int,
        variable: Union[str, StateVariable],
        time_series: Union[np.ndarray, pd.Series, Dict[int, float]],
        years: Optional[np.ndarray] = None,
        mode: Union[str, ForcingMode] = ForcingMode.REPLACE,
        interpolate: bool = True,
    ):
        """Add a forcing function.

        Parameters
        ----------
        group_idx : int
            Index of group to force
        variable : str or StateVariable
            Which state variable to force
        time_series : array-like or dict
            Time series of forced values
            If dict, keys are years and values are forced values
        years : array-like, optional
            Year indices (required if time_series is array)
        mode : str or ForcingMode
            How to apply forcing ("replace", "add", "multiply", "rescale")
        interpolate : bool
            Whether to interpolate between annual values

        Examples
        --------
        >>> forcing = StateForcing()
        >>> # Force phytoplankton biomass to observed series
        >>> forcing.add_forcing(
        ...     group_idx=0,
        ...     variable='biomass',
        ...     time_series={2000: 15.0, 2001: 18.0, 2002: 16.0},
        ...     mode='replace'
        ... )
        >>>
        >>> # Add recruitment pulse for herring in 2005
        >>> forcing.add_forcing(
        ...     group_idx=3,
        ...     variable='recruitment',
        ...     time_series={2005: 2.5},  # 2.5x normal recruitment
        ...     mode='multiply'
        ... )
        """
        # Convert variable to enum
        if isinstance(variable, str):
            variable = StateVariable(variable.lower())

        # Convert mode to enum
        if isinstance(mode, str):
            mode = ForcingMode(mode.lower())

        # Handle dict input
        if isinstance(time_series, dict):
            years = np.array(list(time_series.keys()))
            time_series = np.array(list(time_series.values()))
        elif isinstance(time_series, pd.Series):
            years = time_series.index.values
            time_series = time_series.values
        else:
            time_series = np.asarray(time_series)

        if years is None:
            raise ValueError("years must be provided if time_series is not a dict")

        years = np.asarray(years)

        # Create forcing function
        func = ForcingFunction(
            group_idx=group_idx,
            variable=variable,
            mode=mode,
            time_series=time_series,
            years=years,
            interpolate=interpolate,
            active=True,
        )

        self.functions.append(func)

    def get_forcing(
        self, year: float, variable: StateVariable, group_idx: Optional[int] = None
    ) -> List[Tuple[ForcingFunction, float]]:
        """Get all active forcing values for a variable at given time.

        Parameters
        ----------
        year : float
            Current simulation year
        variable : StateVariable
            Which state variable to query
        group_idx : int, optional
            Specific group index (None = all groups)

        Returns
        -------
        list of (ForcingFunction, float)
            List of (forcing function, forced value) tuples
        """
        results = []

        for func in self.functions:
            if not func.active or func.variable != variable:
                continue

            # Check if this forcing applies to the group
            if group_idx is not None:
                if func.group_idx != -1 and func.group_idx != group_idx:
                    continue

            value = func.get_value(year)
            if not np.isnan(value):
                results.append((func, value))

        return results

    def remove_forcing(self, group_idx: int, variable: Union[str, StateVariable]):
        """Remove forcing for a specific group and variable.

        Parameters
        ----------
        group_idx : int
            Index of group
        variable : str or StateVariable
            Which state variable
        """
        if isinstance(variable, str):
            variable = StateVariable(variable.lower())

        self.functions = [
            f
            for f in self.functions
            if not (f.group_idx == group_idx and f.variable == variable)
        ]


@dataclass
class DietRewiring:
    """Dynamic diet matrix rewiring based on prey availability.

    Allows predator diet preferences to change over time in response to
    changing prey abundance (prey switching, adaptive foraging).

    Attributes
    ----------
    enabled : bool
        Whether diet rewiring is active
    switching_power : float
        Prey switching exponent (higher = more switching)
    min_proportion : float
        Minimum diet proportion to maintain (prevents division by zero)
    update_interval : int
        How often to update diet (in months)
    base_diet : np.ndarray
        Original diet matrix (n_prey x n_pred)
    current_diet : np.ndarray
        Current diet matrix (updated each interval)
    """

    enabled: bool = False
    switching_power: float = 2.0
    min_proportion: float = 0.001
    update_interval: int = 12  # Monthly
    base_diet: Optional[np.ndarray] = None
    current_diet: Optional[np.ndarray] = None

    def initialize(self, diet_matrix: np.ndarray):
        """Initialize with base diet matrix.

        Parameters
        ----------
        diet_matrix : np.ndarray
            Base diet proportions (n_prey x n_pred)
        """
        self.base_diet = diet_matrix.copy()
        self.current_diet = diet_matrix.copy()

    def update_diet(
        self, prey_biomass: np.ndarray, predator_idx: Optional[int] = None
    ) -> np.ndarray:
        """Update diet preferences based on prey availability.

        Uses a prey switching model where diet preferences shift toward
        more abundant prey species.

        Parameters
        ----------
        prey_biomass : np.ndarray
            Current biomass of all prey groups
        predator_idx : int, optional
            Update only this predator (None = update all)

        Returns
        -------
        np.ndarray
            Updated diet matrix

        Notes
        -----
        The prey switching model:

        new_diet[prey, pred] = base_diet[prey, pred] * (biomass[prey] / B_ref[prey])^power

        Then normalize so sum of diet = 1 for each predator.

        Higher switching_power = stronger response to biomass changes.
        """
        if not self.enabled:
            return None

        if self.base_diet is None:
            return self.current_diet

        n_prey, n_pred = self.base_diet.shape
        new_diet = self.current_diet.copy()

        # Calculate relative prey availability
        # Only use first n_prey entries (matching diet matrix)
        prey_biomass_subset = prey_biomass[:n_prey]
        prey_availability = np.maximum(prey_biomass_subset, self.min_proportion)

        # Update diet for specified predator(s)
        pred_range = range(n_pred) if predator_idx is None else [predator_idx]

        for pred in pred_range:
            # Get base diet for this predator
            base_prefs = self.base_diet[:, pred]

            # Only update where there's a base preference
            active_prey = base_prefs > self.min_proportion

            if not np.any(active_prey):
                continue

            # Apply prey switching model
            # new_pref = base_pref * (availability)^power
            new_prefs = base_prefs.copy()
            new_prefs[active_prey] = base_prefs[active_prey] * np.power(
                prey_availability[active_prey]
                / np.mean(prey_availability[active_prey]),
                self.switching_power,
            )

            # Ensure minimum proportions
            new_prefs = np.maximum(new_prefs, self.min_proportion)

            # Normalize to sum to 1
            total = np.sum(new_prefs)
            if total > 0:
                new_diet[:, pred] = new_prefs / total

        self.current_diet = new_diet
        return new_diet

    def reset(self):
        """Reset diet to base values."""
        if self.base_diet is not None:
            self.current_diet = self.base_diet.copy()


def create_biomass_forcing(
    group_idx: int,
    observed_biomass: Union[np.ndarray, pd.Series, Dict[int, float]],
    years: Optional[np.ndarray] = None,
    mode: str = "replace",
    interpolate: bool = True,
) -> StateForcing:
    """Convenience function to create biomass forcing.

    Parameters
    ----------
    group_idx : int
        Index of group to force
    observed_biomass : array-like or dict
        Observed biomass time series
    years : array-like, optional
        Year indices
    mode : str
        Forcing mode ("replace", "add", "multiply")
    interpolate : bool
        Whether to interpolate monthly values

    Returns
    -------
    StateForcing
        Forcing object ready to use

    Examples
    --------
    >>> # Force phytoplankton to observed biomass
    >>> forcing = create_biomass_forcing(
    ...     group_idx=0,
    ...     observed_biomass={2000: 15.0, 2005: 18.0, 2010: 16.0},
    ...     mode='replace'
    ... )
    """
    forcing = StateForcing()
    forcing.add_forcing(
        group_idx=group_idx,
        variable=StateVariable.BIOMASS,
        time_series=observed_biomass,
        years=years,
        mode=mode,
        interpolate=interpolate,
    )
    return forcing


def create_recruitment_forcing(
    group_idx: int,
    recruitment_multiplier: Union[np.ndarray, Dict[int, float]],
    years: Optional[np.ndarray] = None,
    interpolate: bool = False,
) -> StateForcing:
    """Convenience function to create recruitment forcing.

    Parameters
    ----------
    group_idx : int
        Index of group to force
    recruitment_multiplier : array-like or dict
        Recruitment multiplier (1.0 = normal, 2.0 = double, etc.)
    years : array-like, optional
        Year indices
    interpolate : bool
        Whether to interpolate (usually False for recruitment pulses)

    Returns
    -------
    StateForcing
        Forcing object ready to use

    Examples
    --------
    >>> # Strong recruitment in 2005, weak in 2010
    >>> forcing = create_recruitment_forcing(
    ...     group_idx=3,
    ...     recruitment_multiplier={2005: 3.0, 2010: 0.5}
    ... )
    """
    forcing = StateForcing()
    forcing.add_forcing(
        group_idx=group_idx,
        variable=StateVariable.RECRUITMENT,
        time_series=recruitment_multiplier,
        years=years,
        mode=ForcingMode.MULTIPLY,
        interpolate=interpolate,
    )
    return forcing


def create_diet_rewiring(
    switching_power: float = 2.0,
    min_proportion: float = 0.001,
    update_interval: int = 12,
) -> DietRewiring:
    """Convenience function to create diet rewiring configuration.

    Parameters
    ----------
    switching_power : float
        Prey switching exponent (1.0 = proportional, >1 = switching)
    min_proportion : float
        Minimum diet proportion to maintain
    update_interval : int
        How often to update diet (months)

    Returns
    -------
    DietRewiring
        Diet rewiring object ready to use

    Examples
    --------
    >>> # Enable strong prey switching
    >>> rewiring = create_diet_rewiring(switching_power=3.0)
    """
    return DietRewiring(
        enabled=True,
        switching_power=switching_power,
        min_proportion=min_proportion,
        update_interval=update_interval,
    )


# Export main classes and functions
__all__ = [
    "ForcingMode",
    "StateVariable",
    "ForcingFunction",
    "StateForcing",
    "DietRewiring",
    "create_biomass_forcing",
    "create_recruitment_forcing",
    "create_diet_rewiring",
]
