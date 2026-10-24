"""Monkey-patching integration for environmental forcing in RsimScenario.

This module provides automatic patching of the RsimScenario class to support
environmental forcing mechanisms (Week 2 Tasks 2.1-2.3). When imported, it
automatically adds methods to RsimScenario for managing and applying
environmental forcings during ecosystem simulations.

The patching adds the following methods to RsimScenario:
- set_environmental_forcing: Register an environmental forcing
- get_environmental_forcing: Retrieve a registered forcing
- list_environmental_forcings: List all registered forcings
- apply_environmental_forcing: Evaluate forcings at a given time
- compute_habitat_suitability: Calculate habitat suitability from oceanographic data
- get_forcing_trajectory: Batch evaluate forcings across time points

Example
-------
>>> from pypath.core.ecosim import RsimScenario
>>> from pypath.core.environmental_forcing import HabitatProductivity
>>> # After import, RsimScenario has environmental forcing methods
>>> scenario.set_environmental_forcing("habitat", habitat_forcing)
>>> factors = scenario.apply_environmental_forcing(time=100.0)
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Dict, List, Optional, Union

import numpy as np

if TYPE_CHECKING:
    from pypath.core.environmental_forcing import EnvironmentalForcing
    from pypath.core.ecosim import RsimScenario

import logging

logger = logging.getLogger(__name__)


def add_environmental_forcing_support() -> None:
    """Auto-patch RsimScenario with environmental forcing methods.
    
    This function adds six new methods to RsimScenario class to enable
    management and application of environmental forcings during simulation.
    The patching happens at module import time and is non-destructive—it
    does not override existing methods.
    
    Methods Added
    ---------------
    set_environmental_forcing : method
        Register an environmental forcing with the scenario.
    get_environmental_forcing : method
        Retrieve a previously registered forcing.
    list_environmental_forcings : method
        Get list of all registered forcing names.
    apply_environmental_forcing : method
        Evaluate all active forcings at a given time.
    compute_habitat_suitability : method
        Compute habitat suitability from oceanographic conditions.
    get_forcing_trajectory : method
        Batch evaluate forcing across multiple time points.
    
    Notes
    -----
    - Patching is idempotent; calling multiple times has no additional effect
    - Each RsimScenario instance gets its own _environmental_forcings dict
    - All methods validate inputs and provide clear error messages
    - Graceful fallback behavior (None/1.0) for missing forcings
    
    Raises
    ------
    ImportError
        If RsimScenario cannot be imported (should not happen in normal use).
    """
    try:
        from pypath.core.ecosim import RsimScenario
    except ImportError as e:
        logger.error(f"Could not import RsimScenario: {e}")
        raise

    # Skip patching if already done
    if hasattr(RsimScenario, 'set_environmental_forcing'):
        logger.debug("RsimScenario already patched with environmental forcing support")
        return

    def _init_environmental_forcings(self: RsimScenario) -> None:
        """Initialize environmental forcings dictionary if not present."""
        if not hasattr(self, '_environmental_forcings'):
            self._environmental_forcings: Dict[str, EnvironmentalForcing] = {}

    def set_environmental_forcing(
        self: RsimScenario,
        forcing_name: str,
        forcing: EnvironmentalForcing
    ) -> None:
        """Register an environmental forcing with the scenario.
        
        Stores the forcing object for later retrieval and application during
        simulation. Can override existing forcings with the same name.
        
        Parameters
        ----------
        forcing_name : str
            Unique identifier for this forcing (e.g., 'habitat_productivity',
            'oceanographic_conditions'). Must be non-empty.
        forcing : EnvironmentalForcing
            Environmental forcing object (must be instance of EnvironmentalForcing
            base class from pypath.core.environmental_forcing).
        
        Returns
        -------
        None
        
        Raises
        ------
        ValueError
            If forcing_name is empty or None.
        TypeError
            If forcing is not an instance of EnvironmentalForcing.
        
        Examples
        --------
        >>> from pypath.core.environmental_forcing import HabitatProductivity
        >>> habitat = HabitatProductivity(base_productivity=1.0)
        >>> scenario.set_environmental_forcing('habitat', habitat)
        
        Notes
        -----
        - Forcing names should be descriptive and unique per scenario
        - Setting a forcing with an existing name overwrites the previous one
        - Forcing is stored internally and evaluated on demand
        """
        # Validate inputs
        if not forcing_name or not isinstance(forcing_name, str):
            raise ValueError(
                f"forcing_name must be a non-empty string, got {forcing_name!r}"
            )
        
        forcing_name = forcing_name.strip()
        if not forcing_name:
            raise ValueError("forcing_name cannot be empty or whitespace-only")
        
        # Import here to avoid circular dependency
        from pypath.core.environmental_forcing import EnvironmentalForcing
        
        if not isinstance(forcing, EnvironmentalForcing):
            raise TypeError(
                f"forcing must be instance of EnvironmentalForcing, "
                f"got {type(forcing).__name__}"
            )
        
        # Initialize dictionary if needed
        _init_environmental_forcings(self)
        
        # Store the forcing
        self._environmental_forcings[forcing_name] = forcing
        logger.debug(f"Registered environmental forcing: {forcing_name}")

    def get_environmental_forcing(
        self: RsimScenario,
        forcing_name: str
    ) -> Optional[EnvironmentalForcing]:
        """Retrieve a previously registered environmental forcing.
        
        Safely retrieves an environmental forcing by name. Returns None if
        the forcing is not registered, allowing graceful handling of missing
        forcings without exceptions.
        
        Parameters
        ----------
        forcing_name : str
            Name of the forcing to retrieve (must match exactly as registered).
        
        Returns
        -------
        EnvironmentalForcing or None
            The forcing object if found, None otherwise.
        
        Raises
        ------
        TypeError
            If forcing_name is not a string.
        
        Examples
        --------
        >>> forcing = scenario.get_environmental_forcing('habitat')
        >>> if forcing is not None:
        ...     factor = forcing.get_scaling_factor(time=100.0)
        
        Notes
        -----
        - Name matching is case-sensitive
        - Returns None rather than raising exception for missing forcings
        - Type-safe retrieval with no runtime errors
        """
        if not isinstance(forcing_name, str):
            raise TypeError(
                f"forcing_name must be a string, got {type(forcing_name).__name__}"
            )
        
        # Initialize dictionary if needed
        _init_environmental_forcings(self)
        
        # Retrieve forcings (graceful None if not found)
        return self._environmental_forcings.get(forcing_name, None)

    def list_environmental_forcings(self: RsimScenario) -> List[str]:
        """Get list of all registered environmental forcing names.
        
        Returns a list of all forcing names currently registered with this
        scenario. Useful for introspection and validation.
        
        Returns
        -------
        list of str
            Names of all registered forcings, sorted alphabetically.
            Empty list if no forcings are registered.
        
        Examples
        --------
        >>> forcings = scenario.list_environmental_forcings()
        >>> print(f"Active forcings: {', '.join(forcings)}")
        Active forcings: climate_forcing, habitat_productivity
        
        Notes
        -----
        - Returns empty list if scenario has no forcings, never None
        - Names are sorted alphabetically for consistent ordering
        - This is a read-only operation with no side effects
        """
        # Initialize dictionary if needed
        _init_environmental_forcings(self)
        
        return sorted(self._environmental_forcings.keys())

    def apply_environmental_forcing(
        self: RsimScenario,
        time: float
    ) -> Dict[str, float]:
        """Evaluate all active environmental forcings at a given time.
        
        Computes scaling factors for all registered forcings at the specified
        time point. Results are used by deriv_vector() to modify ecosystem
        dynamics (growth rates, recruitment, mortality, etc.).
        
        This method is typically called during integration steps to get current
        forcing values based on time-dependent environmental conditions.
        
        Parameters
        ----------
        time : float
            Time point for evaluation (interpretation depends on context;
            typically in years or months).
        
        Returns
        -------
        dict
            Mapping of forcing names to scaling factors. Each scaling factor
            is a float (typically in range [0.5, 1.5], with 1.0 as neutral).
            Empty dict if no forcings are registered.
        
        Raises
        ------
        TypeError
            If time is not numeric.
        
        Examples
        --------
        >>> factors = scenario.apply_environmental_forcing(time=50.0)
        >>> for name, factor in factors.items():
        ...     print(f"{name}: {factor:.3f}")
        habitat_productivity: 1.050
        oceanographic_forcing: 0.950
        
        Notes
        -----
        - Each forcing independently evaluates to a scalar value
        - Returned factors are not combined; caller decides aggregation
        - Empty dict is returned if no forcings registered (safe)
        - Non-numeric time values will raise TypeError
        - Used in deriv_vector to scale rates by forcing factors
        """
        # Validate input
        try:
            time = float(time)
        except (TypeError, ValueError):
            raise TypeError(
                f"time must be numeric, got {type(time).__name__}: {time!r}"
            )
        
        # Initialize dictionary if needed
        _init_environmental_forcings(self)
        
        # Evaluate all forcings at this time
        factors: Dict[str, float] = {}
        for forcing_name, forcing in self._environmental_forcings.items():
            try:
                factor = forcing.get_scaling_factor(time)
                factors[forcing_name] = float(factor)
            except Exception as e:
                logger.warning(
                    f"Error evaluating forcing '{forcing_name}' at time {time}: {e}"
                )
                # Skip this forcing and continue with others
                continue
        
        return factors

    def compute_habitat_suitability(
        self: RsimScenario,
        group: str,
        time: float
    ) -> float:
        """Compute habitat suitability index from oceanographic forcing.
        
        Routes to registered OceanographicForcing if available, computing
        a suitability index based on environmental conditions. Falls back to
        neutral value (1.0) if no oceanographic forcing is registered.
        
        This method is used to compute spatially-implicit habitat suitability
        that affects settlement success, recruitment rates, and movement rates
        for a given functional group.
        
        Parameters
        ----------
        group : str
            Name or identifier of the functional group. Currently used for
            future expansion; may enable group-specific responses to environment.
        time : float
            Time point for evaluation.
        
        Returns
        -------
        float
            Habitat suitability index (range [0, 1], where 1 is optimal).
            Returns 1.0 (neutral) if no oceanographic forcing available.
        
        Raises
        ------
        TypeError
            If group is not a string or time is not numeric.
        
        Examples
        --------
        >>> suitability = scenario.compute_habitat_suitability('Cod', time=100.0)
        >>> print(f"Habitat suitability: {suitability:.2%}")
        Habitat suitability: 87.50%
        
        Notes
        -----
        - Graceful fallback: returns 1.0 if oceanographic forcing unavailable
        - Group parameter may be used for species-specific thermal responses
        - Suitability is independent of other forcings
        - Range is always [0, 1] (unlike scaling factors which can exceed 1.0)
        - Typically called during recruitment or settlement calculations
        """
        # Validate inputs
        if not isinstance(group, str):
            raise TypeError(
                f"group must be a string, got {type(group).__name__}"
            )
        
        try:
            time = float(time)
        except (TypeError, ValueError):
            raise TypeError(
                f"time must be numeric, got {type(time).__name__}: {time!r}"
            )
        
        # Initialize dictionary if needed
        _init_environmental_forcings(self)
        
        # Look for oceanographic forcing
        oceanographic = self._environmental_forcings.get('oceanographic_forcing', None)
        
        if oceanographic is None:
            # Try alternative name
            oceanographic = self._environmental_forcings.get('oceanographic_conditions', None)
        
        if oceanographic is None:
            # No oceanographic data; return neutral suitability
            logger.debug(
                f"No oceanographic forcing for group '{group}' at time {time}; "
                f"returning neutral suitability (1.0)"
            )
            return 1.0
        
        # Use oceanographic forcing's habitat suitability method if available
        if hasattr(oceanographic, 'compute_habitat_suitability'):
            try:
                suitability = oceanographic.compute_habitat_suitability(
                    group=group, time=time
                )
                return float(np.clip(suitability, 0.0, 1.0))
            except Exception as e:
                logger.warning(
                    f"Error computing habitat suitability for '{group}': {e}; "
                    f"returning neutral (1.0)"
                )
                return 1.0
        else:
            # Oceanographic forcing doesn't have suitability method
            logger.debug(
                f"Oceanographic forcing does not support compute_habitat_suitability; "
                f"returning neutral (1.0)"
            )
            return 1.0

    def get_forcing_trajectory(
        self: RsimScenario,
        forcing_name: str,
        times: np.ndarray
    ) -> np.ndarray:
        """Batch evaluate a forcing across multiple time points.
        
        Efficiently evaluates a single forcing at many time points, returning
        an array of scaling factors. Useful for visualization, analysis, and
        diagnostics of forcing behavior over simulation period.
        
        Parameters
        ----------
        forcing_name : str
            Name of the forcing to evaluate (must be registered).
        times : np.ndarray
            Array of time points (1D array of floats).
        
        Returns
        -------
        np.ndarray
            Array of scaling factors with same shape as input times.
            All values are floats (typically in range [0.5, 1.5]).
        
        Raises
        ------
        ValueError
            If forcing_name is not registered.
        TypeError
            If times is not a numpy array or array-like.
        
        Examples
        --------
        >>> times = np.linspace(0, 100, 1000)
        >>> trajectory = scenario.get_forcing_trajectory('habitat', times)
        >>> plt.plot(times, trajectory)
        >>> plt.xlabel('Time (years)')
        >>> plt.ylabel('Habitat Productivity Forcing')
        
        Notes
        -----
        - Returns empty array if forcing not found (raises ValueError)
        - Input times array is converted to numpy array if needed
        - Each forcing is evaluated independently at each time
        - Useful for diagnostics: "How did this forcing vary during simulation?"
        - Can be memory-intensive for very large time arrays
        """
        # Validate inputs
        if not isinstance(forcing_name, str):
            raise TypeError(
                f"forcing_name must be a string, got {type(forcing_name).__name__}"
            )
        
        try:
            times = np.asarray(times, dtype=float)
        except (TypeError, ValueError):
            raise TypeError(
                f"times must be array-like of numeric values, got {type(times)}"
            )
        
        if times.ndim != 1:
            raise ValueError(
                f"times must be 1D array, got shape {times.shape}"
            )
        
        # Get the forcing
        forcing = self.get_environmental_forcing(forcing_name)
        
        if forcing is None:
            raise ValueError(
                f"Environmental forcing '{forcing_name}' not registered. "
                f"Available forcings: {self.list_environmental_forcings()}"
            )
        
        # Evaluate at each time point
        try:
            trajectory = np.array([
                forcing.get_scaling_factor(t) for t in times
            ], dtype=float)
            return trajectory
        except Exception as e:
            logger.error(
                f"Error evaluating forcing '{forcing_name}' trajectory: {e}"
            )
            raise RuntimeError(
                f"Failed to evaluate forcing '{forcing_name}' across trajectory: {e}"
            ) from e

    # Attach methods to RsimScenario
    RsimScenario.set_environmental_forcing = set_environmental_forcing
    RsimScenario.get_environmental_forcing = get_environmental_forcing
    RsimScenario.list_environmental_forcings = list_environmental_forcings
    RsimScenario.apply_environmental_forcing = apply_environmental_forcing
    RsimScenario.compute_habitat_suitability = compute_habitat_suitability
    RsimScenario.get_forcing_trajectory = get_forcing_trajectory

    logger.info("Successfully patched RsimScenario with environmental forcing support")


# Auto-execute patching on module import
add_environmental_forcing_support()
