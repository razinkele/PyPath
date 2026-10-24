"""Ramping support for Ecosim scenarios.

Extends RsimScenario with time-varying parameter capabilities.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Dict, List, Tuple

import numpy as np

if TYPE_CHECKING:
    from pypath.core.ramping import RampingFunction

from pypath.core.ecosim import RsimScenario


def add_ramping_support() -> None:
    """Monkey-patch RsimScenario with ramping methods."""
    
    def __init_ramping__(self) -> None:
        """Initialize ramping dictionary."""
        if not hasattr(self, "ramps"):
            self.ramps: Dict[Tuple[str, str], "RampingFunction"] = {}
    
    def set_parameter_ramp(
        self, group_name: str, parameter: str, ramp: "RampingFunction"
    ) -> None:
        """Set ramping function for parameter.
        
        Parameters
        ----------
        group_name : str
            Name of ecological group
        parameter : str
            Parameter name (e.g., 'M2', 'K', 'q')
        ramp : RampingFunction
            Ramping function to apply
        """
        self.__init_ramping__()
        if group_name not in self.params.spname:
            raise ValueError(f"Unknown group: {group_name}")
        self.ramps[(group_name, parameter)] = ramp
    
    def get_parameter_value(
        self, group_name: str, parameter: str, time: float
    ) -> float:
        """Get parameter at time, accounting for ramps.
        
        Parameters
        ----------
        group_name : str
        parameter : str
        time : float
            Time in years
        
        Returns
        -------
        float
            Ramp-adjusted parameter value (multiplier form)
        """
        self.__init_ramping__()
        key = (group_name, parameter)
        if key in self.ramps:
            return self.ramps[key].evaluate(time)
        return 1.0  # No ramp = multiplier of 1.0
    
    def get_ramp_trajectory(
        self, group_name: str, parameter: str, times: np.ndarray
    ) -> np.ndarray:
        """Get parameter trajectory over multiple times."""
        self.__init_ramping__()
        key = (group_name, parameter)
        if key not in self.ramps:
            raise ValueError(f"No ramp defined for {group_name}.{parameter}")
        return self.ramps[key].get_trajectory(times)
    
    def list_ramps(self) -> List[Tuple[str, str, str]]:
        """List all active ramps (group, parameter, ramp_type)."""
        self.__init_ramping__()
        result = []
        for (group, param), ramp in self.ramps.items():
            ramp_type = type(ramp).__name__
            result.append((group, param, ramp_type))
        return result
    
    # Add methods to RsimScenario
    RsimScenario.__init_ramping__ = __init_ramping__
    RsimScenario.set_parameter_ramp = set_parameter_ramp
    RsimScenario.get_parameter_value = get_parameter_value
    RsimScenario.get_ramp_trajectory = get_ramp_trajectory
    RsimScenario.list_ramps = list_ramps


# Auto-apply patches on module import
add_ramping_support()