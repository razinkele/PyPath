"""
Advanced Ecosim entry point: state forcing and diet rewiring (not implemented).

``rsim_run_advanced`` runs ``rsim_run``. State forcing (``StateForcing``)
and diet rewiring (``DietRewiring``) are not implemented in this version and
raise ``NotImplementedError``. Earlier versions ran a placeholder loop that
never integrated, so they returned the start biomass in every month. Prey
switching, EwE's per-predator ``SwitchPower``, is available instead of diet
rewiring: set ``scenario.params.SwitchPower``, which ``rsim_run`` applies, or
use the Prey Switching control on the app's Ecosim page.

``apply_state_forcing`` and ``apply_diet_rewiring`` are standalone helpers
that no simulation calls.
"""

from __future__ import annotations

import logging
from typing import Callable, Optional

import numpy as np

from pypath.core.ecosim import RsimOutput, RsimScenario, rsim_run
from pypath.core.forcing import DietRewiring, ForcingMode, StateForcing, StateVariable

# Get logger
logger = logging.getLogger(__name__)


def apply_state_forcing(
    state: np.ndarray,
    year: float,
    state_forcing: Optional[StateForcing],
    variable: StateVariable = StateVariable.BIOMASS,
) -> np.ndarray:
    """Apply state forcing to current state vector.

    Parameters
    ----------
    state : np.ndarray
        Current state vector (biomass, catch, etc.)
    year : float
        Current simulation year (fractional for monthly steps)
    state_forcing : StateForcing or None
        Forcing configuration
    variable : StateVariable
        Which variable is being forced

    Returns
    -------
    np.ndarray
        Modified state vector
    """
    if state_forcing is None:
        return state

    state_modified = state.copy()

    # Get all active forcing for this variable
    forcing_list = state_forcing.get_forcing(year, variable)

    for func, forced_value in forcing_list:
        group_idx = func.group_idx

        if group_idx == -1:
            # Apply to all groups
            indices: range | list[int] = range(len(state))
        else:
            # Apply to specific group
            indices = [group_idx]

        for idx in indices:
            if func.mode == ForcingMode.REPLACE:
                # Replace computed value with forced value
                state_modified[idx] = forced_value

            elif func.mode == ForcingMode.ADD:
                # Add forced value to computed value
                state_modified[idx] += forced_value

            elif func.mode == ForcingMode.MULTIPLY:
                # Multiply computed value by forced value
                state_modified[idx] *= forced_value

            elif func.mode == ForcingMode.RESCALE:
                # Rescale to match forced value
                # (maintains relative proportions)
                if state[idx] > 0:
                    _scale = forced_value / state[idx]
                    state_modified[idx] = forced_value
                else:
                    state_modified[idx] = forced_value

    return state_modified


def apply_diet_rewiring(
    biomass: np.ndarray, diet_rewiring: Optional[DietRewiring], month: int
) -> Optional[np.ndarray]:
    """Update diet matrix based on current biomass.

    Parameters
    ----------
    biomass : np.ndarray
        Current biomass values
    diet_rewiring : DietRewiring or None
        Diet rewiring configuration
    month : int
        Current month (for update interval check)

    Returns
    -------
    np.ndarray or None
        Updated diet matrix, or None if no update
    """
    if diet_rewiring is None or not diet_rewiring.enabled:
        return None

    # Check if it's time to update
    if month % diet_rewiring.update_interval != 0:
        return None

    # Update diet based on current biomass
    new_diet = diet_rewiring.update_diet(biomass)

    return new_diet


def rsim_run_advanced(
    scenario: RsimScenario,
    state_forcing: Optional[StateForcing] = None,
    diet_rewiring: Optional[DietRewiring] = None,
    method: str = "RK4",
    years: Optional[range] = None,
    verbose: bool = False,
    *,
    on_month: Optional[Callable[[int, int], Optional[bool]]] = None,
) -> RsimOutput:
    """Run Ecosim with ``rsim_run``. Diet rewiring and state forcing are refused.

    Without diet rewiring or state forcing, this is
    ``rsim_run(scenario, method=method, years=years, on_month=on_month)``.
    Its output is returned unchanged. An unknown ``method``, or a ``years``
    range that does not start at ``scenario.start_year``, raises
    ``rsim_run``'s ``ValueError``.

    Diet rewiring and state forcing are not implemented in this version.
    Either one raises ``NotImplementedError``. Earlier versions ran a
    placeholder loop that never integrated: every ``method``, even an
    unknown one, returned the start biomass of every unforced group in
    every month, and zero catch. The rewired diet was computed but never
    applied.

    Parameters
    ----------
    scenario : RsimScenario
        Standard Ecosim scenario
    state_forcing : StateForcing, optional
        Must be None or empty (no forcing functions). Any forcing function
        raises ``NotImplementedError``. ``rsim_run`` takes forced biomass
        from ``scenario.forcing.ForcedBio`` (values > 0 replace the
        biomass), a recruitment multiplier from ``scenario.forcing.ForcedRecs``
        and forced catch from ``scenario.fishing.ForcedCatch``.
    diet_rewiring : DietRewiring, optional
        Must be None or disabled (``enabled=False``). An enabled one
        raises ``NotImplementedError``. Prey switching is available
        instead: set EwE's per-predator ``SwitchPower``
        (``scenario.params.SwitchPower``), which ``rsim_run`` applies, or
        use the Prey Switching control on the app's Ecosim page.
    method : str
        Integration method, passed to ``rsim_run``: ``'RK4'``, ``'AB'``
        or one of its continuous methods. Anything else raises
        ``ValueError``.
    years : range, optional
        Years to simulate, passed to ``rsim_run``. It must start at
        ``scenario.start_year`` (the first ``len(years)`` years are run),
        else ``ValueError``.
    verbose : bool
        Unused, kept for compatibility.
    on_month : callable, optional
        Passed to ``rsim_run``: called as ``on_month(month, n_months)`` at
        the end of every simulated month; returning True stops the run with
        ``RunCancelled``.

    Returns
    -------
    RsimOutput
        ``rsim_run``'s output.

    Raises
    ------
    NotImplementedError
        If ``diet_rewiring`` is enabled, or ``state_forcing`` has a forcing
        function.
    ValueError
        From ``rsim_run``: an unknown ``method`` or an offset ``years``.
    RunCancelled
        From ``rsim_run``, when ``on_month`` returns True.
    """
    if diet_rewiring is not None and diet_rewiring.enabled:
        raise NotImplementedError(
            "Diet rewiring is not implemented in this version: "
            "rsim_run_advanced never applied the rewired diet (it returned "
            "the start biomass). Prey switching is available instead: set "
            "EwE's per-predator SwitchPower (scenario.params.SwitchPower, "
            "applied by rsim_run), or use the Prey Switching control on the "
            "app's Ecosim page. Pass diet_rewiring=None, or "
            "DietRewiring(enabled=False), to run rsim_run."
        )
    if state_forcing is not None and state_forcing.functions:
        raise NotImplementedError(
            "State forcing is not implemented in rsim_run_advanced in this "
            "version: it never integrated the unforced groups. Force biomass "
            "with scenario.forcing.ForcedBio (values > 0 replace the "
            "biomass), recruitment with scenario.forcing.ForcedRecs or catch "
            "with scenario.fishing.ForcedCatch, and run rsim_run."
        )
    return rsim_run(scenario, method=method, years=years, on_month=on_month)


def create_advanced_scenario(
    base_scenario: RsimScenario,
    state_forcing: Optional[StateForcing] = None,
    diet_rewiring: Optional[DietRewiring] = None,
) -> tuple[RsimScenario, StateForcing, DietRewiring]:
    """Create advanced scenario with forcing and rewiring.

    Parameters
    ----------
    base_scenario : RsimScenario
        Base Ecosim scenario
    state_forcing : StateForcing, optional
        State forcing configuration
    diet_rewiring : DietRewiring, optional
        Diet rewiring configuration

    Returns
    -------
    scenario : RsimScenario
        Base scenario (unchanged)
    state_forcing : StateForcing
        State forcing (created if None)
    diet_rewiring : DietRewiring
        Diet rewiring (created if None)
    """
    if state_forcing is None:
        state_forcing = StateForcing()

    if diet_rewiring is None:
        diet_rewiring = DietRewiring(enabled=False)

    return base_scenario, state_forcing, diet_rewiring


# Export main functions
__all__ = [
    "rsim_run_advanced",
    "create_advanced_scenario",
    "apply_state_forcing",
    "apply_diet_rewiring",
]
