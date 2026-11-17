"""``rsim_run_advanced`` runs ``rsim_run`` and refuses what was never built.

Its monthly loop never integrated. It was a placeholder from its first
commit (490fa84, v0.3.0): every method, even an unknown one, returned the
start biomass in every month, and the rewired diet was counted and dropped.
It now delegates to ``rsim_run``. Diet rewiring and state forcing were never
implemented, so it refuses them with ``NotImplementedError``. Prey switching
(EwE ``SwitchPower``) is the planned replacement for diet rewiring.
"""

import numpy as np
import pytest

from pypath.core.ecosim import rsim_run
from pypath.core.ecosim_advanced import rsim_run_advanced
from pypath.core.forcing import (
    DietRewiring,
    StateForcing,
    create_biomass_forcing,
    create_diet_rewiring,
)
from tests.continuous_support import needs_reco, reco_plain, scenario

pytestmark = needs_reco


def _scen():
    # Effort 2 for 2 years: rsim_run moves REco by up to 16 % by month 24.
    return scenario(reco_plain(), effort=2.0, years=range(1, 3))


@pytest.mark.parametrize("method", ["RK4", "AB"])
@pytest.mark.parametrize(
    "rewiring, forcing",
    [
        (None, None),
        (DietRewiring(enabled=False), None),
        (None, StateForcing()),
    ],
    ids=["no-rewiring", "rewiring-disabled", "empty-state-forcing"],
)
def test_without_rewiring_it_returns_rsim_runs_output(method, rewiring, forcing):
    ref = rsim_run(_scen(), method=method)
    out = rsim_run_advanced(
        _scen(), state_forcing=forcing, diet_rewiring=rewiring, method=method
    )
    assert np.array_equal(out.out_Biomass, ref.out_Biomass)
    assert np.array_equal(out.out_Catch, ref.out_Catch)
    assert np.array_equal(out.annual_QB, ref.annual_QB)
    # The placeholder returned the start biomass in every month.
    living = slice(1, _scen().params.NUM_LIVING + 1)
    moved = np.abs(out.out_Biomass[24, living] / out.out_Biomass[0, living] - 1)
    assert moved.max() > 0.1


def test_the_years_argument_reaches_rsim_run():
    out = rsim_run_advanced(_scen(), method="AB", years=range(1, 2))
    ref = rsim_run(_scen(), method="AB", years=range(1, 2))
    assert out.out_Biomass.shape[0] == 13
    assert np.array_equal(out.out_Biomass, ref.out_Biomass)


def test_an_unknown_method_raises_value_error():
    with pytest.raises(ValueError, match="Unknown integration method 'nonsense'"):
        rsim_run_advanced(_scen(), method="nonsense")


@pytest.mark.parametrize("method", ["RK4", "AB", "lsoda", "nonsense"])
def test_enabled_diet_rewiring_is_refused(method):
    rewiring = create_diet_rewiring(switching_power=2.5, update_interval=12)
    with pytest.raises(NotImplementedError, match="SwitchPower"):
        rsim_run_advanced(_scen(), diet_rewiring=rewiring, method=method)
    assert rewiring.base_diet is None  # refused before anything ran


def test_state_forcing_is_refused():
    forcing = create_biomass_forcing(group_idx=1, observed_biomass={1: 1.0, 2: 2.0})
    with pytest.raises(NotImplementedError, match="ForcedBio"):
        rsim_run_advanced(_scen(), state_forcing=forcing)
