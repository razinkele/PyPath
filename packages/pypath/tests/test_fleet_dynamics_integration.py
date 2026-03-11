"""Integration tests for Fleet Dynamics with Ecosim."""
import numpy as np
import pytest
import warnings

from pypath.core.ecopath import rpath
from pypath.core.ecosim import rsim_run, rsim_scenario
from pypath.core.fleet_dynamics import create_fleet_econ_params
from pypath.core.params import create_rpath_params


def _make_fleet_model():
    """Create a balanced model with 1 fleet for fleet dynamics testing.

    Groups: Phyto(1), Zoo(0), Fish(0), Det(2), Fleet(3)
    Fleet catches Fish with landing rate 0.5.
    """
    params = create_rpath_params(
        groups=["Phyto", "Zoo", "Fish", "Det", "Fleet"],
        types=[1, 0, 0, 2, 3],
    )
    params.model.loc[0, "Biomass"] = 10.0
    params.model.loc[0, "PB"] = 200.0
    params.model.loc[0, "EE"] = 0.8
    params.model.loc[1, "Biomass"] = 5.0
    params.model.loc[1, "PB"] = 50.0
    params.model.loc[1, "QB"] = 150.0
    params.model.loc[1, "EE"] = 0.9
    params.model.loc[2, "Biomass"] = 3.0
    params.model.loc[2, "PB"] = 10.0
    params.model.loc[2, "QB"] = 30.0
    params.model.loc[2, "EE"] = 0.9
    params.model.loc[3, "Biomass"] = 100.0
    params.model["BioAcc"] = 0.0
    params.model["Unassim"] = 0.2
    params.model.loc[0, "Unassim"] = 0.0
    params.model.loc[3, "Unassim"] = 0.0
    params.model.loc[4, "Unassim"] = 0.0
    params.model["Detritus"] = 1.0
    params.model.loc[3, "Detritus"] = 0.0
    params.model.loc[4, "Detritus"] = 0.0
    # Diet: Zoo eats Phyto, Fish eats Zoo
    params.diet["Zoo"] = [1.0, 0.0, 0.0, 0.0, 0.0]
    params.diet["Fish"] = [0.0, 1.0, 0.0, 0.0, 0.0]
    # Fleet catches Fish
    params.model.loc[2, "Fleet"] = 0.5
    return params


@pytest.mark.slow
class TestFleetDynamicsIntegration:
    def test_rsim_run_with_fleet_dynamics(self):
        """rsim_run returns output with .fleet_dynamics attribute."""
        params = _make_fleet_model()
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            rpath_result = rpath(params)
        scenario = rsim_scenario(rpath_result, params, years=range(1, 6))
        n_links = len(scenario.params.FishFrom)
        fd_params = create_fleet_econ_params(1, n_links)
        fd_params.price[:] = 10.0
        fd_params.cap_base_growth = np.array([0.3])
        fd_params.cap_depreciate = np.array([0.05])

        result = rsim_run(scenario, fleet_dynamics=fd_params)

        assert result.fleet_dynamics is not None
        assert result.fleet_dynamics.out_Effort.shape[1] == 1
        assert result.fleet_dynamics.annual_Effort.shape == (5, 1)
        assert len(result.fleet_dynamics.fleet_names) == 1

    def test_high_price_effort_increases(self):
        """High-price catch -> fleet effort increases over time."""
        params = _make_fleet_model()
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            rpath_result = rpath(params)
        scenario = rsim_scenario(rpath_result, params, years=range(1, 6))
        n_links = len(scenario.params.FishFrom)
        fd_params = create_fleet_econ_params(1, n_links)
        fd_params.price[:] = 100.0  # very high price -> very profitable
        fd_params.cap_base_growth = np.array([0.5])
        fd_params.cap_depreciate = np.array([0.05])

        result = rsim_run(scenario, fleet_dynamics=fd_params)

        effort = result.fleet_dynamics.out_Effort[:, 0]
        assert effort[-1] > effort[0]

    def test_zero_price_effort_decays(self):
        """Zero-price (no revenue) -> fleet effort decays toward floor."""
        params = _make_fleet_model()
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            rpath_result = rpath(params)
        scenario = rsim_scenario(rpath_result, params, years=range(1, 6))
        n_links = len(scenario.params.FishFrom)
        fd_params = create_fleet_econ_params(1, n_links)
        # price stays 0 (default) -> no revenue
        fd_params.cap_depreciate = np.array([0.2])

        result = rsim_run(scenario, fleet_dynamics=fd_params)

        effort = result.fleet_dynamics.out_Effort[:, 0]
        assert effort[-1] < effort[0]

    def test_no_fleet_dynamics_returns_none(self):
        """Without fleet_dynamics kwarg, output.fleet_dynamics is None."""
        params = _make_fleet_model()
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            rpath_result = rpath(params)
        scenario = rsim_scenario(rpath_result, params, years=range(1, 3))

        result = rsim_run(scenario)
        assert result.fleet_dynamics is None

    def test_result_shapes(self):
        """Output arrays have correct shapes."""
        params = _make_fleet_model()
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            rpath_result = rpath(params)
        n_years = 3
        scenario = rsim_scenario(rpath_result, params, years=range(1, n_years + 1))
        n_links = len(scenario.params.FishFrom)
        fd_params = create_fleet_econ_params(1, n_links)

        result = rsim_run(scenario, fleet_dynamics=fd_params)

        n_months = n_years * 12
        assert result.fleet_dynamics.out_Effort.shape == (n_months + 1, 1)
        assert result.fleet_dynamics.out_Revenue.shape == (n_months + 1, 1)
        assert result.fleet_dynamics.annual_Effort.shape == (n_years, 1)
        assert result.fleet_dynamics.annual_Profit.shape == (n_years, 1)
