"""Integration tests for Ecotracer with Ecosim."""
import numpy as np
import pytest

from pypath.core.ecopath import rpath
from pypath.core.ecosim import rsim_run, rsim_scenario
from pypath.core.ecotracer import EcotracerParams, create_ecotracer_params
from pypath.core.params import create_rpath_params


def _make_ecotracer_model():
    """Create a balanced 3-group model for ecotracer testing."""
    params = create_rpath_params(
        groups=["Producer", "Consumer", "Detritus"],
        types=[1, 0, 2],
    )
    params.model.loc[0, "Biomass"] = 10.0
    params.model.loc[0, "PB"] = 200.0
    params.model.loc[0, "EE"] = 0.8
    params.model.loc[1, "Biomass"] = 5.0
    params.model.loc[1, "PB"] = 50.0
    params.model.loc[1, "QB"] = 150.0
    params.model.loc[1, "EE"] = 0.9
    params.model.loc[2, "Biomass"] = 100.0
    params.model["BioAcc"] = 0.0
    params.model["Unassim"] = 0.2
    params.model.loc[0, "Unassim"] = 0.0
    params.model.loc[2, "Unassim"] = 0.0
    params.model["Detritus"] = 1.0
    params.model.loc[2, "Detritus"] = 0.0
    params.diet["Consumer"] = [1.0, 0.0, 0.0, 0.0]
    return params


@pytest.mark.slow
class TestEcotracerIntegration:
    def test_rsim_run_with_ecotracer(self):
        """rsim_run returns output with .ecotracer attribute."""
        params = _make_ecotracer_model()
        rpath_result = rpath(params)
        scenario = rsim_scenario(rpath_result, params, years=range(1, 6))
        eco_params = create_ecotracer_params(3)
        eco_params.czero[0] = 1.0  # contaminate producer

        result = rsim_run(scenario, ecotracer=eco_params)

        assert result.ecotracer is not None
        assert result.ecotracer.out_Conc.shape[0] > 0
        assert result.ecotracer.out_Conc.shape[1] == 3
        assert result.ecotracer.annual_Conc.shape == (5, 3)
        assert len(result.ecotracer.group_names) == 3

    def test_contamination_spreads(self):
        """Consumer eating contaminated Producer gains concentration."""
        params = _make_ecotracer_model()
        rpath_result = rpath(params)
        scenario = rsim_scenario(rpath_result, params, years=range(1, 6))
        eco_params = create_ecotracer_params(3)
        eco_params.czero[0] = 1.0  # contaminate producer
        eco_params.cenv[0] = 0.1   # ongoing environmental input

        result = rsim_run(scenario, ecotracer=eco_params)

        # Consumer (idx 1) should have increasing concentration
        conc_consumer = result.ecotracer.out_Conc[:, 1]
        assert conc_consumer[-1] > conc_consumer[0]

    def test_decay_reduces_concentration(self):
        """With no input and positive decay, concentration decreases."""
        params = _make_ecotracer_model()
        rpath_result = rpath(params)
        scenario = rsim_scenario(rpath_result, params, years=range(1, 3))
        eco_params = create_ecotracer_params(3)
        eco_params.czero = np.array([1.0, 1.0, 0.5])
        eco_params.cdecay = np.array([5.0, 5.0, 1.0])  # high decay to dominate dietary intake
        eco_params.cassim[:] = 0.0  # disable dietary uptake to isolate decay

        result = rsim_run(scenario, ecotracer=eco_params)

        # All concentrations should decrease from initial
        for i in range(3):
            assert result.ecotracer.out_Conc[-1, i] < result.ecotracer.out_Conc[0, i]

    def test_no_ecotracer_returns_none(self):
        """Without ecotracer kwarg, output.ecotracer is None."""
        params = _make_ecotracer_model()
        rpath_result = rpath(params)
        scenario = rsim_scenario(rpath_result, params, years=range(1, 3))

        result = rsim_run(scenario)
        assert result.ecotracer is None

    def test_result_shapes(self):
        """Output arrays have correct shapes."""
        params = _make_ecotracer_model()
        rpath_result = rpath(params)
        n_years = 3
        scenario = rsim_scenario(rpath_result, params, years=range(1, n_years + 1))
        eco_params = create_ecotracer_params(3)

        result = rsim_run(scenario, ecotracer=eco_params)

        n_months = n_years * 12
        assert result.ecotracer.out_Conc.shape == (n_months + 1, 3)
        assert result.ecotracer.annual_Conc.shape == (n_years, 3)
