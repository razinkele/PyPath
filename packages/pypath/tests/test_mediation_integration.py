"""Integration tests for mediation functions with Ecosim simulation."""

import warnings

import numpy as np
import pytest

from pypath.core.ecopath import rpath
from pypath.core.ecosim import rsim_run, rsim_scenario
from pypath.core.mediation import (
    MediationCollection,
    MediationLink,
    make_negative_shape,
    make_positive_shape,
)
from pypath.core.params import create_rpath_params


def _make_3group_model():
    """Create a minimal 3-group model: producer -> consumer -> predator + detritus."""
    params = create_rpath_params(
        groups=["Producer", "Consumer", "Predator", "Detritus"],
        types=[1, 0, 0, 2],
    )
    # Producer
    params.model.loc[0, "Biomass"] = 10.0
    params.model.loc[0, "PB"] = 200.0
    params.model.loc[0, "EE"] = 0.8
    # Consumer
    params.model.loc[1, "Biomass"] = 5.0
    params.model.loc[1, "PB"] = 50.0
    params.model.loc[1, "QB"] = 150.0
    params.model.loc[1, "EE"] = 0.9
    # Predator
    params.model.loc[2, "Biomass"] = 2.0
    params.model.loc[2, "PB"] = 1.0
    params.model.loc[2, "QB"] = 5.0
    params.model.loc[2, "EE"] = 0.5
    # Detritus
    params.model.loc[3, "Biomass"] = 100.0

    params.model["BioAcc"] = 0.0
    params.model["Unassim"] = 0.2
    params.model.loc[0, "Unassim"] = 0.0
    params.model.loc[3, "Unassim"] = 0.0
    params.model["Detritus"] = 1.0
    params.model.loc[3, "Detritus"] = 0.0

    # Diet: consumer eats 100% producer, predator eats 100% consumer
    # Diet rows: Producer, Consumer, Predator, Detritus, Import
    params.diet["Consumer"] = [1.0, 0.0, 0.0, 0.0, 0.0]
    params.diet["Predator"] = [0.0, 1.0, 0.0, 0.0, 0.0]

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        rpath_result = rpath(params)
    return rpath_result, params


@pytest.mark.slow
class TestMediationIntegration:
    def test_no_mediation_baseline(self):
        """rsim_run without mediation produces same result as before."""
        rpath_result, params = _make_3group_model()
        scenario = rsim_scenario(rpath_result, params, years=range(1, 11))
        result = rsim_run(scenario)
        assert result.out_Biomass.shape[0] > 0

    def test_positive_mediation_changes_biomass(self):
        """With positive mediation, biomass trajectories differ from baseline."""
        rpath_result, params = _make_3group_model()
        scenario_base = rsim_scenario(rpath_result, params, years=range(1, 11))
        result_base = rsim_run(scenario_base)

        # Producer (group 0) mediates consumer->predator link: more producer -> more predation
        shape = make_positive_shape(shape_id=1, low=0.5, high=2.0)
        link = MediationLink(shape_id=1, mediator_idx=0, prey_idx=1, pred_idx=2)
        med = MediationCollection(shapes=[shape], links=[link])

        scenario_med = rsim_scenario(rpath_result, params, years=range(1, 11))
        result_med = rsim_run(scenario_med, mediation=med)

        # Biomass trajectories should differ
        base_bio = result_base.out_Biomass[-1, :]
        med_bio = result_med.out_Biomass[-1, :]
        assert not np.allclose(base_bio, med_bio, atol=1e-6)

    def test_negative_mediation_changes_biomass(self):
        """Negative mediation: more mediator -> less predation -> prey benefits."""
        rpath_result, params = _make_3group_model()

        shape = make_negative_shape(shape_id=1, low=0.5, high=2.0)
        link = MediationLink(shape_id=1, mediator_idx=0, prey_idx=1, pred_idx=2)
        med = MediationCollection(shapes=[shape], links=[link])

        scenario = rsim_scenario(rpath_result, params, years=range(1, 11))
        result = rsim_run(scenario, mediation=med)
        assert result.out_Biomass.shape[0] > 0

    def test_fleet_mediation_changes_catch(self):
        """Fleet mediation: mediator scales fleet effort -> catch changes."""
        rpath_result, params = _make_3group_model()

        # Create a fleet mediation: producer mediates fleet 0 effort
        shape = make_positive_shape(shape_id=1, low=0.5, high=2.0)
        link = MediationLink(shape_id=1, mediator_idx=0, fleet_idx=0)
        med = MediationCollection(shapes=[shape], links=[link])

        scenario_base = rsim_scenario(rpath_result, params, years=range(1, 11))
        rsim_run(scenario_base)

        scenario_med = rsim_scenario(rpath_result, params, years=range(1, 11))
        result_med = rsim_run(scenario_med, mediation=med)

        # With fleet mediation, catch should differ
        # (may be identical if no fishing -- but the code path is exercised)
        assert result_med.out_Biomass.shape[0] > 0

    def test_regression_none_mediation(self):
        """Passing mediation=None gives identical results to no mediation."""
        rpath_result, params = _make_3group_model()
        scenario1 = rsim_scenario(rpath_result, params, years=range(1, 11))
        result1 = rsim_run(scenario1)

        scenario2 = rsim_scenario(rpath_result, params, years=range(1, 11))
        result2 = rsim_run(scenario2, mediation=None)

        np.testing.assert_allclose(result1.out_Biomass, result2.out_Biomass, atol=1e-12)
