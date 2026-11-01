"""Tests for Ecosim scenario ramping integration."""

from __future__ import annotations

import numpy as np
import pytest

from pypath.core.ramping import LinearRamp, PolynomialRamp, StepwiseRamp


class TestScenarioRampingAPI:
    """Test RsimScenario ramping API methods."""

    def test_scenario_has_ramps_dict(self, spatial_scenario):
        """Scenario has ramps dict after first use."""
        scenario, _ = spatial_scenario
        ramp = LinearRamp(0, 0.1, 10, 0.5)
        group_name = scenario.params.spname[1]
        scenario.set_parameter_ramp(group_name, "M2", ramp)

        assert hasattr(scenario, "ramps")
        assert isinstance(scenario.ramps, dict)
        assert len(scenario.ramps) == 1

    def test_scenario_set_ramp_basic(self, spatial_scenario):
        scenario, _ = spatial_scenario  # Unpack (scenario, params)
        """Can set ramp for a parameter."""
        ramp = LinearRamp(0, 0.1, 10, 0.5)
        group_name = scenario.params.spname[1]  # First non-Outside group
        scenario.set_parameter_ramp(group_name, "M2", ramp)

        assert (group_name, "M2") in scenario.ramps
        assert scenario.ramps[(group_name, "M2")] is ramp

    def test_scenario_set_ramp_invalid_group(self, spatial_scenario):
        scenario, _ = spatial_scenario  # Unpack (scenario, params)
        """Setting ramp on unknown group raises error."""
        ramp = LinearRamp(0, 0.1, 10, 0.5)
        with pytest.raises(ValueError, match="Unknown group"):
            scenario.set_parameter_ramp("InvalidGroup", "M2", ramp)

    def test_scenario_get_parameter_value_with_ramp(self, spatial_scenario):
        scenario, _ = spatial_scenario  # Unpack (scenario, params)
        """get_parameter_value applies ramp."""
        ramp = LinearRamp(0, 1.0, 10, 2.0)
        group_name = scenario.params.spname[1]
        scenario.set_parameter_ramp(group_name, "M2", ramp)

        # At time 0, ramp returns 1.0
        val = scenario.get_parameter_value(group_name, "M2", 0)
        assert val == pytest.approx(1.0)

        # At time 5 (midpoint), ramp returns 1.5
        val = scenario.get_parameter_value(group_name, "M2", 5)
        assert val == pytest.approx(1.5)

    def test_scenario_get_parameter_value_without_ramp(self, spatial_scenario):
        scenario, _ = spatial_scenario  # Unpack (scenario, params)
        """No ramp returns 1.0 multiplier."""
        group_name = scenario.params.spname[1]
        val = scenario.get_parameter_value(group_name, "K", 5)
        assert val == 1.0

    def test_scenario_get_ramp_trajectory(self, spatial_scenario):
        scenario, _ = spatial_scenario  # Unpack (scenario, params)
        """Can get trajectory for ramp."""
        ramp = LinearRamp(0, 0.1, 10, 0.5)
        group_name = scenario.params.spname[1]
        scenario.set_parameter_ramp(group_name, "M2", ramp)

        times = np.array([0, 5, 10])
        traj = scenario.get_ramp_trajectory(group_name, "M2", times)

        expected = np.array([0.1, 0.3, 0.5])
        np.testing.assert_array_almost_equal(traj, expected)

    def test_scenario_get_ramp_trajectory_no_ramp(self, spatial_scenario):
        scenario, _ = spatial_scenario  # Unpack (scenario, params)
        """Getting trajectory for undefined ramp raises error."""
        group_name = scenario.params.spname[1]
        times = np.array([0, 5, 10])

        with pytest.raises(ValueError, match="No ramp"):
            scenario.get_ramp_trajectory(group_name, "NoRamp", times)

    def test_scenario_list_ramps(self, spatial_scenario):
        scenario, _ = spatial_scenario  # Unpack (scenario, params)
        """Can list all active ramps."""
        ramp1 = LinearRamp(0, 0.1, 10, 0.5)
        ramp2 = PolynomialRamp(0, 1.0, 20, 2.0)
        group_name = scenario.params.spname[1]

        scenario.set_parameter_ramp(group_name, "M2", ramp1)
        scenario.set_parameter_ramp(group_name, "K", ramp2)

        ramps_list = scenario.list_ramps()
        assert len(ramps_list) == 2
        assert (group_name, "M2", "LinearRamp") in ramps_list
        assert (group_name, "K", "PolynomialRamp") in ramps_list

    def test_scenario_multiple_ramps(self, spatial_scenario):
        scenario, _ = spatial_scenario  # Unpack (scenario, params)
        """Multiple ramps on different parameters."""
        group_name = scenario.params.spname[1]

        ramp_M2 = LinearRamp(0, 0.1, 20, 0.3)
        ramp_K = PolynomialRamp(0, 0.5, 20, 1.0)
        ramp_step = StepwiseRamp([(0, 0.1), (10, 0.5)])

        scenario.set_parameter_ramp(group_name, "M2", ramp_M2)
        scenario.set_parameter_ramp(group_name, "K", ramp_K)
        scenario.set_parameter_ramp(group_name, "F", ramp_step)

        ramps_list = scenario.list_ramps()
        assert len(ramps_list) == 3


class TestScenarioRampingIntegration:
    """Test ramping in realistic scenario workflows."""

    def test_multiple_ramps_different_params(self, spatial_scenario):
        """Ramps on different groups don't interfere."""
        scenario, _ = spatial_scenario  # Unpack (scenario, params)
        groups = scenario.params.spname[1:3]  # First 2 groups

        # Use DIFFERENT ramp types for different parameters
        ramp_linear = LinearRamp(0, 0.1, 20, 0.5)
        ramp_poly = PolynomialRamp(0, 0.1, 20, 0.5)

        # Set the same parameter (M2) on two different groups
        scenario.set_parameter_ramp(groups[0], "M2", ramp_linear)
        scenario.set_parameter_ramp(groups[1], "M2", ramp_poly)

        val_linear = scenario.get_parameter_value(groups[0], "M2", 10)
        val_poly = scenario.get_parameter_value(groups[1], "M2", 10)

        # At midpoint (t=10), linear gives 0.3, polynomial also gives ~0.3 via smoothstep
        # They should both be around 0.3 but not necessarily identical
        assert 0.25 < val_linear < 0.35
        assert 0.25 < val_poly < 0.35

    def test_ramp_values_in_valid_range(self, spatial_scenario):
        scenario, _ = spatial_scenario  # Unpack (scenario, params)
        """Ramped values stay within specified range."""
        group_name = scenario.params.spname[1]
        ramp = LinearRamp(0, 0.1, 20, 0.5)
        scenario.set_parameter_ramp(group_name, "M2", ramp)

        times = np.linspace(0, 20, 100)
        traj = scenario.get_ramp_trajectory(group_name, "M2", times)

        assert np.all(traj >= 0.1)
        assert np.all(traj <= 0.5)

    def test_ramp_trajectory_shape(self, spatial_scenario):
        scenario, _ = spatial_scenario  # Unpack (scenario, params)
        """Trajectory shape matches input times."""
        group_name = scenario.params.spname[1]
        ramp = LinearRamp(0, 0.1, 10, 0.5)
        scenario.set_parameter_ramp(group_name, "M2", ramp)

        # 2D times array
        times_2d = np.array([[0, 5], [10, 15]])
        traj = scenario.get_ramp_trajectory(group_name, "M2", times_2d)

        assert traj.shape == times_2d.shape
