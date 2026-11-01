"""Comprehensive integration tests for Ecosim environmental forcing API (Week 2: Tasks 2.1-2.3).

This test suite covers Scenario-level environmental forcing integration:
- Task 2.1: Scenario environmental forcing API (_environmental_forcings dict, setters/getters)
- Task 2.2: Scenario forcing evaluation (apply_environmental_forcing, compute_habitat_suitability)
- Task 2.3: Batch forcing evaluation (get_forcing_trajectory for timeseries analysis)

Test Structure:
- TestScenarioEnvironmentalForcingAPI: 8 tests for Scenario API
- TestScenarioEnvironmentalForcingEvaluation: 4 tests for forcing evaluation
- TestScenarioEnvironmentalForcingIntegration: 3 tests for integration workflows

Total: 15+ tests covering Week 2 requirements.
"""

import numpy as np
import pytest

from pypath.core.environmental_forcing import (
    HabitatProductivity,
    OceanographicForcing,
    TimeSeries,
)

# ============================================================================
# FIXTURES
# ============================================================================


@pytest.fixture
def habitat_productivity_forcing():
    """Create a habitat productivity forcing for testing.

    Returns a HabitatProductivity object with temporal variation.
    Productivity scales from 0.9 to 1.1 over a 100-year period.
    """
    ts = TimeSeries(times=np.array([0.0, 100.0]), values=np.array([0.9, 1.1]))
    return HabitatProductivity(base_productivity=1.0, temporal_forcing=ts)


@pytest.fixture
def oceanographic_forcing():
    """Create an oceanographic forcing for testing.

    Returns an OceanographicForcing object with temperature variation.
    Temperature ranges from 10°C to 25°C over a 365-day cycle.
    """
    temp_ts = TimeSeries(
        times=np.array([0.0, 90.0, 180.0, 270.0, 365.0]),
        values=np.array([10.0, 15.0, 25.0, 20.0, 10.0]),
    )
    return OceanographicForcing(temperature_series=temp_ts)


@pytest.fixture
def spatial_scenario_with_forcings(spatial_scenario, habitat_productivity_forcing):
    """Create a spatial scenario with pre-registered environmental forcings.

    This fixture extends spatial_scenario with a habitat productivity forcing
    already registered for use in integration tests.
    """
    scenario, params = spatial_scenario

    # Ensure the scenario has the environmental forcing support
    # by calling list_environmental_forcings (which will initialize _environmental_forcings)
    if not hasattr(scenario, "_environmental_forcings"):
        scenario._environmental_forcings = {}

    scenario.set_environmental_forcing(
        "habitat_productivity", habitat_productivity_forcing
    )
    return scenario, params


# ============================================================================
# CLASS TestScenarioEnvironmentalForcingAPI (Task 2.1)
# ============================================================================


class TestScenarioEnvironmentalForcingAPI:
    """Test Scenario-level environmental forcing API (Task 2.1).

    Verifies that RsimScenario supports:
    - _environmental_forcings dict storage
    - set_environmental_forcing() registration
    - get_environmental_forcing() retrieval
    - list_environmental_forcings() discovery
    - Type validation and error handling
    """

    def test_scenario_has_environmental_forcings_dict(self, spatial_scenario):
        """Verify that Scenario has _environmental_forcings dict.

        After initialization or first access, the scenario should have an
        _environmental_forcings dictionary attribute for storing forcings.
        """
        scenario, _ = spatial_scenario

        # Initialize by calling list_environmental_forcings
        scenario.list_environmental_forcings()

        # Verify dict exists
        assert hasattr(scenario, "_environmental_forcings")
        assert isinstance(scenario._environmental_forcings, dict)

    def test_scenario_set_forcing_basic(
        self, spatial_scenario, habitat_productivity_forcing
    ):
        """Test basic environmental forcing registration.

        Verify that set_environmental_forcing() successfully registers
        a forcing object under a given name.
        """
        scenario, _ = spatial_scenario
        forcing_name = "habitat_productivity"

        # Should not raise any exceptions
        scenario.set_environmental_forcing(forcing_name, habitat_productivity_forcing)

        # Verify it was registered
        assert forcing_name in scenario.list_environmental_forcings()

    def test_scenario_set_forcing_override(
        self, spatial_scenario, habitat_productivity_forcing, oceanographic_forcing
    ):
        """Test that existing forcings can be overridden.

        When registering a forcing with an existing name, it should
        replace the previous forcing without error.
        """
        scenario, _ = spatial_scenario
        forcing_name = "oceanographic"

        # Register first forcing
        scenario.set_environmental_forcing(forcing_name, habitat_productivity_forcing)
        first_forcing = scenario.get_environmental_forcing(forcing_name)
        assert first_forcing is habitat_productivity_forcing

        # Override with second forcing
        scenario.set_environmental_forcing(forcing_name, oceanographic_forcing)
        second_forcing = scenario.get_environmental_forcing(forcing_name)
        assert second_forcing is oceanographic_forcing
        assert second_forcing is not first_forcing

    def test_scenario_get_forcing_basic(
        self, spatial_scenario, habitat_productivity_forcing
    ):
        """Test retrieving a registered environmental forcing.

        Verify that get_environmental_forcing() returns the exact forcing
        object that was registered.
        """
        scenario, _ = spatial_scenario
        forcing_name = "habitat_test"

        # Register forcing
        scenario.set_environmental_forcing(forcing_name, habitat_productivity_forcing)

        # Retrieve and verify identity
        retrieved = scenario.get_environmental_forcing(forcing_name)
        assert retrieved is habitat_productivity_forcing

    def test_scenario_get_forcing_not_found(self, spatial_scenario):
        """Test that get_environmental_forcing returns None for missing forcings.

        Rather than raising an exception, the method should gracefully
        return None when a forcing is not registered.
        """
        scenario, _ = spatial_scenario

        # Try to retrieve non-existent forcing
        result = scenario.get_environmental_forcing("nonexistent")

        # Should return None without raising
        assert result is None

    def test_scenario_list_forcings_empty(self, spatial_scenario):
        """Test that list_environmental_forcings returns empty list initially.

        A fresh scenario should have no registered forcings, so the method
        should return an empty list (not None).
        """
        scenario, _ = spatial_scenario

        forcings_list = scenario.list_environmental_forcings()

        # Should return empty list, not None
        assert isinstance(forcings_list, list)
        assert len(forcings_list) == 0

    def test_scenario_list_forcings_multiple(
        self, spatial_scenario, habitat_productivity_forcing, oceanographic_forcing
    ):
        """Test listing multiple registered environmental forcings.

        Verify that list_environmental_forcings() returns all registered
        forcing names in a sorted list.
        """
        scenario, _ = spatial_scenario

        # Register multiple forcings
        scenario.set_environmental_forcing("alpha", habitat_productivity_forcing)
        scenario.set_environmental_forcing("zeta", oceanographic_forcing)
        scenario.set_environmental_forcing("gamma", habitat_productivity_forcing)

        # Get list
        forcings_list = scenario.list_environmental_forcings()

        # Verify all present and sorted
        assert len(forcings_list) == 3
        assert forcings_list == ["alpha", "gamma", "zeta"]

    def test_scenario_invalid_forcing_type(self, spatial_scenario):
        """Test that set_environmental_forcing rejects non-EnvironmentalForcing objects.

        The method should validate that the forcing parameter is an instance
        of EnvironmentalForcing base class.
        """
        scenario, _ = spatial_scenario

        # Try to register invalid types
        invalid_objects = [
            None,
            42,
            "string_forcing",
            {"dict": "forcing"},
            ["list", "forcing"],
        ]

        for invalid in invalid_objects:
            with pytest.raises(TypeError, match="EnvironmentalForcing"):
                scenario.set_environmental_forcing("invalid", invalid)


# ============================================================================
# CLASS TestScenarioEnvironmentalForcingEvaluation (Task 2.2)
# ============================================================================


class TestScenarioEnvironmentalForcingEvaluation:
    """Test Scenario forcing evaluation methods (Task 2.2).

    Verifies that RsimScenario can:
    - apply_environmental_forcing() at single time points
    - compute_habitat_suitability() from oceanographic data
    - Handle empty, single, and multiple forcings
    - Return values in expected ranges
    """

    def test_apply_environmental_forcing_single(self, spatial_scenario_with_forcings):
        """Test evaluating a single registered environmental forcing.

        Verify that apply_environmental_forcing() returns a dict with
        the forcing name and its evaluated scaling factor at a given time.
        """
        scenario, _ = spatial_scenario_with_forcings

        # Apply forcing at time 50 (middle of range)
        factors = scenario.apply_environmental_forcing(time=50.0)

        # Should return dict with one entry
        assert isinstance(factors, dict)
        assert "habitat_productivity" in factors

        # Value should be numeric and in reasonable range
        assert isinstance(factors["habitat_productivity"], (int, float, np.number))
        assert 0.5 <= factors["habitat_productivity"] <= 1.5

    def test_apply_environmental_forcing_multiple(
        self, spatial_scenario, habitat_productivity_forcing, oceanographic_forcing
    ):
        """Test evaluating multiple environmental forcings simultaneously.

        Verify that apply_environmental_forcing() evaluates all registered
        forcings and returns a dict with all results.
        """
        scenario, _ = spatial_scenario

        # Register multiple forcings
        scenario.set_environmental_forcing("habitat", habitat_productivity_forcing)
        scenario.set_environmental_forcing("ocean", oceanographic_forcing)

        # Apply forcings
        factors = scenario.apply_environmental_forcing(time=75.0)

        # Should have entries for both
        assert len(factors) == 2
        assert "habitat" in factors
        assert "ocean" in factors

        # Both should be numeric
        assert all(isinstance(v, (int, float, np.number)) for v in factors.values())

    def test_apply_environmental_forcing_empty(self, spatial_scenario):
        """Test applying forcings when none are registered.

        When no forcings are registered, apply_environmental_forcing()
        should return an empty dict (not None or raise exception).
        """
        scenario, _ = spatial_scenario

        # Apply with no forcings registered
        factors = scenario.apply_environmental_forcing(time=100.0)

        # Should return empty dict
        assert isinstance(factors, dict)
        assert len(factors) == 0

    @pytest.mark.parametrize("time_value", [0.0, 25.5, 50.0, 75.3, 100.0])
    def test_forcing_values_in_valid_range(
        self, spatial_scenario_with_forcings, time_value
    ):
        """Test that all evaluated forcing values are in valid range.

        Environmental forcing scaling factors should typically fall within
        [0.5, 1.5], representing 50% to 150% of baseline conditions.
        """
        scenario, _ = spatial_scenario_with_forcings

        # Apply forcing at various times
        factors = scenario.apply_environmental_forcing(time=time_value)

        # All values should be in reasonable range
        for name, factor in factors.items():
            assert 0.5 <= factor <= 1.5, (
                f"Forcing '{name}' at time {time_value} is out of range: {factor}"
            )


# ============================================================================
# CLASS TestScenarioEnvironmentalForcingIntegration (Task 2.3)
# ============================================================================


class TestScenarioEnvironmentalForcingIntegration:
    """Test integrated environmental forcing workflows (Task 2.3).

    Verifies that:
    - Habitat suitability can be computed from oceanographic data
    - Habitat suitability falls back gracefully when data unavailable
    - Forcing trajectories can be evaluated across multiple time points
    """

    def test_habitat_suitability_with_oceanographic(
        self, spatial_scenario, oceanographic_forcing
    ):
        """Test computing habitat suitability when oceanographic forcing is available.

        When an OceanographicForcing is registered, compute_habitat_suitability()
        should use it to compute a suitability index for the habitat.
        """
        scenario, _ = spatial_scenario

        # Register oceanographic forcing
        scenario.set_environmental_forcing(
            "oceanographic_forcing", oceanographic_forcing
        )

        # Compute suitability
        suitability = scenario.compute_habitat_suitability(group="Fish", time=50.0)

        # Should return a numeric value in [0, 1]
        assert isinstance(suitability, (int, float, np.number))
        assert 0.0 <= suitability <= 1.0, (
            f"Suitability out of range [0, 1]: {suitability}"
        )

    def test_habitat_suitability_no_oceanographic(self, spatial_scenario):
        """Test habitat suitability fallback when oceanographic forcing unavailable.

        When no OceanographicForcing is registered, compute_habitat_suitability()
        should gracefully return 1.0 (neutral/optimal conditions).
        """
        scenario, _ = spatial_scenario

        # No forcing registered
        suitability = scenario.compute_habitat_suitability(group="Fish", time=50.0)

        # Should return neutral value 1.0
        assert suitability == 1.0

    def test_get_forcing_trajectory_shape(self, spatial_scenario_with_forcings):
        """Test batch forcing evaluation returns correct shape across time.

        Verify that get_forcing_trajectory() evaluates forcings across
        multiple time points and returns results with correct dimensions.
        """
        scenario, _ = spatial_scenario_with_forcings

        # Define time points
        time_points = np.linspace(0.0, 100.0, 11)  # 11 points from 0 to 100

        # Get trajectory
        trajectory = scenario.get_forcing_trajectory(
            "habitat_productivity", time_points
        )

        # Should return numpy array
        assert isinstance(trajectory, np.ndarray)
        assert trajectory.shape[0] == 11

        # All values should be numeric
        assert all(isinstance(v, (int, float, np.number)) for v in trajectory)

    @pytest.mark.parametrize("num_times", [5, 10, 21, 50])
    def test_forcing_trajectory_multiple_points(
        self, spatial_scenario_with_forcings, num_times
    ):
        """Test forcing trajectory evaluation at multiple time points.

        Verify that get_forcing_trajectory() correctly evaluates forcings
        at various numbers of time points.
        """
        scenario, _ = spatial_scenario_with_forcings

        # Define time range
        time_points = np.linspace(0.0, 100.0, num_times)

        # Get trajectory
        trajectory = scenario.get_forcing_trajectory(
            "habitat_productivity", time_points
        )

        # Trajectory should be numpy array with correct shape
        assert isinstance(trajectory, np.ndarray)
        assert trajectory.shape[0] == num_times, (
            f"Trajectory has {trajectory.shape[0]} points, expected {num_times}"
        )


# ============================================================================
# INTEGRATION WORKFLOW TESTS
# ============================================================================


class TestEnvironmentalForcingWorkflow:
    """Test realistic environmental forcing workflows.

    Covers complete scenarios involving multiple forcings, time-dependent
    evaluation, and integration with ecosystem dynamics.
    """

    def test_scenario_with_multiple_forcings_workflow(
        self, spatial_scenario, habitat_productivity_forcing, oceanographic_forcing
    ):
        """Test complete workflow with multiple forcings across simulation.

        Simulate a realistic scenario where:
        1. Multiple forcings are registered
        2. They are evaluated at different times
        3. Results can be aggregated or used independently
        """
        scenario, _ = spatial_scenario

        # Register multiple forcings
        scenario.set_environmental_forcing("habitat", habitat_productivity_forcing)
        scenario.set_environmental_forcing("ocean", oceanographic_forcing)

        # List registered forcings
        forcings = scenario.list_environmental_forcings()
        assert len(forcings) == 2

        # Evaluate at multiple time points
        time_points = [0.0, 25.0, 50.0, 75.0, 100.0]
        all_results = {}

        for t in time_points:
            factors = scenario.apply_environmental_forcing(time=t)
            all_results[t] = factors

        # Verify results structure
        assert len(all_results) == len(time_points)
        for t, factors in all_results.items():
            assert len(factors) == 2
            assert all(isinstance(v, (int, float, np.number)) for v in factors.values())

    def test_forcing_override_workflow(
        self, spatial_scenario, habitat_productivity_forcing, oceanographic_forcing
    ):
        """Test workflow where forcings are updated during configuration.

        Verify that forcings can be registered, modified, and re-evaluated
        without side effects.
        """
        scenario, _ = spatial_scenario

        # Initial setup
        scenario.set_environmental_forcing("habitat", habitat_productivity_forcing)
        factors_v1 = scenario.apply_environmental_forcing(time=50.0)

        # Update with different forcing
        scenario.set_environmental_forcing("habitat", oceanographic_forcing)
        factors_v2 = scenario.apply_environmental_forcing(time=50.0)

        # Values should differ
        assert factors_v1["habitat"] != factors_v2["habitat"]

        # Verify only one forcing is active
        assert len(scenario.list_environmental_forcings()) == 1

    @pytest.mark.parametrize(
        "forcing_name,forcing_obj",
        [
            ("habitat_prod", "habitat_productivity_forcing"),
            ("ocean_cond", "oceanographic_forcing"),
            ("env_factor_1", "habitat_productivity_forcing"),
        ],
    )
    def test_multiple_forcing_names(
        self,
        spatial_scenario,
        habitat_productivity_forcing,
        oceanographic_forcing,
        forcing_name,
        forcing_obj,
    ):
        """Test registering forcings with various naming conventions.

        Verify that forcing names can vary and all are properly stored/retrieved.
        """
        scenario, _ = spatial_scenario

        # Get the appropriate forcing object
        if forcing_obj == "habitat_productivity_forcing":
            forcing = habitat_productivity_forcing
        else:
            forcing = oceanographic_forcing

        # Register with various names
        scenario.set_environmental_forcing(forcing_name, forcing)

        # Verify retrieval
        retrieved = scenario.get_environmental_forcing(forcing_name)
        assert retrieved is forcing

        # Verify in list
        assert forcing_name in scenario.list_environmental_forcings()


# ============================================================================
# ERROR HANDLING AND EDGE CASES
# ============================================================================


class TestEnvironmentalForcingErrorHandling:
    """Test error handling and edge cases for environmental forcing API.

    Covers invalid inputs, boundary conditions, and graceful degradation.
    """

    def test_set_forcing_empty_name(
        self, spatial_scenario, habitat_productivity_forcing
    ):
        """Test that empty forcing names are rejected.

        Forcing names must be non-empty strings.
        """
        scenario, _ = spatial_scenario

        with pytest.raises(ValueError, match="non-empty string"):
            scenario.set_environmental_forcing("", habitat_productivity_forcing)

        with pytest.raises(ValueError, match="non-empty string"):
            scenario.set_environmental_forcing(None, habitat_productivity_forcing)

    def test_set_forcing_whitespace_name(
        self, spatial_scenario, habitat_productivity_forcing
    ):
        """Test that whitespace-only forcing names are rejected.

        Names that are just whitespace should be treated as empty.
        """
        scenario, _ = spatial_scenario

        with pytest.raises(ValueError, match="empty or whitespace"):
            scenario.set_environmental_forcing("   ", habitat_productivity_forcing)

    def test_apply_forcing_invalid_time_type(self, spatial_scenario_with_forcings):
        """Test that apply_environmental_forcing rejects non-numeric time.

        Time parameter must be convertible to float.
        """
        scenario, _ = spatial_scenario_with_forcings

        invalid_times = ["string", None, [], {}]

        for invalid_time in invalid_times:
            with pytest.raises(TypeError, match="time must be numeric"):
                scenario.apply_environmental_forcing(time=invalid_time)

    def test_compute_suitability_invalid_group_type(
        self, spatial_scenario_with_forcings
    ):
        """Test that compute_habitat_suitability validates group parameter.

        Group must be a string.
        """
        scenario, _ = spatial_scenario_with_forcings

        with pytest.raises(TypeError, match="group must be a string"):
            scenario.compute_habitat_suitability(group=123, time=50.0)

    def test_compute_suitability_invalid_time_type(
        self, spatial_scenario_with_forcings
    ):
        """Test that compute_habitat_suitability validates time parameter.

        Time must be numeric.
        """
        scenario, _ = spatial_scenario_with_forcings

        with pytest.raises(TypeError, match="time must be numeric"):
            scenario.compute_habitat_suitability(group="Fish", time="not_a_number")

    def test_get_forcing_trajectory_with_empty_forcings(self, spatial_scenario):
        """Test error when no forcings registered before trajectory call."""
        scenario, _ = spatial_scenario

        # Should raise ValueError when no forcings registered
        time_points = np.array([0.0, 50.0, 100.0])
        with pytest.raises(ValueError, match="not registered"):
            scenario.get_forcing_trajectory("habitat_productivity", time_points)

    @pytest.mark.parametrize("invalid_name", [123, 45.6, None, [], {}])
    def test_get_forcing_invalid_name_type(
        self, spatial_scenario, habitat_productivity_forcing, invalid_name
    ):
        """Test that get_environmental_forcing rejects non-string names.

        Forcing names must be strings.
        """
        scenario, _ = spatial_scenario
        scenario.set_environmental_forcing("valid", habitat_productivity_forcing)

        with pytest.raises(TypeError, match="must be a string"):
            scenario.get_environmental_forcing(invalid_name)
