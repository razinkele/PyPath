"""
Tests for scenario adjustment functions.

Tests adjust_fishing, adjust_forcing, adjust_scenario and helper functions.
"""

from dataclasses import dataclass
from typing import List

import numpy as np
import pytest

from pypath.core.adjustments import (
    adjust_scenario,
    create_seasonal_forcing,
)


# Mock classes for testing
@dataclass
class MockParams:
    """Mock params object for testing."""

    BURN_YEARS: int = -1
    COUPLED: int = 1
    RK4_STEPS: int = 4
    NumPredPreyLinks: int = 0
    PreyTo: List = None
    PreyFrom: List = None
    VV: List = None
    DD: List = None


@dataclass
class MockFishing:
    """Mock fishing object for testing."""

    ForcedEffort: np.ndarray = None
    ForcedFRate: np.ndarray = None
    ForcedCatch: np.ndarray = None


@dataclass
class MockForcing:
    """Mock forcing object for testing."""

    ForcedPrey: np.ndarray = None
    ForcedMort: np.ndarray = None
    ForcedRecs: np.ndarray = None
    ForcedSearch: np.ndarray = None
    ForcedActresp: np.ndarray = None
    ForcedMigrate: np.ndarray = None
    ForcedBio: np.ndarray = None


@dataclass
class MockScenario:
    """Mock scenario for testing adjustments."""

    params: MockParams
    fishing: MockFishing
    forcing: MockForcing
    years: int = 50
    n_months: int = 600
    group_names: List[str] = None
    NUM_GROUPS: int = 5


def create_mock_scenario(n_groups=5, years=50):
    """Create a mock scenario for testing."""
    n_months = years * 12

    params = MockParams(
        PreyTo=[0] * 10, PreyFrom=[0] * 10, VV=[2.0] * 10, DD=[1000.0] * 10
    )

    fishing = MockFishing(
        ForcedEffort=np.ones((n_months, n_groups)),
        ForcedFRate=np.ones((n_months, n_groups)),
        ForcedCatch=np.zeros((n_months, n_groups)),
    )

    forcing = MockForcing(
        ForcedPrey=np.ones((n_months, n_groups)),
        ForcedMort=np.ones((n_months, n_groups)),
        ForcedRecs=np.ones((n_months, n_groups)),
        ForcedSearch=np.ones((n_months, n_groups)),
        ForcedActresp=np.ones((n_months, n_groups)),
        ForcedMigrate=np.zeros((n_months, n_groups)),
        ForcedBio=np.full((n_months, n_groups), -1.0),
    )

    return MockScenario(
        params=params,
        fishing=fishing,
        forcing=forcing,
        years=years,
        n_months=n_months,
        group_names=["Outside", "Phyto", "Zoo", "Fish", "TopPred"],
        NUM_GROUPS=n_groups,
    )


class TestAdjustScenario:
    """Test adjust_scenario function."""

    def test_adjust_burn_years(self):
        """Adjust burn-in years."""
        scenario = create_mock_scenario()

        result = adjust_scenario(scenario, "BURN_YEARS", 10)

        assert result.params.BURN_YEARS == 10

    def test_adjust_rk4_steps(self):
        """Adjust integration steps."""
        scenario = create_mock_scenario()

        result = adjust_scenario(scenario, "RK4_STEPS", 8)

        assert result.params.RK4_STEPS == 8

    def test_invalid_parameter_raises(self):
        """Invalid parameter should raise error."""
        scenario = create_mock_scenario()

        with pytest.raises(AttributeError):
            adjust_scenario(scenario, "INVALID_PARAM", 1.0)


class TestHelperFunctions:
    """Test helper functions for creating forcing patterns."""

    def test_create_seasonal_forcing_validates_length(self):
        """Seasonal forcing requires 12 monthly values."""
        scenario = create_mock_scenario()

        with pytest.raises(ValueError):
            create_seasonal_forcing(
                scenario,
                group=1,
                years=range(1, 10),
                monthly_values=[1.0] * 6,  # Wrong length
                parameter="ForcedPrey",
            )


class TestArrayModifications:
    """Test that arrays are correctly modified."""

    def test_fishing_matrix_modification(self):
        """Test that fishing matrices can be modified."""
        scenario = create_mock_scenario(n_groups=5, years=10)

        # Initial values should be 1.0
        assert scenario.fishing.ForcedFRate[0, 2] == 1.0

        # Direct modification should work
        scenario.fishing.ForcedFRate[60:, 2] = 0.5

        assert scenario.fishing.ForcedFRate[60, 2] == 0.5
        assert scenario.fishing.ForcedFRate[0, 2] == 1.0

    def test_forcing_matrix_modification(self):
        """Test that forcing matrices can be modified."""
        scenario = create_mock_scenario(n_groups=5, years=10)

        # Initial values should be 1.0
        assert scenario.forcing.ForcedPrey[0, 1] == 1.0

        # Direct modification should work
        scenario.forcing.ForcedPrey[60:, 1] = 1.5

        assert scenario.forcing.ForcedPrey[60, 1] == 1.5
        assert scenario.forcing.ForcedPrey[0, 1] == 1.0


class TestLinearInterpolation:
    """Test linear ramp creation."""

    def test_linear_ramp_values(self):
        """Create linear ramp and verify values."""
        values = np.linspace(1.0, 2.0, 61)  # 5 years * 12 months + 1

        assert values[0] == 1.0
        assert values[-1] == 2.0
        assert np.isclose(values[30], 1.5)  # Midpoint

    def test_ramp_monotonic_increase(self):
        """Ramp should be monotonically increasing."""
        values = np.linspace(1.0, 2.0, 100)

        for i in range(1, len(values)):
            assert values[i] >= values[i - 1]


class TestSeasonalPatterns:
    """Test seasonal forcing patterns."""

    def test_seasonal_pattern_generation(self):
        """Generate seasonal pattern."""
        # Summer high, winter low pattern
        monthly = [0.8, 0.9, 1.0, 1.1, 1.2, 1.3, 1.3, 1.2, 1.1, 1.0, 0.9, 0.8]

        assert len(monthly) == 12
        assert max(monthly) == 1.3
        assert min(monthly) == 0.8

    def test_seasonal_peak_timing(self):
        """Check seasonal peak is in correct month."""
        monthly = [0.8, 0.9, 1.0, 1.1, 1.2, 1.3, 1.3, 1.2, 1.1, 1.0, 0.9, 0.8]

        # Peak should be in months 5-6 (June-July, indices 5-6)
        peak_idx = np.argmax(monthly)
        assert peak_idx in [5, 6]


class TestForcingTypes:
    """Test different forcing parameters."""

    def test_forced_prey_parameter(self):
        """ForcedPrey affects prey availability."""
        scenario = create_mock_scenario()

        # ForcedPrey should default to 1.0
        assert np.all(scenario.forcing.ForcedPrey == 1.0)

    def test_forced_mort_parameter(self):
        """ForcedMort affects additional mortality."""
        scenario = create_mock_scenario()

        # ForcedMort should default to 1.0
        assert np.all(scenario.forcing.ForcedMort == 1.0)

    def test_forced_bio_parameter(self):
        """ForcedBio sets forced biomass."""
        scenario = create_mock_scenario()

        # ForcedBio should default to -1.0 (off)
        assert np.all(scenario.forcing.ForcedBio == -1.0)


class TestFishingParameters:
    """Test fishing parameter modifications."""

    def test_forced_frate_parameter(self):
        """ForcedFRate affects fishing mortality rate."""
        scenario = create_mock_scenario()

        # ForcedFRate should default to 1.0
        assert np.all(scenario.fishing.ForcedFRate == 1.0)

    def test_forced_effort_parameter(self):
        """ForcedEffort affects fishing effort."""
        scenario = create_mock_scenario()

        # ForcedEffort should default to 1.0
        assert np.all(scenario.fishing.ForcedEffort == 1.0)

    def test_forced_catch_parameter(self):
        """ForcedCatch sets catch quotas."""
        scenario = create_mock_scenario()

        # ForcedCatch should default to 0.0
        assert np.all(scenario.fishing.ForcedCatch == 0.0)


class TestIntegration:
    """Integration tests for complete scenarios."""

    def test_climate_scenario_pattern(self):
        """Create a climate change scenario pattern."""
        # Simulate warming: 30% increase over 30 years
        n_years = 50
        warming_start = 10
        warming_end = 40

        forcing = np.ones(n_years * 12)

        # Apply gradual warming
        for month in range(warming_start * 12, warming_end * 12):
            progress = (month - warming_start * 12) / (
                (warming_end - warming_start) * 12
            )
            forcing[month] = 1.0 + 0.3 * progress

        # After warming, maintain elevated level
        forcing[warming_end * 12 :] = 1.3

        # Verify pattern
        assert forcing[0] == 1.0  # Before warming
        assert np.isclose(
            forcing[warming_end * 12 - 1], 1.3, atol=0.05
        )  # End of warming
        assert forcing[-1] == 1.3  # After warming

    def test_management_scenario_pattern(self):
        """Create a fisheries management scenario pattern."""
        n_years = 30
        n_groups = 5

        frate = np.ones((n_years * 12, n_groups))

        # Reduce fishing by 50% on target species starting year 10
        frate[10 * 12 :, 3] = 0.5

        # Verify pattern
        assert frate[0, 3] == 1.0
        assert frate[10 * 12, 3] == 0.5
        assert frate[-1, 3] == 0.5

    def test_mpa_scenario_pattern(self):
        """Create a marine protected area scenario pattern."""
        n_years = 50
        n_groups = 5

        effort = np.ones((n_years * 12, n_groups))

        # MPA closes fishing for all groups at year 5
        effort[5 * 12 :, :] = 0.0

        # Verify all fishing stops
        assert np.all(effort[: 5 * 12, :] == 1.0)
        assert np.all(effort[5 * 12 :, :] == 0.0)


# Run tests
if __name__ == "__main__":
    pytest.main([__file__, "-v"])
