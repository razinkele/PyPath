"""
Tests for state-variable forcing mechanisms.

Tests the ability to force state variables (biomass, catch, recruitment, etc.)
to follow observed or prescribed time series.
"""

import pytest
import numpy as np
import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from pypath.core.forcing import (
    ForcingMode,
    StateVariable,
    ForcingFunction,
    StateForcing,
    create_biomass_forcing,
    create_recruitment_forcing,
)


class TestForcingFunction:
    """Test individual forcing functions."""

    def test_create_forcing_function(self):
        """Should create forcing function correctly."""
        func = ForcingFunction(
            group_idx=0,
            variable=StateVariable.BIOMASS,
            mode=ForcingMode.REPLACE,
            time_series=np.array([10.0, 15.0, 20.0]),
            years=np.array([2000, 2005, 2010]),
            interpolate=True,
            active=True,
        )

        assert func.group_idx == 0
        assert func.variable == StateVariable.BIOMASS
        assert func.mode == ForcingMode.REPLACE
        assert len(func.time_series) == 3

    def test_get_value_exact_year(self):
        """Should return exact value at data points."""
        func = ForcingFunction(
            group_idx=0,
            variable=StateVariable.BIOMASS,
            mode=ForcingMode.REPLACE,
            time_series=np.array([10.0, 15.0, 20.0]),
            years=np.array([2000, 2005, 2010]),
            interpolate=True,
        )

        assert func.get_value(2000) == 10.0
        assert func.get_value(2005) == 15.0
        assert func.get_value(2010) == 20.0

    def test_get_value_interpolated(self):
        """Should interpolate between data points."""
        func = ForcingFunction(
            group_idx=0,
            variable=StateVariable.BIOMASS,
            mode=ForcingMode.REPLACE,
            time_series=np.array([10.0, 20.0]),
            years=np.array([2000, 2010]),
            interpolate=True,
        )

        # Midpoint should be 15.0
        assert func.get_value(2005) == 15.0

        # Quarter point should be 12.5
        assert func.get_value(2002.5) == 12.5

    def test_get_value_no_interpolation(self):
        """Should use nearest value when not interpolating."""
        func = ForcingFunction(
            group_idx=0,
            variable=StateVariable.BIOMASS,
            mode=ForcingMode.REPLACE,
            time_series=np.array([10.0, 20.0]),
            years=np.array([2000, 2010]),
            interpolate=False,
        )

        # Should use nearest (2000 is closer)
        assert func.get_value(2003) == 10.0

        # Should use nearest (2010 is closer)
        assert func.get_value(2007) == 20.0

    def test_get_value_outside_range(self):
        """Should return NaN outside range."""
        func = ForcingFunction(
            group_idx=0,
            variable=StateVariable.BIOMASS,
            mode=ForcingMode.REPLACE,
            time_series=np.array([10.0, 20.0]),
            years=np.array([2000, 2010]),
        )

        # Before range
        assert np.isnan(func.get_value(1995))

        # After range
        assert np.isnan(func.get_value(2015))

    def test_inactive_function(self):
        """Should return NaN when inactive."""
        func = ForcingFunction(
            group_idx=0,
            variable=StateVariable.BIOMASS,
            mode=ForcingMode.REPLACE,
            time_series=np.array([10.0, 20.0]),
            years=np.array([2000, 2010]),
            active=False,
        )

        assert np.isnan(func.get_value(2005))


class TestStateForcing:
    """Test StateForcing collection."""

    def test_add_forcing_with_dict(self):
        """Should add forcing from dictionary."""
        forcing = StateForcing()
        forcing.add_forcing(
            group_idx=0,
            variable="biomass",
            time_series={2000: 10.0, 2005: 15.0, 2010: 20.0},
            mode="replace",
        )

        assert len(forcing.functions) == 1
        assert forcing.functions[0].group_idx == 0

    def test_add_forcing_with_arrays(self):
        """Should add forcing from arrays."""
        forcing = StateForcing()
        forcing.add_forcing(
            group_idx=1,
            variable=StateVariable.RECRUITMENT,
            time_series=np.array([1.0, 2.0, 1.5]),
            years=np.array([2000, 2005, 2010]),
            mode=ForcingMode.MULTIPLY,
        )

        assert len(forcing.functions) == 1
        assert forcing.functions[0].variable == StateVariable.RECRUITMENT
        assert forcing.functions[0].mode == ForcingMode.MULTIPLY

    def test_add_multiple_forcing(self):
        """Should handle multiple forcing functions."""
        forcing = StateForcing()

        # Force biomass for group 0
        forcing.add_forcing(
            group_idx=0,
            variable="biomass",
            time_series={2000: 10.0, 2010: 20.0},
            mode="replace",
        )

        # Force recruitment for group 1
        forcing.add_forcing(
            group_idx=1,
            variable="recruitment",
            time_series={2005: 2.0},
            mode="multiply",
        )

        assert len(forcing.functions) == 2

    def test_get_forcing_single_group(self):
        """Should get forcing for specific group."""
        forcing = StateForcing()
        forcing.add_forcing(
            group_idx=0, variable="biomass", time_series={2000: 10.0, 2010: 20.0}
        )

        # Get forcing for group 0
        results = forcing.get_forcing(2005, StateVariable.BIOMASS, group_idx=0)
        assert len(results) == 1
        assert results[0][1] == 15.0  # Interpolated value

        # Get forcing for group 1 (should be empty)
        results = forcing.get_forcing(2005, StateVariable.BIOMASS, group_idx=1)
        assert len(results) == 0

    def test_get_forcing_all_groups(self):
        """Should get forcing for all matching groups."""
        forcing = StateForcing()

        # Add forcing for group 0
        forcing.add_forcing(group_idx=0, variable="biomass", time_series={2000: 10.0})

        # Add forcing for group 1
        forcing.add_forcing(group_idx=1, variable="biomass", time_series={2000: 20.0})

        # Get all biomass forcing (group_idx=None)
        results = forcing.get_forcing(2000, StateVariable.BIOMASS, group_idx=None)
        assert len(results) == 2

    def test_remove_forcing(self):
        """Should remove specific forcing."""
        forcing = StateForcing()
        forcing.add_forcing(group_idx=0, variable="biomass", time_series={2000: 10.0})
        forcing.add_forcing(
            group_idx=1, variable="recruitment", time_series={2000: 2.0}
        )

        assert len(forcing.functions) == 2

        # Remove biomass forcing for group 0
        forcing.remove_forcing(0, "biomass")

        assert len(forcing.functions) == 1
        assert forcing.functions[0].group_idx == 1


class TestForcingModes:
    """Test different forcing modes."""

    def test_replace_mode(self):
        """Replace mode should override computed value."""
        state = np.array([10.0, 20.0, 30.0])
        forced_value = 50.0

        # Simulate REPLACE mode
        state_new = state.copy()
        state_new[1] = forced_value

        assert state_new[1] == 50.0
        assert state_new[0] == 10.0  # Unchanged
        assert state_new[2] == 30.0  # Unchanged

    def test_add_mode(self):
        """Add mode should add to computed value."""
        state = np.array([10.0, 20.0, 30.0])
        forced_value = 5.0

        # Simulate ADD mode
        state_new = state.copy()
        state_new[1] += forced_value

        assert state_new[1] == 25.0
        assert state_new[0] == 10.0
        assert state_new[2] == 30.0

    def test_multiply_mode(self):
        """Multiply mode should scale computed value."""
        state = np.array([10.0, 20.0, 30.0])
        forced_value = 2.0

        # Simulate MULTIPLY mode
        state_new = state.copy()
        state_new[1] *= forced_value

        assert state_new[1] == 40.0
        assert state_new[0] == 10.0
        assert state_new[2] == 30.0

    def test_rescale_mode(self):
        """Rescale mode should rescale to target."""
        state = np.array([10.0, 20.0, 30.0])
        forced_value = 50.0

        # Simulate RESCALE mode
        state_new = state.copy()
        if state[1] > 0:
            state_new[1] = forced_value

        assert state_new[1] == 50.0


class TestConvenienceFunctions:
    """Test convenience functions for creating forcing."""

    def test_create_biomass_forcing(self):
        """Should create biomass forcing easily."""
        forcing = create_biomass_forcing(
            group_idx=0,
            observed_biomass={2000: 15.0, 2005: 18.0, 2010: 16.0},
            mode="replace",
        )

        assert len(forcing.functions) == 1
        assert forcing.functions[0].variable == StateVariable.BIOMASS
        assert forcing.functions[0].mode == ForcingMode.REPLACE

    def test_create_recruitment_forcing(self):
        """Should create recruitment forcing easily."""
        forcing = create_recruitment_forcing(
            group_idx=3,
            recruitment_multiplier={2005: 3.0, 2010: 0.5},
            interpolate=False,
        )

        assert len(forcing.functions) == 1
        assert forcing.functions[0].variable == StateVariable.RECRUITMENT
        assert forcing.functions[0].mode == ForcingMode.MULTIPLY
        assert forcing.functions[0].interpolate == False


class TestRealisticScenarios:
    """Test realistic forcing scenarios."""

    def test_phytoplankton_seasonal_forcing(self):
        """Simulate seasonal phytoplankton bloom forcing."""
        forcing = StateForcing()

        # Simple seasonal pattern: high in summer, low in winter
        years = np.array([2000.0, 2000.25, 2000.5, 2000.75, 2001.0])
        biomass_seasonal = np.array([15.0, 20.0, 15.0, 10.0, 15.0])  # Explicit pattern

        forcing.add_forcing(
            group_idx=0,
            variable="biomass",
            time_series=biomass_seasonal,
            years=years,
            mode="replace",
            interpolate=True,
        )

        # Check values
        # Spring/summer (0.25) should be high
        spring_value = forcing.functions[0].get_value(2000.25)
        assert spring_value == 20.0

        # Winter (0.0) should be baseline
        winter_value = forcing.functions[0].get_value(2000.0)
        assert winter_value == 15.0

        # Interpolated value between spring and fall
        mid_value = forcing.functions[0].get_value(2000.375)
        assert 15.0 < mid_value < 20.0

    def test_recruitment_pulse(self):
        """Simulate strong recruitment event."""
        forcing = StateForcing()

        # Normal recruitment except 2x in 2005
        forcing.add_forcing(
            group_idx=3,  # Herring
            variable="recruitment",
            time_series={2000: 1.0, 2005: 2.0, 2010: 1.0},
            mode="multiply",
            interpolate=False,
        )

        # Should get 2x in 2005
        pulse = forcing.functions[0].get_value(2005)
        assert pulse == 2.0

        # For nearest without interpolation, 2003 is closer to 2005
        # So it returns 2.0, not 1.0
        # Use 2002 instead (closer to 2000)
        normal = forcing.functions[0].get_value(2002)
        assert normal == 1.0

    def test_fishing_moratorium(self):
        """Simulate fishing ban period."""
        forcing = StateForcing()

        # No fishing 2005-2010
        forcing.add_forcing(
            group_idx=5,  # Target species
            variable="fishing_mortality",
            time_series={2000: 0.2, 2005: 0.0, 2010: 0.0, 2015: 0.2},
            mode="replace",
            interpolate=True,
        )

        # Should have zero fishing in ban period
        assert forcing.functions[0].get_value(2007) == 0.0

        # Should have fishing before/after
        assert forcing.functions[0].get_value(2002) > 0.0
        assert forcing.functions[0].get_value(2012) > 0.0

    def test_climate_driven_primary_production(self):
        """Simulate climate-driven PP changes."""
        forcing = StateForcing()

        # Gradually increasing PP due to warming
        years = np.array([2000, 2010, 2020, 2030])
        pp_multiplier = np.array([1.0, 1.1, 1.25, 1.4])

        forcing.add_forcing(
            group_idx=0,  # Phytoplankton
            variable="primary_production",
            time_series=pp_multiplier,
            years=years,
            mode="multiply",
            interpolate=True,
        )

        # Should increase over time
        assert forcing.functions[0].get_value(2000) == 1.0
        assert forcing.functions[0].get_value(2020) == 1.25
        assert forcing.functions[0].get_value(2030) == 1.4

        # Should interpolate
        mid_value = forcing.functions[0].get_value(2015)
        assert 1.1 < mid_value < 1.25


class TestEdgeCases:
    """Test edge cases and error handling."""

    def test_empty_forcing(self):
        """Should handle empty forcing list."""
        forcing = StateForcing()

        results = forcing.get_forcing(2005, StateVariable.BIOMASS)
        assert len(results) == 0

    def test_single_time_point(self):
        """Should handle single time point."""
        forcing = StateForcing()
        forcing.add_forcing(
            group_idx=0, variable="biomass", time_series={2005: 15.0}, interpolate=False
        )

        # Within range
        assert forcing.functions[0].get_value(2005) == 15.0

        # Outside range
        assert np.isnan(forcing.functions[0].get_value(2000))
        assert np.isnan(forcing.functions[0].get_value(2010))

    def test_negative_values(self):
        """Should handle negative forced values (e.g., migration)."""
        forcing = StateForcing()
        forcing.add_forcing(
            group_idx=2,
            variable="migration",
            time_series={2005: -5.0},  # Emigration
            mode="add",
        )

        assert forcing.functions[0].get_value(2005) == -5.0

    def test_very_large_values(self):
        """Should handle very large forced values."""
        forcing = StateForcing()
        forcing.add_forcing(
            group_idx=0, variable="biomass", time_series={2005: 1e6}, mode="replace"
        )

        assert forcing.functions[0].get_value(2005) == 1e6

    def test_zero_values(self):
        """Should handle zero forced values."""
        forcing = StateForcing()
        forcing.add_forcing(
            group_idx=3,
            variable="recruitment",
            time_series={2005: 0.0},  # Recruitment failure
            mode="replace",
        )

        assert forcing.functions[0].get_value(2005) == 0.0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
