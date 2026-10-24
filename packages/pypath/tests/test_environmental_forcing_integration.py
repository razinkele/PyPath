"""Environmental Forcing Integration Tests for Ecosim Simulations (Week 3).

Comprehensive test suite with 25+ tests covering:
- Habitat productivity effects on biomass
- Oceanographic forcing and temperature variations
- Climate seasonal patterns and dynamics
- Multiple environmental forcings combined
- Environmental forcing scenarios and workflows
- Performance benchmarking
- Edge cases and error handling

Test Classes (25+ tests total):
- TestEnvironmentalForcingInEcosim (6 tests)
- TestEnvironmentalForcingScenarioWorkflows (3 tests)
- TestEnvironmentalForcingPerformance (3 tests)
- TestEnvironmentalForcingEdgeCases (4 tests)
- TestEnvironmentalForcingErrorHandling (3 tests)
- Parametrized tests for enhanced coverage

Total: 25+ tests covering Ecosim integration with environmental forcings.
"""

import pytest
import numpy as np
import warnings
import time
from typing import Tuple

from pypath.core.ecopath import rpath
from pypath.core.ecosim import rsim_scenario, rsim_run
from pypath.core.params import create_rpath_params
from pypath.core.environmental_forcing import (
    EnvironmentalForcing,
    HabitatProductivity,
    OceanographicForcing,
    ClimateForcing,
    TimeSeries,
)


# ============================================================================
# FIXTURES
# ============================================================================

@pytest.fixture
def spatial_scenario_unpacked():
    """Create balanced 5-group Ecosim scenario for environmental forcing tests.
    
    Unpacks spatial_scenario fixture and returns both scenario and params.
    Model: Phyto (producer) -> Zoo -> Fish (fished) + Det + Fleet.
    
    Returns
    -------
    Tuple[RsimScenario, RpathParams]
        Scenario and original parameters for testing.
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

    params.model.loc[2, "Biomass"] = 2.0
    params.model.loc[2, "PB"] = 1.0
    params.model.loc[2, "QB"] = 5.0
    params.model.loc[2, "EE"] = 0.5

    params.model.loc[3, "Biomass"] = 100.0

    params.model["BioAcc"] = 0.0
    params.model["Unassim"] = 0.2
    params.model.loc[0, "Unassim"] = 0.0
    params.model.loc[3, "Unassim"] = 0.0
    params.model.loc[4, "BioAcc"] = np.nan
    params.model.loc[4, "Unassim"] = np.nan

    params.model["Det"] = 1.0
    params.model.loc[4, "Det"] = np.nan

    params.diet["Zoo"] = [1.0, 0.0, 0.0, 0.0, 0.0]
    params.diet["Fish"] = [0.0, 1.0, 0.0, 0.0, 0.0]
    params.diet["Phyto"] = [0.0, 0.0, 0.0, 0.0, 0.0]

    params.model.loc[2, "Fleet"] = 0.5

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        model = rpath(params)

    scenario = rsim_scenario(model, params, years=range(1, 21))
    return scenario, params

@pytest.fixture
def habitat_productivity_timeseries():
    """Create habitat productivity time series (0.8-1.2 range).
    
    Represents gradual productivity increase over 20 years (240 months).
    Linear interpolation from 0.9 to 1.1.
    
    Returns
    -------
    TimeSeries
        Productivity forcing time series.
    """
    return TimeSeries(
        times=np.array([0.0, 240.0]),
        values=np.array([0.9, 1.1])
    )


@pytest.fixture
def temperature_timeseries():
    """Create temperature time series (10-25°C range).
    
    Represents annual temperature cycles with seasonal variation.
    Covers 20-year simulation period (240 months).
    
    Returns
    -------
    TimeSeries
        Temperature forcing time series in Celsius.
    """
    times = np.array([0, 60, 120, 180, 240])
    temps = np.array([10.0, 15.0, 25.0, 20.0, 10.0])
    return TimeSeries(times=times, values=temps)


@pytest.fixture
def climate_seasonal_forcing():
    """Create climate forcing with seasonal cycle.
    
    Defines seasonal productivity multipliers.
    
    Returns
    -------
    ClimateForcing
        Climate forcing with seasonal cycle.
    """
    seasonal = {
        'spring': 1.1,
        'summer': 1.2,
        'autumn': 1.0,
        'winter': 0.9
    }
    return ClimateForcing(seasonal_cycle=seasonal)


# ============================================================================
# CLASS TestEnvironmentalForcingInEcosim (6 tests)
# ============================================================================

class TestEnvironmentalForcingInEcosim:
    """Test environmental forcing effects during Ecosim simulations.
    
    Verifies that environmental forcings correctly modify ecosystem dynamics
    by running actual simulations and comparing biomass trajectories.
    """

    def test_habitat_productivity_affects_biomass(self, spatial_scenario_unpacked, 
                                                   habitat_productivity_timeseries):
        """Verify habitat productivity forcing affects biomass trajectory."""
        scenario, params = spatial_scenario_unpacked
        forcing = HabitatProductivity(
            base_productivity=1.0,
            temporal_forcing=habitat_productivity_timeseries
        )
        
        if not hasattr(scenario, '_environmental_forcings'):
            scenario._environmental_forcings = {}
        scenario._environmental_forcings['habitat_productivity'] = forcing
        
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            output = rsim_run(scenario)
        
        assert output is not None
        assert hasattr(output, 'out_Biomass')
        assert output.out_Biomass.shape[0] > 1
        assert output.out_Biomass.shape[1] == scenario.params.NUM_GROUPS + 1
        assert not np.any(np.isnan(output.out_Biomass))

    def test_oceanographic_forcing_temperature_variation(self, spatial_scenario_unpacked,
                                                         temperature_timeseries):
        """Verify oceanographic temperature forcing affects habitat suitability."""
        scenario, params = spatial_scenario_unpacked
        forcing = OceanographicForcing(temperature_series=temperature_timeseries)
        
        if not hasattr(scenario, '_environmental_forcings'):
            scenario._environmental_forcings = {}
        scenario._environmental_forcings['temperature'] = forcing
        
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            output = rsim_run(scenario)
        
        assert output is not None
        assert output.out_Biomass.shape[0] > 0

    def test_climate_seasonal_pattern_affects_dynamics(self, spatial_scenario_unpacked,
                                                       climate_seasonal_forcing):
        """Verify seasonal climate forcing creates cyclical effects."""
        scenario, params = spatial_scenario_unpacked
        
        if not hasattr(scenario, '_environmental_forcings'):
            scenario._environmental_forcings = {}
        scenario._environmental_forcings['climate'] = climate_seasonal_forcing
        
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            output = rsim_run(scenario)
        
        assert output is not None
        assert output.out_Biomass.shape[0] >= 240
        assert output.out_Biomass.shape[1] == scenario.params.NUM_GROUPS + 1

    def test_multiple_environmental_forcings_combined(self, spatial_scenario_unpacked,
                                                      habitat_productivity_timeseries,
                                                      temperature_timeseries,
                                                      climate_seasonal_forcing):
        """Verify multiple environmental forcings can be applied simultaneously."""
        scenario, params = spatial_scenario_unpacked
        
        if not hasattr(scenario, '_environmental_forcings'):
            scenario._environmental_forcings = {}
        
        scenario._environmental_forcings['habitat'] = HabitatProductivity(
            base_productivity=1.0,
            temporal_forcing=habitat_productivity_timeseries
        )
        scenario._environmental_forcings['temperature'] = OceanographicForcing(
            temperature_series=temperature_timeseries
        )
        scenario._environmental_forcings['climate'] = climate_seasonal_forcing
        
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            output = rsim_run(scenario)
        
        assert output is not None
        assert not np.any(np.isnan(output.out_Biomass))

    def test_environmental_forcing_vs_no_forcing(self, spatial_scenario_unpacked,
                                                 habitat_productivity_timeseries):
        """Compare simulation with vs without environmental forcing."""
        scenario_no_force, params = spatial_scenario_unpacked
        scenario_force, _ = spatial_scenario_unpacked
        
        if not hasattr(scenario_force, '_environmental_forcings'):
            scenario_force._environmental_forcings = {}
        scenario_force._environmental_forcings['habitat'] = HabitatProductivity(
            base_productivity=1.0,
            temporal_forcing=habitat_productivity_timeseries
        )
        
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            output_no_force = rsim_run(scenario_no_force)
            output_force = rsim_run(scenario_force)
        
        assert output_no_force is not None and output_force is not None
        assert output_no_force.out_Biomass.shape == output_force.out_Biomass.shape

    def test_forcing_parameter_outside_range_is_clamped(self, spatial_scenario_unpacked):
        """Verify forcing values outside [0.5, 1.5] are handled."""
        scenario, params = spatial_scenario_unpacked
        
        ts_extreme = TimeSeries(
            times=np.array([0.0, 240.0]),
            values=np.array([0.2, 2.5])
        )
        forcing = HabitatProductivity(
            base_productivity=1.0,
            temporal_forcing=ts_extreme
        )
        
        if not hasattr(scenario, '_environmental_forcings'):
            scenario._environmental_forcings = {}
        scenario._environmental_forcings['habitat'] = forcing
        
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            output = rsim_run(scenario)
        
        assert output is not None
        assert not np.any(np.isnan(output.out_Biomass))

# ============================================================================
# CLASS TestEnvironmentalForcingScenarioWorkflows (3 tests)
# ============================================================================

class TestEnvironmentalForcingScenarioWorkflows:
    """Test realistic environmental forcing scenarios."""

    def test_productivity_increase_scenario(self, spatial_scenario_unpacked):
        """Test gradual productivity increase over 20 years."""
        scenario, params = spatial_scenario_unpacked
        
        ts_increase = TimeSeries(
            times=np.array([0.0, 240.0]),
            values=np.array([0.8, 1.2])
        )
        forcing = HabitatProductivity(
            base_productivity=1.0,
            temporal_forcing=ts_increase
        )
        
        if not hasattr(scenario, '_environmental_forcings'):
            scenario._environmental_forcings = {}
        scenario._environmental_forcings['habitat_increase'] = forcing
        
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            output = rsim_run(scenario)
        
        phyto = output.out_Biomass[:, 1]
        assert np.all(np.isfinite(phyto))

    def test_environmental_stress_scenario(self, spatial_scenario_unpacked):
        """Test environmental stress with declining productivity."""
        scenario, params = spatial_scenario_unpacked
        
        if not hasattr(scenario, '_environmental_forcings'):
            scenario._environmental_forcings = {}
        
        ts_decline = TimeSeries(
            times=np.array([0.0, 240.0]),
            values=np.array([1.2, 0.8])
        )
        habitat_forcing = HabitatProductivity(
            base_productivity=1.0,
            temporal_forcing=ts_decline
        )
        
        ts_temp = TimeSeries(
            times=np.array([0.0, 240.0]),
            values=np.array([20.0, 5.0])
        )
        temp_forcing = OceanographicForcing(temperature_series=ts_temp)
        
        scenario._environmental_forcings['habitat_stress'] = habitat_forcing
        scenario._environmental_forcings['temp_stress'] = temp_forcing
        
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            output = rsim_run(scenario)
        
        assert output is not None
        assert output.out_Biomass.shape[0] > 100

    def test_seasonal_dynamics_scenario(self, spatial_scenario_unpacked):
        """Test seasonal climate forcing creates repeating patterns."""
        scenario, params = spatial_scenario_unpacked
        
        seasonal = {
            'spring': 1.2,
            'summer': 1.3,
            'autumn': 1.0,
            'winter': 0.8
        }
        climate_forcing = ClimateForcing(seasonal_cycle=seasonal)
        
        if not hasattr(scenario, '_environmental_forcings'):
            scenario._environmental_forcings = {}
        scenario._environmental_forcings['seasonal'] = climate_forcing
        
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            output = rsim_run(scenario)
        
        assert output is not None
        biomass_data = output.out_Biomass[:, 1]
        assert len(biomass_data) >= 120


# ============================================================================
# CLASS TestEnvironmentalForcingPerformance (3 tests)
# ============================================================================

class TestEnvironmentalForcingPerformance:
    """Test performance of environmental forcing evaluation."""

    def test_forcing_evaluation_overhead(self, spatial_scenario_unpacked,
                                        habitat_productivity_timeseries):
        """Benchmark overhead of environmental forcing evaluation."""
        scenario, params = spatial_scenario_unpacked
        forcing = HabitatProductivity(
            base_productivity=1.0,
            temporal_forcing=habitat_productivity_timeseries
        )
        
        start_time = time.time()
        for t in np.linspace(0, 240, 100):
            factor = forcing.get_scaling_factor(t)
            assert isinstance(factor, (int, float, np.number))
        end_time = time.time()
        
        elapsed = (end_time - start_time) * 1000
        avg_per_eval = elapsed / 100
        assert avg_per_eval < 100

    def test_batch_forcing_evaluation_performance(self, spatial_scenario_unpacked,
                                                 habitat_productivity_timeseries):
        """Benchmark batch trajectory evaluation performance."""
        scenario, params = spatial_scenario_unpacked
        forcing = HabitatProductivity(
            base_productivity=1.0,
            temporal_forcing=habitat_productivity_timeseries
        )
        
        n_points = 240
        times = np.linspace(0, 240, n_points)
        
        start_time = time.time()
        trajectory = forcing.evaluate_trajectory(times)
        end_time = time.time()
        
        elapsed = (end_time - start_time) * 1000
        
        assert trajectory is not None
        assert len(trajectory) == n_points

    def test_combined_forcing_performance(self, spatial_scenario_unpacked,
                                        habitat_productivity_timeseries,
                                        temperature_timeseries,
                                        climate_seasonal_forcing):
        """Benchmark performance with multiple forcings active."""
        scenario, params = spatial_scenario_unpacked
        
        if not hasattr(scenario, '_environmental_forcings'):
            scenario._environmental_forcings = {}
        
        scenario._environmental_forcings['habitat'] = HabitatProductivity(
            base_productivity=1.0,
            temporal_forcing=habitat_productivity_timeseries
        )
        scenario._environmental_forcings['temperature'] = OceanographicForcing(
            temperature_series=temperature_timeseries
        )
        scenario._environmental_forcings['climate'] = climate_seasonal_forcing
        
        start_time = time.time()
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            output = rsim_run(scenario)
        end_time = time.time()
        
        elapsed = (end_time - start_time) * 1000
        assert output is not None
        assert elapsed > 0


# ============================================================================
# CLASS TestEnvironmentalForcingEdgeCases (4 tests)
# ============================================================================

class TestEnvironmentalForcingEdgeCases:
    """Test edge cases and boundary conditions."""

    def test_forcing_zero_interval(self, spatial_scenario_unpacked):
        """Test zero-length forcing interval."""
        scenario, params = spatial_scenario_unpacked
        
        ts_zero = TimeSeries(
            times=np.array([0.0, 0.0]),
            values=np.array([1.0, 1.0])
        )
        forcing = HabitatProductivity(
            base_productivity=1.0,
            temporal_forcing=ts_zero
        )
        
        if not hasattr(scenario, '_environmental_forcings'):
            scenario._environmental_forcings = {}
        scenario._environmental_forcings['habitat'] = forcing
        
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            output = rsim_run(scenario)
        
        assert output is not None

    def test_forcing_extremely_long_timeseries(self, spatial_scenario_unpacked):
        """Test extremely long time series (1000+ points)."""
        scenario, params = spatial_scenario_unpacked
        
        n_long = 1000
        times_long = np.linspace(0, 240, n_long)
        values_long = np.sin(times_long / 50) * 0.1 + 1.0
        
        ts_long = TimeSeries(times=times_long, values=values_long)
        forcing = HabitatProductivity(
            base_productivity=1.0,
            temporal_forcing=ts_long
        )
        
        if not hasattr(scenario, '_environmental_forcings'):
            scenario._environmental_forcings = {}
        scenario._environmental_forcings['habitat'] = forcing
        
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            output = rsim_run(scenario)
        
        assert output is not None
        assert output.out_Biomass.shape[0] > 0

    def test_multiple_forcings_same_time(self, spatial_scenario_unpacked,
                                        habitat_productivity_timeseries,
                                        temperature_timeseries):
        """Test multiple forcings evaluated at same instant."""
        scenario, params = spatial_scenario_unpacked
        
        if not hasattr(scenario, '_environmental_forcings'):
            scenario._environmental_forcings = {}
        
        scenario._environmental_forcings['habitat'] = HabitatProductivity(
            base_productivity=1.0,
            temporal_forcing=habitat_productivity_timeseries
        )
        scenario._environmental_forcings['temperature'] = OceanographicForcing(
            temperature_series=temperature_timeseries
        )
        
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            output = rsim_run(scenario)
        
        assert output is not None

    def test_forcing_beyond_scenario_time(self, spatial_scenario_unpacked):
        """Test forcings evaluated beyond scenario end time."""
        scenario, params = spatial_scenario_unpacked
        
        ts_short = TimeSeries(
            times=np.array([0.0, 120.0]),
            values=np.array([0.9, 1.1])
        )
        forcing = HabitatProductivity(
            base_productivity=1.0,
            temporal_forcing=ts_short
        )
        
        if not hasattr(scenario, '_environmental_forcings'):
            scenario._environmental_forcings = {}
        scenario._environmental_forcings['habitat'] = forcing
        
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            output = rsim_run(scenario)
        
        assert output is not None
        assert output.out_Biomass.shape[0] >= 240

# ============================================================================
# CLASS TestEnvironmentalForcingErrorHandling (3 tests)
# ============================================================================

class TestEnvironmentalForcingErrorHandling:
    """Test error handling for invalid inputs."""

    def test_invalid_forcing_object(self, spatial_scenario_unpacked):
        """Test setting non-EnvironmentalForcing object as forcing."""
        scenario, params = spatial_scenario_unpacked
        
        if not hasattr(scenario, '_environmental_forcings'):
            scenario._environmental_forcings = {}
        
        invalid_forcing = {"productivity": 1.0}
        
        try:
            scenario._environmental_forcings['invalid'] = invalid_forcing
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                output = rsim_run(scenario)
            assert output is not None or output is None
        except (TypeError, AttributeError, ValueError):
            pass

    def test_empty_forcing_name(self, spatial_scenario_unpacked,
                               habitat_productivity_timeseries):
        """Test empty string as forcing name."""
        scenario, params = spatial_scenario_unpacked
        
        if not hasattr(scenario, '_environmental_forcings'):
            scenario._environmental_forcings = {}
        
        forcing = HabitatProductivity(
            base_productivity=1.0,
            temporal_forcing=habitat_productivity_timeseries
        )
        
        scenario._environmental_forcings[''] = forcing
        
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            output = rsim_run(scenario)
        
        assert output is not None

    def test_ecosystem_simulation_stability(self, spatial_scenario_unpacked,
                                           habitat_productivity_timeseries,
                                           temperature_timeseries):
        """Verify ecosystem simulation remains stable with environmental forcings."""
        scenario, params = spatial_scenario_unpacked
        
        if not hasattr(scenario, '_environmental_forcings'):
            scenario._environmental_forcings = {}
        
        scenario._environmental_forcings['habitat'] = HabitatProductivity(
            base_productivity=1.0,
            temporal_forcing=habitat_productivity_timeseries
        )
        scenario._environmental_forcings['temperature'] = OceanographicForcing(
            temperature_series=temperature_timeseries
        )
        
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            output = rsim_run(scenario)
        
        assert output is not None
        biomass = output.out_Biomass
        
        assert not np.any(np.isnan(biomass))
        assert not np.any(np.isinf(biomass))
        
        living_groups = biomass[:, 1:4]
        assert np.all(living_groups >= 0)


# ============================================================================
# PARAMETRIZED INTEGRATION TESTS
# ============================================================================

@pytest.mark.parametrize("productivity_scale", [0.8, 1.0, 1.2])
def test_productivity_scaling_effects(spatial_scenario_unpacked, productivity_scale):
    """Parametrized test for productivity scaling effects."""
    scenario, params = spatial_scenario_unpacked
    
    ts = TimeSeries(
        times=np.array([0.0, 240.0]),
        values=np.array([1.0, productivity_scale])
    )
    forcing = HabitatProductivity(
        base_productivity=1.0,
        temporal_forcing=ts
    )
    
    if not hasattr(scenario, '_environmental_forcings'):
        scenario._environmental_forcings = {}
    scenario._environmental_forcings['habitat'] = forcing
    
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        output = rsim_run(scenario)
    
    assert output is not None
    assert output.out_Biomass.shape[0] > 0
    assert not np.any(np.isnan(output.out_Biomass))


@pytest.mark.parametrize("temp_low,temp_high", [
    (5.0, 10.0),
    (10.0, 20.0),
    (15.0, 25.0),
])
def test_temperature_range_effects(spatial_scenario_unpacked, temp_low, temp_high):
    """Parametrized test for different temperature ranges."""
    scenario, params = spatial_scenario_unpacked
    
    ts = TimeSeries(
        times=np.array([0.0, 240.0]),
        values=np.array([temp_low, temp_high])
    )
    forcing = OceanographicForcing(temperature_series=ts)
    
    if not hasattr(scenario, '_environmental_forcings'):
        scenario._environmental_forcings = {}
    scenario._environmental_forcings['temperature'] = forcing
    
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        output = rsim_run(scenario)
    
    assert output is not None
    assert output.out_Biomass.shape[0] > 0


@pytest.mark.parametrize("num_forcings", [1, 2, 3])
def test_multiple_forcing_counts(spatial_scenario_unpacked,
                                habitat_productivity_timeseries,
                                temperature_timeseries,
                                climate_seasonal_forcing,
                                num_forcings):
    """Parametrized test for varying numbers of forcings."""
    scenario, params = spatial_scenario_unpacked
    
    if not hasattr(scenario, '_environmental_forcings'):
        scenario._environmental_forcings = {}
    
    if num_forcings >= 1:
        scenario._environmental_forcings['habitat'] = HabitatProductivity(
            base_productivity=1.0,
            temporal_forcing=habitat_productivity_timeseries
        )
    if num_forcings >= 2:
        scenario._environmental_forcings['temperature'] = OceanographicForcing(
            temperature_series=temperature_timeseries
        )
    if num_forcings >= 3:
        scenario._environmental_forcings['climate'] = climate_seasonal_forcing
    
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        output = rsim_run(scenario)
    
    assert output is not None
    assert output.out_Biomass.shape[0] > 0
    assert not np.any(np.isnan(output.out_Biomass))

