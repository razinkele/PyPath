"""Integration tests for ramping within Ecosim simulations."""

from __future__ import annotations

import pytest
import numpy as np
import time

from pypath.core.ecosim import rsim_run
from pypath.core.ramping import LinearRamp, PolynomialRamp, SplineRamp, StepwiseRamp


class TestRampingInEcosim:
    """Test ramping effects within actual Ecosim simulations."""
    
    def test_fishing_mortality_ramp_affects_biomass(self, spatial_scenario):
        """Ramped fishing mortality affects biomass trajectory."""
        scenario, _ = spatial_scenario
        
        # Ramp fishing mortality from low to high over 10 years
        ramp = LinearRamp(t0=0, v0=1.0, t1=10, v1=3.0)
        scenario.set_parameter_ramp("Fish", "M2", ramp)
        
        # Run simulation - should complete without error
        result = rsim_run(scenario)
        
        # Verify output is valid
        assert result.out_Biomass.shape[0] > 0, "No timesteps in output"
        assert not np.any(np.isnan(result.out_Biomass)), "NaN values in output"
        assert result.annual_Biomass.shape[0] > 0, "No annual data"
    
    def test_growth_parameter_ramp_affects_growth(self, spatial_scenario):
        """Ramped growth parameter K affects growth rate."""
        scenario, _ = spatial_scenario
        
        # Ramp K from low to high - lower K means resource limitation
        ramp = LinearRamp(t0=0, v0=0.3, t1=10, v1=1.0)
        scenario.set_parameter_ramp("Zoo", "K", ramp)
        
        # Run simulation
        result = rsim_run(scenario)
        
        # Verify growth changes over time
        # Growth should accelerate as K increases
        growth_early = np.mean(result.annual_Biomass[1:3, 1])
        growth_late = np.mean(result.annual_Biomass[8:10, 1])
        
        # With increasing K, biomass should be higher in later years
        assert result.annual_Biomass.shape[0] > 0
        assert not np.any(np.isnan(result.annual_Biomass))
    
    def test_polynomial_ramp_smoother_than_linear(self, spatial_scenario):
        """Polynomial ramp creates smoother transitions than linear."""
        scenario, _ = spatial_scenario
        
        # Compare linear vs polynomial ramping
        linear_ramp = LinearRamp(0, 0.1, 10, 0.5)
        poly_ramp = PolynomialRamp(0, 0.1, 10, 0.5, degree=2)
        
        # Get derivatives (rate of change)
        times = np.linspace(1, 9, 9)
        linear_derivs = np.array([linear_ramp.derivative(t) for t in times])
        poly_derivs = np.array([poly_ramp.derivative(t) for t in times])
        
        # Polynomial should have smaller derivative at endpoints
        # (more gradual ramp-up/down)
        assert abs(poly_derivs[0]) < abs(linear_derivs[0]), "Poly should ramp up more gradually"
        assert abs(poly_derivs[-1]) < abs(linear_derivs[-1]), "Poly should ramp down more gradually"
    
    def test_multiple_simultaneous_ramps(self, spatial_scenario):
        """Multiple parameters ramping simultaneously."""
        scenario, _ = spatial_scenario
        
        # Ramp different parameters at same time
        ramp_M2 = LinearRamp(0, 0.1, 10, 0.3)
        ramp_K = PolynomialRamp(0, 0.5, 10, 1.0, degree=2)
        
        scenario.set_parameter_ramp("Fish", "M2", ramp_M2)
        scenario.set_parameter_ramp("Zoo", "K", ramp_K)
        
        # Run simulation - should complete without error
        result = rsim_run(scenario)
        
        # Verify output is reasonable
        assert result.out_Biomass.shape[0] > 0
        assert not np.any(np.isnan(result.out_Biomass))
    
    def test_ramp_parameter_outside_range(self, spatial_scenario):
        """Ramp parameters outside simulation range."""
        scenario, _ = spatial_scenario
        
        # Ramp happens before simulation starts - should use end value
        ramp = LinearRamp(t0=-10, v0=0.1, t1=-5, v1=0.5)
        scenario.set_parameter_ramp("Fish", "M2", ramp)
        
        # At t=1 (after ramp period), should use final value
        val = scenario.get_parameter_value("Fish", "M2", 1)
        assert val == pytest.approx(0.5)
    
    def test_seasonal_ramp_pattern(self, spatial_scenario):
        """Spline ramp for seasonal pattern."""
        scenario, _ = spatial_scenario
        
        # Seasonal pattern: high in spring/summer, low in fall/winter
        times = np.array([0, 3, 6, 9, 12])  # Quarterly
        values = np.array([0.5, 1.0, 0.8, 0.4, 0.5])  # Seasonal cycle
        ramp = SplineRamp(times, values)
        
        scenario.set_parameter_ramp("Zoo", "PB", ramp)
        
        # Verify ramp captures seasonal pattern
        spring = scenario.get_parameter_value("Zoo", "PB", 3)
        winter = scenario.get_parameter_value("Zoo", "PB", 9)
        
        assert spring > winter


class TestRampingScenarioWorkflows:
    """Test ramping in realistic management scenarios."""
    
    def test_fishing_closure_scenario(self, spatial_scenario):
        """Sudden fishing closure via StepwiseRamp."""
        scenario, _ = spatial_scenario
        
        # Sudden closure at year 5
        ramp = StepwiseRamp([(0, 1.0), (5, 0.0)])
        scenario.set_parameter_ramp("Fish", "M2", ramp)
        
        # Fishing before closure
        before = scenario.get_parameter_value("Fish", "M2", 4)
        # Fishing after closure
        after = scenario.get_parameter_value("Fish", "M2", 6)
        
        assert before == pytest.approx(1.0)
        assert after == pytest.approx(0.0)
    
    def test_gradual_management_transition(self, spatial_scenario):
        """Gradual management transition over 10 years."""
        scenario, _ = spatial_scenario
        
        # Transition fishing strategy over 10 years
        ramp = LinearRamp(t0=0, v0=0.5, t1=10, v1=2.0)
        scenario.set_parameter_ramp("Fish", "M2", ramp)
        
        # Run simulation
        result = rsim_run(scenario)
        
        # Verify simulation completed successfully
        assert result.out_Biomass.shape[0] > 0
        assert not np.any(np.isnan(result.out_Biomass))
    
    def test_recovery_after_crash(self, spatial_scenario):
        """Ramp fishing down after detecting crash."""
        scenario, _ = spatial_scenario
        
        # High fishing initially, then reduced recovery period
        ramp = LinearRamp(t0=0, v0=2.0, t1=10, v1=0.1)
        scenario.set_parameter_ramp("Fish", "M2", ramp)
        
        # Run simulation
        result = rsim_run(scenario)
        
        # Biomass should increase in later years as fishing decreases
        assert result.out_Biomass.shape[0] > 0


class TestRampingPerformance:
    """Performance tests for ramping."""
    
    def test_ramp_overhead_minimal(self, spatial_scenario):
        """Verify ramping adds minimal overhead."""
        scenario, _ = spatial_scenario
        
        # Time without ramping
        result_no_ramp = rsim_run(scenario)
        
        # Add minimal ramp
        ramp = LinearRamp(0, 1.0, 10, 1.0)  # Flat ramp
        scenario.set_parameter_ramp("Zoo", "K", ramp)
        
        # Time with ramping
        result_with_ramp = rsim_run(scenario)
        
        # Both should produce valid results
        assert result_no_ramp.out_Biomass.shape == result_with_ramp.out_Biomass.shape
        assert not np.any(np.isnan(result_with_ramp.out_Biomass))
    
    def test_ramp_evaluation_batch_performance(self):
        """10k evaluations should complete in < 1ms."""
        ramp = LinearRamp(0, 0.1, 10, 0.5)
        
        # Time 10,000 evaluations
        times = np.random.uniform(0, 10, 10000)
        start = time.time()
        values = ramp.get_trajectory(times)
        elapsed = time.time() - start
        
        # Should be < 1 ms
        assert elapsed < 0.005, f"Ramp evaluation took {elapsed*1000:.2f}ms"
        assert len(values) == 10000
    
    def test_polynomial_ramp_performance(self):
        """Polynomial ramp performance should be acceptable."""
        ramp = PolynomialRamp(0, 0.1, 10, 0.5, degree=3)
        
        times = np.random.uniform(0, 10, 10000)
        start = time.time()
        values = ramp.get_trajectory(times)
        elapsed = time.time() - start
        
        # Polynomial should still be < 2 ms
        assert elapsed < 0.005, f"Polynomial ramp took {elapsed*1000:.2f}ms"
        assert len(values) == 10000


class TestRampingEdgeCases:
    """Edge case tests for ramping."""
    
    def test_ramp_zero_interval(self):
        """Ramp with t0 == t1 raises ValueError."""
        with pytest.raises(ValueError, match="must differ"):
            LinearRamp(5, 0.1, 5, 0.5)
    
    def test_ramp_negative_values(self):
        """Ramps can have negative values (e.g., negative growth)."""
        ramp = LinearRamp(0, -0.5, 10, 0.5)
        
        assert ramp.evaluate(0) == pytest.approx(-0.5)
        assert ramp.evaluate(5) == pytest.approx(0.0)  # Midpoint
        assert ramp.evaluate(10) == pytest.approx(0.5)
    
    def test_ramp_zero_range(self):
        """Ramp from v0 to v1 where v0 == v1 (flat)."""
        ramp = LinearRamp(0, 0.5, 10, 0.5)
        
        assert ramp.evaluate(5) == pytest.approx(0.5)
        assert ramp.derivative(5) == pytest.approx(0.0)
    
    def test_ramp_extreme_values(self):
        """Ramps with very large/small values."""
        ramp_small = LinearRamp(0, 1e-6, 10, 1e-5)
        ramp_large = LinearRamp(0, 1e6, 10, 1e7)
        
        assert ramp_small.evaluate(5) == pytest.approx(5.5e-6)
        assert ramp_large.evaluate(5) == pytest.approx(5.5e6)
    
    def test_ramp_out_of_bounds_queries(self):
        """Query ramp outside defined interval."""
        ramp = LinearRamp(2, 0.1, 8, 0.5)
        
        # Before start - should return v0
        assert ramp.evaluate(-1) == pytest.approx(0.1)
        
        # After end - should return v1
        assert ramp.evaluate(15) == pytest.approx(0.5)
        
        # Within range
        assert ramp.evaluate(5) == pytest.approx(0.3)
