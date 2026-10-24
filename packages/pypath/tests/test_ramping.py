"""Tests for ramping functions."""

from __future__ import annotations

import pytest
import numpy as np
from pypath.core.ramping import (
    RampingFunction,
    LinearRamp,
    StepwiseRamp,
    PolynomialRamp,
    SplineRamp,
)


class TestRampingFunctionABC:
    """Test RampingFunction abstract base class."""

    def test_rampingfunction_is_abstract(self):
        """RampingFunction cannot be instantiated."""
        with pytest.raises(TypeError):
            RampingFunction()

    def test_rampingfunction_subclass_must_implement_evaluate(self):
        """Subclass without evaluate() fails."""
        class BadRamp(RampingFunction):
            def derivative(self, time):
                return 0
            def __repr__(self):
                return "BadRamp"
        with pytest.raises(TypeError):
            BadRamp()

    def test_rampingfunction_get_trajectory(self):
        """get_trajectory() batches evaluate calls."""
        ramp = LinearRamp(0, 1, 10, 2)
        times = np.array([0, 5, 10])
        traj = ramp.get_trajectory(times)
        expected = np.array([1.0, 1.5, 2.0])
        np.testing.assert_array_almost_equal(traj, expected)


class TestLinearRamp:
    """Test LinearRamp implementation."""

    @pytest.fixture
    def basic_ramp(self):
        """Simple linear ramp: 0.1 -> 0.5 over 10 years."""
        return LinearRamp(t0=0, v0=0.1, t1=10, v1=0.5)

    def test_linear_ramp_before_t0(self, basic_ramp):
        """Before ramp start, returns v0."""
        assert basic_ramp.evaluate(-5) == 0.1
        assert basic_ramp.evaluate(0) == 0.1

    def test_linear_ramp_after_t1(self, basic_ramp):
        """After ramp end, returns v1."""
        assert basic_ramp.evaluate(10) == 0.5
        assert basic_ramp.evaluate(15) == 0.5

    def test_linear_ramp_midpoint(self, basic_ramp):
        """Midpoint is correct linear interpolation."""
        assert pytest.approx(basic_ramp.evaluate(5)) == 0.3

    def test_linear_ramp_quarter_point(self, basic_ramp):
        """Quarter point is correct."""
        assert pytest.approx(basic_ramp.evaluate(2.5)) == 0.2

    def test_linear_ramp_derivative_constant(self, basic_ramp):
        """Derivative is constant during ramp."""
        slope = (0.5 - 0.1) / 10
        assert pytest.approx(basic_ramp.derivative(5)) == slope

    def test_linear_ramp_derivative_zero_before(self, basic_ramp):
        """Derivative is zero before ramp."""
        assert basic_ramp.derivative(-5) == 0.0
        assert basic_ramp.derivative(0) == 0.0  # At boundary

    def test_linear_ramp_derivative_zero_after(self, basic_ramp):
        """Derivative is zero after ramp."""
        assert basic_ramp.derivative(10) == 0.0  # At boundary
        assert basic_ramp.derivative(15) == 0.0

    def test_linear_ramp_negative_slope(self):
        """Ramp can go downward."""
        ramp = LinearRamp(0, 0.5, 10, 0.1)
        assert pytest.approx(ramp.evaluate(5)) == 0.3
        assert pytest.approx(ramp.derivative(5)) == -0.04

    def test_linear_ramp_zero_slope(self):
        """Ramp with same start and end value."""
        ramp = LinearRamp(0, 0.5, 10, 0.5)
        assert ramp.evaluate(5) == 0.5
        assert ramp.derivative(5) == 0.0

    def test_linear_ramp_raises_on_equal_times(self):
        """Raises ValueError if t0 == t1."""
        with pytest.raises(ValueError, match="must differ"):
            LinearRamp(5, 0.1, 5, 0.5)

    def test_linear_ramp_trajectory(self, basic_ramp):
        """get_trajectory() returns correct shape."""
        times = np.linspace(0, 10, 11)
        traj = basic_ramp.get_trajectory(times)
        assert traj.shape == times.shape
        assert traj[0] == pytest.approx(0.1)
        assert traj[5] == pytest.approx(0.3)
        assert traj[10] == pytest.approx(0.5)

    def test_linear_ramp_repr(self, basic_ramp):
        """String representation is informative."""
        s = repr(basic_ramp)
        assert "LinearRamp" in s
        assert "0.1" in s
        assert "0.5" in s


class TestStepwiseRamp:
    """Test StepwiseRamp implementation."""

    @pytest.fixture
    def step_ramp(self):
        """Step ramp with 3 breakpoints."""
        ramp = StepwiseRamp()
        ramp.add_breakpoint(0, 0.1)
        ramp.add_breakpoint(5, 0.3)
        ramp.add_breakpoint(10, 0.5)
        return ramp

    def test_stepwise_empty_raises(self):
        """Evaluate on empty stepwise ramp raises error."""
        ramp = StepwiseRamp()
        with pytest.raises(ValueError):
            ramp.evaluate(5)

    def test_stepwise_single_breakpoint(self):
        """Single breakpoint acts as step function."""
        ramp = StepwiseRamp([(5, 0.3)])
        assert ramp.evaluate(0) == 0.3
        assert ramp.evaluate(5) == 0.3
        assert ramp.evaluate(10) == 0.3

    def test_stepwise_before_first(self, step_ramp):
        """Before first breakpoint, return first value."""
        assert step_ramp.evaluate(-5) == 0.1

    def test_stepwise_between_breakpoints(self, step_ramp):
        """Between breakpoints, return lower breakpoint value."""
        assert step_ramp.evaluate(2) == 0.1
        assert step_ramp.evaluate(7) == 0.3

    def test_stepwise_at_breakpoint(self, step_ramp):
        """At breakpoint, use new value."""
        assert step_ramp.evaluate(5) == 0.3
        assert step_ramp.evaluate(10) == 0.5

    def test_stepwise_after_last(self, step_ramp):
        """After last breakpoint, return last value."""
        assert step_ramp.evaluate(15) == 0.5

    def test_stepwise_derivative_always_zero(self, step_ramp):
        """Derivative is always zero (discontinuous)."""
        assert step_ramp.derivative(0) == 0.0
        assert step_ramp.derivative(5) == 0.0
        assert step_ramp.derivative(10) == 0.0

    def test_stepwise_unsorted_breakpoints(self):
        """Breakpoints are auto-sorted."""
        ramp = StepwiseRamp()
        ramp.add_breakpoint(10, 0.5)
        ramp.add_breakpoint(0, 0.1)
        ramp.add_breakpoint(5, 0.3)
        assert ramp.evaluate(2) == 0.1
        assert ramp.evaluate(7) == 0.3


class TestPolynomialRamp:
    """Test PolynomialRamp implementation."""

    @pytest.fixture
    def poly_ramp(self):
        """Polynomial ramp: 0.1 -> 0.5 over 10 years."""
        return PolynomialRamp(t0=0, v0=0.1, t1=10, v1=0.5, degree=2)

    def test_polynomial_endpoints(self, poly_ramp):
        """Passes through both endpoints."""
        assert poly_ramp.evaluate(0) == pytest.approx(0.1)
        assert poly_ramp.evaluate(10) == pytest.approx(0.5)

    def test_polynomial_smoothstep(self, poly_ramp):
        """Uses smoothstep formula."""
        # Midpoint should be 0.3 by smoothstep
        mid = poly_ramp.evaluate(5)
        assert pytest.approx(mid) == 0.3

    def test_polynomial_before_t0(self, poly_ramp):
        """Before ramp, returns v0."""
        assert poly_ramp.evaluate(-5) == 0.1

    def test_polynomial_after_t1(self, poly_ramp):
        """After ramp, returns v1."""
        assert poly_ramp.evaluate(15) == 0.5

    def test_polynomial_derivative_zero_endpoints(self, poly_ramp):
        """Derivative is zero at endpoints (smooth connection)."""
        assert poly_ramp.derivative(0) == pytest.approx(0.0)
        assert poly_ramp.derivative(10) == pytest.approx(0.0)

    def test_polynomial_derivative_nonzero_interior(self, poly_ramp):
        """Derivative is nonzero in interior."""
        deriv_mid = poly_ramp.derivative(5)
        assert deriv_mid != 0.0

    def test_polynomial_invalid_degree(self):
        """Degree < 2 raises ValueError."""
        with pytest.raises(ValueError, match="degree"):
            PolynomialRamp(0, 0.1, 10, 0.5, degree=1)


class TestSplineRamp:
    """Test SplineRamp implementation."""

    @pytest.fixture
    def spline_ramp(self):
        """Spline ramp with 4 knots."""
        times = np.array([0, 5, 10, 15])
        values = np.array([0.1, 0.3, 0.5, 0.2])
        return SplineRamp(times, values)

    def test_spline_passes_through_knots(self, spline_ramp):
        """Exact at knot points."""
        for t, v in [(0, 0.1), (5, 0.3), (10, 0.5), (15, 0.2)]:
            assert spline_ramp.evaluate(t) == pytest.approx(v)

    def test_spline_interpolates(self, spline_ramp):
        """Interpolates between knots."""
        # At t=7.5 (halfway between t=5 and t=10)
        val = spline_ramp.evaluate(7.5)
        # Should be roughly between 0.3 and 0.5
        assert 0.3 < val < 0.5

    def test_spline_few_knots(self):
        """Minimum 2 knots required."""
        times = np.array([0, 10])
        values = np.array([0.1, 0.5])
        ramp = SplineRamp(times, values)
        assert ramp.evaluate(5) == pytest.approx(0.3)

    def test_spline_one_knot_raises(self):
        """One knot raises ValueError."""
        with pytest.raises(ValueError, match=">=.*2"):
            SplineRamp(np.array([5]), np.array([0.3]))

    def test_spline_length_mismatch(self):
        """Mismatched times and values raises."""
        with pytest.raises(ValueError, match="same length"):
            SplineRamp(np.array([0, 5, 10]), np.array([0.1, 0.5]))

    def test_spline_unsorted_times(self):
        """Unsorted times raise ValueError."""
        with pytest.raises(ValueError, match="sorted"):
            SplineRamp(np.array([0, 10, 5]), np.array([0.1, 0.5, 0.3]))


# Integration tests
class TestRampingIntegration:
    """Test ramping functions working together."""

    def test_multiple_ramps_different_params(self):
        """Multiple ramps on different parameters."""
        ramp_M2 = LinearRamp(0, 0.1, 20, 0.3)
        ramp_K = PolynomialRamp(0, 0.5, 20, 1.0)
        
        for t in [0, 5, 10, 20]:
            M2 = ramp_M2.evaluate(t)
            K = ramp_K.evaluate(t)
            assert M2 >= 0.1 and M2 <= 0.3
            assert K >= 0.5 and K <= 1.0

    def test_ramp_trajectory_consistency(self):
        """Single vs batch evaluation consistent."""
        ramp = LinearRamp(0, 0.1, 10, 0.5)
        times = np.linspace(0, 10, 21)
        
        batch = ramp.get_trajectory(times)
        individual = np.array([ramp.evaluate(t) for t in times])
        
        np.testing.assert_array_almost_equal(batch, individual)