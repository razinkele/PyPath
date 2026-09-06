"""Tests for AB2 integrator matching Rpath."""

from unittest.mock import patch

import numpy as np

from pypath.core.ecosim_deriv import integrate_ab


class TestAB2Integration:
    def test_ab2_formula_with_mock(self):
        """AB2: y_{n+1} = y_n + dt/2 * (3*f_n - f_{n-1})."""
        state = np.array([0.0, 100.0, 20.0])
        dt = 1.0 / 12.0
        deriv_current = np.array([0.0, 1.0, -0.5])
        deriv_prev = np.array([0.0, 0.8, -0.3])

        expected = state + (dt / 2.0) * (3.0 * deriv_current - deriv_prev)

        params = {
            "NUM_GROUPS": 2,
            "NUM_LIVING": 2,
            "NoIntegrate": np.zeros(3),
            "Bbase": state.copy(),
        }

        with patch("pypath.core.ecosim_deriv.deriv_vector", return_value=deriv_current):
            new_state, new_deriv = integrate_ab(state, [deriv_prev], params, {}, {}, dt)

        np.testing.assert_allclose(new_state[1:], expected[1:], rtol=1e-6)

    def test_euler_fallback_no_history(self):
        """With no derivative history, AB falls back to Euler."""
        state = np.array([0.0, 100.0, 20.0])
        dt = 1.0 / 12.0
        deriv_current = np.array([0.0, 1.0, -0.5])

        expected = state + dt * deriv_current

        params = {
            "NUM_GROUPS": 2,
            "NUM_LIVING": 2,
            "NoIntegrate": np.zeros(3),
            "Bbase": state.copy(),
        }

        with patch("pypath.core.ecosim_deriv.deriv_vector", return_value=deriv_current):
            new_state, _ = integrate_ab(state, [], params, {}, {}, dt)

        np.testing.assert_allclose(new_state[1:], expected[1:], rtol=1e-6)

    def test_ab2_no_ab4_coefficients(self):
        """AB4 coefficients should not be present in integrate_ab."""
        import inspect

        src = inspect.getsource(integrate_ab)
        assert "55" not in src, "AB4 coefficient 55 should not be present"
        assert "59" not in src, "AB4 coefficient 59 should not be present"

    def test_biomass_bounds_applied(self):
        """Rpath-style bounds: Bbase*EPSILON <= B <= Bbase*BIGNUM."""
        state = np.array([0.0, 100.0, 20.0])
        dt = 1.0 / 12.0
        # Derivative that would drive biomass negative
        deriv_neg = np.array([0.0, -1e8, -1e8])

        params = {
            "NUM_GROUPS": 2,
            "NUM_LIVING": 2,
            "NoIntegrate": np.zeros(3),
            "Bbase": np.array([0.0, 100.0, 20.0]),
        }

        with patch("pypath.core.ecosim_deriv.deriv_vector", return_value=deriv_neg):
            new_state, _ = integrate_ab(state, [], params, {}, {}, dt)

        # Should be bounded above zero (Bbase * EPSILON)
        assert new_state[1] > 0, "Biomass should be bounded above zero"
        assert new_state[2] > 0, "Biomass should be bounded above zero"
