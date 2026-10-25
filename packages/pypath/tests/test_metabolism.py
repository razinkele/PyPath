"""Tests for core metabolic scaling models."""

from __future__ import annotations

import numpy as np
import pytest

from pypath.core.metabolism import (
    MetabolicModel,
    TemperatureDependence,
    apply_size_dependent_rates,
    compute_temperature_adjusted_metabolism,
)


class TestTemperatureDependence:
    def test_default_reference_factor_is_one(self):
        model = TemperatureDependence(q10_coefficient=2.0, reference_temperature=10.0)
        assert model.compute_q10_factor(10.0) == pytest.approx(1.0)

    def test_warming_increases_factor(self):
        model = TemperatureDependence(q10_coefficient=2.0, reference_temperature=10.0)
        assert model.compute_q10_factor(20.0) == pytest.approx(2.0)

    def test_cooling_decreases_factor(self):
        model = TemperatureDependence(q10_coefficient=2.0, reference_temperature=10.0)
        assert model.compute_q10_factor(0.0) == pytest.approx(0.5)

    def test_array_input(self):
        model = TemperatureDependence(q10_coefficient=2.0, reference_temperature=10.0)
        arr = model.compute_q10_factor(np.array([0.0, 10.0, 20.0]))
        np.testing.assert_allclose(arr, np.array([0.5, 1.0, 2.0]))

    def test_clamps_extremes(self):
        model = TemperatureDependence(
            q10_coefficient=3.0,
            reference_temperature=10.0,
            min_factor=0.1,
            max_factor=10.0,
        )
        assert model.compute_q10_factor(-100.0) == pytest.approx(0.1)
        assert model.compute_q10_factor(100.0) == pytest.approx(10.0)

    def test_invalid_q10_raises(self):
        with pytest.raises(ValueError):
            TemperatureDependence(q10_coefficient=0.0)


class TestMetabolicModel:
    @pytest.fixture
    def model(self):
        temp = TemperatureDependence(q10_coefficient=2.0, reference_temperature=10.0)
        return MetabolicModel(
            base_metabolism=0.2,
            mass_exponent=0.75,
            reference_mass=1.0,
            temperature_dependence=temp,
            activity_budget=1.0,
        )

    def test_invalid_base_raises(self):
        with pytest.raises(ValueError):
            MetabolicModel(base_metabolism=0.0)

    def test_invalid_reference_mass_raises(self):
        with pytest.raises(ValueError):
            MetabolicModel(base_metabolism=0.1, reference_mass=0.0)

    def test_invalid_activity_budget_raises(self):
        with pytest.raises(ValueError):
            MetabolicModel(base_metabolism=0.1, activity_budget=0.0)

    def test_default_temperature_model_injected(self):
        model = MetabolicModel(base_metabolism=0.1)
        assert isinstance(model.temperature_dependence, TemperatureDependence)

    def test_mass_factor_reference_mass(self, model):
        assert model.compute_mass_factor(1.0) == pytest.approx(1.0)

    def test_mass_factor_scaling(self, model):
        assert model.compute_mass_factor(4.0) == pytest.approx(4.0**0.75)

    def test_metabolic_rate_scalar(self, model):
        rate = model.compute_metabolic_rate(body_mass=2.0, temperature=20.0)
        expected = 0.2 * (2.0**0.75) * 2.0
        assert rate == pytest.approx(expected)

    def test_activity_multiplier_applies(self, model):
        base = model.compute_metabolic_rate(body_mass=2.0, temperature=10.0)
        active = model.compute_metabolic_rate(
            body_mass=2.0,
            temperature=10.0,
            activity_multiplier=1.5,
        )
        assert active == pytest.approx(base * 1.5)

    def test_activity_multiplier_invalid(self, model):
        with pytest.raises(ValueError):
            model.compute_metabolic_rate(2.0, 10.0, activity_multiplier=0.0)

    def test_batch_computation(self, model):
        masses = np.array([1.0, 2.0, 4.0])
        temps = np.array([10.0, 10.0, 20.0])
        out = model.compute_metabolic_rate_batch(masses, temps)
        assert out.shape == masses.shape
        expected = np.array([
            0.2 * (1.0**0.75) * 1.0,
            0.2 * (2.0**0.75) * 1.0,
            0.2 * (4.0**0.75) * 2.0,
        ])
        np.testing.assert_allclose(out, expected)

    def test_batch_length_mismatch_raises(self, model):
        with pytest.raises(ValueError, match="must equal"):
            model.compute_metabolic_rate_batch(np.array([1.0, 2.0]), np.array([10.0]))

    def test_batch_activity_length_mismatch(self, model):
        with pytest.raises(ValueError, match="must equal"):
            model.compute_metabolic_rate_batch(
                np.array([1.0, 2.0]),
                np.array([10.0, 10.0]),
                activity_multiplier=np.array([1.0]),
            )


class TestMetabolicHelpers:
    def test_apply_size_dependent_rates(self):
        out = apply_size_dependent_rates(
            np.array([1.0, 1.0, 1.0]),
            np.array([1.0, 4.0, 9.0]),
            mass_exponent=0.5,
            reference_mass=1.0,
        )
        np.testing.assert_allclose(out, np.array([1.0, 2.0, 3.0]))

    def test_apply_size_dependent_length_mismatch_raises(self):
        with pytest.raises(ValueError, match="must equal"):
            apply_size_dependent_rates(
                np.array([1.0, 1.0]),
                np.array([1.0]),
                mass_exponent=0.75,
            )

    def test_compute_temperature_adjusted_metabolism_scalar(self):
        temp_model = TemperatureDependence(q10_coefficient=2.0, reference_temperature=10.0)
        out = compute_temperature_adjusted_metabolism(0.5, 20.0, temp_model)
        assert out == pytest.approx(1.0)

    def test_compute_temperature_adjusted_metabolism_array(self):
        temp_model = TemperatureDependence(q10_coefficient=2.0, reference_temperature=10.0)
        out = compute_temperature_adjusted_metabolism(
            np.array([0.5, 0.5, 0.5]),
            np.array([0.0, 10.0, 20.0]),
            temp_model,
        )
        np.testing.assert_allclose(out, np.array([0.25, 0.5, 1.0]))
