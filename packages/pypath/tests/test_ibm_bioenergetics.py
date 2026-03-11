"""
Tests for IBM bioenergetics module (Wisconsin model).

Validates the bioenergetics functions that drive individual fish growth
in the Individual-Based Model, including Q10 temperature scaling,
allometric length conversion, metabolism, assimilation, and the
integrated growth step.
"""

import numpy as np
import pytest

from pypath.ibm.bioenergetics import (
    BioenergParams,
    allometric_length,
    assimilation,
    growth_step,
    growth_step_batch,
    metabolism,
    q10_temperature_factor,
)


@pytest.fixture
def default_params() -> BioenergParams:
    """Return a BioenergParams instance with representative smelt-like values."""
    return BioenergParams(
        ra=0.0018,
        rb=-0.227,
        q10=2.0,
        t_ref=15.0,
        sda_fraction=0.172,
        unassimilated_fraction=0.20,
        a_length=0.01,
        b_length=0.333,
        energy_density=5.0,
        reproduction_fraction=0.3,
    )


class TestQ10:
    """Test q10_temperature_factor function."""

    def test_at_reference_temperature_returns_one(self):
        """Q10 factor should be 1.0 when temp equals reference temperature."""
        result = q10_temperature_factor(temp=15.0, t_ref=15.0, q10=2.0)
        assert result == pytest.approx(1.0)

    def test_plus_10c_returns_q10(self):
        """Q10 factor should equal q10 when temp is t_ref + 10."""
        result = q10_temperature_factor(temp=25.0, t_ref=15.0, q10=2.0)
        assert result == pytest.approx(2.0)

    def test_minus_10c_returns_inverse_q10(self):
        """Q10 factor should equal 1/q10 when temp is t_ref - 10."""
        result = q10_temperature_factor(temp=5.0, t_ref=15.0, q10=2.0)
        assert result == pytest.approx(0.5)

    def test_plus_5c_returns_sqrt_q10(self):
        """Q10 factor at +5C should be sqrt(q10)."""
        result = q10_temperature_factor(temp=20.0, t_ref=15.0, q10=2.0)
        assert result == pytest.approx(2.0**0.5)

    def test_different_q10_value(self):
        """Q10 factor works for different Q10 values."""
        result = q10_temperature_factor(temp=25.0, t_ref=15.0, q10=3.0)
        assert result == pytest.approx(3.0)


class TestAllometry:
    """Test allometric_length function."""

    def test_known_values(self):
        """Allometric length returns a * weight^b for known inputs."""
        # a=0.01, b=0.333, weight=1000 => 0.01 * 1000^0.333
        result = allometric_length(weight=1000.0, a=0.01, b=0.333)
        expected = 0.01 * (1000.0**0.333)
        assert result == pytest.approx(expected, rel=1e-6)

    def test_unit_weight(self):
        """Allometric length at weight=1.0 returns a * 1^b = a."""
        result = allometric_length(weight=1.0, a=0.01, b=0.333)
        assert result == pytest.approx(0.01, rel=1e-6)

    def test_zero_weight_returns_zero(self):
        """Allometric length returns 0.0 for zero weight."""
        result = allometric_length(weight=0.0, a=0.01, b=0.333)
        assert result == pytest.approx(0.0)

    def test_negative_weight_returns_zero(self):
        """Allometric length returns 0.0 for negative weight."""
        result = allometric_length(weight=-5.0, a=0.01, b=0.333)
        assert result == pytest.approx(0.0)


class TestMetabolism:
    """Test metabolism function."""

    def test_at_reference_temperature(self, default_params):
        """Metabolism at reference temp equals ra * weight^rb."""
        weight = 10.0
        temp = default_params.t_ref  # reference temp, so Q10 factor = 1
        result = metabolism(weight, temp, default_params)
        expected = default_params.ra * (weight**default_params.rb)
        assert result == pytest.approx(expected, rel=1e-6)

    def test_higher_temperature_increases_metabolism(self, default_params):
        """Metabolism should increase when temperature rises above t_ref."""
        weight = 10.0
        met_ref = metabolism(weight, default_params.t_ref, default_params)
        met_warm = metabolism(weight, default_params.t_ref + 10.0, default_params)
        assert met_warm > met_ref
        # With Q10=2, should be exactly double
        assert met_warm == pytest.approx(met_ref * 2.0, rel=1e-6)

    def test_heavier_fish(self, default_params):
        """Metabolism changes with weight according to rb exponent."""
        temp = default_params.t_ref
        met_small = metabolism(1.0, temp, default_params)
        met_large = metabolism(100.0, temp, default_params)
        # rb is negative (-0.227), so per-gram rate decreases with size
        # but total = ra * weight^rb, and since rb = -0.227,
        # weight=100 gives 100^(-0.227) which is less than 1^(-0.227)=1
        assert met_large < met_small


class TestAssimilation:
    """Test assimilation function."""

    def test_fraction_removed(self, default_params):
        """Assimilation returns consumption * (1 - unassimilated_fraction)."""
        consumption = 10.0
        result = assimilation(consumption, default_params)
        expected = consumption * (1.0 - default_params.unassimilated_fraction)
        assert result == pytest.approx(expected, rel=1e-6)

    def test_zero_consumption(self, default_params):
        """Assimilation of zero consumption returns zero."""
        result = assimilation(0.0, default_params)
        assert result == pytest.approx(0.0)

    def test_full_assimilation_when_fraction_zero(self):
        """If unassimilated_fraction is 0, full consumption is assimilated."""
        params = BioenergParams(
            ra=0.001,
            rb=-0.2,
            q10=2.0,
            t_ref=15.0,
            sda_fraction=0.1,
            unassimilated_fraction=0.0,
            a_length=0.01,
            b_length=0.333,
        )
        result = assimilation(5.0, params)
        assert result == pytest.approx(5.0)


class TestGrowthStep:
    """Test the integrated growth_step function."""

    def test_positive_growth_with_food(self, default_params):
        """Fish should gain weight when consumption exceeds costs."""
        weight = 10.0
        energy_reserve = 1.0
        consumption = 5.0  # generous consumption
        temperature = default_params.t_ref
        dt = 1.0 / 365.0  # one day in yearly units

        new_weight, new_energy = growth_step(
            weight=weight,
            energy_reserve=energy_reserve,
            consumption=consumption,
            temperature=temperature,
            is_mature=False,
            dt=dt,
            params=default_params,
        )
        assert new_weight > weight, "Fish should gain weight with sufficient food"
        assert new_energy >= 0.0

    def test_starvation_with_no_food(self, default_params):
        """Fish should lose weight when there is no consumption."""
        weight = 10.0
        energy_reserve = 0.0
        consumption = 0.0
        temperature = default_params.t_ref
        dt = 1.0 / 365.0

        new_weight, new_energy = growth_step(
            weight=weight,
            energy_reserve=energy_reserve,
            consumption=consumption,
            temperature=temperature,
            is_mature=False,
            dt=dt,
            params=default_params,
        )
        assert new_weight < weight, "Fish should lose weight during starvation"

    def test_minimum_weight_enforced(self, default_params):
        """Weight should never drop below 0.1 grams."""
        weight = 0.2
        energy_reserve = 0.0
        consumption = 0.0
        temperature = default_params.t_ref + 10.0  # warm = high metabolism
        dt = 1.0  # full year of starvation

        new_weight, new_energy = growth_step(
            weight=weight,
            energy_reserve=energy_reserve,
            consumption=consumption,
            temperature=temperature,
            is_mature=False,
            dt=dt,
            params=default_params,
        )
        assert new_weight >= 0.1, "Weight should not drop below minimum of 0.1"

    def test_mature_fish_reproduction_cost(self, default_params):
        """Mature fish with surplus should allocate reproduction_fraction to reproduction."""
        weight = 10.0
        energy_reserve = 1.0
        consumption = 5.0
        temperature = default_params.t_ref
        dt = 1.0 / 365.0

        # Immature fish
        new_weight_immature, _ = growth_step(
            weight=weight,
            energy_reserve=energy_reserve,
            consumption=consumption,
            temperature=temperature,
            is_mature=False,
            dt=dt,
            params=default_params,
        )

        # Mature fish (same conditions)
        new_weight_mature, _ = growth_step(
            weight=weight,
            energy_reserve=energy_reserve,
            consumption=consumption,
            temperature=temperature,
            is_mature=True,
            dt=dt,
            params=default_params,
        )

        # Mature fish should gain less weight due to reproduction cost
        assert new_weight_mature < new_weight_immature, (
            "Mature fish should gain less weight than immature due to reproduction cost"
        )

    def test_energy_reserve_stores_surplus(self, default_params):
        """When net energy is positive, surplus goes to energy reserve."""
        weight = 10.0
        energy_reserve = 0.5
        consumption = 5.0
        temperature = default_params.t_ref
        dt = 1.0 / 365.0

        _, new_energy = growth_step(
            weight=weight,
            energy_reserve=energy_reserve,
            consumption=consumption,
            temperature=temperature,
            is_mature=False,
            dt=dt,
            params=default_params,
        )
        assert new_energy >= energy_reserve, (
            "Energy reserve should increase or stay the same with surplus energy"
        )

    def test_energy_reserve_drains_under_deficit(self, default_params):
        """When net energy is negative, energy reserve should decrease."""
        weight = 10.0
        energy_reserve = 1.0
        consumption = 0.0  # no food
        temperature = default_params.t_ref
        dt = 1.0 / 365.0

        _, new_energy = growth_step(
            weight=weight,
            energy_reserve=energy_reserve,
            consumption=consumption,
            temperature=temperature,
            is_mature=False,
            dt=dt,
            params=default_params,
        )
        assert new_energy < energy_reserve, (
            "Energy reserve should decrease under deficit"
        )

    def test_energy_conservation(self, default_params):
        """The energy budget should be internally consistent."""
        weight = 10.0
        energy_reserve = 1.0
        consumption = 3.0
        temperature = default_params.t_ref
        dt = 1.0 / 365.0

        new_weight, new_energy = growth_step(
            weight=weight,
            energy_reserve=energy_reserve,
            consumption=consumption,
            temperature=temperature,
            is_mature=False,
            dt=dt,
            params=default_params,
        )

        # Compute expected budget components
        assim = consumption * (1.0 - default_params.unassimilated_fraction)
        sda = consumption * default_params.sda_fraction
        met = metabolism(weight, temperature, default_params) * dt * 365.0
        net_energy = assim - met - sda

        weight_change = net_energy / default_params.energy_density
        expected_weight = max(weight + weight_change, 0.1)

        assert new_weight == pytest.approx(expected_weight, rel=1e-6)

    def test_returns_tuple_of_two_floats(self, default_params):
        """growth_step returns a tuple of (new_weight, new_energy_reserve)."""
        result = growth_step(
            weight=10.0,
            energy_reserve=1.0,
            consumption=2.0,
            temperature=15.0,
            is_mature=False,
            dt=1.0 / 365.0,
            params=default_params,
        )
        assert isinstance(result, tuple)
        assert len(result) == 2
        assert isinstance(result[0], float)
        assert isinstance(result[1], float)


class TestGrowthStepBatch:
    """Verify growth_step_batch matches scalar growth_step element-wise."""

    def test_batch_matches_scalar(self, default_params):
        """Batch results match individual scalar calls."""
        weights = np.array([5.0, 10.0, 20.0])
        energy_reserves = np.array([0.5, 1.0, 0.2])
        consumptions = np.array([3.0, 5.0, 1.0])
        temperature = 15.0
        is_mature = np.array([False, True, False])
        dt = 1.0 / 12.0

        new_w, new_e = growth_step_batch(
            weights,
            energy_reserves,
            consumptions,
            temperature,
            is_mature,
            dt,
            default_params,
        )

        for i in range(3):
            sw, se = growth_step(
                weights[i],
                energy_reserves[i],
                consumptions[i],
                temperature,
                bool(is_mature[i]),
                dt,
                default_params,
            )
            assert new_w[i] == pytest.approx(sw, rel=1e-10)
            assert new_e[i] == pytest.approx(se, rel=1e-10)

    def test_batch_minimum_weight(self, default_params):
        """Batch enforces minimum weight of 0.1."""
        weights = np.array([0.1])
        energy_reserves = np.array([0.0])
        consumptions = np.array([0.0])
        is_mature = np.array([False])

        new_w, _ = growth_step_batch(
            weights,
            energy_reserves,
            consumptions,
            15.0,
            is_mature,
            1.0,
            default_params,
        )
        assert new_w[0] >= 0.1
