"""
Tests for IBM size-structured predation module.

Validates the predation functions that distribute Ecosim group-level
predation mortality across IBM super-individuals based on body size,
including log-normal size selectivity, mortality distribution, and
the integrated predation mortality application.
"""

import math

import pytest

from pypath.ibm.base import SuperIndividual
from pypath.ibm.predation import (
    PredationParams,
    apply_predation_mortality,
    distribute_mortality,
    size_selectivity,
)


@pytest.fixture
def default_predation_params() -> PredationParams:
    """Return PredationParams with representative values."""
    return PredationParams(
        optimal_prey_length=10.0,
        selectivity_sd=0.5,
    )


def _make_individual(
    id: int,
    n_represented: float,
    length: float,
    weight: float = 10.0,
) -> SuperIndividual:
    """Helper to create a SuperIndividual with minimal required fields."""
    return SuperIndividual(
        id=id,
        n_represented=n_represented,
        weight=weight,
        length=length,
        age=1.0,
        energy_reserve=0.8,
        patch_idx=0,
        is_mature=False,
        sex=0,
    )


class TestSizeSelectivity:
    """Test the size_selectivity function."""

    def test_peak_at_optimal_length(self, default_predation_params):
        """Selectivity should be 1.0 at the optimal prey length."""
        result = size_selectivity(
            length=default_predation_params.optimal_prey_length,
            params=default_predation_params,
        )
        assert result == pytest.approx(1.0)

    def test_symmetric_in_log_space(self, default_predation_params):
        """Selectivity should be symmetric around optimal length in log-space.

        That is, a fish twice the optimal length and one half the optimal
        length should have the same selectivity.
        """
        optimal = default_predation_params.optimal_prey_length
        sel_double = size_selectivity(
            length=optimal * 2.0, params=default_predation_params
        )
        sel_half = size_selectivity(
            length=optimal / 2.0, params=default_predation_params
        )
        assert sel_double == pytest.approx(sel_half, rel=1e-6)

    def test_decreases_away_from_optimal(self, default_predation_params):
        """Selectivity should decrease for lengths far from optimal."""
        optimal = default_predation_params.optimal_prey_length
        sel_optimal = size_selectivity(length=optimal, params=default_predation_params)
        sel_far = size_selectivity(
            length=optimal * 5.0, params=default_predation_params
        )
        assert sel_far < sel_optimal

    def test_zero_length_returns_zero(self, default_predation_params):
        """Selectivity should be 0.0 when length is zero."""
        result = size_selectivity(length=0.0, params=default_predation_params)
        assert result == pytest.approx(0.0)

    def test_negative_length_returns_zero(self, default_predation_params):
        """Selectivity should be 0.0 when length is negative."""
        result = size_selectivity(length=-5.0, params=default_predation_params)
        assert result == pytest.approx(0.0)

    def test_selectivity_always_between_zero_and_one(self, default_predation_params):
        """Selectivity should always be in range [0.0, 1.0]."""
        for length in [0.01, 0.1, 1.0, 5.0, 10.0, 20.0, 50.0, 100.0]:
            result = size_selectivity(length=length, params=default_predation_params)
            assert 0.0 <= result <= 1.0, f"Selectivity out of range for length={length}"

    def test_known_value(self):
        """Selectivity matches a hand-computed log-normal value."""
        params = PredationParams(optimal_prey_length=10.0, selectivity_sd=1.0)
        length = math.exp(math.log(10.0) + 1.0)  # one SD above optimal in log-space
        expected = math.exp(-0.5 * 1.0**2)  # exp(-0.5)
        result = size_selectivity(length=length, params=params)
        assert result == pytest.approx(expected, rel=1e-6)


class TestDistributeMortality:
    """Test the distribute_mortality function."""

    def test_total_deaths_preserved(self, default_predation_params):
        """Total deaths across individuals should match expected total deaths."""
        individuals = [
            _make_individual(id=0, n_represented=500.0, length=10.0),
            _make_individual(id=1, n_represented=500.0, length=10.0),
        ]
        mortality_rate = 0.5  # annual rate
        dt = 1.0 / 12.0  # one month

        deaths = distribute_mortality(
            individuals=individuals,
            total_mortality_rate=mortality_rate,
            dt=dt,
            params=default_predation_params,
        )
        total_n = sum(ind.n_represented for ind in individuals)
        expected_total_deaths = total_n * mortality_rate * dt
        assert sum(deaths) == pytest.approx(expected_total_deaths, rel=1e-6)

    def test_smaller_fish_near_optimal_die_more(self, default_predation_params):
        """Individuals closer to optimal prey length should have more deaths.

        With optimal_prey_length=10.0, a fish of length 10 cm should lose
        more individuals than one at 40 cm, given equal n_represented.
        """
        individuals = [
            _make_individual(id=0, n_represented=1000.0, length=10.0),  # at optimal
            _make_individual(
                id=1, n_represented=1000.0, length=40.0
            ),  # far from optimal
        ]
        mortality_rate = 0.3
        dt = 1.0 / 12.0

        deaths = distribute_mortality(
            individuals=individuals,
            total_mortality_rate=mortality_rate,
            dt=dt,
            params=default_predation_params,
        )
        assert deaths[0] > deaths[1], (
            "Fish at optimal prey length should suffer more deaths"
        )

    def test_cannot_kill_more_than_exist(self, default_predation_params):
        """Deaths per individual should not exceed n_represented."""
        individuals = [
            _make_individual(id=0, n_represented=10.0, length=10.0),
        ]
        # Very high mortality rate to force capping
        mortality_rate = 100.0
        dt = 1.0

        deaths = distribute_mortality(
            individuals=individuals,
            total_mortality_rate=mortality_rate,
            dt=dt,
            params=default_predation_params,
        )
        assert deaths[0] <= individuals[0].n_represented

    def test_empty_individuals_returns_empty(self, default_predation_params):
        """Empty individuals list returns empty deaths list."""
        deaths = distribute_mortality(
            individuals=[],
            total_mortality_rate=0.5,
            dt=1.0 / 12.0,
            params=default_predation_params,
        )
        assert deaths == []

    def test_zero_mortality_rate_gives_zero_deaths(self, default_predation_params):
        """Zero mortality rate should produce zero deaths for all individuals."""
        individuals = [
            _make_individual(id=0, n_represented=1000.0, length=10.0),
            _make_individual(id=1, n_represented=500.0, length=5.0),
        ]
        deaths = distribute_mortality(
            individuals=individuals,
            total_mortality_rate=0.0,
            dt=1.0 / 12.0,
            params=default_predation_params,
        )
        assert all(d == pytest.approx(0.0) for d in deaths)

    def test_returns_list_of_correct_length(self, default_predation_params):
        """distribute_mortality returns a list with one entry per individual."""
        individuals = [
            _make_individual(id=i, n_represented=100.0, length=float(5 + i))
            for i in range(5)
        ]
        deaths = distribute_mortality(
            individuals=individuals,
            total_mortality_rate=0.2,
            dt=1.0 / 12.0,
            params=default_predation_params,
        )
        assert isinstance(deaths, list)
        assert len(deaths) == len(individuals)

    def test_proportional_to_selectivity_weighted_abundance(
        self, default_predation_params
    ):
        """Deaths should be proportional to selectivity * n_represented."""
        individuals = [
            _make_individual(id=0, n_represented=1000.0, length=10.0),
            _make_individual(id=1, n_represented=1000.0, length=20.0),
        ]
        mortality_rate = 0.1
        dt = 1.0 / 12.0

        deaths = distribute_mortality(
            individuals=individuals,
            total_mortality_rate=mortality_rate,
            dt=dt,
            params=default_predation_params,
        )

        sel_0 = size_selectivity(10.0, default_predation_params)
        sel_1 = size_selectivity(20.0, default_predation_params)

        # Deaths should be in ratio sel_0 * n_0 : sel_1 * n_1
        expected_ratio = (sel_0 * 1000.0) / (sel_1 * 1000.0)
        actual_ratio = deaths[0] / deaths[1] if deaths[1] > 0 else float("inf")
        assert actual_ratio == pytest.approx(expected_ratio, rel=1e-6)


class TestApplyPredationMortality:
    """Test the apply_predation_mortality function."""

    def test_reduces_n_represented(self, default_predation_params):
        """Survivors should have reduced n_represented."""
        individuals = [
            _make_individual(id=0, n_represented=1000.0, length=10.0),
        ]
        survivors = apply_predation_mortality(
            individuals=individuals,
            total_mortality_rate=0.5,
            dt=1.0 / 12.0,
            params=default_predation_params,
        )
        assert len(survivors) == 1
        assert survivors[0].n_represented < 1000.0

    def test_does_not_modify_weight(self, default_predation_params):
        """Weight of survivors should be unchanged."""
        individuals = [
            _make_individual(id=0, n_represented=1000.0, length=10.0, weight=50.0),
        ]
        survivors = apply_predation_mortality(
            individuals=individuals,
            total_mortality_rate=0.5,
            dt=1.0 / 12.0,
            params=default_predation_params,
        )
        assert survivors[0].weight == pytest.approx(50.0)

    def test_does_not_modify_originals(self, default_predation_params):
        """Original individuals should not be modified (immutability)."""
        individuals = [
            _make_individual(id=0, n_represented=1000.0, length=10.0),
        ]
        original_n = individuals[0].n_represented
        _ = apply_predation_mortality(
            individuals=individuals,
            total_mortality_rate=0.5,
            dt=1.0 / 12.0,
            params=default_predation_params,
        )
        assert individuals[0].n_represented == pytest.approx(original_n)

    def test_removes_individuals_with_zero_n_represented(
        self, default_predation_params
    ):
        """Individuals with n_represented <= 0 after mortality should be removed."""
        individuals = [
            _make_individual(
                id=0, n_represented=5.0, length=10.0
            ),  # near optimal, tiny
        ]
        # Extreme mortality to wipe out
        survivors = apply_predation_mortality(
            individuals=individuals,
            total_mortality_rate=100.0,
            dt=1.0,
            params=default_predation_params,
        )
        assert len(survivors) == 0

    def test_returns_list_of_super_individuals(self, default_predation_params):
        """Return type should be a list of SuperIndividual objects."""
        individuals = [
            _make_individual(id=0, n_represented=1000.0, length=10.0),
            _make_individual(id=1, n_represented=500.0, length=15.0),
        ]
        survivors = apply_predation_mortality(
            individuals=individuals,
            total_mortality_rate=0.1,
            dt=1.0 / 12.0,
            params=default_predation_params,
        )
        assert isinstance(survivors, list)
        for s in survivors:
            assert isinstance(s, SuperIndividual)

    def test_empty_list_returns_empty(self, default_predation_params):
        """Empty input returns empty output."""
        survivors = apply_predation_mortality(
            individuals=[],
            total_mortality_rate=0.5,
            dt=1.0 / 12.0,
            params=default_predation_params,
        )
        assert survivors == []

    def test_preserves_other_attributes(self, default_predation_params):
        """Attributes other than n_represented should be preserved."""
        individuals = [
            _make_individual(id=42, n_represented=1000.0, length=10.0, weight=50.0),
        ]
        individuals[0].age = 3.5
        individuals[0].energy_reserve = 0.9
        individuals[0].patch_idx = 7
        individuals[0].is_mature = True
        individuals[0].sex = 1

        survivors = apply_predation_mortality(
            individuals=individuals,
            total_mortality_rate=0.1,
            dt=1.0 / 12.0,
            params=default_predation_params,
        )
        s = survivors[0]
        assert s.id == 42
        assert s.weight == pytest.approx(50.0)
        assert s.length == pytest.approx(10.0)
        assert s.age == pytest.approx(3.5)
        assert s.energy_reserve == pytest.approx(0.9)
        assert s.patch_idx == 7
        assert s.is_mature is True
        assert s.sex == 1
