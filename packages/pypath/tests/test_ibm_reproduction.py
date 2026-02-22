"""
Tests for IBM stochastic reproduction module.

Validates the reproduction functions implementing spawning, fecundity
calculation, Cushing match/mismatch larval survival, and recruit creation
for Baltic smelt super-individuals.
"""

import math

import pytest

from pypath.ibm.base import SuperIndividual
from pypath.ibm.reproduction import (
    ReproductionParams,
    calculate_fecundity,
    create_recruits,
    larval_survival_probability,
    spawn,
)


@pytest.fixture
def default_repro_params() -> ReproductionParams:
    """Return ReproductionParams with representative values."""
    return ReproductionParams(
        fecundity_coefficient=500.0,
        fecundity_exponent=3.0,
        larval_base_survival=0.01,
        zooplankton_match_window=10.0,
        maturity_energy_threshold=50.0,
        spawning_temp_threshold=8.0,
        larval_duration_days=30,
        recruit_weight=0.5,
        recruit_length=2.0,
    )


def _make_individual(
    id: int = 0,
    n_represented: float = 1000.0,
    weight: float = 50.0,
    length: float = 15.0,
    age: float = 3.0,
    energy_reserve: float = 100.0,
    patch_idx: int = 0,
    is_mature: bool = True,
    sex: int = 0,
) -> SuperIndividual:
    """Helper to create a SuperIndividual for reproduction tests."""
    return SuperIndividual(
        id=id,
        n_represented=n_represented,
        weight=weight,
        length=length,
        age=age,
        energy_reserve=energy_reserve,
        patch_idx=patch_idx,
        is_mature=is_mature,
        sex=sex,
    )


class TestFecundity:
    """Test the calculate_fecundity function."""

    def test_proportional_to_weight(self, default_repro_params):
        """Heavier fish should produce more eggs."""
        eggs_small = calculate_fecundity(weight=10.0, params=default_repro_params)
        eggs_large = calculate_fecundity(weight=50.0, params=default_repro_params)
        assert eggs_large > eggs_small

    def test_known_value(self, default_repro_params):
        """Fecundity should match coeff * weight^exp."""
        weight = 20.0
        expected = 500.0 * 20.0**3.0
        result = calculate_fecundity(weight=weight, params=default_repro_params)
        assert result == pytest.approx(expected, rel=1e-6)

    def test_zero_weight_returns_zero(self, default_repro_params):
        """Fecundity should be 0.0 when weight is zero."""
        result = calculate_fecundity(weight=0.0, params=default_repro_params)
        assert result == pytest.approx(0.0)

    def test_negative_weight_returns_zero(self, default_repro_params):
        """Fecundity should be 0.0 when weight is negative."""
        result = calculate_fecundity(weight=-5.0, params=default_repro_params)
        assert result == pytest.approx(0.0)

    def test_unit_weight(self, default_repro_params):
        """Fecundity at weight=1 should equal the coefficient."""
        result = calculate_fecundity(weight=1.0, params=default_repro_params)
        assert result == pytest.approx(
            default_repro_params.fecundity_coefficient, rel=1e-6
        )


class TestLarvalSurvival:
    """Test the larval_survival_probability function."""

    def test_perfect_match_returns_base_survival(self, default_repro_params):
        """When spawn_day == zoo_peak_day, survival equals base_survival."""
        result = larval_survival_probability(
            spawn_day=100.0, zoo_peak_day=100.0, params=default_repro_params
        )
        assert result == pytest.approx(
            default_repro_params.larval_base_survival, rel=1e-6
        )

    def test_mismatch_reduces_survival(self, default_repro_params):
        """Mismatch between spawn_day and zoo_peak_day should reduce survival."""
        perfect = larval_survival_probability(
            spawn_day=100.0, zoo_peak_day=100.0, params=default_repro_params
        )
        mismatched = larval_survival_probability(
            spawn_day=100.0, zoo_peak_day=120.0, params=default_repro_params
        )
        assert mismatched < perfect

    def test_symmetric_mismatch(self, default_repro_params):
        """Early and late mismatch of the same magnitude should give equal survival."""
        early = larval_survival_probability(
            spawn_day=90.0, zoo_peak_day=100.0, params=default_repro_params
        )
        late = larval_survival_probability(
            spawn_day=110.0, zoo_peak_day=100.0, params=default_repro_params
        )
        assert early == pytest.approx(late, rel=1e-6)

    def test_known_gaussian_value(self, default_repro_params):
        """Survival should match the Gaussian formula for a known mismatch."""
        expected = 0.01 * math.exp(-0.5 * (10.0 / 10.0) ** 2)
        result = larval_survival_probability(
            spawn_day=100.0, zoo_peak_day=110.0, params=default_repro_params
        )
        assert result == pytest.approx(expected, rel=1e-6)

    def test_large_mismatch_approaches_zero(self, default_repro_params):
        """Very large mismatch should yield near-zero survival."""
        result = larval_survival_probability(
            spawn_day=100.0, zoo_peak_day=200.0, params=default_repro_params
        )
        assert result < 1e-10

    def test_survival_always_positive(self, default_repro_params):
        """Survival should always be > 0 (Gaussian never reaches zero)."""
        result = larval_survival_probability(
            spawn_day=100.0, zoo_peak_day=150.0, params=default_repro_params
        )
        assert result > 0.0


class TestSpawning:
    """Test the spawn function."""

    def test_mature_female_spawns(self, default_repro_params):
        """A mature female above energy and temperature thresholds should produce eggs."""
        ind = _make_individual(is_mature=True, sex=0, energy_reserve=100.0)
        eggs = spawn(individual=ind, temperature=10.0, params=default_repro_params)
        assert eggs > 0.0

    def test_spawn_returns_correct_total(self, default_repro_params):
        """Total eggs should equal n_represented * fecundity_per_female."""
        ind = _make_individual(
            n_represented=1000.0,
            weight=20.0,
            is_mature=True,
            sex=0,
            energy_reserve=100.0,
        )
        expected_fecundity = 500.0 * 20.0**3.0
        expected_total = 1000.0 * expected_fecundity
        eggs = spawn(individual=ind, temperature=10.0, params=default_repro_params)
        assert eggs == pytest.approx(expected_total, rel=1e-6)

    def test_immature_female_returns_zero(self, default_repro_params):
        """An immature female should not spawn."""
        ind = _make_individual(is_mature=False, sex=0, energy_reserve=100.0)
        eggs = spawn(individual=ind, temperature=10.0, params=default_repro_params)
        assert eggs == pytest.approx(0.0)

    def test_male_returns_zero(self, default_repro_params):
        """A male should not spawn (sex=1)."""
        ind = _make_individual(is_mature=True, sex=1, energy_reserve=100.0)
        eggs = spawn(individual=ind, temperature=10.0, params=default_repro_params)
        assert eggs == pytest.approx(0.0)

    def test_cold_water_returns_zero(self, default_repro_params):
        """Temperature below threshold should prevent spawning."""
        ind = _make_individual(is_mature=True, sex=0, energy_reserve=100.0)
        eggs = spawn(individual=ind, temperature=5.0, params=default_repro_params)
        assert eggs == pytest.approx(0.0)

    def test_low_energy_returns_zero(self, default_repro_params):
        """Energy below maturity_energy_threshold should prevent spawning."""
        ind = _make_individual(is_mature=True, sex=0, energy_reserve=10.0)
        eggs = spawn(individual=ind, temperature=10.0, params=default_repro_params)
        assert eggs == pytest.approx(0.0)

    def test_exact_threshold_temperature_spawns(self, default_repro_params):
        """Temperature exactly at threshold should allow spawning."""
        ind = _make_individual(is_mature=True, sex=0, energy_reserve=100.0)
        eggs = spawn(individual=ind, temperature=8.0, params=default_repro_params)
        assert eggs > 0.0

    def test_exact_threshold_energy_spawns(self, default_repro_params):
        """Energy exactly at threshold should allow spawning."""
        ind = _make_individual(is_mature=True, sex=0, energy_reserve=50.0)
        eggs = spawn(individual=ind, temperature=10.0, params=default_repro_params)
        assert eggs > 0.0


class TestRecruits:
    """Test the create_recruits function."""

    def test_creates_correct_number_of_recruits(self, default_repro_params):
        """Should create n_super_individuals recruits when survivors > 0."""
        recruits = create_recruits(
            total_eggs=1_000_000.0,
            spawn_day=100.0,
            zoo_peak_day=100.0,
            patch_idx=3,
            next_id=100,
            params=default_repro_params,
            n_super_individuals=5,
        )
        assert len(recruits) == 5

    def test_recruit_attributes(self, default_repro_params):
        """Recruits should have correct weight, length, age, maturity."""
        recruits = create_recruits(
            total_eggs=1_000_000.0,
            spawn_day=100.0,
            zoo_peak_day=100.0,
            patch_idx=3,
            next_id=100,
            params=default_repro_params,
            n_super_individuals=1,
        )
        assert len(recruits) == 1
        r = recruits[0]
        assert r.weight == pytest.approx(default_repro_params.recruit_weight)
        assert r.length == pytest.approx(default_repro_params.recruit_length)
        assert r.age == pytest.approx(0.0)
        assert r.is_mature is False
        assert r.patch_idx == 3
        assert r.energy_reserve == pytest.approx(
            default_repro_params.recruit_weight * 5.0
        )

    def test_recruit_ids_are_sequential(self, default_repro_params):
        """Recruit IDs should start at next_id and increment."""
        recruits = create_recruits(
            total_eggs=1_000_000.0,
            spawn_day=100.0,
            zoo_peak_day=100.0,
            patch_idx=0,
            next_id=50,
            params=default_repro_params,
            n_super_individuals=3,
        )
        ids = [r.id for r in recruits]
        assert ids == [50, 51, 52]

    def test_n_represented_distribution(self, default_repro_params):
        """Total n_represented across recruits should equal total survivors."""
        total_eggs = 1_000_000.0
        survival = larval_survival_probability(
            spawn_day=100.0, zoo_peak_day=100.0, params=default_repro_params
        )
        expected_survivors = total_eggs * survival

        recruits = create_recruits(
            total_eggs=total_eggs,
            spawn_day=100.0,
            zoo_peak_day=100.0,
            patch_idx=0,
            next_id=0,
            params=default_repro_params,
            n_super_individuals=4,
        )
        total_n = sum(r.n_represented for r in recruits)
        assert total_n == pytest.approx(expected_survivors, rel=1e-6)

    def test_empty_if_too_few_survivors(self, default_repro_params):
        """Should return empty list if survivors < 1."""
        # Very few eggs and large mismatch -> near-zero survival
        recruits = create_recruits(
            total_eggs=1.0,
            spawn_day=100.0,
            zoo_peak_day=300.0,
            patch_idx=0,
            next_id=0,
            params=default_repro_params,
        )
        assert recruits == []

    def test_recruit_sex_is_valid(self, default_repro_params):
        """Recruit sex should be 0 or 1."""
        recruits = create_recruits(
            total_eggs=1_000_000.0,
            spawn_day=100.0,
            zoo_peak_day=100.0,
            patch_idx=0,
            next_id=0,
            params=default_repro_params,
            n_super_individuals=10,
        )
        for r in recruits:
            assert r.sex in (0, 1)

    def test_recruits_are_super_individuals(self, default_repro_params):
        """Each recruit should be a SuperIndividual instance."""
        recruits = create_recruits(
            total_eggs=1_000_000.0,
            spawn_day=100.0,
            zoo_peak_day=100.0,
            patch_idx=0,
            next_id=0,
            params=default_repro_params,
            n_super_individuals=2,
        )
        for r in recruits:
            assert isinstance(r, SuperIndividual)

    def test_default_n_super_individuals_is_one(self, default_repro_params):
        """Default n_super_individuals should be 1."""
        recruits = create_recruits(
            total_eggs=1_000_000.0,
            spawn_day=100.0,
            zoo_peak_day=100.0,
            patch_idx=0,
            next_id=0,
            params=default_repro_params,
        )
        assert len(recruits) == 1
