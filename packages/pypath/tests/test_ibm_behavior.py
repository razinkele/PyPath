"""
Tests for IBM behavior module (movement and adaptive foraging).

Validates spatial movement between ECOSPACE patches, migration triggers,
and adaptive prey selection for IBM super-individuals.
"""

import numpy as np
import pytest
import scipy.sparse as sp

from pypath.ibm.base import SuperIndividual
from pypath.ibm.behavior import (
    ForagingParams,
    MovementParams,
    adaptive_forage,
    calculate_movement_probabilities,
    move_individual,
    should_migrate,
)

# ---------------------------------------------------------------------------
# Fixtures / helpers
# ---------------------------------------------------------------------------


def _make_individual(
    id: int = 0,
    patch_idx: int = 0,
    length: float = 15.0,
    weight: float = 50.0,
    n_represented: float = 1000.0,
) -> SuperIndividual:
    """Helper to create a SuperIndividual with minimal required fields."""
    return SuperIndividual(
        id=id,
        n_represented=n_represented,
        weight=weight,
        length=length,
        age=2.0,
        energy_reserve=0.8,
        patch_idx=patch_idx,
        is_mature=False,
        sex=0,
    )


@pytest.fixture
def default_movement_params() -> MovementParams:
    """Return MovementParams with representative values."""
    return MovementParams(
        base_speed=0.5,
        habitat_weight=0.4,
        food_weight=0.4,
        predator_weight=0.2,
        migration_temp_threshold=8.0,
        migration_months=(3, 4, 5),
    )


@pytest.fixture
def simple_adjacency() -> sp.csr_matrix:
    """Return a 4-patch adjacency matrix (linear chain: 0-1-2-3)."""
    # 0 -- 1 -- 2 -- 3
    row = [0, 1, 1, 2, 2, 3]
    col = [1, 0, 2, 1, 3, 2]
    data = [1, 1, 1, 1, 1, 1]
    return sp.csr_matrix((data, (row, col)), shape=(4, 4))


@pytest.fixture
def default_foraging_params() -> ForagingParams:
    """Return ForagingParams for 3 prey groups."""
    return ForagingParams(
        energy_content=np.array([5.0, 10.0, 2.0]),  # kJ/g
        handling_time=np.array([1.0, 2.0, 0.5]),  # time/g
    )


# ===========================================================================
# TestMovement
# ===========================================================================


class TestMovement:
    """Test movement probability calculation and individual movement."""

    def test_probabilities_sum_to_one(self, simple_adjacency, default_movement_params):
        """Movement probabilities must sum to 1.0."""
        habitat_quality = np.array([0.5, 0.8, 0.3, 0.6])
        food_density = np.array([0.4, 0.6, 0.9, 0.2])
        predator_density = np.array([0.1, 0.0, 0.5, 0.3])

        probs = calculate_movement_probabilities(
            current_patch=1,
            adjacency=simple_adjacency,
            habitat_quality=habitat_quality,
            food_density=food_density,
            predator_density=predator_density,
            params=default_movement_params,
        )
        assert probs.sum() == pytest.approx(1.0)

    def test_probabilities_nonnegative(self, simple_adjacency, default_movement_params):
        """All movement probabilities must be >= 0."""
        habitat_quality = np.array([0.5, 0.8, 0.3, 0.6])
        food_density = np.array([0.4, 0.6, 0.9, 0.2])
        predator_density = np.array([0.1, 0.0, 0.5, 0.3])

        probs = calculate_movement_probabilities(
            current_patch=1,
            adjacency=simple_adjacency,
            habitat_quality=habitat_quality,
            food_density=food_density,
            predator_density=predator_density,
            params=default_movement_params,
        )
        assert np.all(probs >= 0.0)

    def test_only_reachable_patches_have_nonzero_probability(
        self, simple_adjacency, default_movement_params
    ):
        """Patches not adjacent (and not current) should have zero probability."""
        habitat_quality = np.array([0.5, 0.8, 0.3, 0.6])
        food_density = np.array([0.4, 0.6, 0.9, 0.2])
        predator_density = np.array([0.1, 0.0, 0.5, 0.3])

        # Patch 0 is adjacent to patch 1 only. Patch 2 and 3 are not reachable.
        probs = calculate_movement_probabilities(
            current_patch=0,
            adjacency=simple_adjacency,
            habitat_quality=habitat_quality,
            food_density=food_density,
            predator_density=predator_density,
            params=default_movement_params,
        )
        # Patch 0 (self) and 1 (neighbor) should have nonzero; 2 and 3 should be 0
        assert probs[2] == pytest.approx(0.0)
        assert probs[3] == pytest.approx(0.0)
        assert probs[0] > 0.0
        assert probs[1] > 0.0

    def test_stays_if_current_patch_is_best(self, default_movement_params):
        """Individual should preferentially stay if current patch is the best."""
        # 2 patches, fully connected
        adj = sp.csr_matrix(np.array([[0, 1], [1, 0]]))
        habitat_quality = np.array([1.0, 0.0])
        food_density = np.array([1.0, 0.0])
        predator_density = np.array([0.0, 10.0])

        probs = calculate_movement_probabilities(
            current_patch=0,
            adjacency=adj,
            habitat_quality=habitat_quality,
            food_density=food_density,
            predator_density=predator_density,
            params=default_movement_params,
        )
        assert (
            probs[0] > probs[1]
        ), "Probability of staying in the best patch should exceed leaving"

    def test_inertia_bonus_with_low_base_speed(self):
        """Low base_speed should give large inertia bonus (stay put)."""
        adj = sp.csr_matrix(np.array([[0, 1], [1, 0]]))
        # Equal quality patches
        habitat_quality = np.array([0.5, 0.5])
        food_density = np.array([0.5, 0.5])
        predator_density = np.array([0.0, 0.0])

        # Very low base_speed -> strong inertia
        params_slow = MovementParams(
            base_speed=0.01,
            habitat_weight=0.5,
            food_weight=0.5,
            predator_weight=0.0,
            migration_temp_threshold=8.0,
        )
        probs = calculate_movement_probabilities(
            current_patch=0,
            adjacency=adj,
            habitat_quality=habitat_quality,
            food_density=food_density,
            predator_density=predator_density,
            params=params_slow,
        )
        # With equal quality and low base_speed, staying should dominate
        assert probs[0] > probs[1]

    def test_all_zero_scores_stays_in_current(self, simple_adjacency):
        """If all scores are zero, individual stays in current patch."""
        # Use predator_weight=0 so the 1/(1+0) avoidance term contributes nothing
        params_zero = MovementParams(
            base_speed=0.5,
            habitat_weight=0.5,
            food_weight=0.5,
            predator_weight=0.0,
            migration_temp_threshold=8.0,
        )
        habitat_quality = np.zeros(4)
        food_density = np.zeros(4)
        predator_density = np.zeros(4)

        probs = calculate_movement_probabilities(
            current_patch=2,
            adjacency=simple_adjacency,
            habitat_quality=habitat_quality,
            food_density=food_density,
            predator_density=predator_density,
            params=params_zero,
        )
        assert probs[2] == pytest.approx(1.0)
        assert probs.sum() == pytest.approx(1.0)

    def test_move_individual_changes_patch(
        self, simple_adjacency, default_movement_params
    ):
        """move_individual should be able to change an individual's patch_idx.

        We run move_individual many times and verify at least one move happens
        (probabilistically near-certain with enough iterations).
        """
        individual = _make_individual(patch_idx=1)
        habitat_quality = np.array([0.9, 0.1, 0.9, 0.1])
        food_density = np.array([0.9, 0.1, 0.9, 0.1])
        predator_density = np.array([0.0, 5.0, 0.0, 5.0])

        moved = False
        rng = np.random.default_rng(42)
        for _ in range(100):
            result = move_individual(
                individual=individual,
                adjacency=simple_adjacency,
                habitat_quality=habitat_quality,
                food_density=food_density,
                predator_density=predator_density,
                params=default_movement_params,
                rng=rng,
            )
            if result.patch_idx != individual.patch_idx:
                moved = True
                break
        assert moved, "move_individual should change patch at least once in 100 trials"

    def test_move_individual_returns_copy(
        self, simple_adjacency, default_movement_params
    ):
        """move_individual must not modify the original individual."""
        individual = _make_individual(patch_idx=0)
        original_patch = individual.patch_idx

        rng = np.random.default_rng(123)
        result = move_individual(
            individual=individual,
            adjacency=simple_adjacency,
            habitat_quality=np.array([0.5, 0.8, 0.3, 0.6]),
            food_density=np.array([0.4, 0.6, 0.9, 0.2]),
            predator_density=np.array([0.1, 0.0, 0.5, 0.3]),
            params=default_movement_params,
            rng=rng,
        )
        assert individual.patch_idx == original_patch
        assert isinstance(result, SuperIndividual)

    def test_move_individual_preserves_other_attributes(
        self, simple_adjacency, default_movement_params
    ):
        """All attributes except patch_idx should be preserved."""
        individual = _make_individual(
            id=42,
            patch_idx=1,
            length=20.0,
            weight=100.0,
            n_represented=500.0,
        )
        individual.age = 3.5
        individual.energy_reserve = 0.9
        individual.is_mature = True
        individual.sex = 1

        rng = np.random.default_rng(99)
        result = move_individual(
            individual=individual,
            adjacency=simple_adjacency,
            habitat_quality=np.array([0.5, 0.8, 0.3, 0.6]),
            food_density=np.array([0.4, 0.6, 0.9, 0.2]),
            predator_density=np.zeros(4),
            params=default_movement_params,
            rng=rng,
        )
        assert result.id == 42
        assert result.n_represented == pytest.approx(500.0)
        assert result.weight == pytest.approx(100.0)
        assert result.length == pytest.approx(20.0)
        assert result.age == pytest.approx(3.5)
        assert result.energy_reserve == pytest.approx(0.9)
        assert result.is_mature is True
        assert result.sex == 1


# ===========================================================================
# TestMigration
# ===========================================================================


class TestMigration:
    """Test the should_migrate function."""

    def test_above_threshold_in_spring_returns_true(self, default_movement_params):
        """Temperature above threshold during migration month -> True."""
        assert (
            should_migrate(
                temperature=10.0,
                month=4,
                params=default_movement_params,
            )
            is True
        )

    def test_below_threshold_returns_false(self, default_movement_params):
        """Temperature below threshold -> False regardless of month."""
        assert (
            should_migrate(
                temperature=5.0,
                month=4,
                params=default_movement_params,
            )
            is False
        )

    def test_wrong_month_returns_false(self, default_movement_params):
        """Non-migration month -> False regardless of temperature."""
        assert (
            should_migrate(
                temperature=15.0,
                month=7,
                params=default_movement_params,
            )
            is False
        )

    def test_at_threshold_returns_false(self, default_movement_params):
        """Temperature exactly at threshold should be False (strictly >)."""
        assert (
            should_migrate(
                temperature=8.0,
                month=3,
                params=default_movement_params,
            )
            is False
        )

    def test_all_migration_months_accepted(self, default_movement_params):
        """All months in migration_months should allow migration."""
        for month in default_movement_params.migration_months:
            assert (
                should_migrate(
                    temperature=12.0,
                    month=month,
                    params=default_movement_params,
                )
                is True
            )

    def test_custom_migration_months(self):
        """Custom migration_months tuple should work."""
        params = MovementParams(
            base_speed=0.5,
            habitat_weight=0.4,
            food_weight=0.4,
            predator_weight=0.2,
            migration_temp_threshold=5.0,
            migration_months=(9, 10, 11),
        )
        assert should_migrate(temperature=7.0, month=10, params=params) is True
        assert should_migrate(temperature=7.0, month=4, params=params) is False


# ===========================================================================
# TestAdaptiveForaging
# ===========================================================================


class TestAdaptiveForaging:
    """Test the adaptive_forage function."""

    def test_selects_most_profitable_prey(self, default_foraging_params):
        """Most profitable prey should receive the largest allocation."""
        prey_available = {0: 100.0, 1: 100.0, 2: 100.0}
        max_consumption = 10.0

        consumption = adaptive_forage(
            prey_available=prey_available,
            max_consumption=max_consumption,
            individual_length=15.0,
            params=default_foraging_params,
        )
        # Profitability: group 0 = 5/1*100=500, group 1 = 10/2*100=500, group 2 = 2/0.5*100=400
        # Groups 0 and 1 tied in profitability, both > group 2
        assert consumption[2] < consumption[0] or consumption[2] < consumption[1]

    def test_total_consumption_does_not_exceed_max(self, default_foraging_params):
        """Total consumption across all prey groups should not exceed max_consumption."""
        prey_available = {0: 100.0, 1: 100.0, 2: 100.0}
        max_consumption = 10.0

        consumption = adaptive_forage(
            prey_available=prey_available,
            max_consumption=max_consumption,
            individual_length=15.0,
            params=default_foraging_params,
        )
        total = sum(consumption.values())
        assert total <= max_consumption + 1e-10

    def test_respects_availability_limits(self, default_foraging_params):
        """Cannot consume more than available for each prey group."""
        prey_available = {0: 2.0, 1: 100.0, 2: 100.0}
        max_consumption = 50.0

        consumption = adaptive_forage(
            prey_available=prey_available,
            max_consumption=max_consumption,
            individual_length=15.0,
            params=default_foraging_params,
        )
        for group_idx, consumed in consumption.items():
            assert consumed <= prey_available[group_idx] + 1e-10, (
                f"Consumed {consumed} > available {prey_available[group_idx]} "
                f"for group {group_idx}"
            )

    def test_returns_dict_with_correct_keys(self, default_foraging_params):
        """Return dict should have same keys as prey_available."""
        prey_available = {0: 10.0, 1: 20.0, 2: 5.0}
        consumption = adaptive_forage(
            prey_available=prey_available,
            max_consumption=10.0,
            individual_length=15.0,
            params=default_foraging_params,
        )
        assert set(consumption.keys()) == set(prey_available.keys())

    def test_all_values_nonnegative(self, default_foraging_params):
        """All consumption values should be non-negative."""
        prey_available = {0: 10.0, 1: 20.0, 2: 5.0}
        consumption = adaptive_forage(
            prey_available=prey_available,
            max_consumption=10.0,
            individual_length=15.0,
            params=default_foraging_params,
        )
        for v in consumption.values():
            assert v >= 0.0

    def test_zero_max_consumption_returns_zeros(self, default_foraging_params):
        """Zero max_consumption should yield zero for all groups."""
        prey_available = {0: 10.0, 1: 20.0}
        consumption = adaptive_forage(
            prey_available=prey_available,
            max_consumption=0.0,
            individual_length=15.0,
            params=default_foraging_params,
        )
        assert all(v == pytest.approx(0.0) for v in consumption.values())

    def test_empty_prey_returns_empty(self, default_foraging_params):
        """Empty prey_available should return empty dict."""
        consumption = adaptive_forage(
            prey_available={},
            max_consumption=10.0,
            individual_length=15.0,
            params=default_foraging_params,
        )
        assert consumption == {}

    def test_single_prey_group(self, default_foraging_params):
        """With a single prey group, all consumption goes to it."""
        prey_available = {0: 100.0}
        max_consumption = 5.0
        consumption = adaptive_forage(
            prey_available=prey_available,
            max_consumption=max_consumption,
            individual_length=15.0,
            params=default_foraging_params,
        )
        assert consumption[0] == pytest.approx(5.0)

    def test_availability_constrains_allocation(self, default_foraging_params):
        """When one prey is scarce, consumption shifts to other prey."""
        # Only 1.0 g available of the most profitable prey
        prey_available = {0: 1.0, 1: 100.0, 2: 100.0}
        max_consumption = 20.0

        consumption = adaptive_forage(
            prey_available=prey_available,
            max_consumption=max_consumption,
            individual_length=15.0,
            params=default_foraging_params,
        )
        assert consumption[0] <= 1.0 + 1e-10
        # The remaining consumption should be distributed to other groups
        total = sum(consumption.values())
        assert total == pytest.approx(max_consumption, abs=1e-8)
