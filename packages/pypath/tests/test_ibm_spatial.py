"""Tests for IBM spatial movement integration."""

import numpy as np
import pytest
import scipy.sparse as sp


class TestSpatialContext:
    """Tests for the SpatialContext dataclass."""

    def test_creation(self):
        """SpatialContext can be constructed with required fields."""
        from pypath.ibm.base import SpatialContext

        adj = sp.csr_matrix(np.array([[0, 1, 0], [1, 0, 1], [0, 1, 0]]))
        ctx = SpatialContext(
            adjacency=adj,
            habitat_quality=np.array([0.8, 0.5, 0.9]),
            food_density=np.array([10.0, 5.0, 15.0]),
            predator_density=np.array([1.0, 2.0, 0.5]),
            n_patches=3,
        )
        assert ctx.n_patches == 3
        assert ctx.adjacency.shape == (3, 3)
        assert ctx.habitat_quality.shape == (3,)

    def test_ibm_step_result_has_patch_biomass(self):
        """IBMStepResult should accept optional patch_biomass field."""
        from pypath.ibm.base import IBMStepResult

        result = IBMStepResult(
            biomass=5.0,
            production=0.1,
            consumption_by_prey=np.zeros(4),
            mortality_count=10.0,
            recruitment_count=5.0,
            patch_biomass=np.array([2.0, 1.5, 1.5]),
        )
        assert result.patch_biomass is not None
        np.testing.assert_allclose(result.patch_biomass.sum(), 5.0)

    def test_ibm_step_result_patch_biomass_defaults_none(self):
        """IBMStepResult.patch_biomass should default to None."""
        from pypath.ibm.base import IBMStepResult

        result = IBMStepResult(
            biomass=5.0,
            production=0.1,
            consumption_by_prey=np.zeros(4),
            mortality_count=10.0,
            recruitment_count=5.0,
        )
        assert result.patch_biomass is None


class TestIBMGroupSpatialSignature:
    """Tests for the updated IBMGroup.compute_step() signature."""

    def test_compute_step_accepts_spatial_context_none(self):
        """compute_step() should accept spatial_context=None (backward compat)."""
        from pypath.ibm.base import IBMGroup, IBMStepResult, SpatialContext

        class MockIBM(IBMGroup):
            def compute_step(self, prey_available, predation_pressure,
                             env_forcing, dt, spatial_context=None):
                return IBMStepResult(
                    biomass=1.0, production=0.0,
                    consumption_by_prey=np.zeros(self.n_groups),
                    mortality_count=0.0, recruitment_count=0.0,
                )

            def get_aggregate_biomass(self):
                return 1.0

            def get_consumption_by_prey(self):
                return np.zeros(self.n_groups)

            def initialize_from_ecosim(self, biomass, params,
                                       n_super_individuals=500):
                pass

        ibm = MockIBM(group_index=1, n_groups=4)
        result = ibm.compute_step(
            prey_available=np.zeros(4),
            predation_pressure=0.0,
            env_forcing={"temperature": 10.0},
            dt=1 / 12,
            spatial_context=None,
        )
        assert result.biomass == 1.0

    def test_spatial_context_importable_from_init(self):
        """SpatialContext should be importable from pypath.ibm."""
        from pypath.ibm import SpatialContext
        assert SpatialContext is not None


def _make_3patch_adjacency():
    """Create a 3-patch linear chain: 0 -- 1 -- 2."""
    adj = sp.lil_matrix((3, 3))
    adj[0, 1] = 1; adj[1, 0] = 1
    adj[1, 2] = 1; adj[2, 1] = 1
    return adj.tocsr()


def _make_smelt_with_spatial(n_patches=3, n_super=30):
    """Create a SmeltIBM initialized for spatial testing."""
    from pypath.ibm.base import SpatialContext
    from pypath.ibm.smelt import SmeltIBM, SmeltParams

    params = SmeltParams.baltic_defaults()
    # Resize foraging arrays for 7-group model
    n_groups = 7
    params.foraging.energy_content = np.full(n_groups, 4.0)
    params.foraging.handling_time = np.ones(n_groups)

    ibm = SmeltIBM(group_index=3, n_groups=n_groups, params=params)
    ibm.initialize_from_ecosim(biomass=5.0, params={}, n_super_individuals=n_super)

    adj = _make_3patch_adjacency()
    ctx = SpatialContext(
        adjacency=adj,
        habitat_quality=np.array([0.5, 0.8, 0.3]),
        food_density=np.array([10.0, 20.0, 5.0]),
        predator_density=np.array([1.0, 0.5, 2.0]),
        n_patches=n_patches,
    )
    return ibm, ctx


class TestSmeltIBMSpatialMovement:
    """Tests for Phase 5 (spatial movement) in SmeltIBM."""

    def test_compute_step_without_spatial_unchanged(self):
        """compute_step without spatial_context should work as before."""
        from pypath.ibm.smelt import SmeltIBM, SmeltParams

        params = SmeltParams.baltic_defaults()
        n_groups = 7
        params.foraging.energy_content = np.full(n_groups, 4.0)
        params.foraging.handling_time = np.ones(n_groups)
        ibm = SmeltIBM(group_index=3, n_groups=n_groups, params=params)
        ibm.initialize_from_ecosim(biomass=5.0, params={}, n_super_individuals=30)

        result = ibm.compute_step(
            prey_available=np.full(n_groups, 2.0),
            predation_pressure=0.1,
            env_forcing={"temperature": 10.0, "month": 6, "zoo_peak_day": 120.0},
            dt=1 / 12,
        )
        assert result.biomass > 0
        assert result.patch_biomass is None

    def test_compute_step_with_spatial_returns_patch_biomass(self):
        """With spatial_context, result should have patch_biomass array."""
        ibm, ctx = _make_smelt_with_spatial()

        result = ibm.compute_step(
            prey_available=np.full(ibm.n_groups, 2.0),
            predation_pressure=0.1,
            env_forcing={"temperature": 10.0, "month": 6, "zoo_peak_day": 120.0},
            dt=1 / 12,
            spatial_context=ctx,
        )
        assert result.patch_biomass is not None
        assert result.patch_biomass.shape == (3,)

    def test_patch_biomass_sums_to_total(self):
        """patch_biomass should sum to total biomass."""
        ibm, ctx = _make_smelt_with_spatial()

        result = ibm.compute_step(
            prey_available=np.full(ibm.n_groups, 2.0),
            predation_pressure=0.1,
            env_forcing={"temperature": 10.0, "month": 6, "zoo_peak_day": 120.0},
            dt=1 / 12,
            spatial_context=ctx,
        )
        np.testing.assert_allclose(
            result.patch_biomass.sum(), result.biomass, rtol=1e-6,
        )

    def test_movement_distributes_across_patches(self):
        """After movement, individuals should be in multiple patches."""
        ibm, ctx = _make_smelt_with_spatial(n_super=100)

        ibm.compute_step(
            prey_available=np.full(ibm.n_groups, 2.0),
            predation_pressure=0.1,
            env_forcing={"temperature": 10.0, "month": 6, "zoo_peak_day": 120.0},
            dt=1 / 12,
            spatial_context=ctx,
        )
        patches_occupied = {ind.patch_idx for ind in ibm.individuals}
        # With 100 individuals and 3 patches, at least 2 patches should be occupied
        assert len(patches_occupied) >= 2

    def test_movement_prefers_food_rich_patches(self):
        """Individuals should tend toward patches with higher food density."""
        # Run 10 trials to reduce stochastic noise
        food_rich_counts = []
        for _ in range(10):
            ibm, ctx = _make_smelt_with_spatial(n_super=200)
            # Extreme food gradient: patch 1 has all the food
            ctx.food_density = np.array([0.1, 100.0, 0.1])
            ctx.habitat_quality = np.array([0.5, 0.5, 0.5])  # uniform habitat
            ctx.predator_density = np.zeros(3)  # no predators

            ibm.compute_step(
                prey_available=np.full(ibm.n_groups, 2.0),
                predation_pressure=0.0,
                env_forcing={"temperature": 10.0, "month": 6, "zoo_peak_day": 120.0},
                dt=1 / 12,
                spatial_context=ctx,
            )
            in_patch_1 = sum(1 for ind in ibm.individuals if ind.patch_idx == 1)
            food_rich_counts.append(in_patch_1)

        avg_in_food_rich = np.mean(food_rich_counts)
        avg_per_patch = np.mean([len(ibm.individuals)]) / 3
        # On average, more individuals should be in the food-rich patch
        assert avg_in_food_rich > avg_per_patch
