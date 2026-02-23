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
