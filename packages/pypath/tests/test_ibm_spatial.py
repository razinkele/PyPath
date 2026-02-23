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
