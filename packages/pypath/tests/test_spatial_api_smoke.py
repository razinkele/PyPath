"""Smoke tests for the spatial (ECOSPACE) public API.

Converted from the root verify_ecospace.py script.
Tests spatial module attributes, grid creation, allocation, and parameter construction.
"""

import logging

import numpy as np
import pytest

import pypath.spatial as spatial

logger = logging.getLogger(__name__)

REQUIRED_SPATIAL_ATTRS = [
    "EcospaceGrid",
    "EcospaceParams",
    "allocate_gravity",
    "allocate_uniform",
    "create_1d_grid",
    "create_regular_grid",
]


class TestSpatialPublicAPI:
    """Verify that the spatial module exposes the expected public interface."""

    @pytest.mark.parametrize("attr_name", REQUIRED_SPATIAL_ATTRS)
    def test_spatial_attribute_exists(self, attr_name):
        assert hasattr(spatial, attr_name), f"Missing spatial attribute: {attr_name}"


class TestSpatialGridCreation:
    """Test basic grid creation and allocation."""

    def test_create_regular_grid(self):
        grid = spatial.create_regular_grid(bounds=(0, 0, 5, 5), nx=5, ny=5)
        assert grid.n_patches == 25
        logger.debug("Created 5x5 grid with %d patches", grid.n_patches)

    def test_create_1d_grid(self):
        grid = spatial.create_1d_grid(n_patches=10, spacing=1.0)
        assert grid.n_patches == 10
        logger.debug("Created 1D grid with %d patches", grid.n_patches)

    def test_allocate_uniform(self):
        effort = spatial.allocate_uniform(n_patches=25, total_effort=100.0)
        assert abs(effort.sum() - 100.0) < 1e-6
        logger.debug("Uniform allocation total: %.2f", effort.sum())

    def test_ecospace_params_construction(self):
        grid = spatial.create_regular_grid(bounds=(0, 0, 5, 5), nx=5, ny=5)
        n_groups = 5
        n_patches = grid.n_patches

        ecospace_params = spatial.EcospaceParams(
            grid=grid,
            habitat_preference=np.ones((n_groups, n_patches)),
            habitat_capacity=np.ones((n_groups, n_patches)),
            dispersal_rate=np.array([0, 5.0, 2.0, 1.0, 3.0]),
            advection_enabled=np.array([False, True, True, False, True]),
            gravity_strength=np.array([0, 0.5, 0.3, 0, 0.7]),
        )
        assert ecospace_params is not None
        logger.debug("Created EcospaceParams for %d groups", n_groups)
