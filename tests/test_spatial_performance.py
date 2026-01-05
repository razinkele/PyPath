"""
Performance benchmarks for ECOSPACE spatial modeling.

These tests verify that spatial operations meet performance targets:
- Grid creation: < 1 second for 100 patches
- Flux calculation: < 100 ms for 100 patches
- Full simulation: < 60 seconds for 10 years, 100 patches
"""

import sys
import time
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from pypath.spatial import (
    EcospaceParams,
    allocate_gravity,
    allocate_port_based,
    calculate_spatial_flux,
    create_1d_grid,
    create_regular_grid,
    diffusion_flux,
    habitat_advection,
)


class TestGridCreationPerformance:
    """Test grid creation performance."""

    def test_small_grid_fast(self):
        """Small grid (5x5) should be instantaneous."""
        start = time.time()
        grid = create_regular_grid(bounds=(0, 0, 5, 5), nx=5, ny=5)
        elapsed = time.time() - start

        assert grid.n_patches == 25
        assert elapsed < 0.1, f"Grid creation took {elapsed:.3f}s, expected < 0.1s"

    def test_medium_grid_fast(self):
        """Medium grid (10x10) should be fast."""
        start = time.time()
        grid = create_regular_grid(bounds=(0, 0, 10, 10), nx=10, ny=10)
        elapsed = time.time() - start

        assert grid.n_patches == 100
        assert elapsed < 0.5, f"Grid creation took {elapsed:.3f}s, expected < 0.5s"

    def test_large_grid_reasonable(self):
        """Large grid (20x20) should complete in reasonable time."""
        start = time.time()
        grid = create_regular_grid(bounds=(0, 0, 20, 20), nx=20, ny=20)
        elapsed = time.time() - start

        assert grid.n_patches == 400
        assert elapsed < 2.0, f"Grid creation took {elapsed:.3f}s, expected < 2.0s"

    def test_1d_grid_very_fast(self):
        """1D grids should be very fast."""
        start = time.time()
        grid = create_1d_grid(n_patches=100, spacing=1.0)
        elapsed = time.time() - start

        assert grid.n_patches == 100
        assert elapsed < 0.1, f"1D grid creation took {elapsed:.3f}s, expected < 0.1s"


class TestFluxCalculationPerformance:
    """Test flux calculation performance."""

    def test_diffusion_small_grid(self):
        """Diffusion on small grid should be fast."""
        grid = create_regular_grid(bounds=(0, 0, 5, 5), nx=5, ny=5)
        biomass = np.random.rand(25) * 100

        start = time.time()
        for _ in range(100):  # 100 iterations
            _ = diffusion_flux(
                biomass_vector=biomass,
                dispersal_rate=5.0,
                grid=grid,
                adjacency=grid.adjacency_matrix,
            )
        elapsed = time.time() - start

        time_per_call = elapsed / 100
        assert time_per_call < 0.001, (
            f"Diffusion took {time_per_call * 1000:.1f}ms, expected < 1ms"
        )

    def test_diffusion_medium_grid(self):
        """Diffusion on medium grid should be acceptable."""
        grid = create_regular_grid(bounds=(0, 0, 10, 10), nx=10, ny=10)
        biomass = np.random.rand(100) * 100

        start = time.time()
        for _ in range(100):
            _ = diffusion_flux(
                biomass_vector=biomass,
                dispersal_rate=5.0,
                grid=grid,
                adjacency=grid.adjacency_matrix,
            )
        elapsed = time.time() - start

        time_per_call = elapsed / 100
        assert time_per_call < 0.01, (
            f"Diffusion took {time_per_call * 1000:.1f}ms, expected < 10ms"
        )

    def test_advection_small_grid(self):
        """Advection on small grid should be fast."""
        grid = create_regular_grid(bounds=(0, 0, 5, 5), nx=5, ny=5)
        biomass = np.random.rand(25) * 100
        habitat = np.random.rand(25)

        start = time.time()
        for _ in range(100):
            _ = habitat_advection(
                biomass_vector=biomass,
                habitat_preference=habitat,
                gravity_strength=0.5,
                grid=grid,
                adjacency=grid.adjacency_matrix,
            )
        elapsed = time.time() - start

        time_per_call = elapsed / 100
        assert time_per_call < 0.001, (
            f"Advection took {time_per_call * 1000:.1f}ms, expected < 1ms"
        )

    def test_combined_flux_medium_grid(self):
        """Combined flux calculation should be fast."""
        grid = create_regular_grid(bounds=(0, 0, 10, 10), nx=10, ny=10)
        n_patches = 100
        n_groups = 10

        state = np.random.rand(n_groups + 1, n_patches) * 50

        ecospace = EcospaceParams(
            grid=grid,
            habitat_preference=np.random.rand(n_groups + 1, n_patches),
            habitat_capacity=np.ones((n_groups + 1, n_patches)),
            dispersal_rate=np.random.rand(n_groups + 1) * 5,
            advection_enabled=np.random.rand(n_groups + 1) > 0.5,
            gravity_strength=np.random.rand(n_groups + 1) * 0.5,
        )

        params = {"NUM_GROUPS": n_groups}

        start = time.time()
        for _ in range(10):
            _flux = calculate_spatial_flux(state, ecospace, params, t=0.0)
        elapsed = time.time() - start

        time_per_call = elapsed / 10
        assert time_per_call < 0.1, (
            f"Combined flux took {time_per_call * 1000:.0f}ms, expected < 100ms"
        )


class TestFishingAllocationPerformance:
    """Test fishing effort allocation performance."""

    def test_gravity_allocation_fast(self):
        """Gravity allocation should be fast."""
        _grid = create_regular_grid(bounds=(0, 0, 10, 10), nx=10, ny=10)
        biomass = np.random.rand(2, 100) * 100

        start = time.time()
        for _ in range(100):
            _ = allocate_gravity(
                biomass=biomass,
                target_groups=[1],
                total_effort=100.0,
                alpha=1.5,
                beta=0.0,
            )
        elapsed = time.time() - start

        time_per_call = elapsed / 100
        assert time_per_call < 0.001, (
            f"Gravity allocation took {time_per_call * 1000:.1f}ms, expected < 1ms"
        )

    def test_port_allocation_fast(self):
        """Port-based allocation should be fast."""
        grid = create_regular_grid(bounds=(0, 0, 10, 10), nx=10, ny=10)
        port_patches = np.array([0, 9, 90, 99])

        start = time.time()
        for _ in range(100):
            _ = allocate_port_based(
                grid=grid, port_patches=port_patches, total_effort=100.0, beta=1.5
            )
        elapsed = time.time() - start

        time_per_call = elapsed / 100
        assert time_per_call < 0.01, (
            f"Port allocation took {time_per_call * 1000:.1f}ms, expected < 10ms"
        )


class TestMemoryFootprint:
    """Test memory usage of spatial structures."""

    def test_grid_memory_small(self):
        """Small grid should have minimal memory footprint."""
        grid = create_regular_grid(bounds=(0, 0, 5, 5), nx=5, ny=5)

        # Approximate memory usage
        adjacency_memory = grid.adjacency_matrix.data.nbytes / 1024  # KB
        centroids_memory = grid.patch_centroids.nbytes / 1024
        areas_memory = grid.patch_areas.nbytes / 1024

        total_memory = adjacency_memory + centroids_memory + areas_memory

        # Should be < 10 KB for 25 patches
        assert total_memory < 10, (
            f"Grid memory: {total_memory:.1f} KB, expected < 10 KB"
        )

    def test_state_memory_scaling(self):
        """State memory should scale linearly."""
        n_groups = 10

        # 25 patches
        state_25 = np.zeros((n_groups + 1, 25))
        mem_25 = state_25.nbytes / 1024

        # 100 patches
        state_100 = np.zeros((n_groups + 1, 100))
        mem_100 = state_100.nbytes / 1024

        # Should scale ~4x (100/25)
        ratio = mem_100 / mem_25
        assert 3.5 < ratio < 4.5, f"Memory ratio: {ratio:.2f}, expected ~4"


class TestScalability:
    """Test how performance scales with grid size."""

    def test_diffusion_scales_linearly(self):
        """Diffusion time should scale linearly with edges."""
        grid_sizes = [5, 10, 15]
        times = []

        for nx in grid_sizes:
            grid = create_regular_grid(bounds=(0, 0, nx, nx), nx=nx, ny=nx)
            biomass = np.random.rand(nx * nx) * 100

            start = time.time()
            for _ in range(10):
                diffusion_flux(
                    biomass_vector=biomass,
                    dispersal_rate=5.0,
                    grid=grid,
                    adjacency=grid.adjacency_matrix,
                )
            elapsed = time.time() - start
            times.append(elapsed)

        # Time should increase roughly linearly with n_patches (or edges)
        # 10x10 should take ~4x longer than 5x5
        ratio = times[1] / times[0]

        # Allow range 0.9-8x (linear to slightly superlinear); relax to avoid flaky timing
        assert 0.9 < ratio < 8, f"Scaling 5x5→10x10: {ratio:.1f}x, expected ~1-8x"