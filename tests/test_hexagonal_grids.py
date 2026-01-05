"""
Tests for hexagonal grid generation in ECOSPACE.

Tests the create_hexagonal_grid_in_boundary function which generates
regular hexagonal grids within boundary polygons.
"""

import sys
from pathlib import Path

import numpy as np
import pytest

# Add src and app to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
sys.path.insert(0, str(Path(__file__).parent.parent / "app"))

try:
    import geopandas as gpd
    from pages.ecospace import create_hexagon, create_hexagonal_grid_in_boundary
    from shapely.geometry import Point, Polygon

    HAS_GIS = True
except ImportError:
    HAS_GIS = False
    pytestmark = pytest.mark.skip(reason="geopandas not available")


class TestHexagonGeometry:
    """Test basic hexagon geometry creation."""

    def test_create_single_hexagon(self):
        """Test creation of a single hexagon."""
        hexagon = create_hexagon(0, 0, 1000)  # 1 km radius at origin

        assert hexagon.geom_type == "Polygon"
        assert len(hexagon.exterior.coords) == 7  # 6 vertices + close

        # Check that hexagon is centered at origin
        centroid = hexagon.centroid
        assert abs(centroid.x) < 1e-10
        assert abs(centroid.y) < 1e-10

    def test_hexagon_has_six_vertices(self):
        """Test that hexagon has exactly 6 vertices."""
        hexagon = create_hexagon(0, 0, 1000)
        # 7 coordinates (6 vertices + closing point)
        coords = list(hexagon.exterior.coords)
        assert len(coords) == 7
        # First and last should be same (closed)
        assert coords[0] == coords[-1]

    def test_hexagon_dimensions(self):
        """Test hexagon dimensions match expected values."""
        radius = 1000  # meters
        hexagon = create_hexagon(0, 0, radius)

        # Get bounds
        minx, miny, maxx, maxy = hexagon.bounds
        width = maxx - minx
        height = maxy - miny

        # Width should be approximately 2 * radius * cos(30°) = radius * sqrt(3)
        expected_width = radius * np.sqrt(3)
        assert abs(width - expected_width) < 1.0  # Within 1 meter

        # Height should be approximately 2 * radius
        expected_height = 2 * radius
        assert abs(height - expected_height) < 1.0

    def test_hexagon_area(self):
        """Test hexagon area calculation."""
        radius = 1000  # meters
        hexagon = create_hexagon(0, 0, radius)

        # Regular hexagon area = (3 * sqrt(3) / 2) * r^2
        expected_area = (3 * np.sqrt(3) / 2) * radius**2
        actual_area = hexagon.area

        # Should be within 1% of expected
        assert abs(actual_area - expected_area) / expected_area < 0.01


class TestSimpleBoundaryGrid:
    """Test hexagon grid generation in simple rectangular boundary."""

    def test_small_square_boundary(self):
        """Test hexagon generation in a small square boundary."""
        # Create 10km x 10km square boundary
        boundary = Polygon(
            [(20.0, 55.0), (20.1, 55.0), (20.1, 55.1), (20.0, 55.1), (20.0, 55.0)]
        )
        boundary_gdf = gpd.GeoDataFrame([{"geometry": boundary}], crs="EPSG:4326")

        # Generate hexagons (1 km size)
        grid = create_hexagonal_grid_in_boundary(boundary_gdf, hexagon_size_km=1.0)

        # Check basic properties
        assert grid.n_patches > 0
        assert len(grid.patch_ids) == grid.n_patches
        assert len(grid.patch_areas) == grid.n_patches
        assert grid.patch_centroids.shape == (grid.n_patches, 2)

    def test_hexagon_count_scales_with_size(self):
        """Test that smaller hexagons produce more patches."""
        boundary = Polygon(
            [(20.0, 55.0), (20.2, 55.0), (20.2, 55.2), (20.0, 55.2), (20.0, 55.0)]
        )
        boundary_gdf = gpd.GeoDataFrame([{"geometry": boundary}], crs="EPSG:4326")

        # Large hexagons
        grid_large = create_hexagonal_grid_in_boundary(
            boundary_gdf, hexagon_size_km=2.0
        )

        # Small hexagons
        grid_small = create_hexagonal_grid_in_boundary(
            boundary_gdf, hexagon_size_km=0.5
        )

        # Small hexagons should produce more patches
        assert grid_small.n_patches > grid_large.n_patches

    def test_rectangular_boundary(self):
        """Test hexagon generation in rectangular boundary."""
        # Create elongated rectangle
        boundary = Polygon(
            [(20.0, 55.0), (20.3, 55.0), (20.3, 55.1), (20.0, 55.1), (20.0, 55.0)]
        )
        boundary_gdf = gpd.GeoDataFrame([{"geometry": boundary}], crs="EPSG:4326")

        grid = create_hexagonal_grid_in_boundary(boundary_gdf, hexagon_size_km=1.0)

        assert grid.n_patches > 0
        # All centroids should be within boundary (approximately)
        for centroid in grid.patch_centroids:
            point = Point(centroid[0], centroid[1])
            assert boundary.buffer(0.01).contains(point)  # Small buffer for edge cases


class TestComplexBoundaryGrid:
    """Test hexagon generation in complex/irregular boundaries."""

    def test_irregular_coastal_boundary(self):
        """Test hexagon generation in irregular coastal shape."""
        # Create irregular polygon mimicking coastline
        boundary = Polygon(
            [
                (20.0, 55.0),
                (20.3, 55.0),
                (20.4, 55.1),
                (20.3, 55.2),
                (20.5, 55.3),
                (20.2, 55.4),
                (20.0, 55.3),
                (19.9, 55.2),
                (20.0, 55.0),
            ]
        )
        boundary_gdf = gpd.GeoDataFrame([{"geometry": boundary}], crs="EPSG:4326")

        grid = create_hexagonal_grid_in_boundary(boundary_gdf, hexagon_size_km=1.0)

        assert grid.n_patches > 0
        assert grid.geometry is not None
        assert len(grid.geometry) == grid.n_patches

    def test_concave_boundary(self):
        """Test hexagon generation in concave (non-convex) boundary."""
        # Create L-shaped boundary
        boundary = Polygon(
            [
                (20.0, 55.0),
                (20.2, 55.0),
                (20.2, 55.1),
                (20.1, 55.1),
                (20.1, 55.2),
                (20.0, 55.2),
                (20.0, 55.0),
            ]
        )
        boundary_gdf = gpd.GeoDataFrame([{"geometry": boundary}], crs="EPSG:4326")

        grid = create_hexagonal_grid_in_boundary(boundary_gdf, hexagon_size_km=0.5)

        assert grid.n_patches > 0
        # Check that hexagons are clipped to boundary
        for geom in grid.geometry.geometry:
            assert boundary.contains(geom) or boundary.intersects(geom)

    def test_multipolygon_boundary(self):
        """Test hexagon generation with multiple boundary polygons."""
        # Create two separate polygons
        poly1 = Polygon(
            [(20.0, 55.0), (20.1, 55.0), (20.1, 55.1), (20.0, 55.1), (20.0, 55.0)]
        )
        poly2 = Polygon(
            [(20.2, 55.0), (20.3, 55.0), (20.3, 55.1), (20.2, 55.1), (20.2, 55.0)]
        )
        boundary_gdf = gpd.GeoDataFrame(
            [{"geometry": poly1}, {"geometry": poly2}], crs="EPSG:4326"
        )

        grid = create_hexagonal_grid_in_boundary(boundary_gdf, hexagon_size_km=0.5)

        # Should generate hexagons in both polygons
        assert grid.n_patches > 2  # At least a few hexagons


class TestHexagonSizes:
    """Test different hexagon sizes."""

    def test_minimum_size_250m(self):
        """Test minimum hexagon size (250m)."""
        boundary = Polygon(
            [(20.0, 55.0), (20.1, 55.0), (20.1, 55.1), (20.0, 55.1), (20.0, 55.0)]
        )
        boundary_gdf = gpd.GeoDataFrame([{"geometry": boundary}], crs="EPSG:4326")

        grid = create_hexagonal_grid_in_boundary(boundary_gdf, hexagon_size_km=0.25)

        assert grid.n_patches > 50  # Should create many small hexagons

    def test_maximum_size_3km(self):
        """Test maximum hexagon size (3km)."""
        boundary = Polygon(
            [(20.0, 55.0), (20.3, 55.0), (20.3, 55.3), (20.0, 55.3), (20.0, 55.0)]
        )
        boundary_gdf = gpd.GeoDataFrame([{"geometry": boundary}], crs="EPSG:4326")

        grid = create_hexagonal_grid_in_boundary(boundary_gdf, hexagon_size_km=3.0)

        assert grid.n_patches < 20  # Should create few large hexagons

    def test_standard_sizes(self):
        """Test common hexagon sizes (0.5, 1.0, 2.0 km)."""
        boundary = Polygon(
            [(20.0, 55.0), (20.2, 55.0), (20.2, 55.2), (20.0, 55.2), (20.0, 55.0)]
        )
        boundary_gdf = gpd.GeoDataFrame([{"geometry": boundary}], crs="EPSG:4326")

        sizes = [0.5, 1.0, 2.0]
        patch_counts = []

        for size in sizes:
            grid = create_hexagonal_grid_in_boundary(boundary_gdf, hexagon_size_km=size)
            patch_counts.append(grid.n_patches)

        # Patch count should decrease with increasing size
        assert patch_counts[0] > patch_counts[1] > patch_counts[2]


class TestGridProperties:
    """Test properties of generated grids."""

    def test_patch_areas(self):
        """Test that patch areas are calculated correctly."""
        boundary = Polygon(
            [(20.0, 55.0), (20.2, 55.0), (20.2, 55.2), (20.0, 55.2), (20.0, 55.0)]
        )
        boundary_gdf = gpd.GeoDataFrame([{"geometry": boundary}], crs="EPSG:4326")

        grid = create_hexagonal_grid_in_boundary(boundary_gdf, hexagon_size_km=1.0)

        # All areas should be positive
        assert np.all(grid.patch_areas > 0)

        # Expected area for 1km hexagon ≈ 2.598 km²
        expected_area = 2.598
        # Interior hexagons should be close to expected
        max_area = np.max(grid.patch_areas)
        assert abs(max_area - expected_area) < expected_area * 0.5  # Within 50%

    def test_patch_centroids_within_boundary(self):
        """Test that all centroids are within or near boundary."""
        boundary = Polygon(
            [(20.0, 55.0), (20.2, 55.0), (20.2, 55.2), (20.0, 55.2), (20.0, 55.0)]
        )
        boundary_gdf = gpd.GeoDataFrame([{"geometry": boundary}], crs="EPSG:4326")

        grid = create_hexagonal_grid_in_boundary(boundary_gdf, hexagon_size_km=1.0)

        # All centroids should be within boundary (with small buffer for clipped hexagons)
        buffered_boundary = boundary.buffer(0.02)  # Small buffer
        for centroid in grid.patch_centroids:
            point = Point(centroid[0], centroid[1])
            assert buffered_boundary.contains(point)

    def test_crs_is_wgs84(self):
        """Test that output CRS is WGS84."""
        boundary = Polygon(
            [(20.0, 55.0), (20.1, 55.0), (20.1, 55.1), (20.0, 55.1), (20.0, 55.0)]
        )
        boundary_gdf = gpd.GeoDataFrame([{"geometry": boundary}], crs="EPSG:4326")

        grid = create_hexagonal_grid_in_boundary(boundary_gdf, hexagon_size_km=1.0)

        assert grid.crs == "EPSG:4326"
        assert grid.geometry.crs.to_string() == "EPSG:4326"


class TestConnectivity:
    """Test hexagon connectivity and adjacency."""

    def test_adjacency_matrix_properties(self):
        """Test basic properties of adjacency matrix."""
        boundary = Polygon(
            [(20.0, 55.0), (20.2, 55.0), (20.2, 55.2), (20.0, 55.2), (20.0, 55.0)]
        )
        boundary_gdf = gpd.GeoDataFrame([{"geometry": boundary}], crs="EPSG:4326")

        grid = create_hexagonal_grid_in_boundary(boundary_gdf, hexagon_size_km=1.0)

        # Adjacency matrix should be square
        assert grid.adjacency_matrix.shape[0] == grid.adjacency_matrix.shape[1]
        assert grid.adjacency_matrix.shape[0] == grid.n_patches

        # Matrix should be symmetric (undirected graph)
        diff = grid.adjacency_matrix - grid.adjacency_matrix.T
        assert np.allclose(diff.data, 0)

        # Diagonal should be zero (no self-loops)
        assert grid.adjacency_matrix.diagonal().sum() == 0

    def test_hexagons_have_up_to_six_neighbors(self):
        """Test that hexagons have at most 6 neighbors."""
        boundary = Polygon(
            [(20.0, 55.0), (20.3, 55.0), (20.3, 55.3), (20.0, 55.3), (20.0, 55.0)]
        )
        boundary_gdf = gpd.GeoDataFrame([{"geometry": boundary}], crs="EPSG:4326")

        grid = create_hexagonal_grid_in_boundary(boundary_gdf, hexagon_size_km=1.0)

        # Count neighbors for each patch
        adj_matrix = grid.adjacency_matrix
        neighbor_counts = np.array(adj_matrix.sum(axis=1)).flatten()

        # All patches should have ≤ 6 neighbors
        assert np.all(neighbor_counts <= 6)

        # At least some interior patches should have 6 neighbors
        if grid.n_patches > 10:  # Only check for larger grids
            assert np.any(neighbor_counts == 6)

    def test_average_connectivity(self):
        """Test average connectivity is reasonable."""
        boundary = Polygon(
            [(20.0, 55.0), (20.3, 55.0), (20.3, 55.3), (20.0, 55.3), (20.0, 55.0)]
        )
        boundary_gdf = gpd.GeoDataFrame([{"geometry": boundary}], crs="EPSG:4326")

        grid = create_hexagonal_grid_in_boundary(boundary_gdf, hexagon_size_km=1.0)

        # Calculate average neighbors
        n_edges = grid.adjacency_matrix.nnz // 2  # Undirected edges
        avg_neighbors = 2 * n_edges / grid.n_patches

        # For hexagonal grids, average should be between 3 and 6
        # (edge hexagons have fewer neighbors)
        assert 3.0 <= avg_neighbors <= 6.0

    def test_edge_lengths(self):
        """Test edge lengths dictionary."""
        boundary = Polygon(
            [(20.0, 55.0), (20.1, 55.0), (20.1, 55.1), (20.0, 55.1), (20.0, 55.0)]
        )
        boundary_gdf = gpd.GeoDataFrame([{"geometry": boundary}], crs="EPSG:4326")

        grid = create_hexagonal_grid_in_boundary(boundary_gdf, hexagon_size_km=1.0)

        # Edge lengths dict should exist
        assert grid.edge_lengths is not None
        assert isinstance(grid.edge_lengths, dict)

        # All edge lengths should be positive
        for length in grid.edge_lengths.values():
            assert length > 0


class TestEdgeCases:
    """Test edge cases and error conditions."""

    def test_very_small_boundary(self):
        """Test with very small boundary."""
        boundary = Polygon(
            [(20.0, 55.0), (20.01, 55.0), (20.01, 55.01), (20.0, 55.01), (20.0, 55.0)]
        )
        boundary_gdf = gpd.GeoDataFrame([{"geometry": boundary}], crs="EPSG:4326")

        # Should create at least one hexagon with small size
        grid = create_hexagonal_grid_in_boundary(boundary_gdf, hexagon_size_km=0.5)
        assert grid.n_patches >= 1

    def test_hexagon_too_large_for_boundary(self):
        """Test error when hexagon is too large for boundary."""
        # Very small boundary
        boundary = Polygon(
            [(20.0, 55.0), (20.01, 55.0), (20.01, 55.01), (20.0, 55.01), (20.0, 55.0)]
        )
        boundary_gdf = gpd.GeoDataFrame([{"geometry": boundary}], crs="EPSG:4326")

        # Try to create very large hexagons
        with pytest.raises(ValueError, match="No hexagons fit within the boundary"):
            create_hexagonal_grid_in_boundary(boundary_gdf, hexagon_size_km=3.0)

    def test_empty_geodataframe(self):
        """Test with empty GeoDataFrame."""
        boundary_gdf = gpd.GeoDataFrame([], geometry=[], crs="EPSG:4326")

        # Should raise an error
        with pytest.raises(Exception):
            create_hexagonal_grid_in_boundary(boundary_gdf, hexagon_size_km=1.0)

    def test_different_hemispheres(self):
        """Test hexagon generation in different hemispheres."""
        # Northern hemisphere
        boundary_north = Polygon(
            [(20.0, 55.0), (20.2, 55.0), (20.2, 55.2), (20.0, 55.2), (20.0, 55.0)]
        )

        # Southern hemisphere
        boundary_south = Polygon(
            [(20.0, -55.0), (20.2, -55.0), (20.2, -55.2), (20.0, -55.2), (20.0, -55.0)]
        )

        gdf_north = gpd.GeoDataFrame([{"geometry": boundary_north}], crs="EPSG:4326")
        gdf_south = gpd.GeoDataFrame([{"geometry": boundary_south}], crs="EPSG:4326")

        # Both should work
        grid_north = create_hexagonal_grid_in_boundary(gdf_north, hexagon_size_km=1.0)
        grid_south = create_hexagonal_grid_in_boundary(gdf_south, hexagon_size_km=1.0)

        assert grid_north.n_patches > 0
        assert grid_south.n_patches > 0


class TestRealWorldScenarios:
    """Test realistic use cases."""

    def test_baltic_sea_like_boundary(self):
        """Test with boundary similar to Baltic Sea example."""
        # Simplified Baltic Sea coastal area
        boundary = Polygon(
            [
                (19.5, 54.8),
                (21.5, 54.8),
                (21.8, 55.0),
                (22.0, 55.3),
                (22.2, 55.6),
                (22.0, 55.9),
                (21.5, 56.2),
                (20.5, 56.3),
                (19.8, 56.1),
                (19.5, 55.8),
                (19.3, 55.4),
                (19.4, 55.0),
                (19.5, 54.8),
            ]
        )
        boundary_gdf = gpd.GeoDataFrame([{"geometry": boundary}], crs="EPSG:4326")

        # Test different sizes
        grid_fine = create_hexagonal_grid_in_boundary(boundary_gdf, hexagon_size_km=0.5)
        grid_medium = create_hexagonal_grid_in_boundary(
            boundary_gdf, hexagon_size_km=1.0
        )
        grid_coarse = create_hexagonal_grid_in_boundary(
            boundary_gdf, hexagon_size_km=2.0
        )

        # All should create grids
        assert grid_fine.n_patches > grid_medium.n_patches > grid_coarse.n_patches

        # Medium grid should have reasonable patch count
        assert 20 < grid_medium.n_patches < 200

    def test_coastal_mpa_scenario(self):
        """Test with small Marine Protected Area boundary."""
        # Small MPA (~5km x 5km)
        boundary = Polygon(
            [(20.0, 55.0), (20.05, 55.0), (20.05, 55.05), (20.0, 55.05), (20.0, 55.0)]
        )
        boundary_gdf = gpd.GeoDataFrame([{"geometry": boundary}], crs="EPSG:4326")

        # Use fine resolution for small MPA
        grid = create_hexagonal_grid_in_boundary(boundary_gdf, hexagon_size_km=0.5)

        # Should create reasonable number of patches
        assert 10 < grid.n_patches < 100

        # All patches should have geometry
        assert grid.geometry is not None
        assert len(grid.geometry) == grid.n_patches


class TestIntegrationWithEcospaceGrid:
    """Test integration with EcospaceGrid structure."""

    def test_grid_has_required_attributes(self):
        """Test that generated grid has all required EcospaceGrid attributes."""
        boundary = Polygon(
            [(20.0, 55.0), (20.1, 55.0), (20.1, 55.1), (20.0, 55.1), (20.0, 55.0)]
        )
        boundary_gdf = gpd.GeoDataFrame([{"geometry": boundary}], crs="EPSG:4326")

        grid = create_hexagonal_grid_in_boundary(boundary_gdf, hexagon_size_km=1.0)

        # Check all required attributes exist
        assert hasattr(grid, "n_patches")
        assert hasattr(grid, "patch_ids")
        assert hasattr(grid, "patch_areas")
        assert hasattr(grid, "patch_centroids")
        assert hasattr(grid, "adjacency_matrix")
        assert hasattr(grid, "edge_lengths")
        assert hasattr(grid, "crs")
        assert hasattr(grid, "geometry")

    def test_patch_ids_are_sequential(self):
        """Test that patch IDs are sequential starting from 0."""
        boundary = Polygon(
            [(20.0, 55.0), (20.1, 55.0), (20.1, 55.1), (20.0, 55.1), (20.0, 55.0)]
        )
        boundary_gdf = gpd.GeoDataFrame([{"geometry": boundary}], crs="EPSG:4326")

        grid = create_hexagonal_grid_in_boundary(boundary_gdf, hexagon_size_km=1.0)

        # IDs should be 0, 1, 2, ..., n-1
        expected_ids = np.arange(grid.n_patches)
        assert np.array_equal(grid.patch_ids, expected_ids)

    def test_array_dimensions_match(self):
        """Test that all arrays have consistent dimensions."""
        boundary = Polygon(
            [(20.0, 55.0), (20.2, 55.0), (20.2, 55.2), (20.0, 55.2), (20.0, 55.0)]
        )
        boundary_gdf = gpd.GeoDataFrame([{"geometry": boundary}], crs="EPSG:4326")

        grid = create_hexagonal_grid_in_boundary(boundary_gdf, hexagon_size_km=1.0)

        n = grid.n_patches

        # Check dimensions
        assert len(grid.patch_ids) == n
        assert len(grid.patch_areas) == n
        assert grid.patch_centroids.shape == (n, 2)
        assert grid.adjacency_matrix.shape == (n, n)
        assert len(grid.geometry) == n


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
