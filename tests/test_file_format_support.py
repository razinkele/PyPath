"""
Test support for different spatial file formats (GeoJSON, GeoPackage, Shapefile).

Tests verify that boundary polygons can be loaded from various file formats
and used for grid generation in ECOSPACE.
"""

import pytest
import sys
from pathlib import Path
import tempfile
import os

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

try:
    import geopandas as gpd
    from shapely.geometry import Polygon
    HAS_GIS = True
except ImportError:
    HAS_GIS = False
    pytestmark = pytest.mark.skip(reason="geopandas not available")


@pytest.fixture
def sample_boundary():
    """Create a sample boundary polygon."""
    return Polygon([
        (20.0, 55.0),
        (20.2, 55.0),
        (20.2, 55.2),
        (20.0, 55.2),
        (20.0, 55.0)
    ])


@pytest.fixture
def sample_gdf(sample_boundary):
    """Create a sample GeoDataFrame with boundary."""
    return gpd.GeoDataFrame(
        [{'id': 0, 'name': 'Test Boundary'}],
        geometry=[sample_boundary],
        crs="EPSG:4326"
    )


class TestGeoJSONSupport:
    """Test GeoJSON file format support."""

    def test_read_geojson(self, sample_gdf):
        """Test reading GeoJSON file."""
        with tempfile.NamedTemporaryFile(suffix='.geojson', delete=False) as f:
            temp_file = f.name

        try:
            # Write GeoJSON
            sample_gdf.to_file(temp_file, driver='GeoJSON')

            # Read back
            loaded = gpd.read_file(temp_file)

            assert len(loaded) == 1
            assert loaded.crs.to_string() == "EPSG:4326"
            assert 'id' in loaded.columns

        finally:
            if os.path.exists(temp_file):
                os.remove(temp_file)

    def test_geojson_with_multiple_features(self, sample_boundary):
        """Test GeoJSON with multiple boundary features."""
        # Create multi-feature boundary
        poly2 = Polygon([
            (20.3, 55.0),
            (20.5, 55.0),
            (20.5, 55.2),
            (20.3, 55.2),
            (20.3, 55.0)
        ])

        gdf = gpd.GeoDataFrame(
            [
                {'id': 0, 'name': 'Area 1'},
                {'id': 1, 'name': 'Area 2'}
            ],
            geometry=[sample_boundary, poly2],
            crs="EPSG:4326"
        )

        with tempfile.NamedTemporaryFile(suffix='.geojson', delete=False) as f:
            temp_file = f.name

        try:
            gdf.to_file(temp_file, driver='GeoJSON')
            loaded = gpd.read_file(temp_file)

            assert len(loaded) == 2
            assert all(loaded.geometry.geom_type == 'Polygon')

        finally:
            if os.path.exists(temp_file):
                os.remove(temp_file)


class TestGeoPackageSupport:
    """Test GeoPackage (GPKG) file format support."""

    def test_read_geopackage(self, sample_gdf):
        """Test reading GeoPackage file."""
        with tempfile.NamedTemporaryFile(suffix='.gpkg', delete=False) as f:
            temp_file = f.name

        try:
            # Write GeoPackage
            sample_gdf.to_file(temp_file, driver='GPKG')

            # Read back
            loaded = gpd.read_file(temp_file)

            assert len(loaded) == 1
            assert loaded.crs.to_string() == "EPSG:4326"
            assert 'id' in loaded.columns

        finally:
            if os.path.exists(temp_file):
                os.remove(temp_file)

    def test_geopackage_preserves_attributes(self, sample_boundary):
        """Test that GeoPackage preserves attributes."""
        gdf = gpd.GeoDataFrame(
            [{
                'id': 0,
                'name': 'Test Area',
                'area_km2': 123.45,
                'type': 'marine'
            }],
            geometry=[sample_boundary],
            crs="EPSG:4326"
        )

        with tempfile.NamedTemporaryFile(suffix='.gpkg', delete=False) as f:
            temp_file = f.name

        try:
            gdf.to_file(temp_file, driver='GPKG')
            loaded = gpd.read_file(temp_file)

            assert loaded['name'].iloc[0] == 'Test Area'
            assert loaded['area_km2'].iloc[0] == 123.45
            assert loaded['type'].iloc[0] == 'marine'

        finally:
            if os.path.exists(temp_file):
                os.remove(temp_file)

    def test_geopackage_with_layers(self, sample_gdf):
        """Test GeoPackage with multiple layers."""
        with tempfile.NamedTemporaryFile(suffix='.gpkg', delete=False) as f:
            temp_file = f.name

        try:
            # Write first layer
            sample_gdf.to_file(temp_file, layer='boundaries', driver='GPKG')

            # Write second layer
            sample_gdf.to_file(temp_file, layer='zones', driver='GPKG')

            # Read specific layer
            loaded = gpd.read_file(temp_file, layer='boundaries')

            assert len(loaded) == 1

        finally:
            if os.path.exists(temp_file):
                os.remove(temp_file)


class TestShapefileSupport:
    """Test Shapefile format support."""

    def test_read_shapefile(self, sample_gdf):
        """Test reading Shapefile."""
        temp_dir = tempfile.mkdtemp()

        try:
            # Write Shapefile
            shp_file = os.path.join(temp_dir, 'test.shp')
            sample_gdf.to_file(shp_file, driver='ESRI Shapefile')

            # Read back
            loaded = gpd.read_file(shp_file)

            assert len(loaded) == 1
            assert loaded.crs.to_string() == "EPSG:4326"

        finally:
            # Clean up
            import shutil
            shutil.rmtree(temp_dir, ignore_errors=True)


class TestFormatComparison:
    """Compare different formats to ensure consistency."""

    def test_formats_produce_same_geometry(self, sample_gdf):
        """Test that all formats produce equivalent geometries."""
        formats = {
            'geojson': ('GeoJSON', '.geojson'),
            'gpkg': ('GPKG', '.gpkg'),
        }

        results = {}

        for fmt_name, (driver, suffix) in formats.items():
            with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as f:
                temp_file = f.name

            try:
                sample_gdf.to_file(temp_file, driver=driver)
                loaded = gpd.read_file(temp_file)
                results[fmt_name] = loaded.geometry.iloc[0]

            finally:
                if os.path.exists(temp_file):
                    os.remove(temp_file)

        # Compare geometries
        geojson_geom = results['geojson']
        gpkg_geom = results['gpkg']

        assert geojson_geom.equals(gpkg_geom) or geojson_geom.equals_exact(gpkg_geom, tolerance=1e-7)


class TestCRSHandling:
    """Test coordinate reference system handling."""

    def test_geojson_crs_preserved(self, sample_gdf):
        """Test that GeoJSON preserves CRS."""
        with tempfile.NamedTemporaryFile(suffix='.geojson', delete=False) as f:
            temp_file = f.name

        try:
            sample_gdf.to_file(temp_file, driver='GeoJSON')
            loaded = gpd.read_file(temp_file)

            assert loaded.crs.to_string() == "EPSG:4326"

        finally:
            if os.path.exists(temp_file):
                os.remove(temp_file)

    def test_geopackage_crs_preserved(self, sample_gdf):
        """Test that GeoPackage preserves CRS."""
        with tempfile.NamedTemporaryFile(suffix='.gpkg', delete=False) as f:
            temp_file = f.name

        try:
            sample_gdf.to_file(temp_file, driver='GPKG')
            loaded = gpd.read_file(temp_file)

            assert loaded.crs.to_string() == "EPSG:4326"

        finally:
            if os.path.exists(temp_file):
                os.remove(temp_file)

    def test_different_crs_conversion(self, sample_boundary):
        """Test loading and converting different CRS."""
        # Create data in different CRS (Web Mercator)
        gdf_mercator = gpd.GeoDataFrame(
            [{'id': 0}],
            geometry=[sample_boundary],
            crs="EPSG:4326"
        ).to_crs("EPSG:3857")

        with tempfile.NamedTemporaryFile(suffix='.gpkg', delete=False) as f:
            temp_file = f.name

        try:
            gdf_mercator.to_file(temp_file, driver='GPKG')
            loaded = gpd.read_file(temp_file)

            # Convert back to WGS84
            loaded_wgs84 = loaded.to_crs("EPSG:4326")

            assert loaded_wgs84.crs.to_string() == "EPSG:4326"

        finally:
            if os.path.exists(temp_file):
                os.remove(temp_file)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
