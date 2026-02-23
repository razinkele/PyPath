"""Tests for marine data module."""

import json
import os
import tempfile
from unittest.mock import MagicMock, patch

import numpy as np
import pytest


class TestMarineDataCache:
    """Tests for MarineDataCache."""

    def setup_method(self):
        self.tmpdir = tempfile.mkdtemp()

    def teardown_method(self):
        import shutil

        shutil.rmtree(self.tmpdir, ignore_errors=True)

    def test_cache_miss_returns_none(self):
        from pypath.io.marine_data import MarineDataCache

        cache = MarineDataCache(cache_dir=self.tmpdir)
        assert cache.get("nonexistent") is None

    def test_cache_put_and_get(self):
        from pypath.io.marine_data import MarineDataCache

        cache = MarineDataCache(cache_dir=self.tmpdir)
        cache.put("test_key", b"hello world")
        assert cache.get("test_key") == b"hello world"

    def test_cache_key_deterministic(self):
        from pypath.io.marine_data import MarineDataCache

        cache = MarineDataCache(cache_dir=self.tmpdir)
        k1 = cache.cache_key(bbox=(1.0, 2.0, 3.0, 4.0), layer="habitats")
        k2 = cache.cache_key(bbox=(1.0, 2.0, 3.0, 4.0), layer="habitats")
        assert k1 == k2

    def test_cache_key_differs_for_different_params(self):
        from pypath.io.marine_data import MarineDataCache

        cache = MarineDataCache(cache_dir=self.tmpdir)
        k1 = cache.cache_key(bbox=(1.0, 2.0, 3.0, 4.0), layer="habitats")
        k2 = cache.cache_key(bbox=(5.0, 6.0, 7.0, 8.0), layer="habitats")
        assert k1 != k2

    def test_cache_creates_directory(self):
        subdir = os.path.join(self.tmpdir, "sub", "cache")
        from pypath.io.marine_data import MarineDataCache

        cache = MarineDataCache(cache_dir=subdir)
        cache.put("key", b"data")
        assert os.path.isdir(subdir)


def _has_geopandas():
    try:
        import geopandas  # noqa: F401

        return True
    except ImportError:
        return False


class TestEMODnetHabitatsClient:
    """Tests for EMODnetHabitatsClient with mocked HTTP."""

    def setup_method(self):
        self.tmpdir = tempfile.mkdtemp()

    def teardown_method(self):
        import shutil

        shutil.rmtree(self.tmpdir, ignore_errors=True)

    def _make_mock_geojson(self):
        """Create a minimal GeoJSON FeatureCollection for testing."""
        return {
            "type": "FeatureCollection",
            "features": [
                {
                    "type": "Feature",
                    "properties": {"EUNIScomb": "A5.23"},
                    "geometry": {
                        "type": "Polygon",
                        "coordinates": [
                            [
                                [20.0, 55.0],
                                [21.0, 55.0],
                                [21.0, 56.0],
                                [20.0, 56.0],
                                [20.0, 55.0],
                            ]
                        ],
                    },
                },
                {
                    "type": "Feature",
                    "properties": {"EUNIScomb": "A5.33"},
                    "geometry": {
                        "type": "Polygon",
                        "coordinates": [
                            [
                                [21.0, 55.0],
                                [22.0, 55.0],
                                [22.0, 56.0],
                                [21.0, 56.0],
                                [21.0, 55.0],
                            ]
                        ],
                    },
                },
            ],
        }

    @pytest.mark.skipif(
        not _has_geopandas(), reason="geopandas not installed"
    )
    @patch("requests.get")
    def test_fetch_euseamap_returns_geodataframe(self, mock_get):
        import geopandas as gpd

        from pypath.io.marine_data import EMODnetHabitatsClient, MarineDataCache

        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.content = json.dumps(self._make_mock_geojson()).encode()
        mock_get.return_value = mock_response

        cache = MarineDataCache(cache_dir=self.tmpdir)
        client = EMODnetHabitatsClient(cache=cache)
        gdf = client.fetch_euseamap(bbox=(20.0, 55.0, 22.0, 56.0))

        assert isinstance(gdf, gpd.GeoDataFrame)
        assert len(gdf) == 2
        assert "EUNIScomb" in gdf.columns

    @pytest.mark.skipif(
        not _has_geopandas(), reason="geopandas not installed"
    )
    def test_get_habitat_types_extracts_level3(self):
        import geopandas as gpd
        from shapely.geometry import Polygon as ShapelyPolygon

        from pypath.io.marine_data import EMODnetHabitatsClient, MarineDataCache

        gdf = gpd.GeoDataFrame(
            {"EUNIScomb": ["A5.23", "A5.33", "A5.23"]},
            geometry=[
                ShapelyPolygon([(0, 0), (1, 0), (1, 1), (0, 1)]),
                ShapelyPolygon([(1, 0), (2, 0), (2, 1), (1, 1)]),
                ShapelyPolygon([(2, 0), (3, 0), (3, 1), (2, 1)]),
            ],
            crs="EPSG:4326",
        )
        cache = MarineDataCache(cache_dir=self.tmpdir)
        client = EMODnetHabitatsClient(cache=cache)
        types = client.get_habitat_types(gdf, level=3)

        assert sorted(types) == ["A5.2", "A5.3"]


class TestEMODnetBathymetryClient:
    """Tests for EMODnetBathymetryClient with mocked HTTP."""

    def setup_method(self):
        import tempfile

        self.tmpdir = tempfile.mkdtemp()

    def teardown_method(self):
        import shutil

        shutil.rmtree(self.tmpdir, ignore_errors=True)

    @pytest.mark.skipif(
        not _has_geopandas(), reason="geopandas not installed"
    )
    def test_sample_to_grid_returns_correct_shape(self):
        from pypath.io.marine_data import EMODnetBathymetryClient, MarineDataCache
        from pypath.spatial.ecospace_params import EcospaceGrid

        grid = EcospaceGrid.from_regular_grid(
            bounds=(20.0, 55.0, 21.0, 56.0), nx=3, ny=3
        )
        # Simulate a depth raster: 10x10 grid of values
        raster = np.arange(100, dtype=float).reshape(10, 10)
        # affine-like transform: (x_origin, pixel_width, 0, y_origin, 0, -pixel_height)
        transform = (20.0, 0.1, 0.0, 56.0, 0.0, -0.1)

        cache = MarineDataCache(cache_dir=self.tmpdir)
        client = EMODnetBathymetryClient(cache=cache)
        depth = client.sample_to_grid(raster, transform, grid)

        assert depth.shape == (grid.n_patches,)
        assert np.all(np.isfinite(depth))


class TestSalinityLoader:
    """Tests for SalinityLoader."""

    def setup_method(self):
        import tempfile

        self.tmpdir = tempfile.mkdtemp()

    def teardown_method(self):
        import shutil

        shutil.rmtree(self.tmpdir, ignore_errors=True)

    @pytest.mark.skipif(
        not _has_geopandas(), reason="geopandas not installed"
    )
    def test_load_csv_creates_environmental_layer(self):
        from pypath.io.marine_data import SalinityLoader
        from pypath.spatial.ecospace_params import EcospaceGrid
        from pypath.spatial.environmental import EnvironmentalLayer

        grid = EcospaceGrid.from_regular_grid(
            bounds=(20, 55, 21, 56), nx=3, ny=3
        )
        csv_path = os.path.join(self.tmpdir, "salinity.csv")
        # Write CSV: lon, lat, salinity
        with open(csv_path, "w") as f:
            f.write("lon,lat,salinity\n")
            for i in range(grid.n_patches):
                lon, lat = grid.patch_centroids[i]
                f.write(f"{lon},{lat},{7.0 + i * 0.1}\n")

        layer = SalinityLoader.load_from_csv(csv_path, grid)
        assert isinstance(layer, EnvironmentalLayer)
        assert layer.name == "salinity"
        assert layer.values.shape == (grid.n_patches,)


class TestHabitatPreferenceBuilder:
    """Tests for HabitatPreferenceBuilder."""

    def test_apply_preset_pelagic_returns_uniform(self):
        from pypath.io.marine_data import HabitatPreferenceBuilder

        builder = HabitatPreferenceBuilder()
        prefs = builder.apply_preset(
            n_groups=3,
            habitat_types=["A5.2", "A5.3", "A6.1"],
            preset="pelagic",
        )
        assert prefs.shape == (3, 3)
        assert np.allclose(prefs, 1.0)

    def test_apply_preset_benthic_varies_by_habitat(self):
        from pypath.io.marine_data import HabitatPreferenceBuilder

        builder = HabitatPreferenceBuilder()
        prefs = builder.apply_preset(
            n_groups=2, habitat_types=["A5.2", "A5.3"], preset="benthic"
        )
        assert prefs.shape == (2, 2)
        assert np.all(prefs >= 0) and np.all(prefs <= 1)

    @pytest.mark.skipif(
        not _has_geopandas(), reason="geopandas not installed"
    )
    def test_build_preference_matrix_correct_shape(self):
        from pypath.io.marine_data import HabitatPreferenceBuilder
        from pypath.spatial.ecospace_params import EcospaceGrid

        grid = EcospaceGrid.from_regular_grid(
            bounds=(20, 55, 21, 56), nx=3, ny=3
        )
        habitat_map = np.array(["A5.2"] * 5 + ["A5.3"] * 4)
        prefs_by_type = np.array(
            [[0.8, 0.2], [0.3, 0.9]]
        )  # 2 groups, 2 types

        builder = HabitatPreferenceBuilder()
        matrix = builder.build_preference_matrix(
            prefs_by_type, ["A5.2", "A5.3"], habitat_map, grid
        )
        assert matrix.shape == (2, grid.n_patches)
        # Patches with A5.2 should have group 0 preference = 0.8
        assert matrix[0, 0] == 0.8
        # Patches with A5.3 should have group 1 preference = 0.9
        assert matrix[1, 5] == 0.9


@pytest.mark.integration
@pytest.mark.slow
def test_fetch_euseamap_real_api():
    """Integration test: fetch real data from EMODnet WFS."""
    import shutil

    from pypath.io.marine_data import EMODnetHabitatsClient, MarineDataCache

    cache_dir = tempfile.mkdtemp()
    cache = MarineDataCache(cache_dir=cache_dir)
    client = EMODnetHabitatsClient(cache=cache)
    # Small area in the Baltic Sea
    gdf = client.fetch_euseamap(bbox=(20.5, 55.5, 21.0, 56.0))
    assert len(gdf) > 0
    assert "EUNIScomb" in gdf.columns
    shutil.rmtree(cache_dir, ignore_errors=True)
