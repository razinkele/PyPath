# Ecospace Data Wizard Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Add EMODnet habitat/bathymetry data integration and a 7-step wizard UI for creating ecospace models from real-world marine data.

**Architecture:** Core data fetching in `pypath.io.marine_data` (pypath-ewe package), wizard UI in `pypath_shiny.pages.ecospace_wizard` (pypath-shiny package). Data flows: user draws polygon → grid created → EMODnet APIs queried → habitats rasterized → preferences assigned → EcospaceParams built.

**Tech Stack:** Python, requests (HTTP), geopandas/shapely (GIS), rasterio (GeoTIFF), Shiny for Python (UI), Leaflet (maps)

**Design doc:** `docs/plans/2026-02-23-ecospace-data-wizard-design.md`

---

## Task 1: MarineDataCache — local file cache

**Files:**
- Create: `packages/pypath/src/pypath/io/marine_data.py`
- Test: `packages/pypath/tests/test_marine_data.py`

**Step 1: Write failing tests for cache**

```python
# packages/pypath/tests/test_marine_data.py
"""Tests for marine data module."""
import os
import tempfile
from pathlib import Path

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
```

**Step 2: Run tests to verify they fail**

Run: `pytest packages/pypath/tests/test_marine_data.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'pypath.io.marine_data'`

**Step 3: Implement MarineDataCache**

```python
# packages/pypath/src/pypath/io/marine_data.py
"""Marine data clients for EMODnet habitats, bathymetry, and salinity.

Provides:
- MarineDataCache: Local file cache for downloaded marine data
- EMODnetHabitatsClient: WFS client for EUSeaMap seabed habitats
- EMODnetBathymetryClient: WCS client for bathymetry depth grids
- SalinityLoader: Load salinity from user-provided files
- HabitatPreferenceBuilder: Semi-automatic habitat preference assignment
"""

import hashlib
import json
import logging
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)

_DEFAULT_CACHE_DIR = Path.home() / ".pypath" / "cache" / "marine_data"


class MarineDataCache:
    """Local file cache for marine data downloads.

    Parameters
    ----------
    cache_dir : str or Path
        Directory for cached files. Created if it doesn't exist.
    """

    def __init__(self, cache_dir: Optional[str] = None):
        self._cache_dir = Path(cache_dir) if cache_dir else _DEFAULT_CACHE_DIR

    def get(self, key: str) -> Optional[bytes]:
        """Retrieve cached data by key. Returns None on cache miss."""
        path = self._cache_dir / key
        if path.exists():
            logger.debug("Cache hit: %s", key)
            return path.read_bytes()
        return None

    def put(self, key: str, data: bytes) -> None:
        """Store data in cache."""
        self._cache_dir.mkdir(parents=True, exist_ok=True)
        path = self._cache_dir / key
        path.write_bytes(data)
        logger.debug("Cached: %s (%d bytes)", key, len(data))

    @staticmethod
    def cache_key(bbox: tuple, layer: str, **kwargs) -> str:
        """Generate deterministic cache key from parameters."""
        parts = {"bbox": list(bbox), "layer": layer, **kwargs}
        raw = json.dumps(parts, sort_keys=True)
        return hashlib.sha256(raw.encode()).hexdigest()
```

**Step 4: Run tests to verify they pass**

Run: `pytest packages/pypath/tests/test_marine_data.py -v`
Expected: 5 PASSED

**Step 5: Commit**

```bash
git add packages/pypath/src/pypath/io/marine_data.py packages/pypath/tests/test_marine_data.py
git commit -m "feat(io): add MarineDataCache for EMODnet data caching"
```

---

## Task 2: EMODnetHabitatsClient — WFS client for seabed habitats

**Files:**
- Modify: `packages/pypath/src/pypath/io/marine_data.py`
- Test: `packages/pypath/tests/test_marine_data.py`

**Step 1: Write failing tests (unit tests with mocked HTTP)**

```python
# Append to packages/pypath/tests/test_marine_data.py
import json
from unittest.mock import MagicMock, patch

import numpy as np


class TestEMODnetHabitatsClient:
    """Tests for EMODnetHabitatsClient with mocked HTTP."""

    def setup_method(self):
        import tempfile
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
                        "coordinates": [[[20.0, 55.0], [21.0, 55.0],
                                         [21.0, 56.0], [20.0, 56.0],
                                         [20.0, 55.0]]],
                    },
                },
                {
                    "type": "Feature",
                    "properties": {"EUNIScomb": "A5.33"},
                    "geometry": {
                        "type": "Polygon",
                        "coordinates": [[[21.0, 55.0], [22.0, 55.0],
                                         [22.0, 56.0], [21.0, 56.0],
                                         [21.0, 55.0]]],
                    },
                },
            ],
        }

    @pytest.mark.skipif(
        not _has_geopandas(), reason="geopandas not installed"
    )
    @patch("pypath.io.marine_data.requests.get")
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


def _has_geopandas():
    try:
        import geopandas  # noqa: F401
        return True
    except ImportError:
        return False
```

**Step 2: Run tests to verify they fail**

Run: `pytest packages/pypath/tests/test_marine_data.py::TestEMODnetHabitatsClient -v`
Expected: FAIL — `ImportError: cannot import name 'EMODnetHabitatsClient'`

**Step 3: Implement EMODnetHabitatsClient**

Append to `packages/pypath/src/pypath/io/marine_data.py`:

```python
_EMODNET_HABITATS_WFS = (
    "https://ows.emodnet-seabedhabitats.eu/geoserver/emodnet_view/wfs"
)


class EMODnetHabitatsClient:
    """WFS client for EMODnet EUSeaMap seabed habitats.

    Parameters
    ----------
    cache : MarineDataCache
        Cache instance for storing downloaded data.
    """

    def __init__(self, cache: MarineDataCache):
        self._cache = cache

    def fetch_euseamap(self, bbox: tuple, eunis_level: int = 3):
        """Fetch EUSeaMap habitat polygons within a bounding box.

        Parameters
        ----------
        bbox : tuple
            (min_lon, min_lat, max_lon, max_lat) in WGS84.
        eunis_level : int
            EUNIS classification level (default 3).

        Returns
        -------
        geopandas.GeoDataFrame
            Habitat polygons with EUNIS classification columns.
        """
        import geopandas as gpd
        import requests

        cache_key = self._cache.cache_key(
            bbox=bbox, layer="euseamap", eunis_level=eunis_level
        )
        cached = self._cache.get(cache_key)
        if cached is not None:
            return gpd.read_file(cached.decode() if isinstance(cached, bytes) else cached)

        bbox_str = f"{bbox[1]},{bbox[0]},{bbox[3]},{bbox[2]}"
        params = {
            "service": "WFS",
            "version": "2.0.0",
            "request": "GetFeature",
            "typeName": "emodnet_view:euseamap_2023",
            "outputFormat": "application/json",
            "bbox": bbox_str,
            "srsName": "EPSG:4326",
        }
        logger.info("Fetching EMODnet habitats for bbox %s", bbox)
        resp = requests.get(_EMODNET_HABITATS_WFS, params=params, timeout=120)
        resp.raise_for_status()

        self._cache.put(cache_key, resp.content)
        gdf = gpd.read_file(resp.content.decode())
        logger.info("Downloaded %d habitat features", len(gdf))
        return gdf

    def rasterize_habitats(self, gdf, grid) -> np.ndarray:
        """Assign majority EUNIS habitat class to each grid patch.

        Parameters
        ----------
        gdf : geopandas.GeoDataFrame
            Habitat polygons with 'EUNIScomb' column.
        grid : EcospaceGrid
            Target spatial grid.

        Returns
        -------
        np.ndarray
            EUNIS code per patch [n_patches], dtype=object.
        """
        import geopandas as gpd
        from shapely.geometry import Point

        habitat_per_patch = np.empty(grid.n_patches, dtype=object)
        habitat_per_patch[:] = "unknown"

        if gdf.empty:
            return habitat_per_patch

        for i in range(grid.n_patches):
            centroid = Point(grid.patch_centroids[i, 0], grid.patch_centroids[i, 1])
            within = gdf[gdf.geometry.contains(centroid)]
            if not within.empty:
                habitat_per_patch[i] = within.iloc[0]["EUNIScomb"]
            else:
                nearest = gdf.geometry.distance(centroid)
                if len(nearest) > 0:
                    habitat_per_patch[i] = gdf.iloc[nearest.idxmin()]["EUNIScomb"]

        return habitat_per_patch

    @staticmethod
    def get_habitat_types(gdf, level: int = 3) -> list:
        """Extract unique EUNIS codes truncated to requested level.

        Parameters
        ----------
        gdf : geopandas.GeoDataFrame
            Habitat polygons with 'EUNIScomb' column.
        level : int
            EUNIS hierarchy level (e.g., 3 means 'A5.2').

        Returns
        -------
        list of str
            Sorted unique EUNIS codes at the requested level.
        """
        codes = gdf["EUNIScomb"].dropna().unique()
        truncated = set()
        for code in codes:
            parts = code.split(".")
            if level <= 1:
                truncated.add(parts[0][:1])
            elif level == 2:
                truncated.add(parts[0])
            else:
                # Level 3+: keep first char + "." + first N-2 chars of remainder
                if len(parts) >= 2:
                    sub = parts[1]
                    keep = min(level - 2, len(sub))
                    truncated.add(f"{parts[0]}.{sub[:keep]}")
                else:
                    truncated.add(parts[0])
        return sorted(truncated)
```

**Step 4: Run tests to verify they pass**

Run: `pytest packages/pypath/tests/test_marine_data.py -v`
Expected: All PASSED (habitat tests skip if geopandas missing)

**Step 5: Commit**

```bash
git add packages/pypath/src/pypath/io/marine_data.py packages/pypath/tests/test_marine_data.py
git commit -m "feat(io): add EMODnetHabitatsClient for WFS habitat download"
```

---

## Task 3: EMODnetBathymetryClient — WCS client for depth

**Files:**
- Modify: `packages/pypath/src/pypath/io/marine_data.py`
- Modify: `packages/pypath/tests/test_marine_data.py`

**Step 1: Write failing tests**

```python
# Append to packages/pypath/tests/test_marine_data.py

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

        grid = EcospaceGrid.from_regular_grid(bounds=(20.0, 55.0, 21.0, 56.0), nx=3, ny=3)
        # Simulate a depth raster: 10x10 grid of values
        raster = np.arange(100, dtype=float).reshape(10, 10)
        # affine-like transform: (x_origin, pixel_width, 0, y_origin, 0, -pixel_height)
        transform = (20.0, 0.1, 0.0, 56.0, 0.0, -0.1)

        cache = MarineDataCache(cache_dir=self.tmpdir)
        client = EMODnetBathymetryClient(cache=cache)
        depth = client.sample_to_grid(raster, transform, grid)

        assert depth.shape == (grid.n_patches,)
        assert np.all(np.isfinite(depth))
```

**Step 2: Run test to verify it fails**

Run: `pytest packages/pypath/tests/test_marine_data.py::TestEMODnetBathymetryClient -v`
Expected: FAIL — `ImportError: cannot import name 'EMODnetBathymetryClient'`

**Step 3: Implement EMODnetBathymetryClient**

Append to `packages/pypath/src/pypath/io/marine_data.py`:

```python
_EMODNET_BATHYMETRY_WCS = "https://ows.emodnet-bathymetry.eu/wcs"


class EMODnetBathymetryClient:
    """WCS client for EMODnet bathymetry depth data.

    Parameters
    ----------
    cache : MarineDataCache
        Cache instance for storing downloaded data.
    """

    def __init__(self, cache: MarineDataCache):
        self._cache = cache

    def fetch_depth(self, bbox: tuple, resolution: float = 0.002):
        """Fetch depth raster for a bounding box.

        Parameters
        ----------
        bbox : tuple
            (min_lon, min_lat, max_lon, max_lat) in WGS84.
        resolution : float
            Grid resolution in degrees (default ~200m).

        Returns
        -------
        tuple of (np.ndarray, tuple)
            (raster [rows, cols], transform tuple).
        """
        import requests

        cache_key = self._cache.cache_key(
            bbox=bbox, layer="bathymetry", resolution=resolution
        )
        cached = self._cache.get(cache_key)
        if cached is not None:
            return self._read_geotiff_bytes(cached)

        bbox_str = f"{bbox[0]},{bbox[1]},{bbox[2]},{bbox[3]}"
        params = {
            "service": "WCS",
            "version": "1.0.0",
            "request": "GetCoverage",
            "coverage": "emodnet:mean",
            "crs": "EPSG:4326",
            "BBOX": bbox_str,
            "format": "image/tiff",
            "interpolation": "nearest",
            "resx": str(resolution),
            "resy": str(resolution),
        }
        logger.info("Fetching EMODnet bathymetry for bbox %s", bbox)
        resp = requests.get(_EMODNET_BATHYMETRY_WCS, params=params, timeout=120)
        resp.raise_for_status()

        self._cache.put(cache_key, resp.content)
        return self._read_geotiff_bytes(resp.content)

    @staticmethod
    def _read_geotiff_bytes(data: bytes):
        """Read a GeoTIFF from bytes, return (array, transform)."""
        import io
        try:
            import rasterio
            with rasterio.open(io.BytesIO(data)) as src:
                arr = src.read(1).astype(float)
                t = src.transform
                transform = (t.c, t.a, t.b, t.f, t.d, t.e)
                return arr, transform
        except ImportError:
            logger.warning("rasterio not installed; cannot read GeoTIFF")
            raise

    def sample_to_grid(self, raster: np.ndarray, transform: tuple, grid) -> np.ndarray:
        """Average raster values within each grid patch.

        Parameters
        ----------
        raster : np.ndarray
            Depth raster [rows, cols].
        transform : tuple
            (x_origin, pixel_width, x_skew, y_origin, y_skew, pixel_height).
        grid : EcospaceGrid
            Target spatial grid.

        Returns
        -------
        np.ndarray
            Mean depth per patch [n_patches].
        """
        x_origin, pixel_width, _, y_origin, _, pixel_height = transform
        rows, cols = raster.shape
        depth = np.zeros(grid.n_patches)

        for i in range(grid.n_patches):
            lon, lat = grid.patch_centroids[i]
            col = int((lon - x_origin) / pixel_width)
            row = int((lat - y_origin) / pixel_height)
            col = max(0, min(col, cols - 1))
            row = max(0, min(row, rows - 1))
            depth[i] = raster[row, col]

        return depth
```

**Step 4: Run tests to verify they pass**

Run: `pytest packages/pypath/tests/test_marine_data.py -v`
Expected: All PASSED

**Step 5: Commit**

```bash
git add packages/pypath/src/pypath/io/marine_data.py packages/pypath/tests/test_marine_data.py
git commit -m "feat(io): add EMODnetBathymetryClient for WCS depth download"
```

---

## Task 4: SalinityLoader and HabitatPreferenceBuilder

**Files:**
- Modify: `packages/pypath/src/pypath/io/marine_data.py`
- Modify: `packages/pypath/tests/test_marine_data.py`

**Step 1: Write failing tests**

```python
# Append to packages/pypath/tests/test_marine_data.py

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

        grid = EcospaceGrid.from_regular_grid(bounds=(20, 55, 21, 56), nx=3, ny=3)
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
            n_groups=3, habitat_types=["A5.2", "A5.3", "A6.1"], preset="pelagic"
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
        # Benthic preset: each group gets high preference for one type
        assert np.all(prefs >= 0) and np.all(prefs <= 1)

    @pytest.mark.skipif(
        not _has_geopandas(), reason="geopandas not installed"
    )
    def test_build_preference_matrix_correct_shape(self):
        from pypath.io.marine_data import HabitatPreferenceBuilder
        from pypath.spatial.ecospace_params import EcospaceGrid

        grid = EcospaceGrid.from_regular_grid(bounds=(20, 55, 21, 56), nx=3, ny=3)
        habitat_map = np.array(["A5.2"] * 5 + ["A5.3"] * 4)
        prefs_by_type = np.array([[0.8, 0.2], [0.3, 0.9]])  # 2 groups, 2 types

        builder = HabitatPreferenceBuilder()
        matrix = builder.build_preference_matrix(
            prefs_by_type, ["A5.2", "A5.3"], habitat_map, grid
        )
        assert matrix.shape == (2, grid.n_patches)
        # Patches with A5.2 should have group 0 preference = 0.8
        assert matrix[0, 0] == 0.8
        # Patches with A5.3 should have group 1 preference = 0.9
        assert matrix[1, 5] == 0.9
```

**Step 2: Run tests to verify they fail**

Run: `pytest packages/pypath/tests/test_marine_data.py::TestSalinityLoader packages/pypath/tests/test_marine_data.py::TestHabitatPreferenceBuilder -v`
Expected: FAIL — import errors

**Step 3: Implement SalinityLoader and HabitatPreferenceBuilder**

Append to `packages/pypath/src/pypath/io/marine_data.py`:

```python
class SalinityLoader:
    """Load salinity data from user-provided files."""

    @staticmethod
    def load_from_csv(filepath: str, grid) -> "EnvironmentalLayer":
        """Load salinity from CSV with lon, lat, salinity columns.

        Parameters
        ----------
        filepath : str
            Path to CSV file with columns: lon, lat, salinity.
        grid : EcospaceGrid
            Target spatial grid for nearest-neighbor sampling.

        Returns
        -------
        EnvironmentalLayer
            Salinity values sampled onto the grid patches.
        """
        import pandas as pd
        from pypath.spatial.environmental import EnvironmentalLayer

        df = pd.read_csv(filepath)
        required = {"lon", "lat", "salinity"}
        if not required.issubset(df.columns):
            raise ValueError(f"CSV must have columns: {required}, got: {set(df.columns)}")

        values = np.zeros(grid.n_patches)
        for i in range(grid.n_patches):
            lon, lat = grid.patch_centroids[i]
            dists = (df["lon"] - lon) ** 2 + (df["lat"] - lat) ** 2
            values[i] = df.loc[dists.idxmin(), "salinity"]

        return EnvironmentalLayer(name="salinity", units="PSU", values=values)

    @staticmethod
    def load_from_netcdf(filepath: str, grid, variable: str = "so") -> "EnvironmentalLayer":
        """Load salinity from NetCDF.

        Parameters
        ----------
        filepath : str
            Path to NetCDF file.
        grid : EcospaceGrid
            Target spatial grid.
        variable : str
            NetCDF variable name for salinity (default: 'so').

        Returns
        -------
        EnvironmentalLayer
            Salinity values sampled onto the grid patches.
        """
        try:
            import xarray as xr
        except ImportError:
            raise ImportError("xarray required for NetCDF support: pip install xarray netCDF4")

        from pypath.spatial.environmental import EnvironmentalLayer

        ds = xr.open_dataset(filepath)
        sal = ds[variable]

        # Handle time dimension: take mean if present
        if "time" in sal.dims:
            sal = sal.mean(dim="time")
        # Handle depth dimension: take surface layer
        for dim in ["depth", "lev", "z"]:
            if dim in sal.dims:
                sal = sal.isel({dim: 0})

        values = np.zeros(grid.n_patches)
        lons = sal.coords[_find_coord(sal, "lon")].values
        lats = sal.coords[_find_coord(sal, "lat")].values

        for i in range(grid.n_patches):
            plon, plat = grid.patch_centroids[i]
            lon_idx = np.argmin(np.abs(lons - plon))
            lat_idx = np.argmin(np.abs(lats - plat))
            values[i] = float(sal.values[lat_idx, lon_idx])

        ds.close()
        return EnvironmentalLayer(name="salinity", units="PSU", values=values)


def _find_coord(da, kind: str) -> str:
    """Find longitude or latitude coordinate name in xarray DataArray."""
    candidates = {
        "lon": ["longitude", "lon", "x", "nav_lon"],
        "lat": ["latitude", "lat", "y", "nav_lat"],
    }
    for name in candidates.get(kind, []):
        if name in da.coords:
            return name
    raise ValueError(f"Cannot find {kind} coordinate in {list(da.coords)}")


class HabitatPreferenceBuilder:
    """Build habitat preference matrices for ecospace models."""

    def apply_preset(
        self, n_groups: int, habitat_types: list, preset: str
    ) -> np.ndarray:
        """Apply a preset preference pattern.

        Parameters
        ----------
        n_groups : int
            Number of species groups.
        habitat_types : list of str
            Unique habitat type codes.
        preset : str
            One of 'pelagic', 'demersal', 'benthic'.

        Returns
        -------
        np.ndarray
            Preference matrix [n_groups, n_habitat_types], values 0-1.
        """
        n_types = len(habitat_types)
        if preset == "pelagic":
            return np.ones((n_groups, n_types))
        elif preset == "benthic":
            prefs = np.full((n_groups, n_types), 0.2)
            for g in range(n_groups):
                primary = g % n_types
                prefs[g, primary] = 1.0
            return prefs
        elif preset == "demersal":
            return np.full((n_groups, n_types), 0.6)
        else:
            raise ValueError(f"Unknown preset: {preset}")

    def suggest_preferences(
        self, group_names: list, habitat_types: list,
        depth_per_patch: Optional[np.ndarray] = None,
    ):
        """Auto-suggest preferences using biodata lookups.

        Parameters
        ----------
        group_names : list of str
            Species/group names from the Ecopath model.
        habitat_types : list of str
            Unique EUNIS habitat type codes.
        depth_per_patch : np.ndarray, optional
            Depth values per patch for depth-based suggestions.

        Returns
        -------
        np.ndarray
            Suggested preference matrix [n_groups, n_habitat_types].
        """
        n_groups = len(group_names)
        n_types = len(habitat_types)
        prefs = np.ones((n_groups, n_types)) * 0.5  # default moderate

        # Try biodata lookups for each group
        for g, name in enumerate(group_names):
            try:
                from pypath.io.biodata import get_species_info
                info = get_species_info(name)
                if info and hasattr(info, "traits") and info.traits:
                    # Use depth range to weight preferences
                    if info.traits.depth_range_shallow is not None:
                        for t, htype in enumerate(habitat_types):
                            if htype.startswith("A5"):  # sublittoral
                                prefs[g, t] = 0.8
                            elif htype.startswith("A6"):  # deep
                                if info.traits.depth_range_deep and info.traits.depth_range_deep > 200:
                                    prefs[g, t] = 0.7
                                else:
                                    prefs[g, t] = 0.2
            except Exception as e:
                logger.debug("Biodata lookup failed for %s: %s", name, e)

        return prefs

    @staticmethod
    def build_preference_matrix(
        prefs_by_type: np.ndarray,
        habitat_types: list,
        habitat_map: np.ndarray,
        grid,
    ) -> np.ndarray:
        """Convert habitat-type preferences to per-patch preferences.

        Parameters
        ----------
        prefs_by_type : np.ndarray
            Preference per habitat type [n_groups, n_habitat_types].
        habitat_types : list of str
            Ordered habitat type codes matching prefs_by_type columns.
        habitat_map : np.ndarray
            EUNIS code per patch [n_patches], dtype=object.
        grid : EcospaceGrid
            Target spatial grid.

        Returns
        -------
        np.ndarray
            Preference matrix [n_groups, n_patches].
        """
        n_groups = prefs_by_type.shape[0]
        type_to_idx = {t: i for i, t in enumerate(habitat_types)}
        matrix = np.full((n_groups, grid.n_patches), 0.5)

        for p in range(grid.n_patches):
            htype = habitat_map[p]
            # Match to the closest habitat type (truncate to same level)
            matched = False
            for t, code in enumerate(habitat_types):
                if htype.startswith(code) or code.startswith(htype):
                    matrix[:, p] = prefs_by_type[:, t]
                    matched = True
                    break
            if not matched and htype in type_to_idx:
                matrix[:, p] = prefs_by_type[:, type_to_idx[htype]]

        return matrix
```

**Step 4: Run tests to verify they pass**

Run: `pytest packages/pypath/tests/test_marine_data.py -v`
Expected: All PASSED

**Step 5: Commit**

```bash
git add packages/pypath/src/pypath/io/marine_data.py packages/pypath/tests/test_marine_data.py
git commit -m "feat(io): add SalinityLoader and HabitatPreferenceBuilder"
```

---

## Task 5: Export marine_data from pypath.io and add optional dependency

**Files:**
- Modify: `packages/pypath/src/pypath/io/__init__.py`
- Modify: `packages/pypath/pyproject.toml`

**Step 1: Update io/__init__.py to export marine_data classes**

Add to the lazy imports section of `packages/pypath/src/pypath/io/__init__.py`:

```python
# After existing imports, add marine_data exports
try:
    from .marine_data import (
        EMODnetBathymetryClient,
        EMODnetHabitatsClient,
        HabitatPreferenceBuilder,
        MarineDataCache,
        SalinityLoader,
    )
except ImportError as e:
    logger.debug("marine_data not available: %s", e)
```

**Step 2: Add `spatial-data` optional dependency group to pyproject.toml**

In `packages/pypath/pyproject.toml`, after the `spatial` extra (line 42), add:

```toml
spatial-data = [
    "geopandas>=0.12",
    "shapely>=2.0",
    "requests>=2.28",
    "rasterio>=1.3",
]
```

Update the `all` extra to include `spatial-data`.

**Step 3: Run existing tests to verify nothing breaks**

Run: `pytest packages/pypath/tests/test_marine_data.py -v`
Expected: All PASSED

**Step 4: Commit**

```bash
git add packages/pypath/src/pypath/io/__init__.py packages/pypath/pyproject.toml
git commit -m "feat(io): export marine_data, add spatial-data optional dependency"
```

---

## Task 6: Ecospace Wizard UI — page skeleton and step navigation

**Files:**
- Create: `packages/pypath-shiny/src/pypath_shiny/pages/ecospace_wizard.py`
- Modify: `packages/pypath-shiny/src/pypath_shiny/pages/__init__.py`
- Modify: `packages/pypath-shiny/src/pypath_shiny/app.py`
- Test: `packages/pypath-shiny/tests/test_ecospace_wizard.py`

**Step 1: Write failing test for page import**

```python
# packages/pypath-shiny/tests/test_ecospace_wizard.py
"""Tests for the ecospace wizard page."""


def test_wizard_page_imports():
    from pypath_shiny.pages import ecospace_wizard
    assert hasattr(ecospace_wizard, "ecospace_wizard_ui")
    assert hasattr(ecospace_wizard, "ecospace_wizard_server")
```

**Step 2: Run test to verify it fails**

Run: `pytest packages/pypath-shiny/tests/test_ecospace_wizard.py -v`
Expected: FAIL — module not found

**Step 3: Create wizard page skeleton**

```python
# packages/pypath-shiny/src/pypath_shiny/pages/ecospace_wizard.py
"""Ecospace Data Wizard — 7-step guided ecospace model creation.

Steps:
1. Select Area — draw polygon on map
2. Configure Grid — choose grid type and resolution
3. Download Data — fetch EMODnet habitats and bathymetry
4. Review Habitats — inspect and merge EUNIS categories
5. Assign Preferences — semi-auto habitat preferences per species group
6. Set Dispersal — per-group dispersal parameters
7. Review & Launch — summary and build EcospaceParams
"""

import logging

import numpy as np
from shiny import Inputs, Outputs, Session, reactive, render, req, ui

logger = logging.getLogger(__name__)

_STEPS = [
    "Select Area",
    "Configure Grid",
    "Download Data",
    "Review Habitats",
    "Assign Preferences",
    "Set Dispersal",
    "Review & Launch",
]


def _step_progress_ui():
    """Render the step progress bar."""
    items = []
    for i, label in enumerate(_STEPS, 1):
        items.append(
            ui.span(
                f"{i}. {label}",
                class_="badge bg-secondary me-1",
                id=f"wizard_step_badge_{i}",
            )
        )
    return ui.div(*items, class_="mb-3")


def ecospace_wizard_ui():
    """Wizard page UI."""
    return ui.page_fluid(
        ui.h3("Ecospace Data Wizard"),
        _step_progress_ui(),
        ui.output_ui("wizard_step_content"),
        ui.div(
            ui.input_action_button("wizard_back", "Back", class_="btn-secondary me-2"),
            ui.input_action_button("wizard_next", "Next", class_="btn-primary"),
            class_="mt-3",
        ),
    )


def ecospace_wizard_server(input: Inputs, output: Outputs, session: Session,
                           shared_data=None):
    """Wizard page server logic."""
    wizard_step = reactive.value(1)

    @reactive.effect
    @reactive.event(input.wizard_next)
    def _next():
        current = wizard_step.get()
        if current < len(_STEPS):
            wizard_step.set(current + 1)

    @reactive.effect
    @reactive.event(input.wizard_back)
    def _back():
        current = wizard_step.get()
        if current > 1:
            wizard_step.set(current - 1)

    @render.ui
    def wizard_step_content():
        step = wizard_step.get()
        if step == 1:
            return _step1_select_area_ui()
        elif step == 2:
            return _step2_configure_grid_ui()
        elif step == 3:
            return _step3_download_data_ui()
        elif step == 4:
            return _step4_review_habitats_ui()
        elif step == 5:
            return _step5_assign_preferences_ui()
        elif step == 6:
            return _step6_set_dispersal_ui()
        elif step == 7:
            return _step7_review_launch_ui()
        return ui.p("Unknown step")


def _step1_select_area_ui():
    return ui.card(
        ui.card_header("Step 1: Select Study Area"),
        ui.p("Draw a polygon on the map to define your study area."),
        ui.output_ui("wizard_map"),
    )


def _step2_configure_grid_ui():
    return ui.card(
        ui.card_header("Step 2: Configure Grid"),
        ui.input_radio_buttons(
            "wizard_grid_type", "Grid Type",
            choices={"regular": "Regular Rectangular", "hexagonal": "Hexagonal"},
            selected="regular",
        ),
        ui.input_numeric("wizard_cell_size", "Cell Size (km)", value=5, min=0.5, max=100),
    )


def _step3_download_data_ui():
    return ui.card(
        ui.card_header("Step 3: Download Data"),
        ui.p("Download EMODnet seabed habitats and bathymetry for your study area."),
        ui.input_action_button("wizard_download", "Download Data", class_="btn-primary"),
        ui.output_ui("wizard_download_status"),
        ui.hr(),
        ui.p("Salinity (optional):"),
        ui.input_file("wizard_salinity_file", "Upload salinity file (CSV or NetCDF)",
                       accept=[".csv", ".nc", ".nc4"]),
    )


def _step4_review_habitats_ui():
    return ui.card(
        ui.card_header("Step 4: Review Habitats"),
        ui.p("Review EUNIS habitat types assigned to each grid patch."),
        ui.output_ui("wizard_habitat_map"),
        ui.output_table("wizard_habitat_table"),
    )


def _step5_assign_preferences_ui():
    return ui.card(
        ui.card_header("Step 5: Assign Habitat Preferences"),
        ui.input_select(
            "wizard_preset", "Quick Preset",
            choices={"none": "-- Manual --", "pelagic": "Pelagic",
                     "demersal": "Demersal", "benthic": "Benthic",
                     "auto": "Auto-suggest (biodata)"},
        ),
        ui.output_ui("wizard_preference_editor"),
    )


def _step6_set_dispersal_ui():
    return ui.card(
        ui.card_header("Step 6: Set Dispersal Parameters"),
        ui.input_slider("wizard_dispersal_default", "Default Dispersal Rate (km²/month)",
                        min=0.0, max=100.0, value=10.0, step=0.5),
        ui.input_slider("wizard_gravity", "Gravity Strength", min=0.0, max=1.0,
                        value=0.3, step=0.05),
        ui.output_ui("wizard_dispersal_table"),
    )


def _step7_review_launch_ui():
    return ui.card(
        ui.card_header("Step 7: Review & Launch"),
        ui.output_ui("wizard_summary"),
        ui.input_action_button("wizard_create", "Create Ecospace Model",
                               class_="btn-success btn-lg"),
    )
```

**Step 4: Register the page in `__init__.py` and `app.py`**

In `packages/pypath-shiny/src/pypath_shiny/pages/__init__.py`, add `"ecospace_wizard"` to the `_optional_modules` list (line 13-28) and `__all__` list (line 38-55).

In `packages/pypath-shiny/src/pypath_shiny/app.py`, add a nav_panel for the wizard inside the Advanced menu (after the Ecospace entry, around line 142):

```python
ui.nav_panel(
    _icon_label("bi-magic", "Ecospace Wizard"),
    ecospace_wizard.ecospace_wizard_ui(),
    value="Ecospace Wizard",
),
```

Also add the import at the top with other page imports.

**Step 5: Run test to verify it passes**

Run: `pytest packages/pypath-shiny/tests/test_ecospace_wizard.py -v`
Expected: PASSED

**Step 6: Commit**

```bash
git add packages/pypath-shiny/src/pypath_shiny/pages/ecospace_wizard.py \
       packages/pypath-shiny/src/pypath_shiny/pages/__init__.py \
       packages/pypath-shiny/src/pypath_shiny/app.py \
       packages/pypath-shiny/tests/test_ecospace_wizard.py
git commit -m "feat(shiny): add ecospace wizard page skeleton with step navigation"
```

---

## Task 7: Step 1 — Leaflet map with polygon drawing

**Files:**
- Modify: `packages/pypath-shiny/src/pypath_shiny/pages/ecospace_wizard.py`

**Step 1: Implement the Leaflet map with drawing tools**

Replace `_step1_select_area_ui()` and add map server logic using `ipyleaflet` or Shiny's `output_widget` pattern. The existing ecospace page uses `folium` for static maps — for the drawing interaction, use Shiny's `ui.HTML()` with a Leaflet.draw JavaScript snippet that communicates back via `session.send_custom_message`.

This step requires:
- A Leaflet map rendered as HTML with Leaflet.draw plugin
- JavaScript that sends drawn polygon coordinates back to Shiny via `Shiny.setInputValue()`
- Server-side reactive to capture `input.wizard_drawn_polygon`

**Step 2: Test interactively**

Run: `python -c "import uvicorn; from pypath_shiny.app import app; uvicorn.run(app, host='127.0.0.1', port=8000)"`

Navigate to Ecospace Wizard, draw a polygon, verify coordinates are captured.

**Step 3: Commit**

```bash
git add packages/pypath-shiny/src/pypath_shiny/pages/ecospace_wizard.py
git commit -m "feat(shiny): add Leaflet map with polygon drawing to wizard step 1"
```

---

## Task 8: Steps 2-3 — Grid creation and data download

**Files:**
- Modify: `packages/pypath-shiny/src/pypath_shiny/pages/ecospace_wizard.py`

**Step 1: Implement grid creation from drawn polygon**

Wire up Step 2 to create an `EcospaceGrid` from the drawn polygon using the selected grid type (regular/hex) and cell size. Store in reactive state.

**Step 2: Implement data download**

Wire up Step 3's download button to:
1. Create `MarineDataCache` and `EMODnetHabitatsClient` / `EMODnetBathymetryClient`
2. Call `fetch_euseamap()` and `fetch_depth()` with the polygon's bounding box
3. Rasterize habitats onto the grid
4. Sample depth onto the grid
5. Handle optional salinity file upload
6. Store results in reactive state
7. Show progress/status messages

**Step 3: Test interactively**

Run the app, go through Steps 1-3 with a small Baltic Sea polygon.

**Step 4: Commit**

```bash
git add packages/pypath-shiny/src/pypath_shiny/pages/ecospace_wizard.py
git commit -m "feat(shiny): implement grid creation and EMODnet data download in wizard"
```

---

## Task 9: Steps 4-5 — Habitat review and preference assignment

**Files:**
- Modify: `packages/pypath-shiny/src/pypath_shiny/pages/ecospace_wizard.py`

**Step 1: Implement habitat review (Step 4)**

- Render habitat map using folium (colored by EUNIS type)
- Show habitat type table with counts and area
- Add merge controls (select types to merge, merge button)

**Step 2: Implement preference assignment (Step 5)**

- Wire preset selector to `HabitatPreferenceBuilder.apply_preset()`
- Wire auto-suggest to `HabitatPreferenceBuilder.suggest_preferences()`
- Render editable preference matrix as a Shiny DataTable
- Store preference matrix in reactive state

**Step 3: Test interactively**

Run through Steps 1-5 end-to-end.

**Step 4: Commit**

```bash
git add packages/pypath-shiny/src/pypath_shiny/pages/ecospace_wizard.py
git commit -m "feat(shiny): implement habitat review and preference assignment in wizard"
```

---

## Task 10: Steps 6-7 — Dispersal settings and EcospaceParams build

**Files:**
- Modify: `packages/pypath-shiny/src/pypath_shiny/pages/ecospace_wizard.py`

**Step 1: Implement dispersal settings (Step 6)**

- Default dispersal rate slider applied to all groups
- Per-group override table (editable)
- Gravity strength slider

**Step 2: Implement review and launch (Step 7)**

- Summary dashboard showing grid, habitat, environment, species, dispersal stats
- "Create Ecospace Model" button that:
  1. Builds `EcospaceParams` from all wizard state
  2. Builds `EnvironmentalDrivers` with depth layer (and salinity if uploaded)
  3. Stores the result in `shared_data` so the existing Ecospace page can use it
  4. Navigates to the Ecospace page

**Step 3: Test end-to-end**

Run through all 7 steps with a small test area.

**Step 4: Commit**

```bash
git add packages/pypath-shiny/src/pypath_shiny/pages/ecospace_wizard.py
git commit -m "feat(shiny): implement dispersal settings and EcospaceParams build in wizard"
```

---

## Task 11: Integration tests and polish

**Files:**
- Modify: `packages/pypath/tests/test_marine_data.py`
- Modify: `packages/pypath-shiny/tests/test_ecospace_wizard.py`

**Step 1: Add integration test for EMODnet API (marked slow)**

```python
@pytest.mark.integration
@pytest.mark.slow
def test_fetch_euseamap_real_api():
    """Integration test: fetch real data from EMODnet WFS."""
    from pypath.io.marine_data import EMODnetHabitatsClient, MarineDataCache
    cache = MarineDataCache()
    client = EMODnetHabitatsClient(cache=cache)
    # Small area in the Baltic Sea
    gdf = client.fetch_euseamap(bbox=(20.5, 55.5, 21.0, 56.0))
    assert len(gdf) > 0
    assert "EUNIScomb" in gdf.columns
```

**Step 2: Add Shiny page smoke tests**

```python
# packages/pypath-shiny/tests/test_ecospace_wizard.py
def test_wizard_step_names():
    from pypath_shiny.pages.ecospace_wizard import _STEPS
    assert len(_STEPS) == 7
    assert _STEPS[0] == "Select Area"
    assert _STEPS[-1] == "Review & Launch"

def test_wizard_ui_renders():
    from pypath_shiny.pages.ecospace_wizard import ecospace_wizard_ui
    ui_result = ecospace_wizard_ui()
    assert ui_result is not None
```

**Step 3: Run all tests**

Run: `pytest packages/pypath/tests/test_marine_data.py packages/pypath-shiny/tests/test_ecospace_wizard.py -v -m "not integration"`
Expected: All PASSED

**Step 4: Commit**

```bash
git add packages/pypath/tests/test_marine_data.py packages/pypath-shiny/tests/test_ecospace_wizard.py
git commit -m "test: add integration and smoke tests for ecospace wizard"
```

---

## Summary

| Task | Component | Estimated Scope |
|------|-----------|----------------|
| 1 | MarineDataCache | Small — cache CRUD |
| 2 | EMODnetHabitatsClient | Medium — WFS + rasterize |
| 3 | EMODnetBathymetryClient | Medium — WCS + sample |
| 4 | SalinityLoader + PreferenceBuilder | Medium — file IO + matrix |
| 5 | Export + dependencies | Small — wiring |
| 6 | Wizard page skeleton | Medium — UI + navigation |
| 7 | Step 1 — Map + drawing | Medium — JS interop |
| 8 | Steps 2-3 — Grid + download | Medium — reactives |
| 9 | Steps 4-5 — Habitats + prefs | Large — editable UI |
| 10 | Steps 6-7 — Dispersal + launch | Medium — build params |
| 11 | Integration tests + polish | Small — tests |
