"""Marine data clients for EMODnet habitats, bathymetry, and salinity.

Provides:
- MarineDataCache: Local file cache for downloaded marine data
- EMODnetHabitatsClient: WFS client for EUSeaMap seabed habitats
- EMODnetBathymetryClient: WCS client for bathymetry depth grids
- SalinityLoader: Load salinity from user-provided files
- HabitatPreferenceBuilder: Semi-automatic habitat preference assignment
"""

import hashlib
import io as _io
import json
import logging
from pathlib import Path
from typing import Optional

import numpy as np

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
            return gpd.read_file(_io.BytesIO(cached))

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
        gdf = gpd.read_file(_io.BytesIO(resp.content))
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
        from shapely.geometry import Point

        habitat_per_patch = np.empty(grid.n_patches, dtype=object)
        habitat_per_patch[:] = "unknown"

        if gdf.empty:
            return habitat_per_patch

        for i in range(grid.n_patches):
            centroid = Point(
                grid.patch_centroids[i, 0], grid.patch_centroids[i, 1]
            )
            within = gdf[gdf.geometry.contains(centroid)]
            if not within.empty:
                habitat_per_patch[i] = within.iloc[0]["EUNIScomb"]
            else:
                nearest = gdf.geometry.distance(centroid)
                if len(nearest) > 0:
                    habitat_per_patch[i] = gdf.iloc[nearest.idxmin()][
                        "EUNIScomb"
                    ]

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
                # Level 3+: keep first part + "." + first (level-2) chars
                if len(parts) >= 2:
                    sub = parts[1]
                    keep = min(level - 2, len(sub))
                    truncated.add(f"{parts[0]}.{sub[:keep]}")
                else:
                    truncated.add(parts[0])
        return sorted(truncated)


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
        try:
            import rasterio

            with rasterio.open(_io.BytesIO(data)) as src:
                arr = src.read(1).astype(float)
                t = src.transform
                transform = (t.c, t.a, t.b, t.f, t.d, t.e)
                return arr, transform
        except ImportError:
            logger.warning("rasterio not installed; cannot read GeoTIFF")
            raise

    def sample_to_grid(
        self, raster: np.ndarray, transform: tuple, grid
    ) -> np.ndarray:
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
