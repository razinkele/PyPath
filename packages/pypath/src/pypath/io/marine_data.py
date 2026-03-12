"""Marine data clients for EMODnet habitats, bathymetry, and salinity.

Provides:
- MarineDataCache: Local file cache for downloaded marine data
- EMODnetHabitatsClient: WFS client for EUSeaMap seabed habitats
- EMODnetBathymetryClient: WCS client for bathymetry depth grids
- SalinityLoader: Load salinity from user-provided files
- HabitatPreferenceBuilder: Semi-automatic habitat preference assignment
"""

from __future__ import annotations

import hashlib
import io as _io
import json
import logging
from pathlib import Path
from typing import TYPE_CHECKING, Optional

if TYPE_CHECKING:
    import geopandas as gpd

    from pypath.spatial.environmental import EnvironmentalLayer

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
    def cache_key(bbox: tuple[float, float, float, float], layer: str, **kwargs) -> str:
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

    def fetch_euseamap(
        self, bbox: tuple[float, float, float, float], eunis_level: int = 3
    ):
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

    def rasterize_habitats(
        self, gdf: "gpd.GeoDataFrame", grid: "gpd.GeoDataFrame"
    ) -> np.ndarray:
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
    def get_habitat_types(gdf: "gpd.GeoDataFrame", level: int = 3) -> list:
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

    def fetch_depth(
        self, bbox: tuple[float, float, float, float], resolution: float = 0.002
    ):
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
        self, raster: np.ndarray, transform: tuple, grid: "gpd.GeoDataFrame"
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

        resolved = Path(filepath).resolve()
        if not resolved.is_file():
            raise FileNotFoundError(f"Salinity CSV not found: {resolved}")
        df = pd.read_csv(resolved)
        required = {"lon", "lat", "salinity"}
        if not required.issubset(df.columns):
            raise ValueError(
                f"CSV must have columns: {required}, got: {set(df.columns)}"
            )

        from scipy.spatial import cKDTree

        csv_coords = np.column_stack([df["lon"].values, df["lat"].values])
        tree = cKDTree(csv_coords)
        centroids = np.asarray(grid.patch_centroids)
        _, indices = tree.query(centroids)
        values = df["salinity"].values[indices]

        return EnvironmentalLayer(name="salinity", units="PSU", values=values)

    @staticmethod
    def load_from_netcdf(
        filepath: str, grid, variable: str = "so"
    ) -> "EnvironmentalLayer":
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
            raise ImportError(
                "xarray required for NetCDF support: pip install xarray netCDF4"
            )

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

        lons = sal.coords[_find_coord(sal, "lon")].values
        lats = sal.coords[_find_coord(sal, "lat")].values
        centroids = np.asarray(grid.patch_centroids)
        lon_idx = np.argmin(np.abs(lons[np.newaxis, :] - centroids[:, 0:1]), axis=1)
        lat_idx = np.argmin(np.abs(lats[np.newaxis, :] - centroids[:, 1:2]), axis=1)
        sal_values = sal.values
        values = sal_values[lat_idx, lon_idx].astype(float)

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
        self,
        group_names: list,
        habitat_types: list,
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

        for g, name in enumerate(group_names):
            try:
                from pypath.io.biodata import get_species_info

                info = get_species_info(name)
                if info and hasattr(info, "traits") and info.traits:
                    if info.traits.depth_range_shallow is not None:
                        for t, htype in enumerate(habitat_types):
                            if htype.startswith("A5"):
                                prefs[g, t] = 0.8
                            elif htype.startswith("A6"):
                                if (
                                    info.traits.depth_range_deep
                                    and info.traits.depth_range_deep > 200
                                ):
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
            matched = False
            for t, code in enumerate(habitat_types):
                if htype.startswith(code) or code.startswith(htype):
                    matrix[:, p] = prefs_by_type[:, t]
                    matched = True
                    break
            if not matched and htype in type_to_idx:
                matrix[:, p] = prefs_by_type[:, type_to_idx[htype]]

        return matrix
