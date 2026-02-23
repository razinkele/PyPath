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
