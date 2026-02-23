"""Tests for marine data module."""

import os
import tempfile


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
