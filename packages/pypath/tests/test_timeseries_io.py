"""Tests for time series I/O (CSV and database)."""
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from pypath.core.timeseries import (
    DATTYPE_CATCH,
    DATTYPE_FORCED_BIOMASS,
    DATTYPE_REL_BIOMASS,
    EweTimeSeries,
    EweTimeSeriesCollection,
)
from pypath.io.ewemdb import read_timeseries


class TestReadTimeseries:
    def test_import(self):
        assert callable(read_timeseries)

    def test_missing_file_returns_empty(self, tmp_path):
        coll = read_timeseries(str(tmp_path / "nonexistent.eweaccdb"))
        assert isinstance(coll, EweTimeSeriesCollection)
        assert len(coll.series) == 0
