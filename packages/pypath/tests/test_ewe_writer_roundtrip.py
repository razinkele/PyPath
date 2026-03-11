"""Integration tests for EwE export round-trip.

Read a real EwE database, write it back, verify values match.
"""

import zipfile
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

pytestmark = pytest.mark.integration


DATA_DIR = Path(__file__).parents[3] / "Data"
DB_PATH = str(DATA_DIR / "LT2022_0.5ST_final7.eweaccdb")


@pytest.fixture
def lt2022_path():
    p = Path(DB_PATH)
    if not p.exists():
        pytest.skip("LT2022 database not found")
    return str(p)


class TestCsvRoundTrip:
    """Read LT2022 -> write CSV bundle -> read back CSVs -> compare."""

    def test_group_names_survive(self, lt2022_path, tmp_path):
        from pypath.io.ewemdb import read_ewemdb
        from pypath.io.ewe_writer import write_ewemdb

        params = read_ewemdb(lt2022_path)
        outpath = tmp_path / "rt_test.ewecsv.zip"
        write_ewemdb(params, str(outpath), backend="csv")

        with zipfile.ZipFile(outpath) as zf:
            groups = pd.read_csv(zf.open("EcopathGroup.csv"))

        original_names = params.model[params.model["Type"] != 3]["Group"].tolist()
        exported_names = groups["GroupName"].tolist()
        assert exported_names == original_names

    def test_biomass_values_match(self, lt2022_path, tmp_path):
        from pypath.io.ewemdb import read_ewemdb
        from pypath.io.ewe_writer import write_ewemdb

        params = read_ewemdb(lt2022_path)
        outpath = tmp_path / "rt_test.ewecsv.zip"
        write_ewemdb(params, str(outpath), backend="csv")

        with zipfile.ZipFile(outpath) as zf:
            groups = pd.read_csv(zf.open("EcopathGroup.csv"))

        bio_mask = params.model["Type"] != 3
        original_biomass = params.model[bio_mask]["Biomass"].values
        exported_biomass = groups["Biomass"].values

        for orig, exp in zip(original_biomass, exported_biomass):
            if np.isnan(orig):
                assert np.isnan(exp)
            else:
                np.testing.assert_allclose(exp, orig, rtol=1e-6)

    def test_pb_qb_values_match(self, lt2022_path, tmp_path):
        from pypath.io.ewemdb import read_ewemdb
        from pypath.io.ewe_writer import write_ewemdb

        params = read_ewemdb(lt2022_path)
        outpath = tmp_path / "rt_test.ewecsv.zip"
        write_ewemdb(params, str(outpath), backend="csv")

        with zipfile.ZipFile(outpath) as zf:
            groups = pd.read_csv(zf.open("EcopathGroup.csv"))

        bio_mask = params.model["Type"] != 3
        for col, ewe_col in [("PB", "ProdBiom"), ("QB", "ConsBiom")]:
            orig = params.model[bio_mask][col].values
            exp = groups[ewe_col].values
            for o, e in zip(orig, exp):
                if np.isnan(o):
                    assert np.isnan(e)
                else:
                    np.testing.assert_allclose(e, o, rtol=1e-6)

    def test_diet_count_matches(self, lt2022_path, tmp_path):
        from pypath.io.ewemdb import read_ewemdb
        from pypath.io.ewe_writer import write_ewemdb

        params = read_ewemdb(lt2022_path)
        outpath = tmp_path / "rt_test.ewecsv.zip"
        write_ewemdb(params, str(outpath), backend="csv")

        with zipfile.ZipFile(outpath) as zf:
            diet = pd.read_csv(zf.open("EcopathDietComp.csv"))

        original_nonzero = 0
        for col in params.diet.columns[1:]:
            original_nonzero += (params.diet[col] > 0).sum()

        assert len(diet) == original_nonzero

    def test_fleet_names_survive(self, lt2022_path, tmp_path):
        from pypath.io.ewemdb import read_ewemdb
        from pypath.io.ewe_writer import write_ewemdb

        params = read_ewemdb(lt2022_path)
        outpath = tmp_path / "rt_test.ewecsv.zip"
        write_ewemdb(params, str(outpath), backend="csv")

        with zipfile.ZipFile(outpath) as zf:
            fleets = pd.read_csv(zf.open("EcopathFleet.csv"))

        original_fleets = params.model[params.model["Type"] == 3]["Group"].tolist()
        exported_fleets = fleets["FleetName"].tolist()
        assert exported_fleets == original_fleets


class TestAccessRoundTrip:
    """Full round-trip: read LT2022 -> write Access -> read back via read_ewemdb."""

    @pytest.fixture(autouse=True)
    def _skip_no_odbc(self):
        try:
            from pypath.io._access_writer import AccessWriter

            AccessWriter._check_odbc()
        except (ImportError, RuntimeError):
            pytest.skip("Access ODBC driver not available")

    def test_access_group_names_roundtrip(self, lt2022_path, tmp_path):
        from pypath.io.ewemdb import read_ewemdb
        from pypath.io.ewe_writer import write_ewemdb

        params_orig = read_ewemdb(lt2022_path)
        outpath = tmp_path / "rt_access.eweaccdb"
        write_ewemdb(params_orig, str(outpath), backend="access")

        params_back = read_ewemdb(str(outpath))

        orig_names = params_orig.model[params_orig.model["Type"] != 3][
            "Group"
        ].tolist()
        back_names = params_back.model[params_back.model["Type"] != 3][
            "Group"
        ].tolist()
        assert back_names == orig_names

    def test_access_biomass_roundtrip(self, lt2022_path, tmp_path):
        from pypath.io.ewemdb import read_ewemdb
        from pypath.io.ewe_writer import write_ewemdb

        params_orig = read_ewemdb(lt2022_path)
        outpath = tmp_path / "rt_access.eweaccdb"
        write_ewemdb(params_orig, str(outpath), backend="access")

        params_back = read_ewemdb(str(outpath))

        bio_orig = params_orig.model[params_orig.model["Type"] != 3]["Biomass"].values
        bio_back = params_back.model[params_back.model["Type"] != 3]["Biomass"].values
        for o, b in zip(bio_orig, bio_back):
            if np.isnan(o):
                assert np.isnan(b)
            else:
                np.testing.assert_allclose(b, o, rtol=1e-6)
