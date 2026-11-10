"""LT2022 Ecopath balance vs EwE-F (independent Fortran EwE implementation).

Reference EE values came from running EwE-F (Akoglu et al. 2015,
doi:10.5194/gmd-8-2687-2015) on the same LT2022 database through the EPath
converter, which reads detritus fates straight from
EcopathDietComp.DetritusFate (2026-11-01). Before read_ewemdb read detritus
fates, PyPath's detritus EE disagreed; living groups already matched.
"""

from pathlib import Path

import pytest

DATA_DIR = Path(__file__).parents[3] / "Data"
DB_PATH = DATA_DIR / "LT2022_0.5ST_final7.eweaccdb"

pytestmark = pytest.mark.integration

# EE from EwE-F Ecopath_Results.dat. EwE-F's input lacked the database's
# BiomAcc (Scoters, Long-tailed duck) and DtImports (Detritus), so those three
# EwE-F values were computed without them and are not pinned here; they are
# pinned against EwE's formula and EwE's stored detritus Export in
# test_ewemdb_ba_detritus_import.py (audit A1, 2026-11-10). BA and the import
# touch no other group's EE.
EWEF_EE = {
    "Piscivorous birds": 0.15331,
    "Turbot": 0.50528,
    "Flounder": 0.37121,
    "Sprat": 0.59795,
    "Sand-eels": 0.50780,
    "Sand goby": 0.76597,
    "Smelt": 0.77984,
    "Perciformes": 0.44494,
    "Cypriniformes": 0.51152,
    "Round goby": 0.38043,
    "Mobile crustaceans": 0.52915,
    "Saduria entomon": 0.32298,
    "Blue mussel (juv.)": 0.01002,
    "Blue mussel (ad.)": 0.01728,
    "Sand bivalves": 0.18315,
    "Worms": 0.51271,
    "Meiobenthos": 0.58495,
    "Zooplankton": 0.78241,
    "Phytoplankton": 0.15622,
}


@pytest.fixture(scope="module")
def balanced():
    if not DB_PATH.exists():
        pytest.skip("LT2022 database not found")
    from pypath.core.ecopath import rpath
    from pypath.io.ewemdb import EwEDatabaseError, read_ewemdb

    try:
        params = read_ewemdb(str(DB_PATH))
    except EwEDatabaseError as e:
        pytest.skip(f"No EwE database driver: {e}")
    return rpath(params)


@pytest.mark.parametrize("group, expected", sorted(EWEF_EE.items()))
def test_ee_matches_ewef(balanced, group, expected):
    i = list(balanced.Group).index(group)
    # Reference values are rounded to 5 decimals, hence abs=1e-5 for small EE
    assert balanced.EE[i] == pytest.approx(expected, rel=1e-4, abs=1e-5)
