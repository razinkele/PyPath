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

# EE from EwE-F Ecopath_Results.dat
EWEF_EE = {
    "Detritus": 0.14918,
    "Phytoplankton": 0.15622,
    "Zooplankton": 0.78241,
    "Meiobenthos": 0.58495,
    "Worms": 0.51271,
    "Sprat": 0.59795,
    "Smelt": 0.77984,
    "Round goby": 0.38043,
    "Blue mussel (ad.)": 0.01728,
    "Piscivorous birds": 0.15331,
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
    assert balanced.EE[i] == pytest.approx(expected, rel=1e-4)
