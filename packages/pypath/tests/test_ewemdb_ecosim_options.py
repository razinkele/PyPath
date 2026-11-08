"""The importer reads the EwE6 Ecosim options a calibrated model relies on.

LT2022 scenario 16: MoPred 1 on every group, FtimeMax 2 (1 for Phytoplankton
and Detritus), UseVarPQ True, ForagingTimeLowerLimit 0.01, and EcopathGroup.vbK
per group (EwE6 uses it for the variable P/Q).
"""

from pathlib import Path

import numpy as np
import pytest

DB_PATH = Path(__file__).parents[3] / "Data" / "LT2022_0.5ST_final7.eweaccdb"

pytestmark = pytest.mark.integration


@pytest.fixture(scope="module")
def scenario():
    if not DB_PATH.exists():
        pytest.skip("LT2022 database not found")
    from pypath.io.ewemdb import EwEDatabaseError, ecosim_scenario_from_ewemdb

    try:
        return ecosim_scenario_from_ewemdb(str(DB_PATH), scenario=16)
    except EwEDatabaseError as e:
        pytest.skip(f"No EwE database driver: {e}")


def _idx(scenario, name):
    return [str(n).strip() for n in scenario.params.spname].index(name)


def test_mopred_and_ftime_max_follow_the_scenario_group_table(scenario):
    p = scenario.params
    names = [str(n).strip() for n in p.spname]
    n_bio = p.NUM_LIVING + p.NUM_DEAD
    assert np.all(p.MoPred[1 : n_bio + 1] == 1.0)
    for i in range(1, n_bio + 1):
        expected = 1.0 if names[i] in ("Phytoplankton", "Detritus") else 2.0
        assert p.FtimeMax[i] == pytest.approx(expected), names[i]


def test_scenario_level_options(scenario):
    assert scenario.params.UseVarPQ is True
    assert scenario.params.ForagingTimeLowerLimit == pytest.approx(0.01)


def test_vbk_and_assimilation_efficiency_from_the_group_table(scenario):
    p = scenario.params
    cod, zoo, scoters = (_idx(scenario, n) for n in ("Cod", "Zooplankton", "Scoters"))
    assert p.VbK[cod] == pytest.approx(0.23)
    assert p.VbK[zoo] == pytest.approx(51.666664)
    assert p.VbK[scoters] == pytest.approx(23.829666)
    ge0 = p.PBopt[cod] / p.FtimeQBOpt[cod]
    assert p.AssimEff[cod] == pytest.approx(
        ge0 * (p.PBopt[cod] + 3 * 0.23) / p.PBopt[cod]
    )


def test_pbmaxs_reaches_the_producer_link(scenario):
    """LT2022 uses EwE's default Pbmaxs of 2: the producer link keeps VV 2."""
    p = scenario.params
    phyto = _idx(scenario, "Phytoplankton")
    link = [
        k
        for k in range(1, len(p.PreyFrom))
        if p.PreyFrom[k] == 0 and p.PreyTo[k] == phyto
    ]
    assert len(link) == 1
    assert p.VV[link[0]] == pytest.approx(2.0)
