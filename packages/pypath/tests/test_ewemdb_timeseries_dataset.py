"""A loaded EwE time-series dataset drives the run, as in EwE6.

EwE6 applies the driver series of the dataset loaded in Ecosim: forced
biomass (type -1, absolute, held for the data year: setForcedBiomass with
toForcingTimeStep), fishing effort (3), fishing mortality (4), forced catches
(-6). LT2022 dataset 35 ("fittingdata2022 04 07 with real estimates") forces
Round goby's biomass from 0.03 to 7.8 and back to 2.4 over the twelve years
and carries the fleet's effort. The scenario tables do not name the dataset,
so the caller passes it.
"""

from pathlib import Path

import numpy as np
import pytest

DB_PATH = Path(__file__).parents[3] / "Data" / "LT2022_0.5ST_final7.eweaccdb"
RG_FORCED = [0.03, 0.09, 0.21, 0.3, 0.36, 0.45, 0.6, 5.7, 7.8, 4.2, 4.35, 2.4]
EFFORT = [
    1.0,
    0.636,
    0.518,
    0.532,
    0.641,
    0.693,
    0.732,
    0.73,
    0.867,
    1.069,
    1.136,
    0.678,
]

pytestmark = pytest.mark.integration


@pytest.fixture(scope="module")
def collection():
    if not DB_PATH.exists():
        pytest.skip("LT2022 database not found")
    from pypath.io.ewemdb import read_timeseries

    return read_timeseries(str(DB_PATH), dataset=35)


def test_dataset_series_are_read_with_their_groups(collection):
    from pypath.core.timeseries import DATTYPE_EFFORT, DATTYPE_FORCED_BIOMASS
    from pypath.io.ewemdb import read_ewemdb

    names = [str(g).strip() for g in read_ewemdb(str(DB_PATH)).model["Group"]]
    assert len(collection.series) == 27
    forced = [s for s in collection.series if s.dat_type == DATTYPE_FORCED_BIOMASS]
    assert len(forced) == 1
    assert names[forced[0].group_idx] == "Round goby"
    np.testing.assert_allclose(forced[0].values, RG_FORCED, rtol=1e-6)
    effort = [s for s in collection.series if s.dat_type == DATTYPE_EFFORT]
    assert len(effort) == 1 and effort[0].fleet_idx == 0
    np.testing.assert_allclose(effort[0].values, EFFORT, atol=5e-4)
    herring = [
        s for s in collection.observed_biomass if names[s.group_idx] == "Herring"
    ]
    assert len(herring) == 1 and len(herring[0].values) == 12


def test_scenario_takes_the_dataset_drivers():
    if not DB_PATH.exists():
        pytest.skip("LT2022 database not found")
    from pypath.core.ecosim import rsim_run
    from pypath.io.ewemdb import ecosim_scenario_from_ewemdb

    sc = ecosim_scenario_from_ewemdb(str(DB_PATH), scenario=16, timeseries_dataset=35)
    names = [str(n).strip() for n in sc.params.spname]
    rg = names.index("Round goby")
    forced = sc.forcing.ForcedBio[:, rg]
    for year, value in enumerate(RG_FORCED):
        np.testing.assert_allclose(
            forced[12 * year : 12 * (year + 1)], value, rtol=1e-6
        )
    others = np.delete(sc.forcing.ForcedBio, rg, axis=1)
    assert np.all(others == -1.0)
    np.testing.assert_allclose(sc.fishing.ForcedEffort[::12, 1], EFFORT, atol=5e-4)

    out = rsim_run(sc, method="AB")
    assert not out.crashed_groups
    # the forced group follows its series year by year
    np.testing.assert_allclose(out.annual_Biomass[1:, rg], RG_FORCED[1:], rtol=0.05)
