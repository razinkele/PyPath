"""Forcing shapes come from EcosimScenarioPredPreyShape, applied as EwE does.

EwE6 stores which shape drives which link in
``EcosimScenarioPredPreyShape(ScenarioID, PredID, PreyID, ShapeID, FunctionType)``
and casts ``FunctionType`` to ``eForcingFunctionApplication`` (1 SearchRate,
2 Vulnerability, 3 ArenaArea, 4 VulAndArea, 6 ProductionRate, 7 Import,
8 MortOther). A production-rate shape multiplies the producer's production and
nothing else; in PyPath that is the producer's search-rate forcing
(``ForcedSearch``), as in Rpath, not its availability to grazers (``ForcedPrey``).

LT2022 scenario 16 has one row: shape 316 ``PTOT_Baltic`` on Phytoplankton,
type 6.
"""

from pathlib import Path

import numpy as np
import pytest

DATA_DIR = Path(__file__).parents[3] / "Data"
DB_PATH = DATA_DIR / "LT2022_0.5ST_final7.eweaccdb"

pytestmark = pytest.mark.integration


@pytest.fixture(scope="module")
def lt2022_scenario():
    if not DB_PATH.exists():
        pytest.skip("LT2022 database not found")
    from pypath.io.ewemdb import EwEDatabaseError, ecosim_scenario_from_ewemdb

    try:
        return ecosim_scenario_from_ewemdb(str(DB_PATH), scenario=16)
    except EwEDatabaseError as e:
        pytest.skip(f"No EwE database driver: {e}")


def _shape_values(shape_id):
    from pypath.io.ewemdb import read_ewemdb_table

    st = read_ewemdb_table(str(DB_PATH), "EcosimShapeTime")
    row = st[st["ShapeID"] == shape_id].iloc[0]
    return np.array([float(v) for v in str(row["zScale"]).split()])


def test_production_shape_drives_the_producer_search_rate(lt2022_scenario):
    phyto = [str(n).strip() for n in lt2022_scenario.params.spname].index(
        "Phytoplankton"
    )
    values = _shape_values(316)
    forced = lt2022_scenario.forcing.ForcedSearch[:, phyto]
    n = min(len(values), len(forced))
    assert n >= 120
    np.testing.assert_allclose(forced[:n], values[:n], rtol=1e-6)


def test_production_shape_does_not_touch_prey_availability(lt2022_scenario):
    assert np.all(lt2022_scenario.forcing.ForcedPrey == 1.0)


def test_shapes_are_not_matched_by_title(lt2022_scenario):
    """Shape 71 'Phytoplankton_biom' exists in the database but is not
    attached to scenario 16; it must not be applied anywhere."""
    biom = _shape_values(71)
    forcing = lt2022_scenario.forcing
    for arr in (forcing.ForcedSearch, forcing.ForcedPrey):
        for col in range(arr.shape[1]):
            n = min(len(biom), arr.shape[0])
            assert not np.allclose(arr[:n, col], biom[:n])
