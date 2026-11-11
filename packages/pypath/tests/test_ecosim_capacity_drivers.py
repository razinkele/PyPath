"""EwE6 environmental response functions in Ecosim ("capacity drivers").

``EcosimScenarioCapacityDrivers(ScenarioID, GroupID, DriverID, ResponseID,
Target)`` ties a group to a driver time series (an ``EcosimShapeTime`` shape)
through a response curve (an ``EcosimShapeMediation`` shape: N points over
[XAxisMin, XAxisMax]). Every month EwE reads the curve at the driver's value
(cMediationDataStructures.getEnviroResponse) and, for Target 115 (the
default), multiplies the group's search rate on all its prey, or a producer's
production, by it (ApplyEnvironmentalResponse in ApplyAVmodifiers,
cEcoSimModel.vb 4931, 2895, 3033). For Target 126 the responses are
proportions of mortality: other mortality becomes mo - 12 ln(1 - p)
(ApplyMortalityResponse, vb 5016-5050).

LT2022's scenario 7 drives the juvenile blue mussels by 'sand' (0 in year 1,
then 1e5 and more) through a response that is 1.0035 at 0 and 0.3345 at the
top of its 0-1 axis: EwE's juveniles crash in year 2. PyPath read nothing of
it, and scenario 7 was 264 % off the EwE oracle.
"""

import logging
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from pypath.core.ecosim import rsim_scenario

from .test_forcing_channels import _simple_model

PHYTO, ZOO, DET = 1, 2, 3  # rsim indices (0 = Outside)
LT2022 = Path(__file__).parents[3] / "Data" / "LT2022_0.5ST_final7.eweaccdb"


@pytest.mark.parametrize(
    "x, expected",
    [
        (-1.0, 10.0),  # at or below XAxisMin: the first point
        (0.0, 10.0),
        (0.24, 10.0),  # 1 + trunc(0.24 * 4 / (1 + 1e-5)) = 1
        (0.26, 20.0),  # 1 + trunc(1.03999) = 2
        (0.5, 20.0),  # 1 + trunc(1.99998) = 2: the point below, not 3
        (0.99, 40.0),
        (1.0, 40.0),  # at or above XAxisMax: the last point
        (1.0e5, 40.0),
        (-9999.0, 1.0),  # cCore.NULL_VALUE: no response
    ],
)
def test_response_is_read_as_ewe_reads_it(x, expected):
    from pypath.io.ewemdb import _enviro_response

    points = np.array([10.0, 20.0, 30.0, 40.0])
    assert _enviro_response(points, 0.0, 1.0, x) == expected


def _tables(rows, driver=(0.0, 0.5, 2.0), response=(1.2, 1.0, 0.5, 0.1)):
    """EwE tables for one scenario (ID 7): groups 901-903 are Phyto, Zoo, Det."""
    zs = " ".join(str(v) for v in driver)
    return {
        "EcosimScenarioCapacityDrivers": pd.DataFrame(rows),
        "EcosimShapeTime": pd.DataFrame(
            {"ShapeID": [96], "zScale": [zs], "Title": ["driver"]}
        ),
        "EcosimShapeMediation": pd.DataFrame(
            {
                "ShapeID": [280],
                "zScale": [" ".join(str(v) for v in response)],
                "XAxisMin": [0.0],
                "XAxisMax": [1.0],
            }
        ),
    }


def _row(group, target=115, scenario=7):
    return {
        "ScenarioID": scenario,
        "GroupID": group,
        "DriverID": 96,
        "ResponseID": 280,
        "Target": target,
    }


def _apply(monkeypatch, tables, scenario_years=range(1, 3)):
    import pypath.io.ewemdb as ew

    model, params = _simple_model()
    sc = rsim_scenario(model, params, years=scenario_years)
    monkeypatch.setattr(
        ew, "_try_read_table_variants", lambda fp, names: tables.get(names[0])
    )
    selected = {
        "id": 7,
        "scenario_group_df": pd.DataFrame(
            {
                "ScenarioID": [7, 7, 7],
                "EcopathGroupID": [11, 12, 13],
                "GroupID": [901, 902, 903],
            }
        ),
        "group_id_to_name": {11: "Phyto", 12: "Zoo", 13: "Det"},
    }
    ew._apply_capacity_drivers("unused.eweaccdb", selected, sc, params)
    return sc


def test_search_rate_response_scales_a_consumers_search(monkeypatch):
    """Driver 0, 0.5, 2 (then its last value) through the curve 1.2 / 1.0 /
    0.5 / 0.1 gives 1.2, 1.0 (point 1 + trunc(1.99998) = 2), 0.1, 0.1, ..."""
    sc = _apply(monkeypatch, _tables([_row(902)]))
    expected = np.full(24, 0.1)
    expected[:2] = [1.2, 1.0]
    np.testing.assert_allclose(sc.forcing.ForcedSearch[:, ZOO], expected)
    np.testing.assert_allclose(sc.forcing.ForcedSearch[:, PHYTO], 1.0)
    np.testing.assert_allclose(sc.forcing.ForcedMort, 1.0)


def test_unset_target_is_the_search_rate_response(monkeypatch):
    row = _row(902)
    del row["Target"]
    sc = _apply(monkeypatch, _tables([row]))
    assert sc.forcing.ForcedSearch[0, ZOO] == pytest.approx(1.2)


def test_search_rate_response_on_a_producer_scales_its_production(monkeypatch):
    sc = _apply(monkeypatch, _tables([_row(901)]))
    assert sc.forcing.ForcedSearch[0, PHYTO] == pytest.approx(1.2)
    assert sc.forcing.ForcedSearch[2, PHYTO] == pytest.approx(0.1)
    np.testing.assert_allclose(sc.forcing.ForcedSearch[:, ZOO], 1.0)


def test_several_search_rate_drivers_multiply(monkeypatch):
    sc = _apply(monkeypatch, _tables([_row(902), _row(902)]))
    assert sc.forcing.ForcedSearch[0, ZOO] == pytest.approx(1.44)


def test_rows_of_other_scenarios_are_ignored(monkeypatch):
    sc = _apply(monkeypatch, _tables([_row(902, scenario=3)]))
    np.testing.assert_allclose(sc.forcing.ForcedSearch, 1.0)


def test_mortality_response_raises_other_mortality(monkeypatch):
    """Target 126: p = response (summed over drivers, capped at 0.9999), and
    other mortality mo -> mo - 12 ln(1 - p), i.e. ForcedMort = that / mo."""
    tables = _tables(
        [_row(902, target=126), _row(902, target=126)],
        response=(0.01, 0.02, 0.6, 0.7),
    )
    sc = _apply(monkeypatch, tables)
    mo = float(sc.params.MzeroMort[ZOO])
    assert mo > 0
    for month, p in ((0, 0.02), (1, 0.04), (2, 0.9999)):  # 2 x 0.7 is capped
        expected = (mo - 12.0 * np.log(1.0 - p)) / mo
        assert sc.forcing.ForcedMort[month, ZOO] == pytest.approx(expected)
    np.testing.assert_allclose(sc.forcing.ForcedSearch, 1.0)


def test_detritus_driver_is_reported_and_skipped(monkeypatch, caplog):
    with caplog.at_level(logging.WARNING, logger="pypath.io.ewemdb"):
        sc = _apply(monkeypatch, _tables([_row(903)]))
    np.testing.assert_allclose(sc.forcing.ForcedSearch, 1.0)
    assert any("Det" in r.getMessage() for r in caplog.records)


def _lt2022_s7():
    if not LT2022.exists():
        pytest.skip("LT2022 database not found")
    from pypath.io.ewemdb import ecosim_scenario_from_ewemdb

    return ecosim_scenario_from_ewemdb(str(LT2022), scenario=7, timeseries_dataset=35)


@pytest.mark.integration
def test_lt2022_scenario_7_drives_the_juvenile_mussels():
    """'sand' is 0 in year 1 and above the 0-1 axis afterwards: the juveniles'
    search rate is x 1.003515 in year 1 and x 0.334501 from month 13."""
    sc = _lt2022_s7()
    names = [str(n).strip() for n in sc.params.spname]
    juv = names.index("Blue mussel (juv.)")
    fs = sc.forcing.ForcedSearch
    np.testing.assert_allclose(fs[:12, juv], 1.003515, rtol=1e-6)
    np.testing.assert_allclose(fs[12:24, juv], 0.334501, rtol=1e-6)
    others = [i for i in range(1, fs.shape[1]) if i != juv]
    np.testing.assert_allclose(fs[:12, others], 1.0)


# EwE 6.7.144 oracle, LT2022 scenario 7 + dataset 35 (run 42895390488): the
# juvenile blue mussels at the end of months 1 and 13. Single precision.
EWE_ORACLE_S7_JUV = {1: 15.028562, 13: 10.4305}


@pytest.mark.integration
def test_lt2022_scenario_7_juvenile_mussels_follow_the_ewe_oracle():
    from pypath.core.ecosim import rsim_run

    sc = _lt2022_s7()
    names = [str(n).strip() for n in sc.params.spname]
    juv = names.index("Blue mussel (juv.)")
    out = rsim_run(sc, method="RK4", years=range(1, 3))
    for month, ewe in EWE_ORACLE_S7_JUV.items():
        assert out.out_Biomass[month, juv] == pytest.approx(ewe, rel=1e-3), month
