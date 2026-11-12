"""The EwE 6 database of the EwE-F Black Sea test case.

``Data/BlackSea_EwEF_testcase.eweaccdb`` was built from the EwE-F input files
in ``tests/data/ewef/Black_Sea`` (Python4EwE-F, validation/ewe6_black_sea).
Read through ``read_ewemdb`` it must balance to EwE-F's Ecopath result, and
its Ecosim scenario must carry the same settings as the EwE-F files read by
``pypath.io.ewef``: two independent import paths for one model.
"""

import warnings
from pathlib import Path

import numpy as np
import pytest

from pypath.core.ecopath import rpath
from tests.test_ewef_import import ECOPATH_NML, ECOSIM_NML, EWEF_ECOPATH

DB_PATH = Path(__file__).parents[3] / "Data" / "BlackSea_EwEF_testcase.eweaccdb"

pytestmark = pytest.mark.integration


def _db_or_skip():
    if not DB_PATH.exists():
        pytest.skip("Black Sea database not found")
    from pypath.io.ewemdb import check_ewemdb_support

    if not any(check_ewemdb_support().values()):
        pytest.skip("No EwE database driver")
    return str(DB_PATH)


@pytest.fixture(scope="module")
def balanced():
    from pypath.io.ewemdb import read_ewemdb

    params = read_ewemdb(_db_or_skip())
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return rpath(params)


@pytest.mark.parametrize("group", list(EWEF_ECOPATH))
def test_database_balances_to_ewef(balanced, group):
    b, _, qb, ee = EWEF_ECOPATH[group]
    i = list(balanced.Group).index(group)
    assert balanced.Biomass[i] == pytest.approx(b, rel=1e-5)
    assert balanced.EE[i] == pytest.approx(ee, rel=1e-5, abs=1e-7)


@pytest.fixture(scope="module")
def scenarios():
    from pypath.io.ewef import ecosim_scenario_from_ewef
    from pypath.io.ewemdb import ecosim_scenario_from_ewemdb

    db = _db_or_skip()
    years = range(1, 11)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        from_db = ecosim_scenario_from_ewemdb(db, scenario=1, years=years)
        from_ewef = ecosim_scenario_from_ewef(ECOPATH_NML, ECOSIM_NML, years=years)
    return from_db, from_ewef


def _links(sc):
    p = sc.params
    names = [str(n).strip() for n in p.spname]
    return {
        (names[int(p.PreyFrom[k])], names[int(p.PreyTo[k])]): float(p.VV[k])
        for k in range(1, len(p.PreyFrom))
    }


def test_vulnerabilities_agree(scenarios):
    from_db, from_ewef = scenarios
    db_links, ewef_links = _links(from_db), _links(from_ewef)
    assert set(db_links) == set(ewef_links)
    for link, vv in ewef_links.items():
        assert db_links[link] == pytest.approx(vv, rel=1e-6), link


@pytest.mark.parametrize("attr", ["MoPred", "FtimeMax", "FtimeAdj", "SwitchPower"])
def test_group_settings_agree(scenarios, attr):
    from_db, from_ewef = scenarios
    db_names = [str(n).strip() for n in from_db.params.spname]
    ewef_names = [str(n).strip() for n in from_ewef.params.spname]
    db_vals = dict(zip(db_names, np.asarray(getattr(from_db.params, attr))))
    for name, value in zip(ewef_names, np.asarray(getattr(from_ewef.params, attr))):
        if name in db_vals and name in EWEF_ECOPATH:
            assert db_vals[name] == pytest.approx(value), (attr, name)


def test_diet_import_is_read():
    """EwE 6 keeps a consumer's imported share of its diet in
    EcopathGroup.ImpVar (cDBDataSource: DC(i, 0) = ImpVar); the reader
    ignored it, so Bonito, Bluefish and A. Mackerel lost 20 % of their diet."""
    from pypath.io.ewemdb import read_ewemdb

    params = read_ewemdb(_db_or_skip())
    diet = params.diet.set_index("Group")
    for group in ("Bonito", "Bluefish", "A. Mackerel"):
        assert diet.loc["Import", group] == pytest.approx(0.2)
    assert float(np.nan_to_num(diet.loc["Import", "Dolphins"])) == 0.0
