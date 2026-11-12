"""EwE6 migration: Immigration, Emigration and EmigRate (deviation A1-s2).

Ecopath (cEcoPathModel.EstimEEAgain, vb 456-490) balances a group's
production against predation, catch, BA and net migration: EE = (M2 + catch
+ BA + Emigration + EmigRate x B - Immigration) / (B x PB), where Emigration
and Immigration are flows (t/km2/yr) and EmigRate is emigration per unit of
biomass, used when Emigration is 0. Ecosim (SetupSimVariables, vb 4528;
Derivt, vb 3081 and 3111) adds Immigration to a living group's derivative and
takes Emig x B, Emig = Emigration / start B, as a loss. It does not use
EmigRate for living groups, so a group whose emigration is entered as a rate
is balanced in Ecopath but grows at EmigRate x B from the start of Ecosim
(reproduced on purpose, deviations.md R4).
"""

import warnings
from pathlib import Path

import numpy as np
import pytest

PHYTO, ZOO, FISH, DET = 1, 2, 3, 4  # rsim indices (0 = Outside)
DB = Path(__file__).parents[3] / "Data" / "LT2022_migration.eweaccdb"


def _params(**migration):
    """Phyto -> Zoo -> Fish. ``migration``: column -> {group: value}."""
    from pypath.core.params import create_rpath_params

    params = create_rpath_params(["Phyto", "Zoo", "Fish", "Det"], [1, 0, 0, 2])
    m = params.model
    m["Biomass"] = [100.0, 20.0, 5.0, 10.0]
    m["PB"] = [10.0, 2.0, 1.0, np.nan]
    m["QB"] = [np.nan, 20.0, 4.0, np.nan]
    m["EE"] = np.nan
    m["BioAcc"] = 0.0
    m["Unassim"] = [0.0, 0.2, 0.2, 0.0]
    m["Det"] = [1.0, 1.0, 1.0, 0.0]
    params.diet["Zoo"] = [1.0, 0.0, 0.0, 0.0, 0.0]
    params.diet["Fish"] = [0.0, 1.0, 0.0, 0.0, 0.0]
    for col, values in migration.items():
        m[col] = 0.0
        for group, value in values.items():
            m.loc[m["Group"] == group, col] = value
    return params


def _balance(params):
    from pypath.core.ecopath import rpath

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return rpath(params)


def test_ecopath_balances_net_migration():
    """Fish (no predators): EE = 1 / (5 x 1). Zoo: (20 eaten - 4 in) / 40.
    Phyto: (400 eaten + 0.1 x 100 out) / 1000."""
    model = _balance(
        _params(
            Emigration={"Fish": 1.0},
            Immigration={"Zoo": 4.0},
            EmigRate={"Phyto": 0.1},
        )
    )
    np.testing.assert_allclose(model.EE[:3], [0.41, 0.4, 0.2], rtol=1e-12)
    np.testing.assert_allclose(model.Immigration[:3], [0.0, 4.0, 0.0])
    np.testing.assert_allclose(model.Emigration[:3], [0.0, 0.0, 1.0])
    np.testing.assert_allclose(model.EmigRate[:3], [0.1, 0.0, 0.0])


def test_emigrate_counts_when_biomass_is_left_to_ecopath():
    """Zoo with EE 0.5 and B unknown: 2 x 0.5 x B = 20 + 0.2 x B, so B = 25."""
    params = _params(EmigRate={"Zoo": 0.2})
    params.model.loc[params.model["Group"] == "Zoo", ["Biomass", "EE"]] = [np.nan, 0.5]
    model = _balance(params)
    assert model.Biomass[ZOO - 1] == pytest.approx(25.0, rel=1e-12)


def test_models_without_migration_are_unchanged():
    plain = _balance(_params())
    zero = _balance(_params(Immigration={}, Emigration={}, EmigRate={}))
    np.testing.assert_array_equal(plain.EE, zero.EE)


def _run(params, years=2):
    from pypath.core.ecosim import rsim_run, rsim_scenario

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        sc = rsim_scenario(_balance(params), params, years=range(1, years + 1))
        # isolates migration; EwE's handling time (D5) has its own tests
        sc.params.EWE_HANDLING_TIME = False
        return rsim_run(sc, method="RK4")


def test_ecosim_keeps_a_migrating_model_at_its_base():
    """Immigration enters as a flow, emigration as Emigration / B0 x B: the
    Ecopath balance, migration included, holds in Ecosim."""
    out = _run(_params(Emigration={"Fish": 1.0}, Immigration={"Zoo": 4.0}))
    np.testing.assert_allclose(out.out_Biomass[-1], out.out_Biomass[0], rtol=1e-9)


def test_ecosim_ignores_emigrate_on_living_groups():
    """EwE's Ecosim sets Emig from Emigration only (vb 4528), so Fish, which
    emigrates 0.2 x B in Ecopath, grows at 0.2 x 5 = 1 a year at the start."""
    out = _run(_params(EmigRate={"Fish": 0.2}))
    step = out.out_Biomass[1, FISH] - out.out_Biomass[0, FISH]
    assert step == pytest.approx(1.0 / 12, rel=2e-2)


def test_the_fast_group_test_counts_emigration():
    """EwE's fast-group test reads the first Derivt's loss, which holds Emig x
    B (SetTimeSteps, vb 2777-2797; Derivt, vb 3081). Zoo (B 2, PB 24.5)
    emigrates 2: Ecopath lowers its other mortality by 2, so its loss is
    still its production, 24.5 a year > 24, and Zoo is solved at its
    equilibrium. Without Emig x B the rate would be 47 / 2 = 23.5."""
    from pypath.core.ecosim import rsim_scenario

    params = _params(Emigration={"Zoo": 2.0})
    m = params.model
    m.loc[m["Group"] == "Zoo", ["Biomass", "PB", "QB"]] = [2.0, 24.5, 100.0]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        sc = rsim_scenario(_balance(params), params, years=range(1, 3))
    assert sc.params.Emig[ZOO] == pytest.approx(1.0)
    assert sc.params.NoIntegrate[ZOO] == 1


# --- EwE 6.7.144 oracle: LT2022 with ewe-oracle patches/lt2022_migration.sql ---
# Herring Emigration 0.5, Sprat Immigration 0.3, Smelt EmigRate 0.05; runs
# 43221485317 (scenario 16, dataset 35). Ecopath EE (ecopath.csv) and biomass
# at the end of months 1, 12, 60 and 144 (ecosim_monthly.csv). Without
# migration the EE are 0.678799, 0.597951 and 0.779837.

EWE_EE = {"Herring": 0.783371, "Sprat": 0.408933, "Smelt": 0.870417}
EWE_BIOMASS = {
    "Smelt": {1: 4.7194128, 12: 4.90708351, 60: 3.42434382, 144: 1.63339484},
    "Herring": {1: 6.13015127, 12: 6.12504196, 60: 4.14777374, 144: 1.20464337},
    "Sprat": {1: 1.27996898, 12: 1.28199673, 60: 1.00676763, 144: 0.200379238},
    "Cod": {1: 0.674025238, 12: 0.676401317, 60: 0.44879809, 144: 0.481496036},
}


def _db():
    if not DB.exists():
        pytest.skip(f"{DB.name} not found")
    return str(DB)


@pytest.mark.integration
def test_reader_reads_migration(caplog):
    import logging

    from pypath.io.ewemdb import read_ewemdb

    with caplog.at_level(logging.WARNING, logger="pypath.io.ewemdb"):
        params = read_ewemdb(_db())
    assert "Immigration" not in caplog.text and "Emig" not in caplog.text
    m = params.model.set_index(params.model["Group"].astype(str).str.strip())
    assert m.loc["Herring", "Emigration"] == pytest.approx(0.5)
    assert m.loc["Sprat", "Immigration"] == pytest.approx(0.3)
    assert m.loc["Smelt", "EmigRate"] == pytest.approx(0.05)


@pytest.mark.integration
def test_lt2022_ecopath_follows_the_ewe_oracle():
    from pypath.io.ewemdb import read_ewemdb

    model = _balance(read_ewemdb(_db()))
    names = [str(g).strip() for g in model.Group]
    for group, ee in EWE_EE.items():
        assert model.EE[names.index(group)] == pytest.approx(ee, rel=1e-5), group


@pytest.mark.integration
def test_lt2022_ecosim_follows_the_ewe_oracle():
    from pypath.core.ecosim import rsim_run
    from pypath.io.ewemdb import ecosim_scenario_from_ewemdb

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        sc = ecosim_scenario_from_ewemdb(_db(), scenario=16, timeseries_dataset=35)
        out = rsim_run(sc, method="RK4")
    names = [str(n).strip() for n in sc.params.spname]
    for group, months in EWE_BIOMASS.items():
        j = names.index(group)
        for month, ewe in months.items():
            assert out.out_Biomass[month, j] == pytest.approx(ewe, rel=2e-3), (
                group,
                month,
            )


def test_ewe_writers_write_migration(tmp_path):
    import zipfile

    import pandas as pd

    from pypath.io.ewe_writer import write_ewemdb

    params = _params(
        Emigration={"Fish": 1.0}, Immigration={"Zoo": 4.0}, EmigRate={"Phyto": 0.1}
    )
    out = tmp_path / "migration.ewecsv.zip"
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        write_ewemdb(params, str(out), backend="csv")
    with zipfile.ZipFile(out) as zf:
        groups = pd.read_csv(zf.open("EcopathGroup.csv")).set_index("GroupName")
    assert groups.loc["Fish", "Emigration"] == pytest.approx(1.0)
    assert groups.loc["Zoo", "Immigration"] == pytest.approx(4.0)
    assert groups.loc["Phyto", "EmigRate"] == pytest.approx(0.1)


# --- EwE 6.7.144 oracle: LT2022 with patches/lt2022_fast_group_emigration.sql --
# Zooplankton (B 3.676, PB 25) emigrates 5: EE 0.836815, and its first loss
# is still 25 x B, so EwE keeps it fast (ecosim_groups_setup.csv NoIntegrate
# 0); run 43284655114 (scenario 16, dataset 35). Without Emig x B in the test
# PyPath integrated it: 0.52 % median deviation, 54 % at worst (Zooplankton at
# month 128).
FAST_DB = Path(__file__).parents[3] / "Data" / "LT2022_fast_group_emigration.eweaccdb"
EWE_FAST_BIOMASS = {
    "Zooplankton": {12: 3.67555237, 60: 2.10694313, 128: 1.04804504, 144: 1.26158595},
    "Mobile crustaceans": {
        12: 15.737113,
        60: 11.631711,
        128: 5.96996832,
        144: 1.01074135,
    },
    "Herring": {12: 6.13273621, 60: 4.22718143, 128: 3.88430452, 144: 1.1590488},
}


@pytest.mark.integration
def test_lt2022_emigrating_fast_group_follows_the_ewe_oracle():
    from pypath.core.ecosim import rsim_run
    from pypath.io.ewemdb import ecosim_scenario_from_ewemdb

    if not FAST_DB.exists():
        pytest.skip(f"{FAST_DB.name} not found")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        sc = ecosim_scenario_from_ewemdb(
            str(FAST_DB), scenario=16, timeseries_dataset=35
        )
        out = rsim_run(sc, method="RK4")
    names = [str(n).strip() for n in sc.params.spname]
    assert sc.params.NoIntegrate[names.index("Zooplankton")] == 1
    for group, months in EWE_FAST_BIOMASS.items():
        j = names.index(group)
        for month, ewe in months.items():
            assert out.out_Biomass[month, j] == pytest.approx(ewe, rel=2e-3), (
                group,
                month,
            )
