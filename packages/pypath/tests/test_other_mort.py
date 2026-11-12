"""EwE6's other mortality input, EcopathGroup.OtherMort (deviation A1-s2).

OtherMort is the share of production lost to other mortality, 1 - EE. EwE
sets EE = 1 - OtherMort wherever OtherMort > 0, over an entered EE
(cEcopathDataStructures.CopyInputToModelArrays, vb 1026-1031); 0 and the
missing marker leave the entered EE.
"""

import numpy as np
import pytest


def test_other_mort_sets_ee():
    from pypath.io.ewemdb import _ee_with_other_mort

    ee = np.array([np.nan, 0.5, 0.4, np.nan, 0.6])
    other_mort = np.array([0.2, 0.3, 0.0, np.nan, -1.0])
    np.testing.assert_allclose(
        _ee_with_other_mort(ee, other_mort),
        [0.8, 0.7, 0.4, np.nan, 0.6],
        rtol=1e-12,
    )


# --- EwE 6.7.144 oracle: LT2022 with ewe-oracle patches/lt2022_othermort.sql ---
# Herring: B left to Ecopath, OtherMort 0.2. Cod: B left to Ecopath, EE
# entered 0.5 and OtherMort 0.3, so EE 0.7. Runs 43230721028 (scenario 16,
# dataset 35): Ecopath B and EE (ecopath.csv) and biomass at the end of months
# 1, 12, 60 and 144 (ecosim_monthly.csv).

import warnings  # noqa: E402
from pathlib import Path  # noqa: E402

DB = Path(__file__).parents[3] / "Data" / "LT2022_othermort.eweaccdb"
EWE_ECOPATH = {  # group: (B, EE)
    "Herring": (4.911343, 0.8),
    "Cod": (0.525598, 0.7),
    "Sprat": (1.279963, 0.530241),
    "Zooplankton": (3.676, 0.745928),
}
EWE_BIOMASS = {
    "Herring": {1: 4.91151762, 12: 4.91339254, 60: 3.35358, 144: 1.0166502},
    "Cod": {1: 0.525606692, 12: 0.525712192, 60: 0.391619205, 144: 0.406788349},
    "Sprat": {1: 1.2799716, 12: 1.28011584, 60: 0.75058198, 144: 0.0927047953},
    "Zooplankton": {1: 3.6759932, 12: 3.67553091, 60: 2.09429431, 144: 1.13368964},
}


def _db():
    if not DB.exists():
        pytest.skip(f"{DB.name} not found")
    return str(DB)


@pytest.mark.integration
def test_reader_sets_ee_from_other_mort(caplog):
    import logging

    from pypath.io.ewemdb import read_ewemdb

    with caplog.at_level(logging.WARNING, logger="pypath.io.ewemdb"):
        params = read_ewemdb(_db())
    assert "OtherMort" not in caplog.text
    m = params.model.set_index(params.model["Group"].astype(str).str.strip())
    assert m.loc["Herring", "EE"] == pytest.approx(0.8)
    assert m.loc["Cod", "EE"] == pytest.approx(0.7)  # over the entered 0.5
    assert np.isnan(m.loc["Herring", "Biomass"]) and np.isnan(m.loc["Cod", "Biomass"])


@pytest.mark.integration
def test_lt2022_ecopath_follows_the_ewe_oracle():
    from pypath.core.ecopath import rpath
    from pypath.io.ewemdb import read_ewemdb

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        model = rpath(read_ewemdb(_db()))
    names = [str(g).strip() for g in model.Group]
    for group, (b, ee) in EWE_ECOPATH.items():
        j = names.index(group)
        assert model.Biomass[j] == pytest.approx(b, rel=1e-5), group
        assert model.EE[j] == pytest.approx(ee, rel=1e-5), group


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
