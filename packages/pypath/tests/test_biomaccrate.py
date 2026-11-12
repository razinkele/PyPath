"""EwE6's biomass accumulation rate, EcopathGroup.BiomAccRate (deviation A1-rate).

Where BiomAcc is 0, EwE counts BiomAccRate x B as the group's biomass
accumulation, like EmigRate (cEcoPathModel.vb:466, 2047, 2097): on the
right-hand side of the balance when B is known, in the denominator when B
is solved, B = (catch + BA + Emigration - Immigration + M2) / (PB x EE -
QB x DC(j, j) - BiomAccRate - EmigRate). A multi-stanza group's EE leaves an
absolute BiomAcc out but keeps BiomAccRate x B (vb 486-489, 2052-2056).
"""

import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

PHYTO, ZOO, FISH, DET = 1, 2, 3, 4  # rsim indices (0 = Outside)
DB = Path(__file__).parents[3] / "Data" / "LT2022_biomaccrate.eweaccdb"


def _params(rates):
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
    m["BioAccRate"] = 0.0
    for group, rate in rates.items():
        m.loc[m["Group"] == group, "BioAccRate"] = rate
    params.diet["Zoo"] = [1.0, 0.0, 0.0, 0.0, 0.0]
    params.diet["Fish"] = [0.0, 1.0, 0.0, 0.0, 0.0]
    return params


def _balance(params):
    from pypath.core.ecopath import rpath

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return rpath(params)


def test_rate_with_biomass_left_to_ecopath():
    """Zoo with EE 0.5 and B unknown: 2 x 0.5 x B = 20 + 0.2 x B, so B = 25
    and BA = 0.2 x 25 = 5."""
    params = _params({"Zoo": 0.2})
    params.model.loc[params.model["Group"] == "Zoo", ["Biomass", "EE"]] = [np.nan, 0.5]
    model = _balance(params)
    assert model.Biomass[ZOO - 1] == pytest.approx(25.0, rel=1e-12)
    assert model.BA[ZOO - 1] == pytest.approx(5.0, rel=1e-12)


def test_rate_with_a_known_biomass():
    """Fish (no predators, B 5): EE = 0.1 x 5 / (5 x 1); BA = 0.5."""
    model = _balance(_params({"Fish": 0.1}))
    assert model.EE[FISH - 1] == pytest.approx(0.1, rel=1e-12)
    assert model.BA[FISH - 1] == pytest.approx(0.5, rel=1e-12)


def test_an_absolute_ba_wins_over_the_rate():
    params = _params({"Fish": 0.1})
    params.model.loc[params.model["Group"] == "Fish", "BioAcc"] = 0.25
    model = _balance(params)
    assert model.EE[FISH - 1] == pytest.approx(0.25 / 5, rel=1e-12)


def _groups(**cols):
    return pd.DataFrame(cols)


def test_reader_keeps_the_rate_where_biomacc_is_zero():
    from pypath.io.ewemdb import _resolve_bioacc_rate

    df = _groups(
        GroupName=["Cod", "Herring", "Sprat", "Detritus"],
        Biomass=[-9999.0, 6.0, 1.0, 2.0],
        BiomAcc=[0.0, 0.0, 0.3, 0.0],
        BiomAccRate=[0.05, -0.03, 0.1, 0.2],
    )
    rate = _resolve_bioacc_rate(df, [0, 0, 0, 2])
    # B unknown or known alike; an absolute BA wins; detritus BA is not modelled
    np.testing.assert_allclose(rate, [0.05, -0.03, 0.0, 0.0])


# --- EwE 6.7.144 oracle: LT2022 with ewe-oracle patches/lt2022_biomaccrate.sql --
# Flounder: B left to Ecopath, EE 0.6, BiomAccRate 0.05. Sand-eels: B known,
# BiomAccRate -0.03. Blue mussel (ad.), a multi-stanza stage: BiomAccRate 0.02.
# Run 43235479966 (scenario 16, dataset 35). EwE ignores the stanza stage's
# rate: its EE falls with its predators' change, it does not rise.

EWE_ECOPATH = {  # group: (B, EE)
    "Flounder": (1.2257154, 0.600000024),
    "Sand-eels": (0.828999996, 0.470788717),
    "Blue mussel (ad.)": (27.0, 0.0163165424),
}
EWE_BIOMASS = {
    "Flounder": {1: 1.23077536, 12: 1.28187919, 60: 1.00122666, 144: 0.717459202},
    "Sand-eels": {1: 0.826970577, 12: 0.810664713, 60: 0.413792312, 144: 0.0897996724},
    "Blue mussel (ad.)": {
        1: 27.0503502,
        12: 27.0513515,
        60: 16.7445374,
        144: 0.162030101,
    },
    "Cod": {1: 0.674010098, 12: 0.674016714, 60: 0.401900053, 144: 0.452364832},
}


def _db():
    if not DB.exists():
        pytest.skip(f"{DB.name} not found")
    return str(DB)


@pytest.mark.integration
def test_lt2022_ecopath_follows_the_ewe_oracle():
    from pypath.io.ewemdb import read_ewemdb

    model = _balance(read_ewemdb(_db()))
    names = [str(g).strip() for g in model.Group]
    for group, (b, ee) in EWE_ECOPATH.items():
        j = names.index(group)
        assert model.Biomass[j] == pytest.approx(b, rel=1e-5), group
        assert model.EE[j] == pytest.approx(ee, rel=1e-5), group
    # EwE's BioAccum: 0.05 x the solved B
    assert model.BA[names.index("Flounder")] == pytest.approx(0.0612857714, rel=1e-5)


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
