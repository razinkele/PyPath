"""EwE6's nutrient limitation of primary production (EcosimScenario.
NutBaseFreeProp, NutForcingShapeID).

EwE6 keeps a total nutrient pool fixed at the start, NutTot = sum(B0) /
(1 - NutBaseFreeProp) over every group, and frees what the biomass does not
hold: NutFree = NutTot x shape(t) - sum(B), at least 0.00101 x NutFree0
(InitialState and Derivt, cEcoSimModel.vb 3248-3280, 2862-2870). A
producer's production is scaled by 2 x NutFree / (NutFree + NutFree0)
(vb 3049), 1 at the base. EwE's default NutBaseFreeProp 0.9999 keeps the
factor within 1e-5 of 1; the Black Sea test case sets 0.9 and a seasonal
nutrient shape (0.7-1.15), which cuts its phytoplankton's production by 9 %
in month 1.
"""

import warnings
from pathlib import Path

import numpy as np
import pytest

PHYTO = 1
BLACKSEA = Path(__file__).parents[3] / "Data" / "BlackSea_EwEF_testcase.eweaccdb"


def test_nutrient_factor_is_ewes_expression():
    from pypath.core.ecosim_deriv import _ewe_nutrient_factor

    b0 = 120.0
    # at the base, with the shape at 1: no limitation
    assert _ewe_nutrient_factor(b0, b0, 0.9, 1.0) == pytest.approx(1.0, rel=1e-14)
    # Black Sea, month 1: NutTot = 10 B0, NutFree0 = 9 B0, shape 0.85
    assert _ewe_nutrient_factor(b0, b0, 0.9, 0.85) == pytest.approx(
        2 * 7.5 / (7.5 + 9.0), rel=1e-14
    )
    # more biomass binds more nutrient
    tot, free0 = b0 / 0.1, b0 / 0.1 - b0
    free = tot - 1.5 * b0
    assert _ewe_nutrient_factor(1.5 * b0, b0, 0.9, 1.0) == pytest.approx(
        2 * free / (free + free0), rel=1e-14
    )
    # NutFree never falls below NutMin = 0.00101 NutFree0
    assert _ewe_nutrient_factor(b0, b0, 0.9, 0.01) == pytest.approx(
        2 * 0.00101 / 1.00101, rel=1e-12
    )
    # NutBaseFreeProp below 0.1 counts as 0.1
    assert _ewe_nutrient_factor(1.5 * b0, b0, 0.0, 1.0) == pytest.approx(
        _ewe_nutrient_factor(1.5 * b0, b0, 0.1, 1.0), rel=1e-14
    )


def _scenario():
    from pypath.core.ecopath import rpath
    from pypath.core.ecosim import rsim_scenario
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
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        sc = rsim_scenario(rpath(params), params, years=range(1, 3))
    # isolates nutrient limitation; EwE's handling time has its own tests
    sc.params.EWE_HANDLING_TIME = False
    return sc


def _run(sc):
    from pypath.core.ecosim import rsim_run

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return rsim_run(sc, method="RK4")


def test_the_base_state_is_unlimited():
    sc = _scenario()
    sc.params.NutBaseFreeProp = 0.9
    out = _run(sc)
    np.testing.assert_allclose(out.out_Biomass[-1], out.out_Biomass[0], rtol=1e-12)


def test_a_low_nutrient_shape_cuts_primary_production():
    low = _scenario()
    low.params.NutBaseFreeProp = 0.9
    low.forcing.NutForcing = np.full(24, 0.85)
    base = _scenario()
    base.params.NutBaseFreeProp = 0.9
    a, b = _run(low), _run(base)
    assert a.out_Biomass[1, PHYTO] < b.out_Biomass[1, PHYTO]
    assert a.out_Biomass[1, PHYTO] < a.out_Biomass[0, PHYTO]


def test_the_rpath_rule_has_no_nutrient_limitation():
    from pypath.core.ecopath import rpath
    from pypath.core.ecosim import rsim_scenario
    from pypath.core.params import create_rpath_params

    params = create_rpath_params(["Phyto", "Det"], [1, 2])
    params.model["Biomass"] = [10.0, 5.0]
    params.model["PB"] = [10.0, np.nan]
    params.model["EE"] = [0.5, np.nan]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        sc = rsim_scenario(
            rpath(params), params, years=range(1, 3), fast_group_rule="rpath"
        )
    assert sc.params.NutBaseFreeProp is None


@pytest.mark.integration
def test_black_sea_nutrient_settings_are_read(caplog):
    import logging

    if not BLACKSEA.exists():
        pytest.skip("Black Sea database not found")
    from pypath.io.ewemdb import ecosim_scenario_from_ewemdb

    with (
        caplog.at_level(logging.WARNING, logger="pypath.io.ewemdb"),
        warnings.catch_warnings(),
    ):
        warnings.simplefilter("ignore")
        sc = ecosim_scenario_from_ewemdb(str(BLACKSEA), scenario=1)
    assert sc.params.NutBaseFreeProp == pytest.approx(0.9)
    np.testing.assert_allclose(sc.forcing.NutForcing[:4], [0.85, 1.0, 1.15, 0.7])
    assert len(sc.forcing.NutForcing) == 120
    assert "nutrient" not in caplog.text.lower()


# EwE 6.7.144 headless oracle, Black Sea scenario 1 (run 43103422428): biomass
# at the end of months 1-3, while the detritus self-fate difference (POM) has
# not yet spread. With nutrient limitation PyPath follows to 0.25 %; without
# it the phytoplankton is 6.3 % high in month 1 and 2.3 % low in month 3.
EWE_BLACKSEA = {
    "Phytoplankton": {1: 0.315152794, 2: 0.321531773, 3: 0.342840672},
    "Zooplankton": {1: 0.378464729, 2: 0.377369165, 3: 0.389126897},
    "Sprat": {1: 0.253572136, 2: 0.250788152, 3: 0.249577299},
}


@pytest.mark.integration
def test_black_sea_follows_the_ewe_oracle_with_nutrient_limitation():
    if not BLACKSEA.exists():
        pytest.skip("Black Sea database not found")
    from pypath.core.ecosim import rsim_run
    from pypath.io.ewemdb import ecosim_scenario_from_ewemdb

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        sc = ecosim_scenario_from_ewemdb(str(BLACKSEA), scenario=1)
        out = rsim_run(sc, method="RK4")
    names = [str(n).strip() for n in sc.params.spname]
    for group, months in EWE_BLACKSEA.items():
        j = names.index(group)
        for month, ewe in months.items():
            assert out.out_Biomass[month, j] == pytest.approx(ewe, rel=5e-3), (
                group,
                month,
            )
