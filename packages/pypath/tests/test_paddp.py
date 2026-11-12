"""EwE6's non-additive predation mortality, AdditivePredMort (PaddP).

EcosimScenarioGroup.AdditivePredMort is the share of predation mortality that
adds to other mortality (default 1). Below 1, other mortality compensates:
EwE sets moMax = mo + M2_0 x (1 - PaddP) and Qh = moMax / mo - 1 from the base
predation rate M2_0 = StartEatenOf / B0 (CalcBaseAdditiveMort, vb 3802-3816),
and each derivative uses moTot = moMax / (1 + Qh x Eatenof / StartEatenOf)
in place of mo (Derivt, vb 3058-3081), so moTot = mo at the base. The flow
to detritus keeps mo (SimDetritusMT, vb 3165), as for MoPred (R2).
"""

import warnings
from pathlib import Path

import numpy as np
import pytest

PHYTO, ZOO, FISH, DET = 1, 2, 3, 4  # rsim indices (0 = Outside)
DB = Path(__file__).parents[3] / "Data" / "LT2022_s16_paddp.eweaccdb"


def _scenario(paddp_zoo=1.0):
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
    # isolates PaddP; EwE's handling time (D5) has its own tests
    sc.params.EWE_HANDLING_TIME = False
    sc.params.PaddP[ZOO] = paddp_zoo
    return sc


def test_compensation_setup_is_ewes():
    """Zoo: eaten 20 by Fish, B0 20, so M2_0 = 1; mo = 2 x (1 - 0.5) = 1.
    PaddP 0: moMax = 1 + 1 = 2, Qh = 2 / 1 - 1 = 1."""
    from pypath.core.ecosim import _mo_tot_setup

    sc = _scenario(paddp_zoo=0.0)
    mo_max, qh, start_eaten_of = _mo_tot_setup(sc.params)
    assert start_eaten_of[ZOO] == pytest.approx(20.0)
    assert mo_max[ZOO] == pytest.approx(2.0)
    assert qh[ZOO] == pytest.approx(1.0)
    assert qh[FISH] == 0.0  # Fish has PaddP 1


def test_other_mortality_compensates():
    from pypath.core.ecosim_deriv import _mo_tot_factor

    mo = np.array([0.0, 1.0, 1.0, 0.5])
    mo_max = np.array([0.0, 2.0, 1.0, 1.0])
    qh = np.array([0.0, 1.0, 0.0, 1.0])
    start = np.array([0.0, 20.0, 5.0, 4.0])
    eaten = np.array([0.0, 40.0, 10.0, 4.0])  # Zoo eaten twice the base
    # Zoo: 2 / (1 + 1 x 2) = 2/3 of... over mo 1; index 2 PaddP 1; index 3 at base
    np.testing.assert_allclose(
        _mo_tot_factor(eaten, start, mo, mo_max, qh), [1.0, 2.0 / 3.0, 1.0, 1.0]
    )


def _run(sc):
    from pypath.core.ecosim import rsim_run

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return rsim_run(sc, method="RK4")


def test_the_base_state_is_kept():
    out = _run(_scenario(paddp_zoo=0.0))
    np.testing.assert_allclose(out.out_Biomass[-1], out.out_Biomass[0], rtol=1e-9)


def test_more_predation_kills_less_through_other_mortality():
    """Fish held at 1.5 x B0: with PaddP 0 Zoo's other mortality falls as the
    predation rises, so Zoo ends higher than with additive mortality."""
    runs = {}
    for paddp in (0.0, 1.0):
        sc = _scenario(paddp_zoo=paddp)
        sc.forcing.ForcedBio[:, FISH] = 1.5 * sc.params.B_BaseRef[FISH]
        runs[paddp] = _run(sc)
    assert runs[0.0].out_Biomass[-1, ZOO] > 1.01 * runs[1.0].out_Biomass[-1, ZOO]


def test_ecospace_keeps_additive_mortality():
    """EwE's Ecospace takes mo, not moTot (cSpaceSolver.vb 1009, 2861): PaddP
    acts in Ecosim only, so the spatial run is the same with PaddP 0 or 1."""
    from pypath.spatial import EcospaceParams, create_1d_grid, rsim_run_spatial

    runs = {}
    for paddp in (0.0, 1.0):
        sc = _scenario(paddp_zoo=paddp)
        sc.forcing.ForcedPrey[12:, PHYTO] = 0.5  # disturb the base state
        n = sc.params.NUM_GROUPS + 1
        eco = EcospaceParams(
            grid=create_1d_grid(n_patches=2, spacing=1.0),
            habitat_preference=np.ones((n, 2)),
            habitat_capacity=np.ones((n, 2)),
            dispersal_rate=np.zeros(n),
            advection_enabled=np.zeros(n, dtype=bool),
            gravity_strength=np.zeros(n),
        )
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            runs[paddp] = rsim_run_spatial(sc, ecospace=eco, years=range(1, 3))
    # the disturbance moves Zoo's predation away from its base
    assert not np.allclose(runs[1.0].out_Biomass[-1], runs[1.0].out_Biomass[0])
    np.testing.assert_allclose(
        runs[0.0].out_Biomass, runs[1.0].out_Biomass, rtol=1e-12, atol=1e-14
    )


# --- EwE 6.7.144 oracle: LT2022 with ewe-oracle patches/lt2022_s16_paddp.sql ---
# Scenario 16 with Sprat PaddP 0, Zooplankton 0.5 and Herring 0.3; run
# 43238238339 (dataset 35): biomass at the end of months 1, 12, 60 and 144.
# With additive mortality Sprat is 0.815 at month 60 and 0.109 at 144.
EWE_BIOMASS = {
    "Sprat": {1: 1.27997541, 12: 1.28008246, 60: 0.200597391, 144: 0.00382067752},
    "Zooplankton": {1: 3.6760087, 12: 3.6754632, 60: 2.0807519, 144: 0.0696699992},
    "Herring": {1: 6.1302352, 12: 6.13256264, 60: 2.74217439, 144: 0.403413892},
    "Cod": {1: 0.674011111, 12: 0.674158514, 60: 0.291892201, 144: 0.345766008},
}


@pytest.mark.integration
def test_lt2022_reads_paddp_and_follows_the_ewe_oracle(caplog):
    import logging

    from pypath.core.ecosim import rsim_run
    from pypath.io.ewemdb import ecosim_scenario_from_ewemdb

    if not DB.exists():
        pytest.skip(f"{DB.name} not found")
    with (
        caplog.at_level(logging.WARNING, logger="pypath.io.ewemdb"),
        warnings.catch_warnings(),
    ):
        warnings.simplefilter("ignore")
        sc = ecosim_scenario_from_ewemdb(str(DB), scenario=16, timeseries_dataset=35)
    assert "AdditivePredMort" not in caplog.text
    names = [str(n).strip() for n in sc.params.spname]
    assert sc.params.PaddP[names.index("Sprat")] == 0.0
    assert sc.params.PaddP[names.index("Zooplankton")] == pytest.approx(0.5)
    assert sc.params.PaddP[names.index("Herring")] == pytest.approx(0.3)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        out = rsim_run(sc, method="RK4")
    for group, months in EWE_BIOMASS.items():
        j = names.index(group)
        for month, ewe in months.items():
            # By month 144 Sprat has collapsed to 0.3 % of B0 and small
            # differences grow (0.02 % at month 12, about 0.5 % at 144)
            rel = 1e-2 if month == 144 else 2e-3
            assert out.out_Biomass[month, j] == pytest.approx(ewe, rel=rel), (
                group,
                month,
            )
