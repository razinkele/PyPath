"""EwE6's handling time (EcosimScenarioGroup.CmCo, deviation D5).

EwE divides each predator's search rate by Hden, the handling-time term of
its total intake (cEcoSimModel.vb 2859-2969, 3286, 3379-3403):

    Htime = pred0 / (CmCo x B0 x QB)
    Hden <- (1 - Dwe) x (1 + Htime x sum(a x Vbiom)) + Dwe x Hden,  Dwe = 0.5

once per Derivt call (the consumption is then recomputed with the new Hden),
starting from Hden0 = CmCo / (CmCo - 1); the base search rates carry the
factor Dzero = Hden0 so the base consumption is Ecopath's. sum(a x Vbiom) is
the predator's consumption x Hden / pred. CmCo is QBmax / QB0: EwE's default
1000 barely acts; the Black Sea test case sets 2 on its 13 fish groups.
"""

import warnings
from pathlib import Path

import numpy as np
import pytest

ZOO, FISH = 2, 3  # spname indices (0 = Outside)
BLACKSEA = Path(__file__).parents[3] / "Data" / "BlackSea_EwEF_testcase.eweaccdb"


def test_hden_update_is_ewes_expression():
    from pypath.core.ecosim_deriv import _ewe_hden_update

    hden_old = np.array([1.0, 1.8, 2.0, 1.5])
    q = np.array([0.0, 3.0, 10.0, 1.0])
    index = np.array([0.0, 0.5, 1.0, 2.0])
    cmco_qb = np.array([0.0, 10.0, 20.0, 0.0])  # 0: no handling
    got = _ewe_hden_update(hden_old, q, index, cmco_qb)
    expected = hden_old.copy()
    for j in (1, 2):
        htime_hdent = hden_old[j] * q[j] / (cmco_qb[j] * index[j])
        expected[j] = 0.5 * (1 + htime_hdent) + 0.5 * hden_old[j]
    np.testing.assert_allclose(got, expected, rtol=1e-14)
    # the base state is a fixed point: Q = QB x B0, Hden = CmCo / (CmCo - 1)
    cmco, qb, b0 = 2.0, 5.0, 3.0
    h0 = cmco / (cmco - 1)
    at_base = _ewe_hden_update(
        np.array([h0]), np.array([qb * b0]), np.array([b0]), np.array([cmco * qb])
    )
    assert at_base[0] == pytest.approx(h0, rel=1e-14)


def _scenario(cmco):
    from pypath.core.ecopath import rpath
    from pypath.core.ecosim import rsim_scenario
    from pypath.core.params import create_rpath_params

    params = create_rpath_params(["Phyto", "Zoo", "Fish", "Det"], [1, 0, 0, 2])
    m = params.model
    m["Biomass"] = [100.0, 20.0, 5.0, 10.0]
    m["PB"] = [10.0, 2.0, 1.0, np.nan]
    m["QB"] = [np.nan, 20.0, 4.0, np.nan]
    m["EE"] = np.nan  # Ecopath solves EE: a balanced base state
    m["BioAcc"] = 0.0
    m["Unassim"] = [0.0, 0.2, 0.2, 0.0]
    m["Det"] = [1.0, 1.0, 1.0, 0.0]
    params.diet["Zoo"] = [1.0, 0.0, 0.0, 0.0, 0.0]
    params.diet["Fish"] = [0.0, 1.0, 0.0, 0.0, 0.0]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        sc = rsim_scenario(rpath(params), params, years=range(1, 3))
    sc.params.CmCo[[ZOO, FISH]] = cmco
    return sc


def _run(sc):
    from pypath.core.ecosim import rsim_run

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return rsim_run(sc, method="RK4")


def test_run_starts_from_ewes_hden_and_settles_on_the_base():
    """EwE starts a run from Hden = CmCo / (CmCo + 1) (SetTimeSteps), while
    the base search rates assume CmCo / (CmCo - 1): with CmCo = 2 the
    predators' search rates start at three times their base value (their
    intake rises less, as the arena saturates). Hden then climbs to
    its fixed point and the run returns to the base state; with EwE's default
    1000 the start transient stays below 1e-4."""
    out = _run(_scenario(2.0))
    assert out.out_Hden[0, FISH] == pytest.approx(2.0 / 3.0)
    assert out.out_Hden[-1, FISH] == pytest.approx(2.0, abs=0.01)
    assert np.abs(out.out_Biomass / out.out_Biomass[0] - 1).max() > 0.01
    np.testing.assert_allclose(out.out_Biomass[-1], out.out_Biomass[0], rtol=5e-3)
    default = _run(_scenario(1000.0))
    assert default.out_Hden[0, FISH] == pytest.approx(1000.0 / 1001.0)
    assert np.abs(default.out_Biomass / default.out_Biomass[0] - 1).max() < 1e-4


def test_handling_damps_the_response_to_more_prey():
    """With CmCo = 2 a predator's intake saturates: doubling its prey raises
    its consumption, and so its biomass, less than with EwE's default 1000."""
    runs = {}
    for cmco in (2.0, 1000.0):
        sc = _scenario(cmco)
        sc.forcing.ForcedBio[:, ZOO] = 40.0
        runs[cmco] = _run(sc)
    gain = {
        c: r.out_Biomass[-1, FISH] / r.out_Biomass[0, FISH] for c, r in runs.items()
    }
    assert 1.0 < gain[2.0] < gain[1000.0]
    assert runs[2.0].out_Hden[-1, FISH] > 2.0  # more intake per head, more handling


@pytest.mark.integration
def test_black_sea_cmco_is_read():
    if not BLACKSEA.exists():
        pytest.skip("Black Sea database not found")
    from pypath.io.ewemdb import ecosim_scenario_from_ewemdb

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        sc = ecosim_scenario_from_ewemdb(str(BLACKSEA), scenario=1, years=range(1, 3))
    cmco = dict(zip([str(n).strip() for n in sc.params.spname], sc.params.CmCo))
    assert cmco["Dolphins"] == 2.0
    assert cmco["Anchovy 1-1+"] == 2.0
    assert cmco["Aurelia"] == 1000.0
    assert cmco["Phytoplankton"] == 1000.0  # EwE's -999 cell is not a setting


# EwE 6.7.144 headless oracle, LT2022 scenario 16 with dataset 35 (run
# 43024049431, ecosim_monthly_internal.csv): Hden at the end of months 1, 2,
# 12 and 120 (CmCo 1000 for every group). Single precision; PyPath follows to
# 2e-7. Starting from CmCo / (CmCo - 1) instead of EwE's CmCo / (CmCo + 1)
# misses month 1 by 6e-5.
EWE_HDEN_S16 = {
    "Cod": {1: 1.00093842, 2: 1.00099707, 12: 1.00100136, 120: 1.00082433},
    "Blue mussel (juv.)": {1: 1.0009383, 2: 1.00099707, 12: 1.0009985, 120: 1.00162506},
    "Zooplankton": {1: 1.0009383, 2: 1.00099707, 12: 1.00100088, 120: 1.00072289},
}


@pytest.mark.integration
def test_lt2022_hden_follows_the_ewe_oracle():
    db = Path(__file__).parents[3] / "Data" / "LT2022_0.5ST_final7.eweaccdb"
    if not db.exists():
        pytest.skip("LT2022 database not found")
    from pypath.core.ecosim import rsim_run
    from pypath.io.ewemdb import ecosim_scenario_from_ewemdb

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        sc = ecosim_scenario_from_ewemdb(str(db), scenario=16, timeseries_dataset=35)
        out = rsim_run(sc, method="RK4")
    names = [str(n).strip() for n in sc.params.spname]
    for group, months in EWE_HDEN_S16.items():
        j = names.index(group)
        for month, ewe in months.items():
            assert out.out_Hden[month, j] == pytest.approx(ewe, abs=5e-7), (
                group,
                month,
            )
