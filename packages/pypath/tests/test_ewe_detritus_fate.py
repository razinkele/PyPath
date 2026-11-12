"""EwE6's detritus fate between detritus pools (audit P7).

Ecopath (cEcoPathModel.CalcFateOfDetritus / CalcBAofDetritus, vb 619-668)
passes each pool's surplus, inflow - consumption, on to the other pools in
group order, so a later pool's surplus includes what earlier pools passed to
it. A pool's fate to itself is not inflow: it is the pool's biomass
accumulation, BA = surplus x DF(i, i).

Ecosim (SimDetritusMT and Derivt, vb 3115-3209) moves DetPassedProp(i) x
B(i) x DF(i, j) from pool i to each other pool j, where DetPassedProp is the
Ecopath passed-on flow over B0 and so already holds DF (EwE applies DF
twice). Besides consumption a pool loses DetritusOut x B, with DetritusOut
fixed by the first derivative of the run (FirstTime) so that the pool then
grows at its BA.

Rpath v1.1 counts a pool's own share as inflow and passes on the first-stage
surplus only. PyPath's Ecopath keeps that for the reported detritus EE;
Ecosim and Ecospace follow EwE under fast_group_rule="ewe".
"""

import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

POM, DET = 3, 4  # spname indices (0 = Outside)
BLACKSEA = Path(__file__).parents[3] / "Data" / "BlackSea_EwEF_testcase.eweaccdb"
RECO = Path(__file__).parent / "data" / "rpath_reference" / "ecopath"


def _params():
    """Two detritus pools with the Black Sea's fates: POM -> POM 0.5,
    POM -> Det 0.5, Det -> POM 0.3, Det -> Det 0.7. Nothing eats Det."""
    from pypath.core.params import create_rpath_params

    params = create_rpath_params(["Phyto", "Zoo", "POM", "Det"], [1, 0, 2, 2])
    m = params.model
    m["Biomass"] = [100.0, 20.0, 5000.0, 20000.0]
    m["PB"] = [10.0, 2.0, np.nan, np.nan]
    m["QB"] = [np.nan, 20.0, np.nan, np.nan]
    m["EE"] = np.nan
    m["BioAcc"] = 0.0
    m["Unassim"] = [0.0, 0.2, 0.0, 0.0]
    m["POM"] = [1.0, 0.5, 0.5, 0.3]
    m["Det"] = [0.0, 0.5, 0.5, 0.7]
    params.diet["Zoo"] = [0.8, 0.0, 0.2, 0.0, 0.0]
    return params


def _model():
    from pypath.core.ecopath import rpath

    params = _params()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return rpath(params), params


# Hand-computed EwE routing for _params(). Living inflow: Phyto dies
# 100 x 10 x (1 - 0.32) = 680 to POM; Zoo's 40 deaths + 80 egested split
# 60 / 60. Zoo eats 80 of POM. POM's surplus 740 - 80 = 660 sends 330 to Det;
# Det's surplus 60 + 330 = 390 sends 117 to POM. BA: POM (857 - 80) x 0.5,
# Det 390 x 0.7.
HAND = {
    "input_to_det": (857.0, 390.0),
    "det_eaten": (80.0, 0.0),
    "passed_on": (330.0, 117.0),
    "ba": (388.5, 273.0),
    "ee": (80.0 / 857.0, 0.0),
}


@pytest.mark.parametrize("field", list(HAND))
def test_routing_passes_surpluses_on_in_group_order(field):
    from pypath.core.ecopath import ewe_detritus_fate

    model, _ = _model()
    fate = ewe_detritus_fate(model)
    got = getattr(fate, field)
    assert len(got) == model.NUM_GROUPS
    np.testing.assert_allclose(got[[POM - 1, DET - 1]], HAND[field], rtol=1e-12)
    assert np.all(got[: model.NUM_LIVING] == 0.0)


def test_a_pools_own_ba_adds_to_its_self_share():
    """EwE adds the self share to the pool's BA input (BA(i) +=, vb 666);
    the EwE reader sets that input to 0 (A1-det), a CSV model may not."""
    from pypath.core.ecopath import ewe_detritus_fate, rpath

    params = _params()
    params.model.loc[params.model["Group"] == "Det", "BioAcc"] = 10.0
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        fate = ewe_detritus_fate(rpath(params))
    assert fate.ba[DET - 1] == pytest.approx(273.0 + 10.0, rel=1e-12)


def test_rpath_ecopath_keeps_rpath_v11_routing():
    """The reported EE stays Rpath's: POM's own share counts as inflow and
    the first-stage surpluses are passed on (740 + 330 + 18 = 1088)."""
    model, _ = _model()
    assert model.EE[POM - 1] == pytest.approx(80.0 / 1088.0, rel=1e-12)


def test_routings_agree_without_self_fates_or_cascades():
    """REco: Discards passes to Detritus, which passes nothing on."""
    from pypath.core.ecopath import ewe_detritus_fate, rpath
    from pypath.core.params import create_rpath_params

    model_df = pd.read_csv(RECO / "model_params.csv")
    params = create_rpath_params(
        model_df["Group"].tolist(), [int(t) for t in model_df["Type"].tolist()]
    )
    params.model = model_df
    params.diet = pd.read_csv(RECO / "diet_matrix.csv")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        model = rpath(params)
    fate = ewe_detritus_fate(model)
    dead = slice(model.NUM_LIVING, model.NUM_LIVING + model.NUM_DEAD)
    np.testing.assert_allclose(fate.ee[dead], model.EE[dead], rtol=1e-10)
    assert np.all(fate.ba[dead] == 0.0)


def _scenario(rule="ewe"):
    from pypath.core.ecosim import rsim_scenario

    model, params = _model()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return rsim_scenario(model, params, years=range(1, 3), fast_group_rule=rule)


def _run(sc):
    from pypath.core.ecosim import rsim_run

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return rsim_run(sc, method="RK4")


def test_ewe_rule_carries_the_routing():
    sc = _scenario()
    b0 = sc.params.B_BaseRef
    np.testing.assert_allclose(
        sc.params.DetPassedProp[[POM, DET]], [330.0 / b0[POM], 117.0 / b0[DET]]
    )
    np.testing.assert_allclose(sc.params.DetBA[[POM, DET]], HAND["ba"])
    assert _scenario("rpath").params.DetPassedProp is None


def test_ewe_fast_group_test_reads_the_routed_loss():
    """EwE's fast-group test (SetTimeSteps, vb 2795) takes a pool's loss from
    the first derivative: inflow - BA, for POM 740 + 0.3 x 117 - 388.5 =
    386.6 a year rather than its living inflow of 740. With B = 20 that is
    19.3 /yr, under the threshold of 24, so POM is integrated."""
    from pypath.core.ecopath import rpath
    from pypath.core.ecosim import rsim_scenario

    params = _params()
    params.model.loc[params.model["Group"] == "POM", "Biomass"] = 20.0
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        sc = rsim_scenario(rpath(params), params, years=range(1, 3))
    assert sc.params.NoIntegrate[POM] == 0


def test_ewe_rule_pools_start_at_their_ba_with_dfs_applied_twice():
    """DetritusOut is fixed at the first derivative so POM grows at its BA,
    388.5 a year. Det gets only 330 x 0.5 = 165 of what POM passes on (EwE
    applies DF twice) plus Zoo's 60: 225 a year, less than its BA of 273, so
    its DetritusOut floors at 1e-20 and it grows at 225. Decay and grazing
    barely act within the month."""
    out = _run(_scenario())
    step = out.out_Biomass[1] - out.out_Biomass[0]
    assert step[POM] == pytest.approx(388.5 / 12, rel=1e-2)
    assert step[DET] == pytest.approx(225.0 / 12, rel=1e-2)


def test_rpath_rule_keeps_the_rpath_links():
    """Rpath: each pool sends M0 x B along every link, its own included, so
    POM gains 740 + 0.3 x 432 + 0.5 x 1008 and loses 80 + 1008 (285.6 / yr);
    Det gains 60 + 0.5 x 1008 + 0.7 x 432 and loses 432 (434.4 / yr)."""
    out = _run(_scenario("rpath"))
    step = out.out_Biomass[1] - out.out_Biomass[0]
    assert step[POM] == pytest.approx(285.6 / 12, rel=2e-2)
    assert step[DET] == pytest.approx(434.4 / 12, rel=2e-2)


def test_ecospace_patches_follow_ecosim():
    """Unequal patch areas, uniform habitat: every patch keeps Ecosim's
    densities, so the map total follows rsim_run."""
    from pypath.spatial import EcospaceParams, create_1d_grid, rsim_run_spatial

    sc = _scenario()
    n = sc.params.NUM_GROUPS + 1
    grid = create_1d_grid(n_patches=3, spacing=1.0)
    grid.patch_areas = np.array([1.0, 2.0, 3.0])
    eco = EcospaceParams(
        grid=grid,
        habitat_preference=np.ones((n, 3)),
        habitat_capacity=np.ones((n, 3)),
        dispersal_rate=np.zeros(n),
        advection_enabled=np.zeros(n, dtype=bool),
        gravity_strength=np.zeros(n),
    )
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        spatial = rsim_run_spatial(sc, ecospace=eco, years=range(1, 3))
    np.testing.assert_allclose(
        spatial.out_Biomass, _run(sc).out_Biomass, rtol=1e-6, atol=1e-10
    )


def _uniform_ecospace(n):
    from pypath.spatial import EcospaceParams, create_1d_grid

    grid = create_1d_grid(n_patches=2, spacing=1.0)
    return EcospaceParams(
        grid=grid,
        habitat_preference=np.ones((n, 2)),
        habitat_capacity=np.ones((n, 2)),
        dispersal_rate=np.zeros(n),
        advection_enabled=np.zeros(n, dtype=bool),
        gravity_strength=np.zeros(n),
    )


def _export_rate_calls(sc, monkeypatch, on_call=None):
    """Run Ecospace and record each call of its Ecosim export-rate helper."""
    import pypath.spatial.integration as integ
    from pypath.spatial import rsim_run_spatial

    calls = []
    original = integ._ecosim_detritus_out

    def spy(*args):
        before = on_call() if on_call else None
        out = original(*args)
        calls.append((args, before, on_call() if on_call else None))
        return out

    monkeypatch.setattr(integ, "_ecosim_detritus_out", spy)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        rsim_run_spatial(
            sc, ecospace=_uniform_ecospace(sc.params.NUM_GROUPS + 1), years=range(1, 2)
        )
    return calls, original


class _CountingIBM:
    """Stands in for an IBM group; counts the steps it takes."""

    def __init__(self, group_index, n_groups, biomass):
        self.group_index, self.n_groups, self.biomass = group_index, n_groups, biomass
        self.steps = 0

    def compute_step(self, **kwargs):
        from types import SimpleNamespace

        self.steps += 1
        return SimpleNamespace(
            biomass=self.biomass, consumption_by_prey=np.zeros(self.n_groups)
        )


def test_ecospace_export_rate_leaves_ibm_groups_alone(monkeypatch):
    """An IBM group steps in every derivative; the extra derivatives that
    set Ecospace's export rate must not advance the run's populations."""
    sc = _scenario()
    ibm = _CountingIBM(2, sc.params.NUM_GROUPS, sc.params.B_BaseRef[2])
    sc.params.ibm_groups = {2: ibm}
    calls, _ = _export_rate_calls(sc, monkeypatch, on_call=lambda: ibm.steps)
    assert len(calls) == 1
    _, before, after = calls[0]
    assert after == before


def test_ecospace_export_rate_starts_from_the_forced_state(monkeypatch):
    """rsim_run fixes DetritusOut on the month's start state with ForcedBio
    applied; Ecospace's shared rate starts from the same state."""
    sc = _scenario()
    forced = 1.5 * sc.params.B_BaseRef[2]
    sc.forcing.ForcedBio[:, 2] = forced
    calls, original = _export_rate_calls(sc, monkeypatch)
    params, forcing, fishing, biomass, hden = calls[0][0]
    forced_state = np.array(biomass, dtype=float)
    forced_state[2] = forced
    np.testing.assert_allclose(
        original(params, forcing, fishing, biomass, hden),
        original(params, forcing, fishing, forced_state, hden),
        rtol=1e-12,
    )


# --- EwE 6.7.144 headless oracle, Black Sea scenario 1 (run 43103422428) -------

# ecopath.csv: InputToDet, FlowToDet (= DetPassedOn), BioAccum, EE
EWE_BLACKSEA_ECOPATH = {
    "POM": (23.0787659, 2.15542555, 5.344872, 0.536814749),
    "Detritus": (21.2816048, 6.37889242, 14.8840818, 0.000875478145),
}

# ecosim_monthly.csv: biomass at the end of months 1, 12 and 120. Before EwE's
# detritus fate PyPath was 1.3 %, 10 % and 45 % high on POM, and Aurelia
# 80 % high by month 120.
EWE_BLACKSEA_ECOSIM = {
    "POM": {1: 20.3936462, 12: 22.9598083, 120: 26.3971748},
    "Detritus": {1: 81.1862946, 12: 92.785881, 120: 164.720718},
    "Aurelia": {1: 0.0369501375, 12: 0.0368066356, 120: 0.0417026728},
    "Phytoplankton": {1: 0.315152794, 12: 0.238087296, 120: 0.201916695},
}


def _blacksea():
    if not BLACKSEA.exists():
        pytest.skip("Black Sea database not found")
    from pypath.io.ewemdb import ecosim_scenario_from_ewemdb

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return ecosim_scenario_from_ewemdb(str(BLACKSEA), scenario=1)


@pytest.mark.integration
def test_black_sea_routing_follows_the_ewe_oracle():
    from pypath.core.ecopath import ewe_detritus_fate, rpath
    from pypath.io.ewemdb import read_ewemdb

    if not BLACKSEA.exists():
        pytest.skip("Black Sea database not found")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        model = rpath(read_ewemdb(str(BLACKSEA)))
    fate = ewe_detritus_fate(model)
    names = [str(g).strip() for g in model.Group]
    for group, expected in EWE_BLACKSEA_ECOPATH.items():
        j = names.index(group)
        got = (fate.input_to_det[j], fate.passed_on[j], fate.ba[j], fate.ee[j])
        np.testing.assert_allclose(got, expected, rtol=1e-6, err_msg=group)


@pytest.mark.integration
def test_black_sea_detritus_follows_the_ewe_oracle():
    sc = _blacksea()
    out = _run(sc)
    names = [str(n).strip() for n in sc.params.spname]
    for group, months in EWE_BLACKSEA_ECOSIM.items():
        j = names.index(group)
        for month, ewe in months.items():
            assert out.out_Biomass[month, j] == pytest.approx(ewe, rel=5e-5), (
                group,
                month,
            )
