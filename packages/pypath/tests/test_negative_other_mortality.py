"""EwE6's other mortality for a group with EE > 1 (deviation M0).

A group whose losses exceed its production has EE > 1 and a negative other
mortality, mo = (1 - EE) x PB (cEcoSimModel.CalcMo, vb 4481). EwE's
CalcBaseAdditiveMort gives every eaten group with mo <= 0 moMax = 1e-20 and
Qh = 0 (vb 3807-3815), and Derivt takes moTot = moMax / (1 + Qh x Eatenof /
StartEatenOf) in place of mo for every eaten group (vb 3058-3081). So an
eaten group with mo < 0 loses no other mortality in Ecosim, while a group
that is not eaten keeps its negative mo. The detritus flow keeps mo
(SimDetritusMT, vb 3165). EwE's fast-group test reads the loss of the first
Derivt (vb 2794), so it sees moTot too. The 'rpath' rule keeps mo.
"""

import copy
import warnings
from pathlib import Path

import numpy as np
import pytest

import pypath.core.ecosim as EC
import pypath.core.ecosim_deriv as D

PHYTO, ZOO, FISH, DET = 1, 2, 3, 4  # rsim indices (0 = Outside)


def _model(zoo_b=20.0, zoo_pb=2.0, fish_b=12.0, zoo_det=0.0):
    """Zoo: production zoo_b x zoo_pb, eaten fish_b x 4 by Fish; zoo_det of
    its diet is detritus. Fish: no predators, production fish_b, landings 15
    (EE > 1 for fish_b < 15)."""
    from pypath.core.ecopath import rpath
    from pypath.core.params import create_rpath_params

    params = create_rpath_params(
        ["Phyto", "Zoo", "Fish", "Det", "Fleet"], [1, 0, 0, 2, 3]
    )
    m = params.model
    m["Biomass"] = [100.0, zoo_b, fish_b, 10.0, 0.0]
    m["PB"] = [10.0, zoo_pb, 1.0, np.nan, np.nan]
    m["QB"] = [np.nan, 20.0, 4.0, np.nan, np.nan]
    m["EE"] = np.nan
    m["BioAcc"] = 0.0
    m["Unassim"] = [0.0, 0.2, 0.2, 0.0, 0.0]
    m["Det"] = [1.0, 1.0, 1.0, 0.0, 0.0]
    m.loc[2, "Fleet"] = 15.0  # Fish landings
    params.diet["Zoo"] = [1.0 - zoo_det, 0.0, 0.0, zoo_det, 0.0]
    params.diet["Fish"] = [0.0, 1.0, 0.0, 0.0, 0.0]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return rpath(params), params


def _scenario(rule, **kw):
    from pypath.core.ecosim import rsim_scenario

    model, params = _model(**kw)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return rsim_scenario(model, params, years=range(1, 3), fast_group_rule=rule)


def _start_rates(monkeypatch, sc):
    """The derivative's rates at the start state of rsim_run (inputs copied
    at the first call: the run updates state such as Hden in place)."""
    cap = {}
    real = D.deriv_vector

    def spy(state, params, forcing, fishing, *a, **k):
        if "P" not in cap:
            cap.update(
                P=copy.deepcopy(params),
                Fo=copy.deepcopy(forcing),
                Fi=copy.deepcopy(fishing),
                s=np.array(state, float),
            )
        return real(state, params, forcing, fishing, *a, **k)

    for module in (D, EC):
        monkeypatch.setattr(module, "deriv_vector", spy)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        EC.rsim_run(sc, method="RK4")
    monkeypatch.undo()
    rates = {}
    D.deriv_vector(cap["s"].copy(), cap["P"], cap["Fo"], cap["Fi"], rates=rates)
    return rates, cap["s"]


def test_the_toy_model_has_negative_other_mortality():
    sc = _scenario("ewe")
    assert sc.params.MzeroMort[ZOO] == pytest.approx(2.0 * (1.0 - 48.0 / 40.0))
    assert sc.params.MzeroMort[FISH] == pytest.approx(1.0 * (1.0 - 15.0 / 12.0))


def test_an_eaten_group_loses_no_negative_other_mortality(monkeypatch):
    rates, state = _start_rates(monkeypatch, _scenario("ewe"))
    # Zoo is eaten: moTot = 1e-20 in place of mo = -0.4
    assert abs(rates["MzeroLoss"][ZOO]) < 1e-15
    # Fish is not eaten: EwE keeps mo (Derivt's Else branch)
    assert rates["MzeroLoss"][FISH] == pytest.approx(-0.25 * state[FISH], rel=1e-12)


def test_the_detritus_flow_keeps_the_negative_mo(monkeypatch):
    """Detritus gets every living group's mo x B, negative ones included:
    6 x 100 - 0.4 x 20 - 0.25 x 12 besides the egestion (0.2 of the intake;
    the first Derivt starts from Hden = CmCo / (CmCo + 1), R3, so the intake
    is not quite the base)."""
    rates, _ = _start_rates(monkeypatch, _scenario("ewe"))
    egestion = 0.2 * (rates["FoodGain"][ZOO] + rates["FoodGain"][FISH])
    assert rates["DetGain"][DET] - egestion == pytest.approx(
        600.0 - 8.0 - 3.0, rel=1e-9
    )


def test_the_rpath_rule_keeps_the_negative_mo(monkeypatch):
    rates, state = _start_rates(monkeypatch, _scenario("rpath"))
    assert rates["MzeroLoss"][ZOO] == pytest.approx(-0.4 * state[ZOO], rel=1e-12)


def test_the_eaten_group_declines_from_its_base():
    """Without the negative mo Zoo's losses exceed its production by
    0.4 x 20 = 8 /yr at the base; under the 'rpath' rule the base holds."""
    runs = {}
    for rule in ("ewe", "rpath"):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            runs[rule] = EC.rsim_run(_scenario(rule), method="RK4")
    b0 = runs["rpath"].out_Biomass[0, ZOO]
    assert runs["rpath"].out_Biomass[1, ZOO] == pytest.approx(b0, rel=1e-9)
    assert runs["ewe"].out_Biomass[1, ZOO] < 0.99 * b0


def test_the_fast_group_test_sees_the_ewe_loss():
    """Zoo B 2, PB 23, eaten 50 by Fish: EE 50 / 46, mo = -2. EwE's base loss
    is 50, 25 /yr > 24, so Zoo is solved at equilibrium; with mo the rate
    would be 46 / 2 = 23 /yr."""
    sc = _scenario("ewe", zoo_b=2.0, zoo_pb=23.0, fish_b=12.5)
    assert sc.params.MzeroMort[ZOO] == pytest.approx(23.0 * (1.0 - 50.0 / 46.0))
    assert sc.params.NoIntegrate[ZOO] == 1


@pytest.mark.parametrize("compat", ["rpath", "ewe"])
def test_ecopath_detritus_counts_negative_other_mortality(compat):
    """Rpath's rpath() and EwE's Ecopath both send M0 x B to detritus
    whatever its sign (Rpath: loss = M0 B + B QB Unassim; EwE: LT2022's
    Sand goby with EE 1.12 has FlowToDet 0.41487 = 0.6 egestion - 0.18513).
    Zoo eats 0.1 detritus, so Phyto is eaten 360 (mo 6.4): inflow 640 - 8 - 3
    + 0.2 x 448 = 718.6 and detritus EE 40 / 718.6."""
    from pypath.core.ecopath import rpath

    _, params = _model(zoo_det=0.1)
    params.compat = compat
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        model = rpath(params)
    assert model.EE[ZOO - 1] == pytest.approx(48.0 / 40.0)
    assert model.EE[DET - 1] == pytest.approx(40.0 / 718.6, rel=1e-9)


# --- EwE 6.7.144 oracle: LT2022 with ewe-oracle patches/lt2022_m0_negative_mo.sql
# Sand goby's entered B 2.2 -> 1.5 with its predation unchanged: EE 1.12 and
# mo = -0.123. Run 43246564948 (scenario 16, dataset 35). Before the fix
# PyPath held Sand goby near its base (1.500 at month 12, 0.264 at 144) and
# its predators drifted 3-6 %; Ecopath's detritus EE was 3e-5 off.
DB = Path(__file__).parents[3] / "Data" / "LT2022_negative_mo.eweaccdb"
EWE_ECOPATH_EE = {"Sand goby": 1.12341988, "Detritus": 0.12477608}
EWE_BIOMASS = {
    "Sand goby": {1: 1.48496771, 12: 1.36973691, 60: 0.741544306, 144: 0.172024578},
    "Flounder": {1: 1.55995739, 12: 1.55582881, 60: 1.06248856, 144: 0.683623016},
    "Cod": {1: 0.673998594, 12: 0.672534943, 60: 0.390916526, 144: 0.447684854},
    "Detritus": {1: 1.80000615, 12: 1.79991055, 60: 1.46237433, 144: 0.818782449},
}


def _db():
    if not DB.exists():
        pytest.skip(f"{DB.name} not found")
    return str(DB)


@pytest.mark.integration
def test_lt2022_ecopath_follows_the_ewe_oracle():
    from pypath.core.ecopath import rpath
    from pypath.io.ewemdb import read_ewemdb

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        model = rpath(read_ewemdb(_db()))
    names = [str(g).strip() for g in model.Group]
    for group, ee in EWE_ECOPATH_EE.items():
        assert model.EE[names.index(group)] == pytest.approx(ee, rel=1e-5), group


@pytest.mark.integration
def test_lt2022_ecosim_follows_the_ewe_oracle():
    from pypath.io.ewemdb import ecosim_scenario_from_ewemdb

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        sc = ecosim_scenario_from_ewemdb(_db(), scenario=16, timeseries_dataset=35)
        out = EC.rsim_run(sc, method="RK4")
    names = [str(n).strip() for n in sc.params.spname]
    assert sc.params.MzeroMort[names.index("Sand goby")] < 0
    for group, months in EWE_BIOMASS.items():
        j = names.index(group)
        for month, ewe in months.items():
            assert out.out_Biomass[month, j] == pytest.approx(ewe, rel=2e-3), (
                group,
                month,
            )
