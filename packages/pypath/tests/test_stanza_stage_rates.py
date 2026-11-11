"""Which rates EwE6's monthly stanza update uses.

EwE6 (one RK4 step a month) calls SplitUpdate after the step with
- growth Gf = Eatenby / pred, Eatenby from the step's last Derivt, i.e. the
  fourth RK4 stage at y + h k3 (cEcoSimModel.rk4, vb 1700-1745, 1851);
- survival Su = exp(-LossAvg / 12 / BAvg) with the loss of the step's first
  Derivt (rk4 resets loss to LossSt) and the stanza groups' start-of-month
  biomass (they are not advanced by the step: NoIntegrate < 0), vb 1745-1756.
PyPath took both from the end-of-month state. In a month with a sharp
change (LT2022's year-2 production step, month 13) the consumption differs,
and the blue mussels stepped 3e-4 off the EwE oracle there; scenario 15
amplified it to 1.7 %. The 'rpath' rule keeps the end-of-month rates.
"""

from pathlib import Path

import numpy as np
import pytest

import pypath.core.ecosim_deriv as D
from pypath.core.ecosim import rsim_run, rsim_scenario
from pypath.core.ecosim_deriv import deriv_vector, integrate_rk4

from .test_forcing_channels import _simple_model
from .test_stanza_plus_group import _ewe_scenario

LT2022 = Path(__file__).parents[3] / "Data" / "LT2022_0.5ST_final7.eweaccdb"


def test_integrate_rk4_reports_its_first_and_last_stage_rates(monkeypatch):
    """No fast groups here, so the fourth stage is exactly y0 + h k3."""
    model, params = _simple_model()
    sc = rsim_scenario(model, params, years=range(1, 3))
    # isolates the mechanism under test; EwE's handling time (D5)
    # has its own tests (test_handling_time.py)
    sc.params.EWE_HANDLING_TIME = False
    cap = {}
    real = D.deriv_vector

    def spy(state, p, forcing, fishing, *a, **k):
        if "P" not in cap and "ForcedPrey" in forcing:
            cap.update(P=p, Fo=forcing, Fi=fishing, s=np.array(state, float))
        return real(state, p, forcing, fishing, *a, **k)

    monkeypatch.setattr(D, "deriv_vector", spy)
    rsim_run(sc, method="RK4")
    monkeypatch.undo()
    P, Fo, Fi, y0 = cap["P"], cap["Fo"], cap["Fi"], cap["s"]
    assert not np.any(D._rk4_fast_mask(P, len(y0)))
    h = 1.0 / 12.0
    stage = {}
    integrate_rk4(y0.copy(), P, Fo, Fi, h, n_steps=1, stage_rates=stage)
    k1 = deriv_vector(y0.copy(), P, Fo, Fi)
    k2 = deriv_vector(y0 + 0.5 * h * k1, P, Fo, Fi)
    k3 = deriv_vector(y0 + 0.5 * h * k2, P, Fo, Fi)
    first, last = {}, {}
    deriv_vector(y0.copy(), P, Fo, Fi, rates=first)
    deriv_vector(y0 + h * k3, P, Fo, Fi, rates=last)
    for key in ("FoodGain", "LossPropToB"):
        np.testing.assert_allclose(stage["first"][key], first[key], rtol=1e-12)
        np.testing.assert_allclose(stage["last"][key], last[key], rtol=1e-12)


def _split_update_inputs(monkeypatch, sc):
    """split_update's arguments, the month-start state and the stage rates,
    for every month."""
    import pypath.core.ecosim as EC

    months = []
    real_rk4, real_split = D.integrate_rk4, EC.split_update

    def rk4_spy(state, p, forcing, fishing, dt, n_steps=1, stage_rates=None):
        months.append({"start": np.array(state, float), "stage": stage_rates})
        return real_rk4(state, p, forcing, fishing, dt, n_steps, stage_rates)

    def split_spy(stanzas, sstate, biomass, food, loss, **k):
        months[-1].update(b=np.array(biomass), food=np.array(food), loss=np.array(loss))
        return real_split(stanzas, sstate, biomass, food, loss, **k)

    monkeypatch.setattr(D, "integrate_rk4", rk4_spy)
    monkeypatch.setattr(EC, "split_update", split_spy)
    rsim_run(sc, method="RK4")
    monkeypatch.undo()
    return months


def test_ewe_rule_stanza_update_takes_ewes_rates(monkeypatch):
    _, sc = _ewe_scenario()
    phyto = sc.params.spname.index("Phyto")
    sc.forcing.ForcedSearch[1:, phyto] = 0.8  # a production step in month 2
    months = _split_update_inputs(monkeypatch, sc)
    m = months[1]
    assert m["stage"] is not None
    np.testing.assert_allclose(m["food"], m["stage"]["last"]["FoodGain"], rtol=1e-12)
    np.testing.assert_allclose(
        m["loss"], m["stage"]["first"]["LossPropToB"], rtol=1e-12
    )
    st = sc.stanzas
    idx = [int(st.ecopath_code[1, k]) for k in (1, 2)]
    np.testing.assert_allclose(m["b"][idx], m["start"][idx], rtol=1e-12)


def test_rpath_rule_stanza_update_keeps_end_of_month_rates(monkeypatch):
    import warnings

    from pypath.core.ecopath import rpath
    from pypath.core.stanzas import rpath_stanzas

    from .test_stanza_plus_group import _mussel_params

    params = _mussel_params()
    rpath_stanzas(params)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        model = rpath(params)
    sc = rsim_scenario(model, params, years=range(1, 3), fast_group_rule="rpath")
    months = _split_update_inputs(monkeypatch, sc)
    m = months[1]
    assert m["stage"] is None
    assert not np.allclose(m["b"], m["start"])  # the end-of-month state


@pytest.mark.integration
def test_lt2022_scenario_15_mussels_follow_the_ewe_oracle_through_the_step():
    """EwE 6.7.144 oracle (run 42895374068), end of months 13, 14 and 24:
    the month of the year-2 production step and after."""
    if not LT2022.exists():
        pytest.skip("LT2022 database not found")
    from pypath.io.ewemdb import EwEDatabaseError, ecosim_scenario_from_ewemdb

    try:
        sc = ecosim_scenario_from_ewemdb(
            str(LT2022), scenario=15, timeseries_dataset=35
        )
    except EwEDatabaseError as e:
        pytest.skip(f"No EwE database driver: {e}")
    oracle = {
        "Blue mussel (juv.)": (14.5212412, 13.8664608, 9.45448875),
        "Blue mussel (ad.)": (26.4216805, 25.5625572, 18.9353294),
    }
    out = rsim_run(sc, method="RK4", years=range(1, 3))
    names = [str(n).strip() for n in sc.params.spname]
    for group, values in oracle.items():
        i = names.index(group)
        for month, ewe in zip((13, 14, 24), values):
            assert out.out_Biomass[month, i] == pytest.approx(ewe, rel=1.5e-4), (
                group,
                month,
            )
