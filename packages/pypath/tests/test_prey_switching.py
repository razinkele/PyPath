"""EwE6 prey switching (SwitchPower).

EwE6 (cEcoSimModel.SetRelaSwitch / InitRelaSwitch) multiplies each link's
search rate by

    RelaSwitch_ij = [A_ij B_i^s / sum_k A_kj B_k^s] / [A_ij B0_i^s / sum_k A_kj B0_k^s]

at every derivative evaluation, with s the predator's SwitchPower and A_ij
the base search rate. For one predator A_ij is proportional to
v_ij Q0_ij / (B0_i (v_ij - 1)) (MakeAMatrixCell, with the vulnerability
exchange rate VulMult * Q0 / B0). A search-rate factor r on a link of the
closed-form kernel Q = Q0 PYY PDY VV/(VV - 1 + PDY) gives
Q = Q0 PYY (r PDY) VV/(VV - 1 + r PDY).

PyPath used to read SwitchPower nowhere: the setting had no effect.
"""

import warnings
from pathlib import Path

import numpy as np
import pytest

from pypath.core.ecopath import rpath
from pypath.core.ecosim import (
    _build_link_matrix,
    _compute_Q_matrix,
    rsim_run,
    rsim_scenario,
)
from pypath.core.ecosim_deriv import deriv_vector
from pypath.core.params import create_rpath_params
from tests.test_ewe_ecosim_options import _fishing, _forcing, _params_dict

DB_PATH = Path(__file__).parents[3] / "Data" / "LT2022_0.5ST_final7.eweaccdb"
PRED, PREY_A, PREY_B = 1, 2, 3  # Ecosim indices
VV = 3.0


def _model():
    params = create_rpath_params(
        groups=["Pred", "PreyA", "PreyB", "Phyto", "Det"], types=[0, 0, 0, 1, 2]
    )
    params.model.loc[0, ["Biomass", "PB", "QB"]] = [1.0, 0.5, 4.0]
    params.model.loc[1, ["Biomass", "PB", "QB"]] = [4.0, 1.5, 6.0]
    params.model.loc[2, ["Biomass", "PB", "QB"]] = [2.0, 2.0, 8.0]
    params.model.loc[3, ["Biomass", "PB"]] = [20.0, 5.0]
    params.model.loc[4, "Biomass"] = 10.0
    params.model["EE"] = np.nan
    params.model["Det"] = 1.0
    params.model.loc[4, "Det"] = 0.0
    params.diet["Pred"] = [0.0, 0.7, 0.3, 0.0, 0.0, 0.0]
    params.diet["PreyA"] = [0.0, 0.0, 0.0, 1.0, 0.0, 0.0]
    params.diet["PreyB"] = [0.0, 0.0, 0.0, 1.0, 0.0, 0.0]
    params.diet["Phyto"] = [0.0] * 6
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        model = rpath(params)
    return model, params


def _scenario(switch_power, years=range(1, 3)):
    model, params = _model()
    overrides = {
        "switch_power": {PRED - 1: switch_power},
        "vv_overrides": {(PREY_A - 1, PRED - 1): VV, (PREY_B - 1, PRED - 1): VV},
    }
    return rsim_scenario(model, params, years=years, scenario_overrides=overrides)


def _dict(sc):
    d = _params_dict(sc.params)
    d["SwitchPower"] = sc.params.SwitchPower
    return d


def _q(sc, state):
    return _compute_Q_matrix(_dict(sc), state, _forcing(sc.params.NUM_GROUPS))


def _prey_a_doubled(sc):
    state = sc.params.B_BaseRef.copy()
    state[PREY_A] *= 2.0
    return state


def test_switch_power_is_read_from_the_overrides():
    sc = _scenario(1.0)
    assert sc.params.SwitchPower[PRED] == 1.0
    assert sc.params.SwitchPower[PREY_A] == 0.0


def test_default_is_no_switching():
    model, params = _model()
    sc = rsim_scenario(model, params, years=range(1, 3))
    assert np.all(sc.params.SwitchPower == 0.0)


def test_no_switching_at_the_base_state():
    sc_off, sc_on = _scenario(0.0), _scenario(1.0)
    state = sc_on.params.B_BaseRef.copy()
    np.testing.assert_allclose(_q(sc_on, state), _q(sc_off, state), rtol=1e-12)


def test_consumption_follows_the_ewe_formula():
    sc_off, sc_on = _scenario(0.0), _scenario(1.0)
    p = sc_on.params
    state = _prey_a_doubled(sc_on)
    q_off, q_on = _q(sc_off, state), _q(sc_on, state)

    b0 = p.B_BaseRef
    q0 = _build_link_matrix(p, p.QQ)
    dd = _build_link_matrix(p, p.DD)
    prey = (PREY_A, PREY_B)
    w = {i: VV * q0[i, PRED] / (b0[i] * (VV - 1.0)) for i in prey}
    now = sum(w[i] * state[i] for i in prey)
    base = sum(w[i] * b0[i] for i in prey)
    for i in prey:
        r = (w[i] * state[i] / now) / (w[i] * b0[i] / base)
        pyy = state[i] / b0[i]
        dd_term = dd[i, PRED] / (dd[i, PRED] - 1.0 + pyy)
        expected = q0[i, PRED] * pyy * dd_term * r * VV / (VV - 1.0 + r)
        assert q_on[i, PRED] == pytest.approx(expected, rel=1e-9)
    # the predator turns to the abundant prey
    assert q_on[PREY_A, PRED] > q_off[PREY_A, PRED]
    assert q_on[PREY_B, PRED] < q_off[PREY_B, PRED]


def test_derivative_uses_the_switching():
    sc_off, sc_on = _scenario(0.0), _scenario(1.0)
    state = _prey_a_doubled(sc_on)
    f = _forcing(sc_on.params.NUM_GROUPS)
    off = deriv_vector(state.copy(), _dict(sc_off), f, _fishing(sc_off.params))
    on = deriv_vector(state.copy(), _dict(sc_on), f, _fishing(sc_on.params))
    assert on[PREY_A] < off[PREY_A]
    assert on[PREY_B] > off[PREY_B]


@pytest.mark.parametrize("method", ["RK4", "AB"])
def test_run_uses_the_switching(method):
    out = {}
    for s in (0.0, 1.0):
        sc = _scenario(s)
        sc.start_state.Biomass[PREY_A] *= 2.0
        out[s] = rsim_run(sc, method=method).out_Biomass
    # more of the abundant prey is eaten from the first month on
    assert out[1.0][1, PREY_A] < out[0.0][1, PREY_A]
    assert out[1.0][1, PREY_B] > out[0.0][1, PREY_B]


def test_ecospace_switches_on_the_map_wide_base():
    """EwE6 Ecospace (cSpaceSolver.SetRelaSwitch) uses the Ecosim base
    biomass: where PreyA's habitat makes it denser, the predator turns to it."""
    from pypath.spatial.environmental import EnvironmentalDrivers
    from tests.test_spatial_habitat_capacity import (
        _ecospace,
        _first_call,
        _local_rates,
    )

    rates = {}
    for s in (0.0, 1.0):
        sc = _scenario(s)
        n_rows = sc.params.NUM_GROUPS + 1
        cap = np.ones((n_rows, 2))
        cap[PREY_A - 1] = [0.2, 1.8]
        args = _first_call(sc, _ecospace(n_rows, cap), EnvironmentalDrivers())
        rates[s] = _local_rates(args)
    # Each patch starts at its own base B * c. Normalised by that, the
    # switching factor would be 1 and the start rates unchanged.
    poor, rich = 0, 1
    assert rates[1.0][PREY_A, rich] < rates[0.0][PREY_A, rich]
    assert rates[1.0][PREY_A, poor] > rates[0.0][PREY_A, poor]


@pytest.mark.integration
def test_lt2022_long_tailed_duck_switches():
    if not DB_PATH.exists():
        pytest.skip("LT2022 database not found")
    from pypath.io.ewemdb import EwEDatabaseError, ecosim_scenario_from_ewemdb

    try:
        sc = ecosim_scenario_from_ewemdb(str(DB_PATH), scenario=16)
    except EwEDatabaseError as e:
        pytest.skip(f"No EwE database driver: {e}")
    names = [str(n).strip() for n in sc.params.spname]
    sp = {names[i]: float(v) for i, v in enumerate(sc.params.SwitchPower) if v}
    assert sp == {"Long-tailed duck": 1.0}
