"""Handling time: Rpath's per-link prey term is not EwE's.

Rpath multiplies each link by DD / (DD - 1 + preyYY) (DD = 1000 by
default), so a predator's intake from one prey saturates as that prey
becomes abundant. EwE6 has no per-link term: handling time acts on the
predator's total intake, Hden = 1 + Htime x sum(a x V) with Htime =
pred / (CmCo x B0 x QB) (cEcoSimModel.Derivt, vb 2896-2969), which at
CmCo = 1000 stays within 0.1 % of its base value unless the predator eats
far more per head than at base. LT2022's dataset 35 forces Round goby from
0.6 to 5.7 (190 x its Ecopath biomass) in year 8: Rpath's term cut Cod's
intake of it by 16 %, and Cod, Turbot and Perciformes fell 3-6 % below the
EwE oracle.
"""

from pathlib import Path

import numpy as np
import pytest

import pypath.core.ecosim as EC
import pypath.core.ecosim_deriv as D
from pypath.core.ecosim import rsim_run, rsim_scenario

from .test_forcing_channels import _simple_model

PHYTO, ZOO = 1, 2
LT2022 = Path(__file__).parents[3] / "Data" / "LT2022_0.5ST_final7.eweaccdb"


def _first_call(monkeypatch, run):
    """Params dict, forcing and fishing of the first full derivative call."""
    cap = {}
    real = D.deriv_vector

    def spy(state, params, forcing, fishing, *a, **k):
        if "P" not in cap and "ForcedPrey" in forcing:
            cap.update(P=params, Fo=forcing, Fi=fishing, s=np.array(state, float))
        return real(state, params, forcing, fishing, *a, **k)

    import pypath.spatial.integration as SI

    for module in (D, EC, SI):  # each imports deriv_vector by name
        monkeypatch.setattr(module, "deriv_vector", spy)
    run()
    return cap


def _intake_ratio(cap, factor):
    """Zoo's intake of Phyto at Phyto x factor, relative to the base."""
    rates0, rates1 = {}, {}
    state = cap["s"].copy()
    real = EC.deriv_vector
    real(state.copy(), cap["P"], cap["Fo"], cap["Fi"], rates=rates0)
    state[PHYTO] *= factor
    real(state.copy(), cap["P"], cap["Fo"], cap["Fi"], rates=rates1)
    return rates1["FoodGain"][ZOO] / rates0["FoodGain"][ZOO]


@pytest.mark.parametrize(
    "rule, expected",
    [
        ("ewe", 190.0),  # no per-link saturation
        ("rpath", 190.0 * 1000.0 / (999.0 + 190.0)),  # DD / (DD - 1 + Y)
    ],
)
def test_intake_of_an_abundant_prey(monkeypatch, rule, expected):
    model, params = _simple_model()
    sc = rsim_scenario(model, params, years=range(1, 3), fast_group_rule=rule)
    cap = _first_call(monkeypatch, lambda: rsim_run(sc, method="RK4"))
    monkeypatch.undo()
    assert _intake_ratio(cap, 190.0) == pytest.approx(expected, rel=1e-9)


@pytest.mark.parametrize("rule", ["ewe", "rpath"])
def test_ecospace_hands_the_derivative_the_same_handling(monkeypatch, rule):
    """rsim_run_spatial builds its own params dict; it must carry the same
    handling matrix as rsim_run under either rule."""
    from pypath.spatial import EcospaceParams, create_1d_grid, rsim_run_spatial

    model, params = _simple_model()
    sc = rsim_scenario(model, params, years=range(1, 3), fast_group_rule=rule)
    flat = _first_call(monkeypatch, lambda: rsim_run(sc, method="RK4"))
    monkeypatch.undo()
    n = sc.params.NUM_GROUPS + 1
    eco = EcospaceParams(
        grid=create_1d_grid(n_patches=1, spacing=1.0),
        habitat_preference=np.ones((n, 1)),
        habitat_capacity=np.ones((n, 1)),
        dispersal_rate=np.zeros(n),
        advection_enabled=np.zeros(n, dtype=bool),
        gravity_strength=np.zeros(n),
    )
    spatial = _first_call(
        monkeypatch, lambda: rsim_run_spatial(sc, ecospace=eco, years=range(1, 3))
    )
    np.testing.assert_array_equal(spatial["P"]["DD"], flat["P"]["DD"])


# EwE 6.7.144 oracle, LT2022 scenario 16 + dataset 35 (runs 42860867155 and
# 42863343066): biomass at the end of months 96 and 108, after Round goby is
# forced to 5.7 and 7.8 in years 8 and 9. Single precision.
EWE_ORACLE_PISCIVORES = {
    "Cod": (0.679357827, 1.07635927),
    "Turbot": (0.123024568, 0.14715293),
    "Perciformes": (0.465022445, 0.552487195),
}


@pytest.mark.integration
def test_piscivores_follow_the_ewe_oracle_through_the_round_goby_boom():
    if not LT2022.exists():
        pytest.skip("LT2022 database not found")
    from pypath.io.ewemdb import EwEDatabaseError, ecosim_scenario_from_ewemdb

    try:
        sc = ecosim_scenario_from_ewemdb(
            str(LT2022), scenario=16, timeseries_dataset=35
        )
    except EwEDatabaseError as e:
        pytest.skip(f"No EwE database driver: {e}")
    out = rsim_run(sc, method="RK4", years=range(1, 10))
    names = [str(n).strip() for n in sc.params.spname]
    for group, (m96, m108) in EWE_ORACLE_PISCIVORES.items():
        i = names.index(group)
        assert out.out_Biomass[96, i] == pytest.approx(m96, rel=5e-3), group
        assert out.out_Biomass[108, i] == pytest.approx(m108, rel=5e-3), group
