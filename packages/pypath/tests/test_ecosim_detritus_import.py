"""Ecosim adds detritus import as a constant inflow (audit A2).

EwE: DetInFlow = DtImp * DtImpMult + Immig, deriv = DetInFlow + ToDetritus -
loss, and biomeq uses DetInFlow + ToDetritus (cEcoSimModel.vb:3119-3126). The
export rate is set at the first call from flows that include DtImp (:3207),
so a balanced Ecopath model starts at equilibrium. PyPath puts DetInput into
the Ecopath detritus inputs (so into M0_det); without the inflow in Ecosim,
detritus falls from t = 0.
"""

import warnings
from pathlib import Path

import numpy as np
import pytest

from pypath.core.ecopath import rpath
from pypath.core.ecosim import (
    _build_active_link_matrix,
    _build_link_matrix,
    rsim_params,
    rsim_run,
    rsim_scenario,
)
from pypath.core.ecosim_deriv import deriv_vector
from pypath.core.link_array import ActiveLinkArray
from pypath.core.params import create_rpath_params

DB_PATH = Path(__file__).parents[3] / "Data" / "LT2022_0.5ST_final7.eweaccdb"

# rsim index: 0 Outside, 1 Fish, 2 Zoo, 3 Phyto, 4 Det
FISH, ZOO, PHYTO, DET = 1, 2, 3, 4


def _model(det_input=0.0, fish_ba=0.0, det_biomass=10.0):
    params = create_rpath_params(
        groups=["Fish", "Zoo", "Phyto", "Det"], types=[0, 0, 1, 2]
    )
    params.model.loc[0, ["Biomass", "PB", "QB", "EE"]] = [1.0, 0.5, 3.0, np.nan]
    params.model.loc[1, ["Biomass", "PB", "QB", "EE"]] = [2.0, 1.5, 5.0, np.nan]
    params.model.loc[2, ["Biomass", "PB", "EE"]] = [10.0, 2.0, np.nan]
    params.model.loc[3, "Biomass"] = det_biomass
    params.model.loc[0, "EE"] = np.nan  # top predator: EE follows from BA
    params.model.loc[1, "EE"] = np.nan
    params.model.loc[2, "EE"] = np.nan
    params.model.loc[0, "BioAcc"] = fish_ba
    params.model.loc[3, "DetInput"] = det_input
    params.model["Det"] = 1.0
    params.model.loc[3, "Det"] = 0.0
    params.diet["Fish"] = [0.0, 1.0, 0.0, 0.0, 0.0]
    params.diet["Zoo"] = [0.0, 0.0, 0.7, 0.3, 0.0]
    params.diet["Phyto"] = [0.0] * 5
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        model = rpath(params)
    return model, params


def _params_dict(sim):
    d = {
        "NUM_GROUPS": sim.NUM_GROUPS,
        "NUM_LIVING": sim.NUM_LIVING,
        "NUM_DEAD": sim.NUM_DEAD,
        "NUM_GEARS": sim.NUM_GEARS,
        "PB": sim.PBopt,
        "QB": sim.FtimeQBOpt,
        "M0": sim.MzeroMort,
        "Unassim": sim.UnassimRespFrac,
        "ActiveLink": _build_active_link_matrix(sim),
        "VV": _build_link_matrix(sim, sim.VV),
        "DD": _build_link_matrix(sim, sim.DD),
        "QQbase": _build_link_matrix(sim, sim.QQ),
        "Bbase": sim.B_BaseRef,
        "PP_type": sim.PP_type,
        "NoIntegrate": np.zeros_like(sim.NoIntegrate),
        "FishFrom": sim.FishFrom,
        "FishTo": getattr(sim, "FishTo", np.array([])),
        "FishQ": sim.FishQ,
        "fleet_idx": getattr(sim, "fleet_idx", None),
        "DetLinkFrom": sim.DetFrom,
        "DetLinkTo": sim.DetTo,
        "DetLinkFrac": sim.DetFrac,
        "RpathPrimaryProduction": True,
        "DetInput": getattr(sim, "DetInput", None),
    }
    links = ActiveLinkArray.from_bool_matrix(d["ActiveLink"])
    d["_link_prey"] = links.prey
    d["_link_pred"] = links.pred
    return d


def _t0_deriv(sim):
    n = sim.NUM_GROUPS
    forcing = {
        "Ftime": np.ones(n + 1),
        "ForcedPrey": np.ones(n + 1),
        "PP_forcing": np.ones(n + 1),
        "ForcedBio": np.zeros(n + 1),
        "ForcedMigrate": np.zeros(n + 1),
        "ForcedEffort": np.ones(2),
    }
    fishing = {
        "FishFrom": sim.FishFrom,
        "FishThrough": sim.FishThrough,
        "FishQ": sim.FishQ,
        "FishingMort": np.zeros(n + 1),
    }
    return deriv_vector(sim.B_BaseRef.copy(), _params_dict(sim), forcing, fishing)


def _det_inputs(model):
    i = list(model.Group).index("Det")
    return model.Biomass[i] * model.PB[i]


# --- DB-free -------------------------------------------------------------------


def test_rpath_carries_det_input():
    model, _ = _model(det_input=50.0)
    i = list(model.Group).index("Det")
    assert model.DetInput[i] == pytest.approx(50.0)
    assert model.DetInput[0] == 0.0


def test_rsim_params_carries_det_input():
    model, _ = _model(det_input=50.0)
    sim = rsim_params(model)
    assert sim.DetInput[DET] == pytest.approx(50.0)
    assert sim.DetInput[FISH] == 0.0


@pytest.mark.parametrize("det_input", [0.0, 50.0])
def test_t0_detritus_balanced(det_input):
    model, _ = _model(det_input=det_input)
    d = _t0_deriv(rsim_params(model))
    assert abs(d[DET]) < 1e-9 * _det_inputs(model)


def test_t0_living_derivative_equals_ba():
    model, _ = _model(fish_ba=0.01)
    d = _t0_deriv(rsim_params(model))
    assert d[FISH] == pytest.approx(0.01, abs=1e-9)


@pytest.mark.parametrize("method", ["AB", "RK4"])
@pytest.mark.parametrize("noint", [True, False])
def test_detritus_flat_over_a_year(method, noint):
    model, params = _model(det_input=50.0)
    scen = rsim_scenario(model, params, years=range(1, 3))
    scen.params.NoIntegrate[DET] = 1 if noint else 0  # PyPath: non-zero = fast eq.
    out = rsim_run(scen, method=method)
    det = out.out_Biomass[:, DET]
    assert abs(det[12] / det[0] - 1) < 1e-6


@pytest.mark.parametrize("det_input, fast", [(0.0, 0), (50.0, 1)])
def test_fast_group_rule_counts_det_import(det_input, fast):
    # EwE: rrate = |loss| / B > 24 (cEcoSimModel.vb:2795), with the detritus
    # loss set from flows that include DtImp (:3207). Det B = 2: inflow from
    # flows is about 13.5 (6.75 /yr); with the import it is 63.5 (31.75 /yr).
    model, _ = _model(det_input=det_input, det_biomass=2.0)
    assert int(rsim_params(model).NoIntegrate[DET] != 0) == fast


def test_det_input_zero_is_bit_identical():
    model, params = _model(det_input=0.0)
    scen = rsim_scenario(model, params, years=range(1, 3))
    with_key = rsim_run(scen, method="RK4").out_Biomass
    scen.params.DetInput = None
    without = rsim_run(scen, method="RK4").out_Biomass
    assert np.array_equal(with_key, without)


# --- LT2022 (integration) ------------------------------------------------------


@pytest.fixture(scope="module")
def lt2022():
    if not DB_PATH.exists():
        pytest.skip("LT2022 database not found")
    from pypath.io.ewemdb import EwEDatabaseError, read_ewemdb

    try:
        params = read_ewemdb(str(DB_PATH))
    except EwEDatabaseError as e:
        pytest.skip(f"No EwE database driver: {e}")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        model = rpath(params)
    return model, rsim_params(model)


@pytest.mark.integration
def test_lt2022_t0_birds_equal_ba(lt2022):
    model, sim = lt2022
    d = _t0_deriv(sim)
    for name in ("Scoters", "Long-tailed duck"):
        i = list(model.Group).index(name)
        assert d[i + 1] == pytest.approx(model.BA[i], abs=1e-8), name


@pytest.mark.integration
def test_lt2022_t0_detritus_balanced(lt2022):
    model, sim = lt2022
    i = list(model.Group).index("Detritus")
    assert abs(_t0_deriv(sim)[i + 1]) < 1e-3


def test_rpath_to_params_keeps_det_input():
    from pypath.io.ewe_writer import _rpath_to_params

    model, _ = _model(det_input=50.0)
    m = _rpath_to_params(model).model.set_index("Group")
    assert m.loc["Det", "DetInput"] == pytest.approx(50.0)


def test_ecospace_detritus_flat_with_import():
    """Unequal patch areas exercise the per-patch density scaling."""
    from pypath.spatial import EcospaceParams, create_1d_grid, rsim_run_spatial

    model, params = _model(det_input=50.0)
    sc = rsim_scenario(model, params, years=range(1, 3))
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
    out = rsim_run_spatial(sc, ecospace=eco, years=range(1, 3))
    # each patch gets its area share of the import, so each stays flat
    det_patch = out.out_Biomass_spatial[:, DET, :]
    np.testing.assert_allclose(det_patch[12] / det_patch[0], 1.0, atol=1e-6)
    det = out.out_Biomass[:, DET]
    assert abs(det[12] / det[0] - 1) < 1e-6
