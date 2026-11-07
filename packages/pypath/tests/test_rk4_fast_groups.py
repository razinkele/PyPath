"""RK4 solves fast-equilibrium groups at equilibrium in every stage (EwE6).

EwE6's rk4 (cEcoSimModel.vb) replaces a NoIntegrate group's stage value by
``(1 - SorWt) * biomeq + SorWt * B`` at each of the four stages and after the
step, with ``biomeq = gain / (loss / B)`` from that stage's derivative. Rpath's
rk4_run integrates every group explicitly and uses the flags only in
Adams-Bashforth, which is why Rpath (and PyPath until now) blow up on a model
with a P/B of 200 and a detritus pool turning over 2850 times a year: the
explicit sub-step of 1/48 year cannot follow them. ``fast_group_rule="rpath"``
keeps Rpath's behaviour for comparisons with Rpath output.
"""

import warnings
from pathlib import Path

import numpy as np
import pytest

from pypath.core.ecopath import rpath
from pypath.core.ecosim import rsim_run, rsim_scenario
from pypath.core.params import create_rpath_params

DB_PATH = Path(__file__).parents[3] / "Data" / "LT2022_0.5ST_final7.eweaccdb"


def _stiff_model():
    """Phyto P/B 300, Zoo P/B 40, a tiny detritus pool: all three fast."""
    params = create_rpath_params(
        groups=["Fish", "Zoo", "Phyto", "Det"], types=[0, 0, 1, 2]
    )
    params.model.loc[0, ["Biomass", "PB", "QB"]] = [1.0, 0.5, 3.0]
    params.model.loc[1, ["Biomass", "PB", "QB"]] = [2.0, 40.0, 150.0]
    params.model.loc[2, ["Biomass", "PB"]] = [10.0, 300.0]
    params.model.loc[3, "Biomass"] = 0.5
    params.model["EE"] = np.nan
    params.model["Det"] = 1.0  # all dead matter goes to Det
    params.model.loc[3, "Det"] = 0.0  # ... except the pool itself
    params.diet["Fish"] = [0.0, 1.0, 0.0, 0.0, 0.0]
    params.diet["Zoo"] = [0.0, 0.0, 1.0, 0.0, 0.0]
    params.diet["Phyto"] = [0.0] * 5
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        model = rpath(params)
    return model, params


def test_fast_groups_are_flagged_in_the_stiff_model():
    model, params = _stiff_model()
    sc = rsim_scenario(model, params, years=range(1, 3))
    names = [str(n) for n in sc.params.spname]
    fast = {names[i] for i, v in enumerate(sc.params.NoIntegrate) if v}
    assert fast == {"Zoo", "Phyto", "Det"}


def test_rk4_keeps_the_stiff_model_at_its_equilibrium():
    """Balanced model, no forcing: nothing moves, under RK4 as under AB."""
    model, params = _stiff_model()
    sc = rsim_scenario(model, params, years=range(1, 3))
    out = rsim_run(sc, method="RK4")
    assert not out.crashed_groups
    bio = out.out_Biomass
    live = bio[0] > 0
    assert np.all(np.isfinite(bio))
    drift = np.max(np.abs(bio[-1, live] / bio[0, live] - 1.0))
    assert drift < 0.05


def test_rpath_rule_leaves_rk4_fast_equilibrium_off():
    """With the Rpath rule RK4 integrates every group, as Rpath's rk4_run."""
    model, params = _stiff_model()
    sc = rsim_scenario(model, params, years=range(1, 3), fast_group_rule="rpath")
    assert sc.params.RK4_FAST_EQUILIBRIUM is False
    sc = rsim_scenario(model, params, years=range(1, 3))
    assert sc.params.RK4_FAST_EQUILIBRIUM is True


@pytest.mark.integration
def test_lt2022_runs_under_rk4_without_crashes():
    if not DB_PATH.exists():
        pytest.skip("LT2022 database not found")
    from pypath.io.ewemdb import EwEDatabaseError, ecosim_scenario_from_ewemdb

    try:
        sc = ecosim_scenario_from_ewemdb(str(DB_PATH), scenario=16)
    except EwEDatabaseError as e:
        pytest.skip(f"No EwE database driver: {e}")
    out = rsim_run(sc, method="RK4")
    assert not out.crashed_groups
    bio = out.annual_Biomass
    n_bio = sc.params.NUM_LIVING + sc.params.NUM_DEAD
    ratio = bio[-1, 1 : n_bio + 1] / bio[0, 1 : n_bio + 1]
    assert np.all(np.isfinite(ratio))
    assert ratio.max() < 20 and ratio.min() > 0.02
