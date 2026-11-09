"""Forced biomass replaces a group's state, as in EwE6.

EwE6 (setForcedBiomass, before and after each rk4 step) sets the group's
biomass to the forced value for the data year; the derivative is evaluated
with that biomass, and the group is simply not integrated. Substituting the
value only inside the derivative while the state keeps integrating its own
rate lets a group forced far from its start blow up (LT2022: Round goby forced
from 0.03 to 7.8).
"""

import warnings

import numpy as np

from pypath.core.ecopath import rpath
from pypath.core.ecosim import rsim_run, rsim_scenario
from pypath.core.params import create_rpath_params

PHYTO, ZOO = 1, 2


def _model():
    params = create_rpath_params(groups=["Phyto", "Zoo", "Det"], types=[1, 0, 2])
    params.model.loc[0, ["Biomass", "PB", "QB", "EE"]] = [10.0, 2.0, 0.0, 0.5]
    params.model.loc[1, ["Biomass", "PB", "QB", "EE"]] = [2.0, 1.0, 5.0, 0.8]
    params.model.loc[2, ["Biomass", "PB", "QB", "EE"]] = [10.0, 0.0, 0.0, 0.0]
    params.diet["Zoo"] = [1.0, 0.0, 0.0, 0.0]
    params.diet["Phyto"] = [0.0, 0.0, 0.0, 0.0]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return rpath(params), params


def _forced_run(method):
    model, params = _model()
    sc = rsim_scenario(model, params, years=range(1, 4))
    forced = np.full(sc.forcing.ForcedBio.shape[0], -1.0)
    forced[:12] = 2.0  # year 1: the Ecopath value
    forced[12:24] = 8.0  # year 2: four times the start
    forced[24:] = 4.0  # year 3
    sc.forcing.ForcedBio[:, ZOO] = forced
    return rsim_run(sc, method=method), forced


def test_forced_group_holds_its_series_under_rk4():
    out, forced = _forced_run("RK4")
    bio = out.out_Biomass[:, ZOO]
    assert not out.crashed_groups
    np.testing.assert_allclose(bio[1:25], forced[:24], rtol=1e-9)
    np.testing.assert_allclose(bio[25:], 4.0, rtol=1e-9)


def test_forced_group_holds_its_series_under_ab():
    out, forced = _forced_run("AB")
    bio = out.out_Biomass[:, ZOO]
    assert not out.crashed_groups
    np.testing.assert_allclose(bio[1:25], forced[:24], rtol=1e-9)


def test_other_groups_respond_to_the_forced_biomass():
    """Four times the grazer depresses its prey relative to the base run."""
    model, params = _model()
    base = rsim_run(rsim_scenario(model, params, years=range(1, 4)), method="RK4")
    out, _ = _forced_run("RK4")
    assert out.annual_Biomass[2, PHYTO] < 0.99 * base.annual_Biomass[2, PHYTO]
