"""Ecospace keeps the multi-stanza age structure (#52).

EwE's default Ecospace stanza model keeps one map-wide age structure per
species, updates it each month with the map-averaged consumption and
mortality, and spreads each life stage's new total back over the cells.
With uniform patches that must equal the non-spatial run.
"""

import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from pypath.core.ecopath import rpath
from pypath.core.ecosim import rsim_run, rsim_scenario
from pypath.core.params import create_rpath_params
from pypath.spatial import EcospaceParams, create_1d_grid, rsim_run_spatial

REF = Path(__file__).parent / "data" / "rpath_reference" / "ecopath"

pytestmark = pytest.mark.skipif(
    not (REF / "stanza_indiv.csv").exists(), reason="Rpath reference data missing"
)


@pytest.fixture(scope="module")
def reco():
    """Rpath's REco model: 4 multi-stanza species, 8 life stages."""
    model_df = pd.read_csv(REF / "model_params.csv")
    params = create_rpath_params(model_df["Group"].tolist(), model_df["Type"].tolist())
    params.model = model_df
    params.diet = pd.read_csv(REF / "diet_matrix.csv")
    params.stanzas.stgroups = pd.read_csv(REF / "stanza_groups.csv")
    params.stanzas.stindiv = pd.read_csv(REF / "stanza_indiv.csv")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        model = rpath(params)
    return model, params


def _scenario(reco, effort=2.0):
    model, params = reco
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        scenario = rsim_scenario(model, params, years=range(1, 4))
    scenario.fishing.ForcedEffort[:] = effort
    return scenario


def _uniform_ecospace(ng, n_patches, dispersal, areas=None):
    grid = create_1d_grid(n_patches=n_patches, spacing=1.0)
    if areas is not None:
        grid.patch_areas = np.asarray(areas, dtype=float)
    return EcospaceParams(
        grid=grid,
        habitat_preference=np.ones((ng, n_patches)),
        habitat_capacity=np.ones((ng, n_patches)),
        dispersal_rate=np.full(ng, dispersal),
        advection_enabled=np.zeros(ng, dtype=bool),
        gravity_strength=np.zeros(ng),
    )


def _stanza_indices(reco):
    model, params = reco
    names = list(model.Group)
    return [names.index(g) + 1 for g in params.stanzas.stindiv["Group"]]


@pytest.mark.parametrize(
    "n_patches,dispersal,areas",
    [(1, 0.0, None), (3, 2.0, None), (2, 0.0, [1.0, 3.0])],
    ids=["one-patch", "three-patches-dispersal", "two-unequal-patches"],
)
def test_uniform_density_patches_keep_stanza_dynamics(
    reco, n_patches, dispersal, areas
):
    years = range(1, 3)
    ng = _scenario(reco).params.NUM_GROUPS + 1

    # rsim_run nudges M0 (PyPath-only steps, not in Rpath): once at the start
    # of a run, by at most 1e-4, and again every month unless switched off.
    # Ecospace does neither. Switch the monthly one off and give the spatial
    # run the M0 rsim_run used, so both runs share one model.
    reference_scenario = _scenario(reco)
    reference_scenario.params.MONTHLY_M0_ADJUST = False
    expected = rsim_run(reference_scenario, years=years)
    spatial_scenario = _scenario(reco)
    spatial_scenario.params.MzeroMort = np.array(expected.params["M0"], dtype=float)
    got = rsim_run_spatial(
        spatial_scenario,
        ecospace=_uniform_ecospace(ng, n_patches, dispersal, areas),
        years=years,
    )

    idx = _stanza_indices(reco)
    # The run must leave the start, or the check proves nothing
    assert not np.allclose(expected.out_Biomass[-1, idx], expected.out_Biomass[0, idx])
    np.testing.assert_allclose(
        got.out_Biomass, expected.out_Biomass, rtol=1e-6, atol=1e-10
    )
    assert got.stanza_biomass is not None
    np.testing.assert_allclose(
        got.stanza_biomass, expected.stanza_biomass, rtol=1e-6, atol=1e-10
    )


def test_stanza_biomass_is_the_sum_over_patches(reco):
    years = range(1, 3)
    ng = _scenario(reco).params.NUM_GROUPS + 1
    got = rsim_run_spatial(
        _scenario(reco), ecospace=_uniform_ecospace(ng, 3, 2.0), years=years
    )
    idx = _stanza_indices(reco)
    np.testing.assert_allclose(
        got.stanza_biomass[:, idx],
        got.out_Biomass_spatial[:, idx, :].sum(axis=2),
        rtol=1e-9,
    )
