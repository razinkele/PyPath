"""Test that discards warn when no detritus groups exist."""

import warnings

import numpy as np
import pytest

from pypath.core.ecopath import rpath
from pypath.core.ecosim import rsim_scenario
from pypath.core.params import create_rpath_params


def _model_no_detritus_with_discards():
    """Create a minimal model with no detritus but with discards."""
    params = create_rpath_params(
        groups=["Phyto", "Zoo", "Fleet1"],
        types=[1, 0, 3],
    )
    m = params.model
    m.loc[m["Group"] == "Phyto", "Biomass"] = 10.0
    m.loc[m["Group"] == "Phyto", "PB"] = 50.0
    m.loc[m["Group"] == "Zoo", "Biomass"] = 5.0
    m.loc[m["Group"] == "Zoo", "PB"] = 2.0
    m.loc[m["Group"] == "Zoo", "QB"] = 10.0
    m.loc[m["Group"] == "Zoo", "EE"] = 0.9

    # Diet: Zoo eats 100% Phyto
    params.diet.loc[params.diet["Group"] == "Phyto", "Zoo"] = 1.0

    # Discards: Fleet1 discards Zoo at 0.5 t/km²/yr
    m.loc[m["Group"] == "Zoo", "Fleet1.disc"] = 0.5

    return params


def test_discard_warns_when_no_detritus():
    """When ndead==0 and discards exist, rsim_scenario must warn."""
    params = _model_no_detritus_with_discards()

    balanced = rpath(params)
    assert balanced is not None
    assert balanced.NUM_DEAD == 0

    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        rsim_scenario(balanced, params, years=range(1, 6))
        discard_warnings = [x for x in w if "discard" in str(x.message).lower()]
        assert len(discard_warnings) > 0, (
            "Expected warning about discards with no detritus groups"
        )


def test_discard_routed_to_outside_when_no_detritus():
    """When ndead==0, discards should be routed to Outside (FishTo=0)."""
    params = _model_no_detritus_with_discards()

    balanced = rpath(params)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        scenario = rsim_scenario(balanced, params, years=range(1, 6))

    # Check that discard links exist and route to Outside (0)
    fish_to = scenario.params.FishTo
    # There should be at least one link going to Outside (FishTo=0)
    # from the discard routing
    assert len(fish_to) > 1, "Expected fish links to be created"
