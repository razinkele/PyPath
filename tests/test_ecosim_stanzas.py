import numpy as np
import pytest

from pypath.core.ecosim import rsim_run, rsim_state, rsim_params
from pypath.core.stanzas import StanzaGroup, create_stanza_params
from pypath.core.params import RpathParams


def make_simple_params_with_stanzas():
    # Minimal params stub: two groups, one with stanzas
    params = RpathParams()
    # Create a simple model DataFrame-like structure expected by the simulation
    params.model = [
        {"Group": "Phytoplankton", "Type": 1},
        {"Group": "Zooplankton", "Type": 2},
    ]
    # Add stanza info for Zooplankton (2 stanzas)
    stanza = StanzaGroup(group="Zooplankton", n_stanzas=2)
    params.stanzas = create_stanza_params([stanza])
    params.n_groups = 2
    return params


def test_rsim_handles_stanzas():
    params = make_simple_params_with_stanzas()

    years = range(1)
    rstate = rsim_state(rsim_params(params))

    # Expect that running a short simulation does not raise and outputs per-stanza info
    out = rsim_run(rstate, years)

    # New behavior requirement: output should include stanza-resolved biomass shape
    assert hasattr(out, "biomass_by_stanza") or hasattr(out, "stanza_biomass"), (
        "Ecosim output must include stanza-resolved biomass (e.g., 'biomass_by_stanza')"
    )
