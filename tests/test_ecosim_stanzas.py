import numpy as np
import pandas as pd

from pypath.core.ecosim import rsim_run, rsim_scenario
from pypath.core.stanzas import StanzaGroup, StanzaIndividual, create_stanza_params
from pypath.core.params import create_rpath_params, RpathParams
from pypath.core.ecopath import rpath


def make_simple_rpath_with_stanzas():
    # Two groups: Phytoplankton (producer) and Zooplankton (consumer with 2 stanzas)
    groups = ['Phytoplankton', 'Zoo_Juv', 'Zoo_Adult']
    types = [1, 0, 0]

    params = create_rpath_params(groups, types, stgroups=[None, 'Zoo', 'Zoo'])

    # Fill basic model values
    params.model['Biomass'] = [10.0, 1.0, 4.0]
    params.model['PB'] = [10.0, 2.0, 2.0]
    params.model['QB'] = [0.0, 10.0, 10.0]
    params.model['EE'] = [1.0, 0.8, 0.8]
    params.model['Unassim'] = [0.0, 0.2, 0.2]

    # Create a simple diet: Phytoplankton eaten by juvenile and adult zoo
    diet = params.diet.copy()
    diet.loc[diet['Group'] == 'Phytoplankton', 'Zoo_Juv'] = 0.5
    diet.loc[diet['Group'] == 'Phytoplankton', 'Zoo_Adult'] = 0.5
    params.diet = diet

    # Define stanza groups and individuals
    groups_def = [{'stanza_group_num': 1, 'n_stanzas': 2, 'vbgf_ksp': 0.3}]
    indivs = [
        {'stanza_group_num': 1, 'stanza_num': 1, 'group_num': 2, 'group_name': 'Zoo_Juv', 'first': 0, 'last': 11, 'z': 1.0, 'leading': False},
        {'stanza_group_num': 1, 'stanza_num': 2, 'group_num': 3, 'group_name': 'Zoo_Adult', 'first': 12, 'last': 60, 'z': 0.5, 'leading': True},
    ]
    params.stanzas = create_stanza_params(groups_def, indivs)

    return params


def test_rsim_handles_stanzas():
    rparams = make_simple_rpath_with_stanzas()
    r = rpath(rparams, eco_name='Test')

    years = range(1, 3)
    scen = rsim_scenario(r, rparams, years=years)
    out = rsim_run(scen, years=years)

    assert hasattr(out, 'stanza_biomass') and out.stanza_biomass is not None, 'Ecosim output must include stanza-resolved biomass (stanza_biomass)'
