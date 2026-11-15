"""Shared builders for the continuous-pathway tests (REco from Rpath)."""

import copy
import warnings
from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from pypath.core.ecopath import rpath
from pypath.core.ecosim import rsim_scenario
from pypath.core.params import create_rpath_params

REF = Path(__file__).parent / "data" / "rpath_reference" / "ecopath"

needs_reco = pytest.mark.skipif(
    not (REF / "stanza_indiv.csv").exists(), reason="Rpath reference data missing"
)


@lru_cache(maxsize=None)
def _build(stanzas: bool):
    """REco's ``(model, params)``, cached: every caller shares the two
    objects, so they must not be mutated (``scenario`` builds a fresh
    scenario from them)."""
    model_df = pd.read_csv(REF / "model_params.csv")
    params = create_rpath_params(model_df["Group"].tolist(), model_df["Type"].tolist())
    params.model = model_df
    params.diet = pd.read_csv(REF / "diet_matrix.csv")
    if stanzas:
        params.stanzas.stgroups = pd.read_csv(REF / "stanza_groups.csv")
        params.stanzas.stindiv = pd.read_csv(REF / "stanza_indiv.csv")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        model = rpath(params)
    return model, params


def reco_plain():
    """Rpath's REco without stanza tables; returns ``(model, params)``."""
    return _build(False)


def reco_stanzas():
    """Rpath's REco with its stanza tables; returns ``(model, params)``."""
    return _build(True)


def scenario(reco, rule="ewe", effort=2.0, years=range(1, 3)):
    model, params = reco
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        scen = rsim_scenario(model, params, years=years, fast_group_rule=rule)
    scen.fishing.ForcedEffort[:] = effort
    return scen


def run_dicts(scen, method="RK4", years=range(1, 2)):
    """``(state, params_dict, forcing_dict, fishing_dict)`` of month 1 of a run.

    Built with ``rsim_run``'s shared setup (``_run_setup`` and
    ``_month_forcing``), with month 1's forced biomass applied to the state,
    as deep copies that a test may change. ``params_dict["_hden"]`` is EwE's
    start value, not advanced: ``rsim_run`` advances it by one derivative at
    the start biomass before month 1. A forced F is not added. ``method``
    does not change the dicts (every method shares the setup).
    """
    from pypath.core.ecosim import _month_forcing, _run_setup

    setup = _run_setup(scen, years)
    forcing_dict = _month_forcing(scen, setup, 1, setup.ftime)
    state = np.array(setup.state)
    forced = np.asarray(forcing_dict["ForcedBio"]) > 0
    state[forced] = np.asarray(forcing_dict["ForcedBio"])[forced]
    return copy.deepcopy((state, setup.params_dict, forcing_dict, setup.fishing_dict))
