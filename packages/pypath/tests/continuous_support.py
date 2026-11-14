"""Shared builders for the continuous-pathway tests (REco from Rpath)."""

import copy
import warnings
from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from pypath.core.ecopath import rpath
from pypath.core.ecosim import rsim_run, rsim_scenario
from pypath.core.params import create_rpath_params

REF = Path(__file__).parent / "data" / "rpath_reference" / "ecopath"

needs_reco = pytest.mark.skipif(
    not (REF / "stanza_indiv.csv").exists(), reason="Rpath reference data missing"
)


@lru_cache(maxsize=None)
def _build(stanzas: bool):
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


class _Captured(Exception):
    def __init__(self, args_):
        super().__init__("captured")
        self.args_ = args_


def run_dicts(scen, method="RK4"):
    """``(state, params_dict, forcing_dict, fishing_dict)`` of ``rsim_run``.

    Captures the arguments of its first ``integrate_rk4`` call and aborts.
    """
    import pypath.core.ecosim_deriv as deriv

    original = deriv.integrate_rk4

    def stub(*args, **kwargs):
        raise _Captured(copy.deepcopy(args[:4]))

    deriv.integrate_rk4 = stub
    try:
        rsim_run(scen, method=method, years=range(1, 2))
    except _Captured as captured:
        state, params_dict, forcing_dict, fishing_dict = captured.args_
        return np.array(state), params_dict, forcing_dict, fishing_dict
    finally:
        deriv.integrate_rk4 = original
    raise RuntimeError("rsim_run never called integrate_rk4")
