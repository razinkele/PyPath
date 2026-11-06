"""Warn about detritus pools whose detritus fate points to themselves."""

import warnings
from pathlib import Path

import pandas as pd
import pytest

from pypath.core.ecopath import rpath
from pypath.core.ecosim import detritus_self_fates, rsim_scenario
from pypath.core.params import create_rpath_params

RE = Path(__file__).parent / "data" / "rpath_reference" / "ecopath"


def _reco_params():
    model_df = pd.read_csv(RE / "model_params.csv")
    params = create_rpath_params(
        model_df["Group"].tolist(), [int(t) for t in model_df["Type"].tolist()]
    )
    params.model = model_df
    params.diet = pd.read_csv(RE / "diet_matrix.csv")
    return params


def _self_fate_warnings(params):
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        rsim_scenario(rpath(params), params, years=range(1, 3))
    return [w for w in caught if "to itself" in str(w.message)]


def test_reco_has_no_detritus_self_fate():
    params = _reco_params()
    model = rpath(params)
    assert detritus_self_fates(model, model.NUM_LIVING, model.NUM_DEAD) == {}
    assert _self_fate_warnings(params) == []


def test_detritus_flowing_to_itself_warns():
    params = _reco_params()
    det_row = params.model.index[params.model["Group"] == "Detritus"][0]
    params.model.loc[det_row, "Detritus"] = 1.0
    model = rpath(params)
    assert detritus_self_fates(model, model.NUM_LIVING, model.NUM_DEAD) == {
        "Detritus": pytest.approx(1.0)
    }
    found = _self_fate_warnings(params)
    assert len(found) == 1
    assert "Detritus (1)" in str(found[0].message)
