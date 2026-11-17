"""Shared pytest configuration for pypath-shiny tests.

The pypath-shiny package should be installed via `pip install -e packages/pypath-shiny`.
"""

import json

import pandas as pd
import pytest
from pypath.core.ecopath import rpath
from pypath.core.ecosim import rsim_scenario
from pypath.core.params import create_rpath_params


@pytest.fixture(scope="session")
def rpath_params():
    """Minimal 3-group RpathParams (Fish/Plankton/Detritus)."""
    params = create_rpath_params(["Fish", "Plankton", "Detritus"], [0, 1, 2])
    params.model.loc[0, "Biomass"] = 10.0
    params.model.loc[0, "PB"] = 1.0
    params.model.loc[0, "QB"] = 5.0
    params.model.loc[0, "EE"] = 0.8
    params.model.loc[1, "Biomass"] = 5.0
    params.model.loc[1, "PB"] = 50.0
    params.model.loc[1, "EE"] = 0.6
    params.model.loc[2, "Biomass"] = 1.0
    params.diet.iloc[1, 1] = 1.0  # Plankton eaten by Fish
    return params


@pytest.fixture(scope="session")
def balanced_rpath_model(rpath_params):
    """Balanced Rpath model from the 3-group params."""
    return rpath(rpath_params)


@pytest.fixture
def sample_model_df():
    """DataFrame matching Ecopath model table structure, including 9999 sentinels."""
    return pd.DataFrame(
        {
            "Group": ["Fish", "Plankton", "Detritus"],
            "Type": [0, 1, 2],
            "Biomass": [10.123, 9999, 1.0],
            "PB": [1.0, 50.0, 0.0],
            "QB": [5.0, 9999, 9999],
            "EE": [0.8, 0.6, 9999],
        }
    )


@pytest.fixture
def tmp_diag_dir(tmp_path):
    """Temporary diagnostics directory with valid meta.json + CSVs."""
    d = tmp_path / "diag"
    d.mkdir()
    meta = {"qq_provided": True, "note": "test note", "version": "1.0"}
    (d / "meta.json").write_text(json.dumps(meta))
    pd.DataFrame({"group": ["Fish"], "value": [1.0]}).to_csv(
        d / "seabirds_qq_rk4.csv", index=False
    )
    pd.DataFrame({"group": ["Fish"], "comp": [0.5]}).to_csv(
        d / "seabirds_components_rk4.csv", index=False
    )
    return d


@pytest.fixture
def switching_scenario():
    """Balanced Phyto/Cod/Seals/Detritus scenario (function scoped: tests edit it).

    Cod eats two prey (Phyto 0.7, Detritus 0.3) so that it can switch;
    Seals eat Cod only.
    """
    params = create_rpath_params(["Phyto", "Cod", "Seals", "Detritus"], [1, 0, 0, 2])
    m = params.model
    for i, (biomass, pb, qb, ee) in enumerate(
        [(10.0, 40.0, None, 0.9), (2.0, 1.0, 5.0, None), (0.1, 0.3, 10.0, None)]
    ):
        m.loc[i, "Biomass"] = biomass
        m.loc[i, "PB"] = pb
        if qb is not None:
            m.loc[i, "QB"] = qb
        if ee is not None:
            m.loc[i, "EE"] = ee
    m.loc[3, "Biomass"] = 5.0
    m["Unassim"] = [0.0, 0.2, 0.2, 0.2]
    m["DetInput"] = [0.0, 0.0, 0.0, 1.0]
    m["Detritus"] = [1.0, 1.0, 1.0, 0.0]
    params.diet.loc[0, "Cod"] = 0.7
    params.diet.loc[3, "Cod"] = 0.3
    params.diet.loc[1, "Seals"] = 1.0
    return rsim_scenario(rpath(params), params, years=range(1, 3))
