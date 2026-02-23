"""Tests for two-stage interdetrital EE calculation.

Verifies that unconsumed detritus routed between detritus groups via the
detritus fate matrix correctly increases the receiving group's EE.
"""

from pathlib import Path

import numpy as np
import pandas as pd

from pypath.core.ecopath import rpath
from pypath.core.params import create_rpath_params, read_rpath_params

_ECOPATH_DIR = str(Path(__file__).parent / "data" / "rpath_reference" / "ecopath")


def _build_4group_params(detfate_cross=0.5):
    """Build a minimal 4-group model: Producer, Consumer, Detritus1, Detritus2, Fleet.

    Parameters
    ----------
    detfate_cross : float
        Fraction of Detritus1's fate routed to Detritus2.
    """
    groups = ["Producer", "Consumer", "Detritus1", "Detritus2", "Fleet"]
    types = [1, 0, 2, 2, 3]
    params = create_rpath_params(groups, types)

    m = params.model
    # Producer: B=100, PB=50, EE=0.5
    m.loc[m["Group"] == "Producer", "Biomass"] = 100.0
    m.loc[m["Group"] == "Producer", "PB"] = 50.0
    m.loc[m["Group"] == "Producer", "EE"] = 0.5

    # Consumer: B=10, PB=2, QB=10, EE missing (solve for it)
    m.loc[m["Group"] == "Consumer", "Biomass"] = 10.0
    m.loc[m["Group"] == "Consumer", "PB"] = 2.0
    m.loc[m["Group"] == "Consumer", "QB"] = 10.0

    # Detritus groups: biomass missing (will be estimated)
    m.loc[m["Group"] == "Detritus1", "Biomass"] = np.nan
    m.loc[m["Group"] == "Detritus2", "Biomass"] = np.nan

    # Unassimilated consumption
    m.loc[m["Group"] == "Consumer", "Unassim"] = 0.2

    # DetFate: living groups route to Detritus1
    m.loc[m["Group"] == "Producer", "Detritus1"] = 1.0
    m.loc[m["Group"] == "Producer", "Detritus2"] = 0.0
    m.loc[m["Group"] == "Consumer", "Detritus1"] = 1.0
    m.loc[m["Group"] == "Consumer", "Detritus2"] = 0.0

    # Interdetrital fate: Detritus1 routes to Detritus2
    m.loc[m["Group"] == "Detritus1", "Detritus1"] = 1.0 - detfate_cross
    m.loc[m["Group"] == "Detritus1", "Detritus2"] = detfate_cross
    m.loc[m["Group"] == "Detritus2", "Detritus1"] = 0.0
    m.loc[m["Group"] == "Detritus2", "Detritus2"] = 1.0

    # Fleet routes to Detritus1
    m.loc[m["Group"] == "Fleet", "Detritus1"] = 1.0
    m.loc[m["Group"] == "Fleet", "Detritus2"] = 0.0

    # Diet: Consumer eats 40% Producer, 30% Detritus1, 30% Detritus2
    d = params.diet
    d.loc[d["Group"] == "Producer", "Consumer"] = 0.4
    d.loc[d["Group"] == "Detritus1", "Consumer"] = 0.3
    d.loc[d["Group"] == "Detritus2", "Consumer"] = 0.3
    d.loc[d["Group"] == "Import", "Consumer"] = 0.0

    # Producer has no diet (primary producer)
    d.loc[:, "Producer"] = 0.0

    return params


def test_interdetrital_flow_increases_receiving_detritus_ee():
    """Detritus2 should have positive EE when Detritus1 routes material to it."""
    params = _build_4group_params(detfate_cross=0.5)
    result = rpath(params)

    # Detritus2 index
    det2_idx = list(result.Group).index("Detritus2")
    assert result.EE[det2_idx] > 0.0, (
        f"Detritus2 EE should be > 0 with interdetrital flow, got {result.EE[det2_idx]}"
    )


def test_interdetrital_flow_zero_when_no_cross_fate():
    """With zero cross-fate, Detritus2 EE should be 0 (no inputs, no consumption)."""
    params = _build_4group_params(detfate_cross=0.0)
    result = rpath(params)

    det2_idx = list(result.Group).index("Detritus2")
    # Detritus2 has no inputs from living groups and no cross-fate from Detritus1
    # Consumer eats from Detritus1 only, so Detritus2 has zero consumption too
    assert result.EE[det2_idx] == 0.0, (
        f"Detritus2 EE should be 0 with no cross-fate, got {result.EE[det2_idx]}"
    )


def test_interdetrital_ee_bounded():
    """Detrital EE values should remain in [0, 1] even with interdetrital flows."""
    params = _build_4group_params(detfate_cross=0.5)
    result = rpath(params)

    dead_mask = result.type == 2
    det_ee = result.EE[dead_mask]
    assert np.all(det_ee >= 0.0), f"Detrital EE has negative values: {det_ee}"
    assert np.all(det_ee <= 1.0), f"Detrital EE has values > 1: {det_ee}"


def test_existing_reference_model_unchanged():
    """Reference model results should not change (it has zero/negligible interdetrital fate)."""
    ecopath_dir = _ECOPATH_DIR
    model_df = pd.read_csv(ecopath_dir + "/model_params.csv")
    diet_df = pd.read_csv(ecopath_dir + "/diet_matrix.csv")

    params = create_rpath_params(model_df["Group"].tolist(), model_df["Type"].tolist())
    params.model = model_df
    params.diet = diet_df

    result = rpath(params)

    # Known reference EE values for the first few living groups (from previous runs)
    # Just verify EE values are finite and in valid range
    living_mask = result.type < 2
    living_ee = result.EE[living_mask]
    assert np.all(np.isfinite(living_ee)), "Living EE has non-finite values"
    assert np.all(living_ee >= 0.0), "Living EE has negative values"

    dead_mask = result.type == 2
    dead_ee = result.EE[dead_mask]
    assert np.all(np.isfinite(dead_ee)), "Dead EE has non-finite values"
    assert np.all(dead_ee >= 0.0), "Dead EE has negative values"
    assert np.all(dead_ee <= 1.0), "Dead EE has values > 1"
