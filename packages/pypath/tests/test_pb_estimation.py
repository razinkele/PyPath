"""Tests for PB estimation from B+EE.

Verifies that the solver can estimate PB when a group has known Biomass
and EE but missing PB.
"""

from pathlib import Path

import numpy as np
import pandas as pd

from pypath.core.ecopath import rpath
from pypath.core.params import create_rpath_params

_ECOPATH_DIR = str(Path(__file__).parent / "data" / "rpath_reference" / "ecopath")


def _build_simple_model():
    """Build a simple 5-group model for PB estimation tests.

    Groups: Producer, Consumer1 (known PB), Consumer2 (PB to estimate),
    Detritus, Fleet
    """
    groups = ["Producer", "Consumer1", "Consumer2", "Detritus", "Fleet"]
    types = [1, 0, 0, 2, 3]
    params = create_rpath_params(groups, types)

    m = params.model
    # Producer
    m.loc[m["Group"] == "Producer", "Biomass"] = 100.0
    m.loc[m["Group"] == "Producer", "PB"] = 50.0
    m.loc[m["Group"] == "Producer", "EE"] = 0.5

    # Consumer1: fully specified
    m.loc[m["Group"] == "Consumer1", "Biomass"] = 10.0
    m.loc[m["Group"] == "Consumer1", "PB"] = 2.0
    m.loc[m["Group"] == "Consumer1", "QB"] = 10.0
    m.loc[m["Group"] == "Consumer1", "EE"] = 0.9

    # Consumer2: B+EE known, PB missing, QB known
    m.loc[m["Group"] == "Consumer2", "Biomass"] = 5.0
    m.loc[m["Group"] == "Consumer2", "PB"] = np.nan
    m.loc[m["Group"] == "Consumer2", "QB"] = 8.0
    m.loc[m["Group"] == "Consumer2", "EE"] = 0.8

    # Detritus
    m.loc[m["Group"] == "Detritus", "Biomass"] = np.nan

    # Unassim
    m.loc[m["Group"] == "Consumer1", "Unassim"] = 0.2
    m.loc[m["Group"] == "Consumer2", "Unassim"] = 0.2

    # Fishing catch on Consumer2 (needed so b_vec > 0 for PB estimation)
    m.loc[m["Group"] == "Consumer2", "Fleet"] = 1.0

    # DetFate: everything to Detritus
    for g in ["Producer", "Consumer1", "Consumer2", "Fleet"]:
        m.loc[m["Group"] == g, "Detritus"] = 1.0
    m.loc[m["Group"] == "Detritus", "Detritus"] = 1.0

    # Diet
    d = params.diet
    d.loc[:, "Producer"] = 0.0  # primary producer

    # Consumer1 eats Producer
    d.loc[d["Group"] == "Producer", "Consumer1"] = 1.0
    d.loc[d["Group"] == "Import", "Consumer1"] = 0.0

    # Consumer2 eats Consumer1 and Producer
    d.loc[d["Group"] == "Producer", "Consumer2"] = 0.5
    d.loc[d["Group"] == "Consumer1", "Consumer2"] = 0.5
    d.loc[d["Group"] == "Import", "Consumer2"] = 0.0

    # Zero out remaining diet entries
    for prey in ["Consumer2", "Detritus"]:
        d.loc[d["Group"] == prey, "Consumer1"] = 0.0
        d.loc[d["Group"] == prey, "Consumer2"] = 0.0
    d.loc[d["Group"] == "Consumer1", "Consumer1"] = 0.0

    return params


def test_pb_estimated_from_b_and_ee():
    """Consumer2 with B=5, EE=0.8, PB=NaN should have PB estimated."""
    params = _build_simple_model()
    result = rpath(params)

    c2_idx = list(result.Group).index("Consumer2")
    estimated_pb = result.PB[c2_idx]

    assert np.isfinite(estimated_pb), f"PB should be finite, got {estimated_pb}"
    assert estimated_pb > 0.0, f"PB should be positive, got {estimated_pb}"


def test_pb_estimation_consistent_with_known_pb():
    """Estimated PB should match when we know the true PB value."""
    params = _build_simple_model()

    # First run with all PBs known to get the true PB for Consumer2
    m = params.model
    # Set a known PB and solve for EE instead
    m.loc[m["Group"] == "Consumer2", "PB"] = 3.0
    m.loc[m["Group"] == "Consumer2", "EE"] = np.nan
    result_known = rpath(params)
    c2_idx = list(result_known.Group).index("Consumer2")
    known_ee = result_known.EE[c2_idx]

    # Now set EE to the solved value and remove PB to force estimation
    params2 = _build_simple_model()
    m2 = params2.model
    m2.loc[m2["Group"] == "Consumer2", "PB"] = np.nan
    m2.loc[m2["Group"] == "Consumer2", "EE"] = known_ee
    result_estimated = rpath(params2)
    estimated_pb = result_estimated.PB[c2_idx]

    np.testing.assert_allclose(
        estimated_pb,
        3.0,
        rtol=0.01,
        err_msg=f"Estimated PB {estimated_pb} should match known PB 3.0",
    )


def test_pb_estimation_recalculates_qb():
    """When both PB and QB are missing (only ProdCons given), both should be estimated."""
    params = _build_simple_model()
    m = params.model

    # Consumer2: B, EE known; PB and QB missing; ProdCons given
    m.loc[m["Group"] == "Consumer2", "PB"] = np.nan
    m.loc[m["Group"] == "Consumer2", "QB"] = np.nan
    m.loc[m["Group"] == "Consumer2", "ProdCons"] = 0.25  # GE = PB/QB

    result = rpath(params)
    c2_idx = list(result.Group).index("Consumer2")

    estimated_pb = result.PB[c2_idx]
    estimated_qb = result.QB[c2_idx]

    assert np.isfinite(estimated_pb), f"PB should be finite, got {estimated_pb}"
    assert estimated_pb > 0.0, f"PB should be positive, got {estimated_pb}"
    assert np.isfinite(estimated_qb), f"QB should be finite, got {estimated_qb}"
    assert estimated_qb > 0.0, f"QB should be positive, got {estimated_qb}"

    # GE consistency: PB/QB should approximately equal ProdCons
    ge_ratio = estimated_pb / estimated_qb
    np.testing.assert_allclose(
        ge_ratio,
        0.25,
        rtol=0.05,
        err_msg=f"PB/QB ratio {ge_ratio} should match ProdCons 0.25",
    )


def test_pb_estimation_does_not_affect_other_groups():
    """Running reference model with all PBs known should give unchanged results."""
    ecopath_dir = _ECOPATH_DIR
    model_df = pd.read_csv(ecopath_dir + "/model_params.csv")
    diet_df = pd.read_csv(ecopath_dir + "/diet_matrix.csv")

    params = create_rpath_params(model_df["Group"].tolist(), model_df["Type"].tolist())
    params.model = model_df
    params.diet = diet_df

    result = rpath(params)

    # All living groups should have valid results
    living_mask = result.type < 2
    assert np.all(
        np.isfinite(result.Biomass[living_mask])
    ), "Biomass has non-finite values"
    assert np.all(np.isfinite(result.PB[living_mask])), "PB has non-finite values"
    assert np.all(np.isfinite(result.EE[living_mask])), "EE has non-finite values"
    assert np.all(result.Biomass[living_mask] >= 0), "Biomass has negative values"
    assert np.all(result.PB[living_mask] >= 0), "PB has negative values"
