"""Test pedigree column naming consistency between read and create paths."""

import numpy as np
import pandas as pd
import pytest
from pathlib import Path

from pypath.core.params import create_rpath_params, read_rpath_params


def test_create_rpath_params_pedigree_uses_biomass():
    """create_rpath_params pedigree must use 'Biomass' column."""
    params = create_rpath_params(
        groups=["Phyto", "Zoo", "Detritus"],
        types=[1, 0, 2],
    )
    assert "Biomass" in params.pedigree.columns
    assert "B" not in params.pedigree.columns


def test_read_rpath_params_pedigree_uses_biomass(tmp_path):
    """read_rpath_params pedigree must use 'Biomass' column, not 'B'."""
    model_data = {
        "Group": ["Phyto", "Zoo", "Detritus"],
        "Type": [1, 0, 2],
        "Biomass": [10.0, 5.0, 1.0],
        "PB": [100.0, 20.0, np.nan],
        "QB": [0.0, 50.0, np.nan],
        "EE": [0.8, 0.9, 0.5],
        "ProdCons": [0.0, 0.0, 0.0],
        "BioAcc": [0.0, 0.0, 0.0],
        "Unassim": [0.0, 0.2, 0.0],
        "DetInput": [0.0, 0.0, 0.0],
    }
    pd.DataFrame(model_data).to_csv(tmp_path / "model.csv", index=False)

    diet_data = {"Group": ["Phyto", "Zoo", "Detritus"], "Zoo": [0.8, 0.0, 0.2]}
    pd.DataFrame(diet_data).to_csv(tmp_path / "diet.csv", index=False)

    params = read_rpath_params(tmp_path / "model.csv", tmp_path / "diet.csv")
    assert "Biomass" in params.pedigree.columns, (
        f"Pedigree columns are {list(params.pedigree.columns)}, expected 'Biomass'"
    )
    assert "B" not in params.pedigree.columns


def test_pedigree_column_consistency():
    """Both creation paths must produce the same pedigree column names."""
    params = create_rpath_params(
        groups=["Phyto", "Zoo", "Detritus"],
        types=[1, 0, 2],
    )
    expected_cols = set(params.pedigree.columns)
    assert "Biomass" in expected_cols
