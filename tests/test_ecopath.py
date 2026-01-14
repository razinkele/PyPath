"""
Tests for PyPath core functionality.
"""

import pytest
import numpy as np
import pandas as pd

from pypath.core.params import (
    RpathParams,
    create_rpath_params,
    check_rpath_params,
)
from pypath.core.ecopath import Rpath, rpath


class TestCreateRpathParams:
    """Tests for create_rpath_params function."""

    def test_basic_creation(self):
        """Test basic parameter creation."""
        groups = ["Phyto", "Zoo", "Fish", "Detritus", "Fleet"]
        types = [1, 0, 0, 2, 3]

        params = create_rpath_params(groups, types)

        assert isinstance(params, RpathParams)
        assert len(params.model) == 5
        assert "Biomass" in params.model.columns
        assert "Diet" in params.diet.columns or "Group" in params.diet.columns

    def test_length_mismatch_raises(self):
        """Test that mismatched lengths raise error."""
        groups = ["A", "B", "C"]
        types = [1, 0]  # Wrong length

        with pytest.raises(ValueError):
            create_rpath_params(groups, types)

    def test_diet_matrix_structure(self):
        """Test diet matrix has correct structure."""
        groups = ["Phyto", "Zoo", "Fish", "Detritus", "Fleet"]
        types = [1, 0, 0, 2, 3]

        params = create_rpath_params(groups, types)

        # Diet should have Import row
        assert "Import" in params.diet["Group"].values

        # Diet columns should be predator groups
        pred_groups = ["Phyto", "Zoo", "Fish"]
        for pg in pred_groups:
            assert pg in params.diet.columns


class TestRpathParams:
    """Tests for RpathParams class."""

    def test_repr(self):
        """Test string representation."""
        params = create_rpath_params(
            groups=["Phyto", "Zoo", "Fish", "Det", "Fleet"], types=[1, 0, 0, 2, 3]
        )

        repr_str = repr(params)
        assert "RpathParams" in repr_str
        assert "groups=5" in repr_str


class TestCheckRpathParams:
    """Tests for parameter validation."""

    def test_valid_params(self):
        """Test validation of valid parameters."""
        params = create_rpath_params(
            groups=["Phyto", "Zoo", "Fish", "Det", "Fleet"], types=[1, 0, 0, 2, 3]
        )

        # Fill in required values
        params.model.loc[0, "Biomass"] = 10.0
        params.model.loc[0, "PB"] = 200.0
        params.model.loc[1, "Biomass"] = 5.0
        params.model.loc[1, "PB"] = 50.0
        params.model.loc[1, "QB"] = 150.0
        params.model.loc[2, "Biomass"] = 2.0
        params.model.loc[2, "PB"] = 1.0
        params.model.loc[2, "QB"] = 5.0
        params.model.loc[3, "Biomass"] = 100.0

        # Fill BioAcc and Unassim
        params.model["BioAcc"] = params.model["BioAcc"].fillna(0.0)
        params.model["Unassim"] = params.model["Unassim"].fillna(0.2)
        params.model.loc[4, "BioAcc"] = np.nan
        params.model.loc[4, "Unassim"] = np.nan

        # Set detritus fate
        params.model["Det"] = params.model["Det"].fillna(1.0)
        params.model.loc[4, "Det"] = np.nan

        # Fill diet (5 rows: Phyto, Zoo, Fish, Det, Import)
        params.diet["Zoo"] = [1.0, 0.0, 0.0, 0.0, 0.0]  # Zoo eats Phyto
        params.diet["Fish"] = [0.0, 1.0, 0.0, 0.0, 0.0]  # Fish eats Zoo
        params.diet["Phyto"] = [0.0, 0.0, 0.0, 0.0, 0.0]  # Producer

        # This should not raise
        check_rpath_params(params)


class TestRpath:
    """Tests for Rpath balanced model."""

    @pytest.fixture
    def simple_params(self):
        """Create simple parameter set for testing."""
        params = create_rpath_params(
            groups=["Phyto", "Zoo", "Fish", "Det", "Fleet"], types=[1, 0, 0, 2, 3]
        )

        # Set up a simple balanced model
        # Phytoplankton (producer)
        params.model.loc[0, "Biomass"] = 10.0
        params.model.loc[0, "PB"] = 200.0
        params.model.loc[0, "EE"] = 0.8

        # Zooplankton (consumer)
        params.model.loc[1, "Biomass"] = 5.0
        params.model.loc[1, "PB"] = 50.0
        params.model.loc[1, "QB"] = 150.0
        params.model.loc[1, "EE"] = 0.9

        # Fish (consumer)
        params.model.loc[2, "Biomass"] = 2.0
        params.model.loc[2, "PB"] = 1.0
        params.model.loc[2, "QB"] = 5.0
        params.model.loc[2, "EE"] = 0.5

        # Detritus
        params.model.loc[3, "Biomass"] = 100.0

        # Fill other required values
        params.model["BioAcc"] = 0.0
        params.model["Unassim"] = 0.2
        params.model.loc[0, "Unassim"] = 0.0  # Producer
        params.model.loc[3, "Unassim"] = 0.0  # Detritus
        params.model.loc[4, "BioAcc"] = np.nan
        params.model.loc[4, "Unassim"] = np.nan

        # Detritus fate
        params.model["Det"] = 1.0
        params.model.loc[4, "Det"] = np.nan

        # Diet matrix (5 rows: Phyto, Zoo, Fish, Det, Import)
        # Zoo eats Phyto
        params.diet["Zoo"] = [1.0, 0.0, 0.0, 0.0, 0.0]
        # Fish eats Zoo
        params.diet["Fish"] = [0.0, 1.0, 0.0, 0.0, 0.0]
        # Phyto is producer (no diet)
        params.diet["Phyto"] = [0.0, 0.0, 0.0, 0.0, 0.0]

        # Landings (Fish caught by Fleet)
        params.model.loc[2, "Fleet"] = 0.5

        return params

    @pytest.fixture
    def baltic_sea_params(self):
        """Create a more realistic Baltic Sea-like test model.

        A simplified 8-group model representing a Baltic Sea ecosystem:
        1. Phytoplankton (producer)
        2. Zooplankton (consumer)
        3. Benthic invertebrates (consumer)
        4. Herring (consumer)
        5. Cod (consumer - top predator)
        6. Detritus
        7. Commercial fishery fleet
        """
        groups = [
            "Phytoplankton",
            "Zooplankton",
            "Benthos",
            "Herring",
            "Cod",
            "Detritus",
            "Fishery",
        ]
        types = [1, 0, 0, 0, 0, 2, 3]

        params = create_rpath_params(groups, types)

        # Set biomass values (t/km²)
        params.model.loc[0, "Biomass"] = 25.0  # Phytoplankton
        params.model.loc[1, "Biomass"] = 12.0  # Zooplankton
        params.model.loc[2, "Biomass"] = 30.0  # Benthos
        params.model.loc[3, "Biomass"] = 8.0  # Herring
        params.model.loc[4, "Biomass"] = 3.0  # Cod
        params.model.loc[5, "Biomass"] = 50.0  # Detritus

        # Set P/B ratios (1/year)
        params.model.loc[0, "PB"] = 150.0  # Phytoplankton - high turnover
        params.model.loc[1, "PB"] = 35.0  # Zooplankton
        params.model.loc[2, "PB"] = 3.0  # Benthos
        params.model.loc[3, "PB"] = 0.8  # Herring
        params.model.loc[4, "PB"] = 0.4  # Cod

        # Set Q/B ratios (1/year) - only for consumers
        params.model.loc[1, "QB"] = 120.0  # Zooplankton
        params.model.loc[2, "QB"] = 12.0  # Benthos
        params.model.loc[3, "QB"] = 4.0  # Herring
        params.model.loc[4, "QB"] = 2.5  # Cod

        # Set EE values (leave some to be calculated)
        params.model.loc[0, "EE"] = 0.85  # Phytoplankton
        params.model.loc[1, "EE"] = 0.90  # Zooplankton
        params.model.loc[2, "EE"] = 0.70  # Benthos
        params.model.loc[3, "EE"] = 0.95  # Herring - heavily predated
        params.model.loc[4, "EE"] = 0.50  # Cod - top predator

        # Set other parameters
        params.model["BioAcc"] = 0.0
        params.model["Unassim"] = 0.2
        params.model.loc[0, "Unassim"] = 0.0  # Producer
        params.model.loc[5, "Unassim"] = 0.0  # Detritus
        params.model.loc[6, "BioAcc"] = np.nan
        params.model.loc[6, "Unassim"] = np.nan

        # Detritus fate - all goes to single detritus pool
        params.model["Detritus"] = 1.0
        params.model.loc[6, "Detritus"] = np.nan

        # Diet matrix (7 rows: Phyto, Zoo, Benthos, Herring, Cod, Detritus, Import)
        # Zooplankton eats phytoplankton (70%) and detritus (30%)
        params.diet["Zooplankton"] = [0.7, 0.0, 0.0, 0.0, 0.0, 0.3, 0.0]

        # Benthos eats detritus (80%) and phytoplankton (20%)
        params.diet["Benthos"] = [0.2, 0.0, 0.0, 0.0, 0.0, 0.8, 0.0]

        # Herring eats zooplankton (90%) and benthos (10%)
        params.diet["Herring"] = [0.0, 0.9, 0.1, 0.0, 0.0, 0.0, 0.0]

        # Cod eats herring (60%), benthos (30%), zooplankton (10%)
        params.diet["Cod"] = [0.0, 0.1, 0.3, 0.6, 0.0, 0.0, 0.0]

        # Producer - no diet
        params.diet["Phytoplankton"] = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]

        # Landings (t/km²/year)
        params.model.loc[3, "Fishery"] = 0.5  # Herring landings
        params.model.loc[4, "Fishery"] = 0.3  # Cod landings

        # Discards
        if "Fishery.disc" in params.model.columns:
            params.model.loc[3, "Fishery.disc"] = 0.05  # Herring discards
            params.model.loc[4, "Fishery.disc"] = 0.02  # Cod discards

        return params

    def test_rpath_creates_balanced_model(self, simple_params):
        """Test that rpath() creates a balanced model."""
        model = rpath(simple_params, eco_name="Test")

        assert isinstance(model, Rpath)
        assert model.NUM_GROUPS == 5
        assert model.NUM_LIVING == 3
        assert model.NUM_DEAD == 1
        assert model.NUM_GEARS == 1

    def test_trophic_levels(self, simple_params):
        """Test trophic level calculation."""
        model = rpath(simple_params)

        # Producers should have TL = 1
        assert np.isclose(model.TL[0], 1.0, atol=0.1)

        # Zoo (eats producer) should have TL ~ 2
        assert model.TL[1] > 1.5

        # Fish (eats Zoo) should have TL ~ 3
        assert model.TL[2] > 2.5

    def test_summary_method(self, simple_params):
        """Test summary DataFrame generation."""
        model = rpath(simple_params)
        summary = model.summary()

        assert isinstance(summary, pd.DataFrame)
        assert "Group" in summary.columns
        assert "TL" in summary.columns
        assert "Biomass" in summary.columns

    def test_baltic_model_balances(self, baltic_sea_params):
        """Test that Baltic Sea model balances correctly."""
        model = rpath(baltic_sea_params, eco_name="Baltic Sea Test")

        assert isinstance(model, Rpath)
        assert model.NUM_GROUPS == 7
        assert model.NUM_LIVING == 5
        assert model.NUM_DEAD == 1
        assert model.NUM_GEARS == 1
        assert model.eco_name == "Baltic Sea Test"

    def test_baltic_trophic_levels(self, baltic_sea_params):
        """Test trophic levels in Baltic model."""
        model = rpath(baltic_sea_params)

        # Phytoplankton (producer) TL = 1
        assert np.isclose(model.TL[0], 1.0, atol=0.01)

        # Zooplankton TL ~ 2 (eats phyto + detritus)
        assert 1.5 < model.TL[1] < 2.5

        # Benthos TL ~ 2 (eats phyto + detritus)
        assert 1.5 < model.TL[2] < 2.5

        # Herring TL ~ 3 (eats zoo + benthos)
        assert 2.5 < model.TL[3] < 3.5

        # Cod TL > 3 (top predator)
        assert model.TL[4] > 3.0

    def test_baltic_ecotrophic_efficiency(self, baltic_sea_params):
        """Test EE values are reasonable."""
        model = rpath(baltic_sea_params)

        # All EE should be between 0 and 1 for a balanced model
        living_ee = model.EE[: model.NUM_LIVING]
        assert all(
            0 <= ee <= 1 for ee in living_ee if not np.isnan(ee)
        ), f"EE values out of range: {living_ee}"

    def test_baltic_gross_efficiency(self, baltic_sea_params):
        """Test GE (P/Q) ratios are reasonable."""
        model = rpath(baltic_sea_params)

        # For consumers, GE = P/Q should typically be 0.1-0.4
        # (production is 10-40% of consumption)
        for i in range(1, model.NUM_LIVING):  # Skip producers
            if model.type[i] == 0:  # Consumer
                ge = model.GE[i]
                assert 0.05 < ge < 0.5, f"GE[{i}] = {ge} is out of typical range"

    def test_baltic_diet_sums_to_one(self, baltic_sea_params):
        """Test that diet compositions sum to 1."""
        model = rpath(baltic_sea_params)

        # For each predator, diet should sum to 1
        for j in range(model.NUM_LIVING):
            if model.type[j] == 0:  # Consumer
                diet_sum = np.nansum(model.DC[:, j])
                assert np.isclose(
                    diet_sum, 1.0, atol=0.01
                ), f"Diet for {model.Group[j]} sums to {diet_sum}"

    def test_baltic_removals(self, baltic_sea_params):
        """Test fishing removals are recorded."""
        model = rpath(baltic_sea_params)

        # Herring should have landings
        herring_idx = 3
        herring_landings = np.nansum(model.Landings[herring_idx, :])
        assert herring_landings > 0, "Herring should have landings"

        # Cod should have landings
        cod_idx = 4
        cod_landings = np.nansum(model.Landings[cod_idx, :])
        assert cod_landings > 0, "Cod should have landings"

    def test_model_repr(self, baltic_sea_params):
        """Test model string representation."""
        model = rpath(baltic_sea_params, eco_name="Baltic Test")

        repr_str = repr(model)
        assert "Baltic Test" in repr_str
        assert "Groups: 7" in repr_str

    def test_model_with_missing_ee(self):
        """Test model can calculate missing EE values."""
        params = create_rpath_params(
            groups=["Phyto", "Zoo", "Det", "Fleet"], types=[1, 0, 2, 3]
        )

        # Set biomass and rates
        params.model.loc[0, "Biomass"] = 10.0
        params.model.loc[0, "PB"] = 100.0
        # Don't set EE for phyto - should be calculated

        params.model.loc[1, "Biomass"] = 5.0
        params.model.loc[1, "PB"] = 30.0
        params.model.loc[1, "QB"] = 100.0
        params.model.loc[1, "EE"] = 0.5

        params.model.loc[2, "Biomass"] = 50.0

        params.model["BioAcc"] = 0.0
        params.model["Unassim"] = 0.2
        params.model.loc[0, "Unassim"] = 0.0
        params.model.loc[2, "Unassim"] = 0.0
        params.model.loc[3, "BioAcc"] = np.nan
        params.model.loc[3, "Unassim"] = np.nan

        params.model["Det"] = 1.0
        params.model.loc[3, "Det"] = np.nan

        # Zoo eats Phyto
        params.diet["Zoo"] = [1.0, 0.0, 0.0, 0.0]
        params.diet["Phyto"] = [0.0, 0.0, 0.0, 0.0]

        model = rpath(params)

        # Model should balance and calculate missing EE
        assert isinstance(model, Rpath)
        assert not np.isnan(model.EE[0])  # EE should be calculated


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
