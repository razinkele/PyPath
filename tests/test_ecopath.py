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
        groups = ['Phyto', 'Zoo', 'Fish', 'Detritus', 'Fleet']
        types = [1, 0, 0, 2, 3]
        
        params = create_rpath_params(groups, types)
        
        assert isinstance(params, RpathParams)
        assert len(params.model) == 5
        assert 'Biomass' in params.model.columns
        assert 'Diet' in params.diet.columns or 'Group' in params.diet.columns
    
    def test_length_mismatch_raises(self):
        """Test that mismatched lengths raise error."""
        groups = ['A', 'B', 'C']
        types = [1, 0]  # Wrong length
        
        with pytest.raises(ValueError):
            create_rpath_params(groups, types)
    
    def test_diet_matrix_structure(self):
        """Test diet matrix has correct structure."""
        groups = ['Phyto', 'Zoo', 'Fish', 'Detritus', 'Fleet']
        types = [1, 0, 0, 2, 3]
        
        params = create_rpath_params(groups, types)
        
        # Diet should have Import row
        assert 'Import' in params.diet['Group'].values
        
        # Diet columns should be predator groups
        pred_groups = ['Phyto', 'Zoo', 'Fish']
        for pg in pred_groups:
            assert pg in params.diet.columns


class TestRpathParams:
    """Tests for RpathParams class."""
    
    def test_repr(self):
        """Test string representation."""
        params = create_rpath_params(
            groups=['Phyto', 'Zoo', 'Fish', 'Det', 'Fleet'],
            types=[1, 0, 0, 2, 3]
        )
        
        repr_str = repr(params)
        assert 'RpathParams' in repr_str
        assert 'groups=5' in repr_str


class TestCheckRpathParams:
    """Tests for parameter validation."""
    
    def test_valid_params(self):
        """Test validation of valid parameters."""
        params = create_rpath_params(
            groups=['Phyto', 'Zoo', 'Fish', 'Det', 'Fleet'],
            types=[1, 0, 0, 2, 3]
        )
        
        # Fill in required values
        params.model.loc[0, 'Biomass'] = 10.0
        params.model.loc[0, 'PB'] = 200.0
        params.model.loc[1, 'Biomass'] = 5.0
        params.model.loc[1, 'PB'] = 50.0
        params.model.loc[1, 'QB'] = 150.0
        params.model.loc[2, 'Biomass'] = 2.0
        params.model.loc[2, 'PB'] = 1.0
        params.model.loc[2, 'QB'] = 5.0
        params.model.loc[3, 'Biomass'] = 100.0
        
        # Fill BioAcc and Unassim
        params.model['BioAcc'] = params.model['BioAcc'].fillna(0.0)
        params.model['Unassim'] = params.model['Unassim'].fillna(0.2)
        params.model.loc[4, 'BioAcc'] = np.nan
        params.model.loc[4, 'Unassim'] = np.nan
        
        # Set detritus fate
        params.model['Det'] = params.model['Det'].fillna(1.0)
        params.model.loc[4, 'Det'] = np.nan
        
        # Fill diet (5 rows: Phyto, Zoo, Fish, Det, Import)
        params.diet['Zoo'] = [1.0, 0.0, 0.0, 0.0, 0.0]  # Zoo eats Phyto
        params.diet['Fish'] = [0.0, 1.0, 0.0, 0.0, 0.0]  # Fish eats Zoo
        params.diet['Phyto'] = [0.0, 0.0, 0.0, 0.0, 0.0]  # Producer
        
        # This should not raise
        check_rpath_params(params)


class TestRpath:
    """Tests for Rpath balanced model."""
    
    @pytest.fixture
    def simple_params(self):
        """Create simple parameter set for testing."""
        params = create_rpath_params(
            groups=['Phyto', 'Zoo', 'Fish', 'Det', 'Fleet'],
            types=[1, 0, 0, 2, 3]
        )
        
        # Set up a simple balanced model
        # Phytoplankton (producer)
        params.model.loc[0, 'Biomass'] = 10.0
        params.model.loc[0, 'PB'] = 200.0
        params.model.loc[0, 'EE'] = 0.8
        
        # Zooplankton (consumer)
        params.model.loc[1, 'Biomass'] = 5.0
        params.model.loc[1, 'PB'] = 50.0
        params.model.loc[1, 'QB'] = 150.0
        params.model.loc[1, 'EE'] = 0.9
        
        # Fish (consumer)
        params.model.loc[2, 'Biomass'] = 2.0
        params.model.loc[2, 'PB'] = 1.0
        params.model.loc[2, 'QB'] = 5.0
        params.model.loc[2, 'EE'] = 0.5
        
        # Detritus
        params.model.loc[3, 'Biomass'] = 100.0
        
        # Fill other required values
        params.model['BioAcc'] = 0.0
        params.model['Unassim'] = 0.2
        params.model.loc[0, 'Unassim'] = 0.0  # Producer
        params.model.loc[3, 'Unassim'] = 0.0  # Detritus
        params.model.loc[4, 'BioAcc'] = np.nan
        params.model.loc[4, 'Unassim'] = np.nan
        
        # Detritus fate
        params.model['Det'] = 1.0
        params.model.loc[4, 'Det'] = np.nan
        
        # Diet matrix (5 rows: Phyto, Zoo, Fish, Det, Import)
        # Zoo eats Phyto
        params.diet['Zoo'] = [1.0, 0.0, 0.0, 0.0, 0.0]
        # Fish eats Zoo
        params.diet['Fish'] = [0.0, 1.0, 0.0, 0.0, 0.0]
        # Phyto is producer (no diet)
        params.diet['Phyto'] = [0.0, 0.0, 0.0, 0.0, 0.0]
        
        # Landings (Fish caught by Fleet)
        params.model.loc[2, 'Fleet'] = 0.5
        
        return params
    
    def test_rpath_creates_balanced_model(self, simple_params):
        """Test that rpath() creates a balanced model."""
        model = rpath(simple_params, eco_name='Test')
        
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
        assert 'Group' in summary.columns
        assert 'TL' in summary.columns
        assert 'Biomass' in summary.columns


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
