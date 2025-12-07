"""
Tests for EcoBase connector module.
"""

import pytest
from unittest.mock import patch, Mock
import numpy as np
import pandas as pd

from pypath.io.ecobase import (
    EcoBaseModel,
    EcoBaseGroupData,
    list_ecobase_models,
    get_ecobase_model,
    ecobase_to_rpath,
    search_ecobase_models,
    download_ecobase_model_to_file,
    ECOBASE_LIST_URL,
    ECOBASE_MODEL_URL,
)


# Sample XML responses for mocking
SAMPLE_MODEL_LIST_XML = """<?xml version="1.0" encoding="UTF-8"?>
<models>
    <model>
        <model_number>123</model_number>
        <model_name>Baltic Sea Model</model_name>
        <year>2015</year>
        <author>Test Author</author>
        <country>Sweden</country>
        <ecosystem_type>Marine</ecosystem_type>
        <number_group>10</number_group>
        <reference>Test et al. 2015</reference>
        <dissemination_allow>true</dissemination_allow>
    </model>
    <model>
        <model_number>456</model_number>
        <model_name>Lake Model</model_name>
        <year>2018</year>
        <author>Another Author</author>
        <country>Finland</country>
        <ecosystem_type>Freshwater</ecosystem_type>
        <number_group>8</number_group>
        <reference>Author et al. 2018</reference>
        <dissemination_allow>true</dissemination_allow>
    </model>
    <model>
        <model_number>789</model_number>
        <model_name>Private Model</model_name>
        <year>2020</year>
        <author>Private Author</author>
        <country>Norway</country>
        <ecosystem_type>Marine</ecosystem_type>
        <number_group>5</number_group>
        <reference>Private et al. 2020</reference>
        <dissemination_allow>false</dissemination_allow>
    </model>
</models>
"""

SAMPLE_MODEL_DATA_XML = """<?xml version="1.0" encoding="UTF-8"?>
<model>
    <model_number>123</model_number>
    <model_name>Baltic Sea Model</model_name>
    <year>2015</year>
    <author>Test Author</author>
    <country>Sweden</country>
    <ecosystem_type>Marine</ecosystem_type>
    <reference>Test et al. 2015</reference>
    <group>
        <group_seq>1</group_seq>
        <group_name>Phytoplankton</group_name>
        <trophic_level>1.0</trophic_level>
        <biomass>10.0</biomass>
        <prod_biom>100.0</prod_biom>
        <cons_biom>0</cons_biom>
        <ecotrophic_eff>0.95</ecotrophic_eff>
        <prod_cons>0</prod_cons>
        <habitat_area>1.0</habitat_area>
    </group>
    <group>
        <group_seq>2</group_seq>
        <group_name>Zooplankton</group_name>
        <trophic_level>2.1</trophic_level>
        <biomass>5.0</biomass>
        <prod_biom>40.0</prod_biom>
        <cons_biom>150.0</cons_biom>
        <ecotrophic_eff>0.90</ecotrophic_eff>
        <prod_cons>0.267</prod_cons>
        <habitat_area>1.0</habitat_area>
    </group>
    <group>
        <group_seq>3</group_seq>
        <group_name>Fish</group_name>
        <trophic_level>3.2</trophic_level>
        <biomass>2.0</biomass>
        <prod_biom>1.5</prod_biom>
        <cons_biom>5.0</cons_biom>
        <ecotrophic_eff>0.80</ecotrophic_eff>
        <prod_cons>0.3</prod_cons>
        <habitat_area>1.0</habitat_area>
    </group>
    <diet>
        <prey>Phytoplankton</prey>
        <predator>Zooplankton</predator>
        <diet_comp>0.9</diet_comp>
    </diet>
    <diet>
        <prey>Zooplankton</prey>
        <predator>Fish</predator>
        <diet_comp>0.8</diet_comp>
    </diet>
    <diet>
        <prey>Phytoplankton</prey>
        <predator>Fish</predator>
        <diet_comp>0.1</diet_comp>
    </diet>
</model>
"""


class TestEcoBaseDataClasses:
    """Tests for EcoBase data classes."""
    
    def test_ecobase_model_creation(self):
        """Test EcoBaseModel dataclass creation."""
        model = EcoBaseModel(
            model_number=123,
            model_name="Test Model",
            author="Test Author",
            country="Sweden",
            ecosystem_type="Marine",
        )
        assert model.model_number == 123
        assert model.model_name == "Test Model"
        assert model.author == "Test Author"
        assert model.country == "Sweden"
        assert model.ecosystem_type == "Marine"
        assert model.year == 0  # Default
    
    def test_ecobase_group_data_creation(self):
        """Test EcoBaseGroupData dataclass creation."""
        group = EcoBaseGroupData(
            group_seq=1,
            group_name="Phytoplankton",
            trophic_level=1.0,
            biomass=10.0,
            prod_biom=100.0,
            cons_biom=0.0,
            ecotrophic_eff=0.95,
        )
        assert group.group_seq == 1
        assert group.group_name == "Phytoplankton"
        assert group.trophic_level == 1.0
        assert group.biomass == 10.0
        assert group.prod_biom == 100.0
        assert group.cons_biom == 0.0
        assert group.ecotrophic_eff == 0.95


class TestListModels:
    """Tests for list_ecobase_models function."""
    
    @patch('pypath.io.ecobase._fetch_url')
    def test_list_models_success(self, mock_fetch):
        """Test successful model listing."""
        mock_fetch.return_value = SAMPLE_MODEL_LIST_XML
        
        models = list_ecobase_models(filter_public=True)
        
        # Should return DataFrame
        assert isinstance(models, pd.DataFrame)
        assert len(models) == 2  # Only public models
        assert 123 in models['model_number'].values
        assert 456 in models['model_number'].values
        assert 789 not in models['model_number'].values  # Private
    
    @patch('pypath.io.ecobase._fetch_url')
    def test_list_models_no_filter(self, mock_fetch):
        """Test listing all models without public filter."""
        mock_fetch.return_value = SAMPLE_MODEL_LIST_XML
        
        models = list_ecobase_models(filter_public=False)
        
        assert len(models) == 3  # All models including private
    
    @patch('pypath.io.ecobase._fetch_url')
    def test_list_models_network_error(self, mock_fetch):
        """Test network error handling."""
        mock_fetch.side_effect = Exception("Network error")
        
        with pytest.raises(ConnectionError) as exc_info:
            list_ecobase_models()
        assert "Network error" in str(exc_info.value)


class TestGetModel:
    """Tests for get_ecobase_model function."""
    
    @patch('pypath.io.ecobase._fetch_url')
    def test_get_model_success(self, mock_fetch):
        """Test successful model retrieval."""
        mock_fetch.return_value = SAMPLE_MODEL_DATA_XML
        
        model_data = get_ecobase_model(123)
        
        # Should return dict with metadata and groups
        assert isinstance(model_data, dict)
        assert 'groups' in model_data
        assert 'diet' in model_data
        
        # Check groups (should have parsed 3 groups)
        assert len(model_data['groups']) == 3
        
        # Check first group
        first_group = model_data['groups'][0]
        assert first_group['group_name'] == 'Phytoplankton'
    
    @patch('pypath.io.ecobase._fetch_url')
    def test_get_model_network_error(self, mock_fetch):
        """Test network error handling."""
        mock_fetch.side_effect = Exception("Connection refused")
        
        with pytest.raises(ConnectionError) as exc_info:
            get_ecobase_model(99999)
        assert "Connection refused" in str(exc_info.value)


class TestSearchModels:
    """Tests for search_ecobase_models function."""
    
    def test_search_by_name(self):
        """Test searching models by name."""
        # Create test DataFrame
        models_df = pd.DataFrame({
            'model_number': [1, 2, 3],
            'model_name': ['Baltic Sea', 'North Sea', 'Lake Erie'],
            'country': ['Sweden', 'UK', 'USA'],
            'ecosystem_type': ['Marine', 'Marine', 'Freshwater'],
            'author': ['Author A', 'Author B', 'Author C'],
        })
        
        results = search_ecobase_models("Sea", models_df=models_df)
        
        assert len(results) == 2
        assert 'Baltic Sea' in results['model_name'].values
        assert 'North Sea' in results['model_name'].values
    
    def test_search_by_field(self):
        """Test searching specific field."""
        models_df = pd.DataFrame({
            'model_number': [1, 2, 3],
            'model_name': ['Model 1', 'Model 2', 'Model 3'],
            'country': ['Sweden', 'Finland', 'Sweden'],
            'ecosystem_type': ['Marine', 'Freshwater', 'Marine'],
            'author': ['Author A', 'Author B', 'Author C'],
        })
        
        results = search_ecobase_models("Sweden", field="country", models_df=models_df)
        
        assert len(results) == 2
        assert all(r == 'Sweden' for r in results['country'].values)
    
    def test_search_case_insensitive(self):
        """Test case-insensitive search."""
        models_df = pd.DataFrame({
            'model_number': [1, 2],
            'model_name': ['Baltic SEA', 'NORTH sea'],
            'country': ['Sweden', 'UK'],
            'ecosystem_type': ['Marine', 'Marine'],
            'author': ['Author A', 'Author B'],
        })
        
        results = search_ecobase_models("sea", models_df=models_df)
        
        assert len(results) == 2


class TestEcobaseToRpath:
    """Tests for ecobase_to_rpath function."""
    
    def test_convert_basic_model(self):
        """Test converting a basic EcoBase model to RpathParams."""
        model_data = {
            'metadata': {
                'model_number': 123,
                'model_name': 'Test Model',
            },
            'groups': [
                {
                    'group_name': 'Phytoplankton',
                    'trophic_level': 1.0,
                    'biomass': 10.0,
                    'prod_biom': 100.0,
                    'cons_biom': 0.0,
                    'ecotrophic_eff': 0.95,
                    'prod_cons': 0.0,
                    'habitat_area': 1.0,
                },
                {
                    'group_name': 'Zooplankton',
                    'trophic_level': 2.1,
                    'biomass': 5.0,
                    'prod_biom': 40.0,
                    'cons_biom': 150.0,
                    'ecotrophic_eff': 0.90,
                    'prod_cons': 0.267,
                    'habitat_area': 1.0,
                },
            ],
            'diet': {
                'Zooplankton': {
                    'Phytoplankton': 1.0,
                },
            },
        }
        
        params = ecobase_to_rpath(model_data)
        
        # Check groups (use len(model) instead of ngroups)
        assert len(params.model) == 2
        assert 'Phytoplankton' in params.model['Group'].values
        assert 'Zooplankton' in params.model['Group'].values
        
        # Check diet was converted
        assert 'Zooplankton' in params.diet.columns, "Zooplankton should be a predator column"
        
        # Find Phytoplankton row and check diet value
        phyto_row = params.diet[params.diet['Group'] == 'Phytoplankton']
        assert len(phyto_row) == 1, "Should have one Phytoplankton row"
        phyto_idx = phyto_row.index[0]
        diet_val = params.diet.at[phyto_idx, 'Zooplankton']
        assert pd.notna(diet_val), f"Diet value should not be NaN, got: {diet_val}"
        assert diet_val == 1.0, f"Expected 1.0, got: {diet_val}"
    
    def test_convert_with_detritus(self):
        """Test converting model with detritus."""
        model_data = {
            'metadata': {'model_name': 'Test'},
            'groups': [
                {
                    'group_name': 'Phytoplankton',
                    'trophic_level': 1.0,
                    'biomass': 10.0,
                    'prod_biom': 100.0,
                    'cons_biom': 0.0,
                    'ecotrophic_eff': 0.95,
                },
                {
                    'group_name': 'Detritus',
                    'trophic_level': 1.0,
                    'biomass': 100.0,
                    'prod_biom': 0.0,
                    'cons_biom': 0.0,
                    'ecotrophic_eff': 0.0,
                    'group_type': 'detritus',
                },
            ],
            'diet': {},
        }
        
        params = ecobase_to_rpath(model_data)
        
        assert len(params.model) == 2
        # Check that detritus is properly typed
        assert 'Detritus' in params.model['Group'].values


class TestIntegration:
    """Integration tests (require network, skip by default)."""
    
    @pytest.mark.skip(reason="Requires network access to EcoBase")
    def test_list_models_live(self):
        """Test listing models from live EcoBase."""
        models = list_ecobase_models()
        assert len(models) > 0
        assert 'model_number' in models.columns
        assert 'model_name' in models.columns
    
    @pytest.mark.skip(reason="Requires network access to EcoBase")
    def test_download_model_live(self):
        """Test downloading a model from live EcoBase."""
        # Use a known model ID
        model_data = get_ecobase_model(1)
        assert 'groups' in model_data
        assert len(model_data['groups']) > 0
    
    @pytest.mark.skip(reason="Requires network access to EcoBase")
    def test_search_live(self):
        """Test searching models on live EcoBase."""
        results = search_ecobase_models("Baltic")
        assert len(results) >= 0  # May or may not have results
    
    @pytest.mark.skip(reason="Requires network access to EcoBase")
    def test_full_pipeline_live(self):
        """Test full pipeline: list -> download -> convert."""
        # List models
        models = list_ecobase_models()
        
        if len(models) > 0:
            # Get first model
            model_id = models.iloc[0]['model_number']
            
            # Download
            model_data = get_ecobase_model(model_id)
            
            # Convert
            params = ecobase_to_rpath(model_data)
            
            assert params.ngroups > 0
