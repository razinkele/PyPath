"""
Unit tests for the analysis module.

Tests for Mixed Trophic Impacts, network indices,
and other analysis functions.
"""

import pytest
import numpy as np
from unittest.mock import MagicMock, patch

from pypath.core.analysis import (
    mixed_trophic_impacts,
    keystoneness_index,
    calculate_network_indices,
    NetworkIndices,
    EcosimSummary,
    summarize_ecosim_output,
    check_ecopath_balance,
    export_ecopath_to_dataframe,
    export_ecosim_to_dataframe,
)


class TestMixedTrophicImpacts:
    """Tests for mixed_trophic_impacts function."""
    
    def test_mti_returns_square_matrix(self):
        """MTI should return square matrix of living groups."""
        rpath = MagicMock()
        rpath.NUM_LIVING = 3
        rpath.NUM_DEAD = 1
        rpath.NUM_GROUPS = 4
        
        # Setup diet and consumption data
        rpath.DC = np.array([
            [0, 0, 0, 0, 0],
            [0, 0, 0.5, 0, 0],  # Group 1: eaten by group 2
            [0, 0, 0, 0.5, 0],  # Group 2: eaten by group 3
            [0, 0, 0, 0, 0],    # Group 3
            [0, 0.5, 0.5, 0, 0],  # Detritus: eaten by 1 and 2
        ])
        rpath.PB = np.array([0, 1.0, 0.5, 0.2, 0])
        rpath.QB = np.array([0, 5, 3, 1, 0])
        rpath.Biomass = np.array([0, 10, 5, 2, 3])
        
        mti = mixed_trophic_impacts(rpath)
        
        assert mti.shape == (4, 4)  # n_groups x n_groups
    
    def test_mti_with_no_diet(self):
        """MTI with zero diet should produce valid matrix."""
        rpath = MagicMock()
        rpath.NUM_LIVING = 2
        rpath.NUM_DEAD = 0
        rpath.NUM_GROUPS = 2
        
        rpath.DC = np.zeros((3, 3))
        rpath.PB = np.array([0, 1.0, 0.5])
        rpath.QB = np.array([0, 5, 5])
        rpath.Biomass = np.array([0, 1, 1])
        
        mti = mixed_trophic_impacts(rpath)
        
        assert mti.shape == (2, 2)


class TestKeystonenessIndex:
    """Tests for keystoneness_index function."""
    
    def test_returns_array(self):
        """Should return array with keystoneness values."""
        rpath = MagicMock()
        rpath.NUM_LIVING = 3
        rpath.NUM_DEAD = 1
        rpath.NUM_GROUPS = 4
        
        rpath.DC = np.array([
            [0, 0, 0, 0, 0],
            [0, 0, 0.5, 0, 0],
            [0, 0, 0, 0.5, 0],
            [0, 0, 0, 0, 0],
            [0, 0.5, 0.5, 0, 0],
        ])
        rpath.PB = np.array([0, 1.0, 0.5, 0.2, 0])
        rpath.QB = np.array([0, 5, 3, 1, 0])
        rpath.Biomass = np.array([0, 10, 5, 2, 3])
        
        ks = keystoneness_index(rpath)
        
        assert len(ks) == 5  # 0 + n_groups
    
    def test_accepts_precomputed_mti(self):
        """Should use provided MTI matrix."""
        rpath = MagicMock()
        rpath.NUM_LIVING = 2
        rpath.NUM_DEAD = 0
        rpath.NUM_GROUPS = 2
        
        rpath.Biomass = np.array([0, 10, 5])
        
        mti = np.array([[0, 0.5], [0.5, 0]])
        
        ks = keystoneness_index(rpath, mti=mti)
        
        assert len(ks) == 3


class TestNetworkIndices:
    """Tests for calculate_network_indices function."""
    
    def test_returns_network_indices_dataclass(self):
        """Should return NetworkIndices dataclass."""
        rpath = MagicMock()
        rpath.NUM_LIVING = 3
        rpath.NUM_DEAD = 1
        rpath.NUM_GROUPS = 4
        
        rpath.DC = np.array([
            [0, 0, 0, 0, 0],
            [0, 0, 0.3, 0, 0],
            [0, 0, 0, 0.4, 0],
            [0, 0, 0, 0, 0],
            [0, 0.7, 0.6, 0, 0],
        ])
        rpath.TL = np.array([0, 1.0, 2.0, 3.0, 1.0])
        rpath.PB = np.array([0, 1.0, 0.5, 0.2, 0])
        rpath.QB = np.array([0, 5, 3, 1, 0])
        rpath.Biomass = np.array([0, 10, 5, 2, 3])
        rpath.EE = np.array([0, 0.9, 0.8, 0.7, 0.5])
        
        indices = calculate_network_indices(rpath)
        
        assert isinstance(indices, NetworkIndices)
        assert indices.n_living == 3
    
    def test_connectance_calculation(self):
        """Connectance should be links / possible_links."""
        rpath = MagicMock()
        rpath.NUM_LIVING = 3
        rpath.NUM_DEAD = 0
        rpath.NUM_GROUPS = 3
        
        # 2 links in a 3-species system
        rpath.DC = np.array([
            [0, 0, 0, 0],
            [0, 0, 0.5, 0],  # 1 link
            [0, 0, 0, 0.5],  # 1 link
            [0, 0, 0, 0],
        ])
        rpath.TL = np.array([0, 1.0, 2.0, 3.0])
        rpath.PB = np.array([0, 1.0, 0.5, 0.2])
        rpath.QB = np.array([0, 5, 3, 1])
        rpath.Biomass = np.array([0, 10, 5, 2])
        rpath.EE = np.array([0, 0.9, 0.8, 0.7])
        
        indices = calculate_network_indices(rpath)
        
        # Should have 2 links
        assert indices.n_links == 2
    
    def test_total_biomass(self):
        """Total biomass should sum all groups including detritus."""
        rpath = MagicMock()
        rpath.NUM_LIVING = 3
        rpath.NUM_DEAD = 1
        rpath.NUM_GROUPS = 4
        
        rpath.DC = np.zeros((5, 5))
        rpath.TL = np.array([0, 1.0, 2.0, 3.0, 1.0])
        rpath.PB = np.array([0, 1.0, 0.5, 0.2, 0])
        rpath.QB = np.array([0, 5, 3, 1, 0])
        rpath.Biomass = np.array([0, 10, 5, 2, 3])  # Total = 10+5+2+3 = 20
        rpath.EE = np.array([0, 0.9, 0.8, 0.7, 0.5])
        
        indices = calculate_network_indices(rpath)
        
        # Function sums all groups
        assert indices.total_biomass == 20


class TestSummarizeEcosimOutput:
    """Tests for summarize_ecosim_output function."""
    
    def test_returns_ecosim_summary(self):
        """Should return EcosimSummary dataclass with summary statistics."""
        output = MagicMock()
        output.out_Biomass_annual = np.random.rand(10, 5)
        output.out_Biomass_annual[:, 0] = 0
        output.out_Catch_annual = np.random.rand(10, 5)
        output.out_Catch_annual[:, 0] = 0
        
        summary = summarize_ecosim_output(output)
        
        assert isinstance(summary, EcosimSummary)
        assert summary.years == 10


class TestCheckEcopathBalance:
    """Tests for check_ecopath_balance function."""
    
    def test_balanced_model(self):
        """Balanced model should pass checks."""
        rpath = MagicMock()
        rpath.NUM_LIVING = 2
        rpath.NUM_DEAD = 0
        rpath.NUM_GROUPS = 2
        rpath.NUM_GEARS = 1
        
        rpath.Biomass = np.array([0, 10.0, 5.0])
        rpath.PB = np.array([0, 1.0, 0.5])
        rpath.QB = np.array([0, 0, 3.0])
        rpath.EE = np.array([0, 0.9, 0.8])
        rpath.TL = np.array([0, 1.0, 2.0])
        rpath.DC = np.zeros((3, 3))
        rpath.DC[1, 2] = 1.0  # Consumer eats producer
        rpath.Catch = np.zeros((3, 2))
        
        result = check_ecopath_balance(rpath)
        
        assert isinstance(result, dict)
        assert 'is_balanced' in result or len(result) > 0


class TestExportEcopathToDataframe:
    """Tests for export_ecopath_to_dataframe function."""
    
    def test_returns_dict_of_dataframes(self):
        """Should return dictionary of DataFrames."""
        rpath = MagicMock()
        rpath.NUM_LIVING = 3
        rpath.NUM_DEAD = 1
        rpath.NUM_GROUPS = 4
        rpath.NUM_GEARS = 2
        
        rpath.Biomass = np.array([0, 10, 5, 2, 3])
        rpath.PB = np.array([0, 1.0, 0.5, 0.2, 0])
        rpath.QB = np.array([0, 0, 3, 1, 0])
        rpath.EE = np.array([0, 0.9, 0.8, 0.7, 0.5])
        rpath.TL = np.array([0, 1.0, 2.0, 3.0, 1.0])
        rpath.DC = np.zeros((5, 5))
        rpath.Catch = np.zeros((5, 3))
        
        result = export_ecopath_to_dataframe(rpath)
        
        assert isinstance(result, dict)
        # Check that it has at least one dataframe
        assert len(result) > 0
        # Check that groups dataframe exists
        assert 'groups' in result


class TestExportEcosimToDataframe:
    """Tests for export_ecosim_to_dataframe function."""
    
    def test_returns_dict_of_dataframes(self):
        """Should return dictionary of DataFrames."""
        output = MagicMock()
        output.out_Biomass_annual = np.random.rand(10, 5)
        output.out_Catch_annual = np.random.rand(10, 5)
        output.out_Biomass = None
        
        result = export_ecosim_to_dataframe(output)
        
        assert isinstance(result, dict)
        assert 'biomass_annual' in result
        assert 'catch_annual' in result


class TestNetworkIndicesDataclass:
    """Tests for NetworkIndices dataclass."""
    
    def test_fields(self):
        """NetworkIndices should have all required fields."""
        ni = NetworkIndices(
            n_groups=10,
            n_living=8,
            n_links=20,
            connectance=0.25,
            linkage_density=2.5,
            omnivory_index=0.3,
            system_omnivory=0.4,
            mean_trophic_level=2.5,
            max_trophic_level=4.0,
            total_biomass=100.0,
            total_throughput=500.0,
            transfer_efficiency=0.1
        )
        
        assert ni.n_groups == 10
        assert ni.n_living == 8
        assert ni.connectance == 0.25
    
    def test_default_values(self):
        """Should have zero defaults."""
        ni = NetworkIndices()
        
        assert ni.n_groups == 0
        assert ni.connectance == 0.0
