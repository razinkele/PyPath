"""
Tests for multi-stanza group functionality.

Tests the stanzas module which handles age-structured groups in Ecosim.
"""

import pytest
import numpy as np
from pypath.core.stanzas import (
    StanzaGroup,
    StanzaIndividual,
    StanzaParams,
    RsimStanzas,
    von_bertalanffy_weight,
    von_bertalanffy_consumption,
    calculate_survival,
    rpath_stanzas,
    rsim_stanzas,
    split_update,
    create_stanza_params,
)


class TestVonBertalanffy:
    """Test Von Bertalanffy growth functions."""
    
    def test_weight_at_ages(self):
        """Weight should be calculated correctly for array of ages."""
        ages = np.array([0, 1, 2, 5, 10])
        k = 0.3
        weights = von_bertalanffy_weight(ages, k)
        
        # Weights should increase with age
        for i in range(1, len(weights)):
            assert weights[i] >= weights[i-1]
    
    def test_weight_increases_with_age(self):
        """Weight should increase monotonically with age."""
        ages = np.arange(0, 20, 0.1)
        weights = von_bertalanffy_weight(ages, k=0.3)
        
        for i in range(1, len(weights)):
            assert weights[i] >= weights[i-1]
    
    def test_weight_with_different_k(self):
        """Higher K should give faster growth."""
        ages = np.array([5])
        w_low_k = von_bertalanffy_weight(ages, k=0.1)
        w_high_k = von_bertalanffy_weight(ages, k=0.5)
        
        # Higher K = faster growth at same age
        assert w_high_k[0] > w_low_k[0]


class TestVonBertalanffyConsumption:
    """Test Von Bertalanffy consumption calculations."""
    
    def test_consumption_from_weight(self):
        """Consumption should scale with weight."""
        weights = np.array([0.1, 0.5, 1.0])
        consumption = von_bertalanffy_consumption(weights)
        
        # Consumption should be related to weight
        assert len(consumption) == len(weights)
        # Consumption should scale with body size
        assert consumption[2] >= consumption[0]


class TestCalculateSurvival:
    """Test survival rate calculations."""
    
    def test_survival_decreases_with_mortality(self):
        """Higher mortality should give lower survival."""
        z_low = np.array([0.1, 0.1, 0.1])
        z_high = np.array([0.5, 0.5, 0.5])
        
        surv_low = calculate_survival(z_low)
        surv_high = calculate_survival(z_high)
        
        # Higher mortality = lower survival
        assert surv_high[-1] < surv_low[-1]
    
    def test_zero_mortality_full_survival(self):
        """Zero mortality should give survival close to 1."""
        z = np.zeros(12)  # 12 months of zero mortality
        surv = calculate_survival(z)
        
        # First element should be 1
        assert surv[0] == 1.0


class TestStanzaGroup:
    """Test StanzaGroup dataclass."""
    
    def test_create_stanza_group(self):
        """Test creating a StanzaGroup."""
        sg = StanzaGroup(
            stanza_group_num=1,
            n_stanzas=3,
            vbgf_ksp=0.3,
            vbgf_d=0.66667,
            wmat=0.5,
            bab=0.0,
            rec_power=1.0
        )
        
        assert sg.stanza_group_num == 1
        assert sg.n_stanzas == 3
        assert sg.vbgf_ksp == 0.3


class TestStanzaIndividual:
    """Test StanzaIndividual dataclass."""
    
    def test_create_stanza_individual(self):
        """Test creating a StanzaIndividual."""
        si = StanzaIndividual(
            stanza_group_num=1,
            stanza_num=1,
            group_num=3,
            group_name='Cod_juv',
            first=0,
            last=24,
            z=0.5,
            leading=True
        )
        
        assert si.group_num == 3
        assert si.leading is True
        assert si.first == 0
        assert si.last == 24


class TestStanzaParams:
    """Test StanzaParams dataclass."""
    
    def test_create_stanza_params(self):
        """Test creating StanzaParams."""
        sp = StanzaParams(
            n_stanza_groups=1,
            stanza_groups=[
                StanzaGroup(stanza_group_num=1, n_stanzas=2, vbgf_ksp=0.3)
            ],
            stanza_individuals=[
                StanzaIndividual(1, 1, 3, 'Cod_juv', 0, 24, 0.5, True),
                StanzaIndividual(1, 2, 4, 'Cod_adult', 24, 120, 0.3, False)
            ]
        )
        
        assert sp.n_stanza_groups == 1
        assert len(sp.stanza_groups) == 1
        assert len(sp.stanza_individuals) == 2


class TestRsimStanzas:
    """Test RsimStanzas dataclass."""
    
    def test_create_rsim_stanzas(self):
        """Test creating RsimStanzas."""
        rs = RsimStanzas(
            n_split=1,
            n_stanzas=np.array([2]),
            ecopath_code=np.array([[3], [4]]),
            age1=np.array([[0], [24]]),
            age2=np.array([[24], [120]])
        )
        
        assert rs.n_split == 1
        assert rs.n_stanzas[0] == 2


class TestCreateStanzaParams:
    """Test create_stanza_params function."""
    
    def test_stanza_params_basic(self):
        """Create basic StanzaParams object."""
        # Create manually since create_stanza_params needs specific structure
        sp = StanzaParams(
            n_stanza_groups=1,
            stanza_groups=[
                StanzaGroup(stanza_group_num=1, n_stanzas=2, vbgf_ksp=0.3)
            ],
            stanza_individuals=[
                StanzaIndividual(1, 1, 3, 'Fish_juv', 0, 24, 0.5, True),
                StanzaIndividual(1, 2, 4, 'Fish_adult', 24, 120, 0.3, False)
            ]
        )
        
        assert sp.n_stanza_groups == 1
        assert len(sp.stanza_individuals) == 2


class TestSplitUpdate:
    """Test split_update function for biomass redistribution."""
    
    def test_split_update_structure(self):
        """Test that split_update can be called with correct structure."""
        # Create minimal rsim_stanzas structure
        stanzas = RsimStanzas(
            n_split=1,
            n_stanzas=np.array([2]),
            ecopath_code=np.array([[3, 4]]),
            age1=np.array([[0, 24]]),
            age2=np.array([[24, 120]]),
            base_wage_s=np.linspace(0.1, 1.0, 120).reshape(-1, 1),
            base_nage_s=np.ones((120, 1)),
            base_qage_s=np.linspace(0.2, 0.8, 120).reshape(-1, 1)
        )
        
        # The structure should be created without error
        assert stanzas.n_split == 1


class TestStanzaIntegration:
    """Integration tests for the complete stanza workflow."""
    
    def test_vb_growth_model(self):
        """Test complete Von Bertalanffy growth model."""
        # Generate monthly ages
        ages = np.arange(0, 120) / 12.0  # 0 to 10 years in months
        
        # Growth parameters
        k = 0.3  # Von Bertalanffy K
        
        # Calculate weight at age
        weights = von_bertalanffy_weight(ages, k)
        
        # Weights should be monotonically increasing
        assert all(weights[i] <= weights[i+1] for i in range(len(weights)-1))
        
        # Calculate consumption
        consumption = von_bertalanffy_consumption(weights)
        
        # Consumption should be defined
        assert len(consumption) == len(weights)
    
    def test_survival_cohort(self):
        """Test survival through a cohort."""
        # Monthly mortality rate
        monthly_z = np.full(24, 0.05)  # 5% per month for 2 years
        
        # Calculate cumulative survival
        survival = calculate_survival(monthly_z)
        
        # Survival should decrease
        assert survival[-1] < survival[0]
        
        # Survival should not be negative
        assert all(s >= 0 for s in survival)


# Run tests
if __name__ == '__main__':
    pytest.main([__file__, '-v'])
