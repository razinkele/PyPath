"""
Tests for PyPath Ecosim simulation functionality.
"""

import pytest
import numpy as np
import warnings

from pypath.core.params import create_rpath_params
from pypath.core.ecopath import rpath
from pypath.core.ecosim import (
    rsim_params,
    rsim_state,
    rsim_forcing,
    rsim_fishing,
    rsim_scenario,
    rsim_run,
    RsimScenario,
    RsimState,
)
from pypath.core.ecosim_deriv import deriv_vector, integrate_rk4
from pypath.core.stanzas import RsimStanzas


@pytest.fixture
def simple_model():
    """Create a simple balanced Ecopath model for testing."""
    params = create_rpath_params(
        groups=['Phyto', 'Zoo', 'Fish', 'Det', 'Fleet'],
        types=[1, 0, 0, 2, 3]
    )
    
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
    
    params.model['BioAcc'] = 0.0
    params.model['Unassim'] = 0.2
    params.model.loc[0, 'Unassim'] = 0.0
    params.model.loc[3, 'Unassim'] = 0.0
    params.model.loc[4, 'BioAcc'] = np.nan
    params.model.loc[4, 'Unassim'] = np.nan
    
    params.model['Det'] = 1.0
    params.model.loc[4, 'Det'] = np.nan
    
    params.diet['Zoo'] = [1.0, 0.0, 0.0, 0.0, 0.0]
    params.diet['Fish'] = [0.0, 1.0, 0.0, 0.0, 0.0]
    params.diet['Phyto'] = [0.0, 0.0, 0.0, 0.0, 0.0]
    
    params.model.loc[2, 'Fleet'] = 0.5
    
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        model = rpath(params)
    
    return model, params


class TestRsimParams:
    """Tests for rsim_params conversion."""
    
    def test_rsim_params_creation(self, simple_model):
        """Test that rsim_params creates valid parameters."""
        model, _ = simple_model
        params = rsim_params(model)
        
        assert params.NUM_GROUPS == 5
        assert params.NUM_LIVING == 3
        assert params.NUM_DEAD == 1
        assert params.NUM_GEARS == 1
        assert len(params.spname) == 6  # Includes "Outside"
        assert params.spname[0] == "Outside"
    
    def test_biomass_reference(self, simple_model):
        """Test that reference biomass is correct."""
        model, _ = simple_model
        params = rsim_params(model)
        
        # B_BaseRef[0] should be 1.0 (Outside)
        assert params.B_BaseRef[0] == 1.0
        
        # Other groups should match original model
        assert np.isclose(params.B_BaseRef[1], 10.0)  # Phyto
        assert np.isclose(params.B_BaseRef[2], 5.0)   # Zoo
        assert np.isclose(params.B_BaseRef[3], 2.0)   # Fish
    
    def test_predprey_links(self, simple_model):
        """Test predator-prey link construction."""
        model, _ = simple_model
        params = rsim_params(model)
        
        # Should have links for:
        # - Primary production (Outside -> Phyto)
        # - Zoo eating Phyto
        # - Fish eating Zoo
        assert params.NumPredPreyLinks >= 2


class TestRsimState:
    """Tests for initial state creation."""
    
    def test_state_creation(self, simple_model):
        """Test that initial state is created correctly."""
        model, _ = simple_model
        params = rsim_params(model)
        state = rsim_state(params)
        
        # Biomass should match reference
        assert np.allclose(state.Biomass, params.B_BaseRef)
        
        # Ftime should be 1.0
        assert np.allclose(state.Ftime, 1.0)


class TestRsimScenario:
    """Tests for scenario creation."""
    
    def test_scenario_creation(self, simple_model):
        """Test full scenario creation."""
        model, rpath_params = simple_model
        scenario = rsim_scenario(model, rpath_params, years=range(1, 11))
        
        assert scenario.params.NUM_GROUPS == 5
        assert scenario.forcing.ForcedBio.shape[0] == 10 * 12  # 10 years * 12 months
        assert scenario.fishing.ForcedEffort.shape[1] == 2  # Outside + 1 gear


class TestEcosimSimulation:
    """Tests for Ecosim simulation run."""
    
    def test_simulation_runs(self, simple_model):
        """Test that simulation runs without error."""
        model, rpath_params = simple_model
        scenario = rsim_scenario(model, rpath_params, years=range(1, 6))
        
        # This should run without raising an exception
        output = rsim_run(scenario, method='RK4')
        
        # Check output structure
        assert output.out_Biomass.shape[0] == 5 * 12 + 1  # 5 years * 12 months + initial
        assert output.out_Biomass.shape[1] == 6  # Outside + 5 groups
    
    def test_biomass_positive(self, simple_model):
        """Test that biomass stays positive."""
        model, rpath_params = simple_model
        scenario = rsim_scenario(model, rpath_params, years=range(1, 6))
        output = rsim_run(scenario, method='RK4')
        
        # All biomass should be positive (or very small epsilon)
        assert np.all(output.out_Biomass >= 0)
    
    def test_annual_output(self, simple_model):
        """Test annual output aggregation."""
        model, rpath_params = simple_model
        scenario = rsim_scenario(model, rpath_params, years=range(1, 6))
        output = rsim_run(scenario, method='RK4')
        
        assert output.annual_Biomass.shape[0] == 5  # 5 years
        assert output.annual_Catch.shape[0] == 5


class TestDerivVector:
    """Tests for derivative calculation function."""
    
    def test_deriv_output_shape(self, simple_model):
        """Test that deriv_vector returns correct shape."""
        model, _ = simple_model
        sim_params = rsim_params(model)
        
        state = np.array([1.0, 10.0, 5.0, 2.0, 100.0, 0.0])  # Outside + groups
        
        params_dict = {
            'NUM_GROUPS': sim_params.NUM_GROUPS,
            'NUM_LIVING': sim_params.NUM_LIVING,
            'NUM_DEAD': sim_params.NUM_DEAD,
            'NUM_GEARS': sim_params.NUM_GEARS,
            'PB': sim_params.PBopt,
            'QB': sim_params.FtimeQBOpt,
            'M0': sim_params.MzeroMort,
            'Unassim': sim_params.UnassimRespFrac,
            'ActiveLink': np.zeros((6, 6), dtype=bool),
            'VV': np.zeros((6, 6)),
            'DD': np.zeros((6, 6)),
            'QQbase': np.zeros((6, 6)),
            'Bbase': sim_params.B_BaseRef,
        }
        
        forcing_dict = {'Ftime': np.ones(6)}
        fishing_dict = {'FishingMort': np.zeros(6)}
        
        deriv = deriv_vector(state, params_dict, forcing_dict, fishing_dict)
        
        assert len(deriv) == 6  # Should match state length


class TestStanzaIntegration:
    """Tests for stanza integration in simulation."""
    
    def test_scenario_with_stanzas(self, simple_model):
        """Test that scenario can hold RsimStanzas."""
        model, rpath_params = simple_model
        scenario = rsim_scenario(model, rpath_params, years=range(1, 6))
        
        # Create a minimal stanza structure
        stanzas = RsimStanzas(n_split=0)
        scenario.stanzas = stanzas
        
        assert scenario.stanzas is not None
        assert scenario.stanzas.n_split == 0
    
    def test_simulation_with_empty_stanzas(self, simple_model):
        """Test simulation runs with empty stanza structure."""
        model, rpath_params = simple_model
        scenario = rsim_scenario(model, rpath_params, years=range(1, 6))
        
        # Add empty stanza structure
        stanzas = RsimStanzas(n_split=0)
        scenario.stanzas = stanzas
        
        # Should run without error (stanzas are skipped when n_split=0)
        output = rsim_run(scenario, method='RK4')
        
        assert output.out_Biomass.shape[0] == 5 * 12 + 1
        assert np.all(output.out_Biomass >= 0)
    
    def test_simulation_with_stanza_groups(self, simple_model):
        """Test simulation runs with initialized stanza groups."""
        model, rpath_params = simple_model
        scenario = rsim_scenario(model, rpath_params, years=range(1, 6))
        
        # Create stanza structure with 1 split group
        n_groups = 6
        max_age = 121  # 10 years in months + 1 (0-indexed, need 0 to 120)
        
        stanzas = RsimStanzas(
            n_split=1,
            n_stanzas=np.array([0, 2]),  # 2 stanzas for group 1
            ecopath_code=np.zeros((2, 3)),
            age1=np.zeros((2, 3)),
            age2=np.zeros((2, 3)),
            base_wage_s=np.zeros((max_age, 2)),
            base_nage_s=np.zeros((max_age, 2)),
            base_qage_s=np.zeros((max_age, 2)),  # Consumption at age
            base_spawn_bio=np.zeros(2),
            vbm=np.array([0.0, 0.2]),  # VB M parameter
            vbgf_d=np.array([0.0, 0.667]),
            wmat=np.array([0.0, 0.5]),
            spawn_x=np.array([0.0, 2.0]),  # Beverton-Holt parameter
            r_zero_s=np.array([0.0, 1000.0]),
            base_eggs_stanza=np.array([0.0, 100.0]),
            recruits=np.array([0.0, 100.0]),
            split_alpha=np.zeros((2, 3)),
            spawn_energy=np.zeros(2),
            r_scale_split=np.zeros(2),
            base_stanza_pred=np.zeros(2),
            rec_power=np.array([0.0, 1.0]),
        )
        
        # Set up ecopath codes and ages
        stanzas.ecopath_code[1, 1] = 2  # juvenile = group 2 (Zoo)
        stanzas.ecopath_code[1, 2] = 3  # adult = group 3 (Fish)
        stanzas.age1[1, 1] = 0
        stanzas.age2[1, 1] = 24  # juvenile 0-24 months
        stanzas.age1[1, 2] = 25
        stanzas.age2[1, 2] = 119  # adult 25-119 months (within bounds)
        
        # Initialize weight, numbers and consumption at age
        for age in range(max_age):
            stanzas.base_wage_s[age, 1] = 0.01 * (age + 1)
            stanzas.base_nage_s[age, 1] = 1000 * np.exp(-0.01 * age)
            stanzas.base_qage_s[age, 1] = (0.01 * (age + 1)) ** 0.667  # Q = W^d
        
        scenario.stanzas = stanzas
        
        # Should run without error
        output = rsim_run(scenario, method='RK4')
        
        assert output.out_Biomass.shape[0] == 5 * 12 + 1
        assert np.all(output.out_Biomass >= 0)


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
