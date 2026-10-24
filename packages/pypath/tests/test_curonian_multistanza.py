"""
Test script for multi-stanza Curonian Lagoon model.

Validates that the multi-stanza configuration is correctly set up
and can integrate with Ecosim.
"""

import sys
from pathlib import Path

import pytest

from pypath.io.ewemdb import read_ewemdb

# Add scripts to path for importing
sys.path.insert(0, str(Path(__file__).parent.parent / "scripts"))

from curonian_multistanza_example import (
    create_stanza_definitions,
    distribute_biomass_across_stanzas,
    setup_multistanza_groups,
)


class TestStanzaDefinitions:
    """Test stanza group definitions."""

    def test_stanza_definitions_created(self):
        """Test that stanza definitions are created."""
        stanza_defs = create_stanza_definitions()
        assert len(stanza_defs) == 2
        assert 8 in stanza_defs  # Demersal
        assert 14 in stanza_defs  # Predatory

    def test_demersal_stanza_config(self):
        """Test demersal fish stanza configuration."""
        stanza_defs = create_stanza_definitions()
        demersal = stanza_defs[8]

        assert demersal.group_idx == 8
        assert demersal.n_stanzas == 2
        assert demersal.vbgf_k == 0.15
        assert demersal.juvenile_z == 1.2
        assert demersal.adult_z == 0.6

    def test_predatory_stanza_config(self):
        """Test predatory fish stanza configuration."""
        stanza_defs = create_stanza_definitions()
        predatory = stanza_defs[14]

        assert predatory.group_idx == 14
        assert predatory.n_stanzas == 3
        assert predatory.vbgf_k == 0.12
        assert predatory.juvenile_z == 1.5
        assert predatory.adult_z == 0.5


class TestBiomassDistribution:
    """Test biomass distribution across stanzas."""

    def test_biomass_conservation(self):
        """Total biomass across stanzas should equal original."""
        import numpy as np
        original = 1.777
        ages = np.array([12, 36])
        weights = np.array([0.1, 0.5])

        stanza_biomass = distribute_biomass_across_stanzas(original, 2, weights)

        # Total should equal original
        np.testing.assert_almost_equal(stanza_biomass.sum(), original)

    def test_biomass_distribution_proportional(self):
        """Biomass distribution should be proportional to weights."""
        import numpy as np
        original = 1.0
        weights = np.array([0.3, 0.7])  # 30-70 split

        stanza_biomass = distribute_biomass_across_stanzas(original, 2, weights)

        # Check proportions
        np.testing.assert_almost_equal(stanza_biomass[0] / stanza_biomass[1], 0.3 / 0.7)


class TestMultistanzaSetup:
    """Test multi-stanza configuration setup."""

    @pytest.fixture
    def ecopath_params(self):
        """Load Curonian Ecopath model."""
        model_path = r"C:\Users\DELL\OneDrive - ku.lt\ECOPATH\ECOPATH paper\curonian pelagic.eweaccdb"
        return read_ewemdb(model_path)

    def test_stanza_setup_creates_groups(self, ecopath_params):
        """Test that stanza setup creates correct number of stanza individuals."""
        stanza_defs = create_stanza_definitions()
        stanzas = setup_multistanza_groups(ecopath_params, stanza_defs)

        # 2 stanzas for demersal + 3 for predatory
        assert len(stanzas.stanza_individuals) == 5

    def test_stanza_group_count(self, ecopath_params):
        """Test that stanza groups are correctly configured."""
        stanza_defs = create_stanza_definitions()
        stanzas = setup_multistanza_groups(ecopath_params, stanza_defs)

        # Should have 2 stanza groups
        assert len(stanzas.stanza_groups) == 2
        assert stanzas.n_stanza_groups == 2

    def test_stanza_individual_properties(self, ecopath_params):
        """Test StanzaIndividual properties."""
        stanza_defs = create_stanza_definitions()
        stanzas = setup_multistanza_groups(ecopath_params, stanza_defs)

        # Check first individual (demersal juvenile)
        demersal_juv = stanzas.stanza_individuals[0]
        assert demersal_juv.first == 0
        assert demersal_juv.last == 24
        assert demersal_juv.z == 1.2
        assert demersal_juv.leading == True

        # Check second individual (demersal adult)
        demersal_ad = stanzas.stanza_individuals[1]
        assert demersal_ad.first == 24
        assert demersal_ad.last == 48
        assert demersal_ad.z == 0.6
        assert demersal_ad.leading == False

    def test_predatory_three_stanzas(self, ecopath_params):
        """Test predatory fish has 3 stanzas."""
        stanza_defs = create_stanza_definitions()
        stanzas = setup_multistanza_groups(ecopath_params, stanza_defs)

        # Predatory stanzas are indices 2, 3, 4
        pred_stanzas = [si for si in stanzas.stanza_individuals if si.stanza_group_num == 2]

        assert len(pred_stanzas) == 3

        # Check mortality gradient
        assert pred_stanzas[0].z > pred_stanzas[1].z > pred_stanzas[2].z
        assert pred_stanzas[0].z == 1.5  # Juvenile
        assert pred_stanzas[2].z == 0.5  # Adult


class TestModelIntegration:
    """Test integration with PyPath core."""

    @pytest.fixture
    def ecopath_params(self):
        """Load Curonian Ecopath model."""
        model_path = r"C:\Users\DELL\OneDrive - ku.lt\ECOPATH\ECOPATH paper\curonian pelagic.eweaccdb"
        return read_ewemdb(model_path)

    def test_can_create_ecopath_model(self, ecopath_params):
        """Test that model can be balanced with Ecopath."""
        from pypath.core.ecopath import rpath

        model = rpath(ecopath_params)

        # Should create successfully
        assert model is not None
        assert model.NUM_GROUPS == len(ecopath_params.model)

    def test_stanzas_preserve_original_biomass(self, ecopath_params):
        """Test that demersal and predatory biomass is preserved."""
        stanza_defs = create_stanza_definitions()
        stanzas = setup_multistanza_groups(ecopath_params, stanza_defs)

        # Get original biomasses
        orig_demersal = ecopath_params.model.iloc[8]["Biomass"]
        orig_predatory = ecopath_params.model.iloc[14]["Biomass"]

        # Sum stanza biomasses
        demersal_stanzas = [si for si in stanzas.stanza_individuals if si.stanza_group_num == 1]
        predatory_stanzas = [si for si in stanzas.stanza_individuals if si.stanza_group_num == 2]

        stanza_demersal = sum(si.biomass for si in demersal_stanzas)
        stanza_predatory = sum(si.biomass for si in predatory_stanzas)

        # Should be approximately equal (within rounding)
        import numpy as np
        np.testing.assert_almost_equal(stanza_demersal, orig_demersal, decimal=2)
        np.testing.assert_almost_equal(stanza_predatory, orig_predatory, decimal=2)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
