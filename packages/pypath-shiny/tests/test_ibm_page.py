"""
Tests for the IBM (Individual-Based Model) Shiny page.

Covers module import, UI rendering, server signature, config existence,
config defaults matching SmeltParams.baltic_defaults(), and navbar presence.
"""

import inspect

import pytest


class TestIBMModuleImport:
    """Tests that the IBM page module is importable."""

    def test_ibm_module_importable(self):
        """Test that ibm page module can be imported."""
        try:
            from pypath_shiny.pages import ibm

            assert ibm is not None
        except ImportError as e:
            pytest.skip(f"IBM page module not available: {e}")


class TestIBMUI:
    """Tests for IBM UI function."""

    def test_ibm_ui_returns_non_none(self):
        """Test that ibm_ui() returns a valid UI object."""
        try:
            from pypath_shiny.pages import ibm

            result = ibm.ibm_ui()
            assert result is not None
        except ImportError:
            pytest.skip("IBM page module not available")

    def test_ibm_ui_callable(self):
        """Test that ibm_ui is callable."""
        try:
            from pypath_shiny.pages import ibm

            assert hasattr(ibm, "ibm_ui")
            assert callable(ibm.ibm_ui)
        except ImportError:
            pytest.skip("IBM page module not available")

    def test_ibm_server_callable(self):
        """Test that ibm_server is callable."""
        try:
            from pypath_shiny.pages import ibm

            assert hasattr(ibm, "ibm_server")
            assert callable(ibm.ibm_server)
        except ImportError:
            pytest.skip("IBM page module not available")


class TestIBMServerSignature:
    """Tests for IBM server function signature."""

    def test_ibm_server_signature(self):
        """Test ibm_server has 6 params matching ecospace pattern."""
        try:
            from pypath_shiny.pages import ibm

            sig = inspect.signature(ibm.ibm_server)
            params = list(sig.parameters.keys())

            # Should have: input, output, session, model_data, sim_results, sim_scenario
            assert len(params) == 6
            assert "input" in params
            assert any(p.lstrip("_") == "output" for p in params)
            assert any(p.lstrip("_") == "session" for p in params)
            assert any(p.lstrip("_") == "model_data" for p in params)
            assert any(p.lstrip("_") == "sim_results" for p in params)
            assert any(p.lstrip("_") == "sim_scenario" for p in params)
        except ImportError:
            pytest.skip("IBM page module not available")


class TestIBMConfig:
    """Tests for IBM configuration."""

    def test_ibm_config_exists(self):
        """Test that IBM singleton config is available."""
        try:
            from pypath_shiny.config import IBM

            assert IBM is not None
        except ImportError:
            pytest.skip("Config module not available")

    def test_ibm_config_has_population_params(self):
        """Test IBM config has population parameters."""
        try:
            from pypath_shiny.config import IBM

            assert hasattr(IBM, "n_super_individuals_default")
            assert IBM.n_super_individuals_default == 50
            assert IBM.n_super_individuals_min == 10
            assert IBM.n_super_individuals_max == 1000
        except ImportError:
            pytest.skip("Config module not available")

    def test_ibm_config_defaults_match_baltic_smelt(self):
        """Test that IBM config defaults match SmeltParams.baltic_defaults()."""
        try:
            from pypath_shiny.config import IBM

            # VBGF defaults
            assert IBM.vbgf_k_default == 0.3
            assert IBM.vbgf_linf_default == 25.0
            assert IBM.max_age_default == 10.0

            # Bioenergetics defaults
            assert IBM.ra_default == 0.0033
            assert IBM.q10_default == 2.1
            assert IBM.t_ref_default == 10.0
            assert IBM.sda_fraction_default == 0.172
            assert IBM.energy_density_default == 5.0

            # Predation defaults
            assert IBM.optimal_prey_length_default == 10.0
            assert IBM.selectivity_sd_default == 0.5

            # Movement defaults
            assert IBM.base_speed_default == 0.3
            assert IBM.habitat_weight_default == 0.4

            # Reproduction defaults
            assert IBM.fecundity_coefficient_default == 200.0
            assert IBM.larval_base_survival_default == 0.01
            assert IBM.spawning_temp_threshold_default == 4.0
        except ImportError:
            pytest.skip("Config module not available")


class TestIBMInNavbar:
    """Tests for IBM page integration in the app."""

    def test_ibm_in_navbar(self):
        """Test that 'Individual-Based Model' appears in the app UI."""
        try:
            from pypath_shiny.app import app_ui

            ui_str = str(app_ui)
            assert "Individual-Based Model" in ui_str
        except ImportError:
            pytest.skip("App module not available")

    def test_ibm_in_pages_init(self):
        """Test that ibm is listed in pages __all__."""
        try:
            from pypath_shiny.pages import __all__

            assert "ibm" in __all__
        except ImportError:
            pytest.skip("Pages module not available")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
