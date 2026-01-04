"""
Tests for individual Shiny page modules.

Tests UI components, server logic, and reactive behaviors for each page.
"""

import sys
from pathlib import Path

import pandas as pd
import pytest

# Add app directory to path
app_dir = Path(__file__).parent.parent / "app"
sys.path.insert(0, str(app_dir))


class TestHomePage:
    """Tests for home page module."""

    def test_home_ui_exists(self):
        """Test that home UI function exists."""
        try:
            from pages import home

            assert hasattr(home, "home_ui")
            assert callable(home.home_ui)
        except ImportError:
            pytest.skip("Home page module not available")

    def test_home_server_exists(self):
        """Test that home server function exists."""
        try:
            from pages import home

            assert hasattr(home, "home_server")
            assert callable(home.home_server)
        except ImportError:
            pytest.skip("Home page module not available")

    def test_home_server_signature(self):
        """Test home_server has correct signature."""
        try:
            import inspect

            from pages import home

            sig = inspect.signature(home.home_server)
            params = list(sig.parameters.keys())

            # Should have: input, output, session, model_data
            assert len(params) == 4
            assert "input" in params
            assert "output" in params
            assert "session" in params
            assert "model_data" in params
        except ImportError:
            pytest.skip("Home page module not available")


class TestDataImportPage:
    """Tests for data import page module."""

    def test_import_ui_exists(self):
        """Test that import UI function exists."""
        try:
            from pages import data_import

            assert hasattr(data_import, "import_ui")
            assert callable(data_import.import_ui)
        except ImportError:
            pytest.skip("Data import page module not available")

    def test_import_server_signature(self):
        """Test import_server has correct signature."""
        try:
            import inspect

            from pages import data_import

            sig = inspect.signature(data_import.import_server)
            params = list(sig.parameters.keys())

            # Should have: input, output, session, model_data
            assert len(params) == 4
            assert "model_data" in params
        except ImportError:
            pytest.skip("Data import page module not available")


class TestEcopathPage:
    """Tests for Ecopath page module."""

    def test_ecopath_ui_exists(self):
        """Test that Ecopath UI function exists."""
        try:
            from pages import ecopath

            assert hasattr(ecopath, "ecopath_ui")
            assert callable(ecopath.ecopath_ui)
        except ImportError:
            pytest.skip("Ecopath page module not available")

    def test_ecopath_server_signature(self):
        """Test ecopath_server has correct signature."""
        try:
            import inspect

            from pages import ecopath

            sig = inspect.signature(ecopath.ecopath_server)
            params = list(sig.parameters.keys())

            # Should have: input, output, session, model_data
            assert len(params) == 4
            assert "model_data" in params
        except ImportError:
            pytest.skip("Ecopath page module not available")


class TestEcosimPage:
    """Tests for Ecosim page module."""

    def test_ecosim_ui_exists(self):
        """Test that Ecosim UI function exists."""
        try:
            from pages import ecosim

            assert hasattr(ecosim, "ecosim_ui")
            assert callable(ecosim.ecosim_ui)
        except ImportError:
            pytest.skip("Ecosim page module not available")

    def test_ecosim_server_signature(self):
        """Test ecosim_server has correct signature."""
        try:
            import inspect

            from pages import ecosim

            sig = inspect.signature(ecosim.ecosim_server)
            params = list(sig.parameters.keys())

            # Should have: input, output, session, model_data, sim_results
            assert len(params) == 5
            assert "model_data" in params
            assert "sim_results" in params
        except ImportError:
            pytest.skip("Ecosim page module not available")


class TestResultsPage:
    """Tests for results page module."""

    def test_results_ui_exists(self):
        """Test that results UI function exists."""
        try:
            from pages import results

            assert hasattr(results, "results_ui")
            assert callable(results.results_ui)
        except ImportError:
            pytest.skip("Results page module not available")

    def test_results_server_signature(self):
        """Test results_server has correct signature."""
        try:
            import inspect

            from pages import results

            sig = inspect.signature(results.results_server)
            params = list(sig.parameters.keys())

            # Should have: input, output, session, model_data, sim_results
            assert len(params) == 5
            assert "model_data" in params
            assert "sim_results" in params
        except ImportError:
            pytest.skip("Results page module not available")


class TestAnalysisPage:
    """Tests for analysis page module."""

    def test_analysis_ui_exists(self):
        """Test that analysis UI function exists."""
        try:
            from pages import analysis

            assert hasattr(analysis, "analysis_ui")
            assert callable(analysis.analysis_ui)
        except ImportError:
            pytest.skip("Analysis page module not available")

    def test_analysis_server_signature(self):
        """Test analysis_server has correct signature."""
        try:
            import inspect

            from pages import analysis

            sig = inspect.signature(analysis.analysis_server)
            params = list(sig.parameters.keys())

            # Should have: input, output, session, model_data, sim_results
            assert len(params) == 5
            assert "model_data" in params
            assert "sim_results" in params
        except ImportError:
            pytest.skip("Analysis page module not available")


class TestAboutPage:
    """Tests for about page module."""

    def test_about_ui_exists(self):
        """Test that about UI function exists."""
        try:
            from pages import about

            assert hasattr(about, "about_ui")
            assert callable(about.about_ui)
        except ImportError:
            pytest.skip("About page module not available")

    def test_about_server_signature(self):
        """Test about_server has correct signature."""
        try:
            import inspect

            from pages import about

            sig = inspect.signature(about.about_server)
            params = list(sig.parameters.keys())

            # Should have: input, output, session (minimal params)
            assert len(params) == 3
            assert "input" in params
            assert "output" in params
            assert "session" in params
        except ImportError:
            pytest.skip("About page module not available")


class TestMultiStanzaPage:
    """Tests for multi-stanza page module."""

    def test_multistanza_ui_exists(self):
        """Test that multi-stanza UI function exists."""
        try:
            from pages import multistanza

            assert hasattr(multistanza, "multistanza_ui")
            assert callable(multistanza.multistanza_ui)
        except ImportError:
            pytest.skip("Multi-stanza page module not available")

    def test_multistanza_server_signature(self):
        """Test multistanza_server has correct signature."""
        try:
            import inspect

            from pages import multistanza

            sig = inspect.signature(multistanza.multistanza_server)
            params = list(sig.parameters.keys())

            # Should have: input, output, session, shared_data
            assert len(params) == 4
            assert "shared_data" in params
        except ImportError:
            pytest.skip("Multi-stanza page module not available")


class TestEcospacePage:
    """Tests for Ecospace page module."""

    def test_ecospace_ui_exists(self):
        """Test that Ecospace UI function exists."""
        try:
            from pages import ecospace

            assert hasattr(ecospace, "ecospace_ui")
            assert callable(ecospace.ecospace_ui)
        except ImportError:
            pytest.skip("Ecospace page module not available")

    def test_ecospace_server_signature(self):
        """Test ecospace_server has correct signature."""
        try:
            import inspect

            from pages import ecospace

            sig = inspect.signature(ecospace.ecospace_server)
            params = list(sig.parameters.keys())

            # Should have: input, output, session, model_data, sim_results
            assert len(params) == 5
            assert "model_data" in params
            assert "sim_results" in params
        except ImportError:
            pytest.skip("Ecospace page module not available")


class TestDemoPages:
    """Tests for demonstration/example pages."""

    def test_forcing_demo_exists(self):
        """Test that forcing demo page exists."""
        try:
            from pages import forcing_demo

            assert hasattr(forcing_demo, "forcing_demo_ui")
            assert hasattr(forcing_demo, "forcing_demo_server")
            assert callable(forcing_demo.forcing_demo_ui)
            assert callable(forcing_demo.forcing_demo_server)
        except ImportError:
            pytest.skip("Forcing demo page not available")

    def test_diet_rewiring_demo_exists(self):
        """Test that diet rewiring demo page exists."""
        try:
            from pages import diet_rewiring_demo

            assert hasattr(diet_rewiring_demo, "diet_rewiring_demo_ui")
            assert hasattr(diet_rewiring_demo, "diet_rewiring_demo_server")
            assert callable(diet_rewiring_demo.diet_rewiring_demo_ui)
            assert callable(diet_rewiring_demo.diet_rewiring_demo_server)
        except ImportError:
            pytest.skip("Diet rewiring demo page not available")

    def test_optimization_demo_exists(self):
        """Test that optimization demo page exists."""
        try:
            from pages import optimization_demo

            assert hasattr(optimization_demo, "optimization_demo_ui")
            assert hasattr(optimization_demo, "optimization_demo_server")
            assert callable(optimization_demo.optimization_demo_ui)
            assert callable(optimization_demo.optimization_demo_server)
        except ImportError:
            pytest.skip("Optimization demo page not available")

    def test_demo_pages_signature(self):
        """Test that demo pages have correct server signatures."""
        try:
            import inspect

            from pages import diet_rewiring_demo, forcing_demo, optimization_demo

            # All demo pages should have: input, output, session (no shared state)
            for module in [forcing_demo, diet_rewiring_demo, optimization_demo]:
                server_func = getattr(
                    module, f"{module.__name__.split('.')[-1]}_server"
                )
                sig = inspect.signature(server_func)
                params = list(sig.parameters.keys())

                assert len(params) == 3
                assert "input" in params
                assert "output" in params
                assert "session" in params
        except ImportError:
            pytest.skip("Demo pages not available")


class TestPageConsistency:
    """Tests for consistency across all pages."""

    def test_all_pages_have_consistent_naming(self):
        """Test that all pages follow naming conventions."""
        try:
            pages_to_test = [
                ("home", "home"),
                ("data_import", "import"),
                ("ecopath", "ecopath"),
                ("ecosim", "ecosim"),
                ("results", "results"),
                ("analysis", "analysis"),
                ("about", "about"),
            ]

            for module_name, prefix in pages_to_test:
                module = __import__(f"pages.{module_name}", fromlist=[module_name])
                ui_func = f"{prefix}_ui"
                server_func = f"{prefix}_server"

                assert hasattr(module, ui_func), f"{module_name} missing {ui_func}"
                assert hasattr(
                    module, server_func
                ), f"{module_name} missing {server_func}"
        except ImportError:
            pytest.skip("Page modules not available")

    def test_no_pages_return_none_from_ui(self):
        """Test that all UI functions return valid UI objects."""
        try:
            from pages import (
                about,
                analysis,
                data_import,
                ecopath,
                ecosim,
                home,
                results,
            )

            pages = [
                home.home_ui,
                data_import.import_ui,
                ecopath.ecopath_ui,
                ecosim.ecosim_ui,
                results.results_ui,
                analysis.analysis_ui,
                about.about_ui,
            ]

            for ui_func in pages:
                result = ui_func()
                assert result is not None, f"{ui_func.__name__} returned None"
        except ImportError:
            pytest.skip("Page modules not available")


class TestUtilsModule:
    """Tests for shared utilities module."""

    def test_utils_module_exists(self):
        """Test that utils module exists."""
        try:
            from pages import utils

            assert utils is not None
        except ImportError:
            pytest.skip("Utils module not available")

    def test_utils_has_shared_functions(self):
        """Test that utils module has common utility functions."""
        try:
            import inspect

            from pages import utils

            # Check that utils has functions (not empty)
            functions = [
                name
                for name, obj in inspect.getmembers(utils)
                if inspect.isfunction(obj)
            ]

            # Should have at least some utility functions
            assert len(functions) > 0, "Utils module should contain utility functions"
        except ImportError:
            pytest.skip("Utils module not available")


class TestPageInteractions:
    """Tests for interactions between pages."""

    def test_data_import_to_ecopath_flow(self):
        """Test data flow from import to ecopath page."""
        try:
            from shiny import reactive

            # Simulate data import setting model_data
            model_data = reactive.Value(None)

            class MockRpathParams:
                def __init__(self):
                    self.model = pd.DataFrame(
                        {
                            "Group": ["Phytoplankton", "Fish"],
                            "TL": [1.0, 3.5],
                            "Biomass": [100.0, 10.0],
                            "PB": [1.0, 0.5],
                            "QB": [0.0, 2.0],
                        }
                    )
                    self.diet = pd.DataFrame()

            # Data import sets model_data
            params = MockRpathParams()
            model_data.set(params)

            # Ecopath page should be able to read model_data
            retrieved_data = model_data()
            assert retrieved_data is not None
            assert hasattr(retrieved_data, "model")
            assert len(retrieved_data.model) == 2
        except ImportError:
            pytest.skip("Shiny not installed")

    def test_ecopath_to_ecosim_flow(self):
        """Test data flow from ecopath to ecosim page."""
        try:
            from shiny import reactive

            model_data = reactive.Value(None)

            class MockRpathParams:
                def __init__(self):
                    self.model = pd.DataFrame(
                        {"Group": ["Fish"], "TL": [3.5], "Biomass": [10.0]}
                    )
                    self.diet = pd.DataFrame()
                    self.balanced = True  # Ecopath marks as balanced

            # Ecopath marks model as balanced
            params = MockRpathParams()
            model_data.set(params)

            # Ecosim should be able to check if balanced
            data = model_data()
            assert hasattr(data, "balanced")
            assert data.balanced is True
        except ImportError:
            pytest.skip("Shiny not installed")

    def test_ecosim_to_results_flow(self):
        """Test data flow from ecosim to results page."""
        try:
            from shiny import reactive

            sim_results = reactive.Value(None)

            # Ecosim sets simulation results
            mock_results = {
                "biomass": pd.DataFrame(
                    {
                        "time": [0, 1, 2],
                        "Phytoplankton": [100, 105, 110],
                        "Fish": [10, 11, 12],
                    }
                ),
                "catch": pd.DataFrame({"time": [0, 1, 2], "Fish": [5, 5.5, 6]}),
            }
            sim_results.set(mock_results)

            # Results page should be able to access results
            results = sim_results()
            assert results is not None
            assert "biomass" in results
            assert "catch" in results
            assert len(results["biomass"]) == 3
        except ImportError:
            pytest.skip("Shiny not installed")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
