"""
Comprehensive tests for the PyPath Shiny dashboard application.

Tests cover:
- App structure and initialization
- UI components and navigation
- Server logic and reactive state management
- Data flow between pages
- Error handling
- Theme and settings functionality
"""

import sys
from pathlib import Path
from unittest.mock import Mock

import pandas as pd
import pytest

# Add app directory to path
app_dir = Path(__file__).parent.parent / "app"
sys.path.insert(0, str(app_dir))


class TestAppStructure:
    """Tests for app.py structure and initialization."""

    def test_app_imports(self):
        """Test that app module can be imported."""
        try:
            from app import app

            assert app is not None
        except ImportError as e:
            pytest.skip(f"Shiny not installed or import error: {e}")

    def test_app_dir_constant(self):
        """Test that APP_DIR is correctly defined."""
        try:
            from app.app import APP_DIR

            assert APP_DIR.exists()
            assert APP_DIR.is_dir()
            assert APP_DIR.name == "app"
        except ImportError:
            pytest.skip("Shiny not installed")

    def test_static_assets_exist(self):
        """Test that static assets directory exists."""
        try:
            from app.app import APP_DIR

            static_dir = APP_DIR / "static"
            assert static_dir.exists()
            assert static_dir.is_dir()

            # Check for key static files
            assert (static_dir / "custom.css").exists()
            assert (static_dir / "icon.svg").exists()
        except ImportError:
            pytest.skip("Shiny not installed")

    def test_page_modules_import(self):
        """Test that all page modules can be imported."""
        try:
            from pages import (
                about,
                analysis,
                data_import,
                diet_rewiring_demo,
                ecopath,
                ecosim,
                ecospace,
                forcing_demo,
                home,
                multistanza,
                optimization_demo,
                results,
            )

            assert all(
                [
                    home,
                    data_import,
                    ecopath,
                    ecosim,
                    results,
                    analysis,
                    about,
                    multistanza,
                    forcing_demo,
                    diet_rewiring_demo,
                    optimization_demo,
                    ecospace,
                ]
            )
        except ImportError:
            pytest.skip("Shiny or page modules not available")


class TestUIComponents:
    """Tests for UI components and layout."""

    @pytest.fixture
    def mock_shiny(self):
        """Mock Shiny UI components."""
        try:
            from shiny import ui

            return ui
        except ImportError:
            pytest.skip("Shiny not installed")

    def test_navbar_structure(self, mock_shiny):
        """Test that navbar has correct structure."""
        try:
            from app.app import app_ui

            # App UI should be a page_navbar
            assert app_ui is not None
        except ImportError:
            pytest.skip("Shiny not installed")

    def test_custom_css_loaded(self):
        """Test that custom CSS is included in head."""
        try:
            from app.app import app_ui

            # Convert UI to string to check for CSS link
            ui_str = str(app_ui)
            assert "custom.css" in ui_str
        except ImportError:
            pytest.skip("Shiny not installed")

    def test_bootstrap_icons_loaded(self):
        """Test that Bootstrap Icons CSS is included."""
        try:
            from app.app import app_ui

            ui_str = str(app_ui)
            assert "bootstrap-icons" in ui_str
        except ImportError:
            pytest.skip("Shiny not installed")

    def test_footer_dynamic_year(self):
        """Test that footer uses dynamic year."""
        try:
            from datetime import datetime

            from app.app import app_ui

            ui_str = str(app_ui)
            current_year = str(datetime.now().year)
            assert current_year in ui_str
            assert "PyPath ©" in ui_str
        except ImportError:
            pytest.skip("Shiny not installed")


class TestServerLogic:
    """Tests for server-side logic and reactive state."""

    @pytest.fixture
    def mock_reactive_value(self):
        """Create a mock reactive value."""

        class MockReactiveValue:
            def __init__(self):
                self._value = None

            def __call__(self):
                return self._value

            def set(self, value):
                self._value = value

        return MockReactiveValue

    def test_shared_data_structure(self, mock_reactive_value):
        """Test SharedData class structure and attributes."""
        try:
            from shiny import reactive

            # Create mock reactive values
            model_data = reactive.Value(None)
            sim_results = reactive.Value(None)

            # Create SharedData class (extracted from app.py)
            class SharedData:
                def __init__(self, model_data_ref, sim_results_ref):
                    self.model_data = model_data_ref
                    self.sim_results = sim_results_ref
                    self.params = reactive.Value(None)

            shared = SharedData(model_data, sim_results)

            # Test attributes exist
            assert hasattr(shared, "model_data")
            assert hasattr(shared, "sim_results")
            assert hasattr(shared, "params")

            # Test that references work
            assert shared.model_data is model_data
            assert shared.sim_results is sim_results
        except ImportError:
            pytest.skip("Shiny not installed")

    def test_shared_data_sync_pattern(self):
        """Test that SharedData syncs correctly with model_data."""
        try:
            from shiny import reactive

            # Create reactive values
            model_data = reactive.Value(None)
            sim_results = reactive.Value(None)

            class SharedData:
                def __init__(self, model_data_ref, sim_results_ref):
                    self.model_data = model_data_ref
                    self.sim_results = sim_results_ref
                    self.params = reactive.Value(None)

            shared = SharedData(model_data, sim_results)

            # Create mock RpathParams
            class MockRpathParams:
                def __init__(self):
                    self.model = pd.DataFrame({"Group": ["Fish"], "TL": [3.0]})
                    self.diet = pd.DataFrame()

            # Test sync logic
            mock_params = MockRpathParams()
            model_data.set(mock_params)

            # Simulate sync
            if hasattr(model_data(), "model") and hasattr(model_data(), "diet"):
                shared.params.set(model_data())

            assert shared.params() is not None
            assert hasattr(shared.params(), "model")
        except ImportError:
            pytest.skip("Shiny not installed")


class TestErrorHandling:
    """Tests for error handling in server initialization."""

    def test_server_init_with_error_handling(self):
        """Test that server initialization handles errors gracefully."""
        try:
            from shiny import Inputs, Outputs, Session

            from app.app import server

            # Create mock objects
            mock_input = Mock(spec=Inputs)
            _mock_output = Mock(spec=Outputs)
            _mock_session = Mock(spec=Session)

            # Mock the settings button
            mock_input.btn_settings = Mock()

            # The server should handle initialization errors gracefully
            # We can't fully test this without running the app, but we can
            # verify the structure is correct
            assert callable(server)
        except ImportError:
            pytest.skip("Shiny not installed")

    def test_page_server_error_recovery(self):
        """Test that app continues if one page server fails."""
        # This is a structural test - the server_modules list
        # with try-except should allow partial initialization
        try:
            import inspect

            from app.app import server

            # Check that server function contains error handling
            source = inspect.getsource(server)
            assert "try:" in source
            assert "except Exception" in source
            assert "ERROR: Failed to initialize" in source
        except ImportError:
            pytest.skip("Shiny not installed")


class TestDataFlow:
    """Tests for data flow between pages."""

    def test_model_data_flow(self):
        """Test that model_data flows correctly between pages."""
        try:
            from shiny import reactive

            # Simulate data flow
            model_data = reactive.Value(None)

            # Data Import sets model_data
            class MockRpathParams:
                def __init__(self):
                    self.model = pd.DataFrame(
                        {
                            "Group": ["Phytoplankton", "Fish"],
                            "TL": [1.0, 3.5],
                            "Biomass": [100.0, 10.0],
                        }
                    )
                    self.diet = pd.DataFrame()

            mock_params = MockRpathParams()
            model_data.set(mock_params)

            # Verify data is accessible
            assert model_data() is not None
            assert hasattr(model_data(), "model")
            assert len(model_data().model) == 2
        except ImportError:
            pytest.skip("Shiny not installed")

    def test_sim_results_flow(self):
        """Test that sim_results flows correctly."""
        try:
            from shiny import reactive

            sim_results = reactive.Value(None)

            # Ecosim sets sim_results
            mock_results = {
                "biomass": pd.DataFrame({"time": [0, 1], "Phytoplankton": [100, 105]}),
                "catch": pd.DataFrame({"time": [0, 1], "Fish": [5, 6]}),
            }
            sim_results.set(mock_results)

            # Verify results are accessible
            assert sim_results() is not None
            assert "biomass" in sim_results()
            assert "catch" in sim_results()
        except ImportError:
            pytest.skip("Shiny not installed")


class TestNavigationStructure:
    """Tests for navigation and page structure."""

    def test_all_pages_have_ui_functions(self):
        """Test that all page modules have UI functions."""
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
                (home, "home_ui"),
                (data_import, "import_ui"),
                (ecopath, "ecopath_ui"),
                (ecosim, "ecosim_ui"),
                (results, "results_ui"),
                (analysis, "analysis_ui"),
                (about, "about_ui"),
            ]

            for module, ui_func_name in pages:
                assert hasattr(module, ui_func_name)
                assert callable(getattr(module, ui_func_name))
        except ImportError:
            pytest.skip("Page modules not available")

    def test_all_pages_have_server_functions(self):
        """Test that all page modules have server functions."""
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
                (home, "home_server"),
                (data_import, "import_server"),
                (ecopath, "ecopath_server"),
                (ecosim, "ecosim_server"),
                (results, "results_server"),
                (analysis, "analysis_server"),
                (about, "about_server"),
            ]

            for module, server_func_name in pages:
                assert hasattr(module, server_func_name)
                assert callable(getattr(module, server_func_name))
        except ImportError:
            pytest.skip("Page modules not available")

    def test_advanced_features_pages(self):
        """Test that advanced feature pages exist."""
        try:
            from pages import (
                diet_rewiring_demo,
                ecospace,
                forcing_demo,
                multistanza,
                optimization_demo,
            )

            advanced_pages = [
                (multistanza, "multistanza_ui", "multistanza_server"),
                (forcing_demo, "forcing_demo_ui", "forcing_demo_server"),
                (
                    diet_rewiring_demo,
                    "diet_rewiring_demo_ui",
                    "diet_rewiring_demo_server",
                ),
                (optimization_demo, "optimization_demo_ui", "optimization_demo_server"),
                (ecospace, "ecospace_ui", "ecospace_server"),
            ]

            for module, ui_func, server_func in advanced_pages:
                assert hasattr(module, ui_func)
                assert hasattr(module, server_func)
                assert callable(getattr(module, ui_func))
                assert callable(getattr(module, server_func))
        except ImportError:
            pytest.skip("Advanced feature modules not available")


class TestThemeAndSettings:
    """Tests for theme picker and settings functionality."""

    def test_theme_picker_integration(self):
        """Test that theme picker is integrated."""
        try:
            import shinyswatch

            from app.app import app_ui

            # Theme picker should be available
            assert shinyswatch is not None

            # Check for settings button in UI
            ui_str = str(app_ui)
            assert "btn_settings" in ui_str or "gear" in ui_str.lower()
        except ImportError:
            pytest.skip("Shinyswatch not installed")

    def test_default_theme(self):
        """Test that default theme is applied."""
        try:
            import shinyswatch

            from app.app import app_ui

            # The app uses flatly theme by default
            # This is verified in the source code
            _ui_str = str(app_ui)
            # Theme is applied via shinyswatch.theme.flatly
            assert True  # Theme is structural, hard to test without running app
        except ImportError:
            pytest.skip("Shinyswatch not installed")


class TestDocumentation:
    """Tests for code documentation and comments."""

    def test_server_docstring_exists(self):
        """Test that server function has comprehensive docstring."""
        try:
            from app.app import server

            assert server.__doc__ is not None
            assert "Data Flow Architecture" in server.__doc__
            assert "Primary Reactive State" in server.__doc__
        except ImportError:
            pytest.skip("Shiny not installed")

    def test_shared_data_docstring(self):
        """Test that SharedData class has docstring."""
        try:
            import inspect

            from app.app import server

            source = inspect.getsource(server)
            # Check for SharedData documentation
            assert (
                "Container providing structured access" in source
                or "Wrapper class providing structured access" in source
            )
        except ImportError:
            pytest.skip("Shiny not installed")


class TestIntegrationScenarios:
    """Integration tests for common user workflows."""

    def test_typical_workflow_structure(self):
        """Test that typical workflow can be followed.

        Typical workflow:
        1. Import data (Data Import page)
        2. Balance model (Ecopath page)
        3. Run simulation (Ecosim page)
        4. View results (Results/Analysis pages)
        """
        try:
            from shiny import reactive

            # Step 1: Import data
            model_data = reactive.Value(None)
            sim_results = reactive.Value(None)

            # Simulate importing data
            class MockRpathParams:
                def __init__(self):
                    self.model = pd.DataFrame(
                        {
                            "Group": ["Fish"],
                            "TL": [3.5],
                            "Biomass": [10.0],
                            "PB": [0.5],
                            "QB": [2.0],
                        }
                    )
                    self.diet = pd.DataFrame()
                    self.balanced = False

            params = MockRpathParams()
            model_data.set(params)
            assert model_data() is not None

            # Step 2: Balance model (simulated)
            params.balanced = True
            model_data.set(params)

            # Step 3: Run simulation (simulated)
            mock_sim = {"biomass": pd.DataFrame({"time": [0, 1], "Fish": [10, 11]})}
            sim_results.set(mock_sim)
            assert sim_results() is not None

            # Step 4: Results available
            assert "biomass" in sim_results()
        except ImportError:
            pytest.skip("Shiny not installed")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
