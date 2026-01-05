"""
Tests for reactive behaviors and state management in the Shiny dashboard.

Tests reactivity patterns, state synchronization, and data propagation.
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

# Add app directory to path
app_dir = Path(__file__).parent.parent / "app"
sys.path.insert(0, str(app_dir))


class TestReactiveValues:
    """Tests for reactive value behavior."""

    def test_reactive_value_creation(self):
        """Test creating reactive values."""
        try:
            from shiny import reactive

            # Create reactive values
            value1 = reactive.Value(None)
            value2 = reactive.Value(0)
            value3 = reactive.Value("test")

            assert value1._value is None
            assert value2._value == 0
            assert value3._value == "test"
        except ImportError:
            pytest.skip("Shiny not installed")

    def test_reactive_value_updates(self):
        """Test updating reactive values."""
        try:
            from shiny import reactive

            value = reactive.Value(0)
            assert value._value == 0

            value.set(10)
            assert value._value == 10

            value.set(None)
            assert value._value is None
        except ImportError:
            pytest.skip("Shiny not installed")

    def test_reactive_value_with_dataframe(self):
        """Test reactive values containing DataFrames."""
        try:
            from shiny import reactive

            df_value = reactive.Value(None)
            assert df_value._value is None

            df = pd.DataFrame({"A": [1, 2, 3], "B": [4, 5, 6]})
            df_value.set(df)

            retrieved_df = df_value._value
            assert retrieved_df is not None
            assert len(retrieved_df) == 3
            pd.testing.assert_frame_equal(retrieved_df, df)
        except ImportError:
            pytest.skip("Shiny not installed")


class TestSharedDataReactivity:
    """Tests for SharedData reactivity patterns."""

    def test_shared_data_initialization(self):
        """Test SharedData initializes correctly."""
        try:
            from shiny import reactive

            model_data = reactive.Value(None)
            sim_results = reactive.Value(None)

            class SharedData:
                def __init__(self, model_data_ref, sim_results_ref):
                    self.model_data = model_data_ref
                    self.sim_results = sim_results_ref
                    self.params = reactive.Value(None)

            shared = SharedData(model_data, sim_results)

            # Test initial state
            assert shared.model_data._value is None
            assert shared.sim_results._value is None
            assert shared.params._value is None
        except ImportError:
            pytest.skip("Shiny not installed")

    def test_shared_data_references(self):
        """Test that SharedData correctly references reactive values."""
        try:
            from shiny import reactive

            model_data = reactive.Value(None)
            sim_results = reactive.Value(None)

            class SharedData:
                def __init__(self, model_data_ref, sim_results_ref):
                    self.model_data = model_data_ref
                    self.sim_results = sim_results_ref
                    self.params = reactive.Value(None)

            shared = SharedData(model_data, sim_results)

            # Update model_data
            test_value = {"test": "data"}
            model_data.set(test_value)

            # SharedData should see the update
            assert shared.model_data._value == test_value
            assert shared.model_data._value is model_data._value
        except ImportError:
            pytest.skip("Shiny not installed")

    def test_shared_data_params_sync(self):
        """Test synchronization of model_data to params."""
        try:
            from shiny import reactive

            model_data = reactive.Value(None)
            sim_results = reactive.Value(None)

            class SharedData:
                def __init__(self, model_data_ref, sim_results_ref):
                    self.model_data = model_data_ref
                    self.sim_results = sim_results_ref
                    self.params = reactive.Value(None)

            shared = SharedData(model_data, sim_results)

            # Create mock params
            class MockRpathParams:
                def __init__(self):
                    self.model = pd.DataFrame({"Group": ["A"]})
                    self.diet = pd.DataFrame()

            params = MockRpathParams()
            model_data.set(params)

            # Simulate sync (as done in app.py)
            data = model_data._value
            if data is not None and hasattr(data, "model") and hasattr(data, "diet"):
                shared.params.set(data)

            # Verify sync worked
            assert shared.params._value is not None
            assert shared.params._value is params
        except ImportError:
            pytest.skip("Shiny not installed")


class TestDataPropagation:
    """Tests for data propagation between reactive values."""

    def test_model_data_propagation(self):
        """Test that model_data changes propagate correctly."""
        try:
            from shiny import reactive

            model_data = reactive.Value(None)

            # Stage 1: Initial import
            initial_data = {"stage": "import", "groups": 5}
            model_data.set(initial_data)
            assert model_data._value["stage"] == "import"

            # Stage 2: After balancing
            balanced_data = {"stage": "balanced", "groups": 5, "balanced": True}
            model_data.set(balanced_data)
            assert model_data._value["stage"] == "balanced"
            assert model_data._value["balanced"] is True

            # Stage 3: Ready for simulation
            sim_ready_data = {
                "stage": "sim_ready",
                "groups": 5,
                "balanced": True,
                "params": "configured",
            }
            model_data.set(sim_ready_data)
            assert model_data._value["params"] == "configured"
        except ImportError:
            pytest.skip("Shiny not installed")

    def test_sim_results_propagation(self):
        """Test that simulation results propagate correctly."""
        try:
            from shiny import reactive

            sim_results = reactive.Value(None)

            # Initially no results
            assert sim_results._value is None

            # After simulation
            results = {
                "biomass": pd.DataFrame({"time": [0, 1], "Fish": [10, 11]}),
                "status": "complete",
            }
            sim_results.set(results)

            # Verify propagation
            assert sim_results._value is not None
            assert sim_results._value["status"] == "complete"
            assert "biomass" in sim_results._value
        except ImportError:
            pytest.skip("Shiny not installed")


class TestReactiveIsolation:
    """Tests for reactive isolation and independence."""

    def test_model_data_and_sim_results_independent(self):
        """Test that model_data and sim_results are independent."""
        try:
            from shiny import reactive

            model_data = reactive.Value(None)
            sim_results = reactive.Value(None)

            # Set model_data
            model_data.set({"test": "model"})
            assert sim_results._value is None  # sim_results unaffected

            # Set sim_results
            sim_results.set({"test": "results"})
            assert model_data._value["test"] == "model"  # model_data unchanged
        except ImportError:
            pytest.skip("Shiny not installed")

    def test_params_isolation_from_model_data(self):
        """Test that params can be different from model_data."""
        try:
            from shiny import reactive

            model_data = reactive.Value(None)

            class SharedData:
                def __init__(self, model_data_ref, sim_results_ref):
                    self.model_data = model_data_ref
                    self.sim_results = sim_results_ref
                    self.params = reactive.Value(None)

            shared = SharedData(model_data, reactive.Value(None))

            # Set model_data
            model_data.set({"data": "original"})

            # Set params to something different
            shared.params.set({"data": "modified"})

            # They should be independent
            assert model_data._value["data"] == "original"
            assert shared.params._value["data"] == "modified"
        except ImportError:
            pytest.skip("Shiny not installed")