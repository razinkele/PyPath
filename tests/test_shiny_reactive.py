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
            # Avoid reactive context registration in unit tests
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


class TestComplexDataStructures:
    """Tests for complex data structures in reactive values."""

    def test_nested_dict_in_reactive_value(self):
        """Test nested dictionaries in reactive values."""
        try:
            from shiny import reactive

            value = reactive.Value(None)

            complex_data = {"level1": {"level2": {"level3": [1, 2, 3]}}}
            value.set(complex_data)

            retrieved = value._value
            assert retrieved["level1"]["level2"]["level3"] == [1, 2, 3]
        except ImportError:
            pytest.skip("Shiny not installed")

    def test_multiple_dataframes_in_reactive_value(self):
        """Test multiple DataFrames in a reactive value."""
        try:
            from shiny import reactive

            value = reactive.Value(None)

            data = {
                "model": pd.DataFrame({"A": [1, 2], "B": [3, 4]}),
                "diet": pd.DataFrame({"Predator": ["Fish"], "Prey": ["Plankton"]}),
                "catch": pd.DataFrame({"Species": ["Fish"], "Catch": [100]}),
            }
            value.set(data)

            retrieved = value._value
            assert "model" in retrieved
            assert "diet" in retrieved
            assert "catch" in retrieved
            assert len(retrieved["model"]) == 2
            assert len(retrieved["diet"]) == 1
        except ImportError:
            pytest.skip("Shiny not installed")

    def test_rpath_params_structure(self):
        """Test RpathParams-like structure in reactive value."""
        try:
            from shiny import reactive

            value = reactive.Value(None)

            class RpathParams:
                def __init__(self):
                    self.model = pd.DataFrame(
                        {
                            "Group": ["Phytoplankton", "Zooplankton", "Fish"],
                            "Type": [1, 1, 0],
                            "TL": [1.0, 2.0, 3.5],
                            "Biomass": [100.0, 50.0, 10.0],
                            "PB": [2.0, 1.5, 0.5],
                            "QB": [0.0, 3.0, 2.0],
                            "EE": [0.95, 0.9, 0.8],
                        }
                    )
                    self.diet = pd.DataFrame(
                        {"Zooplankton": [0.8, 0.0, 0.0], "Fish": [0.0, 1.0, 0.0]},
                        index=["Phytoplankton", "Zooplankton", "Fish"],
                    )
                    self.landing = pd.DataFrame()
                    self.discard = pd.DataFrame()
                    self.balanced = False

            params = RpathParams()
            value.set(params)

            retrieved = value._value
            assert hasattr(retrieved, "model")
            assert hasattr(retrieved, "diet")
            assert len(retrieved.model) == 3
            assert retrieved.balanced is False
        except ImportError:
            pytest.skip("Shiny not installed")


class TestReactiveErrorHandling:
    """Tests for error handling in reactive contexts."""

    def test_reactive_value_with_none(self):
        """Test reactive value handles None correctly."""
        try:
            from shiny import reactive

            value = reactive.Value(None)
            assert value._value is None

            # Setting to None should work
            value.set(None)
            assert value._value is None
        except ImportError:
            pytest.skip("Shiny not installed")

    def test_reactive_value_type_changes(self):
        """Test reactive value can change types."""
        try:
            from shiny import reactive

            value = reactive.Value(None)

            # Start with None
            assert value._value is None

            # Change to int
            value.set(42)
            assert value._value == 42
            assert isinstance(value._value, int)

            # Change to string
            value.set("test")
            assert value._value == "test"
            assert isinstance(value._value, str)

            # Change to dict
            value.set({"key": "value"})
            assert value._value["key"] == "value"
            assert isinstance(value._value, dict)

            # Back to None
            value.set(None)
            assert value._value is None
        except ImportError:
            pytest.skip("Shiny not installed")


class TestMultipleReactiveEffects:
    """Tests for multiple reactive effects watching the same value."""

    def test_multiple_watchers_same_value(self):
        """Test multiple components can watch the same reactive value."""
        try:
            from shiny import reactive

            model_data = reactive.Value(None)

            # Simulate multiple pages watching model_data
            watchers = []

            for i in range(3):

                class Watcher:
                    def __init__(self, model_data_ref, watcher_id):
                        self.model_data = model_data_ref
                        self.id = watcher_id
                        self.last_seen = None

                    def check(self):
                        self.last_seen = self.model_data._value
                        return self.last_seen

                watchers.append(Watcher(model_data, i))

            # Update model_data
            test_data = {"update": "broadcast"}
            model_data.set(test_data)

            # All watchers should see the update
            for watcher in watchers:
                assert watcher.check() == test_data
        except ImportError:
            pytest.skip("Shiny not installed")


class TestReactivePerformance:
    """Tests for reactive value performance characteristics."""

    def test_large_dataframe_in_reactive_value(self):
        """Test reactive value with large DataFrame."""
        try:
            import time

            from shiny import reactive

            value = reactive.Value(None)

            # Create large DataFrame
            large_df = pd.DataFrame(
                {
                    "col1": np.random.rand(10000),
                    "col2": np.random.rand(10000),
                    "col3": np.random.randint(0, 100, 10000),
                }
            )

            # Set value
            start = time.time()
            value.set(large_df)
            set_time = time.time() - start

            # Get value
            start = time.time()
            retrieved = value._value
            get_time = time.time() - start

            # Verify data integrity
            pd.testing.assert_frame_equal(retrieved, large_df)

            # Performance should be reasonable (< 1 second for these operations)
            assert set_time < 1.0
            assert get_time < 1.0
        except ImportError:
            pytest.skip("Shiny not installed")

    def test_frequent_updates(self):
        """Test reactive value with frequent updates."""
        try:
            from shiny import reactive

            value = reactive.Value(0)

            # Perform many updates
            for i in range(1000):
                value.set(i)

            # Final value should be correct
            assert value._value == 999
        except ImportError:
            pytest.skip("Shiny not installed")


if __name__ == "__main__":
    pytest.main([__file__, "-v"]) 
