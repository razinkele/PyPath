"""Test suite for model_importer module.

Tests batch loading, caching, filtering, and comparative analysis of EwE models.
"""

import pytest
import tempfile
import shutil
from pathlib import Path
import pandas as pd
import numpy as np

from pypath.io.model_importer import (
    ModelRegistry,
    ModelMetadata,
    ModelStatistics,
    load_model_directory,
    quick_load,
)
from pypath.io.ewe_official import EweModel


# Use persistent test data directory
TEST_MODELS_DIR = Path("/home/razinka/EPath/docs/TestModels")


class TestModelStatistics:
    """Test ModelStatistics calculation."""
    
    def test_calculate_empty_model(self):
        """Test statistics for empty model."""
        model = EweModel(
            name="Empty",
            groups=pd.DataFrame(),
            diet=None
        )
        stats = ModelStatistics.calculate(model)
        
        assert stats.n_groups == 0
        assert stats.n_links == 0
        assert stats.connectivity == 0.0
    
    def test_calculate_small_model(self):
        """Test statistics for small model with diet."""
        groups = pd.DataFrame({
            'Name': ['Fish', 'Zooplankton', 'Detritus'],
            'Biomass': [1.0, 2.0, 3.0]
        })
        
        diet = np.array([
            [0.0, 0.0, 0.0],  # Fish diet
            [0.5, 0.0, 0.0],  # Zooplankton diet
            [0.0, 0.5, 0.0]   # Detritus diet
        ])
        
        model = EweModel(
            name="Small",
            groups=groups,
            diet=pd.DataFrame(diet)
        )
        
        stats = ModelStatistics.calculate(model)
        
        assert stats.n_groups == 3
        assert stats.n_links == 2
        assert stats.connectivity == pytest.approx(2.0 / 9.0, rel=0.01)
        assert stats.avg_diet_items > 0


class TestModelRegistry:
    """Test ModelRegistry functionality."""
    
    def test_init_empty_registry(self):
        """Test registry initialization."""
        registry = ModelRegistry()
        
        assert len(registry.models) == 0
        assert len(registry.metadata) == 0
        assert len(registry.statistics) == 0
    
    def test_init_with_directory(self):
        """Test registry with directory."""
        registry = ModelRegistry(directory=TEST_MODELS_DIR)
        
        assert registry.directory == TEST_MODELS_DIR
    
    @pytest.mark.skipif(
        not TEST_MODELS_DIR.exists(),
        reason="Test models directory not found"
    )
    def test_load_single_model(self):
        """Test loading a single model."""
        registry = ModelRegistry()
        model_path = TEST_MODELS_DIR / "Generic_37.sqlite"
        
        if model_path.exists():
            model_name = registry.load_model(model_path, validate=False)
            
            assert model_name in registry.models
            assert model_name in registry.metadata
            assert model_name in registry.statistics
            assert registry.models[model_name] is not None
    
    @pytest.mark.skipif(
        not TEST_MODELS_DIR.exists(),
        reason="Test models directory not found"
    )
    def test_load_directory(self):
        """Test loading all models from directory."""
        registry = ModelRegistry()
        loaded = registry.load_directory(
            TEST_MODELS_DIR,
            pattern="*.sqlite",
            validate=False,
            recursive=False
        )
        
        assert len(loaded) > 0, "No models loaded from directory"
        assert all(name in registry.models for name in loaded.values())
    
    @pytest.mark.skipif(
        not TEST_MODELS_DIR.exists(),
        reason="Test models directory not found"
    )
    def test_get_model(self):
        """Test retrieving a model."""
        registry = ModelRegistry()
        registry.load_directory(TEST_MODELS_DIR, validate=False)
        
        if len(registry.models) > 0:
            model_name = list(registry.models.keys())[0]
            model = registry.get_model(model_name)
            
            assert model is not None
            assert isinstance(model, EweModel)
    
    @pytest.mark.skipif(
        not TEST_MODELS_DIR.exists(),
        reason="Test models directory not found"
    )
    def test_get_nonexistent_model(self):
        """Test retrieving nonexistent model."""
        registry = ModelRegistry()
        
        model = registry.get_model("NonExistent")
        assert model is None
    
    @pytest.mark.skipif(
        not TEST_MODELS_DIR.exists(),
        reason="Test models directory not found"
    )
    def test_list_models(self):
        """Test listing models."""
        registry = ModelRegistry()
        registry.load_directory(TEST_MODELS_DIR, validate=False)
        
        model_list = registry.list_models()
        
        assert len(model_list) > 0
        assert all(isinstance(item, tuple) and len(item) == 3 for item in model_list)
        
        # Check format: (name, n_groups, n_links)
        for name, n_groups, n_links in model_list:
            assert isinstance(name, str)
            assert isinstance(n_groups, (int, np.integer))
            assert isinstance(n_links, (int, np.integer))
    
    @pytest.mark.skipif(
        not TEST_MODELS_DIR.exists(),
        reason="Test models directory not found"
    )
    def test_list_models_sorted(self):
        """Test sorting models."""
        registry = ModelRegistry()
        registry.load_directory(TEST_MODELS_DIR, validate=False)
        
        # Sort by groups
        by_groups = registry.list_models(sort_by="groups")
        groups_list = [item[1] for item in by_groups]
        assert groups_list == sorted(groups_list, reverse=True)
        
        # Sort by links
        by_links = registry.list_models(sort_by="links")
        links_list = [item[2] for item in by_links]
        assert links_list == sorted(links_list, reverse=True)
    
    @pytest.mark.skipif(
        not TEST_MODELS_DIR.exists(),
        reason="Test models directory not found"
    )
    def test_filter_by_groups(self):
        """Test filtering by group count."""
        registry = ModelRegistry()
        registry.load_directory(TEST_MODELS_DIR, validate=False)
        
        # Find a reasonable range
        all_groups = [s.n_groups for s in registry.statistics.values()]
        if len(all_groups) > 0:
            min_val = int(np.percentile(all_groups, 25))
            max_val = int(np.percentile(all_groups, 75))
            
            filtered = registry.filter_models(
                min_groups=min_val,
                max_groups=max_val
            )
            
            for name, model in filtered.items():
                n_groups = registry.statistics[name].n_groups
                assert min_val <= n_groups <= max_val
    
    @pytest.mark.skipif(
        not TEST_MODELS_DIR.exists(),
        reason="Test models directory not found"
    )
    def test_compare_models(self):
        """Test model comparison."""
        registry = ModelRegistry()
        registry.load_directory(TEST_MODELS_DIR, validate=False)
        
        if len(registry.models) >= 2:
            model_names = list(registry.models.keys())[:2]
            comparison = registry.compare_models(
                models=model_names,
                metrics=['groups', 'links', 'connectivity']
            )
            
            assert isinstance(comparison, pd.DataFrame)
            assert len(comparison) == 2
            assert 'groups' in comparison.columns
            assert 'links' in comparison.columns
    
    @pytest.mark.skipif(
        not TEST_MODELS_DIR.exists(),
        reason="Test models directory not found"
    )
    def test_compare_all_models(self):
        """Test comparison of all models."""
        registry = ModelRegistry()
        registry.load_directory(TEST_MODELS_DIR, validate=False)
        
        comparison = registry.compare_models()
        
        assert isinstance(comparison, pd.DataFrame)
        assert len(comparison) == len(registry.models)
        assert 'groups' in comparison.columns


class TestModelMetadata:
    """Test ModelMetadata functionality."""
    
    def test_metadata_creation(self):
        """Test creating metadata."""
        path = Path("/tmp/test_model.sqlite")
        meta = ModelMetadata(
            name="TestModel",
            file_path=path,
            load_time_ms=50.5
        )
        
        assert meta.name == "TestModel"
        assert meta.file_path == path
        assert meta.load_time_ms == 50.5
    
    def test_metadata_to_dict(self):
        """Test converting metadata to dictionary."""
        path = Path("/tmp/test_model.sqlite")
        meta = ModelMetadata(
            name="TestModel",
            file_path=path,
            model_type="Reference",
            notes="Test notes"
        )
        
        meta_dict = meta.to_dict()
        
        assert isinstance(meta_dict, dict)
        assert meta_dict['name'] == "TestModel"
        assert meta_dict['file_path'] == str(path)
        assert meta_dict['model_type'] == "Reference"
        assert meta_dict['notes'] == "Test notes"


class TestConvenienceFunctions:
    """Test convenience functions."""
    
    @pytest.mark.skipif(
        not TEST_MODELS_DIR.exists(),
        reason="Test models directory not found"
    )
    def test_load_model_directory(self):
        """Test load_model_directory convenience function."""
        registry = load_model_directory(TEST_MODELS_DIR, validate=False)
        
        assert isinstance(registry, ModelRegistry)
        assert len(registry.models) > 0
        assert registry.directory == TEST_MODELS_DIR
    
    @pytest.mark.skipif(
        not TEST_MODELS_DIR.exists(),
        reason="Test models directory not found"
    )
    def test_quick_load(self):
        """Test quick_load convenience function."""
        model_path = TEST_MODELS_DIR / "Generic_37.sqlite"
        
        if model_path.exists():
            model = quick_load(str(model_path))
            
            assert isinstance(model, EweModel)
            assert model.groups is not None
            assert len(model.groups) > 0


class TestExportFunctionality:
    """Test export functionality."""
    
    @pytest.mark.skipif(
        not TEST_MODELS_DIR.exists(),
        reason="Test models directory not found"
    )
    def test_export_csv(self):
        """Test exporting to CSV."""
        with tempfile.TemporaryDirectory() as tmpdir:
            registry = load_model_directory(TEST_MODELS_DIR, validate=False)
            
            output_path = Path(tmpdir) / "models_summary.csv"
            registry.export_summary(output_path, format="csv")
            
            assert output_path.exists()
            
            # Verify CSV content
            df = pd.read_csv(output_path, index_col=0)
            assert len(df) > 0
    
    @pytest.mark.skipif(
        not TEST_MODELS_DIR.exists(),
        reason="Test models directory not found"
    )
    def test_export_json(self):
        """Test exporting to JSON."""
        with tempfile.TemporaryDirectory() as tmpdir:
            registry = load_model_directory(TEST_MODELS_DIR, validate=False)
            
            output_path = Path(tmpdir) / "models_summary.json"
            registry.export_summary(output_path, format="json")
            
            assert output_path.exists()
            
            # Verify JSON content
            import json
            with open(output_path) as f:
                data = json.load(f)
            assert len(data) > 0
    
    @pytest.mark.skipif(
        not TEST_MODELS_DIR.exists(),
        reason="Test models directory not found"
    )
    def test_export_markdown(self):
        """Test exporting to Markdown."""
        with tempfile.TemporaryDirectory() as tmpdir:
            registry = load_model_directory(TEST_MODELS_DIR, validate=False)
            
            output_path = Path(tmpdir) / "models_summary.md"
            registry.export_summary(output_path, format="md")
            
            assert output_path.exists()
            
            # Verify Markdown content
            content = output_path.read_text()
            assert "Model Comparison" in content
            assert "Model Details" in content


class TestIntegration:
    """Integration tests for model importer."""
    
    @pytest.mark.skipif(
        not TEST_MODELS_DIR.exists(),
        reason="Test models directory not found"
    )
    def test_workflow_load_analyze_export(self):
        """Test complete workflow: load -> analyze -> export."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Load
            registry = load_model_directory(TEST_MODELS_DIR, validate=False)
            
            # Analyze
            assert len(registry.models) > 0
            
            # List
            model_list = registry.list_models()
            assert len(model_list) > 0
            
            # Filter
            filtered = registry.filter_models()
            assert len(filtered) > 0
            
            # Export
            output_path = Path(tmpdir) / "summary.csv"
            registry.export_summary(output_path)
            assert output_path.exists()
    
    @pytest.mark.skipif(
        not TEST_MODELS_DIR.exists(),
        reason="Test models directory not found"
    )
    def test_curonian_70_groups_model(self):
        """Test loading Curonian 70-group model specifically."""
        registry = ModelRegistry()
        model_path = TEST_MODELS_DIR / "Curonian_Lagoon_70_groups.sqlite"
        
        if model_path.exists():
            model_name = registry.load_model(model_path, validate=False)
            
            assert model_name in registry.models
            
            stats = registry.statistics[model_name]
            assert stats.n_groups == 70, f"Expected 70 groups, got {stats.n_groups}"
            assert stats.n_links > 0, "Expected non-zero links"
    
    @pytest.mark.skipif(
        not TEST_MODELS_DIR.exists(),
        reason="Test models directory not found"
    )
    def test_model_comparison_curonian_vs_tampa(self):
        """Test comparing Curonian and Tampa Bay models."""
        registry = ModelRegistry()
        
        curonian_path = TEST_MODELS_DIR / "Curonian_Lagoon_70_groups.sqlite"
        tampa_path = TEST_MODELS_DIR / "Tampa_Bay.sqlite"
        
        models_to_compare = []
        if curonian_path.exists():
            registry.load_model(curonian_path, validate=False)
            models_to_compare.append("Curonian_Lagoon_70_groups")
        
        if tampa_path.exists():
            registry.load_model(tampa_path, validate=False)
            models_to_compare.append("Tampa_Bay")
        
        if len(models_to_compare) == 2:
            comparison = registry.compare_models(models=models_to_compare)
            
            assert len(comparison) == 2
            
            # Curonian should have more groups
            curonian_groups = comparison.loc["Curonian_Lagoon_70_groups", "groups"]
            tampa_groups = comparison.loc["Tampa_Bay", "groups"]
            assert curonian_groups > tampa_groups


class TestErrorHandling:
    """Test error handling."""
    
    def test_load_nonexistent_file(self):
        """Test loading nonexistent file."""
        registry = ModelRegistry()
        
        with pytest.raises(FileNotFoundError):
            registry.load_model("/nonexistent/path/model.sqlite")
    
    def test_load_directory_not_found(self):
        """Test loading from nonexistent directory."""
        registry = ModelRegistry()
        
        with pytest.raises(FileNotFoundError):
            registry.load_directory("/nonexistent/directory")
    
    def test_export_with_no_models(self):
        """Test exporting empty registry."""
        with tempfile.TemporaryDirectory() as tmpdir:
            registry = ModelRegistry()
            
            output_path = Path(tmpdir) / "empty.csv"
            # Should not raise, but create empty file
            registry.export_summary(output_path, format="csv")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
