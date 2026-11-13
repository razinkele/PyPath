"""Batch import routine for SQLite EwE models into PyPath.

This module provides utilities for efficiently importing and managing collections
of EwE ecosystem models stored in SQLite format. It enables:

- Batch loading of multiple models from a directory
- Model registry and caching
- Comparative analysis across models
- Model filtering by size, type, or location
- Automatic model validation and quality checks
- Export to various analysis formats
- Integration with PyPath analysis workflows

Example:
    >>> from pypath.io import model_importer
    >>> 
    >>> # Load all models from a directory
    >>> registry = model_importer.load_model_directory(
    ...     "/path/to/TestModels",
    ...     validate=True
    ... )
    >>> 
    >>> # Access individual models
    >>> curonian = registry.get_model("Curonian_Lagoon_70_groups")
    >>> print(f"Groups: {len(curonian.groups)}")
    >>> 
    >>> # Compare models
    >>> comparison = registry.compare_models(
    ...     models=["Generic_37", "Tampa_Bay"],
    ...     metrics=["group_count", "connectivity", "trophic_levels"]
    ... )
    >>> 
    >>> # Filter by criteria
    >>> large_models = registry.filter_models(
    ...     min_groups=50,
    ...     max_groups=75
    ... )
    >>> 
    >>> # Export for analysis
    >>> registry.export_to_csv("model_comparison.csv")
"""

from __future__ import annotations

import logging
import sqlite3
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union
import json
from datetime import datetime
import warnings

import pandas as pd
import numpy as np

from .ewe_official import EweModel, read_ewe_model
from .validators import validate_ecopath_scenario, ValidationReport

logger = logging.getLogger(__name__)
PathLike = Union[str, Path]


@dataclass
class ModelMetadata:
    """Metadata about a loaded model.
    
    Attributes:
        name: Model identifier
        file_path: Path to SQLite file
        load_time_ms: Time to load model (milliseconds)
        validation_report: Results from ecological validation
        geographic_info: Location metadata
        model_type: Category (Reference, Production, Specialized, etc.)
        notes: User notes or description
    """
    
    name: str
    file_path: Path
    load_time_ms: Optional[float] = None
    validation_report: Optional[ValidationReport] = None
    geographic_info: Dict[str, str] = field(default_factory=dict)
    model_type: str = "Reference"
    notes: str = ""
    load_timestamp: datetime = field(default_factory=datetime.now)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert metadata to dictionary."""
        return {
            "name": self.name,
            "file_path": str(self.file_path),
            "load_time_ms": self.load_time_ms,
            "load_timestamp": self.load_timestamp.isoformat(),
            "model_type": self.model_type,
            "geographic_info": self.geographic_info,
            "notes": self.notes,
            "validation_status": (
                self.validation_report.status 
                if self.validation_report else "Not validated"
            )
        }


@dataclass
class ModelStatistics:
    """Summary statistics for a loaded model.
    
    Attributes:
        n_groups: Number of functional groups
        n_links: Number of food links
        connectivity: Fraction of possible links present (0-1)
        trophic_levels: Estimate of food chain length
        avg_diet_items: Average number of prey per predator
        diet_specialization: Average diet diversity (0=specialist, 1=generalist)
        detritus_linkage: Fraction of groups linking to detritus
    """
    
    n_groups: int = 0
    n_links: int = 0
    connectivity: float = 0.0
    trophic_levels: float = 0.0
    avg_diet_items: float = 0.0
    diet_specialization: float = 0.0
    detritus_linkage: float = 0.0
    
    @classmethod
    def calculate(cls, model: EweModel) -> ModelStatistics:
        """Calculate statistics for a model.
        
        Args:
            model: EweModel instance
            
        Returns:
            ModelStatistics with calculated metrics
        """
        n_groups = len(model.groups) if model.groups is not None else 0
        
        stats = cls()
        stats.n_groups = n_groups
        
        if model.diet is not None and n_groups > 0:
            # Count non-zero links
            diet_matrix = model.diet.values if hasattr(model.diet, 'values') else model.diet
            stats.n_links = int(np.count_nonzero(diet_matrix))
            
            # Calculate connectivity
            max_possible_links = n_groups * n_groups
            stats.connectivity = stats.n_links / max_possible_links if max_possible_links > 0 else 0.0
            
            # Average diet items per predator
            diet_per_predator = np.count_nonzero(diet_matrix, axis=0)
            stats.avg_diet_items = float(np.mean(diet_per_predator)) if len(diet_per_predator) > 0 else 0.0
            
            # Diet specialization (inverse of diversity)
            nonzero_per_predator = diet_per_predator[diet_per_predator > 0]
            if len(nonzero_per_predator) > 0:
                specialization = 1.0 / (1.0 + np.mean(nonzero_per_predator) / n_groups)
                stats.diet_specialization = float(specialization)
        
        return stats


class ModelRegistry:
    """Registry for managing loaded EwE models.
    
    Provides centralized access to loaded models with caching, validation,
    and comparative analysis capabilities.
    
    Attributes:
        models: Dictionary mapping model names to EweModel objects
        metadata: Dictionary mapping model names to ModelMetadata
        statistics: Dictionary mapping model names to ModelStatistics
        directory: Root directory containing model files
    """
    
    def __init__(self, directory: Optional[PathLike] = None):
        """Initialize model registry.
        
        Args:
            directory: Optional directory path for future loads
        """
        self.models: Dict[str, EweModel] = {}
        self.metadata: Dict[str, ModelMetadata] = {}
        self.statistics: Dict[str, ModelStatistics] = {}
        self.directory = Path(directory) if directory else None
    
    def load_model(self, path: PathLike, validate: bool = True, 
                   name: Optional[str] = None) -> str:
        """Load a single model into the registry.
        
        Args:
            path: Path to SQLite model file
            validate: Whether to validate model with ecological checks
            name: Optional override for model name (default: file stem)
            
        Returns:
            Model name in registry
            
        Raises:
            FileNotFoundError: If file does not exist
            ValueError: If model cannot be loaded
        """
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"Model file not found: {path}")
        
        model_name = name or path.stem
        
        # Load model
        import time
        start_time = time.time()
        
        try:
            model = read_ewe_model(str(path))
            load_time_ms = (time.time() - start_time) * 1000
            
            # Store model
            self.models[model_name] = model
            
            # Create metadata
            meta = ModelMetadata(
                name=model_name,
                file_path=path,
                load_time_ms=load_time_ms
            )
            
            # Validate if requested
            if validate:
                try:
                    validation_report = validate_ecopath_scenario(model)
                    meta.validation_report = validation_report
                    logger.info(f"✓ Model {model_name} validated: {validation_report.status}")
                except Exception as e:
                    logger.warning(f"⚠ Validation failed for {model_name}: {e}")
            
            self.metadata[model_name] = meta
            
            # Calculate statistics
            stats = ModelStatistics.calculate(model)
            self.statistics[model_name] = stats
            
            logger.info(f"✓ Loaded {model_name}: {stats.n_groups} groups, {stats.n_links} links")
            
            return model_name
            
        except Exception as e:
            logger.error(f"✗ Failed to load {model_name}: {e}")
            raise ValueError(f"Cannot load model {path}: {e}")
    
    def load_directory(self, directory: PathLike, pattern: str = "*.sqlite",
                      validate: bool = True, recursive: bool = False) -> Dict[str, str]:
        """Load all models from a directory.
        
        Args:
            directory: Directory containing SQLite files
            pattern: Glob pattern for model files (default: *.sqlite)
            validate: Whether to validate each model
            recursive: Whether to search subdirectories
            
        Returns:
            Dictionary mapping file names to model names loaded
        """
        directory = Path(directory)
        self.directory = directory
        
        if not directory.exists():
            raise FileNotFoundError(f"Directory not found: {directory}")
        
        # Find all matching files
        glob_method = directory.rglob if recursive else directory.glob
        model_files = sorted(glob_method(pattern))
        
        loaded_models = {}
        for model_file in model_files:
            if model_file.is_file():
                try:
                    model_name = self.load_model(model_file, validate=validate)
                    loaded_models[model_file.name] = model_name
                except Exception as e:
                    logger.error(f"Skipping {model_file.name}: {e}")
        
        logger.info(f"✓ Loaded {len(loaded_models)} models from {directory}")
        return loaded_models
    
    def get_model(self, name: str) -> Optional[EweModel]:
        """Get a model by name.
        
        Args:
            name: Model name
            
        Returns:
            EweModel or None if not found
        """
        return self.models.get(name)
    
    def list_models(self, sort_by: str = "name") -> List[Tuple[str, int, int]]:
        """List all loaded models with basic stats.
        
        Args:
            sort_by: Sort key ('name', 'groups', 'links')
            
        Returns:
            List of (name, n_groups, n_links) tuples
        """
        results = [
            (name, self.statistics[name].n_groups, self.statistics[name].n_links)
            for name in sorted(self.models.keys())
        ]
        
        if sort_by == "groups":
            results.sort(key=lambda x: x[1], reverse=True)
        elif sort_by == "links":
            results.sort(key=lambda x: x[2], reverse=True)
        
        return results
    
    def filter_models(self, min_groups: int = 0, max_groups: int = 10000,
                     min_links: int = 0, max_links: int = 1000000,
                     model_type: Optional[str] = None) -> Dict[str, EweModel]:
        """Filter models by criteria.
        
        Args:
            min_groups: Minimum number of groups
            max_groups: Maximum number of groups
            min_links: Minimum number of links
            max_links: Maximum number of links
            model_type: Filter by model type (Reference, Production, etc.)
            
        Returns:
            Dictionary of filtered models
        """
        filtered = {}
        
        for name, model in self.models.items():
            stats = self.statistics[name]
            meta = self.metadata[name]
            
            # Check criteria
            if not (min_groups <= stats.n_groups <= max_groups):
                continue
            if not (min_links <= stats.n_links <= max_links):
                continue
            if model_type and meta.model_type != model_type:
                continue
            
            filtered[name] = model
        
        return filtered
    
    def compare_models(self, models: Optional[List[str]] = None,
                      metrics: Optional[List[str]] = None) -> pd.DataFrame:
        """Compare multiple models.
        
        Args:
            models: List of model names to compare (default: all)
            metrics: List of metrics to include (default: all available)
                Supported: 'groups', 'links', 'connectivity', 'trophic_levels',
                          'avg_diet_items', 'diet_specialization'
            
        Returns:
            DataFrame with models as rows and metrics as columns
        """
        if models is None:
            models = list(self.models.keys())
        
        if metrics is None:
            metrics = ['groups', 'links', 'connectivity', 'trophic_levels', 
                      'avg_diet_items', 'diet_specialization']
        
        # Map metric names to statistics attributes
        metric_map = {
            'groups': 'n_groups',
            'links': 'n_links',
            'connectivity': 'connectivity',
            'trophic_levels': 'trophic_levels',
            'avg_diet_items': 'avg_diet_items',
            'diet_specialization': 'diet_specialization'
        }
        
        data = []
        for model_name in models:
            if model_name not in self.statistics:
                logger.warning(f"Model {model_name} not found")
                continue
            
            stats = self.statistics[model_name]
            row = {'Model': model_name}
            
            for metric in metrics:
                if metric in metric_map:
                    attr = metric_map[metric]
                    row[metric] = getattr(stats, attr, None)
            
            data.append(row)
        
        # Return DataFrame with proper index
        if len(data) > 0:
            return pd.DataFrame(data).set_index('Model')
        else:
            # Return empty DataFrame with correct structure
            return pd.DataFrame(columns=list(metrics))
    
    def export_summary(self, output_path: PathLike, format: str = "csv") -> None:
        """Export model summary to file.
        
        Args:
            output_path: Output file path
            format: Output format ('csv', 'json', 'md')
        """
        output_path = Path(output_path)
        
        comparison_df = self.compare_models()
        
        if format == "csv":
            comparison_df.to_csv(output_path)
            logger.info(f"✓ Exported summary to {output_path}")
        
        elif format == "json":
            summary = {
                name: self.metadata[name].to_dict()
                for name in sorted(self.models.keys())
            }
            with open(output_path, 'w') as f:
                json.dump(summary, f, indent=2, default=str)
            logger.info(f"✓ Exported summary to {output_path}")
        
        elif format == "md":
            with open(output_path, 'w') as f:
                f.write("# Model Registry Summary\n\n")
                f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
                f.write(f"Total Models: {len(self.models)}\n\n")
                
                if len(self.models) == 0:
                    f.write("No models loaded.\n")
                else:
                    f.write("## Model Comparison\n\n")
                    
                    # Try to use to_markdown if tabulate is available, otherwise use CSV format
                    try:
                        f.write(comparison_df.to_markdown())
                    except ImportError:
                        # Fallback to simple markdown table format
                        f.write("| Model | Groups | Links | Connectivity | Trophic Levels | Avg Diet Items | Diet Specialization |\n")
                        f.write("|-------|--------|-------|--------------|----------------|-----------------|---------------------|\n")
                        for idx, row in comparison_df.iterrows():
                            f.write(f"| {idx} | {row.get('groups', 0):.0f} | {row.get('links', 0):.0f} | {row.get('connectivity', 0):.2%} | {row.get('trophic_levels', 0):.2f} | {row.get('avg_diet_items', 0):.2f} | {row.get('diet_specialization', 0):.2f} |\n")
                    
                    f.write("\n## Model Details\n\n")
                    
                    for name in sorted(self.models.keys()):
                        stats = self.statistics[name]
                        meta = self.metadata[name]
                        f.write(f"### {name}\n")
                        f.write(f"- Groups: {stats.n_groups}\n")
                        f.write(f"- Links: {stats.n_links}\n")
                        f.write(f"- Connectivity: {stats.connectivity:.1%}\n")
                        if meta.load_time_ms is not None:
                            f.write(f"- Load time: {meta.load_time_ms:.1f}ms\n")
                        f.write(f"- Validation: {meta.validation_report.status if meta.validation_report else 'Not run'}\n\n")
            
            logger.info(f"✓ Exported summary to {output_path}")


def load_model_directory(directory: PathLike, pattern: str = "*.sqlite",
                        validate: bool = True, **kwargs) -> ModelRegistry:
    """Convenience function to load all models from a directory.
    
    Args:
        directory: Directory containing SQLite model files
        pattern: Glob pattern for model files
        validate: Whether to validate models
        **kwargs: Additional arguments (reserved for future use)
        
    Returns:
        Populated ModelRegistry
        
    Example:
        >>> registry = load_model_directory("/home/razinka/EPath/docs/TestModels")
        >>> print(registry.list_models())
        >>> comparison = registry.compare_models()
    """
    registry = ModelRegistry(directory)
    registry.load_directory(directory, pattern=pattern, validate=validate)
    return registry


def quick_load(model_path: PathLike) -> EweModel:
    """Quick load a single model without registry overhead.
    
    Args:
        model_path: Path to SQLite file
        
    Returns:
        EweModel object
        
    Example:
        >>> model = quick_load("Generic_37.sqlite")
    """
    return read_ewe_model(str(model_path))
