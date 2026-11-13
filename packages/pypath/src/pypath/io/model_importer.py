"""Batch import of EwE model databases into PyPath.

Loads every EwE Access database in a directory (``*.ewemdb``,
``*.eweaccdb``) with ``read_ewemdb`` into RpathParams, and summarises each
model for comparison: groups, diet links, connectance, trophic levels and
links to detritus. Models can be validated, filtered, compared and the
summary exported.

Example:
    >>> from pypath.io.model_importer import load_model_directory
    >>> registry = load_model_directory("Data")
    >>> registry.list_models()
    >>> registry.compare_models(metrics=["groups", "links", "trophic_levels"])
    >>> registry.export_summary("model_comparison.csv")
"""

from __future__ import annotations

import json
import logging
import time
from collections import Counter
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import pandas as pd

from pypath.core.params import RpathParams

from .validators import ValidationReport, validate_ecopath_scenario

logger = logging.getLogger(__name__)
PathLike = Union[str, Path]

#: EwE model database files
DEFAULT_PATTERNS = ("*.ewemdb", "*.eweaccdb")


@dataclass
class ModelMetadata:
    """Metadata about a loaded model.

    Attributes:
        name: Model identifier (file stem by default)
        file_path: Path to the model database
        load_time_ms: Time to load the model (milliseconds)
        validation_report: Result of ``validate_ecopath_scenario``
        geographic_info: Location metadata
        model_type: Category (Reference, Production, Specialized, ...)
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
        """Convert metadata to a dictionary."""
        return {
            "name": self.name,
            "file_path": str(self.file_path),
            "load_time_ms": self.load_time_ms,
            "load_timestamp": self.load_timestamp.isoformat(),
            "model_type": self.model_type,
            "geographic_info": self.geographic_info,
            "notes": self.notes,
            "validation_status": (
                "Not validated"
                if self.validation_report is None
                else ("passed" if self.validation_report.passed else "failed")
                + f" ({len(self.validation_report.errors)} errors, "
                f"{len(self.validation_report.warnings)} warnings)"
            ),
        }


@dataclass
class ModelStatistics:
    """Summary statistics of a model's food web.

    Attributes:
        n_groups: living and detritus groups (fleets excluded)
        n_links: positive entries of the prey x predator diet matrix
            (imports excluded)
        connectivity: connectance, n_links / n_groups^2
        trophic_levels: highest trophic level of a living group (the food
            chain length), from the diet without balancing: 1 for producers
            and detritus, 1 + the diet-weighted prey level for consumers
            (imports left out)
        avg_diet_items: links per consumer
        diet_specialization: 1 / (1 + mean prey per consumer / n_groups)
        detritus_linkage: share of consumers eating a detritus group
    """

    n_groups: int = 0
    n_links: int = 0
    connectivity: float = 0.0
    trophic_levels: float = 0.0
    avg_diet_items: float = 0.0
    diet_specialization: float = 0.0
    detritus_linkage: float = 0.0

    @classmethod
    def calculate(cls, params: RpathParams) -> "ModelStatistics":
        """Statistics of an RpathParams model (from ``read_ewemdb``)."""
        from pypath.analysis.prebalance import _calculate_trophic_levels

        model = params.model
        types = model["Type"].astype(float)
        groups = list(model.loc[types < 3, "Group"])
        consumers = list(model.loc[types < 1, "Group"])
        detritus = list(model.loc[types == 2, "Group"])
        stats = cls(n_groups=len(groups))
        if not groups or params.diet is None:
            return stats

        diet = params.diet.set_index("Group")
        prey = [g for g in groups if g in diet.index]
        preds = [g for g in consumers if g in diet.columns]
        matrix = diet.loc[prey, preds].apply(pd.to_numeric, errors="coerce").fillna(0)
        eats = matrix > 0
        stats.n_links = int(eats.values.sum())
        stats.connectivity = stats.n_links / len(groups) ** 2
        if preds:
            per_consumer = eats.sum(axis=0)
            stats.avg_diet_items = float(per_consumer.mean())
            fed = per_consumer[per_consumer > 0]
            if len(fed):
                stats.diet_specialization = float(
                    1.0 / (1.0 + fed.mean() / len(groups))
                )
            on_detritus = eats.loc[[d for d in detritus if d in eats.index]]
            stats.detritus_linkage = float(on_detritus.any(axis=0).sum() / len(preds))

        # The pre-balance routine looks prey up in the diet's index; an
        # RpathParams diet holds prey names in its "Group" column.
        tl = _calculate_trophic_levels(SimpleNamespace(model=model, diet=diet))
        living = [g for g in model.loc[types < 2, "Group"] if g in tl.index]
        if living:
            stats.trophic_levels = float(tl[living].max())
        return stats


class ModelRegistry:
    """Registry of loaded EwE models (RpathParams), for comparison.

    Attributes:
        models: model name -> RpathParams
        metadata: model name -> ModelMetadata
        statistics: model name -> ModelStatistics
        directory: directory last loaded
    """

    def __init__(self, directory: Optional[PathLike] = None):
        self.models: Dict[str, RpathParams] = {}
        self.metadata: Dict[str, ModelMetadata] = {}
        self.statistics: Dict[str, ModelStatistics] = {}
        self.directory = Path(directory) if directory else None

    def load_model(
        self, path: PathLike, validate: bool = True, name: Optional[str] = None
    ) -> str:
        """Load one EwE model database into the registry.

        Args:
            path: Path to an EwE Access database
            validate: Run ``validate_ecopath_scenario`` on the model table
            name: Model name in the registry (default: file stem)

        Returns:
            The model's name in the registry

        Raises:
            FileNotFoundError: If the file does not exist
            ValueError: If the file cannot be read as an EwE model, or another
                file is already loaded under the name
        """
        from pypath.io.ewemdb import read_ewemdb

        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"Model file not found: {path}")
        model_name = name or path.stem
        taken = self.metadata.get(model_name)
        if taken is not None and taken.file_path.resolve() != path.resolve():
            raise ValueError(
                f"A model named {model_name!r} is already loaded from "
                f"{taken.file_path}; pass another name"
            )
        start = time.time()
        try:
            params = read_ewemdb(str(path))
        except Exception as e:  # any reader failure: report the file
            raise ValueError(f"Cannot load model {path}: {e}") from e
        meta = ModelMetadata(
            name=model_name,
            file_path=path,
            load_time_ms=(time.time() - start) * 1000,
        )
        if validate:
            groups = list(params.model["Group"])
            meta.validation_report = validate_ecopath_scenario(params.model, groups)
        self.models[model_name] = params
        self.metadata[model_name] = meta
        self.statistics[model_name] = ModelStatistics.calculate(params)
        stats = self.statistics[model_name]
        logger.info(
            "Loaded %s: %d groups, %d links", model_name, stats.n_groups, stats.n_links
        )
        return model_name

    def load_directory(
        self,
        directory: PathLike,
        pattern: Union[str, Sequence[str]] = DEFAULT_PATTERNS,
        validate: bool = True,
        recursive: bool = False,
    ) -> Dict[str, str]:
        """Load every EwE model database in a directory.

        Args:
            directory: Directory to search
            pattern: Glob pattern(s) (default: ``*.ewemdb``, ``*.eweaccdb``)
            validate: Validate each model
            recursive: Search subdirectories

        Returns:
            Path relative to ``directory`` -> model name, for the files
            loaded. A model is named after its file stem, or after its
            relative path when another file has the same stem. A file that
            cannot be read is logged as an error and skipped.
        """
        directory = Path(directory)
        if not directory.exists():
            raise FileNotFoundError(f"Directory not found: {directory}")
        self.directory = directory
        patterns = [pattern] if isinstance(pattern, str) else list(pattern)
        glob = directory.rglob if recursive else directory.glob
        files = sorted({p for pat in patterns for p in glob(pat) if p.is_file()})
        stems = Counter(p.stem for p in files)
        loaded = {}
        for path in files:
            relative = path.relative_to(directory).as_posix()
            taken = self.metadata.get(path.stem)
            unique = stems[path.stem] == 1 and (
                taken is None or taken.file_path.resolve() == path.resolve()
            )
            try:
                loaded[relative] = self.load_model(
                    path, name=path.stem if unique else relative, validate=validate
                )
            except (ValueError, FileNotFoundError) as e:
                logger.error("Skipping %s: %s", relative, e)
        logger.info("Loaded %d models from %s", len(loaded), directory)
        return loaded

    def get_model(self, name: str) -> Optional[RpathParams]:
        """The model's RpathParams, or None."""
        return self.models.get(name)

    def list_models(self, sort_by: str = "name") -> List[Tuple[str, int, int]]:
        """(name, n_groups, n_links) for each model.

        Args:
            sort_by: 'name', 'groups' or 'links'
        """
        rows = [
            (name, self.statistics[name].n_groups, self.statistics[name].n_links)
            for name in sorted(self.models)
        ]
        if sort_by == "groups":
            rows.sort(key=lambda x: x[1], reverse=True)
        elif sort_by == "links":
            rows.sort(key=lambda x: x[2], reverse=True)
        return rows

    def filter_models(
        self,
        min_groups: int = 0,
        max_groups: int = 10000,
        min_links: int = 0,
        max_links: int = 1000000,
        model_type: Optional[str] = None,
    ) -> Dict[str, RpathParams]:
        """Models within the given group and link counts (and type)."""
        out = {}
        for name, params in self.models.items():
            stats, meta = self.statistics[name], self.metadata[name]
            if not min_groups <= stats.n_groups <= max_groups:
                continue
            if not min_links <= stats.n_links <= max_links:
                continue
            if model_type and meta.model_type != model_type:
                continue
            out[name] = params
        return out

    def compare_models(
        self,
        models: Optional[List[str]] = None,
        metrics: Optional[List[str]] = None,
    ) -> pd.DataFrame:
        """Models (rows) by metrics (columns).

        Metrics: 'groups', 'links', 'connectivity', 'trophic_levels',
        'avg_diet_items', 'diet_specialization', 'detritus_linkage'.
        """
        metric_map = {
            "groups": "n_groups",
            "links": "n_links",
            "connectivity": "connectivity",
            "trophic_levels": "trophic_levels",
            "avg_diet_items": "avg_diet_items",
            "diet_specialization": "diet_specialization",
            "detritus_linkage": "detritus_linkage",
        }
        models = list(self.models) if models is None else models
        metrics = list(metric_map) if metrics is None else metrics
        rows = []
        for name in models:
            if name not in self.statistics:
                logger.warning("Model %s not found", name)
                continue
            stats = self.statistics[name]
            row = {"Model": name}
            for metric in metrics:
                if metric in metric_map:
                    row[metric] = getattr(stats, metric_map[metric])
            rows.append(row)
        if rows:
            return pd.DataFrame(rows).set_index("Model")
        return pd.DataFrame(columns=metrics)

    def export_summary(self, output_path: PathLike, format: str = "csv") -> None:
        """Export the comparison: 'csv', 'json' (metadata and statistics) or 'md'."""
        output_path = Path(output_path)
        table = self.compare_models()
        if format == "csv":
            table.to_csv(output_path)
        elif format == "json":
            summary = {
                name: {
                    **self.metadata[name].to_dict(),
                    **vars(self.statistics[name]),
                }
                for name in sorted(self.models)
            }
            output_path.write_text(json.dumps(summary, indent=2, default=str))
        elif format == "md":
            lines = [
                "# Model Registry Summary",
                "",
                f"Generated: {datetime.now():%Y-%m-%d %H:%M:%S}",
                "",
                f"Total Models: {len(self.models)}",
                "",
            ]
            if not self.models:
                lines.append("No models loaded.")
            else:
                cols = list(table.columns)
                lines += [
                    "| Model | " + " | ".join(cols) + " |",
                    "|---" * (len(cols) + 1) + "|",
                ]
                for name, row in table.iterrows():
                    cells = [f"{row[c]:.3g}" for c in cols]
                    lines.append(f"| {name} | " + " | ".join(cells) + " |")
            output_path.write_text("\n".join(lines) + "\n")
        else:
            raise ValueError(f"Unknown format: {format}")
        logger.info("Exported summary to %s", output_path)


def load_model_directory(
    directory: PathLike,
    pattern: Union[str, Sequence[str]] = DEFAULT_PATTERNS,
    validate: bool = True,
) -> ModelRegistry:
    """Load every EwE model database in a directory into a new registry."""
    registry = ModelRegistry(directory)
    registry.load_directory(directory, pattern=pattern, validate=validate)
    return registry


def quick_load(model_path: PathLike) -> RpathParams:
    """Read one EwE model database (``read_ewemdb``) without a registry."""
    from pypath.io.ewemdb import read_ewemdb

    return read_ewemdb(str(model_path))
