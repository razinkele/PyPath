"""Batch import of EwE model databases into PyPath parameters.

The registry reads each Access database with ``read_ewemdb`` (RpathParams)
and summarises it: groups, diet links (the prey x predator matrix), trophic
levels and detritus links. The checks compare with the databases' own tables
(EcopathGroup, EcopathDietComp), from the tracked test models in ``Data/``.
"""

import json
import os
from pathlib import Path

import numpy as np
import pytest

from pypath.io.model_importer import (
    ModelMetadata,
    ModelRegistry,
    ModelStatistics,
    load_model_directory,
    quick_load,
)

DATA = Path(__file__).resolve().parents[4] / "Data"
BLACK_SEA = DATA / "BlackSea_EwEF_testcase.eweaccdb"
LT2022 = DATA / "LT2022_0.5ST_final7.eweaccdb"


def _available():
    if not (BLACK_SEA.exists() and LT2022.exists()):
        return False
    from pypath.io.ewemdb import check_ewemdb_support

    return any(check_ewemdb_support().values())


needs_access = pytest.mark.skipif(
    not _available(), reason="EwE test databases or driver missing"
)


def _tables(path):
    from pypath.io.ewemdb import read_ewemdb_table

    return (
        read_ewemdb_table(str(path), "EcopathGroup"),
        read_ewemdb_table(str(path), "EcopathDietComp"),
    )


@pytest.fixture
def model_dir(tmp_path):
    for path in (BLACK_SEA, LT2022):
        os.symlink(path, tmp_path / path.name)
    return tmp_path


# --- statistics, against the database's tables ---


@needs_access
@pytest.mark.parametrize("path", [BLACK_SEA, LT2022], ids=["black_sea", "lt2022"])
def test_statistics_count_the_databases_groups_and_links(path):
    groups, diet = _tables(path)
    stats = ModelStatistics.calculate(quick_load(path))
    assert stats.n_groups == len(groups)
    assert stats.n_links == int((diet["Diet"] > 0).sum())
    assert stats.connectivity == pytest.approx(stats.n_links / stats.n_groups**2)


@needs_access
def test_trophic_levels_are_the_longest_food_chain():
    # Independently: TL = 1 for producers and detritus, and for consumers
    # TL_j = 1 + sum_i DC_ij TL_i over the model's prey (imports left out).
    params = quick_load(BLACK_SEA)
    model = params.model
    groups = list(model.loc[model["Type"] < 3, "Group"])
    consumers = set(model.loc[model["Type"] < 1, "Group"])
    diet = params.diet.set_index("Group").reindex(index=groups, columns=groups)
    dc = diet.fillna(0).to_numpy(dtype=float)
    col_sums = dc.sum(axis=0)
    a = np.zeros((len(groups), len(groups)))
    for j, g in enumerate(groups):
        if g in consumers and col_sums[j] > 0:
            a[j] = dc[:, j] / col_sums[j]
    tl = np.linalg.solve(np.eye(len(groups)) - a, np.ones(len(groups)))
    living = [
        i
        for i, g in enumerate(groups)
        if model.loc[model["Group"] == g, "Type"].iloc[0] < 2
    ]
    stats = ModelStatistics.calculate(params)
    assert stats.trophic_levels == pytest.approx(tl[living].max(), rel=1e-2)
    assert stats.trophic_levels > 3  # dolphins and bluefish eat fish that eat plankton


@needs_access
def test_detritus_linkage_is_the_share_of_consumers_eating_detritus():
    groups, diet = _tables(BLACK_SEA)
    detritus = set(groups.loc[groups["Type"] == 2, "GroupID"])
    consumers = set(groups.loc[groups["Type"] < 1, "GroupID"])
    eats = diet[(diet["Diet"] > 0) & diet["PreyID"].isin(detritus)]["PredID"]
    want = len(set(eats) & consumers) / len(consumers)
    stats = ModelStatistics.calculate(quick_load(BLACK_SEA))
    assert stats.detritus_linkage == pytest.approx(want)


# --- the registry ---


@needs_access
def test_a_directory_of_ewe_databases_is_loaded(model_dir):
    registry = load_model_directory(model_dir)
    assert set(registry.models) == {BLACK_SEA.stem, LT2022.stem}
    names = [row[0] for row in registry.list_models()]
    assert names == sorted(names)
    meta = registry.metadata[BLACK_SEA.stem]
    assert meta.validation_report is not None
    assert meta.to_dict()["validation_status"].startswith(("passed", "failed"))


@needs_access
def test_models_can_be_filtered_and_compared(model_dir):
    registry = load_model_directory(model_dir, validate=False)
    small = registry.filter_models(max_groups=30)
    assert BLACK_SEA.stem in small
    table = registry.compare_models(metrics=["groups", "links", "trophic_levels"])
    assert list(table.columns) == ["groups", "links", "trophic_levels"]
    assert (
        table.loc[BLACK_SEA.stem, "groups"]
        == registry.statistics[BLACK_SEA.stem].n_groups
    )


@needs_access
def test_the_summary_exports(model_dir, tmp_path):
    registry = load_model_directory(model_dir, validate=False)
    for fmt in ("csv", "json", "md"):
        out = tmp_path / f"summary.{fmt}"
        registry.export_summary(out, format=fmt)
        assert BLACK_SEA.stem in out.read_text()
    summary = json.loads((tmp_path / "summary.json").read_text())
    assert (
        summary[BLACK_SEA.stem]["n_groups"]
        == registry.statistics[BLACK_SEA.stem].n_groups
    )


def test_a_file_that_is_not_an_ewe_database_is_refused(tmp_path):
    path = tmp_path / "broken.eweaccdb"
    path.write_text("not a database")
    with pytest.raises(ValueError, match="Cannot load model"):
        ModelRegistry().load_model(path)


def test_missing_paths_are_reported(tmp_path):
    with pytest.raises(FileNotFoundError):
        ModelRegistry().load_model(tmp_path / "missing.eweaccdb")
    with pytest.raises(FileNotFoundError):
        ModelRegistry().load_directory(tmp_path / "missing")


def test_metadata_to_dict():
    meta = ModelMetadata(name="M", file_path=Path("m.eweaccdb"), load_time_ms=12.5)
    d = meta.to_dict()
    assert d["name"] == "M" and d["validation_status"] == "Not validated"


def test_an_empty_registry_exports_an_empty_summary(tmp_path):
    out = tmp_path / "summary.md"
    ModelRegistry().export_summary(out, format="md")
    assert "No models loaded" in out.read_text()
    assert np.isfinite(len(ModelRegistry().compare_models().columns))
