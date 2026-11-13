"""Read Official EwE model files.

EwE stores a model in an Access database (.ewemdb, .eweaccdb, .accdb, .mdb)
with tables such as EcopathModel, EcopathGroup, EcopathDietComp, EcopathFleet
and EcopathCatch. ``read_ewe_access`` returns those tables as they are, read
through ``ewemdb.read_ewemdb_table`` (mdb-tools on Linux/macOS, or an ODBC
driver): the same access ``read_ewemdb`` uses to build PyPath parameters, so
it works wherever that does. For a model to run, use ``read_ewemdb``.

EwE's ``.vc.sqlite`` files are not models: they hold its Value Chain
plug-in's economic data (producer, processing, distribution, wholesaler,
retailer and consumer units, trade links and landings). ``read_ewe_sqlite``
refuses them, and any SQLite file without EwE model tables, with a message
that says so.

Example:
    >>> from pypath.io import ewe_official
    >>> model = ewe_official.read_ewe_model("Data/BlackSea_EwEF_testcase.eweaccdb")
    >>> model.groups["GroupName"].head()
"""

from __future__ import annotations

import logging
import sqlite3
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Optional, Union

import pandas as pd

logger = logging.getLogger(__name__)

PathLike = Union[str, Path]

ACCESS_SUFFIXES = (".ewemdb", ".eweaccdb", ".accdb", ".mdb")

# EwE's Value Chain plug-in tables, as in EwE's sample "Anchovy Bay Spatial.vc.sqlite"
VALUE_CHAIN_TABLES = {
    "Unit",
    "Link",
    "LinkLandings",
    "ProducerUnit",
    "ProcessingUnit",
    "DistributionUnit",
    "WholesalerUnit",
    "RetailerUnit",
    "ConsumerUnit",
    "EconomicUnit",
}


@dataclass
class EweModel:
    """The tables of an EwE model database, as stored.

    Attributes:
        name: EcopathModel.Name
        author: EcopathModel.Author
        groups: EcopathGroup, in model order (Sequence)
        diet: EcopathDietComp (one row per predator-prey entry)
        fleet: EcopathFleet
        landings: EcopathCatch (catch by fleet and group)
        model_meta: the EcopathModel row
        source_file: path to the database
        source_format: its suffix (.ewemdb, .eweaccdb, ...)
    """

    name: Optional[str] = None
    author: Optional[str] = None
    groups: Optional[pd.DataFrame] = None
    diet: Optional[pd.DataFrame] = None
    fleet: Optional[pd.DataFrame] = None
    landings: Optional[pd.DataFrame] = None
    model_meta: Dict[str, Any] = field(default_factory=dict)
    source_file: Optional[Path] = None
    source_format: Optional[str] = None

    def __repr__(self) -> str:
        n_groups = len(self.groups) if self.groups is not None else 0
        return (
            f"EweModel(name='{self.name}', n_groups={n_groups}, "
            f"source='{self.source_format}')"
        )


def read_ewe_model(path: PathLike, **kwargs) -> EweModel:
    """Read an EwE model database.

    Args:
        path: .ewemdb / .eweaccdb / .accdb / .mdb (Access). A .sqlite file
            is checked and refused: see ``read_ewe_sqlite``.

    Returns:
        EweModel with the database's tables

    Raises:
        FileNotFoundError: If the file does not exist
        ValueError: If the format is not an EwE model database
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"EwE model file not found: {path}")
    suffix = path.suffix.lower()
    if suffix == ".sqlite":
        return read_ewe_sqlite(path, **kwargs)
    if suffix in ACCESS_SUFFIXES:
        return read_ewe_access(path, **kwargs)
    raise ValueError(f"Unsupported EwE format: {suffix}")


def read_ewe_access(path: PathLike) -> EweModel:
    """Read an EwE Access model database's tables.

    Reads EcopathModel, EcopathGroup (in Sequence order, as EwE loads it),
    EcopathDietComp, EcopathFleet and EcopathCatch through
    ``ewemdb.read_ewemdb_table``. A missing or unreadable core table is a
    warning and leaves that attribute None.

    Args:
        path: Path to the database

    Returns:
        EweModel
    """
    from pypath.io.ewemdb import list_ewemdb_tables, read_ewemdb_table

    path = Path(path)
    tables = set(list_ewemdb_tables(str(path)))
    model = EweModel(source_file=path, source_format=path.suffix.lower())

    def table(name: str) -> Optional[pd.DataFrame]:
        if name not in tables:
            logger.warning("%s has no %s table", path.name, name)
            return None
        try:
            return read_ewemdb_table(str(path), name)
        except Exception as e:  # driver or export failure: report, keep the rest
            logger.warning("Could not read %s from %s: %s", name, path.name, e)
            return None

    meta = table("EcopathModel")
    if meta is not None and len(meta) > 0:
        row = meta.iloc[0].to_dict()
        model.model_meta = row
        model.name = row.get("Name")
        model.author = row.get("Author")

    groups = table("EcopathGroup")
    if groups is not None:
        order = "Sequence" if "Sequence" in groups.columns else "GroupID"
        if order in groups.columns:
            groups = groups.sort_values(order, kind="stable").reset_index(drop=True)
        model.groups = groups
    model.diet = table("EcopathDietComp")
    model.fleet = table("EcopathFleet")
    model.landings = table("EcopathCatch")

    logger.info(
        "Read EwE model %s from %s: %d groups",
        model.name,
        path.name,
        0 if model.groups is None else len(model.groups),
    )
    return model


def read_ewe_sqlite(path: PathLike, **kwargs) -> EweModel:
    """Refuse a SQLite file, explaining what it holds.

    EwE's ``.vc.sqlite`` files are Value Chain (economic) databases, not
    Ecopath models; PyPath has no real EwE model in SQLite to read. Use the
    model's Access database with ``read_ewe_access`` or ``read_ewemdb``.

    Raises:
        FileNotFoundError: If the file does not exist
        ValueError: Always, naming what the file looks like
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"SQLite file not found: {path}")
    con = sqlite3.connect(f"file:{path}?mode=ro", uri=True)
    try:
        names = {
            row[0]
            for row in con.execute("SELECT name FROM sqlite_master WHERE type='table'")
        }
    finally:
        con.close()
    if names & VALUE_CHAIN_TABLES:
        raise ValueError(
            f"{path.name} is an EwE Value Chain database (economic units, "
            f"trade links and landings: {sorted(names & VALUE_CHAIN_TABLES)}), "
            "not an Ecopath model. Read the model from its Access database "
            "(.ewemdb / .eweaccdb) with read_ewe_access or read_ewemdb."
        )
    raise ValueError(
        f"{path.name} has no EwE model tables ({sorted(names)}). Read the model "
        "from its Access database (.ewemdb / .eweaccdb) with read_ewe_access "
        "or read_ewemdb."
    )


def inspect_ewe_sqlite(path: PathLike) -> Dict[str, Any]:
    """List a SQLite database's tables, row counts and columns.

    Args:
        path: Path to .sqlite file

    Returns:
        Dictionary with ``file`` and ``tables`` (name, rows, columns)
    """
    path = Path(path)
    structure: Dict[str, Any] = {"file": str(path), "tables": []}
    con = sqlite3.connect(f"file:{path}?mode=ro", uri=True)
    try:
        tables = [
            row[0]
            for row in con.execute(
                "SELECT name FROM sqlite_master WHERE type='table' ORDER BY name"
            )
        ]
        for name in tables:
            count = con.execute(f"SELECT COUNT(*) FROM [{name}]").fetchone()[0]
            columns = [
                {"name": col[1], "type": col[2]}
                for col in con.execute(f"PRAGMA table_info([{name}])")
            ]
            structure["tables"].append(
                {"name": name, "rows": count, "columns": columns}
            )
    finally:
        con.close()
    return structure


def compare_models(model1: EweModel, model2: EweModel) -> Dict[str, Any]:
    """Compare two EwE models' names, group counts and diet entries.

    Args:
        model1: First model
        model2: Second model

    Returns:
        Dictionary with ``differences`` and ``status``
    """
    comparison: Dict[str, Any] = {
        "model1": str(model1.source_file),
        "model2": str(model2.source_file),
        "differences": [],
    }
    if model1.name != model2.name:
        comparison["differences"].append(f"Name: '{model1.name}' vs '{model2.name}'")
    n1 = len(model1.groups) if model1.groups is not None else 0
    n2 = len(model2.groups) if model2.groups is not None else 0
    if n1 != n2:
        comparison["differences"].append(f"Group count: {n1} vs {n2}")
    d1 = len(model1.diet) if model1.diet is not None else 0
    d2 = len(model2.diet) if model2.diet is not None else 0
    if d1 != d2:
        comparison["differences"].append(f"Diet links: {d1} vs {d2}")
    comparison["status"] = (
        "Differences found" if comparison["differences"] else "Models appear identical"
    )
    return comparison
