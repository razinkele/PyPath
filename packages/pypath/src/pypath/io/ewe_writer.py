"""EwE database writer for PyPath.

Export PyPath models back to native EwE format.

Functions:
- write_ewemdb(params, path, ...): Write an EwE database file from RpathParams
- export_model_to_ewe(model, path, ...): High-level export for Rpath/RpathParams
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from pypath.core.ecopath import Rpath
    from pypath.core.params import RpathParams
    from pypath.io._access_writer import AccessWriter
    from pypath.io._csv_bundle_writer import CsvBundleWriter

logger = logging.getLogger(__name__)


def _odbc_available() -> bool:
    try:
        from pypath.io._access_writer import _find_access_driver

        _find_access_driver()
        return True
    except (ImportError, RuntimeError):
        return False


def _rpath_to_params(model: "Rpath") -> "RpathParams":
    """Reconstruct a best-effort RpathParams from a balanced Rpath model.

    Notes
    -----
    This reconstruction is inherently lossy because balanced Rpath outputs
    do not retain every original input assumption. Prefer passing original
    RpathParams when available.
    """
    from pypath.core.params import create_rpath_params

    groups = [str(g) for g in model.Group.tolist()]
    types = [int(t) for t in model.type.tolist()]

    params = create_rpath_params(groups=groups, types=types)

    n = len(groups)
    params.model["Biomass"] = model.Biomass[:n]
    params.model["PB"] = model.PB[:n]
    params.model["QB"] = model.QB[:n]
    params.model["EE"] = model.EE[:n]
    params.model["ProdCons"] = model.GE[:n]
    params.model["BioAcc"] = model.BA[:n]
    params.model["Unassim"] = model.Unassim[:n]

    # Best-effort diet transfer for living predators/prey where dimensions align.
    # If dimensions differ, keep default shell matrix from create_rpath_params.
    try:
        prey_rows = [g for g, t in zip(groups, types) if t < 3]
        pred_cols = [g for g, t in zip(groups, types) if t < 2]
        if model.DC.shape == (len(prey_rows), len(pred_cols)):
            for i, prey in enumerate(prey_rows):
                for j, pred in enumerate(pred_cols):
                    params.diet.loc[params.diet["Group"] == prey, pred] = model.DC[i, j]
    except Exception:
        logger.debug("Could not transfer DC matrix during Rpath->RpathParams reconstruction")

    return params


def export_model_to_ewe(
    model: "Rpath | RpathParams",
    path: str,
    *,
    params: "RpathParams | None" = None,
    scenarios: list[Any] | None = None,
    ecospace: Any | None = None,
    mpa_config: Any | None = None,
    timeseries: Any | None = None,
    mediation: Any | None = None,
    taxonomy: Any | None = None,
    value_chain: Any | None = None,
    backend: str = "access",
    scenario_id: int = 1,
    source_db: str | None = None,
) -> None:
    """Export a PyPath model to EwE format.

    Parameters
    ----------
    model : Rpath or RpathParams
        Model to export. If ``Rpath`` is provided and ``params`` is omitted,
        a best-effort reconstruction of ``RpathParams`` is used.
    path : str
        Output file path.
    params : RpathParams, optional
        Original parameter object for faithful export when ``model`` is Rpath.
    backend : str
        "access" (native .eweaccdb), "auto", or "csv".
    """
    from pypath.core.ecopath import Rpath
    from pypath.core.params import RpathParams

    if isinstance(model, RpathParams):
        resolved_params = model
    elif isinstance(model, Rpath):
        if params is not None:
            resolved_params = params
        else:
            logger.warning(
                "Reconstructing RpathParams from balanced Rpath model; "
                "this may be lossy. Pass original params for faithful export."
            )
            resolved_params = _rpath_to_params(model)
    else:
        raise TypeError("model must be an Rpath or RpathParams instance")

    out_path = str(Path(path).resolve())
    if backend == "access" and not out_path.lower().endswith(".eweaccdb"):
        raise ValueError(
            "Native Access export requires an output path ending with '.eweaccdb'."
        )

    if backend == "access" and not _odbc_available():
        raise RuntimeError(
            "Native EwE export requires a Microsoft Access ODBC driver. "
            "Install the Access Database Engine, or use backend='csv'."
        )

    write_ewemdb(
        resolved_params,
        out_path,
        scenarios=scenarios,
        ecospace=ecospace,
        mpa_config=mpa_config,
        timeseries=timeseries,
        mediation=mediation,
        taxonomy=taxonomy,
        value_chain=value_chain,
        backend=backend,
        scenario_id=scenario_id,
        source_db=source_db,
    )


def write_ewemdb(
    params: "RpathParams",
    path: str,
    *,
    scenarios: list[Any] | None = None,
    ecospace: Any | None = None,
    mpa_config: Any | None = None,
    timeseries: Any | None = None,
    mediation: Any | None = None,
    taxonomy: Any | None = None,
    value_chain: Any | None = None,
    backend: str = "auto",
    scenario_id: int = 1,
    source_db: str | None = None,
) -> None:
    """Write an EwE database file from PyPath model parameters.

    Parameters
    ----------
    params : RpathParams
        Ecopath model parameters.
    path : str
        Output file path.
    scenarios : list of RsimScenario, optional
        Ecosim scenarios to include.
    ecospace : EcospaceParams or EcospaceReadResult, optional
        Ecospace spatial parameters.
    mpa_config : MPAConfig, optional
        MPA zone configuration to include.
    timeseries : EweTimeSeriesCollection, optional
        Time series data to include.
    mediation : MediationCollection, optional
        Mediation shapes and link assignments to include.
    taxonomy : TaxonomyData, optional
        Taxonomy species records and group assignments to include.
    value_chain : ValueChainData, optional
        Value chain economics data (21 c-prefix tables) to include.
    backend : str
        "auto" (detect ODBC), "access" (require ODBC), or "csv" (CSV bundle).
    scenario_id : int
        ScenarioID for all tables (default 1).
    source_db : str, optional
        Path to an existing EwE database to use as template. The file is
        copied and its data tables are cleared then re-populated. This
        preserves all EwE system tables so the output is recognized as a
        native EwE database. **Recommended for Access backend.**
    """
    if params.model is None or len(params.model) == 0:
        raise ValueError("Cannot export empty model (params.model is empty)")

    path = str(Path(path).resolve())

    if backend == "auto":
        backend = "access" if _odbc_available() else "csv"
        logger.info("Auto-detected backend: %s", backend)

    if backend == "access":
        from pypath.io._access_writer import AccessWriter

        writer: AccessWriter | CsvBundleWriter = AccessWriter(
            params, path, scenario_id=scenario_id, source_db=source_db
        )
    elif backend == "csv":
        from pypath.io._csv_bundle_writer import CsvBundleWriter

        writer = CsvBundleWriter(params, path, scenario_id=scenario_id)
    else:
        raise ValueError(
            f"Unknown backend: {backend!r}. Use 'auto', 'access', or 'csv'."
        )

    try:
        writer.write_ecopath()
        writer.write_ecosim(scenarios)
        writer.write_ecospace(ecospace)
        writer.write_mpa(mpa_config)
        writer.write_timeseries(timeseries)
        writer.write_mediation(mediation)
        writer.write_taxonomy(taxonomy)
        writer.write_value_chain(value_chain)
        writer.close()
    except Exception:
        import os

        # Close connection first (Access writer holds a file lock)
        conn = getattr(writer, "_conn", None)
        if conn is not None:
            try:
                conn.close()
            except Exception:
                pass
            setattr(writer, "_conn", None)
        tmp = getattr(writer, "_tmp_path", None)
        if tmp and os.path.exists(tmp):
            try:
                os.unlink(tmp)
            except OSError:
                pass
        raise


def write_ewe_samples(
    sample_container: "SampleContainer",
    db_path: str | Path,
    model_id: int = 1,
    sample_id: int | None = None,
    truncate: bool = False,
) -> int:
    """Export sampling data to an EwE database.

    Parameters
    ----------
    sample_container : SampleContainer
        Sampling container with group, diet, and catch samples.
    db_path : str | Path
        Path to .eweaccdb database file.
    model_id : int
        Model ID in database (default 1). Used only for metadata.
    sample_id : int, optional
        Sample ID to assign. If None, generates a new one (max existing + 1).
    truncate : bool
        If True, delete all existing samples before writing (default False).

    Returns
    -------
    int
        The sample_id assigned/used.

    Raises
    ------
    FileNotFoundError
        If database file not found.
    RuntimeError
        If pyodbc or Access driver is unavailable.
    ValueError
        If sample_container is empty or has inconsistent replicate counts.

    Notes
    -----
    Writes four tables:
    - EcopathSample: Metadata (SampleID, Hash, Source, Generated, Rating, SS)
    - EcopathGroupSample: Per-group replicates (Biomass, ProdBiom, ConsBiom, etc.)
    - EcopathDietCompSample: Per-diet-link replicates (Diet fraction)
    - EcopathGroupCatchSample: Per-catch replicates (Landing, Discards)

    The function opens a pyodbc connection to the existing database,
    reads group/fleet ID mappings, and INSERTs sample rows into the tables.
    """
    import hashlib
    from datetime import datetime

    try:
        import pyodbc
    except ImportError:
        raise RuntimeError(
            "write_ewe_samples requires pyodbc. "
            "Install pyodbc or use a different export backend."
        )

    # Validate container
    if not sample_container.group_samples and not sample_container.diet_samples:
        raise ValueError("SampleContainer must have at least group or diet samples")

    n_replicates = sample_container.n_replicates
    if n_replicates <= 0:
        raise ValueError("SampleContainer must have n_replicates > 0")

    db_path = str(Path(db_path).resolve())
    if not Path(db_path).exists():
        raise FileNotFoundError(f"Database file not found: {db_path}")

    # Find Access driver
    try:
        from pypath.io._access_writer import _find_access_driver

        driver = _find_access_driver()
    except RuntimeError as e:
        raise RuntimeError(f"Cannot access EwE database: {e}") from e

    # Connect to database
    conn_str = f"DRIVER={{{driver}}};DBQ={db_path};"
    try:
        conn = pyodbc.connect(conn_str)
        conn.autocommit = True
        cursor = conn.cursor()
    except pyodbc.Error as e:
        raise RuntimeError(f"Failed to connect to database: {e}") from e

    try:
        # Read group and fleet ID mappings
        group_name_to_id = {}
        fleet_name_to_id = {}

        try:
            cursor.execute("SELECT EcopathGroupID, GroupName FROM EcopathGroup")
            for group_id, group_name in cursor.fetchall():
                group_name_to_id[group_name] = group_id
        except pyodbc.Error as e:
            logger.warning("Could not read EcopathGroup table: %s", e)

        try:
            cursor.execute("SELECT EcopathFleetID, FleetName FROM EcopathFleet")
            for fleet_id, fleet_name in cursor.fetchall():
                fleet_name_to_id[fleet_name] = fleet_id
        except pyodbc.Error as e:
            logger.warning("Could not read EcopathFleet table: %s", e)

        # Determine sample_id if not provided
        if sample_id is None:
            try:
                cursor.execute("SELECT MAX(SampleID) FROM EcopathSample")
                row = cursor.fetchone()
                sample_id = (row[0] or 0) + 1
            except pyodbc.Error:
                sample_id = 1

        # Truncate existing samples if requested
        if truncate:
            for table in [
                "EcopathGroupCatchSample",
                "EcopathDietCompSample",
                "EcopathGroupSample",
                "EcopathSample",
            ]:
                try:
                    cursor.execute(f"DELETE FROM [{table}]")
                    logger.debug("Cleared table: %s", table)
                except pyodbc.Error as e:
                    logger.warning("Could not clear table %s: %s", table, e)

        # Create EcopathSample metadata record
        hash_val = hashlib.md5(
            f"{sample_id}_{n_replicates}_{len(sample_container.group_samples)}"
            .encode()
        ).hexdigest()
        generated = datetime.now().isoformat()

        sample_row = (sample_id, hash_val, "PyPath", generated, 0.0, 0.0)
        sql_sample = (
            "INSERT INTO [EcopathSample] "
            "(SampleID, Hash, Source, Generated, Rating, SS) "
            "VALUES (?, ?, ?, ?, ?, ?)"
        )
        cursor.execute(sql_sample, sample_row)
        logger.debug("Inserted EcopathSample record (SampleID=%d)", sample_id)

        # Insert group samples
        sql_group = (
            "INSERT INTO [EcopathGroupSample] "
            "(SampleID, GroupID, Biomass, ProdBiom, ConsBiom, "
            "EcoEfficiency, BiomAcc, ImpVar, BiomAccRate) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)"
        )

        total_group_rows = 0
        for (group_name, group_sample) in sample_container.group_samples.items():
            group_id = group_name_to_id.get(group_name)
            if group_id is None:
                logger.warning("Unknown group name: %s", group_name)
                continue

            for i in range(n_replicates):
                row = (
                    sample_id,
                    group_id,
                    float(group_sample.biomass_samples[i])
                    if group_sample.biomass_samples is not None
                    else 0.0,
                    float(group_sample.pb_samples[i])
                    if group_sample.pb_samples is not None
                    else 0.0,
                    float(group_sample.qb_samples[i])
                    if group_sample.qb_samples is not None
                    else 0.0,
                    float(group_sample.ee_samples[i])
                    if group_sample.ee_samples is not None
                    else 0.0,
                    0.0,  # BiomAcc (not in dataclass)
                    0.0,  # ImpVar (not in dataclass)
                    0.0,  # BiomAccRate (not in dataclass)
                )
                cursor.execute(sql_group, row)
                total_group_rows += 1

        logger.debug("Inserted %d EcopathGroupSample rows", total_group_rows)

        # Insert diet samples
        sql_diet = (
            "INSERT INTO [EcopathDietCompSample] "
            "(SampleID, PredID, PreyID, Diet) "
            "VALUES (?, ?, ?, ?)"
        )

        total_diet_rows = 0
        for (pred_name, prey_name), diet_sample in sample_container.diet_samples.items():
            pred_id = group_name_to_id.get(pred_name)
            prey_id = group_name_to_id.get(prey_name)
            if pred_id is None:
                logger.warning("Unknown predator name: %s", pred_name)
                continue
            if prey_id is None:
                logger.warning("Unknown prey name: %s", prey_name)
                continue

            for i in range(n_replicates):
                row = (
                    sample_id,
                    pred_id,
                    prey_id,
                    float(diet_sample.diet_fraction_samples[i]),
                )
                cursor.execute(sql_diet, row)
                total_diet_rows += 1

        logger.debug("Inserted %d EcopathDietCompSample rows", total_diet_rows)

        # Insert catch samples
        sql_catch = (
            "INSERT INTO [EcopathGroupCatchSample] "
            "(SampleID, GroupID, FleetID, Landing, Discards) "
            "VALUES (?, ?, ?, ?, ?)"
        )

        total_catch_rows = 0
        for (group_name, fleet_name), catch_sample in sample_container.catch_samples.items():
            group_id = group_name_to_id.get(group_name)
            fleet_id = fleet_name_to_id.get(fleet_name)
            if group_id is None:
                logger.warning("Unknown group name: %s", group_name)
                continue
            if fleet_id is None:
                logger.warning("Unknown fleet name: %s", fleet_name)
                continue

            for i in range(n_replicates):
                row = (
                    sample_id,
                    group_id,
                    fleet_id,
                    float(catch_sample.catch_samples[i]),
                    0.0,  # Discards (not in dataclass)
                )
                cursor.execute(sql_catch, row)
                total_catch_rows += 1

        logger.debug("Inserted %d EcopathGroupCatchSample rows", total_catch_rows)
        logger.info(
            "Successfully wrote %d sample records to database "
            "(groups=%d, diet=%d, catch=%d)",
            n_replicates,
            total_group_rows,
            total_diet_rows,
            total_catch_rows,
        )

        return sample_id

    finally:
        if conn is not None:
            try:
                conn.close()
            except Exception:
                pass


def write_ewe_samples(
    sample_container: "SampleContainer",
    db_path: str | Path,
    model_id: int = 1,
    sample_id: int | None = None,
    truncate: bool = False,
) -> int:
    """Export sampling data to an EwE database.

    Parameters
    ----------
    sample_container : SampleContainer
        Sampling container with group, diet, and catch samples.
    db_path : str | Path
        Path to .eweaccdb database file.
    model_id : int
        Model ID in database (default 1). Used only for metadata.
    sample_id : int, optional
        Sample ID to assign. If None, generates a new one (max existing + 1).
    truncate : bool
        If True, delete all existing samples before writing (default False).

    Returns
    -------
    int
        The sample_id assigned/used.

    Raises
    ------
    FileNotFoundError
        If database file not found.
    RuntimeError
        If pyodbc or Access driver is unavailable.
    ValueError
        If sample_container is empty or has inconsistent replicate counts.
    """
    import hashlib
    from datetime import datetime
    from pypath.io.ewemdb import read_ewemdb_table, list_ewemdb_tables

    try:
        import pyodbc
    except ImportError:
        raise RuntimeError(
            "write_ewe_samples requires pyodbc. "
            "Install pyodbc or use a different export backend."
        )

    # Validate container
    if not sample_container.group_samples and not sample_container.diet_samples:
        raise ValueError("SampleContainer must have at least group or diet samples")

    n_replicates = sample_container.n_replicates
    if n_replicates <= 0:
        raise ValueError("SampleContainer must have n_replicates > 0")

    db_path = str(Path(db_path).resolve())
    if not Path(db_path).exists():
        raise FileNotFoundError(f"Database file not found: {db_path}")

    # Find Access driver
    try:
        from pypath.io._access_writer import _find_access_driver

        driver = _find_access_driver()
    except RuntimeError as e:
        raise RuntimeError(f"Cannot access EwE database: {e}") from e

    # Connect to database
    conn_str = f"DRIVER={{{driver}}};DBQ={db_path};"
    try:
        conn = pyodbc.connect(conn_str)
        conn.autocommit = True
        cursor = conn.cursor()
    except pyodbc.Error as e:
        raise RuntimeError(f"Failed to connect to database: {e}") from e

    try:
        # Read group and fleet ID mappings using safe read_ewemdb_table
        group_name_to_id = {}
        fleet_name_to_id = {}
        tables = list_ewemdb_tables(db_path)

        if "EcopathGroup" in tables:
            try:
                group_df = read_ewemdb_table(db_path, "EcopathGroup")
                if group_df is not None:
                    for _, row in group_df.iterrows():
                        gid = int(row.get("GroupID", 0))
                        gname = str(row.get("GroupName", ""))
                        if gid > 0 and gname:
                            group_name_to_id[gname] = gid
            except Exception as e:
                logger.warning("Could not read EcopathGroup table: %s", e)

        if "EcopathFleet" in tables:
            try:
                fleet_df = read_ewemdb_table(db_path, "EcopathFleet")
                if fleet_df is not None:
                    for _, row in fleet_df.iterrows():
                        fid = int(row.get("FleetID", 0))
                        fname = str(row.get("FleetName", ""))
                        if fid > 0 and fname:
                            fleet_name_to_id[fname] = fid
            except Exception as e:
                logger.warning("Could not read EcopathFleet table: %s", e)

        # Determine sample_id if not provided
        if sample_id is None:
            if "EcopathSample" in tables:
                try:
                    sample_df = read_ewemdb_table(db_path, "EcopathSample")
                    if sample_df is not None and len(sample_df) > 0:
                        sample_id = int(sample_df["SampleID"].max()) + 1
                    else:
                        sample_id = 1
                except Exception:
                    sample_id = 1
            else:
                sample_id = 1

        # Truncate existing samples if requested
        if truncate:
            for table in [
                "EcopathGroupCatchSample",
                "EcopathDietCompSample",
                "EcopathGroupSample",
                "EcopathSample",
            ]:
                if table in tables:
                    try:
                        cursor.execute(f"DELETE FROM [{table}]")
                        logger.debug("Cleared table: %s", table)
                    except pyodbc.Error as e:
                        logger.warning("Could not clear table %s: %s", table, e)

        # Create EcopathSample metadata record
        hash_val = hashlib.md5(
            f"{sample_id}_{n_replicates}_{len(sample_container.group_samples)}"
            .encode()
        ).hexdigest()
        generated = datetime.now().isoformat()

        sample_row = (sample_id, hash_val, "PyPath", generated, 0.0, 0.0)
        sql_sample = (
            "INSERT INTO [EcopathSample] "
            "(SampleID, Hash, Source, Generated, Rating, SS) "
            "VALUES (?, ?, ?, ?, ?, ?)"
        )
        cursor.execute(sql_sample, sample_row)
        logger.debug("Inserted EcopathSample record (SampleID=%d)", sample_id)

        # Insert group samples
        sql_group = (
            "INSERT INTO [EcopathGroupSample] "
            "(SampleID, GroupID, Biomass, ProdBiom, ConsBiom, "
            "EcoEfficiency, BiomAcc, ImpVar, BiomAccRate) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)"
        )

        total_group_rows = 0
        for (group_name, group_sample) in sample_container.group_samples.items():
            group_id = group_name_to_id.get(group_name)
            if group_id is None:
                logger.warning("Unknown group name: %s", group_name)
                continue

            for i in range(n_replicates):
                row = (
                    sample_id,
                    group_id,
                    float(group_sample.biomass_samples[i])
                    if group_sample.biomass_samples is not None
                    else 0.0,
                    float(group_sample.pb_samples[i])
                    if group_sample.pb_samples is not None
                    else 0.0,
                    float(group_sample.qb_samples[i])
                    if group_sample.qb_samples is not None
                    else 0.0,
                    float(group_sample.ee_samples[i])
                    if group_sample.ee_samples is not None
                    else 0.0,
                    0.0,  # BiomAcc
                    0.0,  # ImpVar
                    0.0,  # BiomAccRate
                )
                cursor.execute(sql_group, row)
                total_group_rows += 1

        logger.debug("Inserted %d EcopathGroupSample rows", total_group_rows)

        # Insert diet samples
        sql_diet = (
            "INSERT INTO [EcopathDietCompSample] "
            "(SampleID, PredID, PreyID, Diet) "
            "VALUES (?, ?, ?, ?)"
        )

        total_diet_rows = 0
        for (pred_name, prey_name), diet_sample in sample_container.diet_samples.items():
            pred_id = group_name_to_id.get(pred_name)
            prey_id = group_name_to_id.get(prey_name)
            if pred_id is None:
                logger.warning("Unknown predator name: %s", pred_name)
                continue
            if prey_id is None:
                logger.warning("Unknown prey name: %s", prey_name)
                continue

            for i in range(n_replicates):
                row = (
                    sample_id,
                    pred_id,
                    prey_id,
                    float(diet_sample.diet_fraction_samples[i]),
                )
                cursor.execute(sql_diet, row)
                total_diet_rows += 1

        logger.debug("Inserted %d EcopathDietCompSample rows", total_diet_rows)

        # Insert catch samples
        sql_catch = (
            "INSERT INTO [EcopathGroupCatchSample] "
            "(SampleID, GroupID, FleetID, Landing, Discards) "
            "VALUES (?, ?, ?, ?, ?)"
        )

        total_catch_rows = 0
        for (group_name, fleet_name), catch_sample in sample_container.catch_samples.items():
            group_id = group_name_to_id.get(group_name)
            fleet_id = fleet_name_to_id.get(fleet_name)
            if group_id is None:
                logger.warning("Unknown group name: %s", group_name)
                continue
            if fleet_id is None:
                logger.warning("Unknown fleet name: %s", fleet_name)
                continue

            for i in range(n_replicates):
                row = (
                    sample_id,
                    group_id,
                    fleet_id,
                    float(catch_sample.catch_samples[i]),
                    0.0,  # Discards
                )
                cursor.execute(sql_catch, row)
                total_catch_rows += 1

        logger.debug("Inserted %d EcopathGroupCatchSample rows", total_catch_rows)
        logger.info(
            "Successfully wrote %d sample records to database "
            "(groups=%d, diet=%d, catch=%d)",
            n_replicates,
            total_group_rows,
            total_diet_rows,
            total_catch_rows,
        )

        return sample_id

    finally:
        if conn is not None:
            try:
                conn.close()
            except Exception:
                pass
