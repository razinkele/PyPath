"""EwE database writer for PyPath.

Export PyPath models back to native EwE format.

Functions:
- write_ewemdb(params, path, ...): Write an EwE database file from RpathParams
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from pypath.core.params import RpathParams

logger = logging.getLogger(__name__)


def _odbc_available() -> bool:
    try:
        from pypath.io._access_writer import _find_access_driver

        _find_access_driver()
        return True
    except (ImportError, RuntimeError):
        return False


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

        writer = AccessWriter(
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
            writer._conn = None
        tmp = getattr(writer, "_tmp_path", None)
        if tmp and os.path.exists(tmp):
            try:
                os.unlink(tmp)
            except OSError:
                pass
        raise
