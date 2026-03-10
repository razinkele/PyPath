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
    backend: str = "auto",
    scenario_id: int = 1,
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
    ecospace : EcospaceParams, optional
        Ecospace spatial parameters.
    backend : str
        "auto" (detect ODBC), "access" (require ODBC), or "csv" (CSV bundle).
    scenario_id : int
        ScenarioID for all tables (default 1).
    """
    if params.model is None or len(params.model) == 0:
        raise ValueError("Cannot export empty model (params.model is empty)")

    path = str(Path(path).resolve())

    if backend == "auto":
        backend = "access" if _odbc_available() else "csv"
        logger.info("Auto-detected backend: %s", backend)

    if backend == "access":
        from pypath.io._access_writer import AccessWriter

        writer = AccessWriter(params, path, scenario_id=scenario_id)
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
