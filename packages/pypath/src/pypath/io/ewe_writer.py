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
