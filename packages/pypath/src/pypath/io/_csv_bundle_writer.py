"""CSV bundle writer for EwE export (.ewecsv.zip).

Creates a zip archive containing one CSV per EwE table plus a manifest.json.
This is the cross-platform fallback writer (no Access/pyodbc dependency).
"""

import io
import json
import logging
import os
import tempfile
import zipfile
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

from pypath.io._ewe_schema import EWE_TABLES, RPATH_TO_EWE_COLUMNS, TYPE_TO_PP

logger = logging.getLogger(__name__)


class CsvBundleWriter:
    """Write RpathParams to an .ewecsv.zip bundle.

    Parameters
    ----------
    params : RpathParams
        The model parameters to export.
    path : str
        Output file path (should end with .ewecsv.zip).
    scenario_id : int, optional
        Scenario ID for Ecosim/Ecospace tables (default 1).
    """

    def __init__(self, params, path: str, scenario_id: int = 1):
        self._params = params
        self._path = os.path.abspath(path)
        self._scenario_id = scenario_id
        self._tables: Dict[str, pd.DataFrame] = {}
        # Create temp file in same directory for atomic rename
        out_dir = os.path.dirname(self._path)
        fd, self._tmp_path = tempfile.mkstemp(
            suffix=".tmp", dir=out_dir if out_dir else "."
        )
        os.close(fd)

    def write_ecopath(self) -> None:
        """Convert RpathParams to EwE Ecopath table DataFrames."""
        model = self._params.model
        diet = self._params.diet

        # Separate biological groups from fleets
        bio_mask = model["Type"] != 3
        fleet_mask = model["Type"] == 3
        bio_groups = model[bio_mask].reset_index(drop=True)
        fleet_groups = model[fleet_mask].reset_index(drop=True)

        n_bio = len(bio_groups)
        n_fleet = len(fleet_groups)
        n_living = int((bio_groups["Type"].isin([0, 1])).sum())
        n_detritus = int((bio_groups["Type"] == 2).sum())

        # --- EcopathGroup ---
        group_rows: List[Dict[str, Any]] = []
        for i, (_, row) in enumerate(bio_groups.iterrows()):
            rpath_type = int(row["Type"]) if not pd.isna(row["Type"]) else 0
            group_rows.append(
                {
                    "GroupID": i + 1,
                    "ModelID": self._scenario_id,
                    "GroupName": row["Group"],
                    "Sequence": i + 1,
                    "Type": rpath_type,
                    "PP": TYPE_TO_PP.get(rpath_type, 0),
                    "Area": 1.0,
                    "Biomass": _nan_to_none(row.get("Biomass")),
                    "BiomassAreaRate": _nan_to_none(row.get("Biomass")),
                    "BiomassHabitat": 1.0,
                    "PB": _nan_to_none(row.get("PB")),
                    "QB": _nan_to_none(row.get("QB")),
                    "EE": _nan_to_none(row.get("EE")),
                    "GE": _nan_to_none(row.get("ProdCons")),
                    "GS": _nan_to_none(row.get("Unassim")),
                    "BA": _nan_to_none(row.get("BioAcc")),
                    "BaBi": None,
                    "Emig": None,
                    "EmigRate": None,
                    "Immig": None,
                    "ImmigEmig": _nan_to_none(row.get("DetInput")),
                    "DetInput": None,
                    "NonMarketValue": None,
                    "pprod": None,
                    "VBK": None,
                }
            )
        self._tables["EcopathGroup"] = pd.DataFrame(group_rows)

        # --- EcopathFleet ---
        fleet_rows: List[Dict[str, Any]] = []
        for i, (_, row) in enumerate(fleet_groups.iterrows()):
            fleet_rows.append(
                {
                    "FleetID": i + 1,
                    "ModelID": self._scenario_id,
                    "FleetName": row["Group"],
                    "Sequence": i + 1,
                    "FixedCost": None,
                    "SailingCost": None,
                    "ProfitMargin": None,
                }
            )
        self._tables["EcopathFleet"] = pd.DataFrame(fleet_rows)

        # --- EcopathDietComp ---
        # Build group name -> 1-based bio ID mapping
        bio_names = bio_groups["Group"].tolist()
        name_to_id = {name: idx + 1 for idx, name in enumerate(bio_names)}

        diet_rows: List[Dict[str, Any]] = []
        prey_names = diet["Group"].tolist()

        # Predator columns are all columns except "Group"
        pred_cols = [c for c in diet.columns if c != "Group"]

        for pred_col in pred_cols:
            if pred_col not in name_to_id:
                continue  # skip non-bio predators
            pred_id = name_to_id[pred_col]
            for prey_idx, prey_name in enumerate(prey_names):
                val = diet.iloc[prey_idx][pred_col]
                if pd.notna(val) and val != 0:
                    if prey_name == "Import":
                        prey_id = n_bio + 1
                    elif prey_name in name_to_id:
                        prey_id = name_to_id[prey_name]
                    else:
                        continue
                    diet_rows.append(
                        {
                            "ModelID": self._scenario_id,
                            "PredID": pred_id,
                            "PreyID": prey_id,
                            "Diet": float(val),
                        }
                    )
        self._tables["EcopathDietComp"] = pd.DataFrame(diet_rows)

        # --- EcopathCatch ---
        catch_rows: List[Dict[str, Any]] = []
        for fi, (_, frow) in enumerate(fleet_groups.iterrows()):
            fleet_name = frow["Group"]
            land_col = f"Landings.{fleet_name}"
            disc_col = f"Discards.{fleet_name}"
            has_land = land_col in model.columns
            has_disc = disc_col in model.columns
            if has_land or has_disc:
                for gi, (_, grow) in enumerate(bio_groups.iterrows()):
                    landing = 0.0
                    discard = 0.0
                    if has_land:
                        v = grow.get(land_col)
                        landing = float(v) if pd.notna(v) else 0.0
                    if has_disc:
                        v = grow.get(disc_col)
                        discard = float(v) if pd.notna(v) else 0.0
                    if landing > 0 or discard > 0:
                        catch_rows.append(
                            {
                                "ModelID": self._scenario_id,
                                "FleetID": fi + 1,
                                "GroupID": gi + 1,
                                "Landing": landing,
                                "Discard": discard,
                                "DiscardMortality": None,
                                "Price": None,
                            }
                        )
        if catch_rows:
            self._tables["EcopathCatch"] = pd.DataFrame(catch_rows)

        # --- Stanza / StanzaLifeStage ---
        stanzas = self._params.stanzas
        if stanzas is not None and hasattr(stanzas, "stgroups"):
            stg = stanzas.stgroups
            if stg is not None and len(stg) > 0:
                stanza_rows = []
                for i, (_, row) in enumerate(stg.iterrows()):
                    stanza_rows.append(
                        {
                            "StanzaID": i + 1,
                            "ModelID": self._scenario_id,
                            "StanzaName": row.get(
                                "StGroupName", row.get("StanzaName", f"Stanza{i+1}")
                            ),
                            "BABsplit": _nan_to_none(row.get("BABsplit")),
                            "WmatWinf": _nan_to_none(row.get("WmatWinf")),
                            "RecPower": _nan_to_none(row.get("RecPower")),
                            "VBK": _nan_to_none(row.get("VBGF_Ksp", row.get("VBK"))),
                        }
                    )
                self._tables["Stanza"] = pd.DataFrame(stanza_rows)

            sti = stanzas.stindiv
            if sti is not None and len(sti) > 0:
                ls_rows = []
                for i, (_, row) in enumerate(sti.iterrows()):
                    group_name = row.get("Group", row.get("GroupName", ""))
                    group_id = name_to_id.get(group_name, 0)
                    ls_rows.append(
                        {
                            "StanzaID": int(row.get("StGroupNum", 1)),
                            "LifeStageID": i + 1,
                            "ModelID": self._scenario_id,
                            "GroupID": group_id,
                            "Months": int(row.get("Last", row.get("Months", 0))),
                            "LeadingLifeStage": bool(
                                row.get("Leading", row.get("LeadingLifeStage", False))
                            ),
                            "LeadingBiomass": bool(
                                row.get(
                                    "LeadingB", row.get("LeadingBiomass", False)
                                )
                            ),
                        }
                    )
                self._tables["StanzaLifeStage"] = pd.DataFrame(ls_rows)

        # --- EcopathModel ---
        self._tables["EcopathModel"] = pd.DataFrame(
            [
                {
                    "ModelID": self._scenario_id,
                    "ModelName": "PyPath Export",
                    "Description": f"Exported by PyPath on "
                    f"{datetime.now(tz=timezone.utc).strftime('%Y-%m-%d')}",
                    "Author": "",
                    "Contact": "",
                    "LastSaved": datetime.now(tz=timezone.utc).isoformat(),
                    "AreaUnit": "km^2",
                    "TimeUnit": "year",
                    "Currency": "t",
                    "NumGroups": n_bio,
                    "NumFleets": n_fleet,
                    "NumLiving": n_living,
                    "NumDetritus": n_detritus,
                    "GroupDigits": 5,
                    "EcopathVersion": 6.6,
                }
            ]
        )

    def write_ecosim(self, scenarios=None) -> None:
        """Store Ecosim tables (stub for future implementation)."""
        logger.debug("write_ecosim: stub, no Ecosim tables written yet")

    def write_ecospace(self, ecospace=None) -> None:
        """Store Ecospace tables (stub for future implementation)."""
        logger.debug("write_ecospace: stub, no Ecospace tables written yet")

    def close(self) -> None:
        """Write all tables to a zip file, then atomically rename to final path."""
        manifest = {
            "ewe_version": "6.6",
            "pypath_export": True,
            "tables": list(self._tables.keys()),
            "created": datetime.now(tz=timezone.utc).isoformat(),
        }

        try:
            with zipfile.ZipFile(
                self._tmp_path, "w", compression=zipfile.ZIP_DEFLATED
            ) as zf:
                # Write manifest
                zf.writestr("manifest.json", json.dumps(manifest, indent=2))

                # Write each table as CSV
                for table_name, df in self._tables.items():
                    buf = io.StringIO()
                    df.to_csv(buf, index=False)
                    zf.writestr(f"{table_name}.csv", buf.getvalue())

            # Atomic replace
            os.replace(self._tmp_path, self._path)
        except Exception:
            # Clean up temp file on error
            if os.path.exists(self._tmp_path):
                os.unlink(self._tmp_path)
            raise


def _nan_to_none(value):
    """Convert NaN/None to None for clean CSV output."""
    if value is None:
        return None
    try:
        if np.isnan(value):
            return None
    except (TypeError, ValueError):
        pass
    return value
