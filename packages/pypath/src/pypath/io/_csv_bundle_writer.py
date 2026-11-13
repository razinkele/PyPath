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
from typing import Any, Dict, List

import numpy as np
import pandas as pd

from pypath.io._ewe_schema import EWE_TABLES

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

        # EwE derives a stanza group's age structure from its life stages'
        # EcopathGroup.vbK (cEcoSimModel.InitStanza): give every stage group
        # its stanza group's K.
        stage_vbk = _stanza_stage_vbk(self._params.stanzas)

        # --- EcopathGroup ---
        group_rows: List[Dict[str, Any]] = []
        for i, (_, row) in enumerate(bio_groups.iterrows()):
            rpath_type = int(row["Type"]) if not pd.isna(row["Type"]) else 0
            group_rows.append(
                {
                    "GroupID": i + 1,
                    "GroupName": row["Group"],
                    "Sequence": i + 1,
                    "Type": rpath_type,
                    "Biomass": _nan_to_none(row.get("Biomass")),
                    "Area": 1.0,
                    "ProdBiom": _nan_to_none(row.get("PB")),
                    "ConsBiom": _nan_to_none(row.get("QB")),
                    "EcoEfficiency": _nan_to_none(row.get("EE")),
                    "ProdCons": _nan_to_none(row.get("ProdCons")),
                    "BiomAcc": _nan_to_none(row.get("BioAcc")),
                    "BiomAccRate": None,
                    "Unassim": _nan_to_none(row.get("Unassim")),
                    "DtImports": _nan_to_none(row.get("DetInput")),
                    "Export": None,
                    "Catch": None,
                    "ImpVar": None,
                    "NonMarketValue": None,
                    "Respiration": None,
                    "PoolColor": None,
                    "Immigration": _nan_to_none(row.get("Immigration")),
                    "Emigration": _nan_to_none(row.get("Emigration")),
                    "EmigRate": _nan_to_none(row.get("EmigRate")),
                    "Production": None,
                    "vbK": stage_vbk.get(str(row["Group"]).strip()),
                    "OtherMort": None,
                }
            )
        self._tables["EcopathGroup"] = pd.DataFrame(group_rows)

        # --- EcopathFleet ---
        fleet_rows: List[Dict[str, Any]] = []
        for i, (_, row) in enumerate(fleet_groups.iterrows()):
            fleet_rows.append(
                {
                    "FleetID": i + 1,
                    "FleetName": row["Group"],
                    "Sequence": i + 1,
                    "FixedCost": None,
                    "VariableCost": None,
                    "SailingCost": None,
                    "PoolColor": None,
                    "NominalEffort": None,
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
                                "FleetID": fi + 1,
                                "GroupID": gi + 1,
                                "Landing": landing,
                                "Discards": discard,
                                "DiscardMortality": None,
                                "Price": None,
                            }
                        )
        if catch_rows:
            self._tables["EcopathCatch"] = pd.DataFrame(catch_rows)

        # --- Stanza / StanzaLifeStage ---
        stanza_rows, ls_rows = _stanza_tables(self._params.stanzas, name_to_id)
        if stanza_rows:
            self._tables["Stanza"] = pd.DataFrame(stanza_rows)
        if ls_rows:
            self._tables["StanzaLifeStage"] = pd.DataFrame(ls_rows)

        # --- EcopathModel ---
        self._tables["EcopathModel"] = pd.DataFrame(
            [
                {
                    "ModelID": self._scenario_id,
                    "Name": "PyPath Export",
                    "Description": f"Exported by PyPath on "
                    f"{datetime.now(tz=timezone.utc).strftime('%Y-%m-%d')}",
                    "Author": "",
                    "Contact": "",
                    "LastSaved": (
                        datetime.now(tz=timezone.utc)
                        - datetime(1899, 12, 30, tzinfo=timezone.utc)
                    ).total_seconds()
                    / 86400.0,
                    "NumDigits": 5,
                    "GroupDigits": 5,
                    "Area": 1.0,
                    "FirstYear": 1,
                    "NumYears": 1,
                    "StepsPerYear": 12,
                    "UnitCurrency": 0,  # 0=default metric (t/km^2)
                    "UnitTime": 0,  # 0=default (year)
                    "UnitMonetary": "",
                    "LastSavedVersion": "6.6",
                    "Country": "",
                    "EcosystemType": "",
                }
            ]
        )

    def write_ecosim(self, scenarios=None) -> None:
        """Convert RsimScenario objects to EwE Ecosim table DataFrames.

        Parameters
        ----------
        scenarios : list of RsimScenario, optional
            Ecosim scenarios to export. If None, no Ecosim tables written.
        """
        if not scenarios:
            return

        scen_rows = []
        group_info_rows = []
        fish_rate_rows = []
        shape_rows = []
        shape_time_rows = []
        forcing_matrix_rows = []
        shape_id_counter = 1

        for si, scen in enumerate(scenarios):
            scen_id = si + 1
            p = scen.params if hasattr(scen, "params") else None

            # Determine num_years from forcing shape
            num_months = 0
            if hasattr(scen, "forcing") and hasattr(scen.forcing, "ForcedBio"):
                num_months = scen.forcing.ForcedBio.shape[0]
            num_years = max(num_months // 12, 1)

            scen_rows.append(
                {
                    "ScenarioID": scen_id,
                    "ScenarioName": getattr(scen, "eco_name", f"Scenario {scen_id}"),
                    "Description": "Exported from PyPath",
                    "Author": "",
                    "Contact": "",
                    "LastSaved": "",
                    "TotalTime": float(num_years),
                    "StepSize": 1.0 / 12.0,
                    "EquilibriumStepSize": 1.0,
                    "SystemRecovery": 0.0,
                    "Discount": 0.0,
                    "ForagingTimeLowerLimit": float(
                        getattr(p, "ForagingTimeLowerLimit", 0.01)
                    )
                    if p is not None
                    else 0.01,
                }
            )

            if p is None:
                continue

            # --- EcosimScenarioGroup: group-level Ecosim settings ---
            n_groups = getattr(p, "NUM_GROUPS", 0) + 1  # +1 for Outside

            def _group_value(attr, gi, default):
                arr = getattr(p, attr, None)
                if arr is None or gi >= len(arr) or not np.isfinite(arr[gi]):
                    return default
                return float(arr[gi])

            # Producers' maximum relative P/B is held as their link's
            # vulnerability VV = PBmaxs / (PBmaxs - 1) (rsim_scenario).
            pb_maxs = {}
            for lk in range(1, len(getattr(p, "PreyFrom", []))):
                if int(p.PreyFrom[lk]) == 0:
                    vv = float(p.VV[lk])
                    pb_maxs[int(p.PreyTo[lk])] = (
                        vv / (vv - 1.0) if 1.0 < vv < 1e9 else 1.0
                    )

            for gi in range(1, n_groups):  # skip Outside (0)
                group_info_rows.append(
                    {
                        "ScenarioID": scen_id,
                        "EcopathGroupID": gi,
                        "GroupID": gi,
                        "Pbmaxs": pb_maxs.get(gi, 2.0),
                        "FtimeMax": _group_value("FtimeMax", gi, 2.0),
                        "FtimeAdjust": _group_value("FtimeAdj", gi, 0.0),
                        "MoPred": _group_value("MoPred", gi, 0.0),
                        "SwitchPower": _group_value("SwitchPower", gi, 0.0),
                    }
                )

            # --- EcosimScenarioForcingMatrix: per-link vulnerabilities ---
            if hasattr(p, "PreyFrom") and hasattr(p, "VV"):
                for link_idx in range(len(p.PreyFrom)):
                    forcing_matrix_rows.append(
                        {
                            "ScenarioID": scen_id,
                            "PredID": int(p.PreyTo[link_idx]),
                            "PreyID": int(p.PreyFrom[link_idx]),
                            "vulnerability": float(p.VV[link_idx]),
                        }
                    )

            # --- Fishing effort shapes (EcosimShapeFishRate) ---
            if hasattr(scen, "fishing") and hasattr(scen.fishing, "FishingEffort"):
                effort = scen.fishing.FishingEffort
                for fi in range(effort.shape[0]):
                    has_non_default = False
                    for t in range(effort.shape[1]):
                        val = float(effort[fi, t])
                        if abs(val - 1.0) > 1e-9:
                            has_non_default = True
                            break
                    if has_non_default:
                        fish_rate_rows.append(
                            {
                                "ShapeID": fi + 1,
                                "zScale": 1.0,
                                "Title": f"FishingEffort_Fleet{fi + 1}",
                            }
                        )

            # --- Environmental forcing (EcosimShape + EcosimShapeTime) ---
            if hasattr(scen, "forcing") and hasattr(scen.forcing, "ForcedBio"):
                forced = scen.forcing.ForcedBio
                for gi in range(forced.shape[0]):
                    # Only export if not all 1.0 (non-trivial forcing)
                    col = forced[gi, :]
                    if np.any(np.abs(col - 1.0) > 1e-9):
                        sid = shape_id_counter
                        shape_id_counter += 1
                        shape_rows.append(
                            {
                                "ShapeID": sid,
                                "ShapeType": 0,
                                "IsSeasonal": False,
                            }
                        )
                        shape_time_rows.append(
                            {
                                "ShapeID": sid,
                                "zScale": 1.0,
                                "Title": f"BioForcing_Group{gi}",
                                "zMaxScale": 1.0,
                                "FunctionType": 0,
                                "ApplicationType": 0,
                                "FunctionParams": "",
                            }
                        )
                if shape_rows:
                    logger.warning(
                        "Forcing time series values cannot be fully "
                        "serialized in EwE 6.6+ format (requires "
                        "EcosimTimeSeries* tables, not yet implemented). "
                        "Shape metadata exported; per-timestep values "
                        "are omitted."
                    )

        self._tables["EcosimScenario"] = pd.DataFrame(scen_rows)
        if group_info_rows:
            self._tables["EcosimScenarioGroup"] = pd.DataFrame(group_info_rows)
        if forcing_matrix_rows:
            self._tables["EcosimScenarioForcingMatrix"] = pd.DataFrame(
                forcing_matrix_rows
            )
        if fish_rate_rows:
            self._tables["EcosimShapeFishRate"] = pd.DataFrame(fish_rate_rows)
        if shape_rows:
            self._tables["EcosimShape"] = pd.DataFrame(shape_rows)
        if shape_time_rows:
            self._tables["EcosimShapeTime"] = pd.DataFrame(shape_time_rows)

        logger.info(
            "write_ecosim: %d scenarios, %d group info rows, %d link rows",
            len(scen_rows),
            len(group_info_rows),
            len(forcing_matrix_rows),
        )

    def write_ecospace(self, ecospace=None) -> None:
        """Convert EcospaceParams to EwE Ecospace table DataFrames.

        Parameters
        ----------
        ecospace : EcospaceParams, optional
            Ecospace spatial parameters to export.
        """
        if ecospace is None:
            return

        # Unwrap EcospaceReadResult -> EcospaceParams for base table logic
        eco = ecospace.ecospace if hasattr(ecospace, "ecospace") else ecospace
        sid = self._scenario_id
        grid = eco.grid if hasattr(eco, "grid") else None

        if grid is not None:
            self._tables["EcospaceScenario"] = pd.DataFrame(
                [
                    {
                        "ScenarioID": sid,
                        "ScenarioName": "PyPath Ecospace",
                        "Description": "",
                        "Inrow": getattr(grid, "n_rows", 0),
                        "Incol": getattr(grid, "n_cols", 0),
                        "CellLength": getattr(grid, "cell_size", 1.0),
                        "CellSize": getattr(grid, "cell_size", 1.0),
                        "MinLat": getattr(grid, "origin_lat", 0.0),
                        "MinLon": getattr(grid, "origin_lon", 0.0),
                    }
                ]
            )

        if hasattr(eco, "dispersal_rate"):
            from pypath.io.ewemdb import _ewe_mvel_from_rate

            # dispersal_rate is EwE's Mrate (read_ecospace): store the Mvel
            # that reads back as it with the CellLength written above (the
            # reader's default 1 without an EcospaceScenario row), not
            # migratory.
            cell_length = (
                float(getattr(grid, "cell_size", 1.0)) if grid is not None else 1.0
            )
            group_rows = []
            for gi in range(len(eco.dispersal_rate)):
                group_rows.append(
                    {
                        "ScenarioID": sid,
                        "GroupID": gi + 1,
                        "EcopathGroupID": gi + 1,
                        "Mvel": _ewe_mvel_from_rate(
                            float(eco.dispersal_rate[gi]), cell_length, False
                        ),
                        "RelMoveBad": 2.0,
                        "RelVulBad": 2.0,
                        "IsAdvected": bool(eco.advection_enabled[gi])
                        if hasattr(eco, "advection_enabled")
                        else False,
                        "IsMigratory": False,
                        "BarrierAvoidanceWeight": 0.0,
                    }
                )
            self._tables["EcospaceScenarioGroup"] = pd.DataFrame(group_rows)

        # Write habitat tables (structured conversion)
        habitat_types = getattr(ecospace, "habitat_types", None)
        if habitat_types:
            hab_rows = []
            for hid_0based, name in habitat_types.items():
                hab_rows.append(
                    {
                        "ScenarioID": sid,
                        "HabitatID": hid_0based + 1,  # 0-based -> 1-based
                        "HabitatName": name,
                        "Sequence": hid_0based + 1,
                        "HabitatMap": None,
                    }
                )
            self._tables["EcospaceScenarioHabitat"] = pd.DataFrame(hab_rows)

        # Write group-habitat preferences (structured conversion)
        if habitat_types and hasattr(eco, "habitat_preference"):
            gh_rows = []
            n_groups = eco.habitat_preference.shape[0]
            for gi in range(n_groups):
                for hid_0based in habitat_types:
                    gh_rows.append(
                        {
                            "ScenarioID": sid,
                            "GroupID": gi + 1,  # 0-based -> 1-based
                            "HabitatID": hid_0based + 1,  # 0-based -> 1-based
                            "Preference": 1.0,
                        }
                    )
            if gh_rows:
                self._tables["EcospaceScenarioGroupHabitat"] = pd.DataFrame(gh_rows)

        # Write DataFrame passthrough tables
        _df_fields = {
            "fleet_info": "EcospaceScenarioFleet",
            "capacity_drivers": "EcospaceScenarioCapacityDrivers",
            "driver_layers": "EcospaceScenarioDriverLayer",
            "migration_maps": "EcospaceScenarioGroupMigration",
            "monthly_maps": "EcospaceScenarioMonth",
            "weight_layers": "EcospaceScenarioWeightLayer",
            "data_connections": "EcospaceScenarioDataConnection",
            "disabled_connections": "EcospaceScenarioDataConnectionDisabled",
            "disabled_drivers": "EcospaceScenarioDriverDisabled",
            "habitat_fishery": "EcospaceScenarioHabitatFishery",
        }
        for attr_name, table_name in _df_fields.items():
            df = getattr(ecospace, attr_name, None)
            if df is not None and len(df) > 0:
                self._tables[table_name] = df

        logger.info("write_ecospace: spatial data written")

    def write_mpa(self, mpa_config=None) -> None:
        """Convert MPAConfig to EwE MPA table DataFrames.

        Parameters
        ----------
        mpa_config : MPAConfig, optional
            MPA zone configuration to export.
        """
        if mpa_config is None:
            return
        zones = getattr(mpa_config, "zones", [])
        if not zones:
            return

        sid = self._scenario_id

        mpa_rows = []
        fishery_rows = []
        for seq, zone in enumerate(zones, start=1):
            mpa_rows.append(
                {
                    "ScenarioID": sid,
                    "MPAID": zone.mpa_id,
                    "Sequence": seq,
                    "MPAname": zone.name,
                    "MPAmonth": zone.start_month,
                }
            )
            if zone.excluded_fleets is not None:
                for fleet_idx in zone.excluded_fleets:
                    fishery_rows.append(
                        {
                            "ScenarioID": sid,
                            "MPAID": zone.mpa_id,
                            "FleetID": fleet_idx + 1,  # 0-based -> 1-based
                            "Excluded": True,
                        }
                    )

        self._tables["EcospaceScenarioMPA"] = pd.DataFrame(mpa_rows)
        if fishery_rows:
            self._tables["EcospaceScenarioMPAFishery"] = pd.DataFrame(fishery_rows)

        logger.info(
            "write_mpa: %d zones, %d fleet exclusions",
            len(mpa_rows),
            len(fishery_rows),
        )

    def write_timeseries(self, timeseries=None) -> None:
        """Write time series tables to the CSV bundle."""
        if timeseries is None or not timeseries.series:
            return

        meta_rows = []
        for s in timeseries.series:
            meta_rows.append(
                {
                    "TimeSeriesID": s.series_id,
                    "ScenarioID": self._scenario_id,
                    "Name": s.name,
                    "DatType": s.dat_type,
                    "GroupID": (s.group_idx + 1) if s.group_idx is not None else 0,
                    "FleetID": (s.fleet_idx + 1) if s.fleet_idx is not None else 0,
                    "DatasetID": s.dataset_id,
                    # The series' weight in EwE's sum of squares (Single)
                    "WtType": float(s.weight),
                    "PoolColor": 0,
                }
            )
        self._tables["EcosimTimeSeries"] = pd.DataFrame(meta_rows)

        val_rows = []
        for s in timeseries.series:
            for t, v in enumerate(s.values):
                if not np.isnan(v):
                    val_rows.append(
                        {
                            "TimeSeriesID": s.series_id,
                            "ScenarioID": self._scenario_id,
                            "TimeStep": t + 1,
                            "Value": v,
                        }
                    )
        if val_rows:
            self._tables["EcosimTimeSeriesValues"] = pd.DataFrame(val_rows)

        dataset_ids = {s.dataset_id for s in timeseries.series}
        ds_rows = [
            {
                "DatasetID": did,
                "ScenarioID": self._scenario_id,
                "DatasourceName": f"Dataset_{did}",
                "Enabled": True,
                # EwE eTSDataSetInterval: 0 annual, 1 time step
                "DataInterval": int(
                    getattr(timeseries, "interval", "annual") == "timestep"
                ),
            }
            for did in sorted(dataset_ids)
        ]
        self._tables["EcosimTimeSeriesDataset"] = pd.DataFrame(ds_rows)

        # Write empty season table so EwE 6 finds the expected table
        self._tables["EcosimTimeSeriesSeason"] = pd.DataFrame(
            columns=["TimeSeriesID", "ScenarioID", "Season", "Value"]
        )

    def write_mediation(self, collection) -> None:
        """Write mediation shapes and link assignments.

        Parameters
        ----------
        collection : MediationCollection
            The mediation collection to write.
        """
        if collection is None:
            return

        # Write shapes table
        shape_rows = []
        for shape in collection.shapes:
            row = {
                "ShapeID": shape.shape_id,
                "Title": shape.name,
                "nPoints": len(shape.y_points),
            }
            # Pad/truncate to 9 YY columns
            for i in range(9):
                if i < len(shape.y_points):
                    row[f"YY{i + 1}"] = float(shape.y_points[i])
                else:
                    row[f"YY{i + 1}"] = 1.0
            shape_rows.append(row)
        self._tables["EcosimShapeMediation"] = pd.DataFrame(shape_rows)

        # Write group links
        group_rows = []
        for link in collection.group_links:
            group_rows.append(
                {
                    "ScenarioID": 1,
                    "ShapeID": link.shape_id,
                    "GroupID": link.mediator_idx + 1,  # 0-based to 1-based
                    "PredID": link.pred_idx + 1,
                    "PreyID": link.prey_idx + 1,
                    "AppliedWeight": link.weight,
                }
            )
        self._tables["EcosimScenarioshapeMedWeightsGroup"] = pd.DataFrame(group_rows)

        # Write fleet links
        fleet_rows = []
        for link in collection.fleet_links:
            fleet_rows.append(
                {
                    "ScenarioID": 1,
                    "ShapeID": link.shape_id,
                    "GroupID": link.mediator_idx + 1,
                    "FleetID": link.fleet_idx + 1,
                    "AppliedWeight": link.weight,
                }
            )
        self._tables["EcosimScenarioShapeMedWeightsFleet"] = pd.DataFrame(fleet_rows)

        # Write landings links
        landing_rows = []
        for link in collection.landing_links:
            landing_rows.append(
                {
                    "ScenarioID": 1,
                    "ShapeID": link.shape_id,
                    "GroupID": link.mediator_idx + 1,
                    "FleetID": link.landing_fleet_idx + 1,
                    "AppliedWeight": link.weight,
                }
            )
        self._tables["EcosimScenarioshapeMedWeightsLandings"] = pd.DataFrame(
            landing_rows
        )

    def write_taxonomy(self, taxonomy=None) -> None:
        """Write taxonomy tables to the CSV bundle."""
        if taxonomy is None:
            return

        from pypath.io.ewemdb import (
            _TAXON_EXTERNAL_KEYS,
            _TAXON_METADATA,
            _TAXON_TRAITS,
            _none_to_sentinel,
        )

        taxon_schema = EWE_TABLES["EcopathTaxon"]
        key_to_col = {v: k for k, v in _TAXON_EXTERNAL_KEYS.items()}
        trait_to_col = {v: k for k, v in _TAXON_TRAITS.items()}
        meta_to_col = {v: k for k, v in _TAXON_METADATA.items()}

        # Build EcopathTaxon rows
        taxon_rows = []
        for t in taxonomy.taxa:
            row = {
                "TaxonID": t.taxon_id,
                "GenusName": t.taxonomy.get("genus_name", ""),
                "SpeciesName": t.taxonomy.get("species_name", ""),
                "ClassName": t.taxonomy.get("class_name", ""),
                "OrderName": t.taxonomy.get("order_name", ""),
                "FamilyName": t.taxonomy.get("family_name", ""),
                "CommonName": t.common_name,
                "SourceName": t.source_name,
                "SourceKey": t.source_key,
            }
            for key, col in key_to_col.items():
                val = t.external_keys.get(key)
                sql_type = taxon_schema.get(col, "INTEGER")
                row[col] = _none_to_sentinel(val, sql_type)

            for key, col in trait_to_col.items():
                val = t.traits.get(key)
                row[col] = _none_to_sentinel(val, "DOUBLE")

            for key, col in meta_to_col.items():
                val = t.metadata.get(key)
                sql_type = taxon_schema.get(col, "INTEGER")
                row[col] = _none_to_sentinel(val, sql_type)

            taxon_rows.append(row)

        self._tables["EcopathTaxon"] = (
            pd.DataFrame(
                taxon_rows,
                columns=list(taxon_schema.keys()),
            )
            if taxon_rows
            else pd.DataFrame(columns=list(taxon_schema.keys()))
        )

        self._tables["EcopathGroupTaxon"] = taxonomy.group_assignments.copy()
        self._tables["EcopathStanzaTaxon"] = taxonomy.stanza_assignments.copy()

    def write_value_chain(self, value_chain=None) -> None:
        """Write value chain economics tables to the CSV bundle."""
        if value_chain is None:
            return

        from pypath.io.ewemdb import _VALUE_CHAIN_TABLES

        for attr_name, table_name in _VALUE_CHAIN_TABLES.items():
            df = getattr(value_chain, attr_name, None)
            if df is not None and len(df) > 0:
                self._tables[table_name] = df

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


def _first_of(row: Any, names: tuple, default: Any = None) -> Any:
    """The first of ``names`` present and not missing in ``row``."""
    for name in names:
        value = row.get(name) if hasattr(row, "get") else None
        if value is not None and not (isinstance(value, float) and np.isnan(value)):
            return value
    return default


def _stanza_frames(stanzas: Any):
    """(stgroups, stindiv) DataFrames of RpathParams.stanzas, or (None, None)."""
    stg = getattr(stanzas, "stgroups", None)
    sti = getattr(stanzas, "stindiv", None)
    if not isinstance(stg, pd.DataFrame) or len(stg) == 0:
        return None, None
    if not isinstance(sti, pd.DataFrame) or len(sti) == 0:
        return stg, None
    return stg, sti


def _stage_order(sti: pd.DataFrame) -> pd.DataFrame:
    """stindiv with a 1-based ``_seq`` within each stanza group."""
    sti = sti.copy()
    sti["_group"] = pd.to_numeric(sti.get("StGroupNum", 1), errors="coerce").fillna(1)
    if "StanzaNum" in sti.columns:
        sti["_seq"] = pd.to_numeric(sti["StanzaNum"], errors="coerce")
    else:
        sti["_seq"] = np.nan
    first = pd.to_numeric(sti.get("First", 0), errors="coerce").fillna(0)
    missing = sti["_seq"].isna() | (sti["_seq"] <= 0)
    if missing.any():
        rank = first.groupby(sti["_group"]).rank(method="first")
        sti.loc[missing, "_seq"] = rank[missing]
    sti["_seq"] = sti["_seq"].astype(int)
    return sti.sort_values(["_group", "_seq"])


def _stanza_stage_vbk(stanzas: Any) -> Dict[str, float]:
    """Life-stage group name -> its stanza group's VBGF K."""
    stg, sti = _stanza_frames(stanzas)
    if stg is None or sti is None:
        return {}
    k_of = {}
    for _, row in stg.iterrows():
        k = _first_of(row, ("VBGF_Ksp", "vbgf_ksp"))
        gid = _first_of(row, ("StGroupNum", "stanza_group_num"))
        if k is not None and gid is not None:
            k_of[int(gid)] = float(k)
    out = {}
    for _, row in sti.iterrows():
        gid = _first_of(row, ("StGroupNum", "stanza_group_num"))
        name = _first_of(row, ("Group", "group_name", "GroupName"))
        if gid is not None and name is not None and int(gid) in k_of:
            out[str(name).strip()] = k_of[int(gid)]
    return out


def _flagged_stage(stages: pd.DataFrame, column: str) -> int:
    """Sequence of the stage whose ``column`` is true, or 0 (EwE: unset)."""
    if column not in stages.columns:
        return 0
    flags = stages[column].map(
        lambda v: str(v).strip().lower() in ("true", "1", "1.0", "yes")
    )
    hits = stages.loc[flags, "_seq"]
    return int(hits.iloc[0]) if len(hits) else 0


def _stanza_tables(stanzas: Any, name_to_id: Dict[str, int]):
    """EwE Stanza and StanzaLifeStage rows from RpathParams.stanzas, so that
    the EwE reader gets the same stanza tables back: the stanza ID, name,
    leading life stages (biomass and consumption), maturity weight,
    recruitment power, BABsplit, fixed fecundity, linked recruitment, and
    each stage's sequence within its group, start age, Z, VBGF K and
    SpawnProp (EwE's default 1). EggAtSpawn is written as EwE's default
    (it has no effect on the model)."""
    stg, sti = _stanza_frames(stanzas)
    if stg is None:
        return [], []
    ids = {str(k).strip(): v for k, v in name_to_id.items()}
    stages = _stage_order(sti) if sti is not None else None
    stanza_rows, ls_rows = [], []
    for i, (_, row) in enumerate(stg.iterrows()):
        sid = int(_first_of(row, ("StGroupNum", "stanza_group_num"), i + 1))
        mine = stages[stages["_group"] == sid] if stages is not None else None
        leading = _flagged_stage(mine, "Leading") if mine is not None else 0
        leading_cb = (
            _flagged_stage(mine, "LeadingCB") if mine is not None else 0
        ) or leading
        stanza_rows.append(
            {
                "StanzaID": sid,
                "StanzaName": _first_of(
                    row,
                    ("StanzaGroup", "StGroupName", "StanzaName", "Name"),
                    f"Stanza{sid}",
                ),
                "HatchCode": 0,
                "BABsplit": float(_first_of(row, ("BAB", "BABsplit", "bab"), 0.0)),
                "WmatWinf": _nan_to_none(_first_of(row, ("Wmat", "WmatWinf", "wmat"))),
                "RecPower": _nan_to_none(_first_of(row, ("RecPower", "rec_power"))),
                "FixedFecundity": int(
                    str(_first_of(row, ("FixedFecundity",), 0)).strip().lower()
                    in ("true", "1", "1.0")
                ),
                "LeadingLifeStage": leading,
                "EggAtSpawn": 1,
                "LeadingCB": leading_cb,
                "RecStanza": int(float(_first_of(row, ("RecStanza",), 0))),
            }
        )
        if mine is None:
            continue
        k = _first_of(row, ("VBGF_Ksp", "vbgf_ksp"))
        for _, stage in mine.iterrows():
            name = str(_first_of(stage, ("Group", "GroupName"), "")).strip()
            ls_rows.append(
                {
                    "GroupID": ids.get(name, 0),
                    "StanzaID": sid,
                    "Sequence": int(stage["_seq"]),
                    "AgeStart": int(float(_first_of(stage, ("First", "AgeStart"), 0))),
                    "Mortality": float(_first_of(stage, ("Z", "Mortality"), 0.0)),
                    "vbK": float(k) if k is not None else 0.0,
                    "SpawnProp": float(_first_of(stage, ("SpawnProp",), 1.0)),
                }
            )
    return stanza_rows, ls_rows
