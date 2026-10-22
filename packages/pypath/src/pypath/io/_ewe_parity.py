"""EwE table parity targets and audit helpers.

This module defines the frozen EwE table target surface used for parity
tracking and provides lightweight static analysis helpers used by tests and CI.
"""

from __future__ import annotations

from pathlib import Path

from pypath.io._ewe_schema import EWE_TABLES

# EwE desktop/system tables intentionally excluded from feature-parity scope.
EXCLUDED_NON_PARITY_TABLES: tuple[str, ...] = (
    "Auxillary",
    "EcopathCatchCode",
    "Quote",
    "UpdateLog",
)

# Frozen target surface for parity tracking (84 tables).
TARGET_EWE_TABLES: tuple[str, ...] = (
    "cConsumerUnit",
    "cConsumerUnitDefault",
    "cDistributionUnit",
    "cDistributionUnitDefault",
    "cEconomicUnit",
    "cFlowDiagram",
    "cFlowPosition",
    "cLink",
    "cLinkDefault",
    "cLinkLandings",
    "cOOPStorable",
    "cParameters",
    "cProcessingUnit",
    "cProcessingUnitDefault",
    "cProducerUnit",
    "cProducerUnitDefault",
    "cRetailerUnit",
    "cRetailerUnitDefault",
    "cUnit",
    "cWholesalerUnit",
    "cWholesalerUnitDefault",
    "EcopathCatch",
    "EcopathDietComp",
    "EcopathDietCompSample",
    "EcopathDiscardFate",
    "EcopathFleet",
    "EcopathGroup",
    "EcopathGroupCatchSample",
    "EcopathGroupPedigree",
    "EcopathGroupSample",
    "EcopathGroupTaxon",
    "EcopathModel",
    "EcopathPSD",
    "EcopathSample",
    "EcopathStanzaTaxon",
    "EcopathTaxon",
    "EcosimModel",
    "EcosimScenario",
    "EcosimScenarioArena",
    "EcosimScenarioCapacityDrivers",
    "EcosimScenarioFleet",
    "EcosimScenarioFleetYear",
    "EcosimScenarioForcingMatrix",
    "EcosimScenarioGroup",
    "EcosimScenarioGroupYear",
    "EcosimScenarioLandingsShape",
    "EcosimScenarioMSE",
    "EcosimScenarioPredPreyShape",
    "EcosimScenarioQuota",
    "EcosimScenarioShapeMedWeightsFleet",
    "EcosimScenarioshapeMedWeightsGroup",
    "EcosimScenarioshapeMedWeightsLandings",
    "EcosimShape",
    "EcosimShapeEggProd",
    "EcosimShapeFishMort",
    "EcosimShapeFishRate",
    "EcosimShapeMediation",
    "EcosimShapeTime",
    "EcosimStanzaShape",
    "EcosimTimeSeries",
    "EcosimTimeSeriesDataset",
    "EcosimTimeSeriesFleet",
    "EcosimTimeSeriesGroup",
    "EcospaceScenario",
    "EcospaceScenarioCapacityDrivers",
    "EcospaceScenarioDataConnection",
    "EcospaceScenarioDataConnectionDisabled",
    "EcospaceScenarioDriverDisabled",
    "EcospaceScenarioDriverLayer",
    "EcospaceScenarioFleet",
    "EcospaceScenarioGroup",
    "EcospaceScenarioGroupHabitat",
    "EcospaceScenarioGroupMigration",
    "EcospaceScenarioHabitat",
    "EcospaceScenarioHabitatFishery",
    "EcospaceScenarioMonth",
    "EcospaceScenarioMPA",
    "EcospaceScenarioMPAFishery",
    "EcospaceScenarioWeightLayer",
    "EcotracerScenario",
    "EcotracerScenarioGroup",
    "Pedigree",
    "Stanza",
    "StanzaLifeStage",
)

PARITY_MATRIX_COLUMNS: tuple[str, ...] = (
    "table_name",
    "domain",
    "in_schema",
    "reader_reference",
    "access_writer_reference",
    "csv_writer_reference",
    "overall_status",
    "notes",
)


def domain_for_table(table_name: str) -> str:
    """Return domain grouping used in parity reports."""
    if table_name.startswith("Ecopath") or table_name.startswith("Stanza"):
        return "ecopath"
    if table_name.startswith("Ecosim"):
        return "ecosim"
    if table_name.startswith("Ecospace"):
        return "ecospace"
    if table_name.startswith("Ecotracer"):
        return "ecotracer"
    if table_name.startswith("c"):
        return "value_chain"
    if table_name.startswith("Pedigree"):
        return "pedigree"
    return "other"


def _contains_table_reference(source_text: str, table_name: str) -> bool:
    return f'"{table_name}"' in source_text or f"'{table_name}'" in source_text


def _read_text(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def derive_table_support() -> dict[str, dict[str, bool]]:
    """Derive schema/reader/writer table coverage from source."""
    base = Path(__file__).resolve().parent
    reader_text = _read_text(base / "ewemdb.py")
    access_text = _read_text(base / "_access_writer.py")
    csv_text = _read_text(base / "_csv_bundle_writer.py")
    schema_tables = set(EWE_TABLES.keys())

    support: dict[str, dict[str, bool]] = {}
    for table_name in TARGET_EWE_TABLES:
        support[table_name] = {
            "in_schema": table_name in schema_tables,
            "reader_reference": _contains_table_reference(reader_text, table_name),
            "access_writer_reference": _contains_table_reference(
                access_text, table_name
            ),
            "csv_writer_reference": _contains_table_reference(csv_text, table_name),
        }
    return support


def status_from_flags(flags: dict[str, bool]) -> str:
    """Return coarse status bucket from support booleans."""
    if all(flags.values()):
        return "implemented"
    if any(flags.values()):
        return "partial"
    return "missing"
