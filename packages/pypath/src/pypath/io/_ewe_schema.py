"""EwE 6.6+ database schema constants for export writer.

Defines table names, column definitions, and mappings between
RpathParams (PyPath) column names and native EwE 6.6+ column names.

These constants are shared by both the Access (.ewemdb) writer
and the CSV bundle writer.

Column names and table names verified against real EwE 6.6+ databases
(e.g. LT2022_0.5ST_final7.eweaccdb).
"""

from collections import OrderedDict

# ---------------------------------------------------------------------------
# EWE_TABLES: {table_name: OrderedDict({column_name: sql_type, ...})}
#
# SQL types: "INTEGER", "DOUBLE", "TEXT", "YESNO"
# Column order matches the canonical EwE 6.6+ Access schema.
# ---------------------------------------------------------------------------

EWE_TABLES = {
    # -----------------------------------------------------------------------
    # Ecopath tables
    # -----------------------------------------------------------------------
    "EcopathModel": OrderedDict(
        [
            ("ModelID", "INTEGER"),
            ("Name", "TEXT"),
            ("Description", "TEXT"),
            ("Author", "TEXT"),
            ("Contact", "TEXT"),
            ("LastSaved", "TEXT"),
            ("NumDigits", "INTEGER"),
            ("GroupDigits", "INTEGER"),
            ("Area", "DOUBLE"),
            ("FirstYear", "INTEGER"),
            ("NumYears", "INTEGER"),
            ("StepsPerYear", "INTEGER"),
            ("UnitCurrency", "TEXT"),
            ("UnitTime", "TEXT"),
            ("UnitMonetary", "TEXT"),
            ("LastSavedVersion", "TEXT"),
            ("Country", "TEXT"),
            ("EcosystemType", "TEXT"),
        ]
    ),
    "EcopathGroup": OrderedDict(
        [
            ("GroupID", "INTEGER"),
            ("GroupName", "TEXT"),
            ("Sequence", "INTEGER"),
            ("Type", "INTEGER"),
            ("Biomass", "DOUBLE"),
            ("Area", "DOUBLE"),
            ("ProdBiom", "DOUBLE"),
            ("ConsBiom", "DOUBLE"),
            ("EcoEfficiency", "DOUBLE"),
            ("ProdCons", "DOUBLE"),
            ("BiomAcc", "DOUBLE"),
            ("BiomAccRate", "DOUBLE"),
            ("Unassim", "DOUBLE"),
            ("DtImports", "DOUBLE"),
            ("Export", "DOUBLE"),
            ("Catch", "DOUBLE"),
            ("ImpVar", "DOUBLE"),
            ("NonMarketValue", "DOUBLE"),
            ("Respiration", "DOUBLE"),
            ("PoolColor", "INTEGER"),
            ("Immigration", "DOUBLE"),
            ("Emigration", "DOUBLE"),
            ("EmigRate", "DOUBLE"),
            ("Production", "DOUBLE"),
            ("vbK", "DOUBLE"),
            ("OtherMort", "DOUBLE"),
        ]
    ),
    "EcopathDietComp": OrderedDict(
        [
            ("PredID", "INTEGER"),
            ("PreyID", "INTEGER"),
            ("Diet", "DOUBLE"),
            ("DetritusFate", "DOUBLE"),
        ]
    ),
    "EcopathFleet": OrderedDict(
        [
            ("FleetID", "INTEGER"),
            ("FleetName", "TEXT"),
            ("Sequence", "INTEGER"),
            ("FixedCost", "DOUBLE"),
            ("VariableCost", "DOUBLE"),
            ("SailingCost", "DOUBLE"),
            ("PoolColor", "INTEGER"),
            ("NominalEffort", "DOUBLE"),
        ]
    ),
    "EcopathCatch": OrderedDict(
        [
            ("GroupID", "INTEGER"),
            ("FleetID", "INTEGER"),
            ("Landing", "DOUBLE"),
            ("Discards", "DOUBLE"),
            ("DiscardMortality", "DOUBLE"),
            ("Price", "DOUBLE"),
        ]
    ),
    "EcopathDiscardFate": OrderedDict(
        [
            ("GroupID", "INTEGER"),
            ("FleetID", "INTEGER"),
            ("DiscardFate", "DOUBLE"),
        ]
    ),
    # -----------------------------------------------------------------------
    # Stanza tables
    # -----------------------------------------------------------------------
    "Stanza": OrderedDict(
        [
            ("StanzaID", "INTEGER"),
            ("StanzaName", "TEXT"),
            ("HatchCode", "INTEGER"),
            ("BABsplit", "DOUBLE"),
            ("WmatWinf", "DOUBLE"),
            ("RecPower", "DOUBLE"),
            ("FixedFecundity", "DOUBLE"),
            ("LeadingLifeStage", "INTEGER"),
            ("EggAtSpawn", "DOUBLE"),
            ("LeadingCB", "DOUBLE"),
            ("RecStanza", "INTEGER"),
        ]
    ),
    "StanzaLifeStage": OrderedDict(
        [
            ("GroupID", "INTEGER"),
            ("StanzaID", "INTEGER"),
            ("Sequence", "INTEGER"),
            ("AgeStart", "INTEGER"),
            ("Mortality", "DOUBLE"),
            ("vbK", "DOUBLE"),
            ("SpawnProp", "DOUBLE"),
        ]
    ),
    # -----------------------------------------------------------------------
    # Ecosim tables
    # -----------------------------------------------------------------------
    "EcosimScenario": OrderedDict(
        [
            ("ScenarioID", "INTEGER"),
            ("ScenarioName", "TEXT"),
            ("Description", "TEXT"),
            ("Author", "TEXT"),
            ("Contact", "TEXT"),
            ("LastSaved", "TEXT"),
            ("TotalTime", "DOUBLE"),
            ("StepSize", "DOUBLE"),
            ("EquilibriumStepSize", "DOUBLE"),
            ("EquilScaleMax", "DOUBLE"),
            ("sorwt", "DOUBLE"),
            ("SystemRecovery", "DOUBLE"),
            ("Discount", "DOUBLE"),
            ("NudgeStart", "DOUBLE"),
            ("NudgeEnd", "DOUBLE"),
            ("NudgeFactor", "DOUBLE"),
            ("DoInteg", "YESNO"),
            ("UseNudge", "YESNO"),
            ("LastSavedVersion", "TEXT"),
            ("ForagingTimeLowerLimit", "DOUBLE"),
        ]
    ),
    "EcosimScenarioGroup": OrderedDict(
        [
            ("ScenarioID", "INTEGER"),
            ("EcopathGroupID", "INTEGER"),
            ("GroupID", "INTEGER"),
            ("FishMortShapeID", "INTEGER"),
            ("Pbmaxs", "DOUBLE"),
            ("FtimeMax", "DOUBLE"),
            ("FtimeAdjust", "DOUBLE"),
            ("MoPred", "DOUBLE"),
            ("FishRateMax", "DOUBLE"),
            ("Show", "YESNO"),
            ("RiskTime", "DOUBLE"),
            ("QmQo", "DOUBLE"),
            ("CmCo", "DOUBLE"),
            ("SwitchPower", "DOUBLE"),
            ("FishMortMax", "DOUBLE"),
            ("Blim", "DOUBLE"),
            ("Bbase", "DOUBLE"),
            ("Fopt", "DOUBLE"),
            ("BiomassCV", "DOUBLE"),
            ("FixedF", "DOUBLE"),
            ("AdditivePredMort", "DOUBLE"),
        ]
    ),
    "EcosimScenarioForcingMatrix": OrderedDict(
        [
            ("ScenarioID", "INTEGER"),
            ("PredID", "INTEGER"),
            ("PreyID", "INTEGER"),
            ("vulnerability", "DOUBLE"),
        ]
    ),
    "EcosimShape": OrderedDict(
        [
            ("ShapeID", "INTEGER"),
            ("ShapeType", "INTEGER"),
            ("IsSeasonal", "YESNO"),
        ]
    ),
    "EcosimShapeTime": OrderedDict(
        [
            ("ShapeID", "INTEGER"),
            ("zScale", "DOUBLE"),
            ("Title", "TEXT"),
            ("zMaxScale", "DOUBLE"),
            ("FunctionType", "INTEGER"),
            ("ApplicationType", "INTEGER"),
            ("FunctionParams", "TEXT"),
        ]
    ),
    "EcosimShapeFishRate": OrderedDict(
        [
            ("ShapeID", "INTEGER"),
            ("zScale", "DOUBLE"),
            ("Title", "TEXT"),
        ]
    ),
    "EcosimTimeSeries": OrderedDict([
        ("TimeSeriesID", "INTEGER"),
        ("ScenarioID", "INTEGER"),
        ("Name", "TEXT"),
        ("DatType", "INTEGER"),
        ("GroupID", "INTEGER"),
        ("FleetID", "INTEGER"),
        ("DatasetID", "INTEGER"),
        ("WtType", "INTEGER"),
        ("PoolColor", "INTEGER"),
    ]),
    "EcosimTimeSeriesValues": OrderedDict([
        ("TimeSeriesID", "INTEGER"),
        ("ScenarioID", "INTEGER"),
        ("TimeStep", "INTEGER"),
        ("Value", "DOUBLE"),
    ]),
    "EcosimTimeSeriesDataset": OrderedDict([
        ("DatasetID", "INTEGER"),
        ("ScenarioID", "INTEGER"),
        ("DatasourceName", "TEXT"),
        ("Enabled", "YESNO"),
    ]),
    "EcosimTimeSeriesSeason": OrderedDict([
        ("TimeSeriesID", "INTEGER"),
        ("ScenarioID", "INTEGER"),
        ("Season", "INTEGER"),
        ("Value", "DOUBLE"),
    ]),
    # -----------------------------------------------------------------------
    # Ecospace tables
    # -----------------------------------------------------------------------
    "EcospaceScenario": OrderedDict(
        [
            ("ScenarioID", "INTEGER"),
            ("ScenarioName", "TEXT"),
            ("Description", "TEXT"),
            ("Author", "TEXT"),
            ("Contact", "TEXT"),
            ("LastSaved", "TEXT"),
            ("EcosimScenarioID", "INTEGER"),
            ("Inrow", "INTEGER"),
            ("Incol", "INTEGER"),
            ("CellLength", "DOUBLE"),
            ("CellSize", "DOUBLE"),
            ("TimeStep", "DOUBLE"),
            ("TotalTime", "DOUBLE"),
            ("MinLon", "DOUBLE"),
            ("MinLat", "DOUBLE"),
            ("LastSavedVersion", "TEXT"),
        ]
    ),
    "EcospaceScenarioGroup": OrderedDict(
        [
            ("ScenarioID", "INTEGER"),
            ("GroupID", "INTEGER"),
            ("EcopathGroupID", "INTEGER"),
            ("Mvel", "DOUBLE"),
            ("RelMoveBad", "DOUBLE"),
            ("RelVulBad", "DOUBLE"),
            ("IsAdvected", "YESNO"),
            ("IsMigratory", "YESNO"),
            ("BarrierAvoidanceWeight", "DOUBLE"),
        ]
    ),
    "EcospaceScenarioHabitat": OrderedDict(
        [
            ("ScenarioID", "INTEGER"),
            ("HabitatID", "INTEGER"),
            ("HabitatName", "TEXT"),
            ("Sequence", "INTEGER"),
        ]
    ),
    "EcospaceScenarioMPA": OrderedDict(
        [
            ("ScenarioID", "INTEGER"),
            ("MPAID", "INTEGER"),
            ("Sequence", "INTEGER"),
            ("MPAname", "TEXT"),
            ("MPAmonth", "INTEGER"),
        ]
    ),
    # -----------------------------------------------------------------------
    # Mediation tables
    # -----------------------------------------------------------------------
    "EcosimShapeMediation": OrderedDict([
        ("ShapeID", "INTEGER"),
        ("Title", "TEXT"),
        ("nPoints", "INTEGER"),
        ("YY1", "DOUBLE"), ("YY2", "DOUBLE"), ("YY3", "DOUBLE"),
        ("YY4", "DOUBLE"), ("YY5", "DOUBLE"), ("YY6", "DOUBLE"),
        ("YY7", "DOUBLE"), ("YY8", "DOUBLE"), ("YY9", "DOUBLE"),
    ]),
    "EcosimScenarioshapeMedWeightsGroup": OrderedDict([
        ("ScenarioID", "INTEGER"),
        ("ShapeID", "INTEGER"),
        ("GroupID", "INTEGER"),
        ("PredID", "INTEGER"),
        ("PreyID", "INTEGER"),
        ("AppliedWeight", "DOUBLE"),
    ]),
    "EcosimScenarioshapeMedWeightsFleet": OrderedDict([
        ("ScenarioID", "INTEGER"),
        ("ShapeID", "INTEGER"),
        ("GroupID", "INTEGER"),
        ("FleetID", "INTEGER"),
        ("AppliedWeight", "DOUBLE"),
    ]),
    "EcosimScenarioshapeMedWeightsLandings": OrderedDict([
        ("ScenarioID", "INTEGER"),
        ("ShapeID", "INTEGER"),
        ("GroupID", "INTEGER"),
        ("FleetID", "INTEGER"),
        ("AppliedWeight", "DOUBLE"),
    ]),
    # -------------------------------------------------------------------
    # Pedigree tables
    # -------------------------------------------------------------------
    "Pedigree": OrderedDict([
        ("LevelID", "INTEGER"),
        ("LevelName", "TEXT"),
        ("VarName", "TEXT"),
        ("Sequence", "INTEGER"),
        ("IndexValue", "DOUBLE"),
        ("Confidence", "DOUBLE"),
        ("LevelColor", "INTEGER"),
        ("Description", "TEXT"),
    ]),
    "EcopathGroupPedigree": OrderedDict([
        ("GroupID", "INTEGER"),
        ("VarName", "TEXT"),
        ("LevelID", "INTEGER"),
    ]),
    # -------------------------------------------------------------------
    # Monte Carlo sample tables
    # -------------------------------------------------------------------
    "EcopathSample": OrderedDict([
        ("SampleID", "INTEGER"),
        ("Hash", "TEXT"),
        ("Source", "TEXT"),
        ("Generated", "TEXT"),
        ("Rating", "DOUBLE"),
        ("SS", "DOUBLE"),
    ]),
    "EcopathGroupSample": OrderedDict([
        ("SampleID", "INTEGER"),
        ("GroupID", "INTEGER"),
        ("Biomass", "DOUBLE"),
        ("ProdBiom", "DOUBLE"),
        ("ConsBiom", "DOUBLE"),
        ("EcoEfficiency", "DOUBLE"),
        ("BiomAcc", "DOUBLE"),
        ("ImpVar", "DOUBLE"),
        ("BiomAccRate", "DOUBLE"),
    ]),
    "EcopathDietCompSample": OrderedDict([
        ("SampleID", "INTEGER"),
        ("PredID", "INTEGER"),
        ("PreyID", "INTEGER"),
        ("Diet", "DOUBLE"),
    ]),
    "EcopathGroupCatchSample": OrderedDict([
        ("SampleID", "INTEGER"),
        ("GroupID", "INTEGER"),
        ("FleetID", "INTEGER"),
        ("Landing", "DOUBLE"),
        ("Discards", "DOUBLE"),
    ]),
    # -------------------------------------------------------------------
    # Fleet dynamics tables
    # -------------------------------------------------------------------
    "EcosimScenarioFleet": OrderedDict([
        ("ScenarioID", "INTEGER"),
        ("EcopathFleetID", "INTEGER"),
        ("CapDepreciate", "DOUBLE"),
        ("CapBaseGrowth", "DOUBLE"),
        ("EffPower", "DOUBLE"),
        ("QmaxQbase", "DOUBLE"),
        ("QchangeRate", "DOUBLE"),
        ("CostOfEffort", "DOUBLE"),
    ]),
    "EcosimScenarioQuota": OrderedDict([
        ("ScenarioID", "INTEGER"),
        ("GroupID", "INTEGER"),
        ("FleetID", "INTEGER"),
        ("QuotaShare", "DOUBLE"),
        ("TAC", "DOUBLE"),
    ]),
    # -------------------------------------------------------------------
    # Ecotracer tables
    # -------------------------------------------------------------------
    "EcotracerScenario": OrderedDict([
        ("ScenarioID", "INTEGER"),
        ("ScenarioName", "TEXT"),
        ("Description", "TEXT"),
        ("Author", "TEXT"),
        ("Contact", "TEXT"),
        ("LastSaved", "TEXT"),
        ("ConForcingShapeID", "INTEGER"),
        ("Czero", "DOUBLE"),
        ("Cinflow", "DOUBLE"),
        ("Coutflow", "DOUBLE"),
        ("Cdecay", "DOUBLE"),
        ("LastSavedVersion", "TEXT"),
    ]),
    "EcotracerScenarioGroup": OrderedDict([
        ("ScenarioID", "INTEGER"),
        ("EcopathGroupID", "INTEGER"),
        ("Czero", "DOUBLE"),
        ("Cimmig", "DOUBLE"),
        ("Cenv", "DOUBLE"),
        ("Cdecay", "DOUBLE"),
        ("CassimProp", "DOUBLE"),
        ("CmetabolismRate", "DOUBLE"),
    ]),
}

# ---------------------------------------------------------------------------
# RPATH_TO_EWE_COLUMNS: RpathParams column name -> EwE 6.6+ column name
# ---------------------------------------------------------------------------

RPATH_TO_EWE_COLUMNS = {
    "Group": "GroupName",
    "Type": "Type",
    "Biomass": "Biomass",
    "PB": "ProdBiom",
    "QB": "ConsBiom",
    "EE": "EcoEfficiency",
    "ProdCons": "ProdCons",
    "Unassim": "Unassim",
    "BioAcc": "BiomAcc",
    "DetInput": "DtImports",
}
