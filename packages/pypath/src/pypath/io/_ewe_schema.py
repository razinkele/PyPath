"""EwE 6 database schema constants for export writer.

Defines table names, column definitions, and mappings between
RpathParams (PyPath) column names and native EwE 6 column names.

These constants are shared by both the Access (.ewemdb) writer
and the CSV bundle writer.
"""

from collections import OrderedDict

# ---------------------------------------------------------------------------
# EWE_TABLES: {table_name: OrderedDict({column_name: sql_type, ...})}
#
# SQL types: "INTEGER", "DOUBLE", "TEXT", "YESNO"
# Column order matches the canonical EwE 6 Access schema.
# ---------------------------------------------------------------------------

EWE_TABLES = {
    # -----------------------------------------------------------------------
    # Ecopath tables
    # -----------------------------------------------------------------------
    "EcopathModel": OrderedDict(
        [
            ("ModelID", "INTEGER"),
            ("ModelName", "TEXT"),
            ("Description", "TEXT"),
            ("Author", "TEXT"),
            ("Contact", "TEXT"),
            ("LastSaved", "TEXT"),
            ("AreaUnit", "TEXT"),
            ("TimeUnit", "TEXT"),
            ("Currency", "TEXT"),
            ("NumGroups", "INTEGER"),
            ("NumFleets", "INTEGER"),
            ("NumLiving", "INTEGER"),
            ("NumDetritus", "INTEGER"),
            ("GroupDigits", "INTEGER"),
            ("EcopathVersion", "DOUBLE"),
        ]
    ),
    "EcopathGroup": OrderedDict(
        [
            ("GroupID", "INTEGER"),
            ("ModelID", "INTEGER"),
            ("GroupName", "TEXT"),
            ("Sequence", "INTEGER"),
            ("Type", "INTEGER"),
            ("PP", "INTEGER"),
            ("Area", "DOUBLE"),
            ("Biomass", "DOUBLE"),
            ("BiomassAreaRate", "DOUBLE"),
            ("BiomassHabitat", "DOUBLE"),
            ("PB", "DOUBLE"),
            ("QB", "DOUBLE"),
            ("EE", "DOUBLE"),
            ("GE", "DOUBLE"),
            ("GS", "DOUBLE"),
            ("BA", "DOUBLE"),
            ("BaBi", "DOUBLE"),
            ("Emig", "DOUBLE"),
            ("EmigRate", "DOUBLE"),
            ("Immig", "DOUBLE"),
            ("ImmigEmig", "DOUBLE"),
            ("DetInput", "DOUBLE"),
            ("NonMarketValue", "DOUBLE"),
            ("pprod", "DOUBLE"),
            ("VBK", "DOUBLE"),
        ]
    ),
    "EcopathDietComp": OrderedDict(
        [
            ("ModelID", "INTEGER"),
            ("PredID", "INTEGER"),
            ("PreyID", "INTEGER"),
            ("Diet", "DOUBLE"),
        ]
    ),
    "EcopathFleet": OrderedDict(
        [
            ("FleetID", "INTEGER"),
            ("ModelID", "INTEGER"),
            ("FleetName", "TEXT"),
            ("Sequence", "INTEGER"),
            ("FixedCost", "DOUBLE"),
            ("SailingCost", "DOUBLE"),
            ("ProfitMargin", "DOUBLE"),
        ]
    ),
    "EcopathCatch": OrderedDict(
        [
            ("ModelID", "INTEGER"),
            ("FleetID", "INTEGER"),
            ("GroupID", "INTEGER"),
            ("Landing", "DOUBLE"),
            ("Discard", "DOUBLE"),
            ("DiscardMortality", "DOUBLE"),
            ("Price", "DOUBLE"),
        ]
    ),
    "EcopathDetritusFate": OrderedDict(
        [
            ("ModelID", "INTEGER"),
            ("GroupID", "INTEGER"),
            ("DetritusID", "INTEGER"),
            ("FleetID", "INTEGER"),
            ("DetritusFate", "DOUBLE"),
        ]
    ),
    # -----------------------------------------------------------------------
    # Stanza tables
    # -----------------------------------------------------------------------
    "Stanza": OrderedDict(
        [
            ("StanzaID", "INTEGER"),
            ("ModelID", "INTEGER"),
            ("StanzaName", "TEXT"),
            ("BABsplit", "DOUBLE"),
            ("WmatWinf", "DOUBLE"),
            ("RecPower", "DOUBLE"),
            ("VBK", "DOUBLE"),
        ]
    ),
    "StanzaLifeStage": OrderedDict(
        [
            ("StanzaID", "INTEGER"),
            ("LifeStageID", "INTEGER"),
            ("ModelID", "INTEGER"),
            ("GroupID", "INTEGER"),
            ("Months", "INTEGER"),
            ("LeadingLifeStage", "YESNO"),
            ("LeadingBiomass", "YESNO"),
        ]
    ),
    # -----------------------------------------------------------------------
    # Ecosim tables
    # -----------------------------------------------------------------------
    "EcosimScenario": OrderedDict(
        [
            ("ScenarioID", "INTEGER"),
            ("ModelID", "INTEGER"),
            ("ScenarioName", "TEXT"),
            ("Description", "TEXT"),
            ("NumYears", "INTEGER"),
            ("StepsPerYear", "INTEGER"),
            ("StepsPerMonth", "INTEGER"),
            ("Discount", "DOUBLE"),
            ("NudgeChecked", "YESNO"),
            ("SystemRecovery", "DOUBLE"),
        ]
    ),
    "EcosimGroupInfo": OrderedDict(
        [
            ("ScenarioID", "INTEGER"),
            ("ModelID", "INTEGER"),
            ("GroupID", "INTEGER"),
            ("MaxRelPB", "DOUBLE"),
            ("MaxRelFeedTime", "DOUBLE"),
            ("FeedTimeAdjRate", "DOUBLE"),
            ("OtherProdRate", "DOUBLE"),
            ("PredEffectRate", "DOUBLE"),
            ("DenDepCatchability", "DOUBLE"),
            ("QBMaxQBO", "DOUBLE"),
            ("SwitchingPower", "DOUBLE"),
        ]
    ),
    "EcosimScenarioGroup": OrderedDict(
        [
            ("ScenarioID", "INTEGER"),
            ("ModelID", "INTEGER"),
            ("GroupID", "INTEGER"),
            ("VulMult", "DOUBLE"),
        ]
    ),
    "EcosimForcing": OrderedDict(
        [
            ("ForcingID", "INTEGER"),
            ("ScenarioID", "INTEGER"),
            ("ModelID", "INTEGER"),
            ("ForcingName", "TEXT"),
            ("ForcingType", "INTEGER"),
        ]
    ),
    "EcosimShapeTime": OrderedDict(
        [
            ("ShapeID", "INTEGER"),
            ("ScenarioID", "INTEGER"),
            ("ModelID", "INTEGER"),
            ("TimeStep", "INTEGER"),
            ("Value", "DOUBLE"),
        ]
    ),
    "EcosimScenarioForcingMatrix": OrderedDict(
        [
            ("ScenarioID", "INTEGER"),
            ("ModelID", "INTEGER"),
            ("ForcingID", "INTEGER"),
            ("GroupID", "INTEGER"),
            ("PreyID", "INTEGER"),
        ]
    ),
    "EcosimShapeFishRate": OrderedDict(
        [
            ("ShapeID", "INTEGER"),
            ("ScenarioID", "INTEGER"),
            ("ModelID", "INTEGER"),
            ("FleetID", "INTEGER"),
            ("GroupID", "INTEGER"),
            ("TimeStep", "INTEGER"),
            ("Value", "DOUBLE"),
        ]
    ),
    # -----------------------------------------------------------------------
    # Ecospace tables
    # -----------------------------------------------------------------------
    "EcospaceScenario": OrderedDict(
        [
            ("ScenarioID", "INTEGER"),
            ("ModelID", "INTEGER"),
            ("ScenarioName", "TEXT"),
            ("Description", "TEXT"),
            ("NumRows", "INTEGER"),
            ("NumCols", "INTEGER"),
            ("CellLength", "DOUBLE"),
            ("BaseLat", "DOUBLE"),
            ("BaseLon", "DOUBLE"),
            ("NumYears", "INTEGER"),
            ("StepsPerYear", "INTEGER"),
        ]
    ),
    "EcospaceGroup": OrderedDict(
        [
            ("ScenarioID", "INTEGER"),
            ("ModelID", "INTEGER"),
            ("GroupID", "INTEGER"),
            ("DispRate", "DOUBLE"),
            ("DispRateM", "DOUBLE"),
            ("RelDisp", "DOUBLE"),
            ("RelVul", "DOUBLE"),
            ("RelFeedRate", "DOUBLE"),
            ("GravityX", "DOUBLE"),
            ("GravityY", "DOUBLE"),
            ("IsAdvected", "YESNO"),
            ("IsMigrating", "YESNO"),
        ]
    ),
    "EcospaceHabitat": OrderedDict(
        [
            ("HabitatID", "INTEGER"),
            ("ScenarioID", "INTEGER"),
            ("ModelID", "INTEGER"),
            ("HabitatName", "TEXT"),
        ]
    ),
    "EcospaceMap": OrderedDict(
        [
            ("ScenarioID", "INTEGER"),
            ("ModelID", "INTEGER"),
            ("Row", "INTEGER"),
            ("Col", "INTEGER"),
            ("Depth", "DOUBLE"),
            ("InModelArea", "YESNO"),
            ("HabitatID", "INTEGER"),
            ("RegionID", "INTEGER"),
            ("MPAID", "INTEGER"),
        ]
    ),
    "EcospaceMPA": OrderedDict(
        [
            ("MPAID", "INTEGER"),
            ("ScenarioID", "INTEGER"),
            ("ModelID", "INTEGER"),
            ("MPAName", "TEXT"),
            ("IsActive", "YESNO"),
        ]
    ),
    "EcospaceRegion": OrderedDict(
        [
            ("RegionID", "INTEGER"),
            ("ScenarioID", "INTEGER"),
            ("ModelID", "INTEGER"),
            ("RegionName", "TEXT"),
        ]
    ),
}

# ---------------------------------------------------------------------------
# RPATH_TO_EWE_COLUMNS: RpathParams column name → EwE 6 column name
# ---------------------------------------------------------------------------

RPATH_TO_EWE_COLUMNS = {
    "Group": "GroupName",
    "Type": "Type",
    "Biomass": "Biomass",
    "PB": "PB",
    "QB": "QB",
    "EE": "EE",
    "ProdCons": "GE",
    "Unassim": "GS",
    "BioAcc": "BA",
    "DetInput": "ImmigEmig",
}

# ---------------------------------------------------------------------------
# TYPE_TO_PP: RpathParams Type value → EwE PP code
#   0 = consumer, 1 = producer, 2 = detritus, 3 = fleet (mapped to 0)
# ---------------------------------------------------------------------------

TYPE_TO_PP = {
    0: 0,  # consumer
    1: 1,  # producer
    2: 2,  # detritus
    3: 0,  # fleet (treated as consumer type in EwE PP)
}
