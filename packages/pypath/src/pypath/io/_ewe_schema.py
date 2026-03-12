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
            ("TimeStep", "DOUBLE"),
            ("PredictEffort", "YESNO"),
            ("IFDPower", "DOUBLE"),
            ("TotalTime", "DOUBLE"),
            ("ModelType", "INTEGER"),
            ("NumThreads", "INTEGER"),
            ("NumPacketsMultiplier", "DOUBLE"),
            ("AdjustSpace", "YESNO"),
            ("UseExact", "YESNO"),
            ("Tolerance", "DOUBLE"),
            ("MinLon", "DOUBLE"),
            ("MinLat", "DOUBLE"),
            ("DepthMap", "LONGBINARY"),
            ("RelPPMap", "LONGBINARY"),
            ("RelCinMap", "LONGBINARY"),
            ("DepthAMap", "LONGBINARY"),
            ("LastSavedVersion", "TEXT"),
            ("NumRegions", "INTEGER"),
            ("RegionMap", "LONGBINARY"),
            ("CellSize", "DOUBLE"),
            ("UseEffortDistrThreshold", "YESNO"),
            ("EffortDistrThreshold", "DOUBLE"),
            ("ExclusionMap", "LONGBINARY"),
            ("AssumeSquareCells", "YESNO"),
            ("CoordinateSystemWKT", "TEXT"),
            ("FlowMap", "LONGBINARY"),
            ("FitResponseType", "INTEGER"),
            ("Q10DriverMap", "LONGBINARY"),
            ("UseSpinup", "YESNO"),
            ("SpinupYears", "INTEGER"),
            ("CellAreaMap", "LONGBINARY"),
            ("NumEffortZones", "INTEGER"),
            ("EffortZoneMap", "LONGBINARY"),
            ("UsePenaltySearch", "YESNO"),
            ("NoFishWeight", "DOUBLE"),
            ("PenaltyPower", "DOUBLE"),
            ("FirstPenaltyMonth", "INTEGER"),
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
            ("CapacityMap", "LONGBINARY"),
            ("CapacityCalType", "INTEGER"),
            ("InMigAreaMovement", "DOUBLE"),
            ("OtherMortMap", "LONGBINARY"),
            ("KMoveFit", "DOUBLE"),
            ("FTarget", "DOUBLE"),
        ]
    ),
    "EcospaceScenarioHabitat": OrderedDict(
        [
            ("ScenarioID", "INTEGER"),
            ("HabitatID", "INTEGER"),
            ("HabitatName", "TEXT"),
            ("Sequence", "INTEGER"),
            ("HabitatMap", "LONGBINARY"),
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
    "EcospaceScenarioMPAFishery": OrderedDict(
        [
            ("ScenarioID", "INTEGER"),
            ("MPAID", "INTEGER"),
            ("FleetID", "INTEGER"),
            ("Excluded", "YESNO"),
        ]
    ),
    "EcospaceScenarioFleet": OrderedDict(
        [
            ("ScenarioID", "INTEGER"),
            ("FleetID", "INTEGER"),
            ("EcopathFleetID", "INTEGER"),
            ("EffPower", "DOUBLE"),
            ("PortMap", "LONGBINARY"),
            ("SailCostMap", "LONGBINARY"),
            ("SEMult", "DOUBLE"),
        ]
    ),
    "EcospaceScenarioGroupHabitat": OrderedDict(
        [
            ("ScenarioID", "INTEGER"),
            ("GroupID", "INTEGER"),
            ("HabitatID", "INTEGER"),
            ("Preference", "DOUBLE"),
        ]
    ),
    "EcospaceScenarioCapacityDrivers": OrderedDict(
        [
            ("ScenarioID", "INTEGER"),
            ("GroupID", "INTEGER"),
            ("VarDBID", "INTEGER"),
            ("ShapeID", "INTEGER"),
            ("Target", "INTEGER"),
        ]
    ),
    "EcospaceScenarioDriverLayer": OrderedDict(
        [
            ("ScenarioID", "INTEGER"),
            ("LayerID", "INTEGER"),
            ("Sequence", "INTEGER"),
            ("LayerName", "TEXT"),
            ("LayerDescription", "TEXT"),
            ("LayerMAP", "LONGBINARY"),
            ("LayerUnits", "TEXT"),
        ]
    ),
    # Additional Ecospace tables (verified against EwE 6.6+ LT2022 database)
    "EcospaceScenarioGroupMigration": OrderedDict([
        ("ScenarioID", "INTEGER"),
        ("GroupID", "INTEGER"),
        ("MonthID", "INTEGER"),
        ("Map", "LONGBINARY"),
    ]),
    "EcospaceScenarioMonth": OrderedDict([
        ("ScenarioID", "INTEGER"),
        ("MonthID", "INTEGER"),
        ("WindXVelMap", "LONGBINARY"),
        ("WindYVelMap", "LONGBINARY"),
        ("AdvectionXVelMap", "LONGBINARY"),
        ("AdvectionYVelMap", "LONGBINARY"),
        ("UpwellingMap", "LONGBINARY"),
    ]),
    "EcospaceScenarioWeightLayer": OrderedDict([
        ("ScenarioID", "INTEGER"),
        ("LayerID", "INTEGER"),
        ("Sequence", "INTEGER"),
        ("Name", "TEXT"),
        ("Description", "TEXT"),
        ("Weight", "DOUBLE"),
        ("LayerMap", "LONGBINARY"),
    ]),
    "EcospaceScenarioDataConnection": OrderedDict([
        ("ScenarioID", "INTEGER"),
        ("VarName", "TEXT"),
        ("LayerID", "INTEGER"),
        ("Sequence", "INTEGER"),
        ("DatasetGUID", "TEXT"),
        ("DatasetTypeName", "TEXT"),
        ("DatasetCfg", "TEXT"),
        ("ConverterTypeName", "TEXT"),
        ("ConverterCfg", "TEXT"),
        ("Scale", "DOUBLE"),
        ("ScaleType", "INTEGER"),
        ("CustomDateStart", "TEXT"),
        ("CustomDateEnd", "TEXT"),
    ]),
    "EcospaceScenarioDataConnectionDisabled": OrderedDict([
        ("ScenarioID", "INTEGER"),
        ("LayerID", "INTEGER"),
        ("Varname", "TEXT"),
    ]),
    "EcospaceScenarioDriverDisabled": OrderedDict([
        ("ScenarioID", "INTEGER"),
        ("LayerID", "INTEGER"),
        ("Target", "TEXT"),
    ]),
    "EcospaceScenarioHabitatFishery": OrderedDict([
        ("ScenarioID", "INTEGER"),
        ("FleetID", "INTEGER"),
        ("HabitatID", "INTEGER"),
    ]),
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
    # -------------------------------------------------------------------
    # Taxonomy tables
    # -------------------------------------------------------------------
    "EcopathTaxon": OrderedDict([
        ("TaxonID", "INTEGER"),
        ("ClassName", "TEXT"),
        ("OrderName", "TEXT"),
        ("FamilyName", "TEXT"),
        ("GenusName", "TEXT"),
        ("SpeciesName", "TEXT"),
        ("CommonName", "TEXT"),
        ("SourceName", "TEXT"),
        ("SourceKey", "TEXT"),
        ("LastUpdated", "DOUBLE"),
        ("EcologyType", "INTEGER"),
        ("OrganismType", "INTEGER"),
        ("Exploited", "INTEGER"),
        ("ConservationStatus", "INTEGER"),
        ("OccurrenceStatus", "INTEGER"),
        ("MeanWeight", "DOUBLE"),
        ("MeanLength", "DOUBLE"),
        ("MaxLength", "DOUBLE"),
        ("MeanLifeSpan", "DOUBLE"),
        ("VulnerabiltyIndex", "DOUBLE"),
        ("CodeSAUP", "INTEGER"),
        ("CodeFB", "INTEGER"),
        ("CodeSLB", "INTEGER"),
        ("CodeLCID", "TEXT"),
        ("CodeFAO", "TEXT"),
        ("Winf", "DOUBLE"),
        ("vbgfK", "DOUBLE"),
        ("ExploitationStatus", "TEXT"),
        ("CodeAquaMaps", "TEXT"),
        ("CodeAphia", "INTEGER"),
        ("CodeOBIS", "INTEGER"),
    ]),
    "EcopathGroupTaxon": OrderedDict([
        ("TaxonID", "INTEGER"),
        ("EcopathGroupID", "INTEGER"),
        ("Proportion", "DOUBLE"),
        ("PropCatch", "DOUBLE"),
    ]),
    "EcopathStanzaTaxon": OrderedDict([
        ("TaxonID", "INTEGER"),
        ("StanzaID", "INTEGER"),
    ]),
    # -----------------------------------------------------------------------
    # Value Chain Economics tables (c-prefix, EwE Value Chain plugin)
    # -----------------------------------------------------------------------
    "cOOPStorable": OrderedDict([
        ("xCLASS_NAMEx", "TEXT"),
        ("DBID", "INTEGER"),
        ("AllowEvents", "YESNO"),
    ]),
    "cParameters": OrderedDict([
        ("EquilibriumEffortMin", "DOUBLE"),
        ("EquilibriumEffortMax", "DOUBLE"),
        ("EquilibriumEffortIncrement", "DOUBLE"),
        ("RunWithEcopath", "YESNO"),
        ("RunWithEcosim", "YESNO"),
        ("RunWithSearches", "YESNO"),
    ]),
    "cUnit": OrderedDict([
        ("Sequence", "INTEGER"),
        ("Name", "TEXT"),
        ("Nationality", "TEXT"),
        ("NameLocal", "TEXT"),
        ("DBID", "INTEGER"),
    ]),
    "cEconomicUnit": OrderedDict([
        ("DBID", "INTEGER"),
        ("RevenueLocalDomestic", "DOUBLE"),
        ("RevenueLocalExport", "DOUBLE"),
        ("RevenueForeignDomestic", "DOUBLE"),
        ("RevenueForeignExport", "DOUBLE"),
        ("CostOperating", "DOUBLE"),
        ("CostCapital", "DOUBLE"),
        ("CostLabour", "DOUBLE"),
        ("CostLabourForeign", "DOUBLE"),
        ("CostRawMaterial", "DOUBLE"),
        ("CostRawMaterialForeign", "DOUBLE"),
        ("CostIntermediate", "DOUBLE"),
        ("CostIntermediateForeign", "DOUBLE"),
        ("TaxDirect", "DOUBLE"),
        ("TaxIndirect", "DOUBLE"),
        ("TaxExport", "DOUBLE"),
        ("TaxImport", "DOUBLE"),
        ("SubsidyDirect", "DOUBLE"),
        ("SubsidyIndirect", "DOUBLE"),
        ("EmploymentDirect", "DOUBLE"),
        ("EmploymentIndirect", "DOUBLE"),
        ("DependentsDirect", "DOUBLE"),
        ("DependentsIndirect", "DOUBLE"),
        ("EmploymentDirectForeign", "DOUBLE"),
        ("EmploymentIndirectForeign", "DOUBLE"),
        ("DependentsDirectForeign", "DOUBLE"),
        ("DependentsIndirectForeign", "DOUBLE"),
        ("RevenueLocalDomesticEquil", "DOUBLE"),
        ("RevenueLocalExportEquil", "DOUBLE"),
        ("RevenueForeignDomesticEquil", "DOUBLE"),
        ("RevenueForeignExportEquil", "DOUBLE"),
        ("CostOperatingEquil", "DOUBLE"),
        ("CostCapitalEquil", "DOUBLE"),
        ("CostLabourEquil", "DOUBLE"),
        ("CostLabourForeignEquil", "DOUBLE"),
        ("CostRawMaterialEquil", "DOUBLE"),
        ("CostRawMaterialForeignEquil", "DOUBLE"),
        ("CostIntermediateEquil", "DOUBLE"),
        ("CostIntermediateForeignEquil", "DOUBLE"),
    ]),
    "cProducerUnit": OrderedDict([
        ("DBID", "INTEGER"),
        ("ObserverCost", "DOUBLE"),
        ("ObserverRate", "DOUBLE"),
        ("TicketProducts", "TEXT"),
        ("EcopathFleetID", "INTEGER"),
    ]),
    "cProcessingUnit": OrderedDict([
        ("DBID", "INTEGER"),
        ("AgriculturalProducts", "TEXT"),
        ("AgriculturalInput", "TEXT"),
    ]),
    "cDistributionUnit": OrderedDict([
        ("DBID", "INTEGER"),
    ]),
    "cWholesalerUnit": OrderedDict([
        ("DBID", "INTEGER"),
    ]),
    "cRetailerUnit": OrderedDict([
        ("DBID", "INTEGER"),
    ]),
    "cConsumerUnit": OrderedDict([
        ("DBID", "INTEGER"),
    ]),
    "cProducerDefault": OrderedDict([
        ("DBID", "INTEGER"),
        ("ObserverCost", "DOUBLE"),
        ("ObserverRate", "DOUBLE"),
        ("TicketProducts", "TEXT"),
        ("EcopathFleetID", "INTEGER"),
    ]),
    "cProcessingDefault": OrderedDict([
        ("DBID", "INTEGER"),
        ("AgriculturalProducts", "TEXT"),
        ("AgriculturalInput", "TEXT"),
    ]),
    "cDistributionDefault": OrderedDict([
        ("DBID", "INTEGER"),
    ]),
    "cWholesalerDefault": OrderedDict([
        ("DBID", "INTEGER"),
    ]),
    "cRetailerDefault": OrderedDict([
        ("DBID", "INTEGER"),
    ]),
    "cConsumerDefault": OrderedDict([
        ("DBID", "INTEGER"),
    ]),
    "cLink": OrderedDict([
        ("DBID", "INTEGER"),
    ]),
    "cLinkDefault": OrderedDict([
        ("LinkType", "INTEGER"),
        ("BiomassRatio", "DOUBLE"),
        ("ValuePerTon", "DOUBLE"),
        ("ValueRatio", "DOUBLE"),
    ]),
    "cLinkLandings": OrderedDict([
        ("EcopathGroupID", "INTEGER"),
        ("ValuePerTon", "DOUBLE"),
    ]),
    "cFlowDiagram": OrderedDict([
        ("DBID", "INTEGER"),
        ("Name", "TEXT"),
        ("Description", "TEXT"),
    ]),
    "cFlowPosition": OrderedDict([
        ("DBID", "INTEGER"),
        ("DiagramDBID", "INTEGER"),
        ("UnitDBID", "INTEGER"),
        ("X", "DOUBLE"),
        ("Y", "DOUBLE"),
        ("Width", "DOUBLE"),
        ("Height", "DOUBLE"),
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
