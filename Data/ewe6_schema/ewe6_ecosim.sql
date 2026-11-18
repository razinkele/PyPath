-- EwE 6 Ecosim tables as SQLite: 27 tables.
-- Source: "Anchovy Bay Spatial.ewemdb" in the Official EwE repository
-- (github.com/Official-EwE/Eii.Ecopath, commit 9a52c24; saved by EwE 6.7.0.19005,
-- database version 67019), read with mdb-tools 1.0.1:
--   mdb-schema --no-not-null -T <table> <db> sqlite
-- which is the command of EwE's Tools/mdb2sqlite bridge.

CREATE TABLE `EcosimModel`
 (
	`ModelID`			INTEGER, 
	`ForcePoints`			INTEGER
	, PRIMARY KEY (`ModelID`)
);

CREATE TABLE `EcosimScenario`
 (
	`ScenarioID`			INTEGER, 
	`ScenarioName`			varchar, 
	`Description`			TEXT, 
	`Author`			varchar, 
	`Contact`			varchar, 
	`LastSaved`			REAL, 
	`NutForcingShapeID`			INTEGER, 
	`TotalTime`			INTEGER, 
	`StepSize`			REAL, 
	`EquilibriumStepSize`			REAL, 
	`EquilScaleMax`			REAL, 
	`sorwt`			REAL, 
	`SystemRecovery`			REAL, 
	`Discount`			REAL, 
	`NudgeStart`			REAL, 
	`NudgeEnd`			REAL, 
	`NudgeFactor`			REAL, 
	`DoInteg`			INTEGER, 
	`UseNudge`			INTEGER, 
	`NutBaseFreeProp`			REAL, 
	`NutPBmax`			REAL, 
	`UseVarPQ`			INTEGER, 
	`LastSavedVersion`			varchar, 
	`ForagingTimeLowerLimit`			REAL
	, PRIMARY KEY (`ScenarioID`)
);

CREATE INDEX `EcosimScenario_NutForceShapeID_idx` ON `EcosimScenario` (`NutForcingShapeID`);
CREATE INDEX `EcosimScenario_ScenarioID_idx` ON `EcosimScenario` (`ScenarioID`);

CREATE TABLE `EcosimScenarioArena`
 (
	`ScenarioID`			INTEGER, 
	`PreyID`			INTEGER, 
	`PredID`			INTEGER, 
	`PredSharedID`			INTEGER, 
	`PeatArena`			REAL, 
	`Sequence`			INTEGER
	, PRIMARY KEY (`ScenarioID`, `PreyID`, `PredID`, `PredSharedID`)
);

CREATE TABLE `EcosimScenarioCapacityDrivers`
 (
	`ScenarioID`			INTEGER, 
	`GroupID`			INTEGER, 
	`DriverID`			INTEGER, 
	`ResponseID`			INTEGER, 
	`Target`			INTEGER
	, PRIMARY KEY (`ScenarioID`, `GroupID`, `DriverID`, `ResponseID`, `Target`)
);

CREATE TABLE `EcosimScenarioFleet`
 (
	`ScenarioID`			INTEGER, 
	`EcopathFleetID`			INTEGER, 
	`FishRateShapeID`			INTEGER, 
	`MaxEffort`			REAL, 
	`QuotaType`			INTEGER, 
	`Epower`			REAL, 
	`PCapBase`			REAL, 
	`CapDepreciate`			REAL, 
	`CapBaseGrowth`			REAL, 
	`CV`			REAL, 
	`QIncrease`			REAL, 
	`CatchRefLower`			REAL, 
	`CatchRefUpper`			REAL, 
	`EffortRefLower`			REAL, 
	`EffortRefUpper`			REAL, 
	`MSYEvaluateFleet`			INTEGER, 
	`FleetID`			INTEGER, 
	`EffortConversionFactor`			REAL
	, PRIMARY KEY (`ScenarioID`, `EcopathFleetID`)
);

CREATE INDEX `EcosimScenarioFleet_EcopathFleetID_idx` ON `EcosimScenarioFleet` (`EcopathFleetID`);
CREATE UNIQUE INDEX `EcosimScenarioFleet_FleetID_idx` ON `EcosimScenarioFleet` (`FleetID`);
CREATE INDEX `EcosimScenarioFleet_ScenarioID_idx` ON `EcosimScenarioFleet` (`ScenarioID`);

CREATE TABLE `EcosimScenarioFleetYear`
 (
	`ScenarioID`			INTEGER, 
	`FleetID`			INTEGER, 
	`TimeYear`			INTEGER, 
	`CV`			REAL
	, PRIMARY KEY (`ScenarioID`, `FleetID`, `TimeYear`)
);

CREATE INDEX `EcosimScenarioFleetYear_EcosimScenarioFleetTimeFleetID_idx` ON `EcosimScenarioFleetYear` (`FleetID`);

CREATE TABLE `EcosimScenarioForcingMatrix`
 (
	`ScenarioID`			INTEGER, 
	`PredID`			INTEGER, 
	`PreyID`			INTEGER, 
	`vulnerability`			REAL
	, PRIMARY KEY (`ScenarioID`, `PreyID`, `PredID`)
);

CREATE TABLE `EcosimScenarioGroup`
 (
	`ScenarioID`			INTEGER, 
	`EcopathGroupID`			INTEGER, 
	`GroupID`			INTEGER, 
	`FishMortShapeID`			INTEGER, 
	`Pbmaxs`			REAL, 
	`FtimeMax`			REAL, 
	`FtimeAdjust`			REAL, 
	`MoPred`			REAL, 
	`FishRateMax`			REAL, 
	`Show`			INTEGER, 
	`RiskTime`			REAL, 
	`QmQo`			REAL, 
	`CmCo`			REAL, 
	`SwitchPower`			REAL, 
	`FishMortMax`			REAL, 
	`Blim`			REAL, 
	`Bbase`			REAL, 
	`Fopt`			REAL, 
	`Kalwt`			REAL, 
	`BiomassCV`			REAL, 
	`LowerRisk`			REAL, 
	`UpperRisk`			REAL, 
	`BiomassRefLower`			REAL, 
	`BiomassRefUpper`			REAL, 
	`CatchRefLower`			REAL, 
	`CatchRefUpper`			REAL, 
	`FixedEscapement`			REAL, 
	`RStockRatio`			REAL, 
	`RHalfB0Ratio`			REAL, 
	`RecruitmentCV`			REAL, 
	`FixedF`			REAL, 
	`AdditivePredMort`			REAL
	, PRIMARY KEY (`ScenarioID`, `EcopathGroupID`)
);

CREATE INDEX `EcosimScenarioGroup_EcopathGroupID_idx` ON `EcosimScenarioGroup` (`EcopathGroupID`);
CREATE UNIQUE INDEX `EcosimScenarioGroup_GroupID_idx` ON `EcosimScenarioGroup` (`GroupID`);
CREATE INDEX `EcosimScenarioGroup_ScenarioID_idx` ON `EcosimScenarioGroup` (`ScenarioID`);

CREATE TABLE `EcosimScenarioGroupYear`
 (
	`ScenarioID`			INTEGER, 
	`GroupID`			INTEGER, 
	`TimeYear`			INTEGER, 
	`CVBiom`			REAL
	, PRIMARY KEY (`ScenarioID`, `GroupID`, `TimeYear`)
);

CREATE INDEX `EcosimScenarioGroupYear_EcosimScenarioGroupTimeGroupID_idx` ON `EcosimScenarioGroupYear` (`GroupID`);

CREATE TABLE `EcosimScenarioLandingsShape`
 (
	`ScenarioID`			INTEGER, 
	`FleetID`			INTEGER, 
	`GroupID`			INTEGER, 
	`ShapeID`			INTEGER, 
	`FunctionType`			INTEGER
	, PRIMARY KEY (`ScenarioID`, `FleetID`, `GroupID`, `ShapeID`)
);

CREATE TABLE `EcosimScenarioMSE`
 (
	`ScenarioID`			INTEGER, 
	`AssessMethod`			INTEGER, 
	`AssessPower`			REAL, 
	`NTrials`			INTEGER, 
	`StartIndex`			INTEGER, 
	`MaxEffort`			REAL
	, PRIMARY KEY (`ScenarioID`)
);

CREATE UNIQUE INDEX `EcosimScenarioMSE_ScenarioID_idx` ON `EcosimScenarioMSE` (`ScenarioID`);

CREATE TABLE `EcosimScenarioPredPreyShape`
 (
	`ScenarioID`			INTEGER, 
	`PredID`			INTEGER, 
	`PreyID`			INTEGER, 
	`ShapeID`			INTEGER, 
	`FunctionType`			INTEGER
	, PRIMARY KEY (`ScenarioID`, `PredID`, `PreyID`, `ShapeID`)
);

CREATE INDEX `EcosimScenarioPredPreyShape_ScenarioID_idx` ON `EcosimScenarioPredPreyShape` (`ScenarioID`);

CREATE TABLE `EcosimScenarioQuota`
 (
	`ScenarioID`			INTEGER, 
	`EcosimGroupID`			INTEGER, 
	`FleetID`			INTEGER, 
	`Quota`			REAL, 
	`Fweight`			REAL, 
	`QuotaShare`			REAL
	, PRIMARY KEY (`ScenarioID`, `EcosimGroupID`, `FleetID`)
);

CREATE INDEX `EcosimScenarioQuota_EcosimScenarioID_idx` ON `EcosimScenarioQuota` (`ScenarioID`);
CREATE INDEX `EcosimScenarioQuota_FleetID_idx` ON `EcosimScenarioQuota` (`FleetID`);
CREATE INDEX `EcosimScenarioQuota_GroupID_idx` ON `EcosimScenarioQuota` (`EcosimGroupID`);

CREATE TABLE `EcosimScenarioShapeMedWeightsFleet`
 (
	`ScenarioID`			INTEGER, 
	`ShapeID`			INTEGER, 
	`FleetID`			INTEGER, 
	`MedWeights`			REAL
	, PRIMARY KEY (`ScenarioID`, `ShapeID`, `FleetID`)
);

CREATE INDEX `EcosimScenarioShapeMedWeightsFleet_ScenarioID_idx` ON `EcosimScenarioShapeMedWeightsFleet` (`ScenarioID`);

CREATE TABLE `EcosimScenarioshapeMedWeightsGroup`
 (
	`ScenarioID`			INTEGER, 
	`ShapeID`			INTEGER, 
	`GroupID`			INTEGER, 
	`MedWeights`			REAL
	, PRIMARY KEY (`ScenarioID`, `ShapeID`, `GroupID`)
);

CREATE INDEX `EcosimScenarioshapeMedWeightsGroup_ScenarioID_idx` ON `EcosimScenarioshapeMedWeightsGroup` (`ScenarioID`);

CREATE TABLE `EcosimScenarioshapeMedWeightsLandings`
 (
	`ScenarioID`			INTEGER, 
	`GroupID`			INTEGER, 
	`FleetID`			INTEGER, 
	`ShapeID`			INTEGER, 
	`MedWeights`			REAL
	, PRIMARY KEY (`ScenarioID`, `GroupID`, `FleetID`, `ShapeID`)
);

CREATE TABLE `EcosimShape`
 (
	`ShapeID`			INTEGER, 
	`ShapeType`			INTEGER, 
	`IsSeasonal`			INTEGER
	, PRIMARY KEY (`ShapeID`)
);

CREATE INDEX `EcosimShape_ShapeID_idx` ON `EcosimShape` (`ShapeID`);

CREATE TABLE `EcosimShapeEggProd`
 (
	`ShapeID`			INTEGER, 
	`zScale`			TEXT, 
	`Title`			varchar, 
	`zMaxScale`			REAL, 
	`FunctionType`			INTEGER, 
	`FunctionParams`			TEXT
	, PRIMARY KEY (`ShapeID`)
);

CREATE INDEX `EcosimShapeEggProd_ShapeID_idx` ON `EcosimShapeEggProd` (`ShapeID`);

CREATE TABLE `EcosimShapeFishMort`
 (
	`ShapeID`			INTEGER, 
	`zScale`			TEXT, 
	`Title`			varchar
	, PRIMARY KEY (`ShapeID`)
);

CREATE INDEX `EcosimShapeFishMort_ShapeID_idx` ON `EcosimShapeFishMort` (`ShapeID`);

CREATE TABLE `EcosimShapeFishRate`
 (
	`ShapeID`			INTEGER, 
	`zScale`			TEXT, 
	`Title`			varchar
	, PRIMARY KEY (`ShapeID`)
);

CREATE INDEX `EcosimShapeFishRate_ShapeID_idx` ON `EcosimShapeFishRate` (`ShapeID`);

CREATE TABLE `EcosimShapeMediation`
 (
	`ShapeID`			INTEGER, 
	`zScale`			TEXT, 
	`Title`			varchar, 
	`IMedBase`			INTEGER, 
	`FunctionType`			INTEGER, 
	`XAxisMin`			REAL, 
	`XAxisMax`			REAL, 
	`FunctionParams`			TEXT, 
	`Units`			varchar
	, PRIMARY KEY (`ShapeID`)
);

CREATE INDEX `EcosimShapeMediation_ShapeID_idx` ON `EcosimShapeMediation` (`ShapeID`);

CREATE TABLE `EcosimShapeTime`
 (
	`ShapeID`			INTEGER, 
	`zScale`			TEXT, 
	`Title`			varchar, 
	`zMaxScale`			REAL, 
	`FunctionType`			INTEGER, 
	`ApplicationType`			INTEGER, 
	`FunctionParams`			TEXT
	, PRIMARY KEY (`ShapeID`)
);

CREATE INDEX `EcosimShapeTime_ShapeID_idx` ON `EcosimShapeTime` (`ShapeID`);

CREATE TABLE `EcosimStanzaShape`
 (
	`StanzaID`			INTEGER, 
	`EggprodShapeID`			INTEGER, 
	`HatchCodeShapeID`			INTEGER
	, PRIMARY KEY (`StanzaID`)
);

CREATE INDEX `EcosimStanzaShape_EggprodShapeID_idx` ON `EcosimStanzaShape` (`EggprodShapeID`);
CREATE INDEX `EcosimStanzaShape_HatchCodeShapeID_idx` ON `EcosimStanzaShape` (`HatchCodeShapeID`);
CREATE INDEX `EcosimStanzaShape_StanzaID_idx` ON `EcosimStanzaShape` (`StanzaID`);

CREATE TABLE `EcosimTimeSeries`
 (
	`TimeSeriesID`			INTEGER, 
	`DatasetID`			INTEGER, 
	`Sequence`			INTEGER, 
	`DatType`			INTEGER, 
	`DatName`			varchar, 
	`TimeValues`			TEXT, 
	`WtType`			REAL, 
	`CV`			REAL
	, PRIMARY KEY (`TimeSeriesID`)
);

CREATE INDEX `EcosimTimeSeries_DatasetID_idx` ON `EcosimTimeSeries` (`DatasetID`);
CREATE INDEX `EcosimTimeSeries_TimeSeriesID_idx` ON `EcosimTimeSeries` (`TimeSeriesID`);

CREATE TABLE `EcosimTimeSeriesDataset`
 (
	`DatasetID`			INTEGER, 
	`DatasetName`			varchar, 
	`Description`			TEXT, 
	`Author`			varchar, 
	`Contact`			varchar, 
	`FirstYear`			INTEGER, 
	`DataInterval`			INTEGER, 
	`NumPoints`			INTEGER
	, PRIMARY KEY (`DatasetID`)
);

CREATE UNIQUE INDEX `EcosimTimeSeriesDataset_DatasetID_idx` ON `EcosimTimeSeriesDataset` (`DatasetID`);

CREATE TABLE `EcosimTimeSeriesFleet`
 (
	`TimeSeriesID`			INTEGER, 
	`FleetID`			INTEGER, 
	`GroupID`			INTEGER
	, PRIMARY KEY (`TimeSeriesID`)
);

CREATE INDEX `EcosimTimeSeriesFleet_GroupID_idx` ON `EcosimTimeSeriesFleet` (`FleetID`);
CREATE INDEX `EcosimTimeSeriesFleet_TimeSeriesID_idx` ON `EcosimTimeSeriesFleet` (`TimeSeriesID`);

CREATE TABLE `EcosimTimeSeriesGroup`
 (
	`TimeSeriesID`			INTEGER, 
	`GroupID`			INTEGER, 
	`VariableName`			varchar
	, PRIMARY KEY (`TimeSeriesID`)
);

CREATE INDEX `EcosimTimeSeriesGroup_GroupID_idx` ON `EcosimTimeSeriesGroup` (`GroupID`);
CREATE INDEX `EcosimTimeSeriesGroup_TimeSeriesID_idx` ON `EcosimTimeSeriesGroup` (`TimeSeriesID`);
