-- EwE 6 Ecospace tables as SQLite: 16 tables.
-- Source: "Anchovy Bay Spatial.ewemdb" in the Official EwE repository
-- (github.com/Official-EwE/Eii.Ecopath, commit 9a52c24; saved by EwE 6.7.0.19005,
-- database version 67019), read with mdb-tools 1.0.1:
--   mdb-schema --no-not-null -T <table> <db> sqlite
-- which is the command of EwE's Tools/mdb2sqlite bridge.

CREATE TABLE `EcospaceScenario`
 (
	`ScenarioID`			INTEGER, 
	`ScenarioName`			varchar, 
	`Description`			TEXT, 
	`Author`			varchar, 
	`Contact`			varchar, 
	`LastSaved`			REAL, 
	`EcosimScenarioID`			INTEGER, 
	`Inrow`			INTEGER, 
	`Incol`			INTEGER, 
	`MinLat`			REAL, 
	`MinLon`			REAL, 
	`CellLength`			REAL, 
	`TimeStep`			REAL, 
	`PredictEffort`			INTEGER, 
	`IFDPower`			REAL, 
	`TotalTime`			REAL, 
	`ModelType`			INTEGER, 
	`NumThreads`			INTEGER, 
	`NumPacketsMultiplier`			REAL, 
	`AdjustSpace`			INTEGER, 
	`UseExact`			INTEGER, 
	`Tolerance`			REAL, 
	`DepthMap`			TEXT, 
	`RelPPMap`			TEXT, 
	`RelCinMap`			TEXT, 
	`DepthAMap`			TEXT, 
	`LastSavedVersion`			varchar, 
	`NumRegions`			INTEGER, 
	`RegionMap`			TEXT, 
	`CellSize`			REAL, 
	`UseEffortDistrThreshold`			INTEGER, 
	`EffortDistrThreshold`			REAL, 
	`ExclusionMap`			TEXT, 
	`AssumeSquareCells`			INTEGER, 
	`CoordinateSystemWKT`			TEXT, 
	`FlowMap`			TEXT, 
	`FitResponseType`			INTEGER, 
	`Q10DriverMap`			TEXT, 
	`UseSpinup`			INTEGER, 
	`SpinupYears`			INTEGER, 
	`CellAreaMap`			TEXT, 
	`NumEffortZones`			INTEGER, 
	`EffortZoneMap`			TEXT, 
	`UsePenaltySearch`			INTEGER, 
	`NoFishWeight`			REAL, 
	`PenaltyPower`			REAL, 
	`FirstPenaltyMonth`			REAL, 
	`UseHabCapGradientCorrections`			INTEGER
	, PRIMARY KEY (`ScenarioID`)
);

CREATE UNIQUE INDEX `EcospaceScenario_ScenarioID_idx` ON `EcospaceScenario` (`ScenarioID`);

CREATE TABLE `EcospaceScenarioCapacityDrivers`
 (
	`ScenarioID`			INTEGER, 
	`GroupID`			INTEGER, 
	`VarDBID`			INTEGER, 
	`ShapeID`			INTEGER, 
	`Target`			INTEGER
	, PRIMARY KEY (`ScenarioID`, `VarDBID`, `GroupID`, `ShapeID`, `Target`)
);

CREATE TABLE `EcospaceScenarioDataConnection`
 (
	`ScenarioID`			INTEGER, 
	`VarName`			varchar, 
	`LayerID`			INTEGER, 
	`Sequence`			INTEGER, 
	`DatasetGUID`			varchar, 
	`DatasetTypeName`			varchar, 
	`DatasetCfg`			TEXT, 
	`ConverterTypeName`			varchar, 
	`ConverterCfg`			TEXT, 
	`Scale`			REAL, 
	`ScaleType`			INTEGER, 
	`CustomDateStart`			varchar, 
	`CustomDateEnd`			varchar
	, PRIMARY KEY (`ScenarioID`, `VarName`, `LayerID`, `Sequence`)
);

CREATE TABLE `EcospaceScenarioDataConnectionDisabled`
 (
	`ScenarioID`			INTEGER, 
	`LayerID`			INTEGER, 
	`Varname`			varchar
	, PRIMARY KEY (`ScenarioID`, `LayerID`, `Varname`)
);

CREATE TABLE `EcospaceScenarioDriverDisabled`
 (
	`ScenarioID`			INTEGER, 
	`LayerID`			INTEGER, 
	`Target`			INTEGER
	, PRIMARY KEY (`ScenarioID`, `LayerID`, `Target`)
);

CREATE TABLE `EcospaceScenarioDriverLayer`
 (
	`ScenarioID`			INTEGER, 
	`LayerID`			INTEGER, 
	`Sequence`			INTEGER, 
	`LayerName`			varchar, 
	`LayerDescription`			TEXT, 
	`LayerMAP`			TEXT, 
	`LayerUnits`			varchar
	, PRIMARY KEY (`ScenarioID`, `LayerID`)
);

CREATE UNIQUE INDEX `EcospaceScenarioDriverLayer_idMap_idx` ON `EcospaceScenarioDriverLayer` (`LayerID`);

CREATE TABLE `EcospaceScenarioFleet`
 (
	`ScenarioID`			INTEGER, 
	`FleetID`			INTEGER, 
	`EcopathFleetID`			INTEGER, 
	`EffPower`			REAL, 
	`PortMap`			TEXT, 
	`SailCostMap`			TEXT, 
	`SEMult`			REAL
	, PRIMARY KEY (`ScenarioID`, `EcopathFleetID`)
);

CREATE INDEX `EcospaceScenarioFleet_EcopathFleetID_idx` ON `EcospaceScenarioFleet` (`EcopathFleetID`);
CREATE UNIQUE INDEX `EcospaceScenarioFleet_EcospaceScenarioFleetFleetID_idx` ON `EcospaceScenarioFleet` (`FleetID`);
CREATE INDEX `EcospaceScenarioFleet_ScenarioID_idx` ON `EcospaceScenarioFleet` (`ScenarioID`);

CREATE TABLE `EcospaceScenarioGroup`
 (
	`ScenarioID`			INTEGER, 
	`GroupID`			INTEGER, 
	`EcopathGroupID`			INTEGER, 
	`Mvel`			REAL, 
	`RelMoveBad`			REAL, 
	`RelVulBad`			REAL, 
	`IsAdvected`			INTEGER, 
	`IsMigratory`			INTEGER, 
	`BarrierAvoidanceWeight`			REAL, 
	`CapacityMap`			TEXT, 
	`CapacityCalType`			INTEGER, 
	`InMigAreaMovement`			REAL, 
	`OtherMortMap`			TEXT, 
	`KMoveFit`			REAL, 
	`FTarget`			REAL
	, PRIMARY KEY (`ScenarioID`, `EcopathGroupID`)
);

CREATE INDEX `EcospaceScenarioGroup_EcospaceGroupID_idx` ON `EcospaceScenarioGroup` (`EcopathGroupID`);
CREATE UNIQUE INDEX `EcospaceScenarioGroup_EcospaceScenarioGroupGroupID_idx` ON `EcospaceScenarioGroup` (`GroupID`);
CREATE INDEX `EcospaceScenarioGroup_ScenarioID_idx` ON `EcospaceScenarioGroup` (`ScenarioID`);

CREATE TABLE `EcospaceScenarioGroupHabitat`
 (
	`ScenarioID`			INTEGER, 
	`GroupID`			INTEGER, 
	`HabitatID`			INTEGER, 
	`Preference`			REAL
	, PRIMARY KEY (`ScenarioID`, `GroupID`, `HabitatID`)
);

CREATE INDEX `EcospaceScenarioGroupHabitat_EcospaceScenarioGroupGroupID_idx` ON `EcospaceScenarioGroupHabitat` (`GroupID`);
CREATE INDEX `EcospaceScenarioGroupHabitat_HabitatID_idx` ON `EcospaceScenarioGroupHabitat` (`HabitatID`);
CREATE INDEX `EcospaceScenarioGroupHabitat_ScenarioID_idx` ON `EcospaceScenarioGroupHabitat` (`ScenarioID`);

CREATE TABLE `EcospaceScenarioGroupMigration`
 (
	`ScenarioID`			INTEGER, 
	`GroupID`			INTEGER, 
	`MonthID`			INTEGER, 
	`Map`			TEXT
	, PRIMARY KEY (`ScenarioID`, `GroupID`, `MonthID`)
);

CREATE TABLE `EcospaceScenarioHabitat`
 (
	`ScenarioID`			INTEGER, 
	`HabitatID`			INTEGER, 
	`HabitatName`			varchar, 
	`Sequence`			INTEGER, 
	`HabitatMap`			TEXT
	, PRIMARY KEY (`ScenarioID`, `HabitatID`)
);

CREATE UNIQUE INDEX `EcospaceScenarioHabitat_HabitatID_idx` ON `EcospaceScenarioHabitat` (`HabitatID`);
CREATE INDEX `EcospaceScenarioHabitat_HabitatNo_idx` ON `EcospaceScenarioHabitat` (`Sequence`);
CREATE INDEX `EcospaceScenarioHabitat_ScenarioID_idx` ON `EcospaceScenarioHabitat` (`ScenarioID`);

CREATE TABLE `EcospaceScenarioHabitatFishery`
 (
	`ScenarioID`			INTEGER, 
	`FleetID`			INTEGER, 
	`HabitatID`			INTEGER
	, PRIMARY KEY (`ScenarioID`, `FleetID`, `HabitatID`)
);

CREATE INDEX `EcospaceScenarioHabitatFishery_FleetID_idx` ON `EcospaceScenarioHabitatFishery` (`FleetID`);
CREATE INDEX `EcospaceScenarioHabitatFishery_HabitatID_idx` ON `EcospaceScenarioHabitatFishery` (`HabitatID`);
CREATE INDEX `EcospaceScenarioHabitatFishery_ScenarioID_idx` ON `EcospaceScenarioHabitatFishery` (`ScenarioID`);

CREATE TABLE `EcospaceScenarioMPA`
 (
	`ScenarioID`			INTEGER, 
	`MPAID`			INTEGER, 
	`Sequence`			INTEGER, 
	`MPAname`			varchar, 
	`MPAmonth`			varchar, 
	`MPAMap`			TEXT
	, PRIMARY KEY (`ScenarioID`, `MPAID`)
);

CREATE UNIQUE INDEX `EcospaceScenarioMPA_MPAID_idx` ON `EcospaceScenarioMPA` (`MPAID`);

CREATE TABLE `EcospaceScenarioMPAFishery`
 (
	`ScenarioID`			INTEGER, 
	`FleetID`			INTEGER, 
	`MPAID`			INTEGER
	, PRIMARY KEY (`ScenarioID`, `FleetID`, `MPAID`)
);

CREATE INDEX `EcospaceScenarioMPAFishery_FleetID_idx` ON `EcospaceScenarioMPAFishery` (`FleetID`);
CREATE INDEX `EcospaceScenarioMPAFishery_HabitatID_idx` ON `EcospaceScenarioMPAFishery` (`MPAID`);
CREATE INDEX `EcospaceScenarioMPAFishery_ScenarioID_idx` ON `EcospaceScenarioMPAFishery` (`ScenarioID`);

CREATE TABLE `EcospaceScenarioMonth`
 (
	`ScenarioID`			INTEGER, 
	`MonthID`			INTEGER, 
	`WindXVelMap`			TEXT, 
	`WindYVelMap`			TEXT, 
	`AdvectionXVelMap`			TEXT, 
	`AdvectionYVelMap`			TEXT, 
	`UpwellingMap`			TEXT
	, PRIMARY KEY (`ScenarioID`, `MonthID`)
);

CREATE TABLE `EcospaceScenarioWeightLayer`
 (
	`ScenarioID`			INTEGER, 
	`LayerID`			INTEGER, 
	`Sequence`			INTEGER, 
	`Name`			varchar, 
	`Description`			TEXT, 
	`Weight`			REAL, 
	`LayerMap`			TEXT
	, PRIMARY KEY (`ScenarioID`, `LayerID`)
);

CREATE UNIQUE INDEX `EcospaceScenarioWeightLayer_LayerID_idx` ON `EcospaceScenarioWeightLayer` (`LayerID`);
CREATE INDEX `EcospaceScenarioWeightLayer_ScenarioID_idx` ON `EcospaceScenarioWeightLayer` (`ScenarioID`);
