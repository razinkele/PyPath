"""Phase 1B EwE export integration tests - Fixed API calls."""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
import pytest

from pypath.core.ecopath import rpath
from pypath.core.ecosim import rsim_scenario
from pypath.core.fleet import FleetDynamics, FleetGroup, GearType
from pypath.core.landings import LandingRecord, LandingsData, LandingType
from pypath.core.nutrients import NutrientContent, NutrientCycle
from pypath.core.params import create_rpath_params
from pypath.core.pedigree import EwePedigreeConfig, PedigreeContainer
from pypath.core.size import SizeClass, SizeDistribution, SizeStructure
from pypath.core.taxonomy import TaxonLink, TaxonomyLinkage


@pytest.fixture
def basic_model():
    """Create a minimal 5-group Ecopath model for testing."""
    params = create_rpath_params(
        groups=["Phyto", "Zoo", "Fish", "Detritus", "Fleet"],
        types=[1, 0, 0, 2, 3],
    )

    params.model.loc[0, "Biomass"] = 10.0
    params.model.loc[0, "PB"] = 200.0
    params.model.loc[0, "EE"] = 0.8

    params.model.loc[1, "Biomass"] = 5.0
    params.model.loc[1, "PB"] = 50.0
    params.model.loc[1, "QB"] = 150.0
    params.model.loc[1, "EE"] = 0.9

    params.model.loc[2, "Biomass"] = 2.0
    params.model.loc[2, "PB"] = 1.0
    params.model.loc[2, "QB"] = 5.0
    params.model.loc[2, "EE"] = 0.5

    params.model.loc[3, "Biomass"] = 100.0

    params.diet["Zoo"] = [1.0, 0.0, 0.0, 0.0, 0.0]
    params.diet["Fish"] = [0.0, 1.0, 0.0, 0.0, 0.0]
    params.diet["Phyto"] = [0.0, 0.0, 0.0, 0.0, 0.0]

    params.model["BioAcc"] = 0.0
    params.model["Unassim"] = 0.2
    params.model.loc[0, "Unassim"] = 0.0
    params.model.loc[3, "Unassim"] = 0.0
    params.model.loc[4, "BioAcc"] = np.nan
    params.model.loc[4, "Unassim"] = np.nan

    params.model.loc[2, "Fleet"] = 0.5

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        model = rpath(params)

    scenario = rsim_scenario(model, params, years=range(1, 6))
    return scenario, params, model


@pytest.fixture
def model_with_pedigree(basic_model):
    """Add PedigreeContainer to basic model."""
    scenario, params, model = basic_model
    pedigree = PedigreeContainer()

    pedigree.add_pedigree("Phyto", EwePedigreeConfig(pedigree_index=1))
    pedigree.add_pedigree("Zoo", EwePedigreeConfig(pedigree_index=2))
    pedigree.add_pedigree("Fish", EwePedigreeConfig(pedigree_index=2))
    pedigree.add_pedigree("Detritus", EwePedigreeConfig(pedigree_index=3))
    pedigree.add_pedigree("Fleet", EwePedigreeConfig(pedigree_index=1))

    return scenario, params, model, pedigree


@pytest.fixture
def model_with_taxonomy(basic_model):
    """Add TaxonomyLinkage to basic model."""
    scenario, params, model = basic_model
    linkage = TaxonomyLinkage()

    phyto_taxon = TaxonLink(
        scientific_name="Diatomeae",
        common_name="Diatoms",
        kingdom="Eukaryota",
        phylum="Bacillariophyta",
        class_="Bacillariophyceae",
        worms_aphia_id=181,
        match_confidence=0.8,
    )
    linkage.add_taxon(phyto_taxon)
    linkage.assign_group("Phyto", phyto_taxon.scientific_name)

    zoo_taxon = TaxonLink(
        scientific_name="Copepoda",
        common_name="Copepods",
        kingdom="Animalia",
        phylum="Arthropoda",
        class_="Copepoda",
        worms_aphia_id=1080,
        match_confidence=0.9,
    )
    linkage.add_taxon(zoo_taxon)
    linkage.assign_group("Zoo", zoo_taxon.scientific_name)

    fish_taxon = TaxonLink(
        scientific_name="Gadus morhua",
        common_name="Atlantic cod",
        kingdom="Animalia",
        phylum="Chordata",
        class_="Actinopterygii",
        worms_aphia_id=126436,
        match_confidence=1.0,
    )
    linkage.add_taxon(fish_taxon)
    linkage.assign_group("Fish", fish_taxon.scientific_name)

    return scenario, params, model, linkage


@pytest.fixture
def model_with_size(basic_model):
    """Add SizeStructure to basic model."""
    scenario, params, model = basic_model

    size_struct = SizeStructure()

    zoo_classes = [
        SizeClass(size_min=0.1, size_max=0.5, size_unit="mm"),
        SizeClass(size_min=0.5, size_max=2.0, size_unit="mm"),
        SizeClass(size_min=2.0, size_max=10.0, size_unit="mm"),
    ]
    zoo_counts = np.array([1000.0, 500.0, 100.0])
    zoo_dist = SizeDistribution(
        group_name="Zoo", size_classes=zoo_classes, counts=zoo_counts
    )
    size_struct.add_distribution(zoo_dist)

    fish_classes = [
        SizeClass(size_min=5.0, size_max=15.0, size_unit="cm"),
        SizeClass(size_min=15.0, size_max=25.0, size_unit="cm"),
        SizeClass(size_min=25.0, size_max=40.0, size_unit="cm"),
        SizeClass(size_min=40.0, size_max=60.0, size_unit="cm"),
    ]
    fish_counts = np.array([200.0, 150.0, 80.0, 20.0])
    fish_dist = SizeDistribution(
        group_name="Fish", size_classes=fish_classes, counts=fish_counts
    )
    size_struct.add_distribution(fish_dist)

    return scenario, params, model, size_struct


@pytest.fixture
def model_with_fleet(basic_model):
    """Add FleetDynamics to basic model."""
    scenario, params, model = basic_model

    fleet_dyn = FleetDynamics()

    fleet1 = FleetGroup(
        name="Small-scale Pelagic",
        fleet_id=0,
        gears=[GearType.NET],
        fuel_efficiency=0.04,
        base_effort=120.0,
        catchability={"Fish": 0.08, "Zoo": 0.02},
    )
    fleet_dyn.add_fleet(fleet1)

    fleet2 = FleetGroup(
        name="Industrial Demersal",
        fleet_id=1,
        gears=[GearType.TRAWL],
        fuel_efficiency=0.12,
        base_effort=250.0,
        catchability={"Fish": 0.15, "Zoo": 0.01},
    )
    fleet_dyn.add_fleet(fleet2)

    return scenario, params, model, fleet_dyn


@pytest.fixture
def model_with_landings(basic_model):
    """Add LandingsData to basic model."""
    scenario, params, model = basic_model

    landings = LandingsData()

    landings.add_record(
        LandingRecord(
            group_name="Fish",
            year=2020,
            month=None,
            landing_type=LandingType.COMMERCIAL,
            catch_tons=5000.0,
            value_usd=2000000.0,
            fleet_id="Fleet1",
        )
    )

    landings.add_record(
        LandingRecord(
            group_name="Fish",
            year=2021,
            month=3,
            landing_type=LandingType.COMMERCIAL,
            catch_tons=1200.0,
            value_usd=None,
            fleet_id="Fleet1",
        )
    )

    landings.add_record(
        LandingRecord(
            group_name="Zoo",
            year=2020,
            month=6,
            landing_type=LandingType.SUBSISTENCE,
            catch_tons=100.0,
            value_usd=5000.0,
            fleet_id="Local",
        )
    )

    return scenario, params, model, landings


@pytest.fixture
def model_with_nutrients(basic_model):
    """Add NutrientCycle to basic model."""
    scenario, params, model = basic_model

    nutrients = NutrientCycle()

    phyto_nut = NutrientContent(
        group_name="Phyto",
        nitrogen_percent=10.0,
        phosphorus_percent=1.5,
        carbon_percent=48.0,
        ash_percent=3.0,
    )
    nutrients.add_content(phyto_nut)

    zoo_nut = NutrientContent(
        group_name="Zoo",
        nitrogen_percent=9.0,
        phosphorus_percent=1.8,
        carbon_percent=49.0,
        ash_percent=8.0,
    )
    nutrients.add_content(zoo_nut)

    fish_nut = NutrientContent(
        group_name="Fish",
        nitrogen_percent=7.5,
        phosphorus_percent=1.2,
        carbon_percent=50.0,
        ash_percent=10.0,
    )
    nutrients.add_content(fish_nut)

    return scenario, params, model, nutrients


class TestPedigreeEweExport:
    """Test PedigreeContainer export to EwE format."""

    @pytest.mark.integration
    def test_pedigree_to_dataframe_format(self, model_with_pedigree):
        """Verify DataFrame export structure for pedigree data."""
        scenario, params, model, pedigree = model_with_pedigree

        df = pedigree.to_dataframe()

        assert isinstance(df, pd.DataFrame)
        assert "group" in df.columns
        assert "pedigree_index" in df.columns
        assert len(df) >= 1

    @pytest.mark.integration
    def test_pedigree_quality_mapping(self, model_with_pedigree):
        """Verify pedigree index is in 0-4 range."""
        scenario, params, model, pedigree = model_with_pedigree

        df = pedigree.to_dataframe()
        indices = df["pedigree_index"].unique()
        for idx in indices:
            assert 0 <= idx <= 4


class TestTaxonomyEweExport:
    """Test TaxonomyLinkage export to EwE format."""

    @pytest.mark.integration
    def test_taxonomy_to_dataframe_format(self, model_with_taxonomy):
        """Verify DataFrame export structure for taxonomy data."""
        scenario, params, model, linkage = model_with_taxonomy

        df = linkage.to_dataframe()

        assert isinstance(df, pd.DataFrame)
        assert "group" in df.columns
        assert "scientific_name" in df.columns

    @pytest.mark.integration
    def test_taxonomy_taxon_retrieval(self, model_with_taxonomy):
        """Test getting taxon by scientific name."""
        scenario, params, model, linkage = model_with_taxonomy

        taxon = linkage.get_taxon("Gadus morhua")
        assert taxon is not None
        assert taxon.common_name == "Atlantic cod"


class TestSizeStructureEweExport:
    """Test SizeStructure export to EwE format."""

    @pytest.mark.integration
    def test_size_to_dataframe_format(self, model_with_size):
        """Verify DataFrame export structure for size data."""
        scenario, params, model, size_struct = model_with_size

        df = size_struct.to_dataframe()

        assert isinstance(df, pd.DataFrame)
        assert "group_name" in df.columns
        assert "size_min" in df.columns
        assert "size_max" in df.columns
        assert "count" in df.columns

    @pytest.mark.integration
    def test_size_distribution_retrieval(self, model_with_size):
        """Test getting size distribution by group name."""
        scenario, params, model, size_struct = model_with_size

        fish_dist = size_struct.get_distribution("Fish")
        assert fish_dist is not None
        assert fish_dist.group_name == "Fish"
        assert len(fish_dist.size_classes) == 4


class TestFleetDynamicsEweExport:
    """Test FleetDynamics export to EwE format."""

    @pytest.mark.integration
    def test_fleet_add_and_retrieve(self, model_with_fleet):
        """Test adding and retrieving fleets."""
        scenario, params, model, fleet_dyn = model_with_fleet

        assert len(fleet_dyn.fleets) >= 2

    @pytest.mark.integration
    def test_fleet_properties(self, model_with_fleet):
        """Test fleet properties and attributes."""
        scenario, params, model, fleet_dyn = model_with_fleet

        fleet = fleet_dyn.fleets.get("Small-scale Pelagic")
        assert fleet is not None
        assert fleet.fleet_id == 0
        assert fleet.base_effort == 120.0
        assert "Fish" in fleet.catchability


class TestLandingsDataEweExport:
    """Test LandingsData export to EwE format."""

    @pytest.mark.integration
    def test_landings_add_and_retrieve(self, model_with_landings):
        """Test adding and retrieving landing records."""
        scenario, params, model, landings = model_with_landings

        assert len(landings.records) >= 3

    @pytest.mark.integration
    def test_landings_annual_catch(self, model_with_landings):
        """Test annual catch aggregation."""
        scenario, params, model, landings = model_with_landings

        fish_2020 = landings.get_annual_catch("Fish", 2020)
        assert fish_2020 == 5000.0


class TestNutrientCycleEweExport:
    """Test NutrientCycle export to EwE format."""

    @pytest.mark.integration
    def test_nutrient_add_and_retrieve(self, model_with_nutrients):
        """Test adding and retrieving nutrient content."""
        scenario, params, model, nutrients = model_with_nutrients

        assert len(nutrients.contents) >= 3

    @pytest.mark.integration
    def test_nutrient_content_values(self, model_with_nutrients):
        """Test nutrient content values."""
        scenario, params, model, nutrients = model_with_nutrients

        phyto_nut = nutrients.contents.get("Phyto")
        assert phyto_nut is not None
        assert phyto_nut.nitrogen_percent == 10.0


class TestModuleInstantiation:
    """Smoke tests to verify all Phase 1B modules can be instantiated."""

    def test_pedigree_instantiation(self):
        """Test PedigreeContainer can be instantiated."""
        pedigree = PedigreeContainer()
        assert isinstance(pedigree, PedigreeContainer)

    def test_taxonomy_instantiation(self):
        """Test TaxonomyLinkage can be instantiated."""
        linkage = TaxonomyLinkage()
        assert isinstance(linkage, TaxonomyLinkage)

    def test_size_structure_instantiation(self):
        """Test SizeStructure can be instantiated."""
        size_struct = SizeStructure()
        assert isinstance(size_struct, SizeStructure)

    def test_fleet_dynamics_instantiation(self):
        """Test FleetDynamics can be instantiated."""
        fleet_dyn = FleetDynamics()
        assert isinstance(fleet_dyn, FleetDynamics)

    def test_landings_data_instantiation(self):
        """Test LandingsData can be instantiated."""
        landings = LandingsData()
        assert isinstance(landings, LandingsData)

    def test_nutrient_cycle_instantiation(self):
        """Test NutrientCycle can be instantiated."""
        nutrients = NutrientCycle()
        assert isinstance(nutrients, NutrientCycle)
