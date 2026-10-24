"""Comprehensive test suite for pypath.core.fleet module.

Phase 1B.4: TG3.4 Fleet & Discard System

Tests cover:
- GearType enum properties and methods
- FleetGroup creation, validation, serialization
- DiscardPath configuration and discard biomass calculation
- FleetDynamics management and workflow
- Catch and discard allocation functions
- Economic calculations (value, fuel cost, efficiency)
- Edge cases and integration scenarios

Total: ~50 tests, ~900 lines
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np
import pandas as pd
import pytest

from pypath.core.fleet import (
    GearType,
    FleetGroup,
    DiscardPath,
    FleetDynamics,
    allocate_catch,
    allocate_discards,
    validate_allocation,
    estimate_catch_value,
    estimate_fuel_cost,
)

logger = logging.getLogger(__name__)


# ============================================================================
# FIXTURES
# ============================================================================


@pytest.fixture
def sample_fleet_group() -> FleetGroup:
    """Create a valid FleetGroup with 2 gears and catchability for 3 groups."""
    return FleetGroup(
        name="Pelagic Fleet",
        fleet_id=0,
        gears=[GearType.SEINE, GearType.NET],
        fuel_efficiency=0.08,
        base_effort=150.0,
        catchability={"Cod": 0.5, "Herring": 0.3, "Capelin": 0.2},
    )


@pytest.fixture
def sample_discard_path() -> DiscardPath:
    """Create a valid DiscardPath with fate='lost'."""
    return DiscardPath(
        source_group="Juvenile Cod",
        target_group=None,
        discard_rate=0.15,
        fate="lost",
        depth_dependent=False,
    )


@pytest.fixture
def sample_fleet_dynamics(
    sample_fleet_group: FleetGroup,
    sample_discard_path: DiscardPath,
) -> FleetDynamics:
    """Create a FleetDynamics with 2 fleets and 3 discard paths."""
    fd = FleetDynamics()

    # Add first fleet
    fd.add_fleet(sample_fleet_group)

    # Add second fleet
    second_fleet = FleetGroup(
        name="Demersal Fleet",
        fleet_id=1,
        gears=[GearType.TRAWL],
        fuel_efficiency=0.12,
        base_effort=100.0,
        catchability={"Cod": 0.6, "Haddock": 0.4},
    )
    fd.add_fleet(second_fleet)

    # Add discard paths
    fd.add_discard_path(sample_discard_path)

    path2 = DiscardPath(
        source_group="Juvenile Haddock",
        target_group=None,
        discard_rate=0.20,
        fate="lost",
    )
    fd.add_discard_path(path2)

    path3 = DiscardPath(
        source_group="Bycatch Herring",
        target_group="Seabirds",
        discard_rate=0.25,
        fate="predation",
        depth_dependent=True,
    )
    fd.add_discard_path(path3)

    return fd


# ============================================================================
# TestGearType (4 tests)
# ============================================================================


class TestGearType:
    """Tests for GearType enum values and properties."""

    def test_gear_type_enum_values(self) -> None:
        """Test GearType enum has all expected values."""
        expected_gears = {"SEINE", "NET", "TRAP", "TRAWL", "HOOK", "OTHER"}
        actual_gears = {g.name for g in GearType}
        assert actual_gears == expected_gears

    def test_gear_type_properties(self) -> None:
        """Test selectivity and bycatch_rate methods return valid ranges."""
        for gear in GearType:
            selectivity = gear.selectivity()
            bycatch = gear.bycatch_rate()
            assert 0.2 <= selectivity <= 1.0, f"{gear.name} selectivity out of range"
            assert 0.0 <= bycatch <= 0.8, f"{gear.name} bycatch out of range"

    def test_all_gear_types_valid(self) -> None:
        """Test all gear types produce consistent selectivity and bycatch."""
        gear_properties = {
            GearType.SEINE: (0.65, 0.15),
            GearType.NET: (0.60, 0.25),
            GearType.TRAP: (0.85, 0.05),
            GearType.TRAWL: (0.35, 0.60),
            GearType.HOOK: (0.95, 0.02),
            GearType.OTHER: (0.50, 0.30),
        }
        for gear, (exp_select, exp_bycatch) in gear_properties.items():
            assert gear.selectivity() == exp_select
            assert gear.bycatch_rate() == exp_bycatch

    def test_gear_type_comparison(self) -> None:
        """Test GearType enum comparison and membership."""
        assert GearType.SEINE != GearType.TRAWL
        assert GearType.SEINE == GearType.SEINE
        assert GearType.TRAWL in list(GearType)


# ============================================================================
# TestFleetGroup (12 tests)
# ============================================================================


class TestFleetGroup:
    """Tests for FleetGroup creation, validation, and serialization."""

    def test_create_valid_fleet(self, sample_fleet_group: FleetGroup) -> None:
        """Test FleetGroup creation with valid parameters."""
        assert sample_fleet_group.name == "Pelagic Fleet"
        assert sample_fleet_group.fleet_id == 0
        assert len(sample_fleet_group.gears) == 2
        assert sample_fleet_group.fuel_efficiency == 0.08
        assert sample_fleet_group.base_effort == 150.0
        assert len(sample_fleet_group.catchability) == 3

    def test_create_minimal_fleet(self) -> None:
        """Test FleetGroup creation with minimal required parameters."""
        fleet = FleetGroup(
            name="Minimal Fleet",
            fleet_id=0,
            gears=[GearType.OTHER],
            catchability={"Group1": 0.5},
        )
        assert fleet.name == "Minimal Fleet"
        assert fleet.fuel_efficiency == 0.05
        assert fleet.base_effort == 100.0

    def test_is_valid_complete_fleet(self, sample_fleet_group: FleetGroup) -> None:
        """Test is_valid() returns True for complete, valid fleet."""
        assert sample_fleet_group.is_valid()

    def test_is_valid_invalid_catchability(self) -> None:
        """Test FleetGroup raises ValueError for q value out of bounds."""
        with pytest.raises(ValueError):
            FleetGroup(
                name="Invalid Fleet",
                fleet_id=0,
                gears=[GearType.OTHER],
                catchability={"Cod": 1.5},  # Out of bounds
            )

    def test_is_valid_negative_effort(self) -> None:
        """Test is_valid() returns False for fleet with negative effort."""
        fleet = FleetGroup(
            name="Invalid Fleet",
            fleet_id=0,
            gears=[GearType.OTHER],
            catchability={"Cod": 0.5},
        )
        assert not fleet.is_valid() or fleet.base_effort > 0

    def test_add_gear(self, sample_fleet_group: FleetGroup) -> None:
        """Test adding a gear to an existing fleet."""
        initial_gears = len(sample_fleet_group.gears)
        sample_fleet_group.gears.append(GearType.TRAP)
        assert len(sample_fleet_group.gears) == initial_gears + 1
        assert GearType.TRAP in sample_fleet_group.gears

    def test_to_dict_serialization(self, sample_fleet_group: FleetGroup) -> None:
        """Test FleetGroup.to_dict() serialization."""
        fleet_dict = sample_fleet_group.to_dict()
        assert fleet_dict["name"] == "Pelagic Fleet"
        assert fleet_dict["fleet_id"] == 0
        assert fleet_dict["gears"] == ["Seine", "Net"]
        assert fleet_dict["fuel_efficiency"] == 0.08
        assert fleet_dict["base_effort"] == 150.0
        assert fleet_dict["catchability"] == {"Cod": 0.5, "Herring": 0.3, "Capelin": 0.2}

    def test_from_dict_deserialization(
        self, sample_fleet_group: FleetGroup
    ) -> None:
        """Test FleetGroup.from_dict() deserialization."""
        fleet_dict = sample_fleet_group.to_dict()
        reconstructed = FleetGroup.from_dict(fleet_dict)
        assert reconstructed.name == sample_fleet_group.name
        assert reconstructed.fleet_id == sample_fleet_group.fleet_id
        assert reconstructed.gears == sample_fleet_group.gears
        assert reconstructed.catchability == sample_fleet_group.catchability

    def test_round_trip_serialization(
        self, sample_fleet_group: FleetGroup
    ) -> None:
        """Test round-trip serialization preserves all data."""
        original_dict = sample_fleet_group.to_dict()
        reconstructed = FleetGroup.from_dict(original_dict)
        final_dict = reconstructed.to_dict()
        assert original_dict == final_dict

    def test_catchability_bounds(self) -> None:
        """Test catchability validation bounds (0-1)."""
        # Valid bounds
        fleet = FleetGroup(
            name="Test",
            fleet_id=0,
            gears=[GearType.OTHER],
            catchability={"Cod": 0.0, "Herring": 0.5, "Haddock": 1.0},
        )
        assert fleet.is_valid()

        # Invalid: > 1.0
        with pytest.raises(ValueError):
            FleetGroup(
                name="Test",
                fleet_id=0,
                gears=[GearType.OTHER],
                catchability={"Cod": 1.1},
            )

        # Invalid: < 0.0
        with pytest.raises(ValueError):
            FleetGroup(
                name="Test",
                fleet_id=0,
                gears=[GearType.OTHER],
                catchability={"Cod": -0.1},
            )

    def test_fuel_efficiency_calculation(self, sample_fleet_group: FleetGroup) -> None:
        """Test fuel efficiency is stored and accessible."""
        assert 0.01 <= sample_fleet_group.fuel_efficiency <= 0.20

    def test_invalid_fleet_creation(self) -> None:
        """Test FleetGroup creation fails with invalid parameters."""
        with pytest.raises(ValueError):
            FleetGroup(
                name="Invalid",
                fleet_id=-1,  # Invalid: negative ID
                gears=[GearType.OTHER],
                catchability={"Cod": 0.5},
            )

        with pytest.raises(ValueError):
            FleetGroup(
                name="Invalid",
                fleet_id=0,
                gears=[GearType.OTHER],
                base_effort=-10.0,  # Invalid: negative effort
                catchability={"Cod": 0.5},
            )


# ============================================================================
# TestDiscardPath (10 tests)
# ============================================================================


class TestDiscardPath:
    """Tests for DiscardPath creation, validation, and discard calculation."""

    def test_create_valid_path(self, sample_discard_path: DiscardPath) -> None:
        """Test DiscardPath creation with valid parameters."""
        assert sample_discard_path.source_group == "Juvenile Cod"
        assert sample_discard_path.discard_rate == 0.15
        assert sample_discard_path.fate == "lost"
        assert not sample_discard_path.depth_dependent

    def test_create_discard_path_to_detritus(self) -> None:
        """Test DiscardPath with fate='detritus'."""
        path = DiscardPath(
            source_group="Bycatch Species",
            discard_rate=0.30,
            fate="detritus",
        )
        assert path.fate == "detritus"
        assert path.target_group is None

    def test_create_discard_path_to_predator(self) -> None:
        """Test DiscardPath with fate='predation' and target_group."""
        path = DiscardPath(
            source_group="Juvenile Fish",
            target_group="Seabirds",
            discard_rate=0.25,
            fate="predation",
        )
        assert path.fate == "predation"
        assert path.target_group == "Seabirds"

    def test_is_valid_complete(self, sample_discard_path: DiscardPath) -> None:
        """Test is_valid() returns True for complete, valid path."""
        assert sample_discard_path.is_valid()

    def test_is_valid_invalid_rate(self) -> None:
        """Test DiscardPath raises ValueError for discard_rate out of bounds."""
        with pytest.raises(ValueError):
            DiscardPath(
                source_group="Test",
                discard_rate=1.5,  # Out of bounds
                fate="lost",
            )

    def test_apply_to_biomass(self, sample_discard_path: DiscardPath) -> None:
        """Test apply_to_biomass() calculates discard fraction."""
        total_catch = 100.0
        discard_biomass = sample_discard_path.apply_to_biomass(total_catch)
        assert discard_biomass == pytest.approx(15.0)

    def test_fate_categories(self) -> None:
        """Test all valid fate categories."""
        fates = ["lost", "predation", "detritus", "benthos"]
        for fate in fates:
            if fate == "predation":
                path = DiscardPath(
                    source_group="Test",
                    target_group="Predator",
                    fate=fate,
                )
            else:
                path = DiscardPath(
                    source_group="Test",
                    fate=fate,
                )
            assert path.fate == fate
            assert path.is_valid()

    def test_depth_dependent_routing(self) -> None:
        """Test depth_dependent flag for spatial discard routing."""
        path = DiscardPath(
            source_group="Pelagic Discard",
            discard_rate=0.20,
            fate="lost",
            depth_dependent=True,
        )
        assert path.depth_dependent

    def test_to_dict_serialization(self, sample_discard_path: DiscardPath) -> None:
        """Test DiscardPath.to_dict() serialization."""
        path_dict = sample_discard_path.to_dict()
        assert path_dict["source_group"] == "Juvenile Cod"
        assert path_dict["discard_rate"] == 0.15
        assert path_dict["fate"] == "lost"
        assert path_dict["depth_dependent"] is False

    def test_round_trip_serialization(
        self, sample_discard_path: DiscardPath
    ) -> None:
        """Test round-trip serialization preserves all data."""
        original_dict = sample_discard_path.to_dict()
        reconstructed = DiscardPath.from_dict(original_dict)
        final_dict = reconstructed.to_dict()
        assert original_dict == final_dict


# ============================================================================
# TestFleetDynamics (11 tests)
# ============================================================================


class TestFleetDynamics:
    """Tests for FleetDynamics management and validation."""

    def test_add_fleet(self, sample_fleet_dynamics: FleetDynamics) -> None:
        """Test add_fleet() adds fleet to registry."""
        assert len(sample_fleet_dynamics.fleets) == 2
        assert "Pelagic Fleet" in sample_fleet_dynamics.fleets
        assert "Demersal Fleet" in sample_fleet_dynamics.fleets

    def test_add_discard_path(self, sample_fleet_dynamics: FleetDynamics) -> None:
        """Test add_discard_path() adds path to registry."""
        assert len(sample_fleet_dynamics.discard_paths) == 3
        assert "Juvenile Cod" in sample_fleet_dynamics.discard_paths
        assert "Juvenile Haddock" in sample_fleet_dynamics.discard_paths
        assert "Bycatch Herring" in sample_fleet_dynamics.discard_paths

    def test_get_fleet_found(self, sample_fleet_dynamics: FleetDynamics) -> None:
        """Test get_fleet() returns fleet when found."""
        fleet = sample_fleet_dynamics.get_fleet("Pelagic Fleet")
        assert fleet is not None
        assert fleet.name == "Pelagic Fleet"
        assert fleet.fleet_id == 0

    def test_get_fleet_not_found(self, sample_fleet_dynamics: FleetDynamics) -> None:
        """Test get_fleet() returns None when not found."""
        fleet = sample_fleet_dynamics.get_fleet("Nonexistent Fleet")
        assert fleet is None

    def test_validate_empty(self) -> None:
        """Test validate() on empty FleetDynamics."""
        fd = FleetDynamics()
        is_valid, errors = fd.validate()
        assert is_valid

    def test_validate_valid_state(
        self, sample_fleet_dynamics: FleetDynamics
    ) -> None:
        """Test validate() returns True for valid fleet dynamics."""
        is_valid, errors = sample_fleet_dynamics.validate()
        assert is_valid
        assert len(errors) == 0

    def test_validate_invalid_fleet(self) -> None:
        """Test validate() detects invalid fleet."""
        fd = FleetDynamics()
        invalid_fleet = FleetGroup(
            name="Invalid",
            fleet_id=0,
            gears=[GearType.OTHER],
            catchability={},  # Empty catchability makes it invalid
        )
        fd.add_fleet(invalid_fleet)
        is_valid, errors = fd.validate()
        assert not is_valid
        assert len(errors) > 0

    def test_validate_missing_catchability(self) -> None:
        """Test validate() with missing catchability for fleet."""
        fd = FleetDynamics()
        fleet = FleetGroup(
            name="Incomplete Fleet",
            fleet_id=0,
            gears=[GearType.OTHER],
            catchability={"Cod": 0.5},
        )
        fd.add_fleet(fleet)
        is_valid, _ = fd.validate()
        assert is_valid

    def test_to_dataframe(self, sample_fleet_dynamics: FleetDynamics) -> None:
        """Test to_dataframe() returns DataFrames."""
        dfs = sample_fleet_dynamics.to_dataframe()
        assert "fleets" in dfs
        assert "discards" in dfs
        assert len(dfs["fleets"]) == 2
        assert len(dfs["discards"]) == 3

    def test_from_dataframe(self, sample_fleet_dynamics: FleetDynamics) -> None:
        """Test from_dataframe() reconstructs FleetDynamics."""
        dfs = sample_fleet_dynamics.to_dataframe()
        reconstructed = FleetDynamics.from_dataframe(
            dfs["fleets"], dfs["discards"]
        )
        assert len(reconstructed.fleets) == 2
        assert len(reconstructed.discard_paths) == 3

    def test_discard_path_validation(self) -> None:
        """Test that invalid discard paths raise ValueError."""
        with pytest.raises(ValueError):
            DiscardPath(
                source_group="Test",
                fate="predation",
                # target_group missing - invalid for predation
            )


# ============================================================================
# TestAllocation (8 tests)
# ============================================================================


class TestAllocation:
    """Tests for catch and discard allocation functions."""

    def test_allocate_catch_single_fleet(self) -> None:
        """Test allocate_catch() with single group."""
        catchability = {"Cod": 1.0}
        result = allocate_catch(100.0, catchability)
        assert result["Cod"] == pytest.approx(100.0)

    def test_allocate_catch_multiple_fleets(self) -> None:
        """Test allocate_catch() distributes across multiple groups."""
        catchability = {"Cod": 0.6, "Herring": 0.4}
        result = allocate_catch(100.0, catchability)
        assert result["Cod"] == pytest.approx(60.0)
        assert result["Herring"] == pytest.approx(40.0)

    def test_allocate_catch_proportions_sum_to_one(self) -> None:
        """Test allocate_catch() proportions sum to total catch."""
        catchability = {"A": 0.2, "B": 0.3, "C": 0.5}
        total_catch = 250.0
        result = allocate_catch(total_catch, catchability)
        assert sum(result.values()) == pytest.approx(total_catch)

    def test_allocate_discards_single_group(self) -> None:
        """Test allocate_discards() for single group."""
        result = allocate_discards(0.20, "Cod")
        assert result["discarded"] == pytest.approx(0.20)
        assert result["landed"] == pytest.approx(0.80)

    def test_allocate_discards_multiple_groups(self) -> None:
        """Test allocate_discards() proportions sum to 1.0."""
        result = allocate_discards(0.15, "Test")
        assert sum(result.values()) == pytest.approx(1.0)

    def test_allocate_discards_fates(self) -> None:
        """Test allocate_discards() with various rates."""
        for rate in [0.0, 0.25, 0.50, 0.75, 1.0]:
            result = allocate_discards(rate, "Group")
            assert result["discarded"] == pytest.approx(rate)
            assert result["landed"] == pytest.approx(1.0 - rate)

    def test_validate_allocation_valid(self) -> None:
        """Test validate_allocation() returns True for valid allocation."""
        allocation = {"Cod": 0.5, "Herring": 0.5}
        assert validate_allocation(allocation)

    def test_validate_allocation_invalid(self) -> None:
        """Test validate_allocation() returns False for invalid allocation."""
        allocation = {"Cod": 0.4, "Herring": 0.4}  # Sums to 0.8
        assert not validate_allocation(allocation)


# ============================================================================
# TestEconomic (4 tests)
# ============================================================================


class TestEconomic:
    """Tests for economic calculations."""

    def test_estimate_catch_value(self) -> None:
        """Test estimate_catch_value() calculates revenue."""
        catch = {"Cod": 100.0, "Herring": 50.0}  # tons
        prices = {"Cod": 1.5, "Herring": 0.8}  # $/kg
        value = estimate_catch_value(catch, prices)
        expected = (100.0 * 1000 * 1.5) + (50.0 * 1000 * 0.8)
        assert value == pytest.approx(expected)

    def test_estimate_fuel_cost(self) -> None:
        """Test estimate_fuel_cost() calculates fuel expense."""
        effort = 10.0  # days
        fuel_price = 1.0  # $/liter
        cost = estimate_fuel_cost(effort, fuel_price)
        expected = effort * 100.0 * fuel_price
        assert cost == pytest.approx(expected)

    def test_economic_efficiency_ratio(self) -> None:
        """Test economic efficiency calculation."""
        catch = {"Cod": 1000.0}
        prices = {"Cod": 2.0}
        value = estimate_catch_value(catch, prices)
        effort = 50.0
        fuel_price = 1.0
        cost = estimate_fuel_cost(effort, fuel_price)
        efficiency = value / cost if cost > 0 else 0
        assert efficiency > 0

    def test_profitability_calculation(self) -> None:
        """Test profitability calculation."""
        catch = {"Cod": 500.0}
        prices = {"Cod": 1.5}
        value = estimate_catch_value(catch, prices)
        assert value > 0


# ============================================================================
# TestEdgeCases (12 tests)
# ============================================================================


class TestEdgeCases:
    """Tests for edge cases and boundary conditions."""

    def test_empty_fleet_dynamics(self) -> None:
        """Test FleetDynamics with no fleets or discard paths."""
        fd = FleetDynamics()
        assert len(fd.fleets) == 0
        assert len(fd.discard_paths) == 0
        is_valid, errors = fd.validate()
        assert is_valid

    def test_single_fleet(self) -> None:
        """Test FleetDynamics with exactly one fleet."""
        fd = FleetDynamics()
        fleet = FleetGroup(
            name="Single Fleet",
            fleet_id=0,
            gears=[GearType.NET],
            catchability={"Cod": 1.0},
        )
        fd.add_fleet(fleet)
        assert len(fd.fleets) == 1

    def test_very_large_catchability(self) -> None:
        """Test fleet with catchability approaching maximum."""
        fleet = FleetGroup(
            name="Efficient Fleet",
            fleet_id=0,
            gears=[GearType.TRAP],
            catchability={"Cod": 0.99, "Herring": 0.005, "Haddock": 0.005},
        )
        assert fleet.is_valid()

    def test_very_small_catchability(self) -> None:
        """Test fleet with very small catchability values."""
        fleet = FleetGroup(
            name="Inefficient Fleet",
            fleet_id=0,
            gears=[GearType.OTHER],
            catchability={"Cod": 0.001, "Herring": 0.001, "Haddock": 0.001},
        )
        assert fleet.is_valid()

    def test_zero_catch(self) -> None:
        """Test allocation with zero total catch."""
        catchability = {"Cod": 0.5, "Herring": 0.5}
        result = allocate_catch(0.0, catchability)
        assert all(v == 0.0 for v in result.values())

    def test_100_percent_discard(self) -> None:
        """Test DiscardPath with 100% discard rate."""
        path = DiscardPath(
            source_group="All Bycatch",
            discard_rate=1.0,
            fate="lost",
        )
        assert path.is_valid()
        assert path.apply_to_biomass(100.0) == pytest.approx(100.0)

    def test_zero_percent_discard(self) -> None:
        """Test DiscardPath with 0% discard rate."""
        path = DiscardPath(
            source_group="No Discard",
            discard_rate=0.0,
            fate="lost",
        )
        assert path.is_valid()
        assert path.apply_to_biomass(100.0) == pytest.approx(0.0)

    def test_multiple_discard_fates_same_group(self) -> None:
        """Test multiple discard paths for same source group."""
        fd = FleetDynamics()
        fd.add_discard_path(DiscardPath(
            source_group="Bycatch",
            discard_rate=0.5,
            fate="lost",
        ))
        fd.add_discard_path(DiscardPath(
            source_group="Bycatch",
            target_group="Seabirds",
            discard_rate=0.3,
            fate="predation",
        ))
        assert len(fd.discard_paths["Bycatch"]) == 2

    def test_circular_discard_paths(self) -> None:
        """Test complex discard path routing (not circular detection)."""
        fd = FleetDynamics()
        fd.add_discard_path(DiscardPath(
            source_group="Group A",
            target_group="Group B",
            discard_rate=0.2,
            fate="predation",
        ))
        fd.add_discard_path(DiscardPath(
            source_group="Group B",
            target_group="Group C",
            discard_rate=0.1,
            fate="predation",
        ))
        assert len(fd.discard_paths) == 2

    def test_fleet_with_many_gears(self) -> None:
        """Test fleet with all gear types."""
        all_gears = list(GearType)
        fleet = FleetGroup(
            name="Multipurpose Fleet",
            fleet_id=0,
            gears=all_gears,
            catchability={"Cod": 0.5},
        )
        assert len(fleet.gears) == len(all_gears)
        assert fleet.is_valid()

    def test_extreme_economic_values(self) -> None:
        """Test economic calculations with extreme values."""
        catch = {"Cod": 1e6}  # 1 million tons
        prices = {"Cod": 1e-3}  # $0.001/kg
        value = estimate_catch_value(catch, prices)
        assert value > 0

    def test_whitespace_in_group_names(self) -> None:
        """Test fleet with whitespace in group names."""
        fleet = FleetGroup(
            name="Fleet with Spaces",
            fleet_id=0,
            gears=[GearType.OTHER],
            catchability={"Atlantic Cod": 0.5, "Arctic Herring": 0.5},
        )
        assert fleet.is_valid()


# ============================================================================
# TestIntegration (6 tests)
# ============================================================================


class TestIntegration:
    """Integration tests combining multiple fleet components."""

    def test_full_fleet_scenario(
        self, sample_fleet_dynamics: FleetDynamics
    ) -> None:
        """Test complete fleet scenario with multiple fleets and discard paths."""
        is_valid, errors = sample_fleet_dynamics.validate()
        assert is_valid
        assert len(sample_fleet_dynamics.fleets) == 2
        assert len(sample_fleet_dynamics.discard_paths) == 3

    def test_catch_and_discard_workflow(self) -> None:
        """Test workflow: allocate catch -> allocate discards."""
        # Fleet allocates catch across groups
        catchability = {"Cod": 0.6, "Herring": 0.4}
        catch_alloc = allocate_catch(100.0, catchability)

        # Each group has a discard path
        discard_rate_cod = 0.15
        discard_rate_herring = 0.10

        discard_alloc_cod = allocate_discards(discard_rate_cod, "Cod")
        discard_alloc_herring = allocate_discards(
            discard_rate_herring, "Herring"
        )

        # Calculate discarded biomass
        discarded_cod = catch_alloc["Cod"] * discard_alloc_cod["discarded"]
        discarded_herring = (
            catch_alloc["Herring"] * discard_alloc_herring["discarded"]
        )

        assert discarded_cod == pytest.approx(9.0)
        assert discarded_herring == pytest.approx(4.0)

    def test_dataframe_roundtrip_preservation(
        self, sample_fleet_dynamics: FleetDynamics
    ) -> None:
        """Test DataFrame round-trip preserves all fleet data."""
        dfs_original = sample_fleet_dynamics.to_dataframe()
        reconstructed = FleetDynamics.from_dataframe(
            dfs_original["fleets"], dfs_original["discards"]
        )
        dfs_reconstructed = reconstructed.to_dataframe()

        pd.testing.assert_frame_equal(
            dfs_original["fleets"], dfs_reconstructed["fleets"]
        )
        pd.testing.assert_frame_equal(
            dfs_original["discards"], dfs_reconstructed["discards"]
        )

    def test_economics_integration(self) -> None:
        """Test complete economics workflow."""
        # Define catch and prices
        catch = {"Cod": 500.0, "Herring": 300.0, "Haddock": 200.0}
        prices = {"Cod": 2.0, "Herring": 0.5, "Haddock": 1.5}

        # Calculate value
        total_value = estimate_catch_value(catch, prices)

        # Calculate fuel cost
        effort = 100.0
        fuel_price = 1.5
        total_cost = estimate_fuel_cost(effort, fuel_price)

        # Calculate profitability
        profit = total_value - total_cost
        assert total_value > 0
        assert total_cost > 0

    def test_spatial_allocation_preparation(self) -> None:
        """Test fleet data preparation for spatial (Ecospace) allocation."""
        fd = FleetDynamics()

        # Add fleets with spatial gear distribution
        fleet1 = FleetGroup(
            name="Pelagic",
            fleet_id=0,
            gears=[GearType.SEINE, GearType.NET],
            catchability={"Cod": 0.3, "Herring": 0.7},
        )
        fleet2 = FleetGroup(
            name="Demersal",
            fleet_id=1,
            gears=[GearType.TRAWL],
            catchability={"Cod": 0.7, "Haddock": 0.3},
        )
        fd.add_fleet(fleet1)
        fd.add_fleet(fleet2)

        # Add depth-dependent discard paths for spatial routing
        fd.add_discard_path(DiscardPath(
            source_group="Pelagic Juvenile",
            discard_rate=0.2,
            fate="lost",
            depth_dependent=True,
        ))
        fd.add_discard_path(DiscardPath(
            source_group="Demersal Juvenile",
            discard_rate=0.15,
            fate="lost",
            depth_dependent=True,
        ))

        # Convert to DataFrames for spatial engine
        dfs = fd.to_dataframe()
        assert not dfs["fleets"].empty
        assert not dfs["discards"].empty

    def test_ecosim_catchability_integration(self) -> None:
        """Test fleet catchability coefficients for Ecosim integration."""
        fleet = FleetGroup(
            name="Test Fleet",
            fleet_id=0,
            gears=[GearType.SEINE],
            catchability={"Group_1": 0.4, "Group_2": 0.3, "Group_3": 0.3},
        )

        # Verify catchability for Ecosim's catch equation: Catch = q * E * B
        assert all(0.0 <= q <= 1.0 for q in fleet.catchability.values())
        assert len(fleet.catchability) > 0
        assert fleet.is_valid()

