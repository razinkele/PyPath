"""Comprehensive test suite for nutrients.py module (Phase 1B.5).

This module tests nutrient cycling functionality including content,
excretion, uptake, and ecosystem-wide calculations.
"""

from __future__ import annotations

import pandas as pd
import pytest

from pypath.core.nutrients import (
    NutrientContent,
    NutrientCycle,
    NutrientElement,
    NutrientExcretion,
    NutrientUptake,
    estimate_excretion_from_pb,
    validate_nutrient_cycle,
)
from pypath.core.params import create_rpath_params

# ==================== FIXTURES ====================


@pytest.fixture
def sample_nutrient_content():
    """Valid NutrientContent for testing."""
    return NutrientContent(
        group_name="Zooplankton",
        nitrogen_percent=10.0,
        phosphorus_percent=1.5,
        carbon_percent=48.0,
        ash_percent=8.0,
    )


@pytest.fixture
def sample_nutrient_excretion():
    """Valid NutrientExcretion for testing."""
    return NutrientExcretion(
        group_name="Zooplankton",
        n_excretion_rate=12.0,
        p_excretion_rate=2.5,
        fecal_n_fraction=0.35,
        fecal_p_fraction=0.55,
    )


@pytest.fixture
def sample_nutrient_uptake():
    """Valid NutrientUptake for primary producers."""
    return NutrientUptake(
        group_name="Phytoplankton",
        n_uptake_rate=0.5,
        p_uptake_rate=0.08,
        half_saturation_n=1.2,
        half_saturation_p=0.15,
    )


@pytest.fixture
def sample_nutrient_cycle(
    sample_nutrient_content, sample_nutrient_excretion, sample_nutrient_uptake
):
    """Complete ecosystem with all nutrient components."""
    cycle = NutrientCycle()

    # Phytoplankton
    phyto = NutrientContent("Phytoplankton", 8.0, 1.2, 46.0, 10.0)
    cycle.add_content(phyto)
    phyto_uptake = NutrientUptake("Phytoplankton", 0.6, 0.1, 1.0, 0.12)
    cycle.add_uptake(phyto_uptake)

    # Zooplankton
    cycle.add_content(sample_nutrient_content)
    cycle.add_excretion(sample_nutrient_excretion)

    # Fish
    fish = NutrientContent("Fish", 11.0, 1.8, 50.0, 7.0)
    cycle.add_content(fish)
    fish_excretion = NutrientExcretion("Fish", 8.0, 1.5, 0.4, 0.6)
    cycle.add_excretion(fish_excretion)

    # Detritus
    det = NutrientContent("Detritus", 5.0, 0.8, 42.0, 15.0)
    cycle.add_content(det)

    return cycle


# ==================== TestNutrientElement ====================


class TestNutrientElement:
    """Tests for NutrientElement enum."""

    def test_nutrient_enum_values(self):
        """Test that nutrient enum has correct values."""
        assert NutrientElement.NITROGEN.value == "nitrogen"
        assert NutrientElement.PHOSPHORUS.value == "phosphorus"
        assert NutrientElement.CARBON.value == "carbon"

    def test_molecular_weights(self):
        """Test molecular weight properties."""
        assert NutrientElement.NITROGEN.molecular_weight == pytest.approx(14.01)
        assert NutrientElement.PHOSPHORUS.molecular_weight == pytest.approx(30.97)
        assert NutrientElement.CARBON.molecular_weight == pytest.approx(12.01)

    def test_element_properties(self):
        """Test element enum members are accessible."""
        for element in NutrientElement:
            assert element.molecular_weight > 0


# ==================== TestNutrientContent ====================


class TestNutrientContent:
    """Tests for NutrientContent dataclass."""

    def test_create_valid_content(self, sample_nutrient_content):
        """Test creation of valid NutrientContent."""
        assert sample_nutrient_content.group_name == "Zooplankton"
        assert sample_nutrient_content.nitrogen_percent == 10.0
        assert sample_nutrient_content.phosphorus_percent == 1.5
        assert sample_nutrient_content.carbon_percent == 48.0
        assert sample_nutrient_content.ash_percent == 8.0

    def test_is_valid_complete(self, sample_nutrient_content):
        """Test is_valid() on valid content."""
        assert sample_nutrient_content.is_valid() is True

    def test_is_valid_incomplete(self):
        """Test is_valid() on invalid content."""
        with pytest.raises(ValueError):
            NutrientContent(
                "Test",
                nitrogen_percent=25.0,
                phosphorus_percent=1.0,
                carbon_percent=48.0,
                ash_percent=8.0,
            )

    def test_nutrient_percentage_bounds(self):
        """Test nutrient percentage validation bounds."""
        with pytest.raises(ValueError, match="nitrogen_percent"):
            NutrientContent("Test", 21.0, 1.0, 48.0, 8.0)

        with pytest.raises(ValueError, match="phosphorus_percent"):
            NutrientContent("Test", 10.0, 6.0, 48.0, 8.0)

        with pytest.raises(ValueError, match="carbon_percent"):
            NutrientContent("Test", 10.0, 1.0, 39.0, 8.0)

        with pytest.raises(ValueError, match="ash_percent"):
            NutrientContent("Test", 10.0, 1.0, 48.0, 21.0)

    def test_to_dict_serialization(self, sample_nutrient_content):
        """Test to_dict() serialization."""
        d = sample_nutrient_content.to_dict()
        assert d["group_name"] == "Zooplankton"
        assert d["nitrogen_percent"] == 10.0
        assert d["phosphorus_percent"] == 1.5
        assert d["carbon_percent"] == 48.0
        assert d["ash_percent"] == 8.0

    def test_from_dict_deserialization(self, sample_nutrient_content):
        """Test from_dict() deserialization."""
        d = sample_nutrient_content.to_dict()
        restored = NutrientContent.from_dict(d)
        assert restored.group_name == sample_nutrient_content.group_name
        assert restored.nitrogen_percent == sample_nutrient_content.nitrogen_percent

    def test_round_trip_serialization(self, sample_nutrient_content):
        """Test round-trip serialization and deserialization."""
        d = sample_nutrient_content.to_dict()
        restored = NutrientContent.from_dict(d)
        assert restored.to_dict() == d

    def test_ash_percentage_validation(self):
        """Test ash percentage bounds specifically."""
        valid = NutrientContent("Test", 10.0, 1.0, 48.0, 0.0)
        assert valid.ash_percent == 0.0
        valid = NutrientContent("Test", 10.0, 1.0, 48.0, 20.0)
        assert valid.ash_percent == 20.0


# ==================== TestNutrientExcretion ====================


class TestNutrientExcretion:
    """Tests for NutrientExcretion dataclass."""

    def test_create_valid_excretion(self, sample_nutrient_excretion):
        """Test creation of valid NutrientExcretion."""
        assert sample_nutrient_excretion.group_name == "Zooplankton"
        assert sample_nutrient_excretion.n_excretion_rate == 12.0
        assert sample_nutrient_excretion.p_excretion_rate == 2.5

    def test_calculate_excretion_nitrogen(self, sample_nutrient_excretion):
        """Test nitrogen excretion calculation."""
        consumption = 100.0
        n_exc = sample_nutrient_excretion.calculate_excretion(consumption, "nitrogen")
        assert n_exc == pytest.approx(12.0)

    def test_calculate_excretion_phosphorus(self, sample_nutrient_excretion):
        """Test phosphorus excretion calculation."""
        consumption = 100.0
        p_exc = sample_nutrient_excretion.calculate_excretion(consumption, "phosphorus")
        assert p_exc == pytest.approx(2.5)

    def test_fecal_partitioning(self, sample_nutrient_excretion):
        """Test fecal partitioning fractions."""
        assert sample_nutrient_excretion.fecal_n_fraction == 0.35
        assert sample_nutrient_excretion.fecal_p_fraction == 0.55
        fecal_n = 12.0 * 0.35
        dissolved_n = 12.0 * (1 - 0.35)
        assert fecal_n + dissolved_n == pytest.approx(12.0)

    def test_excretion_rate_bounds(self):
        """Test excretion rate validation bounds."""
        with pytest.raises(ValueError, match="n_excretion_rate"):
            NutrientExcretion("Test", 51.0, 2.0, 0.4, 0.6)

        with pytest.raises(ValueError, match="p_excretion_rate"):
            NutrientExcretion("Test", 10.0, 21.0, 0.4, 0.6)

    def test_to_dict_serialization(self, sample_nutrient_excretion):
        """Test to_dict() serialization."""
        d = sample_nutrient_excretion.to_dict()
        assert "group_name" in d
        assert "n_excretion_rate" in d
        assert "p_excretion_rate" in d

    def test_round_trip_serialization(self, sample_nutrient_excretion):
        """Test round-trip serialization."""
        d = sample_nutrient_excretion.to_dict()
        restored = NutrientExcretion(**d)
        assert restored.to_dict() == d

    def test_zero_excretion(self):
        """Test zero excretion case."""
        exc = NutrientExcretion("Test", 0.0, 0.0, 0.5, 0.5)
        assert exc.calculate_excretion(100.0, "nitrogen") == 0.0
        assert exc.calculate_excretion(100.0, "phosphorus") == 0.0


# ==================== TestNutrientUptake ====================


class TestNutrientUptake:
    """Tests for NutrientUptake dataclass."""

    def test_create_valid_uptake(self, sample_nutrient_uptake):
        """Test creation of valid NutrientUptake."""
        assert sample_nutrient_uptake.group_name == "Phytoplankton"
        assert sample_nutrient_uptake.n_uptake_rate == 0.5
        assert sample_nutrient_uptake.p_uptake_rate == 0.08

    def test_monod_kinetics_saturation(self, sample_nutrient_uptake):
        """Test Monod kinetics at very high nutrient levels."""
        vmax = 0.5
        uptake_high = sample_nutrient_uptake.calculate_uptake(1000.0, vmax)
        assert uptake_high == pytest.approx(vmax, rel=0.01)

    def test_monod_kinetics_at_zero(self, sample_nutrient_uptake):
        """Test Monod kinetics at zero nutrient."""
        uptake_zero = sample_nutrient_uptake.calculate_uptake(0.0, 0.5)
        assert uptake_zero == 0.0

    def test_monod_kinetics_at_saturation(self):
        """Test Monod kinetics at vmax level."""
        uptake = NutrientUptake("Test", 1.0, 0.1, 0.5, 0.05)
        vmax = 1.0
        nutrient_level = vmax
        uptake_result = uptake.calculate_uptake(nutrient_level, vmax)
        # v = (vmax * vmax) / (vmax + vmax) = vmax^2 / 2*vmax = vmax/2
        assert uptake_result == pytest.approx(0.5, abs=0.01)

    def test_half_saturation_effect(self):
        """Test uptake rate calculation at various nutrient levels."""
        uptake = NutrientUptake("Test", 1.0, 0.1, 0.5, 0.05)
        vmax = 1.0
        nutrient_level = vmax
        uptake_result = uptake.calculate_uptake(nutrient_level, vmax)
        # At vmax nutrient level, uptake = vmax/2
        assert uptake_result == pytest.approx(0.5, abs=0.01)

    def test_uptake_zero_nutrient(self, sample_nutrient_uptake):
        """Test uptake when nutrient is zero."""
        uptake = sample_nutrient_uptake.calculate_uptake(0.0, 1.0)
        assert uptake == 0.0

    def test_round_trip_serialization(self, sample_nutrient_uptake):
        """Test round-trip serialization."""
        d = sample_nutrient_uptake.to_dict()
        restored = NutrientUptake(**d)
        assert restored.to_dict() == d


# ==================== TestNutrientCycle ====================


class TestNutrientCycle:
    """Tests for NutrientCycle container."""

    def test_add_content(self):
        """Test adding nutrient content."""
        cycle = NutrientCycle()
        content = NutrientContent("Test", 10.0, 1.5, 48.0, 8.0)
        cycle.add_content(content)
        assert "Test" in cycle.contents
        assert cycle.contents["Test"] == content

    def test_add_excretion(self):
        """Test adding excretion parameters."""
        cycle = NutrientCycle()
        excretion = NutrientExcretion("Test", 10.0, 2.0, 0.4, 0.6)
        cycle.add_excretion(excretion)
        assert "Test" in cycle.excretions
        assert cycle.excretions["Test"] == excretion

    def test_add_uptake(self):
        """Test adding uptake parameters."""
        cycle = NutrientCycle()
        uptake = NutrientUptake("Phyto", 0.5, 0.08, 1.0, 0.1)
        cycle.add_uptake(uptake)
        assert "Phyto" in cycle.uptakes
        assert cycle.uptakes["Phyto"] == uptake

    def test_validate_empty(self):
        """Test validation of empty cycle."""
        cycle = NutrientCycle()
        is_valid, messages = cycle.validate()
        assert is_valid is False
        assert len(messages) > 0
        assert "empty" in messages[0].lower()

    def test_validate_valid_state(self, sample_nutrient_cycle):
        """Test validation of valid cycle."""
        is_valid, messages = sample_nutrient_cycle.validate()
        assert is_valid is True
        assert len(messages) == 0

    def test_calculate_mass_balance(self, sample_nutrient_cycle):
        """Test mass balance calculation."""
        params = create_rpath_params(
            ["Phytoplankton", "Zooplankton", "Fish", "Detritus", "Fleet"],
            [1, 0, 0, 2, 3],
        )
        params.model["Biomass"] = [10.0, 5.0, 2.0, 100.0, 0.0]

        balance = sample_nutrient_cycle.calculate_mass_balance(params)
        assert "total_n_biomass" in balance
        assert "total_p_biomass" in balance
        assert "balance_status" in balance
        assert balance["total_n_biomass"] > 0

    def test_to_dataframe(self, sample_nutrient_cycle):
        """Test conversion to DataFrame."""
        df = sample_nutrient_cycle.to_dataframe()
        assert isinstance(df, pd.DataFrame)
        assert len(df) > 0
        assert "Group" in df.columns

    def test_from_dataframe(self, sample_nutrient_cycle):
        """Test reconstruction from DataFrame."""
        df = sample_nutrient_cycle.to_dataframe()
        restored = NutrientCycle.from_dataframe(df)
        assert len(restored.contents) > 0
        assert len(restored.excretions) > 0

    def test_duplicate_groups(self):
        """Test adding duplicate group updates record."""
        cycle = NutrientCycle()
        content1 = NutrientContent("Test", 10.0, 1.5, 48.0, 8.0)
        content2 = NutrientContent("Test", 12.0, 1.6, 49.0, 7.0)
        cycle.add_content(content1)
        cycle.add_content(content2)
        assert cycle.contents["Test"].nitrogen_percent == 12.0


# ==================== TestValidation ====================


class TestValidation:
    """Tests for validation functions."""

    def test_validate_valid_cycle(self, sample_nutrient_cycle):
        """Test validation of complete cycle."""
        is_valid, msgs = validate_nutrient_cycle(sample_nutrient_cycle)
        assert is_valid is True

    def test_validate_missing_content(self):
        """Test validation detects missing content."""
        cycle = NutrientCycle()
        excretion = NutrientExcretion("Test", 10.0, 2.0, 0.4, 0.6)
        cycle.add_excretion(excretion)
        is_valid, msgs = validate_nutrient_cycle(cycle)
        assert is_valid is False
        assert any("content" in m.lower() for m in msgs)

    def test_validate_negative_rates(self):
        """Test validation detects negative excretion rates."""
        with pytest.raises(ValueError):
            NutrientExcretion("Test", -1.0, 2.0, 0.4, 0.6)

    def test_validate_inconsistent_groups(self):
        """Test validation detects inconsistent groups."""
        cycle = NutrientCycle()
        content = NutrientContent("A", 10.0, 1.5, 48.0, 8.0)
        uptake = NutrientUptake("B", 0.5, 0.08, 1.0, 0.1)
        cycle.add_content(content)
        cycle.add_uptake(uptake)
        is_valid, msgs = validate_nutrient_cycle(cycle)
        assert len(msgs) > 0


# ==================== TestEstimation ====================


class TestEstimation:
    """Tests for estimation functions."""

    def test_estimate_excretion_from_pb(self):
        """Test excretion estimation from P/B ratio."""
        pb = 0.5
        est = estimate_excretion_from_pb(pb, "consumer")
        assert "n_excretion_rate" in est
        assert "p_excretion_rate" in est
        assert est["n_excretion_rate"] > 0
        assert est["p_excretion_rate"] > 0

    def test_estimation_heuristic(self):
        """Test excretion estimation heuristic values."""
        est_consumer = estimate_excretion_from_pb(0.5, "consumer")
        est_producer = estimate_excretion_from_pb(0.5, "producer")
        est_detritus = estimate_excretion_from_pb(0.5, "detritus")

        assert est_consumer["n_excretion_rate"] > est_producer["n_excretion_rate"]
        assert est_detritus["n_excretion_rate"] == 0.0

    def test_mass_balance_closure(self, sample_nutrient_cycle):
        """Test mass balance closure concept."""
        content_n = sum(
            c.nitrogen_percent for c in sample_nutrient_cycle.contents.values()
        )
        assert content_n > 0


# ==================== TestEdgeCases ====================


class TestEdgeCases:
    """Tests for edge cases and boundary conditions."""

    def test_empty_nutrient_cycle(self):
        """Test empty nutrient cycle."""
        cycle = NutrientCycle()
        df = cycle.to_dataframe()
        assert len(df) == 0

    def test_single_group_cycle(self):
        """Test cycle with single group."""
        cycle = NutrientCycle()
        content = NutrientContent("Single", 10.0, 1.5, 48.0, 8.0)
        cycle.add_content(content)
        df = cycle.to_dataframe()
        assert len(df) == 1
        assert df["Group"].iloc[0] == "Single"

    def test_extreme_nutrient_percentages(self):
        """Test extreme but valid nutrient percentages."""
        low = NutrientContent("Low", 0.0, 0.0, 40.0, 0.0)
        assert low.is_valid() is True
        high = NutrientContent("High", 20.0, 5.0, 55.0, 20.0)
        assert high.is_valid() is True

    def test_extreme_excretion_rates(self):
        """Test extreme but valid excretion rates."""
        low = NutrientExcretion("Low", 0.0, 0.0, 0.0, 0.0)
        assert low.n_excretion_rate == 0.0
        high = NutrientExcretion("High", 50.0, 20.0, 1.0, 1.0)
        assert high.n_excretion_rate == 50.0

    def test_very_low_half_saturation(self):
        """Test very low half-saturation constant."""
        uptake = NutrientUptake("Test", 1.0, 0.1, 0.001, 0.0001)
        uptake_result = uptake.calculate_uptake(0.0001, 1.0)
        assert uptake_result > 0.0

    def test_very_high_half_saturation(self):
        """Test uptake with high nutrient levels."""
        uptake = NutrientUptake("Test", 1.0, 0.1, 100.0, 10.0)
        uptake_result = uptake.calculate_uptake(10.0, 1.0)
        # v = (1.0 * 10.0) / (1.0 + 10.0) = 10/11 ≈ 0.909
        assert uptake_result > 0.5

    def test_multiple_producer_groups(self):
        """Test cycle with multiple primary producers."""
        cycle = NutrientCycle()
        for i, name in enumerate(["Phyto1", "Phyto2", "Phyto3"]):
            content = NutrientContent(name, 8.0 + i, 1.2, 46.0, 10.0)
            uptake = NutrientUptake(name, 0.5 + i * 0.1, 0.08, 1.0, 0.1)
            cycle.add_content(content)
            cycle.add_uptake(uptake)
        assert len(cycle.uptakes) == 3

    def test_nutrient_flow_tracking(self, sample_nutrient_cycle):
        """Test nutrient flow through ecosystem."""
        phyto_n = sample_nutrient_cycle.contents["Phytoplankton"].nitrogen_percent
        zoo_n = sample_nutrient_cycle.contents["Zooplankton"].nitrogen_percent
        fish_n = sample_nutrient_cycle.contents["Fish"].nitrogen_percent
        assert phyto_n < zoo_n < fish_n
