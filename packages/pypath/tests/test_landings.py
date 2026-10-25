"""Comprehensive test suite for pypath.core.landings module.

Tests cover:
- Phase 1B.6 Landings system
- LandingType enum, LandingRecord dataclass, LandingSeasonality
- LandingsData container and aggregation
- Validation, serialization, and round-trip conversions
"""

import logging

import numpy as np
import pytest

from pypath.core.landings import (
    LandingRecord,
    LandingsData,
    LandingSeasonality,
    LandingType,
    aggregate_landings,
    estimate_seasonality_from_records,
    validate_landings,
)

logger = logging.getLogger(__name__)


# ============================================================================
# FIXTURES
# ============================================================================


@pytest.fixture
def sample_landing_record() -> LandingRecord:
    """Create a sample valid landing record."""
    return LandingRecord(
        group_name="Cod",
        year=2020,
        month=3,
        landing_type=LandingType.COMMERCIAL,
        catch_tons=150.5,
        value_usd=450000.0,
        fleet_id="Baltic_Demersal_Fleet",
    )


@pytest.fixture
def sample_landing_record_annual() -> LandingRecord:
    """Create a sample annual (non-seasonal) landing record."""
    return LandingRecord(
        group_name="Herring",
        year=2020,
        month=None,
        landing_type=LandingType.COMMERCIAL,
        catch_tons=1000.0,
        value_usd=200000.0,
    )


@pytest.fixture
def sample_landing_seasonality() -> LandingSeasonality:
    """Create a sample seasonality pattern."""
    proportions = np.array(
        [
            0.05,
            0.05,
            0.08,
            0.10,
            0.12,
            0.15,
            0.18,
            0.15,
            0.12,
            0.10,
            0.05,
            0.05,
        ]
    )
    proportions = proportions / np.sum(proportions)
    return LandingSeasonality(
        group_name="Cod",
        monthly_proportions=proportions,
    )


@pytest.fixture
def sample_landings_data() -> LandingsData:
    """Create a sample LandingsData container."""
    landings = LandingsData()

    # Add commercial records
    landings.add_record(
        LandingRecord(
            group_name="Cod",
            year=2020,
            month=3,
            landing_type=LandingType.COMMERCIAL,
            catch_tons=150.0,
        )
    )
    landings.add_record(
        LandingRecord(
            group_name="Cod",
            year=2020,
            month=6,
            landing_type=LandingType.COMMERCIAL,
            catch_tons=200.0,
        )
    )
    landings.add_record(
        LandingRecord(
            group_name="Herring",
            year=2020,
            month=None,
            landing_type=LandingType.COMMERCIAL,
            catch_tons=1000.0,
        )
    )

    # Add seasonality
    cod_proportions = np.array(
        [
            0.05,
            0.05,
            0.08,
            0.10,
            0.12,
            0.15,
            0.18,
            0.15,
            0.12,
            0.10,
            0.05,
            0.05,
        ]
    )
    cod_proportions = cod_proportions / np.sum(cod_proportions)
    cod_seasonality = LandingSeasonality(
        group_name="Cod",
        monthly_proportions=cod_proportions,
    )
    landings.add_seasonality(cod_seasonality)

    return landings


# ============================================================================
# TEST CLASSES
# ============================================================================


class TestLandingType:
    """Test LandingType enumeration."""

    def test_landing_type_enum_values(self) -> None:
        """Verify all LandingType enum values are defined."""
        assert LandingType.COMMERCIAL.value == "commercial"
        assert LandingType.SUBSISTENCE.value == "subsistence"
        assert LandingType.RECREATIONAL.value == "recreational"
        assert LandingType.UNREPORTED.value == "unreported"
        assert LandingType.DISCARDS.value == "discards"

    def test_all_landing_types_valid(self) -> None:
        """Verify all enum members can be instantiated."""
        for landing_type in LandingType:
            assert isinstance(landing_type, LandingType)
            assert isinstance(landing_type.value, str)

    def test_landing_type_comparison(self) -> None:
        """Verify landing type equality and comparison."""
        assert LandingType.COMMERCIAL == LandingType.COMMERCIAL
        assert LandingType.COMMERCIAL != LandingType.SUBSISTENCE
        assert LandingType.COMMERCIAL is LandingType.COMMERCIAL


class TestLandingRecord:
    """Test LandingRecord dataclass."""

    def test_create_valid_record(self, sample_landing_record: LandingRecord) -> None:
        """Create a fully specified valid record."""
        assert sample_landing_record.group_name == "Cod"
        assert sample_landing_record.year == 2020
        assert sample_landing_record.month == 3
        assert sample_landing_record.landing_type == LandingType.COMMERCIAL
        assert sample_landing_record.catch_tons == pytest.approx(150.5)
        assert sample_landing_record.value_usd == pytest.approx(450000.0)
        assert sample_landing_record.fleet_id == "Baltic_Demersal_Fleet"

    def test_create_minimal_record(self) -> None:
        """Create a minimal valid record."""
        record = LandingRecord(
            group_name="Sprat",
            year=2021,
            month=None,
            landing_type=LandingType.COMMERCIAL,
            catch_tons=500.0,
        )
        assert record.group_name == "Sprat"
        assert record.value_usd is None
        assert record.fleet_id is None

    def test_is_valid_complete(self, sample_landing_record: LandingRecord) -> None:
        """Test is_valid() for complete record."""
        assert sample_landing_record.is_valid() is True

    def test_is_valid_incomplete(self) -> None:
        """Test is_valid() for records with optional fields."""
        record = LandingRecord(
            group_name="Cod",
            year=2020,
            month=None,
            landing_type=LandingType.COMMERCIAL,
            catch_tons=100.0,
        )
        assert record.is_valid() is True

    def test_is_seasonal_with_month(self, sample_landing_record: LandingRecord) -> None:
        """Test is_seasonal() returns True when month is specified."""
        assert sample_landing_record.is_seasonal() is True

    def test_is_seasonal_without_month(
        self, sample_landing_record_annual: LandingRecord
    ) -> None:
        """Test is_seasonal() returns False when month is None."""
        assert sample_landing_record_annual.is_seasonal() is False

    @pytest.mark.parametrize(
        "catch_tons",
        [0.0, 0.1, 1000.0, 1e6],
    )
    def test_catch_amount_validation(self, catch_tons: float) -> None:
        """Test various valid catch amounts."""
        record = LandingRecord(
            group_name="Cod",
            year=2020,
            month=1,
            landing_type=LandingType.COMMERCIAL,
            catch_tons=catch_tons,
        )
        assert record.is_valid() is True

    def test_to_dict_serialization(self, sample_landing_record: LandingRecord) -> None:
        """Test conversion to dictionary."""
        data = sample_landing_record.to_dict()
        assert data["group_name"] == "Cod"
        assert data["year"] == 2020
        assert data["month"] == 3
        assert data["landing_type"] == "commercial"
        assert data["catch_tons"] == pytest.approx(150.5)
        assert data["value_usd"] == pytest.approx(450000.0)
        assert data["fleet_id"] == "Baltic_Demersal_Fleet"

    def test_from_dict_deserialization(self) -> None:
        """Test creation from dictionary."""
        data = {
            "group_name": "Cod",
            "year": 2020,
            "month": 3,
            "landing_type": "commercial",
            "catch_tons": 150.5,
            "value_usd": 450000.0,
            "fleet_id": "Baltic_Demersal_Fleet",
        }
        record = LandingRecord.from_dict(data)
        assert record.group_name == "Cod"
        assert record.landing_type == LandingType.COMMERCIAL

    def test_round_trip_serialization(
        self, sample_landing_record: LandingRecord
    ) -> None:
        """Test serialization and deserialization round-trip."""
        data = sample_landing_record.to_dict()
        reconstructed = LandingRecord.from_dict(data)
        assert reconstructed.group_name == sample_landing_record.group_name
        assert reconstructed.year == sample_landing_record.year
        assert reconstructed.catch_tons == sample_landing_record.catch_tons
        assert reconstructed.landing_type == sample_landing_record.landing_type


class TestLandingSeasonality:
    """Test LandingSeasonality dataclass."""

    def test_create_valid_seasonality(
        self, sample_landing_seasonality: LandingSeasonality
    ) -> None:
        """Create a valid seasonality pattern."""
        assert sample_landing_seasonality.group_name == "Cod"
        assert len(sample_landing_seasonality.monthly_proportions) == 12
        assert np.sum(sample_landing_seasonality.monthly_proportions) == pytest.approx(
            1.0
        )

    def test_is_valid_proper_sum(self) -> None:
        """Test is_valid() for proportions summing to 1.0."""
        proportions = np.ones(12) / 12.0  # Uniform distribution
        seasonality = LandingSeasonality(
            group_name="Herring",
            monthly_proportions=proportions,
        )
        assert seasonality.is_valid() is True

    def test_is_valid_improper_sum(self) -> None:
        """Test is_valid() rejects improper sum."""
        proportions = np.ones(12) / 10.0  # Sums to 1.2
        with pytest.raises(ValueError):
            LandingSeasonality(
                group_name="Cod",
                monthly_proportions=proportions,
            )

    def test_is_valid_negative_proportions(self) -> None:
        """Test is_valid() rejects negative proportions."""
        proportions = np.array(
            [
                0.1,
                0.1,
                0.1,
                0.1,
                0.1,
                0.1,
                -0.05,
                0.2,
                0.1,
                0.1,
                0.1,
                0.05,
            ]
        )
        with pytest.raises(ValueError):
            LandingSeasonality(
                group_name="Cod",
                monthly_proportions=proportions,
            )

    def test_get_monthly_catch(
        self, sample_landing_seasonality: LandingSeasonality
    ) -> None:
        """Test distribution of annual catch to monthly values."""
        annual_catch = 1200.0
        monthly_catch = sample_landing_seasonality.get_monthly_catch(annual_catch)
        assert len(monthly_catch) == 12
        assert np.sum(monthly_catch) == pytest.approx(annual_catch)

    def test_get_monthly_catch_conservation(
        self, sample_landing_seasonality: LandingSeasonality
    ) -> None:
        """Verify that sum of monthly catch equals annual catch."""
        annual_values = [0, 1.0, 100.0, 1000.0, 1e6]
        for annual in annual_values:
            monthly = sample_landing_seasonality.get_monthly_catch(annual)
            assert np.sum(monthly) == pytest.approx(annual)

    def test_flat_seasonality(self) -> None:
        """Test uniform seasonality (equal distribution)."""
        proportions = np.ones(12) / 12.0
        seasonality = LandingSeasonality(
            group_name="Flatfish",
            monthly_proportions=proportions,
        )
        annual_catch = 1200.0
        monthly_catch = seasonality.get_monthly_catch(annual_catch)
        expected = np.ones(12) * 100.0
        np.testing.assert_array_almost_equal(monthly_catch, expected)

    def test_to_dict_serialization(
        self, sample_landing_seasonality: LandingSeasonality
    ) -> None:
        """Test conversion to dictionary."""
        data = sample_landing_seasonality.to_dict()
        assert data["group_name"] == "Cod"
        assert len(data["monthly_proportions"]) == 12
        assert isinstance(data["monthly_proportions"], list)

    def test_round_trip_serialization(
        self, sample_landing_seasonality: LandingSeasonality
    ) -> None:
        """Test serialization and deserialization round-trip."""
        data = sample_landing_seasonality.to_dict()
        reconstructed = LandingSeasonality.from_dict(data)
        assert reconstructed.group_name == sample_landing_seasonality.group_name
        np.testing.assert_array_almost_equal(
            reconstructed.monthly_proportions,
            sample_landing_seasonality.monthly_proportions,
        )


class TestLandingsData:
    """Test LandingsData container."""

    def test_add_record(self) -> None:
        """Test adding a single record."""
        landings = LandingsData()
        record = LandingRecord(
            group_name="Cod",
            year=2020,
            month=3,
            landing_type=LandingType.COMMERCIAL,
            catch_tons=150.0,
        )
        landings.add_record(record)
        assert len(landings.records) == 1
        assert landings.records[0] == record

    def test_add_seasonality(self) -> None:
        """Test adding a seasonality pattern."""
        landings = LandingsData()
        seasonality = LandingSeasonality(
            group_name="Cod",
            monthly_proportions=np.ones(12) / 12.0,
        )
        landings.add_seasonality(seasonality)
        assert "Cod" in landings.seasonality
        assert landings.seasonality["Cod"] == seasonality

    def test_get_annual_catch_found(self, sample_landings_data: LandingsData) -> None:
        """Test getting annual catch for existing group and year."""
        annual = sample_landings_data.get_annual_catch("Cod", 2020)
        assert annual == pytest.approx(150.0 + 200.0)  # Sum of monthly records

    def test_get_annual_catch_not_found(
        self, sample_landings_data: LandingsData
    ) -> None:
        """Test getting annual catch for non-existent group/year."""
        annual = sample_landings_data.get_annual_catch("NonExistent", 2020)
        assert annual == pytest.approx(0.0)

    def test_get_monthly_catch_with_seasonality(
        self, sample_landings_data: LandingsData
    ) -> None:
        """Test getting monthly catch using seasonality pattern."""
        monthly = sample_landings_data.get_monthly_catch("Herring", 2020, 6)
        assert monthly >= 0

    def test_get_monthly_catch_without_seasonality(
        self, sample_landings_data: LandingsData
    ) -> None:
        """Test getting monthly catch for group without seasonality."""
        monthly = sample_landings_data.get_monthly_catch("Herring", 2020, 6)
        assert monthly >= 0

    def test_validate_empty(self) -> None:
        """Test validation of empty LandingsData."""
        landings = LandingsData()
        is_valid, errors = landings.validate()
        assert is_valid is True
        assert len(errors) == 0

    def test_validate_valid_state(self, sample_landings_data: LandingsData) -> None:
        """Test validation of valid LandingsData."""
        is_valid, errors = sample_landings_data.validate()
        assert is_valid is True
        assert len(errors) == 0

    def test_to_dataframe(self, sample_landings_data: LandingsData) -> None:
        """Test conversion to DataFrame."""
        df = sample_landings_data.to_dataframe()
        assert len(df) == 3  # 3 records in sample data
        assert "group_name" in df.columns
        assert "year" in df.columns
        assert "catch_tons" in df.columns

    def test_from_dataframe(self, sample_landings_data: LandingsData) -> None:
        """Test creation from DataFrame."""
        df = sample_landings_data.to_dataframe()
        reconstructed = LandingsData.from_dataframe(df)
        assert len(reconstructed.records) == len(sample_landings_data.records)

    def test_duplicate_records(self) -> None:
        """Test adding duplicate records."""
        landings = LandingsData()
        record = LandingRecord(
            group_name="Cod",
            year=2020,
            month=3,
            landing_type=LandingType.COMMERCIAL,
            catch_tons=150.0,
        )
        landings.add_record(record)
        landings.add_record(record)
        assert len(landings.records) == 2


class TestValidation:
    """Test validation functions."""

    def test_validate_valid_landings(self, sample_landings_data: LandingsData) -> None:
        """Test validation of valid landings."""
        is_valid, errors = validate_landings(sample_landings_data)
        assert is_valid is True
        assert len(errors) == 0

    def test_validate_missing_records(self) -> None:
        """Test validation with invalid records."""
        landings = LandingsData()
        # Manually inject invalid record (bypass add_record validation)
        invalid_record = LandingRecord.__new__(LandingRecord)
        invalid_record.group_name = ""
        invalid_record.year = 2020
        invalid_record.month = 3
        invalid_record.landing_type = LandingType.COMMERCIAL
        invalid_record.catch_tons = 100.0
        invalid_record.value_usd = None
        invalid_record.fleet_id = None
        landings.records.append(invalid_record)

        is_valid, errors = validate_landings(landings)
        assert is_valid is False
        assert len(errors) > 0

    def test_validate_invalid_seasonality(self) -> None:
        """Test validation with invalid seasonality."""
        landings = LandingsData()
        landings.records.append(
            LandingRecord(
                group_name="Cod",
                year=2020,
                month=1,
                landing_type=LandingType.COMMERCIAL,
                catch_tons=100.0,
            )
        )

        # Manually inject invalid seasonality
        invalid_season = LandingSeasonality.__new__(LandingSeasonality)
        invalid_season.group_name = "Cod"
        invalid_season.monthly_proportions = np.ones(12) / 10.0  # Sums to 1.2
        landings.seasonality["Cod"] = invalid_season

        is_valid, errors = validate_landings(landings)
        assert is_valid is False

    def test_validate_negative_catch(self) -> None:
        """Test validation rejects negative catch."""
        with pytest.raises(ValueError):
            LandingRecord(
                group_name="Cod",
                year=2020,
                month=1,
                landing_type=LandingType.COMMERCIAL,
                catch_tons=-100.0,
            )


class TestSeasonalityEstimation:
    """Test seasonality estimation from records."""

    def test_estimate_seasonality_from_monthly_records(self) -> None:
        """Test estimating seasonality from monthly records."""
        records = [
            LandingRecord(
                group_name="Cod",
                year=2020,
                month=i + 1,
                landing_type=LandingType.COMMERCIAL,
                catch_tons=100.0 * (i + 1),
            )
            for i in range(12)
        ]
        seasonality = estimate_seasonality_from_records(records, "Cod")
        assert seasonality is not None
        assert seasonality.group_name == "Cod"
        assert np.sum(seasonality.monthly_proportions) == pytest.approx(1.0)

    def test_estimate_seasonality_uniform(self) -> None:
        """Test seasonality estimation with uniform catches."""
        records = [
            LandingRecord(
                group_name="Herring",
                year=2020,
                month=i + 1,
                landing_type=LandingType.COMMERCIAL,
                catch_tons=100.0,
            )
            for i in range(12)
        ]
        seasonality = estimate_seasonality_from_records(records, "Herring")
        assert seasonality is not None
        expected = np.ones(12) / 12.0
        np.testing.assert_array_almost_equal(
            seasonality.monthly_proportions,
            expected,
        )

    def test_estimate_seasonality_seasonal_pattern(self) -> None:
        """Test seasonality estimation with seasonal pattern."""
        proportions = np.array(
            [
                0.05,
                0.05,
                0.08,
                0.10,
                0.12,
                0.15,
                0.18,
                0.15,
                0.12,
                0.10,
                0.05,
                0.05,
            ]
        )
        proportions = proportions / np.sum(proportions)
        records = [
            LandingRecord(
                group_name="Sprat",
                year=2020,
                month=i + 1,
                landing_type=LandingType.COMMERCIAL,
                catch_tons=proportions[i] * 1000.0,
            )
            for i in range(12)
        ]
        seasonality = estimate_seasonality_from_records(records, "Sprat")
        assert seasonality is not None
        np.testing.assert_array_almost_equal(
            seasonality.monthly_proportions,
            proportions,
        )

    def test_estimate_seasonality_single_record(self) -> None:
        """Test seasonality estimation with single record."""
        records = [
            LandingRecord(
                group_name="Flatfish",
                year=2020,
                month=6,
                landing_type=LandingType.COMMERCIAL,
                catch_tons=100.0,
            )
        ]
        seasonality = estimate_seasonality_from_records(records, "Flatfish")
        assert seasonality is not None
        # Should have 100% in June (index 5), 0% elsewhere
        assert seasonality.monthly_proportions[5] == pytest.approx(1.0)
        np.testing.assert_array_almost_equal(
            seasonality.monthly_proportions[:5],
            np.zeros(5),
        )


class TestAggregation:
    """Test aggregation functions."""

    def test_aggregate_all_records(self, sample_landings_data: LandingsData) -> None:
        """Test aggregating all records."""
        total = aggregate_landings(sample_landings_data.records)
        assert total == pytest.approx(150.0 + 200.0 + 1000.0)

    def test_aggregate_by_group(self, sample_landings_data: LandingsData) -> None:
        """Test aggregating by group name."""
        cod_total = aggregate_landings(
            sample_landings_data.records,
            group_name="Cod",
        )
        assert cod_total == pytest.approx(150.0 + 200.0)

    def test_aggregate_by_year(self, sample_landings_data: LandingsData) -> None:
        """Test aggregating by year."""
        year_2020 = aggregate_landings(
            sample_landings_data.records,
            year=2020,
        )
        assert year_2020 == pytest.approx(1350.0)

    def test_aggregate_by_landing_type(
        self, sample_landings_data: LandingsData
    ) -> None:
        """Test aggregating by landing type."""
        commercial = aggregate_landings(
            sample_landings_data.records,
            landing_type=LandingType.COMMERCIAL,
        )
        assert commercial == pytest.approx(1350.0)

    def test_aggregate_combined_filters(
        self, sample_landings_data: LandingsData
    ) -> None:
        """Test aggregating with multiple filters."""
        result = aggregate_landings(
            sample_landings_data.records,
            group_name="Cod",
            year=2020,
            landing_type=LandingType.COMMERCIAL,
        )
        assert result == pytest.approx(150.0 + 200.0)


class TestEdgeCases:
    """Test edge cases and boundary conditions."""

    def test_empty_landings_data(self) -> None:
        """Test LandingsData with no records."""
        landings = LandingsData()
        assert len(landings.records) == 0
        assert len(landings.seasonality) == 0
        is_valid, errors = landings.validate()
        assert is_valid is True

    def test_single_record(self) -> None:
        """Test LandingsData with single record."""
        landings = LandingsData()
        record = LandingRecord(
            group_name="Cod",
            year=2020,
            month=1,
            landing_type=LandingType.COMMERCIAL,
            catch_tons=100.0,
        )
        landings.add_record(record)
        assert len(landings.records) == 1

    def test_zero_catch(self) -> None:
        """Test record with zero catch."""
        record = LandingRecord(
            group_name="Cod",
            year=2020,
            month=1,
            landing_type=LandingType.COMMERCIAL,
            catch_tons=0.0,
        )
        assert record.is_valid() is True

    def test_very_large_catch(self) -> None:
        """Test record with very large catch."""
        record = LandingRecord(
            group_name="Cod",
            year=2020,
            month=1,
            landing_type=LandingType.COMMERCIAL,
            catch_tons=1e9,  # 1 billion tons
        )
        assert record.is_valid() is True

    def test_multiple_years(self) -> None:
        """Test LandingsData with multiple years."""
        landings = LandingsData()
        for year in range(2015, 2025):
            record = LandingRecord(
                group_name="Cod",
                year=year,
                month=1,
                landing_type=LandingType.COMMERCIAL,
                catch_tons=100.0,
            )
            landings.add_record(record)
        assert len(landings.records) == 10

    def test_all_landing_types(self) -> None:
        """Test records with all landing types."""
        landings = LandingsData()
        for landing_type in LandingType:
            record = LandingRecord(
                group_name="Cod",
                year=2020,
                month=1,
                landing_type=landing_type,
                catch_tons=100.0,
            )
            landings.add_record(record)
        assert len(landings.records) == 5

    def test_missing_value_usd(self) -> None:
        """Test record without economic value."""
        record = LandingRecord(
            group_name="Cod",
            year=2020,
            month=1,
            landing_type=LandingType.COMMERCIAL,
            catch_tons=100.0,
            value_usd=None,
        )
        assert record.is_valid() is True
        assert record.value_usd is None

    @pytest.mark.parametrize("month", [1, 6, 12])
    def test_month_boundary_values(self, month: int) -> None:
        """Test records with month boundary values."""
        record = LandingRecord(
            group_name="Cod",
            year=2020,
            month=month,
            landing_type=LandingType.COMMERCIAL,
            catch_tons=100.0,
        )
        assert record.is_valid() is True
        assert record.month == month

    def test_future_year_handling(self) -> None:
        """Test records with future year."""
        record = LandingRecord(
            group_name="Cod",
            year=2050,
            month=1,
            landing_type=LandingType.COMMERCIAL,
            catch_tons=100.0,
        )
        assert record.is_valid() is True
