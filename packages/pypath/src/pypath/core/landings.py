"""
Landings system for catch/landings tracking with temporal distribution.

This module implements catch and landings data management for fisheries,
including support for multiple landing types, temporal distribution (seasonality),
and validation of catch data.

Classes
-------
LandingType
    Enum defining types of catch/landings.
LandingRecord
    Dataclass representing a single catch record.
LandingSeasonality
    Dataclass for monthly catch distribution patterns.
LandingsData
    Container for managing collections of landing records and seasonality patterns.

Functions
---------
validate_landings
    Validate a LandingsData collection.
estimate_seasonality_from_records
    Infer monthly seasonality from historical records.
aggregate_landings
    Sum catches by group, year, and/or landing type.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


class LandingType(Enum):
    """Enumeration of catch/landings types.

    Attributes
    ----------
    COMMERCIAL
        Commercial fishing catch.
    SUBSISTENCE
        Subsistence/artisanal fishing.
    RECREATIONAL
        Recreational fishing catch.
    UNREPORTED
        Unreported or illegal fishing catch.
    DISCARDS
        Discarded bycatch.
    """

    COMMERCIAL = "commercial"
    SUBSISTENCE = "subsistence"
    RECREATIONAL = "recreational"
    UNREPORTED = "unreported"
    DISCARDS = "discards"


@dataclass
class LandingRecord:
    """A single catch/landings record.

    Parameters
    ----------
    group_name : str
        Name of the functional group (e.g., "Cod", "Herring").
    year : int
        Year of the landing.
    month : Optional[int]
        Month of landing (1-12), or None if annual.
    landing_type : LandingType
        Type of catch/landing.
    catch_tons : float
        Catch in metric tons, must be >= 0.
    value_usd : Optional[float]
        Economic value in USD, optional.
    fleet_id : Optional[str]
        Identifier for the fleet (e.g., "Baltic_Demersal_Fleet").

    Raises
    ------
    ValueError
        During validation if data is inconsistent.
    """

    group_name: str
    year: int
    month: Optional[int]
    landing_type: LandingType
    catch_tons: float
    value_usd: Optional[float] = None
    fleet_id: Optional[str] = None

    def __post_init__(self) -> None:
        """Validate record on instantiation."""
        if not self.is_valid():
            raise ValueError(f"Invalid LandingRecord: {self}")

    def is_valid(self) -> bool:
        """Check if the record is valid.

        Returns
        -------
        bool
            True if group_name is non-empty, year is positive, month (if present)
            is in 1-12, catch_tons >= 0, and value_usd (if present) >= 0.
        """
        if not self.group_name:
            logger.warning("LandingRecord has empty group_name")
            return False
        if self.year <= 0:
            logger.warning(f"LandingRecord has invalid year: {self.year}")
            return False
        if self.month is not None and (self.month < 1 or self.month > 12):
            logger.warning(f"LandingRecord has invalid month: {self.month}")
            return False
        if self.catch_tons < 0:
            logger.warning(f"LandingRecord has negative catch: {self.catch_tons}")
            return False
        if self.value_usd is not None and self.value_usd < 0:
            logger.warning(f"LandingRecord has negative value: {self.value_usd}")
            return False
        return True

    def is_seasonal(self) -> bool:
        """Check if this record has seasonal (monthly) resolution.

        Returns
        -------
        bool
            True if month is specified.
        """
        return self.month is not None

    def to_dict(self) -> dict:
        """Convert to dictionary representation.

        Returns
        -------
        dict
            Dictionary with all fields; LandingType converted to string value.
        """
        return {
            "group_name": self.group_name,
            "year": self.year,
            "month": self.month,
            "landing_type": self.landing_type.value,
            "catch_tons": self.catch_tons,
            "value_usd": self.value_usd,
            "fleet_id": self.fleet_id,
        }

    @classmethod
    def from_dict(cls, data: dict) -> LandingRecord:
        """Create from dictionary representation.

        Parameters
        ----------
        data : dict
            Dictionary with record fields. landing_type can be string (enum value)
            or LandingType instance.

        Returns
        -------
        LandingRecord
            Reconstructed record.
        """
        data_copy = data.copy()
        if isinstance(data_copy["landing_type"], str):
            data_copy["landing_type"] = LandingType(data_copy["landing_type"])
        return cls(**data_copy)


@dataclass
class LandingSeasonality:
    """Temporal (monthly) distribution of catch for a functional group.

    Parameters
    ----------
    group_name : str
        Name of the functional group.
    monthly_proportions : np.ndarray
        Array of shape (12,) with monthly fractions summing to 1.0.

    Raises
    ------
    ValueError
        If monthly_proportions does not sum to ~1.0 or has incorrect shape.
    """

    group_name: str
    monthly_proportions: np.ndarray

    def __post_init__(self) -> None:
        """Validate seasonality on instantiation."""
        if not self.is_valid():
            raise ValueError(f"Invalid LandingSeasonality: {self.group_name}")
        self.monthly_proportions = np.asarray(self.monthly_proportions, dtype=float)

    def is_valid(self) -> bool:
        """Check if seasonality is valid.

        Returns
        -------
        bool
            True if proportions has length 12, all values >= 0, and sum ≈ 1.0.
        """
        if not isinstance(self.monthly_proportions, (np.ndarray, list)):
            return False
        props = np.asarray(self.monthly_proportions, dtype=float)
        if props.shape != (12,):
            logger.warning(
                f"LandingSeasonality has incorrect shape: {props.shape}, expected (12,)"
            )
            return False
        if np.any(props < 0):
            logger.warning("LandingSeasonality has negative proportions")
            return False
        if not np.isclose(np.sum(props), 1.0, atol=1e-6):
            logger.warning(
                f"LandingSeasonality proportions do not sum to 1.0: {np.sum(props)}"
            )
            return False
        return True

    def get_monthly_catch(self, annual_catch: float) -> np.ndarray:
        """Distribute annual catch across months using seasonality.

        Parameters
        ----------
        annual_catch : float
            Total annual catch in metric tons.

        Returns
        -------
        np.ndarray
            Array of shape (12,) with monthly catch values.
        """
        return np.asarray(self.monthly_proportions, dtype=float) * annual_catch

    def to_dict(self) -> dict:
        """Convert to dictionary representation.

        Returns
        -------
        dict
            Dictionary with group_name and monthly_proportions (as list).
        """
        return {
            "group_name": self.group_name,
            "monthly_proportions": self.monthly_proportions.tolist(),
        }

    @classmethod
    def from_dict(cls, data: dict) -> LandingSeasonality:
        """Create from dictionary representation.

        Parameters
        ----------
        data : dict
            Dictionary with group_name and monthly_proportions.

        Returns
        -------
        LandingSeasonality
            Reconstructed seasonality pattern.
        """
        return cls(
            group_name=data["group_name"],
            monthly_proportions=np.asarray(data["monthly_proportions"], dtype=float),
        )


@dataclass
class LandingsData:
    """Container for landing records and seasonality patterns.

    Attributes
    ----------
    records : List[LandingRecord]
        Collection of landing records.
    seasonality : Dict[str, LandingSeasonality]
        Seasonality patterns indexed by group name.
    """

    records: List[LandingRecord] = field(default_factory=list)
    seasonality: Dict[str, LandingSeasonality] = field(default_factory=dict)

    def add_record(self, record: LandingRecord) -> None:
        """Add a landing record.

        Parameters
        ----------
        record : LandingRecord
            Record to add. Must be valid.

        Raises
        ------
        ValueError
            If record is not valid.
        """
        if not record.is_valid():
            raise ValueError(f"Cannot add invalid record: {record}")
        self.records.append(record)
        logger.debug(
            f"Added landing record: {record.group_name} "
            f"({record.year}{'m' + str(record.month) if record.month else 'Y'}): "
            f"{record.catch_tons} tons"
        )

    def add_seasonality(self, seasonality: LandingSeasonality) -> None:
        """Add or update a seasonality pattern.

        Parameters
        ----------
        seasonality : LandingSeasonality
            Seasonality pattern. Must be valid.

        Raises
        ------
        ValueError
            If seasonality is not valid.
        """
        if not seasonality.is_valid():
            raise ValueError(f"Cannot add invalid seasonality: {seasonality}")
        self.seasonality[seasonality.group_name] = seasonality
        logger.debug(f"Added seasonality pattern for {seasonality.group_name}")

    def get_annual_catch(self, group_name: str, year: int) -> float:
        """Get total annual catch for a group and year.

        Aggregates across all landing types and seasons.

        Parameters
        ----------
        group_name : str
            Name of the functional group.
        year : int
            Year to query.

        Returns
        -------
        float
            Total catch in metric tons.
        """
        total = 0.0
        for record in self.records:
            if record.group_name == group_name and record.year == year:
                total += record.catch_tons
        return total

    def get_monthly_catch(self, group_name: str, year: int, month: int) -> float:
        """Get catch for a specific month.

        First checks for exact monthly records; if not found and seasonality
        exists, estimates from annual data using seasonality pattern.

        Parameters
        ----------
        group_name : str
            Name of the functional group.
        year : int
            Year to query.
        month : int
            Month (1-12).

        Returns
        -------
        float
            Monthly catch in metric tons.
        """
        if month < 1 or month > 12:
            raise ValueError(f"Invalid month: {month}")

        monthly_total = 0.0
        for record in self.records:
            if (
                record.group_name == group_name
                and record.year == year
                and record.month == month
            ):
                monthly_total += record.catch_tons

        if monthly_total > 0:
            return monthly_total

        if group_name in self.seasonality:
            annual = self.get_annual_catch(group_name, year)
            if annual > 0:
                seasonality = self.seasonality[group_name]
                monthly_dist = seasonality.get_monthly_catch(annual)
                return float(monthly_dist[month - 1])

        return 0.0

    def validate(self) -> Tuple[bool, List[str]]:
        """Validate all records and seasonality patterns.

        Returns
        -------
        Tuple[bool, List[str]]
            (is_valid, list_of_error_messages). is_valid is True if all
            records and seasonality patterns are valid and consistent.
        """
        errors: List[str] = []

        for i, record in enumerate(self.records):
            if not record.is_valid():
                errors.append(f"Record {i} is invalid: {record}")

        for name, season in self.seasonality.items():
            if not season.is_valid():
                errors.append(f"Seasonality for {name} is invalid")

        if errors:
            logger.warning(f"Validation found {len(errors)} error(s)")

        return len(errors) == 0, errors

    def to_dataframe(self) -> pd.DataFrame:
        """Convert records to DataFrame.

        Returns
        -------
        pd.DataFrame
            DataFrame with columns: group_name, year, month, landing_type,
            catch_tons, value_usd, fleet_id.
        """
        data = [record.to_dict() for record in self.records]
        if not data:
            return pd.DataFrame()
        return pd.DataFrame(data)

    @classmethod
    def from_dataframe(cls, df: pd.DataFrame) -> LandingsData:
        """Create LandingsData from DataFrame.

        Parameters
        ----------
        df : pd.DataFrame
            DataFrame with columns matching LandingRecord fields.

        Returns
        -------
        LandingsData
            Reconstructed container.
        """
        landings = cls()
        for _, row in df.iterrows():
            record = LandingRecord.from_dict(row.to_dict())
            landings.add_record(record)
        return landings


def validate_landings(landings: LandingsData) -> Tuple[bool, List[str]]:
    """Validate a LandingsData collection.

    Parameters
    ----------
    landings : LandingsData
        Collection to validate.

    Returns
    -------
    Tuple[bool, List[str]]
        (is_valid, error_messages).
    """
    return landings.validate()


def estimate_seasonality_from_records(
    records: List[LandingRecord], group_name: str
) -> Optional[LandingSeasonality]:
    """Estimate monthly seasonality from historical records.

    Computes the proportion of catch in each month across all available
    records for a given group.

    Parameters
    ----------
    records : List[LandingRecord]
        List of landing records with monthly resolution.
    group_name : str
        Name of the functional group.

    Returns
    -------
    Optional[LandingSeasonality]
        Seasonality pattern, or None if no monthly data available.
    """
    monthly_catch = np.zeros(12)
    total_catch = 0.0

    for record in records:
        if record.group_name == group_name and record.is_seasonal():
            month_idx = record.month - 1  # type: ignore
            monthly_catch[month_idx] += record.catch_tons
            total_catch += record.catch_tons

    if total_catch <= 0:
        logger.warning(
            f"No monthly catch data available for {group_name} to estimate seasonality"
        )
        return None

    proportions = monthly_catch / total_catch
    logger.debug(f"Estimated seasonality for {group_name} from {len(records)} records")

    return LandingSeasonality(group_name=group_name, monthly_proportions=proportions)


def aggregate_landings(
    records: List[LandingRecord],
    group_name: Optional[str] = None,
    year: Optional[int] = None,
    landing_type: Optional[LandingType] = None,
) -> float:
    """Aggregate landing records by filters.

    Sum catch across records matching specified criteria. All filters are
    optional; if omitted, matches all records.

    Parameters
    ----------
    records : List[LandingRecord]
        Records to aggregate.
    group_name : Optional[str]
        Filter by functional group name.
    year : Optional[int]
        Filter by year.
    landing_type : Optional[LandingType]
        Filter by landing type.

    Returns
    -------
    float
        Sum of catch_tons for matching records.
    """
    total = 0.0
    for record in records:
        if group_name is not None and record.group_name != group_name:
            continue
        if year is not None and record.year != year:
            continue
        if landing_type is not None and record.landing_type != landing_type:
            continue
        total += record.catch_tons
    return total
