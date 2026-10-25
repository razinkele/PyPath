"""Fishing fleet management, gear types, and discard routing.

This module implements fleet-based fishing dynamics, including gear classification,
catchability by group, and discard fate allocation for Ecopath with Ecosim models.

Phase 1B.4: TG3.4 Fleet & Discard System
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional

import pandas as pd

logger = logging.getLogger(__name__)


class GearType(Enum):
    """Fishing gear types with associated selectivity metadata.

    Each gear type carries default bycatch rate and selectivity characteristics
    used in catch allocation and validation.

    Attributes
    ----------
    SEINE : GearType
        Seine net (mid-water or bottom); moderate selectivity, low bycatch.
    NET : GearType
        General net (gill, cast, etc.); moderate-high selectivity, variable bycatch.
    TRAP : GearType
        Fish trap or pot; high selectivity, low bycatch.
    TRAWL : GearType
        Trawl (bottom, mid-water, or pelagic); low selectivity, high bycatch.
    HOOK : GearType
        Hook and line; very high selectivity, minimal bycatch.
    OTHER : GearType
        Unspecified gear; assume moderate selectivity and bycatch.
    """

    SEINE = "Seine"
    NET = "Net"
    TRAP = "Trap"
    TRAWL = "Trawl"
    HOOK = "Hook"
    OTHER = "Other"

    def selectivity(self) -> float:
        """Return default selectivity for this gear type.

        Selectivity (0-1) represents how narrowly the gear targets intended species.
        Higher values = more selective (lower bycatch proportion).

        Returns
        -------
        float
            Selectivity coefficient (0.2-1.0).
        """
        selectivity_map = {
            GearType.SEINE: 0.65,
            GearType.NET: 0.60,
            GearType.TRAP: 0.85,
            GearType.TRAWL: 0.35,
            GearType.HOOK: 0.95,
            GearType.OTHER: 0.50,
        }
        return selectivity_map[self]

    def bycatch_rate(self) -> float:
        """Return default bycatch rate for this gear type.

        Bycatch rate (0-1) represents the proportion of non-target catch
        relative to target catch.

        Returns
        -------
        float
            Bycatch rate (0.0-0.8).
        """
        bycatch_map = {
            GearType.SEINE: 0.15,
            GearType.NET: 0.25,
            GearType.TRAP: 0.05,
            GearType.TRAWL: 0.60,
            GearType.HOOK: 0.02,
            GearType.OTHER: 0.30,
        }
        return bycatch_map[self]


@dataclass
class FleetGroup:
    """A fishing fleet with effort, catchability, and gear composition.

    Represents a commercial or subsistence fishing fleet targeting one or more
    groups. Effort is applied via catchability coefficients (q values) to
    compute catch from available biomass.

    Parameters
    ----------
    name : str
        Fleet name (e.g., "Small-scale pelagic").
    fleet_id : int
        Unique fleet identifier (0-based).
    gears : list[GearType]
        List of gear types used by this fleet.
    fuel_efficiency : float
        Fuel consumption (kg fuel per ton catch). Default 0.05.
    base_effort : float
        Nominal effort (days or trips per year). Default 100.
    catchability : dict[str, float]
        Catchability coefficients (q values) by group name. Maps group_name -> q.

    Attributes
    ----------
    name : str
        Fleet name.
    fleet_id : int
        Fleet identifier.
    gears : list[GearType]
        Gear list.
    fuel_efficiency : float
        Fuel per unit catch (0.01-0.20).
    base_effort : float
        Baseline effort (>0).
    catchability : dict[str, float]
        q values (0.0-1.0).

    Raises
    ------
    ValueError
        If fleet_id < 0, base_effort <= 0, or catchability values out of bounds.
    """

    name: str
    fleet_id: int
    gears: list[GearType]
    fuel_efficiency: float = 0.05
    base_effort: float = 100.0
    catchability: dict[str, float] = field(default_factory=dict)

    def __post_init__(self) -> None:
        """Validate fleet parameters after initialization."""
        if self.fleet_id < 0:
            raise ValueError(f"fleet_id must be >= 0, got {self.fleet_id}")
        if self.base_effort <= 0:
            raise ValueError(f"base_effort must be > 0, got {self.base_effort}")
        if self.fuel_efficiency < 0.01 or self.fuel_efficiency > 0.20:
            logger.warning(
                f"Fleet {self.name}: fuel_efficiency {self.fuel_efficiency} "
                "outside typical range [0.01, 0.20]"
            )
        if not self.gears:
            logger.warning(f"Fleet {self.name}: no gears specified")
        for group, q in self.catchability.items():
            if not (0.0 <= q <= 1.0):
                raise ValueError(f"Catchability for {group} must be in [0, 1], got {q}")

    def is_valid(self) -> bool:
        """Validate fleet definition for completeness.

        Checks that name, fleet_id, and at least one catchability value exist.

        Returns
        -------
        bool
            True if valid, False otherwise.
        """
        if not self.name or self.fleet_id < 0:
            return False
        if not self.catchability:
            return False
        return True

    def to_dict(self) -> dict:
        """Serialize fleet to dictionary.

        Returns
        -------
        dict
            Dictionary with keys: name, fleet_id, gears, fuel_efficiency,
            base_effort, catchability.
        """
        return {
            "name": self.name,
            "fleet_id": self.fleet_id,
            "gears": [g.value for g in self.gears],
            "fuel_efficiency": self.fuel_efficiency,
            "base_effort": self.base_effort,
            "catchability": self.catchability.copy(),
        }

    @classmethod
    def from_dict(cls, data: dict) -> FleetGroup:
        """Deserialize fleet from dictionary.

        Parameters
        ----------
        data : dict
            Dictionary with fleet parameters.

        Returns
        -------
        FleetGroup
            Reconstructed fleet instance.
        """
        gears = [GearType(g) for g in data.get("gears", [])]
        return cls(
            name=data["name"],
            fleet_id=data["fleet_id"],
            gears=gears,
            fuel_efficiency=data.get("fuel_efficiency", 0.05),
            base_effort=data.get("base_effort", 100.0),
            catchability=data.get("catchability", {}),
        )


@dataclass
class DiscardPath:
    """Pathway for discarded catch to its fate.

    Discards can be lost to the system (total mortality), routed to predators
    (as food), or deposited as detritus/benthos. Depth-dependent routing
    enables spatially-explicit discard fate in Ecospace.

    Parameters
    ----------
    source_group : str
        Group name for the discarded catch (e.g., "Juvenile cod").
    target_group : str, optional
        Group that consumes discards (None for lost/detritus routing).
    discard_rate : float
        Proportion discarded (0-1). Example: 0.15 = 15% discarded.
    fate : str
        Discard fate ('lost', 'predation', 'detritus', 'benthos').
        - 'lost': mortality (biomass removed from system)
        - 'predation': biomass routed to target_group
        - 'detritus': enters detritus pool
        - 'benthos': enters benthic organic matter pool
    depth_dependent : bool
        If True, discard fate depends on water column position (Ecospace).

    Attributes
    ----------
    source_group : str
        Source group name.
    target_group : str | None
        Target group name (or None).
    discard_rate : float
        Proportion discarded (0.0-1.0).
    fate : str
        Fate category.
    depth_dependent : bool
        Spatially-explicit routing flag.

    Raises
    ------
    ValueError
        If discard_rate not in [0, 1] or fate not in valid categories.
    """

    source_group: str
    target_group: Optional[str] = None
    discard_rate: float = 0.0
    fate: str = "lost"
    depth_dependent: bool = False

    def __post_init__(self) -> None:
        """Validate discard path parameters after initialization."""
        if not (0.0 <= self.discard_rate <= 1.0):
            raise ValueError(f"discard_rate must be in [0, 1], got {self.discard_rate}")
        valid_fates = ("lost", "predation", "detritus", "benthos")
        if self.fate not in valid_fates:
            raise ValueError(f"fate must be one of {valid_fates}, got {self.fate}")
        if self.fate == "predation" and not self.target_group:
            raise ValueError("fate='predation' requires target_group to be specified")

    def is_valid(self) -> bool:
        """Validate discard path for consistency.

        Checks that source_group exists and fate-group combinations are valid.

        Returns
        -------
        bool
            True if valid, False otherwise.
        """
        if not self.source_group:
            return False
        if self.fate == "predation" and not self.target_group:
            return False
        if not (0.0 <= self.discard_rate <= 1.0):
            return False
        return True

    def apply_to_biomass(self, biomass: float) -> float:
        """Calculate discard biomass from total catch.

        Parameters
        ----------
        biomass : float
            Total catch biomass (tons).

        Returns
        -------
        float
            Discard biomass (tons).
        """
        return biomass * self.discard_rate

    def to_dict(self) -> dict:
        """Serialize discard path to dictionary.

        Returns
        -------
        dict
            Dictionary representation.
        """
        return {
            "source_group": self.source_group,
            "target_group": self.target_group,
            "discard_rate": self.discard_rate,
            "fate": self.fate,
            "depth_dependent": self.depth_dependent,
        }

    @classmethod
    def from_dict(cls, data: dict) -> DiscardPath:
        """Deserialize discard path from dictionary.

        Parameters
        ----------
        data : dict
            Dictionary with discard path parameters.

        Returns
        -------
        DiscardPath
            Reconstructed discard path instance.
        """
        return cls(
            source_group=data["source_group"],
            target_group=data.get("target_group"),
            discard_rate=data.get("discard_rate", 0.0),
            fate=data.get("fate", "lost"),
            depth_dependent=data.get("depth_dependent", False),
        )


@dataclass
class FleetDynamics:
    """Container for fleet groups and discard routing rules.

    Manages collection of fleets and discard paths, with validation and
    application methods for catch allocation and discard fate routing.

    Parameters
    ----------
    fleets : dict[str, FleetGroup]
        Fleet groups indexed by name. Default empty dict.
    discard_paths : dict[str, list[DiscardPath]]
        Discard paths indexed by source_group. Default empty dict.

    Attributes
    ----------
    fleets : dict[str, FleetGroup]
        Active fleets.
    discard_paths : dict[str, list[DiscardPath]]
        Discard routing rules.
    """

    fleets: dict[str, FleetGroup] = field(default_factory=dict)
    discard_paths: dict[str, list[DiscardPath]] = field(default_factory=dict)

    def add_fleet(self, fleet: FleetGroup) -> None:
        """Add or replace a fleet.

        Parameters
        ----------
        fleet : FleetGroup
            Fleet to add. If a fleet with the same name exists, it is replaced.
        """
        if not fleet.is_valid():
            logger.warning(f"Fleet {fleet.name} is not valid; adding anyway")
        self.fleets[fleet.name] = fleet
        logger.debug(f"Added fleet: {fleet.name} (ID: {fleet.fleet_id})")

    def add_discard_path(self, path: DiscardPath) -> None:
        """Add a discard path.

        Parameters
        ----------
        path : DiscardPath
            Discard path to add.
        """
        if not path.is_valid():
            logger.warning(f"Discard path for {path.source_group} is not valid")
        if path.source_group not in self.discard_paths:
            self.discard_paths[path.source_group] = []
        self.discard_paths[path.source_group].append(path)
        logger.debug(
            f"Added discard path: {path.source_group} -> {path.target_group} "
            f"({path.fate})"
        )

    def get_fleet(self, name: str) -> Optional[FleetGroup]:
        """Retrieve a fleet by name.

        Parameters
        ----------
        name : str
            Fleet name.

        Returns
        -------
        FleetGroup | None
            Fleet if found, None otherwise.
        """
        return self.fleets.get(name)

    def validate(self) -> tuple[bool, list[str]]:
        """Validate fleet dynamics configuration.

        Checks that:
        - All fleets are valid
        - All discard paths are valid
        - Referenced groups exist (soft check: logged as warning, not error)

        Returns
        -------
        tuple[bool, list[str]]
            (is_valid, error_messages). is_valid=True only if all checks pass.
        """
        errors = []
        for fleet in self.fleets.values():
            if not fleet.is_valid():
                errors.append(f"Fleet {fleet.name} is not valid")
        for source, paths in self.discard_paths.items():
            for path in paths:
                if not path.is_valid():
                    errors.append(
                        f"Discard path {source} -> {path.target_group} is not valid"
                    )
        return len(errors) == 0, errors

    def to_dataframe(self) -> dict[str, pd.DataFrame]:
        """Convert fleet dynamics to DataFrame representation.

        Returns
        -------
        dict[str, pd.DataFrame]
            Dictionary with keys 'fleets' and 'discards', each a DataFrame.
            - fleets: columns [name, fleet_id, gears, fuel_efficiency,
              base_effort, catchability (dict column)]
            - discards: columns [source_group, target_group, discard_rate,
              fate, depth_dependent]
        """
        fleets_data = [f.to_dict() for f in self.fleets.values()]
        fleets_df = pd.DataFrame(fleets_data) if fleets_data else pd.DataFrame()

        discards_data = []
        for paths in self.discard_paths.values():
            discards_data.extend([p.to_dict() for p in paths])
        discards_df = pd.DataFrame(discards_data) if discards_data else pd.DataFrame()

        return {"fleets": fleets_df, "discards": discards_df}

    @classmethod
    def from_dataframe(
        cls, fleets_df: pd.DataFrame, discards_df: pd.DataFrame
    ) -> FleetDynamics:
        """Reconstruct fleet dynamics from DataFrames.

        Parameters
        ----------
        fleets_df : pd.DataFrame
            Fleet data (as returned by to_dataframe()['fleets']).
        discards_df : pd.DataFrame
            Discard data (as returned by to_dataframe()['discards']).

        Returns
        -------
        FleetDynamics
            Reconstructed fleet dynamics instance.
        """
        fd = cls()
        for _, row in fleets_df.iterrows():
            fleet_dict = row.to_dict()
            fd.add_fleet(FleetGroup.from_dict(fleet_dict))
        for _, row in discards_df.iterrows():
            path_dict = row.to_dict()
            fd.add_discard_path(DiscardPath.from_dict(path_dict))
        return fd


def allocate_catch(
    total_catch: float, catchability: dict[str, float]
) -> dict[str, float]:
    """Allocate catch across groups based on catchability.

    Distributes total catch proportionally to catchability coefficients
    (q values) for each group.

    Parameters
    ----------
    total_catch : float
        Total catch biomass (tons).
    catchability : dict[str, float]
        Catchability by group name. Values should be normalized (sum to ~1)
        for intuitive interpretation.

    Returns
    -------
    dict[str, float]
        Allocated catch per group.

    Examples
    --------
    >>> catchability = {"Cod": 0.6, "Haddock": 0.4}
    >>> catch = allocate_catch(100.0, catchability)
    >>> catch["Cod"]
    60.0
    """
    if not catchability:
        return {}
    total_q = sum(catchability.values())
    if total_q <= 0:
        logger.warning("Total catchability is zero or negative")
        return {g: 0.0 for g in catchability}
    return {g: total_catch * (q / total_q) for g, q in catchability.items()}


def allocate_discards(discard_rate: float, source_group: str) -> dict[str, float]:
    """Allocate discard biomass from a source group.

    Computes discard proportion and residual landed proportion from a
    discard rate. Used to split catch into landed and discarded components.

    Parameters
    ----------
    discard_rate : float
        Proportion discarded (0-1).
    source_group : str
        Source group name (for logging).

    Returns
    -------
    dict[str, float]
        Dictionary with 'landed' and 'discarded' proportions.

    Raises
    ------
    ValueError
        If discard_rate not in [0, 1].
    """
    if not (0.0 <= discard_rate <= 1.0):
        raise ValueError(f"discard_rate must be in [0, 1], got {discard_rate}")
    return {
        "discarded": discard_rate,
        "landed": 1.0 - discard_rate,
    }


def validate_allocation(
    allocation: dict[str, float],
    tolerance: float = 0.01,
) -> bool:
    """Validate that allocation sums to approximately 1.0.

    Parameters
    ----------
    allocation : dict[str, float]
        Allocation dictionary (e.g., from allocate_catch or allocate_discards).
    tolerance : float
        Acceptable deviation from 1.0. Default 0.01 (1%).

    Returns
    -------
    bool
        True if allocation sums to 1.0 ± tolerance, False otherwise.
    """
    if not allocation:
        logger.warning("Empty allocation")
        return False
    total = sum(allocation.values())
    is_valid = abs(total - 1.0) <= tolerance
    if not is_valid:
        logger.warning(
            f"Allocation sum {total} deviates from 1.0 by {abs(total - 1.0)}"
        )
    return is_valid


def estimate_catch_value(catch: dict[str, float], prices: dict[str, float]) -> float:
    """Estimate total economic value of catch.

    Multiplies catch biomass by price per unit (e.g., $/kg) for each group
    and sums the total value.

    Parameters
    ----------
    catch : dict[str, float]
        Catch biomass per group (tons).
    prices : dict[str, float]
        Price per unit (e.g., $/kg) by group name.

    Returns
    -------
    float
        Total catch value (in same currency units as prices).

    Notes
    -----
    This is a stub function for economic analysis. Price data and units
    ($/kg, $/ton, etc.) should be externally documented.
    """
    total_value = 0.0
    for group, catch_biomass in catch.items():
        if group in prices:
            price_per_unit = prices[group]
            value = catch_biomass * 1000 * price_per_unit
            total_value += value
    return total_value


def estimate_fuel_cost(effort: float, fuel_price: float) -> float:
    """Estimate fuel cost for fishing effort.

    Computes cost based on effort (days/trips) and fuel price per unit.
    Assumes a default fuel consumption profile per effort unit.

    Parameters
    ----------
    effort : float
        Effort (days or trips).
    fuel_price : float
        Price per unit fuel (e.g., $/liter).

    Returns
    -------
    float
        Total fuel cost (same currency units as fuel_price).

    Notes
    -----
    This is a stub function. Actual fuel consumption profiles should be
    calibrated to specific fleet characteristics or regional data.
    """
    fuel_consumption_per_day = 100.0
    return effort * fuel_consumption_per_day * fuel_price
