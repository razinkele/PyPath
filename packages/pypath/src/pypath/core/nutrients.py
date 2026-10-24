"""Nutrient cycling (N, P, C) system for Ecopath groups.

This module implements nutrient content, excretion, and uptake
calculations following standard aquatic biogeochemistry conventions.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from enum import Enum
from typing import TYPE_CHECKING, Dict, List, Tuple

import numpy as np
import pandas as pd

if TYPE_CHECKING:
    from pypath.core.params import RpathParams

logger = logging.getLogger(__name__)


class NutrientElement(str, Enum):
    """Nutrient elements tracked in food web models.

    Attributes
    ----------
    NITROGEN : str
        Nitrogen (N), molecular weight 14.01 g/mol
    PHOSPHORUS : str
        Phosphorus (P), molecular weight 30.97 g/mol
    CARBON : str
        Carbon (C), molecular weight 12.01 g/mol
    """

    NITROGEN = "nitrogen"
    PHOSPHORUS = "phosphorus"
    CARBON = "carbon"

    @property
    def molecular_weight(self) -> float:
        """Molecular weight in g/mol.

        Returns
        -------
        float
            Molecular weight of the element.
        """
        weights = {
            NutrientElement.NITROGEN: 14.01,
            NutrientElement.PHOSPHORUS: 30.97,
            NutrientElement.CARBON: 12.01,
        }
        return weights[self]


@dataclass
class NutrientContent:
    """Nutrient elemental composition of a biomass group.

    Attributes
    ----------
    group_name : str
        Name of the functional group
    nitrogen_percent : float
        Nitrogen as % dry weight (typical: 7-15%)
    phosphorus_percent : float
        Phosphorus as % dry weight (typical: 1-2%)
    carbon_percent : float
        Carbon as % dry weight (typical: 45-50%)
    ash_percent : float
        Ash (mineral) as % dry weight (typical: 5-15%)
    """

    group_name: str
    nitrogen_percent: float
    phosphorus_percent: float
    carbon_percent: float
    ash_percent: float

    def __post_init__(self) -> None:
        """Validate nutrient composition ranges."""
        if not (0 <= self.nitrogen_percent <= 20):
            raise ValueError(
                f"{self.group_name}: nitrogen_percent must be 0-20%, "
                f"got {self.nitrogen_percent}"
            )
        if not (0 <= self.phosphorus_percent <= 5):
            raise ValueError(
                f"{self.group_name}: phosphorus_percent must be 0-5%, "
                f"got {self.phosphorus_percent}"
            )
        if not (40 <= self.carbon_percent <= 55):
            raise ValueError(
                f"{self.group_name}: carbon_percent must be 40-55%, "
                f"got {self.carbon_percent}"
            )
        if not (0 <= self.ash_percent <= 20):
            raise ValueError(
                f"{self.group_name}: ash_percent must be 0-20%, "
                f"got {self.ash_percent}"
            )

    def is_valid(self) -> bool:
        """Check if nutrient composition is valid.

        Returns
        -------
        bool
            True if all constraints are satisfied, False otherwise.
        """
        try:
            self.__post_init__()
            return True
        except ValueError:
            return False

    def to_dict(self) -> Dict[str, float]:
        """Convert to dictionary representation.

        Returns
        -------
        dict
            Dictionary with nutrient percentages.
        """
        return {
            "group_name": self.group_name,
            "nitrogen_percent": self.nitrogen_percent,
            "phosphorus_percent": self.phosphorus_percent,
            "carbon_percent": self.carbon_percent,
            "ash_percent": self.ash_percent,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, float]) -> NutrientContent:
        """Create from dictionary representation.

        Parameters
        ----------
        data : dict
            Dictionary with required keys: group_name, nitrogen_percent,
            phosphorus_percent, carbon_percent, ash_percent.

        Returns
        -------
        NutrientContent
            Instantiated object.
        """
        return cls(
            group_name=data["group_name"],
            nitrogen_percent=data["nitrogen_percent"],
            phosphorus_percent=data["phosphorus_percent"],
            carbon_percent=data["carbon_percent"],
            ash_percent=data["ash_percent"],
        )


@dataclass
class NutrientExcretion:
    """Excretion rates and fecal composition for a group.

    Attributes
    ----------
    group_name : str
        Name of the functional group
    n_excretion_rate : float
        Nitrogen excretion as % of daily consumption (typical: 5-15%)
    p_excretion_rate : float
        Phosphorus excretion as % of daily consumption (typical: 1-3%)
    fecal_n_fraction : float
        Fraction of excretion that is fecal vs. dissolved (0-1)
    fecal_p_fraction : float
        Fraction of phosphorus excretion that is fecal (0-1)
    """

    group_name: str
    n_excretion_rate: float
    p_excretion_rate: float
    fecal_n_fraction: float
    fecal_p_fraction: float

    def __post_init__(self) -> None:
        """Validate excretion rates."""
        if not (0 <= self.n_excretion_rate <= 50):
            raise ValueError(
                f"{self.group_name}: n_excretion_rate must be 0-50%, "
                f"got {self.n_excretion_rate}"
            )
        if not (0 <= self.p_excretion_rate <= 20):
            raise ValueError(
                f"{self.group_name}: p_excretion_rate must be 0-20%, "
                f"got {self.p_excretion_rate}"
            )
        if not (0 <= self.fecal_n_fraction <= 1):
            raise ValueError(
                f"{self.group_name}: fecal_n_fraction must be 0-1, "
                f"got {self.fecal_n_fraction}"
            )
        if not (0 <= self.fecal_p_fraction <= 1):
            raise ValueError(
                f"{self.group_name}: fecal_p_fraction must be 0-1, "
                f"got {self.fecal_p_fraction}"
            )

    def calculate_excretion(
        self, consumption: float, element: str
    ) -> float:
        """Calculate excretion of a nutrient element.

        Parameters
        ----------
        consumption : float
            Daily consumption rate (mass units, e.g., kg/day)
        element : str
            Element name ('nitrogen' or 'phosphorus')

        Returns
        -------
        float
            Daily excretion rate (same units as consumption)

        Raises
        ------
        ValueError
            If element is not 'nitrogen' or 'phosphorus'.
        """
        if element.lower() == "nitrogen":
            return consumption * (self.n_excretion_rate / 100.0)
        elif element.lower() == "phosphorus":
            return consumption * (self.p_excretion_rate / 100.0)
        else:
            raise ValueError(
                f"Unknown element: {element}. "
                f"Must be 'nitrogen' or 'phosphorus'."
            )

    def to_dict(self) -> Dict[str, float]:
        """Convert to dictionary representation.

        Returns
        -------
        dict
            Dictionary with excretion parameters.
        """
        return {
            "group_name": self.group_name,
            "n_excretion_rate": self.n_excretion_rate,
            "p_excretion_rate": self.p_excretion_rate,
            "fecal_n_fraction": self.fecal_n_fraction,
            "fecal_p_fraction": self.fecal_p_fraction,
        }


@dataclass
class NutrientUptake:
    """Nutrient uptake kinetics for primary producers.

    Attributes
    ----------
    group_name : str
        Name of the functional group (typically a primary producer)
    n_uptake_rate : float
        Maximum nitrogen uptake rate (mg N / mg biomass / day)
    p_uptake_rate : float
        Maximum phosphorus uptake rate (mg P / mg biomass / day)
    half_saturation_n : float
        Half-saturation constant for nitrogen (mg N/L)
    half_saturation_p : float
        Half-saturation constant for phosphorus (mg P/L)
    """

    group_name: str
    n_uptake_rate: float
    p_uptake_rate: float
    half_saturation_n: float
    half_saturation_p: float

    def __post_init__(self) -> None:
        """Validate uptake parameters."""
        if self.n_uptake_rate < 0:
            raise ValueError(
                f"{self.group_name}: n_uptake_rate must be >= 0, "
                f"got {self.n_uptake_rate}"
            )
        if self.p_uptake_rate < 0:
            raise ValueError(
                f"{self.group_name}: p_uptake_rate must be >= 0, "
                f"got {self.p_uptake_rate}"
            )
        if self.half_saturation_n < 0:
            raise ValueError(
                f"{self.group_name}: half_saturation_n must be >= 0, "
                f"got {self.half_saturation_n}"
            )
        if self.half_saturation_p < 0:
            raise ValueError(
                f"{self.group_name}: half_saturation_p must be >= 0, "
                f"got {self.half_saturation_p}"
            )

    def calculate_uptake(
        self, nutrient_level: float, rate: float
    ) -> float:
        """Calculate uptake using Monod (Michaelis-Menten) kinetics.

        Parameters
        ----------
        nutrient_level : float
            Ambient nutrient concentration (mg/L)
        rate : float
            Maximum uptake rate (Vmax)

        Returns
        -------
        float
            Uptake rate under Monod kinetics (mg/L/day or similar)

        Notes
        -----
        Uses the Monod equation: v = (Vmax * [S]) / (Km + [S])
        where Km is the appropriate half-saturation constant.
        """
        if nutrient_level < 0:
            raise ValueError(
                f"nutrient_level must be >= 0, got {nutrient_level}"
            )
        if rate < 0:
            raise ValueError(f"rate must be >= 0, got {rate}")

        return (rate * nutrient_level) / (rate + nutrient_level)

    def to_dict(self) -> Dict[str, float]:
        """Convert to dictionary representation.

        Returns
        -------
        dict
            Dictionary with uptake parameters.
        """
        return {
            "group_name": self.group_name,
            "n_uptake_rate": self.n_uptake_rate,
            "p_uptake_rate": self.p_uptake_rate,
            "half_saturation_n": self.half_saturation_n,
            "half_saturation_p": self.half_saturation_p,
        }


@dataclass
class NutrientCycle:
    """Container for nutrient cycling parameters across all groups.

    Attributes
    ----------
    contents : dict
        Mapping of group_name -> NutrientContent
    excretions : dict
        Mapping of group_name -> NutrientExcretion
    uptakes : dict
        Mapping of group_name -> NutrientUptake (typically producers only)
    """

    contents: Dict[str, NutrientContent] = field(default_factory=dict)
    excretions: Dict[str, NutrientExcretion] = field(default_factory=dict)
    uptakes: Dict[str, NutrientUptake] = field(default_factory=dict)

    def add_content(self, content: NutrientContent) -> None:
        """Add or update nutrient content for a group.

        Parameters
        ----------
        content : NutrientContent
            Nutrient composition parameters.
        """
        if not content.is_valid():
            raise ValueError(f"Invalid nutrient content for {content.group_name}")
        self.contents[content.group_name] = content
        logger.debug(f"Added nutrient content for {content.group_name}")

    def add_excretion(self, excretion: NutrientExcretion) -> None:
        """Add or update excretion parameters for a group.

        Parameters
        ----------
        excretion : NutrientExcretion
            Excretion and fecal composition parameters.
        """
        self.excretions[excretion.group_name] = excretion
        logger.debug(f"Added excretion parameters for {excretion.group_name}")

    def add_uptake(self, uptake: NutrientUptake) -> None:
        """Add or update uptake parameters for a group.

        Parameters
        ----------
        uptake : NutrientUptake
            Nutrient uptake kinetics (typically for producers).
        """
        self.uptakes[uptake.group_name] = uptake
        logger.debug(f"Added uptake parameters for {uptake.group_name}")

    def validate(self) -> Tuple[bool, List[str]]:
        """Validate nutrient cycle consistency.

        Returns
        -------
        tuple[bool, list[str]]
            (is_valid, list of error/warning messages)
        """
        messages = []

        if not self.contents and not self.excretions and not self.uptakes:
            messages.append("NutrientCycle is empty")
            return len(messages) == 0, messages

        for name, content in self.contents.items():
            if not content.is_valid():
                messages.append(f"Invalid content for {name}")

        for name, excretion in self.excretions.items():
            if name not in self.contents:
                messages.append(
                    f"Excretion defined for {name} but no content"
                )

        for name, uptake in self.uptakes.items():
            if name not in self.contents:
                messages.append(f"Uptake defined for {name} but no content")

        is_valid = len(messages) == 0
        return is_valid, messages

    def calculate_mass_balance(
        self, rpath_params: RpathParams
    ) -> Dict[str, float]:
        """Calculate nutrient mass balance across groups.

        Parameters
        ----------
        rpath_params : RpathParams
            Ecopath model parameters with biomass and flow information.

        Returns
        -------
        dict
            Mass balance summary with keys like 'total_n_biomass',
            'total_p_biomass', 'balance_status'.
        """
        balance = {
            "total_n_biomass": 0.0,
            "total_p_biomass": 0.0,
            "total_c_biomass": 0.0,
            "groups_with_content": 0,
        }

        if not rpath_params.model.empty and "Biomass" in rpath_params.model.columns:
            for idx, row in rpath_params.model.iterrows():
                group = row.get("Group", "")
                biomass = row.get("Biomass", 0.0)

                if group in self.contents and biomass > 0:
                    content = self.contents[group]
                    balance["total_n_biomass"] += (
                        biomass * (content.nitrogen_percent / 100.0)
                    )
                    balance["total_p_biomass"] += (
                        biomass * (content.phosphorus_percent / 100.0)
                    )
                    balance["total_c_biomass"] += (
                        biomass * (content.carbon_percent / 100.0)
                    )
                    balance["groups_with_content"] += 1

        balance["balance_status"] = (
            "partial" if balance["groups_with_content"] > 0 else "undefined"
        )
        return balance

    def to_dataframe(self) -> pd.DataFrame:
        """Convert to pandas DataFrame representation.

        Returns
        -------
        pd.DataFrame
            DataFrame with one row per group, containing nutrient
            content, excretion, and uptake parameters.
        """
        rows = []
        all_groups = set(self.contents.keys()) | set(self.excretions.keys()) | set(
            self.uptakes.keys()
        )

        for group_name in sorted(all_groups):
            row = {"Group": group_name}

            if group_name in self.contents:
                content = self.contents[group_name]
                row.update(
                    {
                        "N_percent": content.nitrogen_percent,
                        "P_percent": content.phosphorus_percent,
                        "C_percent": content.carbon_percent,
                        "Ash_percent": content.ash_percent,
                    }
                )

            if group_name in self.excretions:
                excretion = self.excretions[group_name]
                row.update(
                    {
                        "N_excretion_rate": excretion.n_excretion_rate,
                        "P_excretion_rate": excretion.p_excretion_rate,
                        "Fecal_N_frac": excretion.fecal_n_fraction,
                        "Fecal_P_frac": excretion.fecal_p_fraction,
                    }
                )

            if group_name in self.uptakes:
                uptake = self.uptakes[group_name]
                row.update(
                    {
                        "N_uptake_rate": uptake.n_uptake_rate,
                        "P_uptake_rate": uptake.p_uptake_rate,
                        "Ks_N": uptake.half_saturation_n,
                        "Ks_P": uptake.half_saturation_p,
                    }
                )

            rows.append(row)

        return pd.DataFrame(rows) if rows else pd.DataFrame()

    @classmethod
    def from_dataframe(cls, df: pd.DataFrame) -> NutrientCycle:
        """Create from pandas DataFrame representation.

        Parameters
        ----------
        df : pd.DataFrame
            DataFrame with nutrient parameters, one row per group.

        Returns
        -------
        NutrientCycle
            Instantiated NutrientCycle object.
        """
        cycle = cls()

        for _, row in df.iterrows():
            group_name = row.get("Group", "")
            if not group_name:
                continue

            content_cols = ["N_percent", "P_percent", "C_percent", "Ash_percent"]
            if all(col in df.columns for col in content_cols):
                try:
                    content = NutrientContent(
                        group_name=group_name,
                        nitrogen_percent=float(row["N_percent"]),
                        phosphorus_percent=float(row["P_percent"]),
                        carbon_percent=float(row["C_percent"]),
                        ash_percent=float(row["Ash_percent"]),
                    )
                    cycle.add_content(content)
                except (ValueError, KeyError) as e:
                    logger.warning(f"Could not create content for {group_name}: {e}")

            excretion_cols = [
                "N_excretion_rate",
                "P_excretion_rate",
                "Fecal_N_frac",
                "Fecal_P_frac",
            ]
            if all(col in df.columns for col in excretion_cols):
                try:
                    excretion = NutrientExcretion(
                        group_name=group_name,
                        n_excretion_rate=float(row["N_excretion_rate"]),
                        p_excretion_rate=float(row["P_excretion_rate"]),
                        fecal_n_fraction=float(row["Fecal_N_frac"]),
                        fecal_p_fraction=float(row["Fecal_P_frac"]),
                    )
                    cycle.add_excretion(excretion)
                except (ValueError, KeyError) as e:
                    logger.warning(
                        f"Could not create excretion for {group_name}: {e}"
                    )

            uptake_cols = ["N_uptake_rate", "P_uptake_rate", "Ks_N", "Ks_P"]
            if all(col in df.columns for col in uptake_cols):
                try:
                    uptake = NutrientUptake(
                        group_name=group_name,
                        n_uptake_rate=float(row["N_uptake_rate"]),
                        p_uptake_rate=float(row["P_uptake_rate"]),
                        half_saturation_n=float(row["Ks_N"]),
                        half_saturation_p=float(row["Ks_P"]),
                    )
                    cycle.add_uptake(uptake)
                except (ValueError, KeyError) as e:
                    logger.warning(f"Could not create uptake for {group_name}: {e}")

        return cycle


def validate_nutrient_cycle(cycle: NutrientCycle) -> Tuple[bool, List[str]]:
    """Validate a complete nutrient cycling system.

    Parameters
    ----------
    cycle : NutrientCycle
        Nutrient cycle to validate.

    Returns
    -------
    tuple[bool, list[str]]
        (is_valid, list of error/warning messages)

    Notes
    -----
    This function performs comprehensive validation including:
    - Individual data integrity
    - Consistency across content, excretion, and uptake
    - Reasonable biological ranges
    """
    return cycle.validate()


def estimate_excretion_from_pb(
    pb_ratio: float, group_type: str = "consumer"
) -> Dict[str, float]:
    """Estimate excretion rates from Production/Biomass ratio.

    Parameters
    ----------
    pb_ratio : float
        Production/Biomass ratio (1/year)
    group_type : str, default 'consumer'
        Type of group ('consumer', 'producer', or 'detritus')

    Returns
    -------
    dict
        Estimated excretion parameters with keys:
        - n_excretion_rate (% of consumption per day)
        - p_excretion_rate (% of consumption per day)
        - fecal_n_fraction
        - fecal_p_fraction

    Notes
    -----
    This is a heuristic based on typical aquatic food web stoichiometry.
    Empirical calibration may be required for specific systems.
    """
    if pb_ratio <= 0:
        raise ValueError(f"pb_ratio must be > 0, got {pb_ratio}")

    estimates = {}

    if group_type.lower() == "consumer":
        estimates["n_excretion_rate"] = min(15.0, pb_ratio * 100)
        estimates["p_excretion_rate"] = min(3.0, pb_ratio * 20)
        estimates["fecal_n_fraction"] = 0.4
        estimates["fecal_p_fraction"] = 0.6

    elif group_type.lower() == "producer":
        estimates["n_excretion_rate"] = 2.0
        estimates["p_excretion_rate"] = 0.5
        estimates["fecal_n_fraction"] = 0.1
        estimates["fecal_p_fraction"] = 0.1

    elif group_type.lower() == "detritus":
        estimates["n_excretion_rate"] = 0.0
        estimates["p_excretion_rate"] = 0.0
        estimates["fecal_n_fraction"] = 0.0
        estimates["fecal_p_fraction"] = 0.0

    else:
        raise ValueError(
            f"Unknown group_type: {group_type}. "
            f"Must be 'consumer', 'producer', or 'detritus'."
        )

    return estimates
