"""
Parameter data structures for PyPath.

This module contains the RpathParams class and functions for creating,
reading, writing, and validating Ecopath parameter files.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional, Union

import numpy as np
import pandas as pd


@dataclass
class StanzaParams:
    """Parameters for multi-stanza (age-structured) groups.

    Attributes
    ----------
    n_stanza_groups : int
        Number of stanza group sets (e.g., juvenile + adult = 1 set)
    stgroups : pd.DataFrame
        Stanza group parameters (VBGF_Ksp, VBGF_d, Wmat, etc.)
    stindiv : pd.DataFrame
        Individual stanza parameters (First, Last, Z, Leading)
    """

    n_stanza_groups: int = 0
    stgroups: Optional[pd.DataFrame] = None
    stindiv: Optional[pd.DataFrame] = None


@dataclass
class RpathParams:
    """Container for Rpath model parameters.

    This class holds all parameters needed to create a balanced Ecopath model.

    Attributes
    ----------
    model : pd.DataFrame
        Basic parameters for each group including:
        - Group: Group name
        - Type: 0=consumer, 1=producer, 2=detritus, 3=fleet
        - Biomass: Biomass (t/km²)
        - PB: Production/Biomass ratio (1/year)
        - QB: Consumption/Biomass ratio (1/year)
        - EE: Ecotrophic efficiency
        - ProdCons: Production/Consumption ratio (GE)
        - BioAcc: Biomass accumulation rate
        - Unassim: Unassimilated consumption fraction
        - DetInput: Detrital input (for detritus groups)
        Plus columns for detritus fate and landings/discards by fleet.

    diet : pd.DataFrame
        Diet composition matrix where rows are prey (including Import)
        and columns are predators. Values are fractions (0-1).

    stanzas : StanzaParams
        Multi-stanza (age-structured) group parameters.

    pedigree : pd.DataFrame
        Data quality/pedigree information for parameters.

    remarks : pd.DataFrame
        Comments/remarks for parameter values. Has same structure as model
        with string values containing remarks for each cell.

    Examples
    --------
    >>> params = create_rpath_params(
    ...     groups=['Phyto', 'Zoo', 'Fish', 'Detritus', 'Fleet'],
    ...     types=[1, 0, 0, 2, 3]
    ... )
    >>> params.model['Biomass'] = [10.0, 5.0, 2.0, 100.0, np.nan]
    """

    model: pd.DataFrame
    diet: pd.DataFrame
    stanzas: StanzaParams = field(default_factory=StanzaParams)
    pedigree: Optional[pd.DataFrame] = None
    remarks: Optional[pd.DataFrame] = None

    def __repr__(self) -> str:
        n_groups = len(self.model)
        n_living = len(self.model[self.model["Type"] <= 1])
        n_dead = len(self.model[self.model["Type"] == 2])
        n_fleet = len(self.model[self.model["Type"] == 3])
        return (
            f"RpathParams(\n"
            f"  groups={n_groups} (living={n_living}, detritus={n_dead}, fleets={n_fleet})\n"
            f"  stanzas={self.stanzas.n_stanza_groups}\n"
            f")"
        )


def create_rpath_params(
    groups: List[str], types: List[int], stgroups: Optional[List[str]] = None
) -> RpathParams:
    """Create a shell RpathParams object with empty parameter values.

    Creates the basic structure for an Ecopath model that can be filled
    in with actual parameter values.

    Parameters
    ----------
    groups : list of str
        Names of all groups in the model (living, detritus, and fleets).
    types : list of int
        Type code for each group:
        - 0: Consumer
        - 1: Primary producer (or value 0-1 for mixotrophs)
        - 2: Detritus
        - 3: Fleet/fishery
    stgroups : list of str, optional
        Stanza group assignment for each group. Use None for non-stanza groups.
        Groups with the same stanza group name will be linked (e.g., juvenile/adult).

    Returns
    -------
    RpathParams
        Parameter object with NA values ready to be filled in.

    Examples
    --------
    >>> params = create_rpath_params(
    ...     groups=['Phyto', 'Zoo', 'SmallFish', 'LargeFish', 'Detritus', 'Fleet'],
    ...     types=[1, 0, 0, 0, 2, 3],
    ...     stgroups=[None, None, 'Fish', 'Fish', None, None]
    ... )
    """
    if len(groups) != len(types):
        raise ValueError("groups and types must have the same length")

    n_groups = len(groups)

    # Identify group types
    pred_groups = [g for g, t in zip(groups, types) if t < 2]  # Consumers/producers
    prey_groups = [g for g, t in zip(groups, types) if t < 3]  # All except fleets
    det_groups = [g for g, t in zip(groups, types) if t == 2]
    fleet_groups = [g for g, t in zip(groups, types) if t == 3]

    # Create model DataFrame
    model_data = {
        "Group": groups,
        "Type": types,
        "Biomass": [np.nan] * n_groups,
        "PB": [np.nan] * n_groups,
        "QB": [np.nan] * n_groups,
        "EE": [np.nan] * n_groups,
        "ProdCons": [np.nan] * n_groups,
        "BioAcc": [np.nan] * n_groups,
        "Unassim": [np.nan] * n_groups,
        "DetInput": [np.nan] * n_groups,
    }

    # Add detrital fate columns
    for det in det_groups:
        model_data[det] = [np.nan] * n_groups
        # Set DetInput to 0 for detritus groups
        for i, t in enumerate(types):
            if t == 2:
                model_data["DetInput"][i] = 0.0

    # Add landing and discard columns for each fleet
    n_bio = len([t for t in types if t < 3])  # Non-fleet groups
    for fleet in fleet_groups:
        # Landings
        model_data[fleet] = [0.0] * n_bio + [np.nan] * len(fleet_groups)
        # Discards
        model_data[f"{fleet}.disc"] = [0.0] * n_bio + [np.nan] * len(fleet_groups)

    model = pd.DataFrame(model_data)

    # Create diet DataFrame
    diet_data = {"Group": prey_groups + ["Import"]}
    for pred in pred_groups:
        diet_data[pred] = [np.nan] * (len(prey_groups) + 1)
    diet = pd.DataFrame(diet_data)

    # Create stanza parameters if provided
    stanza_params = StanzaParams()
    if stgroups is not None and any(s is not None for s in stgroups):
        # Get unique stanza groups
        unique_stgroups = sorted(set(s for s in stgroups if s is not None))
        n_stanza_groups = len(unique_stgroups)

        # Count stanzas per group
        nstanzas = [sum(1 for s in stgroups if s == sg) for sg in unique_stgroups]

        stgroups_df = pd.DataFrame(
            {
                "StGroupNum": range(1, n_stanza_groups + 1),
                "StanzaGroup": unique_stgroups,
                "nstanzas": nstanzas,
                "VBGF_Ksp": [np.nan] * n_stanza_groups,
                "VBGF_d": [0.66667] * n_stanza_groups,
                "Wmat": [np.nan] * n_stanza_groups,
                "BAB": [0.0] * n_stanza_groups,
                "RecPower": [1.0] * n_stanza_groups,
            }
        )

        # Individual stanza records
        stindiv_records = []
        for i, (g, t, sg) in enumerate(zip(groups, types, stgroups)):
            if sg is not None:
                st_group_num = unique_stgroups.index(sg) + 1
                stindiv_records.append(
                    {
                        "StGroupNum": st_group_num,
                        "StanzaNum": 0,  # Will be assigned later
                        "GroupNum": i + 1,
                        "Group": g,
                        "First": np.nan,
                        "Last": np.nan,
                        "Z": np.nan,
                        "Leading": np.nan,
                    }
                )

        stindiv_df = pd.DataFrame(stindiv_records)

        stanza_params = StanzaParams(
            n_stanza_groups=n_stanza_groups, stgroups=stgroups_df, stindiv=stindiv_df
        )

    # Create pedigree DataFrame
    pedigree_data = {
        "Group": groups,
        "Biomass": [1.0] * n_groups,
        "PB": [1.0] * n_groups,
        "QB": [1.0] * n_groups,
        "Diet": [1.0] * n_groups,
    }
    # Add fleet pedigree columns
    for fleet in fleet_groups:
        pedigree_data[fleet] = [1.0] * n_groups
    pedigree = pd.DataFrame(pedigree_data)

    return RpathParams(model=model, diet=diet, stanzas=stanza_params, pedigree=pedigree)


def read_rpath_params(
    model_file: Union[str, Path],
    diet_file: Union[str, Path],
    pedigree_file: Optional[Union[str, Path]] = None,
    stanza_group_file: Optional[Union[str, Path]] = None,
    stanza_file: Optional[Union[str, Path]] = None,
) -> RpathParams:
    """Read Rpath parameters from CSV files.

    Parameters
    ----------
    model_file : str or Path
        Path to CSV file with model parameters.
    diet_file : str or Path
        Path to CSV file with diet composition matrix.
    pedigree_file : str or Path, optional
        Path to CSV file with pedigree information.
    stanza_group_file : str or Path, optional
        Path to CSV file with stanza group parameters.
    stanza_file : str or Path, optional
        Path to CSV file with individual stanza parameters.

    Returns
    -------
    RpathParams
        Parameter object populated from files.
    """
    model = pd.read_csv(model_file)
    diet = pd.read_csv(diet_file)

    # Read stanza files if provided
    stanza_params = StanzaParams()
    if stanza_group_file is not None and stanza_file is not None:
        stgroups = pd.read_csv(stanza_group_file)
        stindiv = pd.read_csv(stanza_file)
        stanza_params = StanzaParams(
            n_stanza_groups=len(stgroups), stgroups=stgroups, stindiv=stindiv
        )

    # Read pedigree if provided
    pedigree = None
    if pedigree_file is not None:
        pedigree = pd.read_csv(pedigree_file)
    else:
        # Create default pedigree
        fleet_groups = model[model["Type"] == 3]["Group"].tolist()
        pedigree_data = {
            "Group": model["Group"].tolist(),
            "B": [1.0] * len(model),
            "PB": [1.0] * len(model),
            "QB": [1.0] * len(model),
            "Diet": [1.0] * len(model),
        }
        for fleet in fleet_groups:
            pedigree_data[fleet] = [1.0] * len(model)
        pedigree = pd.DataFrame(pedigree_data)

    return RpathParams(model=model, diet=diet, stanzas=stanza_params, pedigree=pedigree)


def write_rpath_params(
    params: RpathParams, eco_name: str, path: Union[str, Path] = ""
) -> None:
    """Write Rpath parameters to CSV files.

    Parameters
    ----------
    params : RpathParams
        Parameter object to write.
    eco_name : str
        Ecosystem name used in file names.
    path : str or Path
        Directory path for output files.
    """
    path = Path(path)

    params.model.to_csv(path / f"{eco_name}_model.csv", index=False)
    params.diet.to_csv(path / f"{eco_name}_diet.csv", index=False)

    if params.pedigree is not None:
        params.pedigree.to_csv(path / f"{eco_name}_pedigree.csv", index=False)

    if params.stanzas.n_stanza_groups > 0:
        params.stanzas.stgroups.to_csv(
            path / f"{eco_name}_stanza_groups.csv", index=False
        )
        params.stanzas.stindiv.to_csv(path / f"{eco_name}_stanzas.csv", index=False)


def check_rpath_params(params: RpathParams) -> bool:
    """Check Rpath parameter files for consistency.

    Validates that parameter files are filled out correctly and data
    is in the expected locations.

    Parameters
    ----------
    params : RpathParams
        Parameter object to validate.

    Returns
    -------
    bool
        True if parameters are valid, False otherwise.

    Raises
    ------
    warnings.warn
        For each validation issue found.
    """
    model = params.model
    diet = params.diet

    n_warnings = 0

    # Check that all types are represented
    if len(model[model["Type"] == 0]) == 0:
        warnings.warn("Model must contain at least 1 consumer")
        n_warnings += 1

    if len(model[model["Type"] == 1]) == 0:
        warnings.warn("Model must contain a producer group")
        n_warnings += 1

    if len(model[model["Type"] == 2]) == 0:
        warnings.warn("Model must contain at least 1 detrital group")
        n_warnings += 1

    if len(model[model["Type"] == 3]) == 0:
        warnings.warn("Model must contain at least 1 fleet")
        n_warnings += 1

    # Check that either Biomass or EE is provided for living groups
    living = model[model["Type"] < 2]
    missing_both = living[living["Biomass"].isna() & living["EE"].isna()]
    if len(missing_both) > 0:
        groups = missing_both["Group"].tolist()
        warnings.warn(f"Groups missing both Biomass and EE: {groups}")
        n_warnings += 1

    # Check that consumers have QB or ProdCons
    consumers = model[model["Type"] < 1]
    missing_qb = consumers[consumers["QB"].isna() & consumers["ProdCons"].isna()]
    if len(missing_qb) > 0:
        groups = missing_qb["Group"].tolist()
        warnings.warn(f"Consumers missing both QB and ProdCons: {groups}")
        n_warnings += 1

    # Check diet columns sum to ~1 for consumers
    _n_living = len(model[model["Type"] <= 1])
    pred_groups = model[model["Type"] < 2]["Group"].tolist()

    for pred in pred_groups:
        if pred in diet.columns:
            col_sum = diet[pred].sum()
            pred_type = model[model["Group"] == pred]["Type"].values[0]
            expected = 1.0 - pred_type  # Producers have diet = 0
            if not np.isclose(col_sum, expected, atol=0.01) and not np.isnan(col_sum):
                warnings.warn(
                    f"Diet column '{pred}' sums to {col_sum:.3f}, expected ~{expected}"
                )
                n_warnings += 1

    # Check Import row exists
    if "Import" not in diet["Group"].values and "import" not in diet["Group"].values:
        warnings.warn("Diet matrix is missing the Import row")
        n_warnings += 1

    if n_warnings == 0:
        print("Rpath parameter file is functional.")
        return True
    else:
        print(f"Rpath parameter file needs attention! ({n_warnings} warnings)")
        return False
