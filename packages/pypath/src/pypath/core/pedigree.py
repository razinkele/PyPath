"""Pedigree-based parameter distributions and sampling.

Pedigree values (coefficients of variation) define parameter uncertainty.
This module converts pedigree CVs to statistical distributions and generates
parameter samples for Monte Carlo analysis.
"""
from __future__ import annotations

import copy
import logging
import math
import warnings
from dataclasses import dataclass, field
from typing import Union

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


@dataclass
class ScalarDistribution:
    """A single scalar parameter's sampling distribution (log-normal).

    Parameters
    ----------
    param_name : str
        Parameter name (e.g. "Biomass", "PB", "QB").
    group_idx : int
        0-based group index.
    base_value : float
        Current Ecopath value.
    cv : float
        Coefficient of variation from pedigree.
    bounds : tuple[float, float] | None
        Optional hard bounds for rejection sampling.
    """

    param_name: str
    group_idx: int
    base_value: float
    cv: float
    bounds: tuple[float, float] | None = None


@dataclass
class DietDistribution:
    """A predator's diet composition distribution (Dirichlet).

    Parameters
    ----------
    pred_idx : int
        0-based predator group index.
    base_proportions : np.ndarray
        Current diet column (prey proportions, sum=1).
    cv : float
        Controls Dirichlet concentration (higher CV = more spread).
    """

    pred_idx: int
    base_proportions: np.ndarray
    cv: float


ParameterDistribution = Union[ScalarDistribution, DietDistribution]


@dataclass
class PedigreeConfig:
    """Configuration for pedigree-to-CV mapping.

    EwE 6 stores pedigree as (VarName, LevelID) pairs in the Pedigree table,
    where each VarName has its own set of levels with IndexValue (the CV).

    In the Python API, params.pedigree values are treated as CVs directly.
    PedigreeConfig is only needed when importing from EwE databases.
    """

    level_to_cv: dict[str, dict[int, float]] = field(default_factory=dict)


def build_distributions(
    params: "RpathParams",
    config: PedigreeConfig | None = None,
) -> list[ParameterDistribution]:
    """Build parameter distributions from pedigree CVs.

    Parameters
    ----------
    params : RpathParams
        Ecopath parameters with pedigree DataFrame.
    config : PedigreeConfig, optional
        EwE database pedigree mapping (for converting LevelIDs to CVs).

    Returns
    -------
    list[ParameterDistribution]
        Distributions for all parameters with CV > 0.
    """
    from pypath.core.params import RpathParams

    pedigree = params.pedigree
    if pedigree is None:
        return []

    # Warn if all pedigree values are default 1.0
    numeric_cols = [c for c in pedigree.columns if c != "Group"]
    all_vals = pedigree[numeric_cols].values.flatten()
    all_vals = all_vals[~np.isnan(all_vals.astype(float))]
    if len(all_vals) > 0 and np.allclose(all_vals, 1.0):
        warnings.warn(
            "All pedigree values are 1.0 (default = 100% CV). "
            "Consider setting pedigree values before MC analysis.",
            UserWarning,
            stacklevel=2,
        )

    model = params.model
    distributions: list[ParameterDistribution] = []

    # Scalar parameters: Biomass, PB, QB
    scalar_params = ["Biomass", "PB", "QB"]
    for param_name in scalar_params:
        if param_name not in pedigree.columns:
            continue
        for idx in range(len(model)):
            group_type = model.loc[idx, "Type"]
            # Skip detritus (type=2) for PB/QB
            if group_type == 2 and param_name in ("PB", "QB"):
                continue
            # Skip fleets (type=3)
            if group_type == 3:
                continue
            # Producers don't have QB
            if group_type == 1 and param_name == "QB":
                continue

            cv = float(pedigree.loc[idx, param_name])
            if np.isnan(cv) or cv <= 0:
                continue

            base_val = model.loc[idx, param_name]
            if np.isnan(base_val) or base_val <= 0:
                continue

            distributions.append(ScalarDistribution(
                param_name=param_name,
                group_idx=idx,
                base_value=float(base_val),
                cv=cv,
            ))

    # Diet distributions: one per consumer with Diet CV > 0
    if "Diet" in pedigree.columns:
        consumer_mask = model["Type"] == 0
        for idx in range(len(model)):
            if not consumer_mask.iloc[idx]:
                continue
            cv = float(pedigree.loc[idx, "Diet"])
            if np.isnan(cv) or cv <= 0:
                continue

            group_name = model.loc[idx, "Group"]
            if group_name not in params.diet.columns:
                continue
            diet_col = params.diet[group_name].values.astype(float)
            # Only include if diet has non-zero entries
            if np.nansum(diet_col) <= 0:
                continue

            distributions.append(DietDistribution(
                pred_idx=idx,
                base_proportions=diet_col,
                cv=cv,
            ))

    return distributions
