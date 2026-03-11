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
