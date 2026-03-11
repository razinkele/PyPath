"""Ecosim calibration via sum-of-squares fitting to time series data.

Provides fit_to_timeseries() for optimizing Ecosim vulnerability (VV)
and primary production (PP) parameters against observed time series.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from pypath.core.ecosim import RsimScenario
    from pypath.core.timeseries import EweTimeSeriesCollection

logger = logging.getLogger(__name__)


@dataclass
class CalibrationResult:
    """Result of time series calibration.

    Parameters
    ----------
    best_vv : np.ndarray
        Fitted vulnerability values, one per pred-prey link.
    best_pp : np.ndarray or None
        Fitted primary production anomaly (if fit_pp was True).
    ss : float
        Final sum-of-squares value.
    ss_by_group : dict[int, float]
        SS contribution per group index (0-based).
    n_iterations : int
        Number of optimizer iterations performed.
    converged : bool
        Whether the optimizer converged.
    fitted_scenario : RsimScenario or None
        Scenario with best-fit parameters applied.
    link_map : list[tuple[int, int]]
        (prey_idx, pred_idx) 0-based for each entry in best_vv.
    """

    best_vv: np.ndarray
    best_pp: np.ndarray | None
    ss: float
    ss_by_group: dict[int, float]
    n_iterations: int
    converged: bool
    fitted_scenario: object
    link_map: list[tuple[int, int]]


def _compute_ss(
    observed: dict[int, np.ndarray],
    predicted: dict[int, np.ndarray],
    weights: dict[int, float],
    relative: dict[int, bool],
) -> tuple[float, dict[int, float]]:
    """Compute EwE-standard log-ratio sum-of-squares.

    SS = sum_groups( weight_i * sum_t( (log(pred/obs))^2 ) )
    """
    total_ss = 0.0
    ss_by_group: dict[int, float] = {}

    for grp_idx, obs in observed.items():
        if grp_idx not in predicted:
            continue
        pred = predicted[grp_idx]
        w = weights.get(grp_idx, 1.0)

        n = min(len(obs), len(pred))
        obs_n = obs[:n]
        pred_n = pred[:n]

        valid = ~np.isnan(obs_n) & ~np.isnan(pred_n) & (obs_n > 0) & (pred_n > 0)

        if not np.any(valid):
            ss_by_group[grp_idx] = 0.0
            continue

        obs_v = obs_n[valid]
        pred_v = pred_n[valid]

        if relative.get(grp_idx, False):
            scale = np.mean(obs_v) / np.mean(pred_v)
            pred_v = pred_v * scale

        log_ratios = np.log(pred_v / obs_v)
        group_ss = w * np.sum(log_ratios ** 2)
        ss_by_group[grp_idx] = group_ss
        total_ss += group_ss

    return total_ss, ss_by_group
