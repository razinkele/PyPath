"""Ecotracer: contaminant tracking through the food web.

Tracks contaminant concentrations alongside Ecosim biomass dynamics.
Each group has initial concentration, environmental/immigration inputs,
decay, assimilation, and metabolism rates.
"""
from __future__ import annotations

import logging
import math
from dataclasses import dataclass

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class EcotracerParams:
    """Per-group tracer parameters.

    All arrays are 0-based, length n_groups = NUM_LIVING + NUM_DEAD.
    No fleets, no padding column.

    Parameters
    ----------
    czero : np.ndarray
        Initial concentration per group (n_groups,).
    cenv : np.ndarray
        Environmental input concentration (n_groups,).
    cimmig : np.ndarray
        Immigration input concentration (n_groups,).
    cdecay : np.ndarray
        Decay rate (n_groups,).
    cassim : np.ndarray
        Assimilation proportion, 0-1 (n_groups,).
    cmetab : np.ndarray
        Metabolism loss rate (n_groups,).
    """

    czero: np.ndarray
    cenv: np.ndarray
    cimmig: np.ndarray
    cdecay: np.ndarray
    cassim: np.ndarray
    cmetab: np.ndarray


@dataclass
class EcotracerResult:
    """Output time series from Ecotracer simulation.

    Parameters
    ----------
    out_Conc : np.ndarray
        Monthly concentrations (n_months+1, n_groups). Index 0 is initial state.
    annual_Conc : np.ndarray
        Annual average concentrations (n_years, n_groups).
    group_names : list[str]
        Group name labels.
    """

    out_Conc: np.ndarray
    annual_Conc: np.ndarray
    group_names: list[str]


def create_ecotracer_params(n_groups: int) -> EcotracerParams:
    """Create EcotracerParams with sensible defaults.

    Defaults: czero=0, cenv=0, cimmig=0, cdecay=0, cassim=1.0, cmetab=0.
    """
    return EcotracerParams(
        czero=np.zeros(n_groups),
        cenv=np.zeros(n_groups),
        cimmig=np.zeros(n_groups),
        cdecay=np.zeros(n_groups),
        cassim=np.ones(n_groups),
        cmetab=np.zeros(n_groups),
    )


_BIOMASS_THRESHOLD = 1e-10


def ecotracer_deriv(
    conc: np.ndarray,
    biomass: np.ndarray,
    Q_matrix: np.ndarray,
    params: EcotracerParams,
    detritus_fate: np.ndarray | None = None,
    n_living: int = 0,
) -> np.ndarray:
    """Compute dC/dt for all groups.

    Parameters
    ----------
    conc : np.ndarray
        Current concentrations (n_groups,).
    biomass : np.ndarray
        Current biomass (n_groups,).
    Q_matrix : np.ndarray
        Consumption matrix Q[prey, pred] (n_groups, n_groups), 0-based.
    params : EcotracerParams
        Tracer parameters.
    detritus_fate : np.ndarray, optional
        Detritus fate fractions (n_living, n_detritus). When None, detritus
        only decays.
    n_living : int
        Number of living groups (groups 0..n_living-1 are living,
        n_living..n_groups-1 are detritus).

    Returns
    -------
    np.ndarray
        dC/dt for each group (n_groups,).
    """
    n_groups = len(conc)
    deriv = np.zeros(n_groups)

    # Living groups: dietary intake + environmental inputs - losses
    for i in range(n_living):
        # Dietary intake: cassim_i * sum_j(Q[j, i] * C[j]) / B_i
        if biomass[i] > _BIOMASS_THRESHOLD:
            dietary_intake = params.cassim[i] * np.dot(Q_matrix[:, i], conc) / biomass[i]
        else:
            dietary_intake = 0.0

        deriv[i] = (
            dietary_intake
            + params.cenv[i]
            + params.cimmig[i]
            - (params.cdecay[i] + params.cmetab[i]) * conc[i]
        )

    # Detritus groups: contaminant from dead matter + cenv - decay
    for i in range(n_living, n_groups):
        det_input = 0.0
        if detritus_fate is not None:
            det_idx = i - n_living
            if det_idx < detritus_fate.shape[1]:
                # Weighted average of contributor concentrations
                for j in range(n_living):
                    det_input += detritus_fate[j, det_idx] * conc[j]

        deriv[i] = (
            det_input
            + params.cenv[i]
            + params.cimmig[i]
            - (params.cdecay[i] + params.cmetab[i]) * conc[i]
        )

    return deriv


def ecotracer_step(
    conc: np.ndarray,
    biomass: np.ndarray,
    Q_matrix: np.ndarray,
    params: EcotracerParams,
    dt: float,
    detritus_fate: np.ndarray | None = None,
    n_living: int = 0,
) -> np.ndarray:
    """Analytic update for tracer concentration (unconditionally stable).

    For each group i:
      input_i = dietary_intake_i + cenv_i + cimmig_i
      loss_rate_i = cdecay_i + cmetab_i
      if loss_rate_i > 0:
          C_i(t+dt) = input_i/loss_rate_i + (C_i - input_i/loss_rate_i) * exp(-loss_rate_i*dt)
      else:
          C_i(t+dt) = C_i + input_i * dt

    Parameters
    ----------
    conc : np.ndarray
        Current concentrations (n_groups,).
    biomass : np.ndarray
        Current biomass (n_groups,).
    Q_matrix : np.ndarray
        Consumption matrix Q[prey, pred] (n_groups, n_groups), 0-based.
    params : EcotracerParams
        Tracer parameters.
    dt : float
        Timestep (typically 1/12 for monthly).
    detritus_fate : np.ndarray, optional
        Detritus fate fractions (n_living, n_detritus).
    n_living : int
        Number of living groups.

    Returns
    -------
    np.ndarray
        Updated concentrations (n_groups,), clamped to >= 0.
    """
    n_groups = len(conc)
    new_conc = np.zeros(n_groups)

    # Compute instantaneous inputs for each group
    for i in range(n_groups):
        # Dietary intake (living groups only)
        if i < n_living and biomass[i] > _BIOMASS_THRESHOLD:
            dietary_intake = params.cassim[i] * np.dot(Q_matrix[:, i], conc) / biomass[i]
        elif i >= n_living:
            # Detritus input
            dietary_intake = 0.0
            if detritus_fate is not None:
                det_idx = i - n_living
                if det_idx < detritus_fate.shape[1]:
                    for j in range(n_living):
                        dietary_intake += detritus_fate[j, det_idx] * conc[j]
        else:
            dietary_intake = 0.0

        total_input = dietary_intake + params.cenv[i] + params.cimmig[i]
        loss_rate = params.cdecay[i] + params.cmetab[i]

        if loss_rate > 0:
            # Analytic solution: exact for constant input within timestep
            equilibrium = total_input / loss_rate
            new_conc[i] = equilibrium + (conc[i] - equilibrium) * math.exp(-loss_rate * dt)
        else:
            # No loss: simple linear accumulation
            new_conc[i] = conc[i] + total_input * dt

    # Clamp to non-negative
    np.clip(new_conc, 0.0, None, out=new_conc)
    return new_conc
