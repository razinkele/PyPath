"""Coupled Ecosim-Ecospace state-vector utilities."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Callable, Optional

import numpy as np

if TYPE_CHECKING:
    from pypath.core.ecosim import RsimOutput, RsimScenario
    from pypath.spatial.ecospace_params import EcospaceParams


@dataclass(frozen=True)
class CoupledStateVector:
    """Flattened coupled state vector representation.

    Parameters
    ----------
    values : np.ndarray
        Flattened state vector with length ``(n_groups + 1) * n_patches``.
    n_groups : int
        Number of modeled groups (excluding Outside index).
    n_patches : int
        Number of spatial patches.
    """

    values: np.ndarray
    n_groups: int
    n_patches: int

    def __post_init__(self):
        values = np.asarray(self.values, dtype=float).reshape(-1)
        expected = (self.n_groups + 1) * self.n_patches
        if values.size != expected:
            raise ValueError(
                f"State vector length {values.size} does not match expected {expected}"
            )
        if np.any(values < 0):
            raise ValueError("State vector values must be non-negative")
        object.__setattr__(self, "values", values)

    def to_spatial_biomass(self) -> np.ndarray:
        """Return spatial biomass matrix ``[n_groups+1, n_patches]``."""
        return self.values.reshape(self.n_groups + 1, self.n_patches).copy()

    def to_total_biomass(self) -> np.ndarray:
        """Collapse spatial biomass to total biomass by group."""
        return self.to_spatial_biomass().sum(axis=1)

    @classmethod
    def from_spatial_biomass(cls, biomass_spatial: np.ndarray) -> "CoupledStateVector":
        """Create coupled vector from spatial biomass matrix."""
        arr = np.asarray(biomass_spatial, dtype=float)
        if arr.ndim != 2:
            raise ValueError("biomass_spatial must be 2D [n_groups+1, n_patches]")
        n_groups_plus_one, n_patches = arr.shape
        return cls(
            values=arr.reshape(-1), n_groups=n_groups_plus_one - 1, n_patches=n_patches
        )


def build_initial_state_vector(
    total_biomass: np.ndarray,
    n_patches: int,
    habitat_capacity: Optional[np.ndarray] = None,
) -> CoupledStateVector:
    """Build an initial coupled state vector from group-level biomass totals.

    Parameters
    ----------
    total_biomass : np.ndarray
        Group-level biomass vector ``[n_groups+1]``.
    n_patches : int
        Number of patches.
    habitat_capacity : np.ndarray, optional
        Capacity matrix ``[n_groups, n_patches]`` used to weight initial
        distribution for living/detritus groups. Outside group (index 0)
        is always distributed uniformly.
    """
    totals = np.asarray(total_biomass, dtype=float)
    if totals.ndim != 1:
        raise ValueError("total_biomass must be 1D")
    if np.any(totals < 0):
        raise ValueError("total_biomass must be non-negative")
    if n_patches <= 0:
        raise ValueError(f"n_patches must be positive, got {n_patches}")

    n_groups_plus_one = totals.size
    biomass_spatial = np.zeros((n_groups_plus_one, n_patches), dtype=float)

    # Outside / boundary pool
    biomass_spatial[0, :] = totals[0] / n_patches

    if habitat_capacity is None:
        biomass_spatial[1:, :] = totals[1:, np.newaxis] / n_patches
    else:
        cap = np.asarray(habitat_capacity, dtype=float)
        if cap.shape != (n_groups_plus_one - 1, n_patches):
            raise ValueError(
                f"habitat_capacity shape {cap.shape} must equal "
                f"({n_groups_plus_one - 1}, {n_patches})"
            )
        cap = np.clip(cap, 0.0, None)
        row_sum = cap.sum(axis=1, keepdims=True)
        row_sum[row_sum <= 0.0] = 1.0
        shares = cap / row_sum
        biomass_spatial[1:, :] = totals[1:, np.newaxis] * shares

    return CoupledStateVector.from_spatial_biomass(biomass_spatial)


def run_coupled_simulation(
    scenario: "RsimScenario",
    ecospace: "EcospaceParams",
    years: Optional[range] = None,
    *,
    state_vector: Optional[np.ndarray] = None,
    store_state_vectors: bool = False,
    method: str = "RK4",
    mpa=None,
    spatial_fishing=None,
    environmental_drivers=None,
    postprocess: Optional[Callable[["RsimOutput"], None]] = None,
):
    """Run a coupled Ecosim-Ecospace simulation using state-vector IO."""
    from pypath.spatial.integration import rsim_run_spatial

    result = rsim_run_spatial(
        scenario,
        method=method,
        years=years,
        ecospace=ecospace,
        environmental_drivers=environmental_drivers,
        mpa=mpa,
        spatial_fishing=spatial_fishing,
        initial_state_vector=state_vector,
        store_state_vectors=store_state_vectors,
    )
    if postprocess is not None:
        postprocess(result)
    return result
