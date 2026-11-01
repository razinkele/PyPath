"""Marine Protected Area (MPA) support for Ecospace.

Defines MPA zones with fleet-selective, temporally-dynamic closures
and optional habitat capacity bonuses.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class MPAZone:
    """A single Marine Protected Area zone.

    Parameters
    ----------
    mpa_id : int
        Unique identifier.
    name : str
        Human-readable name.
    patches : list[int]
        0-based patch indices covered by this MPA.
    start_month : int
        Month when MPA activates (0 = from simulation start).
    end_month : int or None
        Month when MPA deactivates (None = permanent).
    excluded_fleets : list[int] or None
        0-based fleet indices excluded. None = all fleets (no-take).
    capacity_bonus : float
        Habitat capacity multiplier for patches (1.0 = no bonus).
    """

    mpa_id: int
    name: str
    patches: list[int]
    start_month: int = 0
    end_month: int | None = None
    excluded_fleets: list[int] | None = None
    capacity_bonus: float = 1.0


@dataclass
class MPAConfig:
    """Collection of MPA zones with query and mask interfaces."""

    zones: list[MPAZone] = field(default_factory=list)

    def get_active_zones(self, month: int) -> list[MPAZone]:
        """Return zones active at the given month."""
        active = []
        for z in self.zones:
            if z.start_month <= month and (z.end_month is None or month < z.end_month):
                active.append(z)
        return active

    def is_closed(self, patch: int, fleet: int, month: int) -> bool:
        """Check if a patch is closed to a fleet at a given month."""
        for z in self.get_active_zones(month):
            if patch in z.patches:
                if z.excluded_fleets is None or fleet in z.excluded_fleets:
                    return True
        return False

    def get_effort_mask(self, n_patches: int, n_fleets: int, month: int) -> np.ndarray:
        """Return (n_patches, n_fleets) float mask. 1.0 = open, 0.0 = closed."""
        mask = np.ones((n_patches, n_fleets), dtype=np.float64)
        for z in self.get_active_zones(month):
            for p in z.patches:
                if p < 0 or p >= n_patches:
                    logger.warning(
                        "MPA '%s': patch %d out of range [0, %d), skipped",
                        z.name,
                        p,
                        n_patches,
                    )
                    continue
                if z.excluded_fleets is None:
                    mask[p, :] = 0.0
                else:
                    for f in z.excluded_fleets:
                        if 0 <= f < n_fleets:
                            mask[p, f] = 0.0
        return mask

    def get_capacity_multipliers(self, n_patches: int, month: int) -> np.ndarray:
        """Return (n_patches,) capacity multiplier array."""
        mult = np.ones(n_patches, dtype=np.float64)
        for z in self.get_active_zones(month):
            if z.capacity_bonus != 1.0:
                for p in z.patches:
                    if 0 <= p < n_patches:
                        mult[p] *= z.capacity_bonus
        return mult


def create_mpa_config(zones: list[MPAZone] | None = None) -> MPAConfig:
    """Create MPAConfig, defaulting to empty zones list."""
    return MPAConfig(zones=zones if zones is not None else [])


# Phase 1C.2f reserve network extensions


@dataclass
class MarineProtectedArea:
    """Simplified MPA descriptor for network-level analysis."""

    name: str
    patch_indices: list[int]
    protection_level: float = 1.0


@dataclass
class ReserveNetwork:
    """Collection of MPAs with simple network metrics."""

    reserves: list[MarineProtectedArea] = field(default_factory=list)

    def coverage_fraction(self, n_patches: int) -> float:
        """Compute fraction of patches under protection."""
        if n_patches <= 0:
            return 0.0
        protected = {p for r in self.reserves for p in r.patch_indices}
        return len(protected) / n_patches

    def average_protection_level(self) -> float:
        """Mean protection level across reserves."""
        if not self.reserves:
            return 0.0
        return float(np.mean([r.protection_level for r in self.reserves]))

    def connectivity_score(self, adjacency: np.ndarray) -> float:
        """Score connectedness of protected patches."""
        if not self.reserves:
            return 0.0
        protected = sorted({p for r in self.reserves for p in r.patch_indices})
        if not protected:
            return 0.0
        adj = np.asarray(adjacency, dtype=float)
        sub = adj[np.ix_(protected, protected)]
        if sub.size == 0:
            return 0.0
        return float(sub.sum() / max(len(protected), 1))


def mpa_spillover_index(
    inside_biomass: np.ndarray,
    outside_biomass: np.ndarray,
    exchange_rate: float,
) -> float:
    """Compute biomass spillover index from protected to fished area."""
    inside = float(np.mean(np.asarray(inside_biomass, dtype=float)))
    outside = float(np.mean(np.asarray(outside_biomass, dtype=float)))
    return max(inside - outside, 0.0) * max(exchange_rate, 0.0)
