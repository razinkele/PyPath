"""Mediation functions for Ecosim predation modification.

Mediation allows a third species (mediator) to modify predator-prey
interactions, fleet catchability, or landing proportions based on
the mediator's relative biomass and a user-defined response shape.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class MediationShape:
    """A mediation response shape defined by X-Y point pairs.

    Parameters
    ----------
    shape_id : int
        Unique identifier for this shape.
    name : str
        Human-readable name.
    x_points : np.ndarray
        Relative biomass values (mediator B / B_base).
    y_points : np.ndarray
        Corresponding multiplier values.
    """

    shape_id: int
    name: str
    x_points: np.ndarray
    y_points: np.ndarray

    def evaluate(self, relative_biomass: float) -> float:
        """Evaluate the shape at a given relative biomass via linear interpolation.

        Values outside the x_points range are clamped to the nearest endpoint.
        """
        if len(self.x_points) <= 1:
            return float(self.y_points[0]) if len(self.y_points) > 0 else 1.0
        return float(np.interp(relative_biomass, self.x_points, self.y_points))


@dataclass
class MediationLink:
    """Maps a mediation shape to a specific interaction.

    Exactly one target type should be specified:
    - Group: prey_idx and pred_idx both set
    - Fleet: fleet_idx set
    - Landings: landing_group_idx and landing_fleet_idx both set

    All indices are 0-based.
    """

    shape_id: int
    mediator_idx: int
    prey_idx: int | None = None
    pred_idx: int | None = None
    fleet_idx: int | None = None
    landing_group_idx: int | None = None
    landing_fleet_idx: int | None = None
    weight: float = 1.0


@dataclass
class MediationCollection:
    """Container for mediation shapes and their link assignments."""

    shapes: list[MediationShape]
    links: list[MediationLink]

    @property
    def group_links(self) -> list[MediationLink]:
        return [
            l for l in self.links if l.prey_idx is not None and l.pred_idx is not None
        ]

    @property
    def fleet_links(self) -> list[MediationLink]:
        return [l for l in self.links if l.fleet_idx is not None]

    @property
    def landing_links(self) -> list[MediationLink]:
        return [l for l in self.links if l.landing_group_idx is not None]

    def _get_shape(self, shape_id: int) -> MediationShape | None:
        for s in self.shapes:
            if s.shape_id == shape_id:
                return s
        return None

    def compute_group_multipliers(
        self, BB: np.ndarray, Bbase: np.ndarray, ActiveLink: np.ndarray
    ) -> np.ndarray:
        """Compute 2D (n+1, n+1) multiplier matrix for group mediation.

        For each group link, evaluate the shape at BB[mediator+1]/Bbase[mediator+1]
        and set mult[prey+1, pred+1]. Multiple links on the same pair multiply together.
        """
        n_plus_1 = len(BB)
        mult = np.ones((n_plus_1, n_plus_1))
        for link in self.group_links:
            shape = self._get_shape(link.shape_id)
            if shape is None:
                continue
            med_col = link.mediator_idx + 1
            if med_col >= n_plus_1 or Bbase[med_col] == 0:
                continue
            rel_b = BB[med_col] / Bbase[med_col]
            val = shape.evaluate(rel_b) * link.weight
            prey_row = link.prey_idx + 1
            pred_col = link.pred_idx + 1
            if prey_row < n_plus_1 and pred_col < n_plus_1:
                mult[prey_row, pred_col] *= val
        return mult

    def compute_fleet_multipliers(
        self, BB: np.ndarray, Bbase: np.ndarray, n_fleets: int
    ) -> np.ndarray:
        """Compute per-fleet multiplier array.

        Each fleet link evaluates the shape at mediator relative biomass.
        Default 1.0 for unaffected fleets.
        """
        mult = np.ones(n_fleets)
        for link in self.fleet_links:
            shape = self._get_shape(link.shape_id)
            if shape is None:
                continue
            med_col = link.mediator_idx + 1
            if med_col >= len(BB) or Bbase[med_col] == 0:
                continue
            rel_b = BB[med_col] / Bbase[med_col]
            val = shape.evaluate(rel_b) * link.weight
            if link.fleet_idx is not None and link.fleet_idx < n_fleets:
                mult[link.fleet_idx] *= val
        return mult

    def compute_landing_multipliers(
        self, BB: np.ndarray, Bbase: np.ndarray, n_fleets: int, n_groups: int
    ) -> np.ndarray:
        """Compute (n_fleets, n_groups) multiplier matrix for landings.

        Default 1.0 for unaffected fleet-group combinations.
        """
        mult = np.ones((n_fleets, n_groups))
        for link in self.landing_links:
            shape = self._get_shape(link.shape_id)
            if shape is None:
                continue
            med_col = link.mediator_idx + 1
            if med_col >= len(BB) or Bbase[med_col] == 0:
                continue
            rel_b = BB[med_col] / Bbase[med_col]
            val = shape.evaluate(rel_b) * link.weight
            fi = link.landing_fleet_idx
            gi = link.landing_group_idx
            if fi is not None and gi is not None and fi < n_fleets and gi < n_groups:
                mult[fi, gi] *= val
        return mult


def make_positive_shape(
    shape_id: int = 0,
    name: str = "positive",
    low: float = 0.5,
    high: float = 2.0,
    shape: float = 1.0,
    n_points: int = 9,
) -> MediationShape:
    """Create a positive (increasing) mediation shape.

    y = low + (high - low) * x^shape / (1 + x^shape)
    """
    x = np.linspace(0.0, 2.0, n_points)
    y = low + (high - low) * x**shape / (1.0 + x**shape)
    return MediationShape(shape_id=shape_id, name=name, x_points=x, y_points=y)


def make_negative_shape(
    shape_id: int = 0,
    name: str = "negative",
    low: float = 0.5,
    high: float = 2.0,
    shape: float = 1.0,
    n_points: int = 9,
) -> MediationShape:
    """Create a negative (decreasing) mediation shape.

    y = high - (high - low) * x^shape / (1 + x^shape)
    """
    x = np.linspace(0.0, 2.0, n_points)
    y = high - (high - low) * x**shape / (1.0 + x**shape)
    return MediationShape(shape_id=shape_id, name=name, x_points=x, y_points=y)


def make_ushape(
    shape_id: int = 0,
    name: str = "u-shaped",
    low: float = 0.5,
    high: float = 2.0,
    shape: float = 1.0,
    n_points: int = 9,
) -> MediationShape:
    """Create a U-shaped mediation shape (peaks at x=1.0).

    y = high - (high - low) * |x-1|^shape / (1 + |x-1|^shape)
    """
    x = np.linspace(0.0, 2.0, n_points)
    abs_diff = np.abs(x - 1.0)
    y = high - (high - low) * abs_diff**shape / (1.0 + abs_diff**shape)
    return MediationShape(shape_id=shape_id, name=name, x_points=x, y_points=y)
