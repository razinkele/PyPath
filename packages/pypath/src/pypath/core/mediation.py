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
        return [l for l in self.links if l.prey_idx is not None and l.pred_idx is not None]

    @property
    def fleet_links(self) -> list[MediationLink]:
        return [l for l in self.links if l.fleet_idx is not None]

    @property
    def landing_links(self) -> list[MediationLink]:
        return [l for l in self.links if l.landing_group_idx is not None]
