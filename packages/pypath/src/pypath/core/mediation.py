"""Mediation functions for Ecosim predation modification.

Mediation allows a third species (mediator) to modify predator-prey
interactions, fleet catchability, or landing proportions based on
the mediator's relative biomass and a user-defined response shape.
"""

from __future__ import annotations

import math
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
            link
            for link in self.links
            if link.prey_idx is not None and link.pred_idx is not None
        ]

    @property
    def fleet_links(self) -> list[MediationLink]:
        return [link for link in self.links if link.fleet_idx is not None]

    @property
    def landing_links(self) -> list[MediationLink]:
        return [link for link in self.links if link.landing_group_idx is not None]

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


# --- EwE6 pred/prey mediation (cMediationDataStructures) ----------------------

N_MED_POINTS = 1200  # cMediationDataStructures.N_DEFAULT_MEDIATIONPOINTS
SEARCH_RATE = 1  # eForcingFunctionApplication.SearchRate
VULNERABILITY = 2  # eForcingFunctionApplication.Vulnerability


@dataclass
class EweMediation:
    """EwE6's pred/prey mediation functions, as Ecosim applies them.

    Each shape has 1200 points (``points[s, ip - 1]`` is Medpoints(ip)) and
    a base point ``imedbase[s]``. Its mediators are the groups with a weight
    > 0 in ``weights[s]`` (rsim index) and the fleets with a weight > 0 in
    ``fleet_weights[s]`` (gear index, the column of ForcedEffort). Every
    derivative sets the shape's value from the current biomass and effort
    (SetMedFunctions, cEcoSimModel.vb:2851, cMediationDataStructures.vb:129):
    X = 1e-10 + sum(weight x B) + sum(fleet weight x effort), ip =
    Trunc(IMedBase x X / MedXbase + 0.01) clamped to 1..1200, value =
    Medpoints(ip) / MedYbase, with MedXbase = sum(weight x start biomass) +
    sum(fleet weight), as "Ecosim effort at Ecopath base effort is always
    one", and MedYbase = Medpoints(IMedBase) (InitializeMedFunctions, vb
    3568-3650). Groups come first in both sums, then fleets, as in EwE. A
    shape without mediators, base X or base point value is unused and its
    value stays 1. EwE computes in single precision; so does this, so the
    point index truncates alike.

    ``links`` are ``(prey, pred, shape, application)`` (rsim indices): a
    search-rate function multiplies the link's search rate, a vulnerability
    function its vulnerability (ApplyAVmodifiers, vb 4928-4970). Several
    functions on one link multiply.
    """

    points: np.ndarray
    imedbase: np.ndarray
    weights: np.ndarray
    links: list
    fleet_weights: np.ndarray | None = None
    xbase: np.ndarray | None = None
    ybase: np.ndarray | None = None
    used: np.ndarray | None = None

    def setup(self, start_biomass: np.ndarray) -> None:
        """MedXbase, MedYbase and MedIsUsed from the start biomass."""
        pts = np.asarray(self.points, dtype=np.float32)
        imed = np.asarray(self.imedbase, dtype=int)
        n_shapes = pts.shape[0]
        self.xbase = np.zeros(n_shapes, dtype=np.float32)
        self.ybase = np.zeros(n_shapes, dtype=np.float32)
        self.used = np.zeros(n_shapes, dtype=bool)
        b0 = np.asarray(start_biomass, dtype=np.float32)
        for s in range(n_shapes):
            w = np.asarray(self.weights[s], dtype=np.float32)
            wf = self._fleet_row(s)
            x = np.float32(0.0)
            for j in np.nonzero(w > 0)[0]:
                x = np.float32(x + w[j] * b0[j])
            for g in np.nonzero(wf > 0)[0]:
                x = np.float32(x + wf[g] * np.float32(1.0))  # base effort
            self.xbase[s] = x
            ip = min(max(int(imed[s]), 1), N_MED_POINTS)
            self.ybase[s] = pts[s, ip - 1]
            has_mediator = bool(np.any(w > 0) or np.any(wf > 0))
            self.used[s] = has_mediator and x != 0 and self.ybase[s] != 0

    def _fleet_row(self, s: int) -> np.ndarray:
        """The fleet weights of shape ``s`` (empty without fleet mediators)."""
        if self.fleet_weights is None:
            return np.zeros(0, dtype=np.float32)
        return np.asarray(self.fleet_weights[s], dtype=np.float32)

    def lookup(self, shape: int, x: float) -> float:
        """getMedValue: the shape's value at mediator sum ``x``."""
        ratio = np.float32(np.float32(self.imedbase[shape]) * np.float32(x))
        ratio = np.float32(ratio / self.xbase[shape])
        ip = int(math.trunc(float(ratio) + 0.01))
        ip = max(1, min(N_MED_POINTS, ip))
        return float(np.float32(self.points[shape][ip - 1]) / self.ybase[shape])

    def values(
        self, biomass: np.ndarray, effort: np.ndarray | None = None
    ) -> np.ndarray:
        """MedVal of every shape at ``biomass`` and ``effort`` (by gear, as a
        row of ForcedEffort; None or a missing gear counts as the base effort,
        1). Unused shapes are 1."""
        out = np.ones(len(self.used))
        b = np.asarray(biomass, dtype=np.float32)
        e = None if effort is None else np.asarray(effort, dtype=np.float32)
        for s in np.nonzero(self.used)[0]:
            w = np.asarray(self.weights[s], dtype=np.float32)
            wf = self._fleet_row(s)
            x = np.float32(1e-10)
            for j in np.nonzero(w > 0)[0]:
                x = np.float32(x + b[j] * w[j])
            for g in np.nonzero(wf > 0)[0]:
                eff = e[g] if e is not None and g < len(e) else np.float32(1.0)
                x = np.float32(x + eff * wf[g])
            out[s] = self.lookup(s, float(x))
        return out

    def link_factors(
        self, biomass: np.ndarray, n: int, effort: np.ndarray | None = None
    ):
        """Search-rate and vulnerability multipliers by link, (n, n) arrays
        indexed [prey, pred], or None where no link has that application.
        ``effort`` is the fleets' effort by gear (see :meth:`values`)."""
        if not self.links:
            return None, None
        vals = self.values(biomass, effort)
        search = vul = None
        for prey, pred, shape, application in self.links:
            if application == SEARCH_RATE:
                if search is None:
                    search = np.ones((n, n))
                search[prey, pred] *= vals[shape]
            elif application == VULNERABILITY:
                if vul is None:
                    vul = np.ones((n, n))
                vul[prey, pred] *= vals[shape]
        return search, vul
