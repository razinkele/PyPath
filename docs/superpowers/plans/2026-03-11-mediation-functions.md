# Mediation Functions Implementation Plan

> **For agentic workers:** REQUIRED: Use superpowers:subagent-driven-development (if subagents available) or superpowers:executing-plans to implement this plan. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add EwE mediation function support — shape-based and parametric mediation for group, fleet, and landings interactions.

**Architecture:** New `core/mediation.py` module with MediationShape (X-Y interpolation), MediationLink (target mapping), and MediationCollection (precompute multipliers). Mediation is threaded through the existing params dict into deriv_vector() — no signature changes to integrators. The consumption kernel gains a 2D `med_multipliers` matrix; fleet mediation scales `effort_mult` in the fishing loop.

**Tech Stack:** numpy, dataclasses, scipy (not required — uses np.interp)

**Spec:** `docs/superpowers/specs/2026-03-11-mediation-functions-design.md`

---

## Chunk 1: Data Structures & Unit Tests

### Task 1: MediationShape Dataclass and evaluate()

**Files:**
- Create: `packages/pypath/src/pypath/core/mediation.py`
- Create: `packages/pypath/tests/test_mediation.py`

- [ ] **Step 1: Write failing tests for MediationShape**

Create `packages/pypath/tests/test_mediation.py`:

```python
"""Tests for pypath.core.mediation module."""
import numpy as np
import pytest

from pypath.core.mediation import MediationShape


class TestMediationShape:
    def test_construction(self):
        s = MediationShape(
            shape_id=1, name="test",
            x_points=np.array([0.0, 1.0, 2.0]),
            y_points=np.array([0.5, 1.0, 1.5]),
        )
        assert s.shape_id == 1
        assert s.name == "test"
        assert len(s.x_points) == 3

    def test_evaluate_at_known_points(self):
        s = MediationShape(
            shape_id=1, name="test",
            x_points=np.array([0.0, 1.0, 2.0]),
            y_points=np.array([0.5, 1.0, 1.5]),
        )
        assert s.evaluate(0.0) == pytest.approx(0.5)
        assert s.evaluate(1.0) == pytest.approx(1.0)
        assert s.evaluate(2.0) == pytest.approx(1.5)

    def test_evaluate_interpolation(self):
        s = MediationShape(
            shape_id=1, name="test",
            x_points=np.array([0.0, 1.0, 2.0]),
            y_points=np.array([0.5, 1.0, 1.5]),
        )
        # Midpoint between 0.0->0.5 and 1.0->1.0
        assert s.evaluate(0.5) == pytest.approx(0.75)

    def test_evaluate_clamp_below(self):
        s = MediationShape(
            shape_id=1, name="test",
            x_points=np.array([0.0, 1.0, 2.0]),
            y_points=np.array([0.5, 1.0, 1.5]),
        )
        assert s.evaluate(-1.0) == pytest.approx(0.5)

    def test_evaluate_clamp_above(self):
        s = MediationShape(
            shape_id=1, name="test",
            x_points=np.array([0.0, 1.0, 2.0]),
            y_points=np.array([0.5, 1.0, 1.5]),
        )
        assert s.evaluate(5.0) == pytest.approx(1.5)

    def test_evaluate_single_point(self):
        s = MediationShape(
            shape_id=1, name="const",
            x_points=np.array([1.0]),
            y_points=np.array([2.0]),
        )
        assert s.evaluate(0.0) == pytest.approx(2.0)
        assert s.evaluate(1.0) == pytest.approx(2.0)
        assert s.evaluate(5.0) == pytest.approx(2.0)
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `conda run -n shiny python -m pytest packages/pypath/tests/test_mediation.py -v`
Expected: FAIL with ImportError (mediation module does not exist)

- [ ] **Step 3: Implement MediationShape**

Create `packages/pypath/src/pypath/core/mediation.py`:

```python
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
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `conda run -n shiny python -m pytest packages/pypath/tests/test_mediation.py::TestMediationShape -v`
Expected: 6 PASSED

- [ ] **Step 5: Commit**

```bash
git add packages/pypath/src/pypath/core/mediation.py packages/pypath/tests/test_mediation.py
git commit -m "feat(mediation): add MediationShape dataclass with evaluate()"
```

---

### Task 2: MediationLink, MediationCollection, and Filtered Views

**Files:**
- Modify: `packages/pypath/src/pypath/core/mediation.py`
- Modify: `packages/pypath/tests/test_mediation.py`

- [ ] **Step 1: Write failing tests for MediationLink and MediationCollection**

Append to `packages/pypath/tests/test_mediation.py`:

```python
from pypath.core.mediation import MediationLink, MediationCollection


class TestMediationLink:
    def test_group_link(self):
        link = MediationLink(shape_id=1, mediator_idx=0, prey_idx=1, pred_idx=2)
        assert link.prey_idx == 1
        assert link.pred_idx == 2
        assert link.fleet_idx is None

    def test_fleet_link(self):
        link = MediationLink(shape_id=1, mediator_idx=0, fleet_idx=0)
        assert link.fleet_idx == 0
        assert link.prey_idx is None

    def test_landing_link(self):
        link = MediationLink(
            shape_id=1, mediator_idx=0,
            landing_group_idx=1, landing_fleet_idx=0,
        )
        assert link.landing_group_idx == 1
        assert link.landing_fleet_idx == 0

    def test_default_weight(self):
        link = MediationLink(shape_id=1, mediator_idx=0, prey_idx=0, pred_idx=1)
        assert link.weight == 1.0


class TestMediationCollection:
    def _make_shape(self):
        return MediationShape(
            shape_id=1, name="test",
            x_points=np.array([0.0, 1.0, 2.0]),
            y_points=np.array([0.5, 1.0, 1.5]),
        )

    def test_empty_collection(self):
        coll = MediationCollection(shapes=[], links=[])
        assert coll.group_links == []
        assert coll.fleet_links == []
        assert coll.landing_links == []

    def test_group_links_filter(self):
        links = [
            MediationLink(shape_id=1, mediator_idx=0, prey_idx=0, pred_idx=1),
            MediationLink(shape_id=1, mediator_idx=0, fleet_idx=0),
        ]
        coll = MediationCollection(shapes=[self._make_shape()], links=links)
        assert len(coll.group_links) == 1
        assert coll.group_links[0].prey_idx == 0

    def test_fleet_links_filter(self):
        links = [
            MediationLink(shape_id=1, mediator_idx=0, prey_idx=0, pred_idx=1),
            MediationLink(shape_id=1, mediator_idx=0, fleet_idx=0),
        ]
        coll = MediationCollection(shapes=[self._make_shape()], links=links)
        assert len(coll.fleet_links) == 1
        assert coll.fleet_links[0].fleet_idx == 0

    def test_landing_links_filter(self):
        links = [
            MediationLink(shape_id=1, mediator_idx=0, landing_group_idx=1, landing_fleet_idx=0),
        ]
        coll = MediationCollection(shapes=[self._make_shape()], links=links)
        assert len(coll.landing_links) == 1
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `conda run -n shiny python -m pytest packages/pypath/tests/test_mediation.py::TestMediationLink packages/pypath/tests/test_mediation.py::TestMediationCollection -v`
Expected: FAIL with ImportError

- [ ] **Step 3: Implement MediationLink and MediationCollection**

Append to `packages/pypath/src/pypath/core/mediation.py`:

```python
@dataclass
class MediationLink:
    """Maps a mediation shape to a specific interaction.

    Exactly one target type should be specified:
    - Group: prey_idx and pred_idx both set
    - Fleet: fleet_idx set
    - Landings: landing_group_idx and landing_fleet_idx both set

    All indices are 0-based.

    Parameters
    ----------
    shape_id : int
        ID of the MediationShape to use.
    mediator_idx : int
        0-based group index of the mediating species.
    weight : float
        Weighting factor (AppliedWeight from EwE database).
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
    """Container for mediation shapes and their link assignments.

    Parameters
    ----------
    shapes : list[MediationShape]
        All mediation shapes.
    links : list[MediationLink]
        All mediation link assignments.
    """

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
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `conda run -n shiny python -m pytest packages/pypath/tests/test_mediation.py -v`
Expected: All PASSED

- [ ] **Step 5: Commit**

```bash
git add packages/pypath/src/pypath/core/mediation.py packages/pypath/tests/test_mediation.py
git commit -m "feat(mediation): add MediationLink, MediationCollection with filtered views"
```

---

### Task 3: Precomputation Methods (compute_group/fleet/landing_multipliers)

**Files:**
- Modify: `packages/pypath/src/pypath/core/mediation.py`
- Modify: `packages/pypath/tests/test_mediation.py`

- [ ] **Step 1: Write failing tests for compute methods**

Append to `packages/pypath/tests/test_mediation.py` inside `TestMediationCollection`:

```python
    def test_compute_group_multipliers_basic(self):
        """Single group mediation link: mediator at 2x baseline -> shape gives 1.5."""
        shape = MediationShape(
            shape_id=1, name="test",
            x_points=np.array([0.0, 1.0, 2.0]),
            y_points=np.array([0.5, 1.0, 1.5]),
        )
        # mediator_idx=0 (col 1), prey_idx=1 (col 2), pred_idx=2 (col 3)
        link = MediationLink(shape_id=1, mediator_idx=0, prey_idx=1, pred_idx=2)
        coll = MediationCollection(shapes=[shape], links=[link])

        n = 4  # groups 0..3 -> cols 0..3 in arrays
        BB = np.ones(n + 1)      # 1-based: col 0 unused
        Bbase = np.ones(n + 1)
        BB[1] = 2.0  # mediator (group 0) at 2x baseline
        ActiveLink = np.zeros((n + 1, n + 1), dtype=int)
        ActiveLink[2, 3] = 1  # prey=1->col2, pred=2->col3

        mult = coll.compute_group_multipliers(BB, Bbase, ActiveLink)
        assert mult.shape == (n + 1, n + 1)
        assert mult[2, 3] == pytest.approx(1.5)  # shape(2.0) = 1.5
        assert mult[1, 2] == pytest.approx(1.0)  # unaffected link

    def test_compute_group_multipliers_empty(self):
        coll = MediationCollection(shapes=[], links=[])
        n = 3
        BB = np.ones(n + 1)
        Bbase = np.ones(n + 1)
        ActiveLink = np.zeros((n + 1, n + 1), dtype=int)
        mult = coll.compute_group_multipliers(BB, Bbase, ActiveLink)
        assert np.all(mult == 1.0)

    def test_compute_group_multipliers_multiple_on_same_link(self):
        """Two mediators on same pred-prey link -> multiplied together."""
        shape = MediationShape(
            shape_id=1, name="test",
            x_points=np.array([0.0, 1.0, 2.0]),
            y_points=np.array([0.5, 1.0, 2.0]),
        )
        # Two different mediators (group 0 and group 1) both affect prey=2, pred=3
        link_a = MediationLink(shape_id=1, mediator_idx=0, prey_idx=2, pred_idx=3)
        link_b = MediationLink(shape_id=1, mediator_idx=1, prey_idx=2, pred_idx=3)
        coll = MediationCollection(shapes=[shape], links=[link_a, link_b])

        n = 4
        BB = np.ones(n + 1)
        Bbase = np.ones(n + 1)
        BB[1] = 2.0  # mediator 0 at 2x -> shape(2.0)=2.0
        BB[2] = 0.5  # mediator 1 at 0.5x -> shape(0.5)=0.75
        ActiveLink = np.zeros((n + 1, n + 1), dtype=int)
        ActiveLink[3, 4] = 1  # prey=2->col3, pred=3->col4

        mult = coll.compute_group_multipliers(BB, Bbase, ActiveLink)
        assert mult[3, 4] == pytest.approx(2.0 * 0.75)  # multiplicative

    def test_compute_fleet_multipliers(self):
        shape = MediationShape(
            shape_id=1, name="test",
            x_points=np.array([0.0, 1.0, 2.0]),
            y_points=np.array([0.5, 1.0, 1.5]),
        )
        link = MediationLink(shape_id=1, mediator_idx=0, fleet_idx=1)
        coll = MediationCollection(shapes=[shape], links=[link])

        BB = np.array([0.0, 2.0, 1.0, 1.0])  # col 1 = mediator at 2x
        Bbase = np.ones(4)
        mult = coll.compute_fleet_multipliers(BB, Bbase, n_fleets=3)
        assert mult[0] == pytest.approx(1.0)  # no mediation on fleet 0
        assert mult[1] == pytest.approx(1.5)  # shape(2.0) = 1.5
        assert mult[2] == pytest.approx(1.0)

    def test_compute_fleet_multipliers_empty(self):
        coll = MediationCollection(shapes=[], links=[])
        mult = coll.compute_fleet_multipliers(np.ones(4), np.ones(4), n_fleets=3)
        assert np.all(mult == 1.0)

    def test_compute_landing_multipliers(self):
        shape = MediationShape(
            shape_id=1, name="test",
            x_points=np.array([0.0, 1.0, 2.0]),
            y_points=np.array([0.5, 1.0, 1.5]),
        )
        link = MediationLink(
            shape_id=1, mediator_idx=0,
            landing_group_idx=1, landing_fleet_idx=0,
        )
        coll = MediationCollection(shapes=[shape], links=[link])

        BB = np.array([0.0, 2.0, 1.0])  # mediator at 2x
        Bbase = np.ones(3)
        mult = coll.compute_landing_multipliers(BB, Bbase, n_fleets=2, n_groups=2)
        assert mult.shape == (2, 2)
        assert mult[0, 1] == pytest.approx(1.5)  # fleet 0, group 1
        assert mult[1, 0] == pytest.approx(1.0)  # unaffected

    def test_compute_landing_multipliers_empty(self):
        coll = MediationCollection(shapes=[], links=[])
        mult = coll.compute_landing_multipliers(np.ones(3), np.ones(3), n_fleets=2, n_groups=2)
        assert np.all(mult == 1.0)
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `conda run -n shiny python -m pytest packages/pypath/tests/test_mediation.py -v -k "compute"`
Expected: FAIL with AttributeError

- [ ] **Step 3: Implement compute methods**

Add to `MediationCollection` in `mediation.py`:

```python
    def _get_shape(self, shape_id: int) -> MediationShape | None:
        for s in self.shapes:
            if s.shape_id == shape_id:
                return s
        return None

    def compute_group_multipliers(
        self, BB: np.ndarray, Bbase: np.ndarray, ActiveLink: np.ndarray
    ) -> np.ndarray:
        """Precompute 2D per-link mediation multipliers for consumption kernel.

        Returns array of shape (n_groups+1, n_groups+1) with 1.0 for
        unaffected links. Multiple mediation links on the same pred-prey
        pair are multiplied together.

        BB and Bbase are 1-based arrays (index 0 = Outside).
        MediationLink indices are 0-based; converted to 1-based columns via +1.
        """
        n = ActiveLink.shape[0]
        mult = np.ones((n, n))
        for link in self.group_links:
            shape = self._get_shape(link.shape_id)
            if shape is None:
                continue
            med_col = link.mediator_idx + 1
            if med_col >= len(BB) or Bbase[med_col] <= 0:
                continue
            rel_bio = BB[med_col] / Bbase[med_col]
            m = shape.evaluate(rel_bio) * link.weight
            prey_col = link.prey_idx + 1
            pred_col = link.pred_idx + 1
            if prey_col < n and pred_col < n:
                mult[prey_col, pred_col] *= m
        return mult

    def compute_fleet_multipliers(
        self, BB: np.ndarray, Bbase: np.ndarray, n_fleets: int
    ) -> np.ndarray:
        """Precompute per-fleet effort multipliers.

        Returns array of length n_fleets, default 1.0.
        """
        mult = np.ones(n_fleets)
        for link in self.fleet_links:
            shape = self._get_shape(link.shape_id)
            if shape is None:
                continue
            med_col = link.mediator_idx + 1
            if med_col >= len(BB) or Bbase[med_col] <= 0:
                continue
            rel_bio = BB[med_col] / Bbase[med_col]
            m = shape.evaluate(rel_bio) * link.weight
            if 0 <= link.fleet_idx < n_fleets:
                mult[link.fleet_idx] *= m
        return mult

    def compute_landing_multipliers(
        self, BB: np.ndarray, Bbase: np.ndarray, n_fleets: int, n_groups: int
    ) -> np.ndarray:
        """Precompute per fleet-group landing proportion multipliers.

        Returns (n_fleets, n_groups) array, default 1.0.
        """
        mult = np.ones((n_fleets, n_groups))
        for link in self.landing_links:
            shape = self._get_shape(link.shape_id)
            if shape is None:
                continue
            med_col = link.mediator_idx + 1
            if med_col >= len(BB) or Bbase[med_col] <= 0:
                continue
            rel_bio = BB[med_col] / Bbase[med_col]
            m = shape.evaluate(rel_bio) * link.weight
            fi = link.landing_fleet_idx
            gi = link.landing_group_idx
            if fi is not None and gi is not None and 0 <= fi < n_fleets and 0 <= gi < n_groups:
                mult[fi, gi] *= m
        return mult
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `conda run -n shiny python -m pytest packages/pypath/tests/test_mediation.py -v`
Expected: All PASSED

- [ ] **Step 5: Commit**

```bash
git add packages/pypath/src/pypath/core/mediation.py packages/pypath/tests/test_mediation.py
git commit -m "feat(mediation): add compute_group/fleet/landing_multipliers"
```

---

### Task 4: Parametric Convenience Factories

**Files:**
- Modify: `packages/pypath/src/pypath/core/mediation.py`
- Modify: `packages/pypath/tests/test_mediation.py`

- [ ] **Step 1: Write failing tests for factories**

Append to `packages/pypath/tests/test_mediation.py`:

```python
from pypath.core.mediation import make_positive_shape, make_negative_shape, make_ushape


class TestParametricFactories:
    def test_positive_shape_endpoints(self):
        s = make_positive_shape(low=0.5, high=2.0)
        # At x=0: y should be near low (0.5)
        assert s.evaluate(0.0) == pytest.approx(0.5)
        # At x=2.0 (high relative biomass): should approach high
        assert s.evaluate(2.0) > 1.0

    def test_positive_shape_at_one(self):
        s = make_positive_shape(low=0.5, high=2.0, shape=1.0)
        # At x=1: y = low + (high-low) * 1/(1+1) = 0.5 + 1.5*0.5 = 1.25
        assert s.evaluate(1.0) == pytest.approx(1.25)

    def test_negative_shape_endpoints(self):
        s = make_negative_shape(low=0.5, high=2.0)
        # At x=0: y should be near high (2.0)
        assert s.evaluate(0.0) == pytest.approx(2.0)
        # At x=2.0: should be closer to low
        assert s.evaluate(2.0) < 1.5

    def test_negative_shape_at_one(self):
        s = make_negative_shape(low=0.5, high=2.0, shape=1.0)
        # At x=1: y = high - (high-low) * 1/(1+1) = 2.0 - 1.5*0.5 = 1.25
        assert s.evaluate(1.0) == pytest.approx(1.25)

    def test_ushape_at_one(self):
        s = make_ushape(low=0.5, high=2.0, shape=1.0)
        # At x=1 (optimal): |x-1|=0, so y = high = 2.0
        assert s.evaluate(1.0) == pytest.approx(2.0)

    def test_ushape_at_extremes(self):
        s = make_ushape(low=0.5, high=2.0, shape=1.0)
        # At x=0 and x=2: |x-1|=1, y = high - (high-low)*1/(1+1) = 1.25
        assert s.evaluate(0.0) == pytest.approx(1.25)
        assert s.evaluate(2.0) == pytest.approx(1.25)

    def test_n_points(self):
        s = make_positive_shape(n_points=5)
        assert len(s.x_points) == 5
        assert len(s.y_points) == 5

    def test_shape_exponent(self):
        s1 = make_positive_shape(shape=1.0)
        s2 = make_positive_shape(shape=2.0)
        # At x=0.5: different shape exponents should give different Y
        assert s1.evaluate(0.5) != pytest.approx(s2.evaluate(0.5), abs=0.01)

    def test_factory_ids_and_names(self):
        s = make_positive_shape(shape_id=42, name="my_shape")
        assert s.shape_id == 42
        assert s.name == "my_shape"
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `conda run -n shiny python -m pytest packages/pypath/tests/test_mediation.py::TestParametricFactories -v`
Expected: FAIL with ImportError

- [ ] **Step 3: Implement factories**

Append to `packages/pypath/src/pypath/core/mediation.py`:

```python
def make_positive_shape(
    shape_id: int = 0, name: str = "positive",
    low: float = 0.5, high: float = 2.0, shape: float = 1.0, n_points: int = 9,
) -> MediationShape:
    """Create a positive mediation shape (more mediator = higher multiplier).

    y = low + (high - low) * x^shape / (1 + x^shape)
    """
    x = np.linspace(0.0, 2.0, n_points)
    y = np.where(
        x == 0, low,
        low + (high - low) * (x ** shape) / (1.0 + x ** shape),
    )
    return MediationShape(shape_id=shape_id, name=name, x_points=x, y_points=y)


def make_negative_shape(
    shape_id: int = 0, name: str = "negative",
    low: float = 0.5, high: float = 2.0, shape: float = 1.0, n_points: int = 9,
) -> MediationShape:
    """Create a negative mediation shape (more mediator = lower multiplier).

    y = high - (high - low) * x^shape / (1 + x^shape)
    """
    x = np.linspace(0.0, 2.0, n_points)
    y = np.where(
        x == 0, high,
        high - (high - low) * (x ** shape) / (1.0 + x ** shape),
    )
    return MediationShape(shape_id=shape_id, name=name, x_points=x, y_points=y)


def make_ushape(
    shape_id: int = 0, name: str = "u-shaped",
    low: float = 0.5, high: float = 2.0, shape: float = 1.0, n_points: int = 9,
) -> MediationShape:
    """Create a U-shaped mediation shape (optimal at x=1, declines at extremes).

    y = high - (high - low) * |x-1|^shape / (1 + |x-1|^shape)
    """
    x = np.linspace(0.0, 2.0, n_points)
    diff = np.abs(x - 1.0)
    y = np.where(
        diff == 0, high,
        high - (high - low) * (diff ** shape) / (1.0 + diff ** shape),
    )
    return MediationShape(shape_id=shape_id, name=name, x_points=x, y_points=y)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `conda run -n shiny python -m pytest packages/pypath/tests/test_mediation.py -v`
Expected: All PASSED

- [ ] **Step 5: Commit**

```bash
git add packages/pypath/src/pypath/core/mediation.py packages/pypath/tests/test_mediation.py
git commit -m "feat(mediation): add parametric convenience factories"
```

---

## Chunk 2: I/O Layer

### Task 5: Schema Definitions

**Files:**
- Modify: `packages/pypath/src/pypath/io/_ewe_schema.py`

- [ ] **Step 1: Add 4 mediation table definitions to EWE_TABLES**

Open `packages/pypath/src/pypath/io/_ewe_schema.py` and add these 4 entries to the `EWE_TABLES` OrderedDict (after the existing `EcosimTimeSeriesSeason` entry):

```python
    "EcosimShapeMediation": OrderedDict([
        ("ShapeID", "INTEGER"),
        ("Title", "TEXT"),
        ("nPoints", "INTEGER"),
        ("YY1", "DOUBLE"),
        ("YY2", "DOUBLE"),
        ("YY3", "DOUBLE"),
        ("YY4", "DOUBLE"),
        ("YY5", "DOUBLE"),
        ("YY6", "DOUBLE"),
        ("YY7", "DOUBLE"),
        ("YY8", "DOUBLE"),
        ("YY9", "DOUBLE"),
    ]),
    "EcosimScenarioshapeMedWeightsGroup": OrderedDict([
        ("ScenarioID", "INTEGER"),
        ("ShapeID", "INTEGER"),
        ("GroupID", "INTEGER"),
        ("PredID", "INTEGER"),
        ("PreyID", "INTEGER"),
        ("AppliedWeight", "DOUBLE"),
    ]),
    "EcosimScenarioshapeMedWeightsFleet": OrderedDict([
        ("ScenarioID", "INTEGER"),
        ("ShapeID", "INTEGER"),
        ("GroupID", "INTEGER"),
        ("FleetID", "INTEGER"),
        ("AppliedWeight", "DOUBLE"),
    ]),
    "EcosimScenarioshapeMedWeightsLandings": OrderedDict([
        ("ScenarioID", "INTEGER"),
        ("ShapeID", "INTEGER"),
        ("GroupID", "INTEGER"),
        ("FleetID", "INTEGER"),
        ("AppliedWeight", "DOUBLE"),
    ]),
```

- [ ] **Step 2: Verify schema compiles**

Run: `conda run -n shiny python -c "from pypath.io._ewe_schema import EWE_TABLES; print('EcosimShapeMediation' in EWE_TABLES)"`
Expected: `True`

- [ ] **Step 3: Commit**

```bash
git add packages/pypath/src/pypath/io/_ewe_schema.py
git commit -m "feat(io): add 4 mediation table definitions to EwE schema"
```

---

### Task 6: Database Reader — read_mediation()

**Files:**
- Modify: `packages/pypath/src/pypath/io/ewemdb.py`
- Create: `packages/pypath/tests/test_mediation_io.py`

- [ ] **Step 1: Write failing tests for read_mediation**

Create `packages/pypath/tests/test_mediation_io.py`:

```python
"""Tests for mediation I/O functions."""
import numpy as np
import pytest

from pypath.io.ewemdb import read_mediation


class TestReadMediation:
    def test_import(self):
        """read_mediation is importable."""
        assert callable(read_mediation)

    def test_missing_file_returns_empty(self, tmp_path):
        """Non-existent file returns empty collection."""
        coll = read_mediation(str(tmp_path / "nonexistent.eweaccdb"))
        assert len(coll.shapes) == 0
        assert len(coll.links) == 0
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `conda run -n shiny python -m pytest packages/pypath/tests/test_mediation_io.py -v`
Expected: FAIL with ImportError

- [ ] **Step 3: Implement read_mediation**

Append to the end of `packages/pypath/src/pypath/io/ewemdb.py` (after `read_timeseries`):

```python
def read_mediation(db_path: str) -> "MediationCollection":
    """Read mediation shapes and link assignments from an EwE database.

    Parameters
    ----------
    db_path : str
        Path to an EwE database file (.eweaccdb, .ewemdb, .accdb).

    Returns
    -------
    MediationCollection
        Collection of mediation shapes and links. Empty if tables are missing.
    """
    from pypath.core.mediation import MediationCollection, MediationLink, MediationShape

    filepath = str(db_path)

    # Try reading the shape table
    shape_df = _try_read_table_variants(
        filepath,
        ["EcosimShapeMediation", "Ecosim_Shape_Mediation"],
    )

    shapes: list[MediationShape] = []
    if shape_df is not None and len(shape_df) > 0:
        for _, row in shape_df.iterrows():
            shape_id = int(row.get("ShapeID", 0))
            name = str(row.get("Title", f"Shape_{shape_id}"))
            n_points = int(row.get("nPoints", 9))
            y_vals = []
            for i in range(1, 10):
                col = f"YY{i}"
                if col in row and pd.notna(row[col]):
                    y_vals.append(float(row[col]))
            if not y_vals:
                continue
            n_pts = min(n_points, len(y_vals))
            x_pts = np.linspace(0.0, 2.0, n_pts)
            y_pts = np.array(y_vals[:n_pts])
            shapes.append(MediationShape(
                shape_id=shape_id, name=name,
                x_points=x_pts, y_points=y_pts,
            ))

    links: list[MediationLink] = []

    # Group mediation weights
    grp_df = _try_read_table_variants(
        filepath,
        ["EcosimScenarioshapeMedWeightsGroup"],
    )
    if grp_df is not None and len(grp_df) > 0:
        for _, row in grp_df.iterrows():
            links.append(MediationLink(
                shape_id=int(row.get("ShapeID", 0)),
                mediator_idx=int(row.get("GroupID", 1)) - 1,
                prey_idx=int(row.get("PreyID", 1)) - 1,
                pred_idx=int(row.get("PredID", 1)) - 1,
                weight=float(row.get("AppliedWeight", 1.0)),
            ))

    # Fleet mediation weights
    fleet_df = _try_read_table_variants(
        filepath,
        ["EcosimScenarioshapeMedWeightsFleet"],
    )
    if fleet_df is not None and len(fleet_df) > 0:
        for _, row in fleet_df.iterrows():
            links.append(MediationLink(
                shape_id=int(row.get("ShapeID", 0)),
                mediator_idx=int(row.get("GroupID", 1)) - 1,
                fleet_idx=int(row.get("FleetID", 1)) - 1,
                weight=float(row.get("AppliedWeight", 1.0)),
            ))

    # Landings mediation weights
    # In EwE, GroupID = mediator group, FleetID = fleet whose landings are affected.
    # The landing target group is the mediator itself (mediation modifies
    # the landed proportion of the mediator group's catch by the given fleet).
    land_df = _try_read_table_variants(
        filepath,
        ["EcosimScenarioshapeMedWeightsLandings"],
    )
    if land_df is not None and len(land_df) > 0:
        for _, row in land_df.iterrows():
            med_idx = int(row.get("GroupID", 1)) - 1
            links.append(MediationLink(
                shape_id=int(row.get("ShapeID", 0)),
                mediator_idx=med_idx,
                landing_group_idx=med_idx,
                landing_fleet_idx=int(row.get("FleetID", 1)) - 1,
                weight=float(row.get("AppliedWeight", 1.0)),
            ))

    return MediationCollection(shapes=shapes, links=links)
```

Note: `_try_read_table_variants` already exists in `ewemdb.py` and handles missing tables by returning `None`.

- [ ] **Step 4: Run tests to verify they pass**

Run: `conda run -n shiny python -m pytest packages/pypath/tests/test_mediation_io.py -v`
Expected: 2 PASSED

- [ ] **Step 5: Commit**

```bash
git add packages/pypath/src/pypath/io/ewemdb.py packages/pypath/tests/test_mediation_io.py
git commit -m "feat(io): add read_mediation() for EwE database reading"
```

---

### Task 7: Export — Writer Backends

**Files:**
- Modify: `packages/pypath/src/pypath/io/_csv_bundle_writer.py`
- Modify: `packages/pypath/src/pypath/io/_access_writer.py`
- Modify: `packages/pypath/src/pypath/io/ewe_writer.py`
- Modify: `packages/pypath/tests/test_mediation_io.py`

- [ ] **Step 1: Write failing round-trip test**

Append to `packages/pypath/tests/test_mediation_io.py`:

```python
from pypath.core.mediation import (
    MediationCollection, MediationLink, MediationShape,
)


class TestMediationRoundtrip:
    def test_csv_bundle_roundtrip(self, tmp_path):
        """Write mediation to CSV bundle then read shapes back."""
        from pypath.io._csv_bundle_writer import CsvBundleWriter
        from pypath.core.params import create_rpath_params

        # Minimal model
        params = create_rpath_params(
            model_name="med_test", num_groups=3, num_fleets=1
        )
        params.model.loc[0, "Type"] = 1  # producer
        params.model.loc[1, "Type"] = 0  # consumer
        params.model.loc[2, "Type"] = 2  # detritus

        shape = MediationShape(
            shape_id=1, name="test_shape",
            x_points=np.array([0.0, 0.5, 1.0, 1.5, 2.0]),
            y_points=np.array([0.5, 0.75, 1.0, 1.25, 1.5]),
        )
        link = MediationLink(
            shape_id=1, mediator_idx=0, prey_idx=0, pred_idx=1, weight=0.8,
        )
        coll = MediationCollection(shapes=[shape], links=[link])

        out_path = str(tmp_path / "test.ewecsv.zip")
        writer = CsvBundleWriter(params, out_path, scenario_id=1)
        writer.write_ecopath()
        writer.write_mediation(coll)
        writer.close()

        # Verify the zip contains the mediation tables
        import zipfile
        with zipfile.ZipFile(out_path, "r") as zf:
            names = zf.namelist()
            assert "EcosimShapeMediation.csv" in names
            assert "EcosimScenarioshapeMedWeightsGroup.csv" in names
            assert "EcosimScenarioshapeMedWeightsFleet.csv" in names
            assert "EcosimScenarioshapeMedWeightsLandings.csv" in names
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `conda run -n shiny python -m pytest packages/pypath/tests/test_mediation_io.py::TestMediationRoundtrip -v`
Expected: FAIL with AttributeError (write_mediation does not exist)

- [ ] **Step 3: Implement write_mediation on CsvBundleWriter**

Add method to `CsvBundleWriter` in `packages/pypath/src/pypath/io/_csv_bundle_writer.py` (after `write_timeseries`):

```python
    def write_mediation(self, mediation=None) -> None:
        """Write mediation tables to the CSV bundle."""
        if mediation is None:
            return

        # EcosimShapeMediation
        shape_rows = []
        for s in mediation.shapes:
            row = {
                "ShapeID": s.shape_id,
                "Title": s.name,
                "nPoints": len(s.y_points),
            }
            for i in range(9):
                col = f"YY{i + 1}"
                row[col] = float(s.y_points[i]) if i < len(s.y_points) else 0.0
            shape_rows.append(row)
        self._tables["EcosimShapeMediation"] = pd.DataFrame(shape_rows)

        # Group weights
        grp_rows = []
        fleet_rows = []
        land_rows = []
        for link in mediation.links:
            if link.prey_idx is not None and link.pred_idx is not None:
                grp_rows.append({
                    "ScenarioID": self._scenario_id,
                    "ShapeID": link.shape_id,
                    "GroupID": link.mediator_idx + 1,
                    "PredID": link.pred_idx + 1,
                    "PreyID": link.prey_idx + 1,
                    "AppliedWeight": link.weight,
                })
            elif link.fleet_idx is not None:
                fleet_rows.append({
                    "ScenarioID": self._scenario_id,
                    "ShapeID": link.shape_id,
                    "GroupID": link.mediator_idx + 1,
                    "FleetID": link.fleet_idx + 1,
                    "AppliedWeight": link.weight,
                })
            elif link.landing_group_idx is not None:
                land_rows.append({
                    "ScenarioID": self._scenario_id,
                    "ShapeID": link.shape_id,
                    "GroupID": link.mediator_idx + 1,
                    "FleetID": link.landing_fleet_idx + 1 if link.landing_fleet_idx is not None else 0,
                    "AppliedWeight": link.weight,
                })

        self._tables["EcosimScenarioshapeMedWeightsGroup"] = pd.DataFrame(
            grp_rows, columns=["ScenarioID", "ShapeID", "GroupID", "PredID", "PreyID", "AppliedWeight"]
        ) if grp_rows else pd.DataFrame(
            columns=["ScenarioID", "ShapeID", "GroupID", "PredID", "PreyID", "AppliedWeight"]
        )
        self._tables["EcosimScenarioshapeMedWeightsFleet"] = pd.DataFrame(
            fleet_rows, columns=["ScenarioID", "ShapeID", "GroupID", "FleetID", "AppliedWeight"]
        ) if fleet_rows else pd.DataFrame(
            columns=["ScenarioID", "ShapeID", "GroupID", "FleetID", "AppliedWeight"]
        )
        self._tables["EcosimScenarioshapeMedWeightsLandings"] = pd.DataFrame(
            land_rows, columns=["ScenarioID", "ShapeID", "GroupID", "FleetID", "AppliedWeight"]
        ) if land_rows else pd.DataFrame(
            columns=["ScenarioID", "ShapeID", "GroupID", "FleetID", "AppliedWeight"]
        )
```

- [ ] **Step 4: Add write_mediation to AccessWriter**

Add to `packages/pypath/src/pypath/io/_access_writer.py` (after `write_timeseries`):

```python
    def write_mediation(self, mediation=None) -> None:
        """Write mediation tables to the Access database."""
        if mediation is None:
            return
        self._build_tables_via_csv_writer("write_mediation", mediation=mediation)
```

- [ ] **Step 5: Add mediation parameter to write_ewemdb**

In `packages/pypath/src/pypath/io/ewe_writer.py`:

Add `mediation: Any | None = None,` parameter to `write_ewemdb()` signature (after `timeseries`).

Add `writer.write_mediation(mediation)` call after `writer.write_timeseries(timeseries)` in the try block.

- [ ] **Step 6: Run tests to verify they pass**

Run: `conda run -n shiny python -m pytest packages/pypath/tests/test_mediation_io.py -v`
Expected: All PASSED

- [ ] **Step 7: Commit**

```bash
git add packages/pypath/src/pypath/io/_csv_bundle_writer.py packages/pypath/src/pypath/io/_access_writer.py packages/pypath/src/pypath/io/ewe_writer.py packages/pypath/tests/test_mediation_io.py
git commit -m "feat(io): add write_mediation() to writer backends and ewe_writer"
```

---

## Chunk 3: Runtime Integration

### Task 8: Consumption Kernel — med_multipliers Parameter

**Files:**
- Modify: `packages/pypath/src/pypath/core/ecosim_deriv.py`
- Modify: `packages/pypath/tests/test_mediation.py`

- [ ] **Step 1: Write failing test for kernel with med_multipliers**

Append to `packages/pypath/tests/test_mediation.py`:

```python
from pypath.core.ecosim_deriv import _compute_consumption_sparse_python


class TestKernelMedMultipliers:
    def test_sparse_kernel_with_mediation(self):
        """Consumption kernel applies med_multipliers correctly."""
        n = 4  # 0..3, 0=outside
        QQ = np.zeros((n, n))
        BB = np.array([0.0, 1.0, 1.0, 1.0])
        VV = np.full((n, n), 2.0)
        DD = np.full((n, n), 1000.0)
        QQbase = np.zeros((n, n))
        QQbase[1, 2] = 0.5  # prey=1, pred=2 has base consumption
        preyYY = BB.copy()
        predYY = BB.copy()
        link_prey = np.array([1], dtype=np.int64)
        link_pred = np.array([2], dtype=np.int64)

        # Without mediation
        _compute_consumption_sparse_python(
            QQ, BB, VV, DD, QQbase, preyYY, predYY,
            link_prey, link_pred, 1,
        )
        q_no_med = QQ[1, 2]

        # With mediation multiplier of 0.5
        QQ2 = np.zeros((n, n))
        med_mult = np.ones((n, n))
        med_mult[1, 2] = 0.5
        _compute_consumption_sparse_python(
            QQ2, BB, VV, DD, QQbase, preyYY, predYY,
            link_prey, link_pred, 1,
            med_multipliers=med_mult,
        )
        assert QQ2[1, 2] == pytest.approx(q_no_med * 0.5)

    def test_sparse_kernel_no_mediation_unchanged(self):
        """Without med_multipliers, consumption is unchanged."""
        n = 4
        QQ = np.zeros((n, n))
        BB = np.array([0.0, 1.0, 1.0, 1.0])
        VV = np.full((n, n), 2.0)
        DD = np.full((n, n), 1000.0)
        QQbase = np.zeros((n, n))
        QQbase[1, 2] = 0.5
        preyYY = BB.copy()
        predYY = BB.copy()
        link_prey = np.array([1], dtype=np.int64)
        link_pred = np.array([2], dtype=np.int64)

        _compute_consumption_sparse_python(
            QQ, BB, VV, DD, QQbase, preyYY, predYY,
            link_prey, link_pred, 1,
        )
        assert QQ[1, 2] > 0  # should have some consumption
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `conda run -n shiny python -m pytest packages/pypath/tests/test_mediation.py::TestKernelMedMultipliers -v`
Expected: FAIL with TypeError (unexpected keyword argument 'med_multipliers')

- [ ] **Step 3: Add med_multipliers to both consumption kernels**

In `packages/pypath/src/pypath/core/ecosim_deriv.py`:

**Dense kernel** (`_compute_consumption_python`, starts at line 35): Add `med_multipliers=None` parameter. After `Q_calc = qbase * PDY * PYY_term * dd_term * vv_term` (line 172), add:

```python
            if med_multipliers is not None:
                Q_calc *= med_multipliers[prey, pred]
```

**Sparse kernel** (`_compute_consumption_sparse_python`, starts at line 190): Add `med_multipliers=None` parameter. After the equivalent Q_calc line, add:

```python
            if med_multipliers is not None:
                Q_calc *= med_multipliers[prey, pred]
```

**Also update the call sites in `deriv_vector()`** (lines 1117 and 1140): Pass `med_multipliers=_med_mult` to both kernels. Before the kernel call block (around line 1110), add:

```python
    # Mediation multipliers
    _mediation = params.get("_mediation", None)
    _med_mult = None
    if _mediation is not None:
        _med_mult = _mediation.compute_group_multipliers(BB, Bbase, ActiveLink)
```

Remove the dead `_Mediation = params.get("Mediation", {})` at line 1017.

- [ ] **Step 4: Run tests to verify they pass**

Run: `conda run -n shiny python -m pytest packages/pypath/tests/test_mediation.py::TestKernelMedMultipliers -v`
Expected: 2 PASSED

- [ ] **Step 5: Run existing Ecosim tests to verify no regression**

Run: `conda run -n shiny python -m pytest packages/pypath/tests/test_ecosim.py -v --tb=short`
Expected: All existing tests PASS (mediation defaults to None, zero overhead)

- [ ] **Step 6: Commit**

```bash
git add packages/pypath/src/pypath/core/ecosim_deriv.py packages/pypath/tests/test_mediation.py
git commit -m "feat(ecosim): add med_multipliers to consumption kernels"
```

---

### Task 9: Fleet Mediation in Fishing Loop

**Files:**
- Modify: `packages/pypath/src/pypath/core/ecosim_deriv.py`

- [ ] **Step 1: Add fleet mediation to fishing loop**

In `deriv_vector()`, before the fishing loop (around line 1232), compute fleet multipliers:

```python
    _fleet_med = None
    if _mediation is not None:
        _fleet_med = _mediation.compute_fleet_multipliers(BB, Bbase, NUM_GEARS + 1)
```

Then inside the fishing loop (line 1246), after computing `effort_mult`, multiply by fleet mediation:

```python
        if _fleet_med is not None and 0 < gear_idx < len(_fleet_med):
            effort_mult *= _fleet_med[gear_idx]
```

- [ ] **Step 2: Run existing tests**

Run: `conda run -n shiny python -m pytest packages/pypath/tests/test_ecosim.py -v --tb=short`
Expected: All PASS (no mediation = no change)

- [ ] **Step 3: Commit**

```bash
git add packages/pypath/src/pypath/core/ecosim_deriv.py
git commit -m "feat(ecosim): add fleet mediation to fishing loop"
```

---

### Task 10: rsim_run() Mediation Parameter

**Files:**
- Modify: `packages/pypath/src/pypath/core/ecosim.py`

- [ ] **Step 1: Add keyword-only mediation parameter to rsim_run()**

In `packages/pypath/src/pypath/core/ecosim.py`, modify `rsim_run()` signature (line 991):

```python
def rsim_run(
    scenario: RsimScenario,
    method: str = "RK4",
    years: Optional[range] = None,
    *,
    mediation: "MediationCollection | None" = None,
) -> RsimOutput:
```

Add `TYPE_CHECKING` import at the top of the file (near line 17):

```python
if TYPE_CHECKING:
    from pypath.core.mediation import MediationCollection
```

After `params_dict` is built (around line 1092, after the `PreyPreyWeight` setup), add:

```python
    if mediation is not None:
        params_dict["_mediation"] = mediation
```

- [ ] **Step 2: Run existing tests to verify no regression**

Run: `conda run -n shiny python -m pytest packages/pypath/tests/test_ecosim.py -v --tb=short`
Expected: All PASS

- [ ] **Step 3: Commit**

```bash
git add packages/pypath/src/pypath/core/ecosim.py
git commit -m "feat(ecosim): add keyword-only mediation parameter to rsim_run()"
```

---

### Task 11: Deprecate Old mediation_function()

**Files:**
- Modify: `packages/pypath/src/pypath/core/ecosim_deriv.py`

- [ ] **Step 1: Add deprecation note to mediation_function()**

In `packages/pypath/src/pypath/core/ecosim_deriv.py`, update the docstring of `mediation_function()` (line 716):

```python
def mediation_function(
    mediation_type: int, med_bio: float, med_base: float, med_params: Dict[str, float]
) -> float:
    """
    Calculate mediation effect on predation.

    .. deprecated::
        Use :class:`pypath.core.mediation.MediationShape.evaluate` instead.
        This function is kept for backward compatibility.

    ...rest of existing docstring...
    """
```

- [ ] **Step 2: Run existing tests**

Run: `conda run -n shiny python -m pytest packages/pypath/tests/test_ecosim.py -v --tb=short`
Expected: All PASS

- [ ] **Step 3: Commit**

```bash
git add packages/pypath/src/pypath/core/ecosim_deriv.py
git commit -m "docs: deprecate mediation_function() in favor of MediationShape"
```

---

## Chunk 4: Integration Tests & Package Exports

### Task 12: Integration Tests

**Files:**
- Create: `packages/pypath/tests/test_mediation_integration.py`

- [ ] **Step 1: Write integration tests**

Create `packages/pypath/tests/test_mediation_integration.py`:

```python
"""Integration tests for mediation functions with Ecosim simulation."""
import numpy as np
import pytest

from pypath.core.ecopath import rpath
from pypath.core.ecosim import rsim_run, rsim_scenario
from pypath.core.mediation import (
    MediationCollection,
    MediationLink,
    MediationShape,
    make_positive_shape,
    make_negative_shape,
)
from pypath.core.params import create_rpath_params


def _make_3group_model():
    """Create a minimal 3-group model: producer -> consumer -> predator + detritus."""
    params = create_rpath_params(model_name="med_test", num_groups=4, num_fleets=1)
    # Group 0: producer (type=1)
    params.model.loc[0, "Type"] = 1
    params.model.loc[0, "Biomass"] = 10.0
    params.model.loc[0, "PB"] = 2.0
    # Group 1: consumer (type=0)
    params.model.loc[1, "Type"] = 0
    params.model.loc[1, "Biomass"] = 5.0
    params.model.loc[1, "PB"] = 1.0
    params.model.loc[1, "QB"] = 5.0
    # Group 2: predator (type=0)
    params.model.loc[2, "Type"] = 0
    params.model.loc[2, "Biomass"] = 2.0
    params.model.loc[2, "PB"] = 0.5
    params.model.loc[2, "QB"] = 3.0
    # Group 3: detritus (type=2)
    params.model.loc[3, "Type"] = 2
    params.model.loc[3, "Biomass"] = 50.0
    params.model.loc[3, "PB"] = 0.0

    # Diet: consumer eats producer, predator eats consumer
    params.diet = np.zeros((4, 4))
    params.diet[0, 1] = 1.0  # consumer eats producer
    params.diet[1, 2] = 1.0  # predator eats consumer

    rpath_result = rpath(params)
    return rpath_result, params


@pytest.mark.slow
class TestMediationIntegration:
    def test_no_mediation_baseline(self):
        """rsim_run without mediation produces same result as before."""
        rpath_result, params = _make_3group_model()
        scenario = rsim_scenario(rpath_result, params, years=range(1, 11))
        result = rsim_run(scenario)
        assert result.out_Biomass.shape[0] > 0

    def test_positive_mediation_changes_biomass(self):
        """With positive mediation, biomass trajectories differ from baseline."""
        rpath_result, params = _make_3group_model()
        scenario_base = rsim_scenario(rpath_result, params, years=range(1, 11))
        result_base = rsim_run(scenario_base)

        # Consumer (group 1) mediates producer-predator link: more consumer -> more predation
        shape = make_positive_shape(shape_id=1, low=0.5, high=2.0)
        link = MediationLink(shape_id=1, mediator_idx=1, prey_idx=0, pred_idx=2)
        med = MediationCollection(shapes=[shape], links=[link])

        scenario_med = rsim_scenario(rpath_result, params, years=range(1, 11))
        result_med = rsim_run(scenario_med, mediation=med)

        # Biomass trajectories should differ
        base_bio = result_base.out_Biomass[-1, :]
        med_bio = result_med.out_Biomass[-1, :]
        assert not np.allclose(base_bio, med_bio, atol=1e-6)

    def test_negative_mediation_changes_biomass(self):
        """Negative mediation: more mediator -> less predation -> prey benefits."""
        rpath_result, params = _make_3group_model()

        shape = make_negative_shape(shape_id=1, low=0.5, high=2.0)
        link = MediationLink(shape_id=1, mediator_idx=1, prey_idx=0, pred_idx=2)
        med = MediationCollection(shapes=[shape], links=[link])

        scenario = rsim_scenario(rpath_result, params, years=range(1, 11))
        result = rsim_run(scenario, mediation=med)
        assert result.out_Biomass.shape[0] > 0

    def test_regression_none_mediation(self):
        """Passing mediation=None gives identical results to no mediation."""
        rpath_result, params = _make_3group_model()
        scenario1 = rsim_scenario(rpath_result, params, years=range(1, 11))
        result1 = rsim_run(scenario1)

        scenario2 = rsim_scenario(rpath_result, params, years=range(1, 11))
        result2 = rsim_run(scenario2, mediation=None)

        np.testing.assert_allclose(
            result1.out_Biomass, result2.out_Biomass, atol=1e-12
        )
```

- [ ] **Step 2: Run integration tests**

Run: `conda run -n shiny python -m pytest packages/pypath/tests/test_mediation_integration.py -v --tb=short`
Expected: 4 PASSED (may be slow)

- [ ] **Step 3: Commit**

```bash
git add packages/pypath/tests/test_mediation_integration.py
git commit -m "test(mediation): add integration tests with 3-group Ecosim model"
```

---

### Task 13: Package __init__.py Exports

**Files:**
- Modify: `packages/pypath/src/pypath/core/__init__.py`
- Modify: `packages/pypath/src/pypath/io/__init__.py`

- [ ] **Step 1: Add mediation exports to core/__init__.py**

In `packages/pypath/src/pypath/core/__init__.py`, add import block after the calibration import:

```python
from pypath.core.mediation import (
    MediationCollection,
    MediationLink,
    MediationShape,
    make_negative_shape,
    make_positive_shape,
    make_ushape,
)
```

Add to `__all__` (after the Calibration section):

```python
    # Mediation
    "MediationShape",
    "MediationLink",
    "MediationCollection",
    "make_positive_shape",
    "make_negative_shape",
    "make_ushape",
```

- [ ] **Step 2: Add read_mediation to io/__init__.py**

In `packages/pypath/src/pypath/io/__init__.py`, add `read_mediation` to the ewemdb import:

```python
from pypath.io.ewemdb import (
    EwEDatabaseError,
    check_ewemdb_support,
    get_ewemdb_metadata,
    list_ewemdb_tables,
    read_ewemdb,
    read_ewemdb_table,
    read_mediation,
    read_timeseries,
)
```

Add to `__all__` (in the EwE database section):

```python
    "read_mediation",
```

- [ ] **Step 3: Verify all imports work**

Run: `conda run -n shiny python -c "from pypath.core.mediation import MediationShape, MediationLink, MediationCollection, make_positive_shape, make_negative_shape, make_ushape; from pypath.io.ewemdb import read_mediation; print('All spec import paths verified')"`
Expected: `All spec import paths verified`

- [ ] **Step 4: Commit**

```bash
git add packages/pypath/src/pypath/core/__init__.py packages/pypath/src/pypath/io/__init__.py
git commit -m "feat: add mediation exports to core and io __init__.py"
```

---

### Task 14: Final Verification & Cleanup

**Files:**
- All created/modified files

- [ ] **Step 1: Run all mediation tests**

Run: `conda run -n shiny python -m pytest packages/pypath/tests/test_mediation.py packages/pypath/tests/test_mediation_io.py packages/pypath/tests/test_mediation_integration.py -v`
Expected: All PASSED

- [ ] **Step 2: Run full core test suite**

Run: `conda run -n shiny python -m pytest packages/pypath/tests/ -q -m "not integration" --ignore=packages/pypath/tests/scripts`
Expected: All PASSED, no regressions

- [ ] **Step 3: Verify spec import paths**

Run: `conda run -n shiny python -c "from pypath.core.mediation import MediationShape, MediationLink, MediationCollection, make_positive_shape, make_negative_shape, make_ushape; from pypath.io.ewemdb import read_mediation; print('OK')"`
Expected: `OK`
