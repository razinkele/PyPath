# Monte Carlo / Pedigree Uncertainty Implementation Plan

> **For agentic workers:** REQUIRED: Use superpowers:subagent-driven-development (if subagents available) or superpowers:executing-plans to implement this plan. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add pedigree-based Monte Carlo uncertainty analysis, Morris screening, and optional Sobol sensitivity analysis to PyPath.

**Architecture:** Three new modules: `core/pedigree.py` (distributions + sampling), `core/montecarlo.py` (MC runner + results), `core/sensitivity.py` (Morris + Sobol). Dependency flow: sensitivity → montecarlo → pedigree → params. I/O adds 6 EwE schema tables and `read_pedigree()`.

**Tech Stack:** numpy, pandas, scipy.stats.qmc (LHS), dataclasses. Optional: joblib (parallelism), SALib (Sobol).

**Spec:** `docs/superpowers/specs/2026-03-11-montecarlo-pedigree-design.md`

---

## Chunk 1: Pedigree Data Structures & Sampling

### Task 1: ScalarDistribution and DietDistribution Dataclasses

**Files:**
- Create: `packages/pypath/src/pypath/core/pedigree.py`
- Create: `packages/pypath/tests/test_pedigree.py`

- [ ] **Step 1: Write failing tests for distribution dataclasses**

Create `packages/pypath/tests/test_pedigree.py`:

```python
"""Tests for pypath.core.pedigree module."""
import numpy as np
import pytest

from pypath.core.pedigree import ScalarDistribution, DietDistribution


class TestScalarDistribution:
    def test_construction(self):
        d = ScalarDistribution(
            param_name="Biomass", group_idx=0, base_value=10.0, cv=0.2,
        )
        assert d.param_name == "Biomass"
        assert d.group_idx == 0
        assert d.base_value == 10.0
        assert d.cv == 0.2
        assert d.bounds is None

    def test_with_bounds(self):
        d = ScalarDistribution(
            param_name="PB", group_idx=1, base_value=5.0, cv=0.3,
            bounds=(1.0, 20.0),
        )
        assert d.bounds == (1.0, 20.0)


class TestDietDistribution:
    def test_construction(self):
        props = np.array([0.6, 0.3, 0.1, 0.0])
        d = DietDistribution(pred_idx=1, base_proportions=props, cv=0.2)
        assert d.pred_idx == 1
        assert d.cv == 0.2
        np.testing.assert_array_equal(d.base_proportions, props)

    def test_base_proportions_sum_to_one(self):
        props = np.array([0.5, 0.3, 0.2])
        d = DietDistribution(pred_idx=0, base_proportions=props, cv=0.1)
        assert np.sum(d.base_proportions) == pytest.approx(1.0)
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `conda run -n shiny python -m pytest packages/pypath/tests/test_pedigree.py -v`
Expected: FAIL with ImportError (pedigree module does not exist)

- [ ] **Step 3: Implement ScalarDistribution and DietDistribution**

Create `packages/pypath/src/pypath/core/pedigree.py`:

```python
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
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `conda run -n shiny python -m pytest packages/pypath/tests/test_pedigree.py -v`
Expected: 4 PASSED

- [ ] **Step 5: Commit**

```bash
git add packages/pypath/src/pypath/core/pedigree.py packages/pypath/tests/test_pedigree.py
git commit -m "feat(pedigree): add ScalarDistribution and DietDistribution dataclasses"
```

---

### Task 2: PedigreeConfig and build_distributions()

**Files:**
- Modify: `packages/pypath/src/pypath/core/pedigree.py`
- Modify: `packages/pypath/tests/test_pedigree.py`

- [ ] **Step 1: Write failing tests for PedigreeConfig and build_distributions**

Append to `packages/pypath/tests/test_pedigree.py`:

```python
from pypath.core.pedigree import PedigreeConfig, build_distributions
from pypath.core.params import create_rpath_params


class TestPedigreeConfig:
    def test_default_empty(self):
        config = PedigreeConfig()
        assert config.level_to_cv == {}

    def test_custom_mapping(self):
        config = PedigreeConfig(level_to_cv={
            "PBInput": {6: 0.1, 7: 0.2},
        })
        assert config.level_to_cv["PBInput"][6] == 0.1


class TestBuildDistributions:
    def _make_params(self):
        params = create_rpath_params(
            groups=["Producer", "Consumer", "Detritus"],
            types=[1, 0, 2],
        )
        params.model.loc[0, "Biomass"] = 10.0
        params.model.loc[0, "PB"] = 200.0
        params.model.loc[1, "Biomass"] = 5.0
        params.model.loc[1, "PB"] = 50.0
        params.model.loc[1, "QB"] = 150.0
        params.model.loc[2, "Biomass"] = 100.0
        # Set pedigree CVs
        params.pedigree.loc[0, "Biomass"] = 0.2
        params.pedigree.loc[0, "PB"] = 0.3
        params.pedigree.loc[1, "Biomass"] = 0.1
        params.pedigree.loc[1, "PB"] = 0.2
        params.pedigree.loc[1, "QB"] = 0.15
        params.pedigree.loc[1, "Diet"] = 0.2
        params.pedigree.loc[2, "Biomass"] = 0.0  # known exactly
        return params

    def test_builds_scalar_distributions(self):
        params = self._make_params()
        dists = build_distributions(params)
        scalars = [d for d in dists if isinstance(d, ScalarDistribution)]
        # Producer: Biomass(0.2), PB(0.3); Consumer: Biomass(0.1), PB(0.2), QB(0.15)
        # Detritus Biomass skipped (CV=0)
        assert len(scalars) >= 5

    def test_skips_zero_cv(self):
        params = self._make_params()
        dists = build_distributions(params)
        # Detritus biomass has CV=0, should be skipped
        det_bio = [d for d in dists
                   if isinstance(d, ScalarDistribution)
                   and d.param_name == "Biomass" and d.group_idx == 2]
        assert len(det_bio) == 0

    def test_builds_diet_distribution(self):
        params = self._make_params()
        params.diet["Consumer"] = [1.0, 0.0, 0.0, 0.0]  # 3 groups + import
        dists = build_distributions(params)
        diets = [d for d in dists if isinstance(d, DietDistribution)]
        assert len(diets) >= 1
        assert diets[0].pred_idx == 1  # Consumer

    def test_warns_default_pedigree(self):
        params = create_rpath_params(
            groups=["A", "B", "Detritus"], types=[1, 0, 2],
        )
        params.model.loc[0, "Biomass"] = 10.0
        params.model.loc[1, "Biomass"] = 5.0
        params.model.loc[2, "Biomass"] = 100.0
        with pytest.warns(UserWarning, match="1.0"):
            build_distributions(params)

    def test_skips_producers_qb(self):
        """Producers don't have QB, should not create QB distribution."""
        params = self._make_params()
        dists = build_distributions(params)
        producer_qb = [d for d in dists
                       if isinstance(d, ScalarDistribution)
                       and d.param_name == "QB" and d.group_idx == 0]
        assert len(producer_qb) == 0

    def test_skips_detritus_pedigree(self):
        """Detritus groups (type=2) should not get PB/QB distributions."""
        params = self._make_params()
        dists = build_distributions(params)
        det_pb = [d for d in dists
                  if isinstance(d, ScalarDistribution)
                  and d.param_name == "PB" and d.group_idx == 2]
        assert len(det_pb) == 0
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `conda run -n shiny python -m pytest packages/pypath/tests/test_pedigree.py::TestBuildDistributions -v`
Expected: FAIL with ImportError

- [ ] **Step 3: Implement PedigreeConfig and build_distributions**

Append to `packages/pypath/src/pypath/core/pedigree.py`:

```python
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
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `conda run -n shiny python -m pytest packages/pypath/tests/test_pedigree.py -v`
Expected: All PASSED

- [ ] **Step 5: Commit**

```bash
git add packages/pypath/src/pypath/core/pedigree.py packages/pypath/tests/test_pedigree.py
git commit -m "feat(pedigree): add PedigreeConfig and build_distributions()"
```

---

### Task 3: sample_parameters() and apply_sample()

**Files:**
- Modify: `packages/pypath/src/pypath/core/pedigree.py`
- Modify: `packages/pypath/tests/test_pedigree.py`

- [ ] **Step 1: Write failing tests for sample_parameters and apply_sample**

Append to `packages/pypath/tests/test_pedigree.py`:

```python
from pypath.core.pedigree import sample_parameters, apply_sample


class TestSampleParameters:
    def _make_distributions(self):
        return [
            ScalarDistribution("Biomass", 0, 10.0, 0.2),
            ScalarDistribution("PB", 0, 200.0, 0.3),
            ScalarDistribution("Biomass", 1, 5.0, 0.1),
            DietDistribution(1, np.array([0.6, 0.3, 0.1, 0.0]), 0.2),
        ]

    def test_returns_correct_count(self):
        dists = self._make_distributions()
        samples = sample_parameters(dists, n_samples=5, method="random",
                                     rng=np.random.default_rng(42))
        assert len(samples) == 5

    def test_sample_keys(self):
        dists = self._make_distributions()
        samples = sample_parameters(dists, n_samples=3, method="random",
                                     rng=np.random.default_rng(42))
        s = samples[0]
        assert ("Biomass", 0) in s
        assert ("PB", 0) in s
        assert ("Diet", 1) in s

    def test_scalar_values_positive(self):
        dists = [ScalarDistribution("Biomass", 0, 10.0, 0.2)]
        samples = sample_parameters(dists, n_samples=100, method="random",
                                     rng=np.random.default_rng(42))
        for s in samples:
            assert s[("Biomass", 0)] > 0

    def test_diet_sums_to_one(self):
        dists = [DietDistribution(1, np.array([0.6, 0.3, 0.1, 0.0]), 0.2)]
        samples = sample_parameters(dists, n_samples=50, method="random",
                                     rng=np.random.default_rng(42))
        for s in samples:
            diet = s[("Diet", 1)]
            assert np.sum(diet) == pytest.approx(1.0, abs=1e-10)
            assert diet[3] == 0.0  # zero preserved

    def test_seed_reproducibility(self):
        dists = self._make_distributions()
        s1 = sample_parameters(dists, 5, "random", rng=np.random.default_rng(123))
        s2 = sample_parameters(dists, 5, "random", rng=np.random.default_rng(123))
        for a, b in zip(s1, s2):
            assert a[("Biomass", 0)] == b[("Biomass", 0)]

    def test_lhs_returns_correct_count(self):
        dists = [ScalarDistribution("Biomass", 0, 10.0, 0.2)]
        samples = sample_parameters(dists, n_samples=10, method="lhs",
                                     rng=np.random.default_rng(42))
        assert len(samples) == 10

    def test_lhs_values_positive(self):
        dists = [ScalarDistribution("Biomass", 0, 10.0, 0.2)]
        samples = sample_parameters(dists, n_samples=50, method="lhs",
                                     rng=np.random.default_rng(42))
        for s in samples:
            assert s[("Biomass", 0)] > 0


class TestApplySample:
    def test_applies_scalar_values(self):
        params = create_rpath_params(
            groups=["Producer", "Consumer", "Detritus"], types=[1, 0, 2],
        )
        params.model.loc[0, "Biomass"] = 10.0
        params.model.loc[1, "Biomass"] = 5.0
        sample = {("Biomass", 0): 12.0, ("Biomass", 1): 4.5}
        new_params = apply_sample(params, sample)
        assert new_params.model.loc[0, "Biomass"] == 12.0
        assert new_params.model.loc[1, "Biomass"] == 4.5

    def test_original_unchanged(self):
        params = create_rpath_params(
            groups=["Producer", "Consumer", "Detritus"], types=[1, 0, 2],
        )
        params.model.loc[0, "Biomass"] = 10.0
        sample = {("Biomass", 0): 99.0}
        apply_sample(params, sample)
        assert params.model.loc[0, "Biomass"] == 10.0

    def test_applies_diet(self):
        params = create_rpath_params(
            groups=["Producer", "Consumer", "Detritus"], types=[1, 0, 2],
        )
        params.diet["Consumer"] = [0.8, 0.0, 0.2, 0.0]
        new_diet = np.array([0.7, 0.0, 0.3, 0.0])
        sample = {("Diet", 1): new_diet}
        new_params = apply_sample(params, sample)
        np.testing.assert_array_almost_equal(
            new_params.diet["Consumer"].values, new_diet,
        )
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `conda run -n shiny python -m pytest packages/pypath/tests/test_pedigree.py::TestSampleParameters packages/pypath/tests/test_pedigree.py::TestApplySample -v`
Expected: FAIL with ImportError

- [ ] **Step 3: Implement sample_parameters and apply_sample**

Append to `packages/pypath/src/pypath/core/pedigree.py`:

```python
def sample_parameters(
    distributions: list[ParameterDistribution],
    n_samples: int,
    method: str = "lhs",
    rng: np.random.Generator | None = None,
) -> list[dict]:
    """Generate parameter samples from distributions.

    Parameters
    ----------
    distributions : list[ParameterDistribution]
        Parameter distributions from build_distributions().
    n_samples : int
        Number of samples to generate.
    method : str
        "lhs" for Latin Hypercube Sampling, "random" for direct sampling.
    rng : np.random.Generator, optional
        Random number generator for reproducibility.

    Returns
    -------
    list[dict]
        N parameter sets. Keys are (param_name, group_idx) tuples.
    """
    if rng is None:
        rng = np.random.default_rng()

    scalars = [d for d in distributions if isinstance(d, ScalarDistribution)]
    diets = [d for d in distributions if isinstance(d, DietDistribution)]

    # Generate scalar samples
    if method == "lhs" and len(scalars) > 0:
        from scipy.stats.qmc import LatinHypercube

        sampler = LatinHypercube(d=len(scalars), seed=rng)
        unit_samples = sampler.random(n=n_samples)  # (n_samples, n_scalars)
        scalar_samples = np.empty_like(unit_samples)
        for j, dist in enumerate(scalars):
            sigma = math.sqrt(math.log(1 + dist.cv**2))
            mu = math.log(dist.base_value) - sigma**2 / 2
            # Map uniform [0,1] to lognormal via inverse CDF
            from scipy.stats import lognorm

            scalar_samples[:, j] = lognorm.ppf(
                unit_samples[:, j], s=sigma, scale=math.exp(mu),
            )
    else:
        # Direct random sampling
        scalar_samples = np.empty((n_samples, len(scalars)))
        for j, dist in enumerate(scalars):
            sigma = math.sqrt(math.log(1 + dist.cv**2))
            mu = math.log(dist.base_value) - sigma**2 / 2
            scalar_samples[:, j] = rng.lognormal(mean=mu, sigma=sigma, size=n_samples)

    # Generate diet samples (always direct — no LHS for Dirichlet)
    # Note: the import row (last element) is included in Dirichlet sampling
    # if it has a non-zero proportion. This means import fraction varies
    # with other diet proportions. To fix import, set its proportion to 0.
    diet_samples: list[list[np.ndarray]] = []
    for dist in diets:
        nonzero_mask = dist.base_proportions > 0
        p_nonzero = dist.base_proportions[nonzero_mask]
        alpha = p_nonzero / dist.cv**2
        samples_for_diet = []
        for _ in range(n_samples):
            sampled = rng.dirichlet(alpha)
            full = np.zeros_like(dist.base_proportions)
            full[nonzero_mask] = sampled
            samples_for_diet.append(full)
        diet_samples.append(samples_for_diet)

    # Assemble into list of dicts
    result = []
    for i in range(n_samples):
        sample: dict = {}
        for j, dist in enumerate(scalars):
            sample[(dist.param_name, dist.group_idx)] = float(scalar_samples[i, j])
        for k, dist in enumerate(diets):
            sample[("Diet", dist.pred_idx)] = diet_samples[k][i]
        result.append(sample)

    return result


def apply_sample(params: "RpathParams", sample: dict) -> "RpathParams":
    """Apply a parameter sample to a copy of RpathParams.

    Parameters
    ----------
    params : RpathParams
        Original parameters (not modified).
    sample : dict
        Parameter sample from sample_parameters().

    Returns
    -------
    RpathParams
        New params with sampled values applied.
    """
    from pypath.core.params import RpathParams, RpathStanzaParams

    new_model = params.model.copy()
    new_diet = params.diet.copy()

    # Deep copy stanzas if present
    new_stanzas = copy.deepcopy(params.stanzas)

    for key, value in sample.items():
        param_name, idx = key
        if param_name == "Diet":
            group_name = new_model.loc[idx, "Group"]
            if group_name in new_diet.columns:
                new_diet[group_name] = value
        else:
            new_model.loc[idx, param_name] = value

    return RpathParams(
        model=new_model,
        diet=new_diet,
        stanzas=new_stanzas,
        pedigree=params.pedigree,
        remarks=params.remarks,
        ecosim=params.ecosim,
    )
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `conda run -n shiny python -m pytest packages/pypath/tests/test_pedigree.py -v`
Expected: All PASSED

- [ ] **Step 5: Commit**

```bash
git add packages/pypath/src/pypath/core/pedigree.py packages/pypath/tests/test_pedigree.py
git commit -m "feat(pedigree): add sample_parameters() and apply_sample()"
```

---

## Chunk 2: Monte Carlo Runner

### Task 4: MCConfig and MCResult Dataclasses

**Files:**
- Create: `packages/pypath/src/pypath/core/montecarlo.py`
- Create: `packages/pypath/tests/test_montecarlo.py`

- [ ] **Step 1: Write failing tests for MCConfig and MCResult**

Create `packages/pypath/tests/test_montecarlo.py`:

```python
"""Tests for pypath.core.montecarlo module."""
import numpy as np
import pandas as pd
import pytest

from pypath.core.montecarlo import MCConfig, MCResult


class TestMCConfig:
    def test_defaults(self):
        config = MCConfig()
        assert config.n_samples == 1000
        assert config.method == "lhs"
        assert config.seed is None
        assert config.ecopath_only is False
        assert config.ecosim_years is None
        assert config.store_runs is False
        assert config.n_jobs == 1
        assert config.ecosim_method == "RK4"
        assert config.eco_area == 1.0

    def test_custom_values(self):
        config = MCConfig(n_samples=50, method="random", seed=42, ecopath_only=True)
        assert config.n_samples == 50
        assert config.method == "random"
        assert config.ecopath_only is True


class TestMCResult:
    def test_construction(self):
        result = MCResult(
            n_total=100, n_feasible=80, n_ecosim=0,
            ecopath_stats={"Biomass": pd.DataFrame({"mean": [1.0]})},
            ecosim_stats=None,
            ecopath_runs=None, ecosim_runs=None,
            feasibility_rate=0.8, parameter_samples=None,
        )
        assert result.n_total == 100
        assert result.n_feasible == 80
        assert result.feasibility_rate == 0.8

    def test_to_dataframe(self):
        stats = pd.DataFrame({
            "mean": [10.0, 5.0], "std": [1.0, 0.5],
            "p5": [8.0, 4.0], "p25": [9.0, 4.5],
            "p50": [10.0, 5.0], "p75": [11.0, 5.5], "p95": [12.0, 6.0],
        })
        result = MCResult(
            n_total=100, n_feasible=80, n_ecosim=0,
            ecopath_stats={"Biomass": stats},
            ecosim_stats=None,
            ecopath_runs=None, ecosim_runs=None,
            feasibility_rate=0.8, parameter_samples=None,
        )
        df = result.to_dataframe()
        assert isinstance(df, pd.DataFrame)
        assert len(df) > 0

    def test_to_dict(self):
        result = MCResult(
            n_total=10, n_feasible=8, n_ecosim=0,
            ecopath_stats={},
            ecosim_stats=None,
            ecopath_runs=None, ecosim_runs=None,
            feasibility_rate=0.8, parameter_samples=None,
        )
        d = result.to_dict()
        assert isinstance(d, dict)
        assert d["n_total"] == 10
        assert d["feasibility_rate"] == 0.8
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `conda run -n shiny python -m pytest packages/pypath/tests/test_montecarlo.py -v`
Expected: FAIL with ImportError

- [ ] **Step 3: Implement MCConfig and MCResult**

Create `packages/pypath/src/pypath/core/montecarlo.py`:

```python
"""Monte Carlo uncertainty analysis for Ecopath/Ecosim.

Runs ensemble simulations with parameter sampling from pedigree-defined
distributions. Supports parallel execution and streaming statistics.
"""
from __future__ import annotations

import logging
import warnings
from dataclasses import dataclass, field
from typing import Any, Callable

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# Optional parallel execution
try:
    from joblib import Parallel, delayed

    HAS_JOBLIB = True
except ImportError:
    HAS_JOBLIB = False


@dataclass
class MCConfig:
    """Monte Carlo run configuration."""

    n_samples: int = 1000
    method: str = "lhs"
    seed: int | None = None
    ecopath_only: bool = False
    ecosim_years: range | None = None
    store_runs: bool = False
    n_jobs: int = 1
    mediation: Any = None
    ecosim_method: str = "RK4"
    eco_area: float = 1.0


@dataclass
class MCResult:
    """Monte Carlo ensemble results."""

    n_total: int
    n_feasible: int
    n_ecosim: int
    ecopath_stats: dict[str, pd.DataFrame]
    ecosim_stats: dict[str, np.ndarray] | None
    ecopath_runs: list[dict] | None
    ecosim_runs: list[np.ndarray] | None
    feasibility_rate: float
    parameter_samples: pd.DataFrame | None

    def to_dataframe(self) -> pd.DataFrame:
        """Return ecopath_stats as a single flat DataFrame."""
        frames = []
        for param_name, df in self.ecopath_stats.items():
            df_copy = df.copy()
            df_copy.insert(0, "parameter", param_name)
            frames.append(df_copy)
        if frames:
            return pd.concat(frames, ignore_index=True)
        return pd.DataFrame()

    def to_dict(self) -> dict:
        """Return JSON-serializable summary dict."""
        return {
            "n_total": self.n_total,
            "n_feasible": self.n_feasible,
            "n_ecosim": self.n_ecosim,
            "feasibility_rate": self.feasibility_rate,
            "ecopath_stats": {
                k: v.to_dict(orient="list") for k, v in self.ecopath_stats.items()
            },
        }
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `conda run -n shiny python -m pytest packages/pypath/tests/test_montecarlo.py -v`
Expected: All PASSED

- [ ] **Step 5: Commit**

```bash
git add packages/pypath/src/pypath/core/montecarlo.py packages/pypath/tests/test_montecarlo.py
git commit -m "feat(montecarlo): add MCConfig and MCResult dataclasses"
```

---

### Task 5: run_montecarlo() Implementation

**Files:**
- Modify: `packages/pypath/src/pypath/core/montecarlo.py`
- Modify: `packages/pypath/tests/test_montecarlo.py`

- [ ] **Step 1: Write failing tests for run_montecarlo**

Append to `packages/pypath/tests/test_montecarlo.py`:

```python
import warnings as _warnings

from pypath.core.montecarlo import run_montecarlo
from pypath.core.params import create_rpath_params


def _make_mc_params():
    """3-group model with moderate pedigree CVs."""
    params = create_rpath_params(
        groups=["Producer", "Consumer", "Detritus"],
        types=[1, 0, 2],
    )
    params.model.loc[0, "Biomass"] = 10.0
    params.model.loc[0, "PB"] = 200.0
    params.model.loc[0, "EE"] = 0.8
    params.model.loc[1, "Biomass"] = 5.0
    params.model.loc[1, "PB"] = 50.0
    params.model.loc[1, "QB"] = 150.0
    params.model.loc[1, "EE"] = 0.9
    params.model.loc[2, "Biomass"] = 100.0
    params.model["BioAcc"] = 0.0
    params.model["Unassim"] = 0.2
    params.model.loc[0, "Unassim"] = 0.0
    params.model.loc[2, "Unassim"] = 0.0
    params.model["Detritus"] = 1.0
    params.model.loc[2, "Detritus"] = 0.0
    params.diet["Consumer"] = [1.0, 0.0, 0.0, 0.0]
    # Set moderate pedigree
    params.pedigree["Biomass"] = [0.1, 0.1, 0.0]
    params.pedigree["PB"] = [0.1, 0.1, 0.0]
    params.pedigree["QB"] = [0.0, 0.1, 0.0]
    params.pedigree["Diet"] = [0.0, 0.0, 0.0]
    return params


class TestRunMontecarlo:
    def test_ecopath_only(self):
        params = _make_mc_params()
        config = MCConfig(n_samples=20, method="random", seed=42, ecopath_only=True)
        result = run_montecarlo(params, config)
        assert result.n_total == 20
        assert result.n_feasible > 0
        assert result.feasibility_rate > 0
        assert "Biomass" in result.ecopath_stats
        assert result.ecosim_stats is None

    def test_store_runs(self):
        params = _make_mc_params()
        config = MCConfig(n_samples=10, method="random", seed=42,
                          ecopath_only=True, store_runs=True)
        result = run_montecarlo(params, config)
        assert result.ecopath_runs is not None
        assert len(result.ecopath_runs) == result.n_feasible

    def test_zero_cv_all_identical(self):
        params = _make_mc_params()
        # Set all CVs to 0
        params.pedigree["Biomass"] = [0.0, 0.0, 0.0]
        params.pedigree["PB"] = [0.0, 0.0, 0.0]
        params.pedigree["QB"] = [0.0, 0.0, 0.0]
        config = MCConfig(n_samples=5, method="random", seed=42, ecopath_only=True)
        with _warnings.catch_warnings():
            _warnings.simplefilter("ignore")  # no distributions warning
            result = run_montecarlo(params, config)
        # With zero CV, all samples are identical → all feasible or all fail
        if result.n_feasible > 1:
            bio = result.ecopath_stats["Biomass"]
            assert bio["std"].iloc[0] == pytest.approx(0.0, abs=1e-10)

    def test_with_ecosim(self):
        params = _make_mc_params()
        config = MCConfig(
            n_samples=5, method="random", seed=42,
            ecopath_only=False, ecosim_years=range(1, 6),
        )
        result = run_montecarlo(params, config)
        assert result.n_ecosim > 0, "Expected at least 1 successful Ecosim run"
        assert result.ecosim_stats is not None

    def test_progress_callback(self):
        params = _make_mc_params()
        config = MCConfig(n_samples=5, method="random", seed=42, ecopath_only=True)
        calls = []
        result = run_montecarlo(params, config,
                                 progress_callback=lambda i, n: calls.append((i, n)))
        assert len(calls) > 0
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `conda run -n shiny python -m pytest packages/pypath/tests/test_montecarlo.py::TestRunMontecarlo -v`
Expected: FAIL with ImportError

- [ ] **Step 3: Implement run_montecarlo**

Append to `packages/pypath/src/pypath/core/montecarlo.py`:

```python
def _run_single_ecopath(sampled_params, eco_area):
    """Run a single Ecopath mass balance, return result or None."""
    from pypath.core.ecopath import rpath

    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            return rpath(sampled_params, eco_area=eco_area)
    except Exception:
        return None


def _run_single_ecosim(rpath_result, sampled_params, config):
    """Run a single Ecosim simulation, return out_Biomass or None."""
    from pypath.core.ecosim import rsim_run, rsim_scenario

    try:
        years = config.ecosim_years if config.ecosim_years is not None else range(1, 11)
        scenario = rsim_scenario(rpath_result, sampled_params, years=years)
        result = rsim_run(scenario, method=config.ecosim_method,
                          mediation=config.mediation)
        return result.out_Biomass
    except Exception:
        return None


def run_montecarlo(
    params: "RpathParams",
    config: MCConfig | None = None,
    *,
    pedigree_config: "PedigreeConfig | None" = None,
    progress_callback: Callable[[int, int], None] | None = None,
) -> MCResult:
    """Run Monte Carlo uncertainty analysis.

    Parameters
    ----------
    params : RpathParams
        Base Ecopath parameters.
    config : MCConfig, optional
        MC configuration. Defaults to MCConfig().
    pedigree_config : PedigreeConfig, optional
        EwE pedigree mapping.
    progress_callback : callable, optional
        Called with (current_sample, total_samples) after each run.

    Returns
    -------
    MCResult
    """
    from pypath.core.pedigree import (
        PedigreeConfig,
        apply_sample,
        build_distributions,
        sample_parameters,
    )

    if config is None:
        config = MCConfig()

    rng = np.random.default_rng(config.seed)
    distributions = build_distributions(params, pedigree_config)

    if len(distributions) == 0:
        warnings.warn("No distributions to sample (all CVs are 0).", UserWarning)

    samples = sample_parameters(distributions, config.n_samples, config.method, rng)

    # Collect results — exclude fleet groups (type=3) from biomass stats
    n_groups = int((params.model["Type"] != 3).sum())
    ecopath_biomass = []
    ecopath_runs_list = [] if config.store_runs else None
    ecosim_biomass_list = []
    ecosim_runs_list = [] if config.store_runs else None
    n_feasible = 0
    n_ecosim = 0

    for i, sample in enumerate(samples):
        sampled_params = apply_sample(params, sample)
        rpath_result = _run_single_ecopath(sampled_params, config.eco_area)

        if rpath_result is not None:
            n_feasible += 1
            bio = rpath_result.Biomass[:n_groups] if hasattr(rpath_result, "Biomass") else None
            if bio is not None:
                ecopath_biomass.append(bio.copy())
            if config.store_runs:
                ecopath_runs_list.append({
                    "Biomass": bio.copy() if bio is not None else None,
                })

            if not config.ecopath_only:
                ecosim_bio = _run_single_ecosim(rpath_result, sampled_params, config)
                if ecosim_bio is not None:
                    n_ecosim += 1
                    ecosim_biomass_list.append(ecosim_bio)
                    if config.store_runs:
                        ecosim_runs_list.append(ecosim_bio.copy())

        if progress_callback is not None:
            progress_callback(i + 1, config.n_samples)

    # Compute ecopath statistics
    ecopath_stats = {}
    if ecopath_biomass:
        bio_array = np.array(ecopath_biomass)  # (n_feasible, n_groups)
        ecopath_stats["Biomass"] = pd.DataFrame({
            "mean": np.mean(bio_array, axis=0),
            "std": np.std(bio_array, axis=0),
            "p5": np.percentile(bio_array, 5, axis=0),
            "p25": np.percentile(bio_array, 25, axis=0),
            "p50": np.percentile(bio_array, 50, axis=0),
            "p75": np.percentile(bio_array, 75, axis=0),
            "p95": np.percentile(bio_array, 95, axis=0),
        })

    # Compute ecosim statistics
    ecosim_stats = None
    if ecosim_biomass_list:
        # All arrays should have same shape; exclude padding col 0
        min_t = min(b.shape[0] for b in ecosim_biomass_list)
        stacked = np.array([b[:min_t, 1:n_groups + 1] for b in ecosim_biomass_list])
        # stacked: (n_ecosim, timesteps, n_groups)
        ecosim_stats = {
            "Biomass": np.stack([
                np.mean(stacked, axis=0),
                np.std(stacked, axis=0),
                np.percentile(stacked, 5, axis=0),
                np.percentile(stacked, 25, axis=0),
                np.percentile(stacked, 50, axis=0),
                np.percentile(stacked, 75, axis=0),
                np.percentile(stacked, 95, axis=0),
            ], axis=-1),  # (timesteps, n_groups, 7)
        }

    feasibility_rate = n_feasible / config.n_samples if config.n_samples > 0 else 0.0

    return MCResult(
        n_total=config.n_samples,
        n_feasible=n_feasible,
        n_ecosim=n_ecosim,
        ecopath_stats=ecopath_stats,
        ecosim_stats=ecosim_stats,
        ecopath_runs=ecopath_runs_list,
        ecosim_runs=ecosim_runs_list,
        feasibility_rate=feasibility_rate,
        parameter_samples=None,
    )
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `conda run -n shiny python -m pytest packages/pypath/tests/test_montecarlo.py -v --tb=short`
Expected: All PASSED

- [ ] **Step 5: Commit**

```bash
git add packages/pypath/src/pypath/core/montecarlo.py packages/pypath/tests/test_montecarlo.py
git commit -m "feat(montecarlo): implement run_montecarlo() with Ecopath + Ecosim ensemble"
```

---

## Chunk 3: Sensitivity Analysis

### Task 6: Morris Screening Implementation

**Files:**
- Create: `packages/pypath/src/pypath/core/sensitivity.py`
- Create: `packages/pypath/tests/test_sensitivity.py`

- [ ] **Step 1: Write failing tests for Morris screening**

Create `packages/pypath/tests/test_sensitivity.py`:

```python
"""Tests for pypath.core.sensitivity module."""
import numpy as np
import pytest

from pypath.core.sensitivity import (
    MorrisResult,
    SensitivityConfig,
    _generate_morris_trajectories,
    _compute_elementary_effects,
)


class TestMorrisDesign:
    def test_trajectory_shape(self):
        """Morris trajectories have correct shape: (n_traj * (k+1), k)."""
        k = 3
        n_traj = 5
        traj = _generate_morris_trajectories(k, n_traj, n_levels=4,
                                              rng=np.random.default_rng(42))
        assert traj.shape == (n_traj * (k + 1), k)

    def test_values_in_unit_cube(self):
        traj = _generate_morris_trajectories(4, 3, n_levels=4,
                                              rng=np.random.default_rng(42))
        assert np.all(traj >= 0.0)
        assert np.all(traj <= 1.0)

    def test_one_param_changes_per_step(self):
        """Within each trajectory, exactly one parameter changes per step."""
        k = 3
        n_traj = 2
        traj = _generate_morris_trajectories(k, n_traj, n_levels=4,
                                              rng=np.random.default_rng(42))
        for t in range(n_traj):
            start = t * (k + 1)
            for step in range(k):
                diff = traj[start + step + 1] - traj[start + step]
                n_changed = np.sum(np.abs(diff) > 1e-10)
                assert n_changed == 1


class TestElementaryEffects:
    def test_known_linear_function(self):
        """For y = 2*x0 + 3*x1, EE should be [2, 3]."""
        k = 2
        n_traj = 10
        traj = _generate_morris_trajectories(k, n_traj, n_levels=4,
                                              rng=np.random.default_rng(42))
        # Evaluate y = 2*x0 + 3*x1
        y = 2.0 * traj[:, 0] + 3.0 * traj[:, 1]
        result = _compute_elementary_effects(traj, y, k, n_traj, n_levels=4)
        assert result.mu_star[0] == pytest.approx(2.0, rel=0.2)
        assert result.mu_star[1] == pytest.approx(3.0, rel=0.2)

    def test_result_structure(self):
        k = 2
        n_traj = 5
        traj = _generate_morris_trajectories(k, n_traj, n_levels=4,
                                              rng=np.random.default_rng(42))
        y = traj[:, 0] + traj[:, 1]
        result = _compute_elementary_effects(traj, y, k, n_traj, n_levels=4)
        assert isinstance(result, MorrisResult)
        assert len(result.mu_star) == k
        assert len(result.sigma) == k
        assert len(result.mu) == k


class TestSensitivityConfig:
    def test_defaults(self):
        config = SensitivityConfig()
        assert config.method == "morris"
        assert config.n_trajectories == 10
        assert config.n_levels == 4

    def test_sobol_missing(self):
        """Sobol without SALib raises ImportError."""
        from pypath.core.sensitivity import HAS_SALIB
        if not HAS_SALIB:
            from pypath.core.sensitivity import run_sensitivity
            from pypath.core.params import create_rpath_params
            params = create_rpath_params(
                groups=["A", "B", "Det"], types=[1, 0, 2],
            )
            params.model.loc[0, "Biomass"] = 10.0
            params.model.loc[1, "Biomass"] = 5.0
            params.model.loc[2, "Biomass"] = 100.0
            params.pedigree["Biomass"] = [0.2, 0.2, 0.0]
            config = SensitivityConfig(method="sobol")
            with pytest.raises(ImportError, match="SALib"):
                run_sensitivity(params, config)
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `conda run -n shiny python -m pytest packages/pypath/tests/test_sensitivity.py -v`
Expected: FAIL with ImportError

- [ ] **Step 3: Implement Morris screening and sensitivity runner**

Create `packages/pypath/src/pypath/core/sensitivity.py`:

```python
"""Sensitivity analysis for Ecopath/Ecosim models.

Morris elementary effects screening and optional Sobol variance-based
sensitivity analysis (requires SALib).
"""
from __future__ import annotations

import logging
import math
import warnings
from dataclasses import dataclass
from typing import Callable

import numpy as np

logger = logging.getLogger(__name__)

try:
    import SALib
    from SALib.analyze import sobol as salib_sobol
    from SALib.sample import saltelli

    HAS_SALIB = True
except ImportError:
    HAS_SALIB = False


@dataclass
class MorrisResult:
    """Morris elementary effects screening results."""

    parameter_names: list[str]
    mu_star: np.ndarray
    sigma: np.ndarray
    mu: np.ndarray
    output_name: str = "Biomass"


@dataclass
class SobolResult:
    """Sobol variance-based sensitivity indices."""

    parameter_names: list[str]
    S1: np.ndarray
    ST: np.ndarray
    S1_conf: np.ndarray
    ST_conf: np.ndarray
    output_name: str = "Biomass"


@dataclass
class SensitivityConfig:
    """Sensitivity analysis configuration."""

    method: str = "morris"
    n_trajectories: int = 10
    n_levels: int = 4
    n_samples: int = 1024
    seed: int | None = None
    n_jobs: int = 1
    output_variable: str = "Biomass"
    output_group_idx: int | None = None
    ecopath_only: bool = False
    ecosim_years: range | None = None


def _generate_morris_trajectories(
    k: int, n_trajectories: int, n_levels: int = 4,
    rng: np.random.Generator | None = None,
) -> np.ndarray:
    """Generate Morris OAT trajectories in the unit hypercube.

    Returns array of shape (n_trajectories * (k+1), k).
    """
    if rng is None:
        rng = np.random.default_rng()

    delta = n_levels / (2 * (n_levels - 1))
    grid_values = np.linspace(0, 1, n_levels)
    trajectories = []

    # Restrict base points to grid values that allow perturbation
    valid_grid = grid_values[(grid_values >= delta) & (grid_values <= 1.0 - delta)]
    if len(valid_grid) == 0:
        valid_grid = grid_values  # fallback for extreme n_levels

    for _ in range(n_trajectories):
        # Random base point from valid grid values only
        base = rng.choice(valid_grid, size=k)

        trajectory = [base.copy()]
        order = rng.permutation(k)
        current = base.copy()

        for param_idx in order:
            current = current.copy()
            sign = rng.choice([-1, 1])
            current[param_idx] = np.clip(current[param_idx] + sign * delta, 0.0, 1.0)
            trajectory.append(current.copy())

        trajectories.extend(trajectory)

    return np.array(trajectories)


def _compute_elementary_effects(
    trajectories: np.ndarray,
    y_values: np.ndarray,
    k: int,
    n_trajectories: int,
    n_levels: int = 4,
) -> MorrisResult:
    """Compute Morris elementary effects from trajectories and outputs."""
    delta = n_levels / (2 * (n_levels - 1))
    elementary_effects = [[] for _ in range(k)]

    for t in range(n_trajectories):
        start = t * (k + 1)
        for step in range(k):
            diff = trajectories[start + step + 1] - trajectories[start + step]
            changed_idx = np.argmax(np.abs(diff))
            ee = (y_values[start + step + 1] - y_values[start + step]) / delta
            elementary_effects[changed_idx].append(ee)

    mu_star = np.array([np.mean(np.abs(ee)) if ee else 0.0 for ee in elementary_effects])
    sigma = np.array([np.std(ee) if len(ee) > 1 else 0.0 for ee in elementary_effects])
    mu = np.array([np.mean(ee) if ee else 0.0 for ee in elementary_effects])

    return MorrisResult(
        parameter_names=[f"param_{i}" for i in range(k)],
        mu_star=mu_star,
        sigma=sigma,
        mu=mu,
    )


def run_sensitivity(
    params: "RpathParams",
    config: SensitivityConfig | None = None,
    *,
    pedigree_config: "PedigreeConfig | None" = None,
    progress_callback: Callable[[int, int], None] | None = None,
) -> MorrisResult | SobolResult:
    """Run sensitivity analysis on Ecopath/Ecosim model.

    Parameters
    ----------
    params : RpathParams
        Base Ecopath parameters.
    config : SensitivityConfig, optional
        Sensitivity configuration.
    pedigree_config : PedigreeConfig, optional
        EwE pedigree mapping.
    progress_callback : callable, optional
        Called with (current, total) after each model evaluation.

    Returns
    -------
    MorrisResult or SobolResult
    """
    from pypath.core.pedigree import (
        ScalarDistribution,
        apply_sample,
        build_distributions,
    )
    from pypath.core.ecopath import rpath as run_rpath

    if config is None:
        config = SensitivityConfig()

    if config.method == "sobol" and not HAS_SALIB:
        raise ImportError(
            "Install SALib for Sobol analysis: pip install SALib"
        )

    rng = np.random.default_rng(config.seed)
    distributions = build_distributions(params, pedigree_config)
    scalars = [d for d in distributions if isinstance(d, ScalarDistribution)]

    if len(scalars) == 0:
        raise ValueError("No scalar distributions to analyze (all CVs are 0).")

    k = len(scalars)
    param_names = [f"{d.param_name}_{d.group_idx}" for d in scalars]

    def _evaluate(x_unit: np.ndarray) -> float:
        """Map unit hypercube point to params, run model, extract output."""
        sample = {}
        for j, dist in enumerate(scalars):
            sigma = math.sqrt(math.log(1 + dist.cv**2))
            mu = math.log(dist.base_value) - sigma**2 / 2
            from scipy.stats import lognorm
            val = float(lognorm.ppf(x_unit[j], s=sigma, scale=math.exp(mu)))
            sample[(dist.param_name, dist.group_idx)] = val

        sampled_params = apply_sample(params, sample)
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                rpath_result = run_rpath(sampled_params)
            if config.output_group_idx is not None:
                return float(rpath_result.Biomass[config.output_group_idx])
            # Use nansum to skip fleet groups (which have NaN biomass)
            n_bio = int((params.model["Type"] != 3).sum())
            return float(np.nansum(rpath_result.Biomass[:n_bio]))
        except Exception:
            return np.nan

    if config.method == "morris":
        trajectories = _generate_morris_trajectories(
            k, config.n_trajectories, config.n_levels, rng,
        )
        n_evals = len(trajectories)
        y_values = np.empty(n_evals)
        for i in range(n_evals):
            y_values[i] = _evaluate(trajectories[i])
            if progress_callback:
                progress_callback(i + 1, n_evals)

        result = _compute_elementary_effects(
            trajectories, y_values, k, config.n_trajectories, config.n_levels,
        )
        result.parameter_names = param_names
        result.output_name = config.output_variable
        return result

    elif config.method == "sobol":
        n_runs = config.n_samples * (2 * k + 2)
        if n_runs > 10000:
            warnings.warn(
                f"Sobol analysis requires {n_runs} model evaluations.",
                UserWarning,
            )

        problem = {
            "num_vars": k,
            "names": param_names,
            "bounds": [[0.0, 1.0]] * k,
        }
        X = saltelli.sample(problem, config.n_samples, seed=config.seed)
        Y = np.empty(len(X))
        for i in range(len(X)):
            Y[i] = _evaluate(X[i])
            if progress_callback:
                progress_callback(i + 1, len(X))

        # Impute NaN with mean (SALib requires exact N*(2k+2) rows)
        nan_mask = np.isnan(Y)
        n_nan = np.sum(nan_mask)
        if n_nan > len(Y) * 0.5:
            raise RuntimeError("More than 50% of model evaluations failed.")
        if n_nan > 0:
            Y[nan_mask] = np.nanmean(Y)
            logger.warning("Imputed %d NaN evaluations with mean for Sobol.", n_nan)

        Si = salib_sobol.analyze(problem, Y)
        return SobolResult(
            parameter_names=param_names,
            S1=Si["S1"],
            ST=Si["ST"],
            S1_conf=Si["S1_conf"],
            ST_conf=Si["ST_conf"],
            output_name=config.output_variable,
        )

    raise ValueError(f"Unknown method: {config.method}")
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `conda run -n shiny python -m pytest packages/pypath/tests/test_sensitivity.py -v`
Expected: All PASSED

- [ ] **Step 5: Commit**

```bash
git add packages/pypath/src/pypath/core/sensitivity.py packages/pypath/tests/test_sensitivity.py
git commit -m "feat(sensitivity): add Morris screening and optional Sobol analysis"
```

---

## Chunk 4: I/O Layer & Integration

### Task 7: Pedigree Schema Tables

**Files:**
- Modify: `packages/pypath/src/pypath/io/_ewe_schema.py`

- [ ] **Step 1: Read existing schema**

Read `packages/pypath/src/pypath/io/_ewe_schema.py` to find where to add new table definitions.

- [ ] **Step 2: Add 6 pedigree/sample table definitions**

Add to the `EWE_TABLES` dict in `_ewe_schema.py`. **Important:** Use `OrderedDict` to match the existing schema convention:

```python
    # Pedigree
    "Pedigree": OrderedDict([
        ("LevelID", "INTEGER"),
        ("LevelName", "TEXT"),
        ("VarName", "TEXT"),
        ("Sequence", "INTEGER"),
        ("IndexValue", "DOUBLE"),
        ("Confidence", "DOUBLE"),
        ("LevelColor", "INTEGER"),
        ("Description", "TEXT"),
    ]),
    "EcopathGroupPedigree": OrderedDict([
        ("GroupID", "INTEGER"),
        ("VarName", "TEXT"),
        ("LevelID", "INTEGER"),
    ]),
    # Monte Carlo samples
    "EcopathSample": OrderedDict([
        ("SampleID", "INTEGER"),
        ("Hash", "TEXT"),
        ("Source", "TEXT"),
        ("Generated", "TEXT"),
        ("Rating", "DOUBLE"),
        ("SS", "DOUBLE"),
    ]),
    "EcopathGroupSample": OrderedDict([
        ("SampleID", "INTEGER"),
        ("GroupID", "INTEGER"),
        ("Biomass", "DOUBLE"),
        ("ProdBiom", "DOUBLE"),
        ("ConsBiom", "DOUBLE"),
        ("EcoEfficiency", "DOUBLE"),
        ("BiomAcc", "DOUBLE"),
        ("ImpVar", "DOUBLE"),
        ("BiomAccRate", "DOUBLE"),
    ]),
    "EcopathDietCompSample": OrderedDict([
        ("SampleID", "INTEGER"),
        ("PredID", "INTEGER"),
        ("PreyID", "INTEGER"),
        ("Diet", "DOUBLE"),
    ]),
    "EcopathGroupCatchSample": OrderedDict([
        ("SampleID", "INTEGER"),
        ("GroupID", "INTEGER"),
        ("FleetID", "INTEGER"),
        ("Landing", "DOUBLE"),
        ("Discards", "DOUBLE"),
    ]),
```

- [ ] **Step 3: Commit**

```bash
git add packages/pypath/src/pypath/io/_ewe_schema.py
git commit -m "feat(io): add pedigree and MC sample table definitions to EwE schema"
```

---

### Task 8: read_pedigree() Function

**Files:**
- Modify: `packages/pypath/src/pypath/io/ewemdb.py`

- [ ] **Step 1: Read ewemdb.py to understand patterns**

Read `packages/pypath/src/pypath/io/ewemdb.py` to find `read_mediation()` (the latest addition) and follow the same pattern for `read_pedigree()`.

- [ ] **Step 2: Implement read_pedigree()**

Add to `ewemdb.py` after `read_mediation()`:

```python
def read_pedigree(db_path: str) -> tuple:
    """Read pedigree tables from an EwE database.

    Parameters
    ----------
    db_path : str
        Path to the .eweaccdb database file.

    Returns
    -------
    tuple[PedigreeConfig, pd.DataFrame]
        (config, group_pedigree) where:
        - config: PedigreeConfig with level_to_cv mapping
        - group_pedigree: DataFrame with columns [GroupID, VarName, CV]
    """
    from pypath.core.pedigree import PedigreeConfig

    try:
        tables = list_ewemdb_tables(db_path)
    except Exception:
        return PedigreeConfig(), pd.DataFrame(columns=["GroupID", "VarName", "CV"])

    config = PedigreeConfig()

    # Read Pedigree level definitions
    if "Pedigree" in tables:
        try:
            ped_df = read_ewemdb_table(db_path, "Pedigree")
            for _, row in ped_df.iterrows():
                var_name = str(row.get("VarName", ""))
                level_id = int(row.get("LevelID", 0))
                index_val = float(row.get("IndexValue", 0.0))
                if var_name not in config.level_to_cv:
                    config.level_to_cv[var_name] = {}
                config.level_to_cv[var_name][level_id] = index_val
        except Exception:
            pass

    # Read per-group pedigree assignments
    group_records = []
    if "EcopathGroupPedigree" in tables:
        try:
            gp_df = read_ewemdb_table(db_path, "EcopathGroupPedigree")
            for _, row in gp_df.iterrows():
                group_id = int(row.get("GroupID", 0))
                var_name = str(row.get("VarName", ""))
                level_id = int(row.get("LevelID", 0))
                # Look up CV from pedigree levels
                cv = config.level_to_cv.get(var_name, {}).get(level_id, 0.0)
                group_records.append({
                    "GroupID": group_id,
                    "VarName": var_name,
                    "CV": cv,
                })
        except Exception:
            pass

    group_pedigree = pd.DataFrame(
        group_records if group_records else [],
        columns=["GroupID", "VarName", "CV"],
    )

    return config, group_pedigree
```

- [ ] **Step 3: Verify import works**

Run: `conda run -n shiny python -c "from pypath.io.ewemdb import read_pedigree; print('OK')"`

- [ ] **Step 4: Commit**

```bash
git add packages/pypath/src/pypath/io/ewemdb.py
git commit -m "feat(io): add read_pedigree() for EwE database pedigree tables"
```

---

### Task 9: I/O Tests

**Files:**
- Create: `packages/pypath/tests/test_pedigree_io.py`

- [ ] **Step 1: Write I/O and schema tests**

Create `packages/pypath/tests/test_pedigree_io.py`:

```python
"""I/O tests for pedigree functions."""
import numpy as np
import pandas as pd
import pytest
from unittest.mock import patch

from pypath.core.pedigree import PedigreeConfig


class TestPedigreeSchema:
    def test_pedigree_table_exists(self):
        from pypath.io._ewe_schema import EWE_TABLES
        assert "Pedigree" in EWE_TABLES

    def test_pedigree_table_columns(self):
        from pypath.io._ewe_schema import EWE_TABLES
        tbl = EWE_TABLES["Pedigree"]
        assert tbl["LevelID"] == "INTEGER"
        assert tbl["VarName"] == "TEXT"
        assert tbl["IndexValue"] == "DOUBLE"
        assert tbl["Confidence"] == "DOUBLE"

    def test_group_pedigree_table_exists(self):
        from pypath.io._ewe_schema import EWE_TABLES
        assert "EcopathGroupPedigree" in EWE_TABLES
        tbl = EWE_TABLES["EcopathGroupPedigree"]
        assert "GroupID" in tbl
        assert "VarName" in tbl
        assert "LevelID" in tbl

    def test_sample_tables_exist(self):
        from pypath.io._ewe_schema import EWE_TABLES
        for name in ["EcopathSample", "EcopathGroupSample",
                      "EcopathDietCompSample", "EcopathGroupCatchSample"]:
            assert name in EWE_TABLES


class TestReadPedigree:
    def test_reads_pedigree_levels(self):
        from pypath.io.ewemdb import read_pedigree

        ped_df = pd.DataFrame([
            {"LevelID": 6, "LevelName": "Guesstimate", "VarName": "PBInput",
             "Sequence": 1, "IndexValue": 0.1, "Confidence": 70.0,
             "LevelColor": 0, "Description": ""},
            {"LevelID": 7, "LevelName": "Other model", "VarName": "PBInput",
             "Sequence": 2, "IndexValue": 0.2, "Confidence": 60.0,
             "LevelColor": 0, "Description": ""},
        ])
        gp_df = pd.DataFrame([
            {"GroupID": 1, "VarName": "PBInput", "LevelID": 6},
        ])

        table_map = {
            "Pedigree": ped_df,
            "EcopathGroupPedigree": gp_df,
        }
        with patch("pypath.io.ewemdb.list_ewemdb_tables",
                    return_value=list(table_map.keys())):
            with patch("pypath.io.ewemdb.read_ewemdb_table",
                       side_effect=lambda path, tbl: table_map[tbl]):
                config, group_ped = read_pedigree("fake.eweaccdb")

        assert config.level_to_cv["PBInput"][6] == 0.1
        assert config.level_to_cv["PBInput"][7] == 0.2
        assert len(group_ped) == 1
        assert group_ped.iloc[0]["CV"] == pytest.approx(0.1)

    def test_missing_tables_returns_empty(self):
        from pypath.io.ewemdb import read_pedigree

        with patch("pypath.io.ewemdb.list_ewemdb_tables",
                    return_value=["SomeOtherTable"]):
            config, group_ped = read_pedigree("fake.eweaccdb")

        assert config.level_to_cv == {}
        assert len(group_ped) == 0

    def test_db_exception_returns_empty(self):
        from pypath.io.ewemdb import read_pedigree

        with patch("pypath.io.ewemdb.list_ewemdb_tables",
                    side_effect=Exception("No driver")):
            config, group_ped = read_pedigree("fake.eweaccdb")

        assert len(group_ped) == 0
```

- [ ] **Step 2: Run tests**

Run: `conda run -n shiny python -m pytest packages/pypath/tests/test_pedigree_io.py -v`
Expected: All PASSED

- [ ] **Step 3: Commit**

```bash
git add packages/pypath/tests/test_pedigree_io.py
git commit -m "test(io): add pedigree I/O and schema tests"
```

---

### Task 10: Integration Tests

**Files:**
- Create: `packages/pypath/tests/test_mc_integration.py`

- [ ] **Step 1: Write integration tests**

Create `packages/pypath/tests/test_mc_integration.py`:

```python
"""Integration tests for Monte Carlo and sensitivity analysis."""
import numpy as np
import pytest
import warnings

from pypath.core.ecopath import rpath
from pypath.core.montecarlo import MCConfig, run_montecarlo
from pypath.core.params import create_rpath_params


def _make_mc_model():
    """Create a balanced 3-group model with moderate pedigree CVs."""
    params = create_rpath_params(
        groups=["Producer", "Consumer", "Detritus"],
        types=[1, 0, 2],
    )
    params.model.loc[0, "Biomass"] = 10.0
    params.model.loc[0, "PB"] = 200.0
    params.model.loc[0, "EE"] = 0.8
    params.model.loc[1, "Biomass"] = 5.0
    params.model.loc[1, "PB"] = 50.0
    params.model.loc[1, "QB"] = 150.0
    params.model.loc[1, "EE"] = 0.9
    params.model.loc[2, "Biomass"] = 100.0
    params.model["BioAcc"] = 0.0
    params.model["Unassim"] = 0.2
    params.model.loc[0, "Unassim"] = 0.0
    params.model.loc[2, "Unassim"] = 0.0
    params.model["Detritus"] = 1.0
    params.model.loc[2, "Detritus"] = 0.0
    params.diet["Consumer"] = [1.0, 0.0, 0.0, 0.0]
    # Moderate pedigree
    params.pedigree["Biomass"] = [0.15, 0.15, 0.0]
    params.pedigree["PB"] = [0.1, 0.1, 0.0]
    params.pedigree["QB"] = [0.0, 0.1, 0.0]
    params.pedigree["Diet"] = [0.0, 0.0, 0.0]
    return params


@pytest.mark.slow
class TestMCIntegration:
    def test_ecopath_mc_feasibility(self):
        """Full pipeline: pedigree -> MC(n=50, ecopath_only) -> feasibility > 0."""
        params = _make_mc_model()
        config = MCConfig(n_samples=50, method="random", seed=42, ecopath_only=True)
        result = run_montecarlo(params, config)
        assert result.feasibility_rate > 0
        assert "Biomass" in result.ecopath_stats
        assert result.ecopath_stats["Biomass"].shape[0] > 0

    def test_ecopath_mc_with_ecosim(self):
        """Full pipeline: pedigree -> MC(n=10, ecosim) -> ecosim_stats shape."""
        params = _make_mc_model()
        config = MCConfig(
            n_samples=10, method="random", seed=42,
            ecopath_only=False, ecosim_years=range(1, 6),
        )
        result = run_montecarlo(params, config)
        if result.n_ecosim > 0:
            assert result.ecosim_stats is not None
            assert result.ecosim_stats["Biomass"].shape[2] == 7  # 7 stats

    def test_zero_cv_identical(self):
        """Zero-CV pedigree -> all samples identical."""
        params = _make_mc_model()
        params.pedigree["Biomass"] = [0.0, 0.0, 0.0]
        params.pedigree["PB"] = [0.0, 0.0, 0.0]
        params.pedigree["QB"] = [0.0, 0.0, 0.0]
        config = MCConfig(n_samples=5, method="random", seed=42, ecopath_only=True)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            result = run_montecarlo(params, config)
        if result.n_feasible > 1:
            assert result.ecopath_stats["Biomass"]["std"].max() < 1e-10

    def test_store_runs_accessible(self):
        """store_runs=True -> raw outputs accessible."""
        params = _make_mc_model()
        config = MCConfig(
            n_samples=10, method="random", seed=42,
            ecopath_only=True, store_runs=True,
        )
        result = run_montecarlo(params, config)
        assert result.ecopath_runs is not None
        assert len(result.ecopath_runs) == result.n_feasible

    def test_morris_screening(self):
        """Morris on 3-group model -> all params ranked."""
        from pypath.core.sensitivity import SensitivityConfig, run_sensitivity
        params = _make_mc_model()
        config = SensitivityConfig(
            method="morris", n_trajectories=5, seed=42,
            ecopath_only=True,
        )
        result = run_sensitivity(params, config)
        assert len(result.mu_star) > 0
        assert len(result.parameter_names) == len(result.mu_star)
```

- [ ] **Step 2: Run tests**

Run: `conda run -n shiny python -m pytest packages/pypath/tests/test_mc_integration.py -v --tb=short`
Expected: All PASSED (may be slow)

- [ ] **Step 3: Commit**

```bash
git add packages/pypath/tests/test_mc_integration.py
git commit -m "test(montecarlo): add integration tests for MC and Morris screening"
```

---

### Task 11: Package Exports

**Files:**
- Modify: `packages/pypath/src/pypath/core/__init__.py`
- Modify: `packages/pypath/src/pypath/io/__init__.py`

- [ ] **Step 1: Add pedigree/MC/sensitivity exports to core/__init__.py**

Read `packages/pypath/src/pypath/core/__init__.py` and add after the mediation imports. **Important:** Follow the existing try/except pattern used for optimization imports:

```python
try:
    from pypath.core.pedigree import (
        DietDistribution,
        PedigreeConfig,
        ScalarDistribution,
        apply_sample,
        build_distributions,
        sample_parameters,
    )
    from pypath.core.montecarlo import (
        HAS_JOBLIB,
        MCConfig,
        MCResult,
        run_montecarlo,
    )
    from pypath.core.sensitivity import (
        HAS_SALIB,
        MorrisResult,
        SensitivityConfig,
        SobolResult,
        run_sensitivity,
    )

    HAS_MONTECARLO = True
except ImportError:
    HAS_MONTECARLO = False
```

Add to `__all__` (only if `HAS_MONTECARLO` is True — follow the pattern for optimization exports):

```python
    # Pedigree & Monte Carlo
    "HAS_MONTECARLO",
    "PedigreeConfig",
    "ScalarDistribution",
    "DietDistribution",
    "build_distributions",
    "sample_parameters",
    "apply_sample",
    "MCConfig",
    "MCResult",
    "run_montecarlo",
    "HAS_JOBLIB",
    # Sensitivity
    "MorrisResult",
    "SobolResult",
    "SensitivityConfig",
    "run_sensitivity",
    "HAS_SALIB",
```

- [ ] **Step 2: Add read_pedigree to io/__init__.py**

Read `packages/pypath/src/pypath/io/__init__.py` and add `read_pedigree` to the ewemdb import and `__all__`.

- [ ] **Step 3: Verify imports**

Run: `conda run -n shiny python -c "from pypath.core import MCConfig, run_montecarlo, MorrisResult, run_sensitivity, PedigreeConfig; print('core OK')" && conda run -n shiny python -c "from pypath.io import read_pedigree; print('io OK')"`

- [ ] **Step 4: Commit**

```bash
git add packages/pypath/src/pypath/core/__init__.py packages/pypath/src/pypath/io/__init__.py
git commit -m "feat(api): export MC, pedigree, and sensitivity from package"
```

---

### Task 12: Run Full Test Suite

- [ ] **Step 1: Run all new tests**

Run: `conda run -n shiny python -m pytest packages/pypath/tests/test_pedigree.py packages/pypath/tests/test_montecarlo.py packages/pypath/tests/test_sensitivity.py packages/pypath/tests/test_pedigree_io.py packages/pypath/tests/test_mc_integration.py -v --tb=short`

Expected: All PASSED

- [ ] **Step 2: Run existing ecosim tests for regression**

Run: `conda run -n shiny python -m pytest packages/pypath/tests/test_ecosim.py -v --tb=short`

Expected: 35 PASSED (no regression)
