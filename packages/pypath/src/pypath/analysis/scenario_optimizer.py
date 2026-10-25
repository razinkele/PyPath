"""Scenario optimization utilities for coupled Ecosim-Ecospace management runs."""

from __future__ import annotations

import copy
import itertools
from dataclasses import dataclass, field
from typing import Callable, Dict, Iterable, Optional

import numpy as np


@dataclass(frozen=True)
class ObjectiveSpec:
    """Definition of a scalar objective used in scenario ranking.

    Parameters
    ----------
    name : str
        Objective key expected from an evaluator.
    weight : float
        Objective weight in aggregate score.
    goal : str
        Either ``"maximize"`` or ``"minimize"``.
    min_value : float, optional
        Feasibility lower bound.
    max_value : float, optional
        Feasibility upper bound.
    """

    name: str
    weight: float = 1.0
    goal: str = "maximize"
    min_value: float | None = None
    max_value: float | None = None

    def __post_init__(self):
        if self.goal not in {"maximize", "minimize"}:
            raise ValueError(f"Unsupported goal '{self.goal}'")


@dataclass(frozen=True)
class ScenarioCandidate:
    """Candidate management scenario controls."""

    name: str
    controls: Dict[str, float]
    metadata: Dict[str, str] = field(default_factory=dict)


@dataclass
class ScenarioScore:
    """Evaluated scenario with objective values and aggregate score."""

    name: str
    controls: Dict[str, float]
    objectives: Dict[str, float]
    aggregate_score: float
    feasible: bool
    risk_score: float


class ScenarioOptimizer:
    """Multi-objective scenario optimizer with risk-aware ranking."""

    def __init__(
        self,
        objectives: Iterable[ObjectiveSpec],
        evaluator: Callable[[ScenarioCandidate], Dict[str, float]],
        *,
        risk_objective: str = "collapse_risk",
        infeasible_penalty: float = 1e9,
    ):
        self.objectives = list(objectives)
        self.evaluator = evaluator
        self.risk_objective = risk_objective
        self.infeasible_penalty = float(infeasible_penalty)
        if not self.objectives:
            raise ValueError("At least one objective is required")

    def evaluate(self, candidate: ScenarioCandidate) -> ScenarioScore:
        """Evaluate one candidate scenario."""
        values = self.evaluator(candidate)
        missing = [obj.name for obj in self.objectives if obj.name not in values]
        if missing:
            raise KeyError(f"Evaluator missing objective(s): {missing}")

        feasible = True
        aggregate = 0.0

        for obj in self.objectives:
            value = float(values[obj.name])
            if obj.min_value is not None and value < obj.min_value:
                feasible = False
            if obj.max_value is not None and value > obj.max_value:
                feasible = False

            direction = 1.0 if obj.goal == "maximize" else -1.0
            aggregate += obj.weight * direction * value

        if not feasible:
            aggregate -= self.infeasible_penalty

        risk = float(values.get(self.risk_objective, 0.0))
        return ScenarioScore(
            name=candidate.name,
            controls=dict(candidate.controls),
            objectives={k: float(v) for k, v in values.items()},
            aggregate_score=aggregate,
            feasible=feasible,
            risk_score=risk,
        )

    def rank(self, candidates: Iterable[ScenarioCandidate]) -> list[ScenarioScore]:
        """Evaluate and rank candidates by aggregate score, then risk."""
        scores = [self.evaluate(c) for c in candidates]
        return sorted(scores, key=lambda s: (s.aggregate_score, -s.risk_score), reverse=True)

    def optimize_grid(
        self,
        control_grid: Dict[str, Iterable[float]],
        *,
        name_prefix: str = "scenario",
    ) -> list[ScenarioScore]:
        """Enumerate a Cartesian grid of controls and rank all combinations."""
        if not control_grid:
            raise ValueError("control_grid must not be empty")

        keys = list(control_grid.keys())
        value_lists = [list(control_grid[k]) for k in keys]
        for key, vals in zip(keys, value_lists):
            if not vals:
                raise ValueError(f"control grid for '{key}' is empty")

        candidates = []
        for idx, combo in enumerate(itertools.product(*value_lists), start=1):
            controls = {k: float(v) for k, v in zip(keys, combo)}
            candidates.append(ScenarioCandidate(name=f"{name_prefix}_{idx}", controls=controls))

        return self.rank(candidates)


def make_spatial_management_evaluator(
    base_scenario,
    ecospace,
    *,
    years: Optional[range] = None,
    collapse_threshold: float = 0.2,
    apply_controls: Optional[Callable[[object, dict[str, float]], None]] = None,
) -> Callable[[ScenarioCandidate], Dict[str, float]]:
    """Create an evaluator that runs coupled simulations for controls.

    The default control semantics are:
    - ``effort_scale``: multiply all fleet effort values.
    - ``vulnerability_scale``: scale non-zero vulnerability entries.
    """

    from pypath.spatial.integration import rsim_run_spatial

    def _default_apply_controls(scenario_copy, controls: dict[str, float]) -> None:
        if "effort_scale" in controls:
            scenario_copy.fishing.ForcedEffort *= controls["effort_scale"]
        if "vulnerability_scale" in controls:
            vv = scenario_copy.params.VV
            mask = vv > 0
            vv[mask] *= controls["vulnerability_scale"]

    _apply = apply_controls or _default_apply_controls

    def evaluator(candidate: ScenarioCandidate) -> Dict[str, float]:
        scenario_copy = copy.deepcopy(base_scenario)
        ecospace_copy = copy.deepcopy(ecospace)

        _apply(scenario_copy, candidate.controls)

        result = rsim_run_spatial(scenario_copy, ecospace=ecospace_copy, years=years)

        # Group 0 is "Outside".
        start = result.out_Biomass[0, 1:]
        final = result.out_Biomass[-1, 1:]

        safe_start = np.where(start > 0, start, np.nan)
        ratio = final / safe_start
        ratio = np.nan_to_num(ratio, nan=1.0, posinf=1.0, neginf=0.0)

        collapse_risk = float(np.mean(ratio < collapse_threshold))
        final_total = float(np.sum(final))
        mean_total = float(np.mean(result.out_Biomass[:, 1:].sum(axis=1)))
        catch_total = float(np.sum(result.annual_Catch[:, 1:]))

        return {
            "final_biomass": final_total,
            "mean_biomass": mean_total,
            "catch_total": catch_total,
            "collapse_risk": collapse_risk,
        }

    return evaluator
