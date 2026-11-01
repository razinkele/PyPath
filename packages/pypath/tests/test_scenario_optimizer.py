"""Tests for Phase 1C.3 scenario optimization utilities."""

from __future__ import annotations

import numpy as np

from pypath.analysis.scenario_optimizer import (
    ObjectiveSpec,
    ScenarioCandidate,
    ScenarioOptimizer,
    make_spatial_management_evaluator,
)


def test_scenario_optimizer_rank_respects_goal_direction():
    objectives = [
        ObjectiveSpec("benefit", weight=1.0, goal="maximize"),
        ObjectiveSpec("cost", weight=0.5, goal="minimize"),
    ]

    values = {
        "A": {"benefit": 10.0, "cost": 5.0, "collapse_risk": 0.2},
        "B": {"benefit": 9.0, "cost": 1.0, "collapse_risk": 0.1},
    }

    optimizer = ScenarioOptimizer(objectives, evaluator=lambda c: values[c.name])
    ranked = optimizer.rank(
        [
            ScenarioCandidate(name="A", controls={"effort_scale": 1.0}),
            ScenarioCandidate(name="B", controls={"effort_scale": 0.8}),
        ]
    )

    assert ranked[0].name == "B"
    assert ranked[0].aggregate_score > ranked[1].aggregate_score


def test_scenario_optimizer_infeasible_penalty_applied():
    objectives = [
        ObjectiveSpec("benefit", weight=1.0, goal="maximize", min_value=0.0),
        ObjectiveSpec("risk", weight=1.0, goal="minimize", max_value=0.3),
    ]

    values = {
        "safe": {"benefit": 2.0, "risk": 0.2, "collapse_risk": 0.2},
        "unsafe": {"benefit": 100.0, "risk": 0.9, "collapse_risk": 0.9},
    }

    optimizer = ScenarioOptimizer(objectives, evaluator=lambda c: values[c.name])
    ranked = optimizer.rank(
        [
            ScenarioCandidate(name="unsafe", controls={}),
            ScenarioCandidate(name="safe", controls={}),
        ]
    )

    assert ranked[0].name == "safe"
    assert ranked[0].feasible is True
    assert ranked[1].feasible is False


def test_optimize_grid_enumerates_all_combinations():
    objectives = [ObjectiveSpec("score", weight=1.0, goal="maximize")]

    optimizer = ScenarioOptimizer(
        objectives,
        evaluator=lambda c: {
            "score": c.controls["x"] - c.controls["y"],
            "collapse_risk": 0.0,
        },
    )

    ranked = optimizer.optimize_grid(
        {"x": [1.0, 2.0], "y": [0.0, 1.0]}, name_prefix="grid"
    )
    assert len(ranked) == 4
    assert ranked[0].controls == {"x": 2.0, "y": 0.0}


def test_spatial_management_evaluator_runs(spatial_scenario, single_patch_ecospace):
    scenario, _ = spatial_scenario
    evaluator = make_spatial_management_evaluator(
        scenario,
        single_patch_ecospace,
        years=range(1, 3),
    )

    metrics = evaluator(
        ScenarioCandidate(name="baseline", controls={"effort_scale": 1.0})
    )

    assert set(metrics.keys()) == {
        "final_biomass",
        "mean_biomass",
        "catch_total",
        "collapse_risk",
    }
    assert np.isfinite(metrics["final_biomass"])
    assert 0.0 <= metrics["collapse_risk"] <= 1.0
