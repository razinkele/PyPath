"""Tests for advanced Ecosim control adapters."""

import warnings

import numpy as np
import pytest
from pypath.core.ecopath import rpath
from pypath.core.ecosim import rsim_scenario
from pypath.core.params import create_rpath_params

from pypath_shiny.services.ecosim_controls import (
    RecruitmentControlConfig,
    apply_fishing_management_control,
    apply_metabolism_control,
    apply_recruitment_control,
    build_control_metadata,
    validate_fishing_management_control,
    validate_metabolism_control,
    validate_recruitment_control,
)


@pytest.fixture
def simple_scenario():
    """Create a compact Ecosim scenario for adapter tests."""
    params = create_rpath_params(
        groups=["Phyto", "Zoo", "Fish", "Det", "Fleet"],
        types=[1, 0, 0, 2, 3],
    )

    params.model.loc[0, "Biomass"] = 10.0
    params.model.loc[0, "PB"] = 80.0
    params.model.loc[0, "EE"] = 0.8
    params.model.loc[1, "Biomass"] = 5.0
    params.model.loc[1, "PB"] = 20.0
    params.model.loc[1, "QB"] = 70.0
    params.model.loc[1, "EE"] = 0.8
    params.model.loc[2, "Biomass"] = 2.0
    params.model.loc[2, "PB"] = 1.2
    params.model.loc[2, "QB"] = 5.0
    params.model.loc[2, "EE"] = 0.5
    params.model.loc[3, "Biomass"] = 50.0

    params.model["BioAcc"] = 0.0
    params.model["Unassim"] = 0.2
    params.model.loc[0, "Unassim"] = 0.0
    params.model.loc[3, "Unassim"] = 0.0
    params.model.loc[4, "BioAcc"] = np.nan
    params.model.loc[4, "Unassim"] = np.nan
    params.model["Det"] = 1.0
    params.model.loc[4, "Det"] = np.nan

    params.diet["Zoo"] = [1.0, 0.0, 0.0, 0.0, 0.0]
    params.diet["Fish"] = [0.0, 1.0, 0.0, 0.0, 0.0]
    params.diet["Phyto"] = [0.0, 0.0, 0.0, 0.0, 0.0]
    params.model.loc[2, "Fleet"] = 0.5

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        model = rpath(params)

    return rsim_scenario(model, params, years=range(1, 6))


def test_recruitment_control_applies_multiplier(simple_scenario):
    cfg = validate_recruitment_control(
        enabled=True,
        group_name="Fish",
        model_type="ricker",
        alpha=1.3,
        beta=0.1,
        rmax=1.4,
        k=1.0,
        power=1.0,
        environment_multiplier=1.1,
    )

    summary = apply_recruitment_control(simple_scenario, cfg)

    assert summary["enabled"] is True
    fish_idx = list(simple_scenario.params.spname).index("Fish")
    forced = simple_scenario.forcing.ForcedRecs[:, fish_idx]
    assert np.all(forced > 0)
    assert summary["max_multiplier"] >= summary["min_multiplier"]


def test_recruitment_validation_requires_group():
    with pytest.raises(ValueError):
        validate_recruitment_control(
            enabled=True,
            group_name="",
            model_type="ricker",
            alpha=1.2,
            beta=0.1,
            rmax=1.3,
            k=1.0,
            power=1.0,
            environment_multiplier=1.0,
        )


def test_metabolism_control_scales_pbopt(simple_scenario):
    before = simple_scenario.params.PBopt.copy()
    cfg = validate_metabolism_control(
        enabled=True,
        temperature_delta=3.0,
        q10=2.0,
        mass_exponent=0.75,
        reference_mass=1.0,
        activity_budget=1.2,
        apply_to_consumers=True,
        apply_to_producers=False,
    )

    summary = apply_metabolism_control(simple_scenario, cfg)

    assert summary["enabled"] is True
    assert summary["groups_scaled"] > 0
    assert np.any(simple_scenario.params.PBopt != before)


def test_fishing_management_changes_effort(simple_scenario):
    before = simple_scenario.fishing.ForcedEffort[:, 1:].copy()

    cfg = validate_fishing_management_control(
        enabled=True,
        strategy="sustainability",
        sustainability_reduction=0.7,
        sustainability_floor=0.05,
        effort_budget=1.0,
        bycatch_reduction=0.2,
    )
    summary = apply_fishing_management_control(simple_scenario, cfg)

    assert summary["enabled"] is True
    assert summary["applied"] is True
    assert np.any(simple_scenario.fishing.ForcedEffort[:, 1:] != before)


def test_build_control_metadata_shape():
    metadata = build_control_metadata(
        recruitment={"enabled": True, "model": "ricker"},
        metabolism={"enabled": False},
        fishing_management={"enabled": True, "strategy": "revenue"},
    )

    assert set(metadata.keys()) == {"recruitment", "metabolism", "fishing_management"}
    assert metadata["recruitment"]["model"] == "ricker"


def test_disabled_recruitment_no_change(simple_scenario):
    before = simple_scenario.forcing.ForcedRecs.copy()
    summary = apply_recruitment_control(
        simple_scenario,
        RecruitmentControlConfig(enabled=False),
    )
    assert summary == {"enabled": False}
    assert np.array_equal(before, simple_scenario.forcing.ForcedRecs)
