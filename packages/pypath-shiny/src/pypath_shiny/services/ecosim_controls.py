"""Service adapters for advanced Ecosim control panels."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from pypath.core.ecosim import RsimScenario
from pypath.core.fishing_management import FishingOptimization
from pypath.core.metabolism import TemperatureDependence
from pypath.core.recruitment import (
    BevertonHoltRecruitment,
    MyersRecruitment,
    RickerRecruitment,
    validate_recruitment_params,
)


@dataclass(frozen=True)
class RecruitmentControlConfig:
    """Validated recruitment control settings."""

    enabled: bool
    group_name: str = ""
    model_type: str = "ricker"
    alpha: float = 1.0
    beta: float = 0.1
    rmax: float = 1.0
    k: float = 1.0
    power: float = 1.0
    environment_multiplier: float = 1.0


@dataclass(frozen=True)
class MetabolismControlConfig:
    """Validated metabolism control settings."""

    enabled: bool
    temperature_delta: float = 0.0
    q10: float = 2.0
    mass_exponent: float = 0.75
    reference_mass: float = 1.0
    activity_budget: float = 1.0
    apply_to_consumers: bool = True
    apply_to_producers: bool = True


@dataclass(frozen=True)
class FishingManagementControlConfig:
    """Validated adaptive fishing-management settings."""

    enabled: bool
    strategy: str = "sustainability"
    sustainability_reduction: float = 0.9
    sustainability_floor: float = 0.05
    effort_budget: float = 1.0
    bycatch_reduction: float = 0.2


def validate_recruitment_control(
    *,
    enabled: bool,
    group_name: str,
    model_type: str,
    alpha: float,
    beta: float,
    rmax: float,
    k: float,
    power: float,
    environment_multiplier: float,
) -> RecruitmentControlConfig:
    """Validate recruitment configuration from UI controls."""
    if not enabled:
        return RecruitmentControlConfig(enabled=False)

    if not group_name:
        raise ValueError("Recruitment group must be selected")
    if environment_multiplier <= 0:
        raise ValueError("Environment multiplier must be > 0")

    model = model_type.lower().replace("-", "").replace("_", "")
    if model == "ricker":
        validate_recruitment_params("ricker", alpha=alpha, beta=beta, rmax=rmax)
    elif model == "bevertonholt":
        validate_recruitment_params("bevertonholt", alpha=alpha, beta=beta, rmax=rmax)
    elif model == "myers":
        validate_recruitment_params("myers", rmax=rmax, k=k, power=power)
    else:
        raise ValueError("Recruitment model must be ricker, bevertonholt, or myers")

    return RecruitmentControlConfig(
        enabled=True,
        group_name=group_name,
        model_type=model,
        alpha=float(alpha),
        beta=float(beta),
        rmax=float(rmax),
        k=float(k),
        power=float(power),
        environment_multiplier=float(environment_multiplier),
    )


def validate_metabolism_control(
    *,
    enabled: bool,
    temperature_delta: float,
    q10: float,
    mass_exponent: float,
    reference_mass: float,
    activity_budget: float,
    apply_to_consumers: bool,
    apply_to_producers: bool,
) -> MetabolismControlConfig:
    """Validate metabolism scaling controls."""
    if not enabled:
        return MetabolismControlConfig(enabled=False)
    if q10 <= 0:
        raise ValueError("Q10 must be > 0")
    if reference_mass <= 0:
        raise ValueError("Reference mass must be > 0")
    if activity_budget <= 0:
        raise ValueError("Activity budget must be > 0")
    if not (apply_to_consumers or apply_to_producers):
        raise ValueError("Enable at least one target group for metabolism scaling")

    return MetabolismControlConfig(
        enabled=True,
        temperature_delta=float(temperature_delta),
        q10=float(q10),
        mass_exponent=float(mass_exponent),
        reference_mass=float(reference_mass),
        activity_budget=float(activity_budget),
        apply_to_consumers=bool(apply_to_consumers),
        apply_to_producers=bool(apply_to_producers),
    )


def validate_fishing_management_control(
    *,
    enabled: bool,
    strategy: str,
    sustainability_reduction: float,
    sustainability_floor: float,
    effort_budget: float,
    bycatch_reduction: float,
) -> FishingManagementControlConfig:
    """Validate adaptive fishing management controls."""
    if not enabled:
        return FishingManagementControlConfig(enabled=False)

    normalized = strategy.lower().strip()
    if normalized not in {"sustainability", "revenue", "bycatch"}:
        raise ValueError("Fishing strategy must be sustainability, revenue, or bycatch")
    if not (0 <= sustainability_floor <= 1):
        raise ValueError("Sustainability floor must be between 0 and 1")
    if not (0 < sustainability_reduction <= 1):
        raise ValueError("Sustainability reduction must be in (0, 1]")
    if effort_budget < 0:
        raise ValueError("Effort budget must be >= 0")
    if bycatch_reduction < 0:
        raise ValueError("Bycatch reduction must be >= 0")

    return FishingManagementControlConfig(
        enabled=True,
        strategy=normalized,
        sustainability_reduction=float(sustainability_reduction),
        sustainability_floor=float(sustainability_floor),
        effort_budget=float(effort_budget),
        bycatch_reduction=float(bycatch_reduction),
    )


def apply_recruitment_control(
    scenario: RsimScenario, cfg: RecruitmentControlConfig
) -> dict[str, object]:
    """Apply recruitment forcing multiplier to selected group."""
    if not cfg.enabled:
        return {"enabled": False}

    group_idx = _group_index(scenario, cfg.group_name)
    n_months = int(scenario.forcing.ForcedRecs.shape[0])
    times = np.arange(n_months, dtype=float) / 12.0
    baseline_ssb = max(float(scenario.params.B_BaseRef[group_idx]), 1e-6)
    ssb = np.full(n_months, baseline_ssb, dtype=float)

    if cfg.model_type == "ricker":
        model = RickerRecruitment(alpha=cfg.alpha, beta=cfg.beta, rmax=cfg.rmax)
    elif cfg.model_type == "bevertonholt":
        model = BevertonHoltRecruitment(alpha=cfg.alpha, beta=cfg.beta, rmax=cfg.rmax)
    else:
        model = MyersRecruitment(rmax=cfg.rmax, k=cfg.k, power=cfg.power)

    multipliers = model.get_recruitment_batch(ssb, times) * cfg.environment_multiplier
    scenario.forcing.ForcedRecs[:, group_idx] = multipliers

    return {
        "enabled": True,
        "group": cfg.group_name,
        "model": cfg.model_type,
        "min_multiplier": float(np.min(multipliers)),
        "max_multiplier": float(np.max(multipliers)),
    }


def apply_metabolism_control(
    scenario: RsimScenario, cfg: MetabolismControlConfig
) -> dict[str, object]:
    """Apply metabolic scaling to PBopt and FtimeQBOpt."""
    if not cfg.enabled:
        return {"enabled": False}

    temp_model = TemperatureDependence(q10_coefficient=cfg.q10)
    temp_factor = float(temp_model.compute_q10_factor(10.0 + cfg.temperature_delta))

    pb = np.asarray(scenario.params.PBopt, dtype=float)
    qb = np.asarray(scenario.params.FtimeQBOpt, dtype=float)
    base_ref = np.asarray(scenario.params.B_BaseRef, dtype=float)
    pp_type = np.asarray(scenario.params.PP_type, dtype=int)

    group_mask = np.zeros_like(pb, dtype=bool)
    group_mask[1:] = True
    if len(pp_type) == len(group_mask):
        if cfg.apply_to_consumers and not cfg.apply_to_producers:
            group_mask &= pp_type == 0
        elif cfg.apply_to_producers and not cfg.apply_to_consumers:
            group_mask &= pp_type == 1
        elif not cfg.apply_to_consumers and not cfg.apply_to_producers:
            group_mask[:] = False

    mass_ratio = np.where(base_ref > 0, base_ref / cfg.reference_mass, 1.0)
    mass_scale = np.where(mass_ratio > 0, mass_ratio**cfg.mass_exponent, 1.0)
    scale = np.clip(temp_factor * cfg.activity_budget * mass_scale, 0.01, 100.0)
    scale[~group_mask] = 1.0

    scenario.params.PBopt = np.maximum(pb * scale, 1e-12)
    scenario.params.FtimeQBOpt = np.maximum(qb * scale, 1e-12)

    applied_scale = scale[group_mask]
    median_scale = float(np.median(applied_scale)) if applied_scale.size else 1.0
    return {
        "enabled": True,
        "temperature_delta": cfg.temperature_delta,
        "q10": cfg.q10,
        "groups_scaled": int(np.count_nonzero(group_mask)),
        "median_scale": median_scale,
    }


def apply_fishing_management_control(
    scenario: RsimScenario, cfg: FishingManagementControlConfig
) -> dict[str, object]:
    """Apply fleet-level effort management strategy."""
    if not cfg.enabled:
        return {"enabled": False}

    effort = np.asarray(scenario.fishing.ForcedEffort[:, 1:], dtype=float)
    if effort.size == 0:
        return {"enabled": True, "strategy": cfg.strategy, "applied": False}

    baseline = effort[0].copy()
    if cfg.strategy == "sustainability":
        target = np.maximum(
            baseline * cfg.sustainability_reduction,
            cfg.sustainability_floor,
        )
    elif cfg.strategy == "revenue":
        target = _revenue_effort_target(scenario, baseline, cfg.effort_budget)
    else:
        target = _bycatch_effort_target(scenario, baseline, cfg.bycatch_reduction)

    scenario.fishing.ForcedEffort[:, 1:] = target
    return {
        "enabled": True,
        "strategy": cfg.strategy,
        "applied": True,
        "total_effort_before": float(np.sum(baseline)),
        "total_effort_after": float(np.sum(target)),
    }


def build_control_metadata(
    *,
    recruitment: dict[str, object],
    metabolism: dict[str, object],
    fishing_management: dict[str, object],
) -> dict[str, dict[str, object]]:
    """Build standardized metadata payload for scenario/result traceability."""
    return {
        "recruitment": recruitment,
        "metabolism": metabolism,
        "fishing_management": fishing_management,
    }


def _group_index(scenario: RsimScenario, group_name: str) -> int:
    names = [str(name) for name in getattr(scenario.params, "spname", [])]
    if group_name not in names:
        raise ValueError(f"Unknown group for control: {group_name}")
    return names.index(group_name)


def _revenue_effort_target(
    scenario: RsimScenario, baseline: np.ndarray, effort_budget: float
) -> np.ndarray:
    fish_from = np.asarray(scenario.params.FishFrom, dtype=int)
    fish_through = np.asarray(scenario.params.FishThrough, dtype=int)
    fish_q = np.asarray(scenario.params.FishQ, dtype=float)
    biomass = np.asarray(scenario.params.B_BaseRef, dtype=float)

    n_fleets = len(baseline)
    revenue_signal = np.zeros(n_fleets, dtype=float)
    for idx in range(len(fish_q)):
        fleet_idx = int(fish_through[idx] - 1)
        group_idx = int(fish_from[idx])
        if 0 <= fleet_idx < n_fleets and 0 <= group_idx < len(biomass):
            revenue_signal[fleet_idx] += max(fish_q[idx], 0.0) * max(
                biomass[group_idx], 0.0
            )

    optimizer = FishingOptimization(
        min_effort=0.0,
        max_effort=max(10.0, float(np.max(baseline)) * 10.0),
    )
    raw_target = optimizer.maximize_revenue(
        catchability=np.maximum(revenue_signal, 1e-9),
        biomass=np.ones(n_fleets, dtype=float),
        price=np.ones(n_fleets, dtype=float),
        effort_budget=float(effort_budget),
    )

    total_raw = float(np.sum(raw_target))
    total_baseline = float(np.sum(baseline))
    if total_raw > 0 and total_baseline > 0:
        raw_target = raw_target * (total_baseline / total_raw)
    return np.clip(raw_target, 0.0, None)


def _bycatch_effort_target(
    scenario: RsimScenario, baseline: np.ndarray, bycatch_reduction: float
) -> np.ndarray:
    fish_from = np.asarray(scenario.params.FishFrom, dtype=int)
    fish_through = np.asarray(scenario.params.FishThrough, dtype=int)
    biomass = np.asarray(scenario.params.B_BaseRef, dtype=float)

    n_fleets = len(baseline)
    risk = np.ones(n_fleets, dtype=float)
    fleet_counts = np.zeros(n_fleets, dtype=int)
    for idx in range(len(fish_from)):
        fleet_idx = int(fish_through[idx] - 1)
        group_idx = int(fish_from[idx])
        if 0 <= fleet_idx < n_fleets and 0 <= group_idx < len(biomass):
            risk[fleet_idx] += 1.0 / max(biomass[group_idx], 1e-6)
            fleet_counts[fleet_idx] += 1
    for i in range(n_fleets):
        if fleet_counts[i] > 0:
            risk[i] /= fleet_counts[i]

    optimizer = FishingOptimization(
        min_effort=0.0,
        max_effort=max(10.0, float(np.max(baseline)) * 10.0),
    )
    reduction_abs = float(bycatch_reduction) * float(np.sum(baseline))
    return optimizer.minimize_bycatch(
        effort=baseline,
        bycatch_risk=risk,
        target_reduction=max(reduction_abs, 0.0),
    )
