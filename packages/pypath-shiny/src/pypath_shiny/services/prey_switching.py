"""Prey-switching (EwE ``SwitchPower``) settings for the app.

Pure Python and numpy: no Shiny imports, so it can be used and tested without
a session. Overrides live in an immutable :class:`SwitchingStore`; nothing here
mutates its inputs.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field

import numpy as np

SWITCH_POWER_MAX: float = 2.0
DISPLAY_DECIMALS: int = 3
BALANCED_SOURCE: str = "model"

_TOLERANCE = 1e-12


@dataclass(frozen=True)
class SwitchingStore:
    """Overrides by source key, and the as-built values of EwE sources."""

    overrides: Mapping[str, Mapping[str, float]] = field(default_factory=dict)
    source_values: Mapping[str, Mapping[str, float]] = field(default_factory=dict)


def source_key(source: dict | None, scenario_id=None) -> str:
    """Key identifying where the current model came from."""
    if not source:
        return BALANCED_SOURCE
    kind = source.get("kind") or "ewemdb"
    return f"{kind}|{source.get('path')}|{scenario_id}"


def _names(values) -> list[str]:
    return [str(n).strip() for n in values]


def consumers_from_params(params) -> list[str]:
    """Groups 1..NUM_GROUPS with PP_type == 0, as stripped names."""
    n = int(params.NUM_GROUPS)
    names = _names(params.spname)
    pp = np.asarray(params.PP_type)
    return [names[i] for i in range(1, n + 1) if i < len(names) and pp[i] == 0]


def consumers_from_model(model) -> list[str]:
    """Consumers (0 <= type < 1) of a balanced Rpath or an RpathParams model."""
    if hasattr(model, "type") and hasattr(model, "Group"):
        groups, types = model.Group, model.type
    else:
        table = getattr(model, "model", model)
        groups, types = table["Group"], table["Type"]
    return [
        name
        for name, t in zip(_names(groups), np.asarray(types, dtype=float))
        if 0 <= t < 1
    ]


def switching_by_name(params) -> dict[str, float]:
    """Switching power of every group 1..NUM_GROUPS (zeros if unset or short)."""
    n = int(params.NUM_GROUPS)
    names = _names(params.spname)
    sp = params.SwitchPower
    out = {}
    for i in range(1, n + 1):
        value = 0.0
        if sp is not None and i < len(sp):
            value = float(sp[i])
        out[names[i]] = value
    return out


def snapshot_source(params) -> dict[str, float]:
    """Consumers' powers plus any nonzero non-consumer, as built."""
    consumers = set(consumers_from_params(params))
    return {
        name: v
        for name, v in switching_by_name(params).items()
        if name in consumers or v != 0
    }


def check_power(value) -> float:
    """Return value as a float in [0, 2], or raise ValueError."""
    v = float(value)
    if not math.isfinite(v) or v < 0 or v > SWITCH_POWER_MAX:
        raise ValueError(
            f"Switching power must be between 0 and 2; {value} was not applied."
        )
    return v


def _rebuilt(store, key, group_values):
    overrides = {k: dict(v) for k, v in store.overrides.items()}
    if group_values:
        overrides[key] = group_values
    else:
        overrides.pop(key, None)
    source_values = {k: dict(v) for k, v in store.source_values.items()}
    return SwitchingStore(overrides=overrides, source_values=source_values)


def set_override(store, key, group, value, source_value) -> SwitchingStore:
    """Return a store with one group's override set, or removed if it reverts."""
    if value is None:
        return store
    try:
        v = float(value)
    except (TypeError, ValueError):
        return store
    current = dict(store.overrides.get(key, {}))
    if round(v, DISPLAY_DECIMALS) == round(float(source_value), DISPLAY_DECIMALS):
        current.pop(group, None)
    else:
        current[group] = check_power(v)
    return _rebuilt(store, key, current)


def effective_values(store, key, base: Mapping[str, float]) -> dict[str, float]:
    """``base`` with the key's overrides applied."""
    out = dict(base)
    out.update(store.overrides.get(key, {}))
    return out


def fold_edits(
    store, key, boxes: Mapping[str, object], base: Mapping[str, float]
) -> tuple[SwitchingStore, dict[str, str]]:
    """Fold the boxes' values into the store; report the boxes refused."""
    current = effective_values(store, key, base)
    errors: dict[str, str] = {}
    for name, value in boxes.items():
        if value is None:
            continue
        if name in current and value == round(current[name], DISPLAY_DECIMALS):
            continue
        try:
            store = set_override(store, key, name, value, base.get(name, 0.0))
        except ValueError as exc:
            errors[name] = str(exc).replace(
                "Switching power", f"Switching power for {name}", 1
            )
    return store, errors


def switch_power_with_overrides(params, overrides: Mapping[str, float]):
    """New SwitchPower array with the overrides written, or None if none."""
    if not overrides:
        return None
    size = int(params.NUM_GROUPS) + 1
    sp = params.SwitchPower
    if sp is None or len(sp) < size:
        arr = np.zeros(size)
    else:
        arr = np.array(sp, dtype=float, copy=True)
    names = _names(params.spname)
    consumers = set(consumers_from_params(params))
    for name, value in overrides.items():
        if name in consumers:
            arr[names.index(name)] = value
    return arr


def summary(params) -> dict[str, float]:
    """Nonzero switching powers by name, in group order."""
    return {n: v for n, v in switching_by_name(params).items() if v != 0}


def summary_line(values: Mapping[str, float] | None) -> str:
    if not values:
        return "Prey switching: none"
    return "Prey switching: " + ", ".join(f"{n} {v:g}" for n, v in values.items())


def pending(store, key, base: Mapping[str, float], params) -> bool:
    """True when the settings differ from what the scenario's params hold."""
    eff = effective_values(store, key, base)
    built = switching_by_name(params)
    return any(
        abs(eff.get(name, 0.0) - built.get(name, 0.0)) > _TOLERANCE
        for name in consumers_from_params(params)
    )


def above_range(values: Mapping[str, float]) -> dict[str, float]:
    return {n: v for n, v in values.items() if v > SWITCH_POWER_MAX}


def on_model_change(
    store, *, rebalance: bool, consumers: Sequence[str], keep_ewe: bool
) -> tuple[SwitchingStore, str | None]:
    """Settings after a model load (clear) or a re-balance (keep what applies)."""
    total = sum(len(v) for v in store.overrides.values())
    if not store.overrides:
        return store, None
    if not rebalance:
        return (
            SwitchingStore(),
            f"Prey switching settings cleared ({total} group(s)): "
            "a different model was loaded.",
        )
    allowed = set(consumers)
    overrides: dict[str, dict[str, float]] = {}
    source_values: dict[str, dict[str, float]] = {}
    dropped = 0
    for key, groups in store.overrides.items():
        if key == BALANCED_SOURCE:
            kept = {g: v for g, v in groups.items() if g in allowed}
            dropped += len(groups) - len(kept)
        elif keep_ewe:
            kept = dict(groups)
            if key in store.source_values:
                source_values[key] = dict(store.source_values[key])
        else:
            kept = {}
        if kept:
            overrides[key] = kept
    n_kept = sum(len(v) for v in overrides.values())
    tail = (
        f"; dropped for {dropped} that are gone or no longer consumers."
        if dropped
        else "."
    )
    note = (
        f"Prey switching settings kept for {n_kept} group(s) after re-balancing{tail}"
    )
    return SwitchingStore(overrides=overrides, source_values=source_values), note
