# Prey Switching in the App: Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Let users set EwE's per-predator switching power on the Ecosim page, for models balanced in the app and for imported EwE scenarios and examples. Every run shows the switching it used.

**Architecture:**

- A pure service module, `services/prey_switching.py`, holds every rule: eligibility, overrides per source, applying them, snapshots, the summary line, the range and the model-change rule.
- The Ecosim page renders one numeric input per consumer and applies the overrides at Create Scenario.
- Each run's switching is recorded from the scenario that ran, on the Ecosim, Ecospace and IBM pages.
- A one-shot re-balance mark, shared by `app.py` with the Ecopath and Ecosim pages, tells a re-balance from a model load.
- The core changes only to deprecate `DietRewiring`.

**Tech Stack:** Python ≥ 3.10, Shiny for Python, numpy, pytest.

**Spec:** `docs/plans/2026-11-17-prey-switching-app-design.md` (approved 2026-11-17 after three review rounds). Read it first; this plan argues from it.

## Global Constraints

- **The range.** A switching power is ≥ 0, and 0 means off. EwE limits it to [0, 2] (Christensen & Walters, *EwE textbook*, ch. 21, "Holling functional response": "Pj is limited to the range [0,2]"). That confirms the design's range, so:
  - a typed value outside [0, 2] is refused with a message;
  - an imported value outside it is kept and warned about.
- **Eligibility** is `PP_type == 0` in scenario parameters, which includes partial producers because `rsim_params` casts the type to an integer. For a model with no scenario yet it is `0 <= type < 1` on the model's `type` / `Type` column. Producers, detritus and fleets are never offered.
- **Overrides are keyed by group name**, per source:
  - the balanced route's source key is `"model"`;
  - an EwE source's key is `f"{source.get('kind', 'ewemdb')}|{source['path']}|{scenario_id}"`.
- **Untouched means unchanged.** With no override, `params.SwitchPower` stays the very array the source built. The parity suite and `capture_parity.py` (139 arrays) must stay identical.
- **Do not reuse these constants.** `core/constants.py`'s `*_PREY_SWITCHING_POWER` belongs to autofix's DD. `config.py`'s `PARAM_RANGES.switching_power_*` is the old demo's range, 1-5.
- **The repository's rules:**
  - app tests go in `packages/pypath-shiny/tests/`;
  - run `ruff check` and `ruff format --check` (ruff 0.15.x, CI's gate) before each commit;
  - never run `scripts/deploy.sh`;
  - never merge or push `main` without the user's word.
- **The Python environment.** `/opt/micromamba/envs/shiny/bin/python` with `PYTHONPATH=$PWD/packages/pypath/src:$PWD/packages/pypath-shiny/src`, run from the worktree root. Do not install into that environment.
- **Display precision** is 3 decimals: `DISPLAY_DECIMALS = 3`.

## Review Focus

The five inputs most likely to bite a user that the spec implies but does not test. Each is pinned by a test in the task named:

1. **A numeric box the user clears** (Shiny gives `None`): no override is written, nothing raises, and the box's previous override stays. *(Task 2: `test_set_override_ignores_none`)*
2. **A scenario whose `SwitchPower` is None or shorter than `NUM_GROUPS + 1`** (older readers): it is treated as zeros everywhere, with no IndexError. *(Task 2: `test_summary_and_snapshot_tolerate_missing_or_short_switchpower`)*
3. **An imported value above 2:** shown, kept to the bit, and warned about once at Create Scenario. *(Task 2: `test_imported_values_above_range_are_reported`; Task 3: `test_create_scenario_warns_about_imported_values_above_range`)*
4. **A model with no consumers:** the control says so instead of rendering an empty box. *(Task 3: `test_control_says_so_without_consumers`)*
5. **A large model (50 consumers):** the control renders every input inside a scrolling box. *(Task 3: `test_control_renders_fifty_consumers_in_a_scroll_box`)*

---

### Task 1: Deprecate `DietRewiring` in the core; update the deviations row

**Files:**
- Modify: `packages/pypath/src/pypath/core/forcing.py` (class `DietRewiring`, around line 256; `create_diet_rewiring`, around line 471)
- Modify: `packages/pypath/tests/test_diet_rewiring.py` and `packages/pypath/tests/test_ecosim_advanced.py`: add a module-level filter
- Modify: `docs/parity/deviations.md`, row `DR`
- Test: `packages/pypath/tests/test_diet_rewiring_deprecation.py` (new)

**Interfaces:**
- Produces: `DietRewiring(enabled=True, ...)` emits exactly one `DeprecationWarning` whose message starts with `"DietRewiring is deprecated"`. `DietRewiring()` and `DietRewiring(enabled=False)` emit none. `create_diet_rewiring(...)` emits exactly one (through the constructor).

- [ ] **Step 1: Write the failing tests** in `test_diet_rewiring_deprecation.py`:

```python
import warnings
import pytest
from pypath.core.forcing import DietRewiring, create_diet_rewiring

def _deprecations(fn):
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        fn()
    return [x for x in w if issubclass(x.category, DeprecationWarning)]

def test_enabled_diet_rewiring_warns_once():
    got = _deprecations(lambda: DietRewiring(enabled=True))
    assert len(got) == 1
    assert str(got[0].message).startswith("DietRewiring is deprecated")
    assert "SwitchPower" in str(got[0].message)

def test_disabled_diet_rewiring_is_silent():
    assert _deprecations(lambda: DietRewiring()) == []
    assert _deprecations(lambda: DietRewiring(enabled=False)) == []

def test_create_diet_rewiring_warns_once():
    assert len(_deprecations(lambda: create_diet_rewiring())) == 1

def test_warning_points_at_the_caller():
    got = _deprecations(lambda: DietRewiring(enabled=True))
    assert got[0].filename == __file__
```

- [ ] **Step 2: Run them.** `pytest packages/pypath/tests/test_diet_rewiring_deprecation.py -v`. Expected: the enabled, create and caller tests FAIL; the disabled test passes.

- [ ] **Step 3: Implement `DietRewiring.__post_init__(self) -> None`.** It warns only when `self.enabled`, with `stacklevel=3` so the warning names the caller of `__init__`. The message, exactly:
  `"DietRewiring is deprecated: it never integrated. Use prey switching instead: scenario.params.SwitchPower (EwE's switching power, 0 to 2), or the Ecosim page's Prey Switching control."`
  `create_diet_rewiring` gets no warning of its own.

- [ ] **Step 4: Silence the expected warnings in the old tests.** Add `pytestmark = pytest.mark.filterwarnings("ignore:DietRewiring is deprecated:DeprecationWarning")` at module level in `test_diet_rewiring.py` and `test_ecosim_advanced.py`, importing `pytest` if missing.

- [ ] **Step 5: Update `deviations.md` row DR.**
  - Keep the facts.
  - Replace the last column, "Next: diet rewiring maps onto prey switching", with: "Prey switching is set per predator on the Ecosim page and through `RsimParams.SwitchPower`; `DietRewiring` is deprecated (warns when enabled)".

- [ ] **Step 6: Run.** `pytest packages/pypath/tests/test_diet_rewiring_deprecation.py packages/pypath/tests/test_diet_rewiring.py packages/pypath/tests/test_ecosim_advanced.py -q`. Expected: all pass, and no `DietRewiring is deprecated` warnings in the summary.

- [ ] **Step 7: Commit.**

```bash
git add packages/pypath/src/pypath/core/forcing.py packages/pypath/tests/test_diet_rewiring_deprecation.py packages/pypath/tests/test_diet_rewiring.py packages/pypath/tests/test_ecosim_advanced.py docs/parity/deviations.md
git commit -m "feat(core): deprecate DietRewiring in favour of SwitchPower"
```

---

### Task 2: The prey-switching service

**Files:**
- Create: `packages/pypath-shiny/src/pypath_shiny/services/prey_switching.py`
- Test: `packages/pypath-shiny/tests/test_prey_switching_service.py`

**Interfaces.** Produces these names; Tasks 3-5 and 7 use them exactly as named:

```python
SWITCH_POWER_MAX: float = 2.0
DISPLAY_DECIMALS: int = 3
BALANCED_SOURCE: str = "model"

@dataclass(frozen=True)
class SwitchingStore:
    overrides: Mapping[str, Mapping[str, float]] = field(default_factory=dict)      # source key -> {group: power}
    source_values: Mapping[str, Mapping[str, float]] = field(default_factory=dict)  # EwE source key -> {group: value as built}

def source_key(source: dict | None, scenario_id=None) -> str
def consumers_from_params(params) -> list[str]           # PP_type == 0, in group order, from scenario.params
def consumers_from_model(model) -> list[str]             # 0 <= type < 1, from a balanced Rpath (Group, type) or RpathParams (model["Group"], model["Type"])
def switching_by_name(params) -> dict[str, float]         # every group's SwitchPower by name (zeros when None or short)
def snapshot_source(params) -> dict[str, float]           # switching_by_name restricted to consumers, plus any nonzero non-consumer
def check_power(value) -> float                           # float in [0, 2]; ValueError with the exact message below otherwise
def set_override(store, key, group, value, source_value) -> SwitchingStore
def effective_values(store, key, base: Mapping[str, float]) -> dict[str, float]   # base with the key's overrides applied
def switch_power_with_overrides(params, overrides: Mapping[str, float]) -> np.ndarray | None
def summary(params) -> dict[str, float]                   # nonzero switching powers by name, in group order
def summary_line(values: Mapping[str, float] | None) -> str
def pending(store, key, base: Mapping[str, float], params) -> bool
def above_range(values: Mapping[str, float]) -> dict[str, float]
def on_model_change(store, *, rebalance: bool, consumers: Sequence[str], keep_ewe: bool) -> tuple[SwitchingStore, str | None]
```

Behaviour:

- **`set_override`.** Returns a new store and never mutates.
  - `value` None, or not convertible to float: the store is returned unchanged.
  - Otherwise it calls `check_power` and lets `ValueError` propagate.
  - If `round(value, 3) == round(source_value, 3)`, the group's override is removed (the full-precision source value then applies). Otherwise it is stored.
- **`switch_power_with_overrides`.** Returns None when `overrides` is empty, so the caller leaves the scenario's array alone. Otherwise it returns a new array: a copy of `params.SwitchPower`, or zeros of length `NUM_GROUPS + 1` when that is None or short, with each named override written at the group's index. A name that is not a consumer of `params` is skipped.
- **`pending`.** True when `effective_values(store, key, base)` differs from `switching_by_name(params)` on any consumer of `params` by more than 1e-12.
- **`summary_line`.** Gives `"Prey switching: none"` for None or empty input, otherwise `"Prey switching: " + ", ".join(f"{name} {value:g}" ...)` in the mapping's order.
- **`on_model_change`.**
  - Returns `(store, None)` when the store holds no overrides.
  - When `rebalance` is False, it clears every override and source value, with the note `f"Prey switching settings cleared ({n} group(s)): a different model was loaded."`, where n is the number of overrides cleared.
  - When `rebalance` is True:
    - it keeps the `"model"` overrides whose names are in `consumers`;
    - it keeps the EwE keys only when `keep_ewe`;
    - the note is `f"Prey switching settings kept for {k} group(s) after re-balancing."`, with `f"; dropped for {d} that are gone or no longer consumers."` replacing the final period when d > 0.
- **`check_power`'s message**, for a value outside [0, 2] or non-finite: `f"Switching power must be between 0 and 2; {value} was not applied."`

- [ ] **Step 1: Write the failing tests.**
  - **Fixture.** Build a dedicated `scen` fixture through `create_rpath_params`, `rpath` and `rsim_scenario`: a balanced model of four groups named, in order, "Phyto" (producer), "Cod" and "Seals" (consumers, Seals eating Cod, Cod eating Phyto) and "Detritus". The names are what the tests below assert. Put it in `test_prey_switching_service.py` and import it from the later tasks' tests.
  - **Exact assertions:**

```python
def test_consumers_are_pp_type_zero(scen):
    assert consumers_from_params(scen.params) == ["Cod", "Seals"]

def test_source_keys():
    assert source_key(None) == "model"
    assert source_key({"path": "/x.eweaccdb"}, 16) == "ewemdb|/x.eweaccdb|16"
    assert source_key({"kind": "ewef_example", "path": "/ex"}, 1) == "ewef_example|/ex|1"

def test_untouched_means_none(scen):
    assert switch_power_with_overrides(scen.params, {}) is None

def test_override_written_by_name_on_a_copy(scen):
    before = scen.params.SwitchPower.copy()
    arr = switch_power_with_overrides(scen.params, {"Cod": 1.0, "Phyto": 1.0})
    i = list(scen.params.spname).index("Cod")
    assert arr[i] == 1.0 and arr.sum() == 1.0          # Phyto (producer) skipped
    assert np.array_equal(scen.params.SwitchPower, before)

def test_set_override_ignores_none():
    s = set_override(SwitchingStore(), "model", "Cod", 1.0, 0.0)
    assert set_override(s, "model", "Cod", None, 0.0) == s

def test_untouched_rounded_value_writes_no_override():
    # an untouched input returns the displayed value, round(0.30000001, 3)
    assert set_override(SwitchingStore(), "k", "Cod", 0.3, 0.30000001) == SwitchingStore()

def test_editing_back_to_the_displayed_source_value_removes_the_override():
    s = set_override(SwitchingStore(), "k", "Cod", 0.5, 0.30000001)
    s = set_override(s, "k", "Cod", 0.3, 0.30000001)
    assert "Cod" not in s.overrides.get("k", {})

def test_out_of_range_is_refused():
    with pytest.raises(ValueError, match="between 0 and 2; 3.0 was not applied"):
        set_override(SwitchingStore(), "model", "Cod", 3.0, 0.0)

def test_summary_line():
    assert summary_line(None) == "Prey switching: none"
    assert summary_line({"Cod": 1.0, "Seals": 0.5}) == "Prey switching: Cod 1, Seals 0.5"

def test_summary_and_snapshot_tolerate_missing_or_short_switchpower(scen):
    scen.params.SwitchPower = None
    assert summary(scen.params) == {}
    assert snapshot_source(scen.params) == {"Cod": 0.0, "Seals": 0.0}
    scen.params.SwitchPower = np.zeros(2)
    assert summary(scen.params) == {}

def test_imported_values_above_range_are_reported():
    assert above_range({"Cod": 2.5, "Seals": 1.0}) == {"Cod": 2.5}

def test_pending(scen):
    s = set_override(SwitchingStore(), "model", "Cod", 1.0, 0.0)
    base = {"Cod": 0.0, "Seals": 0.0}
    assert pending(s, "model", base, scen.params)
    scen.params.SwitchPower = switch_power_with_overrides(scen.params, {"Cod": 1.0})
    assert not pending(s, "model", base, scen.params)

def test_load_clears_and_rebalance_keeps_consumers():
    s = set_override(SwitchingStore(), "model", "Cod", 1.0, 0.0)
    s = set_override(s, "model", "Seals", 0.5, 0.0)
    s = set_override(s, "ewemdb|/x|1", "Cod", 2.0, 1.0)
    cleared, note = on_model_change(s, rebalance=False, consumers=["Cod"], keep_ewe=True)
    assert cleared == SwitchingStore() and note.startswith("Prey switching settings cleared (3 group(s))")
    kept, note = on_model_change(s, rebalance=True, consumers=["Cod"], keep_ewe=False)
    assert dict(kept.overrides) == {"model": {"Cod": 1.0}}
    assert note == ("Prey switching settings kept for 1 group(s) after re-balancing; "
                    "dropped for 1 that are gone or no longer consumers.")
    assert on_model_change(SwitchingStore(), rebalance=False, consumers=[], keep_ewe=False) == (SwitchingStore(), None)
```

- [ ] **Step 2: Run.** `pytest packages/pypath-shiny/tests/test_prey_switching_service.py -v`. Expected: FAIL on the import (module missing).

- [ ] **Step 3: Implement the module** with the signatures and behaviour above. `summary`, `switching_by_name` and `snapshot_source` iterate groups `1..NUM_GROUPS` with names from `params.spname`, stripped.

- [ ] **Step 4: Run Step 2's command.** Expected: all pass.

- [ ] **Step 5: Commit** (`feat(shiny): prey-switching service`).

---

### Task 3: The Ecosim page control, applied at Create Scenario

**Files:**
- Modify: `packages/pypath-shiny/src/pypath_shiny/pages/ecosim.py`. Line numbers are as of ac6bac1 and move as you edit:
  - the sidebar diet note, around lines 476-482: replace it;
  - `ewe_scenario_source_ui`'s two notes, around lines 1088-1104;
  - the help text's "Dynamic Diet Rewiring" block, around lines 1330-1338;
  - `_create_scenario`, around lines 1958-2120.
- Modify: `packages/pypath-shiny/tests/test_ecosim_continuous_ui.py`: replace `test_diet_rewiring_controls_are_gone_and_a_note_replaces_them` and the `DIET_NOTE` constant.
- Test: `packages/pypath-shiny/tests/test_prey_switching_ui.py` (new)

**Interfaces:**
- Consumes: Task 2's names.
- Produces, in `ecosim_server`:
  - `switching_store = reactive.Value(SwitchingStore())`;
  - the inputs `switch_power_{i}` for i = 0..n-1 over the control's current group list;
  - the outputs `switching_control` and `switching_pending_note`;
  - the helper `_current_switching_key() -> str`, which is `"model"` unless the "EwE" source is selected and matching, then `source_key(source, scenario_id)`.
- Produces, at module level in `pages/ecosim.py`: `switching_inputs(groups: list[str], base: Mapping[str, float], extra_readonly: Mapping[str, float]) -> ui.Tag`. It is the control's body: the inputs, the read-only lines, or the no-consumers text. `switching_control` renders it, and the tests call it directly.

**Behaviour:**

- **The control.**
  - Before the scenario exists on the EwE route, it shows only `"Create the scenario to see and change its switching powers."`.
  - Otherwise it shows one `ui.input_numeric(f"switch_power_{i}", name, value=round(v, 3), min=0, max=2, step=0.1)` per consumer:
    - on the balanced route the consumers come from `consumers_from_model(model_data.get())` and the base is 0;
    - on the EwE route they come from `consumers_from_params(scenario.get().params)`, with the base taken from `switching_store.get().source_values[key]`.
  - The inputs sit inside `ui.div(..., style="max-height: 320px; overflow-y: auto;")`.
  - A nonzero imported value on a non-consumer is shown read-only as `ui.p(f"{name}: {value:g} (imported; this group cannot switch, kept as imported)", class_="text-muted small")`.
  - With no consumers, the text is `"This model has no consumers to switch."`.
  - The control reads `switching_store` inside `reactive.isolate()`, so an edit does not re-render it.
- **Collecting edits.** One `@reactive.effect` reads every `switch_power_{i}` and calls `set_override` against the group's base value. On a `ValueError` it calls `notify_warning(str(e).replace("Switching power", f"Switching power for {name}"))` and stores nothing.
- **The pending note.**
  - With a current scenario and `pending(...)` true, it reads `"Changed values apply when you create the scenario again."`.
  - With no scenario and overrides for the key, it reads `"Values apply when you create the scenario."`.
  - Otherwise it is empty.
- **`_create_scenario`.** Right after `new_scenario` is built, before fishing and the controls:
  1. On the EwE route, store `snapshot_source(new_scenario.params)` as `source_values[key]` in a new store.
  2. `arr = switch_power_with_overrides(new_scenario.params, overrides_for_key)`. If `arr is not None`, set `new_scenario.params.SwitchPower = arr`.
  3. For each `name, v in above_range(snapshot_source(new_scenario.params)).items()`, call `notify_warning(f"{name} has an imported switching power of {v:g}, above EwE's range of 0 to 2; it is kept.")`.

  The autofix call that follows deep-copies the parameters and leaves `SwitchPower` alone (verified in review), so no change there.
- **Exact copy:**
  - **Sidebar section,** replacing the diet note: `ui.h5("Prey Switching")`, then `ui.p("Predators turn towards prey that become more abundant. 0 = off; higher values switch more sharply (EwE allows 0 to 2). Values apply when you create the scenario.", class_="text-muted small")`, then `ui.output_ui("switching_control")` and `ui.output_ui("switching_pending_note")`.
  - **The database note:** `"Runs the database's Ecopath inputs with the scenario's own vulnerabilities, feeding time, switching, forcing and length, as EwE does. Edits made in this app and the vulnerability slider are not used; prey switching set in the sidebar is."`
  - **The example note:** `"Runs the example's Ecopath inputs with its own EwE-F vulnerabilities, group settings, fishing and run length. Edits made in this app and the vulnerability slider are not used; prey switching set in the sidebar is."`
  - **Help text:** replace `ui.h6("Dynamic Diet Rewiring")` and its paragraph with `ui.h6("Prey Switching")` and `ui.p("Set a predator's switching power in the sidebar to let it turn towards prey that become more abundant (EwE's SwitchPower, 0 to 2; 0 = off). Values apply when the scenario is created; an imported EwE scenario starts from its own values. Every run shows the switching it used. Diet rewiring is deprecated: it never integrated.")`.

- [ ] **Step 1: Write the failing tests** in `test_prey_switching_ui.py`:
  - **Static checks** on `str(ecosim_ui())`:
    - `"Prey Switching"` and `'id="switching_control"'` are present;
    - the old sentence `"Diet rewiring is not available in this version"` is absent;
    - the help text contains `"Diet rewiring is deprecated: it never integrated."`.
  - **Wiring checks** with the AST helpers `_ecosim_tree`, `_find_function` and `_calls`. Import them from `tests.test_ecosim_continuous_ui`; if that import fails under the suite's import mode, move the three helpers to `packages/pypath-shiny/tests/ast_helpers.py` and import them from there in both files. The checks:
    - `_create_scenario` calls `switch_power_with_overrides`, `snapshot_source` and `above_range`;
    - its `switch_power_with_overrides` call comes before `_apply_fishing_scenario`, by line number.
  - **`test_create_scenario_warns_about_imported_values_above_range`.** `_create_scenario` contains a `notify_warning` call whose f-string includes `"above EwE's range of 0 to 2; it is kept."`.
  - **Control behaviour,** through a helper extracted for testability: `switching_inputs(groups: list[str], base: Mapping[str, float], extra_readonly: Mapping[str, float]) -> ui.Tag`, a module-level function in `pages/ecosim.py`.
    - **`test_control_renders_fifty_consumers_in_a_scroll_box`.** 50 names give 50 `id="switch_power_` occurrences and `overflow-y: auto` in the HTML.
    - **`test_control_says_so_without_consumers`.** `switching_inputs([], {}, {})` contains `"This model has no consumers to switch."`.
    - **`test_rounded_display_value`.** A base of 0.30000001 renders `value="0.3"`.
    - **`test_readonly_imported_non_consumer`.** `extra_readonly={"Phyto": 1.0}` renders `"Phyto: 1 (imported; this group cannot switch, kept as imported)"`.
  - **`test_example_route_applies_overrides_and_keeps_the_rest`.**
    - Build `example_ecosim_scenario("Central_Baltic")`, which is what `build_ewe_scenario` returns for an example, and take its first consumer from `consumers_from_params`.
    - With no override, `switch_power_with_overrides` returns None.
    - With `{first: 1.5}`, the returned array has 1.5 at that group and equals the scenario's array everywhere else, bit for bit.
  - **Replace the old diet-note test** in `test_ecosim_continuous_ui.py` with `test_diet_rewiring_controls_are_gone_and_prey_switching_replaces_them`:
    - the four diet input ids are absent;
    - `"Dynamic Diet Rewiring"` is absent;
    - `'id="switching_control"'` is present;
    - no conditional panel encloses it: use `_enclosing_divs(text, 'id="switching_control"')`.

- [ ] **Step 2: Run.** `pytest packages/pypath-shiny/tests/test_prey_switching_ui.py packages/pypath-shiny/tests/test_ecosim_continuous_ui.py -v`. Expected: the new tests FAIL; the others pass.

- [ ] **Step 3: Implement** the behaviour and copy above.

- [ ] **Step 4: Run Step 2's command, then the whole app suite.** `pytest packages/pypath-shiny/tests --ignore=packages/pypath-shiny/tests/ui -q`. Expected: all pass.

- [ ] **Step 5: Commit** (`feat(shiny): per-predator prey switching on the Ecosim page`).

---

### Task 4: Every run shows its switching (Ecosim, Ecospace, IBM)

**Files:**
- Modify: `pages/ecosim.py`:
  - `_land_result`, around line 2298;
  - the UI next to `simulation_status`, around line 754.
- Modify: `pages/ecospace.py`:
  - the state, around line 907;
  - `_run_spatial_simulation`, around lines 1440-1534;
  - every `_spatial_results.set(None)`;
  - the UI next to `spatial_run_status`, around line 799.
- Modify: `pages/ibm.py`:
  - the state, around lines 362-363;
  - `_run_comparison`, around lines 545-600;
  - the "Biomass Comparison" panel, around lines 274-283.
- Test: `packages/pypath-shiny/tests/test_prey_switching_record.py` (new)

**Interfaces:**
- Consumes: `summary(params) -> dict[str, float]` and `summary_line(values) -> str` from Task 2.
- Produces:
  - the attribute `output.switching_used: dict[str, float]` on the Ecosim run output;
  - `output.params["applied_controls"]["prey_switching"]`, the same dict, when `output.params` is a dict;
  - the outputs `switching_line` (Ecosim), `spatial_switching_line` (Ecospace) and `ibm_switching_line` (IBM).

**Behaviour:**

- **Ecosim.** `_land_result` sets `output.switching_used = summary(outcome.source.params)` before `sim_output.set(output)`. `switching_line` renders `ui.p(summary_line(getattr(out, "switching_used", None)), class_="text-muted small")` when `sim_output.get()` is not None, else nothing. It is placed right after `ui.output_ui("simulation_status")`, so it shows for RK4, AB and continuous runs alike.
- **Ecospace.** `_spatial_switching = reactive.Value(None)`.
  - In `_run_spatial_simulation`, take `used = summary(scenario.params)` right after `scenario = _sim_scenario()`.
  - Call `_spatial_switching.set(used)` right after `_spatial_results.set(result)`, and `_spatial_switching.set(None)` wherever `_spatial_results.set(None)` is called.
  - `spatial_switching_line` renders from `_spatial_switching.get()` while `_spatial_results.get()` is not None, never from `_sim_scenario()`.
- **IBM.** `ibm_switching = reactive.Value(None)`.
  - In `_run_comparison`, take `used = summary(scenario.params)` before the deep copies, and call `ibm_switching.set(used)` right after `standard_output.set(std_result)`.
  - `ibm_switching_line` renders in the "Biomass Comparison" panel while `standard_output.get()` is not None.

- [ ] **Step 1: Write the failing tests:**
  - **Wiring** (AST, with the helpers named in Task 3):
    - `_land_result` assigns `switching_used`;
    - `_run_spatial_simulation` calls `summary` and sets `_spatial_switching` after `_spatial_results.set(result)`, by line number;
    - `_run_comparison` calls `summary` before `copy.deepcopy`;
    - the functions rendering `spatial_switching_line` and `ibm_switching_line` never call `_sim_scenario`.
  - **The rendered line** (static): `ecosim_ui()` contains `id="switching_line"` after `id="simulation_status"`; the Ecospace and IBM UIs contain `id="spatial_switching_line"` and `id="ibm_switching_line"`.
  - **`test_record_comes_from_the_scenario_that_ran`.**
    - Build a scenario, set `SwitchPower` for "Cod" to 1, and take `summary(scen.params)`.
    - Then create a new scenario object with Cod at 0.5, as a re-created Ecosim scenario would be.
    - The stored dict still says `{"Cod": 1.0}`, and `summary_line` of it reads `"Prey switching: Cod 1"`.

    This pins the semantics on the service; the page wiring tests above pin where it is called.
  - **`test_line_for_every_method`.** Run the fixture scenario for 1 year with `rsim_run` on `"RK4"`, `"AB"` and `"lsoda"`, with Cod at 1. `summary_line(summary(scen.params))` reads `"Prey switching: Cod 1"` each time, and the run's biomass differs from the same run with Cod at 0. That shows the switching reached the run.

- [ ] **Step 2: Run.** `pytest packages/pypath-shiny/tests/test_prey_switching_record.py -v`. Expected: the wiring and static tests FAIL.

- [ ] **Step 3: Implement** the behaviour above.

- [ ] **Step 4: Run Step 2's command, then the whole app suite.** Expected: all pass.

- [ ] **Step 5: Commit** (`feat(shiny): every run shows the prey switching it used`).

---

### Task 5: Re-balance keeps overrides; any other model change clears them

**Files:**
- Modify: `packages/pypath-shiny/src/pypath_shiny/app.py`:
  - the state, around lines 408-412: add `rebalance_mark = reactive.Value(False)`;
  - pass `rebalance_mark=rebalance_mark` to `ecopath.ecopath_server` (around line 469) and `ecosim.ecosim_server` (around line 476).
- Modify: `pages/ecopath.py`:
  - `ecopath_server` gains the keyword parameter `rebalance_mark: reactive.Value | None = None`;
  - at the re-balance, around line 1365, call `rebalance_mark.set(True)` right before `model_data.set(model)`, when the mark is not None.
- Modify: `pages/ecosim.py`: `ecosim_server` gains `rebalance_mark: reactive.Value | None = None`; extend `_retire_run_on_model_change`, around line 1033.
- Test: `packages/pypath-shiny/tests/test_prey_switching_model_change.py` (new)

**Interfaces:**
- Consumes: `on_model_change` and `consumers_from_model` from Task 2; `switching_store` from Task 3.
- Produces: none for later tasks.

**Behaviour.** In `_retire_run_on_model_change`, inside `with reactive.isolate():`:

1. `rebalance = bool(rebalance_mark.get()) if rebalance_mark is not None else False`, then `rebalance_mark.set(False)`, so the mark is one-shot.
2. Call `on_model_change(switching_store.get(), rebalance=rebalance, consumers=consumers_from_model(model_data.get()), keep_ewe=_matching_ewe_source() is not None)`.
3. Set the returned store, and `notify_info(note, duration=6)` when the note is not None. This notice is independent of `_model_change_notice`.

- [ ] **Step 1: Write the failing tests:**
  - **Signatures:** `ecopath_server` and `ecosim_server` have the keyword parameter `rebalance_mark`, defaulting to None, checked with `inspect.signature`.
  - **AST checks on `ecopath.py`:**
    - the function containing the line `model_data.set(model)` that follows `rpath(` calls `rebalance_mark.set` on the line just before it;
    - no other `model_data.set` call in `ecopath.py` is preceded by `rebalance_mark.set`.
  - **AST checks on `ecosim.py`:**
    - `_retire_run_on_model_change` calls `rebalance_mark.set` with the constant `False`;
    - it calls `on_model_change`;
    - its `notify_info` for the switching note is not inside the `if note:` block of `_model_change_notice`.
  - **`app.py`:** both calls pass `rebalance_mark=`.
  - **The one-shot rule, on the service:**
    1. `on_model_change(rebalance=True)` keeps the overrides.
    2. Then a following `on_model_change(rebalance=False)` clears them.

    Name it `test_a_load_right_after_a_rebalance_clears`.

- [ ] **Step 2: Run.** `pytest packages/pypath-shiny/tests/test_prey_switching_model_change.py -v`. Expected: the signature and AST tests FAIL.

- [ ] **Step 3: Implement** the behaviour above.

- [ ] **Step 4: Run Step 2's command, then the whole app suite.** Expected: all pass.

- [ ] **Step 5: Commit** (`feat(shiny): re-balancing keeps prey-switching settings, a new model clears them`).

---

### Task 6: Wording across the app and the docs

**Files:**
- Modify: `pages/home.py`:
  - the card, around lines 378-394;
  - the feature line, around line 550.
- Modify: `pages/about.py`, around line 91.
- Modify: `pages/forcing_demo.py`, around lines 176-178 and 329.
- Modify: `pages/ecosim.py`, the autofix help label around line 1172.
- Modify: `packages/pypath-shiny/src/pypath_shiny/app.py`, around line 256: change the nav label only. `value="Diet Rewiring"` stays, because it is the navigation id.
- Modify: `docs/parity/continuous_path.md`, "In the Shiny app": replace the **Diet rewiring** bullet.
- Test: `packages/pypath-shiny/tests/test_prey_switching_wording.py` (new); update `test_demo_pages.py` only if Task 7 has not.

**Exact copy:**

- **Home card:**
  - the title `"Prey Switching"`;
  - the paragraph `"Predators turn towards prey that become more abundant: EwE's switching power, set per predator on the Ecosim page."`;
  - the "Explore:" text `"The Prey Switching demo page shows how the switching power shapes a predator's diet."`
- **Home feature line, and `about.py`:** `"Prey switching (EwE's switching power)"`.
- **`forcing_demo.py` markdown,** replacing the clause about prey switching being planned: `"> Use `ForcedBio` on the scenario to force biomass. Diet rewiring is deprecated; prey switching is set per predator (`scenario.params.SwitchPower`, or the Ecosim page)."` The comment at line 329 gets the same sentence as a `#` comment.
- **Autofix label:** `"DD (Prey Switching):"` becomes `"DD (handling time):"`. DD is Rpath's per-link handling term (`core/ecosim.py`: "large DD = no handling time effect"). Also change any other tooltip or label in `pages/` that calls DD prey switching: `grep -rn -i "DD.*switch" packages/pypath-shiny/src` must return nothing afterwards.
- **App nav label:** `"Prey Switching"`.
- **`continuous_path.md` bullet:** `"- **Prey switching.** Set per predator in the sidebar (EwE's SwitchPower, 0 to 2); it applies on every method, and every run shows the switching it used. Diet rewiring is deprecated (it never integrated) and its checkbox is gone for every method."`

- [ ] **Step 1: Write the failing tests:**
  - for each of `home_ui()`, the about page UI, `forcing_demo_ui()` and `ecosim_ui()`, the string `"prey switching is planned"` (case-insensitive) is absent;
  - `"Diet Rewiring"` is absent from the home and about UIs;
  - `"DD (handling time)"` is in `ecosim_ui()`, and `"DD (Prey Switching)"` is not;
  - the home UI contains `"set per predator on the Ecosim page"`.

  Use each page's own UI function, as `test_home_page.py` and `test_about_page.py` do.

- [ ] **Step 2: Run.** `pytest packages/pypath-shiny/tests/test_prey_switching_wording.py -v`. Expected: FAIL.

- [ ] **Step 3: Apply the copy above,** and fix any existing test that pinned the old wording.

- [ ] **Step 4: Run the whole app suite.** Expected: all pass.

- [ ] **Step 5: Commit** (`docs(shiny): describe prey switching, retire diet-rewiring wording`).

---

### Task 7: The demo page shows EwE's switching

**Files:**
- Rewrite: `packages/pypath-shiny/src/pypath_shiny/pages/diet_rewiring_demo.py`. Keep the module name, `diet_rewiring_demo_ui()` and `diet_rewiring_demo_server(input, output, session)` (its exact signature is tested). Drop every use of `DietRewiring`.
- Modify: `packages/pypath-shiny/tests/test_demo_pages.py`: the `build_diet_rewiring_code` test, around lines 61-74.
- Test: `packages/pypath-shiny/tests/test_prey_switching_demo.py` (new)

**Interfaces.** Produces, in the demo module:

```python
def switching_diet(b_rel: Sequence[float], power: float,
                   base_diet: Sequence[float] = (0.5, 0.3, 0.2),
                   vulnerability: float = 2.0) -> np.ndarray
def build_prey_switching_code(power: float) -> str   # replaces build_diet_rewiring_code
```

**`switching_diet`** returns one predator's diet proportions over prey with relative biomasses `b_rel` (B / B0), with the predator at its base biomass.

- **Computation.** It calls `pypath.core.ecosim_deriv.apply_prey_switching` on a small system: prey at base biomass 1, the predator's base flows `base_diet`, and vulnerability `vulnerability` on each link. Consumption is then `Q = Q0' x b_rel` (the closed-form kernel at PDY = 1), normalised to proportions.
- **The page.** Show a plot of diet share against one prey's relative abundance (0.1-10, log axis) for the chosen power and for power 0, with a power slider from 0 to 2 in steps of 0.1. Add a note: "Illustrates EwE's switching formula on one predator with three prey; in a model, set a predator's switching power on the Ecosim page."
- **`build_prey_switching_code`** returns a short Python snippet that sets `scenario.params.SwitchPower` for one named predator and runs `rsim_run`.

- [ ] **Step 1: Write the failing tests.** The oracle is EwE's formula written out, with B0 = 1 and the switching weights in proportion to `base_diet`:

```python
def _oracle(b, s, q0=(0.5, 0.3, 0.2), v=2.0):
    b, q0 = np.asarray(b, float), np.asarray(q0, float)
    r = (q0 * b**s / np.sum(q0 * b**s)) / (q0 / q0.sum())
    q = q0 * b * r * v / (r + v - 1.0)
    return q / q.sum()

def test_base_diet_at_base_abundance():
    for s in (0.0, 1.0, 2.0):
        np.testing.assert_allclose(switching_diet([1, 1, 1], s), [0.5, 0.3, 0.2], rtol=1e-12)

def test_matches_ewe_formula():
    for s in (0.0, 0.5, 1.0, 2.0):
        np.testing.assert_allclose(switching_diet([2.0, 1.0, 0.5], s), _oracle([2.0, 1.0, 0.5], s), rtol=1e-10)

def test_stronger_power_turns_more_to_the_abundant_prey():
    assert switching_diet([2, 1, 1], 2.0)[0] > switching_diet([2, 1, 1], 0.0)[0]

def test_demo_uses_the_core_function_not_diet_rewiring():
    src = Path(demo.__file__).read_text(encoding="utf-8")
    assert "apply_prey_switching" in src and "DietRewiring" not in src

def test_code_snippet_sets_switchpower():
    code = build_prey_switching_code(1.5)
    assert "SwitchPower" in code and "1.5" in code and "rsim_run" in code
```

  If `test_matches_ewe_formula` fails only because `prey_switch_weights` weights the links by `v Q0 / (B0 (v - 1))` and not `Q0`: with one vulnerability for every link and B0 = 1 those are proportional, so the oracle holds. Investigate any other mismatch before changing the oracle.

- [ ] **Step 2: Run.** `pytest packages/pypath-shiny/tests/test_prey_switching_demo.py -v`. Expected: FAIL on the import.

- [ ] **Step 3: Rewrite the module** to the interfaces above, and update `test_demo_pages.py` to call `build_prey_switching_code(1.5)`. Its "prey switching" wording check stays.

- [ ] **Step 4: Run the new tests, then the whole app suite.** Expected: all pass.

- [ ] **Step 5: Commit** (`feat(shiny): the demo page shows EwE's prey switching`).

---

## Final checks (after Task 7)

- `ruff check .` and `ruff format --check .`: clean.
- **App suite:** `pytest packages/pypath-shiny/tests --ignore=packages/pypath-shiny/tests/ui -q`: all pass.
- **Core:** `pytest packages/pypath/tests -q -m "not integration and not slow" --ignore=packages/pypath/tests/scripts`: all pass.
- **Parity capture:** `python packages/pypath/tests/scripts/capture_parity.py`, compared with a capture from `main` at ac6bac1: identical (139 arrays).
- **The app imports:** `python -c "import pypath_shiny.app"`.
- **Files outside `packages/`:** `git diff --name-status origin/main...HEAD` must list only `docs/parity/deviations.md`, `docs/parity/continuous_path.md` and the two `docs/plans/` files.
