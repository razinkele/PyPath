# Prey Switching in the App: Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Let users set EwE's per-predator switching power on the Ecosim page, for models balanced in the app and for imported EwE scenarios and examples. Every run shows the switching it used.

**Architecture:**

- A pure service module, `services/prey_switching.py`, holds every rule: eligibility, overrides per source, applying them, snapshots, the summary line, the range and the model-change rule.
- The Ecosim page renders one numeric input per consumer from a single reactive calc, and applies the overrides at Create Scenario.
- Each run's switching is recorded from the scenario that ran, on the Ecosim, Ecospace and IBM pages.
- A one-shot re-balance mark, shared by `app.py` with the Ecopath and Ecosim pages, tells a re-balance from a model load.
- The core changes only to deprecate `DietRewiring`.

**Tech Stack:** Python ≥ 3.10, Shiny for Python, numpy, pytest.

**Spec:** `docs/plans/2026-11-17-prey-switching-app-design.md` (approved 2026-11-17 after three review rounds). Read it first; this plan argues from it. The plan itself was revised after a review against the code (2026-11-17).

## Global Constraints

- **The range.** A switching power is ≥ 0, and 0 means off. EwE limits it to [0, 2]. That confirms the design's range, so:
  - a typed value outside [0, 2] is refused with a message;
  - an imported value outside it is kept and warned about.

  The source is the EwE textbook (Christensen & Walters), ch. 21 "Holling functional response": "Pj is limited to the range [0,2]". It was checked on the web on 2026-11-17 at https://pressbooks.bccampus.ca/ewemodel/chapter/functional-response/; the local EwE 6.7.144 sources hold no variable metadata for it.
- **Eligibility.**
  - In a scenario's parameters a group is eligible when `PP_type == 0` (groups 1..NUM_GROUPS; slot 0, "Outside", is never one). That includes partial producers, because `rsim_params` casts the type to an integer.
  - For a model with no scenario yet, it is `0 <= type < 1` on the model's `type` / `Type` column.
  - Producers, detritus and fleets are never offered.
  - Group names are always compared stripped.
- **Overrides are keyed by group name**, per source:
  - the balanced route's source key is `"model"`;
  - an EwE source's key is `f"{source.get('kind', 'ewemdb')}|{source['path']}|{scenario_id}"`.
- **Untouched means unchanged.** With no override, `params.SwitchPower` stays the very array the source built. The parity suite and `capture_parity.py` (139 arrays) must stay identical.
- **Do not reuse these constants.** `core/constants.py`'s `*_PREY_SWITCHING_POWER` belongs to autofix's DD. `config.py`'s `PARAM_RANGES.switching_power_*` is the old demo's range, 1-5.
- **The repository's rules:**
  - app tests go in `packages/pypath-shiny/tests/`, and shared fixtures in its `conftest.py`;
  - run `ruff check` and `ruff format --check` (ruff 0.15.x, CI's gate) before each commit;
  - never run `scripts/deploy.sh`;
  - never merge or push `main` without the user's word.
- **The Python environment.** `/opt/micromamba/envs/shiny/bin/python` with `PYTHONPATH=$PWD/packages/pypath/src:$PWD/packages/pypath-shiny/src`, run from the worktree root. Do not install into that environment.
- **Display precision** is 3 decimals: `DISPLAY_DECIMALS = 3`.
- **The deployed Shiny version.** Record it in the final report, from `/opt/micromamba/envs/shiny` on the server (`python -c "import shiny; print(shiny.__version__)"`; 1.9.0 when this was written). Nothing in this plan may need a newer one.

## Review Focus

The five inputs most likely to bite a user that the spec implies but does not test. Each is pinned by a test in the task named:

1. **A numeric box the user clears** (Shiny gives `None`): no override is written, nothing raises, and the box's previous override stays. *(Task 2: `test_set_override_ignores_none`)*
2. **A scenario whose `SwitchPower` is None or shorter than `NUM_GROUPS + 1`** (older readers): it is treated as zeros everywhere, with no IndexError. *(Task 2: `test_summary_and_snapshot_tolerate_missing_or_short_switchpower`)*
3. **An imported value above 2:**
   - it is shown, kept to the bit, and warned about once at Create Scenario;
   - while untouched, its box raises nothing.

   *(Task 2: `test_untouched_imported_value_above_range_is_kept`; Task 3: `test_create_scenario_warns_about_imported_values_above_range`)*
4. **A model with no consumers:** the control says so instead of rendering an empty box. *(Task 3: `test_control_says_so_without_consumers`)*
5. **A large model (50 consumers):** the control renders every input inside a scrolling box. *(Task 3: `test_control_renders_fifty_consumers_in_a_scroll_box`)*

---

### Task 1: Deprecate `DietRewiring` in the core; update the deviations row

**Files:**
- Modify: `packages/pypath/src/pypath/core/forcing.py` (class `DietRewiring`, around line 256; `create_diet_rewiring`, around line 471)
- Modify: `packages/pypath/tests/test_diet_rewiring.py` and `packages/pypath/tests/test_ecosim_advanced.py`: add a warning filter
- Modify: `docs/parity/deviations.md`, row `DR`
- Test: `packages/pypath/tests/test_diet_rewiring_deprecation.py` (new)

**Interfaces:**
- Produces:
  - `DietRewiring(enabled=True, ...)` emits exactly one `DeprecationWarning` whose message starts with `"DietRewiring is deprecated"`, attributed to the caller.
  - `DietRewiring()` and `DietRewiring(enabled=False)` emit none.
  - `create_diet_rewiring(...)` emits exactly one, attributed to its own caller.

- [ ] **Step 1: Write the failing tests** in `test_diet_rewiring_deprecation.py`:

```python
import warnings
from pypath.core.forcing import DietRewiring, create_diet_rewiring

def _deprecations(fn):
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        fn()
    return [x for x in w if issubclass(x.category, DeprecationWarning)]

def test_enabled_diet_rewiring_warns_once_at_the_caller():
    got = _deprecations(lambda: DietRewiring(enabled=True))
    assert len(got) == 1
    assert str(got[0].message).startswith("DietRewiring is deprecated")
    assert "SwitchPower" in str(got[0].message)
    assert got[0].filename == __file__

def test_disabled_diet_rewiring_is_silent():
    assert _deprecations(lambda: DietRewiring()) == []
    assert _deprecations(lambda: DietRewiring(enabled=False)) == []

def test_create_diet_rewiring_warns_once_at_its_caller():
    got = _deprecations(lambda: create_diet_rewiring())
    assert len(got) == 1 and got[0].filename == __file__
```

- [ ] **Step 2: Run them.** `pytest packages/pypath/tests/test_diet_rewiring_deprecation.py -v`. Expected: the enabled and create tests FAIL; the disabled test passes.

- [ ] **Step 3: Implement.**
  - **`DietRewiring.__post_init__(self) -> None`.** It warns only when `self.enabled`, with `stacklevel=3`, which names the caller of `__init__`. The message, exactly:
    `"DietRewiring is deprecated: it never integrated. Use prey switching instead: scenario.params.SwitchPower (EwE's switching power, 0 to 2), or the Ecosim page's Prey Switching control."`
  - **`create_diet_rewiring`.**
    1. Build the instance inside `warnings.catch_warnings()` with `warnings.simplefilter("ignore", DeprecationWarning)`.
    2. Then emit the same message itself with `stacklevel=2`, so its own caller is named. Python's default filters show a DeprecationWarning attributed to user code in `__main__`, not one attributed to `forcing.py`.

- [ ] **Step 4: Silence the expected warnings in the old tests.**
  - In `test_diet_rewiring.py`, add at module level: `pytestmark = pytest.mark.filterwarnings("ignore:DietRewiring is deprecated:DeprecationWarning")`.
  - In `test_ecosim_advanced.py`, which already has `pytestmark = needs_reco` (line 24), make it `pytestmark = [needs_reco, pytest.mark.filterwarnings("ignore:DietRewiring is deprecated:DeprecationWarning")]`, so the REco skip guard stays.

- [ ] **Step 5: Update `deviations.md` row DR.**
  - Keep the facts.
  - Replace the last column, "Next: diet rewiring maps onto prey switching", with: "Prey switching is set per predator on the Ecosim page and through `RsimParams.SwitchPower`; `DietRewiring` is deprecated (warns when enabled)".

- [ ] **Step 6: Run.** `pytest packages/pypath/tests/test_diet_rewiring_deprecation.py packages/pypath/tests/test_diet_rewiring.py packages/pypath/tests/test_ecosim_advanced.py -q`. Expected: all pass (or skip on REco's guard as before), with no `DietRewiring is deprecated` warnings in the summary.

- [ ] **Step 7: Commit** (`feat(core): deprecate DietRewiring in favour of SwitchPower`).

---

### Task 2: The prey-switching service and the shared fixture

**Files:**
- Create: `packages/pypath-shiny/src/pypath_shiny/services/prey_switching.py`
- Modify: `packages/pypath-shiny/tests/conftest.py`: add the `switching_scenario` fixture
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
def consumers_from_params(params) -> list[str]           # PP_type == 0 over groups 1..NUM_GROUPS, stripped names, group order
def consumers_from_model(model) -> list[str]             # 0 <= type < 1; balanced Rpath (Group, type) or RpathParams (model["Group"], model["Type"]); stripped
def switching_by_name(params) -> dict[str, float]         # every group 1..NUM_GROUPS by stripped name (zeros when None or short)
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

Behaviour. All inner mappings are plain dicts, and nothing is mutated:

- **`set_override`.** Returns a new store, in this order:
  1. `value` None, or not convertible to float: the store is returned unchanged.
  2. If `round(value, 3) == round(source_value, 3)`, the group's override is removed and no range check runs. The full-precision source value then applies, which keeps an untouched imported 2.5 legal.
  3. Otherwise `check_power(value)` runs, letting `ValueError` propagate, and the override is stored.
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

- [ ] **Step 1: Add the fixture and write the failing tests.**
  - **The fixture.** In `conftest.py`, add a function-scoped `switching_scenario`, since the tests change its `SwitchPower`. It is built through `create_rpath_params`, `rpath` and `rsim_scenario` from a balanced four-group model, in this order:
    - "Phyto": producer;
    - "Cod": consumer, eating Phyto 0.7 and Detritus 0.3. Two prey are needed: a predator with one prey cannot switch, and the run test in Task 4 must see an effect;
    - "Seals": consumer, eating Cod 1.0;
    - "Detritus".

    Pick B, P/B, Q/B and EE so that `rpath` balances it.
  - **The tests,** in `test_prey_switching_service.py` (`scen` below is the fixture):

```python
def test_consumers_are_pp_type_zero(switching_scenario):
    assert consumers_from_params(switching_scenario.params) == ["Cod", "Seals"]

def test_source_keys():
    assert source_key(None) == "model"
    assert source_key({"path": "/x.eweaccdb"}, 16) == "ewemdb|/x.eweaccdb|16"
    assert source_key({"kind": "ewef_example", "path": "/ex"}, 1) == "ewef_example|/ex|1"

def test_untouched_means_none(switching_scenario):
    assert switch_power_with_overrides(switching_scenario.params, {}) is None

def test_override_written_by_name_on_a_copy(switching_scenario):
    p = switching_scenario.params
    before = p.SwitchPower.copy()
    arr = switch_power_with_overrides(p, {"Cod": 1.0, "Phyto": 1.0})
    i = [str(n).strip() for n in p.spname].index("Cod")
    assert arr[i] == 1.0 and arr.sum() == 1.0          # Phyto (producer) skipped
    assert np.array_equal(p.SwitchPower, before)

def test_set_override_ignores_none():
    s = set_override(SwitchingStore(), "model", "Cod", 1.0, 0.0)
    assert set_override(s, "model", "Cod", None, 0.0) == s

def test_untouched_rounded_value_writes_no_override():
    assert set_override(SwitchingStore(), "k", "Cod", 0.3, 0.30000001) == SwitchingStore()

def test_untouched_imported_value_above_range_is_kept():
    assert set_override(SwitchingStore(), "k", "Cod", 2.5, 2.5) == SwitchingStore()

def test_editing_back_to_the_displayed_source_value_removes_the_override():
    s = set_override(SwitchingStore(), "k", "Cod", 0.5, 0.30000001)
    s = set_override(s, "k", "Cod", 0.3, 0.30000001)
    assert "Cod" not in s.overrides.get("k", {})

def test_out_of_range_is_refused():
    with pytest.raises(ValueError, match="between 0 and 2; 3.0 was not applied"):
        set_override(SwitchingStore(), "model", "Cod", 3.0, 0.0)

def test_effective_values_apply_overrides_on_the_base():
    s = set_override(SwitchingStore(), "model", "Cod", 1.0, 0.0)
    assert effective_values(s, "model", {"Cod": 0.0, "Seals": 0.0}) == {"Cod": 1.0, "Seals": 0.0}

def test_summary_line():
    assert summary_line(None) == "Prey switching: none"
    assert summary_line({"Cod": 1.0, "Seals": 0.5}) == "Prey switching: Cod 1, Seals 0.5"

def test_summary_and_snapshot_tolerate_missing_or_short_switchpower(switching_scenario):
    p = switching_scenario.params
    p.SwitchPower = None
    assert summary(p) == {}
    assert snapshot_source(p) == {"Cod": 0.0, "Seals": 0.0}
    p.SwitchPower = np.zeros(2)
    assert summary(p) == {}

def test_imported_values_above_range_are_reported():
    assert above_range({"Cod": 2.5, "Seals": 1.0}) == {"Cod": 2.5}

def test_pending(switching_scenario):
    p = switching_scenario.params
    s = set_override(SwitchingStore(), "model", "Cod", 1.0, 0.0)
    base = {"Cod": 0.0, "Seals": 0.0}
    assert pending(s, "model", base, p)
    p.SwitchPower = switch_power_with_overrides(p, {"Cod": 1.0})
    assert not pending(s, "model", base, p)

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

def test_a_load_right_after_a_rebalance_clears():
    s = set_override(SwitchingStore(), "model", "Cod", 1.0, 0.0)
    s, _ = on_model_change(s, rebalance=True, consumers=["Cod"], keep_ewe=False)
    s, note = on_model_change(s, rebalance=False, consumers=["Cod"], keep_ewe=False)
    assert s == SwitchingStore() and note is not None
```

- [ ] **Step 2: Run.** `pytest packages/pypath-shiny/tests/test_prey_switching_service.py -v`. Expected: FAIL on the import (module missing).

- [ ] **Step 3: Implement the module** with the signatures and behaviour above.

- [ ] **Step 4: Run Step 2's command.** Expected: all pass.

- [ ] **Step 5: Commit** (`feat(shiny): prey-switching service`).

---

### Task 3: The Ecosim page control, applied at Create Scenario

**Files:**
- Modify: `packages/pypath-shiny/src/pypath_shiny/pages/ecosim.py`. Line numbers are as of ac6bac1 and move as you edit:
  - the sidebar diet note, around lines 476-482: replace it;
  - `ewe_scenario_source_ui`'s two notes, around lines 1088-1104;
  - `help_scenario_setup`'s "Dynamic Diet Rewiring" block, around lines 1330-1338, rendered by the server;
  - `_create_scenario`, around lines 1958-2120;
  - `_retire_run_on_model_change`, around line 1033: only the generation bump and the key reset here; Task 5 adds the store rule.
- Modify: `packages/pypath-shiny/tests/test_ecosim_continuous_ui.py`: replace `test_diet_rewiring_controls_are_gone_and_a_note_replaces_them` and the `DIET_NOTE` constant.
- Test: `packages/pypath-shiny/tests/test_prey_switching_ui.py` (new)

**Interfaces:**
- Consumes: Task 2's names.
- **Produces, at module level in `pages/ecosim.py`:**
  - `switching_inputs(prefix: str, groups: list[str], values: Mapping[str, float], extra_readonly: Mapping[str, float]) -> ui.Tag`. It is the control's body: one `ui.input_numeric(f"{prefix}{i}", name, value=round(values[name], 3), min=0, max=2, step=0.1)` per group, inside `ui.div(..., style="max-height: 320px; overflow-y: auto;")`, then the read-only lines, or the no-consumers text.
- **Produces, in `ecosim_server`:**
  - `switching_store = reactive.Value(SwitchingStore())`;
  - `switching_generation = reactive.Value(0)`;
  - `scenario_switching_key = reactive.Value(None)`, the source key that built the current scenario;
  - `_current_switching_key() -> str`: `"model"` unless the "EwE" source is selected and matching, then `source_key(source, scenario_id)`;
  - the calc `_switching_spec()`;
  - the outputs `switching_control` and `switching_pending_note`.

**Behaviour.** This is the reactivity design; follow it exactly.

- **`_switching_spec()`** is a `@reactive.calc` returning a small frozen record `(prefix, key, groups, base, values, readonly, message)`.
  - **What it depends on.** Only `switching_generation.get()`, `input.scenario_source` (through `_input_or_default`), `input.ewe_scenario_id` and `scenario_switching_key.get()`.
  - **What it reads inside `reactive.isolate()`.** `model_data`, `scenario`, `ewe_source` (through `_current_switching_key`) and `switching_store`.

    So no edit, and no model change before the model-change effect has run, re-renders the control. The model-change effect and Create Scenario bump `switching_generation` after they have updated the store, which fixes the order.
  - **`prefix`** is `f"switch_power_{n}_"`, with n taken from an `itertools.count()` on every recompute. Fresh ids mean a stale browser value can never land on a new group list.
  - **On the balanced route (key `"model"`):**
    - `groups = consumers_from_model(model)`;
    - `base = {g: 0.0}`;
    - with no consumers, `message = "This model has no consumers to switch."`.
  - **On the EwE route:**
    - if `scenario_switching_key.get() != key` or there is no scenario: `groups = []` and `message = "Create the scenario to see and change its switching powers."`;
    - otherwise `groups = consumers_from_params(scenario.params)`, `base = store.source_values[key]`, and `readonly` holds the nonzero entries of `base` that are not consumers.
  - **`values = effective_values(store, key, base)`.** The boxes show the source's values plus the edits, so a re-render never deletes an edit.
- **`switching_control`** renders `ui.p(message)` when there is a message, else `switching_inputs(prefix, groups, values, readonly)`.
- **Collecting edits.** One `@reactive.effect`:
  1. `spec = _switching_spec()`.
  2. Read `input[f"{spec.prefix}{i}"]()` for every group. An input not yet bound raises `SilentException`, and the effect waits.
  3. Then, inside `reactive.isolate()`, fold `set_override` over the groups, starting from `switching_store.get()` and comparing against `spec.base[name]`.
  4. Call `switching_store.set(new)` only when `new != old`; `reactive.Value` compares by identity, so an unconditional set would loop.
  5. On a `ValueError`, call `notify_warning(str(e).replace("Switching power", f"Switching power for {name}"))`. Then reset that box with `ui.update_numeric(id, value=round(effective, 3))`, so the box shows the value that still applies.
- **The pending note.**
  - With a scenario built from the current key (`scenario_switching_key.get() == key`) and `pending(...)` true, it reads `"Changed values apply when you create the scenario again."`.
  - Otherwise, when the key has overrides, it reads `"Values apply when you create the scenario."`.
  - Otherwise it is empty.
- **`_create_scenario`.** Right after `new_scenario` is built, before `_apply_fishing_scenario`:
  1. Compute `key = _current_switching_key()`.
  2. On the EwE route, put `snapshot_source(new_scenario.params)` in a new store as `source_values[key]`. This is taken before any override is applied.
  3. `arr = switch_power_with_overrides(new_scenario.params, store.overrides.get(key, {}))`. If `arr is not None`, set `new_scenario.params.SwitchPower = arr`.
  4. For each `name, v in above_range(snapshot_source(new_scenario.params)).items()`, call `notify_warning(f"{name} has an imported switching power of {v:g}, above EwE's range of 0 to 2; it is kept.")`.
  5. After `scenario.set(new_scenario)`, set the store, `scenario_switching_key.set(key)`, and bump `switching_generation`.

  The autofix call deep-copies the parameters and leaves `SwitchPower` alone (verified in review).
- **`_retire_run_on_model_change`** gains, at its end, `scenario_switching_key.set(None)` and a generation bump. Task 5 adds the store rule just before the bump.
- **Exact copy:**
  - **Sidebar section,** replacing the diet note: `ui.h5("Prey Switching")`, then `ui.p("Predators turn towards prey that become more abundant. 0 = off; higher values switch more sharply (EwE allows 0 to 2). Values apply when you create the scenario.", class_="text-muted small")`, then `ui.output_ui("switching_control")` and `ui.output_ui("switching_pending_note")`.
  - **The database note:** `"Runs the database's Ecopath inputs with the scenario's own vulnerabilities, feeding time, switching, forcing and length, as EwE does. Edits made in this app and the vulnerability slider are not used; prey switching set in the sidebar is."`
  - **The example note:** `"Runs the example's Ecopath inputs with its own EwE-F vulnerabilities, group settings, fishing and run length. Edits made in this app and the vulnerability slider are not used; prey switching set in the sidebar is."`
  - **`help_scenario_setup`:** replace `ui.h6("Dynamic Diet Rewiring")` and its paragraph with `ui.h6("Prey Switching")` and `ui.p("Set a predator's switching power in the sidebar to let it turn towards prey that become more abundant (EwE's SwitchPower, 0 to 2; 0 = off). Values apply when the scenario is created; an imported EwE scenario starts from its own values. Every run shows the switching it used. Diet rewiring is deprecated: it never integrated.")`.

- [ ] **Step 1: Write the failing tests** in `test_prey_switching_ui.py`. For the AST checks, import `_ecosim_tree`, `_find_function` and `_calls` from `tests.test_ecosim_continuous_ui`. If that import fails, move the three helpers to `packages/pypath-shiny/tests/ast_helpers.py` and import them from there in both files. Source checks read `Path(ecosim_page.__file__).read_text()` after joining adjacent string literals: `re.sub(r'"\s+"', "", src)` and whitespace collapsed.
  - **Static checks** on `str(ecosim_ui())`:
    - `"Prey Switching"` and `'id="switching_control"'` are present;
    - `"Diet rewiring is not available in this version"` is absent.
  - **Source checks** on the module text:
    - it contains `"Diet rewiring is deprecated: it never integrated."` and both reworded notes;
    - `re.search(r"(?i)prey switching[^.]{0,60}planned", text)` is None.
  - **Wiring** (AST):
    - `_create_scenario` calls `switch_power_with_overrides`, `snapshot_source` and `above_range`;
    - its `switch_power_with_overrides` call comes before `_apply_fishing_scenario`, by line number;
    - it calls `scenario_switching_key.set` and `switching_generation.set`;
    - `_retire_run_on_model_change` calls `scenario_switching_key.set` and `switching_generation.set`;
    - the effect that calls `set_override` calls `switching_store.set` only inside an `if` whose test compares with `!=`.
  - **`test_create_scenario_warns_about_imported_values_above_range`.** `_create_scenario` contains a `notify_warning` call whose f-string includes `"above EwE's range of 0 to 2; it is kept."`.
  - **Control tests** on `switching_inputs`:
    - **`test_control_renders_fifty_consumers_in_a_scroll_box`.** 50 names with prefix `"switch_power_0_"` give exactly 50 `<input id="switch_power_0_` matches (labels carry `-label` ids, so count the `<input` tag) and `overflow-y: auto`.
    - **`test_control_says_so_without_consumers`.** `switching_inputs("p_", [], {}, {})` contains `"This model has no consumers to switch."`.
    - **`test_rounded_display_value`.** A value of 0.30000001 renders `value="0.3"`.
    - **`test_boxes_show_the_effective_value`.** `switching_inputs("p_", ["Cod"], {"Cod": 1.5}, {})` renders `value="1.5"`, the override, not the source's 0.
    - **`test_readonly_imported_non_consumer`.** `extra_readonly={"Phyto": 1.0}` renders `"Phyto: 1 (imported; this group cannot switch, kept as imported)"`.
  - **`test_example_route_applies_overrides_and_keeps_the_rest`.**
    - Build `example_ecosim_scenario("Central_Baltic")`, which is what `build_ewe_scenario` returns for an example, and take its first consumer.
    - With no override, `switch_power_with_overrides` returns None.
    - With `{first: 1.5}`, the result has 1.5 there and equals the scenario's array everywhere else, bit for bit.
  - **`test_database_route_keeps_imported_values_to_the_bit`.** Reuse the LT2022 database fixture pattern of `test_ewe_scenario_source.py` (lines 143-151); it skips without the database or its driver.
    - Build the scenario 16 with `build_ewe_scenario`; `snapshot_source` gives the Long-tailed duck 1.0.
    - With no override, `switch_power_with_overrides` is None.
    - With `{"Long-tailed duck": 0.5}`, only that entry changes, bit for bit.
  - **Replace the old diet-note test** in `test_ecosim_continuous_ui.py` with `test_diet_rewiring_controls_are_gone_and_prey_switching_replaces_them`:
    - the four diet input ids are absent;
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

- [ ] **Step 1: Write the failing tests.**
  - **Wiring** (AST). Parse each page's own file: `ast.parse(Path(page.__file__).read_text())` for `ecosim`, `ecospace` and `ibm`, with `_find_function` and `_calls` as in Task 3.
    - `_land_result` assigns `switching_used`, from a `summary(...)` call whose argument mentions `outcome.source`;
    - `_run_spatial_simulation` calls `summary`, and sets `_spatial_switching` after `_spatial_results.set(result)`, by line number;
    - every function in `ecospace.py` that calls `_spatial_results.set(None)` also calls `_spatial_switching.set`;
    - `_run_comparison` calls `summary` before `copy.deepcopy`;
    - the functions rendering `spatial_switching_line` and `ibm_switching_line` never call `_sim_scenario`.
  - **The rendered line** (static): `ecosim_ui()` contains `id="switching_line"` after `id="simulation_status"`; the Ecospace and IBM UIs contain `id="spatial_switching_line"` and `id="ibm_switching_line"`.
  - **`test_line_for_every_method`.** Run the `switching_scenario` fixture for 1 year (`years=range(1, 2)`) with `rsim_run` on `"RK4"`, `"AB"` and `"lsoda"`, once with Cod at 1 and once at 0, through `switch_power_with_overrides`.
    - `summary_line(summary(params))` reads `"Prey switching: Cod 1"` for the first run of each pair;
    - the two runs' biomass differs by more than 1e-6 somewhere, so the switching reached the run. Cod eats two prey in the fixture, which is why there is an effect.

- [ ] **Step 2: Run.** `pytest packages/pypath-shiny/tests/test_prey_switching_record.py -v`. Expected: the wiring and static tests FAIL; `test_line_for_every_method` may already pass, since it exercises the core and Task 2.

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
- Modify: `pages/ecosim.py`: `ecosim_server` gains `rebalance_mark: reactive.Value | None = None`; extend `_retire_run_on_model_change`.
- Modify: `packages/pypath-shiny/tests/test_shiny_pages.py`: the parameter counts, `ecopath_server` 4 → 5 (line 113) and `ecosim_server` 7 → 8 (line 143), with the comments naming `rebalance_mark`.
- Test: `packages/pypath-shiny/tests/test_prey_switching_model_change.py` (new)

**Interfaces:**
- Consumes:
  - `on_model_change` and `consumers_from_model` from Task 2;
  - `switching_store`, `switching_generation` and `scenario_switching_key` from Task 3.
- Produces: none for later tasks.

**Behaviour.** In `_retire_run_on_model_change`, before Task 3's generation bump and inside `with reactive.isolate():`:

1. `rebalance = bool(rebalance_mark.get()) if rebalance_mark is not None else False`, then `rebalance_mark.set(False)`, so the mark is one-shot.
2. `new_store, switching_note = on_model_change(switching_store.get(), rebalance=rebalance, consumers=consumers_from_model(model_data.get()), keep_ewe=_matching_ewe_source() is not None)`. The name `switching_note` avoids clashing with the handler's existing `note`.
3. Set the store, and `notify_info(switching_note, duration=6)` when the note is not None. This notice is independent of `_model_change_notice`.

- [ ] **Step 1: Write the failing tests:**
  - **Signatures:** `ecopath_server` and `ecosim_server` have the keyword parameter `rebalance_mark`, defaulting to None, checked with `inspect.signature`.
  - **AST checks on `ecopath.py`:**
    - the only `model_data.set(...)` preceded, within the same block, by `rebalance_mark.set(True)` is the one after `rpath(`;
    - no other `model_data.set` call in `ecopath.py` is preceded by it.
  - **AST checks on `ecosim.py`:**
    - `_retire_run_on_model_change` calls `rebalance_mark.set` with the constant `False`;
    - it calls `on_model_change`;
    - it calls `notify_info` with `switching_note`, outside the `if note:` block.
  - **`app.py`:** both server calls pass `rebalance_mark=`.

- [ ] **Step 2: Run.** `pytest packages/pypath-shiny/tests/test_prey_switching_model_change.py packages/pypath-shiny/tests/test_shiny_pages.py -v`. Expected: the new tests FAIL, and `test_shiny_pages.py` passes until Step 3 changes the signatures. Update its counts in Step 3.

- [ ] **Step 3: Implement** the behaviour above, and update `test_shiny_pages.py`.

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
- Modify: `pages/ecosim.py`: the autofix help modal in `_toggle_autofix_help`, around lines 1146-1180.
- Modify: `packages/pypath-shiny/src/pypath_shiny/app.py`, around line 256: change the nav label only. `value="Diet Rewiring"` stays, because it is the navigation id.
- Modify: `docs/parity/continuous_path.md`, "In the Shiny app": replace the **Diet rewiring** bullet.
- Test: `packages/pypath-shiny/tests/test_prey_switching_wording.py` (new)

**Exact copy:**

- **Home card:**
  - the title `"Prey Switching"`;
  - the paragraph `"Predators turn towards prey that become more abundant: EwE's switching power, set per predator on the Ecosim page."`;
  - the "Explore:" text `"The Prey Switching demo page shows how the switching power shapes a predator's diet."`
- **Home feature line, and `about.py`:** `"Prey switching (EwE's switching power)"`.
- **`forcing_demo.py` markdown,** replacing the clause about prey switching being planned: `"> Use `ForcedBio` on the scenario to force biomass. Diet rewiring is deprecated; prey switching is set per predator (`scenario.params.SwitchPower`, or the Ecosim page)."` The comment at line 329 gets the same sentence as a `#` comment.
- **Autofix modal:** `"DD (Prey Switching):"` becomes `"DD (handling time):"`. DD is Rpath's per-link handling term (`core/ecosim.py`: "large DD = no handling time effect"). Also change any other text in `packages/pypath-shiny/src` that calls DD prey switching: `grep -rn -i "DD.*switch" packages/pypath-shiny/src` must return nothing afterwards.
- **App nav label:** `"Prey Switching"`.
- **`continuous_path.md` bullet:** `"- **Prey switching.** Set per predator in the sidebar (EwE's SwitchPower, 0 to 2); it applies on every method, and every run shows the switching it used. Diet rewiring is deprecated (it never integrated) and its checkbox is gone for every method."`

- [ ] **Step 1: Write the failing tests.** Check each page's module source, joined and collapsed as in Task 3, since modal and markdown text does not appear in the UI functions' output.
  - **Absent from `home.py`, `about.py`, `forcing_demo.py` and `ecosim.py`:** `re.search(r"(?i)prey switching[^.]{0,60}planned", text)`, and `"Diet rewiring (prey switching)"` (case-insensitive).
  - **Present:**
    - `"Prey switching (EwE's switching power)"` in `home.py` and `about.py`;
    - `"set per predator on the Ecosim page"` in `home.py`;
    - `"DD (handling time)"` in `ecosim.py`, and `"DD (Prey Switching)"` is absent;
    - `forcing_demo.py` contains `"Diet rewiring is deprecated; prey switching is set per predator"`.
  - **Not vacuous.** Confirm each absence check FAILS on the unmodified files before Step 3.

- [ ] **Step 2: Run.** `pytest packages/pypath-shiny/tests/test_prey_switching_wording.py -v`. Expected: FAIL.

- [ ] **Step 3: Apply the copy above,** and fix any existing test that pinned the old wording.

- [ ] **Step 4: Run the whole app suite.** Expected: all pass.

- [ ] **Step 5: Commit** (`docs(shiny): describe prey switching, retire diet-rewiring wording`).

---

### Task 7: The demo page shows EwE's switching

**Files:**
- Rewrite: `packages/pypath-shiny/src/pypath_shiny/pages/diet_rewiring_demo.py`. Keep the module name, `diet_rewiring_demo_ui()` and `diet_rewiring_demo_server(input, output, session)` (its exact signature is tested). Drop every use of `DietRewiring`.
- Modify: `packages/pypath-shiny/tests/test_demo_pages.py`, around lines 61-74:
  - call `build_prey_switching_code(1.5)`;
  - keep `"NotImplementedError"` asserted only for the forcing page and code;
  - assert `"SwitchPower"` for the demo page and code.
- Modify: `packages/pypath-shiny/tests/test_review_fixes.py`, line 18 and its uses: import and call `build_prey_switching_code`.
- Test: `packages/pypath-shiny/tests/test_prey_switching_demo.py` (new)

**Interfaces.** Produces, in the demo module:

```python
def switching_diet(b_rel: Sequence[float], power: float,
                   base_diet: Sequence[float] = (0.5, 0.3, 0.2),
                   vulnerability: float = 2.0) -> np.ndarray
def build_prey_switching_code(power: float) -> str   # replaces build_diet_rewiring_code
```

**`switching_diet`** returns one predator's diet proportions over prey with relative biomasses `b_rel` (B / B0), with the predator at its base biomass.

- **Computation.** It calls `pypath.core.ecosim_deriv.apply_prey_switching` on a small system:
  - prey in rows 1-3: row 0, "Outside", gets no weight in `prey_switch_weights`;
  - every prey's base biomass is 1;
  - the predator has a positive base biomass, 1. With 0 the weights deviate (verified in review);
  - the predator's base flows are `base_diet`, and every link's vulnerability is `vulnerability`.

  Consumption is then `Q = Q0' x b_rel` (the closed-form kernel at PDY = 1), normalised to proportions.
- **The page.** Show a plot of diet share against one prey's relative abundance (0.1-10, log axis) for the chosen power and for power 0, with a power slider from 0 to 2 in steps of 0.1. Add a note: "Illustrates EwE's switching formula on one predator with three prey; in a model, set a predator's switching power on the Ecosim page."
- **`build_prey_switching_code`** returns a short Python snippet that sets `scenario.params.SwitchPower` for one named predator and runs `rsim_run`.

- [ ] **Step 1: Write the failing tests.** The oracle is EwE's formula written out, with B0 = 1 and weights in proportion to `base_diet`. Review verified that it matches `apply_prey_switching` to 2e-16:

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

- [ ] **Step 2: Run.** `pytest packages/pypath-shiny/tests/test_prey_switching_demo.py -v`. Expected: FAIL on the import.

- [ ] **Step 3: Rewrite the module** to the interfaces above, and update `test_demo_pages.py` and `test_review_fixes.py` as listed under Files.

- [ ] **Step 4: Run the new tests, then the whole app suite.** Expected: all pass.

- [ ] **Step 5: Commit** (`feat(shiny): the demo page shows EwE's prey switching`).

---

## Final checks (after Task 7)

- `ruff check .` and `ruff format --check .`: clean.
- **App suite:** `pytest packages/pypath-shiny/tests --ignore=packages/pypath-shiny/tests/ui -q`: all pass.
- **Core:** `pytest packages/pypath/tests -q -m "not integration and not slow" --ignore=packages/pypath/tests/scripts`: all pass.
- **Parity capture:** `python packages/pypath/tests/scripts/capture_parity.py`, compared with a capture from `main` at ac6bac1: identical (139 arrays).
- **The app imports:** `python -c "import pypath_shiny.app"`.
- **Record the deployed Shiny version** (see Global Constraints).
- **Files outside `packages/`:** `git diff --name-status origin/main...HEAD` must list only `docs/parity/deviations.md`, `docs/parity/continuous_path.md` and the two `docs/plans/` files.
- **On the PR,** the Shiny smoke check must be green before any merge.
