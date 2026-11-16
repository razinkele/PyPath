# Continuous Pathway, Phase 3b (Shiny) Implementation Plan

> **As built (2026-11-16).** The rest of this plan is a dated record and is not rewritten. Four rulings in the execution ledger (kept outside the repository) changed what was built:
> - **S3.** The diet-rewiring block is hidden for every method, not only the continuous ones, and a note replaces it. Diet rewiring was never implemented (`rsim_run_advanced` integrates nothing), so the server builds no `DietRewiring`; prey switching is planned instead.
> - **S5.** Runs go to a bounded process-wide executor (`MAX_CONCURRENT_RUNS` = 2), not `asyncio.to_thread`. A per-session `RunSlot` refuses a new run while a cancelled one still computes.
> - **S6.** Any model load (an import, or a re-balance on the Ecopath page) clears the Ecosim results, the app-wide results and the scenario. When it cleared something it says "Model changed: create the scenario again before running."
> - **S8.** The sidebar note and the help say continuous runs cost "about 5-25 × RK4", as measured (4.4-23.9 ×), not the 2-25 × this plan gives.

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** On the Ecosim page, offer the continuous methods beside the EwE parity methods, with a tolerance preset, opt-in options, a diagnostics panel built from `solver_info`, and runs that no longer block the server. Invalid combinations notify and never raise. RK4 and AB runs from the page are unchanged.

**Architecture:** Pure logic goes in two service modules, as `services/ecosim_controls.py` does: `services/ecosim_solver.py` (choices, presets, request builder, run dispatch) and `services/solver_diagnostics.py` (summary, badges, month tables). `pages/ecosim.py` gains static UI (`panel_conditional`, `accordion`) and thin server code. Task 5 (isolated, last) moves the run into a `reactive.extended_task` over `asyncio.to_thread`, so the page's existing `TaskState` machinery (`run_id`, `apply_if_latest`, cancel) finally works as designed. `packages/pypath` is not touched: 3b uses only the public API (`CONTINUOUS_METHODS`, `continuous_options`, `default_atol`, `KERNEL_CHECK_TOL`, `QS_RATE_DEFAULT`, `solver_options`, `solver_info`).

**Tech Stack:** Python ≥ 3.10, Shiny for Python (1.9.0 in the shared env, 1.8.0 in the deploy venv; both have `reactive.extended_task` and `ui.accordion`), pandas, pytest. No new dependencies.

**Spec:** `docs/plans/2026-11-14-continuous-pathway-design.md` §6 ("Shiny"), §9 (phase 3 row), §11 items 1-11; the sketch and the binding answers in `docs/plans/2026-11-15-continuous-pathway-phase3.md` (answer 2: LSODA stays the default, BDF is documented for stiff models, no `method="auto"`); `docs/parity/continuous_path.md`.

## Measurements behind this plan

Spikes: `/tmp/claude-1000/-home-razinka-PyPath/3de35854-f464-4f7d-a354-0e5f9d63fe72/scratchpad/p3b/f1-f5*.py`, outputs in the `*.out` files beside them; laguna, SciPy 1.18.1, main 95dec98, 2026-11-15. Ten-year CPU (`process_time`) per simulated year, on the page's protocol (warm-up, interleaved RK4, Fast and Default, minimum of 3; Reference once, as the truth). Error: `|B / B_ref - 1|`, months 1-120, living groups.

| Model, method | Preset | CPU s/yr | × RK4 | RHS evals/mo | Error vs Reference: median, max |
|---|---|---|---|---|---|
| REco with stanzas, effort 2, LSODA | Fast | 0.449 | 4.4 | 37.6 | 6.7e-8, 4.3e-5 (Phytoplankton) |
| | Default | 0.488 | 4.8 | 41.5 | 2.3e-8, 1.9e-7 |
| | Reference | 0.905 | 8.9 | 78.0 | |
| Black Sea s1, LSODA | Fast | 0.724 | 15.1 | 60.2 | 5.0e-6, 8.7e-5 (Phytoplankton) |
| | Default | 1.141 | 23.9 | 95.4 | 3.7e-8, 1.2e-6 |
| | Reference | 2.322 | 48.5 | 193.8 | |
| LT2022 s16, BDF (one sample, load 3.6) | Fast | 0.555 | 5.0 | 36.5 | 8.2e-5, 1.2e-3 (Mobile crustaceans) |
| | Default | 1.277 | 11.4 | 86.4 | 3.2e-7, 7.3e-6 |
| | Reference | 4.426 | 39.5 | 307.9 | |

- **Fast's saving depends on the model:** 8 % on REco, 37 % on the Black Sea, 57 % on LT2022 BDF, and 1-4 % on the three shipped examples (f3: 10-year wall time, one sample each; Central Baltic 3.33 → 3.19 s), where the per-month floor dominates (restart, start check, 6 quadrature nodes, Jacobian columns). No preset clipped, degraded or failed a month; `kernel_check` was 0.0 throughout. The Black Sea's budget residual stays at 3.3e-6 from rtol 1e-6 to 1e-10: not a solver error, so no badge.
- **Run length, not 11 s, is the issue.** A scenario built from the model runs the sidebar's years: default 50, maximum 500. (Home examples and EwE-database scenarios run their own length; the three examples have 10 years.)
  - Extrapolated from the ten-year runs (× 5), 50 years take: Black Sea LSODA 57 s (Reference 116 s); LT2022 BDF 64 s, LSODA 95 s.
  - Measured once (f5) on Lithuanian Coast, under load average 72: RK4 15 s, LSODA 58 s, BDF 26 s.
- **Today's run blocks every session.** `_run_simulation` runs synchronously in a `reactive.effect`, so nothing reaches the browser until it returns (the 20/50/85 % updates never render). Shiny Server Open Source runs one process per app (`/etc/shiny-server/shiny-server.conf`: `location /pypath`, no scheduler directive), so a one-minute run stalls every user.
- **Threads are safe (f4):** four concurrent `rsim_run` calls in threads (REco LSODA and BDF, Central Baltic Fast and RK4) were bit-identical to serial runs; under `asyncio.to_thread` a 10 ms event-loop ticker lagged 0.3 ms median, 1.4 ms p99, and the run took no longer (1.59 against 1.58 s).
- **`rsim_run_advanced` integrates nothing (f2b):** for `"RK4"`, `"lsoda"` and `"nonsense"` alike it returns the start biomass in every month, and the page reports "Diet rewiring was applied." Plain RK4 moves the same scenario 16 % in two years.

## Global Constraints

- **Files.** Only `packages/pypath-shiny/` (src, tests, README) and one new section of `docs/parity/continuous_path.md`. Nothing under `packages/pypath/`.
- **Method values:** `"RK4"`, `"AB"` and exactly `pypath.core.ecosim.CONTINUOUS_METHODS` (lower case), shown as LSODA, BDF, DOP853, Radau. The page's default stays `"RK4"`.
- **Presets** pass only `rtol`: Fast `{"rtol": 1e-4}`, Default `{}`, Reference `{"rtol": 1e-10}`. No `atol`: the core's default already follows `rtol` (`default_atol(b0, rtol)`). Passing `default_atol(b0)` would cap Reference at rtol 1e-6's absolute tolerance.
- **Options**, added only when they differ from the default:
  - log states: `{"positivity": "log"}`;
  - interpolation: `{"forcing_interpolation": k}` for k in `("linear", "smooth")`;
  - quasi-steady: `{"fast_groups": "quasi-steady", "qs_rate": float(r)}`, with r finite and > 0 and a default of `QS_RATE_DEFAULT` (365). `qs_rate` is never sent without quasi-steady, because the core warns that it is unused.
- **Parity methods get `solver_options=None`.** The continuous inputs are neither read into the request nor validated for them, so a stale `qs_rate` can never block an RK4 run.
- **Diet rewiring** reaches `rsim_run_advanced` only for RK4 and AB. The guard is the method, not the checkbox.
- **Badges** come from `solver_info`, not logs. Thresholds: `kernel_check > KERNEL_CHECK_TOL` (1e-9, imported; NaN = not checked, no badge); `qs_separation < 10.0` (`QS_SEPARATION_MIN`, mirroring the core's literal at `ecosim.py:3270`; None and inf, no badge); `degraded` True; `status != 0`; `clips > 0`. No badge on budget residuals.
- **Static inputs.** Every new input is built in `ecosim_ui()` (inside `panel_conditional` or `accordion`), never in `render.ui`, and is read with `read_optional_input`.
- **No exception leaves an effect.** Problems go through `notify_error` and `notify_warning` (`pypath_shiny.services.notifications`).
- **Style.** CI's format gate `ruff format --check packages/` (ruff 0.15.x; `~/.local/bin/ruff` is 0.15.21) and `ruff check packages/` (E, F, I, W); a module `logger`; no bare `except Exception: pass`.
- **Tests.** The Shiny CI command has no marker filter, so every new test runs in CI: no `Data/` files, no `slow` marker, each test under 3 s on laguna. Models: the conftest 3-group model and `pypath.examples` (Central Baltic: stanzas, 17 feeding-time groups). REco's builders live in core's `tests` package and are not used. On the 3-group model a 2-year scenario takes 0.04-1.1 s at Default on every method, and 1.7-3.4 s at Reference (f5, under load average 72), so the one Reference test uses LSODA.
- **Before every commit**, with `/opt/micromamba/envs/shiny/bin/python -m …` and `PYTHONPATH=$PWD/packages/pypath/src:$PWD/packages/pypath-shiny/src:$PWD/packages/pypath`: `ruff check packages/`; `ruff format --check packages/`; `pytest packages/pypath-shiny/tests/ -q --ignore=packages/pypath-shiny/tests/ui` (baseline 873 passed, 30 s); `python -c "import pypath_shiny.app"` (deploy.yml's pre-deploy import check).
- **The smoke workflow** (`ci-shiny-smoke.yml`: shinyswatch tests, the whole Shiny suite, a server start, the Pre-Balance Playwright smoke) must stay green; 3b adds nothing that needs Playwright in CI.

## Review Focus

1. **A stale diet-rewiring tick with a continuous method** sends the run through `rsim_run_advanced`, which returns the start state, and reports success on a flat trajectory. *Task 3,* `test_diet_rewiring_with_a_continuous_method_notifies`, and *Task 1,* `test_run_dispatch` (no `rsim_run_advanced` call).
2. **The UI emits a combination the core rejects or warns about** (an unused `qs_rate`, options on a parity method). *Task 1,* `test_every_request_the_ui_can_emit_is_valid`: all 216 requests (6 methods × 3 presets × 2 log × 3 interpolations × 2 quasi-steady).
3. **Reference silently capped by an explicit `atol`.** *Task 3,* `test_reference_atol_follows_rtol`.
4. **Diagnostics break on edge values:** `solver_info` None (RK4), NaN `kernel_check`, None budgets, `qs_separation` None or inf, `qs_groups == []` under quasi-steady, 600 months. *Task 4,* `test_badge_edges` and `test_month_table_with_none_values`.
5. **Async: a stale result lands on newer state, or the worker touches the session.** A run outlives a cancel, a newer run, Create Scenario or a model load, then writes its output onto the new scenario. *Task 5,* `test_result_is_current` (superseded `run_id`, replaced scenario), `test_new_scenario_retires_the_run`, `test_worker_is_session_free` and `test_threaded_runs_match_serial`.

---

### Task 1: Solver request service

**Files:** Create `packages/pypath-shiny/src/pypath_shiny/services/ecosim_solver.py`. Test: create `packages/pypath-shiny/tests/test_ecosim_solver_service.py`.

**Interfaces:** Produces:
  - `PARITY_METHODS = ("RK4", "AB")`.
  - `METHOD_CHOICES: dict[str, dict[str, str]]`, the optgroups:
    - `"EwE parity"`: `{"RK4": "Runge-Kutta 4", "AB": "Adams-Bashforth"}`;
    - `"Continuous (not EwE parity)"`: `{"lsoda": "LSODA (automatic stiff/non-stiff)", "bdf": "BDF (recommended for stiff models)", "dop853": "DOP853 (non-stiff, high order)", "radau": "Radau (stiff, implicit)"}`.
  - `TOLERANCE_PRESETS` (the Global Constraints' values) and `PRESET_CHOICES` (`"Fast (rtol 1e-4)"`, `"Default (rtol 1e-6)"`, `"Reference (rtol 1e-10)"`).
  - `INTERPOLATION_CHOICES = {"constant": "Held each month (as EwE)", "linear": "Linear (keeps monthly means)", "smooth": "Smooth (keeps monthly means)"}`.
  - `CONTINUOUS_JS = json.dumps(list(CONTINUOUS_METHODS)) + ".includes(input.integration_method)"`.
  - `is_continuous(method: str) -> bool`.
  - `@dataclass(frozen=True) SolverRequest(method: str, solver_options: dict | None, notes: tuple[str, ...] = ())`.
  - `build_solver_request(method, *, preset="default", log_states=False, forcing_interpolation="constant", quasi_steady=False, qs_rate=QS_RATE_DEFAULT, diet_rewiring=False) -> SolverRequest`. It raises `ValueError` with user-facing text:
    - `"Diet rewiring runs on RK4 and AB only. Turn it off, or choose an EwE parity method."`;
    - `"Quasi-steady rate must be a number above 0 (per year)."` (for None, NaN, ≤ 0, bool or text);
    - `"Unknown integration method {m!r}."`;
    - `"Unknown tolerance preset {p!r}."`.

    For a non-DOP853 method under quasi-steady it adds the note `"Quasi-steady groups suit DOP853; on LSODA, BDF and Radau they usually cost more than integrating the fast groups."` (the page's Options measurement).
  - `run_ecosim(scenario, request, *, diet_rewiring: DietRewiring | None = None) -> RsimOutput`, a pure dispatch with no Shiny import:
    - parity with diet rewiring: `rsim_run_advanced(scenario, state_forcing=None, diet_rewiring=…, method=…)`;
    - parity without: `rsim_run(scenario, method=…)`;
    - continuous: `rsim_run(scenario, method=…, solver_options=request.solver_options)`;
    - continuous with diet rewiring: `ValueError` (defence in depth).

- [ ] **Step 1: Write the failing tests.**
  - `test_choices_cover_the_core_methods`: the flattened `METHOD_CHOICES` keys equal `set(PARITY_METHODS) | set(CONTINUOUS_METHODS)`; the group order is parity, then continuous; LSODA comes first in its group.
  - `test_every_request_the_ui_can_emit_is_valid` (Review Focus 2): the 216-request product. Continuous requests pass `continuous_options(m, opts, 3)`, caplog holds no WARNING from `pypath.core`, and `qs_rate` appears only with `fast_groups`. Parity requests have `solver_options is None`.
  - `test_presets_pass_only_rtol`: the three dicts exactly; no request ever has `"atol"`.
  - `test_parity_ignores_stale_continuous_inputs`: `build_solver_request("RK4", quasi_steady=True, qs_rate=None, preset="nonsense")` returns `SolverRequest("RK4", None)`.
  - `test_invalid_inputs_raise_user_text`: each `ValueError` message above, exactly; `qs_rate` in `(None, 0, -1, float("nan"), True, "x")`.
  - `test_quasi_steady_note`: a note for `lsoda`, `bdf` and `radau`, none for `dop853`.
  - `test_run_dispatch` (monkeypatched spies on the module's `rsim_run` and `rsim_run_advanced`): each branch gets exactly the arguments above; continuous with diet rewiring raises before either spy is called.
  - `test_short_runs_on_every_method`: the 3-group model as a 2-year scenario (`rsim_scenario`'s minimum; `run_ecosim` runs the scenario's whole length, as the page does), parametrised over the six methods. `out_Biomass` is finite; each continuous method has `solver_info["method"] == m`; RK4's `solver_info` is None.
- [ ] **Step 2: Run, expect FAIL** (`ModuleNotFoundError: pypath_shiny.services.ecosim_solver`).
- [ ] **Step 3: Implement.** Use NumPy-style docstrings; the module docstring states the package-boundary rule.
- [ ] **Step 4: Run, expect PASS**, then the commit checks.
- [ ] **Step 5: Commit** `feat(shiny): solver request service for the continuous Ecosim methods`.

### Task 2: Sidebar controls

**Files:** Modify `pages/ecosim.py` (`ecosim_ui`, at `integration_method`, line 235, and the diet-rewiring block, 287-345). Test: create `packages/pypath-shiny/tests/test_ecosim_continuous_ui.py`.

**Interfaces:**
- Consumes Task 1's constants.
- **The method select:** `ui.input_select("integration_method", …, choices=METHOD_CHOICES, selected="RK4")`. Its tooltip says the parity methods reproduce EwE (RK4) and Rpath (AB), the continuous ones solve the equations to a tolerance and differ from EwE by design, and BDF suits stiff models.
- **Directly below it,** `ui.panel_conditional(CONTINUOUS_JS, …)` holds:
  - **Numerics (same model):** `ui.input_radio_buttons("solver_preset", "Tolerance", PRESET_CHOICES, selected="default")`; `ui.input_checkbox("solver_log_states", "Log-transformed biomass", value=False)`; a note: "Fast saves time only where the solver takes many steps a month (Black Sea 37 %, REco 8 %); Reference costs 2-3.5 × Default; continuous runs cost about 2-25 × RK4."
  - **Planner's call on log states:** here, with the tolerance, not under "changes the model". They solve the same equations (the page's Options table; REco with stanzas agrees with clipping to 1.3e-6), so listing them as a model change would tell users their results are a different model. The tooltip gives the use (runs that drive a group towards zero) and the cost (+40 % where groups sweep orders of magnitude).
  - **`ui.accordion(ui.accordion_panel("Advanced: changes the model", …), id="solver_advanced", open=False)`:** `ui.input_select("solver_forcing_interpolation", "Forcing within the month", INTERPOLATION_CHOICES, selected="constant")` with the note "Interpolates forcing that varies month to month (an imported EwE scenario's forcing functions); effort stays monthly; the page's own forcing card is constant in time"; `ui.input_checkbox("solver_quasi_steady", "Quasi-steady fast groups", value=False)`; and `ui.panel_conditional("input.solver_quasi_steady", ui.input_numeric("solver_qs_rate", "Quasi-steady above (per year)", value=QS_RATE_DEFAULT, min=1, step=5))`.
- **The diet-rewiring block** is wrapped in `ui.panel_conditional("!" + CONTINUOUS_JS, …)`. A sibling `ui.panel_conditional(CONTINUOUS_JS, ui.p("Diet rewiring runs on the EwE parity methods (RK4, AB) only.", class_="text-muted small"))` shows instead.
- **The `fast_group_rule` select is renamed** (user's answer 4): its label becomes "Equation rule (EwE6 / Rpath)" (the input id `fast_group_rule` and its values stay, so saved state and the core call are unchanged); choice labels "EwE6 (fast groups at equilibrium under RK4/AB)" and "Rpath (fast groups integrated)". Add a test that the label and the id are as stated.
- **The `fast_group_rule` tooltip** gains: "Under a continuous method this picks the equations only (handling time, detritus fate): every group is integrated, or solved quasi-steady if you ask for it below."

- [ ] **Step 1: Write the failing tests** (on `str(ecosim_ui())`, as `test_demo_pages.py` does):
  - `test_method_select_has_two_optgroups`: `<optgroup label="EwE parity">` holds RK4 and AB; `<optgroup label="Continuous (not EwE parity)">` holds the four values in order; RK4 is `selected`.
  - `test_continuous_inputs_are_static`: the ids `solver_preset`, `solver_log_states`, `solver_forcing_interpolation`, `solver_quasi_steady` and `solver_qs_rate` appear in the HTML. An AST check finds none of them inside a `render.ui` function.
  - `test_conditions_name_every_continuous_method`: the `data-display-if` of the solver panel and of the diet-rewiring panels contains each of `CONTINUOUS_METHODS`; the diet-rewiring checkbox sits inside the negated one.
  - `test_advanced_accordion_starts_closed_and_holds_only_model_options`: the accordion's HTML holds `solver_forcing_interpolation` and `solver_quasi_steady`, but not `solver_log_states` or `solver_preset` (whose `default` is checked); the button is `collapsed`.
- [ ] **Step 2: Run, expect FAIL.**
- [ ] **Step 3: Implement.**
- [ ] **Step 4: Run, expect PASS**, then the commit checks (they include the layout, render and hidden-input contract tests).
- [ ] **Step 5: Commit** `feat(shiny): continuous methods, tolerance preset and model options on the Ecosim page`.

### Task 3: Server wiring (synchronous)

**Files:** Modify `pages/ecosim.py` (module-level helper; `ecosim_server`: `_run_simulation`, lines 1888-2036, and a new method-change effect). Test: `test_ecosim_continuous_ui.py`.

**Interfaces:**
- Produces a module-level `_solver_request_or_notify(method, *, diet_rewiring, preset, log_states, forcing_interpolation, quasi_steady, qs_rate) -> SolverRequest | None`, in the pattern of `_require_balanced_model_or_notify`:
  - on `ValueError` it calls `notify_warning(str(e), duration=8)` and returns None;
  - it shows each note with `notify_info(note, duration=8)`.
- **`_run_simulation`** reads the continuous inputs (`_input_or_default`) only when `is_continuous(method)`; builds the request and returns before `_apply_biomass_forcing` on None; builds `DietRewiring` only for parity methods; calls `run_ecosim`; and sets `output.run_request = request` (for the details table). A continuous success says "Simulation completed with LSODA (continuous, not EwE parity)." A continuous failure's notification appends `continuous_failure_hint(method)` (in the service): "Try BDF for stiff models, or the Reference tolerance."
- **A new `@reactive.effect @reactive.event(input.integration_method)`:** when a continuous method is chosen and `enable_diet_rewiring` is ticked, it calls `ui.update_checkbox("enable_diet_rewiring", value=False)` and `notify_info("Diet rewiring turned off: it runs on RK4 and AB only.")`.

- [ ] **Step 1: Write the failing tests:**
  - `test_diet_rewiring_with_a_continuous_method_notifies` (Review Focus 1): with monkeypatched `notify_warning` and `notify_error`, `_solver_request_or_notify("lsoda", diet_rewiring=True, …)` returns None, and the warning text names "RK4 and AB".
  - `test_bad_qs_rate_notifies_instead_of_raising`: `qs_rate=None` with quasi-steady on `"dop853"` returns None with one warning; `notify_error` is not called.
  - `test_reference_atol_follows_rtol` (Review Focus 3): `run_ecosim` on the 3-group scenario with the Reference request gives `solver_info["atol"]` equal to `default_atol(b0, 1e-10)` (rel 1e-12), with `b0` from `B_BaseRef[1 : NUM_LIVING + NUM_DEAD + 1]`.
  - `test_run_simulation_goes_through_the_service`: an AST check of `ecosim_server` finds `_run_simulation` calling `run_ecosim(` and `_solver_request_or_notify(`, with no direct `rsim_run(` or `rsim_run_advanced(`.
  - `test_method_change_unticks_diet_rewiring`: an AST check finds an effect on `input.integration_method` that calls `ui.update_checkbox("enable_diet_rewiring"`.
- [ ] **Step 2: Run, expect FAIL.**
- [ ] **Step 3: Implement.** Keep the crash-message code unchanged; Task 5 moves it.
- [ ] **Step 4: Run, expect PASS**, then the commit checks.
- [ ] **Step 5: Commit** `feat(shiny): run the continuous methods from the Ecosim page; invalid combinations notify`.

### Task 4: Diagnostics panel

**Files:** Create `services/solver_diagnostics.py`. Modify `pages/ecosim.py`: `ui.output_ui("solver_diagnostics")` on the "Simulation Progress" tab after `progress_display`; rows "Method" (and, when continuous, "Tolerance" and "Model options") in `progress_display`; a "Solver diagnostics" section in `help_progress` explaining each badge. Test: create `tests/test_solver_diagnostics.py`.

**Interfaces:**
- Consumes `KERNEL_CHECK_TOL` and `solver_info`'s documented keys (`api/core.md`, `continuous_path.md`).
- Produces:
  - `QS_SEPARATION_MIN = 10.0`.
  - `@dataclass(frozen=True) Badge(level: Literal["danger", "warning", "info", "secondary"], text: str)`.
  - `solver_badges(info: dict | None, group_names: Sequence[str] | None = None) -> list[Badge]` (`[]` for None), in order: secondary "Continuous path (not EwE parity): LSODA"; danger, status ≠ 0; warning, degraded months (up to 5 listed, then "+N more"); warning, `kernel_check` above 1e-9 (count and maximum); warning, "qs_separation 2.1 < 10: raise the quasi-steady rate or integrate the fast groups"; info, "No group qualified as quasi-steady at 365 per year; every group was integrated"; info, "Changes the model: linear forcing; quasi-steady Detritus"; info, clipped evaluations ("log states avoid them").
  - `solver_summary(info, group_names=None) -> list[tuple[str, str]]`: method; `rtol` and "atol: default, follows rtol"; right-hand-side evaluations, total and per month (`rhs_evals` is a run total); the options, with `qs_groups` named and `qs_separation`; corrections applied; EwE switches ignored; feeding-time states; months.
  - `solver_month_table(info) -> pd.DataFrame`, one row per month. Columns: `Month`, `nfev`, `njev`, `Jacobians (fresh)`, `Jacobians (reused)`, `steps`, `status`, `degraded`, `clips`, `hden_passes_max`, `kernel_check`, `budget_residual`, `pool_budget_residual`, `min_state`, and `qs_inner_max` only under quasi-steady. None becomes NaN.
  - `flagged_months(info) -> pd.DataFrame`: the rows with status ≠ 0, degraded, `kernel_check > tol` or clips > 0.
- **Rendering:** `@render.ui solver_diagnostics()` returns None when `getattr(output, "solver_info", None) is None`; otherwise a card "Solver diagnostics" with the badges (Bootstrap `badge text-bg-{level}`), the summary table, the flagged months (or "No month flagged") and `ui.accordion(ui.accordion_panel("Every month", ui.output_data_frame("solver_month_table")), open=False)`. `@render.data_frame solver_month_table()` returns `render.DataGrid(solver_month_table(info), height="400px")`, a pandas frame, never GT (`test_render_contracts.py`). Shown by default: badges, summary, flagged months; the month table is collapsed (open question 2).

- [ ] **Step 1: Write the failing tests** (Review Focus 4):
  - `test_no_badges_for_parity_runs`: `solver_badges(None) == []`, `solver_summary(None) == []`, and an empty frame.
  - `test_badge_edges` on synthetic dicts from a helper `_info(months=…, **top)`: `kernel_check` NaN or 1e-9, no badge, 1.1e-9 a warning; `qs_separation` None, inf or 10.0, none, 9.99 a warning; `degraded` with `status` 0 a warning, `status` 1 danger; `budget_residual` 3.3e-6 at rtol 1e-10, none; quasi-steady with `qs_groups=[]`, the info badge; `"smooth"` forcing, the "changes the model" badge; `positivity="log"`, no such badge.
  - `test_month_table_with_none_values`: None budgets and `qs_inner_max` give NaN; `qs_inner_max` is a column only under quasi-steady; 600 synthetic months give 600 rows, and `flagged_months` returns only the 3 flagged.
  - `test_real_run_renders`: Central Baltic (`example_ecosim_scenario(..., years=range(1, 3), fast_group_rule="ewe")`) on LSODA through `run_ecosim`: the summary names 17 feeding-time states, the corrections `R2`, `DF once` and `Hden0 recalibrated`, and the four ignored switches; the only badge is "not EwE parity"; the table has 24 rows.
  - `test_panel_placeholders`: the Progress tab holds `output_ui("solver_diagnostics")`; the renderer pairs pass `test_render_contracts.py`'s checks.
- [ ] **Step 2: Run, expect FAIL.**
- [ ] **Step 3: Implement.**
- [ ] **Step 4: Run, expect PASS**, then the commit checks.
- [ ] **Step 5: Commit** `feat(shiny): solver diagnostics panel for continuous Ecosim runs`.

### Task 5: Runs off the event loop

Planner's call: **yes, async.** At the sidebar's default 50 years a continuous run takes about a minute (Lithuanian Coast LSODA 58 s, measured under load; Black Sea 57 s extrapolated), and the synchronous effect stalls every session of the app's one process. The threads are bit-identical and add 1.4 ms of loop lag at p99 (f4). This task is isolated and last; dropping it leaves Tasks 1-4 working, synchronously (open question 6).

**Files:** Modify `pages/ecosim.py`: `_run_simulation`, `_cancel_simulation`, `_create_scenario`, `progress_display`, a new `sim_elapsed` output, a new task, a result effect and a `model_data` effect. Modify `services/ecosim_solver.py`: add `RunOutcome` (`run_id`, `source`, the scenario object the run was started from, `output`, `request`, `diet_enabled`) and `result_is_current(state: TaskState, outcome: RunOutcome, current_scenario) -> bool`, true only when `state` is RUNNING with `outcome.run_id` and `current_scenario is outcome.source`. Tests: `test_ecosim_continuous_ui.py` and `test_ecosim_solver_service.py`.

**Interfaces:**
- **The task.** Inside `ecosim_server`:

  ```python
  @reactive.extended_task
  async def _ecosim_task(run_id, source, scen, request, diet):
      out = await asyncio.to_thread(run_ecosim, scen, request, diet_rewiring=diet)
      return RunOutcome(run_id, source, out, request, diet is not None)
  ```

  `scen` is a deep copy and `source` the scenario object from `scenario.get()`. The worker reads no input and sends no notification.
- **Retiring a run.** `_create_scenario`, and a new effect on `model_data`, call `_ecosim_task.cancel()` and set `sim_task_state` to a fresh `TaskState(run_id=f"ecosim-retired-{n}")`, alongside their existing `sim_output.set(None)`. A late outcome then fails `result_is_current` twice over: its `run_id` and its scenario.
- **`_run_simulation`**, while a run is RUNNING, warns "A simulation is already running. Cancel it first." and returns (`invoke` would queue). Otherwise it validates (Task 3), applies the forcing card, deep-copies the scenario, sets RUNNING ("Integrating with LSODA, 50 years") and calls `_ecosim_task.invoke(...)`.
- **A new `@reactive.effect _apply_ecosim_result()`** depends on `_ecosim_task.status()` only (it reads `sim_task_state` and `scenario` under `reactive.isolate()`, so its own `finish_task` does not re-trigger it):
  - `"success"`: when `result_is_current(sim_task_state.get(), outcome, scenario.get())`, it sets `sim_output` and `sim_results`, then runs the crash and success messages (moved from Task 3 unchanged) and `finish_task`. A stale outcome is dropped, logged at DEBUG.
  - `"error"`: `try: _ecosim_task.result()` with `except Exception as e:` gives `notify_error`, the continuous hint and `finish_task(success=False)`.
  - `"cancelled"`: nothing more.
- **`_cancel_simulation`** also calls `_ecosim_task.cancel()`. The message says the computation finishes in the background and its result is discarded: `to_thread` cannot stop the thread, and the `run_id` guard drops its result.
- **Progress.** `progress_display` shows an indeterminate striped bar in place of the fixed 20/50/85 % and keeps the Cancel button. The elapsed seconds go in their own `@render.text sim_elapsed()`, which alone calls `reactive.invalidate_later(1)` while RUNNING, so the button is not rebuilt every second. Per-month progress needs a core callback (Out of scope).

- [ ] **Step 1: Write the failing tests** (Review Focus 5):
  - `test_result_is_current`: true for the RUNNING `run_id` and the same scenario object; false for an older `run_id`, after `finish_task`, or when `current_scenario` is another object (even an equal deep copy).
  - `test_new_scenario_retires_the_run`: an AST check finds `_ecosim_task.cancel()` in `_create_scenario` and in an effect on `model_data`; `invalidate_later` appears only in `sim_elapsed`.
  - `test_worker_is_session_free`: an AST check of `_ecosim_task` and `run_ecosim` finds no `input`, `session`, `notify_*` or `ui.` names; `asyncio.to_thread(run_ecosim` appears in `ecosim.py`.
  - `test_run_simulation_invokes_the_task`: an AST check finds `_run_simulation` calling `_ecosim_task.invoke(` and `copy.deepcopy(`, with no `run_ecosim(` call.
  - `test_threaded_runs_match_serial` (f4 on CI's SciPy): two threads, the 3-group model on LSODA and Central Baltic on RK4, each two years; `out_Biomass` is `np.array_equal` to serial runs.
- [ ] **Step 2: Run, expect FAIL.**
- [ ] **Step 3: Implement.**
- [ ] **Step 4: Run, expect PASS**, then the commit checks.
- [ ] **Step 5: Manual check** with `python -m shiny run packages/pypath-shiny/src/pypath_shiny/app.py --port 8765`: load Central Baltic from Home and run 50 years on LSODA. Confirm that the elapsed seconds tick while the Cancel button stays clickable, that a second browser tab stays responsive, that after Cancel a new run's results survive the old thread finishing, and that Create Scenario mid-run leaves no output from the old run. Record the four observations in the commit body (a Playwright test under `tests/ui` is optional; CI skips it).
- [ ] **Step 6: Commit** `feat(shiny): run Ecosim off the event loop (extended task); cancel discards the result`.

### Task 6: Help, README, docs; final verification

**Files:** `pages/ecosim.py` (`help_scenario_setup`: "Integration method (RK4 recommended)" becomes "RK4 for EwE results; a continuous method to solve the equations to a tolerance"); `pages/tutorial.py` (line 471, one sentence); the README's Ecosim bullet (line 30); `docs/parity/continuous_path.md`, a new section "In the Shiny app" (methods, presets, exposed options, badges, the API-only keys).

- [ ] **Step 1:** Write the texts. Every number is quoted from this plan's measurement table with its date.
- [ ] **Step 2:** Run the commit checks and `pytest packages/pypath-shiny/tests/test_shinyswatch_integration.py -q` (the smoke workflow's first step).
- [ ] **Step 3:** Check the branch's full file list (`git diff --name-only main...HEAD`): only `packages/pypath-shiny/**`, `docs/parity/continuous_path.md` and this plan.
- [ ] **Step 4: Commit** `docs(shiny): continuous methods on the Ecosim page`.

## Versioning (note only; not done in 3b)

Yes, a bump, in order. The tag `pypath-ewe-v1.6.0` has no `ecosim_continuous.py` (phases 1-3a came 70 commits later), yet Shiny pins `pypath-ewe>=1.6.0` in `pyproject.toml`, `conda-recipes/pypath-shiny/meta.yaml` and the README. So pypath-ewe 1.7.0 is released first (minor: `feat` commits), then pypath-shiny 1.5.0, whose release PR raises all three floors to `>=1.7.0`. Deploys run from the source tree (deploy.yml installs both packages from the checkout), so merging 3b deploys correctly before either release.

## Out of scope

- **Fixing `rsim_run_advanced`.** It integrates nothing for any method (f2b); that is a separate core fix (open question 5).
- **Per-month progress.** It needs a callback in `rsim_run`, a core change.
- **Cooperative cancellation.** Cancel cannot stop a running integration: a worker thread cannot be interrupted, so Cancel only discards the result. Stopping a run mid-way needs a per-month callback in `rsim_run` that can abort, a core change. Until then Task 5 bounds the cost (ruling S5): a process-wide executor of `MAX_CONCURRENT_RUNS` (2) threads, and a per-session slot held until the worker finishes, so a session cannot start a new run while its cancelled one still computes.
- **API-only keys:** `atol`, `corrections`, `on_failure`, `hden_tol` and `jacobian`. `jacobian` is cost-only and `"reuse"` is better; `on_failure="warn"` would need a UI for partial results.
- **Core follow-ups found here:** `solver_info` does not say whether any forcing key was interpolated (only an INFO log does); there is no exported constant for the separation threshold of 10; `rsim_run`'s docstring calls `'lsoda'` "the choice for stiff models", against answer 2.
- **Other pages** (Ecospace, IBM, the demos) keep their integrators; Ecospace runs stay synchronous. No diagnostics CSV download. The releases themselves.

## The user's answers (2026-11-16), binding

1. RK4 stays the page's default method.
2. Diagnostics shown by default (badges, summary, flagged months); the month table collapsed.
3. Labels as planned.
4. Rename the Fast Groups select (Task 2, amended).
5. Diet rewiring: the core bug in `rsim_run_advanced` is investigated and fixed in a separate PR; 3b only disables the checkbox for continuous methods, as planned.
6. Async runs (Task 5) accepted.

## Open questions for the user (as asked; answered above)

1. **Default method:** the page keeps RK4 selected (EwE parity, as now). Agree?
2. **Diagnostics by default:** the badges, summary and flagged months show on the Progress tab after every continuous run; the full month table is collapsed. Agree, or collapse everything?
3. **Wording:** the group label "Continuous (not EwE parity)"; the method labels ("BDF (recommended for stiff models)"; LSODA is labelled "automatic stiff/non-stiff", because "LSODA (default)" reads wrongly under a select that defaults to RK4); the presets "Fast / Default / Reference" with the rtol in brackets; the accordion "Advanced: changes the model"?
4. **The existing "Fast Groups" select** ("EwE6: at equilibrium" / "Rpath: integrated") is untrue under a continuous method, where it picks only the equations, and its name collides with "Quasi-steady fast groups". Rename it, for example to "Equation rule (EwE6 / Rpath)", or keep it with the tooltip note?
5. **Diet rewiring:** `rsim_run_advanced` returns the start biomass every month, on RK4 too, while the page says "Diet rewiring was applied". Hide the checkbox for all methods until the core is fixed, or keep it as it is (3b only disables it for continuous methods)?
6. **Async runs (Task 5):** accept, or keep runs synchronous and drop Task 5?

## Spec contradictions found

1. **The brief's "`rsim_run_advanced` is RK4/AB only".** It is no integrator: it ignores `method` (it even accepts `"nonsense"`) and returns the start state.
2. **The sketch's per-month `rhs_evals`.** `solver_info["rhs_evals"]` is a run total; per month there are `nfev`, `njev`, `jac_fresh` and `jac_reused`.
3. **The sketch's "Reference (1e-10, `atol` scaled by `default_atol`)".** Since phase 2 the default `atol` already follows `rtol`. Scaling in Shiny is unnecessary, and passing `default_atol(b0)` would cap Reference.
4. **The sketch's "Fast" implies a speed-up.** It saves 37-57 % on the Black Sea and LT2022, 8 % on REco and 1-4 % on the shipped examples.
5. **The sketch's "smoke run of REco through the server function"** (and spec §9's "Shiny smoke test"): there is no server-session harness (`test_render_contracts.py` says so), and REco's builders live in core's `tests` package. 3b tests the services and the UI HTML on the 3-group model and `pypath.examples`.
6. **Spec §6's "add the two methods"** is superseded by four (decision 3). The brief's 11 s framing misses run length: at the sidebar's default 50 years a run takes about a minute (Lithuanian Coast LSODA 58 s measured; Black Sea 57 s, 116 s at Reference, extrapolated).
