# Checkpoint 2: UI modernization - Phase 2 done, Phase 3 in progress

**Branch:** feat/ui-stack-modernization (not pushed)
**Last updated:** 2026-11-01
**Objective:** ui-stack-modernization (8 phases)

> Earlier versions of this file (2026-10-25, 2026-10-29) are superseded. The 2026-10-25
> "Phase 2 100% complete" claim was wrong; see `PHASE_2_COMPLETION_REPORT.md`.

---

## Phase status

| Phase | Status |
|---|---|
| 1A Dependencies / 1B Architecture | Done |
| 2 great_tables | **Done and browser-checked.** 15 read-only tables are static GT (`@render.ui`); editable tables and row pickers stay DataGrid. **Decision (user, 2026-11-01): keep static GT** - losing sort/filter/highlight on read-only tables is accepted. |
| 3 Plotly + shinywidgets | **In progress.** Plotly via `render_widget` on: Results (simulation biomass, catch), Ecosim (all charts: Time Series, Catch, Summary, effort preview per fleet), Analysis (MTI heatmap, keystoneness, EE bars). Home redesign, `components.py`. Remaining matplotlib (18): Results trophic-level bar/flow + food web (3), Ecopath (2), Pre-Balance (1), Analysis food web / trophic spectrum (2, from `pypath.core.plotting`), Ecospace (3), IBM (6), Feedback (1). Ecosim Summary charts not yet seen in a browser (window was minimised). |
| 4-8 | Not started |

## Browser check (2026-11-01, local app, Chrome)

All passed after fixes: Model Results table renders after balancing; a parameter edit
(Seals EE 0.1 -> 0.12) persists and is used by the balance; pre-balance diagnostics
render; EcoBase picker loads (about 50 s, EcoBase is slow) and selects a row; Ecosim
scenario + run + interactive Plotly biomass/catch charts.

Bugs the check found and fixed:

- **On this branch:** empty `<table>` from `@render.ui` returning `pd.DataFrame()` (11 outputs ->
  `components.empty_state()`); Ecosim "Create Scenario" failed (SilentException from
  inputs inside a hidden `output_ui`, guarded by always-true `hasattr(input, ...)` ->
  `read_optional_input()`); Ecosim header rows stretched ~700 px and pushed charts off
  screen (`fill=False`); format_params_table header was dark text on navy.
- **On main / production (separate PRs):** #31 pre-balance ratio tables always empty
  (merged); #32 Data Import panel invisible because `ui.sidebar` sat in a plain `div`
  (open, ready to merge).

Each has a contract test in `tests/test_render_contracts.py`,
`tests/test_layout_contracts.py` (on main after #32) or `tests/test_ecosim_optional_inputs.py`.

## State

- Branch is up to date with main as of #31 (merge `36d8537`), plus later commits.
  After #32 merges, merge `origin/main` in again.
- Shiny suite: 715 passed. `ruff check packages/` and `ruff format --check packages/` clean.
- CI (`ci-shiny-smoke.yml`) now runs the full Shiny unit suite (commit `4c51264`).

## Next steps

1. (done) #32 merged and deployed; `main` merged into the branch.
2. Phase 3: remaining matplotlib charts, core workflow first (Results trophic charts, Ecopath,
   Pre-Balance); same `plotly_viz` + `render_widget` pattern, builders + tests first.
   Browser-check the Ecosim Summary tab.
3. Cosmetic: the Model Results GT table formats the Type column as `0.000`.
4. Before opening the UI PR: full Shiny suite, a browser pass over every page, and
   remember `main` auto-deploys to laguna.

## How to run here

```bash
export PATH="/c/Users/arturas.baziukas/micromamba/envs/shiny/Library/bin:$PATH"
export PYTHONPATH="$(pwd)/packages/pypath/src:$(pwd)/packages/pypath-shiny/src"
PY=/c/Users/arturas.baziukas/micromamba/envs/shiny/python.exe
$PY -m pytest packages/pypath-shiny/tests -q --ignore=packages/pypath-shiny/tests/ui
$PY -c "import uvicorn; from pypath_shiny.app import app; uvicorn.run(app, host='127.0.0.1', port=8765)"
```

Memory is tight on this laptop (about 1.5 GB free with OneDrive and Chrome open); stop the
app when not checking in the browser. Put git worktrees outside OneDrive (e.g.
`C:\Users\arturas.baziukas\wt\...`): OneDrive's folder permissions block deleting them.
