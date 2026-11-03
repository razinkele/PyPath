# Checkpoint 2: UI modernization - Phase 2 done, Phase 3 in progress

**Branch:** feat/ui-phase3-remaining-plotly (Phase 1-3 core merged in #34)
**Last updated:** 2026-11-03
**Objective:** ui-stack-modernization (8 phases)

> Earlier versions of this file (2026-10-25, 2026-10-29) are superseded. The 2026-10-25
> "Phase 2 100% complete" claim was wrong; see `PHASE_2_COMPLETION_REPORT.md`.

---

## Phase status

| Phase | Status |
|---|---|
| 1A Dependencies / 1B Architecture | Done |
| 2 great_tables | **Done and browser-checked.** 15 read-only tables are static GT (`@render.ui`); editable tables and row pickers stay DataGrid. **Decision (user, 2026-11-01): keep static GT** - losing sort/filter/highlight on read-only tables is accepted. |
| 3 Plotly + shinywidgets | **App pages done.** Merged in #34: Results, Ecosim, Analysis (MTI, keystoneness, EE), Ecopath. Branch `feat/ui-phase3-remaining-plotly` (2026-11-03): Feedback trend, IBM (6 charts) on Plotly. **Ecospace maps on deck.gl, not Plotly (user, 2026-11-03):** grid view (was folium), habitat, effort and biomass maps use shiny-deckgl >= 1.14.0 `MapWidget`s; builders in `services/deckgl_maps.py`. Only matplotlib left: plots drawn by the core package (Analysis food web + trophic spectrum, Pre-Balance diagnostic) - needs a pypath-ewe API decision. |
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
- Shiny suite: 789 passed (2026-11-03, branch feat/ui-phase3-remaining-plotly). `ruff check packages/` and `ruff format --check packages/` clean.
- CI (`ci-shiny-smoke.yml`) now runs the full Shiny unit suite (commit `4c51264`).

## Next steps

1. (done) #32 merged and deployed; `main` merged into the branch.
2. (done 2026-11-02) Browser pass of all converted charts.
   Open decision: core-package plots (pypath.core.plotting / analysis.prebalance) and
   Ecospace / IBM / Feedback charts.
3. (done 2026-11-03) Type column shows whole numbers in GT tables. Example diet data fixed in #33.
5. Branch `feat/ui-phase3-remaining-plotly` (local, not pushed), 2026-11-03.
   Browser-checked (Playwright): Feedback trend (empty state). The Plotly Ecospace maps
   checked that day were then replaced by deck.gl, so those checks no longer apply.
   **deck.gl maps: NOT browser-checked.** Headless Chromium (Playwright MCP) hung while
   loading the app page with the four maps, three times; it may be headless WebGL
   (SwiftShader) on this laptop or a real page hang - check in a real Chrome first:
   Ecospace -> Create Grid -> Grid, Habitat Map, Fishing Effort tabs; then a shapefile /
   GeoJSON grid; then the biomass map + time slider after a spatial run. Also not
   browser-checked: the IBM charts (need a loaded model).
   Deploy: pypath-shiny needs shiny-deckgl >= 1.14.0 (razinka only; PyPI has 1.12).
   `ci-shiny-smoke.yml` builds its env with micromamba from razinka; `conda-build.yml`
   adds `-c razinka`; `deploy.yml` pip-installs the GitHub `v1.14.0` tag into the venv.
   **laguna's `/opt/micromamba/envs/shiny` (the pre-deploy import check and the app's
   interpreter) needs `micromamba install -n shiny -c razinka -c conda-forge
   "shiny-deckgl>=1.14"`**, or deploys stop at the import check.
   Found on the way, not fixed: uploading `example_model_data/model.csv` + `diet.csv` on the
   Ecopath page is rejected ("Diet CSV has groups not in model: Import"); the inline
   `Shiny.setInputValue` call in app.py (fb_browser_info) throws before Shiny is ready.
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
