# Checkpoint 2: Phase 2 (Great Tables) - Fixed, Pending Commit and Browser Check

**Branch:** feat/ui-stack-modernization
**Last updated:** 2026-10-29
**Objective:** ui-stack-modernization (8 phases)

> **Correction note.** The 2026-10-25 version of this checkpoint said Phase 2 was "100% complete,
> 540/540, no breaking changes". That was wrong: the committed Phase 2 broke all 19 converted
> tables and turned off Ecopath parameter and diet editing. The fix is in the working tree;
> see `PHASE_2_COMPLETION_REPORT.md` for the full breakdown.

---

## Phase status (verified)

| Phase | Status |
|---|---|
| 1A Dependencies | Done (`great_tables` had to be installed into the `shiny` env on 2026-10-29) |
| 1B Architecture | Done (decision: optimise, don't redesign; see `MODERNIZATION_PHASE_1B_DECISION.md`) |
| 2 GT integration | **Fixed, uncommitted.** 15 tables use static GT, 2 reverted to DataGrid, edit handlers restored, 78 regression tests. Needs a manual browser check. |
| 3 Plotly + shinywidgets | **Started, uncommitted.** `services/plotly_viz.py` (biomass and catch figures), `components.py` (cards, nav labels, theme toggle), results page now uses `render_widget`, home page redesign, CSS. Its tests pass. |
| 4-8 | Not started |

Tests: **625 passed** (`pytest packages/pypath-shiny/tests -q --ignore=packages/pypath-shiny/tests/ui`, about 70 s).

## Working tree (nothing committed since `7e30c14`)

- **Phase 2 fix only:** `services/gt_formatters.py`, `pages/ecopath.py`, `pages/data_import.py`,
  `pages/feedback.py`, `pages/prebalance.py`, `pages/multistanza.py`, `pages/optimization_demo.py`,
  `tests/test_render_contracts.py` (new), plus these two .md files.
- **Phase 3 work in progress only:** `app.py`, `pages/home.py`, `static/custom.css`, `services/__init__.py`,
  `components.py`, `services/plotly_viz.py`, `tests/test_components.py`, `tests/test_plotly_viz.py` (all new or modified).
- **Both:** `pages/results.py` (render.ui switch from the Phase 2 fix, plus Phase 3 Plotly widgets). Use `git add -p` to split it.

## Next steps

1. **Manual browser check** (automation unavailable on this machine): start the app, balance an
   example model, and confirm:
   - the Model Results table renders;
   - a parameter or diet edit persists;
   - the EcoBase picker still selects a model;
   - prebalance diagnostics render.
2. Commit the Phase 2 fix separately from the Phase 3 work in progress.
3. Decide on the trade-offs: sorting, filtering and cell highlighting lost on the static GT tables, and a probable unreadable header in `format_params_table` (navy fill, dark text).
4. Continue Phase 3: `plotly_viz` on ecosim and analysis pages, following the same service-layer pattern.
5. Before any PR, consider one test that renders a table through a real Shiny session.

## How to run here

```bash
export PATH="/c/Users/arturas.baziukas/micromamba/envs/shiny/Library/bin:$PATH"
export PYTHONPATH="$(pwd)/packages/pypath/src:$(pwd)/packages/pypath-shiny/src"
PY=/c/Users/arturas.baziukas/micromamba/envs/shiny/python.exe
$PY -m pytest packages/pypath-shiny/tests -q --ignore=packages/pypath-shiny/tests/ui
$PY -c "import uvicorn; from pypath_shiny.app import app; uvicorn.run(app, host='127.0.0.1', port=8765)"
```

## User feedback (laguna.ku.lt, checked 2026-10-29)

Both feedback DBs (`/srv/shiny-server/pypath/.../logs/feedback.db` and the fallback at
`/tmp/pypath-shiny-feedback/feedback.db`) have 0 rows and have never had a row inserted. The fallback
existing means the main path was unwritable at least once, and `feedback_store._connect` can fall back
to `:memory:` and lose submissions without warning. Worth one test submission on the live app.
