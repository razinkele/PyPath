# Phase 2 Report - Great Tables Integration

**Status:** Fixed in working tree, not yet committed; awaiting a manual browser check
**Original report:** 2026-10-25 (autopilot)
**Corrected:** 2026-10-29, after verification
**Tests:** 625 passed (`packages/pypath-shiny/tests`, excluding `ui/`), including 78 new render-contract tests

> **Correction note.** The 2026-10-25 version of this report said "COMPLETE, 540/540,
> 0 breaking changes". Checking it showed that the committed Phase 2 code (`e1c94bb`..`6600b5d`)
> broke every table it converted and turned off model editing. The test suite stayed green
> because no test sent a table through Shiny's renderer. The details are below.

---

## What was wrong in the committed Phase 2

| Problem | Effect | Evidence |
|---|---|---|
| 16 `@render.data_frame` functions returned a `GT` object | Shiny 1.8 raises `TypeError: Unsupported dataframe type ... GT`, so every converted table errored when rendered | Reproduced with `shiny.render._data_frame_utils._tbl_data.as_data_frame(GT(df))` |
| 3 `@render.ui` functions (`feedback_grid`, `stanza_table`, `results_table`) still used `ui.output_data_frame` placeholders | The renderer and placeholder types didn't match, so nothing rendered | AST pairing check |
| `ecopath.py` called `format_results_summary` and `format_dataframe` without importing them | `NameError` in `model_results_table` | ruff F821 |
| `@model_params_table.set_patch_fn` and `@diet_matrix_table.set_patch_fn` commented out (the original report listed this as "removed technical debt") | Edits to the Ecopath parameter and diet grids showed in the grid but were never saved to the model | diff against `96621c6` |
| `ecobase_models_table` converted to GT | Its row selection (`input.ecobase_models_table_selected_rows`) chooses which EcoBase model to import; GT can't select rows | code read |
| `feedback_grid` converted to GT | The per-column filters used to sort through feedback were lost | diff against `96621c6` |
| Every formatter called `opt_interactive()` | About 400 KB of HTML per table, plus an `<script type="importmap">` that browsers ignore when the table is inserted after page load | measured `_repr_html_()` output |
| `great_tables` was not installed in the `shiny` env | The test suite failed at collection with 38 errors, so "540/540" could not have been run in this env | |

## The fix (2026-10-29, uncommitted)

- **15 tables → static GT through `@render.ui` + `ui.output_ui`.** Each placeholder gets `max-height` + `overflow: auto` at its old DataGrid height (300 px, 400 px or 500 px), so pages don't grow.
  - ecopath: `model_results_table`
  - prebalance: `table_predator_prey`, `table_pb_ratios`, `table_qb_ratios`
  - results: `params_data_table`, `diet_data_table`, `biomass_data_table`, `catch_data_table`
  - data_import: `imported_groups_table`, `imported_diet_table`, `imported_stanza_groups_table`, `imported_stanza_indiv_table`, `biodata_results_table`
  - multistanza: `stanza_table`; optimization_demo: `results_table`
- **2 tables reverted to `render.DataGrid`:** `ecobase_models_table` (`selection_mode="row"`) and `feedback_grid` (`filters=True`). `feedback.py` is now identical to `96621c6`.
- **Ecopath edit handlers restored:** the params and diet `set_patch_fn` decorators are active again; fisheries was never disabled.
- **Missing imports added** in `ecopath.py`.
- **`opt_interactive()` removed** from all 6 formatters. The reason is recorded in the `gt_formatters.py` module docstring.
- **Dead code removed:** unused `create_cell_styles` calls and import in `data_import.py`, `NO_DATA_STYLE`/`STANZA_STYLE` imports in `ecopath.py`, and an unused `format_dataframe` import in `prebalance.py`.
- **New `tests/test_render_contracts.py` (78 tests).** All 18 bug-related tests failed before the fix. They check:
  - every placeholder matches its renderer type;
  - no `render.data_frame` returns a GT object;
  - every formatter a page calls is imported;
  - formatter output is static HTML with no import map;
  - the Ecopath edit handlers are registered;
  - the EcoBase picker and feedback grid keep their DataGrid features;
  - Shiny rejects GT in `render.data_frame` (a characterisation test that documents the upstream behaviour).

## Trade-offs compared with before Phase 2 (decide whether to keep)

- The 15 static GT tables lost column **sorting and filtering**.
- They also lost **cell highlighting**: the Ecopath results table coloured no-data, stanza and EE/GE/TL cells, and the imported-data tables marked no-data and remarks cells. GT `tab_style` could bring it back.
- `format_params_table` probably has an **unreadable header**: a `#1b3a5c` fill with the default dark text. It affects `params_data_table` and `stanza_table`. Not fixed yet.

## Verification

- Verified: full Shiny suite 625 passed; ruff `F` rules clean (one F841 at `ecopath.py` `title` predates the branch); the app starts from repo source and `/` returns 200 with no errors in the log.
- **Not verified:** a GT table actually appearing in the browser, and a cell edit saving into the model. Browser automation was unavailable: the Chrome extension wasn't connected, the Playwright MCP server was down, and Python Playwright in `shiny` is missing its node driver. Manual check: start the app, balance an example model, confirm the Model Results table shows, then edit a parameter and confirm it sticks.
- Production (laguna.ku.lt) still runs the code from before this branch and is unaffected.

## Architecture (corrected)

```
page.py
  @render.ui         -> services/gt_formatters.py -> static GT  -> ui.output_ui(id, style=max-height)
  @render.data_frame -> render.DataGrid (when editing / row selection / filters are needed)
```
