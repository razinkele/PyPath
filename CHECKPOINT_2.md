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
- Shiny suite: 826 passed (2026-11-03, branch feat/ui-phase3-remaining-plotly). `ruff check packages/` and `ruff format --check packages/` clean.
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
   **deck.gl maps browser-checked 2026-11-03 on laguna** (headless Chrome + Playwright in
   laguna's shiny env, branch code in a /tmp dir on a private port; the laptop's headless
   hang was local only). Grid, Habitat, Effort OK for the default regular grid and an
   uploaded GeoJSON polygon grid, with the CARTO basemap. Fixed on the way: polygon cells
   drew black (`@@d.properties.fill` -> `@@=d.properties.fill`, commit 290f657).
   Also checked there (example model -> balance -> scenario -> spatial run -> IBM): biomass
   map + time slider (fixed colour range, redraws per step) and all six IBM charts.
   Fixed on the way: IBM comparison chart named the wrong group (4115a3c); every Plotly
   chart drew at ~360 px whatever its output_widget height, because height= turns off
   fillable (b8afbd9, `fillable=True` on all 21 calls + contract test).
   Follow-ups, all re-checked on laguna: map fit padding keeps the legend off the grid
   and fb_browser_info is sent on `shiny:sessioninitialized` (b2f3a8a; on
   `shiny:connected` it caused a ProtocolError that ended the session); Home example
   model was not mass-balanced - B, PB, QB and EE all set, Foragefish1 really at EE ~2.4,
   so Ecosim crashed it and Seals - now B or EE per group, flat over 50 years (912e022);
   Ecosim auto-fix off by default (user, 2026-11-03): capping derived QQ/DD moved balanced
   models off equilibrium (c091611). Shiny suite 818 passed (one subprocess test,
   test_prebalance_rpath_verify, failed once under load and passes alone).
   Core stanza bug: PR #45 (`fix/stanza-params-bridge`). `rsim_scenario` failed on every
   DataFrame-form (real) multi-stanza model and logged it at debug level. Now converted by
   `ecosim_stanza_params()`; `rpath_stanzas` matches Rpath's rpath.stanzas to 1e-6 (plus
   group ends at 371, not 419); scenario setup no longer edits the user's params.
   CORRECTION: until #47 PyPath Ecosim never coupled the stanza age structure to group
   biomass (`out_Biomass` identical with or without stanzas).
   #47 (`feat/stanza-coupling`, 2026-11-05): Rpath SplitUpdate/SplitSetPred ported;
   stanza groups within 2% of Rpath 1.2.0 over 50 y at effort x2 / x0 (RK4 and AB).
   The old x2 / x0 references were flat (`forcing$ForcedEffort` instead of
   `fishing$ForcedEffort`) and are regenerated (`scripts/extract_rpath_fishing_scenarios.R`;
   Rpath 1.2.0 built from GitHub into the user R library). Remaining non-stanza gaps:
   RK4 holds NoIntegrate groups at baseline (Discards, hence Seabirds); AB moves
   Zoo/Phyto/Detritus/Discards off baseline in month 1; rsim_run writes its adjusted M0
   into `scenario.params`, so a second run of a scenario differs by ~1e-4.
   Home example stanzas (this branch, 2026-11-04): Wmat 50 -> 0.1 (fraction of Winf),
   juvenile Last 24 -> 23, juvenile Z = its P/B, juvenile B/QB = what rpath_stanzas
   derives from the adult (0.219007 / 8.253302), so the inputs are consistent for when
   #47 lands.
   Deploy: pypath-shiny needs shiny-deckgl >= 1.14.0 (razinka only; PyPI has 1.12).
   `ci-shiny-smoke.yml` builds its env with micromamba from razinka; `conda-build.yml`
   adds `-c razinka`; `deploy.yml` pip-installs the GitHub `v1.14.0` tag into the venv.
   laguna's `/opt/micromamba/envs/shiny` (the pre-deploy import check and the app's
   interpreter): **shiny-deckgl 1.14.0 installed 2026-11-03** (conda, via sudo; the env
   is not writable by razinka). Replaced a root pip 1.9.2; other apps in that env
   (seagarden-dst, ewef viewer, HexSimPy, inSTREAMPY, osmose, sfincs) get 1.14 on restart.
   CSV upload of `example_model_data/model.csv` + `diet.csv` (was rejected over Rpath's
   `Import` row): fixed in e1ec499 - Import row and fleet columns accepted, blank B/PB
   cells mean "Ecopath estimates", 999999 hints removed; browser-checked on laguna.
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
