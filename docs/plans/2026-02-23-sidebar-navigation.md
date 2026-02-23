# Sidebar Navigation Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Replace the top navbar with a left sidebar pill list navigation using bslib components and Bootstrap Icons.

**Architecture:** Replace `page_navbar` in `app.py` with `page_fluid` containing `navset_pill_list`. Each nav item gets a Bootstrap Icon label. Advanced Features are grouped under a collapsible `nav_menu`. The sidebar includes branding at top and footer links at bottom. No page modules change.

**Tech Stack:** Shiny for Python 1.5.1, bslib (`navset_pill_list`, `nav_menu`), Bootstrap Icons 1.11.3, shinyswatch (flatly theme)

**Design doc:** `docs/plans/2026-02-23-sidebar-navigation-design.md`

---

### Task 1: Add `nav_sidebar_width` to UIConfig

**Files:**
- Modify: `packages/pypath-shiny/src/pypath_shiny/config.py:130-168`

**Step 1: Add the config field**

In the `UIConfig` dataclass, after `sidebar_min_width` (line 135), add:

```python
    # Navigation sidebar (pill list)
    nav_sidebar_widths: tuple = (3, 9)  # (nav column, content column) out of 12
```

**Step 2: Verify config imports**

Run:
```bash
python -c "from pypath_shiny.config import UI; print(UI.nav_sidebar_widths)"
```
Expected: `(3, 9)`

**Step 3: Commit**

```bash
git add packages/pypath-shiny/src/pypath_shiny/config.py
git commit -m "feat(shiny): add nav_sidebar_widths to UIConfig"
```

---

### Task 2: Replace `page_navbar` with `navset_pill_list` in app.py

This is the main change. Replace the entire `app_ui` definition.

**Files:**
- Modify: `packages/pypath-shiny/src/pypath_shiny/app.py:66-158`

**Step 1: Define an icon helper**

Add a helper function above the `app_ui` definition (after the page imports, around line 64):

```python
def _icon_label(icon_class: str, text: str) -> ui.TagList:
    """Create a nav label with a Bootstrap Icon and text."""
    return ui.TagList(
        ui.tags.i(class_=f"bi {icon_class}", style="margin-right: 8px;"),
        text,
    )
```

**Step 2: Replace app_ui**

Replace the entire `app_ui = ui.page_navbar(...)` block (lines 66-158) with the new layout. The structure is:

```python
app_ui = ui.page_fluid(
    # Head content (CSS links, DataGrid styles)
    ui.head_content(
        ui.tags.link(
            rel="stylesheet",
            href="https://cdn.jsdelivr.net/npm/bootstrap-icons@1.11.3/font/bootstrap-icons.min.css",
        ),
        ui.tags.link(rel="stylesheet", href="custom.css"),
        ui.tags.style(f"""
            /* Make Group column wider in DataGrids */
            .shiny-data-grid td:first-child,
            .shiny-data-grid th:first-child {{
                min-width: {UI.table_col_min_width_px} !important;
                max-width: {UI.table_col_max_width_px} !important;
            }}
            /* Style for numeric columns */
            .shiny-data-grid td:not(:first-child) {{
                text-align: right;
                font-family: monospace;
            }}
        """),
    ),
    # Branding header
    ui.div(
        ui.tags.img(
            src="icon.svg",
            height=UI.icon_height_px,
            style="margin-right: 8px; vertical-align: middle;",
        ),
        ui.tags.span("PyPath", style="font-weight: 600; font-size: 1.4rem; vertical-align: middle;"),
        class_="p-3 mb-2",
    ),
    # Main navigation
    ui.navset_pill_list(
        # Core workflow pages
        ui.nav_panel(
            _icon_label("bi-house-fill", "Home"),
            home.home_ui(),
        ),
        ui.nav_panel(
            _icon_label("bi-download", "Data Import"),
            data_import.import_ui(),
        ),
        ui.nav_panel(
            _icon_label("bi-gear", "Ecopath Model"),
            ecopath.ecopath_ui(),
        ),
        ui.nav_panel(
            _icon_label("bi-clipboard-check", "Pre-Balance"),
            prebalance.prebalance_ui(),
        ),
        ui.nav_panel(
            _icon_label("bi-graph-up", "Ecosim"),
            ecosim.ecosim_ui(),
        ),
        # Advanced features (collapsible group)
        ui.nav_menu(
            _icon_label("bi-stars", "Advanced"),
            ui.nav_panel(
                _icon_label("bi-globe-americas", "Ecospace"),
                ecospace.ecospace_ui(),
            ),
            ui.nav_panel(
                _icon_label("bi-layers", "Multi-Stanza"),
                multistanza.multistanza_ui(),
            ),
            ui.nav_panel(
                _icon_label("bi-lightning", "Forcing"),
                forcing_demo.forcing_demo_ui(),
            ),
            ui.nav_panel(
                _icon_label("bi-arrow-repeat", "Diet Rewiring"),
                diet_rewiring_demo.diet_rewiring_demo_ui(),
            ),
            ui.nav_panel(
                _icon_label("bi-bullseye", "Optimization"),
                optimization_demo.optimization_demo_ui(),
            ),
            ui.nav_panel(
                _icon_label("bi-cpu", "IBM"),
                ibm.ibm_ui(),
            ),
        ),
        # Output pages
        ui.nav_panel(
            _icon_label("bi-bar-chart-line", "Analysis"),
            analysis.analysis_ui(),
        ),
        ui.nav_panel(
            _icon_label("bi-file-earmark-text", "Results"),
            results.results_ui(),
        ),
        "----",
        ui.nav_panel(
            _icon_label("bi-info-circle", "About"),
            about.about_ui(),
        ),
        ui.nav_control(
            ui.input_action_button(
                "btn_settings",
                _icon_label("bi-gear-fill", "Settings"),
                class_="btn btn-link text-start w-100 p-2",
            ),
        ),
        id="main_nav",
        widths=UI.nav_sidebar_widths,
        well=True,
    ),
    # Footer
    ui.div(
        ui.tags.hr(),
        ui.tags.p(
            f"PyPath © {datetime.now().year} | ",
            ui.tags.a(
                "Documentation",
                href="https://github.com/razinkele/PyPath",
                class_="text-decoration-none",
            ),
            " | ",
            ui.tags.a(
                "Report Issue",
                href="https://github.com/razinkele/PyPath/issues",
                class_="text-decoration-none",
            ),
            class_="text-center text-muted small",
        ),
        class_="p-2",
    ),
    theme=(
        shinyswatch.theme.flatly
        if getattr(shinyswatch, "theme", None) is not None
        and getattr(shinyswatch.theme, "flatly", None) is not None
        else None
    ),
)
```

Key changes from the old layout:
- `page_navbar` → `page_fluid` + `navset_pill_list`
- `id="main_navbar"` → `id="main_nav"`
- `fillable=True` removed (page_fluid handles this)
- Each `nav_panel` title gets `_icon_label()` instead of plain string
- `nav_menu("Advanced Features", ...)` → `nav_menu(_icon_label("bi-stars", "Advanced"), ...)`
- Settings button moved inside the pill list as a `nav_control`
- `"----"` separator before About and Settings
- Branding div added above the navset

**Step 3: Run import check**

```bash
python -c "from pypath_shiny.app import app_ui; print('OK')"
```
Expected: `OK`

**Step 4: Commit**

```bash
git add packages/pypath-shiny/src/pypath_shiny/app.py
git commit -m "feat(shiny): replace top navbar with left sidebar pill list navigation"
```

---

### Task 3: Update custom.css for sidebar pill styling

**Files:**
- Modify: `packages/pypath-shiny/src/pypath_shiny/static/custom.css`

**Step 1: Replace navbar-specific CSS with sidebar pill styling**

Remove the `.navbar-brand` rule (line 38-40). Add new rules for the pill list navigation:

```css
/* Sidebar pill list navigation */
.nav-pills .nav-link {
    color: #2c3e50;
    border-radius: 6px;
    padding: 0.5rem 0.75rem;
    margin-bottom: 2px;
    transition: background-color 0.15s ease-in-out;
}

.nav-pills .nav-link:hover {
    background-color: #e9ecef;
}

.nav-pills .nav-link.active {
    background-color: #2c3e50;
    color: #fff;
    font-weight: 600;
}

/* Nav menu (Advanced section) header */
.nav-pills .dropdown-toggle {
    color: #6c757d;
    font-weight: 600;
    font-size: 0.85rem;
    text-transform: uppercase;
    letter-spacing: 0.05em;
}

/* Pill list well (left column background) */
.well {
    background-color: #f8f9fa;
    border: 1px solid #e9ecef;
    border-radius: 8px;
    padding: 0.75rem;
}
```

Also update the `.nav-tabs .nav-link.active` rule (line 116-119) so it doesn't conflict — scope it to only apply inside tab navsets:

```css
/* Tab panels (inside pages, not the sidebar) */
.navset-card-tab .nav-tabs .nav-link.active {
    font-weight: 600;
    border-bottom: 3px solid #3498db;
}
```

**Step 2: Verify CSS loads**

```bash
python -c "from pypath_shiny.app import APP_DIR; assert (APP_DIR / 'static' / 'custom.css').exists(); print('OK')"
```

**Step 3: Commit**

```bash
git add packages/pypath-shiny/src/pypath_shiny/static/custom.css
git commit -m "style(shiny): update CSS for sidebar pill list navigation"
```

---

### Task 4: Update test assertions for new navigation structure

Some tests check for `"main_navbar"` or navbar-specific strings. Update them to match the new `"main_nav"` id.

**Files:**
- Modify: `packages/pypath-shiny/tests/test_shiny_app.py` (if it references `main_navbar`)

**Step 1: Search for references**

```bash
grep -r "main_navbar" packages/pypath-shiny/tests/
```

If any results, update `"main_navbar"` → `"main_nav"`. Also check for `"page_navbar"` string assertions.

The test at `test_shiny_app.py:113` says `# App UI should be a page_navbar` in a comment — update the comment but the assertion (`app_ui is not None`) doesn't need changing.

**Step 2: Run shiny tests**

```bash
pytest packages/pypath-shiny/tests/ -q --ignore=packages/pypath-shiny/tests/ui
```
Expected: All pass (130+)

**Step 3: Commit (if changes needed)**

```bash
git add packages/pypath-shiny/tests/
git commit -m "test(shiny): update test comments for sidebar navigation"
```

---

### Task 5: Visual verification and final commit

**Step 1: Launch the app**

```bash
python -c "import uvicorn; from pypath_shiny.app import app; uvicorn.run(app, host='127.0.0.1', port=8000)"
```

**Step 2: Verify in browser at http://127.0.0.1:8000**

Check:
- [ ] Left sidebar pill list visible with icons
- [ ] Clicking each pill switches content area
- [ ] "Advanced" section collapses/expands with 6 sub-items
- [ ] Settings button opens theme modal
- [ ] About page accessible
- [ ] Footer visible
- [ ] Logo + "PyPath" branding at top
- [ ] Theme (flatly) applied correctly
- [ ] Individual page layouts (Ecopath sidebar + tabs) still work inside content area

**Step 3: Run full test suite**

```bash
pytest packages/pypath-shiny/tests/ -q --ignore=packages/pypath-shiny/tests/ui
```
Expected: All pass

**Step 4: Final commit (if any tweaks needed)**

```bash
git add -A
git commit -m "feat(shiny): sidebar navigation polish and final adjustments"
```
