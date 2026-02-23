# Design: Left Sidebar Pill List Navigation

**Date:** 2026-02-23
**Status:** Approved

## Goal

Replace the top navbar (`page_navbar`) with a left sidebar pill list navigation using `page_sidebar` + `navset_pill_list`. Add Bootstrap Icons to each nav item and use a collapsible section for Advanced Features.

## Current State

- `app.py` uses `ui.page_navbar()` with 10+ top-level items
- Items overflow on smaller screens
- Advanced Features grouped under a `nav_menu` dropdown
- Theme: shinyswatch "flatly"
- Bootstrap Icons CDN already loaded
- Shiny 1.5.1 installed (supports `page_sidebar`, `navset_pill_list`, `sidebar`)

## Design

### Layout

```
+--------------------+-------------------------------------+
| PyPath logo+title  |                                     |
| ================== |                                     |
| * Home             |                                     |
| * Data Import      |                                     |
| * Ecopath Model    |        Page Content Area            |
| * Pre-Balance      |                                     |
| * Ecosim           |        (navset content panel)       |
|                    |                                     |
| v Advanced         |                                     |
|   * Ecospace       |                                     |
|   * Multi-Stanza   |                                     |
|   * Forcing        |                                     |
|   * Diet Rewiring  |                                     |
|   * Optimization   |                                     |
|   * IBM            |                                     |
|                    |                                     |
| * Analysis         |                                     |
| * Results          |                                     |
| ------------       |                                     |
| Settings   About   |                                     |
|                    |                                     |
| (c) 2026 PyPath    |                                     |
+--------------------+-------------------------------------+
```

### Architecture

- **Top-level container:** `ui.page_sidebar()` with a `ui.sidebar()` on the left
- **Navigation:** `ui.navset_pill_list()` with `widths=(3, 9)` inside a `page_fluid` wrapper
- **Advanced grouping:** `ui.nav_menu("Advanced", ...)` collapses the 6 advanced pages
- **Icons:** Bootstrap Icons (`bi-house-fill`, `bi-download`, etc.) via `ui.tags.i(class_="bi bi-...")`
- **Sidebar header:** Logo SVG + "PyPath" title
- **Sidebar footer:** Settings button + About link + copyright

### Icon Assignments

| Page | Icon |
|------|------|
| Home | `bi-house-fill` |
| Data Import | `bi-download` |
| Ecopath Model | `bi-gear` |
| Pre-Balance | `bi-clipboard-check` |
| Ecosim | `bi-graph-up` |
| Advanced (group) | `bi-stars` |
| Ecospace | `bi-globe-americas` |
| Multi-Stanza | `bi-layers` |
| Forcing | `bi-lightning` |
| Diet Rewiring | `bi-arrow-repeat` |
| Optimization | `bi-bullseye` |
| IBM | `bi-cpu` |
| Analysis | `bi-bar-chart-line` |
| Results | `bi-file-earmark-text` |
| About | `bi-info-circle` |
| Settings | `bi-gear-fill` |

### Files Changed

| File | Change |
|------|--------|
| `app.py` | Replace `page_navbar` with `page_sidebar` + `navset_pill_list` |
| `custom.css` | Add sidebar pill styling, remove navbar-specific styles |
| `config.py` | Add `sidebar_nav_width` config value |

### What Stays the Same

- All page module `*_ui()` and `*_server()` functions unchanged
- Server function reactive state, SharedData, module initialization unchanged
- Theme (flatly), static assets, DataGrid column styles
- Individual page layouts (sidebars + tabs within pages)

## Scope

**In scope:** App-level navigation restructuring only.
**Out of scope:** Individual page layout modernization (value boxes, accordions, etc.).
