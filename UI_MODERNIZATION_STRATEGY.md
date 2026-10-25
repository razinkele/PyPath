# PyPath UI Stack Modernization Strategy
Based on Shiny 1.8+ Recommended Stack

Version: 1.0
Date: 2026-10-25
Status: Strategy Definition - Ready for Phase Implementation

## Recommended Stack (2026)

| Component | Version | Rationale | Priority |
|-----------|---------|-----------|----------|
| shiny | >= 1.8.0 | Core reactive framework; improved native UI | ⭐⭐⭐⭐⭐ |
| shinywidgets | 0.8.2+ | ipywidgets integration; scientific viz | ⭐⭐⭐⭐⭐ |
| plotly | latest | Interactive plots for results | ⭐⭐⭐⭐⭐ |
| great_tables | latest | Professional result/parameter tables | ⭐⭐⭐⭐⭐ |
| shinyswatch | 0.7.0+ | Bootstrap theme management | ⭐⭐⭐⭐ |
| pandas | >= 2.0 | Data manipulation | ⭐⭐⭐⭐⭐ |

## Key Architectural Principle

Separation of Concerns:
- PyPath Engine (NumPy, SciPy) → Standardized Results (DataFrames) → GUI/CLI
- ✅ NO Shiny objects in core engine
- ✅ GUI = consumer of same modeling API
- ✅ Compatible with future compiled backend

## 8-Phase Implementation Roadmap

### Phase 1A: Dependencies Update (0.5-1 days)
- Update pyproject.toml with shinywidgets, plotly, great_tables
- Run full test suite - verify no breaking changes
- Acceptance: All tests pass, app starts

### Phase 1B: Layout Modernization (1-2 days)
- Replace page_fluid + navset_pill_list with page_sidebar
- Use native Shiny ui.card() components
- Ensure responsive behavior on mobile
- Acceptance: All 18 pages load, responsive

### Phase 2: Great Tables Integration (2-3 days)
- Replace DataGrid with great_tables in result pages
- Update Ecopath, Ecosim, Results, Analysis
- Add export functionality
- Acceptance: 4+ pages use great_tables

### Phase 3: Plotly via shinywidgets (2-3 days)
- Create visualization module with Plotly figures
- Replace matplotlib with interactive Plotly
- Update 4+ pages
- Acceptance: Interactive plots render correctly

### Phase 4: Responsive & Accessibility (2-3 days)
- Mobile-first design (320px+)
- WCAG 2.1 AA compliance
- Dark mode support
- Acceptance: Mobile responsive, axe audit passes

### Phase 5: Component Library (1-2 days)
- Create 8+ reusable components
- Refactor pages to use components
- Acceptance: Components created and documented

### Phase 6: Architecture Review (1-2 days)
- Audit Shiny imports in core (should be zero)
- Standardize result objects
- Document API boundary
- Acceptance: Zero Shiny in core

### Phase 7: Testing & Documentation (2-3 days)
- Achieve >= 85% coverage
- Accessibility and mobile testing
- Update README and style guide
- Acceptance: Coverage >= 85%

### Phase 8: Release (1 day)
- Bump version 1.0.0 → 1.1.0
- Publish to PyPI and conda-forge
- Acceptance: Package published

## Timeline: 12-20 days total (44-62 hours)

## Next Steps

1. Review and approve this strategy
2. Create branch: feat/ui-stack-modernization
3. Execute Phase 1A (dependencies)
4. Execute Phases 1B-8 incrementally
5. Commit after each phase
6. PR after Phase 7 (before release)
