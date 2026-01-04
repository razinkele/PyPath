# PyPath Shiny App - Code Style Guide

## Overview

This style guide documents the coding conventions and patterns used in the PyPath Shiny application. Following these guidelines ensures consistency, maintainability, and a smooth developer experience.

---

## Function Naming

### UI and Server Functions

- **UI functions**: `{module_name}_ui()`
  ```python
  def ecopath_ui() -> ui.Tag:
      """Ecopath page UI."""
      return ui.page_fluid(...)
  ```

- **Server functions**: `{module_name}_server(input, output, session)`
  ```python
  def ecopath_server(input: Inputs, output: Outputs, session: Session, model_data: reactive.Value):
      """Ecopath page server logic."""
      # ... server code ...
  ```

### Private Helper Functions

- Prefix with underscore: `_helper_function()`
- Used only within the module
- Not exported or called from other modules

```python
@reactive.effect
@reactive.event(input.btn_balance)
def _balance_model():
    """Private helper to balance the model."""
    # Internal logic only
```

### Public Utility Functions

- No underscore prefix
- Exported from `utils.py`
- Available to all modules
- Include comprehensive NumPy-style docstrings

```python
def is_balanced_model(model) -> bool:
    """Check if model is a balanced Rpath model.

    Parameters
    ----------
    model : object
        Model to check

    Returns
    -------
    bool
        True if model is balanced
    """
    return hasattr(model, 'NUM_LIVING')
```

---

## Import Organization

### Standard Order

1. Standard library imports
2. Third-party imports (shiny, pandas, numpy, matplotlib, etc.)
3. PyPath core imports (`pypath.core.*`)
4. App module imports (`app.config`, `app.pages.utils`, `app.logger`)

```python
# Standard library
from pathlib import Path
from typing import Optional

# Third-party
from shiny import Inputs, Outputs, Session, reactive, render, ui
import pandas as pd
import numpy as np

# PyPath core
from pypath.core.ecopath import rpath
from pypath.core.ecosim import rsim_scenario

# App modules
from app.config import DEFAULTS, UI, THRESHOLDS
from app.pages.utils import is_balanced_model
from app.logger import get_logger
```

### Config Imports

**Standard pattern** (use this):
```python
try:
    from app.config import DEFAULTS, UI, THRESHOLDS
except ModuleNotFoundError:
    from config import DEFAULTS, UI, THRESHOLDS
```

**What to import**:
- `DISPLAY`: Display formatting (NO_DATA_VALUE, decimal_places, type_labels)
- `PLOTS`: Matplotlib settings (default_width, default_height, dpi, style)
- `COLORS`: Color scheme (producer, consumer, detritus, boundary, etc.)
- `DEFAULTS`: Model defaults (unassim_*, default_months, default_vulnerability, etc.)
- `SPATIAL`: Ecospace settings (default_rows/cols, hexagon sizes, zoom)
- `VALIDATION`: Validation rules (min/max ranges for biomass, PB, QB, EE, GE)
- `UI`: UI dimensions and styling (sidebar_width, plot heights, col widths)
- `THRESHOLDS`: Numerical thresholds (vv_cap, crash_threshold, log_offset, etc.)
- `PARAM_RANGES`: UI slider bounds (years_*, vulnerability_*, switching_power_*, etc.)

---

## Error Handling

### User-Facing Operations

Apply to all `@reactive.effect` with `@reactive.event(input.btn_*)`:

```python
from app.logger import get_logger
logger = get_logger(__name__)

@reactive.effect
@reactive.event(input.btn_action)
def _handle_action():
    """Handle user-triggered action."""
    try:
        ui.notification_show("Processing...", duration=3)
        result = perform_operation()
        ui.notification_show("Success!", type="message", duration=2)
    except SpecificError as e:
        logger.error(f"Specific error in {operation}: {e}", exc_info=True)
        ui.notification_show(f"Operation failed: {str(e)}", type="error", duration=5)
    except Exception as e:
        logger.error(f"Unexpected error in {operation}: {e}", exc_info=True)
        ui.notification_show("An unexpected error occurred.", type="error", duration=5)
```

**Key principles**:
- Always log with `exc_info=True` for stack traces
- Show user-friendly notifications
- Catch specific exceptions first, then generic
- Provide context in log messages

### Reactive Calculations

Apply to `@reactive.calc` functions:

```python
@reactive.calc
def expensive_calculation():
    """Perform expensive calculation with error handling."""
    model = get_model()
    if model is None:
        return None

    try:
        result = compute_result(model)
        return result
    except ValueError as e:
        logger.warning(f"Invalid computation parameters: {e}")
        return None
    except Exception as e:
        logger.error(f"Calculation failed: {e}", exc_info=True)
        return None
```

**Key principles**:
- Return `None` on errors (don't raise)
- Log errors for debugging
- NO user notifications (calculations are automatic)
- Validate inputs early

### Data Processing with Graceful Degradation

```python
try:
    # Primary method
    result = primary_method(data)
except ValueError as e:
    logger.warning(f"Primary method failed with invalid data: {e}")
    # Fallback method
    try:
        result = fallback_method(data)
    except Exception as e:
        logger.error(f"Fallback also failed: {e}", exc_info=True)
        result = safe_default_value
except Exception as e:
    logger.error(f"Unexpected error in data processing: {e}", exc_info=True)
    result = safe_default_value
```

---

## Configuration Usage

### When to Use Config

✅ **DO use config for**:
- Algorithmic constants (crash_threshold, vv_cap, etc.)
- Model defaults (unassim_*, default_years, etc.)
- UI dimensions (plot heights, sidebar width, etc.)
- Slider ranges and bounds
- Thresholds for validation and stability
- Color schemes and styling

❌ **DON'T use config for**:
- Function-specific logic
- One-off calculations
- Demo data values (unless reused)
- Temporary variables

### Examples

```python
# GOOD - Uses config
ui.input_numeric(
    "sim_years",
    "Simulation Years",
    value=PARAM_RANGES.years_default,
    min=PARAM_RANGES.years_min,
    max=PARAM_RANGES.years_max
)

if biomass < THRESHOLDS.crash_threshold:
    logger.warning(f"Crash detected: biomass={biomass}")

# BAD - Hardcoded values
ui.input_numeric(
    "sim_years",
    "Simulation Years",
    value=50,  # Should use PARAM_RANGES.years_default
    min=1,     # Should use PARAM_RANGES.years_min
    max=500    # Should use PARAM_RANGES.years_max
)

if biomass < 0.0001:  # Should use THRESHOLDS.crash_threshold
    logger.warning(f"Crash detected")
```

---

## Documentation

### Docstring Format

Use **NumPy-style docstrings** for all public functions:

```python
def format_dataframe_for_display(
    df: pd.DataFrame,
    decimal_places: Optional[int] = None,
    remarks_df: Optional[pd.DataFrame] = None
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Format a DataFrame for display with number formatting and cell styling.

    This function processes a DataFrame to prepare it for display in the Shiny app by
    replacing no-data values, rounding numbers, and creating style masks.

    Parameters
    ----------
    df : pd.DataFrame
        The DataFrame to format for display
    decimal_places : Optional[int], default None
        Number of decimal places to round to. If None, uses DISPLAY.decimal_places
    remarks_df : Optional[pd.DataFrame], default None
        DataFrame containing remarks/tooltips for cells

    Returns
    -------
    formatted : pd.DataFrame
        Formatted DataFrame ready for display
    no_data_mask : pd.DataFrame
        Boolean mask indicating no-data cells
    remark_mask : pd.DataFrame
        Boolean mask indicating cells with remarks
    stanza_mask : pd.DataFrame
        Boolean mask indicating stanza group cells

    Examples
    --------
    >>> df = pd.DataFrame({'Biomass': [1.234, 9999, 5.678]})
    >>> formatted, no_data, remarks, stanzas = format_dataframe_for_display(df)
    >>> formatted['Biomass'].tolist()
    [1.234, nan, 5.678]
    """
    # Implementation
```

**Required sections**:
- One-line summary
- Extended description (if needed)
- Parameters (with types and descriptions)
- Returns (with types and descriptions)
- Examples (showing usage)

**Optional sections**:
- Raises (for exceptions)
- Notes (for implementation details)
- See Also (for related functions)

### Module Docstrings

Every module should have a clear module-level docstring:

```python
"""Ecopath balancing and parameter estimation.

This module provides the UI and server logic for the Ecopath page, which allows
users to create mass-balance food web models, balance them, and view results.
"""
```

---

## Help System

### Decision Matrix

| Page Type | Help Approach | Example |
|-----------|---------------|---------|
| Simple pages | No help | home, about |
| Data pages | Tooltips + collapsible details | ecopath, ecosim, results |
| Complex demos | Dedicated Help tab | forcing_demo, diet_rewiring_demo |
| Analysis pages | Tooltips only | analysis |

### Tooltips

Use Bootstrap icons with `title` attribute:

```python
ui.span(
    "Vulnerability ",
    ui.tags.i(
        class_="bi bi-info-circle",
        title="Controls predator-prey functional response. 1=bottom-up, 2=mixed, higher=top-down.",
        style="cursor: help;"
    )
)
```

### Collapsible Details

For moderate help within tabs:

```python
ui.tags.details(
    ui.tags.summary("Parameter Descriptions"),
    ui.div(
        ui.tags.dl(
            ui.tags.dt("Biomass (B)"),
            ui.tags.dd("Standing stock biomass (t/km²)"),
            # ... more items ...
        )
    )
)
```

### Help Tabs

For complex demos only:

```python
ui.nav_panel(
    "Help",
    ui.card(
        ui.card_header("How to Use This Demo"),
        ui.card_body(
            ui.h5("Overview"),
            ui.p("This demo shows..."),
            # ... more help content ...
        )
    )
)
```

---

## UI Patterns

### Button Classes

Shiny automatically adds `"btn"` class, so:

```python
# GOOD - Shiny adds "btn" automatically
ui.input_action_button("btn_run", "Run", class_="btn-success")

# UNNECESSARY - "btn" prefix is redundant (but harmless)
ui.input_action_button("btn_run", "Run", class_="btn btn-success")
```

### Layout Consistency

**For main pages with settings**:
```python
ui.layout_sidebar(
    ui.sidebar(
        # Settings controls
        width=UI.sidebar_width
    ),
    # Main content with tabs
    ui.navset_card_tab(...)
)
```

**For simple pages**:
```python
ui.page_fluid(
    ui.h2("Page Title"),
    # Content
)
```

**Column layouts**:
```python
ui.layout_columns(
    # Left content
    # Right content
    col_widths=[UI.col_width_wide, UI.col_width_narrow]
)
```

---

## Model Type Checking

### Use Helper Functions

Instead of direct `hasattr()` checks, use utilities from `utils.py`:

```python
# GOOD - Uses helper
from app.pages.utils import is_balanced_model, is_rpath_params, get_model_type

if is_balanced_model(model):
    # Model has been balanced
    indices = calculate_network_indices(model)

# BAD - Direct hasattr check
if hasattr(model, 'NUM_LIVING'):
    indices = calculate_network_indices(model)
```

**Available helpers**:
- `is_balanced_model(model) -> bool`: Check if model is balanced (has NUM_LIVING)
- `is_rpath_params(model) -> bool`: Check if model is RpathParams
- `get_model_type(model) -> str`: Return 'balanced', 'params', or 'unknown'

---

## Testing Guidelines

### Manual Testing Checklist

After changes:
- ✅ App starts without errors
- ✅ All pages load correctly
- ✅ No console errors or warnings
- ✅ Workflows complete successfully (import → balance → simulate)
- ✅ Config values display correctly
- ✅ Error logging works (check logs/pypath_app.log)

### Automated Testing

Recommended test coverage:
- Config dataclass validation
- Helper functions (is_balanced_model, etc.)
- Data formatting functions
- Validation functions

---

## Git Commit Messages

### Format

```
type(scope): Brief description

Extended description if needed.

- Bullet points for details
- More context as necessary

Generated with Claude Code https://claude.com/claude-code

Co-Authored-By: Claude Sonnet 4.5 <noreply@anthropic.com>
```

### Types

- `feat`: New feature
- `fix`: Bug fix
- `refactor`: Code refactoring
- `docs`: Documentation updates
- `test`: Test additions/changes
- `chore`: Maintenance tasks
- `style`: Code style/formatting

### Scopes

- `Phase 1`, `Phase 2`, `Phase 3`: Refactoring phases
- `config`: Configuration changes
- `ui`: UI improvements
- `core`: Core logic changes
- `docs`: Documentation

---

## Common Patterns

### Reactive Value Naming

```python
# GOOD - Descriptive names
model_data = reactive.Value(None)
sim_results = reactive.Value(None)
ecobase_models = reactive.Value(None)

# BAD - Too generic
params = reactive.Value(None)  # params for what?
data = reactive.Value(None)    # what kind of data?
```

### Notification Messages

```python
# Success
ui.notification_show("Model balanced successfully!", type="message", duration=2)

# Info
ui.notification_show("Processing...", duration=5)

# Warning
ui.notification_show("Some groups have EE > 1", type="warning", duration=5)

# Error
ui.notification_show(f"Balance failed: {error}", type="error", duration=10)
```

---

## File Organization

```
app/
├── __init__.py           # Path setup
├── app.py                # Main application
├── config.py             # All configuration
├── logger.py             # Logging setup
├── STYLE_GUIDE.md        # This file
├── static/               # Static assets
│   └── pypath_logo.svg
└── pages/                # Page modules
    ├── __init__.py
    ├── home.py
    ├── about.py
    ├── data_import.py
    ├── ecopath.py
    ├── ecosim.py
    ├── ecospace.py
    ├── results.py
    ├── analysis.py
    ├── multistanza.py
    ├── *_demo.py         # Demo pages
    ├── utils.py          # Shared utilities
    └── validation.py     # Input validation
```

---

## Best Practices

### DO
✅ Use config constants instead of magic numbers
✅ Log errors with `exc_info=True` for debugging
✅ Write comprehensive docstrings with examples
✅ Use helper functions to avoid code duplication
✅ Validate user inputs before processing
✅ Provide user-friendly error messages
✅ Keep functions focused and single-purpose
✅ Use type hints for function signatures

### DON'T
❌ Hardcode magic numbers or UI dimensions
❌ Silently catch exceptions without logging
❌ Skip docstrings on public functions
❌ Duplicate model type checking logic
❌ Mix UI and business logic
❌ Use overly generic variable names
❌ Leave TODO comments unaddressed
❌ Commit without testing

---

## Resources

- [Shiny for Python Documentation](https://shiny.posit.co/py/)
- [NumPy Documentation Guide](https://numpydoc.readthedocs.io/)
- [Python Type Hints (PEP 484)](https://peps.python.org/pep-0484/)
- [Pandas Documentation](https://pandas.pydata.org/docs/)

---

**Last Updated**: 2025-12-18
**Version**: 1.0
**Maintained by**: PyPath Development Team
