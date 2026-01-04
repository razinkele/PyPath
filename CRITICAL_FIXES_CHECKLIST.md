# Critical Fixes Checklist - PyPath

## Immediate Action Items (Can be done today)

### 1. CRITICAL: Fix Bare Except Clause ⚠️
**File:** `src/pypath/io/ewemdb.py:835`
**Risk Level:** CRITICAL
**Effort:** 5 minutes

**Current:**
```python
except:
    pass
```

**Fix:**
```python
except Exception as e:
    logger.warning(f"Could not read optional field: {e}")
```

---

### 2. CRITICAL: Remove Debug Print Statements ⚠️
**File:** `src/pypath/io/ewemdb.py`
**Lines:** 355, 357, 479, 512, 516, 520, 522, 655, 727, 729
**Risk Level:** HIGH
**Effort:** 30 minutes

**Current:**
```python
print(f"[DEBUG] Found Auxillary table with {len(auxillary_df)} remarks")
```

**Fix:**
```python
logger.debug(f"Found Auxillary table with {len(auxillary_df)} remarks")
```

**Steps:**
1. Add `import logging` at top of file
2. Add `logger = logging.getLogger(__name__)`
3. Replace all `print(f"[DEBUG]...` with `logger.debug(...`
4. Remove `[DEBUG]` prefix

---

### 3. HIGH: Fix Overly Broad Exception Catching
**File:** `src/pypath/io/ewemdb.py`
**Lines:** 326, 329, 335, 338, 343, 346
**Risk Level:** HIGH
**Effort:** 1 hour

**Current:**
```python
try:
    value = row['FieldName']
except Exception:
    pass
```

**Fix:**
```python
try:
    value = row['FieldName']
except (KeyError, ValueError, TypeError) as e:
    logger.debug(f"Could not read field 'FieldName': {e}")
    value = None
```

---

## Quick Performance Wins (< 2 hours each)

### 4. Use scipy.spatial.distance for Distance Matrix
**File:** `src/pypath/spatial/connectivity.py:138-147`
**Impact:** 50-100x speedup
**Effort:** 2 hours

**Current:**
```python
for i in range(n_patches):
    for j in range(i + 1, n_patches):
        dx = centroids[i, 0] - centroids[j, 0]
        dy = centroids[i, 1] - centroids[j, 1]
        dist_deg = np.sqrt(dx**2 + dy**2)
```

**Fix:**
```python
from scipy.spatial.distance import cdist
distances = cdist(centroids, centroids, metric='euclidean') * 111.0
```

---

### 5. Replace iterrows() in ewemdb.py
**File:** `src/pypath/io/ewemdb.py`
**Lines:** 404, 485, 574, 592
**Impact:** 10-50x speedup
**Effort:** 1 hour

**Current:**
```python
for _, row in df.iterrows():
    process(row['col1'], row['col2'])
```

**Fix:**
```python
for val1, val2 in zip(df['col1'], df['col2']):
    process(val1, val2)
```

---

### 6. Auto-format with Black and isort
**All Python files**
**Impact:** Consistent style across entire codebase
**Effort:** 1 hour

**Commands:**
```bash
# Install tools
pip install black isort

# Format all files
black src/ app/ tests/
isort src/ app/ tests/

# Add to pre-commit hook
pip install pre-commit
```

**Create `.pre-commit-config.yaml`:**
```yaml
repos:
  - repo: https://github.com/psf/black
    rev: 23.12.1
    hooks:
      - id: black
  - repo: https://github.com/pycqa/isort
    rev: 5.13.2
    hooks:
      - id: isort
```

---

## Medium Priority (Next Week)

### 7. Create Validation Utilities Module
**Impact:** Eliminate ~250 lines of duplicate validation code
**Effort:** 1 day

**Create:** `src/pypath/core/validators.py`

```python
"""Centralized validation utilities for PyPath."""

class ParameterValidator:
    """Validation utilities for model parameters."""

    @staticmethod
    def validate_range(value: float, min_val: float, max_val: float,
                      name: str) -> None:
        """Validate value is within range."""
        if value < min_val or value > max_val:
            raise ValueError(
                f"{name} must be between {min_val} and {max_val}, got {value}"
            )

    @staticmethod
    def validate_shape(array: np.ndarray, expected_shape: tuple,
                      name: str) -> None:
        """Validate array has expected shape."""
        if array.shape != expected_shape:
            raise ValueError(
                f"{name}: expected shape {expected_shape}, got {array.shape}"
            )

    @staticmethod
    def validate_not_none(value, name: str) -> None:
        """Validate value is not None."""
        if value is None:
            raise ValueError(f"{name} cannot be None")

    @staticmethod
    def validate_positive(value: float, name: str) -> None:
        """Validate value is positive."""
        if value <= 0:
            raise ValueError(f"{name} must be positive, got {value}")

    @staticmethod
    def validate_non_negative(value: float, name: str) -> None:
        """Validate value is non-negative."""
        if value < 0:
            raise ValueError(f"{name} must be non-negative, got {value}")
```

**Usage example:**
```python
from pypath.core.validators import ParameterValidator as PV

# Instead of:
if biomass < 0 or biomass > 1e6:
    raise ValueError(f"Biomass must be between 0 and 1e6, got {biomass}")

# Use:
PV.validate_range(biomass, 0, 1e6, "Biomass")
```

---

### 8. Create UI Notification Helper
**Impact:** Eliminate ~60 lines of duplicate notification code
**Effort:** 4 hours

**Create:** `app/pages/ui_helpers.py`

```python
"""UI helper utilities for Shiny app."""
from shiny import ui
import logging

logger = logging.getLogger(__name__)


class NotificationHelper:
    """Centralized notification management."""

    @staticmethod
    def loading(message: str = "Loading...", duration: int = 3):
        """Show loading notification."""
        ui.notification_show(message, duration=duration)

    @staticmethod
    def success(message: str, duration: int = 2):
        """Show success notification."""
        ui.notification_show(message, type="message", duration=duration)

    @staticmethod
    def error(error: Exception, context: str = ""):
        """Show error notification and log it."""
        error_msg = str(error)
        display_msg = f"{context}: {error_msg}" if context else error_msg
        logger.error(f"Error in {context}: {error_msg}", exc_info=True)
        ui.notification_show(display_msg, type="error", duration=5)

    @staticmethod
    def warning(message: str, duration: int = 4):
        """Show warning notification."""
        logger.warning(message)
        ui.notification_show(message, type="warning", duration=duration)

    @staticmethod
    def info(message: str, duration: int = 3):
        """Show info notification."""
        ui.notification_show(message, type="default", duration=duration)


# Convenience shortcuts
notify = NotificationHelper()
```

**Usage example:**
```python
from app.pages.ui_helpers import notify

# Instead of:
ui.notification_show("Loading model...", duration=3)
ui.notification_show(f"Error: {str(e)}", type="error")

# Use:
notify.loading("Loading model...")
notify.error(e, context="Loading model")
```

---

### 9. Add Logging to Core Library
**Impact:** Better debugging, consistency with app layer
**Effort:** 2 days

**Steps:**
1. Add to each module in `src/pypath/core/` and `src/pypath/spatial/`:

```python
import logging
logger = logging.getLogger(__name__)
```

2. Replace `warnings.warn()` with `logger.warning()`:

**Before:**
```python
import warnings
warnings.warn("Biomass is very low")
```

**After:**
```python
logger.warning("Biomass is very low")
```

3. Add debug logging for key operations:

```python
logger.debug(f"Starting Ecopath balance with {n_groups} groups")
logger.info(f"Mass balance converged after {iterations} iterations")
logger.warning(f"Group {group_name} has EE > 1: {ee:.3f}")
logger.error(f"Mass balance failed: {error}")
```

---

## Performance Optimizations (Major Impact)

### 10. Vectorize Spatial Integration Loop
**File:** `src/pypath/spatial/integration.py:86-128`
**Impact:** 10-50x speedup
**Effort:** 2-3 days

See detailed implementation guide in main review document.

---

### 11. Optimize Dispersal Flux Calculation
**File:** `src/pypath/spatial/dispersal.py:58-91`
**Impact:** 10-30x speedup
**Effort:** 1-2 days

See detailed implementation guide in main review document.

---

## Tracking Progress

- [ ] 1. Fix bare except clause
- [ ] 2. Remove debug prints
- [ ] 3. Fix overly broad exceptions
- [ ] 4. Use scipy.spatial.distance
- [ ] 5. Replace iterrows()
- [ ] 6. Auto-format with black/isort
- [ ] 7. Create validation utilities
- [ ] 8. Create UI notification helper
- [ ] 9. Add logging to core library
- [ ] 10. Vectorize spatial integration
- [ ] 11. Optimize dispersal flux

---

## Testing After Changes

After each fix, run:

```bash
# Unit tests
pytest tests/ -v

# Specific module tests
pytest tests/test_ewemdb.py -v
pytest tests/test_spatial_integration.py -v

# With coverage
pytest tests/ --cov=src/pypath --cov-report=html
```

---

**Created:** December 20, 2025
**Priority:** Complete items 1-6 this week
