"""
Verification script for ECOSPACE integration in PyPath.

Run this to confirm ECOSPACE is properly integrated and accessible.
"""

import sys
from pathlib import Path

# Add paths
app_dir = Path(__file__).parent / "app"
src_dir = Path(__file__).parent / "src"
sys.path.insert(0, str(app_dir))
sys.path.insert(0, str(src_dir))

print("=" * 60)
print("ECOSPACE INTEGRATION VERIFICATION")
print("=" * 60)

# Test 1: Import spatial module
print("\n[Test 1] Importing spatial module...")
try:
    from pypath.spatial import (
        EcospaceGrid,
        EcospaceParams,
        allocate_gravity,
        allocate_uniform,
        create_1d_grid,
        create_regular_grid,
    )

    print("  [PASS] Spatial module imported successfully")
except ImportError as e:
    print(f"  [FAIL] Could not import spatial module: {e}")
    sys.exit(1)

# Test 2: Import ECOSPACE page module
print("\n[Test 2] Importing ECOSPACE page module...")
try:
    from pages import ecospace

    print("  [PASS] ECOSPACE page module imported")

    # Check for required functions
    if hasattr(ecospace, "ecospace_ui") and hasattr(ecospace, "ecospace_server"):
        print("  [PASS] UI and Server functions present")
    else:
        print("  [FAIL] Missing UI or Server functions")
        sys.exit(1)
except ImportError as e:
    print(f"  [FAIL] Could not import ECOSPACE page: {e}")
    sys.exit(1)

# Test 3: Import main app
print("\n[Test 3] Importing main app...")
try:
    from app import app, app_ui

    print("  [PASS] Main app imported successfully")
except ImportError as e:
    print(f"  [FAIL] Could not import main app: {e}")
    sys.exit(1)

# Test 4: Verify ECOSPACE in UI
print("\n[Test 4] Verifying ECOSPACE in navigation...")
ui_str = str(app_ui)
if "ECOSPACE" in ui_str:
    print("  [PASS] ECOSPACE found in UI")
else:
    print("  [FAIL] ECOSPACE not found in UI")
    sys.exit(1)

if "Advanced Features" in ui_str:
    print("  [PASS] Advanced Features menu present")
else:
    print("  [FAIL] Advanced Features menu not found")
    sys.exit(1)

# Test 5: Create a simple grid
print("\n[Test 5] Testing grid creation...")
try:
    import numpy as np

    # Create regular grid
    grid = create_regular_grid(bounds=(0, 0, 5, 5), nx=5, ny=5)
    print(f"  [PASS] Created 5x5 grid with {grid.n_patches} patches")

    # Create 1D grid
    grid_1d = create_1d_grid(n_patches=10, spacing=1.0)
    print(f"  [PASS] Created 1D grid with {grid_1d.n_patches} patches")

    # Test allocation
    effort = allocate_uniform(n_patches=25, total_effort=100.0)
    print(f"  [PASS] Uniform allocation: total = {effort.sum():.2f}")

except Exception as e:
    print(f"  [FAIL] Grid creation failed: {e}")
    sys.exit(1)

# Test 6: Test ECOSPACE parameters
print("\n[Test 6] Testing ECOSPACE parameters...")
try:
    n_groups = 5
    n_patches = 25

    ecospace_params = EcospaceParams(
        grid=grid,
        habitat_preference=np.ones((n_groups, n_patches)),
        habitat_capacity=np.ones((n_groups, n_patches)),
        dispersal_rate=np.array([0, 5.0, 2.0, 1.0, 3.0]),
        advection_enabled=np.array([False, True, True, False, True]),
        gravity_strength=np.array([0, 0.5, 0.3, 0, 0.7]),
    )
    print(f"  [PASS] Created ECOSPACE parameters for {n_groups} groups")

except Exception as e:
    print(f"  [FAIL] ECOSPACE parameters failed: {e}")
    sys.exit(1)

# Summary
print("\n" + "=" * 60)
print("VERIFICATION COMPLETE - ALL TESTS PASSED!")
print("=" * 60)
print("\nECOSPACE is properly integrated and accessible.")
print("\nTo use ECOSPACE in the Shiny app:")
print("1. Run: shiny run app/app.py")
print("2. Navigate to: Advanced Features > ECOSPACE Spatial Modeling")
print("3. Create a grid and explore the features")
print("\nFor Python API usage, see: docs/ECOSPACE_USER_GUIDE.md")
print("For quick start, see: ECOSPACE_QUICKSTART.md")
print("\n" + "=" * 60)
