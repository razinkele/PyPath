#!/usr/bin/env python
"""
Quick verification script for biodiversity database dependencies.

Run this after installing dependencies to verify everything is working.
"""

import sys

print("=" * 70)
print("Biodiversity Database Dependencies - Verification")
print("=" * 70)

# Check Python version
print(f"\nPython version: {sys.version}")

# Check pyworms
print("\n1. Checking pyworms...")
try:
    import pyworms

    print(f"   [OK] pyworms installed (version: {pyworms.__version__})")
    HAS_PYWORMS = True
except ImportError:
    print("   [MISSING] pyworms not found")
    print("   Install with: pip install pyworms")
    HAS_PYWORMS = False

# Check pyobis
print("\n2. Checking pyobis...")
try:
    import pyobis

    print(f"   [OK] pyobis installed (version: {pyobis.__version__})")
    HAS_PYOBIS = True
except ImportError:
    print("   [MISSING] pyobis not found")
    print("   Install with: pip install pyobis")
    HAS_PYOBIS = False

# Check requests
print("\n3. Checking requests...")
try:
    import requests

    print(f"   [OK] requests installed (version: {requests.__version__})")
    HAS_REQUESTS = True
except ImportError:
    print("   [MISSING] requests not found")
    print("   Install with: pip install requests")
    HAS_REQUESTS = False

# Check biodata module
print("\n4. Checking pypath.io.biodata module...")
try:
    sys.path.insert(0, "src")
    import pypath.io.biodata as biodata

    required = ["batch_get_species_info", "get_species_info"]
    missing = [name for name in required if not hasattr(biodata, name)]
    if missing:
        raise ImportError(f"Missing biodata attributes: {missing}")

    print("   [OK] biodata module can be imported")
    HAS_BIODATA = True
except ImportError as e:
    print(f"   [ERROR] biodata module import failed: {e}")
    HAS_BIODATA = False

# Summary
print("\n" + "=" * 70)
print("Summary")
print("=" * 70)

all_ok = HAS_PYWORMS and HAS_PYOBIS and HAS_REQUESTS and HAS_BIODATA

if all_ok:
    print("\n[OK] All dependencies installed!")
    print("\nYou can now:")
    print("  1. Run workflow test: python test_biodata_workflow.py")
    print("  2. Start Shiny app: shiny run app/app.py")
    print("  3. Use biodiversity databases in Data Import tab")
else:
    print("\n[ACTION REQUIRED] Some dependencies are missing")
    print("\nInstall missing dependencies with:")
    print("  pip install -e .[biodata]")
    print("\nOr install individually:")
    if not HAS_PYWORMS:
        print("  pip install pyworms")
    if not HAS_PYOBIS:
        print("  pip install pyobis")
    if not HAS_REQUESTS:
        print("  pip install requests")

print("\n" + "=" * 70)
sys.exit(0 if all_ok else 1)
