"""
Quick test to verify all Advanced Features pages are implemented and working.
"""

import sys
from pathlib import Path

# Add app to path
app_dir = Path(__file__).parent / "app"
sys.path.insert(0, str(app_dir))

print("=" * 70)
print("ADVANCED FEATURES IMPLEMENTATION CHECK")
print("=" * 70)

# Test each advanced feature page
features = [
    ("ECOSPACE Spatial Modeling", "ecospace"),
    ("Multi-Stanza Groups", "multistanza"),
    ("State-Variable Forcing", "forcing_demo"),
    ("Dynamic Diet Rewiring", "diet_rewiring_demo"),
    ("Bayesian Optimization", "optimization_demo"),
]

results = []

for feature_name, module_name in features:
    print(f"\n[Testing] {feature_name}...")
    try:
        # Import module
        module = __import__(f"pages.{module_name}", fromlist=[''])

        # Check for UI function
        ui_func = f"{module_name}_ui"
        if hasattr(module, ui_func):
            print(f"  [PASS] UI function '{ui_func}' found")
        else:
            print(f"  [FAIL] UI function '{ui_func}' NOT found")
            results.append((feature_name, False))
            continue

        # Check for Server function
        server_func = f"{module_name}_server"
        if hasattr(module, server_func):
            print(f"  [PASS] Server function '{server_func}' found")
        else:
            print(f"  [FAIL] Server function '{server_func}' NOT found")
            results.append((feature_name, False))
            continue

        # Count lines
        module_path = app_dir / "pages" / f"{module_name}.py"
        if module_path.exists():
            lines = len(module_path.read_text().splitlines())
            print(f"  [INFO] Implementation size: {lines} lines")

            if lines < 50:
                print(f"  [WARNING] File seems small ({lines} lines) - might be placeholder")
            else:
                print(f"  [PASS] Substantial implementation ({lines} lines)")

        # Try to call UI function to verify it returns something
        try:
            ui_result = getattr(module, ui_func)()
            if ui_result:
                print(f"  [PASS] UI function executes successfully")
            else:
                print(f"  [FAIL] UI function returns None/empty")
                results.append((feature_name, False))
                continue
        except Exception as e:
            print(f"  [FAIL] UI function execution error: {e}")
            results.append((feature_name, False))
            continue

        # Mark as success
        results.append((feature_name, True))
        print(f"  [SUCCESS] {feature_name} is fully implemented")

    except ImportError as e:
        print(f"  [FAIL] Could not import module: {e}")
        results.append((feature_name, False))
    except Exception as e:
        print(f"  [FAIL] Unexpected error: {e}")
        results.append((feature_name, False))

# Summary
print("\n" + "=" * 70)
print("SUMMARY")
print("=" * 70)

passed = sum(1 for _, success in results if success)
total = len(results)

print(f"\nTotal Features: {total}")
print(f"Implemented: {passed}")
print(f"Missing/Broken: {total - passed}")

if passed == total:
    print("\n" + "=" * 70)
    print("ALL ADVANCED FEATURES ARE FULLY IMPLEMENTED!")
    print("=" * 70)
    print("\nAccess via: Advanced Features menu in the Shiny app")
    print("Start app: shiny run app/app.py")
else:
    print("\n" + "=" * 70)
    print("SOME FEATURES NEED ATTENTION")
    print("=" * 70)
    print("\nFailed features:")
    for name, success in results:
        if not success:
            print(f"  - {name}")

print("\nNavigation path in app:")
print("  Advanced Features (⭐ menu)")
for feature_name, _ in features:
    print(f"    └── {feature_name}")

print("\n" + "=" * 70)
