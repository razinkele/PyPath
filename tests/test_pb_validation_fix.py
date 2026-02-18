"""
Test script to verify P/B validation fix for phytoplankton.

This script tests that:
1. Producers (type=1) can have P/B up to 250
2. Consumers (type=0) still have P/B limit of 100
3. The validation messages are correct
"""

import sys
from pathlib import Path

# Add app to path
app_dir = Path(__file__).parent / "app"
sys.path.insert(0, str(app_dir))

from config import VALIDATION  # noqa: E402
from pages.validation import validate_pb  # noqa: E402


def test_pb_validation():
    """Test P/B validation with type-specific thresholds."""

    print("=" * 60)
    print("Testing P/B Validation Fix")
    print("=" * 60)

    # Test 1: Consumer with P/B = 50 (should pass)
    print("\n✓ Test 1: Consumer with P/B = 50")
    is_valid, error = validate_pb(50.0, "Small Fish", group_type=0)
    assert is_valid, f"Consumer P/B=50 should be valid. Error: {error}"
    print(f"  Result: PASS (valid={is_valid})")

    # Test 2: Consumer with P/B = 150 (should fail)
    print("\n✗ Test 2: Consumer with P/B = 150")
    is_valid, error = validate_pb(150.0, "Large Fish", group_type=0)
    assert not is_valid, "Consumer P/B=150 should be invalid"
    assert "100.0" in error, "Error should mention threshold of 100.0"
    print("  Result: PASS (correctly rejected)")
    print(f"  Error message: {error[:100]}...")

    # Test 3: Producer with P/B = 200 (should pass now!)
    print("\n✓ Test 3: Producer (Phytoplankton) with P/B = 200")
    is_valid, error = validate_pb(200.0, "Phytoplankton", group_type=1)
    assert is_valid, f"Producer P/B=200 should be valid. Error: {error}"
    print(f"  Result: PASS (valid={is_valid})")

    # Test 4: Producer with P/B = 300 (should fail)
    print("\n✗ Test 4: Producer with P/B = 300 (exceeds limit)")
    is_valid, error = validate_pb(300.0, "Phytoplankton", group_type=1)
    assert not is_valid, "Producer P/B=300 should be invalid"
    assert "250.0" in error, "Error should mention threshold of 250.0"
    print("  Result: PASS (correctly rejected)")
    print(f"  Error message: {error[:100]}...")

    # Test 5: No group type specified (should use default consumer limit)
    print("\n✓ Test 5: No group type specified, P/B = 50")
    is_valid, error = validate_pb(50.0, "Unknown Group", group_type=None)
    assert is_valid, f"P/B=50 should be valid with no type. Error: {error}"
    print(f"  Result: PASS (valid={is_valid})")

    # Test 6: No group type specified, P/B = 150 (should fail with consumer limit)
    print("\n✗ Test 6: No group type specified, P/B = 150")
    is_valid, error = validate_pb(150.0, "Unknown Group", group_type=None)
    assert not is_valid, "P/B=150 should be invalid with no type"
    print("  Result: PASS (correctly rejected)")

    print("\n" + "=" * 60)
    print("Configuration Values:")
    print("=" * 60)
    print(f"  VALIDATION.max_pb (consumers): {VALIDATION.max_pb}")
    print(f"  VALIDATION.max_pb_producer:     {VALIDATION.max_pb_producer}")

    print("\n" + "=" * 60)
    print("✅ ALL TESTS PASSED!")
    print("=" * 60)
    print("\nThe fix is working correctly:")
    print("  • Phytoplankton with P/B=200 will no longer trigger false warnings")
    print("  • Consumers still have stricter P/B limits")
    print("  • Producers can have P/B values up to 250")
    print("\nYou can now balance your example model without warnings for")
    print("Phytoplankton P/B values in the typical range (20-200).")
    print()


if __name__ == "__main__":
    test_pb_validation()
