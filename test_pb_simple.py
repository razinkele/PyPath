"""
Simple test for P/B validation fix (avoids circular imports).
"""

import sys
from pathlib import Path

# Add app to path
app_dir = Path(__file__).parent / "app"
sys.path.insert(0, str(app_dir))

# Import only what we need to avoid circular imports
from config import VALIDATION

def test_config():
    """Test that config has the new producer threshold."""

    print("="*60)
    print("Testing P/B Configuration")
    print("="*60)

    print(f"\n✓ Consumer P/B threshold: {VALIDATION.max_pb}")
    assert VALIDATION.max_pb == 100.0, "Consumer threshold should be 100.0"

    print(f"✓ Producer P/B threshold: {VALIDATION.max_pb_producer}")
    assert VALIDATION.max_pb_producer == 250.0, "Producer threshold should be 250.0"

    print("\n" + "="*60)
    print("✅ Configuration is correct!")
    print("="*60)

    print("\nThis means:")
    print(f"  • Consumers (fish, invertebrates): P/B must be < {VALIDATION.max_pb}")
    print(f"  • Producers (phytoplankton, plants): P/B must be < {VALIDATION.max_pb_producer}")
    print("\nYour Phytoplankton with P/B=200 will now pass validation! ✨")
    print()

if __name__ == "__main__":
    test_config()
