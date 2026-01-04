#!/usr/bin/env python3
"""
Run the R script to extract Rpath reference data.

This script checks if R is available and runs extract_rpath_data.R
to generate reference test data from the Rpath R package.
"""

import subprocess
import sys
from pathlib import Path

def check_r_available():
    """Check if R is installed and available."""
    try:
        result = subprocess.run(
            ['R', '--version'],
            capture_output=True,
            text=True,
            timeout=10
        )
        if result.returncode == 0:
            print("✓ R is installed:")
            print(result.stdout.split('\n')[0])
            return True
        else:
            print("✗ R is not available")
            return False
    except FileNotFoundError:
        print("✗ R command not found")
        return False
    except Exception as e:
        print(f"✗ Error checking R: {e}")
        return False

def run_r_script(script_path):
    """Run the R script to extract reference data."""
    print(f"\nRunning R script: {script_path}")
    print("=" * 60)

    try:
        # Run R script
        result = subprocess.run(
            ['Rscript', str(script_path)],
            capture_output=True,
            text=True,
            timeout=300  # 5 minutes timeout
        )

        # Print output
        print(result.stdout)

        if result.stderr:
            print("STDERR:", file=sys.stderr)
            print(result.stderr, file=sys.stderr)

        if result.returncode == 0:
            print("\n✓ R script completed successfully!")
            return True
        else:
            print(f"\n✗ R script failed with code {result.returncode}")
            return False

    except subprocess.TimeoutExpired:
        print("\n✗ R script timed out (>5 minutes)")
        return False
    except Exception as e:
        print(f"\n✗ Error running R script: {e}")
        return False

def main():
    """Main function."""
    print("Rpath Reference Data Extraction")
    print("=" * 60)

    # Check R availability
    if not check_r_available():
        print("\nPlease install R from https://www.r-project.org/")
        print("Make sure R is in your PATH")
        sys.exit(1)

    # Find the R script
    script_dir = Path(__file__).parent
    r_script = script_dir / "extract_rpath_data.R"

    if not r_script.exists():
        print(f"\n✗ R script not found: {r_script}")
        sys.exit(1)

    # Run the R script
    success = run_r_script(r_script)

    if success:
        output_dir = Path("tests/data/rpath_reference")
        print(f"\nReference data saved to: {output_dir}")
        print("\nYou can now run the validation tests:")
        print("  python -m pytest tests/test_rpath_reference.py -v")
        sys.exit(0)
    else:
        print("\nFailed to extract reference data")
        sys.exit(1)

if __name__ == "__main__":
    main()
