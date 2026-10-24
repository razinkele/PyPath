#!/usr/bin/env python
"""Run PEST calibration for Curonian Lagoon model.

Checkpoint 4 - Phase 2
"""

import subprocess
import sys
from pathlib import Path
import pandas as pd
import logging

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

def verify_pest_files():
    """Verify all required PEST files exist."""
    logger.info("Verifying PEST files...")
    
    required_files = [
        "curonian_pest.pst",
        "curonian_model_template.tpl",
        "curonian_landings_predictions.ins",
        "curonian_landings_observations.csv",
        "pest_runner.py",
    ]
    
    for fname in required_files:
        fpath = Path(fname)
        if fpath.exists():
            fsize = fpath.stat().st_size
            logger.info(f"  ✓ {fname} ({fsize} bytes)")
        else:
            logger.error(f"  ✗ Missing: {fname}")
            return False
    
    return True

def run_pest():
    """Execute PEST calibration."""
    logger.info("=" * 70)
    logger.info("PEST Calibration - Curonian Lagoon Ecosim Model")
    logger.info("=" * 70)
    
    if not verify_pest_files():
        logger.error("PEST files not ready for calibration")
        return False
    
    logger.info("\nStarting PEST calibration...")
    
    # Try to use pestpp-ies (Ensemble Inverse Solver) if available
    try:
        cmd = ["pestpp-ies", "curonian_pest.pst", "--n_threads", "4"]
        logger.info(f"Running: {' '.join(cmd)}")
        result = subprocess.run(cmd, capture_output=True, text=True)
        
        if result.returncode == 0:
            logger.info("✓ PEST calibration completed")
            return True
        else:
            logger.warning(f"PEST exited with code {result.returncode}")
            if result.stdout:
                logger.info(f"STDOUT: {result.stdout[:500]}")
            if result.stderr:
                logger.info(f"STDERR: {result.stderr[:500]}")
    
    except FileNotFoundError:
        logger.warning("pestpp-ies not found - trying standard PEST...")
        try:
            cmd = ["pest", "curonian_pest.pst"]
            result = subprocess.run(cmd, capture_output=True, text=True)
            if result.returncode == 0:
                logger.info("✓ PEST calibration completed")
                return True
        except FileNotFoundError:
            logger.error("Neither pestpp-ies nor pest found in PATH")
            logger.info("Calibration framework is ready, but PEST executables not available")
            logger.info("See PEST_CALIBRATION_GUIDE.md for installation instructions")
            return False
    
    return False

def parse_results():
    """Parse and display calibration results."""
    logger.info("\n" + "=" * 70)
    logger.info("Calibration Results")
    logger.info("=" * 70)
    
    # Look for PEST record file
    rec_files = list(Path(".").glob("*.rec"))
    if not rec_files:
        logger.info("No PEST record file found - calibration may not have run")
        return None
    
    logger.info(f"\nFound record file: {rec_files[0]}")
    
    # Try to parse record file for summary statistics
    try:
        with open(rec_files[0], "r") as f:
            content = f.read()
            
        if "Sum of squared weighted residuals" in content:
            for line in content.split("\n"):
                if "Sum of squared weighted residuals" in line or "Best fit" in line:
                    logger.info(f"  {line.strip()}")
    
    except Exception as e:
        logger.warning(f"Could not parse record file: {e}")
    
    return rec_files[0] if rec_files else None

def main():
    """Main function."""
    try:
        success = run_pest()
        
        if success or not verify_pest_files():
            rec_file = parse_results()
            if rec_file:
                logger.info(f"\nFull results in: {rec_file}")
        
        logger.info("\n" + "=" * 70)
        logger.info("Calibration workflow complete")
        logger.info("See PEST_CALIBRATION_GUIDE.md for next steps")
        logger.info("=" * 70)
        
        return 0
        
    except Exception as e:
        logger.error(f"Calibration error: {e}")
        import traceback
        traceback.print_exc()
        return 1

if __name__ == "__main__":
    sys.exit(main())
