#!/usr/bin/env python
"""Validate PEST setup before calibration.

Checkpoint 4 - Phase 2
"""

import sys
from pathlib import Path
import pandas as pd
import logging

logging.basicConfig(
    level=logging.INFO,
    format="%(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

class PestSetupValidator:
    """Validate all PEST configuration files."""
    
    def __init__(self):
        self.errors = []
        self.warnings = []
        self.checks_passed = 0
        self.checks_total = 0
    
    def check(self, condition, message):
        """Check a condition."""
        self.checks_total += 1
        if condition:
            logger.info(f"✓ {message}")
            self.checks_passed += 1
            return True
        else:
            logger.error(f"✗ {message}")
            self.errors.append(message)
            return False
    
    def validate_pst_file(self):
        """Validate .pst control file."""
        logger.info("\n1. Checking PEST control file (.pst)...")
        
        pst_file = Path("curonian_pest.pst")
        if not self.check(pst_file.exists(), ".pst file exists"):
            return False
        
        try:
            with open(pst_file, "r") as f:
                content = f.read()
            
            self.check("* control data" in content, "Contains control data section")
            self.check("* parameter data" in content, "Contains parameter data section")
            self.check("* observation data" in content, "Contains observation data section")
            self.check("model command line" in content or "pest" in content.lower(), 
                      "Contains model command reference")
            
            # Count parameters and observations
            lines = content.split("\n")
            param_count = sum(1 for l in lines if l.startswith(" ") and "pargp_" in l)
            obs_count = sum(1 for l in lines if "obsgrp_" in l and len(l) > 5)
            
            logger.info(f"  Parameters in .pst: {param_count}")
            logger.info(f"  Observations in .pst: {obs_count}")
            
            return True
        except Exception as e:
            self.check(False, f".pst file readable ({e})")
            return False
    
    def validate_template_file(self):
        """Validate .tpl template file."""
        logger.info("\n2. Checking template file (.tpl)...")
        
        tpl_file = Path("curonian_model_template.tpl")
        if not self.check(tpl_file.exists(), ".tpl file exists"):
            return False
        
        try:
            with open(tpl_file, "r") as f:
                content = f.read()
            
            self.check(content.startswith("ptf %"), "Template file starts with 'ptf %'")
            self.check("%" in content, "Contains parameter placeholders (%...%)")
            
            # Count placeholders
            placeholder_count = content.count(" % ") // 2
            logger.info(f"  Parameter placeholders: {placeholder_count}")
            
            return True
        except Exception as e:
            self.check(False, f".tpl file readable ({e})")
            return False
    
    def validate_instruction_file(self):
        """Validate .ins instruction file."""
        logger.info("\n3. Checking instruction file (.ins)...")
        
        ins_file = Path("curonian_landings_predictions.ins")
        if not self.check(ins_file.exists(), ".ins file exists"):
            return False
        
        try:
            with open(ins_file, "r") as f:
                content = f.read()
            
            self.check(content.startswith("pif %"), "Instruction file starts with 'pif %'")
            self.check("!obs_" in content, "Contains observation placeholders")
            
            # Count extraction instructions
            extraction_count = content.count("w !")
            logger.info(f"  Extraction points: {extraction_count}")
            
            return True
        except Exception as e:
            self.check(False, f".ins file readable ({e})")
            return False
    
    def validate_observations(self):
        """Validate observations CSV."""
        logger.info("\n4. Checking observations file...")
        
        obs_file = Path("curonian_landings_observations.csv")
        if not self.check(obs_file.exists(), "Observations CSV exists"):
            return False
        
        try:
            obs_df = pd.read_csv(obs_file)
            
            self.check(len(obs_df) > 0, f"Contains observations ({len(obs_df)} rows)")
            self.check("year" in obs_df.columns, "Contains 'year' column")
            self.check("species_code" in obs_df.columns, "Contains 'species_code' column")
            self.check("observed_landings" in obs_df.columns, "Contains 'observed_landings' column")
            self.check("weight" in obs_df.columns, "Contains 'weight' column")
            
            # Check data ranges
            year_range = f"{obs_df['year'].min()}-{obs_df['year'].max()}"
            logger.info(f"  Year range: {year_range}")
            logger.info(f"  Species count: {obs_df['species_code'].nunique()}")
            logger.info(f"  Mean observation weight: {obs_df['weight'].mean():.2f}")
            
            return True
        except Exception as e:
            self.check(False, f"Observations CSV readable ({e})")
            return False
    
    def validate_parameters(self):
        """Validate parameter ranges."""
        logger.info("\n5. Checking parameter configuration...")
        
        ranges_file = Path("PEST_CALIBRATION_RANGES.csv")
        if not self.check(ranges_file.exists(), "Parameter ranges file exists"):
            return False
        
        try:
            ranges_df = pd.read_csv(ranges_file)
            pest_params = ranges_df[ranges_df["fit_in_pest"] == True]
            
            self.check(len(pest_params) > 0, f"PEST parameters defined ({len(pest_params)})")
            
            # Check bounds validity
            invalid_bounds = pest_params[pest_params["lower_bound"] >= pest_params["upper_bound"]]
            self.check(len(invalid_bounds) == 0, 
                      f"Parameter bounds valid (all lower < upper)")
            
            if len(invalid_bounds) > 0:
                logger.warning(f"  Invalid bounds found: {invalid_bounds.shape[0]}")
            
            logger.info(f"  Parameter types: {pest_params['param_type'].unique().tolist()}")
            
            return True
        except Exception as e:
            self.check(False, f"Parameter ranges readable ({e})")
            return False
    
    def validate_runner_script(self):
        """Validate PEST runner script."""
        logger.info("\n6. Checking PEST runner script...")
        
        runner_file = Path("pest_runner.py")
        if not self.check(runner_file.exists(), "pest_runner.py exists"):
            return False
        
        try:
            with open(runner_file, "r") as f:
                content = f.read()
            
            self.check("extract_landings_predictions" in content, "Contains prediction function")
            self.check("write_predictions" in content, "Contains output writing function")
            self.check("def main" in content, "Contains main function")
            
            return True
        except Exception as e:
            self.check(False, f"pest_runner.py readable ({e})")
            return False
    
    def validate_all(self):
        """Run all validation checks."""
        logger.info("=" * 70)
        logger.info("PEST Setup Validation")
        logger.info("=" * 70)
        
        self.validate_pst_file()
        self.validate_template_file()
        self.validate_instruction_file()
        self.validate_observations()
        self.validate_parameters()
        self.validate_runner_script()
        
        # Summary
        logger.info("\n" + "=" * 70)
        logger.info("Validation Summary")
        logger.info("=" * 70)
        logger.info(f"Checks passed: {self.checks_passed}/{self.checks_total}")
        
        if self.errors:
            logger.error(f"\n{len(self.errors)} errors found:")
            for error in self.errors:
                logger.error(f"  - {error}")
        
        if self.warnings:
            logger.warning(f"\n{len(self.warnings)} warnings:")
            for warning in self.warnings:
                logger.warning(f"  - {warning}")
        
        success = len(self.errors) == 0
        
        if success:
            logger.info("\n✓ All PEST files valid and ready for calibration!")
            logger.info("\nNext steps:")
            logger.info("  1. Review: PEST_CALIBRATION_GUIDE.md")
            logger.info("  2. Install PEST (see guide)")
            logger.info("  3. Run: python run_pest_calibration.py")
        else:
            logger.error("\n✗ Fix errors above before running PEST calibration")
        
        return success

def main():
    """Main function."""
    validator = PestSetupValidator()
    success = validator.validate_all()
    return 0 if success else 1

if __name__ == "__main__":
    sys.exit(main())
