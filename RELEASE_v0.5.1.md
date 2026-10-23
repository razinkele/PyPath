# PyPath v0.5.1 Release Summary

## Release Information
- **Version**: 0.5.1
- **Release Date**: 2024-10-23
- **Type**: Patch release (feature addition)

## Release Tags
- pypath-ewe-v0.5.1 - Core library
- pypath-shiny-v0.5.1 - Web application

## Major Additions in v0.5.1

### 1. PEST Calibration Framework
- Gauss-Levenberg-Marquardt (GLM) algorithm with Marquardt damping
- Differential Evolution (DE) option for multimodal problems
- Parameter bounds enforcement
- Uncertainty quantification (covariance, std dev, correlation)
- Finite-difference Jacobian computation
- Parameter sensitivity analysis
- Convergence monitoring

Files added:
- packages/pypath/src/pypath/core/pest_calibration.py (500 lines)
- packages/pypath/tests/test_pest_calibration.py (250 lines, 13 tests passing)

### 2. Curonian Lagoon Ecosim Example
- Production-ready calibration workflow
- Real fishery data integration
- 4 management scenarios
- Complete documentation

Files added:
- scripts/pest_calibration_curonian_lagoon.py (350 lines)

### 3. Comprehensive Documentation
- PEST_CALIBRATION_GUIDE.md (400+ lines) - Complete guide with theory and examples
- PEST_IMPLEMENTATION_SUMMARY.md (300+ lines) - Architecture and integration details
- Updated README.md with PEST references

## Installation

### Conda
\\\ash
# Core package
conda install pypath-ewe=0.5.1 -c razinka -c conda-forge

# With web dashboard
conda install pypath-shiny=0.5.1 -c razinka -c conda-forge
\\\

### From source
\\\ash
git clone https://github.com/razinkele/PyPath.git
cd PyPath
git checkout pypath-ewe-v0.5.1
pip install -e "packages/pypath[all]"
\\\

## Test Results
- Core package tests: All passing
- PEST module tests: 13/13 passing
- Integration tests: Successful

## Git Commits Included
- fca3e3c chore: Bump version to 0.5.1 for PEST calibration release
- 4ed737d docs: Add PEST calibration to main README and documentation
- 55f5f97 feat: Add PEST calibration framework for Ecosim models
- 4047167 docs: Document known numerical instability in Curonian Lagoon example
- 790fc92 fix: Improve Curonian Lagoon Ecosim example with better error handling

## Next Steps for Deployment

To deploy v0.5.1 to laguna server:

1. SSH to laguna.ku.lt
2. Activate conda environment: \conda activate pypath-env\
3. Update package: \conda install pypath-shiny=0.5.1 -c razinka -c conda-forge\
4. Restart Shiny server: \sudo systemctl restart shiny-server\
5. Verify: Visit https://laguna.ku.lt/pypath

## Breaking Changes
None - Fully backward compatible with v0.5.0

## Known Issues
See CURONIAN_ECOSIM_README.md for known numerical instabilities in Curonian Lagoon model.

## Contact & Support
- Repository: https://github.com/razinkele/PyPath
- Documentation: https://razinkele.github.io/PyPath/
- Issue tracker: https://github.com/razinkele/PyPath/issues
