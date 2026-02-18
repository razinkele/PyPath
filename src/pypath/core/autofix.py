"""Automatic parameter calibration and fixing for Ecosim stability.

This module provides diagnostic and automatic fixing routines to prevent
simulation crashes and improve model stability.
"""

import logging
from dataclasses import dataclass
from typing import Any, Dict, Tuple

import numpy as np

from .constants import (
    DEFAULT_PREY_SWITCHING_POWER,
    DIET_SUM_THRESHOLD,
    MAX_PREY_SWITCHING_POWER,
    MAX_QB_PB_RATIO,
    MAX_QQ_SAFE,
    MAX_VULNERABILITY_SAFE,
    MIN_BIOMASS_VIABLE,
    MIN_PREY_SWITCHING_POWER,
    MIN_QB_PB_RATIO,
)
from .ecopath import Rpath
from .ecosim import RsimParams, RsimScenario

# Get logger
logger = logging.getLogger(__name__)


@dataclass
class AutofixResult:
    """Results from automatic parameter fixing.

    Attributes
    ----------
    success : bool
        Whether fixes were successful
    fixes_applied : list
        List of fixes that were applied
    warnings : list
        List of warnings about potential issues
    original_params : dict
        Original parameter values before fixing
    """

    success: bool
    fixes_applied: list
    warnings: list
    original_params: dict


def diagnose_crash_causes(
    rpath: Rpath,
    params: RsimParams,
) -> Dict[str, Any]:
    """Diagnose potential causes of simulation crashes.

    Parameters
    ----------
    rpath : Rpath
        Balanced Ecopath model
    params : RsimParams
        Ecosim parameters

    Returns
    -------
    dict
        Diagnostic results with issues and recommendations
    """
    issues = {"critical": [], "warnings": [], "recommendations": []}

    # 1. Check for EE > 1 (overfishing/overconsumption)
    for i in range(1, rpath.NUM_LIVING + 1):
        if rpath.EE[i] > 1.0:
            issues["critical"].append(
                {
                    "type": "ee_too_high",
                    "group": i,
                    "value": rpath.EE[i],
                    "message": f"Group {i} ({rpath.Group[i]}): EE = {rpath.EE[i]:.3f} > 1.0",
                    "fix": "Reduce fishing mortality or consumption by predators",
                }
            )

    # 2. Check for very low initial biomass
    for i in range(1, rpath.NUM_LIVING + 1):
        if params.B_BaseRef[i] < 0.001:
            issues["warnings"].append(
                {
                    "type": "low_biomass",
                    "group": i,
                    "value": params.B_BaseRef[i],
                    "message": f"Group {i} ({rpath.Group[i]}): Very low biomass = {params.B_BaseRef[i]:.6f}",
                    "fix": "Increase initial biomass or remove group",
                }
            )

    # 3. Check for very high vulnerability (VV >> 2) - vectorized
    high_vv_mask = params.VV > MAX_VULNERABILITY_SAFE
    high_vv_indices = np.where(high_vv_mask)[0]
    for i in high_vv_indices:
        prey_idx = params.PreyFrom[i]
        pred_idx = params.PreyTo[i]
        issues["warnings"].append(
            {
                "type": "high_vulnerability",
                "link": i,
                "prey": prey_idx,
                "predator": pred_idx,
                "value": params.VV[i],
                "message": f"Link {i}: VV = {params.VV[i]:.2f} (very high vulnerability)",
                "fix": "Reduce vulnerability to prevent rapid depletion",
            }
        )

    # 4. Check for unrealistic QB/PB ratios - vectorized
    living_indices = np.arange(1, rpath.NUM_LIVING + 1)
    qb = np.array(rpath.QB[1 : rpath.NUM_LIVING + 1])
    pb = np.array(rpath.PB[1 : rpath.NUM_LIVING + 1])

    # Only check where both QB and PB are positive
    valid_mask = (qb > 0) & (pb > 0)
    qb_pb_ratio = np.divide(qb, pb, where=valid_mask, out=np.zeros_like(qb))

    # GE = PB/QB should be between 0.05 and 0.5 for most consumers
    unrealistic_mask = valid_mask & (
        (qb_pb_ratio < MIN_QB_PB_RATIO) | (qb_pb_ratio > MAX_QB_PB_RATIO)
    )
    unrealistic_indices = living_indices[unrealistic_mask]

    for idx, i in enumerate(unrealistic_indices):
        ratio = qb_pb_ratio[i - 1]  # Adjust index for 0-based array
        issues["warnings"].append(
            {
                "type": "unrealistic_qb_pb",
                "group": i,
                "qb": rpath.QB[i],
                "pb": rpath.PB[i],
                "ratio": ratio,
                "message": f"Group {i} ({rpath.Group[i]}): QB/PB = {ratio:.2f} (unusual)",
                "fix": "Check QB and PB values - GE should be 0.05-0.5",
            }
        )

    # 5. Check for very high QQ (density-dependent catchability) - vectorized
    high_qq_mask = params.QQ > MAX_QQ_SAFE
    high_qq_indices = np.where(high_qq_mask)[0]
    for i in high_qq_indices:
        prey_idx = params.PreyFrom[i]
        pred_idx = params.PreyTo[i]
        issues["recommendations"].append(
            {
                "type": "high_qq",
                "link": i,
                "value": params.QQ[i],
                "message": f"Link {i}: QQ = {params.QQ[i]:.2f} (strong density dependence)",
                "fix": "Consider reducing QQ to avoid rapid crashes",
            }
        )

    # 6. Check for missing prey (predator with no food) - vectorized
    # Identify consumers (QB > 0)
    consumer_mask = np.array([rpath.QB[i] > 0 for i in range(1, rpath.NUM_LIVING + 1)])
    consumer_indices = np.arange(1, rpath.NUM_LIVING + 1)[consumer_mask]

    # Calculate diet totals for all consumers at once
    for pred in consumer_indices:
        # Sum diet proportions (only living groups can be prey in DC)
        total_diet = np.sum(rpath.DC[pred, : rpath.NUM_LIVING])

        if total_diet < DIET_SUM_THRESHOLD:  # Diet should sum to ~1
            issues["critical"].append(
                {
                    "type": "incomplete_diet",
                    "group": pred,
                    "diet_sum": total_diet,
                    "message": f"Group {pred} ({rpath.Group[pred]}): Diet sums to {total_diet:.3f} < 1.0",
                    "fix": "Complete diet composition or add import",
                }
            )

    return issues


def autofix_parameters(
    rpath: Rpath, params: RsimParams, aggressive: bool = False
) -> Tuple[RsimParams, AutofixResult]:
    """Automatically fix parameters to improve stability.

    Parameters
    ----------
    rpath : Rpath
        Balanced Ecopath model
    params : RsimParams
        Ecosim parameters to fix
    aggressive : bool
        If True, apply more aggressive fixes

    Returns
    -------
    RsimParams
        Fixed parameters
    AutofixResult
        Summary of fixes applied
    """
    fixes_applied = []
    warnings = []
    original = {}

    # Make a copy to modify
    import copy

    fixed_params = copy.deepcopy(params)

    # Fix 1: Cap vulnerability at reasonable values
    max_vv = 5.0 if not aggressive else 3.0
    for i in range(len(fixed_params.VV)):
        if fixed_params.VV[i] > max_vv:
            original[f"VV_{i}"] = fixed_params.VV[i]
            fixed_params.VV[i] = max_vv
            fixes_applied.append(
                f"Capped VV[{i}] from {original[f'VV_{i}']:.2f} to {max_vv}"
            )

    # Fix 2: Ensure minimum biomass
    for i in range(1, rpath.NUM_LIVING + 1):
        if (
            fixed_params.B_BaseRef[i] < MIN_BIOMASS_VIABLE
            and fixed_params.B_BaseRef[i] > 0
        ):
            original[f"B_{i}"] = fixed_params.B_BaseRef[i]
            fixed_params.B_BaseRef[i] = MIN_BIOMASS_VIABLE
            fixes_applied.append(
                f"Increased B[{i}] from {original[f'B_{i}']:.6f} to {MIN_BIOMASS_VIABLE}"
            )

    # Fix 3: Reduce QQ for very strong density dependence
    max_qq = 3.0 if not aggressive else 2.0
    for i in range(len(fixed_params.QQ)):
        if fixed_params.QQ[i] > max_qq:
            original[f"QQ_{i}"] = fixed_params.QQ[i]
            fixed_params.QQ[i] = max_qq
            fixes_applied.append(
                f"Capped QQ[{i}] from {original[f'QQ_{i}']:.2f} to {max_qq}"
            )

    # Fix 4: Adjust DD (prey switching) for extreme values
    for i in range(len(fixed_params.DD)):
        if fixed_params.DD[i] > MAX_PREY_SWITCHING_POWER:
            original[f"DD_{i}"] = fixed_params.DD[i]
            fixed_params.DD[i] = (
                DEFAULT_PREY_SWITCHING_POWER  # More moderate prey switching
            )
            fixes_applied.append(
                f"Reduced DD[{i}] from {original[f'DD_{i}']:.2f} to {DEFAULT_PREY_SWITCHING_POWER}"
            )
        elif fixed_params.DD[i] < MIN_PREY_SWITCHING_POWER:
            original[f"DD_{i}"] = fixed_params.DD[i]
            fixed_params.DD[i] = 1.0
            fixes_applied.append(
                f"Increased DD[{i}] from {original[f'DD_{i}']:.2f} to 1.0"
            )

    # Fix 5: Warn about EE > 1 (can't fix in Ecosim params)
    for i in range(1, rpath.NUM_LIVING + 1):
        if rpath.EE[i] > 1.0:
            warnings.append(
                f"Group {i} ({rpath.Group[i]}): EE = {rpath.EE[i]:.3f} > 1.0 "
                "(requires Ecopath rebalancing)"
            )

    result = AutofixResult(
        success=len(warnings) == 0,
        fixes_applied=fixes_applied,
        warnings=warnings,
        original_params=original,
    )

    return fixed_params, result


def validate_and_fix_scenario(
    scenario: RsimScenario, rpath: Rpath, auto_fix: bool = True, verbose: bool = True
) -> Tuple[RsimScenario, Dict[str, Any]]:
    """Validate scenario and optionally apply automatic fixes.

    Parameters
    ----------
    scenario : RsimScenario
        Ecosim scenario to validate
    rpath : Rpath
        Original Ecopath model
    auto_fix : bool
        Whether to automatically fix parameters
    verbose : bool
        Whether to print diagnostic messages

    Returns
    -------
    RsimScenario
        Validated (and potentially fixed) scenario
    dict
        Diagnostic report
    """
    report = {"valid": True, "issues": [], "fixes": [], "warnings": []}

    # Diagnose issues
    diagnosis = diagnose_crash_causes(rpath, scenario.params)

    # Check for critical issues
    if diagnosis["critical"]:
        report["valid"] = False
        report["issues"] = diagnosis["critical"]

        if verbose:
            logger.warning("=" * 70)
            logger.warning("CRITICAL ISSUES DETECTED")
            logger.warning("=" * 70)
            for issue in diagnosis["critical"]:
                logger.warning(f"  • {issue['message']}")
                logger.warning(f"    Fix: {issue['fix']}")

    # Apply automatic fixes if requested
    if auto_fix and (diagnosis["critical"] or diagnosis["warnings"]):
        if verbose:
            logger.info("=" * 70)
            logger.info("APPLYING AUTOMATIC FIXES")
            logger.info("=" * 70)

        fixed_params, fix_result = autofix_parameters(rpath, scenario.params)

        if fix_result.fixes_applied:
            scenario.params = fixed_params
            report["fixes"] = fix_result.fixes_applied
            report["valid"] = fix_result.success

            if verbose:
                for fix in fix_result.fixes_applied:
                    logger.info(f"  ✓ {fix}")

        if fix_result.warnings:
            report["warnings"] = fix_result.warnings
            if verbose:
                logger.warning("WARNINGS:")
                for warning in fix_result.warnings:
                    logger.warning(f"  ⚠ {warning}")

    # Log recommendations
    if verbose and diagnosis["recommendations"]:
        logger.info("=" * 70)
        logger.info("RECOMMENDATIONS")
        logger.info("=" * 70)
        for rec in diagnosis["recommendations"]:
            logger.info(f"  • {rec['message']}")

    if verbose:
        logger.info("=" * 70)
        if report["valid"]:
            logger.info("VALIDATION: PASSED ✓")
        else:
            logger.warning("VALIDATION: FAILED - Manual fixes required")
        logger.info("=" * 70)

    return scenario, report
