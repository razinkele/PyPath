"""Automatic parameter calibration and fixing for Ecosim stability.

This module provides diagnostic and automatic fixing routines to prevent
simulation crashes and improve model stability.
"""

import numpy as np
from typing import Dict, Any, Tuple
from dataclasses import dataclass

from .ecopath import Rpath
from .ecosim import RsimScenario, RsimParams
from .params import RpathParams


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
    issues = {
        'critical': [],
        'warnings': [],
        'recommendations': []
    }

    # 1. Check for EE > 1 (overfishing/overconsumption)
    for i in range(1, rpath.NUM_LIVING + 1):
        if rpath.EE[i] > 1.0:
            issues['critical'].append({
                'type': 'ee_too_high',
                'group': i,
                'value': rpath.EE[i],
                'message': f"Group {i} ({rpath.Group[i]}): EE = {rpath.EE[i]:.3f} > 1.0",
                'fix': 'Reduce fishing mortality or consumption by predators'
            })

    # 2. Check for very low initial biomass
    for i in range(1, rpath.NUM_LIVING + 1):
        if params.B_BaseRef[i] < 0.001:
            issues['warnings'].append({
                'type': 'low_biomass',
                'group': i,
                'value': params.B_BaseRef[i],
                'message': f"Group {i} ({rpath.Group[i]}): Very low biomass = {params.B_BaseRef[i]:.6f}",
                'fix': 'Increase initial biomass or remove group'
            })

    # 3. Check for very high vulnerability (VV >> 2)
    for i in range(len(params.VV)):
        if params.VV[i] > 10.0:
            prey_idx = params.PreyFrom[i]
            pred_idx = params.PreyTo[i]
            issues['warnings'].append({
                'type': 'high_vulnerability',
                'link': i,
                'prey': prey_idx,
                'predator': pred_idx,
                'value': params.VV[i],
                'message': f"Link {i}: VV = {params.VV[i]:.2f} (very high vulnerability)",
                'fix': 'Reduce vulnerability to prevent rapid depletion'
            })

    # 4. Check for unrealistic QB/PB ratios
    for i in range(1, rpath.NUM_LIVING + 1):
        if rpath.QB[i] > 0 and rpath.PB[i] > 0:
            qb_pb_ratio = rpath.QB[i] / rpath.PB[i]
            # GE = PB/QB should be between 0.05 and 0.5 for most consumers
            if qb_pb_ratio < 2.0 or qb_pb_ratio > 20.0:
                issues['warnings'].append({
                    'type': 'unrealistic_qb_pb',
                    'group': i,
                    'qb': rpath.QB[i],
                    'pb': rpath.PB[i],
                    'ratio': qb_pb_ratio,
                    'message': f"Group {i} ({rpath.Group[i]}): QB/PB = {qb_pb_ratio:.2f} (unusual)",
                    'fix': 'Check QB and PB values - GE should be 0.05-0.5'
                })

    # 5. Check for very high QQ (density-dependent catchability)
    for i in range(len(params.QQ)):
        if params.QQ[i] > 5.0:
            prey_idx = params.PreyFrom[i]
            pred_idx = params.PreyTo[i]
            issues['recommendations'].append({
                'type': 'high_qq',
                'link': i,
                'value': params.QQ[i],
                'message': f"Link {i}: QQ = {params.QQ[i]:.2f} (strong density dependence)",
                'fix': 'Consider reducing QQ to avoid rapid crashes'
            })

    # 6. Check for missing prey (predator with no food)
    for pred in range(1, rpath.NUM_LIVING + 1):
        if rpath.QB[pred] > 0:  # Is a consumer
            total_diet = 0
            for prey in range(rpath.NUM_LIVING):  # Only living groups can be prey in DC
                total_diet += rpath.DC[pred, prey]

            if total_diet < 0.9:  # Diet should sum to ~1
                issues['critical'].append({
                    'type': 'incomplete_diet',
                    'group': pred,
                    'diet_sum': total_diet,
                    'message': f"Group {pred} ({rpath.Group[pred]}): Diet sums to {total_diet:.3f} < 1.0",
                    'fix': 'Complete diet composition or add import'
                })

    return issues


def autofix_parameters(
    rpath: Rpath,
    params: RsimParams,
    aggressive: bool = False
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
            original[f'VV_{i}'] = fixed_params.VV[i]
            fixed_params.VV[i] = max_vv
            fixes_applied.append(f"Capped VV[{i}] from {original[f'VV_{i}']:.2f} to {max_vv}")

    # Fix 2: Ensure minimum biomass
    min_biomass = 0.001
    for i in range(1, rpath.NUM_LIVING + 1):
        if fixed_params.B_BaseRef[i] < min_biomass and fixed_params.B_BaseRef[i] > 0:
            original[f'B_{i}'] = fixed_params.B_BaseRef[i]
            fixed_params.B_BaseRef[i] = min_biomass
            fixes_applied.append(f"Increased B[{i}] from {original[f'B_{i}']:.6f} to {min_biomass}")

    # Fix 3: Reduce QQ for very strong density dependence
    max_qq = 3.0 if not aggressive else 2.0
    for i in range(len(fixed_params.QQ)):
        if fixed_params.QQ[i] > max_qq:
            original[f'QQ_{i}'] = fixed_params.QQ[i]
            fixed_params.QQ[i] = max_qq
            fixes_applied.append(f"Capped QQ[{i}] from {original[f'QQ_{i}']:.2f} to {max_qq}")

    # Fix 4: Adjust DD (prey switching) for extreme values
    for i in range(len(fixed_params.DD)):
        if fixed_params.DD[i] > 5.0:
            original[f'DD_{i}'] = fixed_params.DD[i]
            fixed_params.DD[i] = 2.0  # More moderate prey switching
            fixes_applied.append(f"Reduced DD[{i}] from {original[f'DD_{i}']:.2f} to 2.0")
        elif fixed_params.DD[i] < 0.1:
            original[f'DD_{i}'] = fixed_params.DD[i]
            fixed_params.DD[i] = 1.0
            fixes_applied.append(f"Increased DD[{i}] from {original[f'DD_{i}']:.2f} to 1.0")

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
        original_params=original
    )

    return fixed_params, result


def validate_and_fix_scenario(
    scenario: RsimScenario,
    rpath: Rpath,
    auto_fix: bool = True,
    verbose: bool = True
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
    report = {
        'valid': True,
        'issues': [],
        'fixes': [],
        'warnings': []
    }

    # Diagnose issues
    diagnosis = diagnose_crash_causes(rpath, scenario.params)

    # Check for critical issues
    if diagnosis['critical']:
        report['valid'] = False
        report['issues'] = diagnosis['critical']

        if verbose:
            print("=" * 70)
            print("CRITICAL ISSUES DETECTED")
            print("=" * 70)
            for issue in diagnosis['critical']:
                print(f"  • {issue['message']}")
                print(f"    Fix: {issue['fix']}")

    # Apply automatic fixes if requested
    if auto_fix and (diagnosis['critical'] or diagnosis['warnings']):
        if verbose:
            print("\n" + "=" * 70)
            print("APPLYING AUTOMATIC FIXES")
            print("=" * 70)

        fixed_params, fix_result = autofix_parameters(rpath, scenario.params)

        if fix_result.fixes_applied:
            scenario.params = fixed_params
            report['fixes'] = fix_result.fixes_applied
            report['valid'] = fix_result.success

            if verbose:
                for fix in fix_result.fixes_applied:
                    print(f"  ✓ {fix}")

        if fix_result.warnings:
            report['warnings'] = fix_result.warnings
            if verbose:
                print("\nWARNINGS:")
                for warning in fix_result.warnings:
                    print(f"  ⚠ {warning}")

    # Print recommendations
    if verbose and diagnosis['recommendations']:
        print("\n" + "=" * 70)
        print("RECOMMENDATIONS")
        print("=" * 70)
        for rec in diagnosis['recommendations']:
            print(f"  • {rec['message']}")

    if verbose:
        print("\n" + "=" * 70)
        if report['valid']:
            print("VALIDATION: PASSED ✓")
        else:
            print("VALIDATION: FAILED - Manual fixes required")
        print("=" * 70)

    return scenario, report
