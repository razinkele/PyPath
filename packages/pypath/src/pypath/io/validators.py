"""Validation and diagnostic checks for EwE models.

Provides energy balance checks, sanity validation, and cross-file
consistency verification for Ecopath and Ecosim models.

Functions:
- validate_ecopath_scenario(): Check Ecopath CSV scenario for issues
- validate_diet_composition(): Verify diet matrix consistency
- validate_detritus_fate(): Check detritus fate matrix
- check_energy_balance(): Validate energy balance closure
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import List, Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


@dataclass
class ValidationIssue:
    """Single validation issue/warning/error."""

    level: str  # "error", "warning", "info"
    category: str  # "energy_balance", "sanity", "consistency", etc.
    message: str
    group: Optional[str] = None
    value: Optional[float] = None

    def __str__(self) -> str:
        level_icon = {"error": "❌", "warning": "⚠️", "info": "ℹ️"}[self.level]
        if self.group:
            return f"{level_icon} {self.level.upper()}: [{self.group}] {self.message}"
        return f"{level_icon} {self.level.upper()}: {self.message}"


@dataclass
class ValidationReport:
    """Summary of validation results."""

    file_path: Optional[str] = None
    issues: List[ValidationIssue] = field(default_factory=list)
    passed: bool = True

    @property
    def errors(self) -> List[ValidationIssue]:
        return [i for i in self.issues if i.level == "error"]

    @property
    def warnings(self) -> List[ValidationIssue]:
        return [i for i in self.issues if i.level == "warning"]

    @property
    def infos(self) -> List[ValidationIssue]:
        return [i for i in self.issues if i.level == "info"]

    def add_issue(
        self,
        level: str,
        category: str,
        message: str,
        group: Optional[str] = None,
        value: Optional[float] = None,
    ) -> None:
        """Add an issue to the report."""
        issue = ValidationIssue(level, category, message, group, value)
        self.issues.append(issue)
        if level == "error":
            self.passed = False

    def summary(self) -> str:
        """Return human-readable summary."""
        lines = [f"Validation Report: {self.file_path or 'Unknown'}"]
        if self.passed:
            lines.append("✅ PASSED - No errors found")
        else:
            lines.append("❌ FAILED - Errors detected")

        if self.errors:
            lines.append(f"\n{len(self.errors)} Errors:")
            for issue in self.errors:
                lines.append(f"  {issue}")

        if self.warnings:
            lines.append(f"\n{len(self.warnings)} Warnings:")
            for issue in self.warnings[:5]:  # Show first 5
                lines.append(f"  {issue}")
            if len(self.warnings) > 5:
                lines.append(f"  ... and {len(self.warnings) - 5} more warnings")

        return "\n".join(lines)


def _as_number(value) -> Optional[float]:
    """``value`` as a float: NaN when missing, None when it is not a number."""
    if value is None or (not isinstance(value, str) and pd.isna(value)):
        return np.nan
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _scenario_labels(scenario_df: pd.DataFrame) -> Optional[List[str]]:
    """Group names of a scenario: its Group column, else its index labels."""
    if "Group" in scenario_df.columns:
        return [str(g) for g in scenario_df["Group"]]
    if pd.api.types.is_integer_dtype(scenario_df.index):
        return None  # row numbers, not names
    return [str(g) for g in scenario_df.index]


def _unknown_labels(labels, group_names: List[str], extra=()) -> List[str]:
    """Labels that are neither group names nor one of ``extra`` (any case)."""
    known = {str(g) for g in group_names}
    allowed = {e.lower() for e in extra}
    return [
        str(label)
        for label in labels
        if str(label) not in known and str(label).lower() not in allowed
    ]


def validate_ecopath_scenario(
    scenario_df: pd.DataFrame,
    group_names: Optional[List[str]] = None,
) -> ValidationReport:
    """Validate Ecopath scenario data.

    Checks for:
    - Missing critical columns (Biomass, P/B, Q/B)
    - Values that are not numbers, and negative biomass (-999 = missing)
    - Physiologically unrealistic values (P/B, Q/B ranges)
    - Group names: the scenario's (Group column, else row labels) must be
      the expected ones

    Args:
        scenario_df: DataFrame with scenario data (rows = groups)
        group_names: Expected group names for consistency check

    Returns:
        ValidationReport with issues found
    """
    report = ValidationReport()

    if scenario_df.empty:
        report.add_issue("error", "data", "Scenario DataFrame is empty")
        return report

    # Check required columns
    required_cols = ["Biomass", "PB", "QB"]
    missing = [col for col in required_cols if col not in scenario_df.columns]
    if missing:
        report.add_issue(
            "error",
            "consistency",
            f"Missing required columns: {missing}",
        )
        return report

    labels = _scenario_labels(scenario_df)

    # Check group names if provided
    if group_names is not None:
        if len(scenario_df) != len(group_names):
            report.add_issue(
                "warning",
                "consistency",
                f"Row count mismatch: {len(scenario_df)} rows vs "
                f"{len(group_names)} group names",
            )
        if labels is not None:
            expected = [str(g) for g in group_names]
            missing = [g for g in expected if g not in set(labels)]
            unexpected = [g for g in labels if g not in set(expected)]
            if missing or unexpected:
                report.add_issue(
                    "error",
                    "consistency",
                    f"Scenario groups differ from the expected ones: missing "
                    f"{missing}, unexpected {unexpected}",
                )

    # Validate each group
    for position, (index_label, row) in enumerate(scenario_df.iterrows()):
        group_name = labels[position] if labels is not None else index_label
        values = {}
        for column in required_cols:
            raw = row.get(column)
            values[column] = _as_number(raw)
            if values[column] is None:
                report.add_issue(
                    "error",
                    "data",
                    f"{column} value {raw!r} is not a number",
                    group=str(group_name),
                )
        biomass, pb, qb = values["Biomass"], values["PB"], values["QB"]

        if biomass is not None:
            if np.isnan(biomass) or biomass == -999:
                report.add_issue(
                    "warning",
                    "sanity",
                    "Missing biomass value",
                    group=str(group_name),
                )
            elif biomass < 0:
                report.add_issue(
                    "error",
                    "sanity",
                    f"Negative biomass {biomass:g}",
                    group=str(group_name),
                    value=biomass,
                )

        # Check P/B range (0.01 - 100 year^-1 typical)
        if pb is not None and not np.isnan(pb) and pb != -999:
            if pb < 0.001 or pb > 100:
                report.add_issue(
                    "warning",
                    "sanity",
                    f"P/B value {pb} outside typical range [0.001, 100]",
                    group=str(group_name),
                    value=pb,
                )

        # Check Q/B range (0.1 - 100 year^-1 typical)
        if qb is not None and not np.isnan(qb) and qb != -999:
            if qb < 0.01 or qb > 100:
                report.add_issue(
                    "warning",
                    "sanity",
                    f"Q/B value {qb} outside typical range [0.01, 100]",
                    group=str(group_name),
                    value=qb,
                )

    report.passed = len(report.errors) == 0
    return report


def validate_diet_composition(
    diet_df: pd.DataFrame,
    group_names: Optional[List[str]] = None,
    tolerance: float = 0.1,
) -> ValidationReport:
    """Validate diet composition matrix.

    Checks for:
    - Missing or non-numeric diet values
    - Sum of diet fractions ≈ 1 for each predator
    - Diet values in range [0, 1]
    - Labels: prey rows and predator columns must be in ``group_names``
      (rows may also hold ``Import``)

    Args:
        diet_df: Diet composition matrix (rows = prey, cols = predators)
        group_names: List of group names for validation
        tolerance: Allowed deviation from 1.0 for diet sums (default: 0.1)

    Returns:
        ValidationReport with issues
    """
    report = ValidationReport()

    if diet_df.empty:
        report.add_issue("error", "data", "Diet composition matrix is empty")
        return report

    values = diet_df.apply(pd.to_numeric, errors="coerce")
    missing = int(values.isna().sum().sum())
    if missing:
        report.add_issue(
            "error", "data", f"{missing} diet values are missing or not numbers"
        )
    if group_names is not None:
        for what, labels, extra in (
            ("prey", diet_df.index, ("Import",)),
            ("predator", diet_df.columns, ()),
        ):
            unknown = _unknown_labels(labels, group_names, extra)
            if unknown:
                report.add_issue(
                    "error", "consistency", f"Unknown {what} labels: {unknown}"
                )

    # Check that all values are in [0, 1]
    invalid_vals = ((values < 0) | (values > 1)).sum().sum()
    if invalid_vals > 0:
        report.add_issue(
            "error",
            "consistency",
            f"Found {invalid_vals} diet values outside [0, 1] range",
        )

    # Check diet sums per predator (column)
    col_sums = values.sum(axis=0)
    for col_name, col_sum in col_sums.items():
        # Skip if NaN or all zeros
        if pd.isna(col_sum) or col_sum == 0:
            continue

        # Allow slight deviation from 1.0
        if not (1.0 - tolerance <= col_sum <= 1.0 + tolerance):
            report.add_issue(
                "warning",
                "consistency",
                f"Diet sum {col_sum:.3f} deviates from 1.0 by {abs(col_sum - 1.0):.3f}",
                group=str(col_name),
                value=col_sum,
            )

    report.passed = len(report.errors) == 0
    return report


def validate_detritus_fate(
    fate_df: pd.DataFrame,
    group_names: Optional[List[str]] = None,
) -> ValidationReport:
    """Validate detritus fate matrix.

    Checks for:
    - Missing or non-numeric fate values
    - Fate fractions sum to 1 per group
    - Fate values in [0, 1]
    - Labels: rows and detritus columns must be in ``group_names``
      (columns may also hold ``Export``)

    Args:
        fate_df: Detritus fate matrix
        group_names: Expected group names

    Returns:
        ValidationReport with issues
    """
    report = ValidationReport()

    if fate_df.empty:
        report.add_issue("error", "data", "Detritus fate matrix is empty")
        return report

    values = fate_df.apply(pd.to_numeric, errors="coerce")
    missing = int(values.isna().sum().sum())
    if missing:
        report.add_issue(
            "error", "data", f"{missing} fate values are missing or not numbers"
        )
    if group_names is not None:
        for what, labels, extra in (
            ("group", fate_df.index, ()),
            ("detritus", fate_df.columns, ("Export",)),
        ):
            unknown = _unknown_labels(labels, group_names, extra)
            if unknown:
                report.add_issue(
                    "error", "consistency", f"Unknown {what} labels: {unknown}"
                )

    # Check that all values are in [0, 1]
    invalid_vals = ((values < 0) | (values > 1)).sum().sum()
    if invalid_vals > 0:
        report.add_issue(
            "error",
            "consistency",
            f"Found {invalid_vals} fate values outside [0, 1] range",
        )

    # Check that each row sums to 1
    row_sums = values.sum(axis=1)
    for row_name, row_sum in row_sums.items():
        if pd.isna(row_sum):
            continue
        if not (0.99 <= row_sum <= 1.01):  # Allow 1% tolerance
            report.add_issue(
                "warning",
                "consistency",
                f"Fate sum {row_sum:.3f} deviates from 1.0",
                group=str(row_name),
                value=row_sum,
            )

    report.passed = len(report.errors) == 0
    return report


def check_energy_balance(
    biomass: np.ndarray,
    production: np.ndarray,
    consumption: np.ndarray,
    import_vec: np.ndarray,
    export_vec: np.ndarray,
    tolerance: float = 0.01,
) -> ValidationReport:
    """Check energy balance closure.

    For a balanced ecosystem:
    sum(Imports) ≈ sum(Exports) ±tolerance

    Args:
        biomass: Biomass vector (n_groups,)
        production: Production vector (n_groups,)
        consumption: Consumption vector (n_groups,)
        import_vec: Import vector (n_groups,)
        export_vec: Export vector (n_groups,)
        tolerance: Acceptable relative error (default: 0.01 = 1%)

    Returns:
        ValidationReport with balance assessment
    """
    report = ValidationReport()

    # Filter out detritus groups and missing values
    valid_idx = (biomass > 0) & (~np.isnan(biomass))

    total_import = np.nansum(import_vec[valid_idx])
    total_export = np.nansum(export_vec[valid_idx])

    if total_import == 0 and total_export == 0:
        report.add_issue(
            "info",
            "energy_balance",
            "No imports or exports detected (closed system)",
        )
        return report

    # Check balance
    if total_import > 0:
        error = abs(total_import - total_export) / total_import
    else:
        error = abs(total_export)

    is_balanced = error <= tolerance

    msg = (
        f"Energy balance: imports={total_import:.2e}, "
        f"exports={total_export:.2e}, error={error:.2%}"
    )

    if is_balanced:
        report.add_issue("info", "energy_balance", f"✅ {msg}")
    else:
        report.add_issue(
            "warning",
            "energy_balance",
            f"⚠️ Balance error exceeds tolerance: {msg}",
        )

    return report


def validate_all_ecopath_files(
    scenario_df: pd.DataFrame,
    diet_df: pd.DataFrame,
    fate_df: pd.DataFrame,
    group_names: Optional[List[str]] = None,
) -> ValidationReport:
    """Run all validation checks on Ecopath input files.

    Combines scenario, diet, and fate validation into single report.

    Args:
        scenario_df: Ecopath scenario data
        diet_df: Diet composition matrix
        fate_df: Detritus fate matrix
        group_names: Group names for consistency checks

    Returns:
        Combined ValidationReport
    """
    report = ValidationReport()

    # Validate each component
    scenario_report = validate_ecopath_scenario(scenario_df, group_names)
    diet_report = validate_diet_composition(diet_df, group_names)
    fate_report = validate_detritus_fate(fate_df, group_names)

    # Merge reports
    report.issues.extend(scenario_report.issues)
    report.issues.extend(diet_report.issues)
    report.issues.extend(fate_report.issues)

    report.passed = scenario_report.passed and diet_report.passed and fate_report.passed

    return report
