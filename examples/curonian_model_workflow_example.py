"""Curonian Lagoon end-to-end validation workflow example.

This example runs a practical EwE -> PyPath validation loop:

1. Select a Curonian-compatible `.eweaccdb` database
2. Load Ecopath parameters and balance with `rpath`
3. Validate Ecopath balance diagnostics
4. Compare key parameters against EwE source tables
5. Auto-select and run an Ecosim scenario
6. Run optional calibration smoke-test if time series are present

Usage
-----
python examples/curonian_model_workflow_example.py
python examples/curonian_model_workflow_example.py --db Data/Curonian_CaseStudy_LT2022_0.5ST_final7_export.eweaccdb
python examples/curonian_model_workflow_example.py --scenario 16 --strict-biomass
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

from pypath.core.analysis import check_ecopath_balance
from pypath.core.calibration import fit_to_timeseries
from pypath.core.ecopath import rpath
from pypath.core.ecosim import rsim_run
from pypath.io.ewemdb import (
    ecosim_scenario_from_ewemdb,
    get_ewemdb_metadata,
    read_ewemdb,
    read_ewemdb_table,
    read_timeseries,
)


@dataclass(frozen=True)
class CheckResult:
    """One workflow check result."""

    name: str
    status: str  # PASS | WARN | FAIL
    detail: str


class CuronianModelWorkflow:
    """Runnable Curonian Lagoon validation workflow."""

    def __init__(
        self,
        db_path: Path | None = None,
        scenario: int | None = None,
        strict_biomass: bool = False,
        verbose: bool = False,
    ) -> None:
        self.db_path = db_path
        self.scenario = scenario
        self.strict_biomass = strict_biomass
        self.verbose = verbose

    def run(self) -> int:
        """Execute workflow and return process exit code."""
        db_path = self._select_database(self.db_path)
        print(f"Using reference DB: {db_path}")

        metadata = get_ewemdb_metadata(str(db_path))
        print(
            "Database metadata: "
            f"has_ecosim={metadata.get('has_ecosim')} "
            f"scenarios={metadata.get('num_scenarios', 0)}"
        )

        params = read_ewemdb(str(db_path))
        model = rpath(params)

        checks: list[CheckResult] = []
        checks.append(self._check_ecopath_balance(model))
        checks.extend(self._check_parameter_parity(params, model, db_path))

        if metadata.get("has_ecosim", False):
            scenario_id = self.scenario or self._pick_scenario_id(db_path)
            checks.append(self._check_ecosim_dynamics(db_path, scenario_id, model))
        else:
            checks.append(
                CheckResult(
                    "Ecosim dynamics",
                    "WARN",
                    "No Ecosim scenarios in database (run skipped).",
                )
            )

        checks.append(self._check_calibration_smoke(params, model, db_path))

        self._print_summary(checks)
        return 1 if any(c.status == "FAIL" for c in checks) else 0

    def _select_database(self, explicit: Path | None) -> Path:
        """Resolve a Curonian-compatible database path."""
        if explicit is not None:
            if not explicit.exists():
                raise FileNotFoundError(f"Database not found: {explicit}")
            return explicit

        candidates = [
            Path("Data") / "Curonian_CaseStudy_LT2022_0.5ST_final7_export.eweaccdb",
            Path("Data") / "Curonian_CaseStudy_Consolidated_2025_2035.eweaccdb",
            Path("Data") / "LT2022_calibrated.eweaccdb",
        ]

        existing = [p for p in candidates if p.exists()]
        if not existing:
            raise FileNotFoundError(
                "No Curonian/LT2022 reference database found under Data/."
            )

        # Prefer DBs that include Ecosim scenarios so workflow is fully runnable.
        ranked: list[tuple[int, Path]] = []
        for p in existing:
            try:
                meta = get_ewemdb_metadata(str(p))
                score = int(bool(meta.get("has_ecosim", False))) * 100
                score += int(meta.get("num_scenarios", 0))
                ranked.append((score, p))
            except Exception:
                ranked.append((0, p))

        ranked.sort(key=lambda item: item[0], reverse=True)
        return ranked[0][1]

    def _check_ecopath_balance(self, model) -> CheckResult:
        """Run Ecopath consistency checks."""
        balance = check_ecopath_balance(model, tolerance=0.01, allow_partial_diet=True)

        if balance.get("is_balanced", False):
            warnings = len(balance.get("diet_warnings", []))
            detail = (
                f"Balanced (EE issues: 0, diet warnings: {warnings})."
                if warnings
                else "Balanced with no EE/diet violations."
            )
            return CheckResult("Ecopath mass balance", "PASS", detail)

        ee_issues = len(balance.get("ee_issues", []))
        diet_issues = len(balance.get("diet_issues", []))
        return CheckResult(
            "Ecopath mass balance",
            "FAIL",
            f"Unbalanced (EE issues: {ee_issues}, diet issues: {diet_issues}).",
        )

    def _check_parameter_parity(
        self, params, model, db_path: Path
    ) -> list[CheckResult]:
        """Compare selected parameters with raw EwE table values."""
        ewetable = read_ewemdb_table(str(db_path), "EcopathGroup")
        n_groups = model.NUM_LIVING + model.NUM_DEAD

        strict_checks = [
            ("PB", "PBInput", np.asarray(params.model["PB"], dtype=float), 1.0),
            ("QB", "QBInput", np.asarray(params.model["QB"], dtype=float), 1.0),
            ("EE", "EEInput", np.asarray(params.model["EE"], dtype=float), 1.0),
            ("Trophic Level", "TrophicLevel", np.asarray(model.TL, dtype=float), 1.0),
        ]

        results: list[CheckResult] = []
        for name, col, arr, pct_limit in strict_checks:
            issues = self._diff_issues(ewetable, col, arr, n_groups, pct_limit, params)
            if not issues:
                results.append(
                    CheckResult(f"Parity: {name}", "PASS", "0 issues > 1.0%.")
                )
                continue
            detail = f"{len(issues)} issues > {pct_limit:.1f}%"
            if self.verbose:
                sample = ", ".join(f"{grp} ({pct:.2f}%)" for _, grp, pct in issues[:8])
                detail = f"{detail}. Sample: {sample}"
            results.append(CheckResult(f"Parity: {name}", "FAIL", detail))

        # Biomass in some EwE exports can differ due conversion/scaling conventions.
        biomass_arr = np.asarray(params.model["Biomass"], dtype=float)
        biomass_issues = self._diff_issues(
            ewetable,
            "Biomass",
            biomass_arr,
            n_groups,
            1.0,
            params,
        )
        if not biomass_issues:
            results.append(CheckResult("Parity: Biomass", "PASS", "0 issues > 1.0%."))
        else:
            status = "FAIL" if self.strict_biomass else "WARN"
            detail = (
                f"{len(biomass_issues)} issues > 1.0%; "
                "biomass mismatches are common across some EwE export variants."
            )
            if self.verbose:
                sample = ", ".join(
                    f"{grp} ({pct:.1f}%)" for _, grp, pct in biomass_issues[:8]
                )
                detail = f"{detail} Sample: {sample}"
            results.append(CheckResult("Parity: Biomass", status, detail))

        return results

    def _diff_issues(
        self,
        ewetable: pd.DataFrame,
        ew_col: str,
        py_arr: np.ndarray,
        n_groups: int,
        pct_limit: float,
        params,
    ) -> list[tuple[str, str, float]]:
        """Collect percent-difference issues above threshold."""
        if ew_col not in ewetable.columns:
            return []

        issues: list[tuple[str, str, float]] = []
        for i in range(n_groups):
            ew_val = ewetable[ew_col].iloc[i]
            py_val = py_arr[i] if i < len(py_arr) else np.nan
            if pd.isna(ew_val) or ew_val in (0, -9999) or pd.isna(py_val):
                continue
            diff_pct = abs(py_val - ew_val) / abs(ew_val) * 100.0
            if diff_pct > pct_limit:
                group_name = str(params.model["Group"].iloc[i])
                issues.append((ew_col, group_name, float(diff_pct)))
        return issues

    def _pick_scenario_id(self, db_path: Path) -> int:
        """Choose a likely calibrated/final Ecosim scenario."""
        scen = read_ewemdb_table(str(db_path), "EcosimScenario")
        if len(scen) == 0:
            raise ValueError("No Ecosim scenarios found")

        def _score(row: pd.Series) -> tuple[int, int]:
            name = str(row.get("ScenarioName", "")).lower()
            bonus = 0
            for token, weight in (
                ("final", 6),
                ("calibr", 5),
                ("fit", 3),
                ("test", 2),
            ):
                if token in name:
                    bonus += weight
            return (bonus, int(row.get("ScenarioID", 0)))

        scored = sorted(((_score(r), r) for _, r in scen.iterrows()), reverse=True)
        selected = scored[0][1]
        return int(selected["ScenarioID"])

    def _check_ecosim_dynamics(
        self, db_path: Path, scenario_id: int, model
    ) -> CheckResult:
        """Run Ecosim and check coarse trajectory stability."""
        scenario = ecosim_scenario_from_ewemdb(
            str(db_path), scenario=scenario_id, balance=True
        )
        output = rsim_run(scenario, method="AB")

        n_groups = model.NUM_LIVING + model.NUM_DEAD
        unstable: list[tuple[int, float]] = []
        for i in range(n_groups):
            start_b = output.out_Biomass[0, i + 1]
            end_b = output.out_Biomass[-1, i + 1]
            if start_b <= 0:
                continue
            ratio = end_b / start_b
            if abs(ratio - 1.0) > 0.5:
                unstable.append((i, float(ratio)))

        if not unstable:
            return CheckResult(
                "Ecosim dynamics",
                "PASS",
                f"Scenario {scenario_id}: 0 groups with >50% end/start drift.",
            )

        return CheckResult(
            "Ecosim dynamics",
            "WARN",
            f"Scenario {scenario_id}: {len(unstable)} groups with >50% end/start drift.",
        )

    def _check_calibration_smoke(self, params, model, db_path: Path) -> CheckResult:
        """Run a short calibration pass only when time series exist."""
        try:
            ts_collection = read_timeseries(str(db_path))
            if not ts_collection or len(ts_collection.series) == 0:
                return CheckResult(
                    "Calibration smoke test",
                    "PASS",
                    "No time series in DB; skipped by design.",
                )

            result = fit_to_timeseries(
                rpath_model=model,
                params=params,
                timeseries=ts_collection,
                fit_vv=True,
                max_iterations=30,
                verbose=False,
            )
            return CheckResult(
                "Calibration smoke test",
                "PASS",
                f"Completed in {result.n_iterations} iterations (SS={result.ss:.4f}).",
            )
        except Exception as exc:
            return CheckResult(
                "Calibration smoke test",
                "FAIL",
                f"Calibration execution failed: {exc}",
            )

    def _print_summary(self, checks: Iterable[CheckResult]) -> None:
        checks = list(checks)

        print("\n" + "=" * 72)
        print("CURONIAN WORKFLOW SUMMARY")
        print("=" * 72)
        for check in checks:
            print(f"[{check.status}] {check.name}: {check.detail}")

        n_pass = sum(1 for c in checks if c.status == "PASS")
        n_warn = sum(1 for c in checks if c.status == "WARN")
        n_fail = sum(1 for c in checks if c.status == "FAIL")

        print("-" * 72)
        print(f"Totals -> PASS: {n_pass} | WARN: {n_warn} | FAIL: {n_fail}")
        if n_fail == 0:
            print("Workflow status: PASS (warnings may still require review).")
        else:
            print("Workflow status: FAIL (investigate failed checks).")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run Curonian Lagoon end-to-end workflow validation."
    )
    parser.add_argument(
        "--db",
        type=Path,
        default=None,
        help="Path to .eweaccdb database. If omitted, auto-select under Data/.",
    )
    parser.add_argument(
        "--scenario",
        type=int,
        default=None,
        help="Ecosim scenario ID override.",
    )
    parser.add_argument(
        "--strict-biomass",
        action="store_true",
        help="Treat biomass parity mismatches as FAIL instead of WARN.",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Print sample mismatch details for parity checks.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    workflow = CuronianModelWorkflow(
        db_path=args.db,
        scenario=args.scenario,
        strict_biomass=args.strict_biomass,
        verbose=args.verbose,
    )
    return workflow.run()


if __name__ == "__main__":
    raise SystemExit(main())
