from __future__ import annotations

import argparse
import json
import os
from pathlib import Path


def _read_float_env(name: str, default: float) -> float:
    value = os.getenv(name)
    if value is None or value.strip() == "":
        return default
    return float(value)


def main() -> int:
    parser = argparse.ArgumentParser(description="Render benchmark comparison report")
    parser.add_argument(
        "--results", required=True, help="Path to benchmark results JSON"
    )
    parser.add_argument(
        "--baseline", required=True, help="Path to benchmark baseline JSON"
    )
    parser.add_argument("--summary-file", help="Optional markdown output file")
    args = parser.parse_args()

    results = json.loads(Path(args.results).read_text(encoding="utf-8"))
    baseline = json.loads(Path(args.baseline).read_text(encoding="utf-8"))

    scale = _read_float_env("PYPATH_BENCH_THRESHOLD_SCALE", 1.0)
    ratio_limit = _read_float_env("PYPATH_BENCH_REGRESSION_MAX_RATIO", 1.5)
    abs_limits = {
        "diffusion_10x10": _read_float_env("PYPATH_BENCH_DIFFUSION_MAX_SECONDS", 0.005)
        * scale,
        "combined_flux_10x10_10groups": _read_float_env(
            "PYPATH_BENCH_COMBINED_FLUX_MAX_SECONDS", 0.05
        )
        * scale,
        "spatial_simulation_5year_10x10": _read_float_env(
            "PYPATH_BENCH_SIMULATION_5Y_10X10_MAX_SECONDS", 30.0
        )
        * scale,
    }

    lines = [
        "## Spatial benchmark report",
        "",
        "| Benchmark | Median (s) | Baseline (s) | Ratio | Abs limit (s) | Status |",
        "|---|---:|---:|---:|---:|---|",
    ]

    failures = 0
    result_rows = results.get("benchmarks", {})
    baseline_rows = baseline.get("benchmarks", {})

    for name in sorted(result_rows):
        row = result_rows[name]
        baseline_median = float(baseline_rows[name]["baseline_median_seconds"])
        median = float(row["median_seconds"])
        ratio = median / baseline_median if baseline_median > 0 else float("inf")
        abs_limit = abs_limits[name]

        status = "PASS"
        if median > abs_limit or ratio > ratio_limit:
            status = "FAIL"
            failures += 1

        lines.append(
            f"| {name} | {median:.6f} | {baseline_median:.6f} | {ratio:.3f}x | {abs_limit:.6f} | {status} |"
        )

    lines.extend(
        [
            "",
            f"- Threshold scale (`PYPATH_BENCH_THRESHOLD_SCALE`): **{scale:.2f}**",
            f"- Max regression ratio (`PYPATH_BENCH_REGRESSION_MAX_RATIO`): **{ratio_limit:.2f}x**",
            f"- Overall status: **{'PASS' if failures == 0 else 'FAIL'}**",
        ]
    )

    report = "\n".join(lines) + "\n"
    print(report)

    if args.summary_file:
        Path(args.summary_file).write_text(report, encoding="utf-8")

    github_summary = os.getenv("GITHUB_STEP_SUMMARY")
    if github_summary:
        with Path(github_summary).open("a", encoding="utf-8") as handle:
            handle.write(report)

    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
