# Performance Benchmarks

This guide defines the deterministic benchmark gate used for spatial-performance regression checks.

## Scope

The benchmark suite is in:

- `packages/pypath/tests/benchmarks/test_spatial_benchmarks.py`
- Baseline medians: `packages/pypath/tests/benchmarks/spatial_benchmark_baseline.json`

Benchmarks currently cover:

1. `diffusion_10x10` (100-patch diffusion flux)
2. `combined_flux_10x10_10groups` (combined diffusion/advection flux path)
3. `spatial_simulation_5year_10x10` (5-year coupled spatial run)

## Local execution

```bash
python -m pytest packages/pypath/tests/benchmarks/test_spatial_benchmarks.py -q -m "benchmark" --tb=short
```

To export machine-readable benchmark results:

```bash
PYPATH_BENCHMARK_OUTPUT=benchmark-results.json python -m pytest packages/pypath/tests/benchmarks/test_spatial_benchmarks.py -q -m "benchmark" --tb=short
```

## CI gating

Workflow: `.github/workflows/ci-performance-benchmarks.yml`

Triggers:

- Pull requests that touch core runtime/tests/workflow files
- Weekly schedule (`0 3 * * 1`)
- Manual `workflow_dispatch`

The workflow produces:

- `benchmark-results.json` artifact (raw timings)
- `benchmark-summary.md` artifact and step summary table (current vs baseline)

## Configurable thresholds

All thresholds are environment-configurable so they can be tuned for CI hardware:

| Variable | Default | Meaning |
|---|---:|---|
| `PYPATH_BENCH_DIFFUSION_MAX_SECONDS` | `0.005` | Absolute max median for `diffusion_10x10` |
| `PYPATH_BENCH_COMBINED_FLUX_MAX_SECONDS` | `0.05` | Absolute max median for `combined_flux_10x10_10groups` |
| `PYPATH_BENCH_SIMULATION_5Y_10X10_MAX_SECONDS` | `30` | Absolute max median for `spatial_simulation_5year_10x10` |
| `PYPATH_BENCH_REGRESSION_MAX_RATIO` | `1.5` | Max allowed median ratio vs baseline |
| `PYPATH_BENCH_THRESHOLD_SCALE` | `1.0` | Multiplier applied to absolute thresholds |

A benchmark must satisfy both checks:

- `median <= absolute_threshold * threshold_scale`
- `median <= baseline_median * regression_ratio`

## Baseline calibration on CI hardware

When CI hardware or workload changes significantly:

1. Run the benchmark workflow on `main` without code changes.
2. Review `benchmark-results.json` artifact medians.
3. Update `spatial_benchmark_baseline.json` with the new medians.
4. Keep `PYPATH_BENCH_REGRESSION_MAX_RATIO` conservative (for example `1.3`-`1.5`) to avoid noisy failures while still detecting regressions.

This keeps gating deterministic while preserving sensitivity to meaningful slowdowns.
