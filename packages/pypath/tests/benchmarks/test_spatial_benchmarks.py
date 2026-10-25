from __future__ import annotations

import json
import os
import statistics
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import pytest

from pypath.spatial import (
    EcospaceParams,
    calculate_spatial_flux,
    create_regular_grid,
    diffusion_flux,
    rsim_run_spatial,
)

BASELINE_PATH = Path(__file__).with_name("spatial_benchmark_baseline.json")


@dataclass
class BenchmarkResult:
    """Measured benchmark statistics."""

    name: str
    samples: list[float]
    median_seconds: float
    p95_seconds: float
    min_seconds: float
    max_seconds: float


_RESULTS: dict[str, BenchmarkResult] = {}


def _read_float_env(name: str, default: float) -> float:
    value = os.getenv(name)
    if value is None or value.strip() == "":
        return default
    parsed = float(value)
    if parsed <= 0:
        raise ValueError(f"{name} must be > 0, got {parsed}")
    return parsed


def _load_baseline() -> dict[str, dict[str, float]]:
    payload = json.loads(BASELINE_PATH.read_text(encoding="utf-8"))
    return payload["benchmarks"]


def _run_benchmark(
    name: str,
    operation,
    *,
    warmups: int,
    repeats: int,
    inner_loops: int,
) -> BenchmarkResult:
    for _ in range(warmups):
        operation()

    samples: list[float] = []
    for _ in range(repeats):
        start = time.perf_counter()
        for _ in range(inner_loops):
            operation()
        elapsed = (time.perf_counter() - start) / inner_loops
        samples.append(elapsed)

    result = BenchmarkResult(
        name=name,
        samples=samples,
        median_seconds=float(statistics.median(samples)),
        p95_seconds=float(np.percentile(samples, 95)),
        min_seconds=float(min(samples)),
        max_seconds=float(max(samples)),
    )
    _RESULTS[name] = result
    return result


def _assert_threshold(
    result: BenchmarkResult,
    *,
    baseline_key: str,
    absolute_env: str,
    absolute_default: float,
) -> None:
    baseline = _load_baseline()
    if baseline_key not in baseline:
        raise KeyError(f"Missing baseline entry for {baseline_key}")

    threshold_scale = _read_float_env("PYPATH_BENCH_THRESHOLD_SCALE", 1.0)
    max_regression_ratio = _read_float_env("PYPATH_BENCH_REGRESSION_MAX_RATIO", 1.5)

    absolute_threshold = _read_float_env(absolute_env, absolute_default) * threshold_scale
    baseline_median = float(baseline[baseline_key]["baseline_median_seconds"])
    regression_threshold = baseline_median * max_regression_ratio

    assert result.median_seconds <= absolute_threshold, (
        f"{result.name} median {result.median_seconds:.6f}s exceeded "
        f"absolute threshold {absolute_threshold:.6f}s (env {absolute_env})"
    )
    assert result.median_seconds <= regression_threshold, (
        f"{result.name} median {result.median_seconds:.6f}s exceeded "
        f"regression threshold {regression_threshold:.6f}s "
        f"(baseline {baseline_median:.6f}s x ratio {max_regression_ratio:.2f})"
    )


@pytest.fixture(scope="session", autouse=True)
def _write_benchmark_output_on_exit(request):
    """Persist benchmark results to JSON when PYPATH_BENCHMARK_OUTPUT is set."""

    def _finalize() -> None:
        output_path = os.getenv("PYPATH_BENCHMARK_OUTPUT")
        if not output_path:
            return

        payload = {
            "benchmarks": {
                name: asdict(result)
                for name, result in sorted(_RESULTS.items(), key=lambda item: item[0])
            }
        }
        path = Path(output_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")

    request.addfinalizer(_finalize)


@pytest.fixture(scope="module")
def _medium_grid_diffusion_inputs() -> tuple[np.ndarray, float, object, object]:
    grid = create_regular_grid(bounds=(0, 0, 10, 10), nx=10, ny=10)
    rng = np.random.default_rng(42)
    biomass = rng.random(100) * 100.0
    return biomass, 5.0, grid, grid.adjacency_matrix


@pytest.fixture(scope="module")
def _combined_flux_inputs() -> tuple[np.ndarray, EcospaceParams, dict[str, int]]:
    n_groups = 10
    grid = create_regular_grid(bounds=(0, 0, 10, 10), nx=10, ny=10)
    n_patches = grid.n_patches

    rng = np.random.default_rng(123)
    state = rng.random((n_groups + 1, n_patches)) * 50.0
    ecospace = EcospaceParams(
        grid=grid,
        habitat_preference=np.ones((n_groups + 1, n_patches)),
        habitat_capacity=np.ones((n_groups + 1, n_patches)),
        dispersal_rate=np.full(n_groups + 1, 2.0),
        advection_enabled=np.zeros(n_groups + 1, dtype=bool),
        gravity_strength=np.zeros(n_groups + 1),
    )
    params = {"NUM_GROUPS": n_groups}
    return state, ecospace, params


@pytest.mark.benchmark
@pytest.mark.slow
def test_diffusion_benchmark_deterministic(_medium_grid_diffusion_inputs):
    """Benchmark diffusion flux on 100 patches with deterministic inputs."""
    biomass, dispersal_rate, grid, adjacency = _medium_grid_diffusion_inputs

    def _operation() -> None:
        diffusion_flux(
            biomass_vector=biomass,
            dispersal_rate=dispersal_rate,
            grid=grid,
            adjacency=adjacency,
        )

    result = _run_benchmark(
        "diffusion_10x10",
        _operation,
        warmups=3,
        repeats=9,
        inner_loops=20,
    )
    _assert_threshold(
        result,
        baseline_key="diffusion_10x10",
        absolute_env="PYPATH_BENCH_DIFFUSION_MAX_SECONDS",
        absolute_default=0.005,
    )


@pytest.mark.benchmark
@pytest.mark.slow
def test_combined_flux_benchmark_deterministic(_combined_flux_inputs):
    """Benchmark combined spatial flux for 10 groups across 100 patches."""
    state, ecospace, params = _combined_flux_inputs

    def _operation() -> None:
        calculate_spatial_flux(state, ecospace, params, t=0.0)

    result = _run_benchmark(
        "combined_flux_10x10_10groups",
        _operation,
        warmups=2,
        repeats=8,
        inner_loops=5,
    )
    _assert_threshold(
        result,
        baseline_key="combined_flux_10x10_10groups",
        absolute_env="PYPATH_BENCH_COMBINED_FLUX_MAX_SECONDS",
        absolute_default=0.05,
    )


@pytest.mark.benchmark
@pytest.mark.slow
def test_spatial_simulation_benchmark_deterministic(spatial_scenario):
    """Benchmark a representative 5-year 10x10 spatial simulation wall time."""
    scenario, _ = spatial_scenario
    ng = scenario.params.NUM_GROUPS + 1
    grid = create_regular_grid(bounds=(0, 0, 10, 10), nx=10, ny=10)
    n_patches = grid.n_patches

    ecospace = EcospaceParams(
        grid=grid,
        habitat_preference=np.ones((ng, n_patches)),
        habitat_capacity=np.ones((ng, n_patches)),
        dispersal_rate=np.full(ng, 2.0),
        advection_enabled=np.zeros(ng, dtype=bool),
        gravity_strength=np.zeros(ng),
    )

    def _operation() -> None:
        rsim_run_spatial(scenario, ecospace=ecospace, years=range(1, 6))

    result = _run_benchmark(
        "spatial_simulation_5year_10x10",
        _operation,
        warmups=1,
        repeats=3,
        inner_loops=1,
    )
    _assert_threshold(
        result,
        baseline_key="spatial_simulation_5year_10x10",
        absolute_env="PYPATH_BENCH_SIMULATION_5Y_10X10_MAX_SECONDS",
        absolute_default=30.0,
    )
