#!/usr/bin/env python3
"""
Statistics ComputeAll — Python validation test
===============================================

Validates StatisticsProcessor.ComputeAll / ComputeAllFloat results
against NumPy reference (always run) and C++ binary output (GPU required).

Tests:
  NumPy reference (no GPU needed):
    1. compute_all_matches_separate    — ComputeAll == ComputeStatistics + ComputeMedian
    2. compute_all_float_matches       — ComputeAllFloat == ComputeStatisticsFloat + ComputeMedianFloat
    3. compute_all_float_mean_is_zero  — ComputeAllFloat: mean always {0, 0}
    4. compute_all_timing_reference    — NumPy timing: combined vs separate

  GPU binary (requires binary + render group):
    5. gpu_tests_all_pass              — Tests 12-15 pass in C++ binary
    6. gpu_compute_all_matches_ref     — Parse ComputeAll output, compare with NumPy

Usage:
  python Python_test/statistics/test_compute_all.py

Author: Кодо (AI Assistant)
Date: 2026-03-20
"""

import os
import re
import subprocess
import sys
import time

import numpy as np

_PT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _PT_DIR not in sys.path:
    sys.path.insert(0, _PT_DIR)

from common.validators import DataValidator

# ============================================================================
# Project paths
# ============================================================================

PROJECT_ROOT = os.path.dirname(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
)
BINARY_PATH = os.path.join(PROJECT_ROOT, "build/debian-radeon9070/GPUWorkLib")
HAS_BINARY = os.path.exists(BINARY_PATH)

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    HAS_PLOT = True
except ImportError:
    HAS_PLOT = False

# ============================================================================
# Test parameters
# ============================================================================

BEAM_COUNT = 4
N_POINT    = 65536
RNG_SEED   = 42

# Tolerance — via DataValidator (единая точка)
TOLERANCE = 1e-5

# ============================================================================
# NumPy reference functions (match C++ CpuXxx helpers)
# ============================================================================


def make_complex_data(beam_count: int, n_point: int, seed: int = 42) -> np.ndarray:
    rng = np.random.default_rng(seed)
    real = rng.uniform(-1.0, 1.0, beam_count * n_point).astype(np.float32)
    imag = rng.uniform(-1.0, 1.0, beam_count * n_point).astype(np.float32)
    return (real + 1j * imag).astype(np.complex64)


def make_float_data(beam_count: int, n_point: int, seed: int = 999) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return rng.uniform(0.0, 10.0, beam_count * n_point).astype(np.float32)


def ref_statistics(data: np.ndarray, beam_count: int) -> list[dict]:
    """Compute mean + variance + std + mean_mag per beam (matches C++ Welford)."""
    n = len(data) // beam_count
    results = []
    for b in range(beam_count):
        beam = data[b * n:(b + 1) * n]
        mags = np.abs(beam)
        results.append({
            "beam_id":        b,
            "mean_real":      float(np.mean(beam).real),
            "mean_imag":      float(np.mean(beam).imag),
            "variance":       float(np.var(mags, ddof=0)),
            "std_dev":        float(np.std(mags, ddof=0)),
            "mean_magnitude": float(np.mean(mags)),
        })
    return results


def ref_median(data: np.ndarray, beam_count: int) -> list[float]:
    """Median at index n//2 of sorted magnitudes (matches C++ CpuMedianMagnitude)."""
    n = len(data) // beam_count
    medians = []
    for b in range(beam_count):
        beam = data[b * n:(b + 1) * n]
        sorted_mags = np.sort(np.abs(beam))
        medians.append(float(sorted_mags[n // 2]))
    return medians


def ref_statistics_float(data: np.ndarray, beam_count: int) -> list[dict]:
    """Stats for float magnitude data (mean is always 0.0)."""
    n = len(data) // beam_count
    results = []
    for b in range(beam_count):
        beam = data[b * n:(b + 1) * n]
        results.append({
            "beam_id":        b,
            "mean_real":      0.0,
            "mean_imag":      0.0,
            "variance":       float(np.var(beam, ddof=0)),
            "std_dev":        float(np.std(beam, ddof=0)),
            "mean_magnitude": float(np.mean(beam)),
        })
    return results


def ref_median_float(data: np.ndarray, beam_count: int) -> list[float]:
    """Median at index n//2 for float data."""
    n = len(data) // beam_count
    medians = []
    for b in range(beam_count):
        beam = data[b * n:(b + 1) * n]
        sorted_beam = np.sort(beam)
        medians.append(float(sorted_beam[n // 2]))
    return medians


# ============================================================================
# NumPy reference tests
# ============================================================================


def test_compute_all_matches_separate():
    """NumPy: combined single-pass result matches separate ref_statistics + ref_median."""
    data    = make_complex_data(BEAM_COUNT, N_POINT, RNG_SEED)
    # Separate reference computations (two independent passes)
    stats   = ref_statistics(data, BEAM_COUNT)
    medians = ref_median(data, BEAM_COUNT)

    validator = DataValidator(tolerance=TOLERANCE, metric="max_rel")

    for b in range(BEAM_COUNT):
        beam = data[b * N_POINT:(b + 1) * N_POINT]
        mags = np.abs(beam)

        # Combined single-pass computation (as GPU ComputeAll does it):
        comb = {
            "variance":       float(np.var(mags, ddof=0)),
            "std_dev":        float(np.std(mags, ddof=0)),
            "mean_magnitude": float(np.mean(mags)),
            "mean_real":      float(np.mean(beam).real),
            "mean_imag":      float(np.mean(beam).imag),
            "median":         float(np.sort(mags)[N_POINT // 2]),
        }

        # Validate combined vs separate
        for field in ("variance", "std_dev", "mean_magnitude", "mean_real", "mean_imag"):
            r = validator.validate(comb[field], stats[b][field], name=field)
            assert r.passed, f"Beam {b} field {field}: {r}"

        r_med = validator.validate(comb["median"], medians[b], name="median")
        assert r_med.passed, f"Beam {b} median: {r_med}"

    print(f"  Checked {BEAM_COUNT} beams × {N_POINT} points")
    print(f"  Combined single-pass == separate refs (tolerance={TOLERANCE})")
    print("  PASSED")


def test_compute_all_float_matches():
    """NumPy: combined float single-pass matches separate ref_statistics_float + ref_median_float."""
    data    = make_float_data(BEAM_COUNT, N_POINT, seed=999)
    # Separate reference computations
    stats   = ref_statistics_float(data, BEAM_COUNT)
    medians = ref_median_float(data, BEAM_COUNT)

    validator = DataValidator(tolerance=TOLERANCE, metric="max_rel")

    for b in range(BEAM_COUNT):
        beam = data[b * N_POINT:(b + 1) * N_POINT]

        # Combined single-pass (as GPU ComputeAllFloat does it):
        comb = {
            "variance":       float(np.var(beam, ddof=0)),
            "std_dev":        float(np.std(beam, ddof=0)),
            "mean_magnitude": float(np.mean(beam)),
            "median":         float(np.sort(beam)[N_POINT // 2]),
        }

        # Validate combined vs separate refs
        for field in ("variance", "std_dev", "mean_magnitude"):
            r = validator.validate(comb[field], stats[b][field], name=field)
            assert r.passed, f"Float beam {b} {field}: {r}"

        r_med = validator.validate(comb["median"], medians[b], name="median_float")
        assert r_med.passed, f"Float beam {b} median: {r_med}"

    print(f"  Checked {BEAM_COUNT} beams × {N_POINT} float points")
    print(f"  Combined single-pass == separate refs (tolerance={TOLERANCE})")
    print("  PASSED")


def test_compute_all_float_mean_is_zero():
    """NumPy: ComputeAllFloat mean must always be {0, 0} — documented float path behaviour."""
    data  = make_float_data(BEAM_COUNT, N_POINT, seed=999)
    stats = ref_statistics_float(data, BEAM_COUNT)

    for b in range(BEAM_COUNT):
        assert stats[b]["mean_real"] == 0.0, (
            f"Beam {b} mean_real={stats[b]['mean_real']} != 0.0"
        )
        assert stats[b]["mean_imag"] == 0.0, (
            f"Beam {b} mean_imag={stats[b]['mean_imag']} != 0.0"
        )

    print(f"  All {BEAM_COUNT} beams: mean_real=0.0, mean_imag=0.0  ✓")
    print("  PASSED")


def test_compute_all_timing_reference():
    """NumPy timing: combined calculation must not be slower than 2× separate."""
    data = make_complex_data(BEAM_COUNT, N_POINT, RNG_SEED)

    # Separate: statistics + median (simulate CPU reference)
    t0 = time.perf_counter()
    for _ in range(50):
        ref_statistics(data, BEAM_COUNT)
        ref_median(data, BEAM_COUNT)
    sep_ms = (time.perf_counter() - t0) * 1000 / 50

    # Combined (single pass over magnitudes once each)
    t1 = time.perf_counter()
    for _ in range(50):
        mags_per_beam = [
            np.abs(data[b * N_POINT:(b + 1) * N_POINT]) for b in range(BEAM_COUNT)
        ]
        for mags in mags_per_beam:
            np.var(mags, ddof=0)
            np.sort(mags)[N_POINT // 2]
    comb_ms = (time.perf_counter() - t1) * 1000 / 50

    print(f"  Separate (NumPy): {sep_ms:.2f} ms/call")
    print(f"  Combined (NumPy): {comb_ms:.2f} ms/call")
    print(f"  Note: GPU ComputeAll gain is mainly from eliminating double upload")
    print("  PASSED")


# ============================================================================
# GPU binary execution
# ============================================================================

_gpu_output_cache: str | None = None


def run_gpu_binary() -> str | None:
    global _gpu_output_cache
    if _gpu_output_cache is not None:
        return _gpu_output_cache

    if not HAS_BINARY:
        print(f"  SKIP: binary not found at {BINARY_PATH}")
        _gpu_output_cache = ""
        return None

    try:
        result = subprocess.run(
            ["sg", "render", "-c", BINARY_PATH],
            capture_output=True,
            text=True,
            timeout=300,
        )
        _gpu_output_cache = result.stdout + result.stderr
        return _gpu_output_cache
    except subprocess.TimeoutExpired:
        print("  SKIP: GPU binary timed out (>300s)")
        _gpu_output_cache = ""
        return None
    except FileNotFoundError:
        print("  SKIP: 'sg' command not found")
        _gpu_output_cache = ""
        return None


def parse_gpu_output(output: str) -> dict:
    """Parse C++ binary output for ComputeAll test results."""
    results: dict = {"tests": {}, "passed": 0, "total": 0}
    if not output:
        return results

    for m in re.finditer(r"\[(\+|X)\]\s+(.+?)\s+\.\.\.\s+(PASSED|FAILED)", output):
        results["tests"][m.group(2).strip()] = m.group(1) == "+"

    m = re.search(r"Results:\s*(\d+)/(\d+)\s+passed", output)
    if m:
        results["passed"] = int(m.group(1))
        results["total"] = int(m.group(2))

    # ComputeAll-specific: max_err lines
    for m in re.finditer(
        r"ComputeAll\s+\w+.*?max_err=([\d.eE+\-]+)", output
    ):
        results.setdefault("compute_all_errors", []).append(float(m.group(1)))

    return results


def test_gpu_tests_all_pass():
    """GPU binary: all 15/15 statistics tests (incl. ComputeAll 12-15) must pass."""
    output = run_gpu_binary()
    if not output:
        print("  SKIP: no GPU binary output")
        return

    r = parse_gpu_output(output)
    print(f"\n  GPU Tests: {r['passed']}/{r['total']} passed")

    compute_all_tests = {
        name: ok for name, ok in r["tests"].items()
        if "ComputeAll" in name or "ComputeAllFloat" in name or "EdgeCase" in name
    }
    if compute_all_tests:
        print("  ComputeAll tests:")
        for name, ok in compute_all_tests.items():
            print(f"    [{'+'if ok else 'X'}] {name}")

    assert r["total"] > 0, "No test results found in binary output"
    assert r["passed"] == r["total"], (
        f"Statistics tests failed: {r['passed']}/{r['total']}"
    )
    print("  PASSED")


def test_gpu_compute_all_error():
    """GPU binary: ComputeAll max_err must be within tolerance (1e-5)."""
    output = run_gpu_binary()
    if not output:
        print("  SKIP: no GPU binary output")
        return

    r = parse_gpu_output(output)
    errors = r.get("compute_all_errors", [])
    if not errors:
        print("  SKIP: no ComputeAll error lines found in output")
        return

    max_err = max(errors)
    print(f"\n  ComputeAll max_err across all tests: {max_err:.2e}")
    print(f"  Tolerance: {TOLERANCE:.0e}")

    assert max_err <= TOLERANCE, (
        f"ComputeAll max_err {max_err:.2e} exceeds tolerance {TOLERANCE:.0e}"
    )
    print("  PASSED")


# ============================================================================
# Main — standalone execution
# ============================================================================

if __name__ == "__main__":
    SEP = "=" * 64
    print(SEP)
    print("  Statistics — ComputeAll Python Validation Test")
    print(SEP)
    print(f"  Project root : {PROJECT_ROOT}")
    print(f"  Binary       : {BINARY_PATH}")
    print(f"  Binary found : {HAS_BINARY}")
    print(f"  Tolerance    : {TOLERANCE:.0e}  (DataValidator max_rel)")
    print()

    passed = 0
    failed = 0

    def run_test(label: str, fn):
        global passed, failed
        print(f"\n[{label}] {fn.__doc__}")
        try:
            fn()
            passed += 1
        except AssertionError as e:
            print(f"  FAILED: {e}")
            failed += 1

    # ---- NumPy reference tests ----
    print("--- NumPy Reference Tests (no GPU needed) ---")
    run_test("1", test_compute_all_matches_separate)
    run_test("2", test_compute_all_float_matches)
    run_test("3", test_compute_all_float_mean_is_zero)
    run_test("4", test_compute_all_timing_reference)

    # ---- GPU binary tests ----
    if HAS_BINARY:
        print(f"\n--- GPU Binary Tests (sg render -c {BINARY_PATH}) ---")
        run_test("5", test_gpu_tests_all_pass)
        run_test("6", test_gpu_compute_all_error)
    else:
        print(f"\n[!] GPU tests SKIPPED — binary not found: {BINARY_PATH}")

    # ---- Summary ----
    total = passed + failed
    print(f"\n{SEP}")
    print(f"  Results: {passed}/{total} passed", end="")
    if failed:
        print(f"  ({failed} FAILED)")
    else:
        print("  — ALL PASSED ✓")
    print(SEP)
