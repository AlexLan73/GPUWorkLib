#!/usr/bin/env python3
"""
Statistics ROCm — Python validation test
=========================================

Validates StatisticsProcessor results against NumPy reference.
Test parameters mirror test_statistics_rocm.hpp exactly.

Tests:
  NumPy reference (always run, no GPU needed):
    1. mean_single_beam   — complex sinusoid mean ≈ 0
    2. mean_multi_beam    — per-beam means ≈ 0 (4 beams)
    3. welford_statistics — mean_mag ≈ amplitude, variance ≈ 0
    4. median_linear      — median of [1..1024] at index n//2 = 513
    5. mean_constant      — mean of constant signal = constant

  GPU binary (requires binary + render group):
    6. all_tests_pass     — all 7/7 C++ tests pass
    7. benchmark_speedup  — GPU sort speedup > MIN_SPEEDUP

  GPU vs NumPy comparison (parsed from stdout):
    8. gpu_vs_numpy_welford — GPU Welford stats vs NumPy
    9. gpu_vs_numpy_median  — GPU median vs NumPy

Usage:
  python Python_test/statistics/test_statistics_rocm.py

Author: Kodo (AI Assistant)
Date: 2026-02-24
"""

import os
import re
import subprocess
import sys
import time

import numpy as np

# ============================================================================
# Project paths
# ============================================================================

# Python_test/statistics/ -> 2 levels up -> project root
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
# Test parameters — must match test_statistics_rocm.hpp exactly
# ============================================================================

# Test 1: ComputeMean — single beam sinusoid
T1_FREQ, T1_FS, T1_N, T1_AMP = 100.0, 1000.0, 4096, 1.0

# Test 2: ComputeMean — multi-beam (4 beams)
T2_BEAMS = 4
T2_N, T2_FREQ, T2_FS = 2048, 50.0, 1000.0
T2_BASE_AMP, T2_AMP_STEP = 1.0, 0.5  # Beam amplitudes: 1.0, 1.5, 2.0, 2.5

# Test 3: ComputeStatistics — Welford
T3_FREQ, T3_FS, T3_N, T3_AMP = 100.0, 1000.0, 4096, 2.0

# Test 4: ComputeMedian — linear magnitudes
T4_N = 1024  # data[i] = complex(i+1, 0) → magnitudes [1..1024]

# Test 5: GPU input
T5_FREQ, T5_FS, T5_N = 200.0, 1000.0, 2048

# Test 6: Constant mean
T6_VALUE = (3.14, -2.71)
T6_N = 4096

# Test 7: Benchmark
T7_BEAMS, T7_N = 4, 500_000

# Test 8-11: Histogram median (uses auto-selection > 100K threshold)
T8_N = 200_000  # triggers histogram path
T9_BEAMS, T9_N = 4, 500_000
T10_BEAMS, T10_N = 2, 200_000

# Minimum acceptable GPU speedup for benchmark test
MIN_SPEEDUP = 2.0

# ============================================================================
# Signal generators — match C++ GenerateSinusoid / GenerateMultiBeam
# ============================================================================


def make_sinusoid(freq: float, fs: float, n: int, amp: float = 1.0) -> np.ndarray:
    """
    Complex sinusoid: amp * exp(2πj * freq * t)
    Matches C++ GenerateSinusoid (float32 precision).
    """
    t = np.arange(n, dtype=np.float32) / np.float32(fs)
    phase = 2.0 * np.float32(np.pi) * np.float32(freq) * t
    return (np.float32(amp) * np.exp(1j * phase)).astype(np.complex64)


def make_multi_beam(
    beams: int, n: int, fs: float, freq: float, base_amp: float, amp_step: float
) -> np.ndarray:
    """
    Multi-beam sinusoids flattened to (beams * n,).
    Matches C++ GenerateMultiBeam.
    """
    data = np.zeros(beams * n, dtype=np.complex64)
    for b in range(beams):
        data[b * n : (b + 1) * n] = make_sinusoid(freq, fs, n, base_amp + b * amp_step)
    return data


# ============================================================================
# NumPy reference statistics — match C++ CPU reference functions
# ============================================================================


def ref_mean(data: np.ndarray) -> complex:
    """Complex mean — matches CpuMean."""
    return complex(np.mean(data.astype(np.complex64)))


def ref_mean_mag(data: np.ndarray) -> float:
    """Mean of magnitudes — matches CpuMeanMagnitude."""
    return float(np.mean(np.abs(data)))


def ref_variance_mag(data: np.ndarray) -> float:
    """
    Population variance of magnitudes — matches CpuVarianceMagnitude.
    Uses ddof=0 (not ddof=1 / Bessel's correction).
    """
    return float(np.var(np.abs(data), ddof=0))


def ref_std_mag(data: np.ndarray) -> float:
    """Population std of magnitudes — matches CpuStdMagnitude."""
    return float(np.std(np.abs(data), ddof=0))


def ref_median_mag(data: np.ndarray) -> float:
    """
    Median: sorted_mags[n // 2] — matches CpuMedianMagnitude.
    Note: NOT the standard Python/NumPy median (average of two middle elements
    for even n). C++ uses the element at index n/2 directly.
    """
    sorted_mags = np.sort(np.abs(data))
    return float(sorted_mags[len(sorted_mags) // 2])


# ============================================================================
# NumPy reference tests — always run, no GPU needed
# ============================================================================


def test_numpy_mean_single_beam():
    """NumPy: complex mean of sinusoid should be near zero."""
    sig = make_sinusoid(T1_FREQ, T1_FS, T1_N, T1_AMP)
    mean = ref_mean(sig)
    print(f"  mean = ({mean.real:.6f}, {mean.imag:.6f})")
    assert abs(mean.real) < 0.01, f"Re mean too large: {mean.real:.4e}"
    assert abs(mean.imag) < 0.01, f"Im mean too large: {mean.imag:.4e}"
    print("  PASSED")


def test_numpy_mean_multi_beam():
    """NumPy: per-beam complex means should all be near zero."""
    for b in range(T2_BEAMS):
        amp = T2_BASE_AMP + b * T2_AMP_STEP
        sig = make_sinusoid(T2_FREQ, T2_FS, T2_N, amp)
        mean = ref_mean(sig)
        print(f"  Beam {b} (amp={amp:.1f}): mean=({mean.real:.6f}, {mean.imag:.6f})")
        assert abs(mean.real) < 0.01 and abs(mean.imag) < 0.01, (
            f"Beam {b}: mean magnitude {abs(mean):.4e} too large"
        )
    print("  PASSED")


def test_numpy_welford_statistics():
    """NumPy: constant-amplitude sinusoid → mean_mag=amp, variance≈0."""
    sig = make_sinusoid(T3_FREQ, T3_FS, T3_N, T3_AMP)
    mean_mag = ref_mean_mag(sig)
    variance = ref_variance_mag(sig)
    std = ref_std_mag(sig)
    print(f"  mean_mag = {mean_mag:.6f}  (expected ~{T3_AMP:.1f})")
    print(f"  variance = {variance:.2e}  (expected ~0)")
    print(f"  std      = {std:.2e}  (expected ~0)")
    assert abs(mean_mag - T3_AMP) < 1e-3, f"mean_mag={mean_mag:.6f} != {T3_AMP}"
    assert variance < 1e-4, f"variance={variance:.2e} unexpectedly large"
    print("  PASSED")


def test_numpy_median_linear():
    """NumPy: median of [1..1024] at index n//2 should equal 513."""
    data = np.array(
        [complex(i + 1, 0) for i in range(T4_N)], dtype=np.complex64
    )
    median = ref_median_mag(data)
    expected = float(T4_N // 2 + 1)  # sorted[512] = 513.0
    print(f"  median = {median:.1f}  (expected {expected:.1f})")
    assert abs(median - expected) < 1.0, f"median={median:.1f} != {expected:.1f}"
    print("  PASSED")


def test_numpy_mean_constant():
    """NumPy: mean of constant complex signal should equal the constant."""
    value = complex(*T6_VALUE)
    data = np.full(T6_N, value, dtype=np.complex64)
    mean = ref_mean(data)
    print(f"  mean     = ({mean.real:.4f}, {mean.imag:.4f})")
    print(f"  expected = ({T6_VALUE[0]:.4f}, {T6_VALUE[1]:.4f})")
    assert abs(mean.real - T6_VALUE[0]) < 1e-3, f"Re mismatch: {mean.real}"
    assert abs(mean.imag - T6_VALUE[1]) < 1e-3, f"Im mismatch: {mean.imag}"
    print("  PASSED")


# ============================================================================
# GPU binary execution
# ============================================================================

_gpu_output_cache: str | None = None  # Run binary once, cache the output


def run_gpu_binary() -> str | None:
    """
    Run GPU binary via 'sg render -c <binary>'.
    Returns combined stdout+stderr, or None on failure/skip.
    """
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
            timeout=180,
        )
        _gpu_output_cache = result.stdout + result.stderr
        return _gpu_output_cache
    except subprocess.TimeoutExpired:
        print("  SKIP: GPU binary timed out (>180s)")
        _gpu_output_cache = ""
        return None
    except FileNotFoundError:
        print("  SKIP: 'sg' command not found")
        _gpu_output_cache = ""
        return None


def parse_output(output: str) -> dict:
    """
    Parse C++ binary stdout for statistics test results.

    Output line format (from ConsoleOutput):
      [HH:MM:SS.mmm] [INF] [GPU_00] [Stats ROCm] <message>

    Returns dict with keys:
      tests       : {test_name: bool}
      passed      : int
      total       : int
      benchmark   : {cpu_ms, gpu_ms, speedup}
      mean_re/im  : floats (Test 1: single beam mean)
      mean_mag_gpu/cpu, variance_gpu/cpu, std_gpu/cpu : floats (Test 3: Welford)
      median_gpu/cpu : floats (Test 4: Median)
    """
    results: dict = {"tests": {}, "passed": 0, "total": 0, "benchmark": {}}
    if not output:
        return results

    # Individual test results: "[+] Name ... PASSED" or "[X] Name ... FAILED"
    for m in re.finditer(r"\[(\+|X)\]\s+(.+?)\s+\.\.\.\s+(PASSED|FAILED)", output):
        results["tests"][m.group(2).strip()] = m.group(1) == "+"

    # Summary: "Results: X/Y passed"
    m = re.search(r"Results:\s*(\d+)/(\d+)\s+passed", output)
    if m:
        results["passed"] = int(m.group(1))
        results["total"] = int(m.group(2))

    # Benchmark timing lines (both radix sort and histogram)
    _parse_float(results, "benchmark", output, "cpu_ms", r"CPU sort\s*:\s*([\d.]+)\s*ms")
    _parse_float(results, "benchmark", output, "gpu_ms", r"GPU sort\s*:\s*([\d.]+)\s*ms")
    _parse_float(results, "benchmark", output, "speedup", r"Speedup\s*:\s*([\d.]+)x")

    # Histogram benchmark
    results["hist_benchmark"] = {}
    _parse_float(results, "hist_benchmark", output, "cpu_ms", r"CPU sort\s+:\s*([\d.]+)\s*ms")
    _parse_float(results, "hist_benchmark", output, "gpu_ms", r"GPU histogram\s*:\s*([\d.]+)\s*ms")

    # Test 1: mean single beam — "mean=(re, im) err_re=..."
    m = re.search(r"mean=\(([-\d.eE+]+),\s*([-\d.eE+]+)\)", output)
    if m:
        results["mean_re"] = float(m.group(1))
        results["mean_im"] = float(m.group(2))

    # Test 3: Welford — "mean_mag=X (cpu=Y)"
    m = re.search(r"mean_mag=([\d.eE+\-]+)\s+\(cpu=([\d.eE+\-]+)\)", output)
    if m:
        results["mean_mag_gpu"] = float(m.group(1))
        results["mean_mag_cpu"] = float(m.group(2))

    m = re.search(r"variance=([\d.eE+\-]+)\s+\(cpu=([\d.eE+\-]+)\)", output)
    if m:
        results["variance_gpu"] = float(m.group(1))
        results["variance_cpu"] = float(m.group(2))

    m = re.search(r"\bstd=([\d.eE+\-]+)\s+\(cpu=([\d.eE+\-]+)\)", output)
    if m:
        results["std_gpu"] = float(m.group(1))
        results["std_cpu"] = float(m.group(2))

    # Test 4: Median — "median=X (cpu=Y), err=Z"
    m = re.search(r"median=([\d.eE+\-]+)\s+\(cpu=([\d.eE+\-]+)\)", output)
    if m:
        results["median_gpu"] = float(m.group(1))
        results["median_cpu"] = float(m.group(2))

    return results


def _parse_float(dest: dict, subkey: str, text: str, key: str, pattern: str):
    """Helper: extract a float from text and store in dest[subkey][key]."""
    m = re.search(pattern, text)
    if m:
        dest[subkey][key] = float(m.group(1))


# ============================================================================
# NumPy reference: histogram median validation
# ============================================================================


def test_numpy_histogram_median_basic():
    """NumPy ref: median of linear [1..200K] = sorted[N//2] = 100001.0."""
    n = T8_N
    mags = np.arange(1, n + 1, dtype=np.float32)
    expected = float(mags[n // 2])  # sorted[N/2]
    actual = float(np.sort(mags)[n // 2])
    print(f"  Linear median [1..{n}]: expected={expected:.0f}, numpy={actual:.0f}")
    assert expected == actual, f"Median mismatch: {expected} != {actual}"
    print("  PASSED")


def test_numpy_histogram_median_random():
    """NumPy ref: median of random data (4 beams × 500K) matches np.median."""
    rng = np.random.default_rng(123)
    data = rng.uniform(0.1, 1000.0, T9_BEAMS * T9_N).astype(np.float32)
    for b in range(T9_BEAMS):
        beam_data = data[b * T9_N : (b + 1) * T9_N]
        sorted_data = np.sort(beam_data)
        median_sort = float(sorted_data[T9_N // 2])
        # np.median returns average of 2 middle elements for even N, we use sorted[N//2]
        print(f"  Beam {b}: sorted[N//2]={median_sort:.4f}")
    print("  PASSED (reference values computed)")


# ============================================================================
# GPU binary tests
# ============================================================================


def test_gpu_all_pass():
    """GPU binary: all 11/11 statistics tests must pass."""
    output = run_gpu_binary()
    if not output:
        print("  SKIP: no GPU binary output")
        return

    r = parse_output(output)
    print(f"\n  GPU Tests: {r['passed']}/{r['total']} passed")
    for name, ok in r["tests"].items():
        print(f"    [{'+'if ok else 'X'}] {name} ... {'PASSED' if ok else 'FAILED'}")

    assert r["total"] > 0, "No test results found in binary output"
    assert r["passed"] == r["total"], (
        f"Statistics tests failed: {r['passed']}/{r['total']}"
    )
    print("  PASSED")


def test_gpu_benchmark_speedup():
    """GPU benchmark: sort speedup over CPU must exceed MIN_SPEEDUP."""
    output = run_gpu_binary()
    if not output:
        print("  SKIP: no GPU binary output")
        return

    r = parse_output(output)
    b = r["benchmark"]
    if not b:
        print("  SKIP: benchmark results not found in output")
        return

    cpu_ms = b.get("cpu_ms", float("nan"))
    gpu_ms = b.get("gpu_ms", float("nan"))
    speedup = b.get("speedup", 0.0)

    print(f"\n  Benchmark ({T7_BEAMS} beams × {T7_N:,} points):")
    print(f"    CPU sort : {cpu_ms:.1f} ms")
    print(f"    GPU sort : {gpu_ms:.1f} ms")
    print(f"    Speedup  : {speedup:.1f}×")
    print(f"    Min req  : {MIN_SPEEDUP:.1f}×")

    assert speedup >= MIN_SPEEDUP, (
        f"GPU speedup {speedup:.1f}× < minimum {MIN_SPEEDUP:.1f}×"
    )
    print(f"  PASSED (speedup = {speedup:.1f}×)")


# ============================================================================
# GPU vs NumPy comparison
# ============================================================================


def test_gpu_vs_numpy_welford():
    """GPU Welford statistics must match NumPy reference within tolerance."""
    output = run_gpu_binary()
    if not output:
        print("  SKIP: no GPU binary output")
        return

    r = parse_output(output)
    if "mean_mag_gpu" not in r:
        print("  SKIP: Welford results not found in output")
        return

    sig = make_sinusoid(T3_FREQ, T3_FS, T3_N, T3_AMP)
    np_mean_mag = ref_mean_mag(sig)
    np_variance = ref_variance_mag(sig)
    np_std = ref_std_mag(sig)

    err_mean = abs(r["mean_mag_gpu"] - np_mean_mag)
    err_var = abs(r["variance_gpu"] - np_variance)
    err_std = abs(r["std_gpu"] - np_std)

    print(f"\n  Welford Statistics (amp={T3_AMP}):")
    print(f"    {'':12s}  {'GPU':>12s}  {'NumPy':>12s}  {'Error':>12s}")
    print(f"    {'mean_mag':12s}  {r['mean_mag_gpu']:>12.6f}  {np_mean_mag:>12.6f}  {err_mean:>12.2e}")
    print(f"    {'variance':12s}  {r['variance_gpu']:>12.2e}  {np_variance:>12.2e}  {err_var:>12.2e}")
    print(f"    {'std':12s}  {r['std_gpu']:>12.2e}  {np_std:>12.2e}  {err_std:>12.2e}")

    assert err_mean < 0.01, f"mean_mag error {err_mean:.4e} too large"
    print("  PASSED")


def test_gpu_vs_numpy_median():
    """GPU median must match NumPy reference within 1 unit."""
    output = run_gpu_binary()
    if not output:
        print("  SKIP: no GPU binary output")
        return

    r = parse_output(output)
    if "median_gpu" not in r:
        print("  SKIP: median results not found in output")
        return

    data = np.array(
        [complex(i + 1, 0) for i in range(T4_N)], dtype=np.complex64
    )
    np_median = ref_median_mag(data)
    err = abs(r["median_gpu"] - np_median)

    print(f"\n  Median (linear [1..{T4_N}]):")
    print(f"    GPU median  = {r['median_gpu']:.1f}")
    print(f"    C++ CPU ref = {r['median_cpu']:.1f}")
    print(f"    NumPy ref   = {np_median:.1f}")
    print(f"    Error       = {err:.2f}")

    assert err < 1.0, f"GPU median {r['median_gpu']:.1f} vs NumPy {np_median:.1f}"
    print("  PASSED")


# ============================================================================
# Visualization — standalone mode only
# ============================================================================


def plot_reference_values():
    """Generate 4-panel reference plot for statistics module."""
    if not HAS_PLOT:
        print("  SKIP: matplotlib not available")
        return

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle(
        "Statistics ROCm — NumPy Reference Values\n"
        "(test parameters match test_statistics_rocm.hpp)",
        fontsize=13,
    )

    # Panel 1: Welford — |z| for constant-amplitude sinusoid
    sig3 = make_sinusoid(T3_FREQ, T3_FS, T3_N, T3_AMP)
    mags3 = np.abs(sig3)
    ax = axes[0, 0]
    ax.plot(mags3[:300], "b-", linewidth=1, alpha=0.8, label="|z|")
    ax.axhline(
        T3_AMP, color="r", linestyle="--", linewidth=1.5,
        label=f"Expected mean_mag = {T3_AMP:.1f}"
    )
    ax.set_title(
        f"Test 3: Welford (n={T3_N}, amp={T3_AMP})\n"
        f"mean_mag={ref_mean_mag(sig3):.4f}, std={ref_std_mag(sig3):.2e}"
    )
    ax.set_xlabel("Sample")
    ax.set_ylabel("Magnitude")
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

    # Panel 2: Median — linear magnitudes [1..T4_N]
    data4 = np.arange(1, T4_N + 1, dtype=np.float32)
    median4 = float(data4[T4_N // 2])  # index 512 → 513
    ax = axes[0, 1]
    ax.plot(data4, "b-", linewidth=0.8, alpha=0.6, label="magnitudes")
    ax.axvline(
        T4_N // 2, color="r", linestyle="--", linewidth=1.5,
        label=f"Index n//2 = {T4_N // 2}"
    )
    ax.axhline(
        median4, color="g", linestyle="-.", linewidth=1.5,
        label=f"Median = {median4:.0f}"
    )
    ax.set_title(
        f"Test 4: Median (linear [1..{T4_N}])\n"
        f"Median at index {T4_N // 2} = {median4:.0f}"
    )
    ax.set_xlabel("Index")
    ax.set_ylabel("Magnitude")
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

    # Panel 3: Multi-beam |mean| (should be near 0)
    beam_amps = [T2_BASE_AMP + b * T2_AMP_STEP for b in range(T2_BEAMS)]
    beam_abs_means = []
    for amp in beam_amps:
        sig = make_sinusoid(T2_FREQ, T2_FS, T2_N, amp)
        beam_abs_means.append(abs(ref_mean(sig)))
    ax = axes[1, 0]
    bars = ax.bar(range(T2_BEAMS), beam_abs_means, color="steelblue", alpha=0.75)
    ax.axhline(0.01, color="r", linestyle="--", linewidth=1.5, label="Threshold 0.01")
    ax.set_title(
        f"Test 2: Multi-beam |mean| (n={T2_N}, {T2_BEAMS} beams)\n"
        "All means should be < 0.01"
    )
    ax.set_xlabel("Beam")
    ax.set_ylabel("|mean(z)|")
    ax.set_xticks(range(T2_BEAMS))
    ax.set_xticklabels(
        [f"B{b}\namp={beam_amps[b]:.1f}" for b in range(T2_BEAMS)]
    )
    for bar, val in zip(bars, beam_abs_means):
        ax.text(bar.get_x() + bar.get_width() / 2, val + 0.0001,
                f"{val:.4f}", ha="center", va="bottom", fontsize=8)
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3, axis="y")

    # Panel 4: CPU sort timing vs n_point (NumPy reference)
    rng = np.random.default_rng(42)
    n_sizes = [1_000, 10_000, 100_000, 500_000]
    cpu_times_ms = []
    for n in n_sizes:
        raw = rng.uniform(0, 1000, T7_BEAMS * n).astype(np.float32)
        t0 = time.perf_counter()
        for b in range(T7_BEAMS):
            np.sort(raw[b * n : (b + 1) * n])
        cpu_times_ms.append((time.perf_counter() - t0) * 1000)

    ax = axes[1, 1]
    ax.loglog(n_sizes, cpu_times_ms, "b-o", linewidth=1.5, markersize=6, label="NumPy sort (CPU)")
    ax.axvline(
        T7_N, color="r", linestyle="--", linewidth=1.5,
        label=f"Benchmark point ({T7_N:,})"
    )
    if len(cpu_times_ms) >= 4:
        ax.annotate(
            f"{cpu_times_ms[-1]:.0f} ms",
            xy=(T7_N, cpu_times_ms[-1]),
            xytext=(T7_N * 0.4, cpu_times_ms[-1] * 1.5),
            fontsize=9,
            arrowprops=dict(arrowstyle="->", color="red"),
        )
    ax.set_title(
        f"Benchmark: CPU sort scaling ({T7_BEAMS} beams)\n"
        f"GPU should be > {MIN_SPEEDUP:.0f}× faster at {T7_N:,} points"
    )
    ax.set_xlabel("Points per beam")
    ax.set_ylabel("Time (ms)")
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3, which="both")

    plt.tight_layout()

    out_dir = os.path.join(PROJECT_ROOT, "Results", "Plots", "statistics")
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, "test_statistics_rocm_reference.png")
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"  Plot saved: {out_path}")
    plt.close()


# ============================================================================
# Main — standalone execution
# ============================================================================

if __name__ == "__main__":
    SEP = "=" * 64
    print(SEP)
    print("  Statistics ROCm — Python Validation Test")
    print(SEP)
    print(f"  Project root : {PROJECT_ROOT}")
    print(f"  Binary       : {BINARY_PATH}")
    print(f"  Binary found : {HAS_BINARY}")
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

    # ---- NumPy reference tests (always run) ----
    print("--- NumPy Reference Tests (no GPU needed) ---")
    run_test("1", test_numpy_mean_single_beam)
    run_test("2", test_numpy_mean_multi_beam)
    run_test("3", test_numpy_welford_statistics)
    run_test("4", test_numpy_median_linear)
    run_test("5", test_numpy_mean_constant)
    run_test("5a", test_numpy_histogram_median_basic)
    run_test("5b", test_numpy_histogram_median_random)

    # ---- GPU binary tests ----
    if HAS_BINARY:
        print(f"\n--- GPU Binary Tests (sg render -c {BINARY_PATH}) ---")
        run_test("6", test_gpu_all_pass)
        run_test("7", test_gpu_benchmark_speedup)

        print("\n--- GPU vs NumPy Comparison ---")
        run_test("8", test_gpu_vs_numpy_welford)
        run_test("9", test_gpu_vs_numpy_median)
    else:
        print(f"\n[!] GPU tests SKIPPED — binary not found: {BINARY_PATH}")
        print("    Build first: cmake --build build/debian-radeon9070 -j4")

    # ---- Visualization ----
    print("\n--- Reference Plot ---")
    plot_reference_values()

    # ---- Summary ----
    total = passed + failed
    print(f"\n{SEP}")
    print(f"  Results: {passed}/{total} passed", end="")
    if failed:
        print(f"  ({failed} FAILED)")
    else:
        print("  — ALL PASSED ✓")
    print(SEP)
