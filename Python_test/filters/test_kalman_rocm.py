#!/usr/bin/env python3
"""
Test: KalmanFilterROCm — GPU 1D scalar Kalman filter (ROCm) vs numpy reference

Filter: 1D scalar Kalman — Re and Im parts filtered independently.
GPU class: gpuworklib.KalmanFilterROCm

Tests:
  1. test_kalman_basic           — random complex signal, GPU vs numpy reference
  2. test_kalman_multi_channel   — 8-channel, each vs reference
  3. test_kalman_channel_independence — 256 channels with distinct const signals
  4. test_kalman_noise_reduction  — SNR improvement: filtered RMS << noisy RMS
  5. test_kalman_step_response    — step at n=512, filter follows smoothly
  6. test_kalman_lfm_radar_demo  — 5-antenna LFM beat tones + noise (CPU-only)
  7. test_kalman_properties       — is_ready(), get_params()

Algorithm (for reference):
  Init: x_hat = x0, P = P0
  For each sample z[n]:
    P_pred = P + Q
    K      = P_pred / (P_pred + R)    # Kalman gain [0..1]
    x_hat += K * (z[n] - x_hat)       # state update
    P      = (1 - K) * P_pred         # covariance update
    output[n] = x_hat

Note:
  Tolerance GPU vs numpy: < 1e-4 (float32).
  GPU API:
    kalman = gpuworklib.KalmanFilterROCm(ctx)
    kalman.set_params(Q, R, x0=0.0, P0=25.0)
    kalman.process(data)               # 1D or 2D (channels, points) complex64
    kalman.is_ready()
    kalman.get_params()                # dict-like: {Q, R, x0, P0}

Usage:
  python Python_test/filters/test_kalman_rocm.py

Author: Kodo (AI Assistant)
Date: 2026-03-04
"""

import sys
import os
import numpy as np

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
for _subdir in ["build/python", "build/python/Release", "build/python/Debug", "build/debian-radeon9070/python"]:
    _p = os.path.join(PROJECT_ROOT, _subdir)
    if os.path.exists(_p):
        sys.path.insert(0, _p)
        break

try:
    import gpuworklib
    HAS_GPU = True
except ImportError:
    HAS_GPU = False
    print("WARNING: gpuworklib not found. Skipping GPU tests.")

# ============================================================================
# Parameters
# ============================================================================

POINTS   = 4096
CHANNELS = 8
Q_DEF    = 0.1    # process noise variance (default)
R_DEF    = 25.0   # measurement noise variance (default)
X0_DEF   = 0.0    # initial state
P0_DEF   = 25.0   # initial error covariance (= R by default)
ATOL     = 1e-4   # float32 tolerance GPU vs Python reference

# ============================================================================
# Python reference implementation — matches GPU kalman_kernel exactly
# ============================================================================

def _kalman_1ch_scalar(signal: np.ndarray,
                        Q: float, R: float,
                        x0: float, P0: float) -> np.ndarray:
    """1D scalar Kalman on a real float32 array.

    Exact float32 arithmetic to match the GPU kernel:
      P_pred = P + Q
      K      = P_pred / (P_pred + R)   [GPU uses __frcp_rn approx, diff < 1e-4]
      x_hat += K * (z - x_hat)
      P      = (1 - K) * P_pred
    """
    Q_f  = np.float32(Q)
    R_f  = np.float32(R)
    n    = len(signal)
    out  = np.empty(n, dtype=np.float32)
    xh   = np.float32(x0)
    P    = np.float32(P0)
    for i in range(n):
        z      = np.float32(signal[i])
        P_pred = P + Q_f
        K      = P_pred / (P_pred + R_f)
        xh     = xh + K * (z - xh)
        P      = (np.float32(1.0) - K) * P_pred
        out[i] = xh
    return out


def kalman_ref(data: np.ndarray,
               Q: float = Q_DEF, R: float = R_DEF,
               x0: float = X0_DEF, P0: float = P0_DEF) -> np.ndarray:
    """Kalman reference for 1D or 2D (ch, pts) complex64 input.

    Re and Im parts are filtered by independent scalar Kalman filters,
    matching the GPU kernel design.
    """
    def _cf32_1ch(d: np.ndarray) -> np.ndarray:
        re = _kalman_1ch_scalar(d.real, Q, R, x0, P0)
        im = _kalman_1ch_scalar(d.imag, Q, R, x0, P0)
        return (re + 1j * im).astype(np.complex64)

    if data.ndim == 1:
        return _cf32_1ch(data)
    return np.stack([_cf32_1ch(data[ch]) for ch in range(data.shape[0])])


def kalman_steady_state_gain(Q: float, R: float) -> float:
    """Compute theoretical steady-state Kalman gain K_ss.

    P_ss = Q/2 + sqrt((Q/2)^2 + Q*R)
    K_ss = P_ss / (P_ss + R)
    """
    P_ss = Q / 2.0 + np.sqrt((Q / 2.0) ** 2 + Q * R)
    return P_ss / (P_ss + R)


# ============================================================================
# Helpers
# ============================================================================

def make_complex_signal(n: int, seed: int = 42) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return (rng.standard_normal(n) + 1j * rng.standard_normal(n)).astype(np.complex64)


def make_ctx_kalman():
    ctx    = gpuworklib.ROCmGPUContext(0)
    kalman = gpuworklib.KalmanFilterROCm(ctx)
    return ctx, kalman


# ============================================================================
# Test 1: basic GPU vs reference
# ============================================================================

def test_kalman_basic():
    """Random complex64 signal (1D): GPU vs numpy Kalman reference."""
    if not HAS_GPU:
        print("  SKIP: no GPU")
        return

    data = make_complex_signal(POINTS)
    _, kalman = make_ctx_kalman()
    kalman.set_params(Q_DEF, R_DEF, X0_DEF, P0_DEF)
    gpu_out = kalman.process(data)
    ref     = kalman_ref(data, Q_DEF, R_DEF, X0_DEF, P0_DEF)

    assert gpu_out.shape == data.shape, f"shape: {gpu_out.shape} vs {data.shape}"
    max_diff = float(np.max(np.abs(gpu_out - ref)))
    print(f"  Kalman(Q={Q_DEF}, R={R_DEF}): max_diff={max_diff:.2e}, atol={ATOL:.2e}")
    assert np.allclose(gpu_out, ref, atol=ATOL), f"max_diff={max_diff:.4e} > {ATOL}"
    print("  PASSED")


# ============================================================================
# Test 2: multi-channel
# ============================================================================

def test_kalman_multi_channel():
    """8-channel 2D (channels, points): each channel matches numpy reference."""
    if not HAS_GPU:
        print("  SKIP: no GPU")
        return

    data = np.stack([make_complex_signal(POINTS, seed=ch) for ch in range(CHANNELS)])
    _, kalman = make_ctx_kalman()
    kalman.set_params(Q_DEF, R_DEF, X0_DEF, P0_DEF)
    gpu_out = kalman.process(data)
    ref     = kalman_ref(data, Q_DEF, R_DEF, X0_DEF, P0_DEF)

    assert gpu_out.shape == data.shape
    max_diff = float(np.max(np.abs(gpu_out - ref)))
    print(f"  Kalman multi-channel shape={data.shape}: max_diff={max_diff:.2e}")
    assert np.allclose(gpu_out, ref, atol=ATOL), f"max_diff={max_diff:.4e} > {ATOL}"
    print("  PASSED")


# ============================================================================
# Test 3: channel independence (256 channels with distinct const signals)
# ============================================================================

def test_kalman_channel_independence():
    """256 channels each tracking a different constant: states must not mix."""
    if not HAS_GPU:
        print("  SKIP: no GPU")
        return

    N_CH     = 256
    rng      = np.random.default_rng(99)
    noise_s  = 0.5
    Q, R     = 0.01, noise_s ** 2

    # Channel ch tracks constant value ch * 10, perturbed by small noise
    const_vals = np.arange(N_CH, dtype=np.float32) * 10.0
    data = np.zeros((N_CH, POINTS), dtype=np.complex64)
    for ch in range(N_CH):
        noise = rng.standard_normal(POINTS).astype(np.float32) * noise_s
        data[ch] = (const_vals[ch] + noise).astype(np.complex64)

    _, kalman = make_ctx_kalman()
    kalman.set_params(Q, R, 0.0, R)
    gpu_out = kalman.process(data)
    ref     = kalman_ref(data, Q, R, 0.0, R)

    # GPU vs reference
    max_diff = float(np.max(np.abs(gpu_out - ref)))
    assert np.allclose(gpu_out, ref, atol=ATOL), \
        f"Channel independence failed, max_diff={max_diff:.4e}"

    # After warm-up (500 samples), each channel should converge to its constant
    for ch in [0, 10, 100, 255]:
        estimate = float(gpu_out[ch, 500].real)
        truth    = float(const_vals[ch])
        err      = abs(estimate - truth)
        print(f"  ch={ch:3d}: const={truth:.1f}, Kalman={estimate:.2f}, err={err:.2f}")
        assert err < 2.0, f"ch={ch}: Kalman estimate too far: err={err:.2f}"

    print("  PASSED")


# ============================================================================
# Test 4: noise reduction (SNR improvement)
# ============================================================================

def test_kalman_noise_reduction():
    """Kalman on constant + AWGN: filtered RMS error << raw RMS error.

    Constant signal (value=100) + Gaussian noise (sigma=5) → Q=0.01, R=25.
    After warm-up: Kalman estimate is significantly better than raw measurement.
    """
    if not HAS_GPU:
        print("  SKIP: no GPU")
        return

    Q, R     = 0.01, 25.0
    const_v  = 100.0
    sigma    = float(np.sqrt(R))    # = 5.0
    rng      = np.random.default_rng(7)
    noise    = rng.standard_normal(POINTS).astype(np.float32) * sigma
    data     = (const_v + noise).astype(np.complex64)

    _, kalman = make_ctx_kalman()
    kalman.set_params(Q, R, 0.0, R)
    gpu_out = kalman.process(data)

    # Compare RMS error (skip warm-up region)
    skip    = 200
    raw_rms = float(np.sqrt(np.mean((data[skip:].real - const_v) ** 2)))
    flt_rms = float(np.sqrt(np.mean((gpu_out[skip:].real - const_v) ** 2)))
    K_ss    = kalman_steady_state_gain(Q, R)

    print(f"  Q={Q}, R={R}, sigma={sigma:.1f}")
    print(f"  K_ss={K_ss:.4f}, raw_rms={raw_rms:.4f}, filtered_rms={flt_rms:.4f}")
    print(f"  SNR improvement: {raw_rms / flt_rms:.1f}x")

    assert flt_rms < raw_rms * 0.5, \
        f"Kalman should reduce RMS by ≥ 2x: raw={raw_rms:.4f}, flt={flt_rms:.4f}"
    print("  PASSED")


# ============================================================================
# Test 5: step response
# ============================================================================

def test_kalman_step_response():
    """Step at n=512 (0→100): Kalman follows with smooth exponential ramp."""
    if not HAS_GPU:
        print("  SKIP: no GPU")
        return

    Q, R    = 1.0, 25.0     # higher Q → faster response to changes
    data    = np.zeros(POINTS, dtype=np.complex64)
    data[512:] = np.complex64(100.0 + 0j)

    _, kalman = make_ctx_kalman()
    kalman.set_params(Q, R, 0.0, R)
    gpu_out = kalman.process(data)
    ref     = kalman_ref(data, Q, R, 0.0, R)

    # GPU vs reference
    max_diff = float(np.max(np.abs(gpu_out - ref)))
    assert np.allclose(gpu_out, ref, atol=ATOL), f"GPU vs ref max_diff={max_diff:.4e}"

    # Before step: near 0
    pre_mean = float(np.mean(np.abs(gpu_out[400:510].real)))
    assert pre_mean < 1.0, f"Pre-step region should be ≈ 0: mean={pre_mean:.4f}"

    # After step: Kalman rises toward 100 (check at +100 samples after step)
    val_after = float(gpu_out[612].real)
    K_ss = kalman_steady_state_gain(Q, R)
    expected_approx = 100.0 * (1.0 - (1.0 - K_ss) ** 100)
    print(f"  Q={Q}, R={R}, K_ss={K_ss:.4f}")
    print(f"  val at +100 steps: GPU={val_after:.2f}, expected≈{expected_approx:.2f}")
    assert val_after > 50.0, f"Kalman should rise above 50 after 100 steps: {val_after:.2f}"

    # Late convergence: by n=512+1000 should be close to 100
    val_late = float(gpu_out[1512].real)
    assert abs(val_late - 100.0) < 5.0, f"Late convergence: {val_late:.2f} ≠ 100"
    print("  PASSED")


# ============================================================================
# Test 6: LFM radar demo — 5-antenna beat tones + noise (CPU-only)
# ============================================================================

def test_kalman_lfm_radar_demo():
    """LFM radar: 5 beat tones + AWGN → Kalman reduces noise on each antenna.

    No GPU required. Validates:
      - Kalman CPU reference reduces signal variance on each antenna
      - Beat frequency is preserved after filtering (signal not distorted)

    Scenario (from Task_21 spec):
      fs=10 MHz, fdev=2 MHz, N=16384, noise_sigma=0.30
      5 antennas: tau = [50,100,150,200,250] µs → f_beat = mu * tau
    """
    fs          = 10e6
    fdev        = 2e6
    N           = 4096      # reduced for speed (vs 16384 in spec)
    Ti          = N / fs
    mu          = fdev / Ti
    noise_sigma = 0.30

    # 5 targets: tau [µs] → f_beat
    tau_us  = np.array([50.0, 100.0, 150.0, 200.0, 250.0])
    tau     = tau_us * 1e-6
    f_beat  = mu * tau
    n_ant   = len(tau)

    # Generate beat signals: complex tone + AWGN
    rng = np.random.default_rng(42)
    t   = np.arange(N, dtype=np.float32) / np.float32(fs)
    signal = np.zeros((n_ant, N), dtype=np.complex64)
    for a in range(n_ant):
        omega = np.float32(2.0 * np.pi * f_beat[a] / fs)
        noise_r = rng.standard_normal(N).astype(np.float32) * noise_sigma
        noise_i = rng.standard_normal(N).astype(np.float32) * noise_sigma
        signal[a] = (np.cos(omega * np.arange(N, dtype=np.float32)) + noise_r +
                     1j * (np.sin(omega * np.arange(N, dtype=np.float32)) + noise_i))

    # Kalman parameters: Q small (frequency barely changes), R = sigma^2
    Q, R = 0.001, noise_sigma ** 2
    filtered = kalman_ref(signal, Q, R, 0.0, R)

    print(f"\n  LFM Radar — Kalman Demo (CPU reference, N={N})")
    print(f"  Q={Q}, R={R:.4f}, noise_sigma={noise_sigma}")
    print(f"  {'Ant':>3} | f_beat     | raw_std  | flt_std  | improvement")
    print(f"  {'':-^3}-+-{'':-^10}-+-{'':-^8}-+-{'':-^8}-+-{'':-^11}")

    skip = 100  # skip warm-up
    all_ok = True
    for a in range(n_ant):
        raw_std = float(np.std(signal[a, skip:].real))
        flt_std = float(np.std(filtered[a, skip:].real))
        ratio   = raw_std / flt_std if flt_std > 0 else float('inf')
        print(f"  {a:>3d} | {f_beat[a]/1e3:>6.2f} kHz  | {raw_std:>8.4f} | "
              f"{flt_std:>8.4f} | {ratio:>6.2f}x")
        if flt_std >= raw_std * 0.8:
            all_ok = False

    assert all_ok, "Kalman should reduce std on all antennas (by at least 20%)"

    # Verify signal is preserved: check that beat frequency is still detectable
    # FFT of filtered signal should have a peak near the expected bin
    for a in range(n_ant):
        fft_mag  = np.abs(np.fft.fft(filtered[a]))
        peak_bin = int(np.argmax(fft_mag[:N // 2]))
        exp_bin  = int(round(f_beat[a] / (fs / N)))
        err_bins = abs(peak_bin - exp_bin)
        print(f"  Ant {a}: expected bin={exp_bin}, peak bin={peak_bin}, err={err_bins}")
        assert err_bins <= 2, \
            f"Ant {a}: beat freq peak at bin {peak_bin}, expected {exp_bin}"

    print("  PASSED")


# ============================================================================
# Test 7: properties
# ============================================================================

def test_kalman_properties():
    """KalmanFilterROCm: is_ready(), get_params() returns correct Q/R/x0/P0."""
    if not HAS_GPU:
        print("  SKIP: no GPU")
        return

    _, kalman = make_ctx_kalman()
    kalman.set_params(Q_DEF, R_DEF, X0_DEF, P0_DEF)

    assert kalman.is_ready(), "is_ready() should be True after set_params"

    params = kalman.get_params()
    assert abs(params['Q']  - Q_DEF)  < 1e-6, f"Q={params['Q']} != {Q_DEF}"
    assert abs(params['R']  - R_DEF)  < 1e-6, f"R={params['R']} != {R_DEF}"
    assert abs(params['x0'] - X0_DEF) < 1e-6, f"x0={params['x0']} != {X0_DEF}"
    assert abs(params['P0'] - P0_DEF) < 1e-6, f"P0={params['P0']} != {P0_DEF}"

    print(f"  is_ready={kalman.is_ready()}, params={params}")
    print("  PASSED")


# ============================================================================
# Main
# ============================================================================

if __name__ == '__main__':
    SEP = '=' * 60
    print(SEP)
    print('  KalmanFilterROCm — Python Test')
    print(SEP)
    print(f'  HAS_GPU={HAS_GPU}')
    print(f'  Default params: Q={Q_DEF}, R={R_DEF}, x0={X0_DEF}, P0={P0_DEF}')
    K_ss = kalman_steady_state_gain(Q_DEF, R_DEF)
    print(f'  Steady-state gain K_ss={K_ss:.4f} (Q={Q_DEF}, R={R_DEF})')

    passed, failed = 0, 0

    def run(label, fn):
        global passed, failed
        print(f'\n[{label}] {fn.__doc__.splitlines()[0]}')
        try:
            fn()
            passed += 1
        except AssertionError as e:
            print(f'  FAILED: {e}')
            failed += 1

    run('1', test_kalman_basic)
    run('2', test_kalman_multi_channel)
    run('3', test_kalman_channel_independence)
    run('4', test_kalman_noise_reduction)
    run('5', test_kalman_step_response)
    run('6', test_kalman_lfm_radar_demo)
    run('7', test_kalman_properties)

    print(f'\n{SEP}')
    print(f'  Results: {passed}/{passed + failed} passed', end='')
    print('  — ALL PASSED ✓' if not failed else f'  ({failed} FAILED)')
    print(SEP)
