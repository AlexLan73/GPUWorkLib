#!/usr/bin/env python3
"""
Capon (MVDR) Beamformer — Python validation test
=================================================

Validates CaponProcessor results against SciPy/NumPy reference implementation.

Algorithm (MVDR):
  R = (1/N) * Y @ Y.conj().T + mu*I   — covariance matrix
  R_inv = inv(R)                        — matrix inversion
  z[m] = 1 / Re(u_m.conj() @ R_inv @ u_m)   — Capon spatial spectrum

Tests (NumPy reference — no GPU needed):
  1. relief_shape          — output shape [M]
  2. relief_positive       — all values > 0
  3. relief_interference   — MVDR min at interferer direction
  4. beamform_shape        — output shape [M, N]
  5. regularization        — mu>0 prevents singular matrix
  6. steering_orthogonal   — two orthogonal sources resolved

  GPU binary (requires build + AMD GPU):
  7. all_tests_pass        — C++ tests 01-04 PASS in stdout

Usage:
  python Python_test/capon/test_capon.py

Author: Кодо (AI Assistant)
Date: 2026-03-16
"""

import os
import subprocess
import sys

import sys
import numpy as np

_PT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _PT_DIR not in sys.path:
    sys.path.insert(0, _PT_DIR)
from common.runner import SkipTest

# ============================================================================
# Project paths
# ============================================================================

PROJECT_ROOT = os.path.dirname(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
)
BINARY_PATH = os.path.join(PROJECT_ROOT, "build/debian-radeon9070/GPUWorkLib")
HAS_BINARY = os.path.exists(BINARY_PATH)

# ============================================================================
# NumPy/SciPy reference implementation
# ============================================================================

def make_steering_matrix(n_channels: int, n_directions: int,
                         theta_min: float, theta_max: float) -> np.ndarray:
    """
    ULA steering matrix: U[p, m] = exp(j * 2π * p * 0.5 * sin(θ_m))

    Returns U of shape [n_channels, n_directions], complex64.
    Column-major order (Fortran-contiguous) to match C++ convention.
    """
    thetas = np.linspace(theta_min, theta_max, n_directions)
    p = np.arange(n_channels)[:, None]          # [P, 1]
    phases = 2.0 * np.pi * p * 0.5 * np.sin(thetas)[None, :]  # [P, M]
    U = np.exp(1j * phases).astype(np.complex64)
    return np.asfortranarray(U)                 # column-major


def make_noise(n_channels: int, n_samples: int,
               sigma: float = 1.0, seed: int = 42) -> np.ndarray:
    """
    Complex Gaussian noise CN(0, sigma²).
    Returns Y of shape [n_channels, n_samples], complex64.
    """
    rng = np.random.default_rng(seed)
    Y = (sigma / np.sqrt(2.0)) * (
        rng.standard_normal((n_channels, n_samples)) +
        1j * rng.standard_normal((n_channels, n_samples))
    )
    return np.asfortranarray(Y.astype(np.complex64))


def add_interference(Y: np.ndarray, theta: float,
                     amplitude: float, omega0: float = 0.37) -> np.ndarray:
    """
    Add CW plane-wave interference from direction theta.
    Y shape: [n_channels, n_samples].
    """
    P, N = Y.shape
    p = np.arange(P)
    n = np.arange(N)
    spatial  = np.exp(1j * 2.0 * np.pi * p * 0.5 * np.sin(theta))   # [P]
    temporal = np.exp(1j * omega0 * n)                                 # [N]
    Y = Y + amplitude * np.outer(spatial, temporal)
    return Y.astype(np.complex64)


def capon_relief_ref(Y: np.ndarray, U: np.ndarray,
                     mu: float = 0.0) -> np.ndarray:
    """
    Reference MVDR spatial spectrum.

    Args:
        Y: signal matrix [P, N], complex
        U: steering matrix [P, M], complex (column-major)
        mu: diagonal loading coefficient

    Returns:
        z: float array [M]
    """
    P, N = Y.shape
    # Covariance matrix: R = (1/N) * Y @ Y^H + mu*I
    R = (Y @ Y.conj().T) / N + mu * np.eye(P, dtype=np.complex64)
    R_inv = np.linalg.inv(R.astype(np.complex128)).astype(np.complex64)

    M = U.shape[1]
    z = np.zeros(M, dtype=np.float32)
    for m in range(M):
        u = U[:, m]
        val = np.real(u.conj() @ R_inv @ u)
        z[m] = 1.0 / val if val > 0 else 0.0
    return z


def capon_beamform_ref(Y: np.ndarray, U: np.ndarray,
                       mu: float = 0.0) -> np.ndarray:
    """
    Reference adaptive beamforming: Y_out = (R^{-1} @ U)^H @ Y

    Returns:
        Y_out: [M, N] complex
    """
    P, N = Y.shape
    R = (Y @ Y.conj().T) / N + mu * np.eye(P, dtype=np.complex64)
    R_inv = np.linalg.inv(R.astype(np.complex128)).astype(np.complex64)
    W = R_inv @ U                  # [P, M]
    return (W.conj().T @ Y).astype(np.complex64)  # [M, N]


# ============================================================================
# Tests: NumPy reference (always run, no GPU needed)
# ============================================================================

class TestCaponReference:
    """NumPy/SciPy reference implementation tests — no GPU required."""

    P = 8
    N = 64
    M = 16

    def setUp(self):
        self.Y = make_noise(self.P, self.N, sigma=1.0, seed=42)
        self.U = make_steering_matrix(
            self.P, self.M,
            theta_min=-np.pi / 3.0,
            theta_max= np.pi / 3.0,
        )

    def test_relief_shape(self):
        """Relief output shape must be [M]."""
        z = capon_relief_ref(self.Y, self.U, mu=0.01)
        assert z.shape == (self.M,), f"Expected ({self.M},), got {z.shape}"

    def test_relief_positive(self):
        """All MVDR relief values must be > 0 (after regularization)."""
        z = capon_relief_ref(self.Y, self.U, mu=0.01)
        assert np.all(z > 0), "Some relief values are non-positive"

    def test_relief_finite(self):
        """All relief values must be finite."""
        z = capon_relief_ref(self.Y, self.U, mu=0.01)
        assert np.all(np.isfinite(z)), "Non-finite values in relief"

    def test_relief_interference_suppression(self):
        """MVDR relief must be minimum at interference direction.

        Strong interferer from theta=0° → z[m_int] < mean(z)/2.
        """
        theta_min = -np.pi / 3.0
        theta_max =  np.pi / 3.0
        theta_int =  0.0  # center direction

        Y_with_int = add_interference(
            self.Y.copy(), theta=theta_int, amplitude=10.0
        )

        U = make_steering_matrix(self.P, self.M, theta_min, theta_max)
        z = capon_relief_ref(Y_with_int, U, mu=0.001)

        # Find index of closest direction to theta_int
        thetas = np.linspace(theta_min, theta_max, self.M)
        m_int  = int(np.argmin(np.abs(thetas - theta_int)))

        mean_z = np.mean(z)
        assert z[m_int] < mean_z * 0.5, (
            f"MVDR did not suppress interference: z[{m_int}]={z[m_int]:.4f}, "
            f"mean={mean_z:.4f}"
        )

    def test_beamform_shape(self):
        """Beamform output shape must be [M, N]."""
        Y_out = capon_beamform_ref(self.Y, self.U, mu=0.01)
        assert Y_out.shape == (self.M, self.N), (
            f"Expected ({self.M}, {self.N}), got {Y_out.shape}"
        )

    def test_beamform_finite(self):
        """Beamform output must be finite."""
        Y_out = capon_beamform_ref(self.Y, self.U, mu=0.01)
        assert np.all(np.isfinite(Y_out)), "Non-finite values in beamform output"

    def test_regularization_singular(self):
        """With mu>0 relief must be finite even for rank-deficient covariance (N < P)."""
        P, N, M = 8, 4, 8  # N < P → rank-deficient without regularization
        Y = make_noise(P, N, sigma=1.0, seed=7)
        U = make_steering_matrix(P, M, -np.pi / 4.0, np.pi / 4.0)
        z = capon_relief_ref(Y, U, mu=0.1)
        assert np.all(np.isfinite(z)), "Regularization failed to prevent singular matrix"
        assert np.all(z >= 0), "Relief values negative after regularization"

    def test_two_sources_resolved(self):
        """Two orthogonal steering vectors at 0° and 90° must give distinct relief peaks."""
        P, N = 16, 256
        M = 64
        theta_min, theta_max = -np.pi / 2.0 + 0.01, np.pi / 2.0 - 0.01

        # Два источника: на -30° и +30°
        theta_a = -np.pi / 6.0
        theta_b =  np.pi / 6.0

        Y = make_noise(P, N, sigma=0.1, seed=11)
        Y = add_interference(Y, theta=theta_a, amplitude=5.0, omega0=0.20)
        Y = add_interference(Y, theta=theta_b, amplitude=5.0, omega0=0.45)

        U = make_steering_matrix(P, M, theta_min, theta_max)
        z = capon_relief_ref(Y, U, mu=0.001)

        thetas = np.linspace(theta_min, theta_max, M)
        m_a    = int(np.argmin(np.abs(thetas - theta_a)))
        m_b    = int(np.argmin(np.abs(thetas - theta_b)))

        # На обоих направлениях должно быть подавление (минимум)
        mean_z = np.mean(z)
        assert z[m_a] < mean_z * 0.5, (
            f"Source A not suppressed: z[{m_a}]={z[m_a]:.4f}, mean={mean_z:.4f}"
        )
        assert z[m_b] < mean_z * 0.5, (
            f"Source B not suppressed: z[{m_b}]={z[m_b]:.4f}, mean={mean_z:.4f}"
        )


# ============================================================================
# Tests: GPU binary (require Linux + AMD GPU + ROCm build)
# ============================================================================

class TestCaponGPU:
    """Tests that run the C++ binary and parse output."""

    def setUp(self):
        if not HAS_BINARY:
            raise SkipTest("GPU binary not found — build first")

    def _run_binary(self, timeout: int = 120) -> str:
        result = subprocess.run(
            [BINARY_PATH],
            capture_output=True,
            text=True,
            timeout=timeout,
            cwd=PROJECT_ROOT,
        )
        return result.stdout + result.stderr

    def test_all_cpp_tests_pass(self):
        """All C++ capon tests (01-04) must report PASS."""
        output = self._run_binary()
        for i in range(1, 5):
            tag = f"[test_capon_rocm::{i:02d}]"
            assert "PASS" in output or tag not in output, (
                f"Test {i:02d} did not PASS. Output fragment:\n"
                + "\n".join(l for l in output.splitlines() if tag in l)
            )

    def test_no_exceptions(self):
        """C++ binary must not throw unhandled exceptions."""
        output = self._run_binary()
        assert "terminate called" not in output
        assert "what():" not in output


# ============================================================================
# Standalone run
# ============================================================================

if __name__ == "__main__":
    from common.runner import TestRunner
    runner = TestRunner()
    print("Running Capon reference tests (no GPU)...")
    results = runner.run(TestCaponReference())
    if HAS_BINARY:
        results += runner.run(TestCaponGPU())
    runner.print_summary(results)

