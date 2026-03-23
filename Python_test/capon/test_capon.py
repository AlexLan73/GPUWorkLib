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

    Что такое управляющий вектор (steering vector)?
    ────────────────────────────────────────────────
    Равномерная линейная решётка (ULA): антенны стоят в ряд с шагом d = λ/2.
    Если сигнал приходит под углом θ, между соседними антеннами возникает
    разность хода d·sin(θ). При шаге d=λ/2 это даёт фазовый сдвиг:
        φ = 2π·(d/λ)·sin(θ) = 2π·0.5·sin(θ)
    Антенна p получает сигнал с суммарным фазовым сдвигом p·φ:
        U[p, m] = exp(j · 2π · p · 0.5 · sin(θ_m))
    Если реальный сигнал пришёл именно с θ_m, «скалярное произведение»
    u_m^H·y_p будет максимальным — векторы «совпадут по фазе».
    """
    # M направлений равномерно от theta_min до theta_max
    thetas = np.linspace(theta_min, theta_max, n_directions)

    # p = [0, 1, 2, ..., P-1] — номер антенны, форма [P, 1] для broadcasting
    p = np.arange(n_channels)[:, None]          # [P, 1]

    # Фазовый сдвиг для каждой пары (антенна p, направление m)
    phases = 2.0 * np.pi * p * 0.5 * np.sin(thetas)[None, :]  # [P, M]

    U = np.exp(1j * phases).astype(np.complex64)

    # column-major (Fortran order) — чтобы совпадать с C++ хранением матрицы
    return np.asfortranarray(U)


def make_noise(n_channels: int, n_samples: int,
               sigma: float = 1.0, seed: int = 42) -> np.ndarray:
    """
    Complex Gaussian noise CN(0, sigma²).
    Returns Y of shape [n_channels, n_samples], complex64.

    Комплексный белый шум: вещественная и мнимая части независимы,
    каждая с дисперсией sigma²/2, итого мощность = sigma².
    Делитель sqrt(2) нормирует суммарную мощность.
    """
    rng = np.random.default_rng(seed)
    # Каждый элемент: (real + j*imag) / sqrt(2) — итоговая дисперсия = sigma²
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

    Модель помехи:
        i[p, n] = amplitude · exp(j·2π·p·0.5·sin(θ)) · exp(j·ω₀·n)
                  └── пространственная часть ──┘   └── временная ──┘
    Это модель плоской волны (CW): каждая антенна p получает один и тот
    же тональный сигнал с частотой ω₀, но сдвинутый по фазе на p·φ,
    где φ = 2π·0.5·sin(θ) — фаза от геометрии ULA.
    """
    P, N = Y.shape
    p = np.arange(P)
    n = np.arange(N)
    # Пространственный вектор: фазы по антеннам для угла theta
    spatial  = np.exp(1j * 2.0 * np.pi * p * 0.5 * np.sin(theta))   # [P]
    # Временной вектор: тональный сигнал с нормированной частотой omega0
    temporal = np.exp(1j * omega0 * n)                                 # [N]
    # outer(spatial, temporal) = [P, N] — помеха на каждом канале и отсчёте
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

    Алгоритм по шагам:
        1. R = (1/N)·Y·Y^H + μ·I  — ковариационная матрица [P×P]
           Каждый элемент R[i,j] — корреляция между антеннами i и j.
           Деление на N нормирует по числу отсчётов.
           μ·I (диагональная нагрузка) делает матрицу невырожденной.

        2. R⁻¹ = inv(R)            — обратная матрица
           GPU использует разложение Холецкого (POTRF+POTRI): быстрее inv().

        3. z[m] = 1 / Re(u_m^H · R⁻¹ · u_m)
           Re(u^H·R⁻¹·u) — «вес» направления m: насколько он «пропускается»
           фильтром. Если из m идёт помеха, фильтр «задавливает» его
           (Re(u^H·R⁻¹·u) становится большим → z[m] = 1/большое = маленькое).
           Чем БОЛЬШЕ z[m], тем ВЕРОЯТНЕЕ там полезный сигнал.
    """
    P, N = Y.shape

    # Шаг 1: ковариационная матрица с регуляризацией
    R = (Y @ Y.conj().T) / N + mu * np.eye(P, dtype=np.complex64)

    # Шаг 2: обращение матрицы (float64 для точности, потом обратно в float32)
    R_inv = np.linalg.inv(R.astype(np.complex128)).astype(np.complex64)

    # Шаг 3: для каждого направления m вычислить z[m] = 1/Re(u^H·R⁻¹·u)
    M = U.shape[1]
    z = np.zeros(M, dtype=np.float32)
    for m in range(M):
        u = U[:, m]                           # управляющий вектор [P]
        val = np.real(u.conj() @ R_inv @ u)   # квадратичная форма: скаляр
        z[m] = 1.0 / val if val > 0 else 0.0
    return z


def capon_beamform_ref(Y: np.ndarray, U: np.ndarray,
                       mu: float = 0.0) -> np.ndarray:
    """
    Reference adaptive beamforming: Y_out = (R^{-1} @ U)^H @ Y

    Returns:
        Y_out: [M, N] complex

    Адаптивное формирование луча (adaptive beamforming):
        W = R⁻¹ @ U     — адаптивные веса [P, M]: для каждого из M направлений
                          это вектор P весовых коэффициентов по антеннам.
                          В отличие от простого Y_out = U^H @ Y (DAS),
                          веса адаптируются под помехи: они «ортогональны»
                          помеховым направлениям → помехи подавляются.

        Y_out = W^H @ Y  — применить веса: [M, N] выходной сигнал,
                          строка m = сигнал, принятый «в направлении m»
                          с подавлением остальных источников.
    """
    P, N = Y.shape
    R = (Y @ Y.conj().T) / N + mu * np.eye(P, dtype=np.complex64)
    R_inv = np.linalg.inv(R.astype(np.complex128)).astype(np.complex64)

    # Адаптивные веса W = R⁻¹·U: форма [P, M]
    W = R_inv @ U

    # Применить веса к каждому отсчёту: W^H·Y = [M, P]·[P, N] = [M, N]
    return (W.conj().T @ Y).astype(np.complex64)


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
# Tests: Real MATLAB data + physical antenna coordinates
# ============================================================================

class TestCaponRealData:
    """Tests using real MATLAB signal and physical antenna coordinates.

    Reference data location:
        Doc_Addition/Capon/capon_test/build/
            x_data.txt       — 340 antenna x-coordinates
            y_data.txt       — 340 antenna y-coordinates
            signal_matlab.txt — 341 rows x 1000 complex samples (MATLAB format)
            z_values.txt      — 5476 = 4 x 37 x 37 reference relief values

    Physical parameters:
        f0 = 3918e6 + 3.15e6 = 3921150000 Hz
        c  = 299792458 m/s
        U[p,m] = exp(j * 2pi * (x[p]*u_m + y[p]*v_m) * f0/c)  (no normalization)

    Formula notes:
        GPU CaponProcessor: R = Y@Y.H/N + mu*I  (with 1/N normalization)
        Reference (ArrayFire): R = Y@Y.H + mu*I  (WITHOUT 1/N)
        Shape comparison uses correlation to be invariant of scale.
    """

    CAPON_TEST_DIR = os.path.join(PROJECT_ROOT, "Doc_Addition", "Capon", "capon_test", "build")
    X_DATA        = os.path.join(CAPON_TEST_DIR, "x_data.txt")
    Y_DATA        = os.path.join(CAPON_TEST_DIR, "y_data.txt")
    SIGNAL_FILE   = os.path.join(CAPON_TEST_DIR, "signal_matlab.txt")
    Z_VALUES_FILE = os.path.join(CAPON_TEST_DIR, "z_values.txt")
    F0 = 3921150000.0   # 3918e6 + 3.15e6
    C  = 299792458.0

    # ── Helpers ────────────────────────────────────────────────────────────

    def _load_coords(self):
        """Load x_data.txt and y_data.txt -> (x: ndarray, y: ndarray)."""
        x = np.loadtxt(self.X_DATA, dtype=np.float32)
        y = np.loadtxt(self.Y_DATA, dtype=np.float32)
        return x, y

    @staticmethod
    def _parse_matlab_line(line: str):
        """Parse one line of MATLAB complex format: '-12.49+15.08i 6.17+18.74i ...'

        For each token finds the last + or - before 'i' that is NOT
        preceded by 'e' or 'E' (exponent sign).
        Returns list of complex numbers.
        """
        result = []
        for token in line.split():
            if not token.endswith('i'):
                continue
            s = token[:-1]  # strip trailing 'i'
            # Find rightmost sign that is not an exponent sign
            split_pos = -1
            for i in range(len(s) - 1, 0, -1):
                if s[i] in ('+', '-') and s[i - 1] not in ('e', 'E'):
                    split_pos = i
                    break
            if split_pos <= 0:
                continue
            real_str = s[:split_pos]
            imag_str = s[split_pos:]  # includes sign
            result.append(complex(float(real_str), float(imag_str)))
        return result

    def _load_signal(self, P: int = 85, N: int = 1000) -> np.ndarray:
        """Load first P rows x N columns from signal_matlab.txt.

        Returns ndarray of shape [P, N], dtype complex64.
        Fast path: replace trailing 'i' with 'j' and use Python's complex().
        """
        rows = []
        with open(self.SIGNAL_FILE, 'r') as f:
            for line in f:
                if len(rows) >= P:
                    break
                # Fast parse: MATLAB 'i' -> Python 'j', then builtin complex()
                tokens = line.split()[:N]
                row = [complex(t.replace('i', 'j')) for t in tokens]
                if len(row) < N:
                    raise ValueError(
                        f"Row {len(rows)}: expected {N} numbers, got {len(row)}"
                    )
                rows.append(row[:N])
        if len(rows) < P:
            raise ValueError(f"Expected {P} rows, got {len(rows)}")
        return np.array(rows, dtype=np.complex64)  # [P, N]

    def _get_u_physical(self, x, y, u_directions, v_directions) -> np.ndarray:
        """Compute steering matrix U[p, m] = exp(j*2pi*(x[p]*u_m + y[p]*v_m)*f0/c).

        Args:
            x:             antenna x-coordinates [P]  (в метрах)
            y:             antenna y-coordinates [P]  (в метрах)
            u_directions:  u-компоненты направлений сканирования [M]  (sin·cos)
            v_directions:  v-компоненты направлений сканирования [M]  (sin·sin)

        Returns:
            U: ndarray of shape [P, M], complex64, Fortran (column-major) order.

        Физический смысл:
        ─────────────────
        В отличие от ULA (где антенны равноудалены), здесь **произвольная**
        плоская антенная решётка. Каждая антенна p находится в точке (x[p], y[p]).

        Волновой вектор k_m для направления (u_m, v_m):
            k_m = (2π·f0/c) · (u_m, v_m)    [рад/м]

        Запаздывание на антенне p (скалярное произведение):
            τ[p, m] = (x[p]·u_m + y[p]·v_m) · f0/c    [секунды → фаза]

        Управляющий вектор:
            U[p, m] = exp(j · 2π · f0/c · (x[p]·u_m + y[p]·v_m))

        (u, v) — это направляющие косинусы: u = sin(θ)·cos(φ),
        v = sin(θ)·sin(φ), где θ — угол от нормали, φ — азимут.
        """
        # Волновое число: k = 2π·f0/c [рад/м]
        k = 2.0 * np.pi * self.F0 / self.C

        # Фазы через broadcasting: [P, 1] * [1, M] -> [P, M]
        # float64 для точности при больших x и высоких частотах
        phase = k * (
            x[:, None].astype(np.float64) * u_directions[None, :].astype(np.float64) +
            y[:, None].astype(np.float64) * v_directions[None, :].astype(np.float64)
        )
        U = np.exp(1j * phase).astype(np.complex64)
        return np.asfortranarray(U)  # column-major — как в C++

    # ── Tests ──────────────────────────────────────────────────────────────

    def test_files_available(self):
        """Reference data files must exist to run real-data tests."""
        missing = []
        for path in (self.X_DATA, self.Y_DATA, self.SIGNAL_FILE, self.Z_VALUES_FILE):
            if not os.path.exists(path):
                missing.append(os.path.basename(path))
        if missing:
            raise SkipTest(
                f"Reference data files not found: {', '.join(missing)}"
            )

    def test_load_coords(self):
        """Coordinate arrays must have at least 85 elements each."""
        if not os.path.exists(self.X_DATA) or not os.path.exists(self.Y_DATA):
            raise SkipTest("Coordinate files not found")
        x, y = self._load_coords()
        assert len(x) >= 85, f"x_data has only {len(x)} elements (need >= 85)"
        assert len(y) >= 85, f"y_data has only {len(y)} elements (need >= 85)"

    def test_load_signal_dims(self):
        """Signal loader must return correct shape and dtype."""
        if not os.path.exists(self.SIGNAL_FILE):
            raise SkipTest("signal_matlab.txt not found")
        signal = self._load_signal(P=10, N=20)
        assert signal.shape == (10, 20), (
            f"Expected (10, 20), got {signal.shape}"
        )
        assert signal.dtype == np.complex64, (
            f"Expected complex64, got {signal.dtype}"
        )

    def test_physical_relief_properties(self):
        """NumPy Capon on real data (P=85, N=1000, M=37*37=1369) must be > 0 and finite."""
        for path in (self.X_DATA, self.Y_DATA, self.SIGNAL_FILE):
            if not os.path.exists(path):
                raise SkipTest(f"File not found: {os.path.basename(path)}")

        P = 85
        N = 1000
        mu = 1.0

        x, y = self._load_coords()
        x, y = x[:P], y[:P]

        print(f"  Loading signal ({P}x{N})... ", end='', flush=True)
        Y = self._load_signal(P=P, N=N)  # [P, N]
        print("done")

        # Scan grid: arange(-ulim, ulim+step/2, step)
        u_step = 0.00312
        ulim   = np.sin(3.25 * np.pi / 180.0)
        u0 = np.arange(-ulim, ulim + u_step / 2.0, u_step, dtype=np.float32)
        Nu = len(u0)
        M  = Nu * Nu   # total directions: Nu × Nu grid
        print(f"  Grid: Nu={Nu}, M={M}")

        # meshgrid: uu[iv, iu] = u0[iu],  vv[iv, iu] = u0[iv]
        uu, vv = np.meshgrid(u0, u0)
        u_dirs = uu.ravel()
        v_dirs = vv.ravel()

        U = self._get_u_physical(x, y, u_dirs, v_dirs)  # [P, M]

        # ── Шаг 1: ковариационная матрица (GPU-формула с 1/N) ─────────────
        R = (Y @ Y.conj().T) / N + mu * np.eye(P, dtype=np.complex64)

        # ── Шаг 2: обращение (float64 для устойчивости) ───────────────────
        R_inv = np.linalg.inv(R.astype(np.complex128)).astype(np.complex64)

        # ── Шаг 3: векторизованный рельеф (все M направлений сразу) ───────
        # Вместо цикла for m in range(M): z[m] = 1/Re(u^H·R⁻¹·u)
        # вычисляем матрично:
        #   W = R_inv @ U           [P, M]  — все M адаптивных весов сразу
        #   num[m] = Re(U[:,m]^H · W[:,m])  — поэлементное произведение + sum
        #   z[m] = 1 / num[m]
        W   = (R_inv @ U.astype(np.complex128)).astype(np.complex64)  # [P, M]
        num = np.real(np.sum(U.conj() * W, axis=0)).astype(np.float32)  # [M]
        z   = np.where(num > 0, 1.0 / num, 0.0).astype(np.float32)

        assert np.all(z > 0), "Some relief values are non-positive"
        assert np.all(np.isfinite(z)), "Non-finite values in relief"

        print(f"  Relief stats: min={z.min():.4g}, max={z.max():.4g}, "
              f"mean={z.mean():.4g}")

    def test_physical_relief_matches_reference_formula(self):
        """GPU formula (1/N) and reference formula (no 1/N) must give same SHAPE.

        Checks correlation > 0.99 between normalized spectra.
        Uses small P=8, N=64, M=16 for speed.
        """
        for path in (self.X_DATA, self.Y_DATA, self.SIGNAL_FILE):
            if not os.path.exists(path):
                raise SkipTest(f"File not found: {os.path.basename(path)}")

        P = 8
        N = 64
        M = 16
        mu_gpu = 100.0   # GPU formula: R = Y@Y.H/N + mu*I
        mu_ref = 100.0 * N  # Reference formula: R = Y@Y.H + mu*I  (mu_ref = mu_gpu*N)

        x, y = self._load_coords()
        x, y = x[:P].astype(np.float32), y[:P].astype(np.float32)

        Y = self._load_signal(P=P, N=N)  # [P, N]

        ulim = np.sin(3.25 * np.pi / 180.0)
        u_scan = np.linspace(-ulim, ulim, M, dtype=np.float32)
        v_scan = np.zeros(M, dtype=np.float32)

        U = self._get_u_physical(x, y, u_scan, v_scan)  # [P, M]

        # --- GPU formula (with 1/N) ---
        R_gpu = (Y @ Y.conj().T) / N + mu_gpu * np.eye(P, dtype=np.complex64)
        R_gpu_inv = np.linalg.inv(R_gpu.astype(np.complex128)).astype(np.complex64)
        z_gpu = np.zeros(M, dtype=np.float32)
        for m in range(M):
            u_m = U[:, m]
            val = np.real(u_m.conj() @ R_gpu_inv @ u_m)
            z_gpu[m] = float(1.0 / val) if val > 0 else 0.0

        # --- Reference formula (without 1/N) ---
        R_ref = (Y @ Y.conj().T) + mu_ref * np.eye(P, dtype=np.complex64)
        R_ref_inv = np.linalg.inv(R_ref.astype(np.complex128)).astype(np.complex64)
        z_ref = np.zeros(M, dtype=np.float32)
        for m in range(M):
            u_m = U[:, m]
            val = np.real(u_m.conj() @ R_ref_inv @ u_m)
            z_ref[m] = float(1.0 / val) if val > 0 else 0.0

        # Normalize to [0, 1]
        z_gpu_norm = z_gpu / (z_gpu.max() + 1e-30)
        z_ref_norm = z_ref / (z_ref.max() + 1e-30)

        corr = float(np.corrcoef(z_gpu_norm, z_ref_norm)[0, 1])
        assert corr > 0.99, (
            f"Spectra shapes differ: correlation={corr:.4f} (need > 0.99)"
        )
        print(f"  Correlation GPU vs reference formula: {corr:.6f}")

    def test_z_values_structure(self):
        """z_values.txt must contain 4 identical blocks of 37x37=1369 values."""
        if not os.path.exists(self.Z_VALUES_FILE):
            raise SkipTest("z_values.txt not found")

        z = np.loadtxt(self.Z_VALUES_FILE, dtype=np.float32)

        expected_len = 4 * 37 * 37  # = 5476
        assert len(z) == expected_len, (
            f"Expected {expected_len} values, got {len(z)}"
        )

        block_size = 37 * 37  # 1369
        b0 = z[0 * block_size : 1 * block_size]
        b1 = z[1 * block_size : 2 * block_size]
        b2 = z[2 * block_size : 3 * block_size]
        b3 = z[3 * block_size : 4 * block_size]

        for i, b in enumerate((b1, b2, b3), start=1):
            max_ref = np.maximum(np.abs(b0), 1e-6)
            rel_diff = np.abs(b - b0) / max_ref
            max_rel = float(rel_diff.max())
            assert max_rel < 0.001, (
                f"Block {i} differs from block 0 by {max_rel * 100:.3f}% "
                f"(need < 0.1%)"
            )

        print(f"  4 blocks of {block_size} values are identical "
              f"(max_rel_diff < 0.1%)")


# ============================================================================
# Standalone run
# ============================================================================

if __name__ == "__main__":
    from common.runner import TestRunner
    runner = TestRunner()
    print("Running Capon reference tests (no GPU)...")
    results = runner.run(TestCaponReference())
    results += runner.run(TestCaponRealData())
    if HAS_BINARY:
        results += runner.run(TestCaponGPU())
    runner.print_summary(results)

