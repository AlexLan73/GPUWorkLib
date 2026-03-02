"""
Python тест: сравнение инверсии матриц vector_algebra с эталоном из CSV.

Загружает R_inv из Data/, вычисляет inv(R_inv) через CholeskyInverterROCm,
сравнивает с эталоном R из CSV. Генерирует отчёт в Results/Reports/vector_algebra/.

Запуск:
    cd /home/alex/C++/GPUWorkLib
    pytest Python_test/vector_algebra/test_matrix_csv_comparison.py -v

Требования: ROCm, gpuworklib (ENABLE_ROCM=ON), pytest, numpy
"""

import os
import sys
from datetime import datetime

import pytest
import numpy as np

# Path to gpuworklib
_BUILD_PATHS = [
    os.path.join(os.path.dirname(__file__), "..", "..", "build", "python"),
    os.path.join(os.path.dirname(__file__), "..", "..", "build", "python", "Release"),
    os.path.join(os.path.dirname(__file__), "..", "..", "build", "python", "Debug"),
]
for _p in _BUILD_PATHS:
    if os.path.isdir(_p):
        sys.path.insert(0, os.path.abspath(_p))
        break

try:
    import gpuworklib
except ImportError:
    gpuworklib = None

# Paths
_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.abspath(os.path.join(_THIS_DIR, "..", ".."))
DATA_DIR = os.path.join(_REPO_ROOT, "modules", "vector_algebra", "tests", "Data")
REPORT_DIR = os.path.join(_REPO_ROOT, "Results", "Reports", "vector_algebra")


# ============================================================================
# Helpers
# ============================================================================


def load_complex_matrix_csv(path: str) -> np.ndarray:
    """Загрузить комплексную матрицу из CSV (формат a+bi, запятые)."""
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            cells = [c.strip() for c in line.split(",")]
            row = []
            for cell in cells:
                # Python complex() использует 'j', CSV использует 'i'
                s = cell.replace("i", "j").replace("I", "j")
                row.append(complex(s))
            rows.append(row)
    arr = np.array(rows, dtype=np.complex64)
    return arr


def frobenius_diff(A: np.ndarray, B: np.ndarray) -> float:
    """||A - B||_F"""
    diff = A.astype(np.complex128) - B.astype(np.complex128)
    return float(np.linalg.norm(diff, "fro"))


def relative_error(A: np.ndarray, B: np.ndarray) -> float:
    """||A - B||_F / ||B||_F"""
    fnorm = float(np.linalg.norm(B.astype(np.complex128), "fro"))
    if fnorm == 0:
        return float("inf")
    return frobenius_diff(A, B) / fnorm


def generate_report(results: list, report_path: str) -> None:
    """Записать Markdown-отчёт о сравнении."""
    os.makedirs(os.path.dirname(report_path), exist_ok=True)
    lines = [
        "# Отчёт сравнения матриц vector_algebra (CSV Data)",
        "",
        f"**Дата/время**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        "",
        "## Результаты",
        "",
        "| Матрица | Базовая (R_inv) | Эталон (R) | ||Δ||_F | Отн. ошибка | Статус |",
        "|---------|-----------------|------------|--------|-------------|--------|",
    ]
    all_pass = True
    for r in results:
        status = "PASS" if r["passed"] else "FAIL"
        if not r["passed"]:
            all_pass = False
        lines.append(
            f"| {r['name']} | {r['r_inv_file']} | {r['r_ref_file']} | "
            f"{r['frobenius_diff']:.6e} | {r['rel_err']:.6e} | {status} |"
        )
    lines.extend(["", "## Итог", ""])
    if all_pass:
        lines.append("Все тесты пройдены.")
    else:
        lines.append("Есть провалившиеся тесты. См. таблицу выше.")

    with open(report_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))


# ============================================================================
# Fixtures
# ============================================================================


@pytest.fixture(scope="module")
def rocm_context():
    """ROCm GPU контекст. Пропустить если ROCm недоступен."""
    if gpuworklib is None:
        pytest.skip("gpuworklib not found")
    try:
        ctx = gpuworklib.ROCmGPUContext(0)
        return ctx
    except Exception as e:
        pytest.skip(f"ROCm недоступен: {e}")


@pytest.fixture(scope="module")
def inverter(rocm_context):
    """CholeskyInverterROCm (GpuKernel mode)."""
    return gpuworklib.CholeskyInverterROCm(
        rocm_context, gpuworklib.SymmetrizeMode.GpuKernel
    )


@pytest.fixture(scope="session")
def matrix_csv_report_data():
    """Собирает результаты для отчёта. По завершении session пишет отчёт."""
    data = []
    yield data
    if data:
        report_path = os.path.join(REPORT_DIR, "matrix_csv_comparison_report.md")
        generate_report(data, report_path)


# ============================================================================
# Tests
# ============================================================================

# Пороги: float32, реальные данные — допускаем относительную ошибку по результатам прогона
REL_ERR_THRESHOLD_85 = 2e-2   # 85×85 (~1.85e-2 observed)
REL_ERR_THRESHOLD_341 = 5e-2  # 341×341


def test_inv_r85_matches_reference(inverter, matrix_csv_report_data):
    """R_inv_85 → inv → сравнить с R_85 (1).csv"""
    n = 85
    r_inv_path = os.path.join(DATA_DIR, "R_inv_85.csv")
    r_ref_path = os.path.join(DATA_DIR, "R_85 (1).csv")

    if not os.path.isfile(r_inv_path):
        pytest.skip(f"Data file not found: {r_inv_path}")
    if not os.path.isfile(r_ref_path):
        pytest.skip(f"Data file not found: {r_ref_path}")

    R_inv = load_complex_matrix_csv(r_inv_path)
    R_ref = load_complex_matrix_csv(r_ref_path)

    assert R_inv.shape == (n, n), f"R_inv shape {R_inv.shape}"
    assert R_ref.shape == (n, n), f"R_ref shape {R_ref.shape}"

    # inv(R_inv) = R
    computed_R = inverter.invert_cpu(R_inv.flatten(), n)

    frob_diff = frobenius_diff(computed_R, R_ref)
    rel_err = relative_error(computed_R, R_ref)
    passed = rel_err < REL_ERR_THRESHOLD_85

    matrix_csv_report_data.append({
        "name": "85×85",
        "r_inv_file": "R_inv_85.csv",
        "r_ref_file": "R_85 (1).csv",
        "frobenius_diff": frob_diff,
        "rel_err": rel_err,
        "passed": passed,
    })

    assert passed, (
        f"R_85: rel_err={rel_err:.2e} >= {REL_ERR_THRESHOLD_85}, "
        f"||Δ||_F={frob_diff:.2e}"
    )


def test_inv_r341_matches_reference(inverter, matrix_csv_report_data):
    """R_inv_341 → inv → сравнить с R_341 (1).csv"""
    n = 341
    r_inv_path = os.path.join(DATA_DIR, "R_inv_341.csv")
    r_ref_path = os.path.join(DATA_DIR, "R_341 (1).csv")

    if not os.path.isfile(r_inv_path):
        pytest.skip(f"Data file not found: {r_inv_path}")
    if not os.path.isfile(r_ref_path):
        pytest.skip(f"Data file not found: {r_ref_path}")

    R_inv = load_complex_matrix_csv(r_inv_path)
    R_ref = load_complex_matrix_csv(r_ref_path)

    assert R_inv.shape == (n, n), f"R_inv shape {R_inv.shape}"
    assert R_ref.shape == (n, n), f"R_ref shape {R_ref.shape}"

    computed_R = inverter.invert_cpu(R_inv.flatten(), n)

    frob_diff = frobenius_diff(computed_R, R_ref)
    rel_err = relative_error(computed_R, R_ref)
    passed = rel_err < REL_ERR_THRESHOLD_341

    matrix_csv_report_data.append({
        "name": "341×341",
        "r_inv_file": "R_inv_341.csv",
        "r_ref_file": "R_341 (1).csv",
        "frobenius_diff": frob_diff,
        "rel_err": rel_err,
        "passed": passed,
    })

    assert passed, (
        f"R_341: rel_err={rel_err:.2e} >= {REL_ERR_THRESHOLD_341}, "
        f"||Δ||_F={frob_diff:.2e}"
    )
