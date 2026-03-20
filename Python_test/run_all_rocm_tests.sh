#!/bin/bash
# =============================================================================
# run_all_rocm_tests.sh — Запуск всех ROCm Python тестов
# =============================================================================
#
# Использование:
#   sg render -c "bash Python_test/run_all_rocm_tests.sh"   # с GPU
#   bash Python_test/run_all_rocm_tests.sh                  # без GPU (только CPU-тесты)
#
# Требования:
#   - Python 3 + numpy + scipy
#   - Собранный .so: build/debian-radeon9070/python/gpuworklib*.so
#   - Группа render для доступа к /dev/dri/render*
#
# =============================================================================

set -euo pipefail

# Путь к .so модулю
PYLIB_DIR="${PYLIB_DIR:-build/debian-radeon9070/python}"

# Цвета
GREEN='\033[0;32m'
RED='\033[0;31m'
YELLOW='\033[1;33m'
NC='\033[0m'  # No Color

PASSED=0
FAILED=0
SKIPPED=0

run_test() {
    local test_file="$1"
    local label="${2:-$test_file}"

    echo ""
    echo -e "${YELLOW}=== $label ===${NC}"

    if [ ! -f "$test_file" ]; then
        echo -e "${YELLOW}  SKIPPED — файл не найден: $test_file${NC}"
        SKIPPED=$((SKIPPED + 1))
        return
    fi

    if PYTHONPATH="$PYLIB_DIR" python3 "$test_file"; then
        echo -e "${GREEN}  PASSED${NC}"
        PASSED=$((PASSED + 1))
    else
        echo -e "${RED}  FAILED${NC}"
        FAILED=$((FAILED + 1))
    fi
}

echo "============================================================"
echo "  GPUWorkLib ROCm Python Tests"
echo "  PYLIB_DIR = $PYLIB_DIR"
echo "============================================================"

# ── Статистика ──────────────────────────────────────────────────
run_test "Python_test/statistics/test_statistics_rocm.py" \
         "StatisticsProcessor (mean/median/statistics, GPU sort)"

run_test "Python_test/statistics/test_statistics_float_rocm.py" \
         "StatisticsProcessor float (mean/median float32)"

# ── Vector Algebra ───────────────────────────────────────────────
run_test "Python_test/vector_algebra/test_cholesky_inverter_rocm.py" \
         "CholeskyInverterROCm (POTRF+POTRI, Roundtrip/GpuKernel)"

# ── Фильтры ─────────────────────────────────────────────────────
run_test "Python_test/filters/test_fir_filter_rocm.py" \
         "FirFilterROCm (LP/BP/HP/multi-channel)"

run_test "Python_test/filters/test_iir_filter_rocm.py" \
         "IirFilterROCm (biquad cascade, multi-channel)"

# ── Гетеродин ───────────────────────────────────────────────────
run_test "Python_test/heterodyne/test_heterodyne_rocm.py" \
         "HeterodyneROCm (dechirp + correct)"

# ── LchFarrow ───────────────────────────────────────────────────
run_test "Python_test/lch_farrow/test_lch_farrow_rocm.py" \
         "LchFarrowROCm (fractional delay, Lagrange 48x5)"

# ── ZeroCopy ─────────────────────────────────────────────────────
run_test "Python_test/zero_copy/test_zero_copy.py" \
         "ZeroCopy (HybridGPUContext, OpenCL→ROCm method detection)"

# ── HybridBackend ────────────────────────────────────────────────
run_test "Python_test/hybrid/test_hybrid_backend.py" \
         "HybridBackend (OpenCL + ROCm on one GPU)"

# ── Итог ────────────────────────────────────────────────────────
TOTAL=$((PASSED + FAILED + SKIPPED))
echo ""
echo "============================================================"
echo -e "  Total:   $TOTAL"
echo -e "  ${GREEN}Passed:  $PASSED${NC}"
if [ "$FAILED" -gt 0 ]; then
    echo -e "  ${RED}Failed:  $FAILED${NC}"
else
    echo -e "  Failed:  $FAILED"
fi
if [ "$SKIPPED" -gt 0 ]; then
    echo -e "  ${YELLOW}Skipped: $SKIPPED${NC}"
fi
echo "============================================================"

if [ "$FAILED" -gt 0 ]; then
    exit 1
fi
exit 0
