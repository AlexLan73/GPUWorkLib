# 🐧 Linux ROCm — что нужно проверить и исправить

> **Создано**: 2026-03-08
> **Контекст**: После регрессии Windows (commit `47edd7b`) и полной проверки C++ & Python
> **Инструкция по ROCm регрессиям**: [`Doc_Addition/ROCm_Regression_Check_Algorithm.md`](../../Doc_Addition/ROCm_Regression_Check_Algorithm.md)

---

## ✅ Исправления уже внесены в код (работают на Windows, нужно проверить на Linux)

### 1. fft_maxima — два бага из оптимизации kernels

| Файл | Что исправлено |
|------|----------------|
| `modules/fft_maxima/src/spectrum_maxima_finder_all_maxima.cpp` | dispatch `detect_all_maxima`: **1D → 2D** NDRange `{nFFT, beam_count}` |
| `modules/fft_maxima/include/kernels/fft_kernel_sources.hpp` | clFFT callback: `>> nFFT_log2` / `& (nFFT-1)` → **безопасные** `/ nFFT` и `% nFFT` |

### 2. Python тесты — порядок путей поиска gpuworklib

**Проблема**: папка `build/python/` существует (там vcxproj/Makefile), поэтому `os.path.isdir()` даёт `True` и break происходит до `build/python/Debug` где реально лежит `.so`/`.pyd`.

**Исправлено во всех тестах** (порядок теперь: Debug → Release → parent):

```python
BUILD_PATHS = [
    os.path.join(..., 'build', 'python', 'Debug'),    # ← первым
    os.path.join(..., 'build', 'python', 'Release'),
    os.path.join(..., 'build', 'python'),              # ← последним
]
```

**Исправленные файлы** (на Linux путь может быть другим — см. ниже):
- `Python_test/fft_maxima/test_spectrum_find_all_maxima.py`
- `Python_test/fft_maxima/test_find_all_maxima_maxvalue.py`
- `Python_test/filters/test_ai_fir_demo.py`
- `Python_test/signal_generators/test_lfm_analytical_delay.py`
- `Python_test/signal_generators/test_form_signal.py`
- `Python_test/signal_generators/test_delayed_form_signal.py`
- `Python_test/lch_farrow/test_lch_farrow.py`
- `Python_test/heterodyne/test_heterodyne.py`
- `Python_test/heterodyne/test_heterodyne_comparison.py`
- `Python_test/heterodyne/test_heterodyne_step_by_step.py`

---

## 🐧 На Linux — путь к .so другой!

На Linux сборка кладёт `.so` в `build/debian-radeon9070/python/` (или аналогичный путь).
Нужно убедиться, что в каждом тесте есть этот путь и он стоит **до** `build/python`:

```python
BUILD_PATHS = [
    os.path.join(..., 'build', 'debian-radeon9070', 'python'),  # ← Linux ROCm
    os.path.join(..., 'build', 'python', 'Debug'),
    os.path.join(..., 'build', 'python', 'Release'),
    os.path.join(..., 'build', 'python'),
]
```

> 💡 Проверить реальный путь командой: `find build/ -name "gpuworklib*.so" 2>/dev/null`

---

## 📋 Что запустить на Linux ROCm (чеклист)

```bash
# Сборка
cmake -B build -DBUILD_PYTHON=ON -DENABLE_ROCM=ON
cmake --build build --target gpuworklib -j$(nproc)

# Найти .so
find build/ -name "gpuworklib*.so"

# C++ тесты (все модули)
./build/GPUWorkLib fft_maxima
./build/GPUWorkLib filters
./build/GPUWorkLib signal_generators
./build/GPUWorkLib lch_farrow
./build/GPUWorkLib heterodyne
./build/GPUWorkLib statistics       # ROCm-only ← важно проверить!
./build/GPUWorkLib fm_correlator    # ROCm-only ← важно проверить!

# Python тесты
python3 Python_test/fft_maxima/test_spectrum_find_all_maxima.py
python3 Python_test/fft_maxima/test_find_all_maxima_maxvalue.py
python3 Python_test/signal_generators/test_form_signal.py --no-plot
python3 Python_test/signal_generators/test_lfm_analytical_delay.py
python3 Python_test/lch_farrow/test_lch_farrow.py
python3 Python_test/heterodyne/test_heterodyne.py
python3 Python_test/filters/test_filters_stage1.py
```

---

## ⚠️ Особое внимание (ROCm-only модули, на Windows не тестировались)

| Модуль | Что проверить |
|--------|---------------|
| `statistics` | welford_fused, extract_medians, radix sort — C++ + Python |
| `fm_correlator` | все тесты закомментированы — раскомментировать и запустить |
| `fft_processor` (ROCm) | hipFFT тесты + matrix benchmark |

---

## 📖 Связанные документы

- **Полный алгоритм поиска регрессий ROCm**: [`Doc_Addition/ROCm_Regression_Check_Algorithm.md`](../../Doc_Addition/ROCm_Regression_Check_Algorithm.md)
- **Оптимизация HIP/ROCm ядер**: [`Doc_Addition/Info_ROCm_HIP_Optimization_Guide.md`](../../Doc_Addition/Info_ROCm_HIP_Optimization_Guide.md)
