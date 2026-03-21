# vector_algebra — Полная документация

> Инверсия эрмитовых положительно определённых матриц на GPU (ROCm, Cholesky)

**Namespace**: `vector_algebra`
**Каталог**: `modules/vector_algebra/`
**Зависимости**: DrvGPU (`IBackend*`), rocBLAS, rocSOLVER, hiprtc, KernelCacheService
**Backend**: ROCm only (AMD GPU)

---

## Содержание

1. [Обзор и назначение](#1-обзор-и-назначение)
2. [Математика алгоритма](#2-математика)
3. [Pipeline](#3-pipeline)
4. [Kernel — symmetrize_upper_to_full](#4-kernel)
5. [API — C++ и Python](#5-api)
6. [Тесты](#6-тесты)
7. [Диаграммы C1–C4](#7-диаграммы)
8. [Файлы модуля](#8-файлы)
9. [Важные нюансы](#важные-нюансы)

---

## 1. Обзор и назначение

`CholeskyInverterROCm` — инверсия эрмитовой положительно определённой (HPD) матрицы методом Холецкого на GPU.

**Вход**: HPD матрица n×n в формате flat `complex<float>[n*n]` (или batched).
**Выход**: `CholeskyResult` — RAII-объект, владеет GPU-памятью. Скачивается через `.AsVector()` / `.matrix()`.

**Два режима симметризации** (`SymmetrizeMode`):
| Режим | Описание | Когда использовать |
|-------|----------|-------------------|
| `Roundtrip` | DtoH → CPU зеркало → HtoD | Fallback, совместимость |
| `GpuKernel` | HIP kernel in-place (hiprtc) | **По умолчанию** — всё на GPU |

**Три формата входных данных** (`InputData<T>`):
| T | Описание |
|---|----------|
| `vector<complex<float>>` | CPU вектор → HtoD внутри |
| `void*` | ROCm device pointer (уже на GPU) |
| `cl_mem` | OpenCL буфер (ZeroCopy через HIP) |

---

## 2. Математика

### Cholesky-инверсия

Для HPD матрицы A инверсия через Cholesky в два шага:

**POTRF** — разложение Холецкого:
$$A = U^H \cdot U$$
где $U$ — верхнетреугольная матрица.

**POTRI** — инверсия через $U$:
$$A^{-1} \text{ из } U \quad \Rightarrow \quad A^{-1} \approx \tilde{A}^{-1}$$

После POTRI результат хранится **только в верхнем треугольнике**. Нижний треугольник нужно заполнить сопряжёнными значениями:

$$A^{-1}_{ij} = \overline{A^{-1}_{ji}}, \quad j > i$$

Это и есть **симметризация** — ключевый постпроцессинг.

### Критерий корректности

Ошибка инверсии по норме Фробениуса:

$$\varepsilon = \|A \cdot A^{-1} - I\|_F$$

Допустимые значения (float32):
| n | Порог |
|---|-------|
| 5 | < 1e-5 |
| 64 | < 1e-3 |
| 341 | < 1e-2 |

---

## 3. Pipeline

```
InputData<vector/void*/cl_mem>
    │
    ▼  (HtoD если CPU вектор)
┌─────────────────────────────────────────┐
│ 1. POTRF (rocSOLVER)                   │  A = U^H * U
│    Cholesky decomposition              │  результат: U в верхнем треугольнике
└─────────────────────────────────────────┘
    │
    ▼
┌─────────────────────────────────────────┐
│ 2. POTRI (rocSOLVER)                   │  A^{-1} из U
│    Cholesky inversion                  │  верхний треугольник: A^{-1}[i][j], i≤j
└─────────────────────────────────────────┘
    │
    ▼
┌────────────────────────────────────────────────────────┐
│ 3. Symmetrize                                         │
│                                                        │
│  Roundtrip:  DtoH → CPU: A[j][i] = conj(A[i][j])     │
│              → HtoD                                    │
│                                                        │
│  GpuKernel:  symmetrize_upper_to_full kernel (hiprtc) │
│              in-place, grid (ceil(n/16), ceil(n/16))  │
└────────────────────────────────────────────────────────┘
    │
    ▼
CholeskyResult { void* d_data, matrix_size=n, batch_count }
    │
    ├─ .AsVector()    → CPU flat vector<complex<float>>
    ├─ .matrix()      → 2D vector [n][n]
    └─ .matrices()    → 3D vector [batch][n][n]
```

### Batched pipeline

Для `InvertBatch` — матрицы хранятся contiguously: `d_data[k * n*n + i*n + j]`.
POTRF/POTRI вызываются последовательно для каждой матрицы (один rocBLAS handle).

---

## 4. Kernel — symmetrize_upper_to_full

**Файл**: `modules/vector_algebra/kernels/symmetrize_upper_to_full.cl`
**Компиляция**: hiprtc в runtime, кешируется через `KernelCacheService`

| Параметр | Тип | Описание |
|----------|-----|----------|
| `data` | `float2* __restrict__` | Матрица in-place |
| `n` | `unsigned int` | Размер матрицы |

**Grid/Block**: `(ceil(n/16), ceil(n/16))` × `(16, 16)` — каждый поток = один элемент.

```c
// Только верхний треугольник: col > row
if (col > row) {
    float2 v = data[row * n + col];
    data[col * n + row] = {v.x, -v.y};  // conjugate: Re сохраняем, Im инвертируем
}
```

**Почему 2D grid**: каждый поток обрабатывает один элемент (row, col) — нет div/mod.

---

## 5. API

### C++

```cpp
#include "cholesky_inverter_rocm.hpp"
#include "vector_algebra_types.hpp"  // CholeskyResult, SymmetrizeMode

using namespace vector_algebra;

// ── Конструктор ──────────────────────────────────────────────────────────────
CholeskyInverterROCm inverter(backend);                            // GpuKernel (default)
CholeskyInverterROCm inverter(backend, SymmetrizeMode::Roundtrip); // Roundtrip

// ── Одна матрица — CPU вектор ─────────────────────────────────────────────────
drv_gpu_lib::InputData<std::vector<std::complex<float>>> input;
input.antenna_count = 1;
input.n_point = n * n;
input.data = matrix_flat;  // vector<complex<float>>, row-major

CholeskyResult result = inverter.Invert(input);
// n автовычисляется из sqrt(n_point), или явно: inverter.Invert(input, n)

// ── Одна матрица — ROCm device pointer ───────────────────────────────────────
drv_gpu_lib::InputData<void*> gpu_input;
gpu_input.data = d_matrix;   // hipDeviceptr
gpu_input.n_point = n * n;
CholeskyResult result = inverter.Invert(gpu_input, n);

// ── Batched — CPU вектор ──────────────────────────────────────────────────────
drv_gpu_lib::InputData<std::vector<std::complex<float>>> batch_input;
batch_input.antenna_count = batch_count;
batch_input.n_point = batch_count * n * n;
batch_input.data = all_matrices_flat;  // batch_count матриц подряд

CholeskyResult batch_result = inverter.InvertBatch(batch_input, n);

// ── Чтение результата ─────────────────────────────────────────────────────────
auto flat = result.AsVector();              // flat vector, size = n*n
auto mat2d = result.matrix();              // vector<vector<...>>, shape [n][n]
auto mat3d = batch_result.matrices();      // [batch][n][n]
void* ptr  = result.AsHipPtr();            // raw device ptr (caller НЕ владеет!)

// ── Настройки ─────────────────────────────────────────────────────────────────
inverter.SetSymmetrizeMode(SymmetrizeMode::GpuKernel);
inverter.SetCheckInfo(false);    // отключить проверку POTRF/POTRI info (для benchmark)
inverter.CompileKernels();       // warmup: принудительная компиляция hiprtc
```

### Python

```python
import numpy as np
import gpuworklib

# ── Context ────────────────────────────────────────────────────────────────────
ctx = gpuworklib.ROCmGPUContext(device_index=0)

# ── Constructor ────────────────────────────────────────────────────────────────
inv = gpuworklib.CholeskyInverterROCm(ctx)                              # GpuKernel
inv = gpuworklib.CholeskyInverterROCm(ctx, gpuworklib.SymmetrizeMode.GpuKernel)
inv_rt = gpuworklib.CholeskyInverterROCm(ctx, gpuworklib.SymmetrizeMode.Roundtrip)

# ── Подготовить HPD матрицу ────────────────────────────────────────────────────
n = 341
B = (np.random.randn(n, n) + 1j * np.random.randn(n, n)).astype(np.complex64)
A = (B @ B.conj().T + n * np.eye(n, dtype=np.complex64)).astype(np.complex64)

# ── Инверсия одной матрицы ────────────────────────────────────────────────────
A_inv = inv.invert_cpu(A.flatten(), n)
# A_inv.shape == (n, n), dtype == complex64

# Проверка
residual = np.linalg.norm(A.astype(np.complex128) @ A_inv.astype(np.complex128)
                          - np.eye(n, dtype=np.complex128), 'fro')
print(f"||A*A⁻¹ - I||_F = {residual:.2e}")  # < 1e-2 для n=341, float32

# ── Batched инверсия ───────────────────────────────────────────────────────────
n, batch_count = 64, 4
matrices = [(lambda B: B @ B.conj().T + n * np.eye(n, dtype=np.complex64))(
    (np.random.randn(n, n) + 1j * np.random.randn(n, n)).astype(np.complex64))
    for _ in range(batch_count)]
flat = np.concatenate([m.flatten() for m in matrices])

results = inv.invert_batch_cpu(flat, n, batch_count)
# results.shape == (4, 64, 64), dtype == complex64

# ── Смена режима ───────────────────────────────────────────────────────────────
inv.set_symmetrize_mode(gpuworklib.SymmetrizeMode.Roundtrip)
print(inv.get_symmetrize_mode())  # SymmetrizeMode.Roundtrip
```

**Полная документация Python API**: [`Doc/Python/vector_algebra_api.md`](../../Python/vector_algebra_api.md)
**API Reference (C++ + Python)**: [`API.md`](API.md)

---

## 6. Тесты

### C++ тесты

**Файлы**: `test_cholesky_inverter_rocm.hpp`, `test_cross_backend_conversion.hpp`, `test_benchmark_symmetrize.hpp`, `test_stage_profiling.hpp`
**Вызов**: через `all_test.hpp` из `main.cpp`
**Оба режима**: каждый тест запускается для `Roundtrip` и `GpuKernel`

| # | Функция | Что проверяет | Параметры | Порог |
|---|---------|---------------|-----------|-------|
| 0 | `TestResolveMatrixSize` | sqrt(n_point) → n без GPU | n=5,64,256,341 | exact |
| 1 | `TestCpuIdentity` | I(5×5): A*A⁻¹ = I | CPU, n=5 | err < 1e-5 |
| 2 | `TestCpu341` | HPD(341): GPU Cholesky | CPU→GPU→CPU | err < 1e-2 |
| 3 | `TestGpuVoidPtr341` | GPU input (void*) | d_ptr, n=341 | err < 1e-2 |
| 4 | `TestZeroCopyClMem` | cl_mem ZeroCopy input | cl_mem, n=85 | err < 1e-2 |
| 5 | `TestBatchCpu_4x64` | Batched 4 × 64×64 | CPU flat | err < 1e-3 each |
| 6 | `TestBatchGpu_4x64` | Batched GPU (void*) | d_ptr batch | err < 1e-3 each |
| 7 | `TestBatchSizes` | Batch sizes 1,2,4,8 | n=64 | passes |
| 8 | `TestMatrixSizes` | Разные n: 5,64,128,256 | CPU | err < порог(n) |
| 9 | `TestResultAccess` | matrix()/matrices() shape | n=64, batch=4 | shape correct |
| 10 | `TestConvert_*` (cross-backend) | Входы CPU/HIP/cl_mem | n=85 | err < 1e-2 |
| 11 | `TestStageProfiling` | GPUProfiler stage timing | Task_12 | passes |
| 12 | `TestProfilerIntegration` | GPUProfiler отчёт | n=341 | passes |
| 13 | `RunComprehensiveBenchmark` | Benchmark MD отчёт | n=341,85; batch=1..128 | warmup=3, runs=20 |

**Итого**: 23 теста (9 тестов × 2 режима + утилиты + cross-backend + benchmark)

### Python тесты

**Файл**: `Python_test/vector_algebra/test_cholesky_inverter_rocm.py`
**Запуск**: `python Python_test/vector_algebra/test_cholesky_inverter_rocm.py`

| # | Тест | Что проверяет | Порог |
|---|------|---------------|-------|
| 1 | `test_invert_5x5` | GPU vs NumPy, n=5 | err < 1e-4 |
| 2 | `test_invert_341x341` | Реалистичный размер | err < 1e-2 |
| 3 | `test_batch_4x64` | Batched 4 × 64×64 | err < 1e-3 each |
| 4 | `test_batch_sizes` | Batch 1, 4, 8 | shape correct |
| 5 | `test_modes_roundtrip_vs_kernel` | Roundtrip == GpuKernel | diff < 1e-5 |
| 6 | `test_set_symmetrize_mode` | Динамическая смена режима | err < 1e-4 после смены |

---

## 7. Диаграммы C1–C4

### C1 — System Context

```
┌─────────────────────────────────────┐
│         Пользовательский код        │
│  (Python / C++ приложение)          │
│                                     │
│  • Радарная обработка               │
│  • MIMO beamforming                 │
│  • Линейная алгебра                 │
└─────────────────┬───────────────────┘
                  │ HPD матрица (CPU / GPU)
                  ▼
┌─────────────────────────────────────┐
│          GPUWorkLib                 │
│     vector_algebra модуль          │
│                                     │
│  CholeskyInverterROCm               │
│  A^{-1} = (U^H U)^{-1}             │
└─────────────────┬───────────────────┘
                  │ POTRF/POTRI, hiprtc
                  ▼
┌─────────────────────────────────────┐
│         AMD GPU (ROCm)              │
│  rocBLAS · rocSOLVER · hiprtc      │
└─────────────────────────────────────┘
```

### C2 — Container

```
┌──────────────────────────────────────────────────────┐
│  vector_algebra                                      │
│                                                      │
│  ┌────────────────────┐    ┌──────────────────────┐  │
│  │ CholeskyInverterROCm│───►│ DrvGPU IBackend*     │  │
│  │                    │    │ (ROCmBackend)         │  │
│  └────────────────────┘    └──────────┬───────────┘  │
│          │                            │               │
│          │                  ┌─────────▼────────────┐  │
│          │                  │  ROCm Runtime        │  │
│          │                  │  rocBLAS · rocSOLVER │  │
│          │                  └──────────────────────┘  │
│          │                                            │
│  ┌───────▼─────────────┐                             │
│  │ KernelCacheService  │──► kernels/bin/ (disk)      │
│  │ (hiprtc compile)    │    HSACO кеш                │
│  └─────────────────────┘                             │
│                                                      │
│  ┌─────────────────────┐                             │
│  │  CholeskyResult     │  (RAII, move-only)          │
│  │  void* d_data       │  владеет GPU памятью        │
│  └─────────────────────┘                             │
└──────────────────────────────────────────────────────┘
```

### C3 — Component

```
CholeskyInverterROCm (Facade)
│
├── CorePotrf()          ── rocSOLVER potrf_bufferSize + potrf
│                           POTRF: A = U^H * U
│
├── CorePotri()          ── rocSOLVER potri_bufferSize + potri
│                           POTRI: A^{-1} из U (верхний треугольник)
│
├── Symmetrize()
│   ├── SymmetrizeRoundtrip()  ── DtoH → CPU conj mirror → HtoD
│   └── SymmetrizeGpuKernel()  ── hiprtc: symmetrize_upper_to_full
│                                  Grid(ceil(n/16),ceil(n/16)), Block(16,16)
│
├── d_info_              ── rocblas_int[2] на GPU (POTRF/POTRI status)
│                           предаллоцирован в конструкторе (Task_12)
│
└── KernelCacheService   ── hiprtc → HSACO → disk cache → reuse
```

### C4 — Code (ключевые классы)

```
namespace vector_algebra {

enum class SymmetrizeMode { Roundtrip, GpuKernel };

struct CholeskyResult {              // move-only, RAII
  void* d_data;                     // HIP device ptr (владеет!)
  IBackend* backend;
  int matrix_size;
  int batch_count;

  AsVector() → vector<complex<float>>
  matrix()   → vector<vector<...>>    // shape [n][n]
  matrices() → vector<...>            // shape [batch][n][n]
  AsHipPtr() → void*                  // raw, НЕ освобождать!
  ~CholeskyResult()                   // hipFree(d_data)
};

class CholeskyInverterROCm {
  + CholeskyInverterROCm(IBackend*, SymmetrizeMode = GpuKernel)
  + Invert(InputData<vector>)   → CholeskyResult
  + Invert(InputData<void*>)    → CholeskyResult
  + Invert(InputData<cl_mem>)   → CholeskyResult   // ZeroCopy
  + InvertBatch(...)            → CholeskyResult
  + SetSymmetrizeMode(mode)
  + GetSymmetrizeMode() → SymmetrizeMode
  + SetCheckInfo(bool)
  + CompileKernels()            // warmup hiprtc

  - rocblas_handle handle_
  - void* d_info_               // предаллоцированный статус
  - KernelCacheService kernel_cache_
  - CorePotrf() / CorePotri()
  - Symmetrize()
};

} // namespace vector_algebra
```

---

## 8. Файлы модуля

```
modules/vector_algebra/
├── include/
│   ├── cholesky_inverter_rocm.hpp          # Facade: CholeskyInverterROCm
│   ├── vector_algebra_types.hpp            # CholeskyResult, SymmetrizeMode
│   └── kernels/
│       └── symmetrize_kernel_sources_rocm.hpp  # HIP kernel source (hiprtc)
├── src/
│   ├── cholesky_inverter_rocm.cpp          # POTRF/POTRI, Invert, InvertBatch
│   └── symmetrize_gpu_rocm.cpp             # SymmetrizeGpuKernel (hiprtc launch)
├── kernels/
│   ├── symmetrize_upper_to_full.cl         # OpenCL-стиль kernel (hiprtc)
│   └── manifest.json                       # KernelCacheService манифест
└── tests/
    ├── all_test.hpp                        # Точка запуска всех тестов
    ├── test_cholesky_inverter_rocm.hpp     # 9 тестов × 2 режима
    ├── test_cross_backend_conversion.hpp   # Конвертация входов (CPU/HIP/cl_mem)
    ├── test_benchmark_symmetrize.hpp       # Benchmark: Roundtrip vs GpuKernel
    └── test_stage_profiling.hpp            # GPUProfiler: стадии POTRF/POTRI/Sym

Python_test/vector_algebra/
└── test_cholesky_inverter_rocm.py          # 6 тестов (2 modes × операции)

Doc/Python/
└── vector_algebra_api.md                  # Python API документация (полная)
```

---

## Важные нюансы

1. **float32 только** — `complex<float>` (complex64). Double не поддерживается rocSOLVER в этом модуле. Точность ограничена: для n=341 ошибка ~1e-3, это нормально.

2. **Матрица обязана быть HPD** — если матрица вырождена или не положительно определена, POTRF вернёт ненулевой info → `RuntimeError`. Проверка включена по умолчанию (`SetCheckInfo(true)`).

3. **Отложенная CheckInfo (Task_12)** — `d_info_` предаллоцирован в конструкторе (одна аллокация на весь lifetime). Проверка делается одним `hipMemcpy` после pipeline — не после каждой операции.

4. **CholeskyResult НЕ копируемый** — только `std::move`. Не передавать по значению, только по ссылке или через move.

5. **`AsHipPtr()` — не освобождать** — raw pointer принадлежит `CholeskyResult`. После деструктора указатель невалиден.

6. **GpuKernel warmup** — первый вызов компилирует hiprtc kernel (~50-100ms). Используй `CompileKernels()` явно для warmup в benchmark/production.

7. **ZeroCopy cl_mem** — требует HIP + OpenCL на одном устройстве (unified memory). На разных устройствах не работает.

8. **batch_count в InputData** — для batched: `input.antenna_count = batch_count`, `input.n_point = batch_count * n * n`.

---

## См. также

- [API.md](API.md) — полный справочник сигнатур C++ и Python
- [Quick.md](Quick.md) — шпаргалка, быстрый старт
- [Doc/Python/vector_algebra_api.md](../../Python/vector_algebra_api.md) — Python API
- [rocSOLVER docs](https://rocm.docs.amd.com/projects/rocSOLVER/en/latest/) — POTRF/POTRI документация AMD

---

*Обновлено: 2026-03-09*
