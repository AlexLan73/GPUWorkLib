# vector_algebra — API Reference

> Инверсия эрмитовых положительно определённых матриц на GPU (ROCm, Cholesky)

**Namespace**: `vector_algebra`
**Backend**: ROCm only (AMD GPU, `ENABLE_ROCM`)
**Заголовки**: `cholesky_inverter_rocm.hpp`, `vector_algebra_types.hpp`

---

## Содержание

1. [Enum SymmetrizeMode](#1-enum-symmetrizemode)
2. [Struct CholeskyResult](#2-struct-choleskyresult)
3. [Class CholeskyInverterROCm](#3-class-choleskyinverterrocm)
4. [Python API](#4-python-api)
5. [Цепочки вызовов](#5-цепочки-вызовов)
6. [Ошибки и исключения](#6-ошибки-и-исключения)

---

## 1. Enum SymmetrizeMode

```cpp
// modules/vector_algebra/include/vector_algebra_types.hpp
namespace vector_algebra {

enum class SymmetrizeMode {
  Roundtrip,  // Download GPU → CPU sym → Upload (fallback)
  GpuKernel   // HIP kernel in-place (hiprtc) — рекомендуется
};

}
```

| Значение | Описание |
|----------|----------|
| `Roundtrip` | Скачивает матрицу на CPU, симметризует там, загружает обратно. Медленнее. |
| `GpuKernel` | Симметризация прямо на GPU через hiprtc kernel. Рекомендуется по умолчанию. |

---

## 2. Struct CholeskyResult

```cpp
// modules/vector_algebra/include/vector_algebra_types.hpp
namespace vector_algebra {

struct CholeskyResult {
  void*  d_data     = nullptr;  // HIP device ptr (владеет памятью!)
  IBackend* backend = nullptr;  // Для DtoH/Free
  int matrix_size   = 0;        // n (сторона одной матрицы)
  int batch_count   = 0;        // количество матриц в batch

  // --- Методы доступа (скачивают GPU → CPU) ---
  std::vector<std::complex<float>> AsVector() const;
  void* AsHipPtr() const;
  std::vector<std::vector<std::complex<float>>> matrix() const;
  std::vector<std::vector<std::vector<std::complex<float>>>> matrices() const;

  // --- Управление памятью ---
  ~CholeskyResult();                              // hipFree(d_data)
  CholeskyResult() = default;
  CholeskyResult(CholeskyResult&&) noexcept;
  CholeskyResult& operator=(CholeskyResult&&) noexcept;
  CholeskyResult(const CholeskyResult&) = delete;
  CholeskyResult& operator=(const CholeskyResult&) = delete;
};

}
```

### Методы CholeskyResult

| Метод | Возвращает | Описание |
|-------|-----------|----------|
| `AsVector()` | `vector<complex<float>>` | Плоский массив, size = `matrix_size * matrix_size * batch_count` |
| `AsHipPtr()` | `void*` | Raw HIP device pointer. **Не освобождать!** |
| `matrix()` | `vector<vector<...>>` shape `[n][n]` | Для одной матрицы (batch_count=1) |
| `matrices()` | `vector<...>` shape `[batch][n][n]` | Для batched результата |

> ⚠️ **RAII**: `CholeskyResult` владеет GPU памятью. Уничтожение объекта → `hipFree`. Не копируемый — только `std::move`.

---

## 3. Class CholeskyInverterROCm

```cpp
// modules/vector_algebra/include/cholesky_inverter_rocm.hpp
namespace vector_algebra {

class CholeskyInverterROCm {
public:
  // ─── Конструктор ────────────────────────────────────────────────────────
  explicit CholeskyInverterROCm(
      drv_gpu_lib::IBackend* backend,
      SymmetrizeMode mode = SymmetrizeMode::GpuKernel);

  ~CholeskyInverterROCm();

  // Не копируемый, не перемещаемый (владеет rocBLAS handle + hipModule)
  CholeskyInverterROCm(const CholeskyInverterROCm&) = delete;
  CholeskyInverterROCm& operator=(const CholeskyInverterROCm&) = delete;

  // ─── Настройки ──────────────────────────────────────────────────────────
  void SetSymmetrizeMode(SymmetrizeMode mode);
  SymmetrizeMode GetSymmetrizeMode() const;
  void SetCheckInfo(bool enabled);
  void CompileKernels();

  // ─── Одна матрица ───────────────────────────────────────────────────────
  CholeskyResult Invert(
      const drv_gpu_lib::InputData<std::vector<std::complex<float>>>& input,
      int n = 0);

  CholeskyResult Invert(
      const drv_gpu_lib::InputData<void*>& input,
      int n = 0);

  CholeskyResult Invert(           // только если CL_VERSION_1_0
      const drv_gpu_lib::InputData<cl_mem>& input,
      int n = 0);

  // ─── Batched ────────────────────────────────────────────────────────────
  CholeskyResult InvertBatch(
      const drv_gpu_lib::InputData<std::vector<std::complex<float>>>& input,
      int n);

  CholeskyResult InvertBatch(
      const drv_gpu_lib::InputData<void*>& input,
      int n);

  CholeskyResult InvertBatch(      // только если CL_VERSION_1_0
      const drv_gpu_lib::InputData<cl_mem>& input,
      int n);
};

}
```

### Методы CholeskyInverterROCm

| Метод | Сигнатура | Описание |
|-------|-----------|----------|
| **Конструктор** | `(IBackend*, SymmetrizeMode=GpuKernel)` | Создаёт rocBLAS handle, аллоцирует `d_info_` |
| `SetSymmetrizeMode` | `(SymmetrizeMode) → void` | Переключает режим симметризации |
| `GetSymmetrizeMode` | `() → SymmetrizeMode` | Текущий режим |
| `SetCheckInfo` | `(bool) → void` | Вкл/выкл проверку POTRF/POTRI info. Default: `true` |
| `CompileKernels` | `() → void` | Принудительная компиляция hiprtc (warmup). Автоматически в конструкторе при GpuKernel mode |
| `Invert` | `(InputData<vector>, n=0) → CholeskyResult` | CPU вектор → GPU → результат. `n=0`: вычисляется из `sqrt(n_point)` |
| `Invert` | `(InputData<void*>, n=0) → CholeskyResult` | ROCm device pointer (уже на GPU) |
| `Invert` | `(InputData<cl_mem>, n=0) → CholeskyResult` | OpenCL буфер (ZeroCopy, требует `CL_VERSION_1_0`) |
| `InvertBatch` | `(InputData<vector>, n) → CholeskyResult` | Batched CPU, `n` обязателен |
| `InvertBatch` | `(InputData<void*>, n) → CholeskyResult` | Batched GPU |
| `InvertBatch` | `(InputData<cl_mem>, n) → CholeskyResult` | Batched cl_mem |

### InputData для batched

```cpp
// Одна матрица (CPU):
drv_gpu_lib::InputData<std::vector<std::complex<float>>> input;
input.antenna_count = 1;
input.n_point = n * n;
input.data = flat_matrix;  // size = n*n

// Batch (CPU):
input.antenna_count = batch_count;
input.n_point = batch_count * n * n;
input.data = all_matrices_flat;  // size = batch_count * n * n
```

---

## 4. Python API

```python
import gpuworklib
import numpy as np
```

### Enum SymmetrizeMode

```python
gpuworklib.SymmetrizeMode.GpuKernel   # HIP kernel in-place (рекомендуется)
gpuworklib.SymmetrizeMode.Roundtrip   # CPU symmetrize (fallback)
```

### Class gpuworklib.CholeskyInverterROCm

```python
class CholeskyInverterROCm:
    def __init__(ctx: ROCmGPUContext,
                 mode: SymmetrizeMode = SymmetrizeMode.GpuKernel): ...

    def invert_cpu(
            matrix_flat: np.ndarray,  # complex64, shape (n*n,)
            n: int
    ) -> np.ndarray: ...              # complex64, shape (n, n)

    def invert_batch_cpu(
            matrices_flat: np.ndarray,  # complex64, shape (batch*n*n,)
            n: int,
            batch_count: int
    ) -> np.ndarray: ...               # complex64, shape (batch, n, n)

    def set_symmetrize_mode(mode: SymmetrizeMode) -> None: ...
    def get_symmetrize_mode() -> SymmetrizeMode: ...
    def __repr__() -> str: ...
```

| Метод | Аргументы | Возвращает |
|-------|-----------|-----------|
| `__init__` | `ctx`, `mode=GpuKernel` | — |
| `invert_cpu` | `matrix_flat` (n*n,), `n` | ndarray `(n, n)` complex64 |
| `invert_batch_cpu` | `matrices_flat` (batch*n*n,), `n`, `batch_count` | ndarray `(batch, n, n)` complex64 |
| `set_symmetrize_mode` | `mode` | `None` |
| `get_symmetrize_mode` | — | `SymmetrizeMode` |

---

## 5. Цепочки вызовов

### C++ — одна матрица (CPU данные)

```cpp
#include "cholesky_inverter_rocm.hpp"
#include "vector_algebra_types.hpp"
using namespace vector_algebra;

// 1. Context
auto ctx = drv_gpu_lib::DrvGPU::Create(/*...*/);
auto* backend = ctx->GetBackend();

// 2. Инвертер
CholeskyInverterROCm inverter(backend);          // GpuKernel (default)
inverter.CompileKernels();                        // warmup (опционально)

// 3. Данные
int n = 341;
std::vector<std::complex<float>> matrix_flat(n * n);
// ... заполнить HPD матрицу ...

drv_gpu_lib::InputData<std::vector<std::complex<float>>> input;
input.antenna_count = 1;
input.n_point = n * n;
input.data = matrix_flat;

// 4. Инверсия
CholeskyResult result = inverter.Invert(input, n);

// 5. Результат
auto flat  = result.AsVector();    // flat n*n
auto mat2d = result.matrix();      // [n][n]
void* dptr = result.AsHipPtr();    // raw GPU ptr (не освобождать!)
```

### C++ — batched инверсия

```cpp
int n = 64, batch_count = 4;
std::vector<std::complex<float>> all_matrices(batch_count * n * n);
// ... заполнить batch_count матриц подряд ...

drv_gpu_lib::InputData<std::vector<std::complex<float>>> batch_input;
batch_input.antenna_count = batch_count;
batch_input.n_point = batch_count * n * n;
batch_input.data = all_matrices;

CholeskyResult batch_result = inverter.InvertBatch(batch_input, n);
auto mat3d = batch_result.matrices();  // [batch_count][n][n]
```

### C++ — GPU input (void*)

```cpp
void* d_matrix = /* hipMalloc или другой источник */;

drv_gpu_lib::InputData<void*> gpu_input;
gpu_input.data = d_matrix;
gpu_input.n_point = n * n;
gpu_input.antenna_count = 1;

CholeskyResult result = inverter.Invert(gpu_input, n);
// d_matrix изменяется in-place — после вызова содержит A^{-1}
```

### Python — полный пример

```python
import numpy as np
import gpuworklib

# 1. Context
ctx = gpuworklib.ROCmGPUContext(device_index=0)

# 2. Инвертер
inv = gpuworklib.CholeskyInverterROCm(ctx, gpuworklib.SymmetrizeMode.GpuKernel)

# 3. Генерация HPD матрицы
n = 341
B = (np.random.randn(n, n) + 1j * np.random.randn(n, n)).astype(np.complex64)
A = (B @ B.conj().T + n * np.eye(n, dtype=np.complex64)).astype(np.complex64)

# 4. Инверсия
A_inv = inv.invert_cpu(A.flatten(), n)      # shape (n, n), complex64

# 5. Проверка
A_d = A.astype(np.complex128)
A_inv_d = A_inv.astype(np.complex128)
err = np.linalg.norm(A_d @ A_inv_d - np.eye(n, dtype=np.complex128), 'fro')
print(f"||A*A⁻¹ - I||_F = {err:.2e}")       # < 1e-2 для n=341, float32
```

### Python — batched

```python
n, batch_count = 64, 4

# Генерация batch HPD матриц
matrices = []
for _ in range(batch_count):
    B = (np.random.randn(n, n) + 1j * np.random.randn(n, n)).astype(np.complex64)
    A = B @ B.conj().T + n * np.eye(n, dtype=np.complex64)
    matrices.append(A.astype(np.complex64))

flat = np.concatenate([m.flatten() for m in matrices])

results = inv.invert_batch_cpu(flat, n, batch_count)
# results.shape == (4, 64, 64), dtype == complex64

# Проверка каждой матрицы
for i in range(batch_count):
    err = np.linalg.norm(
        matrices[i].astype(np.complex128) @ results[i].astype(np.complex128)
        - np.eye(n, dtype=np.complex128), 'fro')
    print(f"  matrix[{i}] error = {err:.2e}")  # < 1e-3
```

### Python — смена режима симметризации

```python
# В конструкторе
inv_rt = gpuworklib.CholeskyInverterROCm(ctx, gpuworklib.SymmetrizeMode.Roundtrip)

# Динамически
inv.set_symmetrize_mode(gpuworklib.SymmetrizeMode.Roundtrip)
mode = inv.get_symmetrize_mode()  # SymmetrizeMode.Roundtrip

# Repr
print(repr(inv))  # <CholeskyInverterROCm (ROCm, POTRF+POTRI, Roundtrip)>
```

---

## 6. Ошибки и исключения

| Ситуация | Исключение | Условие |
|----------|-----------|---------|
| Матрица не HPD (вырождена) | `std::runtime_error` | `POTRF` вернул ненулевой info, `SetCheckInfo(true)` |
| `POTRI` провалился | `std::runtime_error` | info != 0 после POTRI |
| Неверный размер `matrix_flat` | `std::invalid_argument` | Python: `size != n*n` |
| Неверный размер `matrices_flat` | `std::invalid_argument` | Python: `size != batch*n*n` |
| Нет ROCm (compile time) | — | Весь класс обёрнут в `#if ENABLE_ROCM` |

```cpp
// Отключить проверку (для benchmark/production с гарантированно HPD матрицами)
inverter.SetCheckInfo(false);
```

---

## Точность (float32)

| n | Ожидаемая ошибка `||A·A⁻¹ - I||_F` |
|---|--------------------------------------|
| 5 | < 1e-5 |
| 64 | < 1e-3 |
| 128 | < 1e-2 |
| 256 | < 1e-2 |
| 341 | < 1e-2 |

> ⚠️ float32 — единственный поддерживаемый тип (complex64). Double не поддерживается.

---

## См. также

- [Full.md](Full.md) — математика, pipeline, C4 диаграммы, тесты
- [Quick.md](Quick.md) — шпаргалка, быстрый старт
- [Doc/Python/vector_algebra_api.md](../../Python/vector_algebra_api.md) — расширенный Python API

---

*Обновлено: 2026-03-09*
