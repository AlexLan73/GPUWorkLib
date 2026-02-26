# vector_algebra Python API (v2 — Task_11)

## CholeskyInverterROCm

Инверсия эрмитовой положительно определённой матрицы методом Холецкого на ROCm.

**Backend**: AMD GPU — rocSOLVER POTRF + POTRI.
**Symmetrize**: Roundtrip (CPU) или GpuKernel (hiprtc HIP kernel).

---

### SymmetrizeMode (enum)

```python
import gpuworklib

gpuworklib.SymmetrizeMode.Roundtrip   # CPU: download → symmetrize → upload
gpuworklib.SymmetrizeMode.GpuKernel   # GPU: hiprtc in-place kernel (быстрее)
```

| Режим | Описание | Когда использовать |
|-------|----------|-------------------|
| `Roundtrip` | Download GPU → CPU sym → Upload | Fallback, гарантированно работает |
| `GpuKernel` | HIP kernel in-place (hiprtc) | **По умолчанию**, всё на GPU |

---

### Конструктор

```python
import gpuworklib

ctx = gpuworklib.ROCmGPUContext(device_index=0)

# По умолчанию — GpuKernel mode
inverter = gpuworklib.CholeskyInverterROCm(ctx)

# Явный выбор режима
inverter = gpuworklib.CholeskyInverterROCm(ctx, gpuworklib.SymmetrizeMode.GpuKernel)
inverter_rt = gpuworklib.CholeskyInverterROCm(ctx, gpuworklib.SymmetrizeMode.Roundtrip)
```

**Параметры**:
| Параметр | Тип | Описание |
|----------|-----|----------|
| `ctx` | `ROCmGPUContext` | ROCm контекст |
| `mode` | `SymmetrizeMode` | Режим симметризации (default: `GpuKernel`) |

---

### Метод `invert_cpu(matrix_flat, n)`

Инверсия одной матрицы n×n. Данные CPU → GPU → результат CPU.

```python
import numpy as np
import gpuworklib

ctx = gpuworklib.ROCmGPUContext(0)
inverter = gpuworklib.CholeskyInverterROCm(ctx)

n = 341
# Создать положительно определённую матрицу
B = np.random.randn(n, n) + 1j * np.random.randn(n, n)
A = (B @ B.conj().T + n * np.eye(n)).astype(np.complex64)

# GPU инверсия через Cholesky
A_inv = inverter.invert_cpu(A.flatten(), n)
# A_inv.shape == (341, 341), dtype=complex64

# Проверка: A * A⁻¹ ≈ I
residual = np.linalg.norm(A @ A_inv.astype(np.complex128) - np.eye(n))
print(f"Residual ||A*A⁻¹ - I||_F = {residual:.2e}")  # < 1e-2 для float32
```

**Параметры**:
| Параметр | Тип | Описание |
|----------|-----|----------|
| `matrix_flat` | `np.ndarray[complex64]` | Матрица n×n в flat row-major формате, shape `(n*n,)` |
| `n` | `int` | Размер матрицы |

**Возвращает**: `np.ndarray[complex64]`, shape `(n, n)` — обратная матрица A⁻¹

**Исключения**:
- `RuntimeError` — матрица не является положительно определённой
- `ValueError` — размер `matrix_flat` не совпадает с `n*n`

---

### Метод `invert_batch_cpu(matrices_flat, n, batch_count)`

Batched инверсия нескольких матриц. CPU → GPU → CPU результат.

```python
n = 64
batch_count = 4

# Подготовить 4 положительно определённых матрицы
matrices = []
for i in range(batch_count):
    B = np.random.randn(n, n) + 1j * np.random.randn(n, n)
    A = (B @ B.conj().T + n * np.eye(n)).astype(np.complex64)
    matrices.append(A)

# Объединить в flat вектор
flat = np.concatenate([m.flatten() for m in matrices])

# GPU batched инверсия
results = inverter.invert_batch_cpu(flat, n, batch_count)
# results.shape == (4, 64, 64), dtype=complex64

for k in range(batch_count):
    residual = np.linalg.norm(matrices[k] @ results[k] - np.eye(n))
    print(f"Matrix {k}: ||A*A⁻¹ - I||_F = {residual:.2e}")
```

**Параметры**:
| Параметр | Тип | Описание |
|----------|-----|----------|
| `matrices_flat` | `np.ndarray[complex64]` | Все матрицы в flat формате, shape `(batch_count*n*n,)` |
| `n` | `int` | Размер каждой матрицы |
| `batch_count` | `int` | Количество матриц |

**Возвращает**: `np.ndarray[complex64]`, shape `(batch_count, n, n)`

---

### Метод `set_symmetrize_mode(mode)` / `get_symmetrize_mode()`

Динамическая смена режима симметризации.

```python
import gpuworklib

inv = gpuworklib.CholeskyInverterROCm(ctx, gpuworklib.SymmetrizeMode.GpuKernel)
print(inv.get_symmetrize_mode())  # SymmetrizeMode.GpuKernel

inv.set_symmetrize_mode(gpuworklib.SymmetrizeMode.Roundtrip)
print(inv.get_symmetrize_mode())  # SymmetrizeMode.Roundtrip

# Инверсия работает после смены режима
A_inv = inv.invert_cpu(A.flatten(), n)
```

---

### Точность (float32)

| Размер матрицы | Допустимая ошибка `||A*A⁻¹ - I||_F` |
|----------------|---------------------------------------|
| 5×5            | < 1e-4                               |
| 64×64          | < 1e-3                               |
| 128×128        | < 1e-2                               |
| 256×256        | < 1e-2                               |
| 341×341        | < 1e-2                               |

---

### Требования

- **ROCm** — AMD GPU (проверено на Radeon 9070)
- **rocBLAS** + **rocSOLVER** + **hiprtc** (входит в ROCm SDK)
- `gpuworklib` собран с `-DENABLE_ROCM=ON -DBUILD_PYTHON=ON`

### Ограничения

- Только **float32** (`complex64`). Double precision не поддерживается.
- Матрица должна быть **эрмитовой положительно определённой**.
  - Для вырожденных матриц бросается `RuntimeError`.
- **ROCm только** — не работает без AMD GPU.

---

### Связанные модули

- `gpuworklib.ROCmGPUContext` — ROCm контекст
- `gpuworklib.StatisticsProcessor` — статистика сигналов (тоже ROCm)
- `gpuworklib.LchFarrowROCm` — задержка Farrow (ROCm)
