# vector_algebra — Краткий справочник

> Инверсия HPD матриц методом Холецкого на GPU (ROCm)

---

## Алгоритм

```
A (HPD n×n)  →  POTRF: A = U^H·U  →  POTRI: A^{-1} из U  →  Symmetrize  →  A^{-1}
```

---

## Быстрый старт

### C++

```cpp
#include "cholesky_inverter_rocm.hpp"
using namespace vector_algebra;

CholeskyInverterROCm inverter(backend);  // GpuKernel mode (default)

// CPU вектор → инверсия
drv_gpu_lib::InputData<std::vector<std::complex<float>>> input;
input.antenna_count = 1;
input.n_point = n * n;
input.data = matrix_flat;  // vector<complex<float>>, row-major

auto result = inverter.Invert(input);
auto A_inv = result.AsVector();   // flat n*n
auto mat2d = result.matrix();     // [n][n]

// Batched
auto batch = inverter.InvertBatch(batch_input, n);  // [batch][n][n] → result.matrices()
```

### Python

```python
ctx = gpuworklib.ROCmGPUContext(0)
inv = gpuworklib.CholeskyInverterROCm(ctx)  # GpuKernel mode

A_inv = inv.invert_cpu(A.flatten(), n)          # np.ndarray (n, n), complex64
results = inv.invert_batch_cpu(flat, n, batch)  # np.ndarray (batch, n, n)
```

---

## Режимы симметризации

| Режим | Описание |
|-------|----------|
| `GpuKernel` (default) | hiprtc kernel in-place — всё на GPU |
| `Roundtrip` | DtoH → CPU conj → HtoD (fallback) |

```python
inv = gpuworklib.CholeskyInverterROCm(ctx, gpuworklib.SymmetrizeMode.Roundtrip)
inv.set_symmetrize_mode(gpuworklib.SymmetrizeMode.GpuKernel)
```

---

## Форматы входных данных (C++)

| InputData\<T\> | Откуда данные |
|----------------|--------------|
| `vector<complex<float>>` | CPU → HtoD внутри |
| `void*` | ROCm device pointer (уже на GPU) |
| `cl_mem` | OpenCL буфер (ZeroCopy) |

---

## Точность (float32)

| n | ||A·A⁻¹ - I||_F |
|---|----------------|
| 5 | < 1e-5 |
| 64 | < 1e-3 |
| 341 | < 1e-2 |

---

## Ссылки

- [Full.md](Full.md) — математика, pipeline, C4 диаграммы, все тесты
- [Doc/Python/vector_algebra_api.md](../../Python/vector_algebra_api.md) — Python API

---

*Обновлено: 2026-03-01*
