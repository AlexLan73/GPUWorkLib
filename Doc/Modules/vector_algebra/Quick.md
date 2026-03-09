# vector_algebra — Краткий справочник

> Инверсия HPD матриц методом Холецкого на GPU (ROCm)

---

## Концепция — зачем и что это такое

**Зачем нужен модуль?**
В адаптивной обработке антенных решёток (алгоритмы MVDR/Capon) нужно вычислить оптимальные весовые коэффициенты для подавления помех. Для этого требуется решить уравнение с корреляционной матрицей — а значит, нужна её обратная матрица. Именно это и делает модуль.

---

### Что такое HPD матрица

HPD = Эрмитова положительно определённая матрица. Корреляционная матрица принятых сигналов антенной решётки — всегда HPD (если нет вырождения). Это позволяет использовать метод Холецкого — быстрый и численно устойчивый способ инвертирования.

---

### Как это работает (без формул)

1. **POTRF** — разложение Холецкого: матрица A раскладывается в произведение треугольной матрицы на её сопряжённую транспонированную. Это как «квадратный корень» матрицы.
2. **POTRI** — вычисление обратной матрицы через результат POTRF.
3. **Симметризация** — результат POTRI хранит только верхнетреугольную часть. Ядро на GPU зеркалит нижнюю треугольную часть.

---

### Когда брать какой класс

Сейчас только один класс: `CholeskyInverterROCm`. Два режима симметризации:

**GpuKernel** (по умолчанию) — симметризация прямо на GPU. Самый быстрый путь — данные не покидают GPU до финала.

**Roundtrip** — скачать с GPU на CPU, симметризовать там, залить обратно. Запасной вариант если что-то не так с GPU-ядром.

---

### Batched инверсия

Если нужно инвертировать много матриц (например, для каждого временного окна) — используй `InvertBatch`. Все матрицы обрабатываются параллельно на GPU. Входной формат: плоский массив [batch × n × n] в строчном порядке.

---

### ROCm-only

Только AMD GPU + Linux + ROCm. Использует rocSOLVER (POTRF/POTRI) и кастомное HIP-ядро для симметризации. На float32 точность: матрицы 5×5 — ошибка < 1e-5, матрицы 64×64 — < 1e-3.

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

- [API.md](API.md) — полный справочник сигнатур C++ и Python
- [Full.md](Full.md) — математика, pipeline, C4 диаграммы, все тесты
- [Doc/Python/vector_algebra_api.md](../../Python/vector_algebra_api.md) — Python API

---

*Обновлено: 2026-03-09*
