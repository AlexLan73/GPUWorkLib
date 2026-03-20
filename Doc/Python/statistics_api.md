# StatisticsProcessor — Python API

> GPU-статистика для многолучевых комплексных сигналов (ROCm)

**Модуль**: `gpuworklib.StatisticsProcessor`
**Платформа**: Linux + AMD GPU (ROCm/HIP), `-DENABLE_ROCM=ON`
**Источник**: `python/py_statistics.hpp`

---

## Быстрый старт

```python
import sys
sys.path.insert(0, 'build/debian-radeon9070/python')
import gpuworklib
import numpy as np

ctx  = gpuworklib.ROCmGPUContext(0)
proc = gpuworklib.StatisticsProcessor(ctx)

beam_count, n_point = 4, 65536
data = (np.random.randn(beam_count * n_point) +
        1j * np.random.randn(beam_count * n_point)).astype(np.complex64)

# Статистика + медиана за один GPU-вызов (рекомендуется)
results = proc.compute_all(data, beam_count=beam_count)
for r in results:
    print(f"Beam {r['beam_id']}: std={r['std_dev']:.4f}, median={r['median_magnitude']:.4f}")
```

---

## Конструктор

```python
proc = gpuworklib.StatisticsProcessor(ctx)
```

| Параметр | Тип | Описание |
|----------|-----|----------|
| `ctx` | `ROCmGPUContext` | ROCm GPU контекст (**НЕ** `GPUContext`!) |

Инициализация ленивая — GPU-буферы и JIT-компиляция ядер происходят при первом вызове.

---

## Методы

### `compute_all` ⭐ (рекомендуется)

Вычисляет полную статистику **и** медиану за один GPU-вызов.
Быстрее чем `compute_statistics` + `compute_median` отдельно — устраняет двойной PCIe upload.

```python
def compute_all(
    data: np.ndarray,    # complex64, shape (B*N,) или (B,N) C-contiguous
    beam_count: int = 1
) -> list[dict]:
```

**Возвращает** `list` из `beam_count` словарей:

| Ключ | Тип | Описание |
|------|-----|----------|
| `beam_id` | `int` | Индекс луча (0-based) |
| `mean_real` | `float` | Re(комплексного среднего) |
| `mean_imag` | `float` | Im(комплексного среднего) |
| `mean_magnitude` | `float` | E[|z|] — среднее модулей |
| `variance` | `float` | Var(|z|), ddof=0 |
| `std_dev` | `float` | sqrt(variance) |
| `median_magnitude` | `float` | sorted_magnitudes[N/2] |

```python
results = proc.compute_all(data, beam_count=4)
# results[0] == {
#   'beam_id': 0,
#   'mean_real': -0.0003, 'mean_imag': 0.0012,
#   'mean_magnitude': 0.7979, 'variance': 0.1366,
#   'std_dev': 0.3697, 'median_magnitude': 0.7854
# }
```

---

### `compute_all_float`

Статистика + медиана для **float** магнитуд (уже вычисленных, напр. после `|FFT|`).

```python
def compute_all_float(
    data: np.ndarray,    # float32, shape (B*N,) или (B,N)
    beam_count: int = 1
) -> list[dict]:
```

**Возвращает** те же 7 ключей, но:
- `mean_real` всегда `0.0`
- `mean_imag` всегда `0.0`

Это **документированное поведение** float-пути: комплексное среднее не вычисляется.

```python
mags = np.abs(data).astype(np.float32)      # или |FFT| от FFTProcessor
results = proc.compute_all_float(mags, beam_count=4)
assert results[0]['mean_real'] == 0.0       # всегда
assert results[0]['mean_imag'] == 0.0       # всегда
```

---

### `compute_statistics`

Полная статистика (mean + variance + std_dev + mean_magnitude) без медианы.
Реализован через `welford_fused` kernel — один проход по данным.

```python
def compute_statistics(
    data: np.ndarray,    # complex64
    beam_count: int = 1
) -> list[dict]:
# Возвращает: beam_id, mean_real, mean_imag, variance, std_dev, mean_magnitude
```

---

### `compute_median`

Только медиана модулей (GPU radix sort → extract middle).

```python
def compute_median(
    data: np.ndarray,    # complex64
    beam_count: int = 1
) -> list[dict]:
# Возвращает: [{'beam_id': int, 'median_magnitude': float}, ...]
```

---

### `compute_mean`

Только комплексное среднее (двухфазная редукция).

```python
def compute_mean(
    data: np.ndarray,    # complex64
    beam_count: int = 1
) -> list[dict]:
# Возвращает: [{'beam_id': int, 'mean_real': float, 'mean_imag': float}, ...]
```

---

### `compute_statistics_float`

Статистика (без медианы) для float магнитуд.

```python
def compute_statistics_float(
    data: np.ndarray,    # float32
    beam_count: int = 1
) -> list[dict]:
# Возвращает: beam_id, variance, std_dev, mean_magnitude
# (без mean_real / mean_imag)
```

---

### `compute_median_float`

Только медиана для float магнитуд.

```python
def compute_median_float(
    data: np.ndarray,    # float32
    beam_count: int = 1
) -> list[dict]:
# Возвращает: [{'beam_id': int, 'median_magnitude': float}, ...]
```

---

## Полный пример с NumPy верификацией

```python
import sys
sys.path.insert(0, 'build/debian-radeon9070/python')
import gpuworklib
import numpy as np

ctx  = gpuworklib.ROCmGPUContext(0)
proc = gpuworklib.StatisticsProcessor(ctx)

beam_count, n_point = 4, 65536
rng  = np.random.default_rng(42)
data = (rng.uniform(-1, 1, beam_count * n_point) +
        1j * rng.uniform(-1, 1, beam_count * n_point)).astype(np.complex64)

# GPU: ComputeAll
gpu_results = proc.compute_all(data, beam_count=beam_count)

# NumPy reference
for b in range(beam_count):
    beam = data[b * n_point:(b + 1) * n_point]
    mags = np.abs(beam)

    np_std    = float(np.std(mags, ddof=0))
    np_median = float(np.sort(mags)[n_point // 2])

    r = gpu_results[b]
    print(f"Beam {b}:")
    print(f"  std:    GPU={r['std_dev']:.6f}  NumPy={np_std:.6f}  err={abs(r['std_dev']-np_std):.2e}")
    print(f"  median: GPU={r['median_magnitude']:.6f}  NumPy={np_median:.6f}  err={abs(r['median_magnitude']-np_median):.2e}")
```

---

## Формат входных данных

**Beam-major layout**: все сэмплы луча 0, затем луча 1, ...

```python
# 1D: (beam_count * n_point,)  — beam-major
data_1d = np.zeros(4 * 65536, dtype=np.complex64)

# 2D: (beam_count, n_point) C-contiguous — автоматически правильный порядок
data_2d = data_1d.reshape(4, 65536)

# Оба варианта работают одинаково:
r1 = proc.compute_all(data_1d, beam_count=4)
r2 = proc.compute_all(data_2d, beam_count=4)
```

> ⚠️ Transposed `(n_point, beam_count)` — неправильный порядок, результат некорректен!

---

## Определение медианы

Медиана определена как `sorted_magnitudes[N // 2]`, **не** среднее двух средних (как `np.median`).

```python
# Правильный NumPy эквивалент:
median_ref = float(np.sort(np.abs(beam))[n_point // 2])

# НЕ это:
# median_ref = float(np.median(np.abs(beam)))  # ← другая формула для чётного N
```

---

## Запуск (Linux, AMD GPU)

```bash
# Нужна группа render для доступа к GPU:
sg render -c "python3 my_script.py"

# Тесты ComputeAll (NumPy часть — без GPU):
python Python_test/statistics/test_compute_all.py

# Все statistics тесты с GPU:
sg render -c "python Python_test/statistics/test_compute_all.py"
```

---

## Выбор метода

| Задача | Метод |
|--------|-------|
| Нужна статистика + медиана | `compute_all` ⭐ |
| Нужна только статистика | `compute_statistics` |
| Нужна только медиана | `compute_median` |
| Только среднее (быстро) | `compute_mean` |
| Float магнитуды, всё | `compute_all_float` |
| Float магнитуды, только stats | `compute_statistics_float` |

---

## Ссылки

- [Doc/Modules/statistics/API.md](../Modules/statistics/API.md) — C++ API reference
- [Doc/Modules/statistics/Full.md](../Modules/statistics/Full.md) — полная документация с математикой
- [Python_test/statistics/test_compute_all.py](../../Python_test/statistics/test_compute_all.py) — тесты

---

*Обновлено: 2026-03-20*
