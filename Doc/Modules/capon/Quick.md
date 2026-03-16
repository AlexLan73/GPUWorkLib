# Capon (MVDR) — Краткий справочник

> MVDR beamformer: пространственный спектр + адаптивное подавление помех на GPU (ROCm only)

---

## Концепция — зачем и что это такое

**Зачем нужен модуль?**
Когда антенная решётка принимает сигнал вместе с помехами из других направлений, обычное диаграммообразование (фазовое сложение каналов) не справляется — помехи попадают в боковые лепестки. Алгоритм Кейпона (MVDR) адаптирует весовые коэффициенты антенн так, чтобы одновременно не искажать целевой сигнал и максимально подавить всё остальное.

**Аналогия**: классическое ДО — бинокль с фиксированной апертурой. Кейпон — адаптивные очки, которые «знают», откуда идут помехи, и автоматически их ослабляют.

---

### Что делает каждый режим

| Режим | Что вычисляет | Когда брать |
|-------|--------------|-------------|
| `ComputeRelief` | Рельеф MVDR: `z[m] = 1/Re(u^H·R⁻¹·u)` → `float[M]` | Найти направления на источники (пространственный спектр) |
| `AdaptiveBeamform` | Адаптивные лучи: `Y_out = (R⁻¹·U)^H·Y` → `complex[M×N]` | Выходные сигналы по лучам с подавлением помех |

Оба режима: шаги 1–3 одинаковые (ковариация → инверсия), расходятся на шаге 4.

---

### Связи с другими модулями

- **vector_algebra** — `CholeskyInverterROCm`: инверсия R через POTRF+POTRI+symmetrize. Capon сам матрицу **не обращает** — делегирует.
- **statistics** — аналогичные паттерны накопления по отсчётам, но в statistics — скаляры, здесь — матрица [P×P].
- **DrvGPU** — `IBackend*`, `GpuContext`.

---

### Ограничения

- **ROCm-only**: недоступен на Windows/OpenCL.
- **Column-major**: матрицы Y и U в column-major (как rocBLAS/rocSOLVER). NumPy row-major → нужен `np.asfortranarray()`.
- **N < P требует mu > 0**: матрица R вырождена без регуляризации при недостаточном числе отсчётов.
- **Python API не реализован**: запланирован.
- **Статус**: framework готов, rocBLAS CGEMM помечены TODO.

---

## Алгоритм

```
Y [P×N],  U [P×M]
  1. R = Y·Y^H/N + μI       ← rocBLAS CGEMM + HIP kernel add_regularization
  2. R⁻¹                    ← vector_algebra::CholeskyInverterROCm (POTRF+POTRI)
  3a. Relief: z[m] = 1/Re(u^H·R⁻¹·u)     ← CGEMM W=R⁻¹·U + HIP compute_capon_relief
  3b. Beamform: Y_out = (R⁻¹·U)^H·Y      ← CGEMM × 2
```

---

## Быстрый старт

### C++

```cpp
#include "capon_processor.hpp"

capon::CaponProcessor proc(backend);  // ROCm backend

capon::CaponParams params;
params.n_channels   = 8;     // P — каналов
params.n_samples    = 128;   // N — отсчётов (N >= P рекомендуется)
params.n_directions = 32;    // M — направлений
params.mu           = 0.01f; // регуляризация, всегда > 0

// Y: complex<float>[P*N], column-major  (Y[m*P+p] — столбец m, строка p)
// U: complex<float>[P*M], column-major
// ULA: U[m*P+p] = exp(j * 2π * p * 0.5 * sin(θ[m]))
auto relief = proc.ComputeRelief(Y, U, params);
// relief.relief[m] — MVDR-мощность в направлении m

auto beam = proc.AdaptiveBeamform(Y, U, params);
// beam.output[m*N + n] — отсчёт n в луче m (complex<float>)

// GPU-to-GPU (D2D, proc НЕ освобождает буферы)
auto r = proc.ComputeRelief(hip_Y_ptr, hip_U_ptr, params);
```

### Python — NumPy эталон

```python
import numpy as np

def capon_relief_numpy(Y, U, mu=0.01):
    """Y: [P,N], U: [P,M] — complex column-major"""
    P, N = Y.shape
    R = (Y @ Y.conj().T) / N + mu * np.eye(P, dtype=complex)
    R_inv = np.linalg.inv(R)
    W = R_inv @ U
    return (1.0 / np.real(np.sum(U.conj() * W, axis=0))).astype(np.float32)
```

---

## Ключевые параметры

| Параметр | Пример | Описание |
|----------|--------|----------|
| `n_channels` | 8 | P — число каналов (строки Y и U) |
| `n_samples` | 128 | N — число отсчётов (N ≥ P рекомендуется) |
| `n_directions` | 32 | M — число направлений (столбцы U) |
| `mu` | 0.01 | 0.001 (слабая), 0.01 (норма), 0.1 (N<<P) |

---

## Тесты (C++, ROCm)

| # | Тест | Порог |
|---|------|-------|
| 01 | Только шум → relief[m] > 0 | `> 0` |
| 02 | Размерность relief == M | точное равенство |
| 03 | Размерность output == M×N | точное равенство |
| 04 | N < P, mu=0.1 → isfinite && ≥ 0 | `isfinite && ≥ 0` |
| 05 | GPU-to-GPU | SKIP (TODO) |

---

## Ссылки

- [Full.md](Full.md) — математика, C4, тесты, нюансы
- [API.md](API.md) — все классы и методы
- [../vector_algebra/Full.md](../vector_algebra/Full.md) — CholeskyInverterROCm
- [Doc_Addition/Capon/capon_test/](../../../Doc_Addition/Capon/capon_test/) — ArrayFire прототип

*Обновлено: 2026-03-16*
