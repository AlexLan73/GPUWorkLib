# RangeAngleProcessor — Шпаргалка

> 3D FFT модуль: ЛЧМ-дечирп + Range FFT + 2D Beam FFT → дальность + угол

**Namespace**: `range_angle` | **Каталог**: `modules/range_angle/`
**Backend**: ROCm only (`ENABLE_ROCM=1`, AMD GPU gfx1201+)

---

## Концепция — зачем и что это такое

Модуль обрабатывает данные от 2D антенной решётки (по умолчанию 16×16 = 256 антенн) и выдаёт трёхмерный куб мощности `[дальность × азимут × элевация]`, находя цели.

**Физика**: радар посылает ЛЧМ-сигнал с полосой 10 МГц. Отражение с дальности R приходит с задержкой τ = 2R/c. Дечирп (умножение на ophorный сигнал) превращает задержку в частоту биений. Range FFT → бин дальности (15 м). Пространственная 2D FFT по антенной решётке → угол (7.2°).

**Аналогия**: как GPS — только через активный радар: каждая антенна «слышит» эхо ЛЧМ-сигнала, обработка говорит откуда пришло.

### Режимы поиска пиков

| Режим | Что делает | Когда брать |
|-------|-----------|-------------|
| `TOP_1` | Один глобальный максимум мощности | Одна цель в сцене (по умолчанию) |
| `TOP_N` | До n_peaks целей (TODO) | Несколько целей |

### Связи с другими модулями

- **Heterodyne** — то же самое, но для одной антенны (упрощённый аналог)
- **signal_generators** — используется для генерации опорного ЛЧМ (`gen_ref_kernel` внутри)
- **DrvGPU** — `IBackend*`, `GpuContext`, `GPUProfiler`, `ConsoleOutput`

### Ограничения

- ROCm only — не работает на Windows/NVIDIA
- `n_ant_az` и `n_ant_el` должны быть степенью 2 для корректного fftshift
- TOP_N не реализован (TODO)
- SNR в `TargetInfo.snr_db` не вычисляется (= 0)

---

## Быстрый старт (C++)

```cpp
#include "range_angle/include/range_angle_processor.hpp"

// 1. Параметры
range_angle::RangeAngleParams p;
p.n_ant_az = 8; p.n_ant_el = 8; p.n_samples = 50'000;
// Остальные умолчания: 10 МГц, 12 МГц, 435 МГц, шаг 0.345 м

// 2. Создать и настроить
range_angle::RangeAngleProcessor proc(backend);  // IBackend*
proc.SetParams(p);  // вычисляет nfft_range, n_range_bins, range_res_m

// 3. Подготовить данные [n_ant × n_samples] complex<float>, antenna-major
std::vector<std::complex<float>> iq(p.GetNAntennas() * p.n_samples);
// ... заполнить ...

// 4. Обработать
auto result = proc.Process(iq);
if (result.success) {
    // result.targets[0].range_m, angle_az_deg, angle_el_deg
    // result.power_cube  — 3D куб float [n_range_bins × n_az × n_el]
}
```

---

## Быстрый старт (Python)

```python
import numpy as np
import gpu_worklib as gw

ctx = gw.ROCmGPUContext(0)
p = gw.RangeAngleParams()
p.n_ant_az = 8; p.n_ant_el = 8; p.n_samples = 50_000

proc = gw.RangeAngleProcessor(ctx)
proc.set_params(p)

n_ant = p.get_n_antennas()  # 64
iq = np.ones(n_ant * 50_000, dtype=np.complex64)

result = proc.process(iq, download_result=True)
if result.success:
    print(f"R={result.targets[0].range_m:.0f}m")
    cube = result.power_cube_numpy()  # shape: (n_range_bins, 8, 8)
```

---

## Ключевые параметры (умолчания)

| Параметр | Умолчание | Описание |
|---------|-----------|---------|
| `n_ant_az / n_ant_el` | 16 / 16 | Размер решётки (всего 256 антенн) |
| `n_samples` | 1 300 000 | Отсчётов на антенну |
| `f_start / f_end` | -5e6 / +5e6 | Полоса ЛЧМ (baseband), Гц |
| `sample_rate` | 12e6 | Гц |
| `nfft_range` | 0 → авто 2^21 | 0 = авто: следующая 2^n |
| `carrier_freq` | 435e6 | Гц |
| `antenna_spacing` | 0.345 м | λ/2 при 435 МГц |

**Вычисляемые после SetParams**:
- `n_range_bins = nfft_range / 2`
- `range_res_m = c / (2·B) = 15 м`

---

## Типичные ошибки

| Ошибка | Причина | Решение |
|--------|---------|---------|
| FFT в default stream / гонки | Забыли `hipfftSetStream` | Уже сделано в `InitPlan` |
| «Unknown GPU» в profiler | Не вызван `SetGPUInfo` перед `Start()` | `profiler.SetGPUInfo(...)` до `Start()` |
| Падение размерности | `n_ant_az` или `n_ant_el` не степень 2 | fftshift требует чётное число |
| Неверная дальность | Неправильный `chirp_rate` | Убедиться что `f_start/f_end` — baseband |
| OOM на 16×16×1.3M | Нехватка GPU памяти | ~6 ГБ требуется для полного пайплайна |

---

## Вычисление занятой памяти

```
kInput:     n_ant × n_samples × 8 байт   (256 × 1.3M × 8 = ~2.7 ГБ)
kRef:       n_samples × 8                (1.3M × 8 = 10 МБ)
kDechirped: n_ant × nfft_range × 8       (256 × 2.1M × 8 = ~4.3 ГБ)
kTransposed: n_range_bins × n_ant × 8   (1M × 256 × 8 = ~2.1 ГБ) (reuse kDechirped)
kPowerCube:  n_range_bins × n_az × n_el × 4  (1M × 16 × 16 × 4 = ~1 ГБ)
```

Итого ~4-5 ГБ GPU при 16×16×1.3M (буферы переиспользуются in-place).

---

## См. также

- [Full.md](Full.md) — полная документация с математикой и диаграммами
- [API.md](API.md) — справочник API
- [Heterodyne](../heterodyne/Full.md) — одноантенный аналог
