# Signal Generators — Краткий справочник

> Генерация CW/LFM/Noise/FormSignal на GPU: OpenCL (все платформы) + ROCm (AMD, ENABLE_ROCM=1)

**Namespace**: `signal_gen` | **Каталог**: `modules/signal_generators/`

---

## Концепция — зачем и что это такое

**Зачем нужен модуль?**
Мы тестируем фильтры, гетеродин, корреляторы — им нужен входной сигнал. Модуль генерирует этот сигнал прямо на GPU, без перекачки данных с CPU. Генераторы — это «источник» в пайплайне ЦОС.

---

### Простые генераторы — для базовых тестов

**CwGenerator** — тон. Одна синусоида фиксированной частоты, один канал. Используй когда нужно проверить фильтр или FFT на простом предсказуемом сигнале.

**LfmGenerator** — чирп. Сигнал, чья частота линейно растёт от f_start до f_end за время сигнала. Один канал. Используй для базовой проверки дечирпа или совмещённой фильтрации.

**NoiseGenerator** — белый гауссов шум. Один канал. Используй чтобы добавить реалистичный шум к тестам или проверить фильтр на шумовом сигнале.

> Эти три — «одноразовые»: один вызов → один буфер. Простые, но без поддержки массивов антенн и смешивания сигнал+шум.

---

### FormSignalGenerator — основной рабочий генератор

**Что делает**: генерирует многоканальный сигнал (N антенн) на GPU одним вызовом. Каждый канал — это CW или chirp с шумом. Каналы одинаковые, разница между ними — в задержке (если задана `tau_step`).

**Когда брать**: всегда, когда нужна имитация приёмного массива антенн. Большинство тестов гетеродина, фильтров и корреляторов работают именно с этим генератором.

**Три режима задержки между каналами**:
- `FIXED` — все каналы одинаковые, задержка нулевая
- `LINEAR` — каждый следующий канал задержан на фиксированный шаг (имитация плоского фронта волны)
- `RANDOM` — случайные задержки в заданном диапазоне

---

### DelayedFormSignalGenerator — с точной дробной задержкой

**Что делает**: то же самое, что FormSignalGenerator, но задержка на каждую антенну задаётся вручную (в мкс) и применяется через фильтр Farrow — это точная субсэмпловая интерполяция.

**Когда брать**: когда нужна реалистичная разность хода волны между антеннами с точностью до долей отсчёта. Нужно для тестирования алгоритмов локации/пеленгации.

**Ограничение**: чуть медленнее FormSignalGenerator из-за Farrow (48 коэффициентов × 5 порядков).

---

### LfmGeneratorAnalyticalDelay — задержка без интерполяции

**Что делает**: генерирует LFM на N антенн, задержка для каждой антенны вычисляется аналитически (через сдвиг фазы), без интерполяции.

**Когда брать**: когда нужна быстрая генерация с задержками, и субсэмпловая точность не важна. Например, для грубых проверок или больших задержек.

---

### LfmConjugateGenerator — опорный сигнал для дечирпа

**Что делает**: генерирует комплексно сопряжённую копию LFM сигнала — conj(s_tx). Это не имитация принятого сигнала, а опорный сигнал для операции дечирпа.

**Когда брать**: только в связке с модулем Heterodyne (дечирп-процессор). Самостоятельно не используется.

---

### FormScriptGenerator — настраиваемый генератор через DSL

**Что делает**: то же, что FormSignalGenerator, но формула сигнала задаётся текстовым скриптом (мини-язык → OpenCL kernel). Скрипт компилируется один раз и кэшируется на диск.

**Когда брать**: если стандартная формула (CW + chirp + шум) не подходит и нужно своё выражение — без перекомпиляции C++.

**Нюанс**: первый запуск медленный (~50 мс компиляция), после сохранения в файл — ~1 мс загрузка.

---

## Алгоритм getX (FormSignal)

```
X(t) = a·norm·exp(j·phi(t)) + an·norm·(nr + j·ni)
phi(t) = 2π·f0·t + π·fdev/ti·(t - ti/2)² + phase
Окно: X=0 если t<0 или t>ti-dt
```

---

## Какой класс выбрать

| Задача | Класс |
|--------|-------|
| CW/Chirp + шум, N каналов | `FormSignalGenerator` |
| То же + on-disk кэш kernel | `FormScriptGenerator` |
| Дробная задержка (Farrow 48×5) | `DelayedFormSignalGenerator` |
| LFM с аналитической задержкой (нет интерполяции) | `LfmGeneratorAnalyticalDelay` |
| conj(s_tx) для дечирпа | `LfmConjugateGenerator` |
| Простой CW / LFM / Noise | `CwGenerator` / `LfmGenerator` / `NoiseGenerator` |

---

## Быстрый старт

### C++ — FormSignalGenerator

```cpp
#include "generators/form_signal_generator.hpp"

signal_gen::FormParams p;
p.fs = 12e6;  p.f0 = 1e6;  p.antennas = 8;  p.points = 4096;
p.amplitude = 1.0;  p.noise_amplitude = 0.1;
p.tau_step = 1e-5;   // LINEAR 10 мкс (в СЕКУНДАХ!)

signal_gen::FormSignalGenerator gen(backend);
gen.SetParams(p);

auto input = gen.GenerateInputData();   // InputData<cl_mem>
clReleaseMemObject(input.data);         // caller owns!

auto cpu = gen.GenerateToCpu();         // vector<vector<complex<float>>>
```

### C++ — DelayedFormSignalGenerator

```cpp
#include "generators/delayed_form_signal_generator.hpp"

signal_gen::DelayedFormSignalGenerator gen(backend);
gen.SetParams(p);
gen.SetDelays({0.0f, 1.5f, 3.0f, 4.5f});   // МКС per-antenna!

auto input = gen.GenerateInputData();
clReleaseMemObject(input.data);
```

### C++ — LfmGeneratorAnalyticalDelay

```cpp
#include "generators/lfm_generator_analytical_delay.hpp"

signal_gen::LfmGeneratorAnalyticalDelay gen(backend, lfm_params);
gen.SetSampling({12e6, 4096});
gen.SetDelays({0.0f, 2.7f, 5.4f});   // МКС, 3 антенны

auto result = gen.GenerateToGpu();
clReleaseMemObject(result.data);
```

### C++ — CW / LFM / Noise (простые)

```cpp
signal_gen::CwGenerator cw(backend, {.f0=1e6, .amplitude=1.0});
cl_mem buf = cw.GenerateToGpu({12e6, 4096}, /*beams=*/1);
clReleaseMemObject(buf);
```

### Python — FormSignalGenerator

```python
import gpuworklib

ctx = gpuworklib.GPUContext(0)
gen = gpuworklib.FormSignalGenerator(ctx)

gen.set_params(fs=12e6, f0=1e6, antennas=8, points=4096,
               amplitude=1.0, noise_amplitude=0.1,
               tau_step=1e-5)      # LINEAR (в секундах!)
data = gen.generate()              # (8, 4096) complex64
```

### Python — DelayedFormSignalGenerator

```python
gen = gpuworklib.DelayedFormSignalGenerator(ctx)
gen.set_params(fs=1e6, f0=50000, antennas=4, points=4096)
gen.set_delays([0.0, 1.5, 3.0, 4.5])   # МКС!
data = gen.generate()                    # (4, 4096) complex64
```

### Python — LfmAnalyticalDelay

```python
gen = gpuworklib.LfmAnalyticalDelay(ctx)
gen.set_params(f_start=1e6, f_end=2e6, amplitude=1.0)
gen.set_sampling(fs=12e6, length=4096)
gen.set_delays([0.0, 2.7, 5.4])   # МКС
data = gen.generate_cpu()          # list of np.ndarray
```

---

## Ключевые параметры FormParams

| Параметр | Default | Описание |
|----------|---------|----------|
| `fs` | — | Частота дискретизации (Гц) |
| `f0` | 0.0 | Несущая (Гц) |
| `fdev` | 0.0 | Девиация chirp; 0 = CW |
| `antennas` | 1 | Каналов |
| `points` | — | Отсчётов на канал |
| `amplitude` | 1.0 | Амплитуда сигнала (a) |
| `noise_amplitude` | 0.0 | Шум (an); 0 = без шума |
| `tau_step` | 0.0 | Шаг задержки LINEAR (**секунды**) |
| `tau_base` | 0.0 | Базовая задержка (**секунды**) |
| `norm` | 1/√2 | Нормировка |

Режимы: `FIXED` (tau_step=0) · `LINEAR` (tau_step>0) · `RANDOM` (tau_min≠tau_max)

---

## Важные ловушки

| # | Ловушка |
|---|---------|
| ⚠️ | `clReleaseMemObject(input.data)` — caller owns, не забыть! |
| ⚠️ | `SetDelays` → **мкс**; `FormParams.tau_*` → **секунды** |
| ⚠️ | GPU точность ~1e-3 (флаг `-cl-fast-relaxed-math`) |
| ⚠️ | `SetDelays` размер = `params_.antennas` (иначе exception) |
| ⚠️ | `FormScriptGenerator`: первый `Compile()` ~50 мс, после `SaveKernel()` → `LoadKernel()` ~1 мс |

---

## Ссылки

- [Full.md](Full.md) — математика, pipeline, C4 диаграммы, все тесты
- [Doc/Modules/lch_farrow/Full.md](../lch_farrow/Full.md) — математика Farrow 48×5
- [Doc/Modules/heterodyne/Full.md](../heterodyne/Full.md) — использование LfmConjugateGenerator

---

*Обновлено: 2026-03-02*
