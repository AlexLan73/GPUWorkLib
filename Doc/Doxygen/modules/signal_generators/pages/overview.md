@page signal_generators_overview Signal Generators -- Обзор модуля

@tableofcontents

@section sg_overview_purpose Назначение

Модуль **signal_generators** предоставляет GPU-ускоренную генерацию комплексных IQ тест-сигналов
для задач цифровой обработки сигналов (ЦОС). Реализованы генераторы CW, LFM, шума,
многоантенных сигналов с задержками (FormSignal), дробной задержки (Farrow interpolation),
аналитической задержки LFM и DSL-компилятор пользовательских сигналов.

> **Namespace**: `signal_gen` | **Backend**: OpenCL + ROCm (HIP) | **Статус**: Active

@section sg_overview_classes Ключевые классы

| Класс | Описание | Заголовок |
|-------|----------|-----------|
| @ref signal_gen::CwGeneratorROCm | Continuous Wave: чистая несущая частота, multi-beam | `cw_generator_rocm.hpp` |
| @ref signal_gen::LfmGeneratorROCm | Linear Frequency Modulation (chirp-сигнал) | `lfm_generator_rocm.hpp` |
| @ref signal_gen::NoiseGeneratorROCm | Гауссов шум: Philox-2x32-10 PRNG + Box-Muller | `noise_generator_rocm.hpp` |
| @ref signal_gen::LfmConjugateGeneratorROCm | Сопряженный LFM (используется для dechirp) | `lfm_conjugate_generator_rocm.hpp` |
| @ref signal_gen::FormSignalGeneratorROCm | Multi-antenna getX: signal + noise + per-channel delays | `form_signal_generator_rocm.hpp` |
| @ref signal_gen::DelayedFormSignalGeneratorROCm | FormSignal + Farrow 48x5 дробная задержка (Lagrange) | `delayed_form_signal_generator_rocm.hpp` |
| @ref signal_gen::LfmGeneratorAnalyticalDelayROCm | LFM с аналитической задержкой (без интерполяции) | `lfm_generator_analytical_delay_rocm.hpp` |
| @ref signal_gen::FormScriptGeneratorROCm | DSL --> OpenCL kernel compiler + disk cache | `form_script_generator_rocm.hpp` |

@section sg_overview_arch Архитектура

Все генераторы следуют единой 6-слойной архитектуре Ref03:

- **Слой 1** -- `GpuContext`: per-module контекст (backend, stream, compiled module, shared buffers)
- **Слой 2** -- `IGpuOperation`: интерфейс (Name, Initialize, IsReady, Release)
- **Слой 3** -- `GpuKernelOp`: базовый доступ к compiled kernels через GpuContext
- **Слой 5** -- Конкретные операции: `CwGeneratorROCm`, `LfmGeneratorROCm` и т.д.
- **Слой 6** -- Facade: `FormSignalGeneratorROCm` объединяет signal + noise + delays

@note Все генераторы работают с комплексными данными `float2` (IQ) и поддерживают multi-beam/multi-channel режимы.

@warning Для AMD GPU (RDNA4+, gfx1201) используется ROCm/hipFFT. OpenCL/clFFT НЕ работает на новых архитектурах.

@section sg_overview_pipeline Конвейер генерации

```
Параметры --> Initialize() --> Generate() --> float2* (GPU buffer)
                                   |
                           [HIP kernel launch]
                                   |
                              GPU memory
```

Для `FormSignalGeneratorROCm` конвейер расширяется:

```
CwParams/LfmParams --> signal kernel --> + noise kernel --> per-antenna delay --> output
                                              |                    |
                                        Philox PRNG         LINEAR/CUSTOM mode
```

@section sg_overview_quickstart Быстрый старт

@subsection sg_overview_cpp C++ пример

@code{.cpp}
#include "modules/signal_generators/include/generators/cw_generator_rocm.hpp"

// Создание генератора CW-сигнала
signal_gen::CwGeneratorROCm gen(backend);

// Параметры: fs=1 МГц, f0=100 кГц, amplitude=1.0, N=4096, beams=12
signal_gen::CwParams p{1e6, 100e3, 1.0, 4096, 12};
gen.Initialize(p);

// Генерация сигнала на GPU
auto signal = gen.Generate();
// signal -- float2* в GPU памяти, 12 лучей x 4096 отсчетов
@endcode

@code{.cpp}
#include "modules/signal_generators/include/generators/form_signal_generator_rocm.hpp"

// FormSignal: многоантенный сигнал с задержками
signal_gen::FormSignalGeneratorROCm form_gen(backend);
signal_gen::FormSignalParams fp;
fp.fs = 1e6;
fp.n_points = 4096;
fp.n_antennas = 8;
fp.delay_mode = signal_gen::DelayMode::LINEAR;
fp.delay_base = 0.5;   // базовая задержка (мкс)
fp.delay_step = 0.1;   // шаг задержки между антеннами (мкс)
form_gen.Initialize(fp);

auto multi_signal = form_gen.Generate();
// multi_signal -- 8 антенн x 4096 IQ отсчетов
@endcode

@subsection sg_overview_python Python пример

@code{.py}
import gpuworklib as gw
import numpy as np

# Инициализация GPU контекста
ctx = gw.ROCmGPUContext(0)

# FormSignal генератор: 8 антенн, 4096 отсчетов
gen = gw.FormSignalGeneratorROCm(ctx, fs=1e6, n_points=4096, n_antennas=8)
signal = gen.generate()  # np.ndarray complex64, shape=(8, 4096)

# Проверка: спектр первой антенны
spectrum = np.fft.fft(signal[0])
freqs = np.fft.fftfreq(4096, d=1/1e6)
@endcode

@code{.py}
# DelayedFormSignal: дробная задержка через Farrow 48x5
gen_delayed = gw.DelayedFormSignalGeneratorROCm(
    ctx, fs=1e6, n_points=4096, n_antennas=8,
    delay_base=0.5, delay_step=0.1
)
signal_delayed = gen_delayed.generate()
@endcode

@section sg_overview_dependencies Зависимости

- **DrvGPU** -- управление GPU-контекстом, потоками, памятью
- **ROCm / HIP** -- запуск HIP kernels, hipFFT (для связанных модулей)
- **Philox PRNG** -- ГПСЧ для генерации шума (общий kernel `prng.cl`)

@section sg_overview_see_also Смотрите также

- @ref signal_generators_formulas -- Математические формулы генераторов
- @ref signal_generators_tests -- Тесты и бенчмарки
- @ref fft_func_overview -- FFT-обработка сгенерированных сигналов
- @ref filters_overview -- Фильтрация сигналов
