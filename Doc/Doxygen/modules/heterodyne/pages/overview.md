@page heterodyne_overview Heterodyne -- Обзор модуля

@tableofcontents

@section het_overview_purpose Назначение

Модуль **heterodyne** реализует stretch-processing (dechirp) ЛЧМ (LFM) радиолокационных сигналов
на GPU. Вычисляет **beat-частоту** (пропорциональна дальности до цели), **дальность** (м)
и **SNR** (дБ) для каждой антенны в многоканальной системе.

> **Namespace**: `drv_gpu_lib` | **Backend**: OpenCL + ROCm (HIP) | **Статус**: Active

@section het_overview_classes Ключевые классы

| Класс | Описание | Заголовок |
|-------|----------|-----------|
| @ref drv_gpu_lib::HeterodyneDechirp | Facade: SetParams, Dechirp, Correct, Process | `heterodyne_dechirp.hpp` |
| @ref drv_gpu_lib::IHeterodyneProcessor | Strategy interface -- абстракция backend'а | `i_heterodyne_processor.hpp` |
| HeterodyneProcessorOpenCL | OpenCL backend: cl kernels dechirp_multiply, dechirp_correct (planned) | `heterodyne_processor_opencl.hpp` |
| @ref drv_gpu_lib::HeterodyneProcessorROCm | ROCm backend: HIP kernels, hipEvent timing | `heterodyne_processor_rocm.hpp` |

@section het_overview_arch Архитектура

Модуль использует **Strategy pattern** для выбора backend'а:

```
HeterodyneDechirp (Facade)
    |
    +-- IHeterodyneProcessor (Strategy interface)
            |
            +-- HeterodyneProcessorOpenCL  (OpenCL backend)
            +-- HeterodyneProcessorROCm   (ROCm backend)
```

Facade автоматически создаёт нужный processor при инициализации
на основе типа backend'а в DrvGPU.

@section het_overview_pipeline Конвейер обработки (Pipeline)

```
rx_signal (float2*)   ref_signal (float2*)
       \                   /
        \                 /
     [dechirp_multiply kernel]
              |
      beat_signal (float2*)
              |
         [FFT (hipFFT)]
              |
      spectrum (float2*)
              |
     [find_peak + SNR]
              |
      DechirpResult:
        f_beat, range_m, snr_db
```

Для коррекции фазовых ошибок используется дополнительный этап:

```
beat_signal --> [dechirp_correct kernel] --> corrected_signal
                      |
              phase_step, DC removal
```

@section het_overview_kernels GPU Kernels

| Kernel | Grid | Описание |
|--------|------|----------|
| `dechirp_multiply` | 2D (sample, antenna) | \f$ s_{dc}[n, a] = s_{rx}[n, a] \cdot \overline{s_{tx}[n]} \f$ |
| `dechirp_correct` | 2D (sample, antenna) | Phase correction: `sincosf()` (SFU), DC removal |

@note Оба ядра используют **2D grid**: первая размерность -- отсчёты, вторая -- антенны.
Это позволяет обрабатывать все антенны за один kernel launch.

@warning `sincosf()` использует Special Function Unit (SFU) на AMD GPU -- быстрее,
чем раздельные sin/cos, но с ограниченной точностью (~1 ULP).

@section het_overview_quickstart Быстрый старт

@subsection het_overview_cpp C++ пример

@code{.cpp}
#include "modules/heterodyne/include/heterodyne_dechirp.hpp"

// Создание dechirp процессора
drv_gpu_lib::HeterodyneDechirp het(backend);

// Параметры ЛЧМ сигнала
drv_gpu_lib::HeterodyneParams p;
p.sample_rate = 12e6;    // частота дискретизации (Гц)
p.f_start     = 1e6;     // начальная частота ЛЧМ (Гц)
p.f_end       = 3e6;     // конечная частота ЛЧМ (Гц)
p.n_antennas  = 5;       // количество антенн
p.n_points    = 8000;    // отсчётов на антенну
het.SetParams(p);

// Полный конвейер: dechirp -> FFT -> find peak
auto result = het.Process(rx_signal);

// Результаты по каждой антенне
for (int a = 0; a < 5; ++a) {
  auto& r = result.antennas[a];
  // r.f_beat  -- beat-частота (Гц)
  // r.range_m -- дальность (м)
  // r.snr_db  -- SNR (дБ)
}
@endcode

@code{.cpp}
// Поэтапный режим (для отладки)
auto dechirp_buf = het.Dechirp(rx_signal);         // только dechirp multiply
auto corrected   = het.Correct(dechirp_buf);        // phase correction
auto result      = het.ProcessExternal(cl_mem_ptr);  // вход -- уже на GPU
@endcode

@subsection het_overview_python Python пример

@code{.py}
import gpuworklib as gw
import numpy as np

# Инициализация GPU контекста
ctx = gw.ROCmGPUContext(0)

# Создание dechirp процессора: 5 антенн, 8000 отсчётов
het = gw.HeterodyneDechirp(ctx, fs=12e6, f_start=1e6, f_end=3e6,
                            n_ant=5, n_pts=8000)

# Подготовка принятого сигнала (5 антенн x 8000 IQ отсчётов)
rx_signal = generate_lfm_echo(n_ant=5, n_pts=8000, delays=[100e-6, 200e-6, ...])
rx_signal = rx_signal.astype(np.complex64)

# Полный конвейер
result = het.process(rx_signal)

for a in range(5):
    print(f"Ant {a}: f_beat={result[a].f_beat:.1f} Hz, "
          f"range={result[a].range_m:.2f} m, "
          f"SNR={result[a].snr_db:.1f} dB")
@endcode

@code{.py}
# Визуализация: dechirp -> FFT спектр
import matplotlib.pyplot as plt

beat = het.dechirp(rx_signal)  # только dechirp multiply
spectrum = np.fft.fft(beat[0])
freqs = np.fft.fftfreq(8000, d=1/12e6)

plt.plot(freqs[:4000], 20*np.log10(np.abs(spectrum[:4000])))
plt.xlabel("Frequency (Hz)")
plt.ylabel("Magnitude (dB)")
plt.title("Beat spectrum -- Antenna 0")
plt.show()
@endcode

@section het_overview_params Параметры HeterodyneParams

| Параметр | Тип | Описание |
|----------|-----|----------|
| `sample_rate` | `double` | Частота дискретизации \f$ f_s \f$ (Гц) |
| `f_start` | `double` | Начальная частота ЛЧМ (Гц) |
| `f_end` | `double` | Конечная частота ЛЧМ (Гц) |
| `n_antennas` | `int` | Количество антенн (каналов) |
| `n_points` | `int` | Количество отсчётов на антенну |

Производные параметры:
- \f$ B = f_{\text{end}} - f_{\text{start}} \f$ -- полоса ЛЧМ
- \f$ T = N / f_s \f$ -- длительность импульса
- \f$ \mu = B / T \f$ -- скорость перестройки частоты

@section het_overview_deps Зависимости

- **DrvGPU** -- управление GPU-контекстом, потоками, памятью
- **ROCm / HIP** -- запуск HIP kernels (ROCm backend)
- **OpenCL** -- запуск cl kernels (OpenCL backend)
- **hipFFT / clFFT** -- FFT для вычисления спектра beat-сигнала

@section het_overview_see_also Смотрите также

- @ref heterodyne_formulas -- Математические формулы (dechirp, дальность, SNR)
- @ref heterodyne_tests -- Тесты, бенчмарки и графики pipeline
- @ref fft_func_overview -- FFT-обработка (используется внутри Process)
- @ref signal_generators_overview -- Генерация ЛЧМ-сигналов для тестирования
