@page signal_generators_tests Signal Generators -- Тесты и бенчмарки

@tableofcontents

@section sg_tests_intro Введение

Модуль **signal_generators** тестируется на двух уровнях:
- **C++ тесты** -- проверка корректности GPU-вычислений, сравнение GPU vs CPU
- **Python тесты** -- валидация через NumPy/SciPy, визуализация результатов

Все тесты выполняются на реальном GPU (AMD ROCm или NVIDIA OpenCL).

---

@section sg_tests_cpp C++ тесты

Расположение: `modules/signal_generators/tests/`

| Файл | Описание | Кол-во тестов |
|------|----------|:------------:|
| `test_signal_generators_rocm_basic.hpp` | CW, LFM, LfmConjugate, Noise: GPU vs CPU эталон | 4+ |
| `test_form_signal_rocm.hpp` | FormSignalROCm: getX multi-antenna, delay modes, noise | 4+ |

@subsection sg_tests_cpp_basic Базовые генераторы (CW, LFM, Noise)

Тест `test_signal_generators_rocm_basic.hpp` проверяет:

- **CW Generator**: 12 лучей x 4096 точек, частоты \f$ f_i = f_0 + i \cdot \Delta f \f$
- **LFM Generator**: chirp от \f$ f_{start} \f$ до \f$ f_{end} \f$, проверка мгновенной частоты
- **LfmConjugate**: комплексное сопряжение, \f$ s_{conj} = s_{LFM}^* \f$
- **Noise Generator**: статистические проверки -- среднее \f$ \approx 0 \f$, дисперсия \f$ \approx \sigma^2 \f$

@code{.cpp}
// Пример: проверка CW генератора
signal_gen::CwGeneratorROCm gen(backend);
signal_gen::CwParams p{1e6, 100e3, 1.0, 4096, 12};
gen.Initialize(p);
auto gpu_signal = gen.Generate();

// CPU reference
std::vector<float2> cpu_ref(4096);
for (int n = 0; n < 4096; ++n) {
    float phase = 2.0f * M_PI * 100e3f * n / 1e6f;
    cpu_ref[n] = {cosf(phase), sinf(phase)};
}

// Сравнение: максимальная ошибка < 1e-5
ASSERT_MAX_ERROR(gpu_signal, cpu_ref, 1e-5f);
@endcode

@subsection sg_tests_cpp_formsignal FormSignal тесты

Тест `test_form_signal_rocm.hpp` проверяет:

- Генерация getX для 8 антенн
- Режимы задержки: LINEAR, CUSTOM
- Добавление шума: SNR проверка
- Multi-beam конфигурация

---

@section sg_tests_python Python тесты

Расположение: `Python_test/signal_generators/`

| Файл | Описание |
|------|----------|
| `test_form_signal.py` | FormSignalGenerator: getX, noise validation, NumPy reference |
| `test_delayed_form_signal.py` | DelayedFormSignal: Farrow interpolation, integer + fractional delays |
| `test_lfm_analytical_delay.py` | LfmAnalyticalDelay: per-antenna delays, zero-boundary check |
| `example_form_signal.py` | Tutorial / code examples с визуализацией |

@subsection sg_tests_python_formsignal test_form_signal.py

Проверяет генерацию многоантенного сигнала:

@code{.py}
import gpuworklib as gw
import numpy as np

ctx = gw.ROCmGPUContext(0)
gen = gw.FormSignalGeneratorROCm(ctx, fs=1e6, n_points=4096, n_antennas=8)
signal = gen.generate()

# Проверка: 8 антенн, 4096 отсчетов
assert signal.shape == (8, 4096)

# Проверка спектра первой антенны
spectrum = np.abs(np.fft.fft(signal[0]))
peak_idx = np.argmax(spectrum[:2048])
peak_freq = peak_idx * 1e6 / 4096
assert abs(peak_freq - 100e3) < 300  # точность < 300 Гц
@endcode

@subsection sg_tests_python_delayed test_delayed_form_signal.py

Тестирование дробной задержки Farrow 48x5:

@code{.py}
# Integer delay: проверка точного сдвига на целое число отсчетов
gen = gw.DelayedFormSignalGeneratorROCm(ctx, ...)
signal = gen.generate()

# Корреляция между антеннами: пик на позиции задержки
corr = np.correlate(signal[0], signal[1], mode='full')
lag = np.argmax(np.abs(corr)) - len(signal[0]) + 1
assert abs(lag - expected_delay_samples) <= 1

# Fractional delay: ошибка интерполяции < 0.01 sample
@endcode

@subsection sg_tests_python_analytical test_lfm_analytical_delay.py

Аналитическая задержка LFM (без артефактов интерполяции):

@code{.py}
gen = gw.LfmGeneratorAnalyticalDelayROCm(ctx, ...)
signal = gen.generate()

# Проверка: s(t) = 0 для t < tau_ant
for ant in range(n_antennas):
    delay_samples = int(delays[ant] * fs / 1e6)
    assert np.allclose(signal[ant, :delay_samples], 0.0, atol=1e-7)
@endcode

---

@section sg_tests_benchmarks Бенчмарки

Классы бенчмарков наследуют `GpuBenchmarkBase` и измеряют производительность
каждого этапа генерации.

| Класс бенчмарка | Backend | Метрики |
|-----------------|---------|---------|
| `FormSignalGeneratorBenchmark` | OpenCL | FormSignal kernel, FarrowDelay stages, Upload/Download |
| `FormSignalGeneratorROCmBenchmark` | ROCm | Upload (H2D), Kernel execution, Download (D2H) |

@subsection sg_tests_bench_stages Стадии профилирования

Для `FormSignalGeneratorROCm`:

| Стадия | Описание |
|--------|----------|
| Upload | Копирование параметров Host --> Device |
| SignalKernel | Запуск kernel генерации сигнала |
| NoiseKernel | Запуск kernel генерации шума (Philox + Box-Muller) |
| DelayKernel | Применение per-antenna задержек |
| Download | Копирование результата Device --> Host |

Результаты профилирования: `Results/Profiler/GPU_00_SignalGen*/`

@warning Профилирование выполняется ТОЛЬКО через `GPUProfiler` из DrvGPU.
Ручной вывод через `GetStats()` + `con.Print` запрещен.

---

@section sg_tests_plots Графики

@subsection sg_tests_plots_cw FormSignal -- CW спектр

@image html signal_generators/FormSignal/example_01_cw_spectrum.png "CW: FFT спектр (GPU) -- один тон на заданной частоте" width=700px

Спектр CW-сигнала, сгенерированного на GPU. Виден один четкий пик
на частоте \f$ f_0 \f$.

@subsection sg_tests_plots_chirp FormSignal -- Chirp спектрограмма

@image html signal_generators/FormSignal/example_02_chirp_spectrogram.png "LFM chirp spectrogram -- линейное изменение частоты" width=700px

Спектрограмма (время-частота) LFM-сигнала. Частота линейно нарастает
от \f$ f_{start} \f$ до \f$ f_{end} \f$.

@subsection sg_tests_plots_multi FormSignal -- Multichannel waterfall

@image html signal_generators/FormSignal/example_03_multichannel.png "8-antenna multichannel waterfall -- многоканальный сигнал" width=700px

Водопадная диаграмма для 8 антенн. Видны per-antenna задержки
в режиме LINEAR.

@subsection sg_tests_plots_gpu_vs_numpy FormSignal -- GPU vs NumPy

@image html signal_generators/FormSignal/example_04_gpu_vs_numpy.png "GPU vs NumPy reference comparison -- проверка корректности" width=700px

Сравнение GPU-генерации с NumPy-эталоном. Максимальная ошибка < \f$ 10^{-5} \f$.

@subsection sg_tests_plots_integer_delay DelayedFormSignal -- Integer delay

@image html signal_generators/DelayedFormSignal/plot1_integer_delay.png "Integer delay (N samples) -- целочисленная задержка" width=700px

Задержка на целое число отсчетов. Идеальный сдвиг без интерполяции.

@subsection sg_tests_plots_fractional_delay DelayedFormSignal -- Fractional delay (Farrow 48x5)

@image html signal_generators/DelayedFormSignal/plot2_fractional_delay.png "Fractional delay: Lagrange interpolation -- дробная задержка Farrow" width=700px

Дробная задержка через интерполяцию Лагранжа 4-го порядка.
Таблица 48x5 квантует дробную часть с шагом 1/48 sample.

@subsection sg_tests_plots_analytical LfmAnalyticalDelay -- Multi-antenna

@image html signal_generators/LfmAnalyticalDelay/plot3_multiantenna_delays.png "LFM: per-antenna analytical delays -- аналитические задержки" width=700px

LFM-сигнал с аналитическими per-antenna задержками. Нулевой сигнал
до момента \f$ t = \tau_{ant} \f$, далее идеальный LFM без артефактов.

---

@section sg_tests_see_also Смотрите также

- @ref signal_generators_overview -- Обзор модуля и быстрый старт
- @ref signal_generators_formulas -- Математические формулы
