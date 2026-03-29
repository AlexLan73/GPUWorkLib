@page heterodyne_tests Heterodyne -- Тесты и бенчмарки

@tableofcontents

@section het_tests_intro Введение

Тестирование модуля **heterodyne** покрывает полный конвейер dechirp:
от одиночной антенны до 5-антенного массива, проверку beat-частоты, дальности и SNR,
сравнение GPU vs CPU, а также поэтапную отладку pipeline.
Бенчмарки измеряют производительность каждой стадии для OpenCL и ROCm backend'ов.

---

@section het_tests_cpp C++ тесты

@subsection het_tests_cpp_basic Базовые тесты (test_heterodyne_basic.hpp)

Файл: `modules/heterodyne/tests/test_heterodyne_basic.hpp`

| # | Тест | Описание |
|---|------|----------|
| 1 | SingleAntennaDechirp | 1 антенна, задержка \f$ \tau = 100 \f$ мкс, проверка \f$ f_{\text{beat}} \f$ |
| 2 | MultiAntennaLinear | 5 антенн с линейно возрастающими задержками |
| 3 | CorrectionDC | DC removal + phase correction, проверка чистоты спектра |

@subsection het_tests_cpp_pipeline Pipeline тесты (test_heterodyne_pipeline.hpp)

Файл: `modules/heterodyne/tests/test_heterodyne_pipeline.hpp`

| # | Тест | Описание |
|---|------|----------|
| 1 | FullProcess | Полный конвейер Process(): dechirp -> FFT -> peak -> range, SNR |
| 2 | ProcessExternalGPU | Вход -- cl_mem (данные уже на GPU), без upload |
| 3 | StepByStepMatch | Dechirp() + FFT раздельно == Process() |
| 4 | CorrectThenProcess | Correct() -> Process(): влияние коррекции на SNR |

@subsection het_tests_cpp_example Пример C++ теста

@code{.cpp}
// test_heterodyne_basic.hpp -- SingleAntennaDechirp
void TestSingleAntennaDechirp(DrvGPU& drv) {
  auto& con = ConsoleOutput::GetInstance();

  drv_gpu_lib::HeterodyneDechirp het(drv.GetBackend());
  drv_gpu_lib::HeterodyneParams p;
  p.sample_rate = 12e6;
  p.f_start     = 1e6;
  p.f_end       = 3e6;
  p.n_antennas  = 1;
  p.n_points    = 8000;
  het.SetParams(p);

  // Генерация ЛЧМ эха с задержкой 100 мкс -> дальность ~15 км
  auto rx = GenerateLfmEcho(p, /*delay_us=*/100.0);
  auto result = het.Process(rx);

  double expected_range = 100e-6 * 3e8 / 2.0;  // ~15000 м
  ASSERT_NEAR(result.antennas[0].range_m, expected_range, 100.0);
  ASSERT_GT(result.antennas[0].snr_db, 20.0);  // SNR > 20 дБ

  con.Print("TestSingleAntennaDechirp: PASSED");
}
@endcode

---

@section het_tests_python Python тесты

@subsection het_tests_python_basic Базовые тесты (test_heterodyne.py)

Файл: `Python_test/heterodyne/test_heterodyne.py`

| # | Тест | Описание |
|---|------|----------|
| 1 | test_basic_dechirp | Dechirp одной антенны, проверка beat-частоты |
| 2 | test_multi_antenna | 5 антенн, различные задержки, проверка дальностей |
| 3 | test_snr_estimation | SNR > 20 дБ для чистого сигнала |

@subsection het_tests_python_rocm ROCm-специфичные тесты (test_heterodyne_rocm.py)

Файл: `Python_test/heterodyne/test_heterodyne_rocm.py`

| # | Тест | Описание |
|---|------|----------|
| 1 | test_rocm_dechirp | Dechirp через ROCm backend |
| 2 | test_rocm_process | Полный Process() на ROCm |
| 3 | test_rocm_gpu_input | Вход -- device pointer (hipMalloc) |

@subsection het_tests_python_comparison GPU vs CPU (test_heterodyne_comparison.py)

Файл: `Python_test/heterodyne/test_heterodyne_comparison.py`

| # | Тест | Описание |
|---|------|----------|
| 1 | test_fbeat_gpu_vs_cpu | \f$ f_{\text{beat}} \f$: GPU == CPU (scipy.fft) |
| 2 | test_range_gpu_vs_cpu | Дальность: GPU == CPU (atol=1 м) |
| 3 | test_snr_gpu_vs_cpu | SNR: GPU == CPU (atol=0.5 дБ) |

@subsection het_tests_python_step Step-by-step debug (test_heterodyne_step_by_step.py)

Файл: `Python_test/heterodyne/test_heterodyne_step_by_step.py`

| # | Тест | Описание |
|---|------|----------|
| 1 | test_step1_rx_signals | Визуализация принятых ЛЧМ по антеннам |
| 2 | test_step2_ref_conjugate | Комплексно-сопряжённый опорный сигнал |
| 3 | test_step3_dechirp | Beat-сигналы после dechirp multiply |
| 4 | test_step4_fft_spectrum | FFT спектр: пики на beat-частотах |
| 5 | test_step5_find_peaks | Определение \f$ k_{\text{peak}} \f$ и \f$ f_{\text{beat}} \f$ |
| 6 | test_step8_summary | Итоговый отчёт: дальность + SNR по антеннам |

@subsection het_tests_python_example Пример Python теста

@code{.py}
# test_heterodyne_comparison.py -- test_range_gpu_vs_cpu
def test_range_gpu_vs_cpu(self):
    """Сравнение дальности: GPU stretch-processing vs CPU scipy.fft."""
    import scipy.fft

    # Параметры ЛЧМ
    fs, f_start, f_end = 12e6, 1e6, 3e6
    B = f_end - f_start
    n_pts = 8000
    T = n_pts / fs
    delays_us = [50, 100, 150, 200, 250]  # 5 антенн

    rx = generate_lfm_echo(fs, f_start, f_end, n_pts, delays_us)

    # CPU reference (scipy)
    cpu_ranges = []
    for a in range(5):
        ref = generate_lfm_ref(fs, f_start, f_end, n_pts)
        beat = rx[a] * np.conj(ref)
        spectrum = scipy.fft.fft(beat)
        k_peak = np.argmax(np.abs(spectrum[:n_pts//2]))
        f_beat = k_peak * fs / n_pts
        R = 3e8 * T * f_beat / (2 * B)
        cpu_ranges.append(R)

    # GPU
    het = gw.HeterodyneDechirp(self.ctx, fs=fs, f_start=f_start,
                                f_end=f_end, n_ant=5, n_pts=n_pts)
    result = het.process(rx.astype(np.complex64))

    for a in range(5):
        np.testing.assert_allclose(
            result[a].range_m, cpu_ranges[a],
            atol=1.0,  # 1 метр допуск
            err_msg=f"Antenna {a}: range mismatch"
        )
@endcode

---

@section het_tests_benchmarks Бенчмарки

@subsection het_tests_bench_opencl OpenCL бенчмарки

| Класс | Стадии профилирования |
|-------|----------------------|
| `HeterodyneDechirpBenchmark` | Upload_Rx, Upload_Ref, Multiply (dechirp_multiply kernel), Download |
| `HeterodyneCorrectBenchmark` | Upload_DC, PhaseStep (compute), Correct (dechirp_correct kernel), Download |

@subsection het_tests_bench_rocm ROCm бенчмарки

| Класс | Стадии профилирования |
|-------|----------------------|
| `HeterodyneDechirpBenchmarkROCm` | Upload_Rx, Upload_Ref, Multiply, Download (HIP event timing) |
| `HeterodyneCorrectBenchmarkROCm` | Upload_DC, PhaseStep, Correct, Download (HIP event timing) |

@subsection het_tests_bench_example Запуск бенчмарка

@code{.cpp}
// Из all_test.hpp:
#include "modules/heterodyne/tests/test_heterodyne_benchmark.hpp"
#include "modules/heterodyne/tests/test_heterodyne_benchmark_rocm.hpp"

// В RunAllTests():
test_heterodyne_benchmark::RunBenchmarks(drv);
test_heterodyne_benchmark_rocm::RunBenchmarks(drv);
@endcode

Результат: `GPUProfiler::ExportMarkdown()` -> `Results/Profiler/heterodyne_benchmark.md`.

---

@section het_tests_plots Графики

@subsection het_tests_plot_rx Step 1: Принятые ЛЧМ сигналы

@image html heterodyne/step_01_rx_signals.png "Step 1: Принятые ЛЧМ сигналы (5 антенн) -- Re/Im компоненты" width=700px

5 антенн с различными задержками. Видно смещение фазы ЛЧМ между антеннами.

@subsection het_tests_plot_dechirp Step 3: Dechirp -- Beat-сигналы

@image html heterodyne/step_03_dechirp.png "Step 3: Dechirp -- тональные beat-сигналы после умножения на conj(ref)" width=700px

После dechirp multiply ЛЧМ-сигнал превращается в чистый тон.
Частота тона пропорциональна задержке (дальности).

@subsection het_tests_plot_fft Step 4: FFT спектр

@image html heterodyne/step_04_fft_spectrum.png "Step 4: FFT спектр -- пики на beat-частотах для каждой антенны" width=700px

Чёткие пики в спектре на beat-частотах. Расстояние между пиками
соответствует разнице задержек между антеннами.

@subsection het_tests_plot_comparison GPU vs CPU

@image html heterodyne/comparison_gpu_vs_cpu.png "GPU vs CPU reference: f_beat, range, SNR -- совпадение результатов" width=700px

Сравнение GPU-результатов (OpenCL/ROCm) с CPU-эталоном (scipy.fft):
- f_beat: точное совпадение (с точностью до 1 FFT-бина)
- range: погрешность < 1 м
- SNR: погрешность < 0.5 дБ

@subsection het_tests_plot_summary Step 8: Итоговый отчёт

@image html heterodyne/step_08_summary.png "Итоговый отчёт: дальность (м) и SNR (дБ) по антеннам" width=700px

Финальная сводка: дальности линейно растут с номером антенны (задержки заданы линейно),
SNR стабилен (~30-40 дБ) для всех каналов.

---

@section het_tests_tolerances Допустимые погрешности

| Метрика | Абсолютная | Относительная | Примечание |
|---------|------------|---------------|------------|
| \f$ f_{\text{beat}} \f$ | \f$ \Delta f = f_s / N \f$ (1 FFT-бин) | -- | Определяется разрешением FFT |
| range | 1 м | -- | \f$ \Delta R = c / (2B) \f$ |
| SNR | 0.5 дБ | -- | Зависит от оценки шумового пола |

@note Разрешение по дальности можно улучшить, увеличив полосу \f$ B \f$
или используя нулевое дополнение (zero-padding) FFT.

---

@section het_tests_see_also Смотрите также

- @ref heterodyne_overview -- Обзор модуля, классы и быстрый старт
- @ref heterodyne_formulas -- Математические формулы (dechirp, дальность, SNR)
