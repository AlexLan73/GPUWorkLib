@page fft_func_tests FFT Functions — Тесты и бенчмарки

@tableofcontents

@section fft_tests_cpp C++ тесты

Расположение: `modules/fft_func/tests/`

| Файл | Описание |
|------|----------|
| `test_fft_processor_rocm.hpp` | 5 тестов: single_beam, multi_beam, mag_phase, freq, gpu_input |
| `test_complex_to_mag_phase_rocm.hpp` | ComplexToMagPhaseROCm: IQ → mag+phase без FFT |
| `test_process_magnitude_rocm.hpp` | Magnitude-only output, normalization |
| `test_fft_matrix_rocm.hpp` | Матрица beams × nFFT benchmark |
| `test_spectrum_maxima_rocm.hpp` | SpectrumProcessorROCm: ONE_PEAK, TWO_PEAKS |
| `test_helpers_rocm.hpp` | Утилиты генерации тестовых данных |

@section fft_tests_python Python тесты

Расположение: `Python_test/fft_func/`

| Файл | Описание |
|------|----------|
| `test_process_magnitude_rocm.py` | ProcessMagnitude: CW → FFT → magnitude → verify peak |
| `test_spectrum_find_all_maxima_rocm.py` | FindAllMaxima: multi-peak, scipy.signal.find_peaks reference |
| `test_spectrum_maxima_finder_rocm.py` | E2E: NumPy reference (8) + GPU (6) тестов |

@section fft_tests_benchmarks Бенчмарки

| Класс | Метрики |
|-------|---------|
| `FFTProcessorBenchmarkROCm` | Upload, PadData, hipFFT, MagPhase, Download |
| `FFTMaximaBenchmarkROCm` | Upload, FindPeaks, Compact, Download |

Результаты: `Results/Profiler/GPU_00_FFT*/`

@section fft_tests_plots Графики

@subsection fft_plots_single Single tone detection
@image html fft_maxima/test1_single_tone.png "ONE_PEAK: обнаружение одного тона" width=700px

@subsection fft_plots_three Three tones
@image html fft_maxima/test2_three_tones.png "Три тона в спектре" width=700px

@subsection fft_plots_scipy GPU vs SciPy find_peaks
@image html fft_maxima/test4_gpu_vs_scipy.png "ALL_MAXIMA: GPU vs scipy.signal.find_peaks" width=700px

@subsection fft_plots_multibeam FindAllMaxima — multi-beam
@image html fft_maxima/find_all_maxima_multi_beam.png "FindAllMaxima: multi-beam detection" width=700px

@section fft_tests_seealso См. также

- @ref fft_func_overview — Обзор модуля
- @ref fft_func_formulas — Математика FFT
