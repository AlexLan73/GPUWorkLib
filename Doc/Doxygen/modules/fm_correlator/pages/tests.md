@page fm_correlator_tests fm_correlator — Тесты и бенчмарки

@tableofcontents

@section fm_correlator_tests_overview Обзор тестирования

Модуль `fm_correlator` покрыт тестами на уровне M-последовательности (LFSR),
базового pipeline корреляции, а также параметрическими бенчмарками GPU.

@section fm_correlator_tests_cpp C++ тесты

Расположение: `modules/fm_correlator/tests/`

@subsection fm_correlator_tests_cpp_mseq test_fm_msequence.hpp — Тесты M-последовательности

Проверка корректности генерации LFSR с полиномом `0x00400007`:

| Тест | Описание |
|------|----------|
| `ValueRange` | Все значения строго +1.0 или -1.0 |
| `Balance` | Баланс ~50% (+1) / ~50% (-1) |
| `Reproducibility` | Одинаковый seed → одинаковая последовательность |

@subsection fm_correlator_tests_cpp_basic test_fm_basic.hpp — Базовые тесты pipeline

| Тест | Описание |
|------|----------|
| `Autocorrelation` | Автокорреляция M-seq: SNR > 10 (пик при нулевом сдвиге) |
| `Pipeline` | Полный цикл: SetParams → Process → проверка результата |
| `ShiftPattern` | Корреляция с циклически сдвинутой копией — пик на правильном лаге |
| `FullSize` | Большой размер N=32768, K=16, S=8 — проверка масштабируемости |

@section fm_correlator_tests_python Python тесты

Расположение: `Python_test/fm_correlator/`

@subsection fm_correlator_tests_python_cpu test_fm_correlator.py — CPU эталон

4 теста с NumPy реализацией для валидации:

| Тест | Описание |
|------|----------|
| `LFSRGeneration` | CPU LFSR: проверка полинома, баланс |
| `NumpyCorrelation` | NumPy FFT корреляция — эталон |
| `CyclicShift` | Проверка циклического сдвига np.roll |
| `PeakDetection` | Обнаружение пиков в корреляционной функции |

Пример запуска:

@code{.py}
python Python_test/fm_correlator/test_fm_correlator.py
@endcode

@subsection fm_correlator_tests_python_rocm test_fm_correlator_rocm.py — GPU тесты

4 теста на ROCm (требуется AMD GPU + Linux):

| Тест | Описание |
|------|----------|
| `BasicPipeline` | GPU pipeline: SetParams → Process |
| `ShiftPattern` | Циклический сдвиг — пик на GPU |
| `MultiSignal` | Несколько сигналов одновременно |
| `LargeFFT` | Масштабируемость: N=32768 |

Пример запуска:

@code{.py}
python Python_test/fm_correlator/test_fm_correlator_rocm.py
@endcode

@section fm_correlator_tests_benchmarks Бенчмарки

Расположение: `modules/fm_correlator/tests/`

@subsection fm_correlator_tests_bench_alltime test_fm_benchmark_rocm_all_time.hpp — Параметрический бенчмарк

Замер полного времени pipeline для комбинаций параметров:

| Параметр | Значения |
|----------|----------|
| `fft_size` (N) | 1024, 2048, 4096, 8192, 16384, 32768 |
| `num_shifts` (K) | 4, 8, 16, 32 |
| `num_signals` (S) | 1, 2, 4, 8 |

Результат: таблица (N, K, S) → total_time_ms.

@subsection fm_correlator_tests_bench_steps test_fm_step_profiling.hpp — Per-kernel профилирование

Замер каждого этапа pipeline отдельно:

| Стадия | Kernel / Операция |
|--------|-------------------|
| H2D | Копирование данных Host → Device |
| R2C FFT | hipFFT forward (reference + signals) |
| Multiply | `multiply_conj_fused` kernel |
| C2R IFFT | hipFFT inverse |
| Extract | `extract_magnitudes_real` kernel |
| D2H | Копирование результатов Device → Host |

@subsection fm_correlator_tests_bench_avg test_fm_avg_summary.hpp — Агрегированная статистика

Сводка avg / min / max по всем запускам:

| Метрика | Описание |
|---------|----------|
| `avg_ms` | Среднее время (мс) |
| `min_ms` | Минимальное время (мс) |
| `max_ms` | Максимальное время (мс) |
| `std_ms` | Стандартное отклонение |

@section fm_correlator_tests_kernels Тестирование ядер

Каждый HIP kernel тестируется отдельно:

| Kernel | Тест |
|--------|------|
| `apply_cyclic_shifts` | Проверка циклического сдвига + float → float2 конвертация |
| `multiply_conj_fused` | Сравнение с NumPy `np.conj(R) * I` |
| `extract_magnitudes_real` | Bitwise abs: проверка знаковых и беззнаковых значений |
| `generate_test_inputs` | GPU circshift pattern: проверка корректности сдвигов |

@see fm_correlator_overview
@see fm_correlator_formulas
