@page capon_tests capon — Тесты и бенчмарки

@tableofcontents

@section capon_tests_overview Обзор тестирования

Модуль `capon` покрыт обширным набором тестов: от базовой проверки Relief/Beamform
до валидации на реальных MATLAB-данных заказчика и тестов GPU interop (OpenCL <-> ROCm).

@section capon_tests_cpp C++ тесты

Расположение: `modules/capon/tests/`

@subsection capon_tests_cpp_rocm test_capon_rocm.hpp — Базовые ROCm тесты

5 тестов основной функциональности:

| Тест | Описание |
|------|----------|
| `ReliefNoise` | Relief на шумовом сигнале — проверка формы спектра |
| `ReliefInterference` | Relief с помехой — проверка подавления |
| `AdaptiveBeamform` | Полный цикл adaptive beamforming |
| `Regularization` | Влияние \f$ \mu \f$ на стабильность обращения |
| `GPUtoGPU` | Данные уже на GPU (zero-copy path) |

@subsection capon_tests_cpp_reference test_capon_reference_data.hpp — Валидация по MATLAB

3 теста на эталонных данных заказчика (P=85 каналов, N=1000 отсчётов, M=1369 направлений):

| Тест | Описание |
|------|----------|
| `MATLABDataLoad` | Загрузка signal_matlab.txt, x_data.txt, y_data.txt |
| `MATLABRelief` | Сравнение relief с MATLAB-эталоном |
| `PhysicalProperties` | Проверка физических свойств (пики в направлениях источников) |

@note Тестовые данные: `modules/capon/tests/data/` — файлы от заказчика в формате MATLAB.

@subsection capon_tests_cpp_interop test_capon_opencl_to_rocm.hpp — OpenCL-ROCm Interop

5 тестов межбэкендного взаимодействия:

| Тест | Описание |
|------|----------|
| `ZeroCopyInterop` | ZeroCopy передача OpenCL -> ROCm |
| `CustomerPipeline` | Полный pipeline данных заказчика через interop |
| `BufferMapping` | Проверка маппинга буферов между бэкендами |
| `StreamSync` | Синхронизация потоков OpenCL/ROCm |
| `MultiGPU` | Interop на нескольких GPU |

@subsection capon_tests_cpp_hip test_capon_hip_opencl_to_rocm.hpp — HIP+OpenCL Interop

3 теста с hipMalloc и OpenCL совместной работой:

| Тест | Описание |
|------|----------|
| `HipMallocOpenCLWrite` | hipMalloc буфер + OpenCL запись |
| `SVMPath` | Shared Virtual Memory путь |
| `MixedAllocation` | Смешанное выделение памяти |

@section capon_tests_python Python тесты

Расположение: `Python_test/capon/`

@subsection capon_tests_python_main test_capon.py

3 набора тестов (всего 16 тестов):

| Класс | Кол-во | Описание |
|-------|--------|----------|
| `TestCaponReference` | 8 | Эталонная проверка: NumPy MVDR vs GPU |
| `TestCaponGPU` | 2 | GPU pipeline: ComputeRelief, AdaptiveBeamform |
| `TestCaponRealData` | 6 | Реальные данные: загрузка MATLAB, сравнение спектров |

Пример запуска:

@code{.py}
python Python_test/capon/test_capon.py
@endcode

@section capon_tests_benchmarks Бенчмарки

@subsection capon_tests_bench_relief CaponReliefBenchmarkROCm

Профилирование режима ComputeRelief. Замеряемые стадии:

| Стадия | Описание |
|--------|----------|
| `Covariance` | \f$ R = YY^H/N + \mu I \f$ (rocBLAS CGEMM) |
| `Cholesky` | POTRF + POTRI через CholeskyInverterROCm |
| `Relief` | HIP kernel: \f$ z[m] = 1/\text{Re}(u^H R^{-1} u) \f$ |

@subsection capon_tests_bench_beamform CaponBeamformBenchmarkROCm

Профилирование режима AdaptiveBeamform. Замеряемые стадии:

| Стадия | Описание |
|--------|----------|
| `Covariance` | \f$ R = YY^H/N + \mu I \f$ |
| `Cholesky` | POTRF + POTRI |
| `CGEMM_Weights` | \f$ W = R^{-1} U \f$ |
| `Beamform` | \f$ Y_{\text{out}} = W^H Y \f$ |

@section capon_tests_plots Графики

@subsection capon_tests_plots_relief Relief спектр

@image html capon/relief_spectrum.png "Пространственный спектр Capon (MVDR Relief)" width=800px

@subsection capon_tests_plots_beamform Beamforming выход

@image html capon/beamform_output.png "Выходные сигналы адаптивного формирования" width=800px

@section capon_tests_data Тестовые данные

Директория `modules/capon/tests/data/`:

| Файл | Описание |
|------|----------|
| `signal_matlab.txt` | Эталонный сигнал (MATLAB) |
| `x_data.txt` | X-координаты антенных элементов |
| `y_data.txt` | Y-координаты антенных элементов |

Параметры эталона: \f$ P = 85 \f$ каналов, \f$ N = 1000 \f$ отсчётов, \f$ M = 1369 \f$ направлений.

@see capon_overview
@see capon_formulas
