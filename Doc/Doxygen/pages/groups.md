/**

@defgroup infrastructure 0. Инфраструктура
@brief Ядро GPU и базовая обработка — фундамент для всех модулей.

Модули этого уровня не зависят друг от друга и предоставляют `IBackend*` + FFT всем остальным.

@see @ref drvgpu_main "DrvGPU подробно"
@see @ref fft_func_overview "fft_func подробно"

*/

/**
@defgroup grp_drvgpu DrvGPU — Ядро GPU
@ingroup infrastructure
@brief Единая абстракция: OpenCL 3.0, ROCm/HIP, Hybrid + ZeroCopy.

**Namespace**: `drv_gpu_lib`

Все модули получают `IBackend*` от DrvGPU.
Три backend'а, MemoryManager, GPUProfiler, ConsoleOutput, ServiceManager.

Ключевые классы: DrvGPU, GPUManager, IBackend, OpenCLBackend, ROCmBackend, HybridBackend, GPUProfiler

@see @ref drvgpu_main "Полная документация DrvGPU"
@see @ref drvgpu_architecture "Архитектура DrvGPU"
@see @ref drvgpu_profiler "GPUProfiler API"
*/

/**
@defgroup grp_fft_func fft_func — FFT + Spectrum Maxima
@ingroup infrastructure
@brief Пакетный 1D FFT (hipFFT) + поиск максимумов спектра.

**Namespace**: `fft_processor`, `antenna_fft`

Объединяет FFTProcessor + SpectrumMaximaFinder.
Режимы: Complex, MagPhase, MagPhaseFreq. Пики: ONE_PEAK, TWO_PEAKS, ALL_MAXIMA.

@see @ref fft_func_overview "Обзор fft_func"
@see @ref fft_func_formulas "Математика FFT"
@see @ref fft_func_tests "Тесты fft_func"
*/

/**

@defgroup basic_modules 1. Базовые модули
@brief Статистика, матричная алгебра, фильтры, генераторы — строительные блоки pipeline.

@see @ref statistics_overview
@see @ref vector_algebra_overview
@see @ref filters_overview
@see @ref signal_generators_overview

*/

/**
@defgroup grp_statistics statistics — GPU статистика
@ingroup basic_modules
@brief Mean, Std, Variance (Welford one-pass) + Median (Radix Sort / Histogram).

**Namespace**: `statistics`

Per-beam статистика на комплексных данных. Kernel `welford_fused` — один проход.

@see @ref statistics_overview "Обзор statistics"
@see @ref statistics_formulas "Формулы statistics"
*/

/**
@defgroup grp_vector_algebra vector_algebra — Матричная алгебра
@ingroup basic_modules
@brief Cholesky inversion (rocsolver POTRF+POTRI) для HPD матриц.

**Namespace**: `vector_algebra`

Входы: CPU vector, ROCm ptr, cl_mem (ZeroCopy). Два режима symmetrize.

@see @ref vector_algebra_overview "Обзор vector_algebra"
@see @ref vector_algebra_formulas "Формулы vector_algebra"
*/

/**
@defgroup grp_filters filters — GPU фильтры
@ingroup basic_modules
@brief FIR, IIR (Biquad DFII-T), MovingAverage (SMA/EMA/DEMA/TEMA), Kalman, KAMA.

**Namespace**: `filters`

5 типов фильтров для комплексных многоканальных сигналов (float2).

@see @ref filters_overview "Обзор filters"
@see @ref filters_formulas "Формулы filters"
*/

/**
@defgroup grp_signal_generators signal_generators — Генераторы сигналов
@ingroup basic_modules
@brief CW, LFM, Noise (Philox+Box-Muller), FormSignal, DelayedFormSignal, LfmAnalyticalDelay.

**Namespace**: `signal_gen`

8 типов генераторов. Script-генератор компилирует DSL в OpenCL kernel.

@see @ref signal_generators_overview "Обзор signal_generators"
@see @ref signal_generators_formulas "Формулы signal_generators"
*/

/**

@defgroup signal_processing 2. Обработка сигналов
@brief Дробная задержка, дечирп, FM-корреляция — модули обработки.

@see @ref lch_farrow_overview
@see @ref heterodyne_overview
@see @ref fm_correlator_overview

*/

/**
@defgroup grp_lch_farrow lch_farrow — Дробная задержка Farrow
@ingroup signal_processing
@brief Lagrange 48×5 интерполяция для sub-sample fractional delay.

**Namespace**: `lch_farrow`

5-point Lagrange с предвычисленной матрицей. 2D kernel (sample × antenna).

@see @ref lch_farrow_overview "Обзор lch_farrow"
@see @ref lch_farrow_formulas "Формулы lch_farrow"
*/

/**
@defgroup grp_heterodyne heterodyne — LFM Dechirp
@ingroup signal_processing
@brief Stretch-processing ЛЧМ: conj(rx × ref*) → beat frequency → range (м), SNR (дБ).

**Namespace**: `drv_gpu_lib` (heterodyne)

Pipeline: генерация conj ref → multiply → FFT → find peak → range.

@see @ref heterodyne_overview "Обзор heterodyne"
@see @ref heterodyne_formulas "Формулы heterodyne"
*/

/**
@defgroup grp_fm_correlator fm_correlator — FM-корреляция
@ingroup signal_processing
@brief M-sequence LFSR + cyclic shifts + freq-domain correlation (hipFFT R2C/C2R).

**Namespace**: `drv_gpu_lib` (fm_correlator)

Пассивная бистатическая РЛС, многогипотезный поиск задержек.

@see @ref fm_correlator_overview "Обзор fm_correlator"
@see @ref fm_correlator_formulas "Формулы fm_correlator"
*/

/**

@defgroup pipelines 3. Pipeline высокого уровня
@brief Цифровое формирование ДН, MVDR Capon, 3D дальность-угол.

@see @ref strategies_overview
@see @ref capon_overview
@see @ref range_angle_overview

*/

/**
@defgroup grp_strategies strategies — Цифровое ДН
@ingroup pipelines
@brief CGEMM beamforming → Hamming+FFT → post-FFT scenarios (OneMax/AllMaxima/MinMax).

**Namespace**: `strategies`

7-step pipeline, 4 HIP streams, hipBLAS CGEMM. 3 checkpoint-а статистики.

@see @ref strategies_overview "Обзор strategies"
@see @ref strategies_formulas "Формулы strategies"
*/

/**
@defgroup grp_capon capon — MVDR Beamformer
@ingroup pipelines
@brief R=YY^H/N+μI → Cholesky → R^{-1} → relief z[m] / adaptive beamform.

**Namespace**: `capon`

Адаптивное подавление помех. rocBLAS CGEMM + rocsolver POTRF/POTRI.

@see @ref capon_overview "Обзор capon"
@see @ref capon_formulas "Формулы capon"
*/

/**
@defgroup grp_range_angle range_angle — 3D Range-Angle
@ingroup pipelines
@brief Dechirp → Range FFT → 2D Beam FFT → Peak Search → TargetInfo.

**Namespace**: `range_angle`

3D обработка для 2D антенной решётки (16×16). Status: Beta (stubs).

@see @ref range_angle_overview "Обзор range_angle"
@see @ref range_angle_formulas "Формулы range_angle"
*/

/**

@defgroup architecture 4. Архитектура (C4 Model)
@brief C1 System Context, C2 Container, C3 Component, C4 Code, DFD, Sequence Diagrams.

Полная архитектурная документация GPUWorkLib по модели C4.
PlantUML диаграммы + ASCII-art для каждого уровня.

@see @ref architecture_page "Архитектура проекта"

*/

/**

@defgroup testing 5. Тесты и бенчмарки
@brief C++ тесты (*.hpp), Python тесты, GpuBenchmarkBase бенчмарки.

Все C++ тесты: header-only в `modules/*/tests/`, вызываются из `src/main.cpp`.
Python тесты: `Python_test/*/test_*.py` (НЕ pytest, только TestRunner).
Бенчмарки: наследники GpuBenchmarkBase, GPUProfiler → Markdown/JSON.

@see @ref tests_overview_page "Сводка тестов всех модулей"

*/
