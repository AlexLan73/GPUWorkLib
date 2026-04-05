# GPUWorkLib — Архитектурный анализ проекта

> **Версия**: 1.1.0 | **Дата анализа**: 2026-04-05
> **Инструмент**: Claude Code (Explore agent) + repomix

---

## 1. Общая статистика

| Тип файла | Кол-во | LOC |
|-----------|--------|-----|
| Заголовки (*.hpp, *.h) | 335 | 56,395 |
| Исходники (*.cpp) | 66 | 22,797 |
| HIP kernels (*.hip) | 7 | 414 |
| OpenCL kernels (*.cl) | 19 | 2,538 |
| Python (*.py) | 111 | 28,033 |
| Документация (*.md) | 342 | — |
| **ИТОГО** | **880** | **110,177** |

**C++ код**: ~79,192 LOC | **Kernel код**: 2,952 LOC (26 файлов) | **Python**: ~28,033 LOC

---

## 2. Слоистая архитектура

```
┌─────────────────────────────────────┐
│   Python Bindings (pybind11)        │  19 классов → gpuworklib
├─────────────────────────────────────┤
│   Module Processors (Facade)        │  signal_generators, fft_func, ...
├─────────────────────────────────────┤
│   Operation Classes (GpuKernelOp)   │  PadDataOp, MagPhaseOp, ...
├─────────────────────────────────────┤
│   Kernel Bindings & Services        │  KernelCacheService, GPUProfiler
├─────────────────────────────────────┤
│   DrvGPU Core Infrastructure        │  GpuContext, MemoryManager
├─────────────────────────────────────┤
│   Backend Implementations           │  OpenCLBackend, ROCmBackend, HybridBackend
├─────────────────────────────────────┤
│   GPU APIs                          │  OpenCL 3.0, HIP/ROCm 7.2+
└─────────────────────────────────────┘
```

---

## 3. Модули — сводная таблица

| Модуль | Headers | Sources | .hip | .cl | Namespace | Backend |
|--------|---------|---------|------|-----|-----------|---------|
| signal_generators | 34 | 10 | 1 | 6 | `signal_gen` | OpenCL + ROCm |
| fft_func | 42 | 5 | 2 | 2 | `fft_processor`, `antenna_fft` | ROCm (hipFFT) |
| filters | 21 | 5 | 0 | 5 | `filters` | OpenCL + ROCm |
| statistics | 15 | 1 | 1 | 0 | `statistics` | ROCm only |
| heterodyne | 11 | 2 | 0 | 1 | `drv_gpu_lib` * | OpenCL + ROCm |
| vector_algebra | 13 | 4 | 0 | 0 | `vector_algebra` | ROCm (rocSOLVER) |
| capon | 16 | 1 | 0 | 1 | `capon` | ROCm only |
| range_angle | 12 | 1 | 3 | 0 | `range_angle` | ROCm only |
| fm_correlator | 11 | 2 | 0 | 1 | `drv_gpu_lib` * | ROCm only |
| strategies | 38 | 2 | 0 | 1 | `strategies` | ROCm only |
| lch_farrow | 7 | 1 | 0 | 2 | `lch_farrow` | OpenCL + ROCm |
| test_utils | 12 | 0 | 0 | 0 | — | Support |
| **ИТОГО** | **243** | **34** | **7** | **19** | — | — |

> \* `drv_gpu_lib` — legacy namespace. Тесты уже используют `heterodyne::tests` / `fm_correlator::tests`. Миграция production-кода на модульные namespaces — в планах.

---

## 4. DrvGPU — ядро библиотеки

### 4.1 IBackend Interface (287 LOC)

```cpp
class IBackend {
  // Lifecycle
  void Initialize(int device_index);
  void Cleanup();
  void SetOwnsResources(bool owns);

  // Device Info
  BackendType GetType() const;           // OpenCL, ROCm, CUDA, Hybrid
  GPUDeviceInfo GetDeviceInfo() const;
  std::string GetDeviceName() const;

  // Native Handles
  void* GetNativeContext() const;        // cl_context / hipCtx_t
  void* GetNativeDevice() const;         // cl_device_id / hipDevice_t
  void* GetNativeQueue() const;          // cl_command_queue / hipStream_t

  // Memory
  void* Allocate(size_t bytes, unsigned flags);
  void* AllocateManaged(size_t bytes);   // hipMallocManaged
  void Free(void* ptr);
  void MemcpyHostToDevice(void* dst, const void* src, size_t bytes);
  void MemcpyDeviceToHost(void* dst, const void* src, size_t bytes);
  void MemcpyDeviceToDevice(void* dst, const void* src, size_t bytes);

  // Sync
  void Synchronize();
  void Flush();

  // Capabilities
  bool SupportsSVM() const;
  bool SupportsDoublePrecision() const;
  size_t GetMaxWorkGroupSize() const;
  size_t GetGlobalMemorySize() const;
  size_t GetLocalMemorySize() const;
};
```

### 4.2 Backends (4 реализации)

| Backend | Описание | FFT | BLAS |
|---------|----------|-----|------|
| `OpenCLBackend` | OpenCL 1.2+ | clFFT | — |
| `ROCmBackend` | AMD ROCm/HIP | rocFFT | rocBLAS + rocSOLVER |
| `HybridBackend` | OpenCL + ROCm | оба | rocBLAS |
| `ZeroCopyBridge` | Мост между backends | — | — |

### 4.3 Services (15 классов)

| Сервис | Назначение |
|--------|-----------|
| `GPUProfiler` | Профилирование GPU (async) |
| `ConsoleOutput` | Multi-GPU safe консоль |
| `KernelCacheService` | Кеш скомпилированных kernels |
| `ServiceManager` | Реестр сервисов |
| `BatchManager` | Пакетная обработка |
| `GpuBenchmarkBase` | Базовый класс бенчмарков |
| `GpuKernelOp` | Базовый класс GPU-операций |
| `BufferSet` | Управление наборами буферов |
| `AsyncServiceBase` | Async task queue |
| `MemoryManager` | Управление GPU памятью |
| `StreamPool` | Пул HIP streams |
| `CommandQueuePool` | Пул OpenCL queues |
| `ExternalCLBufferAdapter` | Адаптер внешних буферов |
| `HIPBuffer` | HIP buffer wrapper |
| `SVMBuffer` | Shared Virtual Memory |

---

## 5. Граф зависимостей между модулями

```
                           ┌──────────┐
                           │  DrvGPU  │ ← все модули зависят
                           └─────┬────┘
              ┌──────────────────┼──────────────────────────┐
              │                  │                          │
        ┌─────┴──────┐    ┌─────┴──────┐    ┌──────────────┴──────────┐
        │ fft_func   │    │ statistics │    │  vector_algebra         │
        │ (hipFFT)   │    │ (Welford)  │    │  (rocBLAS + rocSOLVER)  │
        └─────┬──────┘    └─────┬──────┘    └──────────┬──────────────┘
              │                 │                       │
              ├─────────────────┤                       │
              ▼                 ▼                       ▼
        ┌─────────────────────────┐              ┌─────────┐
        │      strategies         │              │  capon   │
        │ (fft + statistics +     │              │ (MVDR)   │
        │  CGEMM pipeline)        │              └─────────┘
        └─────────────────────────┘

  ┌──────────────┐    ┌─────────────┐
  │ lch_farrow   │◄───│ signal_gen  │ (DelayedFormSignal uses LchFarrow)
  └──────────────┘    └──────┬──────┘
                             │
                  ┌──────────┘
                  │  signal_gen + fft_func
                  ▼
            ┌─────────────┐
            │  heterodyne  │ (LFM Dechirp pipeline)
            └─────────────┘

  Используют hipFFT напрямую (без fft_func target):
    range_angle   — hipFFT (Range FFT + Beam FFT)
    fm_correlator — hipFFT (R2C/C2R correlation)

  Независимые: filters, lch_farrow
  (зависят только от DrvGPU)
```

**Циклические зависимости**: НЕ ОБНАРУЖЕНЫ ✅

---

## 6. Kernel Inventory (26 файлов, 2,952 LOC)

| Модуль | Файл | Kernels | LOC |
|--------|------|---------|-----|
| **signal_generators** | CwGen_kernels.cl | `generate_cw`, `generate_cw_real` | 68 |
| | FormSignal_kernels.cl | `generate_form_signal` | 155 |
| | LfmConj_kernels.cl | `generate_lfm`, `generate_lfm_conjugate` | 133 |
| | LfmDelay_kernels.cl | `generate_lfm` (delayed) | 133 |
| | LfmGen_kernels.cl | `generate_lfm`, `generate_lfm_real` | 133 |
| | NoiseGen_kernels.cl | `generate_noise_gaussian`, `generate_noise_white` | 101 |
| | form_signal.hip | `form_signal_kernel` | 49 |
| **fft_func** | C2MP_kernels.cl | `complex_to_mag_phase`, `complex_to_magnitude` | 41 |
| | SpectrumMaxima_kernels.cl | `pad_data`, `compute_magnitudes` | 483 |
| | c2mp_kernels.hip | `complex_to_mag_phase` | 32 |
| | fft_processor_kernels.hip | `pad_data`, `complex_to_mag_phase` | 47 |
| **filters** | FIR_kernels.cl | `fir_filter_cf32` | 60 |
| | IIR_kernels.cl | `iir_biquad_cascade_cf32` | 77 |
| | Kalman_kernels.cl | `kalman_kernel` | 58 |
| | KAMA_kernels.cl | `kaufman_kernel` | 103 |
| | SMA_kernels.cl | `sma_kernel`, `ema_kernel`, `mma_kernel` | 195 |
| **heterodyne** | Heterodyne_kernels.cl | `dechirp_multiply`, `dechirp_correct` | 86 |
| **lch_farrow** | LchFarrow_kernels.cl | `lch_farrow_delay` | 145 |
| | LchFarrow_kernels_00.cl | `lch_farrow_delay` (variant) | 145 |
| **range_angle** | dechirp_window_kernel.hip | `dechirp_window_kernel`, `gen_ref_kernel` | 85 |
| | fftshift2d_kernel.hip | `fftshift2d_kernel`, `magnitude_sq_kernel` | 62 |
| | transpose_kernel.hip | `transpose_complex_kernel` | 47 |
| **fm_correlator** | FM_Corr_kernels.cl | `apply_cyclic_shifts`, `multiply_conj_fused` | 109 |
| **statistics** | statistics_sort_gpu.hip | (radix sort) | 92 |
| **strategies** | Strategies_kernels.cl | `hamming_pad_fused`, `compute_magnitudes` | 271 |
| **capon** | Capon_kernels.cl | `compute_capon_relief` | 42 |

---

## 7. Python Bindings (pybind11)

### 7.1 Реализовано (9 классов в gpu_worklib_bindings.cpp)

| Категория | Класс | Модуль Python |
|-----------|-------|---------------|
| **Контекст** | GPUContext | `gw.GPUContext` |
| | ROCmGPUContext | `gw.ROCmGPUContext` |
| | HybridGPUContext | `gw.HybridGPUContext` |
| **Буферы** | GPUBuffer (PyGPUBuffer) | `gw.GPUBuffer` |
| **Генераторы** | SignalGenerator (PySignalGenerator) | `gw.SignalGenerator` |
| | ScriptGenerator (PyScriptGenerator) | `gw.ScriptGenerator` |
| | FormSignalGenerator (PyFormSignalGenerator) | `gw.FormSignalGenerator` |
| | FormScriptGenerator (PyFormScriptGenerator) | `gw.FormScriptGenerator` |
| | DelayedFormSignalGenerator | `gw.DelayedFormSignalGenerator` |

### 7.2 Планируется (ещё не в gpu_worklib_bindings.cpp)

| Категория | Класс | Статус |
|-----------|-------|--------|
| **FFT** | FFTProcessorROCm, ComplexToMagROCm, SpectrumMaximaFinderROCm | ⚪ Planned |
| **Фильтры** | FirFilterROCm, IirFilterROCm, MovingAverageFilterROCm | ⚪ Planned |
| | KalmanFilterROCm, KaufmanFilterROCm | ⚪ Planned |
| **Обработка** | HeterodyneROCm, LchFarrowROCm, FMCorrelatorROCm | ⚪ Planned |
| **Алгебра** | StatisticsProcessor, CholeskyInverterROCm | ⚪ Planned |
| **Pipeline** | AntennaProcessorTest, RangeAngleProcessor | ⚪ Planned |

---

## 8. Паттерны проектирования

| Паттерн | Где используется |
|---------|-----------------|
| **Bridge** | IBackend → OpenCL/ROCm/Hybrid |
| **Factory** | SignalGeneratorFactory, SpectrumProcessorFactory |
| **Strategy** | IPipelineStep, IPostFftScenario, IMatrixRegularizer |
| **Pipeline** | Pipeline + PipelineBuilder (strategies) |
| **Builder** | PipelineBuilder |
| **Template Method** | GpuBenchmarkBase, AsyncServiceBase |
| **Service Locator** | ServiceManager, ModuleRegistry |
| **Facade** | DrvGPU, CaponProcessor, HeterodyneDechirp |
| **Operation** | GpuKernelOp → все *Op классы |

---

## 9. Сборка (CMake 3.20+)

**Порядок сборки модулей**:
```
1.  DrvGPU              (core — обязательный)
2.  fft_func
3.  lch_farrow
4.  signal_generators   (зависит от lch_farrow)
5.  filters
6.  heterodyne
7.  statistics
8.  vector_algebra
9.  fm_correlator
10. strategies          (зависит от fft_func + statistics)
11. capon               (зависит от vector_algebra)
12. range_angle
13. src/main            (exe)
14. python/             (optional, pybind11)
```

**Стек технологий**:

| Компонент | Технология |
|-----------|-----------|
| Язык | C++17 |
| Сборка | CMake 3.20+ |
| GPU API | OpenCL 3.0, HIP/ROCm 7.2+ |
| Линейная алгебра | rocBLAS, rocSOLVER |
| FFT | clFFT (OpenCL), rocFFT (ROCm) |
| Python | pybind11 |
| Профилирование | GPUProfiler (custom) |
| Логирование | plog (per-GPU) |

---

## 10. Ключевые характеристики

1. **Multi-Backend**: OpenCL + ROCm через IBackend абстракцию
2. **ROCm-Optimized**: Kernels оптимизированы под AMD RDNA4+
3. **Модульная**: 11 независимых модулей + DrvGPU core
4. **Pipeline**: strategies композирует FFT + statistics + kernels
5. **Python-First API**: 19 классов через pybind11
6. **Zero-Copy**: ZeroCopyBridge, SVM буферы
7. **Kernel Caching**: KernelCacheService (compile-once)
8. **Batch Processing**: BatchManager для throughput
9. **Нет циклических зависимостей** — чистый DAG

---

*Сгенерировано: 2026-04-05 | Обновлено: 2026-04-05 (code review) | GPUWorkLib v1.1.0 | 880 файлов, 110,177 LOC*
