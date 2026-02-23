# План миграции на AMD Radeon 9070 (ROCm)

> **Сохранил**: Кодо | **Дата**: 2026-02-17
> **Развернём на работе** — полный план с этапами и тестами

**Рабочий план для исполнения**: [PLAN_ROCm_DrvGPU_Full.md](PLAN_ROCm_DrvGPU_Full.md) — детальные шаги, чек-лист, порядок модулей. Этот файл — общий контекст и этапы.

---

## Контекст

- **Radeon RX 9070**: RDNA 4, gfx1201, поддерживается ROCm 6.4.2+ и 7.0.2+
- **Текущее состояние**: DrvGPU имеет только OpenCL backend; ROCm — заглушки (`rocm.txt`, `SpectrumProcessorROCm`, `FormSignalGeneratorROCm`, `FirFilterROCm`, `IirFilterROCm`)
- **Важно**: ROCm для Radeon официально поддерживается только на **Linux**. На Windows — только OpenCL через AMD driver

---

## Этап 1: Установка окружения на новый компьютер

### 1.1 Операционная система

- **Linux (Ubuntu 22.04/24.04)** — для ROCm
- **Windows** — только OpenCL (если нужна разработка без ROCm)

### 1.2 Драйверы и библиотеки

| Компонент          | Назначение                        | Установка                                                                            |
| ------------------ | --------------------------------- | ------------------------------------------------------------------------------------ |
| **AMD GPU Driver** | Базовый драйвер                   | `amdgpu` (встроен в ядро Linux 5.x+)                                                 |
| **ROCm 7.x**       | HIP, hipFFT, rocPRIM, hipCUB      | [ROCm Install Guide](https://rocm.docs.amd.com/projects/install-on-linux/en/latest/) |
| **OpenCL (AMD)**   | OpenCL для AMD GPU                | Входит в ROCm или `rocm-opencl-runtime`                                              |
| **clFFT**          | FFT для OpenCL (уже используется) | Сборка из исходников или пакет                                                       |

### 1.3 Конкретные пакеты (Ubuntu/Debian)

```bash
# ROCm 7.x (проверить совместимость с gfx1201)
wget https://repo.radeon.com/rocm/rocm.gpg.key -O - | gpg --dearmor | sudo tee /etc/apt/keyrings/rocm.gpg
# Добавить repo по инструкции AMD
sudo apt install rocm hipfft rocprim hipcub rocblas
# OpenCL runtime
sudo apt install rocm-opencl-runtime
```

### 1.4 Проверка установки

- `rocminfo` — список GPU
- `hipinfo` — информация о HIP
- Тест: `hipDeviceSynchronize()` в простой программе

### 1.5 Тесты этапа 1

- Скрипт `scripts/check_rocm_env.sh`: проверка `rocminfo`, `hipinfo`, компиляция `hip::hip` через CMake
- Документ `Doc/AMD_Radeon_9070_Setup.md` с чек-листом установки

---

## Этап 2: ROCm Backend в DrvGPU + память + InputData

### 2.1 Структура ROCm backend

Создать в `DrvGPU/backends/rocm/`:

- `rocm_backend.hpp` / `rocm_backend.cpp` — реализация `IBackend`
- `rocm_core.hpp` / `rocm_core.cpp` — `hipDevice_t`, `hipCtx_t`, `hipStream_t`
- `rocm_profiling.hpp` — интеграция с `GPUProfiler` (ROCmProfilingData уже есть)

### 2.2 Ключевые методы ROCmBackend

```cpp
// rocm_backend.hpp
class ROCmBackend : public IBackend {
    void Initialize(int device_index) override;
    void* GetNativeContext() const override;  // hipCtx_t
    void* GetNativeDevice() const override;   // hipDevice_t
    void* GetNativeQueue() const override;    // hipStream_t
    void* Allocate(size_t size_bytes, unsigned int flags) override;  // hipMalloc
    void Free(void* ptr) override;            // hipFree
    void MemcpyHostToDevice(...) override;     // hipMemcpy
    void MemcpyDeviceToHost(...) override;
    void MemcpyDeviceToDevice(...) override;
    void Synchronize() override;               // hipStreamSynchronize
    bool SupportsSVM() const override;         // false для ROCm
    // ...
};
```

### 2.3 Доработка памяти и InputData

**Проблема**: `input_data.hpp` поддерживает `cl_mem`, `void*` (SVM), `vector<complex<float>>`. Для ROCm нужен `hipDeviceptr_t` (или `void*` как указатель на device memory).

**Решение**:

1. В `input_data_traits.hpp` добавить trait `is_gpu_device_ptr<T>` для `cl_mem` и `void*` (ROCm).
2. `IMemoryBuffer` — для ROCm использовать `GPUBuffer` + `IBackend::Allocate` (возвращает `void*`).
3. Файл `DrvGPU/memory/hip_buffer.hpp` (новый): обёртка `HIPBuffer` с `hipMalloc`/`hipFree`, `Write`/`Read` через `hipMemcpy`, `GetDevicePtr()`.

### 2.4 CMake

- В `cmake/gpu-config.cmake` добавить `ENABLE_ROCM` (по умолчанию ON на Linux при наличии `find_package(hip)`).
- В `DrvGPU/CMakeLists.txt` условно подключать `backends/rocm` при `ENABLE_ROCM`.

### 2.5 Тесты этапа 2

- `DrvGPU/tests/test_rocm_backend.hpp`: инициализация, Allocate/Free, Memcpy, Synchronize
- `DrvGPU/tests/test_hip_buffer.hpp`: HIPBuffer Write/Read, размеры
- `DrvGPU/tests/test_input_data_rocm.hpp`: `InputData<void*>` с данными из ROCmBackend::Allocate

---

## Этап 3: Перенос модулей

### 3.1 fft_processor (первый)

**План**:

1. Ввести абстракцию `IFFTBackend` или переключатель по `BackendType` внутри `FFTProcessor`.
2. Реализовать `FFTProcessorROCm`: hipFFT, pre-callback (padding), post-kernel (mag/phase) в HIP.
3. Kernels из `fft_processor_kernels.hpp` → `.hip` файлы

**Тесты**: `modules/fft_processor/tests/test_fft_processor_rocm.hpp`

---

### 3.2 fft_maxima

**План**:

1. Реализовать `SpectrumProcessorROCm`: hipFFT, HIP kernels (detect, scan, compact).
2. Подключить `SpectrumProcessorFactory::Create(BackendType::ROCm, backend)` в `SpectrumMaximaFinder`.
3. Рефакторинг: делегировать обработку `ISpectrumProcessor`.

**Тесты**: `modules/fft_maxima/tests/test_spectrum_maxima_rocm.hpp`

---

### 3.3 Statistics (новый модуль, ROCm only)

**Только ROCm** — в рамках миграции не портируется на OpenCL. Вход: **все антенны сразу**, сигнал **complex float**.

**Библиотеки**:

- **rocPRIM**: `rocprim::reduce()` для mean; `rocprim::radix_sort` или `rocprim::nth_element` для median
- **Custom kernel**: Welford для variance/std (один проход mean+variance+std)

**Анализ**: kernel vs rocPRIM — см. [PLAN_ROCm_DrvGPU_Full.md](PLAN_ROCm_DrvGPU_Full.md) секция 5.3.

**Структура** — по аналогии с `modules/fft_processor/`: `include/`, `src/`, `kernels/`, `tests/`.

**Тесты**: C++ и Python vs `np.mean`, `np.std`, `np.median`

---

### 3.4 signal_generators

**План**: FormSignalGeneratorROCm, DelayedFormSignalGeneratorROCm, LfmAnalyticalDelayROCm — порт .cl → .hip

**Тесты**: расширить существующие Python-тесты режимом ROCm

---

### 3.5 filters

**План**: FirFilterROCm, IirFilterROCm — порт kernels в HIP

**Тесты**: C++/Python vs SciPy `lfilter`, `sosfilt`

---

### 3.6 lch_farrow

**План**: LchFarrowROCm — порт LCH_FARROW_KERNEL_SOURCE в HIP

**Тесты**: расширить существующие тесты

---

## Порядок работ и тесты

| Этап | Задача                                          | Тесты                                                                          |
| ---- | ----------------------------------------------- | ------------------------------------------------------------------------------ |
| 1    | Установка ROCm, OpenCL, проверка GPU            | `check_rocm_env.sh`, `Doc/AMD_Radeon_9070_Setup.md`                            |
| 2.1  | ROCmBackend                                     | `test_rocm_backend.hpp`                                                        |
| 2.2  | HIPBuffer, InputData ROCm                       | `test_hip_buffer.hpp`, `test_input_data_rocm.hpp`                              |
| 3.1  | fft_processor ROCm                              | `test_fft_processor_rocm.hpp`                                                  |
| 3.2  | fft_maxima ROCm                                 | `test_spectrum_maxima_rocm.hpp`                                                |
| 3.3  | statistics (новый): mean, variance, std, median | `test_statistics_rocm.hpp`, Python vs NumPy                                    |
| 3.4  | signal_generators ROCm                          | Расширить существующие Python-тесты                                            |
| 3.5  | filters ROCm                                    | C++/Python vs SciPy                                                            |
| 3.6  | lch_farrow ROCm                                 | Расширить существующие тесты                                                   |

---

## Полезные ссылки

- [ROCm Install (Linux)](https://rocm.docs.amd.com/projects/install-on-linux/en/latest/)
- [rocPRIM Reduce](https://rocm.docs.amd.com/projects/rocPRIM/en/latest/device_ops/reduce.html)
- [rocPRIM Nth Element](https://rocm.docs.amd.com/projects/rocPRIM/en/docs-6.3.0/device_ops/nth_element.html) (median, percentiles)
- [hipFFT](https://rocm.docs.amd.com/projects/hipFFT/en/latest/)
- [AMD Radeon RX 9070](https://amd.com/en/products/graphics/desktops/radeon/9000-series/amd-radeon-rx-9070.html) (gfx1201)

---

## Рекомендации по процессу

- **sequential-thinking**: для выбора архитектуры Statistics, рефакторинга fft_maxima под ISpectrumProcessor
- **GitHub**: искать примеры hipFFT, rocPRIM reduce, Welford на HIP
- **Статьи**: Welford's online algorithm для variance; rocPRIM reduction patterns
