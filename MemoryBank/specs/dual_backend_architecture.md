# Dual-Backend Architecture: OpenCL + ROCm/HIP

**Статус**: 📋 ПЛАНИРОВАНИЕ
**Дата**: 2026-02-11
**Автор**: Кодо

---

## 🎯 Цель

Единый API `SpectrumMaximaFinder::Process()` работает на OpenCL и ROCm/HIP с автоматическим выбором backend'а.

```cpp
// Пользователь НЕ знает какой backend используется
auto finder = SpectrumMaximaFinder::Create(params, BackendType::AUTO);
auto results = finder->Process(input_data);  // Работает одинаково!
```

---

## 📊 Текущая архитектура

```
SpectrumMaximaFinder
        │
        ▼
   IBackend (абстракция)
        │
        ├── OpenCLBackend
        └── [ROCmBackend] (будет добавлен)
```

**Проблема**: `SpectrumMaximaFinder` жёстко использует OpenCL типы:
- `cl_context`, `cl_command_queue`, `cl_mem`
- `clFFT` (библиотека FFT для OpenCL)
- OpenCL kernel sources

---

## 🏗️ Предлагаемая архитектура

### Вариант A: Стратегия (Strategy Pattern) — РЕКОМЕНДУЕТСЯ

```
ISpectrumProcessor (интерфейс)
        │
        ├── SpectrumProcessorOpenCL  (clFFT + OpenCL kernels)
        └── SpectrumProcessorROCm    (hipFFT + HIP kernels)
                                     или (rocFFT если напрямую)

SpectrumMaximaFinder (фасад)
        │
        └── uses ISpectrumProcessor
```

**Плюсы**:
- Чистое разделение backend-специфичного кода
- Легко добавлять новые backend'ы (CUDA, Metal)
- Один файл = один backend (легко поддерживать)
- Тестирование каждого backend'а отдельно

**Минусы**:
- Больше файлов
- Небольшое дублирование логики

### Вариант B: Условная компиляция (#ifdef)

```cpp
void SpectrumMaximaFinder::ExecuteFFT() {
#ifdef USE_OPENCL
    clfftEnqueueTransform(...);
#elif defined(USE_HIP)
    hipfftExecC2C(...);
#endif
}
```

**Плюсы**:
- Один файл
- Меньше кода

**Минусы**:
- Сложная поддержка
- Тяжело тестировать оба backend'а одновременно
- #ifdef спагетти

---

## 📁 Рекомендуемая структура файлов

```
modules/fft_maxima/
├── include/
│   ├── interface/
│   │   ├── i_spectrum_processor.hpp    # Интерфейс
│   │   └── spectrum_maxima_types.h     # Типы (уже есть)
│   ├── spectrum_maxima_finder.h         # Публичный API (фасад)
│   └── kernels/
│       ├── fft_kernel_sources.hpp       # OpenCL kernels (уже есть)
│       └── fft_kernel_sources_hip.hpp   # HIP kernels (новый)
│
├── src/
│   ├── spectrum_maxima_finder.cpp       # Фасад (маленький)
│   ├── opencl/
│   │   └── spectrum_processor_opencl.cpp # OpenCL реализация
│   └── rocm/
│       └── spectrum_processor_rocm.cpp   # ROCm/HIP реализация
```

---

## 🔧 Интерфейс ISpectrumProcessor

```cpp
namespace antenna_fft {

class ISpectrumProcessor {
public:
    virtual ~ISpectrumProcessor() = default;

    /// Инициализация GPU ресурсов
    virtual void Initialize(const SpectrumParams& params) = 0;

    /// Обработка данных (может использовать batch internally)
    virtual std::vector<SpectrumResult> Process(
        const std::vector<std::complex<float>>& input_data) = 0;

    /// Очистка ресурсов
    virtual void Cleanup() = 0;

    /// Профилирование
    virtual ProfilingData GetProfilingData() const = 0;

    /// Тип backend'а
    virtual drv_gpu_lib::BackendType GetBackendType() const = 0;
};

} // namespace antenna_fft
```

---

## 🔄 Factory для создания

```cpp
class SpectrumMaximaFinder {
public:
    /// Создать finder с автоматическим выбором backend'а
    static std::unique_ptr<SpectrumMaximaFinder> Create(
        const SpectrumParams& params,
        BackendType preferred = BackendType::AUTO);

    /// Создать с конкретным IBackend
    static std::unique_ptr<SpectrumMaximaFinder> Create(
        const SpectrumParams& params,
        drv_gpu_lib::IBackend* backend);

    // Публичный API (делегирует в processor_)
    void Initialize();
    std::vector<SpectrumResult> Process(const std::vector<std::complex<float>>& data);

private:
    std::unique_ptr<ISpectrumProcessor> processor_;
};
```

---

## 📋 FFT библиотеки

| Backend | FFT Library | Notes |
|---------|-------------|-------|
| OpenCL  | clFFT       | Уже используется |
| ROCm/HIP | hipFFT     | Обёртка над rocFFT |
| CUDA    | cuFFT       | Если будем поддерживать |

### hipFFT API (аналог clFFT)

```cpp
#include <hipfft.h>

hipfftHandle plan;
hipfftPlan1d(&plan, nFFT, HIPFFT_C2C, batch_size);

hipfftExecC2C(plan, input, output, HIPFFT_FORWARD);

hipfftDestroy(plan);
```

**Важно**: hipFFT НЕ поддерживает callbacks как clFFT.
→ Padding нужно делать отдельным kernel'ом перед FFT.

---

## 📝 План реализации

### Этап 1: Рефакторинг (без ROCm) ⏱️ 2-3 часа
1. Создать `ISpectrumProcessor` интерфейс
2. Переместить OpenCL код в `SpectrumProcessorOpenCL`
3. `SpectrumMaximaFinder` → тонкий фасад
4. Тесты должны проходить!

### Этап 2: ROCm/HIP реализация ⏱️ 4-6 часов
1. Создать `SpectrumProcessorROCm`
2. Добавить HIP kernel sources (padding, post-kernel)
3. Интегрировать hipFFT
4. Тестирование на AMD GPU

### Этап 3: Factory и AUTO-detection ⏱️ 1-2 часа
1. Factory method с автоопределением GPU
2. Fallback логика (OpenCL → ROCm → CPU)
3. Unit tests

---

## ⚠️ Сложности

1. **clFFT callbacks vs hipFFT**:
   - clFFT: pre-callback делает padding "бесплатно" внутри FFT
   - hipFFT: нужен отдельный kernel для padding → +1 kernel launch
   - Решение: для hipFFT использовать [PadKernel] → [FFT] → [PostKernel]

2. **Различия в синхронизации**:
   - OpenCL: cl_event, clWaitForEvents
   - HIP: hipEvent_t, hipStreamWaitEvent
   - Решение: абстракция событий через IBackend

3. **Тестирование на AMD**:
   - Нужен доступ к AMD GPU с ROCm
   - CI pipeline с AMD

---

## ✅ Критерии готовности

- [ ] `SpectrumMaximaFinder::Process()` работает одинаково на OpenCL и ROCm
- [ ] Тесты проходят на обоих backend'ах
- [ ] Benchmark показывает сопоставимую производительность
- [ ] Профилирование работает для обоих backend'ов

---

## 🔗 Связанные документы

- `MemoryBank/specs/fft_module.md` — спецификация FFT модуля
- `DrvGPU/interface/i_backend.hpp` — интерфейс backend'а
- `MemoryBank/specs/double_buffering_analysis.md` — анализ double buffering
