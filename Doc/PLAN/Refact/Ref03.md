# Ref03: Исправление нулевого GPU-профилирования

## Цель

Устранить нулевые значения в GPU-профилировании SpectrumMaximaFinder (Upload, FFT, Post-kernel, Download = 0.000 ms).

---

## Диагноз: почему все значения = 0.000 ms

**Корневая причина**: В `DrvGPU/backends/opencl/opencl_backend.cpp` (строки 131–136) command queue создаётся без `CL_QUEUE_PROFILING_ENABLE`:

- OpenCL 2.0: `props[] = {0}` — пустые свойства
- Legacy: `clCreateCommandQueue(..., 0)` — флаги = 0

Без этого флага OpenCL не заполняет profiling info, и `clGetEventProfilingInfo()` возвращает ошибку. В `spectrum_maxima_finder.cpp` при ошибке `ProfileEvent` возвращает `0.0`.

**Цепочка**: OpenCLBackend (queue без profiling) → SpectrumMaximaFinder использует эту queue → clGetEventProfilingInfo → ошибка → 0.0

---

## План действий

### 1. Исправление OpenCLBackend

**Файл**: `DrvGPU/backends/opencl/opencl_backend.cpp` (строки 131–136)

**Было**:
```cpp
#ifdef CL_VERSION_2_0
    cl_queue_properties props[] = {0};
    queue_ = clCreateCommandQueueWithProperties(context_, device_, props, &err);
#else
    queue_ = clCreateCommandQueue(context_, device_, 0, &err);
#endif
```

**Стало**:
```cpp
#ifdef CL_VERSION_2_0
    cl_queue_properties props[] = {CL_QUEUE_PROPERTIES, CL_QUEUE_PROFILING_ENABLE, 0};
    queue_ = clCreateCommandQueueWithProperties(context_, device_, props, &err);
#else
    queue_ = clCreateCommandQueue(context_, device_, CL_QUEUE_PROFILING_ENABLE, &err);
#endif
```

### 2. Диагностика в ProfileEvent (опционально)

**Файл**: `modules/fft_maxima/src/spectrum_maxima_finder.cpp`

В `ProfileEvent` при `err1 != CL_SUCCESS || err2 != CL_SUCCESS` логировать код ошибки (например, через `std::cerr` в Debug или plog) — для отладки, если профилирование вдруг снова перестанет работать.

**Пример**:
```cpp
if (err1 != CL_SUCCESS || err2 != CL_SUCCESS) {
#ifdef NDEBUG
    // Release: молча возвращаем 0
#else
    std::cerr << "[ProfileEvent] " << name << ": clGetEventProfilingInfo failed "
              << "start_err=" << err1 << " end_err=" << err2 << "\n";
#endif
    return 0.0;
}
```

### 3. Проверка

1. Сборка: `cmake --build build`
2. Запуск: `./build/GPUWorkLib`
3. Ожидание: в блоке «GPU ПРОФИЛИРОВАНИЕ» ненулевые значения (мс или мкс)

### 4. Документация

Обновить этот файл после проверки: указать результат (пример вывода с реальными значениями).

---

## Риски и ограничения

- **Влияние на DrvGPU**: все пользователи OpenCLBackend получат queue с `CL_QUEUE_PROFILING_ENABLE`. Это обычно допустимо и не влияет на производительность до момента вызова `clGetEventProfilingInfo`.
- **OpenCLBackendExternal**: при внешнем context/queue профилирование зависит от того, как создана эта queue.

---

## Результаты проверки

_(заполнить после выполнения)_

---

## Примечания (добавить по необходимости)

_Здесь можно дописать свои моменты._
