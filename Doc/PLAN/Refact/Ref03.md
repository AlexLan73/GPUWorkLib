# Ref03: Исправление нулевого GPU-профилирования

## Цель

Устранить нулевые значения в GPU-профилировании SpectrumMaximaFinder (Upload, FFT, Post-kernel, Download = 0.000 ms). Профилирование управляется через `is_prof` в configGPU.json.

---

## Диагноз: почему все значения = 0.000 ms

**Корневая причина**: В `DrvGPU/backends/opencl/opencl_backend.cpp` (строки 131–136) command queue создаётся без `CL_QUEUE_PROFILING_ENABLE`:

- OpenCL 2.0: `props[] = {0}` — пустые свойства
- Legacy: `clCreateCommandQueue(..., 0)` — флаги = 0

Без этого флага OpenCL не заполняет profiling info, и `clGetEventProfilingInfo()` возвращает ошибку. В `spectrum_maxima_finder.cpp` при ошибке `ProfileEvent` возвращает `0.0`.

**Цепочка**: OpenCLBackend (queue без profiling) → SpectrumMaximaFinder использует эту queue → clGetEventProfilingInfo → ошибка → 0.0

---

## План действий

### 0. Загрузка конфига при старте (проверить реализацию)

**Порядок инициализации**: 1) Сначала читаем configGPU.json, 2) Потом разворачиваем систему DrvGPU.

**CMake (главный или src/)**: Должен копировать `DrvGPU/config/configGPU.json` в каталог запуска (например `$<TARGET_FILE_DIR:GPUWorkLib>` или `CMAKE_BINARY_DIR`), чтобы при старте файл был рядом с исполняемым.

**При старте**:
- Файл configGPU.json должен быть в рабочем каталоге
- Программа инициализирует **только GPU с id из конфига** (например, из 10 GPU в системе нужна только 5‑я → в конфиге `id=5`, создаём `DrvGPU(OPENCL, 5)`)
- Если файла нет: создать конфиг по умолчанию, записать в каталог запуска, затем продолжить инициализацию **с дефолтными настройками** (LoadOrCreate)

**Где**: `DrvGPU::Initialize()` в начале вызывает `GPUConfig::GetInstance().LoadOrCreate(path)` если ещё не загружен. Путь — к файлу в каталоге запуска.

Ожидаемый формат JSON (см. `config_types.hpp`):
```json
{
  "version": "1.0",
  "gpus": [
    {"id": 0, "name": "...", "is_prof": true, "is_logger": true, ...}
  ]
}
```

### 1. OpenCLBackend: queue с учётом is_prof (тернарный оператор, без лишних if)

**Файл**: `DrvGPU/backends/opencl/opencl_backend.cpp` (строки 126–137)

**Примечание про `#ifdef CL_VERSION_2_0`**: В коде есть ветвление — для OpenCL 2.0+ используется `clCreateCommandQueueWithProperties`, для старых 1.x — `clCreateCommandQueue`. У тебя OpenCL 3.0 и AMD AI100 (2.2), поэтому `CL_VERSION_2_0` всегда определён — выполняется только первая ветка. Варианты: (а) оставить оба варианта — для совместимости со старыми системами; (б) убрать `#ifdef` и оставить только `clCreateCommandQueueWithProperties` — если проект ориентирован только на OpenCL 2.0+. Решение — по необходимости.

**Внешнее управление**: добавить set/get для переменных из configGPU.json (доступ к is_prof и др. извне, если потребуется).

**Идея**: выбрать props/флаги через тернарный оператор прямо в вызове OpenCL — один выбор, без ветвлений.

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
#include "config/gpu_config.hpp"  // если ещё не включён

// ...

#ifdef CL_VERSION_2_0
    static const cl_queue_properties PROPS_PROF[] = {CL_QUEUE_PROPERTIES, CL_QUEUE_PROFILING_ENABLE, 0};
    static const cl_queue_properties PROPS_NONE[] = {0};
    bool want_prof = !GPUConfig::GetInstance().IsLoaded()
        || GPUConfig::GetInstance().IsProfilingEnabled(device_index);
    queue_ = clCreateCommandQueueWithProperties(context_, device_,
        want_prof ? PROPS_PROF : PROPS_NONE, &err);
#else
    cl_command_queue_properties flags = (!GPUConfig::GetInstance().IsLoaded()
        || GPUConfig::GetInstance().IsProfilingEnabled(device_index))
        ? static_cast<cl_command_queue_properties>(CL_QUEUE_PROFILING_ENABLE) : 0;
    queue_ = clCreateCommandQueue(context_, device_, flags, &err);
#endif
```

Логика: если конфиг не загружен — включаем профилирование (поведение по умолчанию). Если загружен — смотрим `is_prof` для `device_index`.

### 2. ProfileEvent: диагностика без #ifdef (одна строка)

**Файл**: `modules/fft_maxima/src/spectrum_maxima_finder.cpp`

При ошибке `clGetEventProfilingInfo` — одна строка лога (plog или std::cerr), без `#ifdef NDEBUG` и лишних веток:

```cpp
if (err1 != CL_SUCCESS || err2 != CL_SUCCESS) {
    std::cerr << "[ProfileEvent] " << name << " failed: " << err1 << "," << err2 << "\n";
    return 0.0;
}
```

При наличии plog можно заменить на `PLOG_WARNING << ...` — фильтрация по уровню остаётся в конфиге логгера.

### 3. Проверка

1. Сборка: `cmake --build build`
2. Запуск: `./build/GPUWorkLib`
3. Ожидание: при `is_prof: true` в блоке «GPU ПРОФИЛИРОВАНИЕ» ненулевые значения (мс или мкс)
4. При `is_prof: false` — нули (профилирование отключено, queue без profiling)

### 4. Документация

Обновить этот файл после проверки: указать результат (пример вывода с реальными значениями).

---

## Риски и ограничения

- **Влияние на DrvGPU**: при `is_prof=true` queue создаётся с `CL_QUEUE_PROFILING_ENABLE`. Это обычно допустимо и не влияет на производительность до момента вызова `clGetEventProfilingInfo`. При `is_prof=false` — как раньше, без profiling.
- **OpenCLBackendExternal**: при внешнем context/queue профилирование зависит от того, как создана эта queue.

---

## Результаты проверки

**Выполнено 2026-02-03**

- Сборка: успешно
- Тест: `cd build && ./GPUWorkLib` — PASS
- Профилирование при `is_prof: true`:
  ```
  Upload (Host→GPU):       0.729    ms
  FFT (with pre-callback): 0.131    ms
  Post-kernel:             0.189    ms
  Download (GPU→Host):     0.001    ms
  TOTAL:                   1.051    ms
  ```

---

## Примечания

- Логика is_prof: если конфиг не загружен — профилирование по умолчанию включено. Если загружен — смотрим `is_prof` для `device_index`. Выбор GPU по id описан в п. 0.
