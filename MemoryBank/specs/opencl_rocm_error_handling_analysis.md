# Обработка ошибок OpenCL & ROCm — анализ и предложения

> **Источник**: Context7 (Khronos OpenCL Registry, OpenCL SDK, OpenCL-Wrapper), SaveErrorCret, sequential-thinking
> **Дата**: 2026-02-11
> **Связь**: [SaveErrorCret/Принудительная запись при крит Error.md](../../MemoryBank/DiscussionPlan/SaveErrorCret/Принудительная%20запись%20при%20крит%20Error.md)

---

## ЧАСТЬ 1. Текущее состояние в GPUWorkLib

### 1.1 Паттерны обработки

| Паттерн | Где используется | Пример |
|---------|------------------|--------|
| `if (err != CL_SUCCESS) throw std::runtime_error(...)` | Большинство модулей | [spectrum_maxima_finder.cpp](../../modules/fft_maxima/src/spectrum_maxima_finder.cpp) |
| `CheckCLError(err, "operation")` | opencl_core, external_cl_buffer_adapter, svm_buffer | [opencl_core.hpp](../../DrvGPU/backends/opencl/opencl_core.hpp):201 |
| Прямой `throw std::runtime_error` | Валидация, логика | Модули, DrvGPU |

### 1.2 Существующая утилита CheckCLError

```cpp
// DrvGPU/backends/opencl/opencl_core.hpp
inline void CheckCLError(cl_int error, const std::string& operation) {
    if (error != CL_SUCCESS) {
        std::string error_msg = "OpenCL Error [" + std::to_string(error) + "] in " + operation;
        throw std::runtime_error(error_msg);
    }
}
```

**Проблема**: Код ошибки передаётся как число (например, `-30`), без расшифровки (CL_INVALID_VALUE и т.д.).

### 1.3 Где нет единообразия

- **spectrum_maxima_finder.cpp**, **antenna_fft_core.cpp** — проверки вручную, без `CheckCLError`
- **clfftStatus** (clFFT) — отдельный enum, не cl_int
- **ROCm/HIP** — пока не реализован; будет hipError_t

---

## ЧАСТЬ 2. Паттерны из Context7 и Khronos

### 2.1 Khronos OpenCL Registry — макрос CL_CHECK

```cpp
#define CL_CHECK(ERROR)                             \
  if (ERROR) {                                      \
    std::cerr << "OpenCL error: " << ERROR << "\n"; \
    return ERROR;                                   \
  }
```

- Возвращает код ошибки (не throw)
- Подходит для C-стиля; в C++ чаще используют exceptions

### 2.2 OpenCL SDK — Error-класс

```cpp
class Error : public std::exception {
    cl_int err_;
    const char* errStr_;
public:
    Error(cl_int err, const char* errStr = NULL);
    cl_int err() const { return err_; }
    const char* what() const throw();
};
```

- Специализированное исключение с кодом
- `CL_HPP_ENABLE_EXCEPTIONS` — опция для C++ bindings

### 2.3 OpenCL-Wrapper — print_error / if (error)

```cpp
if (error) print_error("OpenCL Buffer allocation failed with error code " + to_string(error) + ".");
```

- Логирование + ранний выход
- Отдельная функция `print_error` для единообразия

---

## ЧАСТЬ 3. Связь с SaveErrorCret — принудительная запись при критической ошибке

Документ [Принудительная запись при крит Error.md](../../MemoryBank/DiscussionPlan/SaveErrorCret/Принудительная%20запись%20при%20крит%20Error.md) описывает:

- **VEH (Windows)** — `AddVectoredExceptionHandler` для Access Violation, сбоев GPU
- **Сигналы (Linux)** — `SIGSEGV`, `SIGFPE`, `SIGILL` + backtrace
- **OpenCL pfn_notify** — callback при ошибке контекста

**Разделение уровней**:

| Уровень | Что перехватывает | Инструмент |
|---------|-------------------|------------|
| **API (cl_int, hipError_t)** | Ошибки возврата функций | try/catch, CheckCLError, макросы |
| **Асинхронные GPU** | Ошибки в clFinish/clWaitForEvents | Тот же try/catch + callback |
| **Критические падения** | Access Violation, SIGSEGV | VEH, signal handlers, SaveErrorCret |

Текущий try/catch покрывает **уровень API**. SaveErrorCret — это **последняя линия** при полном падении процесса.

---

## ЧАСТЬ 4. Детальные предложения

### 4.1 Централизованный GPU Error Handler

**Цель**: единая точка для проверки, логирования и преобразования в исключение.

**Файл**: `DrvGPU/common/gpu_error_handler.hpp` (новый)

```cpp
#pragma once

#include <CL/cl.h>
#include <string>
#include <stdexcept>

namespace drv_gpu_lib {

enum class GPUBackend { OpenCL, ROCm };

// Расшифровка cl_int в строку (см. ЧАСТЬ 7)
const char* GetOpenCLErrorString(cl_int err);

// Проверка: критическая ли ошибка (для crash_log)
bool IsCriticalOpenCLError(cl_int err);

class GPUErrorHandler {
public:
    // OpenCL: проверка + throw при ошибке
    static void CheckOpenCL(cl_int err, const char* operation, const char* file, int line);

    // Вариант без file/line (для совместимости)
    static void CheckOpenCL(cl_int err, const std::string& operation);

    // Опция: только логировать, не бросать (для некритичных путей)
    static bool LogIfError(cl_int err, const char* operation);
};
}
```

**Макросы**:

```cpp
#define CL_CHECK(expr) do { \
    cl_int _err = (expr); \
    drv_gpu_lib::GPUErrorHandler::CheckOpenCL(_err, #expr, __FILE__, __LINE__); \
} while(0)

// Для функций, возвращающих cl_int в последнем аргументе:
#define CL_CHECK_RET(expr) do { \
    cl_int _err; \
    (expr); \
    drv_gpu_lib::GPUErrorHandler::CheckOpenCL(_err, #expr, __FILE__, __LINE__); \
} while(0)
```

### 4.2 Специализированное исключение GPUException

**Файл**: `DrvGPU/common/gpu_exception.hpp` (новый)

```cpp
#pragma once

#include <stdexcept>
#include <string>

namespace drv_gpu_lib {

enum class GPUBackend { OpenCL, ROCm };

class GPUException : public std::runtime_error {
public:
    GPUException(GPUBackend backend, int error_code, const std::string& operation,
                 const char* file = nullptr, int line = 0);

    int GetErrorCode() const { return error_code_; }
    GPUBackend GetBackend() const { return backend_; }
    std::string GetErrorString() const;  // человекочитаемое описание
    const char* GetFile() const { return file_; }
    int GetLine() const { return line_; }

private:
    GPUBackend backend_;
    int error_code_;
    const char* file_;
    int line_;
};
}
```

**Реализация конструктора**:

```cpp
GPUException::GPUException(GPUBackend backend, int error_code, const std::string& operation,
                           const char* file, int line)
    : std::runtime_error(BuildMessage(backend, error_code, operation, file, line))
    , backend_(backend)
    , error_code_(error_code)
    , file_(file)
    , line_(line) {}

std::string GPUException::BuildMessage(...) {
    std::string msg = (backend_ == GPUBackend::OpenCL ? "OpenCL" : "ROCm")
        + " error " + GetErrorString() + " (" + std::to_string(error_code_) + ") in " + operation;
    if (file_ && line_ > 0) {
        msg += " at " + std::string(file_) + ":" + std::to_string(line_);
    }
    return msg;
}
```

**Использование**:

```cpp
try {
    // ...
} catch (const GPUException& e) {
    std::cerr << "GPU error: " << e.what() << "\n";
    std::cerr << "Code: " << e.GetErrorCode() << ", Backend: " << (int)e.GetBackend() << "\n";
    // Возможна логика recovery по типу ошибки
} catch (const std::exception& e) {
    std::cerr << "Other: " << e.what() << "\n";
}
```

### 4.3 Интеграция с принудительной записью при критической ошибке

**Псевдокод CheckOpenCL**:

```cpp
void GPUErrorHandler::CheckOpenCL(cl_int err, const char* operation, const char* file, int line) {
    if (err == CL_SUCCESS) return;

    std::string err_str = GetOpenCLErrorString(err);
    std::string full_msg = "OpenCL " + err_str + " (" + std::to_string(err) + ") in " + operation;

    // 1. Лог (plog, если доступен)
    if (ILogger* logger = GetLoggerForGPU(0)) {  // или передать gpu_id
        logger->Error("GPU", full_msg);
    }

    // 2. Критическая ошибка — запись в crash_log для VEH/signal handler
    if (IsCriticalOpenCLError(err)) {
        CrashLog::Append("GPU_OPENCL", operation, err, full_msg, file, line);
    }

    // 3. Throw
    throw GPUException(GPUBackend::OpenCL, err, operation, file, line);
}

bool IsCriticalOpenCLError(cl_int err) {
    switch (err) {
        case CL_OUT_OF_HOST_MEMORY:
        case CL_OUT_OF_RESOURCES:
        case CL_MEM_OBJECT_ALLOCATION_FAILURE:
        case CL_MAP_FAILURE:
            return true;
        default:
            return false;
    }
}
```

**Структура CrashLog** (для SaveErrorCret):

```cpp
// crash_log.hpp — последняя запись, доступная VEH/signal handler
namespace CrashLog {
    void Append(const char* component, const char* operation, int code,
                const std::string& message, const char* file, int line);
    std::string GetLastEntry();  // для вывода в VEH/signal handler
}
```

### 4.4 clFFT — детальное приведение к общему паттерну

clFFT возвращает `clfftStatus`. Значения:

- **CLFFT_SUCCESS** — равен CL_SUCCESS
- **CLFFT_*** — первые 47 кодов** — совпадают с OpenCL (CLFFT_INVALID_VALUE = CL_INVALID_VALUE)
- **CLFFT_BUGCHECK = 4096** и далее — расширенные коды clFFT

**Рекомендация**: `CheckCLFFT(clfftStatus status, const char* operation, const char* file, int line)`:

```cpp
void CheckCLFFT(clfftStatus status, const char* operation, const char* file, int line) {
    if (status == CLFFT_SUCCESS) return;

    // Значения < 4096 — OpenCL-совместимые, можно передать в GetOpenCLErrorString
    if (status < CLFFT_BUGCHECK) {
        GPUErrorHandler::CheckOpenCL(static_cast<cl_int>(status), operation, file, line);
    }

    // Расширенные коды clFFT
    const char* clfft_str = GetCLFFTErrorString(status);
    throw GPUException(GPUBackend::OpenCL, static_cast<int>(status),
                       std::string(operation) + " (clFFT: " + clfft_str + ")",
                       file, line);
}

const char* GetCLFFTErrorString(clfftStatus s) {
    switch (s) {
        case CLFFT_BUGCHECK: return "CLFFT_BUGCHECK";
        case CLFFT_NOTIMPLEMENTED: return "CLFFT_NOTIMPLEMENTED";
        case CLFFT_TRANSPOSED_NOTIMPLEMENTED: return "CLFFT_TRANSPOSED_NOTIMPLEMENTED";
        case CLFFT_FILE_NOT_FOUND: return "CLFFT_FILE_NOT_FOUND";
        case CLFFT_FILE_CREATE_FAILURE: return "CLFFT_FILE_CREATE_FAILURE";
        case CLFFT_VERSION_MISMATCH: return "CLFFT_VERSION_MISMATCH";
        case CLFFT_INVALID_PLAN: return "CLFFT_INVALID_PLAN";
        case CLFFT_DEVICE_NO_DOUBLE: return "CLFFT_DEVICE_NO_DOUBLE";
        case CLFFT_DEVICE_MISMATCH: return "CLFFT_DEVICE_MISMATCH";
        default: return "CLFFT_UNKNOWN";
    }
}
```

### 4.5 ROCm/HIP — будущее расширение

HIP предоставляет `hipGetErrorString(hipError_t)` — не нужно дублировать таблицу. Рекомендация:

```cpp
#if defined(CLANG_HIP_SUPPORT) || defined(USE_ROCm)
#include <hip/hip_runtime.h>

void GPUErrorHandler::CheckROCm(hipError_t err, const char* operation, const char* file, int line) {
    if (err == hipSuccess) return;

    const char* err_str = hipGetErrorString(err);
    throw GPUException(GPUBackend::ROCm, static_cast<int>(err),
                       std::string(operation) + ": " + (err_str ? err_str : "unknown"),
                       file, line);
}

#define HIP_CHECK(expr) do { \
    hipError_t _err = (expr); \
    drv_gpu_lib::GPUErrorHandler::CheckROCm(_err, #expr, __FILE__, __LINE__); \
} while(0)
#endif
```

---

## ЧАСТЬ 5. План внедрения (поэтапно)

### Этап 1: Улучшение текущего CheckCLError

1. Добавить `GetOpenCLErrorString(cl_int)` — полная таблица (раздел 8)
2. В `CheckCLError` формировать сообщение: `"CL_INVALID_VALUE (-30) in operation"`
3. Вынести в `DrvGPU/common/` для общего доступа
4. Заменить все `throw std::runtime_error("... " + std::to_string(err))` на вызов `CheckCLError` или `GetOpenCLErrorString`

**Файлы для изменения**:
- spectrum_maxima_finder.cpp (~15 мест)
- antenna_fft_core.cpp, antenna_fft_release.cpp
- opencl_backend.cpp
- vector_ops_module, test_external_context_fft

### Этап 2: GPUException и GPUErrorHandler

1. Создать `gpu_exception.hpp`, `gpu_error_handler.hpp`
2. Реализовать `GPUErrorHandler::CheckOpenCL` с вызовом GPUException
3. Ввести макрос `CL_CHECK`
4. Постепенно заменить ручные проверки на `CL_CHECK`

### Этап 3: Интеграция с логом и SaveErrorCret

1. При критической ошибке — запись в `crash_log.txt` или через plog
2. Согласовать с VEH/signal handler — читать последнюю запись при падении
3. Добавить `pfn_notify` в `clCreateContext` для асинхронных ошибок (опционально)

### Этап 4: ROCm

1. При появлении ROCm backend — `CheckROCm`, макрос `HIP_CHECK`
2. Расширить `GPUException` для `backend_ == ROCm`

---

## ЧАСТЬ 6. Примеры миграции существующего кода

**Было** (spectrum_maxima_finder.cpp):

```cpp
cl_int err = clCreateBuffer(context_, CL_MEM_READ_WRITE, userdata_size, nullptr, &err);
if (err != CL_SUCCESS) {
    throw std::runtime_error("Failed to create pre_callback_userdata buffer: " + std::to_string(err));
}
```

**Стало** (после Этапа 1):

```cpp
cl_int err = clCreateBuffer(context_, CL_MEM_READ_WRITE, userdata_size, nullptr, &err);
if (err != CL_SUCCESS) {
    throw std::runtime_error("Failed to create pre_callback_userdata buffer: " + 
        GetOpenCLErrorString(err) + " (" + std::to_string(err) + ")");
}
```

**Стало** (после Этапа 2):

```cpp
cl_int err;
cl_mem buf = clCreateBuffer(context_, CL_MEM_READ_WRITE, userdata_size, nullptr, &err);
GPUErrorHandler::CheckOpenCL(err, "clCreateBuffer(pre_callback_userdata)", __FILE__, __LINE__);
```

**Или с макросом** (для выражений, возвращающих cl_int):

```cpp
cl_int err;
cl_mem buf = clCreateBuffer(context_, CL_MEM_READ_WRITE, userdata_size, nullptr, &err);
CL_CHECK(err);  // макрос с #expr = "err"
```

---

## ЧАСТЬ 7. Полная таблица OpenCL Error Codes

**Файл**: `DrvGPU/common/opencl_error_strings.cpp`

```cpp
#include "gpu_error_handler.hpp"
#include <CL/cl.h>

namespace drv_gpu_lib {

const char* GetOpenCLErrorString(cl_int err) {
    switch (err) {
        case CL_SUCCESS: return "CL_SUCCESS";
        case CL_DEVICE_NOT_FOUND: return "CL_DEVICE_NOT_FOUND";
        case CL_DEVICE_NOT_AVAILABLE: return "CL_DEVICE_NOT_AVAILABLE";
        case CL_COMPILER_NOT_AVAILABLE: return "CL_COMPILER_NOT_AVAILABLE";
        case CL_MEM_OBJECT_ALLOCATION_FAILURE: return "CL_MEM_OBJECT_ALLOCATION_FAILURE";
        case CL_OUT_OF_RESOURCES: return "CL_OUT_OF_RESOURCES";
        case CL_OUT_OF_HOST_MEMORY: return "CL_OUT_OF_HOST_MEMORY";
        case CL_PROFILING_INFO_NOT_AVAILABLE: return "CL_PROFILING_INFO_NOT_AVAILABLE";
        case CL_MEM_COPY_OVERLAP: return "CL_MEM_COPY_OVERLAP";
        case CL_IMAGE_FORMAT_MISMATCH: return "CL_IMAGE_FORMAT_MISMATCH";
        case CL_IMAGE_FORMAT_NOT_SUPPORTED: return "CL_IMAGE_FORMAT_NOT_SUPPORTED";
        case CL_BUILD_PROGRAM_FAILURE: return "CL_BUILD_PROGRAM_FAILURE";
        case CL_MAP_FAILURE: return "CL_MAP_FAILURE";
        case CL_INVALID_VALUE: return "CL_INVALID_VALUE";
        case CL_INVALID_DEVICE_TYPE: return "CL_INVALID_DEVICE_TYPE";
        case CL_INVALID_PLATFORM: return "CL_INVALID_PLATFORM";
        case CL_INVALID_DEVICE: return "CL_INVALID_DEVICE";
        case CL_INVALID_CONTEXT: return "CL_INVALID_CONTEXT";
        case CL_INVALID_QUEUE_PROPERTIES: return "CL_INVALID_QUEUE_PROPERTIES";
        case CL_INVALID_COMMAND_QUEUE: return "CL_INVALID_COMMAND_QUEUE";
        case CL_INVALID_HOST_PTR: return "CL_INVALID_HOST_PTR";
        case CL_INVALID_MEM_OBJECT: return "CL_INVALID_MEM_OBJECT";
        case CL_INVALID_IMAGE_FORMAT_DESCRIPTOR: return "CL_INVALID_IMAGE_FORMAT_DESCRIPTOR";
        case CL_INVALID_IMAGE_SIZE: return "CL_INVALID_IMAGE_SIZE";
        case CL_INVALID_SAMPLER: return "CL_INVALID_SAMPLER";
        case CL_INVALID_BINARY: return "CL_INVALID_BINARY";
        case CL_INVALID_BUILD_OPTIONS: return "CL_INVALID_BUILD_OPTIONS";
        case CL_INVALID_PROGRAM: return "CL_INVALID_PROGRAM";
        case CL_INVALID_PROGRAM_EXECUTABLE: return "CL_INVALID_PROGRAM_EXECUTABLE";
        case CL_INVALID_KERNEL_NAME: return "CL_INVALID_KERNEL_NAME";
        case CL_INVALID_KERNEL_DEFINITION: return "CL_INVALID_KERNEL_DEFINITION";
        case CL_INVALID_KERNEL: return "CL_INVALID_KERNEL";
        case CL_INVALID_ARG_INDEX: return "CL_INVALID_ARG_INDEX";
        case CL_INVALID_ARG_VALUE: return "CL_INVALID_ARG_VALUE";
        case CL_INVALID_ARG_SIZE: return "CL_INVALID_ARG_SIZE";
        case CL_INVALID_KERNEL_ARGS: return "CL_INVALID_KERNEL_ARGS";
        case CL_INVALID_WORK_DIMENSION: return "CL_INVALID_WORK_DIMENSION";
        case CL_INVALID_WORK_GROUP_SIZE: return "CL_INVALID_WORK_GROUP_SIZE";
        case CL_INVALID_WORK_ITEM_SIZE: return "CL_INVALID_WORK_ITEM_SIZE";
        case CL_INVALID_GLOBAL_OFFSET: return "CL_INVALID_GLOBAL_OFFSET";
        case CL_INVALID_EVENT_WAIT_LIST: return "CL_INVALID_EVENT_WAIT_LIST";
        case CL_INVALID_EVENT: return "CL_INVALID_EVENT";
        case CL_INVALID_OPERATION: return "CL_INVALID_OPERATION";
        case CL_INVALID_GL_OBJECT: return "CL_INVALID_GL_OBJECT";
        case CL_INVALID_BUFFER_SIZE: return "CL_INVALID_BUFFER_SIZE";
        case CL_INVALID_MIP_LEVEL: return "CL_INVALID_MIP_LEVEL";
        case CL_INVALID_GLOBAL_WORK_SIZE: return "CL_INVALID_GLOBAL_WORK_SIZE";
        case CL_INVALID_PROPERTY: return "CL_INVALID_PROPERTY";
        case CL_INVALID_IMAGE_DESCRIPTOR: return "CL_INVALID_IMAGE_DESCRIPTOR";
        case CL_INVALID_COMPILER_OPTIONS: return "CL_INVALID_COMPILER_OPTIONS";
        case CL_INVALID_LINKER_OPTIONS: return "CL_INVALID_LINKER_OPTIONS";
        case CL_INVALID_DEVICE_PARTITION_COUNT: return "CL_INVALID_DEVICE_PARTITION_COUNT";
        case CL_INVALID_PIPE_SIZE: return "CL_INVALID_PIPE_SIZE";
        case CL_INVALID_DEVICE_QUEUE: return "CL_INVALID_DEVICE_QUEUE";
        default: return "CL_UNKNOWN_ERROR";
    }
}
}
```

**Числовые значения** (для справки): CL_SUCCESS=0, CL_INVALID_VALUE=-30, CL_OUT_OF_HOST_MEMORY=-6, CL_OUT_OF_RESOURCES=-5.

---

## ЧАСТЬ 8. ROCm/HIP — основные коды (для будущего)

| hipError_t | Значение | Описание |
|------------|----------|----------|
| hipSuccess | 0 | Нет ошибки |
| hipErrorOutOfMemory | 2 | Недостаточно памяти |
| hipErrorInvalidDevicePointer | 17 | Некорректный указатель устройства |
| hipErrorIllegalState | 401 | Недопустимое состояние |
| hipErrorNotReady | 600 | Устройство не готово |
| hipErrorNotSupported | 801 | Не поддерживается |

**Примечание**: HIP предоставляет `hipGetErrorString(hipError_t)` — использовать готовую функцию.

---

## ЧАСТЬ 9. OpenCL Context Error Callback (pfn_notify)

Для перехвата асинхронных ошибок контекста:

```cpp
void CL_CALLBACK oclContextCallback(const char* errinfo, const void* private_info, size_t cb, void* user_data) {
    (void)private_info;
    (void)cb;
    (void)user_data;
    std::cerr << "[OpenCL Context Error] " << (errinfo ? errinfo : "unknown") << "\n";
    // Опционально: записать в CrashLog для последующего анализа
}

cl_context ctx = clCreateContext(properties, 1, &device, oclContextCallback, user_data, &err);
```

**Рекомендация**: Добавить при создании контекста в OpenCLCore/OpenCLBackend.

---

## ЧАСТЬ 10. Сводка

| Аспект | Рекомендация |
|--------|--------------|
| **Сейчас** | Добавить расшифровку cl_int в CheckCLError |
| **Среднее** | GPUException + GPUErrorHandler + макрос CL_CHECK |
| **Будущее** | ROCm: CheckROCm, HIP_CHECK; единый GPUException |
| **SaveErrorCret** | Интеграция: запись последней GPU-ошибки перед throw; VEH/signal — последняя линия |
| **clFFT** | CheckCLFFT с маппингом на OpenCL или собственные строки |
| **Context callback** | pfn_notify в clCreateContext для асинхронных ошибок |

---

*Создано: 2026-02-11*
*Детализировано: 2026-02-11*
