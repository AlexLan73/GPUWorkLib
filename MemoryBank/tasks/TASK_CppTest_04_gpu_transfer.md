# TASK_CppTest_04 — gpu_transfer.hpp (GPU→CPU)

> **Фаза**: 0 (инфраструктура)
> **Зависимости**: — (независим)
> **Статус**: ⬜ TODO
> **Оценка**: ~1 час
> **Паттерны**: Strategy (OpenCL vs ROCm), RAII, Template

---

## 🎯 Цель

Заменить ~40 мест с 5-строчным GPU readback паттерном одной функцией.

**Было** (5 строк × 40 мест = 200 LOC):
```cpp
std::vector<std::complex<float>> out(n);
auto q = static_cast<cl_command_queue>(backend->GetNativeQueue());
clEnqueueReadBuffer(q, buf, CL_TRUE, 0, n * sizeof(std::complex<float>),
                    out.data(), 0, nullptr, nullptr);
clReleaseMemObject(buf);
```

**Станет** (1 строка × 40 = 40 LOC):
```cpp
auto out = ReadGpuBuffer<std::complex<float>>(backend, buf, n);
```

---

## 📁 Создаваемый файл (1 штука)

```
modules/test_utils/
└── gpu_transfer.hpp     ← ReadGpuBuffer, ReadHipBuffer, ReadClBuffer
```

---

## 📝 Детальное ТЗ

### `modules/test_utils/gpu_transfer.hpp`

```cpp
#pragma once

#include <vector>
#include <cstddef>
#include <stdexcept>
#include <string>

// DrvGPU includes
#include "DrvGPU/backends/i_backend.hpp"

// Conditional includes
#ifdef ENABLE_ROCM
#include <hip/hip_runtime.h>
#endif

// OpenCL — всегда доступен (DrvGPU зависит от него)
#include <CL/cl.h>

namespace gpu_test_utils {

// ══════════════════════════════════════════════════════════════════
// OpenCL: cl_mem → vector<T>
// ══════════════════════════════════════════════════════════════════

/**
 * Прочитать OpenCL буфер → vector<T>.
 * Автоматический clReleaseMemObject если release_buffer=true.
 *
 * @tparam T        float, complex<float>, double и т.д.
 * @param queue     cl_command_queue
 * @param buffer    cl_mem GPU буфер
 * @param count     количество элементов T (не байт!)
 * @param release   освободить cl_mem после чтения (default: true)
 * @return vector<T> с данными из GPU
 */
template<typename T>
inline std::vector<T>
ReadClBuffer(cl_command_queue queue, cl_mem buffer, size_t count,
             bool release = true)
{
    std::vector<T> result(count);
    cl_int err = clEnqueueReadBuffer(
        queue, buffer, CL_TRUE, 0,
        count * sizeof(T), result.data(),
        0, nullptr, nullptr
    );
    if (err != CL_SUCCESS) {
        if (release) clReleaseMemObject(buffer);
        throw std::runtime_error(
            "clEnqueueReadBuffer failed, error=" + std::to_string(err));
    }
    if (release) clReleaseMemObject(buffer);
    return result;
}

/**
 * Записать vector<T> → OpenCL буфер.
 */
template<typename T>
inline void WriteClBuffer(cl_command_queue queue, cl_mem buffer,
                           const std::vector<T>& data)
{
    cl_int err = clEnqueueWriteBuffer(
        queue, buffer, CL_TRUE, 0,
        data.size() * sizeof(T), data.data(),
        0, nullptr, nullptr
    );
    if (err != CL_SUCCESS)
        throw std::runtime_error(
            "clEnqueueWriteBuffer failed, error=" + std::to_string(err));
}

// ══════════════════════════════════════════════════════════════════
// ROCm: void* → vector<T>
// ══════════════════════════════════════════════════════════════════

#ifdef ENABLE_ROCM

/**
 * Прочитать HIP device pointer → vector<T>.
 * Автоматический hipFree если free_buffer=true.
 *
 * ⚠️ КРИТИЧНО (review #1): используем hipMemcpyAsync на конкретном stream
 * + hipStreamSynchronize. Без этого — stale data если kernel ещё не закончился!
 *
 * @param native_queue  hipStream_t от IBackend::GetNativeQueue()
 * @param device_ptr    GPU pointer
 * @param count         количество элементов T
 * @param offset        смещение в элементах T (default: 0) — review #12
 * @param free_buffer   освободить GPU память после чтения (default: true)
 */
template<typename T>
inline std::vector<T>
ReadHipBuffer(void* native_queue, void* device_ptr, size_t count,
              size_t offset = 0, bool free_buffer = true)
{
    auto stream = static_cast<hipStream_t>(native_queue);
    auto src = static_cast<char*>(device_ptr) + offset * sizeof(T);
    std::vector<T> result(count);

    hipError_t err = hipMemcpyAsync(
        result.data(), src,
        count * sizeof(T), hipMemcpyDeviceToHost, stream
    );
    if (err != hipSuccess) {
        if (free_buffer) hipFree(device_ptr);
        throw std::runtime_error(
            std::string("hipMemcpyAsync D2H failed: ") + hipGetErrorString(err));
    }
    // Обязательная синхронизация — дождаться всех операций на этом stream
    err = hipStreamSynchronize(stream);
    if (err != hipSuccess) {
        if (free_buffer) hipFree(device_ptr);
        throw std::runtime_error(
            std::string("hipStreamSynchronize failed: ") + hipGetErrorString(err));
    }
    if (free_buffer) hipFree(device_ptr);
    return result;
}

/**
 * Записать vector<T> → HIP device pointer.
 */
template<typename T>
inline void WriteHipBuffer(void* native_queue, void* device_ptr,
                           const std::vector<T>& data)
{
    auto stream = static_cast<hipStream_t>(native_queue);
    hipError_t err = hipMemcpyAsync(
        device_ptr, data.data(),
        data.size() * sizeof(T), hipMemcpyHostToDevice, stream
    );
    if (err != hipSuccess)
        throw std::runtime_error(
            std::string("hipMemcpyAsync H2D failed: ") + hipGetErrorString(err));
    hipStreamSynchronize(stream);
}

#endif // ENABLE_ROCM

// ══════════════════════════════════════════════════════════════════
// Backend-agnostic: автоопределение OpenCL vs ROCm
// ══════════════════════════════════════════════════════════════════

/**
 * Прочитать GPU буфер → vector<T>. Определяет backend автоматически.
 *
 * Это ОСНОВНАЯ функция для тестов:
 *   auto result = ReadGpuBuffer<complex<float>>(backend, gpu_ptr, n);
 *
 * @tparam T        тип данных
 * @param backend   IBackend* (OpenCL или ROCm)
 * @param buffer    void* (cl_mem или HIP device ptr)
 * @param count     количество элементов T
 * @param release   освободить GPU память после чтения (default: true)
 */
/**
 * Прочитать GPU буфер → vector<T>. Определяет backend автоматически.
 *
 * Это ОСНОВНАЯ функция для тестов:
 *   auto result = ReadGpuBuffer<complex<float>>(backend, gpu_ptr, n);
 *
 * @tparam T        тип данных
 * @param backend   IBackend* (OpenCL или ROCm)
 * @param buffer    void* (cl_mem или HIP device ptr)
 * @param count     количество элементов T
 * @param offset    смещение в элементах T (default: 0) — review #12
 * @param release   освободить GPU память после чтения (default: true)
 */
template<typename T>
inline std::vector<T>
ReadGpuBuffer(drv_gpu_lib::IBackend* backend, void* buffer, size_t count,
              size_t offset = 0, bool release = true)
{
#ifdef ENABLE_ROCM
    if (backend->GetBackendType() == drv_gpu_lib::BackendType::ROCm) {
        return ReadHipBuffer<T>(backend->GetNativeQueue(), buffer, count,
                                offset, release);
    }
#endif
    // OpenCL path (offset через clEnqueueReadBuffer byte offset)
    auto queue = static_cast<cl_command_queue>(backend->GetNativeQueue());
    auto cl_buf = static_cast<cl_mem>(buffer);
    std::vector<T> result(count);
    cl_int err = clEnqueueReadBuffer(
        queue, cl_buf, CL_TRUE, offset * sizeof(T),
        count * sizeof(T), result.data(), 0, nullptr, nullptr);
    if (err != CL_SUCCESS) {
        if (release) clReleaseMemObject(cl_buf);
        throw std::runtime_error(
            "clEnqueueReadBuffer failed, error=" + std::to_string(err));
    }
    if (release) clReleaseMemObject(cl_buf);
    return result;
}

/**
 * Прочитать GPU буфер БЕЗ освобождения (буфер переиспользуется).
 *
 * Пример: читаем антенну #3 из multi-beam буфера [8 × 4096]:
 *   auto ant3 = PeekGpuBuffer<cf>(backend, buf, 4096, /*offset=*/3*4096);
 */
template<typename T>
inline std::vector<T>
PeekGpuBuffer(drv_gpu_lib::IBackend* backend, void* buffer, size_t count,
              size_t offset = 0)
{
    return ReadGpuBuffer<T>(backend, buffer, count, offset, /*release=*/false);
}

} // namespace gpu_test_utils
```

---

## ✅ Критерии завершения

- [ ] `modules/test_utils/gpu_transfer.hpp` создан
- [ ] `ReadClBuffer<float>(queue, buf, 1024)` → vector<float> size=1024
- [ ] `ReadClBuffer<complex<float>>(queue, buf, 4096, true)` → читает + clReleaseMemObject
- [ ] `ReadHipBuffer<float>(d_ptr, 1024)` → vector<float> size=1024 (только если ENABLE_ROCM)
- [ ] `ReadGpuBuffer<complex<float>>(backend, buf, n)` → автоопределение OpenCL/ROCm
- [ ] `PeekGpuBuffer(backend, buf, n)` → читает БЕЗ освобождения
- [ ] Ошибки OpenCL/HIP бросают `std::runtime_error` (не молча теряют данные!)
- [ ] `#ifdef ENABLE_ROCM` корректно: ROCm код не компилируется на nvidia ветке
- [ ] Include пути: `DrvGPU/backends/i_backend.hpp` доступен

---

*Создан: 2026-03-21 | Кодо | Фаза 0*
