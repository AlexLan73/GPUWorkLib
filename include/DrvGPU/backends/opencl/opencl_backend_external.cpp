#include "opencl_backend_external.hpp"
#include "../../common/logger.hpp"
#include <iostream>
#include <cstring>

namespace drv_gpu_lib {

// ════════════════════════════════════════════════════════════════════════════
// Конструктор и деструктор
// ════════════════════════════════════════════════════════════════════════════

OpenCLBackendExternal::OpenCLBackendExternal(
    cl_context external_context,
    cl_device_id external_device,
    cl_command_queue external_queue,
    bool owns_resources)
    : OpenCLBackend()  // Вызываем базовый конструктор
    , is_external_context_(true)
    , owns_resources_(owns_resources)
    , external_context_(external_context)
    , external_device_(external_device)
    , external_queue_(external_queue)
{
    // Валидация входных параметров
    if (!external_context_ || !external_device_ || !external_queue_) {
        throw std::invalid_argument(
            "OpenCLBackendExternal: external context, device, and queue must not be null"
        );
    }

    DRVGPU_LOG_INFO("OpenCLBackendExternal", "Created with external OpenCL context, owns resources: " + 
        std::string(owns_resources_ ? "YES" : "NO"));
}

OpenCLBackendExternal::~OpenCLBackendExternal() {
    // Cleanup вызовется автоматически
    // Внешние ресурсы НЕ будут уничтожены (если owns_resources_ = false)
}

// ════════════════════════════════════════════════════════════════════════════
// Инициализация с внешним контекстом
// ════════════════════════════════════════════════════════════════════════════

void OpenCLBackendExternal::InitializeWithExternalContext() {
    std::lock_guard<std::mutex> lock(const_cast<std::mutex&>(mutex_));

    if (IsInitialized()) {
        DRVGPU_LOG_WARNING("OpenCLBackendExternal", "Already initialized, cleaning up first");
        Cleanup();
    }

    DRVGPU_LOG_INFO("OpenCLBackendExternal", "Initializing with external context...");

    // 1. Валидация внешних объектов
    ValidateExternalObjects();

    // 2. Установить внутренние указатели на внешние объекты
    // (переопределяем члены базового класса OpenCLBackend)
    context_ = external_context_;
    device_ = external_device_;
    queue_ = external_queue_;
    device_index_ = 0;  // Внешний контекст - считаем device 0

    // 3. Инициализировать SVM capabilities для устройства
    svm_capabilities_ = std::make_unique<SVMCapabilities>(
        SVMCapabilities::Query(external_device_)
    );

    // 4. Инициализировать MemoryManager (будет работать с внешним контекстом)
    memory_manager_ = std::make_unique<MemoryManager>(this);

    // 5. Установить флаг инициализации
    initialized_ = true;

    DRVGPU_LOG_INFO("OpenCLBackendExternal", "Initialized successfully, device: " + GetDeviceName());

    // Вывести SVM capabilities
    if (svm_capabilities_->HasAnySVM()) {
        DRVGPU_LOG_INFO("OpenCLBackendExternal", "SVM supported: YES");
        DRVGPU_LOG_DEBUG("OpenCLBackendExternal", svm_capabilities_->ToString());
    } else {
        DRVGPU_LOG_INFO("OpenCLBackendExternal", "SVM not supported (using regular buffers)");
    }
}

// ════════════════════════════════════════════════════════════════════════════
// Override Cleanup для безопасной работы с внешними ресурсами
// ════════════════════════════════════════════════════════════════════════════

void OpenCLBackendExternal::Cleanup() {
    std::lock_guard<std::mutex> lock(const_cast<std::mutex&>(mutex_));

    if (!IsInitialized()) {
        return;
    }

    DRVGPU_LOG_INFO("OpenCLBackendExternal", "Cleanup...");

    // Очистить внутренние менеджеры
    svm_capabilities_.reset();
    memory_manager_.reset();

    // КРИТИЧЕСКИ ВАЖНО: НЕ уничтожаем внешние объекты!
    if (!owns_resources_) {
        DRVGPU_LOG_INFO("OpenCLBackendExternal", "External resources preserved (not owned)");
        // Просто обнуляем указатели (не вызываем clRelease*)
        context_ = nullptr;
        device_ = nullptr;
        queue_ = nullptr;
    } else {
        DRVGPU_LOG_INFO("OpenCLBackendExternal", "Releasing owned resources...");
        // Вызываем базовый Cleanup только если мы владеем ресурсами
        OpenCLBackend::Cleanup();
    }

    initialized_ = false;
    DRVGPU_LOG_INFO("OpenCLBackendExternal", "Cleanup complete");
}

// ════════════════════════════════════════════════════════════════════════════
// Утилиты для работы с внешними буферами
// ════════════════════════════════════════════════════════════════════════════

void OpenCLBackendExternal::WriteToExternalBuffer(
    cl_mem external_cl_mem,
    const void* host_data,
    size_t size_bytes,
    bool blocking)
{
    if (!external_cl_mem || !host_data) {
        throw std::invalid_argument("WriteToExternalBuffer: null parameters");
    }

    if (!IsInitialized()) {
        throw std::runtime_error("WriteToExternalBuffer: backend not initialized");
    }

    // Используем OpenCL API напрямую
    cl_int err = clEnqueueWriteBuffer(
        external_queue_,
        external_cl_mem,
        blocking ? CL_TRUE : CL_FALSE,
        0,                  // offset
        size_bytes,
        host_data,
        0,                  // num_events_in_wait_list
        nullptr,            // event_wait_list
        nullptr             // event
    );

    if (err != CL_SUCCESS) {
        throw std::runtime_error(
            "WriteToExternalBuffer: clEnqueueWriteBuffer failed with error " + 
            std::to_string(err)
        );
    }
}

void OpenCLBackendExternal::ReadFromExternalBuffer(
    cl_mem external_cl_mem,
    void* host_dest,
    size_t size_bytes,
    bool blocking)
{
    if (!external_cl_mem || !host_dest) {
        throw std::invalid_argument("ReadFromExternalBuffer: null parameters");
    }

    if (!IsInitialized()) {
        throw std::runtime_error("ReadFromExternalBuffer: backend not initialized");
    }

    cl_int err = clEnqueueReadBuffer(
        external_queue_,
        external_cl_mem,
        blocking ? CL_TRUE : CL_FALSE,
        0,                  // offset
        size_bytes,
        host_dest,
        0,                  // num_events_in_wait_list
        nullptr,            // event_wait_list
        nullptr             // event
    );

    if (err != CL_SUCCESS) {
        throw std::runtime_error(
            "ReadFromExternalBuffer: clEnqueueReadBuffer failed with error " + 
            std::to_string(err)
        );
    }
}

void OpenCLBackendExternal::CopyExternalBuffers(
    cl_mem src_cl_mem,
    cl_mem dst_cl_mem,
    size_t size_bytes)
{
    if (!src_cl_mem || !dst_cl_mem) {
        throw std::invalid_argument("CopyExternalBuffers: null buffers");
    }

    if (!IsInitialized()) {
        throw std::runtime_error("CopyExternalBuffers: backend not initialized");
    }

    cl_int err = clEnqueueCopyBuffer(
        external_queue_,
        src_cl_mem,
        dst_cl_mem,
        0,                  // src_offset
        0,                  // dst_offset
        size_bytes,
        0,                  // num_events_in_wait_list
        nullptr,            // event_wait_list
        nullptr             // event
    );

    if (err != CL_SUCCESS) {
        throw std::runtime_error(
            "CopyExternalBuffers: clEnqueueCopyBuffer failed with error " + 
            std::to_string(err)
        );
    }
}

// ════════════════════════════════════════════════════════════════════════════
// Приватные методы
// ════════════════════════════════════════════════════════════════════════════

void OpenCLBackendExternal::ValidateExternalObjects() {
    // Проверка context
    if (!external_context_) {
        throw std::runtime_error("External cl_context is null");
    }

    // Проверка device
    if (!external_device_) {
        throw std::runtime_error("External cl_device_id is null");
    }

    // Проверка queue
    if (!external_queue_) {
        throw std::runtime_error("External cl_command_queue is null");
    }

    // Дополнительная валидация: проверить что device принадлежит context
    cl_context queue_context = nullptr;
    cl_int err = clGetCommandQueueInfo(
        external_queue_,
        CL_QUEUE_CONTEXT,
        sizeof(cl_context),
        &queue_context,
        nullptr
    );

    if (err != CL_SUCCESS) {
        throw std::runtime_error(
            "Failed to query command queue context: error " + std::to_string(err)
        );
    }

    if (queue_context != external_context_) {
        throw std::runtime_error(
            "Command queue context does not match provided context"
        );
    }

    DRVGPU_LOG_INFO("OpenCLBackendExternal", "External objects validated successfully");
}

size_t OpenCLBackendExternal::GetBufferSize(cl_mem buffer) const {
    if (!buffer) {
        return 0;
    }

    size_t size = 0;
    cl_int err = clGetMemObjectInfo(
        buffer,
        CL_MEM_SIZE,
        sizeof(size_t),
        &size,
        nullptr
    );

    if (err != CL_SUCCESS) {
        DRVGPU_LOG_ERROR("OpenCLBackendExternal", "GetBufferSize failed: " + std::to_string(err));
        return 0;
    }

    return size;
}

} // namespace drv_gpu_lib
