#include "opencl_backend_external.hpp"
#include "../../logger/logger.hpp"

namespace drv_gpu_lib {

// ════════════════════════════════════════════════════════════════════════════
// Конструктор и деструктор
// ════════════════════════════════════════════════════════════════════════════

/**
 * @brief Создать external backend
 * 
 * ✅ КРИТИЧЕСКИ ВАЖНО: Устанавливает owns_resources_ = false ДО инициализации
 * 
 * По умолчанию:
 * - owns_resources_ = false (НЕ освобождает ресурсы)
 * - initialized_ = false (требуется InitializeFromExternalContext)
 */
OpenCLBackendExternal::OpenCLBackendExternal()
    : OpenCLBackend() {
    
    // ═══════════════════════════════════════════════════════════════════════
    // ✅ КРИТИЧЕСКИ ВАЖНО: Устанавливаем non-owning режим
    // ═══════════════════════════════════════════════════════════════════════
    // Родительский конструктор установил owns_resources_ = true
    // Переопределяем на false для external контекста
    owns_resources_ = false;
    
    DRVGPU_LOG_INFO("OpenCLBackendExternal", 
        "Created in non-owning mode (owns_resources = false)");
}

/**
 * @brief Деструктор
 * 
 * Родительский деструктор вызовет ~OpenCLBackend() → Cleanup()
 * Cleanup() проверит owns_resources_ = false и НЕ освободит ресурсы
 */
OpenCLBackendExternal::~OpenCLBackendExternal() {
    // Родительский деструктор вызовет Cleanup()
    // Cleanup() увидит owns_resources_ = false и не освободит контекст/queue
    DRVGPU_LOG_INFO("OpenCLBackendExternal", 
        "Destructor called - parent will handle cleanup (non-owning)");
}

// ════════════════════════════════════════════════════════════════════════════
// Инициализация из внешнего контекста
// ════════════════════════════════════════════════════════════════════════════

/**
 * @brief Инициализировать из существующего OpenCL контекста
 * 
 * Процесс:
 * 1. Валидация входных параметров
 * 2. ✅ Подтверждаем owns_resources_ = false
 * 3. Сохраняем внешние дескрипторы БЕЗ clRetain*
 * 4. Инициализируем SVM capabilities
 * 5. Создаём MemoryManager
 * 6. Устанавливаем флаг initialized_
 * 
 * @param external_context Ваш cl_context
 * @param external_device Ваш cl_device_id  
 * @param external_queue Ваш cl_command_queue
 */
void OpenCLBackendExternal::InitializeFromExternalContext(
    cl_context external_context,
    cl_device_id external_device,
    cl_command_queue external_queue) {
    
    std::lock_guard<std::mutex> lock(mutex_);
    
    // ═══════════════════════════════════════════════════════════════════════
    // Валидация входных параметров
    // ═══════════════════════════════════════════════════════════════════════
    if (!external_context || !external_device || !external_queue) {
        throw std::invalid_argument(
            "OpenCLBackendExternal::InitializeFromExternalContext - "
            "All parameters (context, device, queue) must be non-null"
        );
    }
    
    DRVGPU_LOG_INFO("OpenCLBackendExternal", 
        "Initializing from external OpenCL context");
    
    // ═══════════════════════════════════════════════════════════════════════
    // ✅ КРИТИЧЕСКИ ВАЖНО: Подтверждаем non-owning режим
    // ═══════════════════════════════════════════════════════════════════════
    // Должно быть false из конструктора, но явно устанавливаем для надёжности
    owns_resources_ = false;
    
    DRVGPU_LOG_INFO("OpenCLBackendExternal", 
        "owns_resources_ = false (external resources will NOT be released)");
    
    // ═══════════════════════════════════════════════════════════════════════
    // Сохраняем внешние дескрипторы БЕЗ clRetain*
    // ═══════════════════════════════════════════════════════════════════════
    // ✅ НЕ вызываем clRetainContext/clRetainDevice/clRetainCommandQueue!
    // Внешний код управляет lifetime этих объектов
    
    context_ = external_context;
    device_ = external_device;
    queue_ = external_queue;
    
    DRVGPU_LOG_INFO("OpenCLBackendExternal", 
        "External OpenCL handles saved (context, device, queue) - NON-OWNING");
    
    // ═══════════════════════════════════════════════════════════════════════
    // Инициализируем SVM capabilities
    // ═══════════════════════════════════════════════════════════════════════
    svm_capabilities_ = std::make_unique<SVMCapabilities>(
        SVMCapabilities::Query(device_)
    );
    
    DRVGPU_LOG_INFO("OpenCLBackendExternal", "SVM capabilities initialized");
    
    // ═══════════════════════════════════════════════════════════════════════
    // Инициализируем MemoryManager
    // ═══════════════════════════════════════════════════════════════════════
    memory_manager_ = std::make_unique<MemoryManager>(this);
    
    DRVGPU_LOG_INFO("OpenCLBackendExternal", "MemoryManager initialized");
    
    // ═══════════════════════════════════════════════════════════════════════
    // Завершение
    // ═══════════════════════════════════════════════════════════════════════
    initialized_ = true;
    device_index_ = 0;  // External контекст = виртуальный device 0
    
    DRVGPU_LOG_INFO("OpenCLBackendExternal", 
        "✅ Successfully initialized from external OpenCL context (owns_resources = false)");
    DRVGPU_LOG_INFO("OpenCLBackendExternal", 
        "⚠️  External code MUST release context/device/queue after use!");
}

// ════════════════════════════════════════════════════════════════════════════
// Блокировка обычной инициализации
// ════════════════════════════════════════════════════════════════════════════

/**
 * @brief ЗАБЛОКИРОВАНО для external backend
 * 
 * External backend должен использовать InitializeFromExternalContext(),
 * а не обычную Initialize(device_index).
 */
void OpenCLBackendExternal::Initialize(int device_index) {
    (void)device_index;
    
    throw std::runtime_error(
        "OpenCLBackendExternal::Initialize(device_index) is not supported.\n"
        "Use InitializeFromExternalContext(context, device, queue) instead."
    );
}

} // namespace drv_gpu_lib
