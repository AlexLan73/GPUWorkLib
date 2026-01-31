#pragma once

/**
 * @file hybrid_buffer.hpp
 * @brief Гибридный буфер с автоматическим выбором стратегии (SVM/Regular)
 * 
 * HybridBuffer автоматически выбирает оптимальную стратегию на основе:
 * - Возможностей GPU (SVM support)
 * - Размера буфера
 * - Паттерна использования
 * 
 * BufferFactory - фабрика для создания буферов с правильной стратегией.
 * 
 * @author Codo (AI Assistant)
 * @date 2026-01-19
 */

#include "i_memory_buffer.hpp"
#include "svm_buffer.hpp"
#include "regular_buffer.hpp"
#include "svm_capabilities.hpp"
#include "memory_type.hpp"
#include <CL/cl.h>
#include <memory>
#include <mutex>
#include <iostream>
#include <iomanip>

namespace ManagerOpenCL {

// ════════════════════════════════════════════════════════════════════════════
// Struct: BufferConfig - конфигурация для создания буфера
// ════════════════════════════════════════════════════════════════════════════

/**
 * @struct BufferConfig
 * @brief Настройки для BufferFactory
 * 
 * Позволяет настроить пороговые значения для автоматического
 * выбора между SVM и традиционными буферами.
 */
struct BufferConfig {
    // ═══════════════════════════════════════════════════════════════
    // Пороговые значения (в байтах)
    // ═══════════════════════════════════════════════════════════════
    
    /// Буферы меньше этого размера всегда используют Regular
    /// (SVM overhead не оправдан для маленьких буферов)
    size_t small_buffer_threshold = 1 * 1024 * 1024;  // 1 MB
    
    /// Буферы больше этого размера предпочитают SVM
    /// (zero-copy даёт преимущество)
    size_t large_buffer_threshold = 64 * 1024 * 1024;  // 64 MB
    
    // ═══════════════════════════════════════════════════════════════
    // Флаги поведения
    // ═══════════════════════════════════════════════════════════════
    
    /// Принудительно использовать SVM если доступен
    bool force_svm = false;
    
    /// Принудительно использовать Regular (отключить SVM)
    bool force_regular = false;
    
    /// Предпочитать coarse-grain SVM (более совместим)
    bool prefer_coarse_grain = true;
    
    /// Выводить диагностику при создании буферов
    bool verbose = false;
    
    // ═══════════════════════════════════════════════════════════════
    // Статические конфигурации
    // ═══════════════════════════════════════════════════════════════
    
    /**
     * @brief Конфигурация по умолчанию (автовыбор)
     */
    static BufferConfig Default() {
        return BufferConfig{};
    }
    
    /**
     * @brief Конфигурация для максимальной производительности
     * (агрессивное использование SVM)
     */
    static BufferConfig Performance() {
        BufferConfig config;
        config.small_buffer_threshold = 256 * 1024;  // 256 KB
        config.large_buffer_threshold = 16 * 1024 * 1024;  // 16 MB
        config.prefer_coarse_grain = false;  // Fine-grain если доступен
        return config;
    }
    
    /**
     * @brief Конфигурация для совместимости
     * (предпочитать Regular buffers)
     */
    static BufferConfig Compatibility() {
        BufferConfig config;
        config.small_buffer_threshold = 256 * 1024 * 1024;  // 256 MB
        config.prefer_coarse_grain = true;
        return config;
    }
    
    /**
     * @brief Конфигурация только SVM
     */
    static BufferConfig SVMOnly() {
        BufferConfig config;
        config.force_svm = true;
        return config;
    }
    
    /**
     * @brief Конфигурация только Regular
     */
    static BufferConfig RegularOnly() {
        BufferConfig config;
        config.force_regular = true;
        return config;
    }
};

// ════════════════════════════════════════════════════════════════════════════
// Class: BufferFactory - Фабрика для создания буферов
// ════════════════════════════════════════════════════════════════════════════

/**
 * @class BufferFactory
 * @brief Фабрика для создания буферов с автоматическим выбором стратегии
 * 
 * Центральная точка для создания GPU буферов. Автоматически определяет
 * возможности устройства и выбирает оптимальную стратегию.
 * 
 * Паттерн: Factory Method + Strategy
 * 
 * @code
 * // Инициализация
 * BufferFactory factory(context, queue, device);
 * 
 * // Создание буфера (автовыбор стратегии)
 * auto buffer = factory.Create(1024 * 1024);  // 1M elements
 * 
 * // Создание с конкретной стратегией
 * auto svm_buffer = factory.Create(1024, MemoryStrategy::SVM_COARSE_GRAIN);
 * 
 * // Создание с данными
 * auto data_buffer = factory.CreateWithData(my_data);
 * @endcode
 */
class BufferFactory {
public:
    // ═══════════════════════════════════════════════════════════════
    // Конструкторы
    // ═══════════════════════════════════════════════════════════════
    
    /**
     * @brief Создать фабрику
     * @param context OpenCL context
     * @param queue Command queue
     * @param device Device ID (для проверки capabilities)
     * @param config Конфигурация (опционально)
     */
    BufferFactory(
        cl_context context,
        cl_command_queue queue,
        cl_device_id device,
        const BufferConfig& config = BufferConfig::Default()
    );
    
    // ═══════════════════════════════════════════════════════════════
    // Создание буферов
    // ═══════════════════════════════════════════════════════════════
    
    /**
     * @brief Создать буфер с автоматическим выбором стратегии
     * @param num_elements Количество complex<float> элементов
     * @param mem_type Тип памяти
     * @param hint Подсказка по использованию
     * @return unique_ptr на IMemoryBuffer
     */
    std::unique_ptr<IMemoryBuffer> Create(
        size_t num_elements,
        MemoryType mem_type = MemoryType::GPU_READ_WRITE,
        const BufferUsageHint& hint = BufferUsageHint::Default()
    );
    
    /**
     * @brief Создать буфер с конкретной стратегией
     * @param num_elements Количество элементов
     * @param strategy Стратегия (REGULAR, SVM_COARSE, etc.)
     * @param mem_type Тип памяти
     * @return unique_ptr на IMemoryBuffer
     */
    std::unique_ptr<IMemoryBuffer> CreateWithStrategy(
        size_t num_elements,
        MemoryStrategy strategy,
        MemoryType mem_type = MemoryType::GPU_READ_WRITE
    );
    
    /**
     * @brief Создать буфер с начальными данными
     * @param data Начальные данные
     * @param mem_type Тип памяти
     * @return unique_ptr на IMemoryBuffer
     */
    std::unique_ptr<IMemoryBuffer> CreateWithData(
        const ComplexVector& data,
        MemoryType mem_type = MemoryType::GPU_READ_WRITE
    );
    
    /**
     * @brief Обернуть внешний cl_mem буфер
     * @param external_buffer Внешний буфер
     * @param num_elements Количество элементов
     * @param mem_type Тип памяти
     * @return unique_ptr на IMemoryBuffer (non-owning)
     */
    std::unique_ptr<IMemoryBuffer> WrapExternal(
        cl_mem external_buffer,
        size_t num_elements,
        MemoryType mem_type = MemoryType::GPU_READ_WRITE
    );
    
    // ═══════════════════════════════════════════════════════════════
    // Информация и конфигурация
    // ═══════════════════════════════════════════════════════════════
    
    /**
     * @brief Получить SVM capabilities устройства
     */
    const SVMCapabilities& GetCapabilities() const { return capabilities_; }
    
    /**
     * @brief Получить текущую конфигурацию
     */
    const BufferConfig& GetConfig() const { return config_; }
    
    /**
     * @brief Установить новую конфигурацию
     */
    void SetConfig(const BufferConfig& config) { config_ = config; }
    
    /**
     * @brief Определить стратегию для данного размера
     * @param size_bytes Размер в байтах
     * @param hint Подсказка по использованию
     * @return Рекомендуемая стратегия
     */
    MemoryStrategy DetermineStrategy(
        size_t size_bytes,
        const BufferUsageHint& hint = BufferUsageHint::Default()
    ) const;
    
    /**
     * @brief Вывести информацию о фабрике
     */
    void PrintInfo() const;
    
    /**
     * @brief Получить статистику созданных буферов
     */
    std::string GetStatistics() const;

private:
    // ═══════════════════════════════════════════════════════════════
    // Члены класса
    // ═══════════════════════════════════════════════════════════════
    
    cl_context       context_;
    cl_command_queue queue_;
    cl_device_id     device_;
    SVMCapabilities  capabilities_;
    BufferConfig     config_;
    
    // Статистика
    mutable std::mutex stats_mutex_;
    size_t total_buffers_created_   = 0;
    size_t svm_buffers_created_     = 0;
    size_t regular_buffers_created_ = 0;
    size_t total_bytes_allocated_   = 0;
};

// ════════════════════════════════════════════════════════════════════════════
// Реализация BufferFactory
// ════════════════════════════════════════════════════════════════════════════

inline BufferFactory::BufferFactory(
    cl_context context,
    cl_command_queue queue,
    cl_device_id device,
    const BufferConfig& config)
    : context_(context),
      queue_(queue),
      device_(device),
      config_(config) {
    
    if (!context_ || !queue_ || !device_) {
        throw std::invalid_argument("BufferFactory: all parameters must not be null");
    }
    
    // Определить capabilities устройства
    capabilities_ = SVMCapabilities::Query(device_);
    
    if (config_.verbose) {
        std::cout << capabilities_.ToString();
    }
}

inline MemoryStrategy BufferFactory::DetermineStrategy(
    size_t size_bytes,
    const BufferUsageHint& hint) const {
    
    // 1. Принудительные режимы
    if (config_.force_regular) {
        return MemoryStrategy::REGULAR_BUFFER;
    }
    
    if (config_.force_svm) {
        if (capabilities_.HasAnySVM()) {
            return config_.prefer_coarse_grain 
                ? MemoryStrategy::SVM_COARSE_GRAIN 
                : capabilities_.GetBestSVMStrategy();
        }
        // Fallback если SVM недоступен
        return MemoryStrategy::REGULAR_BUFFER;
    }
    
    // 2. GPU-only буферы - всегда Regular (нет смысла в SVM)
    if (hint.gpu_only) {
        return MemoryStrategy::REGULAR_BUFFER;
    }
    
    // 3. Маленькие буферы - Regular (SVM overhead не оправдан)
    if (size_bytes < config_.small_buffer_threshold) {
        return MemoryStrategy::REGULAR_BUFFER;
    }
    
    // 4. Средние и большие буферы - SVM если доступен
    if (capabilities_.HasAnySVM()) {
        // Для буферов с частым обменом хост-GPU предпочитаем SVM
        if (hint.frequent_host_read || hint.frequent_host_write) {
            if (capabilities_.fine_grain_buffer && !config_.prefer_coarse_grain) {
                return MemoryStrategy::SVM_FINE_GRAIN;
            }
            if (capabilities_.coarse_grain_buffer) {
                return MemoryStrategy::SVM_COARSE_GRAIN;
            }
        }
        
        // Большие буферы - SVM для zero-copy
        if (size_bytes >= config_.large_buffer_threshold) {
            if (config_.prefer_coarse_grain && capabilities_.coarse_grain_buffer) {
                return MemoryStrategy::SVM_COARSE_GRAIN;
            }
            return capabilities_.GetBestSVMStrategy();
        }
        
        // Средние буферы - coarse-grain SVM если доступен
        if (capabilities_.coarse_grain_buffer) {
            return MemoryStrategy::SVM_COARSE_GRAIN;
        }
    }
    
    // 5. Fallback
    return MemoryStrategy::REGULAR_BUFFER;
}

inline std::unique_ptr<IMemoryBuffer> BufferFactory::Create(
    size_t num_elements,
    MemoryType mem_type,
    const BufferUsageHint& hint) {
    
    size_t size_bytes = num_elements * sizeof(ComplexFloat);
    MemoryStrategy strategy = DetermineStrategy(size_bytes, hint);
    
    return CreateWithStrategy(num_elements, strategy, mem_type);
}

inline std::unique_ptr<IMemoryBuffer> BufferFactory::CreateWithStrategy(
    size_t num_elements,
    MemoryStrategy strategy,
    MemoryType mem_type) {
    
    std::unique_ptr<IMemoryBuffer> buffer;
    size_t size_bytes = num_elements * sizeof(ComplexFloat);
    
    // Validate strategy availability
    if (strategy == MemoryStrategy::AUTO) {
        strategy = DetermineStrategy(size_bytes);
    }
    
    // Fallback если SVM недоступен
    bool need_svm = (strategy == MemoryStrategy::SVM_COARSE_GRAIN ||
                     strategy == MemoryStrategy::SVM_FINE_GRAIN ||
                     strategy == MemoryStrategy::SVM_FINE_SYSTEM);
    
    if (need_svm && !capabilities_.HasAnySVM()) {
        if (config_.verbose) {
            std::cout << "[BufferFactory] SVM requested but not available, falling back to Regular\n";
        }
        strategy = MemoryStrategy::REGULAR_BUFFER;
    }
    
    // Create buffer
    try {
        if (strategy == MemoryStrategy::REGULAR_BUFFER) {
            buffer = std::make_unique<RegularBuffer>(context_, queue_, num_elements, mem_type);
            
            std::lock_guard<std::mutex> lock(stats_mutex_);
            regular_buffers_created_++;
        } else {
            buffer = std::make_unique<SVMBuffer>(context_, queue_, num_elements, strategy, mem_type);
            
            std::lock_guard<std::mutex> lock(stats_mutex_);
            svm_buffers_created_++;
        }
        
        {
            std::lock_guard<std::mutex> lock(stats_mutex_);
            total_buffers_created_++;
            total_bytes_allocated_ += size_bytes;
        }
        
        if (config_.verbose) {
            std::cout << "[BufferFactory] Created " << MemoryStrategyToString(strategy) 
                      << " buffer: " << num_elements << " elements ("
                      << (size_bytes / (1024.0 * 1024.0)) << " MB)\n";
        }
        
    } catch (const std::exception& e) {
        // Fallback to Regular if SVM failed
        if (strategy != MemoryStrategy::REGULAR_BUFFER) {
            if (config_.verbose) {
                std::cout << "[BufferFactory] SVM creation failed (" << e.what() 
                          << "), falling back to Regular\n";
            }
            buffer = std::make_unique<RegularBuffer>(context_, queue_, num_elements, mem_type);
            
            std::lock_guard<std::mutex> lock(stats_mutex_);
            regular_buffers_created_++;
            total_buffers_created_++;
            total_bytes_allocated_ += size_bytes;
        } else {
            throw;
        }
    }
    
    return buffer;
}

inline std::unique_ptr<IMemoryBuffer> BufferFactory::CreateWithData(
    const ComplexVector& data,
    MemoryType mem_type) {
    
    auto buffer = Create(data.size(), mem_type);
    buffer->Write(data);
    return buffer;
}

inline std::unique_ptr<IMemoryBuffer> BufferFactory::WrapExternal(
    cl_mem external_buffer,
    size_t num_elements,
    MemoryType mem_type) {
    
    return std::make_unique<RegularBuffer>(
        context_, queue_, external_buffer, num_elements, mem_type
    );
}

inline void BufferFactory::PrintInfo() const {
    std::cout << "\n" << std::string(70, '═') << "\n";
    std::cout << "BufferFactory Configuration\n";
    std::cout << std::string(70, '═') << "\n\n";
    
    std::cout << capabilities_.ToString();
    
    std::cout << "\nThresholds:\n";
    std::cout << "  Small buffer: < " 
              << (config_.small_buffer_threshold / (1024.0 * 1024.0)) << " MB\n";
    std::cout << "  Large buffer: >= " 
              << (config_.large_buffer_threshold / (1024.0 * 1024.0)) << " MB\n";
    
    std::cout << "\nFlags:\n";
    std::cout << "  Force SVM:       " << (config_.force_svm ? "YES" : "NO") << "\n";
    std::cout << "  Force Regular:   " << (config_.force_regular ? "YES" : "NO") << "\n";
    std::cout << "  Prefer Coarse:   " << (config_.prefer_coarse_grain ? "YES" : "NO") << "\n";
    
    std::cout << "\n" << std::string(70, '═') << "\n";
}

inline std::string BufferFactory::GetStatistics() const {
    std::lock_guard<std::mutex> lock(stats_mutex_);
    
    std::ostringstream oss;
    oss << "\n" << std::string(50, '─') << "\n";
    oss << "BufferFactory Statistics\n";
    oss << std::string(50, '─') << "\n";
    oss << std::left << std::setw(25) << "Total buffers:" << total_buffers_created_ << "\n";
    oss << std::left << std::setw(25) << "SVM buffers:" << svm_buffers_created_ << "\n";
    oss << std::left << std::setw(25) << "Regular buffers:" << regular_buffers_created_ << "\n";
    oss << std::left << std::setw(25) << "Total allocated:" 
        << std::fixed << std::setprecision(2) 
        << (total_bytes_allocated_ / (1024.0 * 1024.0)) << " MB\n";
    
    if (total_buffers_created_ > 0) {
        double svm_percent = 100.0 * svm_buffers_created_ / total_buffers_created_;
        oss << std::left << std::setw(25) << "SVM usage:" 
            << std::fixed << std::setprecision(1) << svm_percent << "%\n";
    }
    
    oss << std::string(50, '─') << "\n";
    return oss.str();
}

} // namespace ManagerOpenCL

