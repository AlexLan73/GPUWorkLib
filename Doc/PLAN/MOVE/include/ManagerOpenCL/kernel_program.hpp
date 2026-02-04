#pragma once

#include "opencl_core.hpp"
#include <CL/cl.h>
#include <string>
#include <memory>
#include <unordered_map>
#include <mutex>

namespace ManagerOpenCL {

// ════════════════════════════════════════════════════════════════════════════
// KernelProgram - Управление OpenCL программами и kernels
// ════════════════════════════════════════════════════════════════════════════

/**
 * @class KernelProgram
 * @brief Обёртка над cl_program с кэшированием и управлением kernels
 *
 * Ответственность:
 * - Компиляция OpenCL программ с обработкой ошибок
 * - Кэширование программ по хешу исходника (избежать перекомпиляции)
 * - Кэширование kernels по имени
 * - Получение информации о kernel
 *
 * Использование:
 * ```cpp
 * auto kernel = engine.LoadKernel("kernel_source.cl", "my_kernel_name");
 * // kernel.GetProgram() вернет cl_program
 * // kernel.GetKernel() вернет cl_kernel
 * ```
 */
class KernelProgram {
public:
    /**
     * @brief Создать программу из исходного кода
     * @param source OpenCL C код
     * @throws std::runtime_error если компиляция не удалась
     */
    explicit KernelProgram(const std::string& source);

    /**
     * @brief Получить или создать kernel по имени
     * @param kernel_name Имя kernel функции в исходнике
     * @return cl_kernel (управляется этим объектом)
     * @throws std::runtime_error если kernel не найден
     */
    cl_kernel GetOrCreateKernel(const std::string& kernel_name);

    /**
     * @brief Получить cl_program
     */
    cl_program GetProgram() const { return program_; }

    /**
     * @brief Проверить, существует ли kernel
     */
    bool HasKernel(const std::string& kernel_name) const;

    /**
     * @brief Получить количество kernels в памяти
     */
    size_t GetKernelCount() const { return kernel_cache_.size(); }

    /**
     * @brief Получить исходный код программы
     */
    const std::string& GetSource() const { return source_; }

    // Деструктор
    ~KernelProgram();

    // Запрет копирования (можно использовать move)
    KernelProgram(const KernelProgram&) = delete;
    KernelProgram& operator=(const KernelProgram&) = delete;

    KernelProgram(KernelProgram&& other) noexcept;
    KernelProgram& operator=(KernelProgram&& other) noexcept;

private:
    cl_program program_;
    std::string source_;
    std::unordered_map<std::string, cl_kernel> kernel_cache_;
    mutable std::mutex cache_mutex_;

    // Компиляция программы (вызывается в конструкторе)
    void CompileProgram();

    // Получить build log при ошибке
    std::string GetBuildLog() const;
};

// ════════════════════════════════════════════════════════════════════════════
// KernelProgramCache - Синглтон для кэширования программ
// ════════════════════════════════════════════════════════════════════════════

/**
 * @class KernelProgramCache
 * @brief Глобальный кэш откомпилированных программ (по хешу исходника)
 *
 * Использование:
 * ```cpp
 * auto program = KernelProgramCache::GetOrCompile(kernel_source);
 * ```
 *
 * Преимущество: Если один и тот же исходник запрашивается дважды,
 * вторая попытка вернет закэшированную программу без перекомпиляции.
 */
class KernelProgramCache {
public:
    /**
     * @brief Получить или откомпилировать программу
     * @param source OpenCL C код
     * @return Shared pointer на KernelProgram (управляется кэшем)
     */
    static std::shared_ptr<KernelProgram> GetOrCompile(const std::string& source);

    /**
     * @brief Получить статистику кэша
     */
    static std::string GetCacheStatistics();

    /**
     * @brief Очистить кэш
     */
    static void Clear();

    /**
     * @brief Получить размер кэша (кол-во программ)
     */
    static size_t GetCacheSize();

private:
    static std::unordered_map<std::string, std::shared_ptr<KernelProgram>> cache_;
    static std::mutex cache_mutex_;
    static size_t cache_hits_;
    static size_t cache_misses_;

    KernelProgramCache() = delete;
};

}  // namespace ManagerOpenCL
