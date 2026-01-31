#pragma once

#include "opencl_core.hpp"
#include <CL/cl.h>
#include <memory>
#include <vector>
#include <mutex>
#include <atomic>

namespace ManagerOpenCL {

// ════════════════════════════════════════════════════════════════════════════
// CommandQueuePool - Управление пулом command queues для асинхронности
// ════════════════════════════════════════════════════════════════════════════

/**
 * @class CommandQueuePool
 * @brief Синглтон, управляющий пулом OpenCL command queues
 *
 * Ответственность:
 * - Создание N command queues (обычно = количество CPU ядер)
 * - Асинхронное выполнение (CL_QUEUE_OUT_OF_ORDER_EXEC_MODE_ENABLE)
 * - Балансировка очередей (round-robin)
 * - Thread-safe доступ к очередям
 *
 * Использование:
 * ```cpp
 * CommandQueuePool::Initialize(4);  // 4 очереди
 * auto queue = CommandQueuePool::GetNextQueue();
 * clEnqueueNDRangeKernel(queue, ...);
 * ```
 *
 * Особенности:
 * - Каждая очередь может работать независимо
 * - Асинхронное выполнение позволяет overlapping операций
 * - Round-robin выбор распределяет нагрузку
 */
class CommandQueuePool {
public:
    // ═══════════════════════════════════════════════════════════════
    // Инициализация
    // ═══════════════════════════════════════════════════════════════

    /**
     * @brief Инициализировать пулл очередей
     * @param num_queues Количество очередей (если 0, используется кол-во ядер)
     * @throws std::runtime_error если не удалось создать очереди
     */
    static void Initialize(size_t num_queues = 0);

    /**
     * @brief Проверить инициализацию
     */
    static bool IsInitialized();

    /**
     * @brief Очистка ресурсов (опционально, вызывается в деструкторе)
     */
    static void Cleanup();

    // ═══════════════════════════════════════════════════════════════
    // Получение очередей
    // ═══════════════════════════════════════════════════════════════

    /**
     * @brief Получить следующую очередь (round-robin)
     * @return cl_command_queue (управляется пулом)
     */
    static cl_command_queue GetNextQueue();

    /**
     * @brief Получить очередь по индексу
     * @param index Индекс (0 до num_queues-1)
     * @return cl_command_queue
     */
    static cl_command_queue GetQueue(size_t index);

    /**
     * @brief Получить случайную очередь
     */
    static cl_command_queue GetRandomQueue();

    /**
     * @brief Получить текущую очередь (для текущего потока)
     */
    static cl_command_queue GetCurrentQueue();

    // ═══════════════════════════════════════════════════════════════
    // Синхронизация
    // ═══════════════════════════════════════════════════════════════

    /**
     * @brief Ждать завершения всех очередей
     */
    static void FinishAll();

    /**
     * @brief Ждать завершения конкретной очереди
     */
    static void FinishQueue(size_t index);

    /**
     * @brief Flush всех очередей (не ждёт завершения)
     */
    static void FlushAll();

    // ═══════════════════════════════════════════════════════════════
    // Информация о пулле
    // ═══════════════════════════════════════════════════════════════

    /**
     * @brief Получить размер пула (количество очередей)
     */
    static size_t GetPoolSize();

    /**
     * @brief Получить текущий индекс очереди
     */
    static size_t GetCurrentQueueIndex();

    /**
     * @brief Получить статистику (load balancing)
     */
    static std::string GetStatistics();



private:
    // ═══════════════════════════════════════════════════════════════
    // Singleton реализация
    // ═══════════════════════════════════════════════════════════════

    CommandQueuePool() = default;
    ~CommandQueuePool() = default;

    CommandQueuePool(const CommandQueuePool&) = delete;
    CommandQueuePool& operator=(const CommandQueuePool&) = delete;

    static std::unique_ptr<CommandQueuePool> instance_;
    static bool initialized_;
    static std::mutex initialization_mutex_;

    // ═══════════════════════════════════════════════════════════════
    // Члены класса (статические для singleton)
    // ═══════════════════════════════════════════════════════════════

    static std::vector<cl_command_queue> queues_;
    static std::atomic<size_t> current_index_;
    static std::vector<size_t> queue_usage_;
    static std::mutex mutex_;
    static size_t queue_counter_;

    // ═══════════════════════════════════════════════════════════════
    // Приватные методы (статические)
    // ═══════════════════════════════════════════════════════════════

    static void CreateQueues(size_t num_queues);
    static void ReleaseQueues();
    size_t GetLeastUsedQueueIndex();



};

}  // namespace ManagerOpenCL
