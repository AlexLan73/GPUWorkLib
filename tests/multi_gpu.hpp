/**
 * @file example_multi_gpu.cpp
 * @brief Пример использования DrvGPU для Multi-GPU
 *
 * Демонстрирует работу с несколькими GPU через GPUManager.
 * КЛЮЧЕВОЙ пример для Multi-GPU сценариев!
 */

#include "gpu_manager.hpp"
#include "backend_type.hpp"
//#include "load_balancing_strategy.hpp"
#include "DrvGPU/balance_state.hpp"
#include <iostream>
#include <vector>
#include <thread>
#include <chrono>

using namespace drv_gpu_lib;
namespace example_drv_gpu_multi
{
  int run()
  {
    try
    {
      std::cout << "=== DrvGPU Multi-GPU Example ===\n\n";

      // ═══════════════════════════════════════════════════════════════
      // 1. Создать GPUManager и инициализировать все GPU
      // ═══════════════════════════════════════════════════════════════

      std::cout << "Initializing all available GPUs...\n";
      GPUManager manager;
      manager.InitializeAll(BackendType::OPENCL);

      size_t gpu_count = manager.GetGPUCount();
      std::cout << "Found " << gpu_count << " GPU(s)\n\n";

      if (gpu_count == 0)
      {
        std::cerr << "No GPUs available!\n";
        return 1;
      }

      // ═══════════════════════════════════════════════════════════════
      // 2. Вывести информацию обо всех GPU
      // ═══════════════════════════════════════════════════════════════

      std::cout << "--- All GPU Devices ---\n";
      manager.PrintAllDevices();

      // ═══════════════════════════════════════════════════════════════
      // 3. Пример 1: Round-Robin распределение
      // ═══════════════════════════════════════════════════════════════

      std::cout << "\n=== Example 1: Round-Robin Load Balancing ===\n";
      manager.SetLoadBalancingStrategy(LoadBalancingStrategy::ROUND_ROBIN);

      const size_t NUM_TASKS = 10;
      const size_t BUFFER_SIZE = 1024;

      std::cout << "Distributing " << NUM_TASKS << " tasks across GPUs...\n";

      for (size_t i = 0; i < NUM_TASKS; ++i)
      {
        // GetNextGPU() автоматически выбирает следующую GPU (Round-Robin)
        auto &gpu = manager.GetNextGPU();

        std::cout << "Task " << i << " -> GPU " << gpu.GetDeviceIndex()
                  << " (" << gpu.GetDeviceName() << ")\n";

        // Создать буфер на выбранной GPU
        auto &mem_mgr = gpu.GetMemoryManager();
        auto buffer = mem_mgr.CreateBuffer<float>(BUFFER_SIZE);

        // Записать данные
        std::vector<float> data(BUFFER_SIZE, static_cast<float>(i));
        buffer->Write(data);
      }

      // ═══════════════════════════════════════════════════════════════
      // 4. Пример 2: Работа с конкретными GPU
      // ═══════════════════════════════════════════════════════════════

      std::cout << "\n=== Example 2: Explicit GPU Selection ===\n";

      // Явно выбрать GPU 0
      auto &gpu0 = manager.GetGPU(0);
      std::cout << "Using GPU 0: " << gpu0.GetDeviceName() << "\n";

      auto buffer0 = gpu0.GetMemoryManager().CreateBuffer<float>(512);

      // Если есть вторая GPU - использовать её
      if (gpu_count > 1)
      {
        auto &gpu1 = manager.GetGPU(1);
        std::cout << "Using GPU 1: " << gpu1.GetDeviceName() << "\n";

        auto buffer1 = gpu1.GetMemoryManager().CreateBuffer<float>(512);
      }

      // ═══════════════════════════════════════════════════════════════
      // 5. Пример 3: Параллельная обработка на разных GPU
      // ═══════════════════════════════════════════════════════════════

      std::cout << "\n=== Example 3: Parallel Processing ===\n";

      std::vector<std::thread> threads;

      for (size_t i = 0; i < gpu_count; ++i)
      {
        threads.emplace_back([&manager, i]()
                             {
                auto& gpu = manager.GetGPU(i);
                
                std::cout << "Thread " << i << " using GPU " << i 
                         << " (" << gpu.GetDeviceName() << ")\n";
                
                // Симулировать работу
                auto buffer = gpu.GetMemoryManager().CreateBuffer<float>(2048);
                
                std::vector<float> data(2048, static_cast<float>(i * 100));
                buffer->Write(data);
                
                // Симулировать вычисления
                std::this_thread::sleep_for(std::chrono::milliseconds(100));
                
                auto result = buffer->Read();
                
                std::cout << "Thread " << i << " completed (first value: " 
                         << result[0] << ")\n"; });
      }

      // Дождаться завершения всех потоков
      for (auto &thread : threads)
      {
        thread.join();
      }

      std::cout << "All threads completed\n";

      // ═════════════════════════════════════════════════════════════════
      // 6. Синхронизация всех GPU
      // ═══════════════════════════════════════════════════════════════

      std::cout << "\n--- Synchronizing all GPUs ---\n";
      manager.SynchronizeAll();
      std::cout << "All GPUs synchronized\n";

      // ═══════════════════════════════════════════════════════════════
      // 7. Статистика по всем GPU
      // ═══════════════════════════════════════════════════════════════

      std::cout << "\n--- Statistics ---\n";
      manager.PrintStatistics();

      std::cout << "\n=== Multi-GPU Example completed successfully! ===\n";
    }
    catch (const std::exception &e)
    {
      std::cerr << "ERROR: " << e.what() << "\n";
      return 1;
    }

    return 0;
  }
}