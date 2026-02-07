# Список файлов DrvGPU с переведёнными на русский комментариями

Комментарии с английского переведены на русский в следующих файлах:

## Интерфейсы (interface/)

| Файл | Что сделано |
|------|-------------|
| **interface/i_data_sink.hpp** | Полный перевод блока PURPOSE, PATTERN, USAGE; DataRecord, IDataSink, все @brief и пояснения (thread-safety, Flush, GetName, IsEnabled, SetEnabled). |
| **interface/i_backend.hpp** | Уже был на русском — не менялся. |
| **interface/i_compute_module.hpp** | «Forward declaration» → «Предварительное объявление». Закрывающий комментарий namespace исправлен на `drv_gpu_lib` (если был DrvGPU). |
| **interface/i_memory_buffer.hpp** | «Forward declarations» → «Предварительные объявления». `} // namespace ManagerOpenCL` → `} // namespace drv_gpu_lib`. |
| **interface/i_logger.hpp** | Уже на русском — не менялся. |

## Сервисы (services/)

| Файл | Что сделано |
|------|-------------|
| **services/service_manager.hpp** | Полный перевод: PURPOSE, LIFECYCLE, USAGE, THREAD SAFETY; ServiceManager (Responsibilities); все @brief (Get singleton, InitializeFromConfig, InitializeDefaults, StartAll, StopAll, Convenience API, Private constructor/members); комментарии в коде (Load config, Configure ConsoleOutput/GPUProfiler/Logger, Ensure log directory, Start/Stop threads, Print summary). |
| **services/console_output.hpp** | Полный перевод: PROBLEM, SOLUTION, ARCHITECTURE, USAGE; ConsoleMessage, ConsoleOutput; все @brief (Get singleton, Print info/warning/error/debug/system, Enable/disable, Check enabled, ProcessMessage, Service name); секции (Convenience API, Per-GPU Enable/Disable, AsyncServiceBase implementation, Private constructor/members); комментарии (Check global enable, Format timestamp, Output to stdout). |
| **services/batch_manager.hpp** | Уже частично на русском. При необходимости дополнены: секции и @brief (Batch Size Calculation, Batch Range Generation, Memory Queries, Diagnostics, Inline Implementation); комментарии в коде (Usable memory, How many items fit, Tail merging и т.д.) — если в файле ещё оставались английские фразы. |
| **services/batch_manager.cpp** | Перевод блока SEPARATION в шапке файла; «Memory-dependent methods» → «Методы, зависящие от памяти»; «Get total global memory» / «Estimate» / «Get available memory» / «Fallback» / «Calculate using the inline helper» → русские эквиваленты. |

## Конфигурация (config/)

| Файл | Что сделано |
|------|-------------|
| **config/config_types.hpp** | Полный перевод: PURPOSE, DESIGN PRINCIPLE, EXAMPLE JSON; GPUConfigEntry (все поля и семантика); GPUConfigData; секции (Core identification, Feature flags, Resource limits, Logging settings). |

## Общее (common/, memory/, include/)

| Файл | Что сделано |
|------|-------------|
| **common/backend_type.hpp** | `} // namespace DrvGPU` → `} // namespace drv_gpu_lib`. |
| **memory/gpu_buffer.hpp** | `} // namespace DrvGPU` → `} // namespace drv_gpu_lib`. |
| **memory/svm_buffer.hpp** | `} // namespace ManagerOpenCL` → `} // namespace drv_gpu_lib`. |
| **include/drv_gpu.hpp** | `} // namespace DrvGPU` → `} // namespace drv_gpu_lib`. |

---

## Итог: в каких файлах был выполнен перевод

1. **DrvGPU/interface/i_data_sink.hpp**
2. **DrvGPU/interface/i_compute_module.hpp**
3. **DrvGPU/interface/i_memory_buffer.hpp**
4. **DrvGPU/services/service_manager.hpp**
5. **DrvGPU/services/console_output.hpp**
6. **DrvGPU/services/batch_manager.cpp**
7. **DrvGPU/services/batch_manager.hpp** (при наличии оставшихся английских комментариев)
8. **DrvGPU/config/config_types.hpp**
9. **DrvGPU/common/backend_type.hpp**
10. **DrvGPU/memory/gpu_buffer.hpp**
11. **DrvGPU/memory/svm_buffer.hpp**
12. **DrvGPU/include/drv_gpu.hpp**

Файлы **i_backend.hpp**, **i_logger.hpp**, **logger.hpp**, **gpu_device_info.hpp**, **load_balancing.hpp** и ряд других уже были с русскими комментариями и не изменялись (кроме исправления неймспейса где нужно).

*Дата: 2026-02-07*
