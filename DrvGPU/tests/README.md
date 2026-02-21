# DrvGPU/tests — Тесты и примеры

Тесты DrvGPU. Вызов через `all_test.hpp` из `main.cpp`.

## Файлы

| Файл | Namespace | Статус | Описание |
|------|-----------|--------|----------|
| `all_test.hpp` | `drvgpu_all_test` | ✅ Активен | Оркестратор — включить/закомментировать тесты |
| `single_gpu.hpp` | `example_drv_gpu_singl` | ✅ Активен | Single GPU: DrvGPU, MemoryManager, device info |
| `test_services.hpp` | `test_services` | ✅ Активен | ConsoleOutput, AsyncServiceBase stress, ServiceManager (многопоточные) |
| `test_gpu_profiler.hpp` | `test_gpu_profiler` | ✅ Активен | GPUProfiler: Record, агрегация, PrintReport + ExportMarkdown |
| `example_external_context_usage.hpp` | `external_context_example` | 💤 Выключен | Внешний OpenCL-контекст (5 примеров) |

## Как запускать

```cpp
// src/main.cpp
#include "DrvGPU/tests/all_test.hpp"
drvgpu_all_test::run();
```

В `all_test.hpp` раскомментировать нужные тесты:
```cpp
#include "single_gpu.hpp"           // ✅ активен
#include "test_services.hpp"        // ✅ активен
#include "test_gpu_profiler.hpp"    // ✅ активен
// #include "example_external_context_usage.hpp"  // 💤 выключен (примеры)
```

## Покрытие по компонентам

| Компонент | Тест | Требует GPU |
|-----------|------|-------------|
| DrvGPU, MemoryManager, GPUDeviceInfo | `single_gpu.hpp` | ✅ Да |
| GPUProfiler (Record, агрегация, PrintReport, ExportMarkdown) | `test_gpu_profiler.hpp` | ✅ Да (InitializeAll) |
| ConsoleOutput (8 потоков × 50 msg) | `test_services.hpp` | ❌ Нет |
| AsyncServiceBase stress (8 потоков × 1000 msg) | `test_services.hpp` | ❌ Нет |
| ServiceManager (все сервисы вместе) | `test_services.hpp` | ❌ Нет |
| OpenCLBackend (внешний контекст, 5 примеров) | `example_external_context_usage.hpp` | ✅ Да |

## Что проверяют активные тесты

### single_gpu — базовый smoke-тест GPU
1. Инициализация DrvGPU с OpenCL backend (GPU #0)
2. Вывод информации об устройстве (`PrintDeviceInfo`, `GetDeviceInfo`)
3. Создание буфера `1024 × float`
4. Write: Host → Device
5. Read: Device → Host + проверка данных
6. Статистика памяти (`PrintStatistics`)
7. Синхронизация (`Synchronize`)

### test_services — многопоточная инфраструктура
1. **ConsoleOutput**: 8 потоков × 50 сообщений → `[PASS/FAIL]`
2. **AsyncServiceBase stress**: 8 потоков × 1000 msg, latency + throughput
3. **ServiceManager**: InitializeDefaults → StartAll → профилирование × 40 событий → StopAll

### test_gpu_profiler — профилирование GPU
1. **PrintReport**: 100 FFT + 100 Padding + 100 MemOps + 20 ROCm событий
2. Real GPU info через `GPUManager::GetGPUReportInfo()`
3. Экспорт: `Results/Profiler/test_report.md`

## Результаты

- Консоль: `[PASS]` / `[FAIL]` для каждого теста
- Markdown: `Results/Profiler/test_report.md` (от test_gpu_profiler)
