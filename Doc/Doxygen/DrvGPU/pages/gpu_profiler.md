@page drvgpu_profiler GPUProfiler --- Профилирование GPU

@tableofcontents

@section profiler_overview Назначение

@ref drv_gpu_lib::GPUProfiler "GPUProfiler" --- асинхронный синглтон-сервис
для централизованного сбора, агрегации и экспорта данных профилирования GPU.

Файл: `DrvGPU/services/gpu_profiler.hpp`

```
Модуль GPU --> Record(gpu_id, "FFT", "Kernel", data) --> Enqueue() --+
                                                                      |
                                                               [Рабочий поток]
                                                                      |
                                                         Агрегация (min/max/avg)
                                                         Экспорт: JSON / Markdown / stdout
```

@note GPUProfiler наследует `AsyncServiceBase<ProfilingMessage>`.
Вызов `Record()` неблокирующий --- потоки GPU не задерживаются.

---

@section profiler_lifecycle Жизненный цикл

@code{.cpp}
// 1. Запуск (обычно через ServiceManager)
auto& profiler = drv_gpu_lib::GPUProfiler::GetInstance();
profiler.Start();

// 2. Запись событий из любого потока
profiler.Record(0, "AntennaFFT", "FFT_Execute", opencl_data);
profiler.Record(0, "AntennaFFT", "Padding_Kernel", opencl_data);
profiler.Record(1, "VectorOps", "VectorAdd", rocm_data);

// 3. Получение результатов
profiler.PrintReport();               // Таблица в stdout
profiler.ExportJSON("report.json");   // JSON-файл
profiler.ExportMarkdown("report.md"); // Markdown-файл

// 4. Остановка
profiler.Stop();
@endcode

@warning Обычно запуск/остановка выполняется через
@ref drv_gpu_lib::ServiceManager "ServiceManager", а не напрямую.

---

@section profiler_setgpuinfo SetGPUInfo --- информация о GPU в отчёте

@warning **Вызывайте `SetGPUInfo()` ПЕРЕД `profiler.Start()`!**
Без этого в отчёте будет "Unknown GPU" и "[нет информации о драйверах]".

@code{.cpp}
auto& profiler = drv_gpu_lib::GPUProfiler::GetInstance();

// Заполнение информации о GPU
drv_gpu_lib::GPUReportInfo info;
info.gpu_name = "AMD Radeon RX 9070 XT";
info.global_mem_mb = 16384;

// Информация о драйверах (vector<map<string,string>>)
std::map<std::string, std::string> rocm_drv;
rocm_drv["driver_type"] = "ROCm";
rocm_drv["version"] = "7.2.0";
rocm_drv["driver_version"] = "6.10.5";
rocm_drv["hip_version"] = "6.2.41134";
info.drivers.push_back(rocm_drv);

std::map<std::string, std::string> ocl_drv;
ocl_drv["driver_type"] = "OpenCL";
ocl_drv["version"] = "3.0";
ocl_drv["driver_version"] = "3614.0";
ocl_drv["platform_name"] = "AMD Accelerated Parallel Processing";
info.drivers.push_back(ocl_drv);

// Установка
profiler.SetGPUInfo(0, info);

// Теперь запуск
profiler.Start();
@endcode

@note При использовании `GPUManager::InitializeAll()` информация передаётся
автоматически (auto-fill).

---

@section profiler_record Record --- запись событий

Два варианта `Record()` для разных бэкендов:

@subsection profiler_record_opencl OpenCL (5 полей cl_profiling_info)

@code{.cpp}
drv_gpu_lib::OpenCLProfilingData data;
data.queued_ns   = queued;    // CL_PROFILING_COMMAND_QUEUED
data.submit_ns   = submit;    // CL_PROFILING_COMMAND_SUBMIT
data.start_ns    = start;     // CL_PROFILING_COMMAND_START
data.end_ns      = end;       // CL_PROFILING_COMMAND_END
data.complete_ns = complete;  // CL_PROFILING_COMMAND_COMPLETE

profiler.Record(gpu_id, "ModuleName", "EventName", data);
@endcode

@subsection profiler_record_rocm ROCm/HIP (расширенные поля)

@code{.cpp}
drv_gpu_lib::ROCmProfilingData data;
// Базовые поля (от ProfilingDataBase)
data.start_ns = start;
data.end_ns   = end;

// Расширенные поля ROCm
data.domain      = 1;                  // 0=HIP API, 1=HIP Activity, 2=HSA
data.kind        = 0;                  // 0=kernel, 1=copy, 2=barrier
data.op          = 42;                 // Код HIP-операции
data.device_id   = 0;
data.queue_id    = stream_id;
data.bytes       = buffer_size;
data.kernel_name = "fft_radix4_kernel";
data.op_string   = "hipLaunchKernel";

profiler.Record(gpu_id, "ModuleName", "EventName", data);
@endcode

---

@section profiler_stats Получение статистики

@subsection profiler_stats_pergpu Статистика по одному GPU

@code{.cpp}
auto stats = profiler.GetStats(0);
// stats: map<string, ModuleStats>
//   ключ = имя модуля

for (const auto& [module_name, mod_stats] : stats) {
    std::cout << module_name
              << " total: " << mod_stats.GetTotalTimeMs() << " ms"
              << " calls: " << mod_stats.GetTotalCalls() << std::endl;

    for (const auto& [event_name, evt_stats] : mod_stats.events) {
        std::cout << "  " << event_name
                  << " avg: " << evt_stats.GetAvgTimeMs() << " ms"
                  << " min: " << evt_stats.min_time_ms << " ms"
                  << " max: " << evt_stats.max_time_ms << " ms"
                  << std::endl;
    }
}
@endcode

@subsection profiler_stats_all Статистика по всем GPU

@code{.cpp}
auto all_stats = profiler.GetAllStats();
// all_stats: map<int, map<string, ModuleStats>>
//   ключ 1 = gpu_id
//   ключ 2 = имя модуля
@endcode

---

@section profiler_export Экспорт

@subsection profiler_export_print PrintReport() --- вывод в stdout

Выводит полный отчёт с шапкой GPU, таблицей событий и итогами.
Автоматически определяет тип таблицы (OpenCL / ROCm) по данным.

@code{.cpp}
profiler.PrintReport();
@endcode

Пример вывода:

```
+============================================+
|     ОТЧЁТ ПРОФИЛИРОВАНИЯ GPU               |
|  Дата: 2026-03-29 14:30:00                 |
+============================================+
|  GPU 0: AMD Radeon RX 9070 XT              |
|  Память: 16384 MB                          |
|  Драйверы:                                 |
|    [ROCm] Версия: 7.2.0 | Драйвер: 6.10.5 |
+--------------------------------------------+
| Модуль: AntennaFFT                         |
+--------------------------------------------+
| Событие       | N   | Среднее | Мин  | Макс |
...
```

@subsection profiler_export_summary PrintSummary() --- краткая сводка

@code{.cpp}
profiler.PrintSummary();
@endcode

Компактный вывод: модуль, количество вызовов, avg/min/max.

@subsection profiler_export_json ExportJSON() --- JSON-файл

@code{.cpp}
bool ok = profiler.ExportJSON("Results/Profiler/2026-03-29_14-30-00.json");
@endcode

Структура JSON:

@code{.json}
{
  "timestamp": "2026-03-29 14:30:00",
  "gpus": [
    {
      "gpu_id": 0,
      "gpu_name": "AMD Radeon RX 9070 XT",
      "memory_mb": 16384,
      "drivers": [
        {"driver_type": "ROCm", "version": "7.2.0", "hip_version": "6.2.41134"}
      ],
      "modules": [
        {
          "name": "AntennaFFT",
          "run_count": 100,
          "avg_run_time_ms": 12.345,
          "events": [
            {
              "name": "FFT_Execute",
              "calls": 100,
              "total_ms": 1234.5,
              "avg_ms": 12.345,
              "min_ms": 11.2,
              "max_ms": 15.8,
              "queue_delay_avg_ms": 0.012,
              "submit_delay_avg_ms": 0.003,
              "exec_time_avg_ms": 12.330,
              "complete_delay_avg_ms": 0.001
            }
          ]
        }
      ]
    }
  ]
}
@endcode

@subsection profiler_export_md ExportMarkdown() --- Markdown-файл

@code{.cpp}
bool ok = profiler.ExportMarkdown("Results/Profiler/2026-03-29_14-30-00.md");
@endcode

Генерирует Markdown-таблицу с теми же данными, что и `PrintReport()`.
Результаты сохраняются в `Results/Profiler/`.

---

@section profiler_enable Включение/выключение

@subsection profiler_enable_global Глобальное

@code{.cpp}
profiler.SetEnabled(true);   // Включить
profiler.SetEnabled(false);  // Выключить (Record() игнорируется)
bool on = profiler.IsEnabled();
@endcode

@subsection profiler_enable_pergpu Per-GPU (из configGPU.json)

@code{.cpp}
profiler.SetGPUEnabled(0, true);   // GPU 0: включить
profiler.SetGPUEnabled(1, false);  // GPU 1: отключить
bool on = profiler.IsGPUEnabled(0);
@endcode

@note При использовании `ServiceManager::InitializeFromConfig()` per-GPU
настройки берутся из поля `is_prof` в `configGPU.json`.

---

@section profiler_reset Сброс данных

@code{.cpp}
profiler.Reset();
// Вся собранная статистика очищена
@endcode

---

@section profiler_rocm_detect Обнаружение ROCm-данных

@code{.cpp}
// Есть ли ROCm-данные для GPU 0?
bool has_rocm = profiler.HasAnyROCmData(0);

// Есть ли ROCm-данные вообще (для любого GPU)?
bool any_rocm = profiler.HasAnyROCmDataGlobal();

// Есть ли ROCm-данные в конкретном модуле?
auto stats = profiler.GetStats(0);
bool mod_rocm = drv_gpu_lib::GPUProfiler::HasModuleROCmData(stats["FFT"]);
@endcode

---

@section profiler_types Типы данных профилирования

Файл: `DrvGPU/services/profiling_types.hpp`

@subsection profiler_types_base ProfilingDataBase

Общие 5 полей времени (nanoseconds):

| Поле | OpenCL аналог | Описание |
|------|---------------|----------|
| `queued_ns` | `CL_PROFILING_COMMAND_QUEUED` | Команда попала в очередь хоста |
| `submit_ns` | `CL_PROFILING_COMMAND_SUBMIT` | Команда отправлена на GPU |
| `start_ns` | `CL_PROFILING_COMMAND_START` | Ядро начало выполняться |
| `end_ns` | `CL_PROFILING_COMMAND_END` | Ядро закончило выполняться |
| `complete_ns` | `CL_PROFILING_COMMAND_COMPLETE` | Данные выгружены/доступны |

@subsection profiler_types_opencl OpenCLProfilingData

Наследник `ProfilingDataBase`. Без дополнительных полей.

@subsection profiler_types_rocm ROCmProfilingData

Наследник `ProfilingDataBase` с дополнительными полями:

| Поле | Тип | Описание |
|------|-----|----------|
| `domain` | `uint32_t` | Область: 0=HIP API, 1=HIP Activity, 2=HSA |
| `kind` | `uint32_t` | Тип: 0=kernel, 1=copy, 2=barrier, 3=marker |
| `op` | `uint32_t` | Код конкретной HIP-операции |
| `correlation_id` | `uint64_t` | Связь API-вызова и выполнения |
| `device_id` | `int` | ID устройства GPU |
| `queue_id` | `uint64_t` | ID очереди/потока (stream) |
| `bytes` | `size_t` | Объём переданных данных |
| `kernel_name` | `string` | Имя ядра |
| `op_string` | `string` | Строка операции |

@subsection profiler_types_gpuinfo GPUReportInfo

Информация о GPU для шапки отчёта:

| Поле | Тип | Описание |
|------|-----|----------|
| `gpu_name` | `string` | Название GPU |
| `global_mem_mb` | `size_t` | Объём глобальной памяти (MB) |
| `drivers` | `vector<map<string,string>>` | Информация о драйверах |

---

@section profiler_rules Правила использования

@warning **ВЫВОД данных профилирования** --- ТОЛЬКО через методы GPUProfiler:
- `PrintReport()` --- полный отчёт в stdout
- `PrintSummary()` --- краткая сводка в stdout
- `ExportMarkdown(path)` --- Markdown-файл
- `ExportJSON(path)` --- JSON-файл

@warning **ЗАПРЕЩЕНО** вручную извлекать `GetStats()` и выводить через
`std::cout` или `ConsoleOutput::Print()`. Это нарушает единый формат отчёта
и не включает шапку GPU.

@note Перед `Start()` вызывайте `SetGPUInfo()` --- иначе в отчёте "Unknown"
и "нет информации о драйверах".

---

@section profiler_related Связанные страницы

- @ref drvgpu_main --- Главная страница DrvGPU
- @ref drvgpu_architecture --- Архитектура DrvGPU
- @ref drvgpu_console --- ConsoleOutput API
