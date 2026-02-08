# План: Правильный вывод профайлера (v2)

**Дата:** 2026-02-08
**Автор:** Кодо

---

## Цель

Вывести ВСЕ значения структур:
- **OpenCL**: 5 полей ProfilingDataBase
- **ROCm**: 5 полей + domain, kind, op, correlation_id, device_id, queue_id, bytes, kernel_name, op_string, counters

---

## 1. Таблица OpenCL (5 полей времени)

### Колонки (по-русски):

| Модуль | Событие | N | Очередь | Отправка | Старт | Конец | Готово |

### Соответствие и СМЫСЛ:

| Русское | Поле | OpenCL константа | Смысл |
|---------|------|------------------|-------|
| Очередь | queued_ns | CL_PROFILING_COMMAND_QUEUED | Команда попала в очередь хоста |
| Отправка | submit_ns | CL_PROFILING_COMMAND_SUBMIT | Команда отправлена на GPU |
| Старт | start_ns | CL_PROFILING_COMMAND_START | Кернел начал выполняться |
| Конец | end_ns | CL_PROFILING_COMMAND_END | Кернел закончил выполняться |
| Готово | complete_ns | CL_PROFILING_COMMAND_COMPLETE | Данные выгружены/доступны |

### Формат значений:
- Время в миллисекундах: `12.345` мс (3 знака после запятой)
- Значения усреднённые (avg) по всем вызовам

---

## 2. Таблица ROCm (ВСЕ поля)

### Основная строка (как OpenCL):

| Модуль | Событие | N | Очередь | Отправка | Старт | Конец | Готово |

### Дополнительные строки ROCm (по-русски):

```
| [ROCm]     | Домен=1 | Тип=2 | Операция=3 | КоррID=12345 | УстрID=1 | ОчерID=100 |
| [ROCm]     | Байты: 10 MB | Ядро: matrix_multiply_kernel | Опер: hipLaunchKernel |
| [Счётчики] | GFLOPS=150.5 | MemBW=200.0 | CacheHit=95.2% |
```

### Легенда ROCm (расшифровка):

| Поле | Описание |
|------|----------|
| Домен (domain) | Область профилирования HIP (0=HIP API, 1=HIP Activity, 2=HSA) |
| Тип (kind) | Тип операции (0=кернел, 1=копирование, 2=барьер, 3=маркер) |
| Операция (op) | Код конкретной HIP операции |
| КоррID | Correlation ID - связь между API вызовом и выполнением |
| УстрID | ID устройства GPU (device_id) |
| ОчерID | ID очереди/потока (queue_id / stream) |
| Байты | Объём переданных данных |
| Ядро | Имя кернела (kernel_name) |
| Опер | Строка операции (op_string) |
| Счётчики | Аппаратные счётчики производительности (counters) - выводить все |

---

## 3. GPUReportInfo — НОВАЯ структура с вектором драйверов

На ОДНОЙ GPU работают сразу OpenCL И ROCm!

```cpp
struct GPUReportInfo {
    std::string gpu_name;
    BackendType backend_type = BackendType::OPENCL;
    size_t global_mem_mb = 0;

    // Вектор драйверов: drivers[0]=OpenCL, drivers[1]=ROCm, ...
    std::vector<std::map<std::string, std::string>> drivers;

    // Получить строку драйверов для отчёта
    std::string GetDriversString() const;
};
```

### Формат drivers[0] для OpenCL:
```cpp
map["driver_type"] = "OpenCL";
map["version"] = "3.0";
map["driver_version"] = "23.10.2";
map["platform_name"] = "AMD Accelerated Parallel Processing";
map["vendor"] = "AMD";
```

### Формат drivers[1] для ROCm:
```cpp
map["driver_type"] = "ROCm";
map["version"] = "5.4.3";
map["driver_version"] = "amdgpu 6.1.0";
map["hip_version"] = "5.4.22801";
map["hip_runtime"] = "5.4.22801-1";
```

---

## 4. Логика работы программы

```
1. Сборка программы
   └── CMake копирует configGPU.json в каталог сборки

2. Старт программы
   └── Читает configGPU.json (только для выбора GPU по id!)
       └── id=0, is_prof=true, is_active=true...

3. Инициализация GPU
   └── GPUManager сканирует все GPU
   └── Находит GPU с id=0
   └── Читает РЕАЛЬНУЮ информацию с устройства/драйверов/окружения:
       │
       ├── OpenCL: через OpenCLCore методы
       │   ├── GetDeviceName()
       │   ├── GetVendor()
       │   ├── GetDriverVersion()
       │   ├── GetPlatformName()
       │   ├── GetOpenCLVersionMajor/Minor()
       │   └── GetGlobalMemorySize()
       │
       └── ROCm: через HIP API (ЗАКОММЕНТИРОВАНО - нет драйверов)
           ├── // hipGetDeviceProperties()
           ├── // hipDriverGetVersion()
           └── // rocm-smi --showdriverversion

4. Формирование drivers vector
   └── drivers[0] = OpenCL info (из реальной системы)
   └── drivers[1] = ROCm info (закомментировано / эмуляция в тесте)

5. Передача в профайлер
   └── GPUProfiler::SetGPUInfo(gpu_id, info)
```

---

## 5. Чтение ROCm info (ЗАКОММЕНТИРОВАНО)

### Файл: `DrvGPU/backends/opencl/opencl_backend.cpp` или отдельный файл

```cpp
// =========================================================================
// ROCm Info - ЗАКОММЕНТИРОВАНО (нет ROCm драйверов)
// Раскомментировать когда будут установлены ROCm/HIP
// =========================================================================

/*
#include <hip/hip_runtime.h>
#include <fstream>

inline std::map<std::string, std::string> GetROCmDriverInfo() {
    std::map<std::string, std::string> info;
    info["driver_type"] = "ROCm";

    // Версия ROCm из /opt/rocm/.info/version
    std::ifstream f("/opt/rocm/.info/version");
    if (f.is_open()) {
        std::string version;
        std::getline(f, version);
        info["version"] = version;  // "5.4.3"
    }

    // Версия драйвера через rocm-smi или modinfo amdgpu
    // info["driver_version"] = "amdgpu 6.1.0";

    // HIP версия
    int hipVersion = 0;
    hipRuntimeGetVersion(&hipVersion);
    info["hip_version"] = std::to_string(hipVersion);

    int driverVersion = 0;
    hipDriverGetVersion(&driverVersion);
    info["hip_driver"] = std::to_string(driverVersion);

    return info;
}
*/
```

---

## 6. Тест

### GPU 0 — реальная OpenCL карта:

```cpp
// GPU 0: читаем реальную информацию
GPUManager manager;
manager.InitializeAll(BackendType::OPENCL);

// Получаем реальные данные с GPU
auto device_info = manager.GetGPU(0).GetDeviceInfo();
auto& core = dynamic_cast<OpenCLBackend*>(&manager.GetGPU(0).GetBackend())->GetCore();

// Формируем drivers[0] - OpenCL (из реальной системы)
std::map<std::string, std::string> opencl_driver;
opencl_driver["driver_type"] = "OpenCL";
opencl_driver["version"] = device_info.opencl_version;
opencl_driver["driver_version"] = device_info.driver_version;
opencl_driver["platform_name"] = core.GetPlatformName();
opencl_driver["vendor"] = device_info.vendor;

// Формируем GPUReportInfo
GPUReportInfo gpu0_info;
gpu0_info.gpu_name = device_info.name;
gpu0_info.global_mem_mb = device_info.global_memory_size / (1024*1024);
gpu0_info.backend_type = BackendType::OPENCL;
gpu0_info.drivers.push_back(opencl_driver);

profiler.SetGPUInfo(0, gpu0_info);
```

### GPU 1 — эмуляция ROCm:

```cpp
// GPU 1: эмулируем ROCm (на той же карте или другой)
GPUReportInfo gpu1_info;
gpu1_info.gpu_name = "AMD Radeon RX 580";
gpu1_info.global_mem_mb = 8192;
gpu1_info.backend_type = BackendType::OPENCLandROCm;

// drivers[0] - OpenCL (можно скопировать с GPU 0 или эмулировать)
std::map<std::string, std::string> opencl_driver;
opencl_driver["driver_type"] = "OpenCL";
opencl_driver["version"] = "2.0";
opencl_driver["driver_version"] = "22.40.5";
opencl_driver["platform_name"] = "AMD Accelerated Parallel Processing";
opencl_driver["vendor"] = "AMD";
gpu1_info.drivers.push_back(opencl_driver);

// drivers[1] - ROCm (ЭМУЛЯЦИЯ)
std::map<std::string, std::string> rocm_driver;
rocm_driver["driver_type"] = "ROCm";
rocm_driver["version"] = "5.4.3";
rocm_driver["driver_version"] = "amdgpu 6.1.0";
rocm_driver["hip_version"] = "5.4.22801";
rocm_driver["hip_runtime"] = "5.4.22801-1";
gpu1_info.drivers.push_back(rocm_driver);

profiler.SetGPUInfo(1, gpu1_info);
```

### Данные профилирования:

```cpp
// GPU 0: только OpenCL данные (5 полей времени)
OpenCLProfilingData ocl_data{};
ocl_data.queued_ns = 1000000000;      // 1000.000 мс
ocl_data.submit_ns = 1000100000;      // 1000.100 мс
ocl_data.start_ns = 1000150000;       // 1000.150 мс
ocl_data.end_ns = 1012900000;         // 1012.900 мс
ocl_data.complete_ns = 1012950000;    // 1012.950 мс
profiler.Record(0, "AntennaFFT", "FFT_Execute", ocl_data);

// GPU 1: ROCm данные (ВСЕ поля)
ROCmProfilingData rocm_data{};
// 5 полей времени
rocm_data.queued_ns = 2000000000;
rocm_data.submit_ns = 2000100000;
rocm_data.start_ns = 2000150000;
rocm_data.end_ns = 2002350000;
rocm_data.complete_ns = 2002400000;
// ROCm специфичные
rocm_data.domain = 1;           // HIP Activity
rocm_data.kind = 0;             // Kernel
rocm_data.op = 11;              // hipLaunchKernel
rocm_data.correlation_id = 12345;
rocm_data.device_id = 1;
rocm_data.queue_id = 100;
rocm_data.bytes = 1024 * 1024 * 10;  // 10 MB
rocm_data.kernel_name = "matrix_multiply_kernel";
rocm_data.op_string = "hipLaunchKernel";
rocm_data.counters["GFLOPS"] = 150.5;
rocm_data.counters["MemBW_GB_s"] = 200.0;
rocm_data.counters["CacheHit_pct"] = 95.2;
profiler.Record(1, "ROCmModule", "MatrixMul", rocm_data);
```

---

## 7. Файлы для изменения

| Файл | Изменения |
|------|-----------|
| `gpu_profiler.hpp` | GPUReportInfo: убрать старые поля, добавить `vector<map> drivers` |
| `gpu_profiler.hpp` | PrintReport() — таблица по-русски, ВСЕ 5 полей времени, ВСЕ поля ROCm |
| `gpu_profiler.hpp` | ExportMarkdown() — аналогично |
| `gpu_manager.hpp` | InitializeGPU() — формировать drivers vector, ROCm закомментировано |
| `gpu_manager.hpp` | **GetGPUReportInfo(int gpu_id)** — получить GPUReportInfo из реальной системы |
| `test_gpu_profiler.hpp` | GPU 0: использовать **GPUManager::GetGPUReportInfo(0)**; GPU 1: эмуляция ROCm |

---

## 7.1. Функция для получения информации о драйверах

**GPUManager::GetGPUReportInfo(int gpu_id)** — возвращает GPUReportInfo с drivers[] из реальной системы.

```cpp
// Использование:
GPUManager manager;
manager.InitializeAll(BackendType::OPENCL);

// Получить GPUReportInfo для GPU 0 (из реальной системы!)
GPUReportInfo gpu0_info = manager.GetGPUReportInfo(0);
profiler.SetGPUInfo(0, gpu0_info);
```

Функция читает:
- `device_info.name` — имя GPU
- `device_info.opencl_version` — версия OpenCL
- `device_info.driver_version` — версия драйвера
- `device_info.vendor` — производитель
- `backend.GetCore().GetPlatformName()` — платформа OpenCL

---

## 8. Пример вывода

### Шапка (GPU с OpenCL только):
```
+========================================================================================+
|              ОТЧЁТ ПРОФИЛИРОВАНИЯ GPU                                                  |
|  Дата: 2026-02-08 14:55:01                                                             |
+========================================================================================+
|  GPU 0: AMD Radeon RX 6700 XT                                                          |
|  Память: 12288 MB                                                                      |
|  Драйверы:                                                                             |
|    [OpenCL] Версия: 3.0 | Драйвер: 23.10.2 | Платформа: AMD APP                        |
+----------------------------------------------------------------------------------------+
```

### Шапка (GPU с OpenCL + ROCm):
```
|  GPU 1: AMD Radeon RX 580                                                              |
|  Память: 8192 MB                                                                       |
|  Драйверы:                                                                             |
|    [OpenCL] Версия: 2.0 | Драйвер: 22.40.5 | Платформа: AMD APP                        |
|    [ROCm] Версия: 5.4.3 | Драйвер: amdgpu 6.1.0 | HIP: 5.4.22801                        |
+----------------------------------------------------------------------------------------+
```

### Таблица OpenCL (5 полей времени):
```
| Модуль     | Событие        | N   | Очередь  | Отправка | Старт    | Конец    | Готово   |
|------------|----------------|-----|----------|----------|----------|----------|----------|
| AntennaFFT | FFT_Execute    | 100 | 1000.000 | 1000.100 | 1000.150 | 1012.900 | 1012.950 |
|            | Padding_Kernel | 100 |  500.000 |  500.050 |  500.080 |  500.850 |  500.860 |
|            | --- ИТОГО ---  | 200 |          |          |          |          |          |
```

### Таблица ROCm (ВСЕ поля):
```
| Модуль     | Событие    | N  | Очередь  | Отправка | Старт    | Конец    | Готово   |
|------------|------------|----| ---------|----------|----------|----------|----------|
| ROCmModule | MatrixMul  | 20 | 2000.000 | 2000.100 | 2000.150 | 2002.350 | 2002.400 |
|            | [ROCm]     |    | Домен=1 | Тип=0 | Операция=11 | КоррID=12345        |
|            | [ROCm]     |    | УстрID=1 | ОчерID=100 | Байты=10 MB                  |
|            | [ROCm]     |    | Ядро: matrix_multiply_kernel                          |
|            | [ROCm]     |    | Опер: hipLaunchKernel                                 |
|            | [Счётчики] |    | GFLOPS=150.5 | MemBW=200.0 | CacheHit=95.2%         |
|            | --- ИТОГО ---|20 |          |          |          |          |          |

+----------------------------------------------------------------------------------------+
```

### Легенда:
```
+--- ЛЕГЕНДА ---+
| Время в миллисекундах (мс), усреднённое значение                                       |
+---------------+------------------------------------------------------------------------+
| Очередь       | Команда попала в очередь хоста                                         |
| Отправка      | Команда отправлена на GPU                                              |
| Старт         | Кернел начал выполняться                                               |
| Конец         | Кернел закончил выполняться                                            |
| Готово        | Данные выгружены/доступны                                              |
+---------------+------------------------------------------------------------------------+
| [ROCm поля]                                                                            |
| Домен         | Область профилирования (0=HIP API, 1=HIP Activity, 2=HSA)              |
| Тип           | Тип операции (0=кернел, 1=копирование, 2=барьер)                       |
| Операция      | Код HIP операции                                                       |
| КоррID        | Correlation ID - связь API вызова и выполнения                         |
| УстрID        | ID устройства GPU                                                      |
| ОчерID        | ID очереди/потока (stream)                                             |
| Байты         | Объём переданных данных                                                |
| Ядро          | Имя кернела                                                            |
| Опер          | Строка операции                                                        |
| Счётчики      | Аппаратные счётчики производительности                                 |
+--------------------------------------------------------------------------------+
```

---

## Подтверждено Алексом:

1. ✅ Время в шапке: `2026-02-08 14:55:01`
2. ✅ Counters: выводить ВСЕ
3. ✅ Легенда Domain/Kind/Op: НУЖНА расшифровка (добавлена выше)
4. ✅ Драйверы: `vector<map<string, string>> drivers` — OpenCL в [0], ROCm в [1]
5. ✅ Тест: GPU 0 = реальная инфа, GPU 1 = эмуляция ROCm в drivers[1]

---

**План готов! Жду подтверждения для реализации.**
