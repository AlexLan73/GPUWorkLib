# План: полный набор измерений в GPUProfiler (OpenCL 5 + ROCm)

## Цель

- В **рабочем (GPU) потоке не делать конвертацию** — передавать данные в **нативном формате** измерения (OpenCL или ROCm/HIP).
- **Конвертацию** в общую форму и агрегацию выполнять **в воркере профайлера** (фоновый поток).
- **Одна общая таблица**: 5 параметров времени как в OpenCL (queued, submit, start, end, complete) — для ROCm те же 5 слотов (заполняемые из begin_ns/end_ns и при необходимости нулями) + **дополнительные параметры (строки)** от ROCm/HIP (correlation_id, device_id, queue_id, domain, op, kernel_name, счётчики и т.д.).

Формат одной записи в профайлер:

- `gpu_id` (int)
- `module_name` (string)
- `event_name` (string)
- **`time_`** — тип `<T>`: либо `OpenCLProfilingData`, либо `ROCmProfilingData` (нативный формат; конвертация в общую таблицу — в ProcessMessage).

---

## 1. Типы измерений: наследование (ООП), общие 5 полей в базе

Поскольку OpenCL и HIP/ROCm используют одни и те же 5 временных полей (queued, submit, start, end, complete), делаем **базовую структуру** с этими полями и **наследование**: OpenCL — только база; ROCm — база + доп. поля. Так не дублируем 5 полей и явно задаём общий контракт.

### 1.1 Базовая структура (5 полей времени)

```cpp
struct ProfilingDataBase {
    uint64_t queued_ns;   // CL_PROFILING_COMMAND_QUEUED / ROCm: 0 или begin_ns
    uint64_t submit_ns;   // CL_PROFILING_COMMAND_SUBMIT / ROCm: 0 или begin_ns
    uint64_t start_ns;    // CL_PROFILING_COMMAND_START / ROCm: begin_ns
    uint64_t end_ns;      // CL_PROFILING_COMMAND_END / ROCm: end_ns
    uint64_t complete_ns; // CL_PROFILING_COMMAND_COMPLETE / ROCm: end_ns
};
```

### 1.2 OpenCL — наследует базу (доп. полей нет)

Вызывающий код заполняет структуру из `clGetEventProfilingInfo` и передаёт в Record. Конвертации в мс в основном потоке нет.

```cpp
struct OpenCLProfilingData : ProfilingDataBase {
    // только 5 полей из базы
};
```

Тип `uint64_t` эквивалентен `cl_ulong`; заголовки OpenCL в общий заголовок профайлера не включать.

### 1.3 ROCm/HIP — наследует базу + все доступные доп. параметры

Структура содержит **все поля** из HIP/ROCTracer/ROCprofiler; 5 времён приходят из базы, остальное — специфика ROCm.

Источники: ROCTracer `activity_record_t`, HIP Event, ROCprofiler-SDK (kernel_name, counters и т.д.).

```cpp
struct ROCmProfilingData : ProfilingDataBase {
    // --- ROCTracer activity_record_t ---
    uint32_t domain;
    uint32_t kind;
    uint32_t op;
    uint64_t correlation_id;
    int      device_id;
    uint64_t queue_id;
    size_t   bytes;

    // --- Строковые/идентифицирующие ---
    std::string kernel_name;
    std::string op_string;

    // --- Счётчики rocprof ---
    std::map<std::string, double> counters;
};
```

База уже задаёт queued_ns, submit_ns, start_ns, end_ns, complete_ns; для ROCm их заполняем из begin_ns/end_ns по соглашению. Конвертация в общую таблицу — в воркере.

---

## 2. Сообщение профилировщика (ProfilingMessage) и смысл variant

- Сейчас в сообщении: `duration_ms` (double), `timestamp` (time_point).
- Нужно: поле **`time_`** типа **`std::variant<OpenCLProfilingData, ROCmProfilingData>`**.

**Что значит variant здесь:** мы явно задаём, **с какими типами измерений работаем** — только два: OpenCL и ROCm. По значению variant профайлер в воркере понимает, откуда пришла запись (OpenCL или ROCm) и как её обрабатывать (достать 5 полей из базы + при ROCm — доп. поля). В основном потоке только заполняем одну из нативных структур и передаём; конвертации не делаем.

В **ProcessMessage** (воркер):

- из `time_` через `std::visit` или `std::get` извлекаем либо `OpenCLProfilingData`, либо `ROCmProfilingData`;
- 5 времён в обоих случаях берём из базовой части (наследование);
- вычисляем duration_ms для агрегации: `(end_ns - start_ns) * 1e-6`;
- обновляем общую таблицу (5 колонок времени + при ROCm — доп. поля).

---

## 3. API Record

**Убрать** старый вызов:

```cpp
void Record(int gpu_id, const std::string& module, const std::string& event, double duration_ms);  // удалить
```

**Оставить только** перегрузки с нативными структурами:

```cpp
void Record(int gpu_id, const std::string& module, const std::string& event, const OpenCLProfilingData& data);
void Record(int gpu_id, const std::string& module, const std::string& event, const ROCmProfilingData& data);
```

Все текущие вызовы Record с `duration_ms` заменить на заполнение `OpenCLProfilingData` (хелпер из cl_event) и вызов `Record(..., opencl_data)`.

---

## 4. Получение полных 5 значений OpenCL в коде

Сейчас в [modules/fft_maxima/src/antenna_fft_core.cpp](modules/fft_maxima/src/antenna_fft_core.cpp) (строки 429–435) вызываются только START и END:

```cpp
clGetEventProfilingInfo(event, CL_PROFILING_COMMAND_START, ...);
clGetEventProfilingInfo(event, CL_PROFILING_COMMAND_END, ...);
```

Нужно:

- Добавить функцию (в том же модуле или в общем OpenCL-хелпере DrvGPU), заполняющую `OpenCLProfilingData` пятью вызовами `clGetEventProfilingInfo` для:
  - `CL_PROFILING_COMMAND_QUEUED`
  - `CL_PROFILING_COMMAND_SUBMIT`
  - `CL_PROFILING_COMMAND_START`
  - `CL_PROFILING_COMMAND_END`
  - `CL_PROFILING_COMMAND_COMPLETE` (если доступно, иначе можно не заполнять или положить 0).
- В местах, где сейчас вызывается `ProfileEvent(fft_event, "FFT")` и затем `GPUProfiler::GetInstance().Record(..., fft_time_ms)`, заменить на: заполнить `OpenCLProfilingData` из `cl_event`, вызвать `GPUProfiler::GetInstance().Record(..., opencl_data)`.

Аналогично проверить [spectrum_maxima_finder.cpp](modules/fft_maxima/src/spectrum_maxima_finder.cpp) и другие вызовы `ProfileEvent` + `Record`.

---

## 5. Агрегация и хранение: одна общая таблица

В **ProcessMessage** (воркер) выполняется вся конвертация:

- Из variant извлекаем либо `OpenCLProfilingData`, либо `ROCmProfilingData`. В обоих случаях **5 полей времени берём из базовой части** (наследование от `ProfilingDataBase`).
- **ROCm**: дополнительно сохраняем correlation_id, device_id, queue_id, domain, op, kernel_name, counters — в виде доп. колонок/ключей в общей таблице/экспорте.

Общая таблица:

- Колонки времени: **queued_ns, submit_ns, start_ns, end_ns, complete_ns** (одна схема для OpenCL и ROCm).
- Доп. колонки/поля для ROCm: correlation_id, device_id, queue_id, domain, op, kernel_name, counters (map<string, double> в JSON как объект).

Агрегация по duration: вычислять `duration_ms = (end_ns - start_ns) * 1e-6` и обновлять `EventStats` (calls, total_time_ms, min/max) как сейчас. Опционально хранить последние N сырых записей (5 времён + ROCm-доп.) для экспорта в JSON.

---

## 6. Экспорт (JSON / PrintSummary)

- **Общая таблица в JSON**: для каждого события выводить 5 полей времени (`queued_ns`, `submit_ns`, `start_ns`, `end_ns`, `complete_ns`) — и при наличии доп. параметров ROCm: `correlation_id`, `device_id`, `queue_id`, `domain`, `op`, `kernel_name`, `counters` (объект имя_метрики -> значение). Агрегаты: calls, total_ms, avg_ms, min_ms, max_ms как сейчас.
- При хранении последних N сэмплов — массив записей с полным набором (5 времён + ROCm-доп.) для детального отчёта.
- PrintSummary: по длительности (avg/min/max); опционально задержки (start_ns - queued_ns), (end_ns - start_ns) в виде агрегатов.

---

## 7. Зависимости и место типов

- Базовый тип `ProfilingDataBase` и наследники `OpenCLProfilingData`, `ROCmProfilingData` не включают заголовки OpenCL/HIP (только `uint64_t`, стандартные контейнеры).
- C++17 для `std::variant`.
- Размещение: общий заголовок в DrvGPU (`profiling_types.hpp` или секция в `gpu_profiler.hpp`), чтобы и модули (fft_maxima), и профайлер использовали одни и те же структуры.

---

## 8. Порядок работ (кратко)

1. Ввести базовый тип `ProfilingDataBase` (5 полей времени) и наследников `OpenCLProfilingData`, `ROCmProfilingData` (ROCm добавляет domain, kind, op, correlation_id, device_id, queue_id, bytes, kernel_name, op_string, counters) в общем заголовке.
2. Расширить `ProfilingMessage` полем `time_` типа `std::variant<OpenCLProfilingData, ROCmProfilingData>` — этим задаём, с какими типами работаем; конвертация только в воркере.
3. **Удалить** старый `Record(..., double duration_ms)`. Реализовать только `Record(..., OpenCLProfilingData)` и `Record(..., ROCmProfilingData)`; в ProcessMessage из variant извлекать данные, конвертировать в общую таблицу, вычислять duration_ms и обновлять EventStats.
4. Добавить хелпер заполнения `OpenCLProfilingData` из `cl_event` (все 5 вызовов `clGetEventProfilingInfo`).
5. Заменить все вызовы Record в antenna_fft_core, spectrum_maxima_finder и др. на заполнение `OpenCLProfilingData` и вызов `Record(..., opencl_data)`.
6. (Опционально) Хранение последних N сэмплов и экспорт полной общей таблицы в JSON.
7. ROCm: при появлении HIP/ROCTracer заполнять `ROCmProfilingData` (база + доп. поля); конвертация — в ProcessMessage.

Итог: запись в GPUProfiler — **gpu_id, module_name, event_name, time_** (variant из двух типов с общей базой 5 полей); конвертация — только в воркере.
