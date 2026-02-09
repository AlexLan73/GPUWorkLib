# Ref04: Вывод профилирования через GPUProfiler — все 5 полей OpenCL

## Цель

1. Перенаправить вывод профилирования из примера `modules/fft_maxima/tests/test_spectrum_maxima.hpp` на профайлер `DrvGPU/services`.
2. Использовать **только существующий полный отчёт и легенду** (Очередь, Отправка, Старт, Конец, Готово) — **без** добавления компактного формата. Нужно отладить поток из разных kernel и получить общее понимание картины по всем 5 значениям OpenCL.

---

## Текущее состояние

- **Тест** вызывает `PrintProfiling(finder.GetProfilingData())` и выводит в stdout блок вида:
  - `Upload (Host→GPU)`, `FFT (with pre-callback)`, `Post-kernel`, `Download (GPU→Host)`, `TOTAL`.
- **SpectrumMaximaFinder** (`spectrum_maxima_finder.cpp`) заполняет локальную `ProfilingData` через `ProfileEvent(event, name)`, читая только START/END и возвращая длительность в мс. В GPUProfiler данные **не отправляет**.
- **GPUProfiler** (`gpu_profiler.hpp`) уже имеет:
  - `Record(gpu_id, module, event, OpenCLProfilingData)`;
  - `PrintReport()` — полная таблица с колонками **Очередь, Отправка, Старт, Конец, Готово** и `PrintLegend()`; их **не менять**;
  - `PrintSummary()` — другой формат.
- В `DrvGPU/backends/opencl/opencl_profiling.hpp` и `.cpp`: **FillOpenCLProfilingData(cl_event, OpenCLProfilingData&)** — заполняет из `cl_event` **все 5 полей** OpenCL: queued_ns, submit_ns, start_ns, end_ns, complete_ns (CL_PROFILING_COMMAND_QUEUED / SUBMIT / START / END / COMPLETE).

**Требование:** от OpenCL нужны **только и именно все 5 значений**. Никакой подстановки длительности (MakeOpenCLFromDurationMs) — только реальные данные из `cl_event` через `FillOpenCLProfilingData`.

---

## План действий

### 1. Отправка данных из SpectrumMaximaFinder в GPUProfiler — все 5 полей OpenCL

**Файлы:** `spectrum_maxima_finder.cpp`, при необходимости `spectrum_maxima_finder.h`.

- **Ветвление по is_prof:** выполнять отправку в профайлер **только если** профилирование включено для данной GPU. Перед блоком проверять, например: `GPUProfiler::GetInstance().IsEnabled() && GPUProfiler::GetInstance().IsGPUEnabled(gpu_id)`. Если выключено — не вызывать FillOpenCLProfilingData и Record.
- В `Process()` для **каждого** из четырёх событий (upload_event, fft_event, post_event, read_event) **до** `clReleaseEvent` **и только при is_prof**:
  - По **каждому** из этих 4 событий получаем **все 5 полей** OpenCL (Очередь, Отправка, Старт, Конец, Готово). Итого: 4 операции × 5 полей в каждой.
  1. Вызвать `FillOpenCLProfilingData(event, data)` — заполняет все 5 полей (queued_ns, submit_ns, start_ns, end_ns, complete_ns) из `cl_event`.
  2. Если успешно: `GPUProfiler::GetInstance().Record(gpu_id, "SpectrumMaxima", eventName, data)` с именами `"Upload"`, `"FFT"`, `"PostKernel"`, `"Download"`.
- Подключить заголовки: `DrvGPU/backends/opencl/opencl_profiling.hpp`, `DrvGPU/services/gpu_profiler.hpp`.
- **gpu_id:** тест использует один GPU (0); в finder пока 0 (при необходимости позже — параметр или получение из бэкенда).
- Локальное заполнение `profiling_` (upload_time_ms, fft_time_ms, …) и `GetProfilingData()` оставить без изменений; **дополнительно** — при is_prof: 4 вызова FillOpenCLProfilingData + Record (каждый вызов отдаёт в профайлер все 5 полей по одному событию).
пше 
### 2. Вывод в тесте — только полный отчёт профайлера (без нового компактного метода)

- **Не добавлять** новый метод компактного вывода (`PrintCompactGPUTiming`). Использовать **существующие** `PrintReport()` и `PrintLegend()` в GPUProfiler — по ним отлаживают поток из разных kernel и получают общую картину по всем 5 полям (Очередь, Отправка, Старт, Конец, Готово).
- В `gpu_profiler.hpp` ничего не менять в части вывода — только приём данных (шаг 1) и вызов существующего отчёта из теста (шаг 3).

### 3. Тест: перенаправить вывод на профайлер (полный отчёт)

**Файл:** `modules/fft_maxima/tests/test_spectrum_maxima.hpp`.

- **Ветвление по is_prof:** блок «ожидание очереди профайлера + PrintReport» выполнять **только при включённом профилировании**. Проверка: `GPUProfiler::GetInstance().IsEnabled()` (и при необходимости `IsGPUEnabled(0)`). Если профилирование выключено — не ждать очередь и не вызывать PrintReport; при желании оставить старый вывод `PrintProfiling(finder.GetProfilingData())` или не выводить блок вовсе.
- Перед вызовом `PrintReport()` при необходимости дать профайлеру обработать асинхронную очередь (например краткая задержка). Оптимизацию ожидания очереди (цикл по GetQueueSize и т.п.) **снимаем** — вернём позже.
- При is_prof: вызвать **существующий** метод `drv_gpu_lib::GPUProfiler::GetInstance().PrintReport()`.
- Подключить `#include "DrvGPU/services/gpu_profiler.hpp"`.
- `PrintProfiling(ProfilingData)` при is_prof не вызывать (вывод идёт через PrintReport); при !is_prof — по желанию оставить для совместимости или не выводить.

### 4. Какие подписи используем

- Рассматриваем **только полный отчёт и легенду**: Очередь (queued), Отправка (submit), Старт (start), Конец (end), Готово (complete). Они заполняются реальными данными OpenCL из `FillOpenCLProfilingData`. Компактный формат с подписями типа "Upload (Host→GPU)" / TOTAL не вводим — нужна отладка по полной таблице.

---

## Зависимости и порядок

- Использовать только **FillOpenCLProfilingData** — все 5 значений из OpenCL (queued, submit, start, end, complete). MakeOpenCLFromDurationMs не использовать.
- При `DrvGPU::Initialize()` сервисы (GPUProfiler) уже поднимаются; в тесте spectrum_maxima отдельно поднимать ServiceManager не нужно.
- Порядок: 1 (finder шлёт в профайлер полные OpenCL-данные) → 3 (тест вызывает существующий PrintReport).

---

## Итог

- От OpenCL передаются **только все 5 значений** (queued, submit, start, end, complete) через FillOpenCLProfilingData; без подстановки длительности.
- Вывод в тесте — **только существующий полный отчёт и легенда** (Очередь, Отправка, Старт, Конец, Готово) через `PrintReport()`; компактный формат не добавляем — нужна отладка потока между kernel и общая картина по всем 5 полям.
- **Ветвление по is_prof:** отправка в профайлер (finder) и вывод отчёта (тест) выполняются **только при включённом профилировании** (конфиг/IsGPUEnabled); иначе лишние вызовы и вывод не делаем.
- Ожидание очереди профайлера перед PrintReport — пока простая задержка при необходимости; оптимизацию (GetQueueSize и т.д.) вернём позже.
