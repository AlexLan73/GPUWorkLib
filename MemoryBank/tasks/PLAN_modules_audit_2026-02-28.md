# 🔍 Аудит модулей: statistics / fft_processor / fft_maxima

> **Дата**: 2026-02-28
> **Автор**: Кодо
> **Цель**: Сравнить планы оптимизации с реальным кодом, найти расхождения, поставить задачи
> **Платформа разработки**: Windows (сборка) → **Debian (цель)**

---

## 0. Эталонный шаблон — `vector_algebra`

Именно по этому модулю сверяем остальные. Что реализовано в `vector_algebra`:

| Паттерн | Как реализован |
|---------|----------------|
| Компиляция с флагами | `hiprtcCompileProgram` с `-O3 --offload-arch=gfxXXXX -std=c++17` |
| Кеш HSACO на диске | `KernelCacheService` → `modules/vector_algebra/kernels/bin/` |
| `__launch_bounds__` | На всех ядрах |
| Fast intrinsics | `__fsqrt_rn`, `__atan2f` |
| Warp shuffle | `__shfl_down` в финальных стадиях редукции |
| RAII буферы | Умные указатели, move-only семантика |
| Profiler + GPUProfiler | `SetGPUInfo()` + `PrintReport()` |
| Тесты C++ | 23 теста, `all_test.hpp` |
| Тесты Python | 6 тестов, `Python_test/vector_algebra/` |
| Doc Python API | `Doc/Python/vector_algebra_api.md` |

---

## 1. `modules/statistics`

### 1.1 Статус плана
Файл `tasks/statistics_optimization_plan.md` → **✅ РЕАЛИЗОВАНО (2026-02-26)**

### 1.2 Проверка кода

| TASK | Описание | Статус |
|------|----------|--------|
| TASK-1 | `welford_fused` — 1 pass по данным | ✅ Реализовано |
| TASK-2 | `extract_medians` GPU kernel — 1 DtoH вместо 256 | ✅ Реализовано |
| TASK-3 | `KernelCacheService` + `--offload-arch` + `-O3` | ✅ Реализовано |
| TASK-4 | `__launch_bounds__(256)`, warp shuffle `__shfl_down`, double-load | ✅ Реализовано |
| TASK-5 | `__fsqrt_rn`, `hipMemcpyAsync` | ✅ Реализовано |

**C++ тесты**: `modules/statistics/tests/test_statistics_rocm.hpp` ✅
**Python тесты**: `Python_test/statistics/test_statistics_rocm.py` ✅

### 1.3 ❌ РАСХОЖДЕНИЯ

| # | Расхождение | Где | Критичность |
|---|-------------|-----|-------------|
| D1 | MASTER_INDEX: статус "**Planned**" — код давно реализован | `MASTER_INDEX.md` | 🔴 Вводит в заблуждение |
| D2 | BACKLOG.md: Statistics в "**Отложенных**" — устарело | `tasks/BACKLOG.md` | 🟡 |
| D3 | CHECKLIST_Statistics_Module.md: **0/149 галочек** (старый чеклист для OpenCL, реализация сделана на ROCm hiprtc иначе) | `tasks/CHECKLIST_Statistics_Module.md` | 🟡 Чеклист устарел |
| D4 | Нет `Doc/Python/statistics_api.md` | `Doc/Python/` | 🟠 Нет документации |
| D5 | Нет Python тестов для OpenCL backend (только ROCm) | `Python_test/statistics/` | 🟢 Низкий |

---

## 2. `modules/fft_processor`

### 2.1 Статус плана
Файл `tasks/fft_processor_optimization_plan.md` → **✅ РЕАЛИЗОВАНО (2026-02-26)**

### 2.2 Проверка кода

| TASK | Описание | Статус |
|------|----------|--------|
| TASK-1 | `KernelCacheService` + `--offload-arch` + `-O3` | ✅ Реализовано |
| TASK-2 | 2D grid (`blockIdx.y = beam_id`) — убран div/mod | ✅ Реализовано |
| TASK-3 | `hipMemsetAsync` trick — устранён divergent else-branch | ✅ Реализовано |
| TASK-4A | `mag_phase_interleaved_` — один DtoH вместо двух | ✅ Реализовано |
| TASK-4B | `plan_last_` кеш — два плана, нет Destroy/Create | ✅ Реализовано |
| TASK-5 | `__launch_bounds__(256)`, `__fsqrt_rn`, `__atan2f` | ✅ Реализовано |

**C++ тесты**: `test_fft_processor.hpp`, `test_fft_processor_rocm.hpp`, `test_fft_vs_cpu.hpp` ✅
**Python тесты**: ❌ НЕТ

### 2.3 ❌ РАСХОЖДЕНИЯ

| # | Расхождение | Где | Критичность |
|---|-------------|-----|-------------|
| D1 | Нет Python тестов для FFT Processor ROCm | `Python_test/fft_processor/` — **не существует** | 🔴 |
| D2 | Нет `Doc/Python/fft_processor_api.md` | `Doc/Python/` | 🟠 |
| D3 | `rocm_modules_api.md` — возможно покрывает FFT, нужно проверить | `Doc/Python/rocm_modules_api.md` | 🟡 |

---

## 3. `modules/fft_maxima`

### 3.1 Статус плана
Файл `tasks/fft_maxima_optimization_plan.md` → **✅ COMPLETED (2026-02-26)**

### 3.2 Проверка кода

| TASK | Описание | Статус |
|------|----------|--------|
| TASK-1 | Parallel tree reduction (LDS stride loop) | ✅ Реализовано |
| TASK-2 | `native_sqrt` вместо `sqrt` | ✅ Реализовано |
| TASK-3 | `nFFT_log2` bitwise вместо div/mod в pre-callback | ✅ Реализовано |
| TASK-4 | padding_kernel → 2D NDRange + `clEnqueueFillBuffer` | ✅ Реализовано |
| TASK-5 | LDS +1 padding (bank conflicts) | ✅ Реализовано |
| TASK-6 | `detect_all_maxima` → 2D NDRange | ✅ Реализовано |
| TASK-7 | `reqd_work_group_size(256,1,1)` | ✅ Реализовано |
| TASK-8 | `__restrict` на все pointer params | ✅ Реализовано |
| TASK-9 | `prefix_sum` LDS +1 padding | ✅ Реализовано |

**C++ тесты**: 8 тестов ✅ + `README.md` ✅
**Python тесты**: `Python_test/fft_maxima/` ✅
**Doc Python API**: `Doc/Python/spectrum_maxima_api.md` ✅
**MASTER_INDEX**: "COMPLETED" ✅

### 3.3 Расхождения
**НЕТ** — модуль полностью соответствует эталонному шаблону vector_algebra 🎉

---

## 4. 📋 ЗАДАЧИ — что нужно сделать

### 🔴 Высокий приоритет

#### STAT-FIX-01 — Обновить MASTER_INDEX
- Statistics: `Planned` → `🟢 Active` с описанием ROCm реализации
- Добавить Python тест в список классов

#### FFT-PROC-01 — Python тесты для FFT Processor ROCm
Создать `Python_test/fft_processor/test_fft_processor_rocm.py`:
- Тест Complex режим: FFT → сравнить с NumPy FFT
- Тест MagPhase режим: проверить magnitude/phase
- Тест batch: 256 лучей × N точек
- Benchmark: время ROCm vs clFFT
- Сравнение результатов OpenCL vs ROCm backend

#### FFT-PROC-02 — Документация `Doc/Python/fft_processor_api.md`
По образцу `vector_algebra_api.md`

### 🟠 Средний приоритет

#### STAT-DOC-01 — Документация `Doc/Python/statistics_api.md`
По образцу `vector_algebra_api.md`:
- `StatisticsProcessor(ctx)` конструктор
- `compute_mean()`, `compute_median()`, `compute_statistics()`
- Параметры, возвращаемые типы, примеры

#### STAT-FIX-02 — Обновить BACKLOG.md
Убрать "Statistics" из "Отложенных" — модуль реализован

#### STAT-FIX-03 — Архивировать старый CHECKLIST_Statistics_Module.md
Чеклист описывает OpenCL-архитектуру с Factory/Facade, которая не была реализована.
Либо: переписать под фактическую ROCm реализацию, либо архивировать.

### 🟢 Низкий приоритет

#### STAT-TEST-01 — C++ тест корректности Statistics (ROCm vs CPU reference)
В `test_statistics_rocm.hpp`: добавить сравнение с CPU-эталоном (numpy-like reference).

#### FFT-PROC-03 — Проверить `Doc/Python/rocm_modules_api.md`
Есть ли там FFT Processor? Если нет — добавить.

---

## 5. ⚠️ Важно: Debian target

**Все скрипты, пути, CMake команды** — писать под **Debian Linux**:
- Разделитель путей: `/` (не `\\`)
- Build команды: `cmake --build build --config Release` (без `.bat`)
- Shell скрипты: `#!/bin/bash`
- Пакеты: `apt-get install librocm-dev` и т.д.
- Python: `python3` (не `python`)
- `.so` вместо `.dll`, `.pyd`

---

## 6. Про папку "Cowork"

Папки с таким именем в проекте **нет**.
Скорее всего речь о `.claude/worktrees/` — это папки, которые Claude Code автоматически создаёт для изолированных сессий работы. Сейчас мы работаем в воркдереве `sleepy-williamson`.
Для Desktop-разработки они не нужны — можно игнорировать.

---

*Составил: Кодо | 2026-02-28*
