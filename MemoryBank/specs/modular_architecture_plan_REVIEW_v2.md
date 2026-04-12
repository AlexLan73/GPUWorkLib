# Code Review v2: modular_architecture_plan.md

> **Дата**: 2026-04-12
> **Ревьюер**: Кодо
> **Объект**: [`modular_architecture_plan.md`](modular_architecture_plan.md) (v2 после первого ревью)
> **Метод**: sequential-thinking (15 шагов) + context7 (CMake, HIP/ROCm) + WebSearch + 3 Explore agents
> **Формат**: чекбоксы для ответа Alex

---

## Как пользоваться

По каждому пункту Alex отвечает:
- **Да** — исправить в плане
- **Нет** — оставить как есть (указать причину)
- **Обсудить** — нужна дискуссия
- **Позже** — учесть, но не сейчас

---

## Статус первого ревью (18 пунктов)

**16/18 полностью внесены** в план v2. Два частично:

| # | Пункт | Статус |
|---|-------|--------|
| #6 | Diamond dependency warning | Решение есть, explicit warning ослаблен |
| #10 | Kernel runtime path | OpenCL закрыт, **HIP kernels не описаны** |

Остальные 16 пунктов (#1-#5, #7-#9, #11, R1-R5, A1-A2) — корректно внесены.

---

## Миграция
### Предварительные работы
1. Умное копирование в папку будущего проекта E:\DSP-GPU все управляющие данные
 - Claude & Cursor, CLAUDE.md
 - MCP-server
 - Настройки VSCode
 - MemoryBank  - и туда поместить задачу по меграции
2. пометить, что базовый каталог с рабочим проектом лежит на E:\C++\GPUWorkLib\ (windows) и ...\C++\GPUWorkLib\ (debian)
3. Я может что то не так представляю поправь
  E:\DSP-GPU - это основная папка проекта- она есть на github.com\dsp-gpu\. Я планировал в ней собирать весь проект для демонстрации из всех проектов + каждый проект живет своей жизню. Но как мне правильно организовать это я не знаю.
4. Предже чем переносить, построй структуру по аналогии Doc\Architecture со всеми связями это будет базовая архитектура. посмотрим обсудим 


## Критические проблемы 🔴

### [ ] #1. `DSP/CMakeLists.txt` — нет conditional build, presets сломаны

**Где**: раздел 8.1 (строки 714-733) vs presets (строки 497-520)

Preset `spectrum-only` задает `DSP_BUILD_STATS=OFF` и т.д., но `DSP/CMakeLists.txt` **безусловно** вызывает все `fetch_dsp_*()`:

```cmake
# Так в плане сейчас (раздел 8.1):
fetch_dsp_core()
fetch_dsp_spectrum()
fetch_dsp_stats()             # <-- тянет даже при DSP_BUILD_STATS=OFF!
fetch_dsp_signal_generators() # <-- аналогично
fetch_dsp_heterodyne()
fetch_dsp_linalg()
fetch_dsp_radar()
fetch_dsp_strategies()
```

**Результат**: `cmake --preset spectrum-only` всё равно клонирует ВСЕ 8 репо с GitHub.

**Исправление**: добавить `option()` + условные вызовы:

```cmake
# DSP/CMakeLists.txt — исправленный
option(DSP_BUILD_SPECTRUM          "Build spectrum"          ON)
option(DSP_BUILD_STATS             "Build stats"             ON)
option(DSP_BUILD_SIGNAL_GENERATORS "Build signal_generators" ON)
option(DSP_BUILD_HETERODYNE        "Build heterodyne"        ON)
option(DSP_BUILD_LINALG            "Build linalg"            ON)
option(DSP_BUILD_RADAR             "Build radar"             ON)
option(DSP_BUILD_STRATEGIES        "Build strategies"        ON)

# core — всегда
fetch_dsp_core()

if(DSP_BUILD_SPECTRUM)          fetch_dsp_spectrum()          endif()
if(DSP_BUILD_STATS)             fetch_dsp_stats()             endif()
if(DSP_BUILD_SIGNAL_GENERATORS) fetch_dsp_signal_generators() endif()
if(DSP_BUILD_HETERODYNE)        fetch_dsp_heterodyne()        endif()
if(DSP_BUILD_LINALG)            fetch_dsp_linalg()            endif()
if(DSP_BUILD_RADAR)             fetch_dsp_radar()             endif()
if(DSP_BUILD_STRATEGIES)        fetch_dsp_strategies()        endif()
```

**Ответ Alex**:
```
Да
```

---

### [ ] #2. `find_package(HIP)` — case-sensitive, не найдет на Linux!

**Где**: строки 218, 303, 324

```cmake
find_package(HIP REQUIRED)      # <-- так в плане
find_dependency(HIP REQUIRED)   # <-- так в Config.cmake.in
```

**Проблема**: ROCm устанавливает `hip-config.cmake` (lowercase). На Linux (case-sensitive!) CMake ищет `HIPConfig.cmake` — **не найдет**.

Подтверждено через [ROCm CMake documentation](https://rocm.docs.amd.com/en/latest/conceptual/cmake-packages.html):
imported targets `hip::host`, `hip::device` доступны через `find_package(hip)`.

**Исправление**: везде `HIP` -> `hip`:

```cmake
find_package(hip REQUIRED)            # CMakeLists.txt
find_dependency(hip REQUIRED)         # Config.cmake.in
target_link_libraries(... hip::host)  # уже правильно в плане
```

**Ответ Alex**:
```
Да
```

---

### [ ] #3. Python API breaking change — нет плана миграции

**Где**: раздел 10.2 описывает **результат** (8 .pyd), но не описывает **как мигрировать**

**Сейчас**: один модуль `gpuworklib.pyd`, все тесты делают:
```python
import gpuworklib
ctx = gpuworklib.GPUContext(...)
fft = gpuworklib.FFTProcessor(ctx, ...)
```

**В плане**: 8 отдельных модулей:
```python
import dsp_core
import dsp_spectrum
ctx = dsp_core.GPUContext(...)
fft = dsp_spectrum.FFTProcessor(ctx, ...)
```

**Что затронуто**:
- Каждый Python-тест (`import gpuworklib` — десятки файлов)
- `GPULoader` в `common/gpu_loader.py` — ищет один `gpuworklib*` файл
- Все `conftest.py` (filters, heterodyne, signal_generators, statistics и др.)
- `run_tests.py` и `run_all_rocm_tests.sh`

**Исправление**: добавить раздел **«10.6 Миграция Python API»**:

1. **Compatibility shim** (переходный период):
```python
# DSP/Python/lib/gpuworklib.py
"""Compatibility: реэкспорт из новых модулей."""
from dsp_core import *
from dsp_spectrum import *
from dsp_stats import *
from dsp_signal_generators import *
from dsp_heterodyne import *
from dsp_linalg import *
from dsp_radar import *
from dsp_strategies import *
```

2. **Рефакторинг GPULoader** — multi-module loading (8 модулей вместо одного)
3. **Обновление conftest.py** — в каждом модуле
4. **Обновление run_tests.py / run_all_rocm_tests.sh**
5. **Timeline**: отдельная Фаза 3.5

Или альтернатива: **не разбивать** .pyd, оставить один `dsp.pyd` в мета-репо (собирается из всех модулей). Тогда Python API не ломается.

**Ответ Alex**:
```
Да
```

---

## Важные замечания 🟡

### [ ] #4. HIP kernels — cache paths и .hip файлы не описаны

**Где**: раздел 11 упоминает только `.cl` kernels

В проекте 7+ файлов `.hip` + inline kernel sources в `*_kernels_rocm.hpp`. Код использует **относительные пути** для cache скомпилированных HSACO:

```cpp
// Текущий код (modules/fft_func/src/complex_to_mag_phase_rocm.cpp):
ctx_(backend, "C2MP", "modules/fft_func/kernels")  // <-- этот путь сломается!
```

В новой структуре `modules/fft_func/` не существует.

**Исправление**: добавить в раздел 11:
> `.hip` файлы и `*_kernels_rocm.hpp` — переносим в `src/{module}/kernels/`.
> В Фазе 3 обновить cache paths в конструкторах GpuContext.

**Ответ Alex**:
```
Предлагаю в каждом проектн создать отднльную папку для файлов с .hpp посмотри как сделано с *.cl сделать нужно на подобе
давай облудим
```

---

### [ ] #5. Namespace migration ~110K LOC — timeline нереалистичен

**Где**: Фаза 3 шаг 24 (строка 680)

Добавить `namespace dsp::` + вложенные namespaces во все 880 файлов — это не задача на «День 2-3». Нужно обернуть каждый `.hpp`, `.cpp`, `.hip`, обновить forward declarations, при этом HIP kernel source strings (R"HIP(...)") **не должны** быть внутри namespace.

**Варианты**:
1. Выделить namespace в отдельную Фазу 3b (3-5 дней)
2. Автоматизировать скриптом + ручная доводка
3. Отложить namespace до стабилизации новых репо

**Ответ Alex**:
```
1. Выделить namespace в отдельную Фазу 3b (3-5 дней)
2. Автоматизировать скриптом + ручная доводка
3. я плаирую перннести (умное копирование с редактированием), затем у нас уже настроенны CMAKE файлы на пустой структуре! затем будем включать  по одному проекты  и настраивать. у нас будет рабочий проект для примера
```

---

### [ ] #6. testPresets отсутствуют в CMakePresets.json

**Где**: раздел 5 (строки 533-539) — есть configurePresets и buildPresets, но нет testPresets

Без них `ctest --preset ci` не работает.

**Исправление**:
```json
"testPresets": [
  { "name": "local-dev", "configurePreset": "local-dev",
    "output": { "outputOnFailure": true } },
  { "name": "ci", "configurePreset": "ci",
    "output": { "outputOnFailure": true } }
]
```

**Ответ Alex**:
```
Да
```

---

### [ ] #7. Diamond dependency — explicit warning не хватает

**Где**: раздел 8.1

Решение (централизованное объявление в мета-репо) — правильное. Но не хватает explicit warning для разработчиков:

> **При поднятии версии core (или любого зависимого) — синхронно обновить тег
> во ВСЕХ зависимых репо.** FetchContent не сообщит о несовпадении версий —
> тихо возьмёт первый объявленный. При изолированной сборке одного репо
> (без мета-репо) разработчик может получить другую версию core.

**Ответ Alex**:
```
" Но не хватает explicit warning для разработчиков:"
 как это исправить?
 -может в CMake у разработчиков нужно проверять версию и если отличается копировать в проект новые файлы
```

---

## Рекомендации 🟢

### [ ] R1. `macro` -> `function` для fetch wrappers

**Где**: раздел 4.3 (строки 432-439)

`macro` расширяется в вызывающем scope — переменные могут «просочиться». `function` безопаснее:

```cmake
# БЫЛО:
macro(fetch_dsp_core) dsp_fetch_package(...) endmacro()
# СТАЛО:
function(fetch_dsp_core) dsp_fetch_package(...) endfunction()
```

**Ответ Alex**:
```
Да
```

---

### [ ] R2. `CMAKE_EXPORT_COMPILE_COMMANDS=ON` в base preset

Для clangd/IDE автодополнения во всех подмодулях:

```json
"cacheVariables": {
  "CMAKE_EXPORT_COMPILE_COMMANDS": "ON"
}
```

**Ответ Alex**:
```
Да
```

---

### [ ] R3. `project()` с DESCRIPTION для Doxygen

```cmake
project(DspCore VERSION 0.1.0 DESCRIPTION "GPU driver and profiler" LANGUAGES CXX HIP)
```

CMake `PROJECT_DESCRIPTION` подхватывается Doxygen и CPack автоматически.

**Ответ Alex**:
```
Да
- у нас еще нужен будет latex - нужно сразу закладываться
```

---

### [ ] R4. `.pyi` type stubs для всех 8 модулей

План упоминает `dsp_core.pyi` для core, но не для остальных 7. Type stubs нужны для IDE autocompletion в Python.

**Ответ Alex**:
```
Да
```

---

## Итоговая оценка плана v2

**Сильные стороны**:
- Все 16/18 пунктов первого ревью корректно внесены
- CMake шаблоны детальные и рабочие (с поправкой на hip case)
- Граф зависимостей — чистый DAG без циклов, хорошо обоснован
- FetchContent + FIND_PACKAGE_ARGS — подтверждён как best practice (context7)
- Фаза 0 (аудит) — правильно добавлена перед началом работы
- Python bindings через pip pybind11 — проверено, совпадает с текущим кодом

**Требует исправления перед Фазой 1**:
- 🔴 #1 — conditional build в DSP/CMakeLists.txt (presets не работают)
- 🔴 #2 — `find_package(hip)` lowercase (сборка упадет на Linux)
- 🔴 #3 — раздел миграции Python API (breaking change)

**Можно отложить до Фазы 3**:
- 🟡 #4 — HIP kernel cache paths
- 🟡 #5 — namespace timeline
- 🟡 #6 — testPresets
- 🟡 #7 — diamond warning

---

## Как отвечать

Впиши ответы в секции `**Ответ Alex**:` под каждым пунктом.
После ответов Кодо внесёт исправления в `modular_architecture_plan.md` (v3).

---

*Создан: 2026-04-12 | Ревьюер: Кодо*
*Инструменты: sequential-thinking (15 шагов) + context7 (CMake, HIP/ROCm) + WebSearch + 3 Explore agents*
*Источники: [CMake FetchContent docs](https://cmake.org/cmake/help/latest/module/FetchContent.html), [ROCm CMake packages](https://rocm.docs.amd.com/en/latest/conceptual/cmake-packages.html), [ROCm 7.2.1 Release](https://github.com/ROCm/ROCm/releases)*
