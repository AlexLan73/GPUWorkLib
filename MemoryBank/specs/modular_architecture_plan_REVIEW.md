# 📋 Code Review: modular_architecture_plan.md

> **Дата ревью**: 2026-04-11
> **Ревьюер**: Кодо
> **Объект**: [`MemoryBank/specs/modular_architecture_plan.md`](modular_architecture_plan.md)
> **Метод**: sequential-thinking (8 шагов) + context7 (CMake docs)
> **Формат**: каждый пункт помечен чекбоксом для ответа Alex

---

## Как пользоваться

По каждому пункту Alex отвечает одним из вариантов:
- ✅ **Принято** — исправить в плане
- ❌ **Отклонено** — оставить как есть (указать причину)
- 💬 **Обсудить** — нужна дискуссия
- 🔄 **Позже** — учесть, но не сейчас

---

## Критические проблемы 🔴

### [ ] #1. `test_utils` протёк в публичный API

**Где**: раздел 4.1, строки 191-195 и 204

```cmake
# ПРОБЛЕМА: тестовая инфраструктура в PUBLIC!
target_include_directories(DspCore
  PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/test_utils/include>  ← ❌
    $<INSTALL_INTERFACE:include>
)

install(DIRECTORY include/ test_utils/include/   ← ❌ устанавливается к потребителям
  DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}
)
```

**Проблема**: `TestRunner`, `GpuBenchmarkBase` и т.д. попадают ко всем потребителям `DspCore`. Тестовая инфраструктура не должна быть частью публичного API библиотеки.

**Исправление**: создать отдельную INTERFACE target `DspCore::TestUtils`:

```cmake
add_library(DspCoreTestUtils INTERFACE)
add_library(DspCore::TestUtils ALIAS DspCoreTestUtils)
target_include_directories(DspCoreTestUtils INTERFACE
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/test_utils/include>
  $<INSTALL_INTERFACE:include/dsp/test>
)
target_link_libraries(DspCoreTestUtils INTERFACE DspCore::DspCore)

# Install отдельно
install(TARGETS DspCoreTestUtils EXPORT DspCoreTargets)
install(DIRECTORY test_utils/include/
        DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/dsp/test)

# tests/ подключают обе:
# target_link_libraries(my_test DspCore::DspCore DspCore::TestUtils)
```

**Ответ Alex**:
```
Согласен - Да
```

---

### [ ] #2. CI preset задаёт переменные, которые `fetch_deps.cmake` не читает

**Где**: разделы 4.3 и 5

CI preset (строки 335-343):
```json
"DSP_CORE_TAG":       "v0.1.0",
"DSP_SPECTRUM_TAG":   "v0.1.0",
...
```

Но в `fetch_deps.cmake` (строка 293):
```cmake
macro(fetch_dsp_core version)
  dsp_fetch_package(DspCore ${version} core v${version}.0)
  #                                         ↑ тег ЗАХАРДКОЖЕН
  #                                         ↑ переменная DSP_CORE_TAG игнорируется
endmacro()
```

**Проблема**: CI preset никогда не использует свои переменные — всегда берёт хардкод. Результат: при попытке CI собрать конкретный тег получит что-то другое.

**Исправление**:
```cmake
macro(fetch_dsp_core version)
  set(_tag "${DSP_CORE_TAG}")
  if(NOT _tag)
    set(_tag "v${version}.0")
  endif()
  dsp_fetch_package(DspCore ${version} core ${_tag})
endmacro()
```

**Ответ Alex**:
```
Да
```

---

### [ ] #3. Противоречие и дублирование разделов 9 и 11

**Где**: раздел 9 (строка 553) и раздел 11 (строка 656)

Одинаковый заголовок **"Что НЕ переносим сразу"**, но разный контент:

| Что | Раздел 9 | Раздел 11 |
|-----|----------|-----------|
| `Doc_Addition/` | «Остаётся в GPUWorkLib **или** переходит в DSP/Doc» | «Переходит в `DSP/Doc/addition/`» |
| Ветка `nvidia` | «Только одна ветка `main` (AMD)» | «Только одна ветка `main` (AMD ROCm)» |

**Проблема**: прямое противоречие по `Doc_Addition/` — непонятно что реально делать.

**Исправление**: удалить раздел 9, оставить раздел 11 с окончательным решением для `Doc_Addition/`.

**Ответ Alex**:
```
## Doc_Addition - переносим - потом посмотрим и почистим
##  Ветка `nvidia` | «Только одна ветка `main` (AMD)» | «Только одна ветка `main` (AMD ROCm)» |
-- я писал про то, что мы в этом проекте на c++ пишем только под rocm под debian. но аналитические модели будем делать и под windows
```

---

### [ ] #4. `find_dependency` не упомянут в `Config.cmake.in`

**Где**: раздел 4.2 (строка 268 «install + export аналогично core»)

**Проблема**: для зависимых репо (spectrum, signal, ...) правила отличаются от core. Они обязаны объявить свои зависимости в `Config.cmake.in`, иначе у потребителя `find_package(DspSpectrum)` упадёт с «target not found: DspCore::DspCore».

**Исправление**: в каждом зависимом репо создать `cmake/Dsp{Repo}Config.cmake.in`:

```cmake
@PACKAGE_INIT@

include(CMakeFindDependencyMacro)
find_dependency(DspCore REQUIRED)
find_dependency(HIP REQUIRED)
find_dependency(hipfft REQUIRED)   # для spectrum

include("${CMAKE_CURRENT_LIST_DIR}/DspSpectrumTargets.cmake")
check_required_components(DspSpectrum)
```

В плане раздела 4.2 нужно явно показать этот файл, а не отсылать к «аналогично core».

**Ответ Alex**:
```
Да
```

---

## Важные замечания 🟡

### [ ] #5. `radar` зависит от `spectrum` без обоснования

**Где**: раздел 2, таблица зависимостей + описание «Почему именно такое разбиение»

**Противоречие в документе**:
- Описание (строка 84): *«Оба используют hipFFT напрямую (не через fft_func модуль)»*
- Таблица (строка 65): `radar` → зависит от `core + spectrum`

**Проблема**: если `radar` использует `hipFFT` напрямую (минуя `fft_func`), зачем тянуть весь `spectrum` со statistics/filters/lch_farrow?

**Варианты исправления**:
1. `radar` зависит только от `core + hipFFT`, убрать spectrum
2. Если что-то из spectrum API всё же используется — уточнить что именно и задокументировать
3. Рассмотреть: может, `range_angle` использует `statistics` или `lch_farrow`?

**Ответ Alex**:
```
radar - должен работать правильно в общем стандарте, Если считает спектр или статистику должен  использовать отлаженную библиотеку spectrum
то же относится и к другим решениям - ненужно создавать новые сущности

```

---

### [ ] #6. Silent version mismatch при diamond dependency

**Где**: раздел 8.1

**Утверждение в плане**: *«FetchContent автоматически дедуплицирует по имени… Второй вызов — no-op»*

**Что упущено**: дедупликация работает по **имени**, но если версии разные — FetchContent тихо возьмёт **первый объявленный**, без warning:

```
strategies → spectrum → FetchContent_Declare(DspCore GIT_TAG v0.1.0)
strategies → signal   → FetchContent_Declare(DspCore GIT_TAG v0.2.0)  ← ИГНОРИРУЕТСЯ!
```

**Исправление**: добавить в раздел 8.1 предупреждение:

> ⚠️ **Важно**: при поднятии версии `core` нужно синхронно обновить тег во всех зависимых репо. FetchContent не сообщит об несовпадении версий — возьмёт первый объявленный.

Альтернатива: в мета-репо DSP централизованно объявлять `FetchContent_Declare(DspCore ...)` ПЕРЕД подключением зависимых.

**Ответ Alex**:
```
думаю этот лучший вариант
Альтернатива: в мета-репо DSP централизованно объявлять `FetchContent_Declare(DspCore ...)` ПЕРЕД подключением зависимых.
```

---

### [ ] #7. `buildPresets` неполный

**Где**: раздел 5 (строки 377-380)

```json
"buildPresets": [
  { "name": "local-dev", "configurePreset": "local-dev" },
  { "name": "ci",        "configurePreset": "ci"        }
  // ← "full-release", "spectrum-only", "linalg-only" отсутствуют
]
```

**Проблема**: `cmake --build --preset full-release` упадёт с «preset not found».

**Исправление**: добавить buildPresets для всех configurePresets:
```json
{ "name": "full-release",  "configurePreset": "full-release"  },
{ "name": "spectrum-only", "configurePreset": "spectrum-only" },
{ "name": "linalg-only",   "configurePreset": "linalg-only"   }
```

**Ответ Alex**:
```
Да
```

---

### [ ] #8. `DSP_PYTHON_LIB_DIR` нет fallback для standalone сборки

**Где**: раздел 10.2 (строки 601-611)

```cmake
# core/python/CMakeLists.txt
install(TARGETS dsp_core
  LIBRARY DESTINATION ${DSP_PYTHON_LIB_DIR}   # ← undefined при standalone!
)
```

**Проблема**: при `cmake --install` из одного только `core/` (без мета-репо DSP) переменная `DSP_PYTHON_LIB_DIR` не задана → установка уйдёт в корень системы или упадёт.

**Исправление**: в каждом `{repo}/python/CMakeLists.txt` добавить fallback:
```cmake
if(NOT DEFINED DSP_PYTHON_LIB_DIR)
  set(DSP_PYTHON_LIB_DIR "${CMAKE_INSTALL_PREFIX}/python/lib"
      CACHE PATH "Destination for compiled Python bindings")
endif()
```

**Ответ Alex**:
```
Да
```

---

### [ ] #9. Потеря git history при копировании не задокументирована

**Где**: раздел 7, Фаза 2 (строки 461-472)

**Проблема**: «копируем DrvGPU/ из GPUWorkLib» — простое копирование теряет весь `git blame` и `git log` для 880 файлов / 110K LOC.

**Осознанное решение или упущение?**

**Альтернативы**:
- `git subtree split --prefix=modules/fft_func -b fft-branch` — сохраняет историю коммитов
- `git filter-repo --subdirectory-filter modules/fft_func` — то же самое, быстрее на больших репо

**Исправление**: добавить в раздел 7 явное решение:
> **Стратегия переноса истории**: копируем без истории (осознанный выбор для упрощения) / используем git subtree split.

**Ответ Alex**:
```
Не нужно истории. если в этом пявится необходимость посмотрим в этом репо
```

---

### [ ] #10. `.cl` kernel runtime path стратегия не описана

**Где**: разделы 3 и 7 (Фаза 2)

**Проблема**: в GPUWorkLib kernels загружаются через `KERNELS_DIR` из `kernel_loader.hpp`. При разбиении на репо каждый модуль хранит свои `.cl` файлы в `src/{module}/kernels/`. Вопросы:
- Как runtime находит kernels после `cmake --install`?
- Куда копируются `.cl` файлы при сборке?
- Как каждый модуль задаёт свой `KERNELS_DIR`?

**Исправление**: добавить в шаблон репо (раздел 3):
```cmake
# Копировать kernels в build/ для разработки
file(COPY src/${module}/kernels/
     DESTINATION ${CMAKE_BINARY_DIR}/kernels/${module})

# Install kernels
install(DIRECTORY src/${module}/kernels/
        DESTINATION ${CMAKE_INSTALL_DATADIR}/dsp/kernels/${module}
        FILES_MATCHING PATTERN "*.cl")

# Runtime path
target_compile_definitions(DspSpectrum PRIVATE
  DSP_KERNELS_DIR="${CMAKE_INSTALL_DATADIR}/dsp/kernels")
```

**Ответ Alex**:
```
кернел для OpenCl не копируем мы не считаем на OpenCl. OpenCl - только для стыковки данных в GPU 
```

---

### [ ] #11. `pybind11` как зависимость не описан

**Где**: раздел 10.2

**Проблема**: в шаблоне используется `pybind11_add_module(...)`, но:
- Откуда берётся `pybind11`?
- Какая версия?
- FetchContent / системный / vcpkg?

**Исправление**: добавить в `fetch_deps.cmake`:
```cmake
macro(fetch_pybind11)
  FetchContent_Declare(pybind11
    GIT_REPOSITORY https://github.com/pybind/pybind11.git
    GIT_TAG        v2.12.0
    FIND_PACKAGE_ARGS NAMES pybind11 CONFIG
  )
  FetchContent_MakeAvailable(pybind11)
endmacro()
```

Вызывать в `{repo}/python/CMakeLists.txt` если `DSP_BUILD_PYTHON=ON`.

**Ответ Alex**:
```
Посмотри как сейчас это работает
```

---

## Рекомендации 🟢

### [ ] R1. Использовать `FIND_PACKAGE_ARGS` (Modern CMake 3.24+)

**Где**: раздел 4.3

Вместо ручного macro `dsp_fetch_package` с `find_package() + if(NOT FOUND)`:

```cmake
FetchContent_Declare(DspCore
  GIT_REPOSITORY https://github.com/dsp-gpu/core.git
  GIT_TAG        v0.1.0
  FIND_PACKAGE_ARGS NAMES DspCore CONFIG
)
FetchContent_MakeAvailable(DspCore)
```

**Преимущества**:
- Официально рекомендован CMake docs (подтверждено через context7)
- `FETCHCONTENT_TRY_FIND_PACKAGE_MODE=ALWAYS` — глобально включается через env
- Меньше кода, единый entry point

Текущий вариант работает, но устарел.

**Ответ Alex**:
```
Да
```

---

### [ ] R2. Аудит зависимостей перед Фазой 1

**Где**: раздел 7, перед Фазой 1

Граф в разделе 2 нарисован «из головы / из кода». Перед началом Фазы 1 сделать автоматическую проверку:

```bash
# Запустить из текущего GPUWorkLib
cmake --graphviz=deps.dot -S . -B build
dot -Tsvg deps.dot -o deps.svg
```

Сравнить с графом в разделе 2. Могут всплыть транзитивные зависимости которые не видны «глазами».

**Ответ Alex**:
```
Создать зависимости как в Doc\Architecture - их проанализировать а потом делать
```

---

### [ ] R3. Logs/ и Results/ в новой структуре

**Где**: разделы 3 и 6 (не упомянуты)

**Вопрос**: куда в новой архитектуре идут:
- `Logs/DRVGPU_XX/` (per-GPU логи plog)
- `Results/Profiler/` (GPUProfiler отчёты)
- `Results/Plots/` (графики из Python тестов)
- `Results/JSON/` (результаты тестов)

**Предложение**: добавить в структуру DSP мета-репо:
```
DSP/
├── Logs/          ← per-GPU логи (gitignore)
├── Results/       ← JSON, Plots, Profiler (gitignore)
├── Python/lib/    ← .pyd/.so (gitignore)
└── Python/        ← тесты (commit)
```

И в `.gitignore`:
```gitignore
Logs/
Results/
Python/lib/
Python/__pycache__/
```

**Ответ Alex**:
```
Да
```

---

### [ ] R4. Проверить `configGPU.json` перед публикацией

**Где**: раздел 10.5 (public visibility)

Перед публикацией в public GitHub:
- Убедиться что `configGPU.json` не содержит локальных путей
- Нет IP-адресов / hostname
- Нет внутренних конфигураций отдела

Добавить в план пункт «audit всех config файлов» перед публикацией.

**Ответ Alex**:
```
Да
```

---

### [ ] R5. Namespace migration план для pybind11 (раздел 8.3)

**Проблема**: при добавлении `namespace dsp::` в C++ — pybind11 module names не изменятся, но внутренние Python-имена классов нужно проверить.

**Вопрос к обсуждению**: добавить namespace сразу при копировании в новые репо, или отложить?

Плюсы «сразу»: избегаем второго рефакторинга
Минусы «сразу»: замедляет Фазу 2, риск ошибок

**Ответ Alex**:
```
Да
```

---

## Архитектурные вопросы к обсуждению 🟣

### [ ] A1. `spectrum` содержит `statistics` — нарушение минимальности

Кто-то, кто хочет только статистику, тянет весь `spectrum` (включая `hipFFT`, `fft_func`, `filters`, `lch_farrow`). `statistics` концептуально не связан с FFT.

**Варианты**:
1. Оставить как есть — прагматично, зависимости одни (ROCm/rocprim)
2. Вынести `statistics` в отдельный `stats` репо
3. Разделить `spectrum` → `analysis` (FFT-based) + `stats`

**Ответ Alex**:
```
согласен 2. Вынести `statistics` в отдельный `stats` репо
```

---

### [ ] A2. `signal` = generators + heterodyne — гранулярность

Если потребителю нужны только генераторы (без heterodyne), он всё равно получит зависимость от `spectrum` (через heterodyne → fft_func).

**Варианты**:
1. Оставить как есть — heterodyne логически связан с LFM генерацией
2. Разделить `signal` → `signal_generators` + `heterodyne` (2 репо)

**Ответ Alex**:
```
Да   Разделить `signal` → `signal_generators` + `heterodyne` (2 репо)
```

---

## Итоговая оценка

**Сильные стороны** ✅:
- Граф зависимостей логичен и обоснован
- Вариант B (не трогаем рабочий GPUWorkLib) — правильная стратегия
- Detailed CMake templates — не просто «надо сделать»
- CMakePresets с local-dev через `FETCHCONTENT_SOURCE_DIR_*` — elegant
- Чёткий Python bindings план
- Закрытые решения в разделе 10 (версионирование, bindings, config, CI, видимость)

**Слабые стороны** ⚠️:
- 4 критические проблемы требуют исправления перед началом Фазы 1
- Неполные CMake шаблоны (test_utils, Config.cmake.in, pybind11, kernels)
- Противоречие разделов 9 и 11
- Несколько «невидимых дыр»: git history, .cl runtime path, Logs/Results

**Вердикт**: план хороший по структуре и мышлению, но **перед Фазой 1 закрыть критические проблемы**.

---

## Как отвечать

Открой этот файл и впиши ответы в секции `**Ответ Alex**:` под каждым пунктом. После ответов Кодо соберёт исправления в обновлённую версию `modular_architecture_plan.md`.

---

*Создан: 2026-04-11 | Ревьюер: Кодо*
*Инструменты: sequential-thinking (8 шагов) + context7 (CMake docs)*

