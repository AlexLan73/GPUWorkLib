# Consolidated Review — CMake Planning (GPUWorkLib → LibGPU → Закрытый проект)

> **Дата**: 2026-04-09
> **Reviewer**: Кодо
> **Цель**: Ревью всех CMake-планов перед понедельником, подготовка roadmap «первопроходцев»
> **Контекст**: Alex попросил ревью планов перехода на другое формирование CMake

---

## 🎯 TL;DR

У нас **3 параллельно существующих плана** + 1 туториал + 1 таск:

| # | Файл | Дата | Что | Статус |
|---|---|---|---|---|
| 1 | [cmake_modular_architecture_plan.md](cmake_modular_architecture_plan.md) | 05.04 | Модульная CMake архитектура (Options + Presets + выборочная сборка) | 📋 готов к реализации, 10 тасков |
| 2 | [cmake_libgpu_integration_plan.md](cmake_libgpu_integration_plan.md) | 29.03 | LibGPU как «конверт» GPUWorkLib → закрытый проект | 📋 готов, уже ревьюен (см. review_issues) |
| 3 | [cmake_libgpu_research_agent.md](cmake_libgpu_research_agent.md) | 29.03 | Research: 6 вариантов интеграции, рекомендация FetchContent | 📖 справочник |
| 4 | [cmake_libgpu_review_issues.md](cmake_libgpu_review_issues.md) | 29.03 | Ревью плана LibGPU — 3🔴 + 6🟡 + 4🟢, все решения приняты | ✅ закрыт |
| 5 | [TASK_LibGPU_Integration.md](../tasks/TASK_LibGPU_Integration.md) | 29.03 | Таск A/B/C/D — блоки работ для фазы 2 | 📋 BACKLOG |
| 6 | [Primer.md](../../Doc_Addition/CMake/Primer.md) | 25.03 | Учебник: 5 подходов к CMake-интеграции, рекомендует **Подход 5** (FetchContent + FIND_PACKAGE_ARGS + Presets) | 📖 справочник |
| 7 | [Git_ALL.md](../../Doc_Addition/CMake/Git_ALL.md) | 25.03 | Учебник: multi-remote Git (не про CMake) | 📖 справочник |

**Главный вывод**: планы **не противоречат** друг другу, но **не согласованы между собой**:
- Одни и те же опции названы по-разному (`GPUWORKLIB_ENABLE_*` vs `LIBGPU_BUILD_*`)
- Разные версии `CMakePresets.json` (v3 / v4 / v6)
- modular_architecture использует `CMakeDependentOption` (автоматика), LibGPU использует `if(TARGET ...)` (ручная проверка)
- LibGPU-план создавался **до** modular_architecture — не знает о новом формировании

**Готовый roadmap для работы** — в конце документа (раздел "Консолидированный roadmap").

---

## 🗺️ Карта: что уже есть, чего нет

### Текущее состояние CMake в GPUWorkLib

| Файл | Есть? | Состояние | Что нужно |
|---|---|---|---|
| [CMakeLists.txt](../../CMakeLists.txt) | ✅ | 70 строк, жёстко `add_subdirectory` всех 11 модулей, без опций | Обернуть в `if(GPUWORKLIB_ENABLE_*)`, добавить FATAL_ERROR на ROCm |
| [CMakePresets.json](../../CMakePresets.json) | ✅ | version **3**, только 2 Windows-preset (vs2026 + nvidia) | Обновить до version 4, добавить Linux presets (minimal/signal-lab/radar-dsp/full) |
| [cmake/modules-options.cmake](../../cmake/) | ❌ | Нет файла | Создать по шаблону из modular_architecture, Шаг 1 |
| [modules/CMakeLists.txt](../../modules/CMakeLists.txt) | ⚠️ | 37 строк, пустой (только `add_drvgpu_module` функция, не вызывается) | Удалить — мёртвый код |
| [src/CMakeLists.txt](../../src/CMakeLists.txt) | ✅ | 259 строк, 12 блоков `if(TARGET)` по одному на модуль + 20 строк дублирующих include_directories | Заменить на `gpuworklib_link_module()` функцию (DRY) — **33 блока → 11 строк** |
| [python/CMakeLists.txt](../../python/CMakeLists.txt) | ✅ | 123 строки, **жёстко** линкует `drvgpu, signal_generators, fft_func, lch_farrow, filters, heterodyne` | Сделать условной через `gpuworklib_bind_module()` — иначе `--preset minimal` сломается на Python сборке |
| [DrvGPU/CMakeLists.txt](../../DrvGPU/CMakeLists.txt) | ✅ | **0 install() правил** (grep: 0 совпадений) | Добавить `install(TARGETS drvgpu EXPORT GPUWorkLibTargets …)` для Фазы 2 |

### Что существует в планах, но ещё не реализовано в коде

- ❌ `cmake/modules-options.cmake` (весь файл)
- ❌ `CMakePresets.json` Linux-пресеты (minimal, signal-lab, correlator-prod, radar-dsp, full, full-debug, full-python)
- ❌ `src/main.cpp` таблица-реестр `ModuleEntry` (сейчас три `#ifdef`-секции)
- ❌ `sync_to_libgpu.py` (скрипт синхронизации)
- ❌ `E:\C++\LibGPU` репозиторий (ещё не создан)
- ❌ `LibGPU/CMakeLists.txt`, `LibGPU/CMakePresets.json`, `GPUWorkLibConfig.cmake.in`
- ❌ Install/export правила в DrvGPU и модулях

---

## 🔄 Как 3 фазы связаны

```
┌─────────────────────────────────────────────────────────────────┐
│  ФАЗА 1 — Модульная CMake архитектура (GPUWorkLib)              │
│  План: cmake_modular_architecture_plan.md (05.04)                │
│  Что делает: выборочная сборка через GPUWORKLIB_ENABLE_*         │
│  Результат: cmake --preset signal-lab → только нужные модули     │
└──────────────────────────────┬──────────────────────────────────┘
                               │ даёт единый SSOT опций
                               ▼
┌─────────────────────────────────────────────────────────────────┐
│  ФАЗА 2 — LibGPU «конверт» (создание промежуточного репо)       │
│  План: cmake_libgpu_integration_plan.md (29.03)                  │
│  Что делает: sync GPUWorkLib → LibGPU + CMake для фазы 1         │
│  Результат: git submodule LibGPU в закрытом проекте работает    │
└──────────────────────────────┬──────────────────────────────────┘
                               │ даёт устойчивый target GPUWorkLib::*
                               ▼
┌─────────────────────────────────────────────────────────────────┐
│  ФАЗА 3 — Подключение в закрытом проекте                         │
│  Базовый вариант: add_subdirectory(libs/LibGPU)                  │
│  Улучшенный: Подход 5 из Primer.md (FetchContent + FIND_PKG)    │
│  Результат: закрытый проект переключается между submodule       │
│             и локальной копией через CMakePresets               │
└─────────────────────────────────────────────────────────────────┘
```

**Ключ**: фаза 2 НЕ МОЖЕТ корректно работать без фазы 1 — потому что LibGPU должен включать тот же `cmake/modules-options.cmake`. Поэтому порядок строгий.

---

## 🔴 Несогласованности между планами (4)

### ❗ 1. Разные имена опций

| Модуль | modular_architecture | libgpu_integration |
|---|---|---|
| FFT | `GPUWORKLIB_ENABLE_FFT` | `LIBGPU_BUILD_FFT` |
| Statistics | `GPUWORKLIB_ENABLE_STATISTICS` | `LIBGPU_BUILD_STATISTICS` |
| Signal generators | `GPUWORKLIB_ENABLE_SIGNAL_GENERATORS` | `LIBGPU_BUILD_GENERATORS` (короче!) |
| Vector algebra | `GPUWORKLIB_ENABLE_VECTOR_ALGEBRA` | `LIBGPU_BUILD_VECTOR_ALG` (короче!) |
| FM correlator | `GPUWORKLIB_ENABLE_FM_CORRELATOR` | `LIBGPU_BUILD_FM_CORR` (короче!) |
| Range angle | `GPUWORKLIB_ENABLE_RANGE_ANGLE` | `LIBGPU_BUILD_RANGE_ANGLE` |

**Риск**: если реализовать оба плана как есть — получим **две параллельные системы опций**, они не будут знать друг о друге. Пользователь в закрытом проекте задаст `LIBGPU_BUILD_FFT=OFF`, но модуль всё равно соберётся, потому что в `cmake/modules-options.cmake` он управляется через `GPUWORKLIB_ENABLE_FFT`.

**Решение**: ✅ Использовать **единую схему `GPUWORKLIB_ENABLE_*`** (из modular_architecture — более свежий, более гранулярный, согласован по именам). `LIBGPU_BUILD_*` оставить **только** для meta-опций сборки, которые не касаются модулей:
- `LIBGPU_BUILD_TESTS` (ON/OFF C++ тесты)
- `LIBGPU_BUILD_MAIN` (ON/OFF standalone exe)
- `LIBGPU_BUILD_PYTHON` (ON/OFF Python bindings)

**Где исправить**: в таске B2 (корневой CMakeLists.txt для LibGPU) — использовать `GPUWORKLIB_ENABLE_*` вместо `LIBGPU_BUILD_*` для модулей.

---

### ❗ 2. `CMakeDependentOption` vs `if(TARGET)`

**modular_architecture** использует автоматическое отключение зависимых модулей:

```cmake
CMakeDependentOption(GPUWORKLIB_ENABLE_SIGNAL_GENERATORS
    "Enable Signal Generators" ON
    "GPUWORKLIB_ENABLE_LCH_FARROW" OFF)
# ^ если lch_farrow=OFF, signal_generators автоматически OFF
```

**libgpu_integration** проверяет вручную:

```cmake
if(LIBGPU_BUILD_GENERATORS AND TARGET lch_farrow AND ROCM_ENABLED)
    add_subdirectory(modules/signal_generators)
endif()
```

**Риск**: разные подходы к одной проблеме. В LibGPU-плане если пользователь задаст `LIBGPU_BUILD_GENERATORS=ON`, `LIBGPU_BUILD_LCH_FARROW=OFF` — модуль **тихо не соберётся** без объяснения. В modular_architecture он явно в диагностике покажет `signal_generators : OFF`.

**Решение**: ✅ Использовать `CMakeDependentOption` везде. LibGPU корневой CMakeLists.txt **включает** тот же `cmake/modules-options.cmake`, что и GPUWorkLib (файл же попадает в LibGPU через sync_to_libgpu.py в составе `cmake/` whitelist).

---

### ❗ 3. Разные версии `CMakePresets.json`

| План | version | min CMake |
|---|---|---|
| Текущий `CMakePresets.json` | **3** | 3.21 |
| modular_architecture (исправлено по ревью) | **4** | 3.21 |
| libgpu_integration | **6** | 3.25 |

**Риск**: если LibGPU использует v6, а корневой проект v4, при обновлении всё может развалиться. CMake 3.25 — свежее обязательного минимума `cmake_minimum_required(VERSION 3.20)` в корневом CMakeLists.txt.

**Решение**: ✅ Везде **version 4** (покрывает все нужные фичи: inherits, cacheVariables, hidden, condition, displayName, description). LibGPU тоже v4.

---

### ❗ 4. ROCm check в разных местах

**modular_architecture**:
```cmake
# корневой CMakeLists.txt
include(cmake/dependencies.cmake)
if(NOT ROCM_ENABLED)
    message(FATAL_ERROR "ROCm/HIP not found! ...")
endif()
include(cmake/modules-options.cmake)  # ← опции объявляются ПОСЛЕ FATAL_ERROR
```

**libgpu_integration**:
```cmake
# корневой LibGPU/CMakeLists.txt
option(LIBGPU_BUILD_FFT "..." ON)      # ← опции объявлены ДО проверки ROCm
# ...
include(cmake/dependencies.cmake)
# ... модули проверяют ROCM_ENABLED по одному
if(LIBGPU_BUILD_FFT AND ROCM_ENABLED)
    add_subdirectory(modules/fft_func)
endif()
```

**Риск**: в LibGPU 11 повторных проверок `AND ROCM_ENABLED`. Если ROCm не найден, всё равно объявляются все `LIBGPU_BUILD_*` опции и CMake проходит конфигурацию с 11 предупреждениями «модуль не собирается». Это шум, и это **противоположно fail-fast** принципу из modular_architecture.

**Решение**: ✅ Унифицировать — в LibGPU тоже FATAL_ERROR на отсутствие ROCm сразу после `include(dependencies.cmake)`, и ОДИН set `cmake/modules-options.cmake`. Из модулей **убрать** `if(NOT ROCM_ENABLED) return()` — ROCm гарантирован (это уже TASK-CMAKE-07 в modular_architecture).

---

## 🟡 Гэпы в коде (чего нет для реализации)

| # | Файл | Что нет | Кто чинит |
|---|---|---|---|
| G1 | [cmake/modules-options.cmake](../../cmake/) | Весь файл | TASK-CMAKE-01 |
| G2 | [CMakePresets.json](../../CMakePresets.json) | version 3 → 4, нет Linux-пресетов | TASK-CMAKE-08 |
| G3 | [src/CMakeLists.txt](../../src/CMakeLists.txt) | 12 блоков if(TARGET) = DRY нарушено; 20 строк дублирующих include_directories | TASK-CMAKE-03 |
| G4 | [python/CMakeLists.txt](../../python/CMakeLists.txt) | Жёсткие `target_link_libraries(gpuworklib PRIVATE drvgpu signal_generators fft_func ...)` — не условные | TASK-CMAKE-05 |
| G5 | [src/main.cpp](../../src/main.cpp) | Нет таблицы-реестра `ModuleEntry`, три #ifdef-секции | TASK-CMAKE-04 |
| G6 | [modules/CMakeLists.txt](../../modules/CMakeLists.txt) | Мёртвый файл (пустая функция не вызывается) | TASK-CMAKE-06 (удалить) |
| G7 | [DrvGPU/CMakeLists.txt](../../DrvGPU/CMakeLists.txt) | 0 `install()` правил (grep подтверждён) | TASK_LibGPU_A1 |
| G8 | [модули]/CMakeLists.txt | `if(NOT ROCM_ENABLED) return()` — блокирует фазу 1 | TASK-CMAKE-07 |
| G9 | `sync_to_libgpu.py` | Весь файл, ~250 строк | TASK_LibGPU_A2 |
| G10 | `E:\C++\LibGPU\*` | Всего репо | TASK_LibGPU_B1-B5 |

---

## 📘 Подход 5 из Primer.md — когда применять

Primer.md рекомендует **гибрид FetchContent + FIND_PACKAGE_ARGS + CMakePresets**:

```cmake
# В CMakeLists.txt закрытого проекта:
include(FetchContent)

FetchContent_Declare(
    GPUWorkLib
    GIT_REPOSITORY ssh://server/srv/repos/LibGPU.git
    GIT_TAG v1.2.0
    FIND_PACKAGE_ARGS NAMES GPUWorkLib   # ← сначала пытается find_package
)

FetchContent_MakeAvailable(GPUWorkLib)

add_executable(closed_app src/main.cpp)
target_link_libraries(closed_app PRIVATE
    GPUWorkLib::drvgpu
    GPUWorkLib::fft_func
    GPUWorkLib::strategies)
```

```jsonc
// CMakePresets.json закрытого проекта:
{
  "configurePresets": [
    {
      "name": "dev-local",
      "description": "Разработчик правит GPUWorkLib локально",
      "cacheVariables": {
        "FETCHCONTENT_SOURCE_DIR_GPUWORKLIB": "${sourceDir}/../GPUWorkLib"
      }
    },
    {
      "name": "prod",
      "description": "Production — submodule + tag v1.2.0",
      "cacheVariables": {
        "FETCHCONTENT_SOURCE_DIR_GPUWORKLIB": ""
      }
    }
  ]
}
```

### Приоритеты при configure (автоматически):
1. **Если `FETCHCONTENT_SOURCE_DIR_GPUWORKLIB` задан** → берёт локальный путь (dev-local)
2. **Если `find_package(GPUWorkLib)` успешен** → использует установленный пакет
3. **Иначе** → клонирует из Git по тегу

### Преимущества Подхода 5 vs текущий LibGPU-план (submodule + add_subdirectory)

| Критерий | submodule + add_subdirectory | **Подход 5 (гибрид)** |
|---|---|---|
| Начальная настройка | 2 команды (submodule add + commit) | 1 строка в CMakeLists.txt |
| Локальная разработка | git submodule (сложно переключать между ветками) | ✅ Preset `dev-local` → указать путь и всё |
| CI/CD | `git submodule update --init` | ✅ Без дополнительных шагов |
| Отладка / Step Into | ✅ Работает | ✅ Работает |
| Фиксация версии | git commit SHA в родительском репо | ✅ GIT_TAG в CMakeLists.txt |
| Забыть `--recursive` | ⚠️ Частая проблема | ✅ Нет |
| Переключение между tag/local | 2 команды git | ✅ 1 параметр preset |

### Когда применять Подход 5?

**НЕ СРАЗУ.** Подход 5 требует чтобы в LibGPU работали:
1. ✅ `install(EXPORT GPUWorkLibTargets …)` — нужно для find_package fallback (сейчас только в модулях, нет в DrvGPU)
2. ✅ `GPUWorkLibConfig.cmake.in` — корректный (готов в integration_plan, Шаг 5)
3. ✅ Транзитивные зависимости через `find_dependency()` — готово
4. ✅ Kernel-файлы устанавливаются через `install(DIRECTORY kernels/ …)` — TODO Фазы 2

**Рекомендация**:
- **Фаза 2 (сейчас)**: `add_subdirectory(libs/LibGPU)` + `git submodule` — ПРОСТО, быстро, работает.
- **Фаза 2.5 (через 1–2 недели после запуска LibGPU)**: добавить `install/export` в DrvGPU и kernel-файлы, проверить что `find_package(GPUWorkLib CONFIG)` работает.
- **Фаза 3 (когда всё стабильно)**: перевести закрытый проект на Подход 5 — **можно даже БЕЗ ломания** текущей submodule-интеграции, потому что `FIND_PACKAGE_ARGS` сначала ищет установленный пакет и только потом клонирует.

---

## 🗺️ Консолидированный roadmap

> **Допущение**: работаем на ветке `main` (Debian + ROCm). Все 3 фазы независимы и могут выполняться последовательно одним человеком или параллельно двумя.

### 🔴 Фаза 0 — Выравнивание планов (10 минут, можно прямо сегодня)

**Цель**: исправить несогласованности между планами **до** начала реализации.

- [ ] **F0.1** — Обновить [cmake_libgpu_integration_plan.md](cmake_libgpu_integration_plan.md): заменить все `LIBGPU_BUILD_*` для модулей на `GPUWORKLIB_ENABLE_*`. Оставить `LIBGPU_BUILD_TESTS`/`MAIN`/`PYTHON` (это meta-опции, не модули)
- [ ] **F0.2** — Обновить `version: 6` → `version: 4` в примере CMakePresets.json внутри libgpu_integration_plan
- [ ] **F0.3** — В libgpu_integration_plan перевести ручные `if(X AND ROCM_ENABLED)` на `include(cmake/modules-options.cmake)` + `FATAL_ERROR` на ROCm
- [ ] **F0.4** — В libgpu_integration_plan из секции «корневой CMakeLists» убрать 11 повторных `AND ROCM_ENABLED` — ROCm гарантирован в одном месте
- [ ] **F0.5** — Добавить в libgpu_integration «примечание»: «Опции модулей определяются в cmake/modules-options.cmake (SSOT) — этот файл синхронизируется через sync_to_libgpu.py»

Эти правки делают планы **совместимыми**. После Фазы 0 можно реализовывать.

---

### 🟢 Фаза 1 — Модульная CMake архитектура в GPUWorkLib (10 тасков)

**Основа**: [cmake_modular_architecture_plan.md](cmake_modular_architecture_plan.md), Changelog v3.

| ID | Файл/Действие | Сложность |
|---|---|---|
| **CMAKE-01** | Создать [cmake/modules-options.cmake](../../cmake/) — `option` + `CMakeDependentOption` + диагностика (Шаг 1) | ⭐ |
| **CMAKE-02** | Обновить [CMakeLists.txt](../../CMakeLists.txt) — FATAL_ERROR на ROCm + условные `add_subdirectory` (Шаг 2) | ⭐⭐ |
| **CMAKE-03** | Обновить [src/CMakeLists.txt](../../src/CMakeLists.txt) — функция `gpuworklib_link_module()`, убрать дублирующие include_directories (Шаг 3) | ⭐⭐ |
| **CMAKE-04** | Обновить [src/main.cpp](../../src/main.cpp) — таблица-реестр `ModuleEntry` + graceful skip (Шаг 4) | ⭐⭐ |
| **CMAKE-05** | Обновить [python/CMakeLists.txt](../../python/CMakeLists.txt) — условная линковка `gpuworklib_bind_module()` (Шаг 5) | ⭐⭐ |
| **CMAKE-06** | Удалить мёртвый [modules/CMakeLists.txt](../../modules/CMakeLists.txt) (Шаг 5.5) | ⭐ |
| **CMAKE-07** | Убрать `if(NOT ROCM_ENABLED) return()` из 11 модулей (ROCm теперь гарантирован) | ⭐ |
| **CMAKE-08** | Создать/обновить [CMakePresets.json](../../CMakePresets.json) — version 4, 8 пресетов (minimal / signal-lab / correlator-prod / radar-dsp / full / full-debug / full-python + base-linux hidden) (Шаг 6) | ⭐ |
| **CMAKE-09** | Smoke: `cmake --preset minimal && cmake --build --preset minimal -j$(nproc)` — собирается только DrvGPU | ⭐ |
| **CMAKE-10** | Smoke: `cmake --preset full && cmake --build --preset full -j$(nproc)` — ничего не сломалось | ⭐ |

**Критерий готовности Фазы 1**: `cmake --preset signal-lab` собирает только lch_farrow + fft_func + statistics + signal_generators, а все остальные модули показаны как `OFF` в диагностическом выводе.

---

### 🟢 Фаза 2 — LibGPU «конверт» (блоки A/B/C/D из TASK_LibGPU)

**Основа**: [TASK_LibGPU_Integration.md](../tasks/TASK_LibGPU_Integration.md) + [cmake_libgpu_integration_plan.md](cmake_libgpu_integration_plan.md) **после правок Фазы 0**.

#### Блок A — подготовка GPUWorkLib (можно параллельно с Фазой 1)

- **A1** — Добавить `install(TARGETS drvgpu EXPORT GPUWorkLibTargets …)` + `install(DIRECTORY include/ …)` в [DrvGPU/CMakeLists.txt](../../DrvGPU/CMakeLists.txt). Сейчас 0 install()-правил (проверено).
- **A2** — Написать `sync_to_libgpu.py` (~250 строк Python). Whitelist `SYNC_DIRS`/`SYNC_FILES` + `EXCLUDE_PATTERNS` из плана. Тест: `--dry-run`.

#### Блок B — создание LibGPU (зависит от Фазы 1: `cmake/modules-options.cmake` должен существовать)

- **B1** — `mkdir E:\C++\LibGPU && git init && git checkout -b main`, создать `.gitignore` и `VERSION=1.1.0`
- **B2** — Корневой `LibGPU/CMakeLists.txt` (~150 строк, `project(GPUWorkLib ...)`, `LIBGPU_IS_TOP_LEVEL`, `include(cmake/modules-options.cmake)` — **тот же файл** что в GPUWorkLib)
- **B3** — `LibGPU/CMakePresets.json` — **version 4** (не 6!), 3 пресета: `debug`/`release`/`local-debug` (Ninja)
- **B4** — `GPUWorkLibConfig.cmake.in` — с `find_dependency` для Threads, OpenCL, hip, hipfft, rocblas, rocsolver
- **B5** — `LibGPU/README.md` — инструкция подключения

#### Блок C — первая синхронизация + smoke-тест

- **C1** — `python sync_to_libgpu.py --src E:\C++\GPUWorkLib --dst E:\C++\LibGPU --dry-run` → проверить whitelist. Потом `--commit --tag v1.0.0`.
- **C2** — На Linux с ROCm: `cmake --preset local-debug && cmake --build build-local-debug && ./build-local-debug/GPUWorkLib` — C++ тесты прошли.
- **C3** — Создать минимальный тестовый проект с `add_subdirectory(../LibGPU …)` + `target_link_libraries(test_app PRIVATE GPUWorkLib::drvgpu GPUWorkLib::fft_func)` — линковка OK, Step Into в исходники GPUWorkLib работает.

#### Блок D — интеграция в закрытый проект (на сервере)

- **D1** — `git init --bare /srv/repos/LibGPU.git` на сервере, `git push -u origin main --tags`. Добавить как submodule в закрытый проект: `git submodule add ssh://server/srv/repos/LibGPU.git libs/GPUWorkLib`.
- **D2** — В `CMakeLists.txt` закрытого проекта: `set(LIBGPU_BUILD_TESTS OFF CACHE BOOL "" FORCE)` + `add_subdirectory(libs/GPUWorkLib)` + `target_link_libraries(closed_app PRIVATE GPUWorkLib::drvgpu GPUWorkLib::fft_func …)`.
- **D3** — Debug-сборка, Release-сборка, `configGPU.json` в рабочей директории, Step Into из IDE.

**Критерий готовности Фазы 2**: `git submodule update --remote libs/GPUWorkLib` + `cmake --build` в закрытом проекте обновляет код без ручного копирования.

---

### 🔵 Фаза 3 — Подход 5 из Primer.md (улучшение, опционально)

**Когда**: через 1–2 недели после успешной Фазы 2, когда LibGPU обкатан.

- **P5.1** — В LibGPU добавить `install(DIRECTORY kernels/ …)` для всех модулей которые используют `CMAKE_CURRENT_SOURCE_DIR/kernels`. См. секцию «Runtime Assets» в integration_plan.
- **P5.2** — Проверить что `cmake --install build-release --prefix /tmp/test-install` создаёт работающий пакет с `lib/cmake/GPUWorkLib/GPUWorkLibConfig.cmake`
- **P5.3** — Проверить `find_package(GPUWorkLib CONFIG PATHS /tmp/test-install)` в отдельном тестовом проекте — линкует и запускается
- **P5.4** — Перевести закрытый проект на Подход 5:
  ```cmake
  FetchContent_Declare(GPUWorkLib
      GIT_REPOSITORY ssh://server/srv/repos/LibGPU.git
      GIT_TAG v1.2.0
      FIND_PACKAGE_ARGS NAMES GPUWorkLib)
  FetchContent_MakeAvailable(GPUWorkLib)
  ```
- **P5.5** — Добавить в закрытый проект `CMakePresets.json` с `dev-local` пресетом (`FETCHCONTENT_SOURCE_DIR_GPUWORKLIB=${sourceDir}/../GPUWorkLib`)
- **P5.6** — Проверить что **оба** варианта работают: `cmake --preset prod` (из submodule) и `cmake --preset dev-local` (из локального каталога)

**Критерий готовности Фазы 3**: разработчик закрытого проекта переключается между production-версией и локальной правкой GPUWorkLib через смену пресета, **без** изменения CMakeLists.txt.

> **Важно**: после Фазы 3 можно **удалить git submodule** если команда привыкла к FetchContent. Но можно и оставить — `FIND_PACKAGE_ARGS` работает с обоими путями одинаково хорошо.

---

## 🚀 Что сделать прямо сейчас (Quick Wins)

Если хочется попробовать сегодня/завтра, **без ожидания**:

### QW-1 (15 минут) — Создать `cmake/modules-options.cmake`

Взять шаблон из [modular_architecture_plan Шаг 1](cmake_modular_architecture_plan.md) и вставить в [cmake/modules-options.cmake](../../cmake/). Файл самодостаточный, не ломает существующую сборку (ничего ещё не использует эти опции).

### QW-2 (5 минут) — Обновить CMakePresets.json до v4

```jsonc
{
  "version": 4,            // ← было 3
  "cmakeMinimumRequired": { "major": 3, "minor": 21 },
  ...
}
```

Версия 4 поддерживает `inherits`, `hidden`, `description`, `condition` — всё, что нужно для Linux-пресетов.

### QW-3 (10 минут) — Добавить Linux base-preset

Не трогая Windows-пресеты, добавить скрытый base и один minimal:

```jsonc
{
  "name": "base-linux", "hidden": true,
  "generator": "Ninja",
  "binaryDir": "${sourceDir}/build/${presetName}",
  "cacheVariables": {
    "CMAKE_BUILD_TYPE": "Release",
    "CMAKE_CXX_COMPILER": "/opt/rocm/bin/hipcc",
    "ENABLE_ROCM": "ON"
  }
},
{
  "name": "minimal",
  "inherits": "base-linux",
  "displayName": "Minimal — DrvGPU only"
}
```

Всё остальное (minimal/signal-lab/…) можно доложить в TASK-CMAKE-08.

### QW-4 (2 минуты) — Удалить мёртвый `modules/CMakeLists.txt`

Файл не вызывается корневым CMake (проверено — корневой `CMakeLists.txt` добавляет модули напрямую). Можно смело удалить. Это **TASK-CMAKE-06** одной командой.

### QW-5 (30 минут) — DRY в src/CMakeLists.txt

Заменить 12 блоков `if(TARGET …) target_link_libraries(…)` на одну функцию `gpuworklib_link_module()` из [modular_architecture Шаг 3](cmake_modular_architecture_plan.md). Также убрать 20 строк дублирующих `target_include_directories` — они придут через PUBLIC link.

---

## ⚠️ Риски и меры

| Риск | Вероятность | Последствие | Митигация |
|---|---|---|---|
| Имена опций не согласованы → две параллельные системы | **Высокая** (если Фаза 0 не сделана) | Закрытый проект не может отключить модуль | Фаза 0 — выровнять планы до начала |
| `cmake/modules-options.cmake` не попадёт в LibGPU | Средняя | LibGPU не сможет парсить `GPUWORKLIB_ENABLE_*` | В `SYNC_DIRS` sync_to_libgpu.py **уже** есть `cmake/` — проверить после A2 |
| ROCm 7.2 не установлен → не собираются Фаза 1 smoke-тесты | Низкая (на dev-машине Alex'а ROCm есть) | Блокирует CMAKE-09/10 | Smoke делать на AMD-машине в понедельник (есть Radeon9070) |
| `python/CMakeLists.txt` жёстко линкует — сломается при `--preset minimal` | **Высокая** | CMake error при minimal | CMAKE-05 должен быть сделан ДО CMAKE-09 |
| `sync_to_libgpu.py --clean` случайно удалит `LibGPU/CMakeLists.txt` | Средняя | Потеря ручных правок в LibGPU | В whitelist исключений скрипта **уже** прописано (`CMakeLists.txt, CMakePresets.json, VERSION, GPUWorkLibConfig.cmake.in`) — проверить |
| Подход 5 не взлетит без install/export DrvGPU | Высокая | Падение `find_package(GPUWorkLib)` | A1 (DrvGPU install) — **обязательный prereq** для Фазы 3 |
| Submodule `--recursive` забывается | Средняя | Пустые директории в libs/GPUWorkLib | Задокументировать в LibGPU/README.md. Или сразу перейти на Подход 5 |

---

## 📦 Итоговая сводка

| Параметр | Значение |
|---|---|
| **Планов** | 4 (modular + libgpu + research + review_issues) |
| **Справочников** | 2 (Primer.md — да; Git_ALL.md — не про CMake) |
| **Тасков открытых** | 1 (TASK_LibGPU_Integration, блоки A-D) |
| **Тасков в roadmap** | 10 (Фаза 1) + 4+5+3+3 = 15 (Фаза 2) + 6 (Фаза 3) = **31** |
| **Критических несогласованностей между планами** | 4 (опции, deps check, version presets, ROCm check) |
| **Гэпов в текущем коде** | 10 |
| **Quick wins (< 1 час)** | 5 |
| **Критических проблем в планах (самих по себе)** | 0 ✅ — все решения уже приняты в review_issues |

**Вердикт**: планы хорошие, готовы к реализации. Нужно только **согласовать их между собой** (Фаза 0, 10 минут) перед началом работы. После этого Фазу 1 можно делать независимо, Фазу 2 — после Фазы 1, Фазу 3 — когда будет время.

**Рекомендация для понедельника**:
1. ☀️ **Утро** — Фаза 0 (10 минут) + QW-1..QW-4 (40 минут) — обновить планы и начать с безболезненных изменений.
2. 🌤️ **До обеда** — CMAKE-01..CMAKE-05 — модули-options, корневой CMake, src, main, python.
3. 🌇 **После обеда** — CMAKE-06..CMAKE-10 — очистка + пресеты + smoke-тесты на Debian+Radeon9070.
4. 🌙 **Вечер** — если всё работает — начать Блок A (DrvGPU install + sync_to_libgpu.py черновик).

**Первопроходцы на работе** 🚀: Фаза 3 (Подход 5 + CMakePresets) — это редкий у нас в индустрии паттерн, но он **именно** решает «разработка локально vs production submodule». Это и станет нашим конкурентным преимуществом.

---

## 🔗 Источники

- **Context7**: не требовался (планы уже подробные, Primer.md покрывает теорию).
- **Sequential-thinking**: 1 итерация для консолидации 4 планов + карта гэпов.
- **Grep verification**:
  - `install` в DrvGPU/CMakeLists.txt → **0** совпадений (G7 подтверждён)
  - `if(TARGET` в src/CMakeLists.txt → 12 блоков (G3 подтверждён)
  - `target_link_libraries(gpuworklib PRIVATE` в python/CMakeLists.txt → жёсткий список (G4 подтверждён)
- **git rev-parse**: `E:/C++/GPUWorkLib` — основной репозиторий, не worktree ✅

---

*Review by: Кодо (AI Assistant) | 2026-04-09*
*Для обсуждения на работе в понедельник 2026-04-12*
