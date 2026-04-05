# Plan: CMake Modular Architecture — GPUWorkLib (Debian / ROCm)

> Дата: 2026-04-05 (обновлено: 2026-04-05 после code review)
> Цель: Реализовать выборочную сборку модулей через CMake Options + Presets
> Платформа: Debian Linux, ROCm 7.2+, AMD GPU (ветка main)
> Статус: 📋 PLAN — готов к созданию тасков

---

## Проблема (текущее состояние)

```cmake
# Текущий корневой CMakeLists.txt — добавляет ВСЁ всегда:
add_subdirectory(modules/fft_func)
add_subdirectory(modules/lch_farrow)
add_subdirectory(modules/signal_generators)
add_subdirectory(modules/filters)
add_subdirectory(modules/heterodyne)
add_subdirectory(modules/statistics)
add_subdirectory(modules/vector_algebra)
add_subdirectory(modules/fm_correlator)
add_subdirectory(modules/strategies)
add_subdirectory(modules/capon)
add_subdirectory(modules/range_angle)
```

Единственная защита — `if(NOT ROCM_ENABLED) return()` внутри каждого модуля.
**Нет возможности** собрать "только FFT + Statistics без генераторов" для продакшена.

> **Примечание**: `modules/CMakeLists.txt` сейчас пустой — все модули добавляются
> из корневого `CMakeLists.txt` напрямую. План работает с корневым файлом.

---

## Решение — 3 уровня

```
Уровень 1: cmake/modules-options.cmake   — флаги GPUWORKLIB_ENABLE_*
Уровень 2: CMakePresets.json             — готовые конфигурации (наборы флагов)
Уровень 3: CMakeLists.txt (корневой)     — условное подключение модулей
```

---

## Граф зависимостей модулей

```
DrvGPU (CORE — всегда обязателен)
│
├── lch_farrow                    (ROCm, нет зависимостей внутри проекта)
│     └── signal_generators       (требует: lch_farrow)
│           └── heterodyne        (требует: signal_generators + fft_func)
│
├── fft_func                      (ROCm, hipFFT)
│     ├── heterodyne              (требует: signal_generators + fft_func)
│     ├── strategies              (требует: fft_func + statistics)
│     ├── range_angle             (требует: fft_func)
│     └── fm_correlator           (требует: fft_func)
│
├── statistics                    (ROCm, rocprim + rocblas)
│     └── strategies              (требует: fft_func + statistics)
│
├── vector_algebra                (ROCm, rocblas + rocsolver)
│     └── capon                   (требует: vector_algebra)
│
├── filters                       (ROCm, независимый)
├── range_angle                   (требует: fft_func)
├── fm_correlator                 (требует: fft_func)
└── capon                         (требует: vector_algebra)
```

**Правило**: если модуль-зависимость выключен — зависимый модуль автоматически OFF.
`CMakeDependentOption` делает это автоматически.

---

## Шаг 1 — cmake/modules-options.cmake

Создать новый файл `cmake/modules-options.cmake`.

> **ВАЖНО**: Подключать ПОСЛЕ `include(cmake/dependencies.cmake)`.
> К этому моменту `ROCM_ENABLED` уже проверен и гарантированно TRUE
> (иначе сборка прервалась на FATAL_ERROR раньше).

```cmake
# cmake/modules-options.cmake
# Опции выборочной сборки модулей GPUWorkLib
# Подключается из корневого CMakeLists.txt ПОСЛЕ dependencies.cmake,
# ПЕРЕД add_subdirectory(modules/...)
#
# ROCM_ENABLED гарантированно TRUE на этом этапе — проверка в корневом CMakeLists.txt

include(CMakeDependentOption)

# ─────────────────────────────────────────────────────────────
# БАЗОВЫЕ МОДУЛИ (нет зависимостей внутри GPUWorkLib)
# ROCm гарантирован — проверка один раз в корневом CMakeLists.txt
# ─────────────────────────────────────────────────────────────

option(GPUWORKLIB_ENABLE_LCH_FARROW
    "Enable LCH Farrow fractional delay module"
    ON)

option(GPUWORKLIB_ENABLE_FFT
    "Enable FFT processor + SpectrumMaximaFinder (hipFFT)"
    ON)

option(GPUWORKLIB_ENABLE_FILTERS
    "Enable FIR/IIR/Kalman/Kaufman/MovingAverage filters"
    ON)

option(GPUWORKLIB_ENABLE_STATISTICS
    "Enable Statistics module (mean, median, std, variance via rocprim)"
    ON)

option(GPUWORKLIB_ENABLE_VECTOR_ALGEBRA
    "Enable Vector Algebra (Cholesky inversion, matrix ops via rocblas/rocsolver)"
    ON)

# ─────────────────────────────────────────────────────────────
# СОСТАВНЫЕ МОДУЛИ (зависят от базовых)
# CMakeDependentOption(name doc default_if_deps_met deps_expression default_otherwise)
# ─────────────────────────────────────────────────────────────

# signal_generators требует lch_farrow
CMakeDependentOption(GPUWORKLIB_ENABLE_SIGNAL_GENERATORS
    "Enable Signal Generators (CW, LFM, Noise, Script DSL)"
    ON
    "GPUWORKLIB_ENABLE_LCH_FARROW"
    OFF)

# heterodyne требует signal_generators И fft
CMakeDependentOption(GPUWORKLIB_ENABLE_HETERODYNE
    "Enable Heterodyne LFM Dechirp pipeline"
    ON
    "GPUWORKLIB_ENABLE_SIGNAL_GENERATORS;GPUWORKLIB_ENABLE_FFT"
    OFF)

# strategies требует fft И statistics
CMakeDependentOption(GPUWORKLIB_ENABLE_STRATEGIES
    "Enable Strategies antenna array pipeline (GEMM -> FFT)"
    ON
    "GPUWORKLIB_ENABLE_FFT;GPUWORKLIB_ENABLE_STATISTICS"
    OFF)

# capon требует vector_algebra
CMakeDependentOption(GPUWORKLIB_ENABLE_CAPON
    "Enable Capon MVDR beamformer"
    ON
    "GPUWORKLIB_ENABLE_VECTOR_ALGEBRA"
    OFF)

# range_angle требует fft
CMakeDependentOption(GPUWORKLIB_ENABLE_RANGE_ANGLE
    "Enable Range-Angle processor (LFM dechirp -> Range FFT -> Beam FFT)"
    ON
    "GPUWORKLIB_ENABLE_FFT"
    OFF)

# fm_correlator требует fft
CMakeDependentOption(GPUWORKLIB_ENABLE_FM_CORRELATOR
    "Enable FM Correlator (M-sequence frequency-domain correlation)"
    ON
    "GPUWORKLIB_ENABLE_FFT"
    OFF)

# ─────────────────────────────────────────────────────────────
# ДИАГНОСТИКА — вывод активных модулей при конфигурации
# ─────────────────────────────────────────────────────────────

message(STATUS "")
message(STATUS "══════════════════════════════════════════════")
message(STATUS "  GPUWorkLib — Active Modules")
message(STATUS "══════════════════════════════════════════════")
message(STATUS "  [CORE]  DrvGPU                          always")
message(STATUS "  [BASE]  lch_farrow         : ${GPUWORKLIB_ENABLE_LCH_FARROW}")
message(STATUS "  [BASE]  fft_func           : ${GPUWORKLIB_ENABLE_FFT}")
message(STATUS "  [BASE]  filters            : ${GPUWORKLIB_ENABLE_FILTERS}")
message(STATUS "  [BASE]  statistics         : ${GPUWORKLIB_ENABLE_STATISTICS}")
message(STATUS "  [BASE]  vector_algebra     : ${GPUWORKLIB_ENABLE_VECTOR_ALGEBRA}")
message(STATUS "  [COMP]  signal_generators  : ${GPUWORKLIB_ENABLE_SIGNAL_GENERATORS}")
message(STATUS "  [COMP]  heterodyne         : ${GPUWORKLIB_ENABLE_HETERODYNE}")
message(STATUS "  [COMP]  strategies         : ${GPUWORKLIB_ENABLE_STRATEGIES}")
message(STATUS "  [COMP]  capon              : ${GPUWORKLIB_ENABLE_CAPON}")
message(STATUS "  [COMP]  range_angle        : ${GPUWORKLIB_ENABLE_RANGE_ANGLE}")
message(STATUS "  [COMP]  fm_correlator      : ${GPUWORKLIB_ENABLE_FM_CORRELATOR}")
message(STATUS "══════════════════════════════════════════════")
message(STATUS "")
```

---

## Шаг 2 — Изменить корневой CMakeLists.txt

> **Ключевое отличие от v1 плана**: модули добавляются из корневого CMakeLists.txt,
> не из modules/CMakeLists.txt (который сейчас пустой).

**Было:**
```cmake
add_subdirectory(modules/fft_func)
add_subdirectory(modules/lch_farrow)
add_subdirectory(modules/signal_generators)
# ... и т.д. — безусловно
```

**Стало:**
```cmake
# CMakeLists.txt (корень)

include(cmake/platform-detection.cmake)
include(cmake/gpu-config.cmake)
include(cmake/dependencies.cmake)

# ═══ ROCm — единственная проверка на весь проект (ветка main) ═══
if(NOT ROCM_ENABLED)
    message(FATAL_ERROR
        "ROCm/HIP not found! Ветка main требует ROCm 7.2+.\n"
        "  Установи: https://rocm.docs.amd.com/\n"
        "  Ubuntu: sudo apt install rocm-dev hip-dev\n"
        "  Или переключись на ветку nvidia для OpenCL/Windows.")
endif()

include(cmake/modules-options.cmake)    # <-- ПОСЛЕ dependencies + ROCm check

add_subdirectory(DrvGPU)

# Базовые модули
if(GPUWORKLIB_ENABLE_LCH_FARROW)
    add_subdirectory(modules/lch_farrow)
endif()
if(GPUWORKLIB_ENABLE_FFT)
    add_subdirectory(modules/fft_func)
endif()
if(GPUWORKLIB_ENABLE_FILTERS)
    add_subdirectory(modules/filters)
endif()
if(GPUWORKLIB_ENABLE_STATISTICS)
    add_subdirectory(modules/statistics)
endif()
if(GPUWORKLIB_ENABLE_VECTOR_ALGEBRA)
    add_subdirectory(modules/vector_algebra)
endif()

# Составные модули (CMakeDependentOption гарантирует OFF если зависимость OFF)
if(GPUWORKLIB_ENABLE_SIGNAL_GENERATORS)
    add_subdirectory(modules/signal_generators)
endif()
if(GPUWORKLIB_ENABLE_HETERODYNE)
    add_subdirectory(modules/heterodyne)
endif()
if(GPUWORKLIB_ENABLE_STRATEGIES)
    add_subdirectory(modules/strategies)
endif()
if(GPUWORKLIB_ENABLE_CAPON)
    add_subdirectory(modules/capon)
endif()
if(GPUWORKLIB_ENABLE_RANGE_ANGLE)
    add_subdirectory(modules/range_angle)
endif()
if(GPUWORKLIB_ENABLE_FM_CORRELATOR)
    add_subdirectory(modules/fm_correlator)
endif()

# Main executable
add_subdirectory(src)

# Python bindings
option(BUILD_PYTHON "Build Python bindings (pybind11)" OFF)
if(BUILD_PYTHON)
    add_subdirectory(python)
endif()
```

> **Примечание**: НЕ проверяем `ROCM_ENABLED` в каждом модуле — одна проверка
> `FATAL_ERROR` в корневом CMakeLists.txt гарантирует что ROCm есть.
> Внутренние `if(NOT ROCM_ENABLED) return()` в модулях можно убрать при реализации.

---

## Шаг 3 — Compile definitions в src/CMakeLists.txt

> **Важно**: НЕ использовать глобальную `add_compile_definitions()` —
> она загрязняет ВСЕ таргеты (DrvGPU, plog, pybind11).
> Вместо этого — `target_compile_definitions` только для exe `GPUWorkLib`.

Вместо 33 отдельных `if(TARGET)` блоков — **одна функция**,
которая объединяет link + define + include:

```cmake
# src/CMakeLists.txt — DRY-функция: link + compile_definition + include

function(gpuworklib_link_module MODULE_ALIAS DEFINE_NAME)
    if(TARGET ${MODULE_ALIAS})
        target_link_libraries(GPUWorkLib PRIVATE ${MODULE_ALIAS})
        target_compile_definitions(GPUWorkLib PRIVATE ${DEFINE_NAME})
        # Include dirs уже приходят через PUBLIC target_include_directories модуля
        message(STATUS "  Linked: ${MODULE_ALIAS}")
    endif()
endfunction()
```

> **Почему не нужен отдельный target_include_directories?**
> Каждый модуль уже объявляет `target_include_directories(... PUBLIC ...)`.
> При `target_link_libraries(GPUWorkLib PRIVATE module)` CMake автоматически
> подтягивает PUBLIC include dirs зависимости. Текущие 20+ строк ручных
> include-путей в src/CMakeLists.txt — **дублирование**, их можно удалить.

Заменить существующие блоки (строки 64-191 текущего src/CMakeLists.txt):

```cmake
# ════════════════════════════════════════════════════════════
# ЛИНКОВКА МОДУЛЕЙ (link + compile definition — одним вызовом)
# ════════════════════════════════════════════════════════════

# DrvGPU — всегда
target_link_libraries(GPUWorkLib PRIVATE DrvGPU::drvgpu)
if(TARGET DrvGPU::VectorOps)
    target_link_libraries(GPUWorkLib PRIVATE DrvGPU::VectorOps)
endif()
if(TARGET DrvGPU::Search)
    target_link_libraries(GPUWorkLib PRIVATE DrvGPU::Search)
endif()

# Опциональные модули
gpuworklib_link_module(GPUWorkLib::fft_func           GPUWORKLIB_HAS_FFT)
gpuworklib_link_module(GPUWorkLib::signal_generators   GPUWORKLIB_HAS_SIGNAL_GENERATORS)
gpuworklib_link_module(GPUWorkLib::lch_farrow          GPUWORKLIB_HAS_LCH_FARROW)
gpuworklib_link_module(GPUWorkLib::filters             GPUWORKLIB_HAS_FILTERS)
gpuworklib_link_module(GPUWorkLib::heterodyne          GPUWORKLIB_HAS_HETERODYNE)
gpuworklib_link_module(GPUWorkLib::statistics          GPUWORKLIB_HAS_STATISTICS)
gpuworklib_link_module(GPUWorkLib::vector_algebra      GPUWORKLIB_HAS_VECTOR_ALGEBRA)
gpuworklib_link_module(GPUWorkLib::fm_correlator       GPUWORKLIB_HAS_FM_CORRELATOR)
gpuworklib_link_module(GPUWorkLib::strategies          GPUWORKLIB_HAS_STRATEGIES)
gpuworklib_link_module(GPUWorkLib::capon               GPUWORKLIB_HAS_CAPON)
gpuworklib_link_module(GPUWorkLib::range_angle         GPUWORKLIB_HAS_RANGE_ANGLE)

# ════════════════════════════════════════════════════════════
# INCLUDE DIRECTORIES — только базовые (модульные приходят через PUBLIC link)
# ════════════════════════════════════════════════════════════

target_include_directories(GPUWorkLib PRIVATE
    ${CMAKE_SOURCE_DIR}
    ${CMAKE_SOURCE_DIR}/include
    ${CMAKE_SOURCE_DIR}/include/interface
    ${CMAKE_SOURCE_DIR}/src
    ${CMAKE_SOURCE_DIR}/DrvGPU
)
```

> **Итого**: 33 блока `if(TARGET)` → 11 вызовов `gpuworklib_link_module` +
> удалён дублирующий `target_include_directories` на 20 строк.
> При добавлении нового модуля — **одна строка** вместо трёх блоков.

---

## Шаг 4 — Защита в src/main.cpp

Вместо трёх отдельных #ifdef-секций (includes, get_default_order, run_module)
используем **таблицу-реестр** — каждый модуль упоминается один раз:

```cpp
// src/main.cpp

#include <iostream>
#include <string>
#include <vector>
#include <fstream>
#include <algorithm>
#include <cctype>
#include <functional>

// ═══════════════════════════════════════════════════════════
// Условные includes — #ifdef только здесь
// ═══════════════════════════════════════════════════════════
#include "DrvGPU/tests/all_test.hpp"

#ifdef GPUWORKLIB_HAS_FFT
  #include "modules/fft_func/tests/all_test.hpp"
#endif
#ifdef GPUWORKLIB_HAS_SIGNAL_GENERATORS
  #include "modules/signal_generators/tests/all_test.hpp"
#endif
#ifdef GPUWORKLIB_HAS_LCH_FARROW
  #include "modules/lch_farrow/tests/all_test.hpp"
#endif
#ifdef GPUWORKLIB_HAS_FILTERS
  #include "modules/filters/tests/all_test.hpp"
#endif
#ifdef GPUWORKLIB_HAS_HETERODYNE
  #include "modules/heterodyne/tests/all_test.hpp"
#endif
#ifdef GPUWORKLIB_HAS_STATISTICS
  #include "modules/statistics/tests/all_test.hpp"
#endif
#ifdef GPUWORKLIB_HAS_VECTOR_ALGEBRA
  #include "modules/vector_algebra/tests/all_test.hpp"
#endif
#ifdef GPUWORKLIB_HAS_FM_CORRELATOR
  #include "modules/fm_correlator/tests/all_test.hpp"
#endif
#ifdef GPUWORKLIB_HAS_STRATEGIES
  #include "modules/strategies/tests/all_test.hpp"
#endif
#ifdef GPUWORKLIB_HAS_CAPON
  #include "modules/capon/tests/all_test.hpp"
#endif
#ifdef GPUWORKLIB_HAS_RANGE_ANGLE
  #include "modules/range_angle/tests/all_test.hpp"
#endif

namespace {

// ═══════════════════════════════════════════════════════════
// Реестр модулей — одна таблица вместо трёх #ifdef-секций
// Модуль упоминается ровно один раз. Порядок = default order.
// ═══════════════════════════════════════════════════════════
struct ModuleEntry {
    const char* name;
    std::function<void()> run;
};

const std::vector<ModuleEntry>& get_registry() {
    static const std::vector<ModuleEntry> reg = {
        {"drvgpu",            drvgpu_all_test::run},
#ifdef GPUWORKLIB_HAS_FFT
        {"fft_func",          fft_func_all_test::run},
#endif
#ifdef GPUWORKLIB_HAS_STATISTICS
        {"statistics",        statistics_all_test::run},
#endif
#ifdef GPUWORKLIB_HAS_VECTOR_ALGEBRA
        {"vector_algebra",    vector_algebra_all_test::run},
#endif
#ifdef GPUWORKLIB_HAS_FILTERS
        {"filters",           filters_all_test::run},
#endif
#ifdef GPUWORKLIB_HAS_SIGNAL_GENERATORS
        {"signal_generators", signal_generators_all_test::run},
#endif
#ifdef GPUWORKLIB_HAS_LCH_FARROW
        {"lch_farrow",        lch_farrow_all_test::run},
#endif
#ifdef GPUWORKLIB_HAS_HETERODYNE
        {"heterodyne",        heterodyne_all_test::run},
#endif
#ifdef GPUWORKLIB_HAS_FM_CORRELATOR
        {"fm_correlator",     fm_correlator_all_test::run},
#endif
#ifdef GPUWORKLIB_HAS_STRATEGIES
        {"strategies",        strategies_all_test::run},
#endif
#ifdef GPUWORKLIB_HAS_CAPON
        {"capon",             capon_all_test::run},
#endif
#ifdef GPUWORKLIB_HAS_RANGE_ANGLE
        {"range_angle",       range_angle_all_test::run},
#endif
    };
    return reg;
}

// Lookup по реестру (nullptr если модуль не скомпилирован)
const ModuleEntry* find_module(const std::string& name) {
    for (const auto& m : get_registry())
        if (name == m.name) return &m;
    return nullptr;
}

// ... to_lower, read_modules_from_file — без изменений ...

}  // namespace

int main(int argc, char* argv[]) {
    // ... парсинг аргументов без изменений ...

    // Если нет аргументов/all — берём порядок из config/tests_order.txt
    // Если файла нет — default order = порядок из реестра
    if (to_run.empty()) {
        for (const auto& m : get_registry())
            to_run.push_back(m.name);
    }

    for (const auto& name : to_run) {
        auto* mod = find_module(to_lower(name));
        if (mod) {
            std::cout << "\n>>> " << name << " <<<\n";
            mod->run();
        } else {
            // Модуль в tests_order.txt, но не скомпилирован → skip с warning
            std::cerr << "  [SKIP] " << name
                      << " — not compiled in this build\n";
        }
    }

    std::cout << "\nВсе тесты завершены!" << std::endl;
    return 0;
}
```

> **Что изменилось по сравнению с v2:**
> - `get_default_order()` и `run_module()` **удалены** — заменены `get_registry()` + `find_module()`
> - #ifdef остался только в двух местах: includes (неизбежно) и таблица реестра
> - **Graceful skip**: если `config/tests_order.txt` содержит модуль который
>   не был скомпилирован (например `capon` в preset `signal-lab`) — не crash,
>   а `[SKIP]` с пояснением. Один `tests_order.txt` для всех пресетов.
> - Порядок по умолчанию (без tests_order.txt) = порядок в реестре

---

## Шаг 5 — python/CMakeLists.txt (условные модули)

Текущий `python/CMakeLists.txt` жёстко линкует `signal_generators`, `fft_func`,
`lch_farrow`, `filters`, `heterodyne` — при отключённом модуле будет ошибка линковки.

Применить ту же DRY-функцию + условные define для биндингов:

```cmake
# python/CMakeLists.txt — условная линковка (аналогично src/)

function(gpuworklib_bind_module MODULE_ALIAS DEFINE_NAME)
    if(TARGET ${MODULE_ALIAS})
        target_link_libraries(gpuworklib PRIVATE ${MODULE_ALIAS})
        target_compile_definitions(gpuworklib PRIVATE ${DEFINE_NAME})
    endif()
endfunction()

# DrvGPU — всегда
target_link_libraries(gpuworklib PRIVATE drvgpu)

# Опциональные модули
gpuworklib_bind_module(GPUWorkLib::fft_func           GPUWORKLIB_HAS_FFT)
gpuworklib_bind_module(GPUWorkLib::signal_generators   GPUWORKLIB_HAS_SIGNAL_GENERATORS)
gpuworklib_bind_module(GPUWorkLib::lch_farrow          GPUWORKLIB_HAS_LCH_FARROW)
gpuworklib_bind_module(GPUWorkLib::filters             GPUWORKLIB_HAS_FILTERS)
gpuworklib_bind_module(GPUWorkLib::heterodyne          GPUWORKLIB_HAS_HETERODYNE)
gpuworklib_bind_module(GPUWorkLib::statistics          GPUWORKLIB_HAS_STATISTICS)
gpuworklib_bind_module(GPUWorkLib::vector_algebra      GPUWORKLIB_HAS_VECTOR_ALGEBRA)
gpuworklib_bind_module(GPUWorkLib::fm_correlator       GPUWORKLIB_HAS_FM_CORRELATOR)
gpuworklib_bind_module(GPUWorkLib::strategies          GPUWORKLIB_HAS_STRATEGIES)
gpuworklib_bind_module(GPUWorkLib::capon               GPUWORKLIB_HAS_CAPON)
gpuworklib_bind_module(GPUWorkLib::range_angle         GPUWORKLIB_HAS_RANGE_ANGLE)
```

В `gpu_worklib_bindings.cpp` — обернуть секции биндинга в `#ifdef GPUWORKLIB_HAS_*`.

> **Аналогичная проблема**: `target_include_directories` в python/CMakeLists.txt
> тоже жёстко перечисляет все modules — заменить на базовые (модульные придут через PUBLIC link).

---

## Шаг 5.5 — Удалить пустой modules/CMakeLists.txt

`modules/CMakeLists.txt` сейчас пустой — все модули добавляются из корня.
Корневой `CMakeLists.txt` не вызывает `add_subdirectory(modules)`.
Файл — мёртвый код, удалить чтобы не путать.

---

## Шаг 6 — CMakePresets.json (корень проекта)

> **Версия схемы**: `4` (совместимо с CMake 3.21+, покрывает все нужные фичи).
> v1 плана использовала version 6 (требует CMake 3.25) — несовместимо с
> текущим `cmake_minimum_required(VERSION 3.20)`.

```json
{
  "version": 4,
  "cmakeMinimumRequired": { "major": 3, "minor": 21 },

  "configurePresets": [

    {
      "name": "base-linux",
      "hidden": true,
      "description": "Базовые настройки Linux/Debian ROCm",
      "generator": "Ninja",
      "binaryDir": "${sourceDir}/build/${presetName}",
      "cacheVariables": {
        "CMAKE_BUILD_TYPE": "Release",
        "CMAKE_CXX_COMPILER": "/opt/rocm/bin/hipcc",
        "ENABLE_ROCM": "ON",
        "BUILD_PYTHON": "OFF"
      }
    },

    {
      "name": "minimal",
      "inherits": "base-linux",
      "displayName": "Minimal — DrvGPU only",
      "description": "Только DrvGPU core. Для тестирования драйвера и бэкенда.",
      "cacheVariables": {
        "GPUWORKLIB_ENABLE_LCH_FARROW":        "OFF",
        "GPUWORKLIB_ENABLE_FFT":               "OFF",
        "GPUWORKLIB_ENABLE_FILTERS":           "OFF",
        "GPUWORKLIB_ENABLE_STATISTICS":        "OFF",
        "GPUWORKLIB_ENABLE_VECTOR_ALGEBRA":    "OFF"
      }
    },

    {
      "name": "signal-lab",
      "inherits": "base-linux",
      "displayName": "Signal Lab — генераторы + FFT + статистика",
      "description": "Тестовый стенд. Генераторы + FFT + Statistics.",
      "cacheVariables": {
        "GPUWORKLIB_ENABLE_LCH_FARROW":        "ON",
        "GPUWORKLIB_ENABLE_FFT":               "ON",
        "GPUWORKLIB_ENABLE_FILTERS":           "OFF",
        "GPUWORKLIB_ENABLE_STATISTICS":        "ON",
        "GPUWORKLIB_ENABLE_VECTOR_ALGEBRA":    "OFF"
      }
    },

    {
      "name": "correlator-prod",
      "inherits": "base-linux",
      "displayName": "Correlator Production — без генераторов",
      "description": "Продакшн FM-коррелятор. Генераторы сигналов не нужны.",
      "cacheVariables": {
        "GPUWORKLIB_ENABLE_LCH_FARROW":        "OFF",
        "GPUWORKLIB_ENABLE_FFT":               "ON",
        "GPUWORKLIB_ENABLE_FILTERS":           "ON",
        "GPUWORKLIB_ENABLE_STATISTICS":        "ON",
        "GPUWORKLIB_ENABLE_VECTOR_ALGEBRA":    "OFF"
      }
    },

    {
      "name": "radar-dsp",
      "inherits": "base-linux",
      "displayName": "Radar DSP — гетеродин + range/angle + capon",
      "description": "Полный радарный стек. Статистика и capon для обнаружения целей.",
      "cacheVariables": {
        "GPUWORKLIB_ENABLE_LCH_FARROW":        "ON",
        "GPUWORKLIB_ENABLE_FFT":               "ON",
        "GPUWORKLIB_ENABLE_FILTERS":           "ON",
        "GPUWORKLIB_ENABLE_STATISTICS":        "ON",
        "GPUWORKLIB_ENABLE_VECTOR_ALGEBRA":    "ON"
      }
    },

    {
      "name": "full",
      "inherits": "base-linux",
      "displayName": "Full — все модули (CI / разработка)",
      "description": "Все модули. Используется в CI и при полной разработке."
    },

    {
      "name": "full-debug",
      "inherits": "full",
      "displayName": "Full Debug",
      "description": "Все модули, Debug build.",
      "cacheVariables": {
        "CMAKE_BUILD_TYPE": "Debug"
      }
    },

    {
      "name": "full-python",
      "inherits": "full",
      "displayName": "Full + Python bindings",
      "description": "Все модули + pybind11. Для Python тестов.",
      "cacheVariables": {
        "BUILD_PYTHON": "ON"
      }
    }
  ],

  "buildPresets": [
    { "name": "minimal",         "configurePreset": "minimal" },
    { "name": "signal-lab",      "configurePreset": "signal-lab" },
    { "name": "correlator-prod", "configurePreset": "correlator-prod" },
    { "name": "radar-dsp",       "configurePreset": "radar-dsp" },
    { "name": "full",            "configurePreset": "full" },
    { "name": "full-debug",      "configurePreset": "full-debug" },
    { "name": "full-python",     "configurePreset": "full-python" }
  ]
}
```

> **Примечание по minimal preset**: указаны только базовые модули OFF.
> Составные (signal_generators, heterodyne, strategies, capon, range_angle,
> fm_correlator) выключатся автоматически через `CMakeDependentOption`.

> **Примечание по full preset**: не указывает ENABLE-флаги — все опции
> по умолчанию ON (если ROCm найден), что и нужно для полной сборки.

---

## Ожидаемый вывод при cmake --preset signal-lab

```
-- ══════════════════════════════════════════════
--   GPUWorkLib — Active Modules
-- ══════════════════════════════════════════════
--   [CORE]  DrvGPU                          always
--   [BASE]  lch_farrow         : ON
--   [BASE]  fft_func           : ON
--   [BASE]  filters            : OFF
--   [BASE]  statistics         : ON
--   [BASE]  vector_algebra     : OFF
--   [COMP]  signal_generators  : ON
--   [COMP]  heterodyne         : OFF
--   [COMP]  strategies         : OFF
--   [COMP]  capon              : OFF
--   [COMP]  range_angle        : OFF
--   [COMP]  fm_correlator      : OFF
-- ══════════════════════════════════════════════
```

---

## Итоговая последовательность файлов для создания/изменения

| # | Файл | Действие | Сложность |
|---|------|----------|-----------|
| 1 | `cmake/modules-options.cmake` | Создать новый | * |
| 2 | `CMakeLists.txt` (корневой) | FATAL_ERROR check + обернуть `add_subdirectory` | ** |
| 3 | `src/CMakeLists.txt` | DRY-функция `gpuworklib_link_module` + убрать дублирующие include dirs | ** |
| 4 | `src/main.cpp` | Таблица-реестр + graceful skip | ** |
| 5 | `python/CMakeLists.txt` | Условные модули + DRY-функция | ** |
| 5.5 | `modules/CMakeLists.txt` | Удалить (пустой) | * |
| 6 | `CMakePresets.json` | Создать новый в корне (version 4) | * |
| 7 | Модули: убрать `if(NOT ROCM_ENABLED) return()` | Очистка (ROCm гарантирован) | * |
| 8 | Тест: `cmake --preset minimal` | Проверить на Debian | * |
| 9 | Тест: `cmake --preset full` | Проверить что ничего не сломалось | * |

---

## Команды сборки

```bash
# Быстрая сборка — только то что нужно:
cmake --preset signal-lab
cmake --build --preset signal-lab -j$(nproc)

# Полная сборка (как раньше):
cmake --preset full
cmake --build --preset full -j$(nproc)

# Ручная конфигурация (имена переменных!):
cmake -B build \
  -DGPUWORKLIB_ENABLE_SIGNAL_GENERATORS=OFF \
  -DGPUWORKLIB_ENABLE_CAPON=OFF
```

---

## Таски для реализации

- [ ] **TASK-CMAKE-01**: Создать `cmake/modules-options.cmake` (option + CMakeDependentOption + диагностика)
- [ ] **TASK-CMAKE-02**: Обновить корневой `CMakeLists.txt` — FATAL_ERROR ROCm check + условные `add_subdirectory`
- [ ] **TASK-CMAKE-03**: Обновить `src/CMakeLists.txt` — DRY-функция `gpuworklib_link_module`, убрать дублирующие include dirs
- [ ] **TASK-CMAKE-04**: Обновить `src/main.cpp` — таблица-реестр `ModuleEntry` + graceful skip
- [ ] **TASK-CMAKE-05**: Обновить `python/CMakeLists.txt` — условная линковка + DRY-функция
- [ ] **TASK-CMAKE-06**: Удалить пустой `modules/CMakeLists.txt`
- [ ] **TASK-CMAKE-07**: Убрать `if(NOT ROCM_ENABLED) return()` из всех модулей (ROCm гарантирован)
- [ ] **TASK-CMAKE-08**: Создать `CMakePresets.json` (version 4, 8 пресетов)
- [ ] **TASK-CMAKE-09**: Тест: `cmake --preset minimal` на Debian (только DrvGPU)
- [ ] **TASK-CMAKE-10**: Тест: `cmake --preset full` на Debian (все модули, как раньше)

---

## Changelog

**v3 (2026-04-05)** — оптимизация DRY + graceful skip:
- src/CMakeLists.txt: DRY-функция `gpuworklib_link_module()` — link + define одним вызовом (33 блока → 11 строк)
- src/CMakeLists.txt: удалены дублирующие `target_include_directories` (PUBLIC приходят через link)
- main.cpp: таблица-реестр `ModuleEntry` вместо трёх #ifdef-секций (get_default_order + run_module → get_registry + find_module)
- main.cpp: graceful skip — модуль в tests_order.txt но не скомпилирован → `[SKIP]` вместо crash
- python/CMakeLists.txt: добавлен Шаг 5 — условная линковка + DRY (раньше жёстко линковал все модули)
- Шаг 5.5: удалить мёртвый `modules/CMakeLists.txt`
- TASK-CMAKE-07: убрать `if(NOT ROCM_ENABLED) return()` из модулей (ROCm гарантирован FATAL_ERROR)
- Тасков: 7 → 10

**v2 (2026-04-05)** — исправления по результатам code review:
- `add_compile_definitions` (глобальная) → `target_compile_definitions(GPUWorkLib PRIVATE ...)`
- ROCm проверяется ОДИН РАЗ (`FATAL_ERROR` в корневом CMakeLists.txt)
- CMakePresets version 6 → **version 4** (CMake 3.21+)
- Исправлена опечатка: `ENABLE_GENERATORS` → `ENABLE_SIGNAL_GENERATORS`
- Шаг 2 актуализирован: корневой CMakeLists.txt (не modules/)
- full preset не перечисляет ENABLE-флаги (все ON по умолчанию)
- Убрана ветка nvidia — план только для main/ROCm
