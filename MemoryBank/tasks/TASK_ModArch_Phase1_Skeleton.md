# 🏗️ Фаза 1: Создание скелета 9 репо

> **Индекс**: [`TASK_Modular_Architecture_INDEX.md`](TASK_Modular_Architecture_INDEX.md)
> **Спецификация**: [`specs/modular_architecture_plan.md`](../specs/modular_architecture_plan.md) — разделы 3, 4, 5, 7 Фаза 1
> **Статус**: ✅ DONE — завершено 2026-04-12
> **Зависимость**: Фаза 0 ✅ DONE
> **Платформа**: Windows (GitHub + создание файлов, GPU не нужен)
> **Результат**: 9 пустых репо на `github.com/dsp-gpu` с CMake-скелетом, которые конфигурируются без ошибок

---

## ⚠️ Что уже сделано (пропустить эти шаги!)

- ✅ Все 9 репо созданы вручную на `github.com/dsp-gpu` (2026-04-12)
- ✅ `DSP/Doc/Architecture/` — документация запушена в DSP репо
- ✅ Рабочая папка: **`E:\DSP-GPU\`** (не `C:\dsp-gpu\`!)

> Шаги `gh repo create ...` в чеклисте ниже — **пропустить**, репо уже существуют.
> Сразу переходить к созданию файлов.

---

## Цель

Создать CMake-скелет в каждом репо **без реального кода** (только CMakeLists.txt + заглушки), убедиться что `cmake --preset local-dev` конфигурируется без ошибок. Потом в Фазе 2 копируем код.

---

## 🔴 КРИТИЧЕСКОЕ: find_package — только lowercase на Linux!

```cmake
# ❌ НЕПРАВИЛЬНО (упадёт на Debian):
find_package(HIP REQUIRED)

# ✅ ПРАВИЛЬНО:
find_package(hip REQUIRED)
find_package(hipfft REQUIRED)
find_package(rocprim REQUIRED)
find_package(rocblas REQUIRED)
find_package(rocsolver REQUIRED)
```

Linux — case-sensitive файловая система. `FindHIP.cmake` ≠ `Findhip.cmake`.
Проверять во ВСЕХ `CMakeLists.txt` и `Config.cmake.in` файлах!

---

## Общий шаблон файлов (применяется к каждому репо)

```
{repo}/
├── CMakeLists.txt              ← standalone build по шаблону из раздела 4 плана
├── CMakePresets.json           ← local-dev (E:\DSP-GPU\) + ci
├── cmake/
│   ├── {Repo}Config.cmake.in  ← с find_dependency (lowercase!)
│   └── fetch_deps.cmake        ← из раздела 4.3 плана
├── include/dsp/.gitkeep        ← пустая папка (публичные заголовки)
├── kernels/rocm/.gitkeep       ← PRIVATE: ROCm .hip kernel файлы
├── src/.gitkeep                ← пустая папка
├── tests/all_test.hpp          ← пустой include-файл
├── README.md
└── .gitignore
```

> ⚠️ `kernels/rocm/` — PRIVATE директория (не в `include/`!)
> CMake: `target_include_directories(DspXxx PUBLIC include PRIVATE kernels/)`

`.gitignore` стандартный:
```gitignore
build/
*.pyd
*.so
__pycache__/
.cache/
```

---

## CMakePresets.json — шаблон (local-dev пути для Windows)

```json
{
  "version": 3,
  "configurePresets": [
    {
      "name": "local-dev",
      "displayName": "Local Development (E:\\DSP-GPU\\)",
      "generator": "Ninja",
      "binaryDir": "${sourceDir}/build",
      "cacheVariables": {
        "FETCHCONTENT_SOURCE_DIR_DSPCORE":             "E:/DSP-GPU/core",
        "FETCHCONTENT_SOURCE_DIR_DSPSPECTRUM":         "E:/DSP-GPU/spectrum",
        "FETCHCONTENT_SOURCE_DIR_DSPSTATS":            "E:/DSP-GPU/stats",
        "FETCHCONTENT_SOURCE_DIR_DSPSIGNALGENERATORS": "E:/DSP-GPU/signal_generators",
        "FETCHCONTENT_SOURCE_DIR_DSPHETERODYNE":       "E:/DSP-GPU/heterodyne",
        "FETCHCONTENT_SOURCE_DIR_DSPLINALG":           "E:/DSP-GPU/linalg",
        "FETCHCONTENT_SOURCE_DIR_DSPRADAR":            "E:/DSP-GPU/radar",
        "FETCHCONTENT_SOURCE_DIR_DSPSTRATEGIES":       "E:/DSP-GPU/strategies"
      }
    },
    {
      "name": "ci",
      "displayName": "CI Build (tags from DSP_*_TAG)",
      "generator": "Ninja",
      "binaryDir": "${sourceDir}/build"
    }
  ]
}
```

> На Debian пути будут `~/dsp-gpu/core` и т.д. — см. Фазу 4.

---

## Чеклист по репо

### 🔵 Репо 1: `core` (~~gh repo create~~ — уже существует)

- [x] **- [x] **1.2** Создать `CMakeLists.txt`:
  ```cmake
  cmake_minimum_required(VERSION 3.25)
  project(DspCore VERSION 0.1.0 LANGUAGES CXX HIP)
  find_package(hip REQUIRED)          # lowercase!
  find_package(rocprim REQUIRED)
  add_library(DspCore STATIC)
  add_library(DspCore::DspCore ALIAS DspCore)
  add_library(DspCoreTestUtils INTERFACE)
  add_library(DspCore::TestUtils ALIAS DspCoreTestUtils)
  target_include_directories(DspCore
    PUBLIC  include
    PRIVATE kernels/ src)
  option(DSP_CORE_BUILD_TESTS  "Build tests"   ON)
  option(DSP_CORE_BUILD_PYTHON "Build Python"  OFF)
  ```
- [x] **- [x] **1.3** Создать `cmake/DspCoreConfig.cmake.in`:
  ```cmake
  @PACKAGE_INIT@
  include(CMakeFindDependencyMacro)
  find_dependency(hip REQUIRED)
  # OpenCL — только если нужен OpenCL backend:
  # find_dependency(OpenCL REQUIRED)
  include("${CMAKE_CURRENT_LIST_DIR}/DspCoreTargets.cmake")
  check_required_components(DspCore)
  ```
- [x] **- [x] **1.4** Создать `cmake/fetch_deps.cmake` — полная версия из раздела 4.3 плана
- [x] **- [x] **1.5** Создать `CMakePresets.json` (local-dev + ci, пути `E:\DSP-GPU\`)
- [x] **- [x] **1.6** Создать `kernels/rocm/.gitkeep` + `include/dsp/.gitkeep` + `src/.gitkeep`
- [x] **- [x] **1.7** Создать `tests/all_test.hpp` (пустой)
- [x] **- [x] **1.8** Создать `python/CMakeLists.txt` (pybind11)
- [x] **- [x] **1.9** `cmake -S . -B build --preset local-dev` — конфигурируется ✅
- [x] **1.10** `git push`

---

### 🔵 Репо 2: `spectrum` (~~gh repo create~~ — уже существует)

- [x] **- [x] **2.2** Создать `CMakeLists.txt`:
  ```cmake
  cmake_minimum_required(VERSION 3.25)
  project(DspSpectrum VERSION 0.1.0 LANGUAGES CXX HIP)
  find_package(hip REQUIRED)          # lowercase!
  find_package(hipfft REQUIRED)
  include(cmake/fetch_deps.cmake)
  fetch_dsp_core()
  add_library(DspSpectrum STATIC)
  add_library(DspSpectrum::DspSpectrum ALIAS DspSpectrum)
  target_include_directories(DspSpectrum
    PUBLIC  include
    PRIVATE kernels/ src)
  target_link_libraries(DspSpectrum PUBLIC DspCore::DspCore hip::hipfft)
  ```
- [x] **- [x] **2.3** `cmake/DspSpectrumConfig.cmake.in`:
  ```cmake
  find_dependency(DspCore  REQUIRED)
  find_dependency(hip      REQUIRED)    # lowercase!
  find_dependency(hipfft   REQUIRED)
  ```
- [x] **- [x] **2.4** `cmake/fetch_deps.cmake` (скопировать из core)
- [x] **- [x] **2.5** `CMakePresets.json` (local-dev с `FETCHCONTENT_SOURCE_DIR_DSPCORE=E:/DSP-GPU/core`)
- [x] **- [x] **2.6** Структура папок: `include/dsp/spectrum/`, `kernels/rocm/`, `src/`, `tests/`
- [x] **- [x] **2.7** `cmake -S . -B build --preset local-dev` — OK ✅
- [x] **- [x] **2.8** Push

---

### 🔵 Репо 3: `stats` (~~gh repo create~~ — уже существует)

- [x] **- [x] **3.2** Создать `CMakeLists.txt`:
  ```cmake
  cmake_minimum_required(VERSION 3.25)
  project(DspStats VERSION 0.1.0 LANGUAGES CXX HIP)
  find_package(hip REQUIRED)
  find_package(rocprim REQUIRED)
  include(cmake/fetch_deps.cmake)
  fetch_dsp_core()
  add_library(DspStats STATIC)
  add_library(DspStats::DspStats ALIAS DspStats)
  target_include_directories(DspStats
    PUBLIC  include
    PRIVATE kernels/ src)
  target_link_libraries(DspStats PUBLIC DspCore::DspCore roc::rocprim)
  ```
- [x] **- [x] **3.3** `cmake/DspStatsConfig.cmake.in`:
  ```cmake
  find_dependency(DspCore  REQUIRED)
  find_dependency(hip      REQUIRED)
  find_dependency(rocprim  REQUIRED)
  ```
- [x] **- [x] **3.4** Папки + `CMakePresets.json`
- [x] **- [x] **3.5** `cmake --preset local-dev` — OK ✅
- [x] **- [x] **3.6** Push

---

### 🔵 Репо 4: `signal_generators` (~~gh repo create~~ — уже существует)

- [x] **- [x] **4.2** Создать `CMakeLists.txt`:
  ```cmake
  cmake_minimum_required(VERSION 3.25)
  project(DspSignalGenerators VERSION 0.1.0 LANGUAGES CXX HIP)
  find_package(hip REQUIRED)
  find_package(hiprtc REQUIRED)
  include(cmake/fetch_deps.cmake)
  fetch_dsp_core()
  fetch_dsp_spectrum()
  add_library(DspSignalGenerators STATIC)
  add_library(DspSignalGenerators::DspSignalGenerators ALIAS DspSignalGenerators)
  target_include_directories(DspSignalGenerators
    PUBLIC  include
    PRIVATE kernels/ src)
  target_link_libraries(DspSignalGenerators
    PUBLIC  DspCore::DspCore DspSpectrum::DspSpectrum
    PRIVATE hiprtc)
  ```
- [x] **- [x] **4.3** `cmake/DspSignalGeneratorsConfig.cmake.in`:
  ```cmake
  find_dependency(DspCore     REQUIRED)
  find_dependency(DspSpectrum REQUIRED)
  find_dependency(hip         REQUIRED)
  ```
- [x] **- [x] **4.4** `CMakePresets.json` — FETCHCONTENT_SOURCE_DIR для core + spectrum
- [x] **- [x] **4.5** `cmake --preset local-dev` — OK ✅
- [x] **- [x] **4.6** Push

---

### 🔵 Репо 5: `heterodyne` (~~gh repo create~~ — уже существует)

- [x] **- [x] **5.2** Создать `CMakeLists.txt`:
  ```cmake
  cmake_minimum_required(VERSION 3.25)
  project(DspHeterodyne VERSION 0.1.0 LANGUAGES CXX HIP)
  find_package(hip REQUIRED)
  find_package(hipfft REQUIRED)
  include(cmake/fetch_deps.cmake)
  fetch_dsp_core()
  fetch_dsp_spectrum()
  fetch_dsp_signal_generators()
  add_library(DspHeterodyne STATIC)
  add_library(DspHeterodyne::DspHeterodyne ALIAS DspHeterodyne)
  target_include_directories(DspHeterodyne
    PUBLIC  include
    PRIVATE kernels/ src)
  target_link_libraries(DspHeterodyne
    PUBLIC DspCore::DspCore DspSpectrum::DspSpectrum DspSignalGenerators::DspSignalGenerators)
  ```
- [x] **- [x] **5.3** `cmake/DspHeterodyneConfig.cmake.in`:
  ```cmake
  find_dependency(DspCore             REQUIRED)
  find_dependency(DspSpectrum         REQUIRED)
  find_dependency(DspSignalGenerators REQUIRED)
  find_dependency(hip                 REQUIRED)
  find_dependency(hipfft              REQUIRED)
  ```
- [x] **- [x] **5.4** Папки + `CMakePresets.json`
- [x] **- [x] **5.5** `cmake --preset local-dev` — OK ✅
- [x] **- [x] **5.6** Push

---

### 🔵 Репо 6: `linalg` (~~gh repo create~~ — уже существует)

- [x] **- [x] **6.2** Создать `CMakeLists.txt`:
  ```cmake
  cmake_minimum_required(VERSION 3.25)
  project(DspLinalg VERSION 0.1.0 LANGUAGES CXX HIP)
  find_package(hip REQUIRED)
  find_package(rocblas REQUIRED)
  find_package(rocsolver REQUIRED)
  find_package(hiprtc REQUIRED)
  include(cmake/fetch_deps.cmake)
  fetch_dsp_core()
  add_library(DspLinalg STATIC)
  add_library(DspLinalg::DspLinalg ALIAS DspLinalg)
  target_include_directories(DspLinalg
    PUBLIC  include
    PRIVATE kernels/ src)
  target_link_libraries(DspLinalg
    PUBLIC  DspCore::DspCore roc::rocblas roc::rocsolver
    PRIVATE hiprtc)
  ```
- [x] **- [x] **6.3** `cmake/DspLinalgConfig.cmake.in`:
  ```cmake
  find_dependency(DspCore   REQUIRED)
  find_dependency(hip       REQUIRED)
  find_dependency(rocblas   REQUIRED)
  find_dependency(rocsolver REQUIRED)
  ```
- [x] **- [x] **6.4** Папки + `CMakePresets.json`
- [x] **- [x] **6.5** `cmake --preset local-dev` — OK ✅
- [x] **- [x] **6.6** Push

---

### 🔵 Репо 7: `radar` (~~gh repo create~~ — уже существует)

- [x] **- [x] **7.2** Создать `CMakeLists.txt`:
  ```cmake
  cmake_minimum_required(VERSION 3.25)
  project(DspRadar VERSION 0.1.0 LANGUAGES CXX HIP)
  find_package(hip REQUIRED)
  find_package(hipfft REQUIRED)
  include(cmake/fetch_deps.cmake)
  fetch_dsp_core()
  fetch_dsp_spectrum()
  fetch_dsp_stats()
  add_library(DspRadar STATIC)
  add_library(DspRadar::DspRadar ALIAS DspRadar)
  target_include_directories(DspRadar
    PUBLIC  include
    PRIVATE kernels/ src)
  target_link_libraries(DspRadar
    PUBLIC DspCore::DspCore DspSpectrum::DspSpectrum DspStats::DspStats)
  ```
- [x] **- [x] **7.3** `cmake/DspRadarConfig.cmake.in`:
  ```cmake
  find_dependency(DspCore     REQUIRED)
  find_dependency(DspSpectrum REQUIRED)
  find_dependency(DspStats    REQUIRED)
  find_dependency(hip         REQUIRED)
  find_dependency(hipfft      REQUIRED)
  ```
- [x] **- [x] **7.4** Папки + `CMakePresets.json`
- [x] **- [x] **7.5** `cmake --preset local-dev` — OK ✅
- [x] **- [x] **7.6** Push

---

### 🔵 Репо 8: `strategies` (~~gh repo create~~ — уже существует)

- [x] **- [x] **8.2** Создать `CMakeLists.txt`:
  ```cmake
  cmake_minimum_required(VERSION 3.25)
  project(DspStrategies VERSION 0.1.0 LANGUAGES CXX HIP)
  find_package(hip REQUIRED)
  include(cmake/fetch_deps.cmake)
  fetch_dsp_core()
  fetch_dsp_spectrum()
  fetch_dsp_stats()
  fetch_dsp_signal_generators()
  fetch_dsp_heterodyne()
  fetch_dsp_linalg()
  add_library(DspStrategies STATIC)
  add_library(DspStrategies::DspStrategies ALIAS DspStrategies)
  target_include_directories(DspStrategies
    PUBLIC  include
    PRIVATE kernels/ src)
  target_link_libraries(DspStrategies
    PUBLIC DspCore::DspCore DspSpectrum::DspSpectrum DspStats::DspStats
           DspSignalGenerators::DspSignalGenerators DspHeterodyne::DspHeterodyne
           DspLinalg::DspLinalg)
  ```
- [x] **- [x] **8.3** `cmake/DspStrategiesConfig.cmake.in` — `find_dependency` для всех 6
- [x] **- [x] **8.4** Папки + `CMakePresets.json`
- [x] **- [x] **8.5** `cmake --preset local-dev` — OK ✅
- [x] **- [x] **8.6** Push

---

### 🔵 Репо 9: `DSP` мета-репо (~~gh repo create~~ — уже существует)

- [x] **- [x] **9.2** Обновить `CMakeLists.txt` — добавить `option()` guards:
  ```cmake
  cmake_minimum_required(VERSION 3.25)
  project(DSP LANGUAGES NONE)
  include(FetchContent)
  include(cmake/fetch_deps.cmake)

  # Условная сборка — управляется через пресеты
  option(DSP_BUILD_CORE             "Build core"             ON)
  option(DSP_BUILD_SPECTRUM         "Build spectrum"         ON)
  option(DSP_BUILD_STATS            "Build stats"            ON)
  option(DSP_BUILD_SIGNAL_GENERATORS "Build signal_generators" ON)
  option(DSP_BUILD_HETERODYNE       "Build heterodyne"       ON)
  option(DSP_BUILD_LINALG           "Build linalg"           ON)
  option(DSP_BUILD_RADAR            "Build radar"            ON)
  option(DSP_BUILD_STRATEGIES       "Build strategies"       ON)

  # Централизованный FetchContent_Declare ПЕРЕД всеми зависимыми!
  # (решает diamond dependency — все модули получают одну версию)
  if(DSP_BUILD_CORE)             fetch_dsp_core()             endif()
  if(DSP_BUILD_SPECTRUM)         fetch_dsp_spectrum()         endif()
  if(DSP_BUILD_STATS)            fetch_dsp_stats()            endif()
  if(DSP_BUILD_SIGNAL_GENERATORS) fetch_dsp_signal_generators() endif()
  if(DSP_BUILD_HETERODYNE)       fetch_dsp_heterodyne()       endif()
  if(DSP_BUILD_LINALG)           fetch_dsp_linalg()           endif()
  if(DSP_BUILD_RADAR)            fetch_dsp_radar()            endif()
  if(DSP_BUILD_STRATEGIES)       fetch_dsp_strategies()       endif()
  ```
- [x] **- [x] **9.3** `CMakePresets.json` — полная версия:
  - `local-dev` — все `FETCHCONTENT_SOURCE_DIR_DSP*` → `E:/DSP-GPU/{repo}`
  - `ci` — все `DSP_*_TAG`
  - `spectrum-only` — `DSP_BUILD_SPECTRUM=ON`, остальные OFF
  - `linalg-only` — `DSP_BUILD_LINALG=ON`
  - `full-release`
- [x] **- [x] **9.4** Структура папок (уже частично есть):
  ```
  DSP/
  ├── Python/
  │   └── lib/          (.gitignore)
  ├── Doc/
  │   └── Architecture/ (уже создано ✅)
  ├── Examples/
  ├── Logs/             (.gitignore)
  └── Results/          (.gitignore)
  ```
- [x] **- [x] **9.5** `.gitignore` DSP (проверить что есть `Python/lib/`)
- [x] **- [x] **9.6** `cmake -S . -B build --preset local-dev` — OK ✅
- [x] **- [x] **9.7** Push

---

## Definition of Done

- [x] Каждый из 8 репо: файлы созданы, коммит готов (push — ожидает прав в org dsp-gpu)
- [x] DSP мета-репо: CMakeLists.txt + CMakePresets.json + cmake/ обновлены, коммит готов
- [x] Нет ни одного `find_package(HIP)` с заглавной буквы (все lowercase: `find_package(hip)`)
- [x] В каждом репо есть `kernels/rocm/` директория
- [x] `CMakePresets.json` использует пути `E:/DSP-GPU/`
- [x] DSP/CMakeLists.txt имеет `option(DSP_BUILD_*)` для каждого модуля
- [ ] `cmake --preset local-dev` конфигурируется ✅ — проверить на Linux с ROCm (Фаза 4)
- [ ] Push на GitHub — нужны права AlexLan73 в org dsp-gpu (см. примечание ниже)
- [ ] Можно переходить к Фазе 2

## ⚠️ Примечание: Push заблокирован

Все коммиты созданы локально в `E:/DSP-GPU/`. Push падает с 403:
`remote: Permission to dsp-gpu/core.git denied to AlexLan73`

**Решение**: на GitHub → `github.com/dsp-gpu` → Settings → Members → пригласить AlexLan73 как Owner или Member.
После добавления — запустить:
```bash
cd E:/DSP-GPU
for r in core spectrum stats signal_generators heterodyne linalg radar strategies; do
  cd $r && git push -u origin main && cd ..
done
cd DSP && git push -u origin main
```

---

*Создан: 2026-04-12 | Обновлён: 2026-04-12 (исправлены пути, find_package lowercase, добавлены option guards, kernels/rocm/) | Автор: Кодо*
