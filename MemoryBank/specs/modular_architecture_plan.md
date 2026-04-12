# 🏛️ План модульной архитектуры DSP-GPU

> **Статус**: v2 — обновлён после ревью Alex (2026-04-11)
> **Дата**: 2026-04-11
> **Цель**: Перенос GPUWorkLib в GitHub Organization `github.com/dsp-gpu`
> **Вариант**: B — создаём новые репо, не трогаем рабочий GPUWorkLib
> **Ревью**: [`modular_architecture_plan_REVIEW.md`](modular_architecture_plan_REVIEW.md)

---

## 1. Итоговая карта репозиториев

> **После ревью A1 + A2**: `statistics` вынесен в отдельный `stats`, `signal` разделён на `signal_generators` + `heterodyne`. Итого 9 репо вместо 7.

```
github.com/dsp-gpu/
│
├── core              ← DrvGPU (test_utils — отдельная target DspCore::TestUtils)
├── spectrum          ← fft_func + filters + lch_farrow
├── stats             ← statistics (welford, medians, radix sort)
├── signal_generators ← CW / LFM / Noise / Script / FormSignal
├── heterodyne        ← LFM Dechirp, NCO, MixDown/MixUp
├── linalg            ← vector_algebra + capon
├── radar             ← range_angle + fm_correlator
├── strategies        ← strategies (pipeline v1, v2...)
│
└── DSP               ← МЕТА-РЕПО
    ├── CMakeLists.txt      централизованно объявляет FetchContent_Declare
    ├── CMakePresets.json   сценарии сборки
    ├── Python/             ВСЕ тесты + аналитика
    ├── Doc/                общая документация + Doxygen
    │   └── addition/       ← бывший Doc_Addition/ (перенесём, потом почистим)
    ├── Examples/           примеры конфигураций
    ├── Logs/               ← per-GPU логи plog (gitignore)
    └── Results/            ← JSON, Plots, Profiler (gitignore)
```

---

## 2. Граф зависимостей

> ⚠️ **Перед Фазой 1** — сгенерировать реальный граф зависимостей (см. R2 в ревью):
> «Создать зависимости как в `Doc/Architecture/`, их проанализировать, потом делать».
> Граф ниже — предварительный, должен быть подтверждён автоматическим анализом
> (`cmake --graphviz` + ручной просмотр модульных CMakeLists.txt).

```
Уровень 0:  core (DrvGPU)
                │
        ┌───────┼──────────┬───────────┐
        ▼       ▼          ▼           ▼
Уровень 1:  spectrum    stats      linalg
            (fft_func, (welford,  (vector_algebra,
             filters,   medians,   capon)
             lch_farrow) radix)
                │                        │
                ▼                        │
Уровень 2:  signal_generators            │
            (CW/LFM/Noise/                │
             Script/FormSignal)           │
                │                         │
                ▼                         │
Уровень 3:  heterodyne                   │
            (Dechirp/NCO/                 │
             MixDown/MixUp)               │
                │                         │
                ▼                         │
Уровень 4:  radar                         │
            (range_angle,                 │
             fm_correlator)               │
                │                         │
                └───────────┬─────────────┘
                            ▼
Уровень 5:             strategies
                            │
                            ▼
Уровень 6:              DSP (meta)
```

### Таблица зависимостей

| Репо | Зависит от | Внешние SDK |
|------|-----------|-------------|
| `core` | — | HIP/ROCm, HSA, OpenCL (только для стыковки данных в GPU, вычисления на ROCm) |
| `spectrum` | core | hipFFT |
| `stats` | core | rocprim |
| `linalg` | core | rocBLAS, rocSOLVER, hiprtc |
| `signal_generators` | core + spectrum | HIP, hiprtc (ScriptGenerator) |
| `heterodyne` | core + spectrum + signal_generators | HIP, hipFFT |
| `radar` | core + spectrum + stats | HIP, hipFFT, hiprtc |
| `strategies` | core + spectrum + stats + signal_generators + heterodyne + linalg | hipBLAS, hipFFT |
| `DSP` | все | — |

### Почему именно такое разбиение

**`stats` — отдельный репо** (решение A1 ревью):
- `statistics` концептуально не связан с FFT
- Потребитель, которому нужна только статистика, не должен тянуть hipFFT
- Зависимости ортогональны spectrum (rocprim vs hipFFT)
- `radar` использует `stats` напрямую — не плодим свою статистику

**`signal_generators` и `heterodyne` — разные репо** (решение A2 ревью):
- Генераторы самодостаточны (только spectrum::lch_farrow для DelayedFormSignal)
- `heterodyne` нужен для LFM Dechirp pipeline — использует fft_func + signal_generators
- Потребителю генераторов не нужен hipFFT через heterodyne

**`lch_farrow` → spectrum** (не в signal_generators):
- Farrow-интерполяция — DSP-алгоритм обработки, не генерации
- `signal_generators` использует `lch_farrow` (DelayedFormSignalGenerator) →
  signal_generators зависит от spectrum, это осознанно

**`radar` = range_angle + fm_correlator** (решение #5 ревью):
- Оба специфичны для радарного приложения
- range_angle: дальность + угол через dechirp + FFT
- fm_correlator: M-sequence FM корреляция
- **Radar должен работать в общем стандарте**: если считает спектр или статистику —
  использует отлаженные `spectrum` и `stats`, **не плодим новых сущностей**

**`linalg` = vector_algebra + capon**:
- capon зависит от vector_algebra (Cholesky инверсия для MVDR)
- Оба — линейная алгебра на GPU
- rocBLAS + rocSOLVER — одна группа зависимостей

**`strategies` — отдельный репо навсегда**:
- Pipeline-стратегии антенной решётки — прикладная задача
- При смене алгоритма → новый репо: strategies-music, strategies-mvdr2
- Не навязывать другим отделам свою стратегию

---

## 3. Структура каждого репо

### Шаблон (применяется ко всем)

```
dsp-gpu/{repo}/
├── CMakeLists.txt              ← standalone build
├── CMakePresets.json           ← local-dev + ci presets
├── cmake/
│   ├── {Repo}Config.cmake.in  ← для find_package
│   └── fetch_deps.cmake        ← FetchContent зависимостей
├── include/
│   └── dsp/
│   │    └── {module}.hpp        ← публичные заголовки
│   └── (кернелы для rocm)/
│          └── файлы
├── src/
│   └── {module}/               ← исходники + kernels
├── tests/                      ← C++ тесты
│   ├── all_test.hpp
│   └── test_*.hpp
├── examples/                   ← минимальные примеры запуска
│   └── basic_usage.cpp
└── README.md
```

### core — детальная структура

```
dsp-gpu/core/
├── CMakeLists.txt
├── CMakePresets.json
├── cmake/
│   ├── DspCoreConfig.cmake.in   ← с find_dependency(HIP, OpenCL)
│   └── FindROCm.cmake           ← поиск ROCm/HIP SDK
├── include/
│   └── dsp/
│       ├── drv_gpu.hpp          ← главный интерфейс
│       ├── backends/
│       │   ├── rocm_backend.hpp
│       │   └── opencl_backend.hpp     ← только стыковка данных в GPU
│       ├── common/
│       │   ├── gpu_device_info.hpp
│       │   └── backend_type.hpp
│       ├── profiler/
│       │   └── gpu_profiler.hpp
│       └── config/
│           └── gpu_config.hpp
├── src/
│   ├── backends/
│   │   ├── rocm/
│   │   └── opencl/
│   ├── config/
│   └── profiler/
├── test_utils/                  ← ОТДЕЛЬНАЯ target DspCore::TestUtils
│   ├── include/dsp/test/
│   │   ├── test_runner.hpp
│   │   ├── gpu_test_base.hpp
│   │   └── gpu_benchmark_base.hpp
│   └── CMakeLists.txt           ← add_library(DspCoreTestUtils INTERFACE)
├── python/                      ← pybind11 биндинги (опционально)
│   ├── CMakeLists.txt
│   ├── dsp_core_module.cpp      ← PYBIND11_MODULE(dsp_core, m) { ... }
│   └── dsp_core.pyi
├── tests/
│   └── test_drv_gpu.hpp
└── configGPU.json               ← конфиг GPU (копируется в build)
```

> ⚠️ **Важно (#1 ревью)**: `test_utils/` — это **отдельная target** `DspCore::TestUtils`,
> не часть публичного API `DspCore::DspCore`. Тестовые заголовки не протекают к
> пользователям библиотеки.

---

## 4. CMakeLists.txt — шаблоны

### 4.1 core/CMakeLists.txt

```cmake
cmake_minimum_required(VERSION 3.25)
project(DspCore VERSION 0.1.0 LANGUAGES CXX HIP)

# ── Опции ──────────────────────────────────────────────────────────
option(DSP_CORE_BUILD_TESTS    "Build tests"         ON)
option(DSP_CORE_BUILD_EXAMPLES "Build examples"      OFF)
option(DSP_CORE_BUILD_PYTHON   "Build Python bindings" OFF)

# ── ROCm/HIP ───────────────────────────────────────────────────────
find_package(HIP REQUIRED)
find_package(OpenCL REQUIRED)

# ── Основная библиотека ────────────────────────────────────────────
add_library(DspCore STATIC
  src/backends/rocm/rocm_backend.cpp
  src/backends/opencl/opencl_backend.cpp
  src/config/gpu_config.cpp
  src/profiler/gpu_profiler.cpp
)
add_library(DspCore::DspCore ALIAS DspCore)

target_include_directories(DspCore
  PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:include>
)
target_link_libraries(DspCore PUBLIC hip::host OpenCL::OpenCL)
target_compile_features(DspCore PUBLIC cxx_std_17)

# ── TestUtils — ОТДЕЛЬНАЯ target (#1 ревью) ────────────────────────
# Не протекает в публичный API DspCore — подключается тестами явно.
add_library(DspCoreTestUtils INTERFACE)
add_library(DspCore::TestUtils ALIAS DspCoreTestUtils)

target_include_directories(DspCoreTestUtils INTERFACE
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/test_utils/include>
  $<INSTALL_INTERFACE:include/dsp/test>
)
target_link_libraries(DspCoreTestUtils INTERFACE DspCore::DspCore)

# ── Install + Export ────────────────────────────────────────────────
include(GNUInstallDirs)

install(TARGETS DspCore DspCoreTestUtils
  EXPORT DspCoreTargets
  ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
)

# Основные заголовки
install(DIRECTORY include/
  DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}
)
# Тестовые заголовки — в отдельный подкаталог
install(DIRECTORY test_utils/include/
  DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/dsp/test
)

include(CMakePackageConfigHelpers)
configure_package_config_file(cmake/DspCoreConfig.cmake.in
  "${CMAKE_CURRENT_BINARY_DIR}/DspCoreConfig.cmake"
  INSTALL_DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/DspCore
)
write_basic_package_version_file(
  "${CMAKE_CURRENT_BINARY_DIR}/DspCoreConfigVersion.cmake"
  VERSION ${PROJECT_VERSION}  COMPATIBILITY SameMajorVersion
)
install(EXPORT DspCoreTargets NAMESPACE DspCore::
  DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/DspCore
)
install(FILES
  "${CMAKE_CURRENT_BINARY_DIR}/DspCoreConfig.cmake"
  "${CMAKE_CURRENT_BINARY_DIR}/DspCoreConfigVersion.cmake"
  DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/DspCore
)

# ── Тесты ──────────────────────────────────────────────────────────
if(DSP_CORE_BUILD_TESTS)
  add_subdirectory(tests)
  # tests/CMakeLists.txt:
  #   target_link_libraries(test_drv_gpu
  #     PRIVATE DspCore::DspCore DspCore::TestUtils)
endif()

# ── Python bindings ────────────────────────────────────────────────
if(DSP_CORE_BUILD_PYTHON)
  add_subdirectory(python)
endif()
```

### 4.1a cmake/DspCoreConfig.cmake.in (#4 ревью)

```cmake
@PACKAGE_INIT@

include(CMakeFindDependencyMacro)
find_dependency(HIP REQUIRED)
find_dependency(OpenCL REQUIRED)

include("${CMAKE_CURRENT_LIST_DIR}/DspCoreTargets.cmake")
check_required_components(DspCore)
```

> Без `find_dependency` потребитель `find_package(DspCore)` получит targets,
> но при линковке упадёт на `hip::host not found`.

### 4.2 spectrum/CMakeLists.txt (паттерн для всех зависимых)

```cmake
cmake_minimum_required(VERSION 3.25)
project(DspSpectrum VERSION 0.1.0 LANGUAGES CXX HIP)

option(DSP_SPECTRUM_BUILD_TESTS  "Build tests"         ON)
option(DSP_SPECTRUM_BUILD_PYTHON "Build Python bindings" OFF)

# ── Зависимости ────────────────────────────────────────────────────
find_package(HIP REQUIRED)
find_package(hipfft REQUIRED)

# Ищем DspCore — FetchContent с FIND_PACKAGE_ARGS (R1 ревью)
include(cmake/fetch_deps.cmake)
fetch_dsp_core()         # ← версия берётся из DSP_CORE_TAG или по умолчанию

# ── Библиотека ─────────────────────────────────────────────────────
add_library(DspSpectrum STATIC
  src/fft_func/fft_processor_rocm.cpp
  src/filters/fir_filter_rocm.cpp
  src/filters/iir_filter_rocm.cpp
  src/lch_farrow/lch_farrow_rocm.cpp
  # ... остальные (statistics вынесен в отдельный репо stats!)
)
add_library(DspSpectrum::DspSpectrum ALIAS DspSpectrum)

target_include_directories(DspSpectrum
  PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:include>
)
target_link_libraries(DspSpectrum
  PUBLIC DspCore::DspCore hip::hipfft
)

# ── Install + Export ────────────────────────────────────────────────
include(GNUInstallDirs)
install(TARGETS DspSpectrum EXPORT DspSpectrumTargets
  ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
)
install(DIRECTORY include/ DESTINATION ${CMAKE_INSTALL_INCLUDEDIR})

include(CMakePackageConfigHelpers)
configure_package_config_file(cmake/DspSpectrumConfig.cmake.in
  "${CMAKE_CURRENT_BINARY_DIR}/DspSpectrumConfig.cmake"
  INSTALL_DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/DspSpectrum
)
write_basic_package_version_file(
  "${CMAKE_CURRENT_BINARY_DIR}/DspSpectrumConfigVersion.cmake"
  VERSION ${PROJECT_VERSION} COMPATIBILITY SameMajorVersion
)
install(EXPORT DspSpectrumTargets NAMESPACE DspSpectrum::
  DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/DspSpectrum
)
install(FILES
  "${CMAKE_CURRENT_BINARY_DIR}/DspSpectrumConfig.cmake"
  "${CMAKE_CURRENT_BINARY_DIR}/DspSpectrumConfigVersion.cmake"
  DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/DspSpectrum
)

if(DSP_SPECTRUM_BUILD_TESTS)
  add_subdirectory(tests)
endif()
if(DSP_SPECTRUM_BUILD_PYTHON)
  add_subdirectory(python)
endif()
```

### 4.2a cmake/DspSpectrumConfig.cmake.in (#4 ревью)

```cmake
@PACKAGE_INIT@

include(CMakeFindDependencyMacro)
find_dependency(DspCore  REQUIRED)   # ← критично!
find_dependency(HIP      REQUIRED)
find_dependency(hipfft   REQUIRED)

include("${CMAKE_CURRENT_LIST_DIR}/DspSpectrumTargets.cmake")
check_required_components(DspSpectrum)
```

> Шаблон для всех зависимых репо (`stats`, `signal_generators`, `heterodyne`,
> `linalg`, `radar`, `strategies`): каждый `{Repo}Config.cmake.in` обязан
> содержать `find_dependency(DspCore)` + свои внешние SDK.

### 4.3 cmake/fetch_deps.cmake (#2 ревью + R1)

Использует `FIND_PACKAGE_ARGS` (CMake 3.24+, Modern CMake) — сначала пробует
системный пакет, иначе тянет с GitHub. Теги берутся из cmake-переменных
(`DSP_*_TAG`), которые задаются в `CMakePresets.json` через CI preset.

```cmake
include_guard(GLOBAL)
include(FetchContent)

# Значения по умолчанию — перекрываются через -D или preset
set(DSP_CORE_TAG              "v0.1.0" CACHE STRING "Tag for DspCore")
set(DSP_SPECTRUM_TAG          "v0.1.0" CACHE STRING "Tag for DspSpectrum")
set(DSP_STATS_TAG             "v0.1.0" CACHE STRING "Tag for DspStats")
set(DSP_SIGNAL_GENERATORS_TAG "v0.1.0" CACHE STRING "Tag for DspSignalGenerators")
set(DSP_HETERODYNE_TAG        "v0.1.0" CACHE STRING "Tag for DspHeterodyne")
set(DSP_LINALG_TAG            "v0.1.0" CACHE STRING "Tag for DspLinalg")
set(DSP_RADAR_TAG             "v0.1.0" CACHE STRING "Tag for DspRadar")
set(DSP_STRATEGIES_TAG        "v0.1.0" CACHE STRING "Tag for DspStrategies")

function(dsp_fetch_package name repo tag_var)
  FetchContent_Declare(${name}
    GIT_REPOSITORY https://github.com/dsp-gpu/${repo}.git
    GIT_TAG        ${${tag_var}}
    GIT_SHALLOW    TRUE
    FIND_PACKAGE_ARGS NAMES ${name} CONFIG
  )
  FetchContent_MakeAvailable(${name})
endfunction()

# Удобные обёртки — читают *_TAG из cache:
macro(fetch_dsp_core)              dsp_fetch_package(DspCore              core              DSP_CORE_TAG)              endmacro()
macro(fetch_dsp_spectrum)          dsp_fetch_package(DspSpectrum          spectrum          DSP_SPECTRUM_TAG)          endmacro()
macro(fetch_dsp_stats)             dsp_fetch_package(DspStats             stats             DSP_STATS_TAG)             endmacro()
macro(fetch_dsp_signal_generators) dsp_fetch_package(DspSignalGenerators  signal_generators DSP_SIGNAL_GENERATORS_TAG) endmacro()
macro(fetch_dsp_heterodyne)        dsp_fetch_package(DspHeterodyne        heterodyne        DSP_HETERODYNE_TAG)        endmacro()
macro(fetch_dsp_linalg)            dsp_fetch_package(DspLinalg            linalg            DSP_LINALG_TAG)            endmacro()
macro(fetch_dsp_radar)             dsp_fetch_package(DspRadar             radar             DSP_RADAR_TAG)             endmacro()
macro(fetch_dsp_strategies)        dsp_fetch_package(DspStrategies        strategies        DSP_STRATEGIES_TAG)        endmacro()
```

**Как это работает**:
1. `FIND_PACKAGE_ARGS NAMES DspCore CONFIG` — сначала пробует `find_package(DspCore CONFIG)`
2. Если в системе/кеше есть — использует его, не клонирует
3. Если нет — клонирует `GIT_TAG ${DSP_CORE_TAG}` (значение из CACHE)
4. В CI preset: `"DSP_CORE_TAG": "v0.3.0"` → будет использоваться
5. При локальной разработке: `FETCHCONTENT_SOURCE_DIR_DSPCORE=/path/to/core` → локальная папка

---

## 5. CMakePresets.json для DSP (мета-репо)

```json
{
  "version": 6,
  "configurePresets": [
    {
      "name": "base",
      "hidden": true,
      "generator": "Ninja",
      "binaryDir": "${sourceDir}/build/${presetName}"
    },
    {
      "name": "local-dev",
      "displayName": "Local: все репо рядом на диске",
      "inherits": "base",
      "description": "Не клонирует с GitHub — использует локальные папки",
      "cacheVariables": {
        "FETCHCONTENT_SOURCE_DIR_DSPCORE":             "${sourceDir}/../core",
        "FETCHCONTENT_SOURCE_DIR_DSPSPECTRUM":         "${sourceDir}/../spectrum",
        "FETCHCONTENT_SOURCE_DIR_DSPSTATS":            "${sourceDir}/../stats",
        "FETCHCONTENT_SOURCE_DIR_DSPSIGNALGENERATORS": "${sourceDir}/../signal_generators",
        "FETCHCONTENT_SOURCE_DIR_DSPHETERODYNE":       "${sourceDir}/../heterodyne",
        "FETCHCONTENT_SOURCE_DIR_DSPLINALG":           "${sourceDir}/../linalg",
        "FETCHCONTENT_SOURCE_DIR_DSPRADAR":            "${sourceDir}/../radar",
        "FETCHCONTENT_SOURCE_DIR_DSPSTRATEGIES":       "${sourceDir}/../strategies",
        "CMAKE_BUILD_TYPE": "Debug"
      }
    },
    {
      "name": "ci",
      "displayName": "CI: FetchContent с тегами из GitHub",
      "inherits": "base",
      "description": "Тянет конкретные версии каждого репо",
      "cacheVariables": {
        "DSP_CORE_TAG":              "v0.1.0",
        "DSP_SPECTRUM_TAG":          "v0.1.0",
        "DSP_STATS_TAG":             "v0.1.0",
        "DSP_SIGNAL_GENERATORS_TAG": "v0.1.0",
        "DSP_HETERODYNE_TAG":        "v0.1.0",
        "DSP_LINALG_TAG":            "v0.1.0",
        "DSP_RADAR_TAG":             "v0.1.0",
        "DSP_STRATEGIES_TAG":        "v0.1.0",
        "CMAKE_BUILD_TYPE": "Release"
      }
    },
    {
      "name": "spectrum-only",
      "displayName": "Только spectrum (отдел анализа сигналов)",
      "inherits": "local-dev",
      "cacheVariables": {
        "DSP_BUILD_STATS":             "OFF",
        "DSP_BUILD_SIGNAL_GENERATORS": "OFF",
        "DSP_BUILD_HETERODYNE":        "OFF",
        "DSP_BUILD_LINALG":            "OFF",
        "DSP_BUILD_RADAR":             "OFF",
        "DSP_BUILD_STRATEGIES":        "OFF"
      }
    },
    {
      "name": "linalg-only",
      "displayName": "Только linalg (отдел линейной алгебры)",
      "inherits": "local-dev",
      "cacheVariables": {
        "DSP_BUILD_SPECTRUM":          "OFF",
        "DSP_BUILD_STATS":             "OFF",
        "DSP_BUILD_SIGNAL_GENERATORS": "OFF",
        "DSP_BUILD_HETERODYNE":        "OFF",
        "DSP_BUILD_RADAR":             "OFF",
        "DSP_BUILD_STRATEGIES":        "OFF"
      }
    },
    {
      "name": "full-release",
      "displayName": "Полная сборка (релиз)",
      "inherits": "ci",
      "cacheVariables": {
        "CMAKE_BUILD_TYPE": "Release",
        "DSP_BUILD_PYTHON": "ON"
      }
    }
  ],
  "buildPresets": [
    { "name": "local-dev",    "configurePreset": "local-dev"    },
    { "name": "ci",           "configurePreset": "ci"           },
    { "name": "spectrum-only","configurePreset": "spectrum-only"},
    { "name": "linalg-only",  "configurePreset": "linalg-only"  },
    { "name": "full-release", "configurePreset": "full-release" }
  ]
}
```

---

## 6. Структура Python/ в мета-репо DSP

```
DSP/Python/
├── lib/                        ← .pyd/.so (gitignore, собирается cmake --install)
│
├── common/                     ← из текущего Python_test/common/
│   ├── runner.py               (TestRunner, SkipTest)
│   ├── gpu_loader.py           (Singleton GPULoader)
│   └── base.py                 (TestBase)
│
├── spectrum/                   ← тесты + аналитика (fft + filters + lch_farrow)
│   ├── test_fft.py
│   ├── test_filters.py
│   ├── test_lch_farrow.py
│   └── analytics/
│       ├── spectrum_quality.py  ← SNR, THD, SFDR
│       └── filter_response.py   ← АЧХ, ФЧХ
│
├── stats/                      ← тесты + аналитика (вынесен из spectrum)
│   ├── test_statistics.py
│   └── analytics/
│       └── distribution.py     ← mean, std, median анализ
│
├── signal_generators/          ← тесты + аналитика
│   ├── test_cw.py
│   ├── test_lfm.py
│   ├── test_noise.py
│   ├── test_form_signal.py
│   └── analytics/
│       └── lfm_analysis.py     ← параметры ЛЧМ
│
├── heterodyne/                 ← тесты + аналитика
│   ├── test_dechirp.py
│   ├── test_nco.py
│   └── analytics/
│       └── snr_analysis.py     ← ОСШ по дальности
│
├── linalg/                     ← тесты + аналитика
│   ├── test_vector_algebra.py
│   ├── test_capon.py
│   └── analytics/
│       └── beampattern.py      ← ДН антенной решётки
│
├── radar/                      ← тесты + аналитика
│   ├── test_range_angle.py
│   ├── test_fm_correlator.py
│   └── analytics/
│       ├── range_resolution.py ← разрешение по дальности
│       └── fm_correlation.py   ← характеристики корреляции
│
├── strategies/                 ← тесты + аналитика
│   ├── test_antenna_processor.py
│   └── analytics/
│       └── pipeline_perf.py    ← производительность pipeline
│
└── integration/                ← сквозные тесты
    └── test_full_pipeline.py   ← core → spectrum → signal_generators → strategies
```

**Запуск** (как сейчас, без pytest):
```bash
"F:\Program Files (x86)\Python314\python.exe" Python/spectrum/test_fft.py
"F:\Program Files (x86)\Python314\python.exe" Python/integration/test_full_pipeline.py
```

---

## 7. Порядок создания репо и переноса (Вариант B)

> **Стратегия переноса истории git**: копируем без истории (решение #9 ревью).
> При появлении необходимости смотрим в исходный репо GPUWorkLib.
> `git subtree split` / `git filter-repo` не используем.

### Фаза 0: Аудит зависимостей (R2 ревью) — ПЕРЕД всем остальным

```
  0.1. Создать Doc/Architecture/dependencies.md — реальный граф зависимостей
  0.2. Проанализировать: кто кого include'ит, кто кого линкует
  0.3. Сверить с графом раздела 2 этого плана
  0.4. Если нашлись расхождения — обновить раздел 2
  0.5. Только после согласия с графом — начинать Фазу 1
```

### Фаза 1: Создаём скелет (без реального кода) — делаем сейчас на Windows

```
День 1 (сегодня/завтра):
  1.  github.com/dsp-gpu/core              → CMakeLists.txt + DspCoreConfig.cmake.in
  2.  github.com/dsp-gpu/spectrum          → CMakeLists.txt + fetch core
  3.  github.com/dsp-gpu/stats             → CMakeLists.txt + fetch core
  4.  github.com/dsp-gpu/signal_generators → CMakeLists.txt + fetch core + spectrum
  5.  github.com/dsp-gpu/heterodyne        → CMakeLists.txt + fetch core + spectrum + signal_generators
  6.  github.com/dsp-gpu/linalg            → CMakeLists.txt + fetch core
  7.  github.com/dsp-gpu/radar             → CMakeLists.txt + fetch core + spectrum + stats
  8.  github.com/dsp-gpu/strategies        → CMakeLists.txt + fetch all
  9.  github.com/dsp-gpu/DSP               → мета-репо + CMakePresets.json

  Цель: убедиться что CMake конфигурируется без ошибок (без реального кода)
```

### Фаза 2: Копируем реальный код — делаем на Windows

```
День 2:
  10. core/              ← копируем DrvGPU/ из GPUWorkLib
  11. core/test_utils/   ← копируем modules/test_utils/ (отдельная target!)
  12. spectrum/          ← копируем modules/fft_func/ + filters/ + lch_farrow/
                          (БЕЗ statistics — оно идёт в stats!)
  13. stats/             ← копируем modules/statistics/
  14. signal_generators/ ← копируем modules/signal_generators/
  15. heterodyne/        ← копируем modules/heterodyne/
  16. linalg/            ← копируем modules/vector_algebra/ + capon/
  17. radar/             ← копируем modules/range_angle/ + fm_correlator/
  18. strategies/        ← копируем modules/strategies/
  19. DSP/Python/        ← копируем Python_test/ (переименовываем папки)

  Что НЕ копируем:
  - OpenCL .cl kernels — мы не считаем на OpenCl, только стыковка данных в GPU (#10)
  - .claude/worktrees/ — только локально
  - ветка nvidia / OpenCL-специфичный код

  Цель: всё скопировано, но ещё не тестировано на GPU
```

### Фаза 3: Правим CMakeLists.txt в каждом модуле

```
День 2 (продолжение) — День 3:
  20. В каждом модуле: заменить target_link_libraries
      БЫЛО: drvgpu, lch_farrow (пути в GPUWorkLib)
      СТАЛО: DspCore::DspCore, DspSpectrum::DspSpectrum, ...
  21. Убрать абсолютные пути (${CMAKE_SOURCE_DIR}/modules/...)
  22. Добавить $<BUILD_INTERFACE:...> в include directories
  23. Добавить install() + export() + Config.cmake.in в каждый репо
  24. Добавить namespace dsp:: в C++ код (R5 ревью — сразу, не откладывая):
      - класс DrvGPU          → dsp::DrvGPU
      - класс FFTProcessor    → dsp::spectrum::FFTProcessor
      - класс Dechirp         → dsp::heterodyne::Dechirp
      - PYBIND11_MODULE names остаются как есть (dsp_core, dsp_spectrum...)
```

### Фаза 4: Тестирование на GPU — понедельник на работе (Debian + Radeon)

```
Понедельник:
  25. git clone https://github.com/dsp-gpu/DSP на Debian
  26. cmake --preset local-dev (или ci)
  27. cmake --build build/local-dev
  28. Запускаем C++ тесты модуль за модулем:
      core → spectrum → stats → linalg → signal_generators → heterodyne → radar → strategies
  29. Запускаем Python тесты из DSP/Python/
  30. Если ошибки — правим, коммитим, пушим
```

---

## 8. Важные технические решения

### 8.1 Diamond dependency — централизованно в мета-репо (#6 ревью)

**Проблема**: если `strategies` зависит от `core` через `spectrum`, `stats`,
`signal_generators`, `heterodyne`, `linalg`, а каждый из них объявляет
`FetchContent_Declare(DspCore GIT_TAG ...)`, то при несовпадении версий
FetchContent возьмёт **первый объявленный** — без warning. Silent version mismatch.

**Решение**: в мета-репо `DSP/CMakeLists.txt` **централизованно** объявляем все
`FetchContent_Declare` ПЕРЕД подключением любых зависимых репо:

```cmake
# DSP/CMakeLists.txt
cmake_minimum_required(VERSION 3.25)
project(DSP LANGUAGES NONE)

include(FetchContent)
include(cmake/fetch_deps.cmake)   # определяет DSP_*_TAG и функцию dsp_fetch_package

# ── Централизованное объявление ВСЕХ репо ─────────────────────────
# Когда зависимые репо потом сделают FetchContent_Declare(DspCore ...) —
# CMake увидит, что оно уже объявлено, и возьмёт наши теги.
fetch_dsp_core()              # тянет DspCore по DSP_CORE_TAG
fetch_dsp_spectrum()          # spectrum сам попытается объявить DspCore — игнор
fetch_dsp_stats()
fetch_dsp_signal_generators()
fetch_dsp_heterodyne()
fetch_dsp_linalg()
fetch_dsp_radar()
fetch_dsp_strategies()
```

**Правило**: при подъёме версии `core` (например до `v0.2.0`) — меняем
`DSP_CORE_TAG` в `CMakePresets.json` мета-репо (или через `-D`). Зависимые
репо в CI собираются изолированно со своим `DSP_CORE_TAG`, но при сборке DSP
действует единая версия. Subtle, но работает.

> ⚠️ При изолированной сборке одного репо (например только `heterodyne`) —
> он возьмёт свой `DSP_CORE_TAG`. Это нормально для локальной разработки,
> но в production — собираем из мета-репо DSP.

### 8.2 Локальная разработка (FETCHCONTENT_SOURCE_DIR)

Когда правим core и сразу тестируем в spectrum:
```bash
cmake -S . -B build \
  -DFETCHCONTENT_SOURCE_DIR_DSPCORE=/home/alex/dsp-gpu/core
# → CMake использует локальную папку, не клонирует с GitHub
```
Или через preset `local-dev` в CMakePresets.json.

### 8.3 Namespace C++ (R5 ревью — делаем сразу в Фазе 3)

Все классы переходят в namespace `dsp::` **в момент копирования + правки
CMakeLists** (Фаза 3, шаг 24). Не откладываем — избегаем второго рефакторинга.

```cpp
namespace dsp {
  class DrvGPU { ... };                 // core
  namespace spectrum {
    class FFTProcessor { ... };         // spectrum/fft_func
    class FirFilter    { ... };         // spectrum/filters
    class LchFarrow    { ... };         // spectrum/lch_farrow
  }
  namespace stats {
    class StatisticsProcessor { ... };  // stats (бывший spectrum/statistics)
  }
  namespace signal_generators {
    class CwGenerator   { ... };
    class LfmGenerator  { ... };
    class FormSignal    { ... };
  }
  namespace heterodyne {
    class Dechirp       { ... };
    class NcoMixDown    { ... };
  }
  namespace linalg {
    class VectorAlgebra { ... };
    class Capon         { ... };
  }
  namespace radar {
    class RangeAngle    { ... };
    class FmCorrelator  { ... };
  }
  namespace strategies {
    class AntennaProcessor { ... };
  }
}
```

**Python bindings**: имена модулей не меняются (`dsp_core`, `dsp_spectrum`, ...).
Внутри `PYBIND11_MODULE` используем `py::class_<dsp::spectrum::FFTProcessor>`,
но в Python видно как `FFTProcessor` — Python API остаётся совместимым.

### 8.4 Doxygen (мета-репо собирает всё)

Каждый репо генерирует `.tag` файл:
```
core.tag, spectrum.tag, signal.tag, linalg.tag, radar.tag, strategies.tag
```
DSP/Doc/Doxygen/Doxyfile:
```ini
TAGFILES = \
  ../../core/docs/core.tag=https://dsp-gpu.github.io/core/api \
  ../../spectrum/docs/spectrum.tag=https://dsp-gpu.github.io/spectrum/api \
  ...
```

---

## 9. Что НЕ переносим (см. объединённый раздел 11)

> Раздел 9 старой версии удалён — дублировался с разделом 11 (#3 ревью).
> Актуальная таблица — ниже в разделе 11.

---

## 10. Принятые решения ✅

### 10.1 Версионирование
**`0.1.0` для всех репо на старте.** Единая версия — проще синхронизировать.  
При следующем релизе поднимаем все сразу. Тег в каждом репо: `v0.1.0`.

### 10.2 Python bindings — схема работы

**Принцип**: при `DSP_{REPO}_BUILD_PYTHON=ON` репо компилирует свой `.pyd`/`.so`.
`cmake --install` автоматически кладёт их в `DSP/Python/lib/`. Тесты берут оттуда.

```
dsp-gpu/core/              → сборка → dsp_core.pyd
dsp-gpu/spectrum/          → сборка → dsp_spectrum.pyd
dsp-gpu/stats/             → сборка → dsp_stats.pyd
dsp-gpu/signal_generators/ → сборка → dsp_signal_generators.pyd
dsp-gpu/heterodyne/        → сборка → dsp_heterodyne.pyd
dsp-gpu/linalg/            → сборка → dsp_linalg.pyd
dsp-gpu/radar/             → сборка → dsp_radar.pyd
dsp-gpu/strategies/        → сборка → dsp_strategies.pyd
...
         ↓  cmake --install build/local-dev
DSP/Python/lib/            ← ВСЕ .pyd/.so собраны здесь (в .gitignore!)
DSP/Python/spectrum/test_fft.py   ← импортирует из ../lib/
```

**В каждом репо** — папка `python/` с pybind11 кодом (собирается при
`DSP_{REPO}_BUILD_PYTHON=ON`, схема как в текущем GPUWorkLib):

```
core/python/
├── CMakeLists.txt         Python3 + pybind11 через pip + pybind11_add_module
├── dsp_core_module.cpp    PYBIND11_MODULE(dsp_core, m) { ... }
└── dsp_core.pyi           type stubs для IDE
```

**Как работает поиск pybind11** (как сейчас в `GPUWorkLib/python/CMakeLists.txt`):

```cmake
# {repo}/python/CMakeLists.txt

# 1. Python3 + NumPy — системный или venv
find_package(Python3 COMPONENTS Interpreter Development NumPy REQUIRED)

# 2. pybind11 — ставится через pip, CMake находит его через python -m pybind11
if(NOT pybind11_DIR)
  execute_process(
    COMMAND "${Python3_EXECUTABLE}" -m pybind11 --cmakedir
    OUTPUT_VARIABLE _pybind11_cmakedir
    OUTPUT_STRIP_TRAILING_WHITESPACE
    RESULT_VARIABLE pybind11_RESULT
  )
  if(NOT pybind11_RESULT EQUAL 0)
    message(FATAL_ERROR "pybind11 not found. Install: pip install pybind11")
  endif()
  string(REPLACE "\"" "" _pybind11_cmakedir "${_pybind11_cmakedir}")
  set(pybind11_DIR "${_pybind11_cmakedir}" CACHE PATH "pybind11 cmake dir")
endif()
find_package(pybind11 CONFIG REQUIRED)

# 3. Fallback для DSP_PYTHON_LIB_DIR (#8 ревью) — standalone сборка
if(NOT DEFINED DSP_PYTHON_LIB_DIR)
  set(DSP_PYTHON_LIB_DIR "${CMAKE_INSTALL_PREFIX}/python/lib"
      CACHE PATH "Destination for compiled Python bindings")
endif()

# 4. Собственно сборка модуля
pybind11_add_module(dsp_core dsp_core_module.cpp)
target_link_libraries(dsp_core PRIVATE DspCore::DspCore)

install(TARGETS dsp_core
  LIBRARY DESTINATION ${DSP_PYTHON_LIB_DIR}
  RUNTIME DESTINATION ${DSP_PYTHON_LIB_DIR}   # Windows .pyd
)
install(FILES dsp_core.pyi DESTINATION ${DSP_PYTHON_LIB_DIR})
```

**В DSP/CMakeLists.txt** — задаём централизованный путь установки:
```cmake
set(DSP_PYTHON_LIB_DIR "${CMAKE_SOURCE_DIR}/Python/lib"
    CACHE PATH "Destination for compiled Python bindings")
```

> 🔑 **Почему так**: схема уже отлажена в текущем `GPUWorkLib/python/CMakeLists.txt`
> (`find_package(Python3)` + `python -m pybind11 --cmakedir` + `find_package(pybind11)`).
> Не нужно FetchContent для pybind11 — достаточно `pip install pybind11`
> в CI и у разработчиков.

**В DSP/.gitignore**:
```gitignore
Python/lib/          # скомпилированные .pyd/.so — не коммитим
Python/__pycache__/
```

**Запуск тестов** — PYTHONPATH указывает на lib/:
```bash
# Linux / Debian (на работе):
PYTHONPATH=./Python/lib "python3" Python/spectrum/test_fft.py

# Windows:
$env:PYTHONPATH=".\Python\lib"
& "F:\Program Files (x86)\Python314\python.exe" Python/spectrum/test_fft.py
```

Или `Python/common/runner.py` добавляет `lib/` в `sys.path` автоматически — тогда просто:
```bash
"F:\Program Files (x86)\Python314\python.exe" Python/spectrum/test_fft.py
```

### 10.3 configGPU.json
**Хранится в `core/`** как шаблон по умолчанию (1 GPU, стандартные настройки).  
CMake копирует его в `build/` при конфигурации:
```cmake
configure_file(configGPU.json ${CMAKE_BINARY_DIR}/configGPU.json COPYONLY)
```
В будущем — отдельный файл `configCluster.json` под 10 GPU + 10 сетевых карт.

### 10.4 GitHub Actions CI
**Добавляем постепенно, после написания тасков.**  
Порядок:
1. Сначала — скелет + копирование кода + ручная проверка на GPU
2. Потом — таски в MemoryBank на CI
3. Добавляем `.github/workflows/build.yml` в каждый репо по очереди (начиная с `core`)

### 10.5 Видимость репо
**Публичные** — платить за GitHub Pro не нужно.
Код открытый, это нормально для DSP/GPU библиотеки.
В будущем можно перевести в приватные одной кнопкой (Settings → Change visibility).

> ⚠️ **Перед публикацией (R4 ревью)** — аудит всех config-файлов:
> - `configGPU.json` — не содержит ли локальных путей / hostname / IP
> - `CMakePresets.json` — нет ли абсолютных путей с `C:/Users/...`
> - Python test config — нет ли путей к конкретным рабочим машинам
> - Logs/ / Results/ — в `.gitignore` (не случайно закоммичены)

---

## 11. Что НЕ переносим / что переносим с оговорками

| Что | Куда / Причина |
|-----|----------------|
| Ветка `nvidia` / OpenCL-only вычисления | **НЕ переносим**. C++ пишем только под ROCm на Debian. Аналитические модели (Python) будем делать и под Windows, но без C++ OpenCL вычислений. |
| OpenCL `.cl` kernels | **НЕ переносим** (#10 ревью). Мы не считаем на OpenCL. OpenCL остаётся только для стыковки данных в GPU (через `core::opencl_backend`), без вычислительных ядер. |
| `src/main.cpp` (тестовый main) | Заменяется `examples/basic_usage.cpp` в каждом репо |
| `Doc_Addition/` | **Переносим** в `DSP/Doc/addition/` — потом посмотрим и почистим |
| `.claude/` | Только в GPUWorkLib (worktrees, локальные настройки) |
| Старые CMake заглушки | Переписываем с нуля под новую архитектуру |
| Git history | **НЕ переносим** (#9 ревью). При необходимости смотрим в исходный GPUWorkLib. |

---

## 12. Чеклист критических исправлений после ревью 2026-04-11

- [x] #1. `test_utils` → отдельная target `DspCore::TestUtils`
- [x] #2. CI preset переменные `DSP_*_TAG` читаются в `fetch_deps.cmake`
- [x] #3. Дубликат раздела 9 удалён, противоречие по `Doc_Addition/` устранено
- [x] #4. `find_dependency` показан явно в `DspSpectrumConfig.cmake.in`
- [x] #5. Обоснование зависимости `radar → spectrum + stats` (не плодим сущностей)
- [x] #6. Diamond dep решён через централизованное объявление в `DSP/CMakeLists.txt`
- [x] #7. `buildPresets` содержит все 5 конфигураций
- [x] #8. `DSP_PYTHON_LIB_DIR` имеет fallback для standalone сборки
- [x] #9. Политика git history задокументирована (без истории)
- [x] #10. OpenCL `.cl` kernels не переносим (только стыковка данных в GPU)
- [x] #11. `pybind11` — схема как в текущем GPUWorkLib (pip + `python -m pybind11 --cmakedir`)
- [x] R1. `FIND_PACKAGE_ARGS` использован в `fetch_deps.cmake`
- [x] R2. Фаза 0 — аудит зависимостей в `Doc/Architecture/`
- [x] R3. `Logs/` и `Results/` добавлены в структуру `DSP/`
- [x] R4. Чеклист аудита config-файлов перед публикацией
- [x] R5. `namespace dsp::` — в Фазе 3 сразу
- [x] A1. `stats` — отдельный репо (9 репо вместо 7)
- [x] A2. `signal` разделён на `signal_generators` + `heterodyne`

---

*Создан: 2026-04-11 | Автор: Кодо*
*Версия: v2 — после ревью Alex*
*Статус решений: ✅ все 18 пунктов ревью закрыты*
*Следующий шаг: **Фаза 0** — сгенерировать `Doc/Architecture/dependencies.md` и проанализировать реальный граф*
