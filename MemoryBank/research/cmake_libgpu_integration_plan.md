# 📦 План: Интеграция GPUWorkLib → LibGPU → Закрытый проект

## Context

**Проблема:** Сейчас ребята вручную копируют файлы из GPUWorkLib в закрытый репозиторий на локальном сервере. Это ненадёжно, теряется версионирование, нет контроля качества, Alex не может использовать Кодо для работы.

**Цель:** Создать промежуточный репозиторий `E:\C++\LibGPU` с валидированным кодом, который закрытый проект подключает через CMake. Обеспечить изоляцию между открытой разработкой и закрытым контуром.

**Результат:** 3 режима сборки (Debug/Release/LocalDev), автоматизированная синхронизация, правильная CMake интеграция, документированная защита закрытого контура.

### Решения Alex'а:
- **Платформа закрытого проекта:** Linux + AMD GPU (ROCm)
- **LibGPU:** Отдельный git-репозиторий на локальном сервере (LibGPU — только название директории/repo, проект остаётся GPUWorkLib)
- **Модули:** Все 12 модулей
- **Python:** Да, Python bindings включены
- **Тесты, документация, main.cpp:** Включены в LibGPU
- **Naming:** Проект НЕ переименовывается. `project(GPUWorkLib)`, namespace `GPUWorkLib::`, export `GPUWorkLibTargets` — всё остаётся как есть. LibGPU = "конверт" для передачи кода
- **ROCm:** Если нет ROCm — просто выходим из сборки. Если есть ROCm то и OpenCL есть
- **configGPU.json:** Копировать из LibGPU, на сервере настроить под свои GPU

---

## 🔒 Обоснование решения: Безопасность и защита закрытого контура

### Почему промежуточный репозиторий, а не прямое подключение?

**1. Изоляция разработки от закрытого контура**

Рабочий репозиторий GPUWorkLib содержит артефакты разработки, которые НЕ ДОЛЖНЫ попадать в закрытый контур:
- `.claude/` — конфигурация AI-ассистента, MCP-серверы, API-ключи
- `MemoryBank/` — внутренние задачи, планы, черновики, сессии работы с AI
- `Results/` — промежуточные результаты профилирования и тестов
- `Logs/` — логи отладки с GPU (могут содержать данные обработки)
- `.mcp.json`, `api_keys.json` — секреты и токены
- `.vscode/`, `.idea/`, `.cursor/` — персональные настройки IDE
- `scripts/` — вспомогательные скрипты разработки (run_agent_tests.py)
- `MemoryBank/sessions/` — полная история взаимодействия с AI

Без промежуточного репозитория любой из этих файлов может случайно попасть в закрытый контур через `git add -A` или неаккуратное копирование.

**2. Контроль качества (Quality Gate)**

LibGPU — это контрольная точка (gate):
- В LibGPU попадает **только код, прошедший валидацию и тесты** в GPUWorkLib
- Alex лично контролирует момент синхронизации (запуск скрипта)
- Git tags в LibGPU фиксируют версии, можно откатиться
- Закрытый проект привязан к конкретному тегу/коммиту через git submodule

**3. Защита от утечки данных закрытого проекта**

- GPUWorkLib находится на GitHub (публичный/приватный) — данные закрытого проекта не должны туда попасть
- LibGPU — однонаправленный поток: GPUWorkLib → LibGPU → Закрытый проект
- Обратный поток (Закрытый проект → LibGPU → GPUWorkLib) НЕ предусмотрен
- Закрытый проект подключает LibGPU как read-only submodule — не может менять код библиотеки
- Изменения в библиотеку идут ТОЛЬКО через GPUWorkLib → sync → LibGPU

**4. Аудит и прослеживаемость**

- Каждая синхронизация — git commit в LibGPU с сообщением `sync from GPUWorkLib vX.Y.Z`
- Git tags позволяют точно знать какая версия библиотеки используется в закрытом проекте
- `git log` в LibGPU — полная история всех обновлений
- При инциденте: можно точно определить какая версия была в production

**5. Принцип минимальных привилегий**

- Разработчики закрытого проекта видят ТОЛЬКО код LibGPU, не видят внутреннюю кухню GPUWorkLib
- Нет доступа к MemoryBank (задачи, планы, исследования)
- Нет доступа к AI-инструментам и их конфигурации
- Нет доступа к промежуточным результатам и логам разработки

**6. Отсутствие прямого сетевого доступа**

- LibGPU на локальном сервере — нет связи с интернетом
- GPUWorkLib на GitHub — нет связи с локальным сервером
- Синхронизация — ручная (Alex запускает скрипт) или через защищённый канал
- Невозможна автоматическая утечка: нет CI/CD цепочки GPUWorkLib → LibGPU

---

## 🏗️ Архитектура решения

```
                              ЗАЩИЩЁННЫЙ КОНТУР (локальный сервер)
                              ┌─────────────────────────────────────────────┐
┌──────────────────┐          │  ┌──────────────────┐    ┌──────────────────┐│
│   GPUWorkLib     │  sync    │  │     LibGPU       │    │  Закрытый проект ││
│  (рабочий репо)  │ ────────>│  │ (git repo, tags) │<───│  (git submodule) ││
│                  │  ручная  │  │                  │    │                  ││
│  GitHub/приватный│  операция│  │  Локальный сервер│    │  Локальный сервер││
└──────────────────┘          │  └──────────────────┘    └──────────────────┘│
       ↑                      └─────────────────────────────────────────────┘
  Alex + Кодо                         ↑
  разработка                    Однонаправленный поток
  тесты, CI                    Обратная связь ЗАПРЕЩЕНА
```

**Ключевые принципы:**
1. **Однонаправленность**: GPUWorkLib → LibGPU → Закрытый проект (никогда обратно)
2. **Ручной контроль**: Alex лично запускает синхронизацию
3. **Версионирование**: git tags в LibGPU (v1.1.0, v1.2.0...)
4. **Изоляция**: артефакты разработки не покидают GPUWorkLib

---

## 📂 Структура LibGPU (что попадает)

```
LibGPU/
├── CMakeLists.txt              # Облегчённый корневой CMake (library + optional main)
├── CMakePresets.json            # linux-debug / linux-release / localdev
├── GPUWorkLibConfig.cmake.in        # Для find_package() (фаза 2)
├── VERSION                      # 1.1.0
├── .gitignore                   # build/, Logs/, __pycache__
│
├── cmake/                       # CMake модули
│   ├── platform-detection.cmake
│   ├── gpu-config.cmake
│   ├── compiler-options.cmake
│   └── dependencies.cmake
│
├── DrvGPU/                      # Базовый GPU драйвер (ПОЛНОСТЬЮ включая тесты)
│   ├── CMakeLists.txt
│   ├── backends/
│   │   ├── opencl/
│   │   ├── rocm/
│   │   └── hybrid/
│   ├── include/
│   ├── interface/
│   ├── src/
│   ├── common/
│   ├── config/
│   ├── logger/
│   ├── memory/
│   ├── services/
│   └── tests/                   # ✅ C++ тесты DrvGPU
│
├── modules/                     # Все 12 модулей (ПОЛНОСТЬЮ включая тесты)
│   ├── CMakeLists.txt
│   ├── test_utils/              # ✅ Общие утилиты тестирования
│   ├── fft_func/
│   │   ├── CMakeLists.txt
│   │   ├── include/
│   │   ├── kernels/
│   │   ├── src/
│   │   └── tests/               # ✅ C++ тесты модуля
│   ├── filters/                 # (та же структура)
│   ├── statistics/
│   ├── signal_generators/
│   ├── heterodyne/
│   ├── lch_farrow/
│   ├── vector_algebra/
│   ├── fm_correlator/
│   ├── strategies/
│   ├── capon/
│   └── range_angle/
│
├── src/                         # ✅ Standalone executable
│   ├── CMakeLists.txt
│   └── main.cpp
│
├── python/                      # ✅ Python bindings (pybind11)
│   ├── CMakeLists.txt
│   ├── gpu_worklib_bindings.cpp
│   └── py_*.hpp
│
├── Python_test/                 # ✅ Python тесты
│   ├── conftest.py
│   ├── common/
│   ├── fft_func/
│   ├── filters/
│   ├── statistics/
│   ├── ... (все модули)
│   └── integration/
│
├── Doc/                         # ✅ Документация проекта
│   ├── Architecture/
│   ├── DrvGPU/
│   ├── Modules/
│   ├── Python/
│   ├── Python_test/
│   ├── Doxygen/                 # ✅ Графики, таблицы тестов, примеры
│   ├── Full_Reference.md
│   ├── INDEX.md
│   └── Quick_Reference.md
│
├── Doc_Addition/                # ✅ Дополнительная документация
│   ├── Info_*
│   ├── PLAN/
│   └── Mermaid_DarkTheme_Guide.md
│
├── include/                     # Общие headers
├── third_party/                 # plog, nlohmann_json (headers)
├── config/                      # tests_order.txt и др.
├── configGPU.json               # Конфигурация GPU
├── requirements.txt             # Python зависимости
├── run.sh / run.bat             # Скрипты запуска
└── README.md
```

### ❌ НЕ попадает в LibGPU (артефакты разработки):

| Что исключено | Причина |
|--------------|---------|
| `MemoryBank/` | Внутренние задачи, планы, сессии AI — конфиденциальная информация разработки |
| `.claude/` | Конфигурация AI-ассистента, MCP-серверы, API-ключи |
| `.mcp.json` | Секреты и токены MCP-серверов |
| `api_keys.json` | API-ключи (если есть) |
| `Results/` | Промежуточные результаты профилирования и тестов |
| `Logs/` | Логи отладки GPU (могут содержать обрабатываемые данные) |
| `.vscode/`, `.idea/`, `.cursor/` | Персональные настройки IDE |
| `scripts/` | Внутренние скрипты разработки (run_agent_tests.py) |
| `CLAUDE.md` | Конфигурация AI-ассистента (содержит внутреннюю информацию) |
| `Doc_Addition/PLAN/` | Внутренние планы рефакторинга (Ref01, Ref02...) — не для закрытого контура |
| `build/` | Артефакты сборки |
| `CMakeUserPresets.json` | Персональные настройки CMake |
| `setup_linux_claude.sh` | Скрипт настройки AI-окружения |
| `*.pyc`, `__pycache__/` | Python кеш |

---

## 🔧 Детальные шаги реализации

### Шаг 1: Инициализация репозитория LibGPU

```bash
# Создать директорию и git репозиторий
mkdir -p /path/to/LibGPU
cd /path/to/LibGPU
git init
git checkout -b main

# Создать базовые файлы
echo "1.1.0" > VERSION
```

Создать `.gitignore`:
```
build*/
Logs/
__pycache__/
*.pyc
*.pyo
*.o
*.a
*.so
*.lib
*.dll
*.exe
CMakeUserPresets.json
compile_commands.json
.cache/
```

### Шаг 2: Скрипт синхронизации `sync_to_libgpu.py`

**Файл:** `E:\C++\GPUWorkLib\sync_to_libgpu.py` (~250 строк Python)

**Интерфейс:**
```
usage: sync_to_libgpu.py [-h] [--src SRC] [--dst DST] [--dry-run]
                          [--clean] [--commit] [--tag TAG]
                          [--exclude MOD1,MOD2]

Синхронизация GPUWorkLib → LibGPU (промежуточный репозиторий)

arguments:
  --src PATH        Путь к GPUWorkLib (default: текущая директория)
  --dst PATH        Путь к LibGPU (default: ../LibGPU)
  --dry-run         Показать что будет скопировано без выполнения
  --clean           Удалить файлы в dst которых нет в src (whitelist)
  --commit          git add + commit в LibGPU после синхронизации
  --tag TAG         Создать git tag в LibGPU (например: --tag v1.2.0)
  --exclude MODS    Исключить модули через запятую (например: --exclude capon,range_angle)
```

**Whitelist — что копируется:**
```python
# Директории для синхронизации (ВКЛЮЧАЯ все поддиректории)
SYNC_DIRS = [
    "cmake/",
    "config/",
    "DrvGPU/",              # ПОЛНОСТЬЮ включая tests/
    "modules/",             # ПОЛНОСТЬЮ включая */tests/ и test_utils/
    "src/",                 # main.cpp
    "python/",              # Python bindings
    "Python_test/",         # Python тесты
    "Doc/",                 # Документация
    "Doc_Addition/",        # Доп. документация
    "include/",             # Общие headers
    "third_party/",         # Зависимости (headers)
]

# Отдельные файлы
SYNC_FILES = [
    "configGPU.json",
    "requirements.txt",
    "run.sh",
    "run.bat",
    "README.md",
    ".clangd",
    ".gitattributes",
]

# Что НИКОГДА не копируется (исключения из whitelist)
EXCLUDE_PATTERNS = [
    "MemoryBank/",           # Внутренние задачи и планы
    "Doc_Addition/PLAN/",    # Внутренние планы рефакторинга
    ".claude/",              # AI конфигурация
    ".mcp.json",             # MCP секреты
    "api_keys.json",         # API ключи
    "Results/",              # Промежуточные результаты
    "Logs/",                 # Логи GPU
    ".vscode/",              # IDE
    ".idea/",                # IDE
    ".cursor/",              # IDE
    "scripts/",              # Внутренние скрипты
    "CLAUDE.md",             # AI конфигурация
    "build*/",               # Артефакты сборки
    "CMakeUserPresets.json", # Персональные настройки
    "setup_linux_claude.sh", # AI setup
    "__pycache__/",          # Python кеш
    "*.pyc",
    ".git/",                 # Git данные исходного репо
]
```

**Алгоритм работы:**
1. Проверить что `--src` содержит `CMakeLists.txt` (это GPUWorkLib)
2. Проверить что `--dst` существует и содержит `.git/` (это LibGPU)
3. Сканировать whitelist директорий и файлов в `--src`
4. Для каждого файла:
   - Проверить что не попадает в EXCLUDE_PATTERNS
   - Сравнить timestamp/hash с файлом в `--dst`
   - Если новее или не существует → копировать
5. При `--clean`: удалить файлы в `--dst` которых нет в `--src` (кроме LibGPU-specific: CMakeLists.txt, CMakePresets.json, VERSION, GPUWorkLibConfig.cmake.in)
6. Вывести статистику:
   ```
   ✅ Синхронизация завершена:
     Добавлено:    15 файлов
     Обновлено:    8 файлов
     Удалено:      2 файла (--clean)
     Без изменений: 342 файла
   ```
7. При `--commit`:
   ```bash
   cd /path/to/LibGPU
   git add -A
   git commit -m "sync from GPUWorkLib $(date +%Y-%m-%d)"
   ```
8. При `--tag`:
   ```bash
   git tag v1.2.0
   ```

---

### Шаг 3: Корневой CMakeLists.txt для LibGPU (ДЕТАЛЬНО)

**Файл:** `LibGPU/CMakeLists.txt`

Этот файл НЕ копируется из GPUWorkLib — он создаётся отдельно и живёт только в LibGPU. Он должен быть совместим как с автономной сборкой, так и с подключением через `add_subdirectory()` из закрытого проекта.

```cmake
# ═══════════════════════════════════════════════════════════════════
# LibGPU — GPU Signal Processing Library
# Промежуточный репозиторий для интеграции в закрытые проекты
# ═══════════════════════════════════════════════════════════════════
cmake_minimum_required(VERSION 3.20)

# Читаем версию из файла VERSION
file(READ "${CMAKE_CURRENT_SOURCE_DIR}/VERSION" LIBGPU_VERSION)
string(STRIP "${LIBGPU_VERSION}" LIBGPU_VERSION)

# project() вызываем ТОЛЬКО если это корневой проект
# Если подключен через add_subdirectory — project() уже вызван родителем
# ВАЖНО: проект называется GPUWorkLib, НЕ LibGPU!
# LibGPU — только название директории/репозитория ("конверт" для передачи кода)
if(CMAKE_CURRENT_SOURCE_DIR STREQUAL CMAKE_SOURCE_DIR)
    project(GPUWorkLib VERSION ${LIBGPU_VERSION} LANGUAGES CXX)
    set(LIBGPU_IS_TOP_LEVEL TRUE)
else()
    set(LIBGPU_IS_TOP_LEVEL FALSE)
endif()

# ═══════════════════════════════════════════════════════════════════
# ОПЦИИ СБОРКИ
# Закрытый проект может задать эти переменные ПЕРЕД add_subdirectory()
# Пример:
#   set(LIBGPU_BUILD_TESTS OFF CACHE BOOL "" FORCE)
#   set(LIBGPU_BUILD_MAIN OFF CACHE BOOL "" FORCE)
#   add_subdirectory(libs/LibGPU)
# ═══════════════════════════════════════════════════════════════════

# --- Общие опции ---
# LIBGPU_BUILD_TESTS: собирать C++ тесты модулей
#   ON  — тесты компилируются, вызываются из main.cpp
#   OFF — тесты пропускаются, уменьшается время сборки
option(LIBGPU_BUILD_TESTS   "Build LibGPU C++ tests"        ${LIBGPU_IS_TOP_LEVEL})

# LIBGPU_BUILD_MAIN: собирать standalone executable (src/main.cpp)
#   ON  — создаётся исполняемый файл LibGPU/GPUWorkLib
#   OFF — только библиотеки, без exe (для подключения как зависимость)
option(LIBGPU_BUILD_MAIN    "Build standalone executable"    ${LIBGPU_IS_TOP_LEVEL})

# LIBGPU_BUILD_PYTHON: собирать Python bindings (pybind11)
#   ON  — создаётся gpuworklib.so/.pyd
#   OFF — Python не требуется
option(LIBGPU_BUILD_PYTHON  "Build Python bindings"          OFF)

# --- Выборочная сборка модулей ---
# Каждый модуль можно отключить отдельно.
# Зависимости проверяются автоматически (если отключён lch_farrow,
# signal_generators тоже не соберётся)
option(LIBGPU_BUILD_FFT         "Build fft_func module"          ON)
option(LIBGPU_BUILD_FILTERS     "Build filters module"           ON)
option(LIBGPU_BUILD_STATISTICS  "Build statistics module"        ON)
option(LIBGPU_BUILD_GENERATORS  "Build signal_generators module" ON)
option(LIBGPU_BUILD_HETERODYNE  "Build heterodyne module"        ON)
option(LIBGPU_BUILD_LCH_FARROW "Build lch_farrow module"        ON)
option(LIBGPU_BUILD_VECTOR_ALG "Build vector_algebra module"     ON)
option(LIBGPU_BUILD_FM_CORR    "Build fm_correlator module"      ON)
option(LIBGPU_BUILD_STRATEGIES "Build strategies module"         ON)
option(LIBGPU_BUILD_CAPON      "Build capon module"              ON)
option(LIBGPU_BUILD_RANGE_ANGLE "Build range_angle module"       ON)

# ═══════════════════════════════════════════════════════════════════
# ПЛАТФОРМА И ЗАВИСИМОСТИ
# cmake/ директория содержит helper-модули для определения платформы,
# GPU конфигурации, флагов компилятора и поиска зависимостей
# ═══════════════════════════════════════════════════════════════════

# platform-detection.cmake:
#   Определяет IS_WINDOWS, IS_LINUX, PLATFORM_NAME
#   Устанавливает C++17, CMAKE_BUILD_TYPE (default: Release)
#   Включает CMAKE_EXPORT_COMPILE_COMMANDS
include(${CMAKE_CURRENT_SOURCE_DIR}/cmake/platform-detection.cmake)

# gpu-config.cmake:
#   Опции ENABLE_CUDA, ENABLE_OPENCL, ENABLE_ROCM
#   Автоопределение GPU платформы
#   Поиск CUDA, HIP, hiprtc
include(${CMAKE_CURRENT_SOURCE_DIR}/cmake/gpu-config.cmake)

# compiler-options.cmake:
#   MSVC: /MD, /O2, /arch:AVX2, /W4
#   GCC/Clang: -O3, -march=native, -Wall
#   Debug: -g -ggdb3 / /Zi /RTC1
include(${CMAKE_CURRENT_SOURCE_DIR}/cmake/compiler-options.cmake)

# dependencies.cmake:
#   OpenCL (System32, CUDA SDK, системные пути)
#   clFFT (локальный /clFFT/, AMD APP SDK)
#   nlohmann_json (vcpkg, система)
#   ROCm: hip, hipfft, hiprtc
#   rocprim (для statistics)
include(${CMAKE_CURRENT_SOURCE_DIR}/cmake/dependencies.cmake)

# ═══════════════════════════════════════════════════════════════════
# CORE: DrvGPU (всегда собирается)
# Базовый GPU драйвер: backends, memory management, profiler, logger
# Targets: drvgpu (DrvGPU::drvgpu)
# ═══════════════════════════════════════════════════════════════════
add_subdirectory(DrvGPU)

# ═══════════════════════════════════════════════════════════════════
# МОДУЛИ (в порядке зависимостей!)
# Порядок важен: signal_generators зависит от lch_farrow,
# heterodyne зависит от signal_generators + fft_func,
# capon зависит от vector_algebra
# ═══════════════════════════════════════════════════════════════════

# fft_func: FFT процессор (hipFFT)
# Target: fft_func (GPUWorkLib::fft_func)
# Зависимости: drvgpu, hip::host, hipfft
if(LIBGPU_BUILD_FFT AND ROCM_ENABLED)
    add_subdirectory(modules/fft_func)
endif()

# lch_farrow: дробная задержка (Lagrange 48x5)
# Target: lch_farrow (GPUWorkLib::lch_farrow)
# Зависимости: drvgpu, hip::host
if(LIBGPU_BUILD_LCH_FARROW AND ROCM_ENABLED)
    add_subdirectory(modules/lch_farrow)
endif()

# signal_generators: CW, LFM, Noise, FormSignal, Script
# Target: signal_generators (GPUWorkLib::signal_generators)
# Зависимости: drvgpu, lch_farrow, hip::host
if(LIBGPU_BUILD_GENERATORS AND TARGET lch_farrow AND ROCM_ENABLED)
    add_subdirectory(modules/signal_generators)
endif()

# filters: FIR, IIR, MovingAverage, Kalman, Kaufman
# Target: filters (GPUWorkLib::filters)
# Зависимости: drvgpu, hip::host
if(LIBGPU_BUILD_FILTERS AND ROCM_ENABLED)
    add_subdirectory(modules/filters)
endif()

# heterodyne: LFM Dechirp
# Target: heterodyne (GPUWorkLib::heterodyne)
# Зависимости: drvgpu, signal_generators, fft_func, hip::host
if(LIBGPU_BUILD_HETERODYNE AND TARGET signal_generators AND TARGET fft_func)
    add_subdirectory(modules/heterodyne)
endif()

# statistics: mean, std, variance, median (Welford + radix sort)
# Target: statistics (GPUWorkLib::statistics)
# Зависимости: drvgpu, hip::host, rocprim
# Особенность: содержит .hip файлы, нужен enable_language(HIP)
if(LIBGPU_BUILD_STATISTICS AND ROCM_ENABLED)
    add_subdirectory(modules/statistics)
endif()

# vector_algebra: Cholesky инверсия (rocsolver)
# Target: vector_algebra (GPUWorkLib::vector_algebra)
# Зависимости: drvgpu, hip::host, rocsolver
if(LIBGPU_BUILD_VECTOR_ALG AND ROCM_ENABLED)
    add_subdirectory(modules/vector_algebra)
endif()

# fm_correlator: M-sequence LFSR + freq-domain корреляция
# Target: fm_correlator (GPUWorkLib::fm_correlator)
# Зависимости: drvgpu, hip::host, hipfft
if(LIBGPU_BUILD_FM_CORR AND ROCM_ENABLED)
    add_subdirectory(modules/fm_correlator)
endif()

# strategies: цифровое ДН (CGEMM beamforming)
# Target: strategies (GPUWorkLib::strategies)
# Зависимости: drvgpu, hip::host
if(LIBGPU_BUILD_STRATEGIES AND ROCM_ENABLED)
    add_subdirectory(modules/strategies)
endif()

# capon: MVDR Capon beamformer
# Target: capon (GPUWorkLib::capon)
# Зависимости: drvgpu, vector_algebra, hip::host
if(LIBGPU_BUILD_CAPON AND TARGET vector_algebra AND ROCM_ENABLED)
    add_subdirectory(modules/capon)
endif()

# range_angle: 3D обработка (Dechirp → range FFT → 2D beam FFT)
# Target: range_angle (GPUWorkLib::range_angle)
# Зависимости: drvgpu, hip::host
if(LIBGPU_BUILD_RANGE_ANGLE AND ROCM_ENABLED)
    add_subdirectory(modules/range_angle)
endif()

# ═══════════════════════════════════════════════════════════════════
# STANDALONE EXECUTABLE (опционально)
# src/main.cpp — запускает все тесты модулей через all_test.hpp
# Используется для LocalDev режима и автономного тестирования
# ═══════════════════════════════════════════════════════════════════
if(LIBGPU_BUILD_MAIN)
    add_subdirectory(src)
endif()

# ═══════════════════════════════════════════════════════════════════
# PYTHON BINDINGS (опционально)
# python/gpu_worklib_bindings.cpp + py_*.hpp → gpuworklib.so/.pyd
# Требует: Python3, pybind11, NumPy
# ═══════════════════════════════════════════════════════════════════
if(LIBGPU_BUILD_PYTHON)
    add_subdirectory(python)
endif()

# ═══════════════════════════════════════════════════════════════════
# EXPORT TARGETS (для find_package в фазе 2)
# Позволяет другим проектам использовать:
#   find_package(GPUWorkLib 1.1 REQUIRED)
#   target_link_libraries(app PRIVATE GPUWorkLib::fft_func)
#
# ВАЖНО: Export name = GPUWorkLibTargets (совпадает с модулями!)
# Namespace = GPUWorkLib:: (единый для всех, включая DrvGPU)
# ═══════════════════════════════════════════════════════════════════
if(LIBGPU_IS_TOP_LEVEL)
    include(CMakePackageConfigHelpers)

    # Экспорт всех targets с namespace GPUWorkLib::
    # Имя GPUWorkLibTargets совпадает с install(EXPORT) в модулях
    install(EXPORT GPUWorkLibTargets
        NAMESPACE GPUWorkLib::
        DESTINATION lib/cmake/GPUWorkLib
    )

    # Файл версии для find_package(GPUWorkLib 1.1 REQUIRED)
    write_basic_package_version_file(
        ${CMAKE_CURRENT_BINARY_DIR}/GPUWorkLibConfigVersion.cmake
        VERSION ${LIBGPU_VERSION}
        COMPATIBILITY SameMajorVersion
    )

    # Конфигурационный файл
    configure_package_config_file(
        ${CMAKE_CURRENT_SOURCE_DIR}/GPUWorkLibConfig.cmake.in
        ${CMAKE_CURRENT_BINARY_DIR}/GPUWorkLibConfig.cmake
        INSTALL_DESTINATION lib/cmake/GPUWorkLib
    )

    install(FILES
        ${CMAKE_CURRENT_BINARY_DIR}/GPUWorkLibConfig.cmake
        ${CMAKE_CURRENT_BINARY_DIR}/GPUWorkLibConfigVersion.cmake
        DESTINATION lib/cmake/GPUWorkLib
    )
endif()

# ═══════════════════════════════════════════════════════════════════
# ИНФОРМАЦИЯ О СБОРКЕ
# ═══════════════════════════════════════════════════════════════════
message(STATUS "")
message(STATUS "═══════════════════════════════════════")
message(STATUS "LibGPU v${LIBGPU_VERSION} Configuration:")
message(STATUS "  Platform:    ${PLATFORM_NAME}")
message(STATUS "  Build type:  ${CMAKE_BUILD_TYPE}")
message(STATUS "  ROCm:        ${ROCM_ENABLED}")
message(STATUS "  OpenCL:      ${OPENCL_ENABLED}")
message(STATUS "  Build main:  ${LIBGPU_BUILD_MAIN}")
message(STATUS "  Build tests: ${LIBGPU_BUILD_TESTS}")
message(STATUS "  Build Python:${LIBGPU_BUILD_PYTHON}")
message(STATUS "  Top-level:   ${LIBGPU_IS_TOP_LEVEL}")
message(STATUS "═══════════════════════════════════════")
message(STATUS "")
```

**Ключевые решения в CMake:**

1. **`LIBGPU_IS_TOP_LEVEL`**: Определяется автоматически. Если LibGPU — корневой проект (самостоятельная сборка), то `project()` вызывается, тесты и main включены по умолчанию. Если подключен через `add_subdirectory` — `project()` НЕ вызывается, тесты и main выключены по умолчанию.

2. **Порядок модулей**: Строго по зависимостям. `if(TARGET lch_farrow)` гарантирует что signal_generators не соберётся без lch_farrow.

3. **Совместимость**: CMake файлы модулей из GPUWorkLib используются БЕЗ ИЗМЕНЕНИЙ. Они уже написаны для `add_subdirectory` и используют `if(TARGET ...)` для условной линковки.

---

### Шаг 4: CMakePresets.json для LibGPU (ДЕТАЛЬНО)

**Файл:** `LibGPU/CMakePresets.json`

```json
{
    "version": 6,
    "cmakeMinimumRequired": {
        "major": 3,
        "minor": 20,
        "patch": 0
    },
    "configurePresets": [
        {
            "name": "debug",
            "displayName": "Debug — общая отладка",
            "description": "Все модули с debug symbols. Для закрытого проекта и отладки",
            "generator": "Ninja",
            "binaryDir": "${sourceDir}/build-debug",
            "cacheVariables": {
                "CMAKE_BUILD_TYPE": "Debug",
                "ENABLE_ROCM": "ON",
                "ENABLE_OPENCL": "ON",
                "LIBGPU_BUILD_TESTS": "OFF",
                "LIBGPU_BUILD_MAIN": "OFF",
                "LIBGPU_BUILD_PYTHON": "OFF"
            }
        },
        {
            "name": "release",
            "displayName": "Release — продакшен",
            "description": "Оптимизация -O3, минимальный код. Для production сборки закрытого проекта",
            "generator": "Ninja",
            "binaryDir": "${sourceDir}/build-release",
            "cacheVariables": {
                "CMAKE_BUILD_TYPE": "Release",
                "ENABLE_ROCM": "ON",
                "ENABLE_OPENCL": "ON",
                "LIBGPU_BUILD_TESTS": "OFF",
                "LIBGPU_BUILD_MAIN": "OFF",
                "LIBGPU_BUILD_PYTHON": "OFF"
            }
        },
        {
            "name": "local-debug",
            "displayName": "Local Debug — полная сборка для локальной отладки",
            "description": "Все модули + тесты + main + Python. Для автономной отладки",
            "generator": "Ninja",
            "binaryDir": "${sourceDir}/build-local-debug",
            "cacheVariables": {
                "CMAKE_BUILD_TYPE": "Debug",
                "ENABLE_ROCM": "ON",
                "ENABLE_OPENCL": "ON",
                "LIBGPU_BUILD_TESTS": "ON",
                "LIBGPU_BUILD_MAIN": "ON",
                "LIBGPU_BUILD_PYTHON": "ON"
            }
        }
    ],
    "buildPresets": [
        {
            "name": "debug",
            "configurePreset": "debug",
            "jobs": 8
        },
        {
            "name": "release",
            "configurePreset": "release",
            "jobs": 8
        },
        {
            "name": "local-debug",
            "configurePreset": "local-debug",
            "jobs": 8
        }
    ]
}
```

**Как закрытый проект использует presets:**

Закрытый проект НЕ использует presets LibGPU напрямую. Presets нужны только для автономной сборки. Когда подключен через `add_subdirectory()`, `CMAKE_BUILD_TYPE` наследуется от родительского проекта автоматически.

**Генератор Ninja** выбран вместо Unix Makefiles — значительно быстрее для параллельной сборки.

---

### Шаг 5: GPUWorkLibConfig.cmake.in

**Файл:** `LibGPU/GPUWorkLibConfig.cmake.in`

```cmake
# GPUWorkLib CMake Config File
# Используется find_package(GPUWorkLib) в фазе 2
@PACKAGE_INIT@

include(CMakeFindDependencyMacro)

# Транзитивные зависимости — без них линковка упадёт!
find_dependency(Threads)
if(@OPENCL_ENABLED@)
  find_dependency(OpenCL)
endif()
if(@ROCM_ENABLED@)
  find_dependency(hip)
  find_dependency(hipfft)
  find_dependency(rocblas)
  find_dependency(rocsolver)
endif()

include("${CMAKE_CURRENT_LIST_DIR}/GPUWorkLibTargets.cmake")
check_required_components(GPUWorkLib)

# Сообщение для разработчика
message(STATUS "Found GPUWorkLib @PROJECT_VERSION@ at ${CMAKE_CURRENT_LIST_DIR}")
```

---

### Шаг 6: Интеграция в закрытый проект (ДЕТАЛЬНО)

**Рекомендуемый способ: git submodule + add_subdirectory**

#### 6.1 Добавление LibGPU как submodule

```bash
# В корне закрытого проекта:
cd /path/to/ClosedProject

# Добавить LibGPU как submodule
# URL — адрес git-репозитория LibGPU на локальном сервере
git submodule add ssh://server/repos/LibGPU.git libs/LibGPU

# Зафиксировать конкретную версию (тег)
cd libs/LibGPU
git checkout v1.1.0
cd ../..
git add libs/LibGPU
git commit -m "pin LibGPU to v1.1.0"
```

#### 6.2 CMakeLists.txt закрытого проекта

```cmake
# ClosedProject/CMakeLists.txt
cmake_minimum_required(VERSION 3.20)
project(ClosedProject LANGUAGES CXX)

# ═══════════════════════════════════════════════════
# Настройки LibGPU — задаются ПЕРЕД add_subdirectory
# ═══════════════════════════════════════════════════

# Режим Debug/Release наследуется автоматически от CMAKE_BUILD_TYPE
# Явно отключаем то что не нужно в закрытом проекте:
set(LIBGPU_BUILD_TESTS OFF CACHE BOOL "LibGPU: skip tests" FORCE)
set(LIBGPU_BUILD_MAIN  OFF CACHE BOOL "LibGPU: skip standalone exe" FORCE)
set(LIBGPU_BUILD_PYTHON OFF CACHE BOOL "LibGPU: skip Python" FORCE)

# Опционально: отключить ненужные модули
# set(LIBGPU_BUILD_RANGE_ANGLE OFF CACHE BOOL "" FORCE)

# ═══════════════════════════════════════════════════
# Подключение LibGPU
# ═══════════════════════════════════════════════════
# Вариант 1: submodule в libs/
add_subdirectory(libs/LibGPU)

# Вариант 2: внешняя директория (если LibGPU рядом)
# add_subdirectory(${CMAKE_SOURCE_DIR}/../LibGPU ${CMAKE_BINARY_DIR}/libgpu-build)

# Вариант 3: задать путь через переменную
# add_subdirectory(${LIBGPU_DIR} ${CMAKE_BINARY_DIR}/libgpu-build)

# ═══════════════════════════════════════════════════
# Использование библиотек LibGPU
# ═══════════════════════════════════════════════════
add_executable(closed_app src/main.cpp)

# Линковка с нужными модулями
# Каждый модуль автоматически подтягивает свои зависимости
# (drvgpu, hip::host, hipfft и т.д.)
# ALIAS'ы из CMakeLists.txt модулей — используются как есть, без изменений:
# DrvGPU::drvgpu — ядро, GPUWorkLib::* — модули
target_link_libraries(closed_app PRIVATE
    DrvGPU::drvgpu              # Базовый GPU драйвер (ALIAS из DrvGPU/CMakeLists.txt)
    GPUWorkLib::fft_func        # FFT процессор
    GPUWorkLib::signal_generators  # Генераторы сигналов
    GPUWorkLib::filters         # Фильтры
    GPUWorkLib::heterodyne      # Гетеродин
    GPUWorkLib::statistics      # Статистика
    GPUWorkLib::capon           # Capon MVDR
    GPUWorkLib::strategies      # ДН
    GPUWorkLib::range_angle     # 3D обработка
)
# При фазе 2 (find_package) все targets будут GPUWorkLib::*
# (включая GPUWorkLib::drvgpu вместо DrvGPU::drvgpu)
```

#### 6.3 Как происходит сборка в закрытом проекте

```bash
# Debug режим (отладка):
cd /path/to/ClosedProject
mkdir build-debug && cd build-debug
cmake -DCMAKE_BUILD_TYPE=Debug ..
cmake --build . -j8

# Release режим (продакшен):
mkdir build-release && cd build-release
cmake -DCMAKE_BUILD_TYPE=Release ..
cmake --build . -j8
```

**Что происходит при cmake configure:**
1. CMake входит в `add_subdirectory(libs/LibGPU)`
2. LibGPU определяет `LIBGPU_IS_TOP_LEVEL=FALSE` (не вызывает `project()`)
3. Читает опции (`LIBGPU_BUILD_TESTS=OFF`, `LIBGPU_BUILD_MAIN=OFF`)
4. Подключает `cmake/platform-detection.cmake` → определяет Linux
5. Подключает `cmake/gpu-config.cmake` → находит ROCm/HIP
6. Подключает `cmake/dependencies.cmake` → находит OpenCL, hipfft, rocprim
7. Собирает `DrvGPU` → target `drvgpu` (DrvGPU::drvgpu)
8. Собирает включённые модули → targets `fft_func`, `filters`, ...
9. Пропускает `src/` (LIBGPU_BUILD_MAIN=OFF) и `python/` (LIBGPU_BUILD_PYTHON=OFF)
10. Возвращается в закрытый проект → линкует `closed_app` с targets LibGPU

**Что происходит при cmake build:**
1. Компилятор собирает DrvGPU (`.a` статическая библиотека)
2. HIP компилирует `.hip` файлы (statistics)
3. Компилятор собирает каждый модуль (`.a`)
4. Линкер собирает `closed_app` → подтягивает все `.a` + системные библиотеки

**Debug vs Release:**
- `CMAKE_BUILD_TYPE=Debug` → все `.a` собираются с `-g -O0` → полная отладка, Step Into в код LibGPU
- `CMAKE_BUILD_TYPE=Release` → все `.a` собираются с `-O3 -DNDEBUG` → максимальная производительность

#### 6.4 Обновление LibGPU в закрытом проекте

```bash
# Когда Alex выпустил новую версию:
cd /path/to/ClosedProject

# Обновить submodule до последнего коммита
git submodule update --remote libs/LibGPU

# Или перейти на конкретный тег
cd libs/LibGPU
git fetch
git checkout v1.2.0
cd ../..

# Зафиксировать обновление
git add libs/LibGPU
git commit -m "update LibGPU to v1.2.0"
```

#### 6.5 Первый clone закрытого проекта с LibGPU

```bash
# Для новых разработчиков:
git clone --recurse-submodules ssh://server/ClosedProject.git

# Или если уже склонировали без submodules:
git submodule update --init --recursive
```

---

## 📊 3 режима сборки (итого)

| Режим | Preset | CMAKE_BUILD_TYPE | Опции | Что собирается | Кто использует |
|-------|--------|-----------------|-------|----------------|---------------|
| **debug** | `--preset debug` | Debug | TESTS=OFF, MAIN=OFF | Все модули как `.a` с -g -O0 | Ребята отлаживают закрытый проект, Step Into в код GPUWorkLib |
| **release** | `--preset release` | Release | TESTS=OFF, MAIN=OFF | Все модули как `.a` с -O3 | Продакшен сборка (минимальный код) |
| **local-debug** | `--preset local-debug` | Debug | TESTS=ON, MAIN=ON, PYTHON=ON | ВСЁ: модули + тесты + main + Python | Alex + Кодо, разработка и отладка |

---

## 🔄 Полный workflow

```
╔═══════════════════════════════════════════════════════════════════╗
║  1. Alex разрабатывает в GPUWorkLib (с Кодо)                     ║
║     - Пишет код, запускает тесты, отлаживает                     ║
║     - GPUWorkLib живёт на GitHub (публичный/приватный)            ║
╠═══════════════════════════════════════════════════════════════════╣
║  2. Тесты проходят — код валидирован                              ║
║     - C++ тесты: cmake --build . --target GPUWorkLib && ./GPUWorkLib║
║     - Python тесты: python Python_test/module/test_xxx.py         ║
╠═══════════════════════════════════════════════════════════════════╣
║  3. Alex запускает синхронизацию:                                  ║
║     python sync_to_libgpu.py --commit --tag v1.2.0                ║
║     - Копирует код по whitelist                                   ║
║     - Исключает MemoryBank, .claude, Logs, Results                ║
║     - Создаёт коммит и тег в LibGPU                               ║
╠═══════════════════════════════════════════════════════════════════╣
║  4. Alex пушит LibGPU на локальный сервер:                        ║
║     cd ../LibGPU && git push origin main --tags                   ║
╠═══════════════════════════════════════════════════════════════════╣
║  5. Ребята в закрытом проекте обновляют:                          ║
║     git submodule update --remote libs/LibGPU                     ║
║     cmake --build build-release                                    ║
╚═══════════════════════════════════════════════════════════════════╝
```

---

## 📦 Runtime Assets (kernel-файлы и конфигурация)

### Kernel-файлы (.cl, .hip)

Модули используют `CMAKE_CURRENT_SOURCE_DIR/kernels` для пути к kernel-файлам:
```cmake
# Пример из filters/CMakeLists.txt:
set(FILTERS_KERNELS_DIR "${CMAKE_CURRENT_SOURCE_DIR}/kernels")
target_compile_definitions(filters PRIVATE FILTERS_KERNELS_DIR="${FILTERS_KERNELS_DIR}")
```

**Фаза 1 (add_subdirectory):** `CMAKE_CURRENT_SOURCE_DIR` автоматически указывает на правильную директорию внутри submodule. Работает без изменений.

**Фаза 2 (install + find_package):** Kernel-файлы нужно будет устанавливать:
```cmake
install(DIRECTORY kernels/ DESTINATION share/GPUWorkLib/fft_func/kernels)
```
И передавать путь через Config.cmake. TODO для фазы 2.

### configGPU.json

DrvGPU загружает `configGPU.json` из рабочей директории:
```cpp
GPUConfig::GetInstance().LoadOrCreate("configGPU.json");
```

**Закрытый проект** должен скопировать `configGPU.json` из LibGPU в свою рабочую директорию и настроить под GPU сервера.

---

## 🔄 Rollback-стратегия

Если синхронизация сломала LibGPU:

```bash
# Откат LibGPU к предыдущему тегу:
cd /path/to/LibGPU
git log --oneline --tags    # посмотреть теги
git reset --hard v1.1.0     # откатить к рабочей версии
git push --force origin main

# В закрытом проекте — зафиксировать рабочую версию:
cd /path/to/ClosedProject/libs/GPUWorkLib
git checkout v1.1.0
cd ../..
git add libs/GPUWorkLib
git commit -m "rollback GPUWorkLib to v1.1.0"
```

---

## 🧪 CI/CD валидация после синхронизации

Минимальный smoke-тест после каждого sync:

```bash
cd /path/to/LibGPU
cmake --preset local-debug
cmake --build build-local-debug -j$(nproc)
./build-local-debug/GPUWorkLib    # запуск C++ тестов
```

Если smoke-тест упал — НЕ пушить, исправить в GPUWorkLib и пересинхронизировать.

---

## ✅ Верификация (чек-лист)

1. [ ] `python sync_to_libgpu.py --dry-run` — whitelist корректен
2. [ ] LibGPU: `cmake --preset debug && cmake --build build-debug` — сборка OK
3. [ ] LibGPU: `cmake --preset release && cmake --build build-release` — сборка OK
4. [ ] LibGPU: `cmake --preset local-debug && cmake --build build-local-debug` — main + тесты OK
5. [ ] Тестовый проект: `add_subdirectory(../LibGPU ...)` → сборка OK
6. [ ] Тестовый проект: `target_link_libraries(app PRIVATE DrvGPU::drvgpu GPUWorkLib::fft_func)` → линковка OK
7. [ ] Step Into из тестового проекта заходит в исходники GPUWorkLib (debug mode)
8. [ ] GPUWorkLib продолжает работать как раньше (ничего не сломано)
9. [ ] `sync_to_libgpu.py --clean` — устаревшие файлы удаляются
10. [ ] MemoryBank/, .claude/, Doc_Addition/PLAN/, Logs/, Results/ НЕ попадают в LibGPU
11. [ ] git tag в LibGPU → закрытый проект фиксирует версию через submodule
12. [ ] configGPU.json скопирован и настроен в рабочей директории закрытого проекта

---

## 🔑 Создаваемые файлы (полный список)

### В GPUWorkLib (рабочий репо):
1. **`sync_to_libgpu.py`** — скрипт синхронизации (~250 строк Python)
2. **`MemoryBank/research/cmake_libgpu_integration.md`** — документ исследования (для отчёта)

### В LibGPU (новый репо, создаётся с нуля):
3. **`CMakeLists.txt`** — корневой CMake (library mode, ~150 строк)
4. **`CMakePresets.json`** — presets: debug / release / local-debug (Ninja)
5. **`GPUWorkLibConfig.cmake.in`** — для find_package() (фаза 2)
6. **`VERSION`** — файл версии (`1.1.0`)
7. **`.gitignore`** — стандартный (build/, Logs/, __pycache__)
8. **`README.md`** — описание LibGPU и инструкции подключения

### Существующие файлы GPUWorkLib — БЕЗ ИЗМЕНЕНИЙ:
- CMakeLists.txt модулей уже совместимы с `add_subdirectory`
- cmake/ хелперы уже настроены правильно
- install(EXPORT GPUWorkLibTargets) уже есть в модулях

### Нужно добавить в GPUWorkLib (мелкие правки):
- **`DrvGPU/CMakeLists.txt`** — добавить `install(TARGETS drvgpu EXPORT GPUWorkLibTargets ...)` (сейчас отсутствует, нужно для фазы 2)

---

## ⚠️ Ключевое правило: LibGPU = "конверт", НЕ проект

> **LibGPU** — это название директории и git-репозитория для передачи кода.
> **GPUWorkLib** — это название проекта. Оно НЕ меняется.
>
> - `project(GPUWorkLib ...)` — остаётся
> - `GPUWorkLib::fft_func` — остаётся
> - `DrvGPU::drvgpu` — остаётся
> - `EXPORT GPUWorkLibTargets` — остаётся
> - Код в LibGPU = копия кода из GPUWorkLib (без MemoryBank, .claude и т.д.)
> - В LibGPU код НЕ переписывается, НЕ переименовывается
