# Исследование: Интеграция GPUWorkLib в закрытый проект через CMake

## Контекст

GPUWorkLib -- GPU-библиотека (OpenCL 3.0, ROCm 7.2+, HIP) с 12 модулями.
Нужно подключить к закрытому проекту. Платформы: Windows (MSVC/Ninja) + Linux (ROCm).

Текущая структура:
- `DrvGPU/` -- ядро (backends: OpenCL, ROCm, Hybrid)
- `modules/` -- 12 модулей (fft_func, filters, heterodyne, statistics, vector_algebra, capon, range_angle, fm_correlator, strategies, lch_farrow, signal_generators, test_utils)
- `cmake/` -- 5 cmake-скриптов (platform-detection, gpu-config, dependencies, compiler-options, debug-config)
- `third_party/plog/` -- header-only логгер
- `CMakePresets.json` -- windows-vs2026 + windows-nvidia
- CMake >= 3.20, C++17, `CMAKE_POSITION_INDEPENDENT_CODE ON`

---

## 1. Сводная таблица: 6 вариантов интеграции

| Критерий | git submodule | git subtree | FetchContent | ExternalProject_Add | find_package | add_subdirectory |
|---|---|---|---|---|---|---|
| **Простота настройки** | 4/5 | 3/5 | **5/5** | 2/5 | 2/5 | **5/5** |
| **Отладка (step-through)** | **5/5** | **5/5** | 4/5 | 1/5 | 3/5 | **5/5** |
| **3 режима сборки** | **5/5** | **5/5** | **5/5** | 3/5 | 4/5 | **5/5** |
| **Частичная синхронизация** | 2/5 | 2/5 | 2/5 | 2/5 | **5/5** | 1/5 |
| **Windows MSVC/Ninja** | **5/5** | **5/5** | 4/5 | 2/5 | 3/5 | **5/5** |
| **Linux ROCm** | **5/5** | **5/5** | **5/5** | 4/5 | 4/5 | **5/5** |
| **ИТОГО** | **26/30** | **25/30** | **25/30** | **14/30** | **21/30** | **26/30** |

---

## 2. Детальный анализ каждого варианта

### 2.1 git submodule + add_subdirectory

**Механика:**
```bash
cd closed-project
git submodule add git@company.com:LibGPU.git extern/LibGPU
```
```cmake
# closed-project/CMakeLists.txt
add_subdirectory(extern/LibGPU)
target_link_libraries(myapp PRIVATE DrvGPU fft_processor_lib)
```

**Плюсы:**
- Простая начальная настройка -- одна команда git + одна строка CMake
- Точная фиксация версии (конкретный коммит SHA)
- Полный доступ к исходникам -- отладка и step-through работают идеально
- IDE (CLion, VS Code, Visual Studio) видят все файлы
- CI/CD: `git submodule update --init --recursive`
- Windows и Linux работают одинаково

**Минусы:**
- Тянет весь репозиторий (включая тесты, документацию, скрипты)
- git submodule "pain" -- разработчики забывают `--recursive`, "грязные" коммиты
- Обновление требует ручного `git submodule update --remote` + commit
- Вложенные submodules -- если LibGPU тоже имеет submodules (plog)
- Нет фильтрации файлов -- нужен промежуточный репозиторий

**Когда использовать:** Команда привыкла к submodules, нужен максимально быстрый старт.

---

### 2.2 git subtree

**Механика:**
```bash
git subtree add --prefix=extern/LibGPU git@company.com:LibGPU.git main --squash
# Обновление:
git subtree pull --prefix=extern/LibGPU git@company.com:LibGPU.git main --squash
```

**Плюсы:**
- Код физически в репозитории -- обычный `git clone` содержит все
- CI/CD проще -- нет `submodule update`
- Можно локально патчить код под свой проект
- Отладка идеальная -- исходники рядом

**Минусы:**
- История загрязняется (squash = один большой коммит)
- Merge conflicts при subtree pull, особенно если были локальные патчи
- `git subtree` менее известен -- путаница в команде
- Размер репозитория растет
- Обратная синхронизация (push) -- сложная и хрупкая

**Когда использовать:** Нужно патчить LibGPU под свой проект, команда не хочет submodules.

---

### 2.3 CMake FetchContent (РЕКОМЕНДУЕТСЯ)

**Механика:**
```cmake
include(FetchContent)
FetchContent_Declare(LibGPU
  GIT_REPOSITORY git@company.com:team/LibGPU.git
  GIT_TAG v1.2.0
)
FetchContent_MakeAvailable(LibGPU)
```

**Плюсы:**
- Чистейшее CMake-решение -- нет зависимости от git submodule/subtree
- Автоматическое скачивание при configure
- Исходники доступны в `_deps/libgpu-src/` -- отладка работает
- **KILLER FEATURE**: `FETCHCONTENT_SOURCE_DIR_LIBGPU` для локальной разработки!
  ```bash
  cmake -DFETCHCONTENT_SOURCE_DIR_LIBGPU=/home/dev/LibGPU -DCMAKE_BUILD_TYPE=Debug ..
  ```
- CI/CD: ничего дополнительного, cmake сам скачает
- Можно переопределять через CMakePresets.json

**Минусы:**
- Первый configure медленнее (git clone)
- Тянет весь репозиторий (нужен промежуточный LibGPU)
- Вложенные FetchContent могут конфликтовать (diamond dependency)
- На Windows за proxy/firewall -- git clone может не работать
- Нет возможности патчить код (только через PATCH_COMMAND)

**Killer Feature -- LocalDev через CMakePresets.json:**
```json
{
  "configurePresets": [
    {
      "name": "local-dev",
      "displayName": "Local Development (LibGPU from disk)",
      "cacheVariables": {
        "FETCHCONTENT_SOURCE_DIR_LIBGPU": "${sourceDir}/../LibGPU",
        "CMAKE_BUILD_TYPE": "Debug"
      }
    },
    {
      "name": "ci-release",
      "displayName": "CI Release Build",
      "cacheVariables": {
        "CMAKE_BUILD_TYPE": "Release"
      }
    }
  ]
}
```

**Когда использовать:** Основной рекомендуемый вариант. Чистый CMake, удобный LocalDev.

---

### 2.4 CMake ExternalProject_Add (НЕ РЕКОМЕНДУЕТСЯ)

**Механика:**
```cmake
include(ExternalProject)
ExternalProject_Add(LibGPU_ext
  GIT_REPOSITORY git@company.com:LibGPU.git
  GIT_TAG v1.2.0
  CMAKE_ARGS -DCMAKE_INSTALL_PREFIX=<INSTALL_DIR>
             -DENABLE_ROCM=${ENABLE_ROCM}
  INSTALL_DIR ${CMAKE_BINARY_DIR}/LibGPU-install
)
```

**Плюсы:**
- Полная изоляция -- LibGPU как независимый проект
- Свой CMAKE_BUILD_TYPE (LibGPU Release, основной Debug)

**Минусы:**
- **КРИТИЧЕСКИЙ**: Targets НЕ доступны на этапе configure! Нельзя `target_link_libraries(myapp LibGPU::DrvGPU)` напрямую
- Нужны хаки с IMPORTED targets и `ExternalProject_Get_Property`
- Отладка практически невозможна -- нет исходников в IDE
- Два этапа сборки -- значительно сложнее
- Не работает с multi-config generators (MSVC) без костылей
- Невозможен step-through debugging

**Когда использовать:** НИКОГДА для разработческой интеграции. Только если LibGPU -- стабильная "черная коробка" в CI/CD pipeline.

---

### 2.5 find_package() с предварительной установкой

**Механика:**
```bash
# Шаг 1: Собрать и установить LibGPU
cd LibGPU && cmake -B build -DCMAKE_INSTALL_PREFIX=/opt/LibGPU && cmake --build build --target install

# Шаг 2: В закрытом проекте
cmake -DCMAKE_PREFIX_PATH=/opt/LibGPU ..
```
```cmake
find_package(LibGPU 1.2 REQUIRED CONFIG)
target_link_libraries(myapp PRIVATE LibGPU::DrvGPU LibGPU::FFTProcessor)
```

**Требует создания в LibGPU:**
- `LibGPUConfig.cmake.in` -- шаблон конфигурации
- `install(EXPORT LibGPUTargets ...)` -- экспорт targets
- `install(TARGETS ... EXPORT LibGPUTargets ...)` -- для каждого модуля
- `write_basic_package_version_file()` -- версионирование
- Правильный экспорт транзитивных зависимостей (OpenCL, HIP, hipFFT)

**Плюсы:**
- Самый "правильный" по стандартам CMake
- Чистое разделение -- LibGPU = независимый пакет
- Быстрый configure (ничего не скачивает/собирает)
- `find_package` дает targets, include paths, compiler definitions
- Версионирование через ConfigVersion.cmake
- **Частичная синхронизация**: install копирует ТОЛЬКО нужные файлы (headers + libs)

**Минусы:**
- Нужно НАПИСАТЬ install/export инфраструктуру (значительная работа!)
- Разработчик должен пересобрать/переустановить LibGPU при каждом изменении
- Отладка: step-through только если Debug install + исходники доступны
- Windows: нет `/usr/local`, нужен `CMAKE_PREFIX_PATH`
- GPU-зависимости (OpenCL, ROCm, hipFFT) должны быть правильно экспортированы

**Шаблон LibGPUConfig.cmake.in:**
```cmake
@PACKAGE_INIT@
include(CMakeFindDependencyMacro)

# Транзитивные зависимости
if(@OPENCL_ENABLED@)
  find_dependency(OpenCL)
endif()
find_dependency(Threads)
if(@ROCM_ENABLED@)
  find_dependency(hip)
  if(@HIPFFT_FOUND@)
    find_dependency(hipfft)
  endif()
endif()

include("${CMAKE_CURRENT_LIST_DIR}/LibGPUTargets.cmake")
check_required_components(LibGPU)
```

**Когда использовать:** Когда API стабилизировался, для production-deployments.

---

### 2.6 add_subdirectory() напрямую

**Механика:**
```cmake
# LibGPU лежит рядом
add_subdirectory(../LibGPU ${CMAKE_BINARY_DIR}/LibGPU)
# или в подпапке
add_subdirectory(extern/LibGPU)
```

**Плюсы:**
- Максимально просто -- одна строка CMake
- Полный доступ к исходникам, идеальная отладка
- Один build tree, один CMAKE_BUILD_TYPE
- IDE видит все файлы
- Нет промежуточных шагов

**Минусы:**
- Нужно как-то получить LibGPU рядом (clone, symlink, submodule)
- Может загрязнять namespace targets
- Нет фиксации версии -- какая копия лежит, та используется
- Option pollution -- все опции LibGPU видны в основном проекте

**Совместимость с GPUWorkLib:** DrvGPU/CMakeLists.txt написан без `project()` -- специально для subdirectory! Уже совместим.

**Когда использовать:** Для быстрого прототипирования, когда LibGPU лежит рядом.

---

## 3. Промежуточный репозиторий LibGPU

### 3.1 Зачем нужен

GPUWorkLib содержит много лишнего для интеграции:
- `Python_test/`, `Results/`, `Logs/` -- артефакты
- `Doc/`, `Doc_Addition/`, `MemoryBank/` -- документация и управление
- `.claude/` -- AI-инструментарий
- `src/main.cpp` -- главный исполняемый файл (потребитель не должен видеть)
- `tests/` в модулях -- опционально

### 3.2 Что включить в LibGPU

```
LibGPU/
  CMakeLists.txt          # Адаптированный (без src/, python/)
  CMakePresets.json        # С presets для Windows + Linux
  cmake/
    platform-detection.cmake
    gpu-config.cmake
    dependencies.cmake
    compiler-options.cmake
  DrvGPU/                  # Полностью
    include/
    interface/
    services/
    backends/
    memory/
    common/
    logger/
    src/
    CMakeLists.txt
  modules/
    fft_func/              # Выбранные модули
    filters/
    heterodyne/
    statistics/
    vector_algebra/
    capon/
    range_angle/
    fm_correlator/
    strategies/
    lch_farrow/
    signal_generators/
  third_party/
    plog/include/          # Только headers
  config/
    configGPU.json         # Если нужен
```

### 3.3 Способы синхронизации GPUWorkLib -> LibGPU

#### A) Скрипт синхронизации (РЕКОМЕНДУЕТСЯ)

**Linux (sync_to_libgpu.sh):**
```bash
#!/bin/bash
set -euo pipefail
SRC="${1:-$HOME/projects/GPUWorkLib}"
DST="${2:-$HOME/projects/LibGPU}"

echo "Syncing GPUWorkLib -> LibGPU..."

# Build system
rsync -av --delete "$SRC/cmake/" "$DST/cmake/"
cp "$SRC/CMakePresets.json" "$DST/CMakePresets.json"
# CMakeLists.txt -- адаптированный, НЕ перезаписывать автоматически!

# DrvGPU core (без tests/)
rsync -av --delete \
  --exclude='tests/' \
  "$SRC/DrvGPU/" "$DST/DrvGPU/"

# Modules
MODULES="fft_func filters heterodyne statistics vector_algebra capon range_angle fm_correlator strategies lch_farrow signal_generators"
for mod in $MODULES; do
  if [ -d "$SRC/modules/$mod" ]; then
    rsync -av --delete \
      --exclude='tests/' \
      --exclude='__pycache__/' \
      "$SRC/modules/$mod/" "$DST/modules/$mod/"
  fi
done

# Third-party (только headers)
mkdir -p "$DST/third_party/plog"
rsync -av --delete "$SRC/third_party/plog/include/" "$DST/third_party/plog/include/"

echo "Done! Now review changes in $DST and commit."
```

**Windows (sync_to_libgpu.bat):**
```bat
@echo off
set SRC=E:\C++\GPUWorkLib
set DST=E:\C++\LibGPU

robocopy "%SRC%\cmake" "%DST%\cmake" /MIR /XD .git
robocopy "%SRC%\DrvGPU" "%DST%\DrvGPU" /MIR /XD tests .git __pycache__
rem ... аналогично для каждого модуля
```

**Плюсы:** Полный контроль, легко адаптировать, работает на обеих ОС.
**Минусы:** Ручной запуск, нет автоматической истории.

#### B) git sparse-checkout

```bash
# В закрытом проекте -- submodule с sparse-checkout
git submodule add git@company.com:GPUWorkLib.git extern/GPUWorkLib
cd extern/GPUWorkLib
git sparse-checkout init --cone
git sparse-checkout set DrvGPU modules/fft_func modules/filters cmake third_party/plog/include
```

**Плюсы:** Не нужен промежуточный репо!
**Минусы:** sparse-checkout + submodules = bleeding edge, возможны баги. Не все git GUI поддерживают.

#### C) CMake install(EXPORT ...) -- пакетная поставка

```bash
cd GPUWorkLib
cmake -B build -DCMAKE_INSTALL_PREFIX=$HOME/LibGPU-install -DCMAKE_BUILD_TYPE=Release
cmake --build build --target install
# Получаем: include/ + lib/ + cmake/LibGPUConfig.cmake
```

**Плюсы:** Только нужные файлы, версионирование, чистый пакет.
**Минусы:** Нужно писать install() правила, теряем исходники для отладки.

#### D) GitHub Actions автосинхронизация

```yaml
name: Sync to LibGPU
on:
  push:
    branches: [main]
    paths:
      - 'DrvGPU/**'
      - 'modules/**'
      - 'cmake/**'
jobs:
  sync:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - name: Sync
        run: ./scripts/sync_to_libgpu.sh . /tmp/LibGPU
      - name: Push to LibGPU
        run: |
          cd /tmp/LibGPU
          git add -A && git commit -m "sync from GPUWorkLib $(date)" && git push
```

---

## 4. РЕКОМЕНДАЦИЯ: Поэтапный план

### Этап 1 -- Быстрый старт (1-2 дня)

**Подход:** git submodule + add_subdirectory

1. Создать репозиторий LibGPU (промежуточный)
2. Написать `sync_to_libgpu.sh` скрипт
3. Адаптировать корневой CMakeLists.txt (убрать src/, python/, option для тестов)
4. В закрытом проекте:
   ```bash
   git submodule add git@company.com:team/LibGPU.git extern/LibGPU
   ```
   ```cmake
   add_subdirectory(extern/LibGPU)
   target_link_libraries(myapp PRIVATE DrvGPU fft_processor_lib)
   ```

**Почему:** Минимум настройки, максимум удобства, отладка "из коробки".

### Этап 2 -- Переход на FetchContent (через 1-2 месяца)

**Подход:** FetchContent + CMakePresets для LocalDev

1. Заменить submodule на FetchContent в CMakeLists.txt
2. Добавить CMakePresets.json с `local-dev` preset:
   ```json
   {
     "name": "local-dev",
     "cacheVariables": {
       "FETCHCONTENT_SOURCE_DIR_LIBGPU": "${sourceDir}/../LibGPU"
     }
   }
   ```
3. Разработчики, которые активно работают с LibGPU, клонируют его рядом и используют `local-dev` preset
4. CI использует FetchContent с GIT_TAG (автоскачивание)

**Почему:** Чистый CMake, удобный переключатель LocalDev/CI.

### Этап 3 -- Production-ready (когда API стабилен)

**Подход:** install(EXPORT ...) + find_package()

1. Добавить install/export инфраструктуру в LibGPU
2. CI собирает и публикует пакет LibGPU (артефакт)
3. Закрытый проект:
   ```cmake
   find_package(LibGPU 1.2 REQUIRED CONFIG)
   target_link_libraries(myapp PRIVATE LibGPU::DrvGPU LibGPU::FFTProcessor)
   ```
4. Для разработки -- FetchContent fallback:
   ```cmake
   find_package(LibGPU 1.2 CONFIG QUIET)
   if(NOT LibGPU_FOUND)
     include(FetchContent)
     FetchContent_Declare(LibGPU GIT_REPOSITORY ... GIT_TAG ...)
     FetchContent_MakeAvailable(LibGPU)
   endif()
   ```

**Почему:** Самый правильный по CMake-стандартам, быстрый configure, версионирование.

---

## 5. Адаптация CMakeLists.txt для LibGPU

Текущий корневой `CMakeLists.txt` GPUWorkLib нужно адаптировать для LibGPU:

```cmake
# LibGPU/CMakeLists.txt
cmake_minimum_required(VERSION 3.20)

# Если включен как subdirectory -- не объявляем project
if(CMAKE_SOURCE_DIR STREQUAL CMAKE_CURRENT_SOURCE_DIR)
  project(LibGPU VERSION 1.1.0 LANGUAGES CXX)
endif()

# Модули сборки
include(cmake/platform-detection.cmake)
include(cmake/gpu-config.cmake)
include(cmake/dependencies.cmake)

set(CMAKE_POSITION_INDEPENDENT_CODE ON)

# Core
add_subdirectory(DrvGPU)

# Modules (можно сделать option для каждого)
option(LIBGPU_BUILD_FFT "Build FFT module" ON)
option(LIBGPU_BUILD_FILTERS "Build Filters module" ON)
# ... и т.д.

if(LIBGPU_BUILD_FFT)
  add_subdirectory(modules/fft_func)
endif()
# ... аналогично для остальных

# НЕ включаем: src/, python/, tests
```

---

## 6. Особенности GPU-зависимостей

### OpenCL
- Windows: ищется в CUDA SDK или System32 (dependencies.cmake уже умеет)
- Linux: `find_package(OpenCL)` или pkg-config

### ROCm/HIP
- Только Linux: `find_package(hip)`, `find_package(hipfft)`
- Все ROCM-зависимости (rocBLAS, rocSOLVER) нужны для vector_algebra, capon
- При export нужно `find_dependency(hip)` в Config.cmake

### clFFT
- Windows: локальная копия в `clFFT/` (headers + lib + dll)
- Linux: системный пакет `libclfft-dev`
- На RDNA4 (gfx1201) НЕ работает -- только hipFFT

### Транзитивные зависимости
При install/export обязательно указать:
```cmake
# LibGPUConfig.cmake.in
find_dependency(Threads)
if(@OPENCL_ENABLED@)
  find_dependency(OpenCL)
endif()
if(@ROCM_ENABLED@)
  find_dependency(hip)
  find_dependency(hipfft)
  find_dependency(rocblas)    # для vector_algebra
  find_dependency(rocsolver)  # для vector_algebra
endif()
```

---

## 7. Итоговая рекомендация (из исследования)

**Для текущего этапа (активная разработка):**
- **FetchContent** как основной метод интеграции
- **Промежуточный репозиторий LibGPU** со скриптом синхронизации
- **CMakePresets.json** с `local-dev` preset для разработчиков

**Ключевые преимущества выбора:**
1. Нет "git submodule pain"
2. `FETCHCONTENT_SOURCE_DIR_LIBGPU` -- мгновенное переключение на локальный код
3. CI/CD работает автоматически (git clone при configure)
4. Один CMake workflow для Debug/Release/LocalDev
5. Работает идентично на Windows и Linux

> **Итоговое решение Alex'а (2026-03-29):** Выбран **git submodule + add_subdirectory** вместо FetchContent. Причина: закрытый проект на локальном сервере без интернета, git submodule проще для ручного контроля. Детали -- см. `cmake_libgpu_integration_plan.md`.
