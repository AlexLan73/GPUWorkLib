# TASK: Интеграция GPUWorkLib -> LibGPU -> Закрытый проект

> **Статус**: BACKLOG (запланировано на 2026-03-30)
> **Дата создания**: 2026-03-29  |  **Автор**: Кодо
> **Исполнитель**: Alex + Кодо
> **Ревьюер**: Кодо
>
> **План**: `MemoryBank/research/cmake_libgpu_integration_plan.md`
> **Исследование**: `MemoryBank/research/cmake_libgpu_research_agent.md`
> **Ревью**: `MemoryBank/research/cmake_libgpu_review_issues.md`

---

## Цель

Создать промежуточный репозиторий LibGPU (конверт для передачи кода GPUWorkLib в закрытый проект). Заменить ручное копирование файлов на автоматизированную синхронизацию + git submodule.

---

## Задачи

### A. Подготовка GPUWorkLib (мелкие правки в рабочем репо)

#### A1. DrvGPU: добавить install() правила
- **Файл**: `DrvGPU/CMakeLists.txt`
- **Что сделать**: Добавить `install(TARGETS drvgpu EXPORT GPUWorkLibTargets ARCHIVE DESTINATION lib LIBRARY DESTINATION lib RUNTIME DESTINATION bin)` + `install(DIRECTORY include/ DESTINATION include/DrvGPU)` для public headers
- **Зачем**: Сейчас DrvGPU единственный target без install() — при find_package (фаза 2) не войдёт в export
- [ ] Добавить install(TARGETS) с EXPORT GPUWorkLibTargets
- [ ] Добавить install(DIRECTORY) для public headers
- [ ] Проверить что GPUWorkLib собирается без ошибок после правки

#### A2. Написать sync_to_libgpu.py
- **Файл**: `E:\C++\GPUWorkLib\sync_to_libgpu.py` (~250 строк)
- **Спецификация**: integration_plan.md, Шаг 2
- **Интерфейс**: `--src`, `--dst`, `--dry-run`, `--clean`, `--commit`, `--tag`, `--exclude`
- [ ] Whitelist директорий (SYNC_DIRS): cmake/, config/, DrvGPU/, modules/, src/, python/, Python_test/, Doc/, Doc_Addition/, include/, third_party/
- [ ] Whitelist файлов (SYNC_FILES): configGPU.json, requirements.txt, run.sh, run.bat, README.md, .clangd, .gitattributes
- [ ] EXCLUDE_PATTERNS: MemoryBank/, Doc_Addition/PLAN/, .claude/, .mcp.json, api_keys.json, Results/, Logs/, .vscode/, .idea/, .cursor/, scripts/, CLAUDE.md, build*/, CMakeUserPresets.json, setup_linux_claude.sh, __pycache__/, *.pyc, .git/
- [ ] Алгоритм: проверка src/dst -> сканирование whitelist -> exclude -> hash comparison -> copy
- [ ] --dry-run: показать что будет скопировано без выполнения
- [ ] --clean: удалить файлы в dst которых нет в src (кроме LibGPU-specific: CMakeLists.txt, CMakePresets.json, VERSION, GPUWorkLibConfig.cmake.in)
- [ ] --commit: git add -A + git commit в LibGPU
- [ ] --tag: git tag в LibGPU
- [ ] Статистика: Добавлено / Обновлено / Удалено / Без изменений
- [ ] Тест: `python sync_to_libgpu.py --dry-run` на GPUWorkLib

---

### B. Создание репозитория LibGPU

#### B1. Инициализация git-репозитория
- **Где**: `E:\C++\LibGPU\` (локально) + bare repo на сервере
- [ ] `mkdir E:\C++\LibGPU && cd E:\C++\LibGPU && git init`
- [ ] `git remote add origin ssh://server/srv/repos/LibGPU.git` (когда сервер готов)
- [ ] Создать `.gitignore` (build*/, Logs/, __pycache__/, *.pyc, *.o, *.a, *.so, *.lib, *.dll, *.exe, CMakeUserPresets.json, compile_commands.json, .cache/)
- [ ] Создать `VERSION` (содержимое: `1.1.0`)

#### B2. Корневой CMakeLists.txt для LibGPU
- **Файл**: `E:\C++\LibGPU\CMakeLists.txt` (~150 строк)
- **Спецификация**: integration_plan.md, Шаг 3
- **ВАЖНО**: `project(GPUWorkLib ...)` НЕ `project(LibGPU ...)`!
- [ ] `cmake_minimum_required(VERSION 3.20)`
- [ ] Чтение VERSION из файла
- [ ] Проверка top-level: `if(CMAKE_CURRENT_SOURCE_DIR STREQUAL CMAKE_SOURCE_DIR)` -> `project(GPUWorkLib VERSION ...)`
- [ ] LIBGPU_IS_TOP_LEVEL: TRUE/FALSE
- [ ] Опции: LIBGPU_BUILD_TESTS, LIBGPU_BUILD_MAIN, LIBGPU_BUILD_PYTHON (default: зависит от IS_TOP_LEVEL)
- [ ] Опции модулей: LIBGPU_BUILD_FFT, LIBGPU_BUILD_FILTERS, ... (default: ON)
- [ ] include() cmake хелперов через CMAKE_CURRENT_SOURCE_DIR
- [ ] add_subdirectory(DrvGPU) — всегда
- [ ] add_subdirectory модулей — в порядке зависимостей, с проверкой ROCM_ENABLED + TARGET
- [ ] Секция EXPORT: `install(EXPORT GPUWorkLibTargets NAMESPACE GPUWorkLib:: DESTINATION lib/cmake/GPUWorkLib)`
- [ ] write_basic_package_version_file + configure_package_config_file
- [ ] Блок message(STATUS) с информацией о сборке

#### B3. CMakePresets.json для LibGPU
- **Файл**: `E:\C++\LibGPU\CMakePresets.json`
- **Спецификация**: integration_plan.md, Шаг 4
- [ ] Preset `debug`: Ninja, Debug, ROCM=ON, OPENCL=ON, TESTS=OFF, MAIN=OFF, PYTHON=OFF
- [ ] Preset `release`: Ninja, Release, ROCM=ON, OPENCL=ON, TESTS=OFF, MAIN=OFF, PYTHON=OFF
- [ ] Preset `local-debug`: Ninja, Debug, ROCM=ON, OPENCL=ON, TESTS=ON, MAIN=ON, PYTHON=ON
- [ ] buildPresets для каждого (jobs: 8)

#### B4. GPUWorkLibConfig.cmake.in
- **Файл**: `E:\C++\LibGPU\GPUWorkLibConfig.cmake.in`
- **Спецификация**: integration_plan.md, Шаг 5
- [ ] @PACKAGE_INIT@
- [ ] find_dependency(Threads)
- [ ] find_dependency(OpenCL) — if @OPENCL_ENABLED@
- [ ] find_dependency(hip, hipfft, rocblas, rocsolver) — if @ROCM_ENABLED@
- [ ] include GPUWorkLibTargets.cmake
- [ ] check_required_components

#### B5. README.md для LibGPU
- [ ] Описание: GPUWorkLib — GPU Signal Processing Library
- [ ] Инструкция подключения через add_subdirectory
- [ ] Пример target_link_libraries (DrvGPU::drvgpu + GPUWorkLib::*)
- [ ] 3 режима сборки (debug / release / local-debug)
- [ ] Требования: CMake >= 3.20, C++17, ROCm 7.2+, Ninja

---

### C. Первая синхронизация и проверка

#### C1. Синхронизация GPUWorkLib -> LibGPU
- [ ] `python sync_to_libgpu.py --src E:\C++\GPUWorkLib --dst E:\C++\LibGPU --dry-run` — проверить whitelist
- [ ] `python sync_to_libgpu.py --src E:\C++\GPUWorkLib --dst E:\C++\LibGPU --commit --tag v1.0.0`
- [ ] Проверить что MemoryBank/, .claude/, Doc_Addition/PLAN/, Logs/, Results/ НЕ попали

#### C2. Smoke-тест сборки LibGPU (на Linux с ROCm)
- [ ] `cmake --preset local-debug`
- [ ] `cmake --build build-local-debug -j$(nproc)`
- [ ] `./build-local-debug/GPUWorkLib` — C++ тесты проходят
- [ ] `cmake --preset debug && cmake --build build-debug` — сборка OK
- [ ] `cmake --preset release && cmake --build build-release` — сборка OK

#### C3. Тест интеграции (тестовый проект)
- [ ] Создать минимальный тестовый проект с `add_subdirectory(../LibGPU ...)`
- [ ] `target_link_libraries(test_app PRIVATE DrvGPU::drvgpu GPUWorkLib::fft_func)` — линковка OK
- [ ] Step Into в исходники GPUWorkLib из debug (IDE)
- [ ] Проверить что GPUWorkLib (рабочий) продолжает собираться без ошибок

---

### D. Интеграция в закрытый проект (на сервере)

#### D1. Git submodule
- [ ] На сервере: `git init --bare /srv/repos/LibGPU.git`
- [ ] Пуш LibGPU: `cd E:\C++\LibGPU && git push -u origin main --tags`
- [ ] В закрытом проекте: `git submodule add ssh://server/srv/repos/LibGPU.git libs/GPUWorkLib`
- [ ] `cd libs/GPUWorkLib && git checkout v1.0.0`
- [ ] `git add libs/GPUWorkLib && git commit -m "add GPUWorkLib v1.0.0 as submodule"`

#### D2. CMakeLists.txt закрытого проекта
- [ ] Добавить опции LIBGPU_BUILD_TESTS=OFF, LIBGPU_BUILD_MAIN=OFF, LIBGPU_BUILD_PYTHON=OFF
- [ ] `add_subdirectory(libs/GPUWorkLib)`
- [ ] `target_link_libraries(closed_app PRIVATE DrvGPU::drvgpu GPUWorkLib::fft_func ...)` — нужные модули

#### D3. Верификация на сервере
- [ ] Debug сборка: `cmake -DCMAKE_BUILD_TYPE=Debug .. && cmake --build . -j8`
- [ ] Release сборка: `cmake -DCMAKE_BUILD_TYPE=Release .. && cmake --build . -j8`
- [ ] configGPU.json скопирован в рабочую директорию и настроен
- [ ] Step Into в GPUWorkLib из IDE ребят

---

## Порядок выполнения

```
A1 ──┐
     ├──> C1 ──> C2 ──> C3 ──> D1 ──> D2 ──> D3
A2 ──┤
B1 ──┤
B2 ──┤
B3 ──┤
B4 ──┤
B5 ──┘
```

- **Блок A+B** — можно параллельно (A1-A2 в GPUWorkLib, B1-B5 в LibGPU)
- **Блок C** — после A+B, smoke-тест
- **Блок D** — на сервере, после успешного C

---

## Критерии готовности

- [ ] sync_to_libgpu.py работает (--dry-run, --commit, --tag, --clean)
- [ ] LibGPU собирается в 3 режимах (debug, release, local-debug)
- [ ] Закрытый проект собирается с LibGPU как submodule (Debug + Release)
- [ ] MemoryBank, .claude, PLAN — гарантированно НЕ в LibGPU
- [ ] Step Into из закрытого проекта в исходники GPUWorkLib работает

---

*Создано: 2026-03-29*
*На завтра: 2026-03-30*
