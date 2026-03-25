Отлично, давай разберём это детально. Покажу все подходы на конкретной файловой структуре.Итак, у нас `/workspace/my-project/` зависит от `/workspace/lib-network/`, `/workspace/lib-protocol/` и `/workspace/third-party-lib/`. Разбираем все подходы от простого к правильному.

---

## Подход 1: `add_subdirectory()` с абсолютным путём

Самый прямолинейный способ. `add_subdirectory()` принимает абсолютный путь к исходникам, но в этом случае обязательно нужно указать второй аргумент — путь для бинарного дерева внутри вашей сборки.

```cmake
# my-project/CMakeLists.txt
cmake_minimum_required(VERSION 3.24)
project(my-project LANGUAGES CXX)

# Соседний каталог — указываем абсолютный путь + binary dir
add_subdirectory(
  ${CMAKE_CURRENT_SOURCE_DIR}/../lib-network   # SOURCE_DIR  (абсолютный!)
  ${CMAKE_BINARY_DIR}/_deps/lib-network-build   # BINARY_DIR  (обязательно!)
)

add_subdirectory(
  ${CMAKE_CURRENT_SOURCE_DIR}/../lib-protocol
  ${CMAKE_BINARY_DIR}/_deps/lib-protocol-build
)

add_executable(myapp src/main.cpp)
target_link_libraries(myapp PRIVATE network::network protocol::protocol)
```

При этом соседний проект должен определять таргеты с алиасами, чтобы их можно было одинаково линковать и через `add_subdirectory`, и через `find_package`:

```cmake
# lib-network/CMakeLists.txt
cmake_minimum_required(VERSION 3.24)
project(lib-network LANGUAGES CXX)

add_library(network src/network.cpp)
add_library(network::network ALIAS network)  # ← ключевой момент!

target_include_directories(network
  PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:include>
)
```

**Плюсы**: просто, работает "из коробки", IDE видит все таргеты.

**Минусы**: переменные и настройки вашего проекта утекают в зависимость и могут сломать её сборку; глобальные переменные загрязняют namespace; модули становятся тесно связанными. Также переменные, заданные в подкаталоге, локальны для его скоупа, если не пробросить их через `PARENT_SCOPE`, что часто приводит к путанице.

---

## Подход 2: `FetchContent` с локальным путём

Более элегантный способ — использовать FetchContent, но указать на локальную папку вместо Git URL. `FETCHCONTENT_SOURCE_DIR_<uppercaseName>` позволяет переопределить декларацию и использовать содержимое по указанному локальному пути вместо скачивания.

Два варианта:

**Вариант A — через переменную кеша (рекомендуемый):**

```cmake
# my-project/CMakeLists.txt
cmake_minimum_required(VERSION 3.24)
project(my-project LANGUAGES CXX)

include(FetchContent)

# Объявляем "нормально" — с Git URL (для CI/новых разработчиков)
FetchContent_Declare(
  lib-network
  GIT_REPOSITORY https://github.com/yourorg/lib-network.git
  GIT_TAG        v2.3.1
)

FetchContent_Declare(
  lib-protocol
  GIT_REPOSITORY https://github.com/yourorg/lib-protocol.git
  GIT_TAG        v1.0.0
)

FetchContent_MakeAvailable(lib-network lib-protocol)

add_executable(myapp src/main.cpp)
target_link_libraries(myapp PRIVATE network::network protocol::protocol)
```

А разработчик, работающий с локальными соседними каталогами, просто передаёт переменную:

```bash
cmake -B build \
  -DFETCHCONTENT_SOURCE_DIR_LIB-NETWORK=../lib-network \
  -DFETCHCONTENT_SOURCE_DIR_LIB-PROTOCOL=../lib-protocol
```

**Вариант B — через CMake Presets (идеальный для команды):**

```jsonc
// my-project/CMakePresets.json
{
  "version": 6,
  "configurePresets": [
    {
      "name": "dev-local",
      "displayName": "Dev: local sibling dirs",
      "generator": "Ninja",
      "binaryDir": "${sourceDir}/build",
      "cacheVariables": {
        "FETCHCONTENT_SOURCE_DIR_LIB-NETWORK": "${sourceDir}/../lib-network",
        "FETCHCONTENT_SOURCE_DIR_LIB-PROTOCOL": "${sourceDir}/../lib-protocol",
        "CMAKE_CXX_COMPILER_LAUNCHER": "ccache"
      }
    },
    {
      "name": "ci",
      "displayName": "CI: fetch from Git",
      "generator": "Ninja",
      "binaryDir": "${sourceDir}/build"
      // FetchContent скачает из Git — переменные не заданы
    }
  ]
}
```

```bash
# Разработчик с локальными каталогами:
cmake --preset dev-local

# CI без локальных каталогов:
cmake --preset ci
```

**Это лучший паттерн**: когда `FETCHCONTENT_SOURCE_DIR` задана, никакие шаги загрузки и обновления не выполняются — CMake просто использует существующие исходники по указанному пути, что даёт разработчикам возможность свободно редактировать зависимость без конфликтов с системой сборки.

---

## Подход 3: `find_package()` + предварительная установка

Для больших проектов с CMake 3.24+ — наиболее "чистый" подход, разделяющий сборку и потребление:

```cmake
# lib-network/CMakeLists.txt — должен уметь устанавливаться
cmake_minimum_required(VERSION 3.24)
project(lib-network VERSION 2.3.1 LANGUAGES CXX)

add_library(network src/network.cpp)
add_library(network::network ALIAS network)

target_include_directories(network
  PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:include>
)

# Установка и экспорт конфигурации
include(GNUInstallDirs)
install(TARGETS network EXPORT networkTargets)
install(EXPORT networkTargets
  NAMESPACE network::
  DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/network
)
install(DIRECTORY include/ DESTINATION ${CMAKE_INSTALL_INCLUDEDIR})

# Генерация Config файла
include(CMakePackageConfigHelpers)
configure_package_config_file(
  cmake/networkConfig.cmake.in
  ${CMAKE_CURRENT_BINARY_DIR}/networkConfig.cmake
  INSTALL_DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/network
)
install(FILES ${CMAKE_CURRENT_BINARY_DIR}/networkConfig.cmake
  DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/network
)
```

Сначала собираем и устанавливаем зависимость в общий prefix:

```bash
# Собираем lib-network в общий prefix
cmake -B ../lib-network/build -S ../lib-network \
  -DCMAKE_INSTALL_PREFIX=/workspace/local-install
cmake --build ../lib-network/build
cmake --install ../lib-network/build
```

Потом используем через `find_package`:

```cmake
# my-project/CMakeLists.txt
cmake_minimum_required(VERSION 3.24)
project(my-project LANGUAGES CXX)

find_package(network 2.3.1 REQUIRED)

add_executable(myapp src/main.cpp)
target_link_libraries(myapp PRIVATE network::network)
```

```bash
cmake -B build -DCMAKE_PREFIX_PATH=/workspace/local-install
```

---

## Подход 4: Dependency Provider (CMake 3.24+) — универсальный мост

Это самый мощный подход: при вызове `find_package()` или `FetchContent_MakeAvailable()` запрос может быть перенаправлен в Dependency Provider, который сам решает, как удовлетворить зависимость.

```cmake
# my-project/cmake/local_provider.cmake
# Этот файл подключается через CMAKE_PROJECT_TOP_LEVEL_INCLUDES

# Маппинг: имя пакета → локальный путь
set(_LOCAL_DEPS_MAP
  "lib-network|${CMAKE_CURRENT_LIST_DIR}/../../lib-network"
  "lib-protocol|${CMAKE_CURRENT_LIST_DIR}/../../lib-protocol"
)

macro(local_dependency_provider METHOD DEP_NAME)
  # Только для FIND_PACKAGE запросов
  if("${METHOD}" STREQUAL "FIND_PACKAGE")
    foreach(_entry IN LISTS _LOCAL_DEPS_MAP)
      string(REPLACE "|" ";" _parts "${_entry}")
      list(GET _parts 0 _name)
      list(GET _parts 1 _path)

      if("${DEP_NAME}" STREQUAL "${_name}" AND EXISTS "${_path}/CMakeLists.txt")
        # Превращаем find_package в add_subdirectory
        message(STATUS "Provider: resolving ${_name} from ${_path}")
        add_subdirectory("${_path}" "${CMAKE_BINARY_DIR}/_deps/${_name}-build")
        # Сообщаем FetchContent, что зависимость уже подключена
        set(${DEP_NAME}_FOUND TRUE)
        return()
      endif()
    endforeach()
  endif()
  # Если не нашли локально — CMake вызовет стандартный find_package
endmacro()

cmake_language(
  SET_DEPENDENCY_PROVIDER local_dependency_provider
  SUPPORTED_METHODS FIND_PACKAGE
)
```

Потребитель пишет идиоматичный CMake:

```cmake
# my-project/CMakeLists.txt — чистый и переносимый
cmake_minimum_required(VERSION 3.24)
project(my-project LANGUAGES CXX)

find_package(lib-network REQUIRED)
find_package(lib-protocol REQUIRED)

add_executable(myapp src/main.cpp)
target_link_libraries(myapp PRIVATE network::network protocol::protocol)
```

```bash
# Локально — провайдер подставляет соседние каталоги
cmake -B build -DCMAKE_PROJECT_TOP_LEVEL_INCLUDES=cmake/local_provider.cmake

# CI — стандартный find_package ищет в системе / vcpkg / Conan
cmake -B build -DCMAKE_TOOLCHAIN_FILE=$VCPKG_ROOT/scripts/buildsystems/vcpkg.cmake
```

---

## Подход 5: Гибридный — FetchContent + find_package (best practice)

Начиная с CMake 3.24, FetchContent может сначала попытаться найти пакет через `find_package`, и только при неудаче — скачать и собрать из исходников:

```cmake
# my-project/CMakeLists.txt
cmake_minimum_required(VERSION 3.24)
project(my-project LANGUAGES CXX)

include(FetchContent)

FetchContent_Declare(
  lib-network
  GIT_REPOSITORY https://github.com/yourorg/lib-network.git
  GIT_TAG        v2.3.1
  FIND_PACKAGE_ARGS NAMES network  # ← сначала попробовать find_package!
)

FetchContent_Declare(
  lib-protocol
  GIT_REPOSITORY https://github.com/yourorg/lib-protocol.git
  GIT_TAG        v1.0.0
  FIND_PACKAGE_ARGS NAMES protocol
)

FetchContent_MakeAvailable(lib-network lib-protocol)

add_executable(myapp src/main.cpp)
target_link_libraries(myapp PRIVATE network::network protocol::protocol)
```

Порядок приоритетов при вызове `FetchContent_MakeAvailable`:

1. Если задан `FETCHCONTENT_SOURCE_DIR_LIB-NETWORK` → берёт локальный путь
2. Если `FIND_PACKAGE_ARGS` задан и `find_package(network)` успешен → использует системную/vcpkg версию
3. Иначе → клонирует из Git и делает `add_subdirectory`

---

## Сравнительная таблица## Что я рекомендую для вас

Раз вы ориентируетесь на CMake 3.24+, лучшая стратегия — **подход 5 (гибридный FetchContent + FIND_PACKAGE_ARGS)** в связке с **CMakePresets.json**:

**В CMakeLists.txt** — пишете чистый, переносимый код с `FetchContent_Declare(..., FIND_PACKAGE_ARGS ...)`. Никаких хардкодов путей.

**В CMakePresets.json** — для каждого сценария свой пресет: `dev-local` с `FETCHCONTENT_SOURCE_DIR_*` на соседние каталоги, `ci` без переопределений (тянет из Git), `prod` с vcpkg toolchain (бинарники).

Таким образом каждый разработчик работает со своими локальными ветками соседних проектов, CI всё тянет из репозиториев с фиксированными тегами, а production-сборка использует предкомпилированные бинарники из vcpkg — и всё это без единого изменения в `CMakeLists.txt`.

Хотите, могу собрать готовый шаблон проекта с этой структурой?
