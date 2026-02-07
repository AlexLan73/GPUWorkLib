# Что сделать: C4996 (sscanf) и LNK2019 (spdlog/STL)

## 1. Предупреждение C4996 в `svm_capabilities.hpp` (строка 123, `sscanf`)

MSVC помечает `sscanf` как небезопасную функцию. Варианты:

### Вариант A — подавить только для этого места (быстро)

В **DrvGPU/memory/svm_capabilities.hpp** перед строкой с `sscanf` (строка 123) добавить:

```cpp
#ifdef _MSC_VER
#pragma warning(push)
#pragma warning(disable : 4996)
#endif
            if (sscanf(version_str, "OpenCL %d.%d", &major, &minor) == 2) {
#ifdef _MSC_VER
#pragma warning(pop)
#endif
```

(остальной блок с `caps.opencl_major_version` и т.д. без изменений)

### Вариант B — использовать безопасную функцию MSVC (только Windows)

Заменить вызов на:

```cpp
#ifdef _MSC_VER
            if (sscanf_s(version_str, "OpenCL %d.%d", &major, &minor) == 2)
#else
            if (sscanf(version_str, "OpenCL %d.%d", &major, &minor) == 2)
#endif
```

### Вариант C — убрать sscanf, парсить вручную (C++17, без предупреждений)

Добавить в начало файла: `#include <cstring>` и `#include <charconv>`.  
Заменить блок с `sscanf` на разбор через `std::strncmp(version_str, "OpenCL ", 7)` и два вызова `std::from_chars` для `major` и `minor` (как делалось ранее в отменённых изменениях).

---

## 2. Ошибки линковки LNK2019 (spdlog + символы `__std_find_*`)

Ошибки говорят о том, что:
- часть кода ожидает spdlog как **библиотеку** (drvgpu импортирует символы из DLL/lib);
- при этом в проекте включён **SPDLOG_HEADER_ONLY**;
- плюс spdlog собран с другой версией/настройками MSVC (символы `__std_find_last_trivial_1` и т.д. из STL).

### Что сделать по шагам

1. **Сделать везде один режим spdlog — только header-only, без линковки библиотеки.**

   В **DrvGPU/CMakeLists.txt**:
   - Найти блок «Линковка spdlog» (примерно строки 224–250).
   - Чтобы **всегда** использовать header-only и не тянуть spdlogd.lib:
     - либо закомментировать/удалить вызовы `target_link_libraries(drvgpu PUBLIC spdlog::spdlog)` (и аналоги для `spdlog_header_only`/fmt), оставив только `target_include_directories` и определение `SPDLOG_HEADER_ONLY`;
     - либо в начале этого блока добавить условие: если мы хотим header-only, не выполнять `target_link_libraries` для spdlog (например, задать переменную `DRVGPU_SPDLOG_HEADER_ONLY ON` и ветку «только include, без link»).

   Итог: drvgpu не должен линковать `spdlog::spdlog` или `spdlogd.lib`, если используется `SPDLOG_HEADER_ONLY`.

2. **Добавить FMT_HEADER_ONLY**, если используется fmt отдельно.

   В **DrvGPU/CMakeLists.txt** рядом с `target_compile_definitions(drvgpu PUBLIC SPDLOG_HEADER_ONLY)` добавить при необходимости:
   ```cmake
   target_compile_definitions(drvgpu PUBLIC FMT_HEADER_ONLY)
   ```
   (если в модулях уже есть `FMT_HEADER_ONLY`, лучше задать его и для drvgpu.)

3. **Пересобрать проект с нуля.**

   - Закрыть решение, удалить папку `build` (или содержимое).
   - Заново: Configure + Generate (CMake), затем сборка.

Если после этого LNK2019 на spdlog останутся — значит какой-то другой таргет (например, exe или модуль) всё ещё тянет spdlog как библиотеку; тогда в настройках этого таргета тоже нужно отключить линковку spdlog и оставить только header-only.

---

## Краткий чек-лист

- [ ] C4996: выбрать один из вариантов A/B/C и поправить `svm_capabilities.hpp`.
- [ ] CMake: при желании использовать только header-only — убрать/условить `target_link_libraries(drvgpu ... spdlog::spdlog)`, оставить include + `SPDLOG_HEADER_ONLY` (и при необходимости `FMT_HEADER_ONLY`).
- [ ] Полная пересборка (clean build).

После этого можно снова собрать и проверить линковку и отсутствие C4996.
