# Review: Разногласия и проблемы CMake LibGPU Integration

> **Дата ревью:** 2026-03-29
> **Файлы:** `cmake_libgpu_research_agent.md` + `cmake_libgpu_integration_plan.md`
> **Статус:** ✅ Все решения приняты (2026-03-29), integration_plan.md обновлён

---

## 🔴 Критические проблемы (3)

### 1. Export Name Mismatch

**Где:** `integration_plan.md`, строка 571

**Суть:** Корневой CMakeLists.txt LibGPU делает:
```cmake
install(EXPORT LibGPUTargets NAMESPACE LibGPU:: ...)
```
Но все 11 модулей в реальном коде экспортируют **другое** имя:
```cmake
install(TARGETS ${MODULE_NAME} EXPORT GPUWorkLibTargets ...)
```

`LibGPUTargets` != `GPUWorkLibTargets` -> при `cmake --install` экспорт `LibGPUTargets` будет пуст.

**Варианты решения:**
- **A)** Переименовать export в модулях: `GPUWorkLibTargets` -> `LibGPUTargets`
- **B)** В корневом CMake LibGPU использовать `GPUWorkLibTargets`:
  ```cmake
  install(EXPORT GPUWorkLibTargets NAMESPACE LibGPU:: DESTINATION lib/cmake/LibGPU)
  ```
- **C)** Оставить как есть для фазы 1 (add_subdirectory не использует export), исправить перед фазой 2

> **Решение Alex'а:** Вариант B — НЕ менять export name в модулях. Корневой CMake использует `install(EXPORT GPUWorkLibTargets NAMESPACE GPUWorkLib::)`. Проект НЕ переименовывается.

---

### 2. Triple Namespace Confusion

**Где:** `integration_plan.md`, строки 789-799

**Суть:** В проекте сосуществуют 3 namespace'а:

| Где | Namespace | Пример |
|-----|-----------|--------|
| DrvGPU ALIAS | `DrvGPU::` | `DrvGPU::drvgpu` |
| Модули ALIAS | `GPUWorkLib::` | `GPUWorkLib::fft_func` |
| ~~Export в плане~~ | ~~`LibGPU::`~~ | ~~`LibGPU::fft_func`~~ |

**Фаза 1** (add_subdirectory): закрытый проект видит ALIAS'ы — `DrvGPU::drvgpu` + `GPUWorkLib::fft_func`. Работает.

**Фаза 2** (find_package): targets будут `GPUWorkLib::*` (единый namespace через export).

> **Решение Alex'а:** LibGPU — только название директории! Код НЕ переписывается. Namespace `LibGPU::` убран из плана. Остаются `DrvGPU::` + `GPUWorkLib::` (ALIAS'ы как в GPUWorkLib). При find_package — `GPUWorkLib::*`.

---

### 3. GPUWorkLibConfig.cmake.in — нет транзитивных зависимостей

**Где:** `integration_plan.md`, строки 714-723

**Суть:** Config-файл был минималистичный, без `find_dependency()`.

> **Решение Alex'а:** Взять полный вариант из `research_agent.md` с find_dependency() для Threads, OpenCL, hip, hipfft, rocblas, rocsolver. Обновлено в integration_plan.md.

---

## 🟡 Важные замечания (6)

### 4. Kernel-файлы: нет описания runtime path

> **Решение Alex'а:** Согласен, не должно быть абсолютных адресов. Добавлена секция "Runtime Assets" в план. Для фазы 1 (add_subdirectory) работает без изменений. Для фазы 2 — TODO.

---

### 5. configGPU.json — рабочая директория

> **Решение Alex'а:** Скопировать из LibGPU/configGPU.json, потом на сервере отрегулировать. Добавлено в план.

---

### 6. OpenCL-only сборка невозможна

> **Решение Alex'а:** Если нет ROCm — просто выходим из сборки. Если есть ROCm то и OpenCL есть. Добавлено в "Решения Alex'а" в плане.

---

### 7. Doc_Addition/PLAN/ в whitelist — утечка внутренних планов?

> **Решение Alex'а:** Не нужно передавать `Doc_Addition/PLAN/`. Добавлено в EXCLUDE_PATTERNS.

---

### 8. DrvGPU не имеет install() правил

> **Решение Alex'а:** Добавить `install(TARGETS drvgpu EXPORT GPUWorkLibTargets ...)` в DrvGPU/CMakeLists.txt. Добавлено в TODO в плане.

---

### 9. Research vs Plan — противоречие в рекомендации

> **Решение Alex'а:** Хорошо, добавить пометку. Добавлена в research_agent.md.

---

## 🟢 Рекомендации (4)

### 10. Ninja вместо "Unix Makefiles"

В CMakePresets все пресеты используют `"generator": "Unix Makefiles"`. Ninja значительно быстрее для параллельной сборки.

**Предложение:** Добавить альтернативные пресеты `linux-debug-ninja`, `linux-release-ninja` с `"generator": "Ninja"`.

---

### 11. Rollback-стратегия отсутствует

Что делать если sync сломал LibGPU?

**Предложение:** Добавить секцию:
```bash
# Откат LibGPU к предыдущему тегу:
cd LibGPU && git reset --hard v1.1.0
# В закрытом проекте:
cd libs/LibGPU && git checkout v1.1.0
```

---

### 12. CI/CD валидация LibGPU после синхронизации

**Предложение:** Минимальный smoke-тест после sync:
```bash
cd LibGPU
cmake --preset localdev
cmake --build build-localdev -j$(nproc)
./build-localdev/GPUWorkLib  # запуск C++ тестов
```

---

### 13. VERSION файл vs project(VERSION)

Текущий подход (читать из файла) работает. Но `project(VERSION X.Y.Z)` автоматически даёт `PROJECT_VERSION_MAJOR/MINOR/PATCH` — удобнее для `write_basic_package_version_file()`.

---

## Сводка

| Категория | 🔴 | 🟡 | 🟢 |
|-----------|---|---|---|
| CMake technical | 3 | 3 | 2 |
| Безопасность | 0 | 1 | 0 |
| Полнота | 0 | 2 | 2 |
| **ИТОГО** | **3** | **6** | **4** |

/5/5/5/5/5/5/5/5/5/5/5/5/5/5/5/5/5/5/5/5/
### Ответы
## 🔴 Критические проблемы (3)

### 1. Export Name Mismatch
> **Решение Alex'а:** Переименовать export в модулях: `GPUWorkLibTargets` -> `LibGPUTargets`

### 2. Triple Namespace Confusion
МЫ создаем проект LibGPU - только для передачи/обмена кода нашего открытого проекта GPUWorkLib с закрыты локальным проектом.
поэтому считаю -> | Export в плане | `LibGPU::` | `LibGPU::fft_func` | это не правильно!! в LibGPU код не переписываем!!


### 3. LibGPUConfig.cmake.in — нет транзитивных зависимостей

**Где:** `integration_plan.md`, строки 714-723

В `research_agent.md` (строки 217-235) правильный вариант уже написан:
```cmake
@PACKAGE_INIT@
include(CMakeFindDependencyMacro)
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
include("${CMAKE_CURRENT_LIST_DIR}/LibGPUTargets.cmake")
check_required_components(LibGPU)
```

**Варианты решения:**
- **A)** Заменить Config в `integration_plan` на полный из `research_agent`
вроде этот правильно!
---

## 🟡 Важные замечания (6)

### 4. Kernel-файлы: нет описания runtime path

Модули задают абсолютный путь к .cl/.hip файлам через `CMAKE_CURRENT_SOURCE_DIR/kernels`. При add_subdirectory — работает. При install + find_package (фаза 2) — kernels нужно устанавливать и менять путь.

**Нужно:** Добавить секцию "Runtime Assets" в план.

> **Решение Alex'а:** Согласен не должно быть абсолютных адресв


### 5. configGPU.json — рабочая директория

DrvGPU загружает `configGPU.json` из CWD (`GPUConfig::GetInstance().LoadOrCreate("configGPU.json")`). Закрытому проекту нужно обеспечить наличие этого файла в рабочей директории.

**Нужно:** Документировать в плане: "Закрытый проект должен иметь configGPU.json в рабочей директории, либо скопировать из LibGPU/configGPU.json".

> **Решение Alex'а:** скопировать из LibGPU/configGPU.json" - потом на сервере отрегулируем

---

### 6. OpenCL-only сборка невозможна

Все 11 модулей делают `if(NOT ROCM_ENABLED) return()`. Без ROCm собирается только голый DrvGPU.

Это осознанное ограничение (закрытый проект = Linux + AMD GPU), но стоит явно задокументировать: "LibGPU требует ROCm 7.2+. OpenCL-only сборка не поддерживается."

> **Решение Alex'а:**  СЧИТАЮ что если нет ROCm - просто выходим из сборки. Если есть ROCm то и OpenCL есть

---

### 7. Doc_Addition/PLAN/ в whitelist — утечка внутренних планов?

`SYNC_DIRS` включает `Doc_Addition/` -> содержит `PLAN/` с внутренними планами рефакторинга (Ref01, Ref02, Ref03...). Нужны ли они в закрытом контуре?

**Варианты:**
- **A)** Добавить `Doc_Addition/PLAN/` в EXCLUDE_PATTERNS
- **B)** Оставить — планы не содержат секретов, полезны для понимания архитектуры

> **Решение Alex'а:** ты права это не нужно `Doc_Addition/PLAN/` передавать

---

### 8. DrvGPU не имеет install() правил

В `DrvGPU/CMakeLists.txt` нет `install(TARGETS drvgpu EXPORT GPUWorkLibTargets ...)`. Для фазы 2 (find_package) нужно будет добавить.

**Нужно:** Для фазы 1 не критично. Пометить как TODO для фазы 2.

> **Решение Alex'а:** Добавь

---

### 9. Research vs Plan — противоречие в рекомендации

`research_agent.md` раздел 7 рекомендует **FetchContent** как основной метод.
`integration_plan.md` выбрал **git submodule**.

Предполагаю это осознанное решение Alex'а. Стоит добавить пометку в research: "Итоговое решение — submodule, см. integration_plan.md".

> **Решение Alex'а:** Хорошо

---

## 🟢 Рекомендации (4)

### 10. Ninja вместо "Unix Makefiles"

В CMakePresets все пресеты используют `"generator": "Unix Makefiles"`. Ninja значительно быстрее для параллельной сборки.

**Предложение:** Добавить альтернативные пресеты `linux-debug-ninja`, `linux-release-ninja` с `"generator": "Ninja"`.
НЕ понимаю (( Ninja - да 
linux-debug-ninja -длинные названия ((
  думаю должны быть такие 
   - debug          - это общая отладка - файлы про сборке отрегудируем 
   - release        - это продакшен- файлы про сборке отрегудируем - минимальное кол-во кода
   - local-debug    - это полная сборка для окальной отладки
---

### 11. Rollback-стратегия отсутствует

Что делать если sync сломал LibGPU?

**Предложение:** Добавить секцию:
```bash
# Откат LibGPU к предыдущему тегу:
cd LibGPU && git reset --hard v1.1.0
# В закрытом проекте:
cd libs/LibGPU && git checkout v1.1.0
```
- Согласен!
---

### 12. CI/CD валидация LibGPU после синхронизации

**Предложение:** Минимальный smoke-тест после sync:
```bash
cd LibGPU
cmake --preset localdev
cmake --build build-localdev -j$(nproc)
./build-localdev/GPUWorkLib  # запуск C++ тестов
```
-ДА

---

### 13. VERSION файл vs project(VERSION)

Текущий подход (читать из файла) работает. Но `project(VERSION X.Y.Z)` автоматически даёт `PROJECT_VERSION_MAJOR/MINOR/PATCH` — удобнее для `write_basic_package_version_file()`.

-ДА