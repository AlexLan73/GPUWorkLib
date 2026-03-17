Кодо! Ты супер-оркестратор для GPUWorkLib проекта. Твоя задача: создать и запустить цепочку sub-агентов для полного цикла разработки нового ROCm-модуля.

---

## PROJECT_CONTEXT

- **Платформа**: AMD RX 9070 (Navi 48), ROCm 7.2+, HIP kernels
- **Архитектура**: Ref03 — 6-слойная модель (`Doc_Addition/PLAN/Ref03_Unified_Architecture.md`)
- **Структура модуля** (как в `modules/fft_func` или `modules/statistics`):
  ```
  modules/NEW_MODULE/
  ├── CMakeLists.txt
  ├── include/               # .hpp заголовки
  ├── src/                   # .cpp реализации
  ├── kernels/               # .cl/.hip kernel-файлы
  ├── operations/            # Concrete Ops (Слой 5)
  ├── steps/                 # IPipelineStep (Слой 6)
  └── tests/                 # *.hpp тесты + all_test.hpp + README.md
  ```
- **Интеграция**: Только через DrvGPU (`IBackend`, `GpuContext`). Никаких новых синглтонов.
- **Память**: Через DrvGPU memory management (не hipMallocManaged напрямую).
- **Логи/консоль**: Только `console_output` (3-arg: `Print(gpu_id, module, message)`).
- **Профилирование**: Только `GPUProfiler` (`SetGPUInfo()` → `Start()` → `PrintReport()`/`ExportMarkdown()`).
- **Вывод данных** в тестах: через `GPUProfiler.PrintReport()` / `ExportMarkdown()` / `ExportJSON()`. ЗАПРЕЩЕНО: `GetStats()` + цикл + con.Print.
- **Стиль написание программы** стиль написание программы ООП, SOLID, GRASP, GoF
---

## WORKFLOW

Запускай строго по шагам. После каждого — `cmake --build` + проверка. Если fail → resume agent и fix.

---

### Шаг 1 — ModuleWriter агент

Создай скелет модуля по описанию пользователя.

**Алгоритм:**
1. Прочитай `Doc_Addition/PLAN/Ref03_Unified_Architecture.md` — понять 6-слойную модель
2. Прочитай ближайший по смыслу готовый модуль как референс (statistics → `modules/statistics/`, vector_algebra → `modules/vector_algebra/`)
3. Создай структуру по Ref03:
   - **Слой 1**: `GpuContext` per-module
   - **Слой 2**: `IGpuOperation` интерфейс
   - **Слой 3**: `GpuKernelOp` base (hiprtc compile)
   - **Слой 4**: `BufferSet<N>` (enum индексы, compile-time)
   - **Слой 5**: Concrete Op-классы (один класс — один файл в `operations/`)
   - **Слой 6**: Facade + Strategy (тонкий фасад, `IPipelineStep` в `steps/`)
4. Создай `CMakeLists.txt` с `if(NOT ROCM_ENABLED) return() endif()`
5. Добавь `target_compile_definitions(... PUBLIC ENABLE_ROCM=1)`
6. Добавь `add_subdirectory` в корневой `CMakeLists.txt`

**Результат**: `"Модуль создан: [список файлов] + summary архитектуры"`

---

### Шаг 2 — KernelOptimizer агент (`/optimizer`)

Оптимизируй все HIP kernels нового модуля.

**Алгоритм** — полный алгоритм описан в `/optimizer`, ключевые критерии:
1. **Приоритет 1**: Минимальное время выполнения на GPU
2. **Приоритет 2**: Надёжность (корректность результата при любых входных данных)

**Что проверить:**
- Workgroup size (оптимальный для gfx1201 Navi 48)
- Коалесцентный доступ к памяти
- LDS/shared memory для matrix/FFT операций
- Unroll, vectorize (float2/float4 где возможно)
- Отсутствие лишних CPU↔GPU копий в hot path

**Тестируй** на реалистичных размерах данных (как в существующих тестах модуля).

**Результат**: `"Kernels оптимизированы: before/after timing + рекомендации"`

---

### Шаг 3 — CodeReviewer агент (`/review`)

Полный ревью модуля — полный алгоритм описан в `/review`, ключевые проверки для GPUWorkLib:

**Архитектура Ref03:**
- [ ] Соответствие 6-слойной модели
- [ ] Один класс — один файл (Ops в `operations/`, Steps в `steps/`)
- [ ] `BufferSet<N>` вместо raw `void*` полей
- [ ] `GpuContext` per-module (не глобальный)
- [ ] Facade API не меняется (Python bindings не ломаются)

**GPU / ROCm:**
- [ ] Консоль только через `console_output` (`Print(gpu_id, module, msg)`)
- [ ] Профилирование только через `GPUProfiler` (`SetGPUInfo()` перед `Start()`)
- [ ] Нет blocking-вызовов в hot path
- [ ] Multi-GPU safe (thread-safe между модулями)
- [ ] Нет лишних `hipMalloc/hipFree` в цикле

**Стиль:**
- [ ] Google C++ Style Guide + 2-пробельная табуляция
- [ ] CamelCase классы, snake_case методы, kConstant константы

**Результат**: `"Ревью passed/failed: [список issues + конкретные fixes с файл:строка]"`

---

### Шаг 4 — Tester агент

Напиши тесты по образцу `modules/strategies/tests/`.

**Структура тестов** (обязательно):
```
tests/
├── all_test.hpp                    # Точка входа — вызывается из src/main.cpp
├── README.md                       # Описание тестов
├── test_[module]_pipeline.hpp      # Функциональный тест pipeline
├── test_[module]_profiling.hpp     # Пошаговое профилирование (GPUProfiler)
├── test_[module]_benchmark.hpp     # Benchmark (время выполнения)
└── base_[module]_test.hpp          # Общие утилиты тестов
```

**Что тестировать:**
- Корректность результата (сравнение с эталоном — NumPy/SciPy через Python тест)
- Профилирование каждого шага через `GPUProfiler` (SetGPUInfo → Start → PrintReport)
- Benchmark: время на реалистичных данных
- Edge cases: минимальный/максимальный размер буферов

**Вызов из main**: добавить в `src/main.cpp`:
```cpp
#include "modules/NEW_MODULE/tests/all_test.hpp"
// ...
run_new_module_tests(backend);
```

**Результат**: `"Tests: X/Y passed, profiling OK, benchmark [время мс]"`

---

### Шаг 5 — PythonBindings агент

Создай Python биндинги через pybind11.

**Структура:**
- Файл: `python/py_NEW_MODULE_rocm.hpp`
- Регистрация: `register_new_module(m)` в `python/gpu_worklib_bindings.cpp` внутри `#if ENABLE_ROCM`
- Линковка: `python/CMakeLists.txt` под `if(ROCM_ENABLED AND ...)`

**Что экспортировать:**
- Основной Facade-класс (Слой 6)
- Конструктор с `IBackend*` / `ROCmGPUContext`
- Публичные методы обработки
- Properties для параметров

**Python тест**: создай `Python_test/new_module/test_new_module.py`
- Сравнение с NumPy/SciPy эталоном
- Запуск: `PYTHONPATH=build/python pytest Python_test/new_module/`

**Результат**: `"Python bindings: [список методов], тест passed/failed"`

---

### Шаг 6 — DocWriter агент (`/doc`)

Создай документацию модуля.

**Полный алгоритм описан в `/doc`**, ключевые разделы:
- `Doc/Modules/NEW_MODULE/Full.md` — полная документация (C++ API, Python API, pipeline-диаграмма, таблица тестов, kernels)
- `Doc/Modules/NEW_MODULE/API.md` — краткий API справочник
- `Doc/Modules/NEW_MODULE/Quick.md` — быстрый старт

Обновить `MemoryBank/MASTER_INDEX.md` — добавить строку модуля.

**Результат**: `"Документация создана: [список файлов]"`

---

## ФИНАЛ (Meta-Orchestrator)

После всех шагов:
1. `cmake --build` — финальная проверка сборки
2. Запустить тесты
3. Сформировать git commit message:
   ```
   feat(NEW_MODULE): add [название] module — [краткое описание]

   - Ref03 6-layer architecture
   - HIP kernels optimized for gfx1201
   - Python bindings (pybind11)
   - X/Y tests passed, benchmark [время]
   ```

**Готово**: `"Полный цикл завершён: модуль [название] production-ready"`

---

## ИСПОЛЬЗОВАНИЕ

Напиши описание модуля в свободной форме, например:
- `"capon beamforming для 16 антенн, 256 частот, входные данные complex float32"`
- `"FIR фильтр на GPU, 1024 отводов, batch processing"`
- `"матричное умножение complex float, до 512×512"`

Кодо запустит workflow с sequential-thinking на каждом сложном шаге! 🚀
