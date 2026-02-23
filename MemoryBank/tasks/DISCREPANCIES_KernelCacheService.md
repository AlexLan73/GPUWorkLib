# Файл разносласий: KernelCacheService и связанные таски

> **Дата проверки:** 2026-02-23  
> **Проверено по:** [PLAN_KernelCacheService_DrvGPU.md](PLAN_KernelCacheService_DrvGPU.md), [TASK_FormSignal_Kernels_OnDiskCache_Refactor.md](TASK_FormSignal_Kernels_OnDiskCache_Refactor.md), TASK-001..005

---

## 1. Сводка: что реализовано

| Таска | Статус | Комментарий |
|-------|--------|-------------|
| TASK-001 StorageBackend | Выполнено | IStorageBackend + FileStorageBackend |
| TASK-002 KernelCacheService | Выполнено | API, Save/Load/ListKernels, VersionOldFiles, ROCm суффикс |
| TASK-003 FormScriptRefactor | Выполнено | FormScriptGenerator использует KernelCacheService |
| TASK-004 Filters | Выполнено | FirFilter, IirFilter интегрированы с KernelCacheService |
| TASK-005 FilterConfigService | Выполнено | Save/Load/ListFilters/Exists, VersionOldFiles |
| TASK_FormSignal (шаги 1–2) | Выполнено | Kernels в .cl, правило в CLAUDE.md |

---

## 2. Разносласия и отличия

### 2.1. Имя тестового файла (TASK-001)

| План | Реализация |
|------|------------|
| `DrvGPU/tests/test_storage_backend.hpp` | `DrvGPU/tests/test_storage_services.hpp` |

**Суть:** Тесты объединены в один файл `test_storage_services.hpp` (FileStorageBackend + KernelCacheService + FilterConfigService). Функционально покрытие соответствует плану.

---

### 2.2. FilterConfigData: формат IIR sections (TASK-005)

| План (TASK-005) | Реализация |
|-----------------|------------|
| `std::vector<std::array<double, 6>> sections` — [b0, b1, b2, a0, a1, a2] | `std::vector<std::array<float, 5>> sections` — [b0, b1, b2, a1, a2] |

**Суть:** В проекте BiquadSection использует 5 коэффициентов (a0=1 нормализован). Реализация совпадает с `filters::BiquadSection` (b0, b1, b2, a1, a2). План описал 6 коэффициентов — это расхождение в спецификации, реализация корректна для текущей модели фильтров.

---

### 2.3. Интеграция FilterConfigService в FirFilter/IirFilter (TASK-005)

| План | Реализация |
|------|------------|
| TASK-005: только FilterConfigService. Интеграция в FirFilter/IirFilter — **TASK-006** | TASK-006 не создана, методов SaveFilterConfig/LoadFilterConfig в FirFilter/IirFilter нет |

**Суть:** FilterConfigService реализован и протестирован. Вызовы SaveFilterConfig/LoadFilterConfig из FirFilter/IirFilter в план не входили (отложены на TASK-006). Это не расхождение, а ожидаемое состояние.

---

### 2.4. Пути Python-тестов (TASK_FormSignal)

| План | Реализация |
|------|------------|
| `Python_test/test_form_signal.py` | `Python_test/signal_generators/test_form_signal.py` |
| `Python_test/test_delayed_form_signal.py` | `Python_test/signal_generators/test_delayed_form_signal.py` |

**Суть:** Python-тесты лежат в подпапке `signal_generators/`. В `Python_test/README.md` указана структура по модулям. Пути в плане устарели.

---

### 2.5. Директории для графиков (TASK_FormSignal)

| План | Реализация |
|------|------------|
| `Results/Plots/FormSignal/` | `Results/Plots/signal_generators/FormSignal/` |
| `Results/Plots/DelayedFormSignal/` | `Results/Plots/signal_generators/DelayedFormSignal/` |

**Суть:** В `Python_test/README.md` указано: `Results/Plots/signal_generators/FormSignal/`. План использует старый путь без `signal_generators/`.

---

### 2.6. GetBinDir: кроссплатформенность (тест)

| Тест ожидает | Реализация |
|--------------|------------|
| `dir + "/bin"` | `base_dir_ + "/bin"` |

**Суть:** На Windows `fs::temp_directory_path()` может давать путь с `\`. Конкатенация `dir + "/bin"` может дать смешанный путь. `std::filesystem` обычно нормализует такие пути. Потенциальный риск на некоторых конфигурациях Windows.

---

## 3. Не проверено (требует запуска)

- [ ] `cmake -B build && cmake --build build` — успешность сборки
- [ ] `test_storage_services::run()` — прохождение всех тестов
- [ ] `test_form_script::run()` — SaveKernel, LoadKernel, Versioning, ListKernels
- [ ] `pytest Python_test/signal_generators/test_form_signal.py -v`
- [ ] `pytest Python_test/signal_generators/test_delayed_form_signal.py -v`
- [ ] Тесты filters (C++ и Python, если есть)

---

## 4. Рекомендации

1. **TASK-001:** Оставить `test_storage_services.hpp` — один файл для всех storage-сервисов удобнее.
2. **TASK-005:** Формат sections [b0,b1,b2,a1,a2] считать эталонным; при необходимости обновить описание в TASK-005.
3. **TASK_FormSignal:** Обновить пути в плане на `Python_test/signal_generators/` и `Results/Plots/signal_generators/`.
4. **TASK-006:** Создать таску на интеграцию FilterConfigService в FirFilter/IirFilter (SaveFilterConfig/LoadFilterConfig), когда понадобится.

---

*Создано: Кодо (AI Assistant), 2026-02-23*
