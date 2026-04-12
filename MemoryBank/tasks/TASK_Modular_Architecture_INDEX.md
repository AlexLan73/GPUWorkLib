# 🏛️ Индекс задач: Модульная архитектура DSP-GPU

> **Проект**: Перенос GPUWorkLib в `github.com/dsp-gpu` (9 публичных репо)
> **Спецификация**: [`specs/modular_architecture_plan.md`](../specs/modular_architecture_plan.md)
> **Ревью**: [`specs/modular_architecture_plan_REVIEW_v2.md`](../specs/modular_architecture_plan_REVIEW_v2.md)
> **Версия плана**: v2 (после ревью 2026-04-11)
> **Статус**: IN_PROGRESS (Фаза 0 ✅, Фаза 1 в работе)

---

## Структура репо (итоговая, 9 штук)

```
github.com/dsp-gpu/
├── core              ← DrvGPU  [DspCore::DspCore + DspCore::TestUtils]
├── spectrum          ← fft_func + filters + lch_farrow
├── stats             ← statistics (welford, medians, radix sort)
├── signal_generators ← CW / LFM / Noise / Script / FormSignal
├── heterodyne        ← LFM Dechirp, NCO, MixDown/MixUp
├── linalg            ← vector_algebra + capon
├── radar             ← range_angle + fm_correlator
├── strategies        ← antenna pipeline
└── DSP               ← мета-репо (FetchContent, Python, Doc)
```

> ✅ Все 9 репо созданы на GitHub (2026-04-12, вручную)
> ✅ Doc/Architecture/ запушена в DSP репо

---

## Задачи по фазам

| Фаза | Файл задачи | Статус |
|------|-------------|--------|
| 0 — Аудит зависимостей | [`TASK_ModArch_Phase0_Audit.md`](TASK_ModArch_Phase0_Audit.md) | ✅ DONE |
| 1 — Скелет 9 репо | [`TASK_ModArch_Phase1_Skeleton.md`](TASK_ModArch_Phase1_Skeleton.md) | ⬜ BACKLOG |
| 2 — Копирование кода | [`TASK_ModArch_Phase2_Copy.md`](TASK_ModArch_Phase2_Copy.md) | ⬜ BACKLOG |
| 3 — CMake + namespace | [`TASK_ModArch_Phase3_CMake.md`](TASK_ModArch_Phase3_CMake.md) | ⬜ BACKLOG |
| **3b — Python migration** | [`TASK_ModArch_Phase3b_Python.md`](TASK_ModArch_Phase3b_Python.md) | ⬜ BACKLOG |
| 4 — Тестирование GPU | [`TASK_ModArch_Phase4_Test.md`](TASK_ModArch_Phase4_Test.md) | ⬜ BACKLOG |

---

## Зависимости между фазами

```
Фаза 0 (аудит) ✅
  → Фаза 1 (скелет CMake)
    → Фаза 2 (копирование кода)
      → Фаза 3 (CMake + namespace dsp::)
        → Фаза 3b (Python migration — 8 .pyd)
          → Фаза 4 (GPU тесты на Debian)
```

---

## Ключевые решения (зафиксированы после ревью)

| Тема | Решение |
|------|---------|
| git history | Без истории — копируем как есть |
| OpenCL `.cl` kernels | Не переносим — только стыковка данных в GPU |
| ROCm `.hip` kernels | В `kernels/rocm/` — PRIVATE директория каждого репо |
| `test_utils` | Отдельная target `DspCore::TestUtils` (не в публичном API) |
| `pybind11` | pip + `python -m pybind11 --cmakedir` (как сейчас) |
| `namespace dsp::` | Добавляем в Фазе 3, не откладываем |
| Diamond dep | Централизованный `FetchContent_Declare` в `DSP/CMakeLists.txt` |
| `statistics` | Отдельный репо `stats` (не в `spectrum`) |
| `signal` | Разделён: `signal_generators` + `heterodyne` |
| `Doc_Addition/` | Переносим в `DSP/Doc/addition/`, потом чистим |
| Windows / nvidia ветка | Только аналитические Python-модели, C++ только ROCm |
| **Python API** | **gpuworklib.pyd → 8 отдельных .pyd + shim gpuworklib.py** |
| **find_package** | **Только lowercase: `find_package(hip)` не `find_package(HIP)`** |
| **MemoryBank** | **Остаётся в GPUWorkLib, в DSP не переносится** |
| **Рабочая папка** | **`E:\DSP-GPU\` (Windows), `~/dsp-gpu/` (Debian)** |
| **Benchmark baseline** | **Устанавливается на gfx1201, не сравниваем с GPUWorkLib** |

---

*Создан: 2026-04-12 | Обновлён: 2026-04-12 (Phase 0 DONE, добавлена Phase 3b, исправлены ключевые решения) | Автор: Кодо*
