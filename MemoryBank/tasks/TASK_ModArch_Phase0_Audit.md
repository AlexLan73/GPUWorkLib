# 🔍 Фаза 0: Аудит зависимостей GPUWorkLib

> **Индекс**: [`TASK_Modular_Architecture_INDEX.md`](TASK_Modular_Architecture_INDEX.md)
> **Спецификация**: [`specs/modular_architecture_plan.md`](../specs/modular_architecture_plan.md) — раздел 7, Фаза 0
> **Статус**: ✅ DONE — завершено 2026-04-12
> **Платформа**: Windows (анализ файлов, GPU не нужен)
> **Результат**: `DSP/Doc/Architecture/dependencies.md` создан и запушен в `github.com/dsp-gpu/DSP`

---

## Результаты аудита

Аудит выполнен в сессии 2026-04-12 с помощью Explore-агента.
Граф зависимостей в разделе 2 плана **подтверждён** — расхождений не найдено.

### Реальные зависимости (target_link_libraries)

| Модуль | Зависит от | External SDK |
|--------|-----------|-------------|
| DrvGPU (core) | — | HIP, OpenCL, HSA, rocprim |
| fft_func (spectrum) | DrvGPU | hipFFT |
| filters (spectrum) | DrvGPU | hipFFT |
| lch_farrow (spectrum) | DrvGPU | hipFFT |
| statistics (stats) | DrvGPU | rocprim |
| signal_generators | DrvGPU + lch_farrow (spectrum) | hiprtc |
| heterodyne | DrvGPU + signal_generators + fft_func | — |
| vector_algebra (linalg) | DrvGPU | rocBLAS, rocSOLVER, hiprtc |
| capon (linalg) | DrvGPU + vector_algebra | rocBLAS |
| range_angle (radar) | DrvGPU + fft_func + statistics | — |
| fm_correlator (radar) | DrvGPU + fft_func | — |
| strategies | DrvGPU + spectrum + stats + signal_gen + heterodyne + linalg | — |

### Скрытые include-зависимости (проверено)

- `statistics/include/` — нет скрытых include из fft_func ✅
- `heterodyne/include/` — нет прямых include из signal_generators без CMake ✅
- `range_angle/include/` — нет include statistics без CMake ✅
- `signal_generators/include/` — есть include lch_farrow → через spectrum ✅

### Документация

- Создано: `E:\DSP-GPU\DSP\Doc\Architecture\dependencies.md` — граф Mermaid + таблицы
- Создано: `E:\DSP-GPU\DSP\Doc\Architecture\repo_map.md`
- Создано: `E:\DSP-GPU\DSP\Doc\Architecture\repo_structure.md`
- Создано: `E:\DSP-GPU\DSP\Doc\Architecture\README.md`
- Всё запушено в `github.com/dsp-gpu/DSP`

---

## Чеклист (выполнен)

### А. CMakeLists.txt каждого модуля

- [x] **A1** DrvGPU — HIP, OpenCL, HSA, rocprim
- [x] **A2** fft_func — DrvGPU + hipFFT
- [x] **A3** filters — DrvGPU + hipFFT
- [x] **A4** lch_farrow — DrvGPU + hipFFT
- [x] **A5** statistics — DrvGPU + rocprim (не тянет fft_func ✅)
- [x] **A6** signal_generators — DrvGPU + lch_farrow + hiprtc
- [x] **A7** heterodyne — DrvGPU + signal_generators + fft_func
- [x] **A8** vector_algebra — DrvGPU + rocBLAS + rocSOLVER + hiprtc
- [x] **A9** capon — DrvGPU + vector_algebra + rocBLAS
- [x] **A10** range_angle — DrvGPU + fft_func + statistics
- [x] **A11** fm_correlator — DrvGPU + fft_func
- [x] **A12** strategies — DrvGPU + spectrum + stats + signal_gen + heterodyne + linalg

### Б. Скрытые include-зависимости

- [x] **B1-B5** — проверены, нарушений нет

### В. Соответствие плану

- [x] **G1-G5** — `dependencies.md` создан, граф подтверждён, план не требует правок

---

## Definition of Done — ✅ ВЫПОЛНЕНО

- [x] `Doc/Architecture/dependencies.md` создан и заполнен
- [x] Граф в разделе 2 плана соответствует реальности
- [x] Можно запускать Фазу 1

---

*Создан: 2026-04-12 | Завершён: 2026-04-12 | Автор: Кодо*
