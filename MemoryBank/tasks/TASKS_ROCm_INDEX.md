# ROCm Tasks — Индекс и инструкции для ИИ-исполнителя

> **Дата**: 2026-02-23
> **Назначение**: Передать ИИ-исполнителю при портировании GPUWorkLib на ROCm

---

## Что передать ИИ-исполнителю

Скопируй и вставь в чат с ИИ:

---

**Задача**: Портирование GPUWorkLib на ROCm. Выполняй таски по порядку из `MemoryBank/tasks/`.

**Памятка**:
- Если работаешь под **Windows** — пиши весь код + тесты. **Тестировать будем только под Linux** (Debian/Ubuntu с Radeon 9070 / MI100). ROCm не поддерживается на Windows.
- Тесты под `#if ENABLE_ROCM` — на Windows без ROCm не вызывать.
- Создавай тесты, но **не запускай** — запуск будет на Linux.

**Порядок тасков**:
1. [Task_00_DrvGPU.md](Task_00_DrvGPU.md) — ROCmBackend, rocm_core, HIPBuffer, test_rocm_backend
2. [Task_01_FFTProcessorROCm.md](Task_01_FFTProcessorROCm.md) — FFT на hipFFT
3. [Task_02_StatisticsProcessorROCm.md](Task_02_StatisticsProcessorROCm.md) — mean, median, variance, std (ROCm only)
4. [Task_03_SpectrumProcessorROCm.md](Task_03_SpectrumProcessorROCm.md) — поиск максимумов спектра
5. [Task_04_FiltersROCm.md](Task_04_FiltersROCm.md) — FIR, IIR
6. [Task_05_LchFarrowROCm.md](Task_05_LchFarrowROCm.md) — LCH Farrow
7. [Task_06_FormSignalGeneratorROCm.md](Task_06_FormSignalGeneratorROCm.md) — генератор формы сигнала
8. [Task_07_HeterodyneProcessorROCm.md](Task_07_HeterodyneProcessorROCm.md) — гетеродин
9. [Task_08_ZeroCopy.md](Task_08_ZeroCopy.md) — опционально: OpenCL↔ROCm без копирования
10. [Task_09_HybridBackend.md](Task_09_HybridBackend.md) — опционально: OPENCLandROCm режим
11. [Task_10_VectorAlgebraCholesky.md](Task_10_VectorAlgebraCholesky.md) — Cholesky Inverter (vector_algebra)
12. [Task_11_VectorAlgebraCholesky_v2.md](Task_11_VectorAlgebraCholesky_v2.md) — vector_algebra v2: RAII CholeskyResult, SymmetrizeMode, hiprtc, benchmarks

**Референсы**:
- [PLAN_ROCm_DrvGPU_Full.md](PLAN_ROCm_DrvGPU_Full.md) — полный план
- [ROCm_Setup_Instructions.md](../../ROCm_Setup_Instructions.md) — CMake, установка ROCm

**Начни с Task_00_DrvGPU.md.**

---

## Список тасков

| # | Файл | Описание | Обязательный |
|---|------|----------|--------------|
| 00 | [Task_00_DrvGPU.md](Task_00_DrvGPU.md) | ROCmBackend, rocm_core, HIPBuffer, тесты | Да |
| 01 | [Task_01_FFTProcessorROCm.md](Task_01_FFTProcessorROCm.md) | FFT на hipFFT | Да |
| 02 | [Task_02_StatisticsProcessorROCm.md](Task_02_StatisticsProcessorROCm.md) | mean, median, variance, std | Да |
| 03 | [Task_03_SpectrumProcessorROCm.md](Task_03_SpectrumProcessorROCm.md) | Поиск максимумов спектра | Да |
| 04 | [Task_04_FiltersROCm.md](Task_04_FiltersROCm.md) | FIR, IIR фильтры | Да |
| 05 | [Task_05_LchFarrowROCm.md](Task_05_LchFarrowROCm.md) | LCH Farrow | Да |
| 06 | [Task_06_FormSignalGeneratorROCm.md](Task_06_FormSignalGeneratorROCm.md) | Генератор формы сигнала | Да |
| 07 | [Task_07_HeterodyneProcessorROCm.md](Task_07_HeterodyneProcessorROCm.md) | Гетеродин | Да |
| 08 | [Task_08_ZeroCopy.md](Task_08_ZeroCopy.md) | ZeroCopy OpenCL↔ROCm | Опционально |
| 09 | [Task_09_HybridBackend.md](Task_09_HybridBackend.md) | OPENCLandROCm режим | Опционально |
| 10 | [Task_10_VectorAlgebraCholesky.md](Task_10_VectorAlgebraCholesky.md) | Cholesky Inverter: vector_algebra (rocBLAS+rocSOLVER) | Да |
| 11 | [Task_11_VectorAlgebraCholesky_v2.md](Task_11_VectorAlgebraCholesky_v2.md) | vector_algebra v2: RAII CholeskyResult, SymmetrizeMode (Roundtrip/GpuKernel), hiprtc, benchmarks | Да |

---

## Зависимости между тасками

```
Task_00 (DrvGPU) ──────────────────────────────────────────────┐
    │                                                          │
    ├── Task_01 (FFTProcessor)                                 │
    ├── Task_02 (Statistics)                                   │
    ├── Task_03 (SpectrumMaxima) ──► Task_07 (Heterodyne)       │
    ├── Task_04 (Filters)                                      │
    ├── Task_05 (LchFarrow)                                    │
    ├── Task_06 (FormSignal)                                   │
    │                                                          │
    └── Task_08 (ZeroCopy) ──► Task_09 (HybridBackend)         │
                         └──► Task_10 (VectorAlgebra/Cholesky) ──► Task_11 (Cholesky GPU-only + Profiling)
```

---

## Чек-лист для исполнителя

После выполнения каждого таска:
- [ ] Код написан
- [ ] Тесты созданы и добавлены в all_test.hpp
- [ ] Компиляция без ошибок (на Windows: ENABLE_ROCM=OFF, ROCm код не компилируется)
- [ ] Тесты **не запускались** (запуск на Linux)
