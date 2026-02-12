# 📋 Задачи по доработке API Refactoring (code review 2026-02-12)

> **Источник**: Code review изменений api_refactoring, batch, test_gpu_generator_integration  
> **План**: Cursor plan `profiler_md_api_docs_90da5ede` (разделы 4, 5)

---

## T-API-001: DriverType driver — заглушка и документация

**Проблема**: Параметр `DriverType driver` в `Process()` не используется — в шаблоне `(void)driver;`.

**Что сделать**:
- Документировать в `spectrum_maxima_finder.h`: «зарезервировано для ROCm, реализация — на днях»
- Добавить заглушку внутри метода `Process<T>`: проверка `driver`, при `DriverType::ROCM` — комментарий `// TODO: ROCm backend`
- В `spectrum_maxima_api_usage.md` описать и примеры выбора OpenCL vs ROCm (сейчас — заглушка)

**Файлы**: `spectrum_maxima_finder.h`, `spectrum_input_data.hpp`, `spectrum_maxima_api_usage.md`

**Приоритет**: Средний (описание и примеры сейчас, реализация — на днях)

---

## T-API-002: Удаление мёртвого кода в ProcessFromGPU

**Проблема**: `PrepareParams()` вызывается в `Process()` до `ProcessFromGPU()` и всегда заполняет `params_.nFFT`. Блок `if (params_.nFFT == 0) { CalculateFFTSize(); ... }` никогда не выполняется.

**Что сделать**: Удалить этот блок в `ProcessFromGPU()` (позже, после стабилизации).

**Файл**: `modules/fft_maxima/src/spectrum_maxima_finder.cpp`, строки ~1196–1204

**Приоритет**: Низкий (потом удалим)
