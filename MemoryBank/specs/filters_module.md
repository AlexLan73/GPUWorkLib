# 📝 Filters Module — Спецификация

> **Модуль**: `Filters`
> **Статус**: ⚪ Planned
> **Платформы**: OpenCL, ROCm, HIP
> **Автор**: Alex
> **Создано**: 2026-02-09
> **Обновлено**: 2026-02-09

---

## 🎯 Назначение

Библиотека ЦОС-фильтров на GPU для обработки сигналов в реальном времени.

---

## 📋 Требования

### Функциональные
- [ ] REQ-001: FIR-фильтры (произвольные коэффициенты)
- [ ] REQ-002: IIR-фильтры (биквадратные секции)
- [ ] REQ-003: Адаптивные фильтры (LMS, NLMS, RLS)
- [ ] REQ-004: Децимация / Интерполяция
- [ ] REQ-005: Полифазные фильтры
- [ ] REQ-006: Overlap-Save / Overlap-Add для длинных FIR

### Нефункциональные
- [ ] NFR-001: Latency < 1ms для блоков до 4096 samples
- [ ] NFR-002: Поддержка streaming (непрерывный поток)

---

## 🔧 API (планируемый)

```cpp
class FIRFilter {
    void SetCoefficients(const float* coeffs, size_t count);
    void Process(const float* input, float* output, size_t samples);
};

class IIRFilter {
    void SetBiquadSections(const BiquadCoeffs* sections, size_t count);
    void Process(const float* input, float* output, size_t samples);
};

class AdaptiveFilter {
    void SetReference(const float* reference);
    void Adapt(const float* input, float* output, float* error);
};
```

---

## 📊 Метрики производительности

| Операция | Размер | Порядок | GPU | Время |
|----------|--------|---------|-----|-------|
| FIR | 1M samples | 128 taps | - | TBD |
| IIR | 1M samples | 8 biquads | - | TBD |

---

## 🔗 Зависимости

- `DrvGPU` — базовый драйвер
- `FFT Module` — для Overlap-Save/Add

---

## 📝 История изменений

| Дата | Автор | Изменение |
|------|-------|-----------|
| 2026-02-09 | Alex | Создание спецификации |
