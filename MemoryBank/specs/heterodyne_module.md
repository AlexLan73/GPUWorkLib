# 📝 Heterodyne Module — Спецификация

> **Модуль**: `Heterodyne`
> **Статус**: ⚪ Planned
> **Платформы**: OpenCL, ROCm, HIP
> **Автор**: Alex
> **Создано**: 2026-02-09
> **Обновлено**: 2026-02-09

---

## 🎯 Назначение

Гетеродин на GPU — перенос сигнала по частоте (frequency shifting).
Используется в SDR, радиоприёмниках, спектроанализаторах.

---

## 📋 Требования

### Функциональные
- [ ] REQ-001: Комплексный гетеродин (умножение на exp(j·2π·f·t))
- [ ] REQ-002: NCO (Numerically Controlled Oscillator)
- [ ] REQ-003: Настраиваемая частота и фаза
- [ ] REQ-004: Плавное изменение частоты (frequency sweep)
- [ ] REQ-005: Квадратурный детектор (I/Q демодуляция)
- [ ] REQ-006: SSB модуляция/демодуляция

### Нефункциональные
- [ ] NFR-001: Фазовая когерентность между блоками
- [ ] NFR-002: Spurious-free dynamic range > 90 dB
- [ ] NFR-003: Поддержка непрерывного потока (streaming)

---

## 🔧 API (планируемый)

```cpp
class Heterodyne {
    // Установить частоту переноса (Гц) и sample rate
    void SetFrequency(double frequency_hz, double sample_rate);

    // Установить начальную фазу (радианы)
    void SetPhase(double phase_rad);

    // Перенос вниз по частоте (входной сигнал реальный → комплексный)
    void MixDown(const float* input_real,
                 float* output_i, float* output_q,
                 size_t count);

    // Перенос вверх по частоте (комплексный → реальный)
    void MixUp(const float* input_i, const float* input_q,
               float* output_real,
               size_t count);

    // Комплексный гетеродин (complex × complex)
    void MixComplex(const float* in_i, const float* in_q,
                    float* out_i, float* out_q,
                    size_t count);
};

class NCO {
    void SetFrequency(double frequency_hz, double sample_rate);
    void Generate(float* cos_out, float* sin_out, size_t count);
};
```

---

## 🏗️ Архитектура

```
Input Signal x(t)
      │
      ▼
┌─────────────────────────────────────┐
│         GPU Kernel                  │
│                                     │
│   x(t) × exp(-j·2π·f₀·t)           │
│                                     │
│   cos_table[phase] → I component    │
│   sin_table[phase] → Q component    │
│                                     │
│   phase += delta_phase              │
└──────────────┬──────────────────────┘
               │
               ▼
      I(t), Q(t) — baseband signal
```

**Реализация NCO:**
- Direct Digital Synthesis (DDS)
- Taylor series или lookup table для sin/cos
- Фазовый аккумулятор с высокой точностью

---

## 📊 Метрики производительности

| Операция | Размер | GPU | Время |
|----------|--------|-----|-------|
| MixDown | 1M samples | - | TBD |
| NCO Generate | 1M samples | - | TBD |

---

## 🔗 Зависимости

- `DrvGPU` — базовый драйвер
- `Filters Module` — для LPF после mixing

---

## 📚 Ссылки

- [Direct Digital Synthesis](https://en.wikipedia.org/wiki/Direct_digital_synthesis)
- [NCO Design](https://www.analog.com/en/analog-dialogue/articles/all-about-direct-digital-synthesis.html)

---

## 📝 История изменений

| Дата | Автор | Изменение |
|------|-------|-----------|
| 2026-02-09 | Alex | Создание спецификации |
