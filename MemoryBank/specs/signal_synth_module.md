# 📝 Signal Synthesizer Module — Спецификация

> **Модуль**: `SignalSynth`
> **Статус**: ⚪ Planned
> **Платформы**: OpenCL, ROCm, HIP
> **Автор**: Alex
> **Создано**: 2026-02-09
> **Обновлено**: 2026-02-09

---

## 🎯 Назначение

Синтезатор сигналов на GPU — генерация тестовых и модулированных сигналов для SDR, тестирования, симуляции.

---

## 📋 Требования

### Функциональные

**Базовые сигналы:**
- [ ] REQ-001: Синусоида (с настройкой частоты, амплитуды, фазы)
- [ ] REQ-002: Меандр (square wave)
- [ ] REQ-003: Пилообразный (sawtooth)
- [ ] REQ-004: Треугольный (triangle)
- [ ] REQ-005: Шум (белый, розовый, Гауссов)
- [ ] REQ-006: Chirp (линейная/логарифмическая развёртка)

**Модулированные сигналы:**
- [ ] REQ-007: AM (Amplitude Modulation)
- [ ] REQ-008: FM (Frequency Modulation)
- [ ] REQ-009: PM (Phase Modulation)
- [ ] REQ-010: PSK (Phase Shift Keying: BPSK, QPSK, 8PSK)
- [ ] REQ-011: QAM (16-QAM, 64-QAM, 256-QAM)
- [ ] REQ-012: OFDM (с настраиваемыми поднесущими)

**Специальные:**
- [ ] REQ-013: Импульсы (прямоугольный, Гауссов, sinc)
- [ ] REQ-014: Псевдослучайные последовательности (PN, Gold, m-sequences)

### Нефункциональные
- [ ] NFR-001: Sample rate до 100 MHz (GPU-bound)
- [ ] NFR-002: Поддержка непрерывной генерации (streaming)
- [ ] NFR-003: Детерминированность (seed для RNG)

---

## 🔧 API (планируемый)

```cpp
// Базовые генераторы
class SineGenerator {
    void SetParams(double freq_hz, double amplitude, double phase_rad, double sample_rate);
    void Generate(float* output, size_t count);
};

class NoiseGenerator {
    enum Type { WHITE, PINK, GAUSSIAN };
    void SetType(Type type);
    void SetSeed(uint64_t seed);
    void Generate(float* output, size_t count);
};

class ChirpGenerator {
    void SetParams(double f_start, double f_end, double duration_sec, double sample_rate);
    void Generate(float* output, size_t count);
};

// Модуляторы
class AMModulator {
    void SetCarrierFreq(double freq_hz);
    void SetModulationIndex(double m);
    void Modulate(const float* baseband, float* output, size_t count);
};

class FMModulator {
    void SetCarrierFreq(double freq_hz);
    void SetFrequencyDeviation(double deviation_hz);
    void Modulate(const float* baseband, float* output, size_t count);
};

class PSKModulator {
    void SetOrder(int order);  // 2=BPSK, 4=QPSK, 8=8PSK
    void SetSymbolRate(double symbols_per_sec);
    void Modulate(const uint8_t* symbols, float* i_out, float* q_out, size_t count);
};

class QAMModulator {
    void SetOrder(int order);  // 16, 64, 256
    void Modulate(const uint8_t* symbols, float* i_out, float* q_out, size_t count);
};
```

---

## 🏗️ Архитектура

```
┌─────────────────────────────────────────────┐
│              SignalSynth                    │
├─────────────────────────────────────────────┤
│                                             │
│  ┌─────────────┐  ┌─────────────┐           │
│  │ Oscillators │  │    Noise    │           │
│  │ sin/cos/NCO │  │  white/pink │           │
│  └──────┬──────┘  └──────┬──────┘           │
│         │                │                  │
│         ▼                ▼                  │
│  ┌─────────────────────────────┐            │
│  │        Modulators           │            │
│  │   AM / FM / PM / PSK / QAM  │            │
│  └──────────────┬──────────────┘            │
│                 │                           │
│                 ▼                           │
│  ┌─────────────────────────────┐            │
│  │     Pulse Shaping Filter    │            │
│  │    (raised cosine, etc.)    │            │
│  └──────────────┬──────────────┘            │
│                 │                           │
│                 ▼                           │
│            Output Buffer                    │
└─────────────────────────────────────────────┘
```

---

## 📊 Метрики производительности

| Операция | Sample Rate | GPU | Throughput |
|----------|-------------|-----|------------|
| Sine gen | 100 MHz | - | TBD |
| Noise gen | 100 MHz | - | TBD |
| QPSK mod | 10 Msym/s | - | TBD |

---

## 🔗 Зависимости

- `DrvGPU` — базовый драйвер
- `Heterodyne Module` — NCO
- `Filters Module` — pulse shaping

---

## 📚 Ссылки

- [GNU Radio Signal Processing](https://wiki.gnuradio.org/)
- [Digital Modulation Techniques](https://www.ni.com/en/shop/wireless-design-test/what-is-modulation.html)

---

## 📝 История изменений

| Дата | Автор | Изменение |
|------|-------|-----------|
| 2026-02-09 | Alex | Создание спецификации |
