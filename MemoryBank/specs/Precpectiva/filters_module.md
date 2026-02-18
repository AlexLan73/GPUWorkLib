# 📝 Filters Module — Спецификация

> **Модуль**: `Filters`
> **Статус**: 🟢 **Active** (Stage 1: FIR + IIR MVP — DONE)
> **Платформы**: OpenCL (active), ROCm (stub)
> **Автор**: Alex + Кодо
> **Создано**: 2026-02-09
> **Обновлено**: 2026-02-18

---

## 🎯 Назначение

Библиотека ЦОС-фильтров на GPU для параллельной обработки multi-channel сигналов.

---

## 📋 Требования

### Функциональные
- [x] REQ-001: FIR-фильтры (произвольные коэффициенты, direct-form)
- [x] REQ-002: IIR-фильтры (биквадратные секции, DFII-Transposed)
- [ ] REQ-003: Адаптивные фильтры (LMS, NLMS, RLS) — post-MVP
- [ ] REQ-004: Децимация / Интерполяция — post-MVP
- [ ] REQ-005: Полифазные фильтры — post-MVP
- [ ] REQ-006: Overlap-Save / Overlap-Add для длинных FIR — post-MVP

### Нефункциональные
- [ ] NFR-001: Latency < 1ms для блоков до 4096 samples
- [ ] NFR-002: Поддержка streaming (непрерывный поток)

---

## 🔧 Реализованный API

### C++ API
```cpp
namespace filters {

// FIR Filter
class FirFilter {
    FirFilter(IBackend* backend);
    void LoadConfig(const std::string& json_path);
    void SetCoefficients(const std::vector<float>& coeffs);
    InputData<cl_mem> Process(cl_mem input_buf, uint32_t channels, uint32_t points);
    std::vector<std::complex<float>> ProcessCpu(...);  // reference
    uint32_t GetNumTaps() const;
};

// IIR Filter (biquad cascade)
class IirFilter {
    IirFilter(IBackend* backend);
    void LoadConfig(const std::string& json_path);
    void SetBiquadSections(const std::vector<BiquadSection>& sections);
    InputData<cl_mem> Process(cl_mem input_buf, uint32_t channels, uint32_t points);
    std::vector<std::complex<float>> ProcessCpu(...);  // reference
    uint32_t GetNumSections() const;
};

struct BiquadSection { float b0, b1, b2, a1, a2; };

} // namespace filters
```

### Python API
```python
# FIR
fir = gpuworklib.FirFilter(ctx)
fir.set_coefficients(scipy.signal.firwin(64, 0.1).tolist())
result = fir.process(signal)  # (channels, points) complex64

# IIR
iir = gpuworklib.IirFilter(ctx)
iir.set_sections([{'b0':0.02, 'b1':0.04, 'b2':0.02, 'a1':-1.56, 'a2':0.64}])
result = iir.process(signal)
```

---

## 🏗️ Архитектура

### OpenCL Kernels
| Kernel | NDRange | Описание |
|--------|---------|----------|
| `fir_filter_cf32` | 2D (channels, points) | Direct-form FIR, __constant coeffs |
| `fir_filter_cf32_global` | 2D (channels, points) | FIR с __global coeffs (>16K taps) |
| `iir_biquad_cascade_cf32` | 1D (channels) | Biquad cascade, все секции в одном kernel |

### Decision Tree
```
FIR → num_taps ≤ 16000 → fir_filter_cf32 (__constant)
    → num_taps > 16000 → fir_filter_cf32_global (__global)
IIR → iir_biquad_cascade_cf32 (все секции в цикле)
```

### Файловая структура
```
modules/filters/
├── CMakeLists.txt
├── include/
│   ├── filters/          ← FirFilter, IirFilter, ROCm stubs
│   ├── kernels/          ← Inline R"CL(...)CL" kernels
│   └── types/            ← BiquadSection, FilterConfig, enums
├── src/
│   ├── fir_filter.cpp
│   └── iir_filter.cpp
└── tests/
    ├── all_test.hpp      ← Entry point: filters_all_test::run()
    ├── test_fir_basic.hpp
    ├── test_iir_basic.hpp
    └── README.md
```

---

## 📊 Метрики производительности

| Операция | Каналов | Порядок | Ожидание GPU vs CPU | Факт. точность |
|----------|---------|---------|---------------------|----------------|
| FIR direct | 8 | 64 taps | ~30-50x | err = 1e-6 ✅ |
| FIR direct | 64 | 64 taps | ~40-60x | — |
| IIR cascade | 8 | 2 biquads | ~30-50x | err = 1e-6 ✅ |
| IIR cascade | 1 | 2 biquads | ~0.5x (CPU быстрее!) | — |

### Результаты тестирования (2026-02-18, RTX 2080 Ti)

| Тест | GPU | Ошибка GPU vs CPU | Порог | Статус |
|------|-----|-------------------|-------|--------|
| C++ FIR 64tap 8ch×4096 | RTX 2080 Ti | 1e-6 | 1e-3 | ✅ PASSED |
| C++ IIR biquad 8ch×4096 | RTX 2080 Ti | 1e-6 | 1e-3 | ✅ PASSED |
| Python FIR vs scipy | RTX 2080 Ti | 4.77e-7 | 1e-2 | ✅ PASSED |
| Python IIR vs scipy.sosfilt | RTX 2080 Ti | 1.31e-6 | 5e-2 | ✅ PASSED |

---

## 🔗 Зависимости

- `DrvGPU` — базовый драйвер (IBackend, GPUProfiler, console_output)
- `OpenCL` — GPU вычисления
- `FFT Module` — для Overlap-Save/Add (будущее)

---

## 📝 История изменений

| Дата | Автор | Изменение |
|------|-------|-----------|
| 2026-02-09 | Alex | Создание спецификации |
| 2026-02-18 | Кодо | Реализация FIR + IIR MVP, Python bindings, тесты |
| 2026-02-18 | Кодо | Сборка + тесты PASSED (C++ err=1e-6, Python err=4.77e-7), статус → Active |
