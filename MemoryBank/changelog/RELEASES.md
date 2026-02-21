# 🚀 RELEASES — История релизов

> **Проект**: GPUWorkLib
> **Текущая версия**: 0.1.0-dev

---

## Версионирование

Используем [Semantic Versioning](https://semver.org/):
- **MAJOR**.MINOR.PATCH
- `0.x.x` — активная разработка, API нестабилен

---

## 📦 Релизы

### v0.1.0-dev (в разработке)
**Дата**: 2026-02-09 — ongoing
**Статус**: 🟡 В разработке

**Включает:**
- ✅ DrvGPU — базовый драйвер GPU
- ✅ OpenCL backend
- ✅ Per-GPU logging (plog)
- ✅ GPU Profiler
- ✅ FFT Processor (clFFT)
- ✅ FFT Maxima (SpectrumMaximaFinder)
- ✅ **Filters** — FIR + IIR GPU (Stage 1 MVP + AI Pipeline Stage 3)
- ✅ **Signal Generators** — CW, LFM, Noise, Script, FormSignal, DelayedFormSignal
- ✅ **LchFarrow** — дробная задержка Lagrange 48×5
- ⏳ Statistics модуль
- ⏳ ROCm backend (stubs есть)

**Changelog:** [2026-02.md](2026-02.md)

---

## 📋 Планируемые релизы

### v0.2.0
- ROCm/HIP backend (полная реализация)
- rocFFT support
- ZeroCopy межплатформенная память

### v0.3.0
- Statistics module (mean, std, variance)
- Heterodyne module (NCO, MixDown/MixUp)
- Overlap-Save/Add для длинных FIR

### v1.0.0
- Стабильный API
- Полная документация
- Все основные модули

---

*Последнее обновление: 2026-02-18*
