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
- ✅ **ROCm backend** — полная реализация (hipFFT, hiprtc, ZeroCopy, HybridBackend)
- ✅ Per-GPU logging (plog)
- ✅ GPU Profiler
- ✅ FFT Processor (clFFT + hipFFT ROCm)
- ✅ FFT Maxima (SpectrumMaximaFinder, OpenCL + ROCm)
- ✅ **Filters** — FIR + IIR GPU (Stage 1 MVP + AI Pipeline Stage 3)
- ✅ **Signal Generators** — CW, LFM, Noise, Script, FormSignal, DelayedFormSignal
- ✅ **LchFarrow** — дробная задержка Lagrange 48×5 (OpenCL + ROCm)
- ✅ **Statistics** — mean, median, variance, std (ROCm hiprtc, C++ 7/7, Python 9/9)
- ✅ **vector_algebra** — Cholesky инверсия (rocBLAS+rocSOLVER, SymmetrizeMode v2)
- ✅ **Heterodyne** — LFM Dechirp (OpenCL + ROCm)
- ✅ **FormSignal** — мультиканальный генератор (OpenCL + ROCm)
- ✅ Профилирование: fft_maxima, filters, heterodyne, lch_farrow (GpuBenchmarkBase)
- ✅ Python тесты ROCm — 30/30 PASSED по 5 модулям

**Changelog:** [2026-02.md](2026-02.md) | [2026-03.md](2026-03.md)

---

## 📋 Планируемые релизы

### v0.2.0
- ~~ROCm/HIP backend~~ ✅ Done
- ~~rocFFT support~~ ✅ Done (hipFFT)
- ~~ZeroCopy межплатформенная память~~ ✅ Done
- FM Correlator (planned)
- Moving Average Filter ROCm

### v0.3.0
- ~~Statistics module~~ ✅ Done
- ~~Heterodyne module~~ ✅ Done (LFM Dechirp)
- Overlap-Save/Add для длинных FIR

### v1.0.0
- Стабильный API
- Полная документация
- Все основные модули

---

*Последнее обновление: 2026-03-06*
