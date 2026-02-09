# 📝 FFT/IFFT Module — Спецификация

> **Модуль**: `fft_maxima`, `DrvGPU/FFT`
> **Статус**: 🟡 WIP
> **Платформы**: OpenCL (clFFT), ROCm (rocFFT), HIP
> **Автор**: Alex
> **Создано**: 2026-02-09
> **Обновлено**: 2026-02-09

---

## 🎯 Назначение

Высокопроизводительное преобразование Фурье на GPU для обработки сигналов.
- Прямое FFT (time → frequency)
- Обратное IFFT (frequency → time)
- Поиск максимумов спектра

---

## 📋 Требования

### Функциональные
- [x] REQ-001: FFT для размеров 2^n (n = 8..24)
- [x] REQ-002: IFFT с восстановлением фазы
- [x] REQ-003: Поиск максимумов спектра (spectrum_maxima_finder)
- [ ] REQ-004: Оконные функции (Hann, Hamming, Blackman, Kaiser)
- [ ] REQ-005: Real-to-Complex FFT (R2C)
- [ ] REQ-006: Batch FFT (несколько сигналов одновременно)

### Нефункциональные
- [ ] NFR-001: Производительность: ≥80% от пиковой теор. GPU
- [ ] NFR-002: Память: ZeroCopy где поддерживается
- [x] NFR-003: Профилирование через GPUProfiler

---

## 🔧 API

### Основные функции

```cpp
// spectrum_maxima_finder.hpp
class SpectrumMaximaFinder {
    void Initialize(cl_context ctx, cl_device_id device, size_t fft_size);
    void FindMaxima(const float* input, size_t count, std::vector<MaximaResult>& results);
    void Shutdown();
};
```

### Параметры конфигурации

| Параметр | Тип | Default | Описание |
|----------|-----|---------|----------|
| `fft_size` | size_t | 1024 | Размер FFT (должен быть 2^n) |
| `batch_size` | size_t | 1 | Количество FFT за один вызов |
| `use_zero_copy` | bool | true | Использовать ZeroCopy память |

---

## 🏗️ Архитектура

```
Input Signal (host)
      │
      ▼
┌─────────────────┐
│  Upload to GPU  │ ← cl_event upload_event
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│   clFFT/rocFFT  │ ← cl_event fft_event
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│  Maxima Kernel  │ ← cl_event maxima_event
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│ Download Result │ ← cl_event download_event
└────────┬────────┘
         │
         ▼
   Results (host)
```

---

## 📊 Метрики производительности

| Операция | Размер | GPU | Время | Пропускная способность |
|----------|--------|-----|-------|------------------------|
| FFT | 1M points | RX 7900 XTX | ~0.5ms | ~4 GB/s |
| FFT | 4M points | RX 7900 XTX | ~2ms | ~8 GB/s |

---

## 🔗 Зависимости

- `DrvGPU` — базовый драйвер
- `clFFT` — OpenCL FFT библиотека
- `rocFFT` — ROCm FFT библиотека (planned)

---

## 📚 Ссылки

- [clFFT GitHub](https://github.com/clMathLibraries/clFFT)
- [rocFFT Docs](https://rocm.docs.amd.com/projects/rocFFT/)

---

## 📝 История изменений

| Дата | Автор | Изменение |
|------|-------|-----------|
| 2026-02-09 | Alex + Кодо | Исправлен crash в clfftEnqueueTransform (clReleaseEvent ordering) |
| 2026-02-09 | Alex + Кодо | Добавлен GPUProfiler для замеров |
