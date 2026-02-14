# Quick Reference: GPU Maxima Detection Methods

**Дата**: 2026-02-14
**Задача**: Поиск максимумов в спектре FFT (256 лучей × 4M точек)

---

## 🎯 Быстрый выбор метода

| Нужно | Метод | Время (4M) | Сложность |
|-------|-------|------------|-----------|
| **1 глобальный max** | Reduction | 1-2 ms | ⭐ Низкая |
| **K наибольших** | RadiK / Bitonic Select | 0.5-2 ms | ⭐⭐ Средняя |
| **ВСЕ локальные max** | Custom + Scan | 2-3 ms | ⭐⭐⭐ Высокая |

---

## 📊 Сравнение методов

### Reduction-based
```
Время: 1-5 ms | Память: O(n) | OpenCL: ✅ | ROCm: ✅
```
**Плюсы**: Простой, быстрый для 1 максимума
**Минусы**: Только глобальный максимум
**Когда**: Нужен 1 максимум на луч

### Top-K Selection (RadiK/Bitonic)
```
Время: 0.2-2 ms | Память: O(n+k) | OpenCL: ⚠️ | ROCm: ⚠️
```
**Плюсы**: Быстро для K максимумов, масштабируется
**Минусы**: Требует портирования, K заранее известно
**Когда**: Нужно фиксированное число максимумов

### Scan-based (Prefix Sum)
```
Время: 2-4 ms | Память: O(n) | OpenCL: ✅ | ROCm: ✅
```
**Плюсы**: ВСЕ максимумы, готовые библиотеки
**Минусы**: 3 прохода, дополнительная память
**Когда**: Универсальное решение

### Custom Kernel
```
Время: 1.5-2.5 ms | Память: O(n) | OpenCL: ✅ | ROCm: ✅
```
**Плюсы**: Максимальная производительность
**Минусы**: Сложная разработка и отладка
**Когда**: Критична производительность

---

## 🔧 Рекомендованное решение

### Для 256 лучей × 4M точек

**Подход**: Custom Kernel + Scan-based Compaction

**Алгоритм**:
1. Custom kernel детектирует локальные максимумы → флаги
2. Scan (rocPRIM/PyOpenCL) подсчитывает позиции
3. Compaction kernel копирует результаты

**Производительность**:
- На луч: ~2-3 ms
- 256 лучей на 1 GPU: ~500-750 ms
- 256 лучей на 10 GPU: ~50-75 ms

**Код** (псевдокод):
```c
// 1. Детекция
__kernel void detect_maxima(
    __global float* data,
    __global int* flags
) {
    __local float cache[WORK_SIZE + 2*WINDOW];
    // Load to local memory
    // Compare with neighbors
    flags[gid] = is_local_max ? 1 : 0;
}

// 2. Scan (rocPRIM)
rocprim::exclusive_scan(flags, positions, ...);

// 3. Compaction
__kernel void compact(
    __global float* data,
    __global int* flags,
    __global int* positions,
    __global float* output
) {
    if (flags[gid]) {
        output[positions[gid]] = data[gid];
    }
}
```

---

## 📚 Библиотеки

### OpenCL
- `PyOpenCL.ReductionKernel` — reduction
- `PyOpenCL.InclusiveScanKernel` — scan

### ROCm
- `rocprim::reduce` — reduction
- `rocprim::exclusive_scan` — scan
- `rocThrust` — высокоуровневые алгоритмы

---

## 🎓 Ключевые источники

1. [RadiK Paper (2025)](https://arxiv.org/html/2501.14336v1) — Top-K selection
2. [GPU Gems 3 Ch.39](https://developer.nvidia.com/gpugems/gpugems3/part-vi-gpu-computing/chapter-39-parallel-prefix-sum-scan-cuda) — Scan алгоритм
3. [NVIDIA OpenCL Reduction](https://github.com/sschaetz/nvidia-opencl-examples) — Optimized kernels
4. [gpu-topk](https://github.com/anilshanbhag/gpu-topk) — Top-K примеры

---

## ⚡ Оптимизации

1. **Local memory**: Кеширование для уменьшения global memory access
2. **Work-group size**: 64-256 (профилировать)
3. **Coalesced access**: Выравнивание доступа к памяти
4. **Batch processing**: Обработка нескольких лучей параллельно
5. **Async transfer**: `clEnqueueMapBuffer` для overlap

---

## 📋 Next Steps

1. ✅ Прототип Custom Kernel (OpenCL)
2. ✅ Интеграция с PyOpenCL scan
3. ⏳ Benchmark на реальных данных
4. ⏳ Оптимизация через GPUProfiler
5. ⏳ Портирование на ROCm/HIP
6. ⏳ Python bindings + тесты

---

**См. полный отчёт**: `GPU_Maxima_Detection_Methods_2026-02-14.md`
