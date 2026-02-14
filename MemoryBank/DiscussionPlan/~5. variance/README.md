# Variance Research Summary

> **Дата**: 2026-02-14
> **Статус**: Исследование завершено
> **Для модуля**: Statistics

---

## 📄 Документы

- **[GPU_Variance_Research.md](GPU_Variance_Research.md)** — Полное исследование (45 KB)

---

## 🎯 Краткие выводы

### Рекомендуемые алгоритмы

#### 1. Welford's Online Algorithm ⭐ (Primary Choice)
- **Точность**: Отлично (численно устойчивый)
- **Скорость**: Single pass, ~5N FLOPs
- **Параллелизация**: Хорошая (через combine)
- **Bonus**: Вычисляет mean + variance одновременно
- **Use case**: Default алгоритм для GPUWorkLib

#### 2. Pairwise Algorithm (Alternative)
- **Точность**: Хорошая
- **Скорость**: Single pass, ~6N FLOPs
- **Параллелизация**: Отличная (tree reduction)
- **Use case**: Когда нужна максимальная параллелизация

#### 3. Compensated Two-Pass
- **Точность**: Отлично
- **Скорость**: Two passes, 2× bandwidth
- **Use case**: Когда точность критична и два прохода допустимы

### ❌ НЕ использовать

- **One-Pass Naïve** — catastrophic cancellation
- **Kahan Compensated** — плохая параллелизация на GPU

---

## 📊 Для 256 лучей × 4M точек

### Архитектура
```
Memory Layout: SoA (Structure of Arrays)
├── Ray 0: [4M points]
├── Ray 1: [4M points]
├── ...
└── Ray 255: [4M points]

GPU Kernel:
├── 256 workgroups (one per ray)
├── 256 threads per workgroup
└── Welford reduction в shared memory
```

### Performance Estimate
```
Hardware: AMD RX 6800 XT
Data size: 256 × 4M × 4 bytes = 4 GB
Expected time: ~10-15 ms
Speedup vs CPU: 15-30×
```

---

## 💻 Implementation Plan

### API Design
```cpp
class StatisticsProcessor {
public:
    // Single array
    float compute_variance(cl_mem data, size_t N);

    // Batched (256 rays)
    void compute_variance_batch(
        cl_mem data,
        size_t points_per_array,
        size_t num_arrays,
        cl_mem output_variance
    );

    // Mean + Variance
    struct Stats { float mean, variance, std_dev; };
    Stats compute_statistics(cl_mem data, size_t N);
};
```

### Key Features
- ✅ Welford algorithm (default)
- ✅ Batched processing для multiple rays
- ✅ float32 + optional float64
- ✅ Integration с DrvGPU
- ✅ Memory coalescing (SoA layout)
- ✅ GPUProfiler support

---

## 🔬 Key Research Findings

### Numerical Stability Issues

**Catastrophic Cancellation:**
```
Формула: σ² = E[X²] - (E[X])²
Проблема: E[X²] ≈ (E[X])² → вычитание близких чисел

Пример:
Data: [1e8, 1e8+1, 1e8+2, 1e8+3]
E[X²] ≈ 1.00002e16  (float32)
(E[X])² ≈ 1.00002e16  (float32)
Result: 0.0 или negative!  ❌

Solution: Welford или Pairwise алгоритм ✅
```

### Memory Access Patterns

**SoA vs AoS:**
```
AoS (плохо):  ray0[4M], ray1[4M], ... → non-coalesced
SoA (хорошо): все rays в одном массиве → coalesced
```

### Precision Requirements

**float32 достаточно когда:**
- Численно устойчивый алгоритм (Welford/Pairwise)
- Данные в нормальном диапазоне (1e-6 ... 1e6)
- Относительная дисперсия σ²/μ² > 1e-7

**float64 нужен когда:**
- Очень малая дисперсия (σ² << μ²)
- Большие абсолютные значения
- N > 10⁷ точек

**Performance impact:**
- Consumer GPU: FP64 = 1/32 × FP32
- Professional GPU: FP64 = 1/2 × FP32

---

## 📚 Key References

### Algorithms
- [Chan-Golub-LeVeque (1979) - Pairwise Algorithm](http://i.stanford.edu/pub/cstr/reports/cs/tr/79/773/CS-TR-79-773.pdf)
- [Algorithms for calculating variance - Wikipedia](https://en.wikipedia.org/wiki/Algorithms_for_calculating_variance)
- [Welford's online algorithm](https://changyaochen.github.io/welford/)

### GPU Programming
- [NVIDIA - Parallel Reduction](https://developer.download.nvidia.com/assets/cuda/files/reduction.pdf)
- [AMD - Memory Coalescing](https://gpuopen.com/learn/gcn-memory-coalescing/)
- [rocPRIM Documentation](https://rocm.docs.amd.com/projects/rocPRIM/en/latest/)

### Libraries
- [CuPy variance implementation](https://github.com/cupy/cupy)
- [PyTorch variance source](https://github.com/pytorch/pytorch)
- [NVIDIA OpenCL Examples - Reduction](https://github.com/sschaetz/nvidia-opencl-examples)

---

## ✅ Next Steps для Statistics Module

1. **Реализация** (MemoryBank/specs/statistics_module.md)
   - Welford kernel OpenCL
   - Batched variance для 256 rays
   - Integration с DrvGPU

2. **Тестирование**
   - Numerical accuracy tests
   - Performance benchmarks
   - Comparison с NumPy/SciPy

3. **Python Bindings**
   - `StatisticsProcessor` class
   - Batch operations
   - Documentation в `Doc/Python/statistics_api.md`

4. **Документация**
   - API reference
   - Usage examples
   - Performance guide

---

*Исследование: Кодо*
*Дата: 2026-02-14*
