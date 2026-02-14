# GPU Variance — Quick Reference

> Быстрая справка по алгоритмам variance для GPU

---

## 📐 Формулы

### Variance (Population)
```
σ² = (1/N) Σ(xᵢ - μ)²
   = E[X²] - (E[X])²
```

### Variance (Sample)
```
s² = (1/(N-1)) Σ(xᵢ - μ)²
```

### Standard Deviation
```
σ = √(σ²)
```

---

## ⚡ Алгоритмы (Quick Comparison)

| Algorithm | Passes | Stability | GPU-Friendly | Recommended |
|-----------|--------|-----------|--------------|-------------|
| **Welford** | 1 | ⭐⭐⭐ | ✅ | **YES** |
| **Pairwise** | 1 | ⭐⭐ | ✅✅ | **YES** |
| **Compensated 2-pass** | 2 | ⭐⭐⭐ | ✅ | OK |
| **Two-pass** | 2 | ⭐⭐ | ✅ | OK |
| **One-pass naïve** | 1 | ❌ | ✅ | **NO** |
| **Shifted data** | 1 | ⭐ | ✅ | Maybe |

---

## 💻 Code Snippets

### Welford's Algorithm (Sequential)

```cpp
float M = 0.0f;  // Running mean
float S = 0.0f;  // Sum of squared diffs

for (int i = 0; i < N; i++) {
    float delta = x[i] - M;
    M += delta / (i + 1);
    float delta2 = x[i] - M;
    S += delta * delta2;
}

float mean = M;
float variance = S / N;
```

### Welford Combine (Parallel)

```cpp
struct Welford {
    int n;
    float mean;
    float M2;
};

Welford combine(Welford a, Welford b) {
    int n = a.n + b.n;
    float delta = b.mean - a.mean;
    float mean = (a.n * a.mean + b.n * b.mean) / n;
    float M2 = a.M2 + b.M2 + delta * delta * a.n * b.n / n;
    return {n, mean, M2};
}
```

### Pairwise Combine

```cpp
struct Stats {
    int n;
    float mean;
    float var;
};

Stats combine(Stats a, Stats b) {
    int n = a.n + b.n;
    float mean = (a.n * a.mean + b.n * b.mean) / n;
    float delta = b.mean - a.mean;
    float var = (a.n * a.var + b.n * b.var) / n
              + (a.n * b.n * delta * delta) / (n * n);
    return {n, mean, var};
}
```

---

## 🎯 OpenCL Kernel Template (Welford)

```c
__kernel void welford_batched(
    __global const float* data,     // [num_arrays * N]
    __global float* out_mean,       // [num_arrays]
    __global float* out_var,        // [num_arrays]
    int N,
    int num_arrays
) {
    int array_id = get_group_id(0);
    int tid = get_local_id(0);
    int wg_size = get_local_size(0);

    __local float local_mean[256];
    __local float local_M2[256];
    __local int local_count[256];

    // Local Welford
    __global const float* arr = data + array_id * N;
    int count = 0;
    float mean = 0.0f;
    float M2 = 0.0f;

    for (int i = tid; i < N; i += wg_size) {
        float x = arr[i];
        count++;
        float delta = x - mean;
        mean += delta / count;
        M2 += delta * (x - mean);
    }

    local_count[tid] = count;
    local_mean[tid] = mean;
    local_M2[tid] = M2;
    barrier(CLK_LOCAL_MEM_FENCE);

    // Reduction (Welford combine)
    for (int s = wg_size / 2; s > 0; s >>= 1) {
        if (tid < s) {
            int na = local_count[tid];
            int nb = local_count[tid + s];
            int n = na + nb;

            float ma = local_mean[tid];
            float mb = local_mean[tid + s];
            float delta = mb - ma;

            local_mean[tid] = (na * ma + nb * mb) / (float)n;
            local_M2[tid] = local_M2[tid] + local_M2[tid + s]
                          + delta * delta * na * nb / (float)n;
            local_count[tid] = n;
        }
        barrier(CLK_LOCAL_MEM_FENCE);
    }

    if (tid == 0) {
        out_mean[array_id] = local_mean[0];
        out_var[array_id] = local_M2[0] / local_count[0];
    }
}
```

---

## 🚀 Launch Configuration (256 rays × 4M points)

```cpp
// OpenCL dispatch
const size_t num_rays = 256;
const size_t points_per_ray = 4000000;
const size_t workgroup_size = 256;

size_t global_size = num_rays * workgroup_size;
size_t local_size = workgroup_size;

clEnqueueNDRangeKernel(
    queue,
    kernel,
    1,                  // work_dim
    NULL,               // global_work_offset
    &global_size,       // global_work_size
    &local_size,        // local_work_size
    0, NULL, NULL
);
```

---

## ⚠️ Common Pitfalls

### 1. Catastrophic Cancellation
```cpp
// ❌ BAD (one-pass naïve)
float sum_sq = 0, sum = 0;
for (...) { sum += x; sum_sq += x*x; }
float var = sum_sq/N - (sum/N)*(sum/N);
// Problem: sum_sq/N ≈ (sum/N)² → loss of precision!

// ✅ GOOD (Welford)
float M = 0, S = 0;
for (int i = 0; i < N; i++) {
    float delta = x[i] - M;
    M += delta / (i+1);
    S += delta * (x[i] - M);
}
float var = S / N;
```

### 2. Non-Coalesced Memory Access
```cpp
// ❌ BAD (strided access)
// Thread 0: data[0], data[256], data[512], ...
// Thread 1: data[1], data[257], data[513], ...

// ✅ GOOD (coalesced)
// Thread 0: data[0], data[1], data[2], ...
// Thread 1: data[256*4M], data[256*4M+1], ...
```

### 3. Wrong Workgroup Size
```cpp
// ❌ BAD
size_t local_size = 100;  // Not multiple of warp/wavefront

// ✅ GOOD
size_t local_size = 256;  // Good for both NVIDIA (32) and AMD (64)
```

---

## 📊 Performance Tips

### Memory Bandwidth Optimization
```cpp
// Single kernel pass > Multiple kernels
// Example: combined mean+variance Welford kernel
// instead of separate mean kernel + variance kernel
```

### Workgroup Optimization
```cpp
// NVIDIA: multiples of 32 (warp size)
// AMD: multiples of 64 (wavefront size)
// Recommended: 256 or 512
```

### Precision Choice
```cpp
// float32: ~10-15 ms for 256×4M
// float64: ~320-640 ms (consumer GPU)
//          ~20-40 ms (professional GPU)

// Use float32 + Welford for most cases
// Use float64 only when:
//   - σ² << μ² (very small variance)
//   - High precision required
```

---

## 🧪 Testing Checklist

### Numerical Accuracy
```python
# Test 1: Small variance (catastrophic cancellation)
data = [1e8, 1e8+1, 1e8+2, 1e8+3]
assert abs(gpu_var - ref_var) / ref_var < 1e-5

# Test 2: Large N
data = randn(4_000_000)
assert abs(gpu_var - numpy_var) / numpy_var < 1e-4

# Test 3: Known distribution
data = normal(μ=5, σ=2, size=1M)
assert abs(sqrt(gpu_var) - 2.0) < 0.01
```

### Performance
```python
# Benchmark different sizes
sizes = [1K, 10K, 100K, 1M, 4M, 10M]
for N in sizes:
    cpu_time = benchmark_cpu(N)
    gpu_time = benchmark_gpu(N)
    speedup = cpu_time / gpu_time
    print(f"N={N}: Speedup={speedup:.1f}x")
```

---

## 🔗 Quick Links

- [Full Research Document](GPU_Variance_Research.md)
- [Summary](README.md)
- [Chan-Golub-LeVeque Paper](http://i.stanford.edu/pub/cstr/reports/cs/tr/79/773/CS-TR-79-773.pdf)
- [Welford Algorithm](https://en.wikipedia.org/wiki/Algorithms_for_calculating_variance#Welford's_online_algorithm)
- [NVIDIA Reduction Guide](https://developer.download.nvidia.com/assets/cuda/files/reduction.pdf)

---

*Quick Reference v1.0 — 2026-02-14*
