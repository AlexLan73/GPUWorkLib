# Пошаговый алгоритм реализации дробной задержки фильтром Фарроу
# C++ / OpenCL (GPU)

**На основе:** Математическое описание КОМЦОС (раздел 3.1.2)
**Исследование:** Farrow C.W. "A continuously variable digital delay element" (1988),
MATLAB Farrow Structures, AMD/Xilinx Fractional Delay Farrow Filter, clFFT, VkFFT

---

## 1. ОБЗОР АЛГОРИТМА

### 1.1 Задача
Задержать комплексную сигнальную выборку антенного канала на произвольное
(нецелое) количество отсчётов для формирования приёмной диаграммы направленности
(ДН) фазированной антенной решётки.

### 1.2 Почему GPU?
- Антенная решётка содержит N×M каналов (десятки-сотни)
- Каждый канал обрабатывается **независимо** → идеальный параллелизм
- Свёртка через БПФ на GPU даёт ускорение 10-100x vs CPU
- OpenCL обеспечивает кроссплатформенность (NVIDIA, AMD, Intel, FPGA)

### 1.3 Структура алгоритма (из документа КОМЦОС)

```
Вход: s[M] — комплексная сигнальная выборка канала [x,y]
       D[x,y,u,v] — задержка (сек), f_d — частота дискретизации

  ┌─────────────────────────────────────┐
  │ Шаг 1: int_delay = floor(D * f_d)  │  ← целочисленная задержка
  │ Шаг 2: Сдвиг массива на int_delay  │  ← GPU kernel (тривиальный)
  │ Шаг 3: frd = -(D - int_delay/f_d)  │  ← дробная часть задержки
  │         * f_d                       │
  │ Шаг 4: pw[48×5] — матрица степеней │  ← CPU (маленькая матрица)
  │ Шаг 5: h[48] = sum(Farrow.*pw, 2)  │  ← CPU (48 чисел)
  │ Шаг 6: Дополнение нулями s и h     │  ← GPU
  │ Шаг 7: Y = IFFT(FFT(s) * FFT(h))  │  ← GPU (быстрая свёртка)
  └─────────────────────────────────────┘

Выход: Y — задержанная комплексная выборка
```

**ВАЖНО (из документа):** Задержка применяется **отдельно** для Re и Im частей
комплексной выборки. Сигнал без задержки тоже должен пройти через фильтр
Фарроу с frd=0 для компенсации групповой задержки фильтра = 3*(N-1)/(2*f_d).

---

## 2. ПОДГОТОВКА ОКРУЖЕНИЯ

### 2.1 Зависимости

```
Библиотека      │ Назначение               │ Установка
────────────────┼──────────────────────────┼─────────────────────────
OpenCL SDK      │ GPU compute              │ vendor SDK (NVIDIA/AMD/Intel)
clFFT           │ FFT на GPU               │ github.com/clMathLibraries/clFFT
 или VkFFT      │ FFT (лучше, актуальнее)  │ github.com/DTolm/VkFFT
```

### 2.2 Структура проекта

```
farrow_gpu/
├── CMakeLists.txt
├── src/
│   ├── main.cpp              — точка входа, инициализация OpenCL
│   ├── farrow_filter.cpp     — основная логика фильтра
│   ├── farrow_filter.h       — заголовок
│   ├── opencl_utils.cpp      — обёртки OpenCL (контекст, очередь, буферы)
│   └── opencl_utils.h
├── kernels/
│   ├── int_delay.cl          — kernel целочисленной задержки
│   ├── zero_pad.cl           — kernel дополнения нулями
│   ├── complex_multiply.cl   — kernel поэлементного умножения спектров
│   └── extract_result.cl     — kernel извлечения результата
├── data/
│   └── Farrow_coeff.bin      — константная матрица 48×5 (из .mat файла)
└── tests/
    └── test_farrow.cpp       — контрольный пример из документа
```

---

## 3. ПОШАГОВАЯ РЕАЛИЗАЦИЯ

### ═══════════════════════════════════════
### ШАГ 0: Инициализация OpenCL
### ═══════════════════════════════════════

**Что делаем:** Создаём контекст, очередь, компилируем kernels.
**Где:** CPU (один раз при запуске).

```cpp
// opencl_utils.cpp
#include <CL/cl.h>
#include <vector>
#include <fstream>
#include <stdexcept>

struct OpenCLContext {
    cl_platform_id   platform;
    cl_device_id     device;
    cl_context       context;
    cl_command_queue  queue;

    // Скомпилированные kernels
    cl_kernel k_int_delay;
    cl_kernel k_complex_multiply;
    cl_kernel k_zero_pad;
    cl_kernel k_extract;
};

OpenCLContext initOpenCL() {
    OpenCLContext ctx;
    cl_int err;

    // 1. Платформа и устройство
    err = clGetPlatformIDs(1, &ctx.platform, nullptr);
    err = clGetDeviceIDs(ctx.platform, CL_DEVICE_TYPE_GPU, 1, &ctx.device, nullptr);

    // 2. Контекст и очередь
    ctx.context = clCreateContext(nullptr, 1, &ctx.device, nullptr, nullptr, &err);
    ctx.queue = clCreateCommandQueue(ctx.context, ctx.device,
                                     CL_QUEUE_PROFILING_ENABLE, &err);

    // 3. Компиляция kernels (из файлов .cl)
    auto buildKernel = [&](const char* filename, const char* kernelName) -> cl_kernel {
        std::ifstream file(filename);
        std::string src((std::istreambuf_iterator<char>(file)),
                         std::istreambuf_iterator<char>());
        const char* srcPtr = src.c_str();
        size_t srcLen = src.size();
        cl_program prog = clCreateProgramWithSource(ctx.context, 1,
                                                     &srcPtr, &srcLen, &err);
        err = clBuildProgram(prog, 1, &ctx.device, "-cl-fast-relaxed-math", nullptr, nullptr);

        // Проверка ошибок компиляции
        if (err != CL_SUCCESS) {
            char log[4096];
            clGetProgramBuildInfo(prog, ctx.device, CL_PROGRAM_BUILD_LOG,
                                  sizeof(log), log, nullptr);
            throw std::runtime_error(std::string("Kernel build error: ") + log);
        }
        return clCreateKernel(prog, kernelName, &err);
    };

    ctx.k_int_delay        = buildKernel("kernels/int_delay.cl", "apply_int_delay");
    ctx.k_complex_multiply = buildKernel("kernels/complex_multiply.cl", "complex_multiply");
    ctx.k_zero_pad         = buildKernel("kernels/zero_pad.cl", "zero_pad");
    ctx.k_extract          = buildKernel("kernels/extract_result.cl", "extract_result");

    return ctx;
}
```

### ═══════════════════════════════════════
### ШАГ 1: Вычислить целочисленную задержку
### ═══════════════════════════════════════

**Формула (из документа):**
```
int_delay = floor(D[x,y,u,v] · f_d)
```

**Где:** CPU (одно скалярное вычисление на канал).

```cpp
// farrow_filter.cpp

struct FarrowParams {
    int   int_delay;     // целочисленная задержка (отсчёты)
    float frd;           // дробная задержка (-1..0]
    int   M;             // длина входной выборки
    int   N;             // длина фильтра Фарроу (48)
    int   L;             // длина свёртки = M + N - 1 (дополнение нулями)
    int   L_fft;         // ближайшая степень 2 >= L (для БПФ)
};

FarrowParams computeParams(float D_delay, float f_d, int M) {
    FarrowParams p;
    p.M = M;
    p.N = 48;  // длина фильтра Фарроу (из документа КОМЦОС)

    // ШАГ 1: целочисленная задержка
    p.int_delay = (int)floor(D_delay * f_d);

    // Длина свёртки и размер FFT
    p.L = p.M + p.N - 1;
    p.L_fft = 1;
    while (p.L_fft < p.L) p.L_fft <<= 1;  // ближайшая степень 2

    return p;
}
```

**Замечание:** `int_delay` может быть отрицательным (для отрицательных задержек).
В этом случае сдвиг выполняется в обратную сторону.

---

### ═══════════════════════════════════════
### ШАГ 2: Применить целочисленную задержку (GPU)
### ═══════════════════════════════════════

**Алгоритм (из документа):**
1. Удалить `int_delay` отсчётов в **конце** выборки
2. Добавить `int_delay` нулевых отсчётов в **начале** выборки

**Параллелизм:** Каждый поток обрабатывает один отсчёт → M потоков.

```c
// kernels/int_delay.cl
//
// Применение целочисленной задержки к комплексному сигналу.
// float2 = (Re, Im) — interleaved complex format.

__kernel void apply_int_delay(
    __global const float2* s_in,    // входной сигнал [M]
    __global       float2* s_out,   // выходной сигнал [M]
    const int int_delay,            // целочисленная задержка (отсчёты)
    const int M)                    // длина выборки
{
    int i = get_global_id(0);
    if (i >= M) return;

    if (i < int_delay) {
        // Начало выборки заполняем нулями
        s_out[i] = (float2)(0.0f, 0.0f);
    } else {
        // Копируем со сдвигом
        s_out[i] = s_in[i - int_delay];
    }
}
```

**Запуск kernel на CPU:**
```cpp
void applyIntDelay(OpenCLContext& ctx, cl_mem d_in, cl_mem d_out,
                   int int_delay, int M)
{
    cl_int err;
    err = clSetKernelArg(ctx.k_int_delay, 0, sizeof(cl_mem), &d_in);
    err = clSetKernelArg(ctx.k_int_delay, 1, sizeof(cl_mem), &d_out);
    err = clSetKernelArg(ctx.k_int_delay, 2, sizeof(int), &int_delay);
    err = clSetKernelArg(ctx.k_int_delay, 3, sizeof(int), &M);

    size_t globalSize = ((M + 255) / 256) * 256;  // округление до workgroup
    size_t localSize  = 256;
    err = clEnqueueNDRangeKernel(ctx.queue, ctx.k_int_delay,
                                  1, nullptr, &globalSize, &localSize,
                                  0, nullptr, nullptr);
    clFinish(ctx.queue);
}
```

---

### ═══════════════════════════════════════
### ШАГ 3: Вычислить дробную задержку frd
### ═══════════════════════════════════════

**Формула (из документа):**
```
frd = -(D[x,y,u,v] - int_delay / f_d) · f_d
```

**Где:** CPU.

```cpp
// Продолжение computeParams()
FarrowParams computeParams(float D_delay, float f_d, int M) {
    FarrowParams p;
    // ... (Шаг 1) ...

    // ШАГ 3: дробная задержка
    p.frd = -(D_delay - (float)p.int_delay / f_d) * f_d;

    // frd должна быть в диапазоне (-1, 0]
    // Если нет — скорректировать int_delay
    return p;
}
```

**Физический смысл:** `frd` — это доля интервала дискретизации,
на которую нужно дополнительно сдвинуть сигнал после целочисленного сдвига.

---

### ═══════════════════════════════════════
### ШАГ 4: Сформировать матрицу степеней pw (48×5)
### ═══════════════════════════════════════

**Формула (из документа):**
```
pw[row][col] = frd^col,  row = 0..47, col = 0..4
```

Все строки одинаковые: `[1, frd, frd², frd³, frd⁴]`

**Где:** CPU (матрица мизерная — 48×5 = 240 float).

```cpp
void buildPowerMatrix(float frd, float pw[48][5]) {
    float frd2 = frd * frd;
    float frd3 = frd2 * frd;
    float frd4 = frd3 * frd;

    for (int row = 0; row < 48; row++) {
        pw[row][0] = 1.0f;
        pw[row][1] = frd;
        pw[row][2] = frd2;
        pw[row][3] = frd3;
        pw[row][4] = frd4;
    }
}
```

---

### ═══════════════════════════════════════
### ШАГ 5: Вычислить коэффициенты фильтра Фарроу h[48]
### ═══════════════════════════════════════

**Формула (из документа, эквивалент Matlab: `h = sum(Farrow_coeff .* pw, 2)`):**
```
h[i] = Σ(j=0..4) Farrow_coeff[i][j] × pw[i][j],  i = 0..47
```

**Где:** CPU (48 скалярных произведений по 5 элементов).

```cpp
// Загрузка константной матрицы из файла
void loadFarrowCoeff(const char* filename, float coeff[48][5]) {
    FILE* f = fopen(filename, "rb");
    if (!f) throw std::runtime_error("Cannot open Farrow_coeff file");
    fread(coeff, sizeof(float), 48 * 5, f);
    fclose(f);
}

// Вычисление коэффициентов фильтра
void computeFarrowCoeffs(const float Farrow_coeff[48][5],
                          const float pw[48][5],
                          float h[48])
{
    for (int i = 0; i < 48; i++) {
        h[i] = 0.0f;
        for (int j = 0; j < 5; j++) {
            h[i] += Farrow_coeff[i][j] * pw[i][j];
        }
    }
}
```

**Замечание:** Структура Фарроу — это по сути набор из 5 параллельных
FIR-фильтров (ветвей), выходы которых взвешиваются степенями `frd`
и суммируются. В документе КОМЦОС используется «свёрнутая» форма
через поэлементное умножение матриц.

---

### ═══════════════════════════════════════
### ШАГ 6: Дополнение нулями (zero-padding) на GPU
### ═══════════════════════════════════════

**Из документа:** Перед быстрой свёрткой оба вектора `s` (размер M) и `h` (размер N=48)
дополняются нулями до размера `L = N + M - 1`.

**На практике:** Для FFT удобно дополнять до ближайшей степени 2 (`L_fft`).

```c
// kernels/zero_pad.cl
//
// Дополнение нулями: копирует src[src_len] в dst[dst_len],
// остаток заполняет нулями.
// Для комплексных данных используется float2.

__kernel void zero_pad(
    __global const float2* src,     // входные данные
    __global       float2* dst,     // выходные данные (дополненные)
    const int src_len,              // длина исходных данных
    const int dst_len)              // целевая длина (степень 2)
{
    int i = get_global_id(0);
    if (i >= dst_len) return;

    if (i < src_len) {
        dst[i] = src[i];
    } else {
        dst[i] = (float2)(0.0f, 0.0f);
    }
}
```

**Для фильтра h (действительный → комплексный):**
```c
// Специальный kernel для преобразования float[48] → float2[L_fft]
__kernel void zero_pad_real_to_complex(
    __global const float* src,      // действительные коэффициенты h[48]
    __global       float2* dst,     // комплексный выход [L_fft]
    const int src_len,
    const int dst_len)
{
    int i = get_global_id(0);
    if (i >= dst_len) return;

    if (i < src_len) {
        dst[i] = (float2)(src[i], 0.0f);   // Im = 0
    } else {
        dst[i] = (float2)(0.0f, 0.0f);
    }
}
```

---

### ═══════════════════════════════════════
### ШАГ 7: Быстрая свёртка через FFT на GPU
### ═══════════════════════════════════════

**Формула (из документа):**
```
Y = ОБПФ( БПФ(s) * БПФ(h) )
```
Где `*` — поэлементное комплексное умножение.

Этот шаг состоит из 4 подшагов:

#### 7.1 Прямое FFT сигнала и фильтра

Используем библиотеку **clFFT** или **VkFFT**:

```cpp
#include <clFFT.h>

void performFFTConvolution(OpenCLContext& ctx,
                           cl_mem d_s_padded,     // сигнал [L_fft] complex
                           cl_mem d_h_padded,     // фильтр [L_fft] complex
                           cl_mem d_result,        // результат [L_fft] complex
                           int L_fft)
{
    cl_int err;

    // --- Инициализация clFFT ---
    clfftSetupData fftSetup;
    clfftInitSetupData(&fftSetup);
    clfftSetup(&fftSetup);

    // --- Создание плана FFT ---
    clfftPlanHandle planHandle;
    size_t clLengths[1] = { (size_t)L_fft };

    clfftCreateDefaultPlan(&planHandle, ctx.context, CLFFT_1D, clLengths);
    clfftSetPlanPrecision(planHandle, CLFFT_SINGLE);
    clfftSetLayout(planHandle, CLFFT_COMPLEX_INTERLEAVED,
                               CLFFT_COMPLEX_INTERLEAVED);
    clfftSetResultLocation(planHandle, CLFFT_INPLACE);
    clfftBakePlan(planHandle, 1, &ctx.queue, nullptr, nullptr);

    // --- 7.1: FFT(s) — прямое преобразование сигнала ---
    clfftEnqueueTransform(planHandle, CLFFT_FORWARD,
                          1, &ctx.queue, 0, nullptr, nullptr,
                          &d_s_padded, nullptr, nullptr);

    // --- 7.1: FFT(h) — прямое преобразование фильтра ---
    clfftEnqueueTransform(planHandle, CLFFT_FORWARD,
                          1, &ctx.queue, 0, nullptr, nullptr,
                          &d_h_padded, nullptr, nullptr);

    clFinish(ctx.queue);
```

#### 7.2 Поэлементное комплексное умножение спектров (GPU)

```c
// kernels/complex_multiply.cl
//
// Поэлементное умножение двух комплексных массивов:
//   (a + bi)(c + di) = (ac - bd) + (ad + bc)i

__kernel void complex_multiply(
    __global const float2* A,       // FFT(s)
    __global const float2* B,       // FFT(h)
    __global       float2* C,       // результат
    const int len)
{
    int i = get_global_id(0);
    if (i >= len) return;

    float2 a = A[i];
    float2 b = B[i];

    // Комплексное умножение
    C[i] = (float2)(
        a.x * b.x - a.y * b.y,     // Re = ac - bd
        a.x * b.y + a.y * b.x      // Im = ad + bc
    );
}
```

#### 7.3 Обратное FFT (IFFT)

```cpp
    // --- 7.2: Поэлементное умножение ---
    err = clSetKernelArg(ctx.k_complex_multiply, 0, sizeof(cl_mem), &d_s_padded);
    err = clSetKernelArg(ctx.k_complex_multiply, 1, sizeof(cl_mem), &d_h_padded);
    err = clSetKernelArg(ctx.k_complex_multiply, 2, sizeof(cl_mem), &d_result);
    err = clSetKernelArg(ctx.k_complex_multiply, 3, sizeof(int), &L_fft);

    size_t globalSize = ((L_fft + 255) / 256) * 256;
    size_t localSize = 256;
    clEnqueueNDRangeKernel(ctx.queue, ctx.k_complex_multiply,
                            1, nullptr, &globalSize, &localSize,
                            0, nullptr, nullptr);

    // --- 7.3: IFFT(результат) ---
    clfftEnqueueTransform(planHandle, CLFFT_BACKWARD,
                          1, &ctx.queue, 0, nullptr, nullptr,
                          &d_result, nullptr, nullptr);

    clFinish(ctx.queue);

    // --- Очистка ---
    clfftDestroyPlan(&planHandle);
    clfftTeardown();
}
```

#### 7.4 Извлечение результата

Из массива длиной `L_fft` берём только первые `M` отсчётов
(или `L = M + N - 1` если нужна полная свёртка).
Также нужно нормализовать на `L_fft` (clFFT не нормализует IFFT).

```c
// kernels/extract_result.cl

__kernel void extract_result(
    __global const float2* conv_full,   // результат IFFT [L_fft]
    __global       float2* output,       // итоговый сигнал [M]
    const int M,                         // длина выходного сигнала
    const float norm)                    // = 1.0f / L_fft (нормализация IFFT)
{
    int i = get_global_id(0);
    if (i >= M) return;

    output[i] = conv_full[i] * norm;
}
```

---

## 4. ПОЛНАЯ ФУНКЦИЯ: Обработка одного канала

```cpp
void farrowDelayChannel(
    OpenCLContext& ctx,
    const float2* h_signal,    // входной комплексный сигнал [M] на CPU
    float2*       h_output,    // выходной сигнал [M] на CPU
    float         D_delay,     // задержка (секунды)
    float         f_d,         // частота дискретизации (Гц)
    int           M,           // длина выборки
    const float   Farrow_coeff[48][5])  // константные коэффициенты
{
    cl_int err;

    // === CPU: Шаги 1, 3, 4, 5 ===
    FarrowParams p = computeParams(D_delay, f_d, M);

    float pw[48][5];
    buildPowerMatrix(p.frd, pw);

    float h[48];
    computeFarrowCoeffs(Farrow_coeff, pw, h);

    // === GPU: Создание буферов ===
    cl_mem d_signal   = clCreateBuffer(ctx.context, CL_MEM_READ_ONLY,
                                        M * sizeof(float2), nullptr, &err);
    cl_mem d_shifted  = clCreateBuffer(ctx.context, CL_MEM_READ_WRITE,
                                        M * sizeof(float2), nullptr, &err);
    cl_mem d_s_padded = clCreateBuffer(ctx.context, CL_MEM_READ_WRITE,
                                        p.L_fft * sizeof(float2), nullptr, &err);
    cl_mem d_h_padded = clCreateBuffer(ctx.context, CL_MEM_READ_WRITE,
                                        p.L_fft * sizeof(float2), nullptr, &err);
    cl_mem d_h_real   = clCreateBuffer(ctx.context, CL_MEM_READ_ONLY,
                                        48 * sizeof(float), nullptr, &err);
    cl_mem d_conv     = clCreateBuffer(ctx.context, CL_MEM_READ_WRITE,
                                        p.L_fft * sizeof(float2), nullptr, &err);
    cl_mem d_output   = clCreateBuffer(ctx.context, CL_MEM_WRITE_ONLY,
                                        M * sizeof(float2), nullptr, &err);

    // === GPU: Загрузка данных ===
    clEnqueueWriteBuffer(ctx.queue, d_signal, CL_TRUE,
                         0, M * sizeof(float2), h_signal, 0, nullptr, nullptr);
    clEnqueueWriteBuffer(ctx.queue, d_h_real, CL_TRUE,
                         0, 48 * sizeof(float), h, 0, nullptr, nullptr);

    // === GPU: Шаг 2 — целочисленная задержка ===
    applyIntDelay(ctx, d_signal, d_shifted, p.int_delay, M);

    // === GPU: Шаг 6 — zero-padding ===
    // Сигнал: d_shifted[M] → d_s_padded[L_fft]
    // (запуск kernel zero_pad)
    {
        clSetKernelArg(ctx.k_zero_pad, 0, sizeof(cl_mem), &d_shifted);
        clSetKernelArg(ctx.k_zero_pad, 1, sizeof(cl_mem), &d_s_padded);
        clSetKernelArg(ctx.k_zero_pad, 2, sizeof(int), &M);
        clSetKernelArg(ctx.k_zero_pad, 3, sizeof(int), &p.L_fft);
        size_t gs = ((p.L_fft + 255) / 256) * 256;
        size_t ls = 256;
        clEnqueueNDRangeKernel(ctx.queue, ctx.k_zero_pad,
                                1, nullptr, &gs, &ls, 0, nullptr, nullptr);
    }

    // Фильтр: h[48] real → d_h_padded[L_fft] complex
    // (нужен отдельный kernel zero_pad_real_to_complex — см. Шаг 6)

    // === GPU: Шаг 7 — быстрая свёртка через FFT ===
    performFFTConvolution(ctx, d_s_padded, d_h_padded, d_conv, p.L_fft);

    // === GPU: Извлечение результата ===
    {
        float norm = 1.0f / (float)p.L_fft;
        clSetKernelArg(ctx.k_extract, 0, sizeof(cl_mem), &d_conv);
        clSetKernelArg(ctx.k_extract, 1, sizeof(cl_mem), &d_output);
        clSetKernelArg(ctx.k_extract, 2, sizeof(int), &M);
        clSetKernelArg(ctx.k_extract, 3, sizeof(float), &norm);
        size_t gs = ((M + 255) / 256) * 256;
        size_t ls = 256;
        clEnqueueNDRangeKernel(ctx.queue, ctx.k_extract,
                                1, nullptr, &gs, &ls, 0, nullptr, nullptr);
    }

    // === CPU: Чтение результата ===
    clEnqueueReadBuffer(ctx.queue, d_output, CL_TRUE,
                        0, M * sizeof(float2), h_output, 0, nullptr, nullptr);

    // === Освобождение памяти ===
    clReleaseMemObject(d_signal);
    clReleaseMemObject(d_shifted);
    clReleaseMemObject(d_s_padded);
    clReleaseMemObject(d_h_padded);
    clReleaseMemObject(d_h_real);
    clReleaseMemObject(d_conv);
    clReleaseMemObject(d_output);
}
```

---

## 5. ПАРАЛЛЕЛЬНАЯ ОБРАБОТКА ВСЕХ КАНАЛОВ РЕШЁТКИ

В антенной решётке NxM каналов. Каждый канал обрабатывается независимо.

### Вариант A: Последовательный запуск (простой)
```cpp
for (int x = 0; x < Nx; x++) {
    for (int y = 0; y < Ny; y++) {
        float D = computeDelay(x, y, u, v, c);  // D = (x*u + y*v) / c
        farrowDelayChannel(ctx, signal[x][y], output[x][y],
                          D, f_d, M, Farrow_coeff);
    }
}
```

### Вариант B: Пакетная обработка (batched FFT) — рекомендуется!

Если задержки для всех каналов разные, но длины одинаковые,
то можно:
1. Вычислить все `h[48]` для каждого канала на CPU
2. Запустить batched FFT для всех каналов одновременно

```cpp
// clFFT поддерживает batched FFT:
size_t batchSize = Nx * Ny;  // количество каналов
clfftSetPlanBatchSize(planHandle, batchSize);
```

Это даёт **максимальную загрузку GPU** и минимизирует overhead
от запуска kernels.

---

## 6. КОНТРОЛЬНЫЙ ПРИМЕР (из документа)

Из раздела 3.1.2 документа:
- Задержка между сигналами s1 и s2: **0.145 мкс**
- Сигналы немодулированные, несущая частота на отсчёте БПФ
- s2 также проходит через фильтр Фарроу с `frd = 0`
- Ожидаемая ошибка: **67 пкс** (пикосекунд)

```cpp
void runControlExample() {
    const float delay = 0.145e-6f;  // 0.145 мкс
    const float f_d   = 50e6f;      // пример: 50 МГц дискретизация
    const int   M     = 1024;

    // ... генерация s1, s2 ...
    // ... применение задержки к s1 ...
    // ... применение фильтра Фарроу с frd=0 к s2 ...
    // ... вычисление разности фаз через FFT ...
    // ... вычисление ошибки Derr = phi / (360 * f_carrier) ...
}
```

---

## 7. ОПТИМИЗАЦИИ

### 7.1 Память
- **Переиспользуйте буферы** между каналами (не создавать/удалять каждый раз)
- Используйте **pinned memory** (`CL_MEM_ALLOC_HOST_PTR`) для быстрого
  копирования CPU↔GPU
- Храните `Farrow_coeff` в **constant memory** GPU (`__constant`)

### 7.2 Вычисления
- **Batched FFT** — обрабатывать все каналы одним вызовом FFT
- **VkFFT вместо clFFT** — активно развивается, быстрее на современных GPU,
  поддерживает свёртку встроенно (zero-padding + convolution в один проход)
- Для коротких фильтров (N=48) и длинных сигналов (M >> 48) можно
  рассмотреть **прямую свёртку** вместо FFT-based (GPU хорошо параллелит FIR)

### 7.3 Точность
- Для двойной точности: заменить `float` → `double`, `float2` → `double2`
- Документ допускает целочисленный формат для `Farrow_coeff` —
  можно использовать **fixed-point** для FPGA через OpenCL

---

## 8. БЛОК-СХЕМА ПОЛНОГО PIPELINE

```
┌─────────────────────────────────────────────────────────┐
│                    АНТЕННАЯ РЕШЁТКА                      │
│              N×M приёмных каналов                        │
└──────────────────────┬──────────────────────────────────┘
                       │ s[x][y][M] — комплексные выборки
                       ▼
┌──────────────────────────────────────────────────────────┐
│  CPU: Для каждого канала [x,y]:                         │
│    D = (x·u + y·v) / c                                  │
│    int_delay = floor(D · f_d)                            │
│    frd = -(D - int_delay/f_d) · f_d                      │
│    pw = [1, frd, frd², frd³, frd⁴]  (48 строк)          │
│    h = sum(Farrow_coeff .* pw, 2)    (48 коэфф.)        │
└──────────────────────┬───────────────────────────────────┘
                       │ h[channel][48], int_delay[channel]
                       ▼
┌──────────────────────────────────────────────────────────┐
│  GPU Kernel 1: Целочисленная задержка (сдвиг массива)   │
│  → N×M потоков параллельно                               │
└──────────────────────┬───────────────────────────────────┘
                       ▼
┌──────────────────────────────────────────────────────────┐
│  GPU Kernel 2: Zero-padding до L_fft                     │
│  → s[M] → s_pad[L_fft], h[48] → h_pad[L_fft]           │
└──────────────────────┬───────────────────────────────────┘
                       ▼
┌──────────────────────────────────────────────────────────┐
│  GPU: Batched FFT (clFFT / VkFFT)                        │
│  → FFT(s_pad), FFT(h_pad) для всех каналов               │
└──────────────────────┬───────────────────────────────────┘
                       ▼
┌──────────────────────────────────────────────────────────┐
│  GPU Kernel 3: Поэлементное комплексное умножение        │
│  → S_freq[i] × H_freq[i] для всех каналов               │
└──────────────────────┬───────────────────────────────────┘
                       ▼
┌──────────────────────────────────────────────────────────┐
│  GPU: Batched IFFT                                        │
│  → обратное преобразование для всех каналов               │
└──────────────────────┬───────────────────────────────────┘
                       ▼
┌──────────────────────────────────────────────────────────┐
│  GPU Kernel 4: Извлечение + нормализация                 │
│  → первые M отсчётов × (1/L_fft)                        │
└──────────────────────┬───────────────────────────────────┘
                       ▼
┌──────────────────────────────────────────────────────────┐
│  Суммирование каналов → формирование луча ДН             │
└──────────────────────────────────────────────────────────┘
```

---

## 9. ССЫЛКИ И ИСТОЧНИКИ

1. **Документ КОМЦОС** — раздел 3.1.2 «Формирование приёмных ДН
   посредством управления временной задержкой»
2. **Farrow C.W.** "A continuously variable digital delay element" (1988) —
   оригинальная статья по структуре Фарроу
3. **MATLAB** — Fractional Delay Filters Using Farrow Structures
   (mathworks.com/help/dsp/ug/fractional-delay-filters-using-farrow-structures.html)
4. **AMD/Xilinx** — Fractional Delay Farrow Filter (Vitis AI Engine)
5. **clFFT** — OpenCL FFT library (github.com/clMathLibraries/clFFT)
6. **VkFFT** — Vulkan/CUDA/OpenCL FFT library (github.com/DTolm/VkFFT)
7. **Stanford CCRMA** — Farrow Structure (J.O. Smith)
8. **IntechOpen** — Fractional Delay Digital Filters (обзорная глава)
9. **Digital Beamforming using a GPU** — Nilsen, Hafizovic (2011)
10. **MIMO Radar Parallel Simulation on CPU/GPU** — PMC (2022)
