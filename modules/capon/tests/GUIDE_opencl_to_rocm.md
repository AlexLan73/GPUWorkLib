# Руководство: Данные от OpenCL — Расчёт Кейпона на ROCm (Zero Copy)

> **Для кого**: Разработчики, которые хотят скопировать этот паттерн в свой проект.
> Читай сверху вниз — каждый блок объясняет одну концепцию.

---

## Зачем это нужно?

В реальных DSP-системах данные редко появляются «из воздуха» — они выходят из предыдущего
этапа обработки. Например:

```
OpenCL-модуль        OpenCL-модуль        ROCm-модуль
[FFT обработка]  →  [Фильтрация]  →  [Алгоритм Кейпона]
    cl_mem               cl_mem              ???
```

Как передать `cl_mem` в ROCm? Есть два способа:

### ❌ Медленный путь — через CPU (не делай так!)

```
VRAM(OpenCL)  →  RAM(CPU)  →  VRAM(ROCm)
   cl_mem        hipMemcpy    hipMalloc
```

Два DMA-трансфера через шину PCIe. На современных GPU каждый стоит **~1-2 мс** для матрицы
1000×340 комплексных чисел. Для задачи реального времени это катастрофа.

### ✅ Быстрый путь — Zero Copy (используй это!)

```
VRAM(OpenCL)  ──────────────────►  VRAM(ROCm)
   cl_mem     ZeroCopyBridge        hip_ptr
              (только указатель,
               данные не трогаются)
```

OpenCL и ROCm на одном AMD GPU работают с **одной физической видеопамятью**.
Нужно только передать GPU-адрес — `ZeroCopyBridge` делает это за ~0-10 нс.

---

## Как работает Zero Copy изнутри

### Метод 1 — AMD GPU VA (предпочтительный)

AMD GPU может дать нам **virtual address** буфера прямо из OpenCL.
ROCm принимает этот адрес как обычный device pointer — никаких системных вызовов.

```
cl_mem
  │
  │  clGetMemObjectInfo(CL_MEM_AMD_GPU_VA)  ← расширение AMD
  ▼
void* gpu_va  ← физический адрес в VRAM
  │
  │  ZeroCopyBridge::ImportFromGpuVA()
  ▼
void* hip_ptr  ← тот же адрес, вид со стороны HIP
```

**Overhead**: ~0 наносекунд (просто сохраняем указатель).

### Метод 2 — DMA-BUF (Linux kernel)

Linux предоставляет механизм **dma-buf** для обмена GPU-буферами между драйверами.

```
cl_mem
  │
  │  clGetMemObjectInfo(CL_MEM_LINUX_DMA_BUF_FD_KHR)
  ▼
int dma_buf_fd  ← file descriptor (как обычный файл, но в памяти GPU)
  │
  │  hipImportExternalMemory(fd)
  ▼
hipExternalMemory_t  →  hip_ptr
```

**Overhead**: ~1-5 микросекунд (один системный вызов ядра).

---

## Минимальный пример кода

Вот полный паттерн «5 строк Zero Copy», который можно скопировать в свой проект:

```cpp
#include "backends/opencl/opencl_backend.hpp"
#include "backends/opencl/opencl_export.hpp"
#include "backends/rocm/rocm_backend.hpp"
#include "backends/rocm/zero_copy_bridge.hpp"

using namespace drv_gpu_lib;

void MyProcessing() {
    OpenCLBackend cl_backend;
    cl_backend.Initialize(0);

    ROCmBackend rocm_backend;
    rocm_backend.Initialize(0);

    // ── 1. Аллоцировать и заполнить данные через OpenCL ──────────────────
    const size_t N     = 1024;
    const size_t bytes = N * sizeof(float);

    void* cl_buf = cl_backend.Allocate(bytes);       // clCreateBuffer
    cl_backend.MemcpyHostToDevice(cl_buf, my_data, bytes); // clEnqueueWriteBuffer

    // ── 2. ОБЯЗАТЕЛЬНО: clFinish перед ZeroCopy! ─────────────────────────
    cl_backend.Synchronize();

    // ── 3. Zero Copy: cl_mem → HIP pointer ───────────────────────────────
    cl_device_id cl_device = static_cast<cl_device_id>(cl_backend.GetNativeDevice());

    ZeroCopyBridge bridge;
    bridge.ImportFromOpenCl(
        static_cast<cl_mem>(cl_buf),  // ← cl_mem буфер
        bytes,                        // ← размер в байтах
        cl_device                     // ← OpenCL device для проверки capabilities
    );

    // ── 4. Использовать в любом ROCm алгоритме ───────────────────────────
    void* hip_ptr = bridge.GetHipPtr();  // ← обычный HIP device pointer
    my_hip_kernel<<<grid, block>>>(static_cast<float*>(hip_ptr), N);
    // или передать в любой ROCm-модуль GPUWorkLib

    // ── 5. Освободить ─────────────────────────────────────────────────────
    // bridge.Release() вызывается автоматически в деструкторе bridge
    cl_backend.Free(cl_buf);
}
```

---

## Пример с алгоритмом Кейпона

```cpp
#include "capon_processor.hpp"
#include "backends/opencl/opencl_backend.hpp"
#include "backends/rocm/rocm_backend.hpp"
#include "backends/rocm/zero_copy_bridge.hpp"

using namespace drv_gpu_lib;

capon::CaponReliefResult ComputeCaponFromOpenCL(
    cl_mem          cl_signal,    // матрица сигнала Y [P×N] из OpenCL
    cl_mem          cl_steering,  // управляющие векторы U [P×M] из OpenCL
    cl_device_id    cl_device,
    OpenCLBackend&  cl_backend,
    ROCmBackend&    rocm_backend,
    const capon::CaponParams& params)
{
    const size_t bytes_Y = params.n_channels * params.n_samples    * sizeof(std::complex<float>);
    const size_t bytes_U = params.n_channels * params.n_directions * sizeof(std::complex<float>);

    // Шаг 1: clFinish — данные от OpenCL должны быть в VRAM
    cl_backend.Synchronize();

    // Шаг 2: Zero Copy
    ZeroCopyBridge bridge_Y, bridge_U;
    bridge_Y.ImportFromOpenCl(cl_signal,   bytes_Y, cl_device);
    bridge_U.ImportFromOpenCl(cl_steering, bytes_U, cl_device);

    // Шаг 3: Кейпон на ROCm с HIP-указателями
    capon::CaponProcessor processor(&rocm_backend);
    return processor.ComputeRelief(
        bridge_Y.GetHipPtr(),   // void* — тот же VRAM что и cl_signal
        bridge_U.GetHipPtr(),   // void* — тот же VRAM что и cl_steering
        params
    );
    // bridge_Y и bridge_U освобождаются здесь (деструкторы)
}
```

---

## Тесты в этом файле

Файл `test_capon_opencl_to_rocm.hpp` содержит 4 теста:

### Test 01 — detect_interop

**Что делает**: Определяет, поддерживает ли GPU Zero Copy. Выводит доступные методы.

```
[Capon[OCL→ROCm]] [01] detect_interop — проверить возможности Zero Copy
[Capon[OCL→ROCm]]   Capabilities: DMA-BUF=YES  AMD-GPU-VA=YES
[Capon[OCL→ROCm]]   Selected method: AMD GPU VA (CL_MEM_AMD_GPU_VA)
[Capon[OCL→ROCm]] [01] PASS — Zero Copy готов к использованию
```

**Не падает никогда** — только информирует. Если метод == NONE, остальные тесты пропустятся.

---

### Test 02 — signal_from_opencl

**Что делает**: Полный pipeline теста.

```
CPU data  →  OpenCL cl_mem  →  ZeroCopy  →  CaponProcessor  →  z[m]
```

Проверяет: `all z[m] > 0` и `all finite`.

**Шаги** (задокументированы inline в коде):
1. Генерация Y [8×64] и U [8×16] на CPU
2. `cl.Allocate()` + `cl.MemcpyHostToDevice()` → данные в VRAM
3. `cl.Synchronize()` — **критично!**
4. `bridge.ImportFromOpenCl()` → hip_ptr
5. `CaponProcessor::ComputeRelief(hip_ptr_Y, hip_ptr_U, params)`
6. Проверка результата

---

### Test 03 — results_match_ref

**Что доказывает**: Zero Copy **математически прозрачен**.

Вычисляет рельеф двумя путями с одними и теми же данными:
- **Путь A**: `CaponProcessor::ComputeRelief(vector<cx>, vector<cx>, params)` — стандартный
- **Путь B**: OpenCL upload → Zero Copy → `ComputeRelief(void*, void*, params)`

Проверяет: `max|z_A[m] - z_B[m]| < 1e-4`

Если тест падает — Zero Copy что-то сломал в данных (критическая ошибка!).

---

### Test 04 — beamform_from_opencl

**Что делает**: Адаптивное формирование луча `AdaptiveBeamform(hip_Y, hip_U, params)`.

Проверяет:
- Размерность выхода `[M × N]`
- Все элементы конечны (нет NaN, нет Inf)

---

## Частые ошибки

### ❌ Забыл `Synchronize()` перед `ImportFromOpenCl()`

```cpp
cl.MemcpyHostToDevice(cl_buf, data, bytes);
// cl.Synchronize();  ← ЗАБЫЛ!
bridge.ImportFromOpenCl(cl_buf, bytes, device);  // ← читает мусор!
```

**Эффект**: ROCm начинает читать буфер пока OpenCL ещё пишет → **случайные числа**, тест может
пройти (шум же шумом и должен быть), а может упасть. Очень трудно поймать!

**Правило**: `cl.Synchronize()` ВСЕГДА перед `ImportFromOpenCl()`.

### ❌ cl_mem освобождён раньше bridge

```cpp
ZeroCopyBridge bridge;
bridge.ImportFromOpenCl(cl_buf, bytes, device);
cl.Free(cl_buf);   // ← cl_mem освобождён!
my_kernel(bridge.GetHipPtr());  // ← UB: указатель на освобождённую память
```

**Правило**: `cl.Free()` только ПОСЛЕ того как bridge перестал использоваться.
Деструкторы в C++ вызываются в обратном порядке объявления — используй это!

```cpp
{
    void* cl_buf = cl.Allocate(bytes);
    cl.MemcpyHostToDevice(cl_buf, data, bytes);
    cl.Synchronize();

    ZeroCopyBridge bridge;                          // объявлен ПОСЛЕ cl_buf
    bridge.ImportFromOpenCl(cl_buf, bytes, device);
    processor.Process(bridge.GetHipPtr());

    cl.Free(cl_buf);   // ← освобождаем cl_mem вручную первым
    // bridge.~ZeroCopyBridge() вызывается здесь — порядок правильный
}
```

### ❌ OpenCL и ROCm на разных GPU

```cpp
OpenCLBackend cl;
cl.Initialize(0);   // GPU 0

ROCmBackend rocm;
rocm.Initialize(1); // GPU 1 — другое устройство!

// bridge.ImportFromOpenCl() выдаст ошибку или UB
```

**Правило**: оба backend должны инициализироваться на **одном и том же** физическом GPU
(одинаковый `device_index`).

### ❌ ENABLE_ROCM не определён

Zero Copy требует ROCm. Без `-DENABLE_ROCM=ON` при CMake тест просто вернётся из `run()`.
Это нормально — предусмотрена заглушка `#else`.

---

## Что нужно для компиляции

### CMakeLists.txt (минимум)

```cmake
find_package(HIP REQUIRED)
find_package(OpenCL REQUIRED)

target_include_directories(MyTarget PRIVATE
    ${PROJECT_SOURCE_DIR}/DrvGPU/include
    ${PROJECT_SOURCE_DIR}/DrvGPU
    ${PROJECT_SOURCE_DIR}/modules/capon/include
)

target_link_libraries(MyTarget PRIVATE
    DrvGPU        # OpenCL + ROCm backend
    capon_module  # CaponProcessor
    OpenCL::OpenCL
    hip::host
)

target_compile_definitions(MyTarget PUBLIC ENABLE_ROCM=1)
```

### Требования к системе

| Требование | Версия |
|-----------|--------|
| Linux | kernel 5.10+ (для DMA-BUF) |
| AMD GPU | RDNA2+ (gfx1010+) |
| ROCm | 6.0+ |
| OpenCL | 3.0 (Mesa/AMDGPU PRO) |
| C++ | C++17+ |

---

## Связанные файлы

| Файл | Описание |
|------|----------|
| `modules/capon/tests/test_capon_opencl_to_rocm.hpp` | **Этот тест** |
| `DrvGPU/backends/rocm/zero_copy_bridge.hpp` | ZeroCopyBridge API |
| `DrvGPU/backends/opencl/opencl_export.hpp` | ExportClBuffer* функции |
| `DrvGPU/tests/test_zero_copy.hpp` | Базовые тесты ZeroCopyBridge |
| `modules/capon/tests/test_capon_rocm.hpp` | Базовые тесты Кейпона (без ZeroCopy) |
| `DrvGPU/backends/hybrid/hybrid_backend.hpp` | HybridBackend (OpenCL+ROCm вместе) |

---

## Краткая шпаргалка

```
┌─────────────────────────────────────────────────────┐
│              ПАТТЕРН ZERO COPY                       │
├─────────────────────────────────────────────────────┤
│                                                     │
│  // 1. Выделить + заполнить через OpenCL            │
│  void* cl_buf = cl.Allocate(bytes);                 │
│  cl.MemcpyHostToDevice(cl_buf, data, bytes);        │
│                                                     │
│  // 2. !!! ОБЯЗАТЕЛЬНО !!!                          │
│  cl.Synchronize();                                  │
│                                                     │
│  // 3. Zero Copy                                    │
│  ZeroCopyBridge bridge;                             │
│  bridge.ImportFromOpenCl(                           │
│      (cl_mem)cl_buf, bytes, cl_device);             │
│                                                     │
│  // 4. Использовать в ROCm                          │
│  rocm_module.Process(bridge.GetHipPtr());           │
│                                                     │
│  // 5. Освободить (порядок важен!)                  │
│  cl.Free(cl_buf);   // cl_mem — вручную             │
│  // bridge — автоматически в деструкторе            │
└─────────────────────────────────────────────────────┘
```
