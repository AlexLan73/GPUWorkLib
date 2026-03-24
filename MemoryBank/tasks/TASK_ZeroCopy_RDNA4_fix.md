# TASK: ZeroCopy OpenCL -> ROCm на RDNA4 (gfx1201)

## Статус: IN_PROGRESS

## Проблема

ZeroCopyBridge не работает на gfx1201 (RDNA4, Radeon RX 9070).
Оба существующих метода недоступны:

| Метод | Расширение | Статус на gfx1201 |
|-------|-----------|-------------------|
| **DMA-BUF** | `cl_khr_external_memory_dma_buf` | **НЕТ** в extensions |
| **AMD GPU VA** | `CL_MEM_AMD_GPU_VA` (0x4100) | `clGetMemObjectInfo` -> err=-30 (CL_INVALID_VALUE) |
| **SVM** | core OpenCL 2.0+ | Детектится, но **не реализован** в ZeroCopyBridge |

DrvGPU `test_zero_copy` тоже SKIP на тех же тестах — это не баг теста capon, а ограничение инфраструктуры.

## Что работает

- ROCm backend: OK (все capon тесты PASS)
- OpenCL backend: OK (инициализация, allocate, memcpy)
- HIP runtime: OK (hipMalloc, hipMemcpy)
- `CL_DEVICE_SVM_CAPABILITIES`: есть `CL_DEVICE_SVM_FINE_GRAIN_BUFFER`

## Расширения OpenCL на gfx1201

```
cl_khr_fp64 cl_khr_global_int32_base_atomics cl_khr_global_int32_extended_atomics
cl_khr_local_int32_base_atomics cl_khr_local_int32_extended_atomics
cl_khr_int64_base_atomics cl_khr_int64_extended_atomics
cl_khr_3d_image_writes cl_khr_byte_addressable_store cl_khr_fp16
cl_khr_gl_sharing cl_amd_device_attribute_query cl_amd_media_ops
cl_amd_media_ops2 cl_khr_image2d_from_buffer cl_khr_subgroups
cl_khr_depth_images cl_amd_copy_buffer_p2p cl_amd_assembly_program
```

**Отсутствуют**: `cl_khr_external_memory*`, `cl_amd_svm`, `cl_khr_svm`

## Варианты решения (исследовать)

### Вариант 1: SVM Path (наиболее вероятный)

OpenCL 2.0+ SVM + HIP unified memory на одном AMD GPU:

```cpp
// OpenCL: аллокация через SVM (не clCreateBuffer!)
void* svm_ptr = clSVMAlloc(cl_context, CL_MEM_READ_WRITE, size, 0);
clEnqueueSVMMemcpy(queue, CL_TRUE, svm_ptr, host_data, size, 0, nullptr, nullptr);

// HIP: тот же указатель напрямую (unified address space на AMD)
hipMemcpy(dst, svm_ptr, size, hipMemcpyDeviceToHost);  // или передать в kernel
```

**Проблема**: Нужно менять аллокацию на стороне OpenCL (не `clCreateBuffer` а `clSVMAlloc`).
Это значит ZeroCopyBridge не может импортировать произвольный `cl_mem`.

**Исследовать**:
- `clSVMAlloc` + `hipHostRegister` на этом указателе
- Можно ли передать SVM-указатель в HIP kernel напрямую

### Вариант 2: hipImportExternalMemory с DMA-BUF через ioctl

Даже без OpenCL расширения, AMD KFD (Kernel Fusion Driver) поддерживает DMA-BUF.
Можно получить fd через `/dev/kfd` ioctl вместо OpenCL API.

**Исследовать**:
- `rocr_lib` / HSA API: `hsa_amd_ipc_memory_create` / `hsa_amd_ipc_memory_attach`
- Прямой доступ через KFD: `AMDKFD_IOC_EXPORT_DMABUF` из `/usr/include/linux/kfd_ioctl.h`

### Вариант 3: hipHostRegister для mapped host memory

```cpp
// OpenCL: создать буфер с CL_MEM_ALLOC_HOST_PTR
cl_mem buf = clCreateBuffer(ctx, CL_MEM_READ_WRITE | CL_MEM_ALLOC_HOST_PTR, size, nullptr, &err);
void* host_ptr = clEnqueueMapBuffer(queue, buf, CL_TRUE, CL_MAP_READ, 0, size, ...);

// HIP: зарегистрировать host pointer
hipHostRegister(host_ptr, size, hipHostRegisterDefault);
void* device_ptr;
hipHostGetDevicePointer(&device_ptr, host_ptr, 0);
// device_ptr можно передать в HIP kernel
```

**Проблема**: Данные идут через host memory (pinned, но всё равно PCIe).
Не настоящий zero-copy в VRAM.

### Вариант 4: HSA IPC (Inter-Process Communication)

ROCm runtime (HSA) поддерживает IPC для разделения GPU памяти.
OpenCL и HIP оба используют HSA underneath.

```cpp
// HSA API
hsa_amd_ipc_memory_create(gpu_ptr, size, &ipc_handle);
hsa_amd_ipc_memory_attach(&ipc_handle, size, 1, &agents, &mapped_ptr);
```

**Исследовать**: работает ли IPC внутри одного процесса (OpenCL context + HIP context).

## Файлы для изменения

| Файл | Что менять |
|------|-----------|
| `DrvGPU/backends/opencl/opencl_export.hpp` | Добавить SVM detection / новый метод |
| `DrvGPU/backends/rocm/zero_copy_bridge.hpp/cpp` | Добавить новый Import метод |
| `DrvGPU/backends/hybrid/hybrid_backend.cpp` | Обновить CreateZeroCopyBridge |
| `DrvGPU/tests/test_zero_copy.hpp` | Тесты для нового метода |
| `modules/capon/tests/test_capon_opencl_to_rocm.hpp` | Обновить CheckZeroCopyAvailable |

## План работы

1. **Исследование** — проверить Вариант 1 (SVM) и Вариант 3 (hipHostRegister) быстрым скриптом
2. **Прототип** — реализовать рабочий вариант в отдельном test_*.cpp
3. **Интеграция** — добавить в ZeroCopyBridge как новый метод
4. **Тесты** — обновить test_zero_copy, test_capon_opencl_to_rocm
5. **Документация** — обновить GUIDE_opencl_to_rocm.md

## Быстрый тест для проверки (скопировать и запустить)

```bash
# Тест 1: SVM + HIP
cat > /tmp/test_svm_hip.cpp << 'TESTEOF'
#include <CL/cl.h>
#include <hip/hip_runtime.h>
#include <cstdio>
#include <cstring>

int main() {
    // OpenCL init
    cl_platform_id platform;
    cl_device_id device;
    cl_int err;
    clGetPlatformIDs(1, &platform, nullptr);
    clGetDeviceIDs(platform, CL_DEVICE_TYPE_GPU, 1, &device, nullptr);
    cl_context ctx = clCreateContext(nullptr, 1, &device, nullptr, nullptr, &err);
    cl_command_queue q = clCreateCommandQueueWithProperties(ctx, device, nullptr, &err);

    // SVM allocate
    size_t sz = 1024 * sizeof(float);
    void* svm_ptr = clSVMAlloc(ctx, CL_MEM_READ_WRITE | CL_MEM_SVM_FINE_GRAIN_BUFFER, sz, 0);
    printf("clSVMAlloc: %p\n", svm_ptr);
    if (!svm_ptr) { printf("SVM alloc failed\n"); return 1; }

    // Write data via host (fine-grain SVM = CPU-accessible)
    float* data = (float*)svm_ptr;
    for (int i = 0; i < 1024; i++) data[i] = (float)i * 0.5f;

    // Try reading via HIP
    float buf[4] = {};
    hipError_t herr = hipMemcpy(buf, svm_ptr, 4*sizeof(float), hipMemcpyDeviceToHost);
    printf("hipMemcpy from SVM: err=%d (%s)\n", herr, hipGetErrorString(herr));
    printf("  buf[0..3] = %.1f %.1f %.1f %.1f (expected: 0.0 0.5 1.0 1.5)\n",
           buf[0], buf[1], buf[2], buf[3]);

    clSVMFree(ctx, svm_ptr);
    clReleaseCommandQueue(q);
    clReleaseContext(ctx);
    return 0;
}
TESTEOF

# Компиляция:
g++ /tmp/test_svm_hip.cpp -o /tmp/test_svm_hip \
    -I/opt/rocm/include -L/opt/rocm/lib -lOpenCL -lamdhip64

# Запуск:
/tmp/test_svm_hip
```

```bash
# Тест 2: hipHostRegister на mapped cl_mem
cat > /tmp/test_host_register.cpp << 'TESTEOF'
#include <CL/cl.h>
#include <hip/hip_runtime.h>
#include <cstdio>

int main() {
    cl_platform_id platform;
    cl_device_id device;
    cl_int err;
    clGetPlatformIDs(1, &platform, nullptr);
    clGetDeviceIDs(platform, CL_DEVICE_TYPE_GPU, 1, &device, nullptr);
    cl_context ctx = clCreateContext(nullptr, 1, &device, nullptr, nullptr, &err);
    cl_command_queue q = clCreateCommandQueueWithProperties(ctx, device, nullptr, &err);

    size_t sz = 1024 * sizeof(float);

    // OpenCL: буфер с host-accessible памятью
    cl_mem buf = clCreateBuffer(ctx, CL_MEM_READ_WRITE | CL_MEM_ALLOC_HOST_PTR, sz, nullptr, &err);
    printf("clCreateBuffer (ALLOC_HOST_PTR): err=%d\n", err);

    // Map to host
    void* host_ptr = clEnqueueMapBuffer(q, buf, CL_TRUE, CL_MAP_WRITE, 0, sz, 0, nullptr, nullptr, &err);
    printf("clEnqueueMapBuffer: err=%d host_ptr=%p\n", err, host_ptr);

    // Write test data
    float* fdata = (float*)host_ptr;
    for (int i = 0; i < 1024; i++) fdata[i] = (float)i * 0.25f;
    clEnqueueUnmapMemObject(q, buf, host_ptr, 0, nullptr, nullptr);
    clFinish(q);

    // Re-map for read
    host_ptr = clEnqueueMapBuffer(q, buf, CL_TRUE, CL_MAP_READ, 0, sz, 0, nullptr, nullptr, &err);

    // HIP: register host pointer
    hipError_t herr = hipHostRegister(host_ptr, sz, hipHostRegisterDefault);
    printf("hipHostRegister: err=%d (%s)\n", herr, hipGetErrorString(herr));

    if (herr == hipSuccess) {
        void* device_ptr = nullptr;
        herr = hipHostGetDevicePointer(&device_ptr, host_ptr, 0);
        printf("hipHostGetDevicePointer: err=%d device_ptr=%p\n", herr, device_ptr);

        // Try reading back through device_ptr
        float rbuf[4] = {};
        herr = hipMemcpy(rbuf, device_ptr, 4*sizeof(float), hipMemcpyDeviceToHost);
        printf("hipMemcpy: err=%d\n", herr);
        printf("  rbuf[0..3] = %.2f %.2f %.2f %.2f (expected: 0.00 0.25 0.50 0.75)\n",
               rbuf[0], rbuf[1], rbuf[2], rbuf[3]);

        hipHostUnregister(host_ptr);
    }

    clEnqueueUnmapMemObject(q, buf, host_ptr, 0, nullptr, nullptr);
    clReleaseMemObject(buf);
    clReleaseCommandQueue(q);
    clReleaseContext(ctx);
    return 0;
}
TESTEOF

g++ /tmp/test_host_register.cpp -o /tmp/test_host_register \
    -I/opt/rocm/include -L/opt/rocm/lib -lOpenCL -lamdhip64
/tmp/test_host_register
```

## Контекст

- **GPU**: AMD Radeon RX 9070 (gfx1201, RDNA4)
- **ROCm**: 7.2.0 at `/opt/rocm-7.2.0/`
- **OS**: Debian 13 (Linux 6.12)
- **Тесты capon**: 01-05 ROCm PASS, reference_data 01-03 PASS, opencl_to_rocm SKIP
- **Связанный тест**: `DrvGPU/tests/test_zero_copy.hpp` — тоже SKIP
