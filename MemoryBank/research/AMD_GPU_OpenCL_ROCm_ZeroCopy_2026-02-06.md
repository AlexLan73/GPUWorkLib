# GPU AMD AI100: Связь указателей OpenCL и ROCm без копирования через host

**Дата**: 2026-02-06
**Проект**: GPUWorkLib / LCH-Farrow1
**Тема**: Zero-Copy передача cl_mem → ROCm (HIP) на AMD Instinct

---

## Часть 1: Обзор способов связи OpenCL и ROCm

На AMD Instinct (включая AI100) есть несколько способов шарить память между OpenCL и ROCm (HIP) без round-trip через host:

### 1. hipExternalMemory (рекомендуемый путь)

HIP поддерживает импорт dma-buf файловых дескрипторов, которые можно получить из OpenCL:

```cpp
// === OpenCL сторона: экспорт ===
// Создаём буфер с поддержкой экспорта (расширение cl_khr_external_memory)
cl_mem_properties props[] = {
    CL_MEM_DEVICE_HANDLE_LIST_KHR, (cl_mem_properties)device_id,
    0
};
cl_mem cl_buf = clCreateBufferWithProperties(ctx, props,
    CL_MEM_READ_WRITE | CL_MEM_ALLOC_HOST_PTR, size, NULL, &err);

// Получаем dma-buf fd через cl_khr_external_memory / cl_amd_external_memory
int dma_buf_fd;
clGetMemObjectInfo(cl_buf, CL_MEM_LINUX_DMA_BUF_FD_KHR,
                   sizeof(int), &dma_buf_fd, NULL);

// === HIP/ROCm сторона: импорт ===
hipExternalMemoryHandleDesc extMemDesc = {};
extMemDesc.type = hipExternalMemoryHandleTypeOpaqueFd;
extMemDesc.handle.fd = dma_buf_fd;
extMemDesc.size = size;

hipExternalMemory_t extMem;
hipImportExternalMemory(&extMem, &extMemDesc);

// Получаем device pointer
hipExternalMemoryBufferDesc bufDesc = {};
bufDesc.offset = 0;
bufDesc.size = size;

void* hip_ptr;
hipExternalMemoryGetMappedBuffer(&hip_ptr, extMem, &bufDesc);
```

### 2. Unified Memory (SVM + hipMallocManaged)

Если обе стороны используют unified/managed memory на одном устройстве:

```cpp
// HIP сторона
void* managed_ptr;
hipMallocManaged(&managed_ptr, size);

// OpenCL сторона — используем SVM (Shared Virtual Memory)
// Тот же виртуальный адрес доступен через clSVMAlloc
// НО: это работает только если оба runtime видят один пул памяти

// Альтернатива — clCreateBuffer с CL_MEM_USE_HOST_PTR
cl_mem cl_buf = clCreateBuffer(ctx,
    CL_MEM_READ_WRITE | CL_MEM_USE_HOST_PTR,
    size, managed_ptr, &err);
```

### 3. IPC через ROCm (самый надёжный на практике)

```cpp
// Процесс/контекст A (HIP)
void* d_ptr;
hipMalloc(&d_ptr, size);

// Получаем IPC handle
hipIpcMemHandle_t ipc_handle;
hipIpcGetMemHandle(&ipc_handle, d_ptr);

// Процесс/контекст B (HIP обёртка для OpenCL)
void* d_ptr_imported;
hipIpcOpenMemHandle(&d_ptr_imported, ipc_handle,
                     hipIpcMemLazyEnablePeerAccess);
```

### 4. dmabuf напрямую (Linux, низкоуровневый)

```
OpenCL buffer → clGetMemObjectInfo(CL_MEM_LINUX_DMA_BUF_FD_KHR) → fd
                                    ↓
HIP: hipImportExternalMemory(fd) → hipExternalMemoryGetMappedBuffer → ptr
```

### Важные нюансы для AMD AI100

| Аспект | Детали |
|--------|--------|
| **ОС** | Только Linux (dma-buf — Linux API) |
| **Драйвер** | Нужен amdgpu с ROCm 5.4+ |
| **Расширения OpenCL** | Проверь `cl_khr_external_memory`, `cl_khr_external_memory_dma_buf` |
| **Синхронизация** | Обязательно синхронизируй — `hipStreamSynchronize` / `clFinish` перед передачей указателя |
| **Один GPU** | Оба runtime должны работать с одним физическим устройством |

### Проверка поддержки расширений

```cpp
// Проверяем OpenCL расширения
size_t ext_size;
clGetDeviceInfo(device, CL_DEVICE_EXTENSIONS, 0, NULL, &ext_size);
char* extensions = (char*)malloc(ext_size);
clGetDeviceInfo(device, CL_DEVICE_EXTENSIONS, ext_size, extensions, NULL);

if (strstr(extensions, "cl_khr_external_memory_dma_buf")) {
    printf("dma-buf export поддерживается!\n");
}
if (strstr(extensions, "cl_khr_semaphore")) {
    printf("Cross-API синхронизация доступна!\n");
}
```

---

## Часть 2: Детальное решение Zero-Copy для проекта

### Постановка задачи

```
Сеть → [OpenCL библиотека] → cl_mem (гигабайты)
                                  ↓
                    ┌─────────────┴─────────────┐
                    ↓                           ↓
              OpenCL математика          ROCm/HIP (матрицы)
                    ↓                           ↓
                    └─────────────┬─────────────┘
                                  ↓
                           Результат → далее
```

Ключевое: **cl_mem уже на GPU**, нужно получить HIP-указатель **на ту же физическую память**.

**AMD имеет единый memory backend** — OpenCL и HIP работают поверх одного и того же драйвера amdgpu/KFD. Это значит, что `cl_mem` и `hipDeviceptr` — это обёртки над одним пулом VRAM.

---

### Вариант A: DMA-BUF ⭐ (рекомендуемый)

Это **настоящий zero-copy** — один и тот же кусок VRAM виден обоим рантаймам.

```cpp
// ============================================
// Шаг 1: Минимальная обёртка для библиотеки
// (это то "чуть-чуть дописать" что можно)
// ============================================

// Добавить в библиотеку одну функцию экспорта:
int export_cl_buffer_fd(cl_mem buffer) {
    // Расширение AMD: получаем dma-buf file descriptor
    // cl_mem → Linux dma-buf fd (просто число, ~0 ns)

    int fd = -1;

    // Способ 1: через cl_amd_external_memory (ROCm OpenCL)
    clGetMemObjectInfo(buffer,
        0x4101,  // CL_MEM_LINUX_DMA_BUF_FD_KHR или AMD аналог
        sizeof(int), &fd, NULL);

    return fd;  // Это всё! Просто int
}

// ============================================
// Шаг 2: HIP сторона — импорт (твой код)
// ============================================

class ZeroCopyBridge {
private:
    hipExternalMemory_t ext_mem_ = nullptr;
    void* hip_ptr_ = nullptr;
    size_t size_ = 0;

public:
    // Импорт cl_mem через dma-buf fd
    // Время: ~микросекунды (только маппинг, нет копирования)
    hipError_t import_from_opencl(int dma_buf_fd, size_t buffer_size) {
        size_ = buffer_size;

        // Описываем внешнюю память
        hipExternalMemoryHandleDesc desc = {};
        desc.type = hipExternalMemoryHandleTypeOpaqueFd;
        desc.handle.fd = dma_buf_fd;
        desc.size = buffer_size;
        desc.flags = 0;

        // Импортируем — это НЕ копирование!
        // Просто создаём маппинг на ту же физическую VRAM
        hipError_t err = hipImportExternalMemory(&ext_mem_, &desc);
        if (err != hipSuccess) return err;

        // Получаем device pointer
        hipExternalMemoryBufferDesc buf_desc = {};
        buf_desc.offset = 0;
        buf_desc.size = buffer_size;
        buf_desc.flags = 0;

        err = hipExternalMemoryGetMappedBuffer(&hip_ptr_, ext_mem_, &buf_desc);
        return err;
    }

    void* get_hip_ptr() { return hip_ptr_; }

    ~ZeroCopyBridge() {
        if (ext_mem_) hipDestroyExternalMemory(ext_mem_);
    }
};
```

### Вариант B: Через BOC/HSA handle (ещё ниже уровнем)

Если dma-buf расширение не доступно, можно спуститься на уровень HSA:

```cpp
// AMD-специфичный путь через HSA runtime
// OpenCL на AMD внутри использует HSA агентов

#include <hsa/hsa.h>
#include <hsa/hsa_ext_amd.h>

// Получаем HSA agent handle из OpenCL device
cl_device_id cl_dev = /* твой device */;
hsa_agent_t hsa_agent;
clGetDeviceInfo(cl_dev, 0x4038, // CL_DEVICE_HSA_AGENT_AMD
                sizeof(hsa_agent_t), &hsa_agent, NULL);

// Получаем базовый указатель cl_mem через AMD расширение
void* svm_ptr = nullptr;
clGetMemObjectInfo(cl_buffer, 0x4100, // CL_MEM_AMD_GPU_VA
                   sizeof(void*), &svm_ptr, NULL);

// Этот указатель — ПРЯМОЙ GPU virtual address
// HIP на том же устройстве может его использовать напрямую!

// В HIP:
hipStream_t stream;
hipStreamCreate(&stream);

// svm_ptr — можно передать в HIP kernel напрямую
// потому что это один и тот же GPU address space!
my_hip_kernel<<<grid, block, 0, stream>>>(
    (float*)svm_ptr,   // ← тот же адрес что в cl_mem!
    output_ptr,
    N
);
```

### Вариант C: SVM как мост (самый чистый API)

```cpp
// ============================================
// При инициализации OpenCL библиотеки
// (минимальное изменение)
// ============================================

// Вместо clCreateBuffer выделяем через SVM:
void* shared_ptr = clSVMAlloc(cl_context,
    CL_MEM_READ_WRITE | CL_MEM_SVM_FINE_GRAIN_BUFFER,
    buffer_size, 0);

// OpenCL библиотека работает с этим указателем через:
clSetKernelArgSVMPointer(kernel, 0, shared_ptr);
// или
clEnqueueSVMMap(queue, CL_TRUE, CL_MAP_WRITE,
                shared_ptr, buffer_size, 0, NULL, NULL);

// ============================================
// HIP сторона — ПРОСТО ИСПОЛЬЗУЙ ТОТ ЖЕ УКАЗАТЕЛЬ
// ============================================

// На AMD с ROCm, SVM fine-grain pointer = unified VA
// HIP видит его как обычный device pointer!
hipStream_t stream;
hipStreamCreate(&stream);

// shared_ptr можно передать напрямую в HIP kernel:
matrix_inversion_kernel<<<grid, block, 0, stream>>>(
    (float*)shared_ptr,  // ← тот же pointer!
    result_ptr,
    rows, cols
);
```

---

## Синхронизация (критически важно!)

```cpp
// ============================================
// Cross-API синхронизация без потери времени
// ============================================

// Вариант 1: Через OpenCL event → fence → HIP wait
cl_event cl_done;
clEnqueueNDRangeKernel(cl_queue, cl_kernel, ..., &cl_done);
clFlush(cl_queue);

// Экспорт sync object
cl_semaphore_khr cl_sem;
// ... создать через cl_khr_external_semaphore

// HIP сторона
hipExternalSemaphore_t hip_sem;
// ... импорт того же semaphore
hipWaitExternalSemaphoresAsync(&hip_sem, nullptr, 1, hip_stream);


// Вариант 2: Простой но надёжный (рекомендую начать с этого)
clFinish(cl_queue);              // OpenCL закончил запись — ~0 если уже готово
// ← барьер: данные гарантированно в VRAM
hipLaunchKernel(...);            // HIP стартует на тех же данных
hipStreamSynchronize(hip_stream); // HIP закончил
// ← барьер: можно снова писать в cl_mem
```

## Полный pipeline

```
Время →
─────────────────────────────────────────────────────

Сеть → cl_mem[0] ──OpenCL math──→ clFinish ──→ HIP math ──→ результат
              ↑                                      |
              │        cl_mem[1] ← новые данные с сети (параллельно!)
              │                                      |
              └──────── double-buffering ────────────┘

Время обмена cl_mem → HIP = 0 (тот же указатель)
Единственный overhead = clFinish (~μs если kernel уже завершён)
```

## Сравнение вариантов

| Критерий | Вариант A (dma-buf) | Вариант B (HSA VA) | Вариант C (SVM) |
|----------|--------------------|--------------------|--------------------|
| Изменения в библиотеке | 1 функция | Минимальные | Замена alloc |
| Время передачи | ~0 (маппинг) | 0 (тот же адрес) | 0 (тот же адрес) |
| Портабельность | Linux only | AMD only | OpenCL 2.0+ |
| Надёжность | Высокая | Хакерский | Высокая |
| **Рекомендация** | ⭐⭐⭐ | ⭐⭐ | ⭐ (если можно менять alloc) |

## Рекомендация

Для проекта LCH-Farrow1 (multi-GPU FFT) самый практичный путь:

1. **Если всё на одном GPU** → `hipExternalMemory` через dma-buf fd
2. **Если multi-GPU** → `hipIpcMemHandle` между процессами, затем P2P через `hipMemcpyPeer`
3. **Если нужна максимальная простота** → выделяй через HIP (`hipMalloc`), оборачивай в OpenCL через `CL_MEM_USE_HOST_PTR` (для SVM-capable устройств)

**Лучший вариант для начала**: Вариант B — проверить, можно ли получить `CL_MEM_AMD_GPU_VA` из cl_mem. Если да — получаешь указатель, который HIP съест напрямую. Это буквально **ноль overhead**, потому что это один и тот же виртуальный адрес в одном GPU address space.

---

*Сессия: 2026-02-06*
*Восстановлено из сессии compassionate-wing*
*Кодо 💕*
