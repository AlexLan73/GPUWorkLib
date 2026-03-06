<!-- ДАЛЬШЕ ПЕРЕНОСИТЬ: Doc_Addition/Info_ZeroCopy_OpenCL_ROCm.md -->

# GPU AMD: связь OpenCL и ROCm без копирования через host

> Источник: MemoryBank/research/AMD_GPU_OpenCL_ROCm_ZeroCopy_2026-02-06.md  
> Дата: 2026-02-06

---

## Способы связи OpenCL и ROCm (кратко)

1. **hipExternalMemory** (рекомендуемый): OpenCL экспортирует dma-buf fd → HIP импортирует через `hipImportExternalMemory` / `hipExternalMemoryGetMappedBuffer`.
2. **Unified Memory (SVM + hipMallocManaged)**: один виртуальный адрес для OpenCL SVM и HIP managed на одном устройстве.
3. Другие варианты (vendor-specific) — см. полный файл в MemoryBank/research/.

---

## Код-пример (hipExternalMemory)

- OpenCL: `clCreateBufferWithProperties` (CL_MEM_DEVICE_HANDLE_LIST_KHR), `clGetMemObjectInfo(CL_MEM_LINUX_DMA_BUF_FD_KHR)`.
- HIP: `hipExternalMemoryHandleDesc` (type OpaqueFd), `hipImportExternalMemory`, `hipExternalMemoryGetMappedBuffer`.

Полный текст с примерами и ограничениями — в MemoryBank/research/AMD_GPU_OpenCL_ROCm_ZeroCopy_2026-02-06.md.
