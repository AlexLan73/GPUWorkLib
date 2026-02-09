# 📚 AMD GPU: OpenCL ↔ ROCm Zero-Copy Memory

> **Источник**: [Doc/Info_RAM_OpenCl_ROCm/AMD_GPU_OpenCL_ROCm_ZeroCopy_2026-02-06.md](../../Doc/Info_RAM_OpenCl_ROCm/AMD_GPU_OpenCL_ROCm_ZeroCopy_2026-02-06.md)
> **Категория**: Memory
> **Дата**: 2026-02-06
> **Платформа**: AMD Instinct (AI100), OpenCL, ROCm/HIP

---

## 🎯 Краткое содержание

Способы передачи памяти между OpenCL и ROCm (HIP) **без копирования через host**:

### 1. hipExternalMemory (рекомендуемый)
- Экспорт `cl_mem` как dma-buf fd через `cl_khr_external_memory`
- Импорт в HIP через `hipImportExternalMemory`
- Получение device pointer через `hipExternalMemoryGetMappedBuffer`

### 2. Unified Memory (SVM + hipMallocManaged)
- Fine-grained SVM в OpenCL
- hipMallocManaged в HIP
- Общий указатель для обеих API

### 3. cl_amd_bus_addressable_memory
- Получение физического адреса GPU
- Прямой маппинг в другой runtime

---

## 📋 Ключевые API

```cpp
// OpenCL → экспорт dma-buf
clGetMemObjectInfo(cl_buf, CL_MEM_LINUX_DMA_BUF_FD_KHR, ...)

// HIP → импорт
hipImportExternalMemory(&extMem, &extMemDesc);
hipExternalMemoryGetMappedBuffer(&hip_ptr, extMem, &bufDesc);
```

---

## 🔗 Полный документ

👉 [Открыть полную документацию](../../Doc/Info_RAM_OpenCl_ROCm/AMD_GPU_OpenCL_ROCm_ZeroCopy_2026-02-06.md)

---

## 📝 Применение в проекте

- **DrvGPU**: Потенциальный Multi-API backend (OpenCL + ROCm одновременно)
- **FFT**: Передача буферов между clFFT и rocFFT без копирования
