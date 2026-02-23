# Task_08_ZeroCopy — OpenCL ↔ ROCm без копирования (опционально)

> **Памятка для ИИ**: Если работаешь под Windows — пиши весь код + тесты. **Тестировать будем только под Linux** (Debian/Ubuntu). ZeroCopy через dma-buf — Linux-only.

---

## 1. Цель

Реализовать ZeroCopy: передача буфера из OpenCL в ROCm без копирования через dma-buf (Вариант A). Резерв: CL_MEM_AMD_GPU_VA (Вариант B).

---

## 2. Зависимости

- Task_00_DrvGPU (ROCmBackend, OpenCLBackend)
- Linux (dma-buf не поддерживается на Windows)

---

## 3. Источник

[MemoryBank/research/AMD_GPU_OpenCL_ROCm_ZeroCopy_2026-02-06.md](../research/AMD_GPU_OpenCL_ROCm_ZeroCopy_2026-02-06.md)

---

## 4. Задачи

### 4.1 OpenCL: экспорт cl_mem в dma-buf fd

- Создать `opencl_export.hpp` или утилиту в `DrvGPU/backends/opencl/`
- `int ExportClBufferToFd(cl_mem buffer)` — возвращает dma-buf fd или -1
- Использовать `cl_khr_external_memory_dma_buf` или AMD-аналог
- Проверка: `cl_khr_external_memory_dma_buf` в CL_DEVICE_EXTENSIONS

### 4.2 ZeroCopyBridge класс

- Файлы: `zero_copy_bridge.hpp`, `zero_copy_bridge.cpp`
- `hipError_t ImportFromOpenCl(int dma_buf_fd, size_t buffer_size)`
- `void* GetHipPtr() const`
- Деструктор: `hipDestroyExternalMemory`
- Логика: `hipImportExternalMemory` + `hipExternalMemoryGetMappedBuffer`

### 4.3 Синхронизация

- Документировать: `clFinish` перед передачей, `hipStreamSynchronize` после ROCm
- Файл: `zero_copy_sync.hpp` или в ZeroCopyBridge

### 4.4 Тесты

- Export fd: OpenCL buffer → fd != -1
- Import: fd → ZeroCopyBridge → hip_ptr != nullptr
- Data integrity: записать в cl_mem, прочитать через hip_ptr — совпадение

---

## 5. Чек-лист

- [ ] ExportClBufferToFd
- [ ] ZeroCopyBridge
- [ ] Синхронизация (документация)
- [ ] Тесты
- [ ] Компиляция (Linux)

---

## Ссылки

- [PLAN_ROCm_DrvGPU_Full.md](PLAN_ROCm_DrvGPU_Full.md) — Часть 2, 6
- [AMD_GPU_OpenCL_ROCm_ZeroCopy_2026-02-06.md](../research/AMD_GPU_OpenCL_ROCm_ZeroCopy_2026-02-06.md)
