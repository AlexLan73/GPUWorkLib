# Task_00_DrvGPU — ROCm Backend и инфраструктура

> **Памятка для ИИ**: Если работаешь под Windows — пиши весь код + тесты. **Тестировать будем только под Linux** (Debian/Ubuntu). ROCm не поддерживается на Windows.

---

## 1. Структура файлов

Создать в `DrvGPU/backends/rocm/` по аналогии с `DrvGPU/backends/opencl/`:

| Файл | Назначение |
|------|------------|
| `rocm_backend.hpp` | Класс ROCmBackend : public IBackend |
| `rocm_backend.cpp` | Реализация Initialize, Allocate, Free, Memcpy, Synchronize |
| `rocm_core.hpp` | hipDevice_t, hipCtx_t, hipStream_t, инициализация HIP |
| `rocm_core.cpp` | hipDeviceGet, hipCtxCreate, hipStreamCreate |

---

## 2. ROCmBackend — ключевые методы

Реализовать все методы `DrvGPU/interface/i_backend.hpp`:

```cpp
// Инициализация
void Initialize(int device_index) override;
bool IsInitialized() const override;
void Cleanup() override;

// Нативные хэндлы (void* → hipCtx_t, hipDevice_t, hipStream_t)
void* GetNativeContext() const override;
void* GetNativeDevice() const override;
void* GetNativeQueue() const override;

// Память (hipMalloc/hipFree, hipMemcpy)
void* Allocate(size_t size_bytes, unsigned int flags = 0) override;
void Free(void* ptr) override;
void MemcpyHostToDevice(void* dst, const void* src, size_t size_bytes) override;
void MemcpyDeviceToHost(void* dst, const void* src, size_t size_bytes) override;
void MemcpyDeviceToDevice(void* dst, const void* src, size_t size_bytes) override;

// Синхронизация
void Synchronize() override;
void Flush() override;

// Возможности
bool SupportsSVM() const override { return false; }
BackendType GetType() const override { return BackendType::ROCm; }
```

**Важно**: `Allocate` возвращает `void*` (фактически `hipDeviceptr_t`). `Free` принимает этот указатель и вызывает `hipFree`.

---

## 3. Интеграция в DrvGPU

В `DrvGPU/src/drv_gpu.cpp`, метод `CreateBackend()`:

```cpp
case BackendType::ROCm:
    backend_ = std::make_unique<ROCmBackend>();
    break;
```

Добавить `#include "backends/rocm/rocm_backend.hpp"` и условную компиляцию `#if ENABLE_ROCM`.

---

## 4. CMake — подключение ROCm backend

В `DrvGPU/CMakeLists.txt`:

- Условно добавлять `backends/rocm/rocm_backend.cpp`, `rocm_core.cpp` при `ROCM_ENABLED` (или `ENABLE_ROCM`)
- `target_link_libraries(drvgpu hip::hip)` при ROCm
- `target_include_directories` для `backends/rocm`
- В корневом `cmake/gpu-config.cmake` или `cmake/dependencies.cmake` — `find_package(hip)` и опция `ENABLE_ROCM` (ON на Linux при наличии hip)

---

## 5. HIPBuffer

Создать `DrvGPU/memory/hip_buffer.hpp` по аналогии с GPUBuffer:

- Конструктор: `HIPBuffer(void* hip_ptr, size_t num_elements, IBackend* backend)`
- `Write(host_data, size_bytes)` — hipMemcpy H2D
- `Read(host_data, size_bytes)` — hipMemcpy D2H
- `GetDevicePtr()` — возврат `void*`
- Деструктор: `hipFree` (если владеет) или нет (для ZeroCopy)

---

## 6. InputData для ROCm

`InputData` уже поддерживает `T = void*`. Для ROCm использовать `InputData<void*>` с `data` = указатель от `ROCmBackend::Allocate`.

Добавить в документацию или `input_data_traits.hpp` (если есть): `void*` для ROCm трактуется как `hipDeviceptr_t`.

---

## 7. Тесты

Создать `DrvGPU/tests/test_rocm_backend.hpp` и добавить вызов в `DrvGPU/tests/all_test.hpp`.

| Тест | Что проверяет |
|------|---------------|
| ROCm init | Initialize(0), IsInitialized |
| Allocate/Free | Allocate, Free, повторное выделение |
| Memcpy | MemcpyHostToDevice, MemcpyDeviceToHost, сравнение данных |
| Synchronize | Synchronize после kernel |

**Условие**: тесты под `#if ENABLE_ROCM` — компилируются только при включённом ROCm. На Windows без ROCm — не вызывать.

---

## 8. Чек-лист

- [ ] rocm_backend.hpp, rocm_backend.cpp
- [ ] rocm_core.hpp, rocm_core.cpp
- [ ] CreateBackend(ROCm) в drv_gpu.cpp
- [ ] CMake: ROCm sources, hip::hip, include_directories
- [ ] HIPBuffer (hip_buffer.hpp)
- [ ] test_rocm_backend.hpp + all_test.hpp
- [ ] Компиляция без ошибок (на Windows: ROCm код отключён через ENABLE_ROCM=OFF)

---

## Ссылки

- [PLAN_ROCm_DrvGPU_Full.md](PLAN_ROCm_DrvGPU_Full.md) — Часть 1, 4
- [DrvGPU/interface/i_backend.hpp](../../DrvGPU/interface/i_backend.hpp)
- [DrvGPU/backends/opencl/](../../DrvGPU/backends/opencl/) — референс структуры
