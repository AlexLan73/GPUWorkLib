# DrvGPU/memory — Управление памятью GPU

Backend-агностичное управление GPU-памятью.

## Классы и файлы

| Файл | Описание |
|------|----------|
| `memory_manager.hpp/cpp` | `MemoryManager` — создание буферов, `CreateBuffer<T>()`, отслеживание аллокаций |
| `gpu_buffer.hpp` | `GPUBuffer<T>` — RAII-обёртка. `Write()`, `Read()`, `GetSize()`, `GetRaw()` |
| `svm_buffer.hpp` | SVM-буферы (Shared Virtual Memory) |
| `svm_capabilities.hpp` | `SVMCapabilities` — проверка поддержки SVM |
| `memory_type.hpp` | Типы памяти |
| `external_cl_buffer_adapter.hpp` | Адаптер внешних OpenCL-буферов |
| `i_memory_buffer.hpp` | `IMemoryBuffer` — интерфейс (redirect) |

## Использование

```cpp
auto& mem = gpu.GetMemoryManager();
auto buffer = mem.CreateBuffer<float>(1024);
std::vector<float> data(1024, 1.0f);
buffer->Write(data);
auto result = buffer->Read();
```

## Как тестировать

- **MemoryManager, GPUBuffer**: `tests/single_gpu.hpp` — `example_drv_gpu_singl::run()` — CreateBuffer, Write, Read
- **SVM**: проверка через `SVMCapabilities` в `single_gpu.hpp` (device info)
