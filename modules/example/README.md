# 📦 DrvGPU Compute Modules

Модульная система вычислительных библиотек для DrvGPU.

## 🎯 Концепция

Каждый модуль реализует `IComputeModule` интерфейс:

```cpp
class IComputeModule {
    virtual void Initialize() = 0;
    virtual void Cleanup() = 0;
    virtual std::string GetName() const = 0;
    virtual IBackend* GetBackend() const = 0;
};
```

## ✅ VectorOps - Векторные операции

**Операции:**
- `AddOneOut()` / `AddOneInPlace()` - добавление скаляра
- `SubOneOut()` / `SubOneInPlace()` - вычитание скаляра
- `AddVectorsOut()` / `AddVectorsInPlace()` - сложение векторов

**Пример:**
```cpp
auto module = std::make_shared<VectorOpsModule>(backend);
module->Initialize();
registry.RegisterModule("VectorOps", module);

auto A = mem_mgr.CreateBuffer<float>(1024);
auto C = mem_mgr.CreateBuffer<float>(1024);
module->AddOneOut(A, C, 1024);
```

## 🔧 CMake опции

```cmake
option(DRVGPU_BUILD_MODULE_VECTOR_OPS "Build Vector Operations Module" ON)
option(DRVGPU_BUILD_MODULE_MATRIX     "Build Matrix Module"            OFF)
option(DRVGPU_BUILD_MODULE_FFT        "Build FFT Module"               OFF)
```

## 📁 Структура модуля

```
modules/
├── CMakeLists.txt
├── README.md
└── module_name/
    ├── CMakeLists.txt
    ├── include/
    │   └── module_name_module.hpp
    ├── src/
    │   └── module_name_module.cpp
    └── kernels/
        └── module_name.cl
```

## 🚀 Создание нового модуля

1. Создать структуру папок
2. Реализовать IComputeModule
3. Написать OpenCL kernels
4. Добавить в modules/CMakeLists.txt

См. `exampele/` как reference implementation.

## 🧪 Тестирование

```bash
cmake -DDRVGPU_BUILD_MODULES=ON ..
make
./build/examples/test_vector_ops
```

---

**Автор:** DrvGPU Team  
**Дата:** 2026-02-03
