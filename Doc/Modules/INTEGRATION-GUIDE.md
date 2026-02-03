# 🔧 Инструкция по интеграции modules в проект DrvGPU

## 📋 Что создано

```
modules/
├── CMakeLists.txt
├── README.md
└── vector_ops/
    ├── CMakeLists.txt
    ├── include/vector_ops_module.hpp
    ├── src/vector_ops_module.cpp
    └── kernels/vector_ops.cl
examples/
└── test_vector_ops.cpp
```

## 🚀 Шаги интеграции

### ШАГ 1: Добавить в главный CMakeLists.txt

```cmake
option(DRVGPU_BUILD_MODULES "Build compute modules" ON)

if(DRVGPU_BUILD_MODULES)
    add_subdirectory(modules)
    add_subdirectory(examples)
endif()
```

### ШАГ 2: Скопировать файлы

```bash
# Создать структуру
mkdir -p YOUR_PROJECT/modules/vector_ops/{include,src,kernels}
mkdir -p YOUR_PROJECT/examples

# Скопировать файлы
cp modules-CMakeLists.txt YOUR_PROJECT/modules/CMakeLists.txt
cp vector_ops-CMakeLists.txt YOUR_PROJECT/modules/vector_ops/CMakeLists.txt
cp vector_ops_module.hpp YOUR_PROJECT/modules/vector_ops/include/
cp vector_ops_module-*.cpp YOUR_PROJECT/modules/vector_ops/src/vector_ops_module.cpp
cp vector_ops.cl YOUR_PROJECT/modules/vector_ops/kernels/
cp test_vector_ops.cpp YOUR_PROJECT/examples/
cp modules-README.md YOUR_PROJECT/modules/README.md
```

### ШАГ 3: Собрать проект

```bash
cd YOUR_PROJECT
mkdir -p build && cd build
cmake .. -DDRVGPU_BUILD_MODULES=ON -DDRVGPU_BUILD_MODULE_VECTOR_OPS=ON
make -j$(nproc)
```

### ШАГ 4: Запустить тест

```bash
./examples/test_vector_ops
```

**Ожидаемый вывод:**
```
✅ DrvGPU initialized
✅ VectorOpsModule зарегистрирован
✅ Все 6 тестов пройдены успешно!
```

## 🎯 Использование в коде

```cpp
#include "drv_gpu.hpp"
#include "vector_ops_module.hpp"

// 1. Инициализация
DrvGPU gpu(BackendType::OPENCL, 0);
gpu.Initialize();

// 2. Создание модуля
auto module = std::make_shared<VectorOpsModule>(&gpu.GetBackend());
module->Initialize();
gpu.GetModuleRegistry().RegisterModule("VectorOps", module);

// 3. Использование
auto& mem = gpu.GetMemoryManager();
auto A = mem.CreateBuffer<float>(1024);
auto C = mem.CreateBuffer<float>(1024);
module->AddOneOut(A, C, 1024);
```

## ✅ Чеклист

- [ ] Скопированы все файлы
- [ ] Добавлен `add_subdirectory(modules)` в CMakeLists.txt
- [ ] CMake конфигурация проходит
- [ ] Модуль собирается
- [ ] Тест запускается и проходит

---

**Готово к использованию!** 🚀
