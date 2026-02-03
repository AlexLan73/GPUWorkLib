# 🔧 Как собрать vector_ops_module.cpp из частей

Файл `vector_ops_module.cpp` был разделён на 3 части для удобства. Чтобы собрать финальный файл:

## Вариант 1: Вручную

Объедините файлы в следующем порядке:

1. **vector_ops_module-part1.cpp** (начало + AddOne операции)
2. **vector_ops_module-part2.cpp** (SubOne + AddVectors + CompileKernels)
3. **vector_ops_module-part3.cpp** (CreateKernelObjects + Release + LoadKernelSource)

Удалите дублирующиеся includes и namespace блоки между частями.

## Вариант 2: Команда cat

```bash
cat vector_ops_module-part1.cpp \
    vector_ops_module-part2.cpp \
    vector_ops_module-part3.cpp \
    > modules/vector_ops/src/vector_ops_module.cpp
```

## Финальная структура файла:

```cpp
#include "vector_ops_module.hpp"
#include "common/logger.hpp"
#include <fstream>
#include <sstream>
#include <stdexcept>
#include <cmath>

#ifndef VECTOR_OPS_KERNELS_PATH
#define VECTOR_OPS_KERNELS_PATH "kernels"
#endif

namespace drv_gpu_lib {

// Конструктор/деструктор
VectorOpsModule::VectorOpsModule(...) { ... }
~VectorOpsModule() { ... }

// Жизненный цикл
void Initialize() { ... }
void Cleanup() { ... }

// Операции AddOne
void AddOneOut(...) { ... }
void AddOneInPlace(...) { ... }

// Операции SubOne
void SubOneOut(...) { ... }
void SubOneInPlace(...) { ... }

// Операции AddVectors
void AddVectorsOut(...) { ... }
void AddVectorsInPlace(...) { ... }

// Приватные методы
void CompileKernels() { ... }
void CreateKernelObjects() { ... }
void ReleaseKernels() { ... }
std::string LoadKernelSource(...) { ... }

} // namespace drv_gpu_lib
```

## ✅ Проверка

После сборки файл должен компилироваться без ошибок:

```bash
g++ -c vector_ops_module.cpp -I../include -I../../include
```

---

**Все части содержат полный рабочий код - просто объедините их!**
