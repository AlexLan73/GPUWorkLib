# 🚀 Быстрый старт - DrvGPU CMake конфигурация

## 📌 Краткая инструкция

Вы получили **4 CMake файла** для настройки DrvGPU как отдельной библиотеки:

### 1️⃣ **DrvGPU-CMake-Simple.txt** ⭐ НАЧНИТЕ С ЭТОГО!
**Для: Быстрый старт, все файлы в одной папке**

Простейший вариант для начала работы:
```bash
# 1. Положите все ваши .cpp/.hpp файлы в одну папку
# 2. Переименуйте файл
mv DrvGPU-CMake-Simple.txt CMakeLists.txt

# 3. Создайте build директорию
mkdir build && cd build

# 4. Запустите CMake
cmake ..

# 5. Соберите
make -j$(nproc)

# 6. Готово! У вас есть libdrvgpu.a
```

**Что делает:**
- Собирает все ваши OpenCL .cpp файлы
- Создаёт статическую библиотеку `libdrvgpu.a`
- Собирает примеры `example_single_gpu` и `example_multi_gpu`

---

### 2️⃣ **DrvGPU-CMakeLists-Main.txt**
**Для: Профессиональная структура проекта**

Полнофункциональная CMake конфигурация с:
- ✅ Разделением на include/ и src/
- ✅ Поддержкой install (установка в систему)
- ✅ Экспортом targets для других проектов
- ✅ find_package(DrvGPU) поддержкой
- ✅ Опциями сборки (SHARED/STATIC, CUDA/VULKAN)

**Требует структуру:**
```
DrvGPU/
├── CMakeLists.txt          ← Этот файл
├── include/                ← Публичные .hpp
├── src/                    ← Реализации .cpp
├── opencl/                 ← Ваш OpenCL код
└── examples/               ← Примеры
```

**Использование:**
```bash
mv DrvGPU-CMakeLists-Main.txt CMakeLists.txt
mkdir build && cd build
cmake .. -DCMAKE_BUILD_TYPE=Release
make -j$(nproc)
sudo make install  # Установить в /usr/local/
```

---

### 3️⃣ **DrvGPU-CMake-Examples.txt**
**Для: Сборка примеров**

Положите в `examples/CMakeLists.txt`:
```bash
mkdir -p examples
mv DrvGPU-CMake-Examples.txt examples/CMakeLists.txt
```

Собирает:
- `example_single_gpu` - базовое использование
- `example_multi_gpu` - Multi-GPU сценарии

---

### 4️⃣ **DrvGPU-CMake-Config.in**
**Для: find_package() поддержки**

Положите в `cmake/DrvGPUConfig.cmake.in`:
```bash
mkdir -p cmake
mv DrvGPU-CMake-Config.in cmake/DrvGPUConfig.cmake.in
```

Позволяет другим проектам использовать:
```cmake
find_package(DrvGPU 1.0 REQUIRED)
target_link_libraries(my_app PRIVATE DrvGPU::drvgpu)
```

---

## 🎯 Какой файл выбрать?

### Вариант A: У меня всё в одной папке → **DrvGPU-CMake-Simple.txt**
```
your_project/
├── CMakeLists.txt           ← Simple
├── opencl_core.cpp
├── opencl_manager.cpp
├── drv_gpu.hpp
├── gpu_manager.hpp
├── single_gpu.cpp
└── multi_gpu.cpp
```

### Вариант B: Хочу правильную структуру → **DrvGPU-CMakeLists-Main.txt**
```
DrvGPU/
├── CMakeLists.txt           ← Main
├── cmake/
│   └── DrvGPUConfig.cmake.in
├── include/
│   ├── drv_gpu.hpp
│   └── ...
├── src/
│   └── core/
│       └── drv_gpu.cpp
├── opencl/
│   ├── opencl_core.cpp
│   └── ...
└── examples/
    ├── CMakeLists.txt       ← Examples
    └── single_gpu.cpp
```

---

## 🔧 Минимальная настройка (5 минут)

### Шаг 1: Создайте проект
```bash
mkdir DrvGPU && cd DrvGPU
```

### Шаг 2: Положите ваши файлы
```bash
# Ваши OpenCL файлы (.cpp и .hpp)
cp /path/to/opencl_*.{cpp,hpp} .
cp /path/to/command_queue_pool.* .
cp /path/to/kernel_program.* .
cp /path/to/gpu_memory*.{cpp,hpp} .

# DrvGPU заголовки (из выгруженных файлов)
cp DrvGPU-Core-*.hpp .
cp DrvGPU-Backend-*.hpp .
cp DrvGPU-Memory-*.hpp .
cp DrvGPU-Modules-*.hpp .
cp DrvGPU-Common-*.hpp .

# Примеры
cp DrvGPU-Examples-*.cpp .
```

### Шаг 3: CMakeLists.txt
```bash
cp DrvGPU-CMake-Simple.txt CMakeLists.txt
```

### Шаг 4: Соберите
```bash
mkdir build && cd build
cmake ..
make -j$(nproc)
```

### Шаг 5: Готово! ✅
```bash
ls -l
# Вы должны увидеть:
# libdrvgpu.a          ← Ваша библиотека
# example_single_gpu   ← Пример Single GPU
# example_multi_gpu    ← Пример Multi-GPU
```

---

## 📦 Использование библиотеки

### В том же CMake проекте:
```cmake
# В вашем главном CMakeLists.txt
add_subdirectory(DrvGPU)

add_executable(my_app main.cpp)
target_link_libraries(my_app PRIVATE DrvGPU::drvgpu)
```

### После установки (make install):
```cmake
find_package(DrvGPU REQUIRED)

add_executable(my_app main.cpp)
target_link_libraries(my_app PRIVATE DrvGPU::drvgpu)
```

### Компиляция вручную:
```bash
g++ -std=c++17 main.cpp -I/usr/local/include/drvgpu \
    -L/usr/local/lib -ldrvgpu -lOpenCL -lpthread
```

---

## ⚙️ Опции CMake

### DrvGPU-CMake-Simple.txt:
```bash
cmake .. -DBUILD_SHARED_LIBS=ON      # Собрать .so вместо .a
cmake .. -DBUILD_EXAMPLES=OFF        # Не собирать примеры
```

### DrvGPU-CMakeLists-Main.txt:
```bash
cmake .. -DDRVGPU_BUILD_SHARED=ON           # Shared library
cmake .. -DDRVGPU_BUILD_EXAMPLES=ON         # Примеры
cmake .. -DDRVGPU_BUILD_TESTS=ON            # Тесты
cmake .. -DDRVGPU_ENABLE_OPENCL=ON          # OpenCL backend
cmake .. -DDRVGPU_ENABLE_CUDA=OFF           # CUDA (будущее)
cmake .. -DCMAKE_INSTALL_PREFIX=/opt/drvgpu # Путь установки
```

---

## 🐛 Частые проблемы

### ❌ "OpenCL not found"
**Решение:**
```bash
# Ubuntu/Debian
sudo apt-get install opencl-headers ocl-icd-opencl-dev

# CentOS/RHEL
sudo yum install opencl-headers ocl-icd-devel

# Arch Linux
sudo pacman -S opencl-headers ocl-icd
```

### ❌ "Cannot find source file: drv_gpu.cpp"
**Причина:** Используете Main CMakeLists, но нет .cpp файлов

**Решение 1:** Используйте Simple CMakeLists (header-only)
```bash
mv DrvGPU-CMake-Simple.txt CMakeLists.txt
```

**Решение 2:** Создайте .cpp файлы (см. DrvGPU-Setup-Guide.md)

### ❌ Ошибки include: "drv_gpu.hpp: No such file"
**Причина:** Неправильные пути

**Решение:** Проверьте структуру:
```bash
ls -la
# Должно быть:
# drv_gpu.hpp
# gpu_manager.hpp
# opencl_backend.hpp
# ...
```

---

## 📚 Дополнительные файлы

- **DrvGPU-Setup-Guide.md** - Детальное руководство по структуре проекта
- **README.md** - Полная документация DrvGPU
- Все `DrvGPU-*.hpp` - Заголовочные файлы библиотеки
- Все `DrvGPU-*.cpp` - Примеры использования

---

## ✅ Проверочный список

После сборки проверьте:

- [ ] Библиотека собралась (`libdrvgpu.a` или `libdrvgpu.so`)
- [ ] Примеры собрались (`example_single_gpu`, `example_multi_gpu`)
- [ ] Нет предупреждений компиляции
- [ ] `cmake ..` показывает "OpenCL found"

---

## 🎉 Готово!

Теперь у вас есть полностью рабочая библиотека DrvGPU!

**Следующие шаги:**
1. Попробуйте запустить `./example_single_gpu`
2. Изучите код примеров
3. Интегрируйте DrvGPU в свой проект
4. Прочитайте полную документацию в README.md

**Вопросы?** Смотрите:
- `DrvGPU-Setup-Guide.md` - детальная настройка
- `README.md` - API документация
- `examples/` - рабочие примеры

---

## 📞 Полезные команды

```bash
# Пересобрать проект
cd build && cmake --build . --clean-first

# Пересобрать с отладкой
cmake .. -DCMAKE_BUILD_TYPE=Debug

# Посмотреть все опции
cmake -L ..

# Показать подробный вывод сборки
make VERBOSE=1

# Установить в систему
sudo make install

# Удалить из системы
sudo make uninstall  # (если есть)
```

**Удачи! 🚀**
