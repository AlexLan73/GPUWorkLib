# Сборка проекта {#build_guide_page}

> CMake + ROCm/HIP + OpenCL 3.0. Две ветки — параллельное развитие.

---

## Ветки

| Ветка | Платформа | Сборка | FFT |
|-------|-----------|--------|-----|
| **main** | Linux, AMD GPU | CMake + GCC/Clang | ROCm / hipFFT |
| **nvidia** | Windows, NVIDIA GPU | Ninja + MSVC (VS 2026) | OpenCL / clFFT |

@warning Ветки **не планируется объединять** — параллельное развитие.

---

## Зависимости

| Компонент | Версия | Назначение |
|-----------|--------|-----------|
| ROCm | 7.2+ | HIP runtime, hipFFT, rocBLAS, rocSOLVER |
| OpenCL | 3.0 | Кросс-платформенный backend |
| pybind11 | 2.11+ | Python bindings |
| plog | 1.1.10 | Логирование (per-GPU) |
| Graphviz | 2.50+ | Doxygen диаграммы |

---

## Сборка (Linux/AMD)

```bash
mkdir build && cd build
cmake .. -DENABLE_ROCM=ON -DCMAKE_BUILD_TYPE=Release
make -j$(nproc)
```

## Сборка (Windows/NVIDIA)

```powershell
mkdir build; cd build
cmake .. -G Ninja -DCMAKE_C_COMPILER=cl -DCMAKE_CXX_COMPILER=cl
ninja
```

---

## Запуск тестов

```bash
# C++ — все модули
./GPUWorkLib all

# C++ — один модуль
./GPUWorkLib fft_func

# Python (НЕ pytest!)
python Python_test/fft_func/test_process_magnitude_rocm.py
```

---

## Сборка Doxygen

```powershell
cd Doc/Doxygen
.\build_docs.bat     # Windows
# или
./build_docs.sh      # Linux
```

Порядок: DrvGPU (.tag) → модули (.tag) → главный (TAGFILES).
