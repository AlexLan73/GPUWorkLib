# Установка зависимостей — GPUWorkLib (Ubuntu / RTX 3060)

## Обязательные пакеты

```bash
# OpenCL (для NVIDIA RTX 3060)
sudo apt install ocl-icd-opencl-dev opencl-headers

# Драйвер NVIDIA (если ещё не установлен)
# Должен быть установлен для работы OpenCL на RTX 3060
# sudo apt install nvidia-driver-XXX  # или через proprietary driver
```

## Рекомендуемые (ускоряют первую сборку)

```bash
# spdlog — если установлен, FetchContent не скачивает его из GitHub
sudo apt install libspdlog-dev
```

## Опционально (clFFT для FFT)

```bash
sudo apt install libclfft-dev
```

## Сборка

```bash
./run.sh re    # Release
./run.sh de    # Debug
./run.sh build # только сборка, без запуска
```

Или вручную:

```bash
cmake -B build -DCMAKE_BUILD_TYPE=Release -DENABLE_OPENCL=ON
cmake --build build
./build/GPUWorkLib
```
