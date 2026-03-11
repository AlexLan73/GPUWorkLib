# Ветка: nvidia

## Для кого

Эта ветка — для **Windows + NVIDIA GPU** с OpenCL/clFFT.  
Активная ветка: домашние эксперименты, прототипы, тестирование новых идей на NVIDIA.

Ветка `main` — для **Linux + AMD GPU** с ROCm.

---

## Архитектура модуля fft_func

| Ветка  | Backend       | Платформа      | Реализации                        |
|--------|---------------|----------------|-----------------------------------|
| main   | ROCm/hipFFT   | Linux / AMD    | FFTProcessorROCm, SpectrumMaximaFinderROCm |
| nvidia | OpenCL/clFFT  | Windows / NVIDIA | FFTProcessor, SpectrumMaximaFinder |

DrvGPU (OpenCL инфраструктура) — присутствует в обеих ветках.

---

## Зависимости для сборки на Windows

1. **MSVC 2022** (Visual Studio 17) или **Ninja + MSVC toolchain**
2. **OpenCL SDK** — устанавливается вместе с NVIDIA драйвером или отдельно:
   - [NVIDIA CUDA Toolkit](https://developer.nvidia.com/cuda-downloads) (включает OpenCL headers)
   - Или [Khronos OpenCL Headers](https://github.com/KhronosGroup/OpenCL-Headers)
3. **clFFT** — AMD open-source FFT library for OpenCL:
   - Скачать: https://github.com/clMathLibraries/clFFT/releases
   - Или собрать из исходников (рекомендуется для Windows)

---

## Как собирать

### Visual Studio 2022

```bash
# Настроить пути в CMakePresets.json (windows-nvidia пресет):
#   OpenCL_ROOT → путь к CUDA Toolkit (где есть include/CL/cl.h)
#   CLFFT_DIR   → путь к clFFT (где есть include/clFFT.h и lib/clFFT.lib)

cmake --preset windows-nvidia
cmake --build build-nvidia --config Debug
```

### Ninja (быстрее)

```bash
cmake --preset windows-nvidia-ninja
cmake --build build-nvidia
```

---

## Запуск тестов

```bash
cd build-nvidia
./gpu_worklib_test.exe fft_func
# или
./gpu_worklib_test fft_func
```

---

## Синхронизация с main

Новые ROCm-нейтральные фичи из `main` можно подтянуть через:
```bash
git fetch origin
git merge origin/main --no-ff -m "sync: merge updates from main"
```

Конфликты в `modules/fft_func/` — ожидаемы (разные файлы), решаются в пользу `nvidia` ветки.

---

*Последнее обновление: 2026-03-11*
