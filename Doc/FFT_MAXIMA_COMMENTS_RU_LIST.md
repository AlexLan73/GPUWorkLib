# Список файлов fft_maxima с переведёнными на русский комментариями

Комментарии с английского переведены на русский в следующих файлах модуля **modules/fft_maxima**:

## Заголовки (include/)

| Файл | Что сделано |
|------|-------------|
| **include/antenna_fft_core.h** | Полный перевод: описание файла и класса AntennaFFTCore, секции (Public types, Constructor/Destructor, Public interface, Virtual methods, Protected utilities, Protected fields), все @brief и пояснения к полям (Processing parameters, DrvGPU backend, OpenCL resources, clFFT resources, Common GPU buffers, Userdata buffers, Profiling, Batch configuration). |
| **include/antenna_fft_release.h** | Описание файла и класса AntennaFFTProcMax уже частично на русском; дополнены/уточнены секции (Реализации виртуальных методов), @brief (Initialize, ProcessSingleBatch, ProcessBatch, AllocateBuffers, ReleaseBuffers), приватные методы (CreateFFTPlanWithCallbacks, ExecuteFFTWithCallbacks, ReadResults), приватные поля (буферы выбранного спектра, кэш планов). |
| **include/spectrum_maxima_finder.h** | Переведены комментарии к полям: pre_callback_userdata_, fft_input_, fft_output_, maxima_output_ (FFT input buffer → Входной буфер FFT и т.д.). Остальное уже было на русском. |
| **include/fft_plan_cache.hpp** | Полный перевод: FFTPlanKey, FFTPlanEntry, FFTPlanCache (описание класса, конструктор/деструктор, Core API, GetOrCreate с Cache HIT/MISS, Create plan, Configure plan), HasPlan, IsBaked, MarkBaked, Remove, ClearAll, секция Statistics (GetCacheSize, GetTotalCreates, GetTotalHits, GetHitRatio, PrintStats), Private members. |
| **include/fft_logger.h** | Перевод описания файла и класса FFTLogger, секции Configuration → Настройки, все @brief (Enable or disable → Включить или отключить, Check if logging is enabled → Проверить включено ли логирование, Set minimum log level, Set output stream, Set custom log callback), Logging methods → Методы логирования. |
| **include/fft_batch_adapter.hpp** | Секция «FFTBatchAdapter — Bridge between AntennaFFT and DrvGPU::BatchManager» → «FFTBatchAdapter — связка между AntennaFFT и DrvGPU::BatchManager». Остальные комментарии уже на русском. |
| **include/kernels/fft_kernel_sources.hpp** | Секция «GetPaddingKernelSource() - Standalone Kernel для Batch Processing» → «GetPaddingKernelSource() — отдельное ядро для пакетной обработки»; «MEMORY LAYOUT» → «РАЗМЕЩЕНИЕ В ПАМЯТИ»; уточнены формулировки (padding нулями → дополнением нулями, ОТДЕЛЬНЫЕ → отдельные, КЛЮЧЕВАЯ ФИЧА - beam_offset → КЛЮЧЕВАЯ ФИЧА — beam_offset). |

## Исходники (src/)

| Файл | Что сделано |
|------|-------------|
| **src/antenna_fft_core.cpp** | Секция «Constructor / Destructor» → «Конструктор / Деструктор»; комментарии в конструкторе (Validate parameters → Проверка параметров, Check backend → Проверка бэкенда, Get context → Получить context, Calculate nFFT → Вычисление nFFT, Initialize clFFT library → Инициализация библиотеки clFFT, Manually initialize → Ручная инициализация, Initialize profiling → Инициализация профилирования, Calculate batch config → Расчёт конфигурации пакетов); в деструкторе (Note: derived classes → Примечание: производные классы). |
| **src/antenna_fft_release.cpp** | Секции «Constructor / Destructor» и «Virtual method implementations» → «Конструктор / Деструктор», «Реализации виртуальных методов»; комментарии (Call virtual Initialize → Вызов виртуального Initialize, Create FFT plan cache → Создание кэша FFT-планов, Allocate buffers → Выделение буферов, Ensure buffers and plan → Убедиться что буферы и план готовы, Copy input data → Копирование входных данных, 32 bytes header offset → смещение заголовка 32 байта, Execute FFT with callbacks → Выполнение FFT с колбэками, Wait for completion → Ожидание завершения). |

## Параметры и прочее

| Файл | Статус |
|------|--------|
| **include/interface/antenna_fft_params.h** | Уже на русском — не менялся. |
| **include/fft_result_writer.hpp** | Секции и комментарии уже на русском — не менялся. |

---

## Итог: в каких файлах был выполнен перевод

1. **modules/fft_maxima/include/antenna_fft_core.h**
2. **modules/fft_maxima/include/antenna_fft_release.h**
3. **modules/fft_maxima/include/spectrum_maxima_finder.h**
4. **modules/fft_maxima/include/fft_plan_cache.hpp**
5. **modules/fft_maxima/include/fft_logger.h**
6. **modules/fft_maxima/include/fft_batch_adapter.hpp**
7. **modules/fft_maxima/include/kernels/fft_kernel_sources.hpp**
8. **modules/fft_maxima/src/antenna_fft_core.cpp**
9. **modules/fft_maxima/src/antenna_fft_release.cpp**

*Дата: 2026-02-07*
