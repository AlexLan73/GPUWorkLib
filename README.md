# GPUWorkLib - Модульная GPU библиотека для вычислений

## Описание

**GPUWorkLib** — модульная, расширяемая библиотека для GPU-вычислений с поддержкой нескольких бэкендов (OpenCL, ROCm) и独立的LID-модулей для обработки сигналов.

## Архитектура

Проект построен на основе **LID (Library Interface Definition)** архитектуры, обеспечивающей:

- **Абстракция бэкенда**: Единый API для OpenCL и ROCm
- **Модульность**: Независимые библиотеки для разных функциональных областей
- **Расширяемость**: Легкое добавление новых модулей и бэкендов
- **Тестируемость**: Полный набор unit-тестов

### Структура LID

```
┌─────────────────────────────────────────────────────────────┐
│                     GPUWorkLib                              │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│   ┌─────────────────────────────────────────────────────┐   │
│   │              LID-BACKEND LAYER                      │   │
│   │  ┌──────────────┐  ┌──────────────┐                │   │
│   │  │ OpenCL LID   │  │  ROCm LID    │ (перспектива)  │   │
│   │  │ Backend      │  │  Backend     │                │   │
│   │  └──────────────┘  └──────────────┘                │   │
│   └─────────────────────────────────────────────────────┘   │
│                              │                               │
│                              ▼                               │
│   ┌─────────────────────────────────────────────────────┐   │
│   │              LID-CORE LAYER                         │   │
│   │  • Memory Manager    • Module Registry              │   │
│   │  • Backend Factory   • Context Management           │   │
│   └─────────────────────────────────────────────────────┘   │
│                              │                               │
│                              ▼                               │
│   ┌─────────────────────────────────────────────────────┐   │
│   │            LID-PROCESSING LAYER                     │   │
│   │  ┌─────────┐ ┌─────────┐ ┌─────────┐ ┌─────────┐   │   │
│   │  │LID-FFT  │ │LID-STAT │ │LID-FILT │ │LID-HET  │   │   │
│   │  │         │ │         │ │         │ │         │   │   │
│   │  └─────────┘ └─────────┘ └─────────┘ └─────────┘   │   │
│   │  ┌─────────┐ ┌─────────┐ ┌─────────┐ ┌─────────┐   │   │
│   │  │LID-DELAY│ │LID-DOA  │ │LID-ANR  │ │LID-BEAM │   │   │
│   │  └─────────┘ └─────────┘ └─────────┘ └─────────┘   │   │
│   └─────────────────────────────────────────────────────┘   │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

## LID Модули

### LID-Backend (Слой абстракции бэкендов)

| Модуль | Статус | Описание |
|--------|--------|----------|
| LID-Backend-OpenCL | ✅ Готов | OpenCL 1.2/2.0 реализация |
| LID-Backend-ROCm | 🔜 Планируется | ROCm/HIP реализация |
| LID-Backend-CUDA | 📋 Перспектива | CUDA реализация |

### LID-Processing (Слой обработки сигналов)

| Модуль | Статус | Описание |
|--------|--------|----------|
| LID-FFT | ✅ Готов | FFT и пост-обработка (пики, интерполяция) |
| LID-Statistics | ✅ Готов | Статистика (медиана, среднее, stddev) |
| LID-Filter | 🔜 Планируется | Фильтрация (lowpass, bandpass, FIR, IIR) |
| LID-Heterodyne | 🔜 Планируется | Гетеродинная обработка |
| LID-FractionalDelay | 📋 Перспектива | Дробная задержка сигналов |
| LID-DOA | 📋 Перспектива | Direction of Arrival |
| LID-ANR | 📋 Перспектива | Adaptive Noise Reduction |

## Установка

### Требования

- C++17 совместимый компилятор (GCC 9+, Clang 10+, MSVC 2019+)
- CMake 3.18+
- OpenCL SDK (для OpenCL бэкенда)
- ROCm SDK (опционально, для ROCm бэкенда)

### Сборка

```bash
# Клонирование репозитория
git clone https://github.com/your-org/GPUWorkLib.git
cd GPUWorkLib

# Создание директории сборки
mkdir build && cd build

# Конфигурация CMake
cmake .. \
    -DCMAKE_BUILD_TYPE=Release \
    -DLID_BUILD_TESTS=ON \
    -DLID_BUILD_EXAMPLES=ON \
    -DLID_ENABLE_OPENCL=ON \
    -DLID_ENABLE_ROCM=OFF

# Сборка
cmake --build . --parallel

# Запуск тестов
ctest
```

### CMake Опции

| Опция | По умолчанию | Описание |
|-------|--------------|----------|
| `LID_BUILD_TESTS` | ON | Сборка unit-тестов |
| `LID_BUILD_EXAMPLES` | ON | Сборка примеров |
| `LID_BUILD_BENCHMARKS` | OFF | Сборка benchmarks |
| `LID_ENABLE_OPENCL` | ON | Включить OpenCL бэкенд |
| `LID_ENABLE_ROCM` | OFF | Включить ROCm бэкенд |

## Быстрый старт

### Пример использования

```cpp
#include <lid/backend/lid_backend_factory.hpp>
#include <lid/processing/fft/lid_fft.hpp>

using namespace gpu_lib::lid;

int main() {
    // Создание бэкенда
    auto backend = BackendFactory::create(Backend_Type::OpenCL);
    backend->initialize(0);

    // Создание FFT модуля
    auto fft = std::make_unique<processing::FFTPostProcessing>();
    fft->initialize(*backend);

    // Выделение памяти
    size_t fft_size = 1024;
    auto input = backend->create_buffer(fft_size * sizeof(std::complex<float>));
    auto output = backend->create_buffer(fft_size * sizeof(std::complex<float>));

    // Копирование данных на GPU
    input->copy_from(host_data.data(), host_data.size() * sizeof(std::complex<float>));

    // Поиск пиков
    processing::FFTPostProcessing::PeakConfig config;
    config.max_peaks = 3;
    config.threshold = 0.1f;

    auto peaks = fft->find_top_peaks(*input, fft_size, config);

    // Вывод результатов
    for (const auto& peak : peaks) {
        printf("Peak: %.2f Hz, Magnitude: %.2f\n", 
               peak.frequency, peak.magnitude);
    }

    return 0;
}
```

### Runtime переключение бэкендов

```cpp
// Переключение с OpenCL на ROCm
auto opencl_backend = BackendFactory::create(Backend_Type::OpenCL);
opencl_backend->initialize(0);

// ... работа с OpenCL ...

// Переключение на ROCm
auto rocm_backend = BackendFactory::create(Backend_Type::ROCm);
rocm_backend->initialize(0);

// Миграция буферов
BackendMigrator::migrate(*opencl_backend, *rocm_backend, buffers);

// Продолжение работы с ROCm
```

## Документация

- [Архитектура LID](Doc/Архитекура/LID-Architecture.md)
- [Архитектура GPU библиотеки](Doc/Архитекура/GPU-Library-Architecture.md)
- [Руководство по разработке модулей](Doc/Архитекура/Module-Development.md)
- [API Reference](docs/api_reference.md)

## Структура проекта

```
GPUWorkLib/
├── CMakeLists.txt                    # Root CMake
├── README.md                          # Этот файл
├── run.bat                            # Windows launcher
├── run.sh                             # Linux launcher
├── .gitignore                         # Git ignore rules
├── CLAUDE.md                          # Claude AI instructions
│
├── clFFT/                            # clFFT библиотека
│   └── include/
│       ├── clFFT.h
│       ├── clFFT.version.h
│       ├── clAmdFft.h
│       └── clAmdFft.version.h
│
├── Doc/                              # Проектная документация
│   ├── Архитекура/
│   │   ├── GPU-Library-Architecture.md
│   │   ├── LID-Architecture.md
│   │   ├── Quick-Start-Guide.md
│   │   ├── Implementation-Examples.md
│   │   └── ... (другие документы)
│   └── DrvGPU/
│       └── Arс/
│           └── Примеры кода
│
├── include/DrvGPU/                   # Заголовочные файлы DrvGPU
│   ├── CMakeLists.txt
│   ├── drv_gpu.hpp                   # Основной класс DrvGPU
│   ├── drv_gpu.cpp
│   ├── i_backend.hpp                 # Интерфейс бэкенда
│   ├── i_compute_module.hpp          # Интерфейс вычислительного модуля
│   ├── module_registry.hpp           # Реестр модулей
│   ├── module_registry.cpp
│   ├── backend_type.hpp              # Типы бэкендов
│   ├── gpu_device_info.hpp           # Информация об устройстве
│   ├── gpu_manager.hpp               # Менеджер GPU
│   ├── load_balancing.hpp            # Балансировка нагрузки
│   │
│   ├── backends/                    # Реализации бэкендов
│   │   └── opencl/                   # OpenCL бэкенд
│   │       ├── opencl_backend.hpp
│   │       ├── opencl_backend.cpp
│   │       ├── opencl_core.hpp
│   │       ├── opencl_core.cpp
│   │       ├── command_queue_pool.hpp
│   │       └── command_queue_pool.cpp
│   │
│   └── memory/                      # Управление памятью
│       ├── memory_manager.hpp
│       ├── gpu_buffer.hpp            # GPU буфер
│       ├── i_memory_buffer.hpp       # Интерфейс буфера памяти
│       ├── memory_type.hpp           # Типы памяти
│       ├── svm_buffer.hpp            # SVM буфер
│       └── svm_capabilities.hpp      # SVM возможности
│
├── interface/                        # Внешние интерфейсы
│   ├── antenna_fft_params.h
│   ├── combined_delay_param.h
│   ├── DelayParameter.h
│   ├── lfm_parameters.h
│
├── modules/                          # Вычислительные модули
│   └── ...                           # Модули обработки сигналов
│
├── src/                              # Исходники
│   ├── CMakeLists.txt
│   ├── main.cpp
│   └── backends/
│
└── tests/                            # Тесты
    ├── single_gpu.hpp
    └── multi_gpu.hpp
```

## Тестирование

### Запуск тестов

```bash
# Из директории build
ctest --output-on-failure

# Или запуск конкретного теста
./tests/unit/test_fft
```

### Тестовые метрики

| Метрика | Целевое значение | Текущее |
|---------|------------------|---------|
| Coverage Core | ≥ 80% | - |
| Coverage Modules | ≥ 70% | - |
| Unit Tests | 100% passed | - |
| Integration Tests | 100% passed | - |

## Производительность

| Операция | Целевое время | Статус |
|----------|---------------|--------|
| Memory Allocation (< 1MB) | < 1ms | - |
| Kernel Launch | < 100μs | - |
| Backend Switching | < 1s | - |
| Module Init | < 100ms | - |

## Разработка

### Добавление нового LID модуля

1. Создайте структуру директорий в `lid/processing/new_module/`
2. Определите интерфейс в `include/lid/processing/new_module/`
3. Реализуйте модуль в `src/`
4. Добавьте CMakeLists.txt
5. Напишите тесты в `tests/unit/`
6. Зарегистрируйте модуль в CMakeLists.txt родительской директории

### Добавление нового бэкенда

1. Создайте реализацию интерфейса `ILID_Backend`
2. Обновите `BackendFactory::create()` для нового типа
3. Добавьте тесты совместимости
4. Обновите документацию по миграции

## Лицензия

MIT License

## Контакты

- Repository: https://github.com/your-org/GPUWorkLib
- Issues: https://github.com/your-org/GPUWorkLib/issues

---

**Статус проекта**: 🚧 Активная разработка
**Версия**: 1.0.0-alpha
**Дата**: 2026-02-01
