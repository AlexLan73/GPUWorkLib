# LID (Library Interface Definition) Архитектура модульной GPU библиотеки

## Оглавление
1. [Введение и цели проекта](#введение-и-цели-проекта)
2. [Концепция LID (Library Interface Definition)](#концепция-lid-library-interface-definition)
3. [Архитектурные принципы](#архитектурные-принципы)
4. [Структура LID интерфейсов](#структура-lid-интерфейсов)
5. [Реализация бэкендов (LID-Backend)](#реализация-бэкендов-lid-backend)
6. [Модули обработки сигналов (LID-Processing)](#модули-обработки-сигналов-lid-processing)
7. [Система сборки и интеграции](#система-сборки-и-интеграции)
8. [ROCm адаптация](#rocm-адаптация)
9. [Дорожная карта разработки](#дорожная-карта-разработки)

---

## Введение и цели проекта

### Цели
Создание долгосрочной, модульной, расширяемой библиотеки для GPU-вычислений с поддержкой:

- **Абстракция бэкенда**: Единое LID API для работы с OpenCL и ROCm
- **Модульность**: Независимые LID-библиотеки для разных функциональных областей
- **Тестируемость**: Полный набор unit-тестов с возможностью условной компиляции
- **Расширяемость**: Простое добавление новых LID модулей без изменения существующего кода
- **Производительность**: Эффективное управление памятью GPU и минимизация копирований

### Scope проекта

#### LID-Backend Layer (Слой абстракции бэкендов)
- LID-Backend-OpenCL: OpenCL 1.2/2.0 реализация
- LID-Backend-ROCm: ROCm/HIP реализация (перспектива)
- LID-Backend-CUDA: CUDA реализация (перспектива)

#### LID-Processing Layer (Слой обработки сигналов)
- LID-DSP: Цифровая обработка сигналов
- LID-FFT: FFT и спектральный анализ
- LID-Statistics: Статистическая обработка
- LID-Filter: Фильтрация сигналов
- LID-Heterodyne: Гетеродинная обработка

---

## Концепция LID (Library Interface Definition)

### Что такое LID?

**LID (Library Interface Definition)** — это контракт между компонентами системы, определяющий:

1. **Публичный API**: Функции, классы, структуры данных
2. **Контракт поведения**: Гарантии производительности и семантики
3. **Зависимости**: От каких других LID зависит данный
4. **Версионирование**: API versioning и backward compatibility

### Принципы LID

```
┌─────────────────────────────────────────────────────────────────┐
│                        LID Architecture                          │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│   ┌─────────────┐    ┌─────────────┐    ┌─────────────┐        │
│   │ LID-CORE    │    │ LID-BACKEND │    │ LID-UTILS   │        │
│   │             │    │             │    │             │        │
│   │ - DrvGPU    │◄──►│ - OpenCL    │◄──►│ - Logging   │        │
│   │ - Memory    │    │ - ROCm      │    │ - Config    │        │
│   │ - Registry  │    │ - CUDA      │    │ - Timer     │        │
│   └──────┬──────┘    └──────┬──────┘    └──────┬──────┘        │
│          │                  │                  │               │
│          └──────────────────┼──────────────────┘               │
│                             │                                  │
│                             ▼                                  │
│   ┌─────────────────────────────────────────────────────┐      │
│   │                  LID-PROCESSING                      │      │
│   │  ┌─────────┐ ┌─────────┐ ┌─────────┐ ┌─────────┐   │      │
│   │  │LID-FFT  │ │LID-DSP  │ │LID-FILTER│ │LID-STAT│   │      │
│   │  └─────────┘ └─────────┘ └─────────┘ └─────────┘   │      │
│   │  ┌─────────┐ ┌─────────┐ ┌─────────┐ ┌─────────┐   │      │
│   │  │LID-HET  │ │LID-ANR  │ │LID-DOA  │ │LID-EXTRA│   │      │
│   │  └─────────┘ └─────────┘ └─────────┘ └─────────┘   │      │
│   └─────────────────────────────────────────────────────┘      │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

### Преимущества LID подхода

1. **Независимая разработка**: Команды могут разрабатывать LID модули параллельно
2. **Plug-and-play**: Легкое подключение/отключение модулей
3. **Тестирование**: Каждый LID может тестироваться изолированно
4. **Документирование**: LID служит контрактом и документацией
5. **Версионирование**: Четкое управление совместимостью

---

## Архитектурные принципы

### SOLID Principles в контексте LID

#### Single Responsibility Principle (SRP)
```
LID-Backend:     Только управление GPU контекстом и памятью
LID-DSP:         Только операции цифровой обработки сигналов
LID-FFT:         Только FFT и спектральный анализ
LID-Statistics:  Только статистические функции
```

#### Open/Closed Principle (OCP)
```
                    ┌─────────────────────┐
                    │   LID-INTERFACE     │
                    │   (Закрыт)          │
                    └──────────┬──────────┘
                               │
              ┌────────────────┼────────────────┐
              ▼                ▼                ▼
     ┌────────────────┐ ┌────────────────┐ ┌────────────────┐
     │ LID-FFT-OPENCL │ │ LID-FFT-ROCM   │ │ LID-FFT-CUDA   │
     │   (Открыт)     │ │   (Открыт)     │ │   (Открыт)     │
     └────────────────┘ └────────────────┘ └────────────────┘
```

#### Liskov Substitution Principle (LSP)
```cpp
// Все LID-Backend реализации взаимозаменяемы
std::unique_ptr<LID_Backend> backend = LID_Backend_Factory::create(BackendType::OpenCL);
// Можно заменить на ROCm без изменения кода клиента
std::unique_ptr<LID_Backend> backend = LID_Backend_Factory::create(BackendType::ROCm);
```

#### Interface Segregation Principle (ISP)
```cpp
// Маленькие, focused интерфейсы
class ILID_Memory {};       // Только управление памятью
class ILID_Compute {};       // Только вычисления
class ILID_Synchronize {};   // Только синхронизация
// Клиенты зависят только от того, что используют
```

#### Dependency Inversion Principle (DIP)
```
┌─────────────────────────────────────────┐
│         LID-PROCESSING (Высокоуровневый) │
└─────────────────┬───────────────────────┘
                  │ зависит от
                  ▼
┌─────────────────────────────────────────┐
│         LID-BACKEND (Абстракция)         │
└─────────────────┬───────────────────────┘
                  │ реализует
                  ▼
┌─────────────────────────────────────────┐
│         OpenCLBackend / ROCmBackend     │
└─────────────────────────────────────────┘
```

### GoF Patterns для LID архитектуры

#### Abstract Factory для LID
```cpp
// LID-Backend Factory
class LID_Backend_Factory {
public:
    static std::unique_ptr<ILID_Backend> create(Backend_Type type);
    static std::vector<Backend_Type> available_backends();
};

// LID-Module Factory
class LID_Module_Factory {
public:
    template<typename ModuleType>
    static std::unique_ptr<ILID_Module> create();
    
    static void register_module(const std::string& name, ModuleFactory factory);
};
```

#### Bridge Pattern
```cpp
// LID-Backend абстракция
class ILID_Backend {
public:
    virtual void initialize() = 0;
    virtual std::unique_ptr<ILID_Memory_Buffer> allocate(size_t size) = 0;
    // ...
};

// Реализация отделена от абстракции
class OpenCL_Backend : public ILID_Backend { /* ... */ };
class ROCm_Backend : public ILID_Backend { /* ... */ };
```

#### Template Method для модулей
```cpp
class ILID_Module {
public:
    virtual void initialize(ILID_Backend& backend) = 0;
    virtual void execute() = 0;
    virtual void shutdown() = 0;
    
    // Template Method
    void process(const LID_Context& ctx) {
        if (!initialized_) initialize(ctx.backend);
        execute();
        // optional: synchronize
    }
};
```

---

## Структура LID интерфейсов

### LID-Backend Interface (LID-BACKEND)

```cpp
// lid/backend/lid_backend.hpp
#pragma once

#include <cstddef>
#include <cstdint>
#include <memory>
#include <string>
#include <vector>

namespace gpu_lib {
namespace lid {
namespace backend {

/**
 * @brief Типы поддерживаемых бэкендов
 */
enum class Backend_Type : uint8_t {
    OpenCL,
    ROCm,
    CUDA,
    SYCL,
    Metal
};

/**
 * @brief Информация об устройстве
 */
struct Device_Info {
    std::string name;
    std::string vendor;
    std::string version;
    uint64_t global_memory_size;
    uint64_t max_memory_alloc_size;
    size_t max_work_group_size;
    uint32_t compute_units;
    bool supports_double;
    bool supports_images;
};

/**
 * @brief Флаги памяти
 */
enum class Memory_Flags : uint32_t {
    None            = 0,
    Read            = 1 << 0,
    Write           = 1 << 1,
    ReadWrite       = Read | Write,
    HostVisible     = 1 << 2,
    HostCoherent    = 1 << 3,
    CopyHostPtr     = 1 << 4
};

/**
 * @brief Размеры работы для kernel
 */
struct Work_Size {
    size_t global_x = 0;
    size_t global_y = 0;
    size_t global_z = 0;
    size_t local_x = 0;
    size_t local_y = 0;
    size_t local_z = 0;
    
    Work_Size() = default;
    Work_Size(size_t gx) : global_x(gx) {}
    Work_Size(size_t gx, size_t gy) : global_x(gx), global_y(gy) {}
    Work_Size(size_t gx, size_t gy, size_t gz) : global_x(gx), global_y(gy), global_z(gz) {}
};

/**
 * @brief LID-Backend Interface
 * 
 * Абстрактный интерфейс для всех GPU бэкендов.
 * Реализует паттерн Bridge для отделения абстракции от реализации.
 */
class ILID_Backend {
public:
    virtual ~ILID_Backend() = default;

    // === Lifecycle ===
    virtual void initialize(int device_id = 0) = 0;
    virtual void shutdown() = 0;
    virtual bool is_initialized() const = 0;

    // === Device Info ===
    virtual Device_Info get_device_info() const = 0;
    virtual Backend_Type get_backend_type() const = 0;
    virtual std::string get_backend_name() const = 0;
    virtual std::string get_backend_version() const = 0;

    // === Memory Management ===
    virtual std::unique_ptr<ILID_Memory_Buffer> create_buffer(
        size_t size,
        Memory_Flags flags = Memory_Flags::ReadWrite
    ) = 0;

    virtual std::unique_ptr<ILID_Memory_Buffer> create_buffer(
        size_t size,
        const void* host_ptr,
        Memory_Flags flags = Memory_Flags::CopyHostPtr
    ) = 0;

    // === Kernel Management ===
    virtual std::unique_ptr<ILID_Kernel> create_kernel(
        const std::string& source,
        const std::string& kernel_name,
        const std::string& build_options = ""
    ) = 0;

    virtual void build_kernel(
        ILID_Kernel& kernel,
        const std::string& build_options = ""
    ) = 0;

    // === Execution ===
    virtual std::unique_ptr<ILID_Command_Queue> create_command_queue() = 0;
    virtual void synchronize() = 0;

    // === Capabilities ===
    virtual size_t get_max_work_group_size() const = 0;
    virtual uint64_t get_max_memory_alloc_size() const = 0;
    virtual bool supports_extension(const std::string& extension) const = 0;
};

/**
 * @brief Интерфейс буфера памяти GPU
 */
class ILID_Memory_Buffer {
public:
    virtual ~ILID_Memory_Buffer() = default;

    virtual size_t get_size() const = 0;
    virtual void* get_host_ptr() = 0;
    virtual const void* get_host_ptr() const = 0;
    virtual void* get_device_ptr() = 0;

    // === Data Transfer ===
    virtual void copy_from(const void* host_ptr, size_t size, size_t offset = 0) = 0;
    virtual void copy_to(void* host_ptr, size_t size, size_t offset = 0) = 0;
    virtual void copy_to(ILID_Memory_Buffer& dst, size_t size, size_t src_offset = 0, size_t dst_offset = 0) = 0;

    // === Mapping ===
    virtual void* map(Memory_Flags flags) = 0;
    virtual void unmap() = 0;

    // === Synchronization ===
    virtual void wait() = 0;
};

/**
 * @brief Интерфейс Kernel
 */
class ILID_Kernel {
public:
    virtual ~ILID_Kernel() = default;

    virtual std::string get_name() const = 0;

    // === Argument Setting ===
    template<typename T>
    void set_arg(int index, const T& value) {
        set_arg_impl(index, sizeof(T), &value);
    }

    void set_arg_buffer(int index, ILID_Memory_Buffer& buffer) {
        set_arg_impl(index, sizeof(void*), &buffer.get_device_ptr());
    }

    void set_arg_size(int index, size_t size) {
        set_arg_impl(index, sizeof(size_t), &size);
    }

    // === Execution ===
    virtual void execute(
        const Work_Size& global_size,
        const Work_Size& local_size = Work_Size()
    ) = 0;

    virtual void execute_async(
        ILID_Command_Queue& queue,
        const Work_Size& global_size,
        const Work_Size& local_size = Work_Size()
    ) = 0;

protected:
    virtual void set_arg_impl(int index, size_t size, const void* value) = 0;
};

/**
 * @brief Интерфейс Command Queue
 */
class ILID_Command_Queue {
public:
    virtual ~ILID_Command_Queue() = default;

    virtual void enqueue_kernel(
        ILID_Kernel& kernel,
        const Work_Size& global_size,
        const Work_Size& local_size = Work_Size()
    ) = 0;

    virtual void enqueue_copy(
        ILID_Memory_Buffer& src,
        ILID_Memory_Buffer& dst,
        size_t size
    ) = 0;

    virtual void flush() = 0;
    virtual void finish() = 0;
};

} // namespace backend
} // namespace lid
} // namespace gpu_lib
```

### LID-Processing Interface (LID-PROCESSING)

```cpp
// lid/processing/lid_dsp.hpp
#pragma once

#include "../backend/lid_backend.hpp"
#include <complex>
#include <vector>
#include <string>

namespace gpu_lib {
namespace lid {
namespace processing {

/**
 * @brief Базовый интерфейс для всех DSP модулей
 */
class ILID_DSP_Module {
public:
    virtual ~ILID_DSP_Module() = default;

    // === Identity ===
    virtual std::string get_name() const = 0;
    virtual std::string get_version() const = 0;
    virtual std::string get_description() const = 0;

    // === Lifecycle ===
    virtual void initialize(backend::ILID_Backend& backend) = 0;
    virtual bool is_initialized() const = 0;
    virtual void shutdown() = 0;

    // === Dependencies ===
    virtual std::vector<std::string> get_dependencies() const {
        return {};
    }

    // === Configuration ===
    virtual void set_parameter(const std::string& name, double value) = 0;
    virtual void set_parameter(const std::string& name, const std::string& value) = 0;
    virtual double get_parameter_double(const std::string& name) const = 0;
    virtual std::string get_parameter_string(const std::string& name) const = 0;
};

/**
 * @brief Интерфейс FFT модуля
 */
class ILID_FFT : public ILID_DSP_Module {
public:
    enum class FFT_Direction : uint8_t {
        Forward,
        Inverse
    };

    enum class FFT_Type : uint8_t {
        Complex_to_Complex,
        Real_to_Complex,
        Complex_to_Magnitude,
        Real_to_Magnitude
    };

    struct FFT_Config {
        size_t fft_size;
        FFT_Type type = FFT_Type::Complex_to_Complex;
        FFT_Direction direction = FFT_Direction::Forward;
        bool normalize = true;
        bool window_enabled = false;
        double window_parameter = 0.0;
    };

    struct FFT_Plan {
        std::unique_ptr<backend::ILID_Memory_Buffer> input;
        std::unique_ptr<backend::ILID_Memory_Buffer> output;
        std::unique_ptr<backend::ILID_Kernel> kernel;
    };

    // === FFT Operations ===
    virtual FFT_Plan create_plan(const FFT_Config& config) = 0;
    virtual void execute_plan(const FFT_Plan& plan) = 0;
    virtual void destroy_plan(FFT_Plan& plan) = 0;

    // === In-place operations ===
    virtual void execute(
        backend::ILID_Memory_Buffer& data,
        size_t size,
        FFT_Direction direction
    ) = 0;

    // === Utility ===
    virtual size_t get_optimal_size(size_t requested_size) const = 0;
};

/**
 * @brief Интерфейс статистического модуля
 */
class ILID_Statistics : public ILID_DSP_Module {
public:
    struct Stats_Config {
        bool compute_min = true;
        bool compute_max = true;
        bool compute_mean = true;
        bool compute_stddev = true;
        bool compute_median = false;
        bool compute_magnitude = true;
    };

    struct Stats_Result {
        float min;
        float max;
        float mean;
        float stddev;
        float median;
        float magnitude;
    };

    // === Computation ===
    virtual Stats_Result compute_stats(
        const backend::ILID_Memory_Buffer& data,
        size_t size,
        const Stats_Config& config = {}
    ) = 0;

    // === Streaming statistics ===
    virtual void reset_running_stats() = 0;
    virtual void update_running_stats(const backend::ILID_Memory_Buffer& data, size_t size) = 0;
    virtual Stats_Result get_running_stats() const = 0;
};

/**
 * @brief Интерфейс фильтрации
 */
class ILID_Filter : public ILID_DSP_Module {
public:
    enum class Filter_Type : uint8_t {
        LowPass,
        HighPass,
        BandPass,
        BandStop,
        MovingAverage,
        Median,
        FIR,
        IIR
    };

    struct Filter_Config {
        Filter_Type type = Filter_Type::LowPass;
        double cutoff_frequency = 0.0;
        double bandwidth = 0.0;
        size_t filter_order = 0;
        double ripple = 0.0;
        std::vector<float> fir_coefficients;  // Для FIR
    };

    // === Filter Operations ===
    virtual void design_filter(const Filter_Config& config) = 0;
    virtual void apply_filter(
        const backend::ILID_Memory_Buffer& input,
        backend::ILID_Memory_Buffer& output,
        size_t size
    ) = 0;

    // === In-place ===
    virtual void apply_filter_inplace(
        backend::ILID_Memory_Buffer& data,
        size_t size
    ) = 0;

    // === Group Delay ===
    virtual double get_group_delay() const = 0;
    virtual void set_group_delay_compensation(bool enable) = 0;
};

/**
 * @brief Интерфейс гетеродинной обработки
 */
class ILID_Heterodyne : public ILID_DSP_Module {
public:
    struct Heterodyne_Config {
        double center_frequency = 0.0;
        double sample_rate = 1.0;
        double phase_offset = 0.0;
        bool enable_nco = true;
        bool enable_mixer = true;
    };

    // === Operations ===
    virtual void configure(const Heterodyne_Config& config) = 0;
    
    virtual void downconvert(
        const backend::ILID_Memory_Buffer& input,
        backend::ILID_Memory_Buffer& output,
        size_t size
    ) = 0;

    virtual void upconvert(
        const backend::ILID_Memory_Buffer& input,
        backend::ILID_Memory_Buffer& output,
        size_t size
    ) = 0;

    // === NCO Control ===
    virtual void set_phase(double phase) = 0;
    virtual void set_frequency(double frequency) = 0;
    virtual void advance_phase(double delta) = 0;
    virtual double get_current_phase() const = 0;
};

/**
 * @brief Интерфейс дробной задержки
 */
class ILID_FractionalDelay : public ILID_DSP_Module {
public:
    enum class Interpolation_Method : uint8_t {
        Linear,
        Cubic,
        Lagrange,
        Farrow,
        Sinc
    };

    struct Delay_Config {
        double delay_samples = 0.0;
        Interpolation_Method method = Interpolation_Method::Farrow;
        size_t fir_length = 32;
        bool enable_compensation = true;
    };

    // === Operations ===
    virtual void set_delay(double delay_samples) = 0;
    virtual void apply_delay(
        const backend::ILID_Memory_Buffer& input,
        backend::ILID_Memory_Buffer& output,
        size_t size,
        double delay_samples
    ) = 0;

    // === High precision ===
    virtual void apply_variable_delay(
        const backend::ILID_Memory_Buffer& input,
        const backend::ILID_Memory_Buffer& delays,
        backend::ILID_Memory_Buffer& output,
        size_t size
    ) = 0;
};

} // namespace processing
} // namespace lid
} // namespace gpu_lib
```

---

## Реализация бэкендов (LID-Backend)

### OpenCL Backend Implementation

```cpp
// lid/backend/opencl/lid_opencl_backend.hpp
#pragma once

#include "../lid_backend.hpp"
#include <CL/cl.h>

namespace gpu_lib {
namespace lid {
namespace backend {

/**
 * @brief OpenCL реализация LID-Backend
 */
class OpenCL_Backend : public ILID_Backend {
public:
    OpenCL_Backend();
    ~OpenCL_Backend() override;

    // === Lifecycle ===
    void initialize(int device_id = 0) override;
    void shutdown() override;
    bool is_initialized() const override;

    // === Device Info ===
    Device_Info get_device_info() const override;
    Backend_Type get_backend_type() const override { return Backend_Type::OpenCL; }
    std::string get_backend_name() const override { return "OpenCL"; }
    std::string get_backend_version() const override;

    // === Memory Management ===
    std::unique_ptr<ILID_Memory_Buffer> create_buffer(
        size_t size,
        Memory_Flags flags = Memory_Flags::ReadWrite
    ) override;

    std::unique_ptr<ILID_Memory_Buffer> create_buffer(
        size_t size,
        const void* host_ptr,
        Memory_Flags flags = Memory_Flags::CopyHostPtr
    ) override;

    // === Kernel Management ===
    std::unique_ptr<IKernel> create_kernel(
        const std::string& source,
        const std::string& kernel_name,
        const std::string& build_options = ""
    ) override;

    void build_kernel(IKernel& kernel, const std::string& build_options = "") override;

    // === Execution ===
    std::unique_ptr<ICommand_Queue> create_command_queue() override;
    void synchronize() override;

    // === Capabilities ===
    size_t get_max_work_group_size() const override;
    uint64_t get_max_memory_alloc_size() const override;
    bool supports_extension(const std::string& extension) const override;

private:
    cl_context context_ = nullptr;
    cl_device_id device_ = nullptr;
    cl_command_queue default_queue_ = nullptr;
    bool initialized_ = false;

    // Helper methods
    void check_error(cl_int error, const std::string& message);
    std::string get_cl_error_string(cl_int error);
};

/**
 * @brief OpenCL Memory Buffer реализация
 */
class OpenCL_Memory_Buffer : public ILID_Memory_Buffer {
public:
    explicit OpenCL_Memory_Buffer(cl_mem buffer, size_t size, Memory_Flags flags);
    ~OpenCL_Memory_Buffer() override;

    size_t get_size() const override;
    void* get_host_ptr() override;
    const void* get_host_ptr() const override;
    void* get_device_ptr() override;

    void copy_from(const void* host_ptr, size_t size, size_t offset = 0) override;
    void copy_to(void* host_ptr, size_t size, size_t offset = 0) override;
    void copy_to(ILID_Memory_Buffer& dst, size_t size, size_t src_offset = 0, size_t dst_offset = 0) override;

    void* map(Memory_Flags flags) override;
    void unmap() override;

    void wait() override;

    // OpenCL specific
    cl_mem get_cl_mem() const { return buffer_; }

private:
    cl_mem buffer_;
    size_t size_;
    Memory_Flags flags_;
    void* mapped_ptr_ = nullptr;
    cl_context context_;
    cl_command_queue queue_;
};

/**
 * @brief OpenCL Kernel реализация
 */
class OpenCL_Kernel : public ILID_Kernel {
public:
    OpenCL_Kernel(cl_program program, const std::string& name);
    ~OpenCL_Kernel() override;

    std::string get_name() const override { return name_; }

    void execute(const Work_Size& global_size, const Work_Size& local_size = Work_Size()) override;
    void execute_async(ICommand_Queue& queue, const Work_Size& global_size, const Work_Size& local_size = Work_Size()) override;

    // OpenCL specific
    cl_kernel get_cl_kernel() const { return kernel_; }

protected:
    void set_arg_impl(int index, size_t size, const void* value) override;

private:
    cl_kernel kernel_;
    std::string name_;
};

/**
 * @brief OpenCL Command Queue реализация
 */
class OpenCL_Command_Queue : public ICommand_Queue {
public:
    explicit OpenCL_Command_Queue(cl_command_queue queue);
    ~OpenCL_Command_Queue() override;

    void enqueue_kernel(IKernel& kernel, const Work_Size& global_size, const Work_Size& local_size = Work_Size()) override;
    void enqueue_copy(ILID_Memory_Buffer& src, ILID_Memory_Buffer& dst, size_t size) override;
    void flush() override;
    void finish() override;

private:
    cl_command_queue queue_;
};

} // namespace backend
} // namespace lid
} // namespace gpu_lib
```

### ROCm Backend Implementation (Перспектива)

```cpp
// lid/backend/rocm/lid_rocm_backend.hpp
#pragma once

#include "../lid_backend.hpp"
#include <hip/hip_runtime.h>

namespace gpu_lib {
namespace lid {
namespace backend {

/**
 * @brief ROCm/HIP реализация LID-Backend
 * 
 * TODO: Реализация при добавлении ROCm поддержки
 */
class ROCm_Backend : public ILID_Backend {
public:
    ROCm_Backend();
    ~ROCm_Backend() override;

    // === Lifecycle ===
    void initialize(int device_id = 0) override;
    void shutdown() override;
    bool is_initialized() const override;

    // === Device Info ===
    Device_Info get_device_info() const override;
    Backend_Type get_backend_type() const override { return Backend_Type::ROCm; }
    std::string get_backend_name() const override { return "ROCm"; }
    std::string get_backend_version() const override;

    // === Memory Management ===
    std::unique_ptr<ILID_Memory_Buffer> create_buffer(
        size_t size,
        Memory_Flags flags = Memory_Flags::ReadWrite
    ) override;

    std::unique_ptr<ILID_Memory_Buffer> create_buffer(
        size_t size,
        const void* host_ptr,
        Memory_Flags flags = Memory_Flags::CopyHostPtr
    ) override;

    // === Kernel Management ===
    std::unique_ptr<IKernel> create_kernel(
        const std::string& source,
        const std::string& kernel_name,
        const std::string& build_options = ""
    ) override;

    void build_kernel(IKernel& kernel, const std::string& build_options = "") override;

    // === Execution ===
    std::unique_ptr<ICommand_Queue> create_command_queue() override;
    void synchronize() override;

    // === Capabilities ===
    size_t get_max_work_group_size() const override;
    uint64_t get_max_memory_alloc_size() const override;
    bool supports_extension(const std::string& extension) const override;

private:
    hipCtx_t context_ = nullptr;
    hipDevice_t device_ = nullptr;
    hipStream_t default_stream_ = nullptr;
    bool initialized_ = false;

    // Helper methods
    void check_error(hipError_t error, const std::string& message);
};

/**
 * @brief ROCm Memory Buffer реализация
 */
class ROCm_Memory_Buffer : public ILID_Memory_Buffer {
public:
    ROCm_Memory_Buffer(void* device_ptr, size_t size, Memory_Flags flags);
    ~ROCm_Memory_Buffer() override;

    size_t get_size() const override;
    void* get_host_ptr() override;
    const void* get_host_ptr() const override;
    void* get_device_ptr() override;

    void copy_from(const void* host_ptr, size_t size, size_t offset = 0) override;
    void copy_to(void* host_ptr, size_t size, size_t offset = 0) override;
    void copy_to(ILID_Memory_Buffer& dst, size_t size, size_t src_offset = 0, size_t dst_offset = 0) override;

    void* map(Memory_Flags flags) override;
    void unmap() override;

    void wait() override;

private:
    void* device_ptr_;
    void* host_ptr_ = nullptr;
    size_t size_;
    Memory_Flags flags_;
};

} // namespace backend
} // namespace lid
} // namespace gpu_lib
```

---

## Модули обработки сигналов (LID-Processing)

### LID-FFT Module Implementation

```cpp
// lid/processing/fft/lid_fft.hpp
#pragma once

#include "../lid_dsp.hpp"
#include "../../backend/lid_backend.hpp"
#include <vector>

namespace gpu_lib {
namespace lid {
namespace processing {

/**
 * @brief FFT Post-Processing Module
 * 
 * LID модуль для обработки результатов FFT:
 * - Поиск максимумов
 * - Параболическая интерполяция
 * - Поиск точек перегиба
 * - Поиск локальных максимумов
 */
class FFTPostProcessing : public ILID_DSP_Module {
public:
    FFTPostProcessing();
    ~FFTPostProcessing() override;

    // === Identity ===
    std::string get_name() const override { return "FFTPostProcessing"; }
    std::string get_version() const override { return "1.0.0"; }
    std::string get_description() const override { 
        return "FFT post-processing: peak finding, interpolation, inflection points"; 
    }

    // === Lifecycle ===
    void initialize(backend::ILID_Backend& backend) override;
    bool is_initialized() const override { return initialized_; }
    void shutdown() override;

    // === Configuration ===
    void set_parameter(const std::string& name, double value) override;
    void set_parameter(const std::string& name, const std::string& value) override;
    double get_parameter_double(const std::string& name) const override;
    std::string get_parameter_string(const std::string& name) const override;

    // === Peak Finding ===
    struct PeakInfo {
        float frequency;
        float magnitude;
        float phase;
        int bin_index;
        float interpolated_frequency;
    };

    struct PeakConfig {
        int max_peaks = 3;
        float threshold = 0.0f;
        bool enable_parabolic_interpolation = true;
    };

    std::vector<PeakInfo> find_top_peaks(
        const backend::ILID_Memory_Buffer& fft_data,
        size_t size,
        const PeakConfig& config = {}
    );

    std::vector<PeakInfo> find_all_peaks(
        const backend::ILID_Memory_Buffer& fft_data,
        size_t size,
        float threshold
    );

    // === Interpolation ===
    float refine_frequency_parabolic(
        const backend::ILID_Memory_Buffer& fft_data,
        int peak_index,
        float sample_rate
    );

    // === Inflection Points ===
    std::vector<int> find_inflection_points(
        const backend::ILID_Memory_Buffer& fft_data,
        size_t size
    );

private:
    bool initialized_ = false;
    backend::ILID_Backend* backend_ = nullptr;

    // Kernels
    std::unique_ptr<backend::ILID_Kernel> magnitude_kernel_;
    std::unique_ptr<backend::ILID_Kernel> find_peaks_kernel_;
    std::unique_ptr<backend::ILID_Kernel> parabolic_kernel_;

    // Buffers
    std::unique_ptr<backend::ILID_Memory_Buffer> magnitude_buffer_;
    std::unique_ptr<backend::ILID_Memory_Buffer> peaks_buffer_;

    // Configuration
    PeakConfig config_;

    void compile_kernels();
    void create_buffers(size_t max_size);
};

} // namespace processing
} // namespace lid
} // namespace gpu_lib
```

### LID-Statistics Module Implementation

```cpp
// lid/processing/statistics/lid_statistics.hpp
#pragma once

#include "../lid_dsp.hpp"
#include "../../backend/lid_backend.hpp"

namespace gpu_lib {
namespace lid {
namespace processing {

/**
 * @brief Signal Statistics Module
 * 
 * LID модуль для статистической обработки сигналов:
 * - Медиана
 * - Среднее значение
 * - Среднеквадратичное отклонение
 * - Магнитуда
 */
class SignalStatistics : public ILID_DSP_Module {
public:
    SignalStatistics();
    ~SignalStatistics() override;

    // === Identity ===
    std::string get_name() const override { return "SignalStatistics"; }
    std::string get_version() const override { return "1.0.0"; }
    std::string get_description() const override {
        return "Signal statistics: mean, median, stddev, magnitude";
    }

    // === Lifecycle ===
    void initialize(backend::ILID_Backend& backend) override;
    bool is_initialized() const override { return initialized_; }
    void shutdown() override;

    // === Configuration ===
    void set_parameter(const std::string& name, double value) override;
    void set_parameter(const std::string& name, const std::string& value) override;
    double get_parameter_double(const std::string& name) const override;
    std::string get_parameter_string(const std::string& name) const override;

    // === Statistics Computation ===
    Stats_Result compute_stats(
        const backend::ILID_Memory_Buffer& data,
        size_t size,
        const Stats_Config& config = {}
    ) override;

    // === Running Statistics ===
    void reset_running_stats() override;
    void update_running_stats(const backend::ILID_Memory_Buffer& data, size_t size) override;
    Stats_Result get_running_stats() const override;

    // === Complex Magnitude ===
    void compute_magnitude(
        const backend::ILID_Memory_Buffer& complex_data,
        backend::ILID_Memory_Buffer& magnitude_output,
        size_t size
    );

    // === Power Spectrum ===
    void compute_power_spectrum(
        const backend::ILID_Memory_Buffer& complex_fft,
        backend::ILID_Memory_Buffer& power_output,
        size_t size
    );

private:
    bool initialized_ = false;
    backend::ILID_Backend* backend_ = nullptr;

    // Kernels
    std::unique_ptr<backend::ILID_Kernel> magnitude_kernel_;
    std::unique_ptr<backend::ILID_Kernel> stats_kernel_;
    std::unique_ptr<backend::ILID_Kernel> reduction_kernel_;

    // Running stats state
    float running_count_ = 0.0f;
    float running_sum_ = 0.0f;
    float running_sum_sq_ = 0.0f;
    float running_min_ = 0.0f;
    float running_max_ = 0.0f;

    // Buffers
    std::unique_ptr<backend::ILID_Memory_Buffer> temp_buffer_;
    std::unique_ptr<backend::ILID_Memory_Buffer> running_stats_buffer_;

    void compile_kernels();
};

} // namespace processing
} // namespace lid
} // namespace gpu_lib
```

---

## Система сборки и интеграции

### Структура проекта LID

```
GPUWorkLib/
├── CMakeLists.txt                    # Root CMake
├── lid/
│   ├── CMakeLists.txt
│   ├── core/                         # LID Core интерфейсы
│   │   ├── CMakeLists.txt
│   │   ├── include/lid/
│   │   │   └── core/
│   │   │       ├── lid_version.hpp
│   │   │       └── lid_types.hpp
│   │   └── src/
│   │
│   ├── backend/                      # LID Backend Layer
│   │   ├── CMakeLists.txt
│   │   ├── opencl/                   # OpenCL реализация
│   │   │   ├── CMakeLists.txt
│   │   │   ├── include/lid/backend/opencl/
│   │   │   ├── src/
│   │   │   └── kernels/
│   │   └── rocm/                     # ROCm реализация (future)
│   │       └── CMakeLists.txt
│   │
│   └── processing/                   # LID Processing Modules
│       ├── CMakeLists.txt
│       ├── fft/                      # FFT модуль
│       │   ├── CMakeLists.txt
│       │   ├── include/lid/processing/fft/
│       │   ├── src/
│       │   └── kernels/
│       ├── statistics/               # Statistics модуль
│       │   └── ...
│       ├── filter/                   # Filter модуль
│       │   └── ...
│       ├── heterodyne/               # Heterodyne модуль
│       │   └── ...
│       └── fractional_delay/         # Fractional Delay модуль
│           └── ...
│
├── examples/
│   ├── CMakeLists.txt
│   ├── basic_usage.cpp
│   └── fft_example.cpp
│
├── tests/
│   ├── CMakeLists.txt
│   ├── unit/
│   ├── integration/
│   └── benchmarks/
│
└── docs/
    ├── architecture.md
    ├── api_reference.md
    └── module_development.md
```

### Root CMakeLists.txt для LID

```cmake
cmake_minimum_required(VERSION 3.18)
project(GPUWorkLib VERSION 1.0.0 LANGUAGES CXX)

# C++ Standard
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# Options
option(LID_BUILD_TESTS "Build unit tests" ON)
option(LID_BUILD_EXAMPLES "Build examples" ON)
option(LID_BUILD_BENCHMARKS "Build benchmarks" OFF)
option(LID_ENABLE_OPENCL "Enable OpenCL backend" ON)
option(LID_ENABLE_ROCM "Enable ROCm backend" OFF)

# Build type
if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release)
endif()

# Global settings
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

# Dependencies
if(LID_ENABLE_OPENCL)
    find_package(OpenCL REQUIRED)
endif()

if(LID_ENABLE_ROCM)
    find_package(ROCm REQUIRED)
endif()

# Testing
if(LID_BUILD_TESTS)
    enable_testing()
    find_package(GTest REQUIRED)
endif()

# Subdirectories
add_subdirectory(lid/core)
add_subdirectory(lid/backend)
add_subdirectory(lid/processing)

if(LID_BUILD_EXAMPLES)
    add_subdirectory(examples)
endif()

if(LID_BUILD_TESTS)
    add_subdirectory(tests)
endif()

if(LID_BUILD_BENCHMARKS)
    add_subdirectory(benchmarks)
endif()

# Package config
include(CMakePackageConfigHelpers)
configure_package_config_file(
    ${CMAKE_CURRENT_SOURCE_DIR}/cmake/LIDConfig.cmake.in
    ${CMAKE_CURRENT_BINARY_DIR}/LIDConfig.cmake
    INSTALL_DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/LID
)
write_basic_package_version_file(
    LIDConfigVersion.cmake
    VERSION ${PROJECT_VERSION}
    COMPATIBILITY SameMajorVersion
)

install(FILES
    ${CMAKE_CURRENT_BINARY_DIR}/LIDConfig.cmake
    ${CMAKE_CURRENT_BINARY_DIR}/LIDConfigVersion.cmake
    DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/LID
)
```

### LID Backend CMakeLists.txt

```cmake
# lid/backend/CMakeLists.txt

# OpenCL Backend
if(LID_ENABLE_OPENCL)
    add_subdirectory(opencl)
endif()

# ROCm Backend
if(LID_ENABLE_ROCM)
    add_subdirectory(rocm)
endif()

# Backend Factory
add_library(lid_backend_factory
    src/backend_factory.cpp
)

target_include_directories(lid_backend_factory
    PUBLIC
        ${CMAKE_CURRENT_SOURCE_DIR}/include
)

target_link_libraries(lid_backend_factory
    PRIVATE
        $<TARGET_NAME_IF_EXISTS:lid_backend_opencl>
        $<TARGET_NAME_IF_EXISTS:lid_backend_rocm>
)

# Create alias for easy linking
add_library(lid::backend_factory ALIAS lid_backend_factory)
```

### LID Processing Module CMakeLists.txt

```cmake
# lid/processing/fft/CMakeLists.txt

# FFT Module Library
add_library(lid_processing_fft
    src/FFTPostProcessing.cpp
    src/FFTKernels.cpp
)

target_include_directories(lid_processing_fft
    PUBLIC
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
        $<INSTALL_INTERFACE:include>
    PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/src
)

target_link_libraries(lid_processing_fft
    PUBLIC
        lid::core
    PRIVATE
        $<TARGET_NAME_IF_EXISTS:lid_backend_opencl>
        $<TARGET_NAME_IF_EXISTS:lid_backend_rocm>
)

# Copy OpenCL kernels
add_custom_command(
    OUTPUT ${CMAKE_CURRENT_BINARY_DIR}/kernels/fft_kernels.hpp
    COMMAND ${CMAKE_COMMAND} -E make_directory "${CMAKE_CURRENT_BINARY_DIR}/kernels"
    COMMAND ${CMAKE_COMMAND} -E copy
        ${CMAKE_CURRENT_SOURCE_DIR}/kernels/fft_postproc.cl
        ${CMAKE_CURRENT_BINARY_DIR}/kernels/fft_kernels.hpp
    DEPENDS ${CMAKE_CURRENT_SOURCE_DIR}/kernels/fft_postproc.cl
)

target_sources(lid_processing_fft PRIVATE
    ${CMAKE_CURRENT_BINARY_DIR}/kernels/fft_kernels.hpp
)

# Tests
if(LID_BUILD_TESTS)
    add_subdirectory(tests)
endif()

# Install
install(TARGETS lid_processing_fft
    EXPORT LIDTargets
    LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
    ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
)

install(DIRECTORY include/
    DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/lid/processing/fft
)

# Alias
add_library(lid::processing::fft ALIAS lid_processing_fft)
```

---

## ROCm адаптация

### Стратегия миграции OpenCL → ROCm

```cpp
// lid/utils/lid_backend_migrator.hpp
#pragma once

#include "../backend/lid_backend.hpp"
#include <string>

namespace gpu_lib {
namespace lid {
namespace utils {

/**
 * @brief Утилита для миграции между бэкендами
 */
class Backend_Migrator {
public:
    /**
     * @brief Миграция состояния с одного бэкенда на другой
     */
    static void migrate(
        ILID_Backend& source,
        ILID_Backend& target,
        const std::vector<ILID_Memory_Buffer*>& buffers
    );

    /**
     * @brief Адаптация kernel для нового бэкенда
     * 
     * Преобразует OpenCL kernel в ROCm/HIP kernel
     */
    static std::string adapt_kernel(
        const std::string& opencl_source,
        Backend_Type target_backend
    );

    /**
     * @brief Проверка совместимости kernel
     */
    static bool is_kernel_compatible(
        const std::string& kernel_source,
        Backend_Type backend_type
    );
};

} // namespace utils
} // namespace lid
} // namespace gpu_lib
```

### ROCm Kernel Migration Example

```
OpenCL Kernel                    ROCm/HIP Kernel
─────────────────                ─────────────────
__kernel void foo(               __global__ void foo(
    __global float* a,               float* a,
    __global float* b,               float* b,
    int n)                           int n)
{                                  {
    int i = get_global_id(0);        int i = hipThreadIdx_x;
    ...                              ...
}                                  }
```

### Backend Switching Example

```cpp
// Пример переключения бэкенда в runtime
#include "lid/backend/lid_backend_factory.hpp"

int main() {
    // Создание бэкенда (OpenCL)
    auto backend = lid::BackendFactory::create(lid::Backend_Type::OpenCL);
    backend->initialize(0);
    
    // Работа с OpenCL бэкендом
    auto buffer = backend->create_buffer(size, lid::Memory_Flags::ReadWrite);
    // ... операции ...
    
    // Переключение на ROCm (перспектива)
    // Все буферы должны быть пересозданы
    auto rocm_backend = lid::BackendFactory::create(lid::Backend_Type::ROCm);
    rocm_backend->initialize(0);
    
    // Миграция буферов
    std::vector<lid::ILID_Memory_Buffer*> buffers = { buffer.get() };
    lid::utils::BackendMigrator::migrate(*backend, *rocm_backend, buffers);
    
    // Продолжение работы с ROCm
    // ...
    
    return 0;
}
```

---

## Дорожная карта разработки

### Phase 1: LID Foundation (Месяцы 1-2)

**Week 1-2: Core LID Interfaces**
- [ ] LID-Backend интерфейс (IBackend)
- [ ] LID-Memory интерфейс
- [ ] LID-Kernel интерфейс
- [ ] Базовые тесты интерфейсов

**Week 3-4: OpenCL Backend Implementation**
- [ ] OpenCL_Backend реализация
- [ ] OpenCL_Memory_Buffer реализация
- [ ] OpenCL_Kernel реализация
- [ ] OpenCL тесты

**Week 5-6: LID Core Services**
- [ ] LID_Backend_Factory
- [ ] LID_Registry
- [ ] LID_Memory_Manager
- [ ] Интеграционные тесты

**Week 7-8: Build System**
- [ ] Полный CMake setup
- [ ] Google Test integration
- [ ] CI/CD basics

### Phase 2: LID Processing Modules (Месяцы 3-4)

**Week 9-10: LID-FFT Module**
- [ ] FFTPostProcessing класс
- [ ] Kernels для поиска пиков
- [ ] Parabolic interpolation
- [ ] Тесты и benchmarks

**Week 11-12: LID-Statistics Module**
- [ ] Median, mean, stddev calculation
- [ ] Magnitude computation
- [ ] Тесты и benchmarks

**Week 13-14: LID-Filter & LID-Heterodyne**
- [ ] Filter design и application
- [ ] Heterodyne processing
- [ ] Тесты

**Week 15-16: Integration**
- [ ] Pipeline examples
- [ ] Integration tests
- [ ] Documentation

### Phase 3: LID Advanced Features (Месяцы 5-6)

**Week 17-18: LID-FractionalDelay**
- [ ] Farrow structure implementation
- [ ] Variable delay support
- [ ] Тесты

**Week 19-20: Performance Optimization**
- [ ] Memory pool optimization
- [ ] Kernel optimization
- [ ] Async execution

**Week 21-22: Additional Modules**
- [ ] LID-DOA (Direction of Arrival)
- [ ] LID-ANR (Adaptive Noise Reduction)
- [ ] LID-Beamforming

**Week 23-24: Documentation & Polish**
- [ ] API documentation
- [ ] Migration guides
- [ ] Performance tuning

### Phase 4: ROCm Support (Месяцы 7-8)

**Week 25-26: ROCm Backend Implementation**
- [ ] ROCm_Backend реализация
- [ ] HIP kernel compilation
- [ ] Memory management

**Week 27-28: Backend Switching**
- [ ] Runtime switching mechanism
- [ ] State migration
- [ ] Cross-backend tests

**Week 29-30: Optimization**
- [ ] ROCm-specific optimizations
- [ ] Performance benchmarks
- [ ] Bug fixes

**Week 31-32: Release**
- [ ] Version 2.0 release
- [ ] Documentation updates
- [ ] Migration guides

### Phase 5: Production & Extensions (Месяц 9+)

**Continuous:**
- [ ] Bug fixes
- [ ] Performance optimization
- [ ] New LID modules
- [ ] CUDA/SYCL backends
- [ ] Community support

---

## Критерии качества LID

### Code Quality

1. **Test Coverage**: Минимум 80% для Core, 70% для Modules
2. **Documentation**: 100% public API documented
3. **Static Analysis**: 0 критических предупреждений
4. **Performance**: Benchmarks для всех модулей

### LID Interface Quality Checklist

- [ ] Интерфейс минимален и focused
- [ ] Все методы имеют документацию
- [ ] Понятные имена методов и параметров
- [ ] Обработка ошибок определена
- [ ] Thread safety документирована
- [ ] Версионирование соблюдается

### Performance Requirements

- **Memory Allocation**: < 1ms для буферов < 1MB
- **Kernel Launch Overhead**: < 100μs
- **Backend Switching**: < 1s для полной миграции
- **Module Initialization**: < 100ms

---

## Заключение

LID архитектура обеспечивает:

✅ **Модульность**: Независимые LID-библиотеки с четкими границами  
✅ **Расширяемость**: Простое добавление модулей и бэкендов  
✅ **Тестируемость**: Полное покрытие тестами  
✅ **Производительность**: Эффективное управление памятью GPU  
✅ **Гибкость**: Runtime переключение OpenCL ↔ ROCm  
✅ **Документированность**: LID как контракт и документация  

Архитектура готова к долгосрочному развитию и легко адаптируется к новым требованиям!
